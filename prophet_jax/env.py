"""THE LOAD-BEARING BRIDGE  (prophet_jax/env.py)
================================================

Wraps pgx's native ``chess`` environment but exposes **prophet's** world to
the rest of the JAX port, so that torch weights trained against
``prophet/encoding.py`` transfer to JAX **unchanged**:

  * a 24-feature ``[B, 64, 24]`` board encoding (NOT pgx's ``(8,8,119)`` obs),
  * a 4096-action from*64+to space (NOT pgx's 4672),
  * side-to-move / ^56-flipped "model space" (NOT pgx's raw square layout),

all of it fully ``vmap``/``jit``-able and GPU-resident.

Why we re-derive instead of using pgx outputs
---------------------------------------------
pgx natively gives ``state.observation`` of shape ``(8, 8, 119)`` and a 4672
action space (64 from-squares x 73 move-types = 56 queen-slides + 8 knight +
9 underpromotions). **Both are wrong for prophet.** prophet's transformer eats
``[64, 24]`` square tokens and emits a ``from*64+to`` policy with queen-only
promotions. So this module ignores ``state.observation`` and the 4672 action
ids and RE-DERIVES prophet's encoding + action map from the pgx ``State``
internals (the ``GameState`` piece bitboards / flags and the
``legal_action_mask``).

The single most important correctness property in the whole port
----------------------------------------------------------------
The semantic correctness of (a) the action map and (b) the ^56 flip parity is
the #1 thing to unit-test. This mirrors the Rust core's 31099-position
bit-identity check. A fresh test MUST assert, over a batch of random
positions, that:

  * ``legal_mask(state)``      == ``encoding.legal_move_map`` (as a 4096 bool),
  * ``encode_state(state)``    == ``encoding.encode_board`` (FEN-reconstructed),
  * ``prophet_to_pgx(state)``  steps to the same position python-chess reaches.

See ``tests/test_env_bridge.py`` (the parity harness) — until that test is
green, treat every "# VERIFY:" below as a live risk.

pgx internals this module assumes (VERIFY against installed pgx/chess.py)
------------------------------------------------------------------------
pgx is **not installed** on the box this file was written on, so the exact
private attribute names on ``State`` / ``GameState`` could not be executed.
Everything touching pgx internals is centralised in the small helper layer
at the top of this file (``_PgxBoard`` accessors) and marked "# VERIFY:" so
a single place needs fixing if a name is off. The public surface
(``encode_state``, ``legal_mask``, ``prophet_to_pgx``, ``env_step``,
``terminal_info``) does NOT change regardless of how those internals are
spelled.

Public API (matches the module plan in the spec)
------------------------------------------------
    make_chess_env() -> ChessEnv
    start_keys(master_key, B)        -> keys[B]
    env_init(keys[B])                -> State            (vmapped pgx init)
    env_step(state, prophet_action[B]) -> State          (4096->4672, no reset)
    encode_state(state)              -> x f32[B,64,24]
    legal_mask(state)                -> bool[B,4096]      (queen-promo only)
    prophet_to_pgx(state)            -> int32[B,4096]     (pgx action or -1)
    terminal_info(state)             -> (is_terminal bool[B], value f32[B])

``ChessEnv`` also caches the jitted vmapped init/step. The pgx ``State`` is the
opaque pytree threaded through ``mctx`` as the search embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import pgx
from pgx._src.games.chess import FROM_PLANE as _PGX_FROM_PLANE_TABLE

# ---------------------------------------------------------------------------
# Constants (kept local; config.py re-exports the canonical copies). These are
# duplicated here only so env.py is importable stand-alone for the parity test.
# ---------------------------------------------------------------------------

NUM_ACTIONS = 64 * 64           # prophet action space: from*64 + to   (4096)
FEATURES = 24                   # prophet encode_board feature columns
PGX_NUM_ACTIONS = 64 * 73       # pgx chess action space                (4672)
PGX_FROM_STRIDE = 73            # pgx action = from_sq*73 + move_type
DRAW_HALFMOVE_CAP = 100         # fifty-move rule (halfmove clock cap)

# python-chess / prophet square convention is LERF: square = rank*8 + file,
# A1 = 0, H1 = 7, A8 = 56, H8 = 63. The ^56 model-space flip is a *rank* flip
# (a1<->a8, files preserved). We assert pgx uses the SAME LERF convention for
# its (from_sq, to_sq) once decoded from a move_type; if pgx instead numbers
# squares differently, the single ``_pgx_sq_to_lerf`` permutation below is the
# one knob to fix (and the parity test will catch it).

# Mate sentinel for the terminal value of the *side to move*.
_MATE_VALUE = -1.0


# ===========================================================================
#  pgx State accessor layer  (THE ONLY place that touches pgx internals)
# ===========================================================================
#
# pgx/chess.py stores game logic in ``State._x`` (a ``GameState`` pytree). The
# board is kept *from the side-to-move's point of view* (pgx rotates the board
# each ply so the mover always plays "up the board"), as a flat length-64
# ``int8`` array of piece codes:
#
#     0           empty
#     +1..+6      OWN     pawn, knight, bishop, rook, queen, king
#     -1..-6      OPPONENT pawn, knight, bishop, rook, queen, king
#
# This already matches prophet's "us in cols 0..5, them in cols 6..11"
# semantics — EXCEPT for the square numbering, which pgx defines in its own
# coordinate system. We convert pgx-square -> LERF below.
#
# VERIFY (all of this section) against the installed pgx/chess.py:
#   * field name ``State._x``                         -> GameState
#   * ``GameState.board``      int8[64] piece codes    (own +, opp -)
#   * ``GameState.can_castle_queen_side`` bool[2]      ([my, opp])
#   * ``GameState.can_castle_king_side``  bool[2]
#   * ``GameState.en_passant``  int (pgx square of ep target, -1 if none)
#   * ``GameState.halfmove_count`` / ``.fifty_move_count`` int
#   * the pgx square numbering (see ``_pgx_sq_to_lerf``)
#
# If a name differs, fix it HERE; nothing downstream needs to change.


def _gamestate(state: "pgx.State"):
    """Return pgx's internal GameState pytree. VERIFY field name ``_x``."""
    # pgx public State wraps the game logic in a private GameState. On the
    # versions we target this is ``state._x``; some builds expose ``_state``.
    gs = getattr(state, "_x", None)
    if gs is None:  # VERIFY: fallback attribute name
        gs = getattr(state, "_state")
    return gs


def _pgx_board(state: "pgx.State"):
    """Side-to-move-relative int8[..., 64] piece-code board (own +, opp -).

    VERIFY: attribute ``GameState.board``. pgx keeps this already flipped to
    the mover's POV, so we do NOT re-flip for color; we only remap squares.
    """
    return _gamestate(state).board


def _pgx_can_castle_king(state: "pgx.State"):
    """bool[..., 2] = (my, opp) king-side castling rights.

    pgx GameState exposes ``castling_rights`` bool[2, 2] = [player, side].
    Last-axis side index 0 -> queen-side, 1 -> king-side.
    """
    return _gamestate(state).castling_rights[..., 1]


def _pgx_can_castle_queen(state: "pgx.State"):
    """bool[..., 2] = (my, opp) queen-side castling rights. See above."""
    return _gamestate(state).castling_rights[..., 0]


def _pgx_en_passant(state: "pgx.State"):
    """int[...] pgx-square of the en-passant target, or -1 if none. VERIFY."""
    return _gamestate(state).en_passant


def _pgx_halfmove(state: "pgx.State"):
    """int[...] fifty-move (halfmove) clock. VERIFY name.

    pgx tends to call this ``halfmove_count`` (full plies) AND keep a separate
    no-progress counter. prophet's ``halfmove_clock`` is the *no-progress*
    (reset-on-capture-or-pawn) counter, which is pgx's ``fifty_move_count`` /
    the plane-118 "no-progress" counter. We prefer the no-progress field.
    """
    gs = _gamestate(state)
    for name in ("fifty_move_count", "no_progress_count", "halfmove_count"):
        v = getattr(gs, name, None)
        if v is not None:
            return v
    # Last resort: 0 (only costs the col17 / 50-move-draw features).
    return jnp.zeros_like(state._step_count)  # VERIFY


# --- pgx square <-> LERF square -------------------------------------------
#
# THE FLIP / SQUARE-MAP CRUX. Two independent facts must be pinned by the test:
#
#  (1) pgx's per-square integer ordering vs LERF (rank*8+file, A1=0). If pgx is
#      already LERF, ``_pgx_sq_to_lerf`` is the identity. pgx/chess.py is
#      documented to use a layout where its action geometry is computed; the
#      safest, test-pinnable form is a static length-64 permutation.
#
#  (2) Whether pgx's mover-relative board needs the SAME ^56 that prophet
#      applies. prophet flips by ^56 **only when turn==BLACK** (it does NOT
#      flip for white). pgx ALREADY presents the board from the mover's POV, so
#      after mapping pgx-square -> LERF the result is in *white-relative* (i.e.
#      "as if I were white") coordinates. prophet's model space is ALSO
#      mover-relative (it XORs ^56 for black exactly to reach mover-relative).
#      => Once we are in pgx's mover-relative LERF squares, that IS prophet's
#         model space, and NO extra ^56 is applied here. The ``flipped`` bit is
#         then only relevant for converting back to *absolute* squares (which we
#         never need on the JAX side — we stay in model space throughout).
#
#  *** VERIFY THIS PARITY AGAINST A KNOWN POSITION IN A TEST. ***
#  Concretely: from the start position with BLACK to move (e.g. FEN after 1.e4),
#  prophet's encode_board puts black's e7 pawn (absolute e7=52, ^56 -> 12, i.e.
#  model-space e2) at model square 12. encode_state(state) MUST place that same
#  pawn (pgx code +1, a "my pawn") at LERF square 12. If it lands at 52 instead,
#  flip the parity by setting ``_PGX_NEEDS_EXTRA_FLIP = True`` below.

# If, after pgx-square->LERF remap, the board is NOT yet mover-relative the way
# prophet expects, set this True to apply an extra ^56. Default False because
# pgx already rotates to the mover's POV. PIN WITH THE PARITY TEST.
_PGX_NEEDS_EXTRA_FLIP = False  # VERIFY (flip parity)

# Static pgx-square -> LERF-square permutation. pgx chess numbers squares
# file-major (a1=0, a2=1, ..., h8=63); python-chess/prophet use LERF
# rank-major (a1=0, b1=1, ..., h8=63). Convert pgx square s=(file*8+rank) to
# LERF square (rank*8+file). pgx already keeps the board mover-relative, so this
# transpose is sufficient; no extra ^56 is applied.
_PGX_SQ_TO_LERF = jnp.asarray(
    [(s % 8) * 8 + (s // 8) for s in range(64)], dtype=jnp.int32
)
_LERF_TO_PGX_SQ = jnp.argsort(_PGX_SQ_TO_LERF).astype(jnp.int32)


def _pgx_sq_to_model(sq):  # pgx square (int array) -> prophet model square
    """Map a pgx square index to prophet model-space square (LERF, then the
    optional parity flip). Pure, vectorised, jit-safe."""
    lerf = _PGX_SQ_TO_LERF[sq]
    if _PGX_NEEDS_EXTRA_FLIP:
        lerf = lerf ^ 56
    return lerf


# ===========================================================================
#  ACTION GEOMETRY:  pgx 4672 action  ->  (from_sq, to_sq) in pgx squares
# ===========================================================================
#
# pgx action a = from_sq*73 + move_type, move_type in [0,73). pgx's plane
# layout is private and has changed before, so do not duplicate the geometry
# here. Import pgx's own FROM_PLANE table and use it as the source of truth for
# action -> destination. Underpromotions collapse onto the same prophet
# from-to index as queen promotions; prophet_to_pgx prefers the normal
# queen-promotion action when several pgx labels collide.

import numpy as _np  # static table construction only (host-side, at import)

def _build_pgx_action_tables() -> Tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Build static int32[4672] tables: from_sq, to_sq, valid.

    ``valid`` is False where the geometric destination falls off the board
    (those pgx action ids are never legal, so legal_action_mask is already
    False there; we still build a defined to_sq=from_sq for them).
    """
    from_sq = _np.zeros(PGX_NUM_ACTIONS, dtype=_np.int32)
    to_sq = _np.zeros(PGX_NUM_ACTIONS, dtype=_np.int32)
    valid = _np.zeros(PGX_NUM_ACTIONS, dtype=_np.bool_)

    pgx_to_sq = _np.asarray(_PGX_FROM_PLANE_TABLE, dtype=_np.int32)
    for s in range(64):
        for mt in range(73):
            a = s * PGX_FROM_STRIDE + mt
            to = int(pgx_to_sq[s, mt])
            from_sq[a] = s
            if 0 <= to < 64:
                to_sq[a] = to
                valid[a] = True
            else:
                to_sq[a] = s  # off-board: undefined dest, kept in-range
    return from_sq, to_sq, valid


_PGX_FROM_SQ_NP, _PGX_TO_SQ_NP, _PGX_VALID_NP = _build_pgx_action_tables()

# Device-resident static lookups (pgx-square coordinates).
_PGX_FROM_SQ = jnp.asarray(_PGX_FROM_SQ_NP)   # int32[4672]
_PGX_TO_SQ = jnp.asarray(_PGX_TO_SQ_NP)       # int32[4672]

# Precompute the prophet model-space (from, to) and the prophet index for every
# pgx action id, ONCE. prophet_index = from_model*64 + to_model. Off-board
# actions get index 0 but are masked out by ``valid`` / legal_action_mask.
_PGX_FROM_MODEL = _pgx_sq_to_model(_PGX_FROM_SQ)            # int32[4672]
_PGX_TO_MODEL = _pgx_sq_to_model(_PGX_TO_SQ)               # int32[4672]
_PGX_PROPHET_INDEX = (_PGX_FROM_MODEL * 64 + _PGX_TO_MODEL).astype(jnp.int32)
_PGX_ACTION_VALID = jnp.asarray(_PGX_VALID_NP)             # bool[4672]


# ===========================================================================
#  Environment wrapper
# ===========================================================================


@dataclass
class ChessEnv:
    """Thin handle around pgx's chess env + cached jitted vmapped fns.

    Attributes
    ----------
    env : pgx.Env
        The underlying ``pgx.make("chess")`` environment.
    init : callable
        ``jit(vmap(env.init))`` : keys[B] -> batched State.
    step : callable
        ``jit(vmap(env.step))`` : (State, action[B]) -> batched State.
    """

    env: Any
    init: Any
    step: Any

    @property
    def num_prophet_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def num_pgx_actions(self) -> int:
        return int(self.env.num_actions)  # 4672


# ---------------------------------------------------------------------------
# Process-default ChessEnv registry.
#
# CROSS-MODULE CONTRACT: the free functions ``env_init`` / ``env_step`` are
# called by search.py / selfplay.py / reflection.py WITHOUT a ChessEnv handle
# (``env_init(keys)`` / ``env_step(state, action)``) -- that is the interface in
# the module plan. We therefore register the most-recently-built ChessEnv as a
# process default (one env per process) and let the no-handle calls fall back to
# it, while still accepting an explicit handle as the first positional argument.
# ``make_chess_env`` registers automatically; pin a specific env with
# :func:`set_default_env`.
# ---------------------------------------------------------------------------
_DEFAULT_ENV: "ChessEnv | None" = None


def set_default_env(env: "ChessEnv") -> None:
    """Register ``env`` as the process default used by handle-less env_*calls."""
    global _DEFAULT_ENV
    _DEFAULT_ENV = env


def get_default_env() -> "ChessEnv":
    """Return the registered process-default ChessEnv (building one on first
    use if needed). One chess env per process, so a lazy default is safe."""
    global _DEFAULT_ENV
    if _DEFAULT_ENV is None:
        _DEFAULT_ENV = make_chess_env()
    return _DEFAULT_ENV


def make_chess_env() -> ChessEnv:
    """Construct the chess bridge: ``pgx.make('chess')`` plus jitted vmapped
    init/step. Call once and reuse (compilation is cached on the handle). Also
    registers the result as the process default (see :func:`set_default_env`) so
    the handle-less ``env_init(keys)`` / ``env_step(state, action)`` forms used
    by search/selfplay/reflection work without threading the handle."""
    env = pgx.make("chess")
    # Sanity (cheap, host-side): pgx chess must be the 4672 / (8,8,119) env.
    # We do NOT hard-fail if a future pgx changes num_actions, but the action
    # tables assume 4672; mismatch is a loud error rather than silent wrong.
    n = int(env.num_actions)
    if n != PGX_NUM_ACTIONS:  # VERIFY: pgx chess action count
        raise AssertionError(
            f"pgx chess num_actions={n}, expected {PGX_NUM_ACTIONS}; the "
            "action-geometry tables in prophet_jax/env.py assume 64*73."
        )
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    handle = ChessEnv(env=env, init=init, step=step)
    set_default_env(handle)
    return handle


# Eagerly build the default env at IMPORT so pgx's *lazy* chess-module import
# (and its module-level board constants, e.g. the int32[64] start board) happen
# OUTSIDE any jit. Otherwise the first env_init() inside a jitted self-play/
# search traces those constants, caches them, and they leak across the next
# differently-shaped call (UnexpectedTracerError). Cheap (~one pgx.make).
try:
    make_chess_env()
except Exception:  # pragma: no cover - pgx unavailable on some dev boxes
    pass


def start_keys(master_key: jax.Array, B: int) -> jax.Array:
    """Split ``master_key`` into ``B`` per-game PRNG keys for ``env_init``.

    pgx's ``env.init`` consumes its key (it picks the starting player), so each
    parallel game needs its own key.
    """
    return jax.random.split(master_key, B)


def env_init(env_or_keys, keys: jax.Array | None = None) -> "pgx.State":
    """Vmapped pgx init over a batch of keys -> batched ``State`` (leading B).

    Polymorphic on arity so both the module-plan signature and the explicit
    handle form work:
      * ``env_init(keys)``        -- uses the process-default ChessEnv
        (registered by :func:`make_chess_env`). This is the form search /
        selfplay / reflection use.
      * ``env_init(env, keys)``   -- uses the explicit ``ChessEnv`` handle.
    """
    if keys is None:  # env_init(keys): first arg is the key batch
        env = get_default_env()
        keys = env_or_keys
    else:  # env_init(env, keys): first arg is the handle
        env = env_or_keys
    return env.init(keys)


def env_step(a, b, c=None) -> "pgx.State":
    """Apply one prophet move per game and return the next batched ``State``.

    Polymorphic on arity:
      * ``env_step(state, prophet_action)``      -- process-default ChessEnv
        (the form search / selfplay / reflection use).
      * ``env_step(env, state, prophet_action)`` -- explicit ``ChessEnv`` handle.

    ``prophet_action`` is int32[B] in prophet's ``from*64+to`` model space. We
    translate each to the pgx 4672 action via the per-state ``prophet_to_pgx``
    table, then call pgx's (deterministic) ``step``. **No auto-reset** — pgx
    never auto-resets and we don't either; the caller loops on
    ``terminal_info`` / ``state.terminated | state.truncated``.

    Stepping a terminated game is a pgx no-op (rewards stay 0, state frozen),
    so it is safe to keep stepping finished games in a batch.
    """
    if c is None:  # env_step(state, action)
        env = get_default_env()
        state, prophet_action = a, b
    else:  # env_step(env, state, action)
        env, state, prophet_action = a, b, c
    p2pgx = prophet_to_pgx(state)                       # int32[B,4096]
    pgx_action = jnp.take_along_axis(
        p2pgx, prophet_action.astype(jnp.int32)[:, None], axis=1
    )[:, 0]                                             # int32[B]
    # Illegal / unmapped prophet actions yield -1; clamp to 0 so pgx.step gets a
    # valid index. Callers must only ever pass *legal* prophet actions (search
    # masks with legal_mask), so this clamp is a guard, not a code path.
    pgx_action = jnp.where(pgx_action < 0, 0, pgx_action).astype(jnp.int32)
    return env.step(state, pgx_action)                 # chess is deterministic


# ===========================================================================
#  ACTION MAPPING  (the crux):  prophet 4096  <->  pgx 4672  per state
# ===========================================================================


def _prophet_maps_single(legal_action_mask: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """For ONE state's pgx ``legal_action_mask`` (bool[4672]) produce:

        prophet_legal : bool[4096]   (queen-promo only; underpromos collapsed)
        prophet_to_pgx: int32[4096]  (a pgx action to step for each prophet idx,
                                       -1 where illegal)

    Method (pure, jit/vmap-safe — no data-dependent shapes):
      * Every legal pgx action id ``a`` already knows its prophet index
        ``_PGX_PROPHET_INDEX[a]`` and validity ``_PGX_ACTION_VALID[a]``
        (precomputed statically). A legal action contributes its prophet index.
      * Scatter ``True`` into prophet_legal at those indices (collapsing the 9
        underpromotion ids onto the single queen-promo prophet index, since they
        share the same (from_model, to_model) destination).
      * Scatter the pgx action id into prophet_to_pgx at those indices. When
        several pgx ids map to the same prophet index (queen + underpromotions),
        we deterministically keep the QUEEN one. Queen-promo pgx ids live in the
        56-ray block (move_type<56), underpromotions in 64..72; taking the
        MINIMUM pgx id among colliders selects the ray/queen action (mt<56 <
        64..72), which is the move prophet's ``index_to_move`` plays
        (promotion=QUEEN). For non-promotion squares there is no collision.
    """
    legal = legal_action_mask & _PGX_ACTION_VALID            # bool[4672]
    idx = _PGX_PROPHET_INDEX                                 # int32[4672]

    # prophet_legal[4096]: OR of legality over all pgx ids sharing each index.
    prophet_legal = (
        jnp.zeros((NUM_ACTIONS,), dtype=jnp.int32)
        .at[idx]
        .max(legal.astype(jnp.int32))            # max == logical-or for 0/1
        .astype(jnp.bool_)
    )

    # prophet_to_pgx[4096]: among the LEGAL pgx ids mapping to a given prophet
    # index, keep a deterministic pgx id. If a queen promotion and one or more
    # underpromotions collide on the same from-to index, prefer the normal
    # queen-promotion label. In pgx, underpromotion planes are 0..8 and normal
    # moves are 9..72, so a raw min would incorrectly choose underpromotion.
    BIG = PGX_NUM_ACTIONS * 3
    pgx_ids = jnp.arange(PGX_NUM_ACTIONS, dtype=jnp.int32)
    is_underpromo = (pgx_ids % PGX_FROM_STRIDE) < 9
    ranked_ids = pgx_ids + is_underpromo.astype(jnp.int32) * PGX_NUM_ACTIONS
    candidate = jnp.where(legal, ranked_ids, BIG)            # int32[4672]
    best = (
        jnp.full((NUM_ACTIONS,), BIG, dtype=jnp.int32)
        .at[idx]
        .min(candidate)                                       # int32[4096]
    )
    prophet_to_pgx_ = jnp.where(best >= BIG, -1, best % PGX_NUM_ACTIONS).astype(jnp.int32)
    return prophet_legal, prophet_to_pgx_


def legal_mask(state: "pgx.State") -> jax.Array:
    """prophet legality as bool[B,4096] (queen-promotion only).

    Collapses pgx's 9 underpromotion actions onto the single queen-promo
    prophet index, exactly as ``encoding.legal_move_map`` drops non-queen
    promotions. Derived purely from ``state.legal_action_mask`` — independent
    of the encoding, so the parity test cross-checks the two.
    """
    masks = jax.vmap(lambda m: _prophet_maps_single(m)[0])(state.legal_action_mask)
    return masks  # bool[B,4096]


def prophet_to_pgx(state: "pgx.State") -> jax.Array:
    """int32[B,4096]: for each prophet action index, the pgx 4672 action to
    actually step (queen-ray preferred over underpromotions), or -1 if that
    prophet action is illegal in this state."""
    return jax.vmap(lambda m: _prophet_maps_single(m)[1])(state.legal_action_mask)


# ===========================================================================
#  ENCODING:  pgx GameState  ->  prophet x[B,64,24]   (EXACT encode_board)
# ===========================================================================


def _encode_single(state_slice: "pgx.State") -> jax.Array:
    """Reproduce ``encoding.encode_board`` for ONE (unbatched) pgx state.

    Returns x[64, 24] float32 in prophet model space (token t == model-space
    square t; channels-last, already token-major -> NO transpose). This reads
    pgx's piece bitboards/flags, NOT the 119-plane observation.

    Column layout (must be byte-identical to prophet/encoding.py):
      0..5   us pieces   (pawn,knight,bishop,rook,queen,king) at sq^flip
      6..11  opp pieces  (same order)
      12     en-passant target flag
      13     us  king-side  castling right   (broadcast scalar)
      14     us  queen-side castling right   (broadcast scalar)
      15     opp king-side  castling right   (broadcast scalar)
      16     opp queen-side castling right   (broadcast scalar)
      17     min(halfmove,100)/100           (broadcast scalar)
      18     last move from-square flag      (0 here: history not reconstructed)
      19     last move to-square   flag      (0 here)
      20     2nd-last move from-square flag  (0 here)
      21     2nd-last move to-square   flag  (0 here)
      22     is_repetition(2) flag           (broadcast; computed when hm>=4)
      23     len(move_stack) % 2             (broadcast == step_count % 2)
    """
    x = jnp.zeros((64, FEATURES), dtype=jnp.float32)

    # --- piece planes (cols 0..11) -----------------------------------------
    # pgx board: int8[64] in pgx-square order; +1..+6 own P..K, -1..-6 opp.
    # Already mover-relative for color (own vs opp), so cols 0..5 = positive
    # codes, cols 6..11 = negative codes. Remap pgx square -> model square.
    board = _pgx_board(state_slice)                       # int8[64], pgx squares
    model_sq = _pgx_sq_to_model(jnp.arange(64, dtype=jnp.int32))  # int32[64]

    # For each pgx square s (carrying a piece code c), set
    #   x[model_sq[s], (0 if c>0 else 6) + (|c|-1)] = 1
    # Build via scatter. |c|-1 in [0,5] is the piece-type plane; base 0/6 by sign.
    code = board.astype(jnp.int32)                       # pgx-square order
    has_piece = code != 0
    ptype = jnp.abs(code) - 1                             # 0..5 (garbage where empty)
    base = jnp.where(code > 0, 0, 6)                      # us=0, opp=6
    plane = jnp.clip(base + ptype, 0, FEATURES - 1)       # col index 0..11
    rows = model_sq                                       # destination model square
    # Only scatter where a piece exists: multiply the set-value by has_piece and
    # use .add into a fresh array (each (row, plane) hit at most once).
    vals = has_piece.astype(jnp.float32)
    x = x.at[rows, plane].add(vals)
    # Guard against the empty-square garbage plane writing 0 anyway (vals=0 there
    # so the add is a no-op even though `plane` is junk — safe).

    # --- en passant (col 12) ------------------------------------------------
    ep = _pgx_en_passant(state_slice).astype(jnp.int32)   # pgx square or -1
    ep_model = _pgx_sq_to_model(jnp.clip(ep, 0, 63))
    ep_present = ep >= 0
    x = x.at[ep_model, 12].add(ep_present.astype(jnp.float32))

    # --- castling rights (cols 13..16, broadcast) ---------------------------
    ck = _pgx_can_castle_king(state_slice)                # bool[2] (my, opp)
    cq = _pgx_can_castle_queen(state_slice)               # bool[2] (my, opp)
    x = x.at[:, 13].set(ck[0].astype(jnp.float32))        # us  king-side
    x = x.at[:, 14].set(cq[0].astype(jnp.float32))        # us  queen-side
    x = x.at[:, 15].set(ck[1].astype(jnp.float32))        # opp king-side
    x = x.at[:, 16].set(cq[1].astype(jnp.float32))        # opp queen-side

    # --- halfmove clock (col 17, broadcast) ---------------------------------
    hm = _pgx_halfmove(state_slice).astype(jnp.float32)
    x = x.at[:, 17].set(jnp.minimum(hm, 100.0) / 100.0)

    # --- history (cols 18..21) ---------------------------------------------
    # DOCUMENTED FALLBACK: pgx's public State does not cheaply expose the last
    # two *moves* (only an 8-step *board* history of planes). prophet zeroes
    # these columns for any position lacking a move_stack (bare-FEN / study
    # branches), and the project explicitly tolerates that as a known accuracy
    # cost, not a transfer breaker. We therefore leave 18..21 = 0.
    #
    # If a future port reconstructs the last move from pgx's two most recent
    # board-history planes (diff of consecutive own/opp piece planes -> the
    # from/to of the move that produced the current board), populate cols 18/19
    # (and 20/21 from the prior pair) here, remapping squares with
    # ``_pgx_sq_to_model``. Leaving zero is the safe, spec-sanctioned default.
    # (cols 18..21 already zero)

    # --- repetition (col 22, broadcast; only when halfmove>=4) -------------
    # prophet uses board.is_repetition(2) (this exact position seen >=2 times).
    # pgx tracks repetition via its 2 repetition planes per history step; the
    # public, jit-clean signal is whether the current position's repetition
    # count >= 2. VERIFY the exact field; we look for a hashed-history count.
    rep2 = _is_repetition_ge2(state_slice)                # bool scalar
    hm_int = _pgx_halfmove(state_slice).astype(jnp.int32)
    rep_flag = jnp.where(hm_int >= 4, rep2.astype(jnp.float32), 0.0)
    x = x.at[:, 22].set(rep_flag)

    # --- side parity (col 23, broadcast) -----------------------------------
    # prophet: float(len(move_stack) % 2) == number of plies played, mod 2,
    #          == state._step_count % 2 (start position has 0 -> 0.0).
    parity = (state_slice._step_count.astype(jnp.int32) % 2).astype(jnp.float32)
    x = x.at[:, 23].set(parity)

    return x


def _is_repetition_ge2(state_slice: "pgx.State") -> jax.Array:
    """Whether the current position has occurred at least twice (== prophet's
    ``board.is_repetition(2)``). Returns a bool scalar.

    VERIFY: pgx stores repetition either as explicit per-step repetition planes
    in the observation history or as a hashed position-count in GameState. The
    cleanest internal source is a per-position occurrence count; we look for it
    and threshold at >= 2 (current occurrence inclusive). If pgx instead only
    exposes the 2 boolean repetition planes (rep>=2 and rep>=3 of the *latest*
    board), the first plane is exactly this signal.
    """
    gs = _gamestate(state_slice)
    hist = getattr(gs, "hash_history", None)
    if hist is not None:
        # pgx keeps the current position hash at history[0]. prophet's
        # board.is_repetition(2) asks whether the current position has occurred
        # at least twice, counting the current occurrence.
        h = hist[0]
        count = (hist == h).all(axis=-1).sum()
        nonzero = ~(h == 0).all()
        return nonzero & (count >= 2)

    # Alternative: an integer "how many times has this position occurred" field.
    for name in ("hash_history_count", "repetition_count", "position_count"):
        v = getattr(gs, name, None)
        if v is not None:
            return v.astype(jnp.int32) >= 2
    # Fallback: derive from the observation's repetition planes if present.
    # pgx obs layout: per history step, planes [12]=rep==0 and [13]=rep>=1.
    # The latest step's second repetition plane encodes "seen before".
    obs = getattr(state_slice, "observation", None)
    if obs is not None:  # obs (8,8,119): plane 12 of the latest step = rep flag
        return jnp.any(obs[..., 13] > 0)
    # Last resort: no repetition info -> 0 (only costs the col22 feature when a
    # genuine 2-fold occurs with hm>=4; the threefold *draw* is still handled by
    # terminal_info via pgx.terminated). VERIFY.
    return jnp.bool_(False)


def encode_state(state: "pgx.State") -> jax.Array:
    """Batched prophet encoding: ``State`` -> x f32[B, 64, 24].

    Channels-last and token-major already (64 tokens x 24 features) — the model
    consumes ``[B, 64, F]`` directly, NO transpose. Reproduces
    ``encoding.encode_board`` from pgx GameState bitboards/flags (NOT the
    119-plane observation), in prophet model space (^56-flip parity folded into
    ``_pgx_sq_to_model``).
    """
    return jax.vmap(_encode_single)(state)


# ===========================================================================
#  TERMINAL VALUE  (side-to-move perspective)
# ===========================================================================


def terminal_info(state: "pgx.State") -> Tuple[jax.Array, jax.Array]:
    """Return ``(is_terminal: bool[B], value: f32[B])`` for the side to move.

    Semantics (matching ``search._terminal_value`` / ``board.terminal_value``):
      * checkmate (side to move is mated)            -> value = -1.0
      * stalemate / insufficient material / 50-move /
        threefold repetition (any draw)              -> value =  0.0
      * not terminal                                 -> value =  0.0  (sentinel;
        read ``is_terminal`` to know it's meaningless — JAX has no None, so the
        caller MUST gate on ``is_terminal`` rather than testing value==0.)

    Derivation: pgx ``rewards`` are indexed by **player-id** (0/1), and are the
    GAME reward (mate = +1 for the winner, -1 for the loser) emitted ONLY on the
    transition that sets ``terminated``. The side-to-move value is the reward of
    the *current* player:
        v_side_to_move = rewards[current_player]
    On checkmate the side to move is the one that just got mated, so its reward
    is -1 -> value -1. On any draw all rewards are 0 -> value 0. Truncation
    (512-step cap) is treated as a non-terminal sentinel here (it is not a chess
    result); callers that want to *stop* a game should use
    ``state.terminated | state.truncated`` for the loop guard but only feed
    ``terminal_info``'s value when ``is_terminal`` (== terminated) is True.

    NOTE on pgx reward sign/scale: pgx chess rewards are in {-1,0,+1}. If a
    given pgx build emits the reward as "reward to the player who just moved"
    rather than "to current_player", the parity test (mate position) catches it
    and the fix is to index by the mover instead. VERIFY against a known mate.
    """
    cur = state.current_player.astype(jnp.int32)          # int8 -> int32, [B]
    rewards = state.rewards                                # f32[B, 2]
    v = jnp.take_along_axis(rewards, cur[:, None], axis=1)[:, 0]  # f32[B]
    is_terminal = state.terminated                        # bool[B]
    # Force the value to exactly 0 when not terminal so callers that ignore the
    # gate at least see a benign 0 rather than a stray reward.
    value = jnp.where(is_terminal, v, 0.0).astype(jnp.float32)
    return is_terminal, value


def mover_white(state: "pgx.State") -> jax.Array:
    """bool[B]: whether the chess side to move is White.

    pgx separates chess color (``state._x.color``: 0 white, 1 black) from
    player id (``state.current_player``), and ``env.init`` may randomize the
    player-id order. Training/outcome code needs chess color, not player id.
    """
    return _gamestate(state).color.astype(jnp.int32) == 0


# ===========================================================================
#  Convenience: bundle the per-state derived tensors (one vmapped pass)
# ===========================================================================


def derive_all(state: "pgx.State"):
    """One-shot helper returning everything the search/selfplay layer needs:

        x            : f32[B, 64, 24]   prophet encoding
        legal        : bool[B, 4096]    prophet legal mask (queen-promo only)
        to_pgx       : int32[B, 4096]   prophet index -> pgx action (or -1)
        is_terminal  : bool[B]
        term_value   : f32[B]           side-to-move terminal value

    Kept as a single function so a caller can ``jax.jit`` the whole bridge step
    and reuse the legal/to_pgx computation between masking and stepping.
    """
    x = encode_state(state)
    legals, to_pgxs = jax.vmap(_prophet_maps_single)(state.legal_action_mask)
    is_terminal, term_value = terminal_info(state)
    return x, legals, to_pgxs, is_terminal, term_value


__all__ = [
    "ChessEnv",
    "make_chess_env",
    "set_default_env",
    "get_default_env",
    "start_keys",
    "env_init",
    "env_step",
    "encode_state",
    "legal_mask",
    "prophet_to_pgx",
    "terminal_info",
    "mover_white",
    "derive_all",
    "NUM_ACTIONS",
    "FEATURES",
    "PGX_NUM_ACTIONS",
    "DRAW_HALFMOVE_CAP",
]
