"""Vectorized, fully on-device self-play for the JAX port of prophet.

The PyTorch engine plays one game per Python generator and batches network
evaluations *across* concurrently-running games (``worker.run_vector_selfplay``).
In JAX we flip the structure inside out: every one of ``B`` games advances in
**lockstep** under ``vmap``, and the whole rollout is a single ``jax.lax.scan``
over plies that is JIT-compiled end to end. "Thousands of games" is just a
bigger ``B``; there is no Python-level game loop at all.

What one scan step does (all batched over ``B``):
  1. ``batched_search(params, key, states, scfg)`` -> per game:
       move_index[B], policy_target[B,4096] (sums to 1 over legal),
       root_value[B] (pre-outcome), q_target[B,4096] / q_weight[B,4096]
       (empirical root-perspective Q and visit counts, scattered DENSE),
       q_head_played[B] (the Q-head value of the played move).
  2. Record the parent features ``x = encode_state(states)``, the legal mask,
     the played action, the (pre-outcome) root value, and the dense policy/Q
     targets into a fixed-size ``[B, max_plies, ...]`` buffer, written by ply
     index.
  3. ``env_step`` to the child position and record ``child_x`` (for the
     negamax consistency loss Q(s,a) ~= -V(s')).
  4. Freeze finished games: stepping a terminated pgx ``State`` is a no-op, and
     the sample written for an already-finished game is masked out via
     ``valid``.

Resignation (gated): per game we track consecutive plies where the *mover's*
own root value fell below ``resign_threshold``; once that streak reaches
``resign_plies`` and the per-game ``resign_active`` flag is set, the game is
adjudicated (done := True, with the opponent recorded as the winner). The
master ``gate`` bool is pure orchestration (the learner opens it once the value
head matures); it is passed in, not decided here -- exactly mirroring
``worker.episode_gen`` / ``selfplay.SelfPlayConfig``.

After the scan we apply the **outcome blend** identically to
``selfplay.play_game_gen``: derive ``z_white`` per game from the terminal
state (checkmate -> the side to move at the terminal node lost; draw -> 0;
truncated / unfinished -> NaN), then for every recorded ply with a known
outcome::

    z         = z_white if mover_was_white else -z_white
    wdl       = int(z) + 1            # 0/1/2  (-1 when z unknown)
    z_eff     = -contempt if z == 0 else z * win_discount ** (total - t)
    value     = (1 - outcome_mix) * root_value + outcome_mix * z_eff

all vectorized with ``jnp.where`` over a ``[B, T]`` grid.

The flat ``SamplesBatch`` it returns is the *training contract*: dense,
fixed-shape arrays of ``N = B * max_plies`` rows with a ``valid`` mask marking
real samples versus post-terminal padding. ``GameMeta`` additionally carries
the per-ply pgx ``State`` snapshots so ``reflection.py`` can re-search from any
ply *without* python-chess FENs -- the pgx State *is* the reconstructable
position.

Public API::

    generate_selfplay(params, key, B, scfg, spcfg, gate) -> (SamplesBatch, GameMeta)

Real third-party APIs only: pgx (``env.init`` / ``env.step`` / ``State``) and
the project's own ``env``/``search`` JAX modules. Anything genuinely uncertain
about the pgx/mctx surface is implemented in its most-likely-correct form and
flagged with a ``# VERIFY:`` comment.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

# ---------------------------------------------------------------------------
# Sibling-module imports.
#
# These JAX modules (config/env/search) are written against the shared
# interface contract in the module plan. They are imported lazily-tolerantly so
# this file still *parses* during incremental development even if a sibling is
# not present yet; at run time they must exist. We deliberately do NOT
# re-implement encoding/search here -- selfplay only orchestrates them.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only once the package is complete
    from .config import FEATURES, NUM_ACTIONS, SearchConfig, SelfPlayConfig
except Exception:  # pragma: no cover
    FEATURES = 24
    NUM_ACTIONS = 4096

    @struct.dataclass
    class SearchConfig:  # minimal stand-in matching config.SearchConfig
        sims: int = 32
        root_candidates: int = 8
        c_puct: float = 1.5
        c_visit: float = 50.0
        c_scale: float = 1.0
        q_trust: float = 1.0

    @struct.dataclass
    class SelfPlayConfig:  # minimal stand-in matching config.SelfPlayConfig
        max_plies: int = 200
        outcome_mix: float = 0.5
        resign_threshold: float = -0.92
        resign_plies: int = 8
        resign_off_prob: float = 0.1
        contempt: float = 0.15
        win_discount: float = 0.997


try:  # pragma: no cover
    from . import env as env_mod
except Exception:  # pragma: no cover
    env_mod = None

try:  # pragma: no cover
    from .search import batched_search
except Exception:  # pragma: no cover
    batched_search = None


# ---------------------------------------------------------------------------
# Output pytrees (the training + reflection contract).
#
# Defined HERE because no sibling owns them; train.py / reflection.py import
# SamplesBatch / GameMeta from this module. Both are flax.struct dataclasses so
# they are valid JAX pytrees (jit args/returns, tree_map, device put, ...).
# ---------------------------------------------------------------------------
@struct.dataclass
class SamplesBatch:
    """Dense, fixed-shape training rows. ``N = B * max_plies``.

    Every field is a leading-``N`` array; rows where ``valid`` is False are
    padding for plies that occurred after a game finished (or never happened)
    and must be ignored by the loss (the loss multiplies per-sample terms by
    ``weight``; the trainer additionally masks/zeros via ``valid``).

    Indexing of the 4096 action axis is prophet's ``from * 64 + to`` (queen
    promotions only) -- identical to ``encoding.move_to_index`` and the model's
    policy/Q heads.
    """

    x: jnp.ndarray            # f32 [N, 64, FEATURES]  parent position
    child_x: jnp.ndarray      # f32 [N, 64, FEATURES]  position after played move
    played: jnp.ndarray       # i32 [N]                played action (from*64+to)
    value: jnp.ndarray        # f32 [N]                blended value target
    weight: jnp.ndarray       # f32 [N]                loss weight (1.0 for self-play)
    wdl: jnp.ndarray          # i32 [N]                terminal class 0/1/2, -1 unknown
    mask: jnp.ndarray         # bool [N, NUM_ACTIONS]  legal-move mask
    policy: jnp.ndarray       # f32 [N, NUM_ACTIONS]   improved policy target (sums to 1)
    q_target: jnp.ndarray     # f32 [N, NUM_ACTIONS]   empirical root-perspective Q (dense)
    q_weight: jnp.ndarray     # f32 [N, NUM_ACTIONS]   per-move visit counts (dense)
    valid: jnp.ndarray        # bool [N]               real sample vs padding


@struct.dataclass
class GameMeta:
    """Per-game / per-ply metadata for logging and for reflection re-search.

    ``states_per_ply`` is a batched pgx ``State`` with a leading ``[B, T]``
    (game, ply) shape -- the snapshot of the position *before* each ply. It is
    the JAX replacement for python-chess FENs: reflection reconstructs a
    position simply by slicing out ``states_per_ply[g, t]`` and re-running
    search from it, so no FEN parsing is needed (study.py's ``board_from_fen``
    becomes a pure array slice).
    """

    root_values: jnp.ndarray    # f32  [B, T]   search root value per ply (pre-outcome)
    q_head_played: jnp.ndarray  # f32  [B, T]   Q-head value of the played move per ply
    mover_white: jnp.ndarray    # bool [B, T]   was the side to move White at this ply
    z_white: jnp.ndarray        # f32  [B]      final outcome for White (NaN if truncated)
    plies: jnp.ndarray          # i32  [B]      number of real plies played
    result: jnp.ndarray         # i32  [B]      0='0-1', 1='1/2-1/2', 2='1-0', -1='*'
    valid_ply: jnp.ndarray      # bool [B, T]   which (game, ply) cells are real samples
    states_per_ply: Any = struct.field(pytree_node=True, default=None)  # pgx State [B, T]


# Result-code constants for GameMeta.result (compact, JAX-friendly).
RESULT_BLACK_WIN = 0   # "0-1"
RESULT_DRAW = 1        # "1/2-1/2"
RESULT_WHITE_WIN = 2   # "1-0"
RESULT_UNFINISHED = -1  # "*"  (truncated without a terminal)


# ---------------------------------------------------------------------------
# Scan carry.
# ---------------------------------------------------------------------------
@struct.dataclass
class _Carry:
    states: Any            # pgx State, batched over B (the live positions)
    history: jnp.ndarray   # i32 [B, 4] last-two-move squares in current model space
    key: jnp.ndarray       # [2] PRNG key, split each step
    done: jnp.ndarray      # bool [B]  game finished (terminal OR resigned)
    ply: jnp.ndarray       # i32 scalar  current ply index (0..max_plies-1)
    # resignation bookkeeping (per game), keyed by mover colour like the torch
    # ``low_streak`` dict {white_mover: streak, black_mover: streak}:
    streak_white: jnp.ndarray   # i32 [B]  consecutive low-value plies, White to move
    streak_black: jnp.ndarray   # i32 [B]  consecutive low-value plies, Black to move
    resign_active: jnp.ndarray  # bool [B] this game obeys resignation (vs exempt)
    resigned_white_won: jnp.ndarray  # i8 [B]  1 White won by resign, 0 Black, -1 none
    z_white: jnp.ndarray   # f32 [B] final outcome captured at termination, NaN if unknown
    plies: jnp.ndarray     # i32 [B]  count of real plies recorded so far


def _lite_state(state: Any) -> Any:
    """Strip the fat, reflection-unused ``observation`` ([B,8,8,119] ~= 30KB/state)
    before snapshotting per-ply. Reflection re-searches from ``_x`` +
    ``legal_action_mask`` (env_step recomputes a fresh observation), so the stored
    observation is never read. Cuts states_per_ply memory ~4x (the OOM-at-B1024
    cause). Keeps the leading [B] axis -> stacks to [B, T, 1]."""
    B = jnp.asarray(state.observation).shape[0]
    return state.replace(
        observation=jnp.zeros((B, 1), dtype=state.observation.dtype)
    )


def _zeros_like_state_stack(state: Any, max_plies: int) -> Any:
    """Pre-allocate a ``[B, max_plies, ...]`` snapshot buffer shaped like the
    batched pgx ``State`` (a leading ``max_plies`` axis inserted after ``B``).

    pgx States are registered pytrees, so ``tree_map`` over a single-step state
    gives us a correctly-typed empty buffer to scatter per-ply snapshots into.
    """
    def _alloc(leaf):
        leaf = jnp.asarray(leaf)
        # leaf is [B, ...]; we want [B, max_plies, ...].
        return jnp.zeros((leaf.shape[0], max_plies) + leaf.shape[1:], dtype=leaf.dtype)

    return jax.tree_util.tree_map(_alloc, state)


def _write_state_snapshot(buffer: Any, state: Any, ply: jnp.ndarray) -> Any:
    """Write the current batched ``state`` into ``buffer[:, ply]`` for every
    pytree leaf (dynamic index update along the ply axis)."""
    def _put(buf_leaf, st_leaf):
        st_leaf = jnp.asarray(st_leaf)
        # buf_leaf: [B, T, ...]; st_leaf: [B, ...]. Set buf_leaf[:, ply] = st_leaf.
        return buf_leaf.at[:, ply].set(st_leaf)

    return jax.tree_util.tree_map(_put, buffer, state)


def _mover_is_white(states: Any) -> jnp.ndarray:
    """Boolean [B]: is the side to move at ``states`` White?

    The prophet convention is "White == the player who moved first from the
    initial position". pgx exposes ``current_player`` (0/1, the player-id to
    move) and ``_step_count``; with chess starting from the standard position
    and player 0 moving first, side-to-move-is-White == (step_count is even).
    We defer to ``env.mover_white`` when the env module provides it (it owns the
    pgx<->prophet colour mapping), and fall back to the step-count parity.
    """
    if env_mod is not None and hasattr(env_mod, "mover_white"):
        return env_mod.mover_white(states)  # type: ignore[attr-defined]
    # VERIFY: pgx State exposes a private ``_step_count`` int32; parity gives
    # side-to-move colour when the game starts from the standard position with
    # player 0 = White to move. If env.py defines a canonical helper, that path
    # above is preferred.
    step = jnp.asarray(states._step_count).astype(jnp.int32)  # type: ignore[attr-defined]
    return (step % 2) == 0


def _terminal_z_white(states: Any, mover_white: jnp.ndarray) -> jnp.ndarray:
    """Per-game ``z_white`` in {+1, -1, 0, NaN} from a (possibly terminal) state.

    Matches ``selfplay.play_game_gen``: checkmate -> the side to move at the
    terminal node LOST, so White's score is -1 if White is to move else +1;
    any other terminal (stalemate / insufficient material / 50-move /
    repetition) -> draw 0; not terminated (i.e. truncated or still running)
    -> NaN (unknown outcome). ``env.terminal_info`` returns the side-to-move
    value (mate -> -1, draw -> 0) and we negate/sign it into White's frame.
    """
    is_term, stm_value = env_mod.terminal_info(states)  # type: ignore[union-attr]
    # stm_value is from the side-to-move perspective: -1 = side to move lost,
    # 0 = draw. Convert to White's frame: z_white = stm_value if mover White
    # else -stm_value. (mate: stm_value=-1 -> White=-1 when White to move.)
    z_white = jnp.where(mover_white, stm_value, -stm_value).astype(jnp.float32)
    # A self-play ply cap is not a chess result. Keep cap-truncated games as
    # unknown outcomes: their policy/Q/search-value rows still enter replay, but
    # outcome blending and WDL supervision are skipped, matching prophet/selfplay.py.
    return jnp.where(is_term, z_white, jnp.nan)


def _scan_step(static, carry: _Carry, _):
    """One ply for all ``B`` games in lockstep. ``static`` bundles the python/
    config values closed over by the jitted scan (search cfg, self-play cfg,
    params)."""
    params, scfg, spcfg = static

    key, search_key = jax.random.split(carry.key)

    # --- 1) batched search on the live positions -------------------------
    # search returns dense per-game targets (see search.SearchOut).
    out = batched_search(params, search_key, carry.states, scfg, carry.history)

    move_index = out.move_index.astype(jnp.int32)     # [B]
    policy_target = out.policy_target.astype(jnp.float32)  # [B, A] sums to 1 over legal
    root_value = out.root_value.astype(jnp.float32)   # [B]
    q_target = out.q_target.astype(jnp.float32)       # [B, A] dense
    q_weight = out.q_weight.astype(jnp.float32)       # [B, A] dense visit counts
    q_head_played = out.q_head_played.astype(jnp.float32)  # [B]

    # --- 2) record parent-side features / targets ------------------------
    x = env_mod.encode_state(carry.states, carry.history).astype(jnp.float32)   # [B, 64, F]
    mask = env_mod.legal_mask(carry.states)                      # bool [B, A]
    mover_white = _mover_is_white(carry.states)                 # bool [B]

    # Active games: those NOT already finished. Stepping / recording for a
    # finished game is a no-op (pgx step on a terminated state is a no-op) and
    # the written row is masked out below.
    active = ~carry.done                                         # bool [B]

    # --- 3) step to the child position -----------------------------------
    # For finished games this is a no-op (terminated pgx state). We still call
    # it uniformly so the scan stays branch-free; the snapshot/sample is masked.
    child_states = env_mod.env_step(carry.states, move_index)
    raw_child_history = env_mod.update_history(carry.history, move_index)
    child_history = jnp.where(active[:, None], raw_child_history, carry.history)
    child_x = env_mod.encode_state(child_states, child_history).astype(jnp.float32)  # [B, 64, F]

    # --- 4) terminal detection on the child ------------------------------
    child_is_term, _ = env_mod.terminal_info(child_states)       # bool [B]
    # A game also ends if the child got truncated (pgx 512-step cap). pgx marks
    # this on ``truncated``; fold it in so the scan can stop.
    child_truncated = _state_truncated(child_states)             # bool [B]

    # --- 5) resignation bookkeeping (per mover colour) -------------------
    # Mirror torch ``low_streak`` keyed by mover: only the colour that just
    # moved updates; increment on a low root value, else reset to 0.
    low = root_value < spcfg.resign_threshold                    # bool [B]
    moved_white = active & mover_white
    moved_black = active & (~mover_white)
    inc_white = jnp.where(low, carry.streak_white + 1, 0)
    inc_black = jnp.where(low, carry.streak_black + 1, 0)
    streak_white = jnp.where(moved_white, inc_white, carry.streak_white)
    streak_black = jnp.where(moved_black, inc_black, carry.streak_black)

    mover_streak = jnp.where(mover_white, streak_white, streak_black)  # [B]
    resign_now = (
        active
        & carry.resign_active
        & (mover_streak >= spcfg.resign_plies)
    )
    # mover resigns -> opponent wins. resigned_white_won = 1 if White wins.
    resign_white_won = jnp.where(mover_white, 0, 1).astype(jnp.int8)   # opp of mover
    resigned_white_won = jnp.where(
        resign_now & (carry.resigned_white_won < 0),
        resign_white_won,
        carry.resigned_white_won,
    )

    # Capture the outcome on the exact transition that ends the game. pgx
    # rewards are transition-local; after later padding/no-op steps on an
    # already-finished game they can read as zero, which turns mates into draws
    # if we recompute from the final fixed-scan state.
    child_mover_white = _mover_is_white(child_states)
    z_terminal = _terminal_z_white(child_states, child_mover_white)
    z_resign = jnp.where(resign_white_won == 1, 1.0, -1.0).astype(jnp.float32)
    z_new = jnp.where(
        resign_now,
        z_resign,
        jnp.where(child_is_term, z_terminal, carry.z_white),
    )
    z_white = jnp.where(active & jnp.isnan(carry.z_white), z_new, carry.z_white)

    # --- 6) bookkeeping: counts + done -----------------------------------
    plies = carry.plies + active.astype(jnp.int32)               # count real plies
    new_done = carry.done | (active & (child_is_term | child_truncated | resign_now))

    # --- 7) emit this ply's per-game record (scanned, stacked over T) ----
    valid_ply = active  # rows that correspond to a genuine, not-yet-finished ply
    record = dict(
        x=x,                       # [B, 64, F]
        child_x=child_x,           # [B, 64, F]
        played=move_index,         # [B]
        root_value=root_value,     # [B]  (pre-outcome; blended after the scan)
        q_target=q_target,         # [B, A]
        q_weight=q_weight,         # [B, A]
        policy=policy_target,      # [B, A]
        mask=mask,                 # [B, A]
        mover_white=mover_white,   # [B]
        q_head_played=q_head_played,  # [B]
        valid=valid_ply,           # [B]
        state=_lite_state(carry.states),  # lite State (no 30KB observation)
        history=carry.history,     # [B, 4] root move-stack history for this ply
    )

    new_carry = _Carry(
        states=child_states,
        history=child_history,
        key=key,
        done=new_done,
        ply=carry.ply + 1,
        streak_white=streak_white,
        streak_black=streak_black,
        resign_active=carry.resign_active,
        resigned_white_won=resigned_white_won,
        z_white=z_white,
        plies=plies,
    )
    return new_carry, record


def _state_truncated(states: Any) -> jnp.ndarray:
    """Boolean [B] truncation flag (pgx 512-step cap), tolerant of envs that do
    not expose ``truncated``."""
    trunc = getattr(states, "truncated", None)
    if trunc is None:  # pragma: no cover - all pgx envs have it
        return jnp.zeros((jnp.asarray(states._step_count).shape[0],), dtype=bool)  # type: ignore[attr-defined]
    return jnp.asarray(trunc).astype(bool)


def _apply_outcome_blend(record_stack, carry: _Carry, spcfg: SelfPlayConfig):
    """Vectorize ``play_game_gen``'s outcome blend over the whole ``[B, T]``
    grid and produce the flat training tensors.

    ``record_stack`` leaves are ``[T, B, ...]`` (scan stacks along axis 0); we
    transpose the per-game scalars/targets to ``[B, T, ...]`` so a single
    ``jnp.where`` grid applies. Returns ``(SamplesBatch, GameMeta)``.
    """
    # scan stacks time on axis 0 -> [T, B, ...]; move to [B, T, ...].
    def t_to_bt(leaf):
        return jnp.swapaxes(leaf, 0, 1)

    x = t_to_bt(record_stack["x"])                 # [B, T, 64, F]
    child_x = t_to_bt(record_stack["child_x"])     # [B, T, 64, F]
    played = t_to_bt(record_stack["played"])       # [B, T]
    root_value = t_to_bt(record_stack["root_value"])  # [B, T]
    q_target = t_to_bt(record_stack["q_target"])   # [B, T, A]
    q_weight = t_to_bt(record_stack["q_weight"])   # [B, T, A]
    policy = t_to_bt(record_stack["policy"])       # [B, T, A]
    mask = t_to_bt(record_stack["mask"])           # [B, T, A]
    mover_white = t_to_bt(record_stack["mover_white"])  # [B, T]
    q_head_played = t_to_bt(record_stack["q_head_played"])  # [B, T]
    valid = t_to_bt(record_stack["valid"])         # [B, T] bool
    states_per_ply = jax.tree_util.tree_map(t_to_bt, record_stack["state"])  # State[B,T]

    B, T = root_value.shape

    # ----- final outcome per game (z_white) ------------------------------
    # Captured inside the scan at the exact terminal/resignation transition.
    # Do not recompute from ``carry.states`` here: finished games keep flowing
    # through padding scan steps, and pgx's transition reward is not guaranteed
    # to remain readable at the end of the fixed-length scan.
    z_white = carry.z_white                                    # [B] (may be NaN)

    # ----- per-ply outcome, mirrored to the mover ------------------------
    z_white_bt = z_white[:, None]                               # [B, 1] -> [B, T]
    z_known = ~jnp.isnan(z_white_bt)                            # [B, T] bool
    z_known = z_known & valid

    # z = z_white if mover_was_white else -z_white
    z = jnp.where(mover_white, z_white_bt, -z_white_bt)         # [B, T]
    z_safe = jnp.where(z_known, z, 0.0)                         # avoid NaN math

    # wdl class = int(z)+1 in {0,1,2}; -1 where unknown.
    wdl = jnp.where(z_known, jnp.round(z_safe).astype(jnp.int32) + 1, -1)  # [B, T]

    # z_eff = -contempt if z==0 else z * win_discount ** (total - t)
    total = carry.plies.astype(jnp.float32)[:, None]           # [B, 1] real plies
    t_grid = jnp.arange(T, dtype=jnp.float32)[None, :]          # [1, T]
    discount = spcfg.win_discount ** (total - t_grid)          # [B, T]
    is_draw = jnp.abs(z_safe) < 0.5                             # z == 0 (within rounding)
    z_eff = jnp.where(is_draw, -spcfg.contempt, z_safe * discount)  # [B, T]

    # value_target = (1-mix)*root_value + mix*z_eff  (only where z known)
    blended = (1.0 - spcfg.outcome_mix) * root_value + spcfg.outcome_mix * z_eff
    value = jnp.where(z_known, blended, root_value)            # [B, T]

    # ----- flatten [B, T] -> [N] (training contract) ---------------------
    N = B * T

    def flat(a):
        return a.reshape((N,) + a.shape[2:])

    weight = jnp.where(valid, 1.0, 0.0).astype(jnp.float32)    # self-play weight 1.0
    samples = SamplesBatch(
        x=flat(x).astype(jnp.float32),
        child_x=flat(child_x).astype(jnp.float32),
        played=flat(played).astype(jnp.int32),
        value=flat(value).astype(jnp.float32),
        weight=flat(weight),
        wdl=flat(wdl).astype(jnp.int32),
        mask=flat(mask).astype(bool),
        policy=flat(policy).astype(jnp.float32),
        q_target=flat(q_target).astype(jnp.float32),
        q_weight=flat(q_weight).astype(jnp.float32),
        valid=flat(valid).astype(bool),
    )

    # ----- result code per game (for logging / surprise detection) -------
    # 0='0-1', 1='1/2-1/2', 2='1-0', -1='*' (unfinished/truncated).
    result = jnp.where(
        jnp.isnan(z_white),
        RESULT_UNFINISHED,
        (jnp.round(z_white).astype(jnp.int32) + 1),  # -1->0, 0->1, +1->2
    ).astype(jnp.int32)

    meta = GameMeta(
        root_values=root_value.astype(jnp.float32),
        q_head_played=q_head_played.astype(jnp.float32),
        mover_white=mover_white.astype(bool),
        z_white=z_white.astype(jnp.float32),
        plies=carry.plies.astype(jnp.int32),
        result=result,
        valid_ply=valid.astype(bool),
        states_per_ply=states_per_ply,
    )
    return samples, meta


def _generate_selfplay_impl(params, key, B, scfg, spcfg, gate):
    """Inner (traceable) implementation. ``B``, ``max_plies``, and ``gate`` are
    Python ints/bools (static) so the scan length and buffer shapes are
    compile-time constants; ``params``/``key`` are traced."""
    if env_mod is None or batched_search is None:  # pragma: no cover
        raise RuntimeError(
            "prophet_jax.selfplay requires sibling modules `env` and `search`; "
            "they were not importable. Ensure prophet_jax/env.py and "
            "prophet_jax/search.py exist."
        )

    max_plies = int(spcfg.max_plies)

    # --- initial batched state + per-game rng ----------------------------
    key, init_key, resign_key = jax.random.split(key, 3)
    start_keys = env_mod.start_keys(init_key, B)     # [B] PRNGKeys for env.init
    states = env_mod.env_init(start_keys)            # batched pgx State
    history = env_mod.empty_history(states)          # [B, 4], moonshot move-stack flags

    # --- per-game resignation activation ---------------------------------
    # resign_active = gate AND (draw >= resign_off_prob). The gate is a python
    # bool (orchestration); when closed, no game resigns. Mirrors
    # selfplay.play_game_gen: `resign_active = resign_enabled and rng.random()
    # >= resign_off_prob`.
    if gate:
        draws = jax.random.uniform(resign_key, (B,))
        resign_active = draws >= spcfg.resign_off_prob
    else:
        resign_active = jnp.zeros((B,), dtype=bool)

    carry0 = _Carry(
        states=states,
        history=history,
        key=key,
        done=jnp.zeros((B,), dtype=bool),
        ply=jnp.int32(0),
        streak_white=jnp.zeros((B,), dtype=jnp.int32),
        streak_black=jnp.zeros((B,), dtype=jnp.int32),
        resign_active=resign_active,
        resigned_white_won=-jnp.ones((B,), dtype=jnp.int8),
        z_white=jnp.full((B,), jnp.nan, dtype=jnp.float32),
        plies=jnp.zeros((B,), dtype=jnp.int32),
    )

    # --- the one scan over plies -----------------------------------------
    static = (params, scfg, spcfg)
    step = partial(_scan_step, static)
    carry, record_stack = jax.lax.scan(step, carry0, xs=None, length=max_plies)

    # --- outcome blend + flatten -----------------------------------------
    return _apply_outcome_blend(record_stack, carry, spcfg)


# ``B``, ``spcfg`` (shape-bearing: max_plies), and ``gate`` are static so the
# scan length / buffer shapes specialise at compile time. ``scfg`` is also
# static (its int fields parameterise the search/halving schedule, which must
# unroll to fixed shapes inside search). Recompiles only when these change.
_generate_selfplay_jit = jax.jit(
    _generate_selfplay_impl,
    static_argnums=(2, 3, 4, 5),  # B, scfg, spcfg, gate
)


def generate_selfplay(
    params: Any,
    key: jnp.ndarray,
    B: int,
    scfg: SearchConfig,
    spcfg: SelfPlayConfig,
    gate: bool,
):
    """Play ``B`` self-play games fully in parallel on-device and collect
    prophet-format training targets.

    Args:
      params: the network parameters (flax PyTree) fed to ``batched_search``.
      key: a ``jax.random.PRNGKey``; split internally for env init, the
        resignation lottery, and each scan step's Gumbel search.
      B: number of parallel games (static; thousands == just a bigger ``B``).
      scfg: ``SearchConfig`` driving each ply's Gumbel search (its ``q_trust``
        and sim/candidate counts come from the schedule; passed in unchanged).
      spcfg: ``SelfPlayConfig`` (``max_plies`` sets the scan length and buffer
        size; also ``outcome_mix`` / ``contempt`` / ``win_discount`` /
        resignation knobs).
      gate: master orchestration bool -- when False, resignation is disabled
        for every game (the learner opens the gate once the value head has
        matured; see ``worker.episode_gen``). Reflection/study gating is the
        caller's concern, not this function's.

    Returns:
      ``(samples, meta)`` where

      * ``samples`` is a :class:`SamplesBatch` of dense fixed-shape arrays with
        ``N = B * spcfg.max_plies`` rows (``samples.valid`` marks the real
        rows; padding rows carry ``weight == 0``);
      * ``meta`` is a :class:`GameMeta` carrying per-game outcomes, per-ply
        diagnostics, and the per-ply pgx ``State`` snapshots reflection
        re-searches from.

    The whole rollout (init -> ``lax.scan`` over plies -> outcome blend ->
    flatten) is a single JIT-compiled, ``vmap``-batched computation.
    """
    return _generate_selfplay_jit(params, key, int(B), scfg, spcfg, bool(gate))


__all__ = [
    "SamplesBatch",
    "GameMeta",
    "generate_selfplay",
    "RESULT_BLACK_WIN",
    "RESULT_DRAW",
    "RESULT_WHITE_WIN",
    "RESULT_UNFINISHED",
]
