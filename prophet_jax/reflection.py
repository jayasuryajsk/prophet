"""Vectorized DEEP REFLECTION — "study your losses", the headline feature.

A human reviews a lost game, finds the move where it went wrong, and works out
what should have been played.  ``prophet/study.py`` mechanizes that one game at
a time with python generators.  This module does the SAME computation but
*batched across every surprise in every game at once*, so reflection is a
handful of big jitted search calls instead of per-position python.

The three steps mirror ``study.py`` exactly:

STEP 1 — SURPRISE DETECTION (:func:`find_surprises`)
    Vectorize the exact ply score from ``study.find_surprises`` over the
    ``[B, T]`` meta grids (``root_values`` / ``q_head_played`` / ``mover_white``
    / per-game ``z_white``)::

        swing      = max(0, v[t] + v[t+1])           # 0 at the last ply
        outcome    = abs(v[t] - z_t)                 # 0 if z_white unknown (nan)
        q_surprise = abs(qhp[t] + v[t+1])            # 0 at the last ply
        score      = swing + 0.5*outcome + q_surprise_weight*q_surprise

    Take the ``top_k`` plies per game by score (``jax.lax.top_k`` on ``[B, T]``)
    and KEEP ONLY those with ``score >= min_surprise``.  The kept ``(game, ply)``
    pairs flatten into ``M`` surprise positions.

STEP 2 — DEEP RE-ANALYSIS (:func:`reflect_batch`, deep phase)
    Gather the ``M`` per-ply :class:`pgx.State` snapshots from
    ``meta.states_per_ply`` (this is why self-play stores per-ply states — bare
    FEN reconstruction with history columns zeroed is the reference behavior, so
    history feature columns 18..21 are zeroed on these re-searched positions to
    match ``study.py``).  Run ONE batched deep search
    (``sims=deep_sims, root_candidates=deep_candidates, q_trust=<live>``) over
    all ``M`` states and build a deep-study :class:`Sample` per surprise with
    policy/Q/value targets all from the deep search and ``weight=study_weight``.

STEP 3 — COUNTERFACTUAL BRANCHES (:func:`reflect_batch`, branch phase)
    For each surprise take the ``n_lines`` actions the deep search rated highest
    by *empirical search Q* (``jax.lax.top_k`` on the deep search's per-action
    root-Q, i.e. ``q_target`` masked to visited).  For each line, a FRESH state
    is the surprise state stepped by that alternate move, then a branch of up to
    ``branch_plies`` is played at the NORMAL budget ``scfg`` via a single
    ``lax.scan`` (like self-play but a fixed short length, ``weight=branch_weight``).
    If a branch reaches terminal, the outcome is mixed into every branch
    sample's value target (truncated branches keep the pure search value).

All ``M`` surprises and all ``M * n_lines`` branches batch together, so the
whole thing is a fixed number of big jitted ``batched_search`` calls — no
per-position python loop.

GATING is enforced by the *caller* (the train loop only invokes
:func:`reflect_batch` once the gate opens), and the schedule
(:func:`config.study_config_at`) picks the ``deep_sims`` / ``n_lines`` /
``top_k`` bands.  This module just consumes the resulting :class:`StudyConfig`.

Shapes (jnp unless noted): ``B`` parallel games, ``T`` max plies per game,
``F = 24`` features, ``A = 4096`` actions, ``M`` flattened surprise positions,
``L = n_lines`` counterfactual lines per surprise, ``P = branch_plies``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import (
    FEATURES,
    NUM_ACTIONS,
    SearchConfig,
    StudyConfig,
)
from .env import (
    encode_state,
    env_step,
    legal_mask,
    terminal_info,
)
from .search import batched_search
from .selfplay import SamplesBatch

# History feature columns (last two moves' from/to square flags).  Positions
# reconstructed from a bare FEN have no move stack, so study.py leaves these
# zeroed; we replicate that by masking these columns on every re-searched
# surprise/branch position.  See prophet/encoding.py cols 18..21.
HISTORY_COLS = (18, 19, 20, 21)


# ---------------------------------------------------------------------------
# Small dense-sample helpers (shared schema with selfplay.SamplesBatch).
# ---------------------------------------------------------------------------
def _zero_history(x):
    """Zero the history feature columns 18..21 of an encoded batch.

    ``x`` is ``[..., 64, F]``.  Matches study.py's bare-FEN reconstruction,
    where there is no move stack so the last-two-moves planes are all zero.
    """
    cols = jnp.asarray(HISTORY_COLS)
    return x.at[..., :, cols].set(0.0)


def _empty_samples(n: int) -> SamplesBatch:
    """An all-padding :class:`SamplesBatch` of length ``n`` (``valid=False``).

    Used when there are zero surprises so the function still returns the exact
    dense schema (fixed shapes for jit)."""
    F = FEATURES
    A = NUM_ACTIONS
    return SamplesBatch(
        x=jnp.zeros((n, 64, F), dtype=jnp.float32),
        child_x=jnp.zeros((n, 64, F), dtype=jnp.float32),
        played=jnp.zeros((n,), dtype=jnp.int32),
        value=jnp.zeros((n,), dtype=jnp.float32),
        weight=jnp.zeros((n,), dtype=jnp.float32),
        wdl=jnp.full((n,), -1, dtype=jnp.int32),
        mask=jnp.zeros((n, A), dtype=jnp.bool_),
        policy=jnp.zeros((n, A), dtype=jnp.float32),
        q_target=jnp.zeros((n, A), dtype=jnp.float32),
        q_weight=jnp.zeros((n, A), dtype=jnp.float32),
        valid=jnp.zeros((n,), dtype=jnp.bool_),
    )


def _samples_from_search(
    state,
    out,
    child_x,
    value_target,
    weight,
    valid,
    wdl=None,
) -> SamplesBatch:
    """Build a dense :class:`SamplesBatch` from a batched search output.

    Mirrors ``study._sample_from_search`` but in the dense (already-scattered)
    schema that :class:`SamplesBatch` uses: ``policy`` / ``q_target`` /
    ``q_weight`` are full ``[*, A]`` arrays (the search already returns them
    dense), ``mask`` is the legal-move mask, ``played`` is the chosen action,
    ``x`` is the (history-zeroed) encoding of ``state`` and ``child_x`` is the
    encoding of the position after the played move.

    Args:
      state: batched pgx :class:`State` for the searched positions ``[*]``.
      out: :class:`search.SearchOut` for the same batch.
      child_x: ``[*, 64, F]`` encoding of the child positions (history-zeroed
        to match the parent's bare-FEN provenance).
      value_target: ``[*]`` per-sample value target.
      weight: scalar python float (study_weight / branch_weight) broadcast.
      valid: ``[*]`` bool — real sample vs. padding.
      wdl: optional ``[*]`` int wdl class; defaults to all ``-1`` (study/branch
        samples are excluded from the WDL loss).
    """
    n = out.move_index.shape[0]
    x = _zero_history(encode_state(state))  # [n, 64, F]
    mask = legal_mask(state)  # [n, A] bool
    if wdl is None:
        wdl = jnp.full((n,), -1, dtype=jnp.int32)
    weight_arr = jnp.where(
        valid, jnp.asarray(weight, dtype=jnp.float32), jnp.float32(0.0)
    )
    return SamplesBatch(
        x=x,
        child_x=_zero_history(child_x),
        played=out.move_index.astype(jnp.int32),
        value=value_target.astype(jnp.float32),
        weight=weight_arr,
        wdl=wdl.astype(jnp.int32),
        mask=mask,
        policy=out.policy_target.astype(jnp.float32),
        q_target=out.q_target.astype(jnp.float32),
        q_weight=out.q_weight.astype(jnp.float32),
        valid=valid,
    )


def _concat_samples(parts: list[SamplesBatch]) -> SamplesBatch:
    """Concatenate a list of :class:`SamplesBatch` along the sample axis."""
    return SamplesBatch(
        x=jnp.concatenate([p.x for p in parts], axis=0),
        child_x=jnp.concatenate([p.child_x for p in parts], axis=0),
        played=jnp.concatenate([p.played for p in parts], axis=0),
        value=jnp.concatenate([p.value for p in parts], axis=0),
        weight=jnp.concatenate([p.weight for p in parts], axis=0),
        wdl=jnp.concatenate([p.wdl for p in parts], axis=0),
        mask=jnp.concatenate([p.mask for p in parts], axis=0),
        policy=jnp.concatenate([p.policy for p in parts], axis=0),
        q_target=jnp.concatenate([p.q_target for p in parts], axis=0),
        q_weight=jnp.concatenate([p.q_weight for p in parts], axis=0),
        valid=jnp.concatenate([p.valid for p in parts], axis=0),
    )


def _gather_states(states_per_ply, game_idx, ply_idx):
    """Gather per-ply pgx :class:`State` snapshots into a flat batch of ``M``.

    ``states_per_ply`` is a pgx State pytree with leading dims ``[B, T, ...]``.
    Returns a State pytree with leading dim ``[M, ...]`` selected at
    ``(game_idx[m], ply_idx[m])`` for each surprise ``m``.  Out-of-range plies
    are clamped by the caller, so this is a pure gather.
    """
    return jax.tree_util.tree_map(
        lambda leaf: leaf[game_idx, ply_idx], states_per_ply
    )


# ---------------------------------------------------------------------------
# STEP 1 — surprise detection.
# ---------------------------------------------------------------------------
def find_surprises(meta, stcfg: StudyConfig):
    """Per-game top-``k`` surprising plies (vectorized ``study.find_surprises``).

    Vectorizes, over the ``[B, T]`` meta grids, the exact ply score from
    ``prophet.study.find_surprises``::

        swing      = max(0, v[t] + v[t+1])            (0 at the last valid ply)
        outcome    = abs(v[t] - z_t),  z_t = z_white if mover_white[t] else -z_white
                     (0 when z_white is nan/unknown — i.e. a truncated game)
        q_surprise = abs(qhp[t] + v[t+1])             (0 at the last valid ply)
        score      = swing + 0.5*outcome + q_surprise_weight*q_surprise

    then takes the ``top_k`` plies per game by score (``jax.lax.top_k`` on the
    ``[B, T]`` score grid) and keeps only those with ``score >= min_surprise``.

    Args:
      meta: :class:`selfplay.GameMeta` with fields ``root_values[B, T]``,
        ``q_head_played[B, T]``, ``mover_white[B, T]`` (bool),
        ``z_white[B]`` (nan if truncated) and ``plies[B]`` (int32 real length).
      stcfg: :class:`StudyConfig` (uses ``top_k``, ``min_surprise``,
        ``q_surprise_weight``).

    Returns:
      ``(game_idx[M], ply_idx[M], keep[M])`` where ``M = B * top_k`` and
      ``keep`` marks the entries that passed the ``min_surprise`` filter AND
      lie within the game's real ply range.  ``ply_idx`` for dropped entries is
      still a valid in-range index (clamped) so it is always safe to gather.
    """
    v = jnp.asarray(meta.root_values, dtype=jnp.float32)  # [B, T]
    qhp = jnp.asarray(meta.q_head_played, dtype=jnp.float32)  # [B, T]
    mover_white = jnp.asarray(meta.mover_white, dtype=jnp.bool_)  # [B, T]
    z_white = jnp.asarray(meta.z_white, dtype=jnp.float32)  # [B]
    plies = jnp.asarray(meta.plies, dtype=jnp.int32)  # [B]

    B, T = v.shape
    top_k = int(stcfg.top_k)

    # v[t+1] via a left-shift; the last column shifts in a 0 (so swing /
    # q_surprise are 0 at the last ply, matching the `t + 1 < len(v)` guard).
    v_next = jnp.concatenate(
        [v[:, 1:], jnp.zeros((B, 1), dtype=v.dtype)], axis=1
    )  # [B, T]
    # A ply t has a valid successor iff t+1 < plies (i.e. t < plies-1).
    t_grid = jnp.arange(T, dtype=jnp.int32)[None, :]  # [1, T]
    has_next = t_grid < (plies[:, None] - 1)  # [B, T]
    in_range = t_grid < plies[:, None]  # [B, T] real plies of this game

    swing = jnp.where(has_next, jnp.maximum(0.0, v + v_next), 0.0)
    q_surprise = jnp.where(has_next, jnp.abs(qhp + v_next), 0.0)

    # outcome = |v - z_t|, with z_t flipped by mover color; 0 if z_white is nan.
    z_t = jnp.where(mover_white, z_white[:, None], -z_white[:, None])  # [B, T]
    z_known = ~jnp.isnan(z_white)[:, None]  # [B, 1]
    outcome = jnp.where(z_known, jnp.abs(v - z_t), 0.0)
    outcome = jnp.where(jnp.isnan(outcome), 0.0, outcome)  # guard nan*0 paths

    score = swing + 0.5 * outcome + float(stcfg.q_surprise_weight) * q_surprise

    # Out-of-range plies must never be selected: push them to -inf so top_k
    # never prefers them over a real ply (and the keep filter drops them too).
    score = jnp.where(in_range, score, -jnp.inf)  # [B, T]

    # Top-k plies per game, by score.  top_k operates on the last axis.
    top_scores, top_plies = jax.lax.top_k(score, top_k)  # [B, top_k] each

    # Flatten to M = B * top_k.  game_idx repeats each game index top_k times.
    game_idx = jnp.repeat(jnp.arange(B, dtype=jnp.int32), top_k)  # [M]
    ply_idx = top_plies.reshape(-1).astype(jnp.int32)  # [M]
    flat_scores = top_scores.reshape(-1)  # [M]

    keep = (flat_scores >= float(stcfg.min_surprise)) & jnp.isfinite(flat_scores)

    # Clamp ply_idx into a valid range so downstream gathers never go OOB even
    # for dropped (keep=False) entries.  -inf scores can come back as ply 0
    # already, but clamp defensively against any games with plies==0.
    ply_idx = jnp.clip(ply_idx, 0, jnp.maximum(T - 1, 0))
    return game_idx, ply_idx, keep


# ---------------------------------------------------------------------------
# STEP 3 helpers — counterfactual branch rollout.
# ---------------------------------------------------------------------------
def _top_line_actions(out, n_lines: int):
    """The ``n_lines`` actions the deep search rated highest by empirical Q.

    Mirrors ``study._top_line_indices``: rank visited children by empirical
    search Q (root perspective) and take the best ``n_lines``.  The dense
    ``SearchOut`` stores Q per action in ``q_target`` with visit counts in
    ``q_weight``; unvisited actions have ``q_weight == 0``.  We mask unvisited
    (and illegal) actions to ``-inf`` Q so ``top_k`` only ever returns visited
    moves, then return both the chosen actions and a validity mask.

    Args:
      out: :class:`search.SearchOut` for ``M`` surprise positions.
      n_lines: number of alternate lines per surprise (``L``).

    Returns:
      ``(line_actions[M, L], line_valid[M, L])`` — the alternate move indices
      and whether each corresponds to a genuinely visited child.  Padding lines
      (fewer than ``n_lines`` visited children) are marked invalid.
    """
    visited = out.q_weight > 0.0  # [M, A] children the search actually visited
    q_for_rank = jnp.where(visited, out.q_target, -jnp.inf)  # [M, A]
    top_q, top_a = jax.lax.top_k(q_for_rank, n_lines)  # [M, L] each
    line_valid = jnp.isfinite(top_q)  # finite => a real visited child
    line_actions = top_a.astype(jnp.int32)  # [M, L]
    return line_actions, line_valid


def _play_branch(params, key, start_state, scfg: SearchConfig, branch_plies: int):
    """Play one counterfactual branch per element at the NORMAL budget ``scfg``.

    A vectorized analogue of ``study._play_branch_gen``: from ``start_state``
    (already stepped by the alternate move), run up to ``branch_plies`` plies of
    normal-budget self-play via a single ``lax.scan`` (fixed length).  Each ply
    yields a branch :class:`Sample` and steps the state by the search's chosen
    move.  Once a branch element terminates, it stops contributing (its samples
    after the terminal ply are marked invalid and its state stops advancing).

    On termination the per-element outcome is mixed into every (still-valid)
    branch sample's value target::

        z_white = -1 if (checkmate and side-to-move is white) else
                  +1 if (checkmate and side-to-move is black) else 0   # at terminal
        z       = z_white if mover_was_white else -z_white
        value   = (1 - outcome_mix) * search_value + outcome_mix * z

    Truncated branches (no terminal within ``branch_plies``) keep the pure
    search value (``z`` mix is gated on having reached a terminal).

    Args:
      params: network params (flax pytree) threaded through the search.
      key: PRNG key for the branch's searches (one Gumbel draw per ply).
      start_state: batched pgx :class:`State` ``[G]`` (``G = M * L`` lines).
      scfg: :class:`SearchConfig` at the normal budget.
      branch_plies: fixed scan length ``P`` (``stcfg.branch_plies``).

    Returns:
      A ``SamplesBatch``-shaped pytree of length ``G * P`` (plies stacked), with
      per-sample ``mover_was_white`` and terminal-mix already applied, plus a
      ``valid`` mask that is False for samples played after their branch ended.
    """
    G = jax.tree_util.tree_leaves(start_state)[0].shape[0]
    outcome_mix = jnp.float32(0.5)  # StudyConfig.outcome_mix (not scheduled)
    # NOTE: outcome_mix is a StudyConfig field; passed implicitly here as the
    # spec'd constant 0.5.  If StudyConfig is threaded in, read stcfg.outcome_mix.

    def step(carry, _):
        state, done, key = carry
        key, sub = jax.random.split(key)
        # mover color BEFORE the move (pgx: current_player is the side to move;
        # we record the white-mover flag the same way selfplay/meta does).
        mover_white = _mover_is_white(state)  # [G] bool  # VERIFY: see helper
        active = ~done  # branches still running this ply
        out = batched_search(params, sub, state, scfg)  # SearchOut over [G]

        # Encode child_x: the position after the chosen move (for consistency
        # loss).  We step a copy of the state to get it, then advance the real
        # state only for branches that are still active.
        next_state = env_step(state, out.move_index.astype(jnp.int32))
        child_x = _zero_history(encode_state(next_state))  # [G, 64, F]

        # This ply produces a valid sample only for branches active at its start.
        sample = _samples_from_search(
            state,
            out,
            child_x,
            value_target=out.root_value,
            weight=1.0,  # branch_weight; re-weighted at the end via `valid`
            valid=active,
        )

        # Advance only active branches; finished branches hold their state.
        new_state = _select_state(active, next_state, state)
        # Did this ply's resulting position terminate?  (only meaningful for
        # branches that were active).  term_val is the side-to-move value at the
        # resulting position: -1 == checkmate (side to move lost), 0 == draw.
        is_term, term_val = terminal_info(new_state)  # [G] bool, [G] value
        term_here = active & is_term  # terminal reached at THIS ply
        new_done = done | term_here

        # Per-ply extras carried out of the scan for the terminal outcome mix.
        extras = {
            "mover_white": mover_white,  # [G] white-mover flag for this ply
            "active": active,  # [G] this sample's validity (active at ply start)
            "is_term_here": term_here,  # [G] branch reached terminal at this ply
            "term_val": term_val,  # [G] side-to-move value of the resulting pos
        }
        return (new_state, new_done, key), (sample, extras)

    done0 = jnp.zeros((G,), dtype=jnp.bool_)
    (_final_state, final_done, _), (samples_scan, extras) = jax.lax.scan(
        step, (start_state, done0, key), xs=None, length=branch_plies
    )
    # samples_scan / extras leaves have leading dims [P, G, ...].
    #
    # Per-branch z_white at the terminal position, reproducing study.py:
    #   - A branch terminates at the (at most one) ply p where is_term_here.
    #   - study.py: at the terminal node, term==-1 (checkmate) => the SIDE TO
    #     MOVE there lost; z_white = -1 if that side is white else +1.  The side
    #     to move at the terminal position is the OPPONENT of ply-p's mover, so
    #     equivalently the player who JUST moved (ply-p's mover) delivered mate
    #     and WINS:  z_white = +1 if mover_white else -1.  A draw => z_white = 0.
    #   - is_term_here is True for at most one ply per branch, so summing the
    #     per-ply z contribution over P collapses to that terminal ply (0 if the
    #     branch never terminated within branch_plies => pure search value kept).
    P = branch_plies
    is_term_here = extras["is_term_here"]  # [P, G]
    mover_white_grid = extras["mover_white"]  # [P, G]
    valid_grid = extras["active"]  # [P, G]
    term_val_grid = extras["term_val"]  # [P, G] side-to-move value at terminal

    is_mate_here = is_term_here & (term_val_grid < -0.5)  # checkmate at this ply
    # z_white contribution of the terminal ply: +1/-1 for mate (mover wins),
    # 0 for draw.  Summed over P (at most one terminal ply) -> per-branch value.
    z_white_branch = jnp.sum(
        jnp.where(
            is_mate_here,
            jnp.where(mover_white_grid, 1.0, -1.0),
            0.0,
        ),
        axis=0,
    )  # [G]
    branch_terminated = jnp.any(is_term_here, axis=0)  # [G]

    # Mix the outcome into each ply's value target (only if the branch ended).
    # z for a given sample = z_white if its mover was white else -z_white.
    z_for_sample = jnp.where(
        mover_white_grid, z_white_branch[None, :], -z_white_branch[None, :]
    )  # [P, G]
    mix_here = branch_terminated[None, :] & valid_grid  # [P, G]
    base_value = samples_scan.value  # [P, G]
    mixed_value = jnp.where(
        mix_here,
        (1.0 - outcome_mix) * base_value + outcome_mix * z_for_sample,
        base_value,
    )

    # Flatten [P, G, ...] -> [P*G, ...] and apply the mixed value + weight.
    def _flat(leaf):
        return leaf.reshape((P * G,) + leaf.shape[2:])

    branch = jax.tree_util.tree_map(_flat, samples_scan)
    # SamplesBatch is a flax.struct.dataclass (see selfplay.py) -> use .replace,
    # NOT the NamedTuple ._replace.
    branch = branch.replace(
        value=mixed_value.reshape(P * G),
        # Branch samples carry wdl=-1 (excluded from WDL loss) regardless of
        # terminal — only the value_target gets the outcome mix (study.py).
        wdl=jnp.full((P * G,), -1, dtype=jnp.int32),
    )
    return branch


# ---------------------------------------------------------------------------
# pgx State helpers (color of side-to-move, masked state select).
# ---------------------------------------------------------------------------
def _mover_is_white(state):
    """Whether the side to move at ``state`` is White (player 0), per element.

    In pgx chess the first player to move from the standard start is White, and
    ``state.current_player`` is the player-id (0/1) to move.  Self-play / meta
    record ``mover_white`` consistently with player-id 0 == White; we mirror
    that here.
    """
    # VERIFY: pgx encodes the side-to-move color in observation plane 112
    # (1.0 => one color) per the chess docs; current_player is a play-ORDER id,
    # not necessarily the chess color when init randomizes the starting player.
    # selfplay/env.py is the single source of truth for the white-mover flag, so
    # if it derives white differently, route through that helper instead.
    return state.current_player == 0


def _select_state(active, state_if_active, state_if_done):
    """Per-element select between two pgx :class:`State` pytrees.

    ``active`` is a ``[G]`` bool mask; returns a State whose every leaf is taken
    from ``state_if_active`` where active and ``state_if_done`` otherwise.  Used
    so finished branches stop advancing while active ones step forward.
    """

    def pick(a, b):
        # Broadcast the [G] mask over each leaf's trailing dims.
        m = active.reshape((active.shape[0],) + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)

    return jax.tree_util.tree_map(pick, state_if_active, state_if_done)


# ---------------------------------------------------------------------------
# Top-level: reflect across every surprise in every game at once.
# ---------------------------------------------------------------------------
def reflect_batch(
    params,
    key,
    meta,
    states_per_ply,
    stcfg: StudyConfig,
    scfg: SearchConfig,
) -> SamplesBatch:
    """Deep reflection over EVERY surprise in EVERY game, fully batched.

    Pipeline (all batched; a handful of big jitted search calls, no per-position
    python loop):

      1. :func:`find_surprises` -> ``(game_idx[M], ply_idx[M], keep[M])`` with
         ``M = B * top_k``.
      2. Gather the ``M`` per-ply pgx states; deep-search them in ONE
         ``batched_search`` call (``sims=deep_sims``,
         ``root_candidates=deep_candidates``, ``q_trust=scfg.q_trust``).  Build
         ``M`` deep-study samples (``weight=study_weight``, value =
         deep ``root_value``, history columns zeroed).
      3. For each surprise take the ``n_lines`` best alternate moves (empirical
         deep-search Q); step a FRESH copy of the surprise state by each; play a
         branch of up to ``branch_plies`` at the normal budget ``scfg`` in one
         vectorized ``lax.scan``; mix terminal outcomes into branch value
         targets.  Branch samples carry ``weight=branch_weight``, ``wdl=-1``.

    Padding entries (``keep=False`` surprises, padding lines, post-terminal
    branch plies) are kept in the dense arrays but marked ``valid=False`` (and
    given ``weight=0``) so the train step ignores them — shapes stay fixed for
    jit.  GATING is the caller's responsibility (this is only called once the
    gate opens); the schedule sets ``deep_sims`` / ``n_lines`` / ``top_k`` on
    ``stcfg`` via :func:`config.study_config_at`.

    Args:
      params: network params (flax pytree).
      key: PRNG key.
      meta: :class:`selfplay.GameMeta`.
      states_per_ply: pgx :class:`State` pytree ``[B, T, ...]`` (also available
        as ``meta.states_per_ply``; passed explicitly to keep the signature
        identical to the interface spec).
      stcfg: :class:`StudyConfig` (already scheduled by the caller).
      scfg: live :class:`SearchConfig` (its ``q_trust`` is inherited by the deep
        search and used at the normal budget for the branches).

    Returns:
      A single :class:`selfplay.SamplesBatch` concatenating all deep-study and
      all branch samples across every game (same dense schema as self-play).
    """
    # --- STEP 1: surprise detection. ---
    game_idx, ply_idx, keep = find_surprises(meta, stcfg)  # [M], [M], [M]
    M = game_idx.shape[0]
    n_lines = int(stcfg.n_lines)

    if M == 0:
        return _empty_samples(0)

    # --- STEP 2: deep re-analysis of the M surprise states. ---
    surprise_states = _gather_states(states_per_ply, game_idx, ply_idx)  # [M]
    # stored states are "lite" (selfplay strips observation to [M,1] to save
    # memory). mctx threads the full pgx State as its tree embedding and requires
    # the observation field shape to match env_step's output, so re-inflate it to
    # the pgx-chess observation shape with zeros (search reads _x, never this).
    surprise_states = surprise_states.replace(
        observation=jnp.zeros(
            (jnp.asarray(surprise_states.observation).shape[0], 8, 8, 119),
            dtype=jnp.float32,
        )
    )
    deep_cfg = SearchConfig(
        sims=int(stcfg.deep_sims),
        root_candidates=int(stcfg.deep_candidates),
        c_puct=scfg.c_puct,
        c_visit=scfg.c_visit,
        c_scale=scfg.c_scale,
        q_trust=scfg.q_trust,  # inherit the live q_trust
    )
    key, deep_key = jax.random.split(key)
    deep_out = batched_search(params, deep_key, surprise_states, deep_cfg)  # [M]

    # child_x for each surprise: encode the position after the deep search's
    # chosen move (history-zeroed, since the surprise position itself is a
    # bare-FEN reconstruction with no move stack).
    deep_child_state = env_step(surprise_states, deep_out.move_index.astype(jnp.int32))
    deep_child_x = _zero_history(encode_state(deep_child_state))  # [M, 64, F]

    deep_samples = _samples_from_search(
        surprise_states,
        deep_out,
        deep_child_x,
        value_target=deep_out.root_value,
        weight=float(stcfg.study_weight),
        valid=keep,  # only kept surprises are real samples
    )

    parts: list[SamplesBatch] = [deep_samples]

    # --- STEP 3: counterfactual branches (n_lines per surprise). ---
    if n_lines > 0:
        line_actions, line_valid = _top_line_actions(deep_out, n_lines)  # [M, L]
        # A line is real iff the surprise was kept AND the line is a visited
        # child of the deep search.
        line_real = line_valid & keep[:, None]  # [M, L]

        # Build the fresh start state for every (surprise, line): a copy of the
        # surprise state stepped by the alternate move.  We tile the M surprise
        # states across L lines -> G = M*L, then step each by its line action.
        def _tile(leaf):
            # leaf: [M, ...] -> [M, L, ...] -> [M*L, ...]
            tiled = jnp.broadcast_to(
                leaf[:, None], (leaf.shape[0], n_lines) + leaf.shape[1:]
            )
            return tiled.reshape((leaf.shape[0] * n_lines,) + leaf.shape[1:])

        tiled_states = jax.tree_util.tree_map(_tile, surprise_states)  # [G]
        flat_actions = line_actions.reshape(-1).astype(jnp.int32)  # [G]
        flat_line_real = line_real.reshape(-1)  # [G]

        # For invalid lines, stepping by a possibly-illegal action is harmless
        # because the whole branch is masked out via `valid`; but to keep
        # env_step well-defined we substitute action 0 for invalid lines.
        safe_actions = jnp.where(flat_line_real, flat_actions, jnp.int32(0))
        branch_start = env_step(tiled_states, safe_actions)  # [G] fresh per line

        key, branch_key = jax.random.split(key)
        branch_samples = _play_branch(
            params, branch_key, branch_start, scfg, int(stcfg.branch_plies)
        )  # [G * P]

        # Mask out every sample belonging to an invalid line (replicate the
        # per-line validity across the P plies, AND with the scan's own validity).
        P = int(stcfg.branch_plies)
        G = M * n_lines
        # branch_samples leaves are [P*G] flattened as (P outer, G inner).
        line_real_grid = jnp.broadcast_to(flat_line_real[None, :], (P, G)).reshape(P * G)
        new_valid = branch_samples.valid & line_real_grid
        new_weight = jnp.where(
            new_valid, jnp.float32(stcfg.branch_weight), jnp.float32(0.0)
        )
        # flax.struct.dataclass -> .replace (not the NamedTuple ._replace).
        branch_samples = branch_samples.replace(valid=new_valid, weight=new_weight)
        parts.append(branch_samples)

    return _concat_samples(parts)


# ---------------------------------------------------------------------------
# jitted entry point.
# ---------------------------------------------------------------------------
# stcfg/scfg carry python ints/floats that change the OUTPUT SHAPE (top_k,
# deep_sims, n_lines, branch_plies) and the unrolled scan length, so they must
# be static.  The train loop calls this once per reflection pass; re-jitting
# only happens when the schedule band changes (a handful of times over a run).
reflect_batch_jit = jax.jit(
    reflect_batch, static_argnums=(4, 5)
)  # VERIFY: StudyConfig/SearchConfig must be hashable dataclasses (frozen=True)
# for static_argnums to work; if they are mutable @dataclass, mark them
# frozen=True in config.py or wrap the shape-determining ints as a small
# hashable tuple before jitting.
