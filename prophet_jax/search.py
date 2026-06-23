"""Batched Gumbel-MuZero search (mctx) reproducing prophet's search semantics.

This is the JAX port of ``prophet/search.py``. The reference engine runs a
Gumbel top-k + sequential-halving selection at the *root* and PUCT in the
*interior*, with unvisited children seeded by the network's per-move Q-head
(``q_trust * q_init``). We reproduce that as closely as the Gumbel-MuZero
*formulation* allows by wrapping ``mctx.gumbel_muzero_policy`` and running it
fully batched / vmapped over thousands of positions in a single compiled call.

What maps exactly
-----------------
* **Root algorithm.** ``gumbel_muzero_policy`` *natively* does Gumbel top-k
  candidate selection + Sequential Halving over
  ``min(num_simulations, max_num_considered_actions)`` actions. Setting
  ``num_simulations = cfg.sims`` (32), ``max_num_considered_actions =
  cfg.root_candidates`` (8) and ``gumbel_scale = 1.0`` makes mctx's root match
  prophet's root algorithm (Gumbel draw on raw logits, deterministic argmax of
  the halving). This *is* what the policy does out of the box.
* **Action space.** We keep mctx's action space equal to prophet's 4096
  (``from*64 + to``) and pass ``invalid_actions = ~legal_mask`` so search only
  ever considers legal from-to moves. The env layer (``env.py``) handles the
  4096 -> pgx-4672 remap inside ``env_step``; mctx never sees pgx's 4672 layout.
* **Terminal handling.** ``recurrent_fn`` returns ``discount = 0`` at terminal
  children (cutting the value bootstrap, matching prophet's terminal handling)
  and overrides the child value with the true mate/draw ``terminal_value`` so
  the backup uses the real outcome.
* **Policy target.** ``policy_output.action_weights`` (the Gumbel-improved,
  completed-Q distribution that already sums to 1 over legal moves) is the
  training target — the analogue of prophet's ``policy_target``.

What is *approximated* (NOT bit-reproducible under mctx)
-------------------------------------------------------
Prophet's interior PUCT seeds an *unvisited* child with ``q_trust * q_init``
(the Q-head value of the move into that child) as its first-play value. mctx's
default ``qtransform_completed_by_mix_value`` instead completes unvisited-action
Q from a *value mixture* of the parent value and the visited children's Q. There
is no exact, bit-for-bit way to inject prophet's per-action ``q_init`` as the
unvisited first-play value through the public mctx API:

  - Folding ``q_trust * q`` into ``prior_logits`` (as a temperature shift) is
    **not** equivalent — it changes the prior/Gumbel weighting, not the
    completed-Q used by the qtransform.
  - The qtransform is what consumes per-action Q, and it derives unvisited Q
    from the value mixture, not from a user-supplied per-action initial value.

So we *document* that exact PUCT-with-``q_init`` is not reproducible here and
accept that the interior selection differs. We still seed the Q-head into the
tree everywhere it *is* expressible: the per-action Q-head value flows in as the
root/child ``value`` and (for the root) as the network priors, and we rely on
the Gumbel root algorithm (which prophet itself uses at the root) to be the
behaviourally-dominant part at these tiny sim budgets. The interior PUCT
difference is the accepted approximation.

Everything here is pure JAX so an entire batch of games searches in one
``jax.jit``-compiled call (no per-search python/numpy tree ops, unlike the
reference generator engine).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mctx

# --- prophet_jax siblings -------------------------------------------------
# env.py threads the *pgx State* through mctx as the opaque embedding.
from .env import encode_state, env_step, legal_mask, terminal_info

# Network forward: (params, x[B,64,F]) -> (policy_logits[B,4096], q[B,4096], v[B]).
#
# NOTE: the module-plan lists model.forward as forward(model, params, x), but the
# search interface (root_fn(params, state), recurrent_fn(params, rng, action,
# state)) only threads `params` — it never receives the flax `model` object. So
# this module requires a *params-only* forward in scope. We import `forward` and
# call it as model_forward(params, x). model.py must therefore expose forward as
# either (a) forward(params, x) directly, or (b) a partial/closure with the
# PolicyQValueNet already bound (e.g. functools.partial(forward, model)). If
# model.py keeps the 3-arg forward(model, params, x), bind it at module import
# there or re-export a 2-arg alias named `forward`. See API risks in the summary.
from .model import forward as model_forward

# Action-space / config constants. We try the real config module first and fall
# back to local constants so this file stays importable before config.py lands.
try:  # pragma: no cover - exercised once config.py exists
    from .config import NUM_ACTIONS, SearchConfig
except Exception:  # pragma: no cover
    NUM_ACTIONS = 4096

    from dataclasses import dataclass

    @dataclass
    class SearchConfig:  # type: ignore[no-redef]
        """Fallback mirror of prophet's SearchConfig (+ deep variants).

        The interior-PUCT knobs (``c_puct`` / ``c_visit`` / ``c_scale``) are
        retained for interface parity but are NOT used by the mctx wrapper —
        mctx fixes its own Gumbel/qtransform constants. ``q_trust`` is likewise
        carried for parity (and surprise-detection callers) but cannot be
        injected into mctx's completed-Q; see the module docstring.
        """

        sims: int = 32
        root_candidates: int = 8
        c_puct: float = 1.5
        c_visit: float = 50.0
        c_scale: float = 1.0
        q_trust: float = 1.0
        deep_sims: int = 128
        deep_candidates: int = 16


# mctx marks ILLEGAL actions with 1.0 in ``invalid_actions`` (legal -> 0.0).
# legal_mask(state) is True at LEGAL actions, so invalid = logical-not.
def _invalid_actions(state: Any) -> jnp.ndarray:
    """[B, NUM_ACTIONS] float32, 1.0 at illegal actions (mctx convention)."""
    return (~legal_mask(state)).astype(jnp.float32)


# --------------------------------------------------------------------------
# Root and recurrent functions (the mctx model contract).
# --------------------------------------------------------------------------
def root_fn(params: Any, state: Any) -> mctx.RootFnOutput:
    """Build the mctx root from a batched pgx ``State``.

    Runs ``model.forward`` on ``encode_state(state)`` to get
    ``(policy_logits[B,4096], q[B,4096], v[B])`` and returns:

      * ``prior_logits`` = the **RAW** policy logits (mctx wants logits, NOT a
        softmax — do not normalise them here). Illegal actions are masked out
        separately via ``invalid_actions`` passed to the policy call, so we keep
        the full 4096 logit vector untouched.
      * ``value``        = ``v`` (the network's P(win) - P(loss) scalar in
        ``[-1, 1]`` from the side-to-move's perspective).
      * ``embedding``    = the pgx ``State`` itself. mctx treats this as an
        opaque pytree and only threads it through ``recurrent_fn``; it never
        inspects it.

    The Q-head ``q`` is not a field of ``RootFnOutput`` (mctx has no per-action
    root value slot); it is seeded into the search via ``recurrent_fn`` (each
    child's ``value`` is the network value of that child, and the played-move
    Q-head value is surfaced separately by ``search_result`` for surprise
    detection). See the module docstring for why exact ``q_init`` first-play
    seeding is not reproducible under mctx.
    """
    x = encode_state(state)  # f32[B, 64, FEATURES]
    policy_logits, _q, v = model_forward(params, x)
    return mctx.RootFnOutput(
        prior_logits=policy_logits,          # RAW logits, [B, 4096]
        value=v,                             # [B]
        embedding=state,                     # opaque pgx State pytree
    )


def recurrent_fn(params: Any, rng_key: Any, action: jnp.ndarray, state: Any):
    """One simulation step: apply ``action`` to ``state`` and evaluate the child.

    ``action`` is a batched prophet action index ``[B]`` (``from*64 + to``).
    Contract (mctx): return ``(RecurrentFnOutput, next_embedding)`` in that
    order, where ``next_embedding`` is the child pgx ``State``.

    Field-by-field (all ``[B]`` / ``[B, 4096]``):

      * ``reward``   = 0.0. prophet's reward signal is its *dense Q-head*, but
        mctx's backup is value-based (it bootstraps from ``value``), so the
        per-transition reward is zero and the child ``value`` carries the
        signal. (Prophet's "dense reward" lives in the value/Q targets at train
        time, not in the MCTS transition reward.)
      * ``discount`` = ``where(child_terminal, 0.0, 1.0)`` — cut the bootstrap
        at terminal/absorbing children, matching prophet's terminal handling.
      * ``prior_logits`` = the child's RAW policy logits.
      * ``value``    = ``where(child_terminal, terminal_value, child_v)`` — for
        terminal children we override the network value with the true mate/draw
        value so the backup uses the real outcome; otherwise the network value.

    NOTE on sign convention. pgx and prophet both express the child value from
    the *child's* side-to-move perspective, and mctx itself performs the
    negamax flip across plies when backing values up the tree. So we return the
    child value as-is (side-to-move-relative); we do NOT pre-negate it.
    ``terminal_info`` already returns the side-to-move value at the child
    (checkmate -> -1.0, draw -> 0.0), which is exactly this convention.
    """
    # action: [B] int32 (prophet index). env_step maps 4096 -> pgx 4672 and
    # steps without auto-reset.
    next_state = env_step(state, action.astype(jnp.int32))

    x_child = encode_state(next_state)                 # f32[B, 64, F]
    child_logits, _child_q, child_v = model_forward(params, x_child)

    is_terminal, terminal_value = terminal_info(next_state)  # bool[B], f32[B]

    # TWO-PLAYER NEGAMAX (the drawish-collapse root cause). The child's value/
    # outcome is from the CHILD's (opponent's) side-to-move perspective, so it
    # must FLIP sign when backed up to the parent. mctx applies that flip via a
    # NEGATIVE discount (-1). The old code used +1 -> no flip -> both sides
    # effectively COOPERATE -> nobody seeks the win -> ~0% decisive shuffling.
    # Terminal outcomes are routed through `reward` (flipped to the parent's
    # view) so the search still values a mating move (+1) even though discount=0
    # cuts the bootstrap there.
    discount = jnp.where(is_terminal, 0.0, -1.0)
    reward = jnp.where(is_terminal, -terminal_value, 0.0)
    value = jnp.where(is_terminal, 0.0, child_v)

    out = mctx.RecurrentFnOutput(
        reward=reward,                                 # [B] terminal outcome, parent's view
        discount=discount,                             # [B] -1 non-terminal (negamax), 0 terminal
        prior_logits=child_logits,                     # [B, 4096] RAW logits
        value=value,                                   # [B] network value (non-terminal)
    )
    return out, next_state


# --------------------------------------------------------------------------
# The compiled search call.
# --------------------------------------------------------------------------
# We jit the ENTIRE search pipeline as one compiled call: build the root
# (root_fn -> one batched model.forward), build the legal-move mask, then run
# gumbel_muzero_policy (which traces recurrent_fn -> one batched env_step +
# model.forward per simulation step). So a whole batch of games searches in a
# single jit boundary, exactly as the spec wants. The Gumbel/qtransform knobs
# are keyword-only (note the bare ``*`` in the mctx signature); gumbel_scale=1.0
# + the default qtransform (qtransform_completed_by_mix_value) is the correct
# Gumbel configuration. num_simulations / max_num_considered_actions are python
# ints (static to mctx), so we cache one compiled callable per (sims,
# candidates) pair instead of retracing each call.
#
# VERIFY: argument names/positions of mctx.gumbel_muzero_policy
# (params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=...,
#  *, qtransform=..., max_num_considered_actions=..., gumbel_scale=...).
# These match the verified mctx/_src/policies.py signature. qtransform is left
# at its default (qtransform_completed_by_mix_value), the correct Gumbel default.
_SEARCH_CACHE: dict[tuple[int, int], Any] = {}


def _make_search_fn(num_simulations: int, max_num_considered_actions: int):
    """Build a jitted ``(params, key, state) -> PolicyOutput`` search closure.

    Closes over the two python-int budgets (static to mctx) so each distinct
    budget compiles once. The whole body — root_fn, the invalid-actions mask,
    and gumbel_muzero_policy — lives inside the single ``jax.jit``.
    """

    def _search(params, key, state):
        root = root_fn(params, state)               # one batched model.forward
        invalid = _invalid_actions(state)           # [B, 4096], 1.0 = illegal
        return mctx.gumbel_muzero_policy(
            params,
            key,
            root,
            recurrent_fn,
            num_simulations,
            invalid_actions=invalid,
            max_num_considered_actions=max_num_considered_actions,
            gumbel_scale=1.0,
            # qtransform -> qtransform_completed_by_mix_value (Gumbel default).
        )

    return jax.jit(_search)


def _get_search_fn(num_simulations: int, max_num_considered_actions: int):
    """Memoised :func:`_make_search_fn`, keyed by the (sims, candidates) ints."""
    key = (int(num_simulations), int(max_num_considered_actions))
    fn = _SEARCH_CACHE.get(key)
    if fn is None:
        fn = _make_search_fn(key[0], key[1])
        _SEARCH_CACHE[key] = fn
    return fn


def run_search(params: Any, key: Any, state: Any, cfg: SearchConfig) -> mctx.PolicyOutput:
    """Run one batched Gumbel-MuZero search over ``state`` and return PolicyOutput.

    ``state`` is a *batched* pgx ``State`` (leading batch dim B). Returns the
    full ``mctx.PolicyOutput`` (``action[B]``, ``action_weights[B,4096]``,
    ``search_tree``). Use :func:`search_result` to extract the prophet-shaped
    SearchOut, or :func:`batched_search` to do both in one call.

    Everything (root forward, every per-simulation env step + child forward,
    and the Gumbel top-k / sequential-halving selection) runs inside one
    ``jax.jit`` boundary, so the whole batch searches in a single compiled call.
    """
    search_fn = _get_search_fn(cfg.sims, cfg.root_candidates)
    return search_fn(params, key, state)


def deep_search(params: Any, key: Any, state: Any, cfg: SearchConfig) -> mctx.PolicyOutput:
    """Deep-reflection variant: identical to :func:`run_search` but at the deep
    budget (``cfg.deep_sims`` simulations, ``cfg.deep_candidates`` root
    candidates). Used by reflection.py to re-analyse surprising positions.
    """
    deep_sims = getattr(cfg, "deep_sims", 128)
    deep_candidates = getattr(cfg, "deep_candidates", 16)
    search_fn = _get_search_fn(deep_sims, deep_candidates)
    return search_fn(params, key, state)


# --------------------------------------------------------------------------
# SearchResult-equivalent extraction (prophet's SearchResult -> dense SearchOut).
# --------------------------------------------------------------------------
class SearchOut(NamedTuple):
    """Dense, fully-batched analogue of prophet's ``SearchResult``.

    All arrays carry the leading batch dim B. Unlike the reference (which uses
    ragged per-position ``legal_indices`` / ``q_indices``), the JAX port keeps
    everything dense over the 4096 action space so it is a clean jit/vmap
    pytree and a fixed-shape training contract.

      * ``move_index``    int32[B]   — chosen action (``policy_output.action``).
      * ``policy_target`` f32[B,4096] — the Gumbel-improved distribution
        (``policy_output.action_weights``); already sums to 1 over legal moves.
        **This is the policy training target.**
      * ``root_value``    f32[B]     — backed-up root value
        (``tree.node_values[:, ROOT_INDEX]``).
      * ``q_target``      f32[B,4096] — root-perspective empirical Q per action
        (``tree.summary().qvalues``); dense, 0 where the qtransform leaves it.
      * ``q_weight``      f32[B,4096] — child visit counts at the root
        (``tree.children_visits[:, ROOT_INDEX]``); the per-action Q regression
        weight (0 for unvisited children).
      * ``q_head_played`` f32[B]     — the network Q-head's value of the played
        move (``gather(q[B,4096], action)``), for Q-surprise detection.
    """

    move_index: jnp.ndarray      # int32[B]
    policy_target: jnp.ndarray   # f32[B, 4096]
    root_value: jnp.ndarray      # f32[B]
    q_target: jnp.ndarray        # f32[B, 4096]
    q_weight: jnp.ndarray        # f32[B, 4096]
    q_head_played: jnp.ndarray   # f32[B]


def _extract_tree_stats(policy_output: mctx.PolicyOutput):
    """Pull the policy/value/Q stats out of a PolicyOutput + its tree.

    Returns ``(move_index[B], policy_target[B,4096], root_value[B],
    q_target[B,4096], q_weight[B,4096])`` — everything except the network
    Q-head's ``q_head_played`` (which needs ``params``; see callers).
    """
    tree = policy_output.search_tree
    move_index = policy_output.action.astype(jnp.int32)            # [B]
    policy_target = policy_output.action_weights                   # [B, 4096]

    root_index = mctx.Tree.ROOT_INDEX                              # == 0
    root_value = tree.node_values[:, root_index]                  # [B]

    # Per-action root-perspective Q (the qtransform's completed Q) and the
    # per-child visit counts at the root.
    # VERIFY: SearchSummary fields are (visit_counts, visit_probs, value,
    # qvalues); qvalues is [B, num_actions] root-perspective Q.
    summary = tree.summary()
    q_target = summary.qvalues                                    # [B, 4096]
    q_weight = tree.children_visits[:, root_index].astype(jnp.float32)  # [B, 4096]

    return move_index, policy_target, root_value, q_target, q_weight


def search_result(
    policy_output: mctx.PolicyOutput,
    state: Any,
    params: Any = None,
) -> SearchOut:
    """Extract the prophet-shaped, batched :class:`SearchOut` from a PolicyOutput.

    Signature matches the spec's ``search_result(policy_output, state)``;
    ``params`` is an optional third argument needed only to recompute
    ``q_head_played`` (the network Q-head's value of the played move), since
    mctx stores the *empirical/completed* Q in the tree, not the raw network
    Q-head. If ``params`` is omitted, ``q_head_played`` is returned as zeros
    (and you should prefer :func:`batched_search`, which always threads params).

    Reads (all ``[B, ...]``):

      * ``move_index``    = ``policy_output.action``.
      * ``policy_target`` = ``policy_output.action_weights`` (Gumbel-improved,
        sums to 1 over legal — the train target).
      * ``root_value``    = ``tree.node_values[:, Tree.ROOT_INDEX]``.
      * ``q_target``      = ``tree.summary().qvalues`` (root-perspective Q).
      * ``q_weight``      = ``tree.children_visits[:, Tree.ROOT_INDEX]`` (visit
        counts per child; 0 -> unvisited, used as the Q-regression weight so
        only verified children contribute).
      * ``q_head_played`` = ``gather(q4096, action)`` from one extra batched
        ``model.forward`` on the root ``state`` (or zeros if ``params is None``).

    ``state`` is the *root* batched pgx ``State`` (the embedding originally fed
    to :func:`run_search`); used only to recompute the root Q-head.
    """
    move_index, policy_target, root_value, q_target, q_weight = _extract_tree_stats(
        policy_output
    )

    if params is None:
        # No params -> cannot read the network Q-head; surface zeros rather than
        # a hard error so the bare two-arg call still returns a valid SearchOut.
        q_head_played = jnp.zeros_like(root_value)
    else:
        _logits, q4096, _v = model_forward(params, encode_state(state))  # [B,4096]
        batch = move_index.shape[0]
        q_head_played = q4096[jnp.arange(batch), move_index]             # [B]

    return SearchOut(
        move_index=move_index,
        policy_target=policy_target,
        root_value=root_value,
        q_target=q_target,
        q_weight=q_weight,
        q_head_played=q_head_played,
    )


def batched_search(params: Any, key: Any, state: Any, cfg: SearchConfig) -> SearchOut:
    """``run_search`` + :func:`search_result` in one call: PolicyOutput -> SearchOut.

    This is the high-level entry the rest of prophet_jax uses (selfplay /
    reflection). The whole batch of games searches in a single compiled mctx
    call, then the dense, batched SearchOut is extracted (one extra batched
    forward for the root Q-head used in ``q_head_played``).
    """
    policy_output = run_search(params, key, state, cfg)
    return search_result(policy_output, state, params)


def batched_deep_search(params: Any, key: Any, state: Any, cfg: SearchConfig) -> SearchOut:
    """Deep-reflection analogue of :func:`batched_search` (uses ``deep_search``).

    Re-analyses positions at ``cfg.deep_sims`` / ``cfg.deep_candidates`` and
    returns the same dense SearchOut, so reflection.py can build high-weight
    study samples with sharper targets.
    """
    policy_output = deep_search(params, key, state, cfg)
    return search_result(policy_output, state, params)
