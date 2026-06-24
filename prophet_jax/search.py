"""Batched Gumbel-MuZero search (mctx) reproducing prophet's search semantics.

This is the JAX port of ``prophet/search.py``. The reference engine runs a
Gumbel top-k + sequential-halving selection at the *root* and PUCT in the
*interior*, with unvisited children seeded by the network's per-move Q-head
(``q_trust * q_init``). We reproduce that as closely as the Gumbel-MuZero
*formulation* allows by wrapping ``mctx.gumbel_muzero_policy`` and running it
fully batched / vmapped over thousands of positions in a single compiled call.

What maps closely
-----------------
* **Root algorithm.** ``gumbel_muzero_policy`` *natively* does Gumbel top-k
  candidate selection + Sequential Halving over
  ``min(num_simulations, max_num_considered_actions)`` actions. Setting
  ``num_simulations = cfg.sims`` (32), ``max_num_considered_actions =
  cfg.root_candidates`` (8) and ``gumbel_scale = 1.0`` makes mctx's root match
  prophet's root algorithm (Gumbel draw on raw logits, deterministic argmax of
  the halving). This *is* what the policy does out of the box.
* **Action space.** We keep mctx's action space equal to prophet's 4096
  (``from*64 + to``). mctx only has an explicit invalid-action mask at the
  root, so we also mask policy logits at every expanded child node; otherwise
  interior simulations can select illegal chess moves and poison Q targets with
  pgx illegal-action terminal rewards. The env layer handles the 4096 -> pgx
  4672 remap inside ``env_step``; mctx never sees pgx's 4672 layout.
* **Q-head completion.** Each mctx node embedding carries the raw per-action
  Q-head vector for that node. The custom qtransform completes unvisited
  children with ``q_trust * q_init`` and uses empirical tree Q for visited
  children, matching prophet's first-play-value idea while keeping mctx's
  batched tree machinery.
* **Terminal handling.** ``recurrent_fn`` returns ``discount = 0`` at terminal
  children (cutting the value bootstrap, matching prophet's terminal handling)
  and routes the true terminal outcome through ``reward`` from the parent's
  perspective, so a mating move backs up as +1.
* **Policy target.** ``policy_output.action_weights`` (the Gumbel-improved,
  completed-Q distribution that already sums to 1 over legal moves) is the
  training target — the analogue of prophet's ``policy_target``.

Remaining approximation
-----------------------
mctx's interior action-selection rule is still Full Gumbel MuZero, not
prophet's exact Python PUCT formula. The important load-bearing semantics are
preserved: legal-only child selection, two-player negamax backup, and Q-head
first-play completion for unvisited actions.

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
# env.py provides the pgx State half of the mctx embedding.
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

        The interior-PUCT knobs are retained for interface parity. ``c_visit``,
        ``c_scale``, and ``q_trust`` are used by the custom qtransform; mctx's
        interior action-selection rule itself is still Gumbel MuZero rather
        than prophet's exact Python PUCT formula.
        """

        sims: int = 32
        root_candidates: int = 8
        c_puct: float = 1.5
        c_visit: float = 50.0
        c_scale: float = 1.0
        q_trust: float = 1.0
        deep_sims: int = 128
        deep_candidates: int = 16


class _SearchEmbedding(NamedTuple):
    """mctx node embedding: the pgx state plus this node's raw Q-head vector."""

    state: Any
    q_init: jnp.ndarray


# mctx marks ILLEGAL actions with 1.0 in ``invalid_actions`` (legal -> 0.0).
# legal_mask(state) is True at LEGAL actions, so invalid = logical-not.
def _invalid_actions(state: Any) -> jnp.ndarray:
    """[B, NUM_ACTIONS] float32, 1.0 at illegal actions (mctx convention)."""
    return (~legal_mask(state)).astype(jnp.float32)


def _mask_illegal_logits(logits: jnp.ndarray, state: Any) -> jnp.ndarray:
    """Mask illegal actions in logits for both root and interior mctx nodes."""
    invalid = _invalid_actions(state).astype(bool)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid, min_logit, logits)


def _qtransform_qinit(
    tree: mctx.Tree,
    node_index: jnp.ndarray,
    *,
    q_trust: float,
    c_visit: float,
    c_scale: float,
) -> jnp.ndarray:
    """Reference-style completed Q for mctx action selection.

    Prophet's Python search completes an unvisited child with
    ``q_trust * q_init`` from the parent node's Q-head. mctx's default
    completion only has scalar V, so the Q-head was not participating in JAX
    search at all. We carry each node's raw Q-head in the mctx embedding and use
    it here for unvisited actions; visited actions still use empirical tree Q.
    """
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    q_init = tree.embeddings.q_init[node_index]
    completed = jnp.where(visit_counts > 0, qvalues, q_trust * q_init)
    visit_scale = c_visit + jnp.max(visit_counts, axis=-1)
    return visit_scale * c_scale * completed


# --------------------------------------------------------------------------
# Root and recurrent functions (the mctx model contract).
# --------------------------------------------------------------------------
def root_fn(params: Any, state: Any) -> mctx.RootFnOutput:
    """Build the mctx root from a batched pgx ``State``.

    Runs ``model.forward`` on ``encode_state(state)`` to get
    ``(policy_logits[B,4096], q[B,4096], v[B])`` and returns:

      * ``prior_logits`` = policy logits with illegal actions masked out. mctx
        also receives a root invalid-action mask, but child nodes need their
        logits pre-masked because mctx has no child invalid-action argument.
      * ``value``        = ``v`` (the network's P(win) - P(loss) scalar in
        ``[-1, 1]`` from the side-to-move's perspective).
      * ``embedding``    = ``(state, q)``. mctx treats this as an opaque pytree;
        the custom qtransform reads ``q`` back out for unvisited-action
        completion.
    """
    x = encode_state(state)  # f32[B, 64, FEATURES]
    policy_logits, q, v = model_forward(params, x)
    return mctx.RootFnOutput(
        prior_logits=_mask_illegal_logits(policy_logits, state),  # [B, 4096]
        value=v,                             # [B]
        embedding=_SearchEmbedding(state=state, q_init=q),  # opaque to mctx
    )


def recurrent_fn(params: Any, rng_key: Any, action: jnp.ndarray, embedding: Any):
    """One simulation step: apply ``action`` to ``state`` and evaluate the child.

    ``action`` is a batched prophet action index ``[B]`` (``from*64 + to``).
    Contract (mctx): return ``(RecurrentFnOutput, next_embedding)`` in that
    order, where ``next_embedding`` is the child ``(pgx State, q_head)`` pair.

    Field-by-field (all ``[B]`` / ``[B, 4096]``):

      * ``reward``   = terminal outcome in the parent's perspective, else 0.
      * ``discount`` = ``where(child_terminal, 0.0, -1.0)``. The negative
        discount is the negamax sign flip for non-terminal children.
      * ``prior_logits`` = the child's policy logits with illegal actions masked.
      * ``value``    = ``child_v`` for non-terminals and 0 at terminals because
        terminal outcomes are carried by ``reward``.

    NOTE on sign convention. pgx and prophet both express the child value from
    the *child's* side-to-move perspective. mctx backs up ``reward + discount *
    value``; using ``discount = -1`` flips child values into the parent's frame.
    ``terminal_info`` returns the side-to-move terminal value at the child
    (checkmate -> -1.0, draw -> 0.0), so ``reward = -terminal_value`` gives a
    mating move +1 for the parent.
    """
    state = embedding.state

    # action: [B] int32 (prophet index). env_step maps 4096 -> pgx 4672 and
    # steps without auto-reset.
    next_state = env_step(state, action.astype(jnp.int32))

    x_child = encode_state(next_state)                 # f32[B, 64, F]
    child_logits, child_q, child_v = model_forward(params, x_child)

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
        prior_logits=_mask_illegal_logits(child_logits, next_state),  # [B, 4096]
        value=value,                                   # [B] network value (non-terminal)
    )
    return out, _SearchEmbedding(state=next_state, q_init=child_q)


# --------------------------------------------------------------------------
# The compiled search call.
# --------------------------------------------------------------------------
# We jit the ENTIRE search pipeline as one compiled call: build the root
# (root_fn -> one batched model.forward), build the legal-move mask, then run
# gumbel_muzero_policy (which traces recurrent_fn -> one batched env_step +
# model.forward per simulation step). So a whole batch of games searches in a
# single jit boundary, exactly as the spec wants. The Gumbel/qtransform knobs
# are keyword-only (note the bare ``*`` in the mctx signature); gumbel_scale=1.0
# keeps the reference root exploration. num_simulations / candidates / Q-scale
# knobs are static to the compiled function, so we cache by all of them.
#
# VERIFY: argument names/positions of mctx.gumbel_muzero_policy
# (params, rng_key, root, recurrent_fn, num_simulations, invalid_actions=...,
#  *, qtransform=..., max_num_considered_actions=..., gumbel_scale=...).
# These match the verified mctx/_src/policies.py signature. We pass a custom
# qtransform so unvisited children use prophet's Q-head first-play value.
_SEARCH_CACHE: dict[tuple[int, int, float, float, float], Any] = {}


def _make_search_fn(
    num_simulations: int,
    max_num_considered_actions: int,
    q_trust: float,
    c_visit: float,
    c_scale: float,
):
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
            qtransform=lambda tree, node_index: _qtransform_qinit(
                tree,
                node_index,
                q_trust=q_trust,
                c_visit=c_visit,
                c_scale=c_scale,
            ),
        )

    return jax.jit(_search)


def _get_search_fn(
    num_simulations: int,
    max_num_considered_actions: int,
    q_trust: float,
    c_visit: float,
    c_scale: float,
):
    """Memoised :func:`_make_search_fn`, keyed by static search knobs."""
    key = (
        int(num_simulations),
        int(max_num_considered_actions),
        float(q_trust),
        float(c_visit),
        float(c_scale),
    )
    fn = _SEARCH_CACHE.get(key)
    if fn is None:
        fn = _make_search_fn(key[0], key[1], key[2], key[3], key[4])
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
    search_fn = _get_search_fn(
        cfg.sims,
        cfg.root_candidates,
        cfg.q_trust,
        cfg.c_visit,
        cfg.c_scale,
    )
    return search_fn(params, key, state)


def deep_search(params: Any, key: Any, state: Any, cfg: SearchConfig) -> mctx.PolicyOutput:
    """Deep-reflection variant: identical to :func:`run_search` but at the deep
    budget (``cfg.deep_sims`` simulations, ``cfg.deep_candidates`` root
    candidates). Used by reflection.py to re-analyse surprising positions.
    """
    deep_sims = getattr(cfg, "deep_sims", 128)
    deep_candidates = getattr(cfg, "deep_candidates", 16)
    search_fn = _get_search_fn(
        deep_sims,
        deep_candidates,
        cfg.q_trust,
        cfg.c_visit,
        cfg.c_scale,
    )
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
