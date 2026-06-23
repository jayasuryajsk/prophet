#!/usr/bin/env python3
"""Minimal CPU sanity check for the prophet_jax port — NO GPU required.

Runs the three load-bearing cross-module paths at tiny dims so the whole thing
finishes on a laptop CPU in well under a minute:

  1. build the Flax model           (model.build_model)
  2. one batched mctx search step    (search.batched_search over a pgx batch)
  3. one optax train step            (train.train_step on a dense SamplesBatch)

It also runs a tiny end-to-end self-play rollout (selfplay.generate_selfplay)
and a reflection pass (reflection.reflect_batch) when those import cleanly, so
the env<->search<->selfplay<->train interface contract is exercised together.

This is a SMOKE test (does it wire up and run?), not a correctness test. The
real correctness check is the env-bridge parity harness described in env.py
(encode/legal/action-map vs python-chess) — that needs pgx + python-chess and a
known-position fixture and is out of scope here.

Usage:
    python -m prophet_jax.smoke_test          # quiet-ish, exits 0 on success
    python prophet_jax/smoke_test.py -v       # verbose shapes

Requires jax/flax/optax/mctx/pgx installed (CPU wheels are fine — see
requirements.txt). It pins JAX to CPU so it never touches a GPU.
"""

from __future__ import annotations

# --- force CPU + jax-first import (the package import-order guard) ----------
# Pin CPU BEFORE importing jax/prophet_jax so the test is GPU-independent and
# deterministic. Must precede any numpy import (prophet_jax enforces this too).
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # CPU only — this is a no-GPU sanity check
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import sys
import traceback

import jax
import jax.numpy as jnp

# Import the package (this triggers the jax-before-numpy guard) and the pieces
# we exercise. Done at top level so an ImportError fails loudly and early.
import prophet_jax  # noqa: F401  (ensures the aggregator import path works)
from prophet_jax import config as cfg_mod
from prophet_jax import model as model_mod
from prophet_jax import env as env_mod
from prophet_jax import search as search_mod
from prophet_jax import train as train_mod


# Tiny architecture: a 2-layer/32-dim transformer keeps init + forward cheap
# while still exercising every code path (attention, the 3 heads, the loader-
# shaped param tree). in_features stays at the real 24 so encode_state lines up.
TINY_CFG = cfg_mod.ModelConfig(
    d_model=32,
    n_layers=2,
    n_heads=2,
    d_ff=64,
    head_dim=16,
    dropout=0.0,
    in_features=cfg_mod.FEATURES,  # 24 — must match encode_state's output width
)

B = 4               # parallel games / batch rows (tiny)
NUM_ACTIONS = cfg_mod.NUM_ACTIONS  # 4096
FEATURES = cfg_mod.FEATURES        # 24

# Tiny search budget: a couple of sims / candidates is enough to confirm the
# mctx root + recurrent_fn wiring and that a whole batch searches in one call.
TINY_SCFG = cfg_mod.SearchConfig(sims=2, root_candidates=2)


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}", flush=True)


def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return "?"


def stage_build_model(verbose: bool):
    """Stage 1 — build the model and confirm a batched forward runs."""
    key = jax.random.PRNGKey(0)
    model, params = model_mod.build_model(TINY_CFG, key)  # also registers default
    n = model_mod.num_params(params)
    assert n > 0, "model has no parameters"

    # Both forward conventions must work: the 3-arg (train.py) and the 2-arg
    # (search.py) form that relies on the just-registered default model.
    x = jnp.zeros((B, 64, FEATURES), dtype=jnp.float32)
    p3, q3, v3, wdl3 = model_mod.forward_wdl(model, params, x)  # explicit model
    p2, q2, v2 = model_mod.forward(params, x)                   # default model
    assert _shape(p3) == (B, NUM_ACTIONS), f"policy shape {_shape(p3)}"
    assert _shape(q3) == (B, NUM_ACTIONS), f"q shape {_shape(q3)}"
    assert _shape(v3) == (B,), f"v shape {_shape(v3)}"
    assert _shape(wdl3) == (B, 3), f"wdl shape {_shape(wdl3)}"
    # 2-arg (default-model) path must agree with the explicit-model path.
    assert jnp.allclose(p2, p3) and jnp.allclose(v2, v3), \
        "2-arg forward(params,x) disagrees with 3-arg forward(model,params,x)"
    # WDL is a probability distribution; v = P(win) - P(loss) in [-1, 1].
    assert jnp.allclose(wdl3.sum(-1), 1.0, atol=1e-4), "wdl rows must sum to 1"
    assert bool((jnp.abs(v3) <= 1.0 + 1e-4).all()), "v out of [-1,1]"

    if verbose:
        print(f"    params={n:,} d_model={TINY_CFG.d_model} "
              f"n_layers={TINY_CFG.n_layers}")
    _ok(f"build_model + forward (both arities) — {n:,} params")
    return model, params


def stage_search(params, verbose: bool):
    """Stage 2 — one batched mctx search step over a real pgx batch."""
    env = env_mod.make_chess_env()  # registers the process-default env
    key = jax.random.PRNGKey(1)
    keys = env_mod.start_keys(key, B)
    states = env_mod.env_init(keys)  # handle-less form (uses default env)

    # Sanity on the bridge before search: encode / legal mask / terminal info.
    x = env_mod.encode_state(states)
    legal = env_mod.legal_mask(states)
    is_term, term_v = env_mod.terminal_info(states)
    assert _shape(x) == (B, 64, FEATURES), f"encode shape {_shape(x)}"
    assert _shape(legal) == (B, NUM_ACTIONS), f"legal shape {_shape(legal)}"
    # From the start position every game has legal moves and is non-terminal.
    assert bool(legal.any(axis=-1).all()), "a start position has no legal moves?"
    assert not bool(is_term.any()), "start position flagged terminal"

    skey = jax.random.PRNGKey(2)
    out = search_mod.batched_search(params, skey, states, TINY_SCFG)
    # SearchOut field shapes (the training-target contract consumed by selfplay).
    assert _shape(out.move_index) == (B,), f"move_index {_shape(out.move_index)}"
    assert _shape(out.policy_target) == (B, NUM_ACTIONS), \
        f"policy_target {_shape(out.policy_target)}"
    assert _shape(out.root_value) == (B,), f"root_value {_shape(out.root_value)}"
    assert _shape(out.q_target) == (B, NUM_ACTIONS), f"q_target {_shape(out.q_target)}"
    assert _shape(out.q_weight) == (B, NUM_ACTIONS), f"q_weight {_shape(out.q_weight)}"
    assert _shape(out.q_head_played) == (B,), \
        f"q_head_played {_shape(out.q_head_played)}"
    # The chosen move must be legal in its position; the policy target must be a
    # distribution over the legal moves.
    chosen_legal = legal[jnp.arange(B), out.move_index]
    assert bool(chosen_legal.all()), "search chose an illegal move"
    assert jnp.all(jnp.isfinite(out.policy_target)), "non-finite policy target"
    assert jnp.allclose(out.policy_target.sum(-1), 1.0, atol=1e-3), \
        "policy target rows must sum to ~1"

    if verbose:
        print(f"    move_index={np_list(out.move_index)} "
              f"root_value={np_list(out.root_value)}")
    _ok("batched_search — one compiled mctx search over a pgx batch")
    return env, states, out


def stage_train(model, params, states, out, verbose: bool):
    """Stage 3 — one optax train step on a dense SamplesBatch.

    Builds a minimal-but-valid SamplesBatch directly from the search output and
    the env encodings (rather than running full self-play), so the loss/optax
    path is tested in isolation and fast. Confirms the loss is finite and the
    params actually moved.
    """
    SamplesBatch = train_mod.SamplesBatch  # canonical flax.struct dataclass

    x = env_mod.encode_state(states).astype(jnp.float32)          # [B,64,F]
    child_states = env_mod.env_step(states, out.move_index)       # handle-less
    child_x = env_mod.encode_state(child_states).astype(jnp.float32)
    mask = env_mod.legal_mask(states)                            # [B,A] bool

    n = B
    batch = SamplesBatch(
        x=x,
        child_x=child_x,
        played=out.move_index.astype(jnp.int32),
        value=out.root_value.astype(jnp.float32),
        weight=jnp.ones((n,), jnp.float32),
        # Give a couple of rows a known WDL class so the WDL CE term is active;
        # -1 means "unknown / excluded", which must also be handled. Tiled to N
        # so this stays valid if B changes.
        wdl=jnp.asarray([( [2, 0, -1, 1][i % 4]) for i in range(n)], dtype=jnp.int32),
        mask=mask,
        policy=out.policy_target.astype(jnp.float32),
        q_target=out.q_target.astype(jnp.float32),
        q_weight=out.q_weight.astype(jnp.float32),
        valid=jnp.ones((n,), bool),
    )

    state = train_mod.make_train_state(model, params, lr=1e-3)
    weights = train_mod.loss_weights_to_array(cfg_mod.LossWeights())

    new_state, metrics = train_mod.train_step(state, batch, weights)

    for k, v in metrics.items():
        assert jnp.isfinite(v), f"loss term {k!r} is not finite: {v}"
    # Params must have moved after a step (non-zero gradient somewhere).
    moved = jax.tree_util.tree_reduce(
        lambda acc, leaf: acc or bool(leaf),
        jax.tree_util.tree_map(
            lambda a, b: bool(jnp.any(a != b)), new_state.params, state.params
        ),
        False,
    )
    assert moved, "params did not change after a train step"
    assert int(new_state.step) == int(state.step) + 1, "step did not increment"

    if verbose:
        print("    losses: " + ", ".join(
            f"{k}={float(v):.4f}" for k, v in metrics.items()))
    _ok("train_step — clip+adamw+EMA optax step, loss finite, params updated")
    return new_state


def stage_selfplay_and_reflect(params, verbose: bool):
    """Optional integration stage — tiny generate_selfplay + reflect_batch.

    Exercises the full env<->search<->selfplay<->reflection<->train-contract
    loop at tiny dims. Reported separately so a failure here is informative but
    does not mask the core 3 stages above.
    """
    from prophet_jax import selfplay as selfplay_mod
    from prophet_jax import reflection as reflection_mod

    spcfg = cfg_mod.SelfPlayConfig(max_plies=4)   # 4-ply rollout
    stcfg = cfg_mod.StudyConfig(top_k=1, deep_sims=2, deep_candidates=2,
                                branch_plies=2, n_lines=1)
    key = jax.random.PRNGKey(3)

    samples, meta = selfplay_mod.generate_selfplay(
        params, key, B, TINY_SCFG, spcfg, gate=True
    )
    N = B * spcfg.max_plies
    assert _shape(samples.x) == (N, 64, FEATURES), f"samples.x {_shape(samples.x)}"
    assert _shape(samples.valid) == (N,), f"samples.valid {_shape(samples.valid)}"
    assert _shape(meta.plies) == (B,), f"meta.plies {_shape(meta.plies)}"
    assert _shape(meta.root_values)[0] == B, "meta.root_values batch dim wrong"
    if verbose:
        print(f"    selfplay: N={N} valid={int(samples.valid.sum())} "
              f"plies={np_list(meta.plies)}")
    _ok(f"generate_selfplay — {int(samples.valid.sum())}/{N} valid samples")

    rkey = jax.random.PRNGKey(4)
    study = reflection_mod.reflect_batch(
        params, rkey, meta, meta.states_per_ply, stcfg, TINY_SCFG
    )
    # Reflection returns the same dense SamplesBatch schema (possibly all-padding
    # if no ply was surprising at these tiny dims — that is still a valid run).
    assert hasattr(study, "valid"), "reflect_batch did not return a SamplesBatch"
    if verbose:
        print(f"    reflect: rows={_shape(study.valid)[0]} "
              f"valid={int(study.valid.sum())}")
    _ok("reflect_batch — deep-reflection pass over the self-play meta")

    # Train once on a replay batch drawn from a buffer fed by both sources, to
    # close the loop end to end through the host-side ReplayBuffer + sampler.
    buf = train_mod.ReplayBuffer(capacity=4096)
    buf.add(samples)
    buf.add(study)
    if len(buf) > 0:
        model = model_mod.get_default_model()
        state = train_mod.make_train_state(model, params, lr=1e-3)
        weights = train_mod.loss_weights_to_array(cfg_mod.LossWeights())
        bkey = jax.random.PRNGKey(5)
        bs = min(8, len(buf))
        rbatch = train_mod.sample_batch(buf, bkey, bs)
        state, metrics = train_mod.train_step(state, rbatch, weights)
        assert all(bool(jnp.isfinite(v)) for v in metrics.values()), \
            "replay train step produced a non-finite loss"
        _ok(f"replay buffer add + sample_batch + train_step (buf={len(buf)})")


def np_list(a):
    import numpy as _np
    return _np.asarray(a).tolist()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verbose = ("-v" in argv) or ("--verbose" in argv)

    print(f"prophet_jax smoke test  (jax {jax.__version__}, "
          f"backend={jax.default_backend()}, B={B}, "
          f"d_model={TINY_CFG.d_model})", flush=True)
    if jax.default_backend() != "cpu":
        print("  note: backend is not CPU; this test pins JAX_PLATFORMS=cpu but "
              "a device was still selected — that's fine, it still validates.",
              flush=True)

    try:
        print("[1/4] build model", flush=True)
        model, params = stage_build_model(verbose)

        print("[2/4] batched mctx search", flush=True)
        env, states, out = stage_search(params, verbose)

        print("[3/4] optax train step", flush=True)
        stage_train(model, params, states, out, verbose)

        print("[4/4] self-play + reflection (integration)", flush=True)
        stage_selfplay_and_reflect(params, verbose)

    except Exception as exc:  # noqa: BLE001 - surface the first failure clearly
        print("\nSMOKE TEST FAILED", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1

    print("\nSMOKE TEST PASSED — model builds, search runs, train step runs.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
