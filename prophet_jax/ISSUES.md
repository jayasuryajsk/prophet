# prophet_jax — status & open bugs (handoff)

A JAX/GPU rewrite of prophet (the PyTorch engine in `../prophet/`), built to run
self-play ~50-90× faster. Stack: **flax** (model) + **pgx** (chess env) +
**mctx** (Gumbel-MuZero search) + **optax**. Validated on an A100-80GB.

## ✅ What works
- Full pipeline runs end-to-end: model + pgx env + mctx search + vmapped
  self-play + deep reflection + optax training. `smoke_test.py` passes.
- **Throughput: ~94× the Mac** (565 self-play games/min at batch 1024-2048,
  sims=32, 10M model). Saturates ~B1024 (memory-bandwidth-bound).
- Torch checkpoint export works: `latest.pt` loads into the PyTorch
  `prophet.model` and runs (so the gauntlet/openings tooling works on it).
- pgx terminal/reward detection verified correct (4 real mates → mated side
  gets value −1 via `env.terminal_info`).

## 🔴 THE OPEN BUG (engine does not train properly yet)
**Self-play produces ~0% decisive games** (all draws, ~180-200 plies) where the
PyTorch run is **71-100% decisive** at the same early stage. With no wins/losses,
the value & Q heads can't learn to *win* — only the policy head trains.

### 2026-06-24 Codex fix update
- Fixed the remaining search corruption path: mctx only applies
  `invalid_actions` at the root, so JAX interior simulations were free to pick
  illegal chess actions. Those illegal actions went through the `env_step`
  guard/clamp and could inject fake terminal rewards into root Q targets.
  `search.py` now masks policy logits at every expanded node.
- Wired the Q-head back into search. Each mctx node embedding now carries that
  node's raw per-action Q-head vector, and the custom qtransform completes
  unvisited actions with `q_trust * q_init` instead of mctx's scalar-V mixed
  value. `q_trust`, `c_visit`, and `c_scale` are now live search knobs.
- Restored PyTorch-reference truncation semantics: a self-play ply cap is
  unknown (`z_white = NaN`, `result = -1`), not a fake draw. The replay rows are
  still kept, but outcome blending/WDL supervision are skipped for those games.
- Local CPU validation passed: `python -m prophet_jax.smoke_test`, a mate-in-1
  search probe at full candidate coverage, no nonzero fake rewards on visited
  start-position edges, and a tiny `prophet_jax.train` run.

### Root cause = wrong sign/convention in the **mctx two-player search**, partly fixed:
1. **FIXED — negamax discount.** `search.py recurrent_fn` returned
   `discount = +1` for non-terminal children; a two-player zero-sum game needs
   `discount = −1` so mctx flips the child value's sign each ply. With +1 the
   search backs up the opponent's value as its own → both sides *cooperate* →
   nobody seeks the win. **Fix applied:** `discount = where(is_terminal, 0, −1)`;
   terminal outcome routed through `reward = where(is_terminal, −terminal_value, 0)`;
   `value = where(is_terminal, 0, child_v)`.
   - **Effect:** woke the heads up (q-loss 0.000 → ~0.12, v 10×) — BUT decisive
     rate stayed 0%, so this was *a* bug, not the *whole* bug.
2. **STILL BROKEN — a residual two-player sign/convention mismatch.** Symptom:
   **consistency loss stuck at ~1.07** (PyTorch ≈ 0.075). The consistency term is
   `(q_played + v_child)²` (formula is correct), so Q-head and value-head signals
   still **disagree** → likely one of:
   - mctx may want the **value pre-negated** in `recurrent_fn` with `discount=+1`,
     rather than `discount=−1` (verify which mechanism *this* mctx version uses —
     read `mctx/_src/search.py` backup + `qtransform_completed_by_mix_value`).
   - the **q_target perspective** extracted from the mctx tree (`search.py`
     `SearchOut.q_target`) may be in the child's frame, not the root's.
   - the qtransform (`qtransform_completed_by_mix_value`) interaction with a
     negative discount.

### 👉 NEXT STEP (do this BEFORE any more full runs)
Build a tiny **search-on-mate test rig** — no training loop, runs in seconds:
load a known mate-in-1 (need pgx FEN or hand-build the State), run
`search.batched_search`, assert it **picks the mating move** and `root_value`
≈ +1. Iterate the mctx value convention against *that* (seconds), not against a
14-hour 100k run ($). Once the search reliably finds mates, decisive-rate will
follow and the value head will learn.

## 🟡 Secondary issues (not blocking, but known)
- **Memory: OOM beyond batch ~2048 even on 80GB.** Hog = dense per-ply training
  targets `policy/q_target/q_weight/mask` each `[B, T, 4096]` f32 (~22 GiB at
  B4096). Fix later with sparse (legal-only) or bf16 target storage. Not needed
  for correctness; throughput already saturates ~B1024.
- **Throughput is memory-bandwidth-bound, not compute-bound** (declines slightly
  with batch). The dense-target fix above would also raise the throughput ceiling.
- **Bugs already fixed along the way** (keep, don't regress):
  - Tracer leak: pgx lazily imports its chess module on first `pgx.make("chess")`;
    if that happens inside the jitted self-play, pgx constants get traced & leak.
    Fix: `env.py` eagerly calls `make_chess_env()` at import.
  - `states_per_ply` stored full pgx State incl `observation[8,8,119]` (~30KB):
    `selfplay._lite_state` strips it; `reflection` re-inflates to zeros before
    `batched_search` (mctx threads the State as its embedding, needs the shape).
  - Truncated games were NaN-masked out of training: `selfplay._terminal_z_white`
    now scores a truncated game as a draw (0) so it isn't dropped. (Note: with the
    search fixed, fewer games should truncate anyway.)
  - `max-plies`: pgx random games need ~225 plies to mate; keep `--max-plies`
    high enough (≥160) that games can finish. (Was set to 120 → made it worse.)

## Run notes
- Deps: `pip install -U "jax[cuda12]" pgx mctx flax optax chex` (Python 3.12,
  validated jax 0.10.2 / pgx 2.6 / flax 0.12).
- Always `python3 -u` (stdout buffers otherwise → looks hung; the on-disk
  `runs/<out>/metrics.csv` + `progress.json` are the real progress signal).
- Smoke test: `JAX_PLATFORMS=cpu python3 -m prophet_jax.smoke_test`
- Throughput: `python3 -m prophet_jax.stress_test 1024`
- Train: `python3 -u -m prophet_jax.train --games 100000 --batch-games 256
  --sims 32 --max-plies 256 --d-model 320 --n-layers 8 --study --schedule
  --gate 2000 --eval-every 999999 --log-every 100 --out runs/jaxrun`
- **The metric to watch is `decisive` — it must climb off ~0% (toward 50%+),
  or the search is still broken.**
