# prophet_jax

A JAX/Flax port of the **prophet** chess engine — a compute-efficient,
AlphaZero-style self-play engine with an explicit **per-move Q ("intuition")
head** alongside the usual policy and value heads.

The reference engine (`prophet/`) is PyTorch + python-chess and plays one game
per Python generator, batching network evals *across* concurrently running
games. This port flips that inside out: the environment, search, self-play, and
training all run **on-accelerator in one process**, with thousands of games
advancing in lockstep under `vmap` inside a single `jax.lax.scan`. There are no
worker processes — "more games" is just a larger batch `B`.

The architecture and the 24-feature board encoding / 4096-action (`from*64+to`)
move space are kept **bit-for-functional-identical** to the reference, so a
PyTorch checkpoint loads into the JAX model unchanged, and a JAX-trained net
exports back to a PyTorch checkpoint that `scripts/gauntlet.py` can rate.

```
  pgx.make("chess")  ──►  env.py   ──►  prophet 24-feature [B,64,24] encoding
   (batched, jittable)                  + 4096 from*64+to action space
                                              │
                                              ▼
   model.py  PolicyQValueNet (Flax linen)  ── policy[4096] / Q[4096] / WDL value
                                              │
                                              ▼
   search.py  mctx.gumbel_muzero_policy   ── batched Gumbel-MuZero root search
                                              │
            ┌─────────────────────────────────┤
            ▼                                  ▼
   selfplay.py  one lax.scan over plies   reflection.py  "study your losses"
   (B games in lockstep) ──► SamplesBatch  (deep re-search of surprises)
                                              │
                                              ▼
   train.py  5 losses + optax adamw + EMA  ── single-process learner loop
```

## Module map

| file            | what it is |
|-----------------|------------|
| `config.py`     | Frozen dataclasses (`ModelConfig`, `SearchConfig`, `SelfPlayConfig`, `StudyConfig`, `LossWeights`), constants, and the game-count curricula (`q_trust_at` / `study_config_at` / `loss_weights_at`). Pure Python; no jax/numpy. |
| `model.py`      | Flax linen `PolicyQValueNet` (pre-LN transformer over 64 square tokens + 3 coupled heads) and torch↔flax checkpoint interop (`load_torch_checkpoint`, `export_torch_checkpoint`, native `save_checkpoint`). |
| `env.py`        | The load-bearing bridge. Wraps `pgx.make("chess")` but re-derives prophet's encoding + the 4096↔4672 action map from the pgx `State` internals. |
| `search.py`     | `mctx.gumbel_muzero_policy` wrapper. Whole batch searches in one compiled call; returns a dense `SearchOut`. |
| `selfplay.py`   | `generate_selfplay(...)` — vectorized on-device self-play (one `lax.scan`), emits dense `SamplesBatch` + `GameMeta`. |
| `reflection.py` | `find_surprises` + `reflect_batch` — deep re-analysis of surprising plies and counterfactual branches, batched across every game at once. |
| `train.py`      | The 5 training losses (ported verbatim), an optax `adamw` + global-norm-clip + EMA step, a host-side replay buffer, and the single-process `main()` learner loop. |
| `smoke_test.py` | CPU sanity check (build model → one search step → one train step). No GPU. |

## Install

Everything is in `requirements.txt`. **Pick the right `jax` line for your
hardware** (the file has both, with the GPU one active by default):

```bash
# GPU (NVIDIA, CUDA 12):
pip install -r prophet_jax/requirements.txt

# CPU-only / laptop / Apple Silicon (for the smoke test):
#   edit requirements.txt: comment out  jax[cuda12]>=...  and uncomment  jax>=...
pip install -r prophet_jax/requirements.txt
```

Core deps and why:

- **`jax[cuda12]`** (or plain **`jax`** for CPU) — array/JIT/autodiff backend.
- **`flax`** — the linen NN (`PolicyQValueNet`) and `flax.struct` pytrees
  (`SamplesBatch`, `GameMeta`).
- **`optax`** — `adamw` + `clip_by_global_norm` + `incremental_update` (the EMA).
- **`mctx`** — `gumbel_muzero_policy`, the batched Gumbel-MuZero search.
- **`pgx`** — `pgx.make("chess")`, the batched/jittable chess environment.
- **`torch` + `python-chess` + `numpy`** — used *outside* JAX only, for
  PyTorch-format checkpoint interop and the env-bridge parity test.

> **Import order matters.** `prophet_jax/__init__.py` imports `jax` *before* any
> submodule (hence before `numpy`). On Homebrew Python, numpy and jax/torch each
> bundle their own OpenMP/BLAS and only a jax-first init is stable. Keep
> `import prophet_jax` (or `import jax`) as the first heavy import in any entry
> point.

## Run the smoke test (no GPU)

This pins JAX to CPU, builds a tiny model, runs one batched mctx search step,
and runs one optax train step — plus a tiny self-play + reflection integration
pass. It finishes in well under a minute on a laptop CPU.

```bash
python -m prophet_jax.smoke_test          # exits 0 on success
python prophet_jax/smoke_test.py -v       # verbose: print shapes + losses
```

Expected tail on success:

```
SMOKE TEST PASSED — model builds, search runs, train step runs.
```

Run this **first** after installing — it shakes out interface/shape issues
before you spend GPU time. It is a wiring check, **not** a correctness check
(see *Caveats* below for the parity test that validates the env bridge).

## Run a training run

The learner is single-process: self-play, reflection, and training all run in
one process on the accelerator. Defaults mirror the reference `train_loop.py`
(lr 3e-4, wd 1e-4, train-ratio 4.0, buffer 200k, warmup 5k, batch 256, EMA
0.999, gate at 2000 games).

```bash
# Small/medium GPU smoke of the real loop (a few hundred games, default 320/8 net):
python -m prophet_jax.train --games 2000 --batch-games 256 --out runs/jax_smoke

# Moonshot-style run with the game-count curricula (q_trust / study / q-loss ramps):
python -m prophet_jax.train \
    --games 100000 --batch-games 1024 \
    --d-model 320 --n-layers 8 --n-heads 8 \
    --sims 32 --candidates 8 --max-plies 160 \
    --study --schedule \
    --out runs/jax_moonshot

# Warm-start from an existing PyTorch checkpoint (config is read FROM the ckpt):
python -m prophet_jax.train --init-from runs/torch_run/latest.pt --out runs/jax_warm
```

Useful flags (`python -m prophet_jax.train --help` for all):

- `--batch-games B` — parallel self-play games per round. This replaces the
  reference's `workers × batch_games` fan-out; **scale this up to fill the GPU**.
- `--d-model / --n-layers / --n-heads` — model size (config-driven; ignored when
  `--init-from` supplies a checkpoint whose `config` is loaded instead).
- `--study` — enable "study your losses" deep reflection (after the gate).
- `--schedule` — turn on the game-count curricula for `q_trust`, study
  intensity, and the Q/consistency loss weights.
- `--sync-every / --eval-every` — how often to checkpoint / write milestone
  checkpoints + a torch export + a `metrics.csv` row.
- `--no-export-torch` — skip the per-milestone PyTorch export (the export is what
  lets `scripts/gauntlet.py` rate the JAX net).

Outputs (under `--out`): `latest.npz` (native EMA checkpoint), `latest.pt`
(PyTorch export for the gauntlet), `ckpt_{games}.npz/.pt` milestones,
`progress.json`, and `metrics.csv`.

## GPU notes

- **Backend selection.** `JAX_PLATFORMS` is left unset (auto-select) by the
  package; export `JAX_PLATFORMS=cuda` to force GPU or `=cpu` to force CPU. The
  smoke test always pins CPU regardless.
- **CUDA wheels.** `jax[cuda12]` bundles the CUDA 12 runtime; you still need a
  recent enough NVIDIA driver. If `jax.default_backend()` prints `cpu` on a GPU
  box, the CUDA jax install didn't take — reinstall the `jax[cuda12]` wheel.
- **One big batch, not many processes.** Throughput comes from a large
  `--batch-games` (thousands), not from launching workers. Pick the largest `B`
  that fits memory; self-play, search, and the env all `vmap` over it.
- **Memory.** `GameMeta.states_per_ply` keeps a full pgx `State` (including its
  observation/history tensors) for every `(B, max_plies)` cell so reflection can
  re-search any ply without FENs. For very large `B × max_plies` this dominates
  host/device memory — reduce `--max-plies` or `--batch-games` if you OOM. By
  default XLA preallocates ~75% of GPU memory; set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or `XLA_PYTHON_CLIENT_PREALLOCATE=false`)
  to tune.
- **Compilation.** The first self-play round, the first search at each
  `(sims, candidates)` budget, and the first train step each trigger an XLA
  compile (tens of seconds). Schedule **bands** are static, so search/study/loss
  re-specialize only at band boundaries — a handful of recompiles over a full
  run, not per step.
- **Precision.** fp32 is the reference path (matches the torch weights). A bf16
  forward exists in `model.py` (`dtype=` arg) but is best-effort, not bit-matched
  to torch's CUDA bf16 — leave it fp32 unless you specifically want the speedup.

## Caveats (read before trusting a run)

This port was authored in an environment where **jax/flax/optax/mctx/pgx were
not installed**, so the modules are written against the verified library APIs but
have **not been executed end to end**. Before a real run:

1. **Run `smoke_test.py`** (CPU) to catch interface/shape/JIT issues.
2. **Write and run the env-bridge parity test** (described at the top of
   `env.py`). It must cross-check, over random positions, that `encode_state`,
   `legal_mask`, and the `prophet_to_pgx` action map agree with python-chess /
   the reference `encoding.py`. The pgx-internal field names, the LERF
   square-numbering assumption, and the `^56` flip parity are all marked
   `# VERIFY:` in `env.py` and are the highest-value things to pin — they are the
   JAX analogue of the Rust core's 31099-position bit-identity check.
3. **Known approximation:** mctx's interior qtransform completes unvisited-action
   Q from a value mixture, so prophet's exact PUCT-with-`q_init` interior is *not*
   bit-reproducible here (the Gumbel **root** algorithm, which prophet also uses,
   *is*). See the `search.py` docstring. `q_trust` / `c_puct` / `c_visit` /
   `c_scale` are carried for interface parity but do not change the mctx search.
