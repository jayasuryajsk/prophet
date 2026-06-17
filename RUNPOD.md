# Moonshot run: deep reflection in 100k games (RunPod)

Goal: from random weights, **2000 Elo in 100k self-play games** — by reflecting
*much* harder per game, not by playing more games. Same Prophet architecture
(from-to policy + per-move Q + WDL); the new machinery is reflection depth and
a game-count curriculum.

## What's new on this branch (`moonshot-deepreflect`)

- **Multi-line Q-surprise study** (`study.py`): study now fires where the
  **Q-head was surprised** — `|Q_head(s, a_played) + V(child)|`, the move-level
  "my intuition was wrong here" signal — and at each such position plays out the
  deep search's **top `n_lines` moves** as separate branches. One surprising
  position becomes a whole tree of studied lines, not a single counterfactual.
- **Q-trust curriculum** (`search.py` `q_trust`): search ignores the Q-head for
  unvisited children early (when Q is noise) and ramps to full trust as it
  matures. Does not touch the Q-head architecture.
- **Game-count schedules** (`schedule.py`): study intensity (`top_k`,
  `deep_sims`, `branch_plies`, `n_lines`), `q_trust`, and Q/consistency loss
  weights all ramp with total games. Learner writes `progress.json`; workers
  poll it (like the gate file).

Enable all of it with `--schedule`.

## Step 1 — throughput probe FIRST (do not skip)

Half of self-play time is python-chess + tree ops, which a GPU does **not**
accelerate. Before committing to a multi-day run, measure real games/min on the
box and confirm the GPU isn't idle:

```sh
python3 scripts/bench_selfplay.py --batch-games 64 --episodes 64 \
  --d-model 256 --n-layers 8 --n-heads 8 --device cuda
```

If throughput is too low (deep study at 100k games would take many days), the
fix is the Rust move-gen/search core — that's the bottleneck, not the GPU.

## Step 2 — the run

Recommended **moderate** first model (d=256, ~6.5M) — NOT d=320. At a fixed 100k
games a 10M net risks underfilling (see `results/` v4: a 6.5M from-scratch net
was already throughput-starved on the Mac). Prove the study ramp can fill 6.5M
before reaching for 10M.

```sh
python3 scripts/train_loop.py \
  --games 100000 \
  --device cuda --worker-device cuda --compile \
  --workers 8 --batch-games 32 --worker-threads 2 \
  --d-model 256 --n-layers 8 --n-heads 8 \
  --sims 32 --candidates 12 \
  --study --schedule \
  --train-ratio 7 --batch 1024 --buffer 800000 \
  --warmup 10000 --gate 2000 \
  --lr 2e-4 --ema 0.9995 \
  --contempt 0.0 --win-discount 0.999 \
  --eval-every 1000 \
  --out /workspace/prophet-runs/moonshot_d256
```

`--workers 8` parallelizes the CPU-side tree ops across processes (tune to the
box's core count); they share the GPU for batched inference. `--compile`
torch.compiles worker inference (one-time ~1-2 min per worker at startup).

## CUDA optimizations (what's automatic on `--device cuda`)

- **bf16 autocast** for self-play inference AND training (no GradScaler needed)
  — typically ~2x on matmul-bound work. No-op off CUDA.
- **TF32 + cuDNN autotuning** — free on Ampere+.
- **`torch.compile` worker inference** (with `--compile`) — fixed batch size, so
  no per-step recompile.
- **`inference_mode`** for self-play forward.

These are guarded so MPS/CPU runs are byte-for-byte unchanged.

### Not yet optimized (the remaining big win)

Each worker process holds its **own** model copy and hits the GPU with a
batch-of-`batch_games` forward. On one GPU, N workers = N small batches
serialized — the GPU is underutilized. The proper fix is a **central inference
server**: many CPU worker processes for tree ops feeding ONE GPU process that
batches all their leaf evals into large forwards (the KataGo/Leela design). That
is the highest-value remaining optimization but a real rewrite; left for after
the throughput probe tells us whether it's needed. Interim tuning: if the GPU is
underutilized, use fewer workers with larger `--batch-games` (e.g. `--workers 4
--batch-games 96`) to grow the per-forward batch.

## Step 3 — evaluate with cores FREE

Gauntlets run alongside training **inflate** the score (time-limited Stockfish
gets starved). Always pause/finish training before an official number:

```sh
python3 scripts/gauntlet.py <ckpt> --procs <cores> --out <ckpt>.gauntlet.json
```

## Expectations (honest)

- 2000 in 100k games would be a genuine first — no public precedent. The
  realistic near-term target is **>1500-1700** if the study ramp fills 6.5M.
- Decision gate at ~30k games: gauntlet (cores free). If climbing past 1400 →
  push to d=320 and full 100k. If stuck at ~1400 → the bottleneck is target
  quality / Q expressivity, and *that's* when the Q-head redesign becomes the
  next experiment (not before).

## Do NOT change yet

Q-head structure, 64×64 action space, no JEPA, no legal-only scorer. The clean
question this run answers is whether *this* architecture scales with capacity +
reflection. Changing the head would confound the answer.
