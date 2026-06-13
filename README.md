# prophet

A compute-efficient, AlphaZero-style chess engine that learns from
**self-play only** — no human games, no engine labels, no opening books —
built to maximize *Elo per game* rather than Elo per datacenter.

From random weights to **~1255 Elo in 14 hours on a single Mac Studio**,
playing classical opening theory it discovered on its own. To the limit of
what is publicly indexed, there is no prior record of pure self-play chess
reaching this strength on consumer hardware in hours (see `results/`).

## The core idea

Humans don't just evaluate positions — they glance at a position and *feel*
which moves are good before calculating anything. AlphaZero's network has no
organ for that: it scores positions, V(s), and must search to learn what a
move is worth. prophet adds an explicit **per-move value head** — a Q(s, a)
estimate for every legal move in a single forward pass — so search starts
where intuition already points, and tiny search budgets (16 sims) compete
with AlphaZero's 800.

Three coupled heads share one transformer trunk over 64 square-tokens:

| Head | Output | Role |
|---|---|---|
| **Policy** | 4096 from×to logits | which moves to consider |
| **Q** | per-move value in [-1, 1] | how good each move is (the intuition organ) |
| **Value (WDL)** | win/draw/loss | how the game stands |

Moves are factorized as `from*64 + to` (4096 actions, queen-default
promotion) — smaller than AlphaZero's 4672, and the from×to bilinear form
is what makes per-move Q cheap to produce for all moves at once.

## What makes it sample-efficient

The founding constraint was that terminal reward is too sparse — one bit per
game shared across ~80 positions. Every design choice answers that:

- **Gumbel search at 16–32 sims** (DeepMind 2022): sound policy improvement
  at ~25-50x less search than vanilla MCTS.
- **Dense targets at every move**: search root values (not just game
  outcome), per-move Q targets from every visited child, and a negamax
  consistency loss Q(s, a) ≈ −V(s′).
- **Study-your-losses**: after each game, find the highest-*surprise*
  positions (value swings = blunders), re-analyze them with 8× deeper
  search, and play out *counterfactual branches* when the deep search
  disagrees with the move played — the lesson the game never showed.
  Measured worth ~+100 Elo in a controlled A/B.
- **Resignation, contempt, urgency**: hopeless games end early; draws train
  as mildly negative; faster wins are worth more (kills slow conversion).

See `results/README.md` for the ablations behind each of these.

## Results at a glance

| Run | Games | Net | Elo | Notes |
|---|---|---|---|---|
| v1 baseline | 10k | 0.85M | ~550 | proves the loop learns |
| v1 + study | 10k | 0.85M | ~650 | study-your-losses A/B (+100) |
| **v2** | **100k** | **2.8M** | **1255** | overnight; official baseline |

Test-time search scaling of the v2 net saturates at ~1400 Elo (1024
forwards/move) — it *knows* ~150 Elo more than the 256-forward eval shows.

**Opening theory emerged unprompted** over v2's 100k games:
`random → 1.Na3 → 1.b3 (Larsen) → 1.g3 system → 1.e4 e5 (classical)`.

## Layout

```
prophet/        engine package
  encoding.py     board/move <-> tensor, side-to-move perspective, history
  model.py        transformer trunk + policy/Q/WDL heads, checkpoint surgery
  search.py       Gumbel root + PUCT interior, generator core for batching
  selfplay.py     game generation with dense + contempt/urgency targets
  study.py        study-your-losses: surprise detection + counterfactuals
  train.py        4+1 loss training step
  worker.py       vectorized self-play workers (batched leaf eval)
  buffer.py       replay buffer
  evaluate.py     vs random / material-greedy opponents
scripts/
  train_loop.py     the self-play training loop (warm-start, EMA, gating)
  gauntlet.py       official prophet-bench eval vs a Stockfish ladder
  eval_vs_stockfish.py, final_eval.py, openings.py, watch_games.py, bench_selfplay.py
  smoke_test.py     end-to-end sanity check
SPEC.md         prophet-bench: a from-scratch chess speedrun benchmark
results/        metrics, logs, gauntlet results, and the full experiment log
weights/        curated milestone checkpoints (full curve at 10k granularity)
```

## Quickstart

```sh
pip install -r requirements.txt          # torch, python-chess, numpy
python3 scripts/smoke_test.py            # end-to-end sanity (~3s)

# train from scratch
python3 scripts/train_loop.py --games 100000 --workers 6 --study --out runs/myrun

# evaluate a checkpoint against the Stockfish ladder
python3 scripts/gauntlet.py weights/run100k/ckpt_100000.pt --procs 6
```

Requires the `stockfish` binary on PATH for the gauntlet. **Entry points
must `import torch` before numpy** (Homebrew libomp clash — see
`requirements.txt`).

## prophet-bench

`SPEC.md` defines a nanoGPT-speedrun-style benchmark: reach the highest Elo
from random weights within a fixed **FLOPs** budget (hardware-independent,
search-depth-honest), evaluated against a pinned Stockfish ladder. v2 is
baseline #1. The summit is 2500. Climb.

## Credits

Self-play RL after AlphaZero (DeepMind 2017); Gumbel low-sim search after
Gumbel MuZero (DeepMind 2022); search-value targets and gating after
KataGo; from-to encoding is Leela-adjacent. The explicit per-move Q head
with negamax consistency, the study/counterfactual loop, and the
single-position transformer at this scale are this project's own.
