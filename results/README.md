# Experimental log

All runs are pure self-play from random init (Zero division: no human games,
no engine labels, no tablebases), on one Mac Studio (M-series, 14 cores,
36 GB), PyTorch on MPS for the learner + CPU workers for self-play.

## Runs

| Run | Games | Net params | Sims | Wall-clock | Official Elo |
|---|---|---|---|---|---|
| `run10k` (v1 baseline) | 10,000 | 853k | 16 | 45 min | ~550 |
| `run10k_study` (v1 + study) | 10,000 | 853k | 16 | 51 min | ~650 |
| `run100k` (v2) | 100,000 | 2.77M | 16 | ~14 h | **1255** |
| `run_v3` (warm-start, in progress) | 60,000 | 2.77M | 32 | ~16 h | 1236 @ 27k (interim) |

Elo is the maximum-likelihood performance rating from `scripts/gauntlet.py`:
360 games vs a pinned Stockfish `UCI_Elo` ladder (1320–2600), model capped
at 256 forward passes/move. Per-run JSON in each run directory.

## The study-your-losses ablation (v1)

Identical runs, 10k games each, with and without the study module:

| vs material-greedy | baseline | + study |
|---|---|---|
| Q-greedy wins | 3% | 10% |
| policy-greedy wins | 17% | 20% |
| search | 91% | 91% |
| vs Stockfish skill-0/d1 | score 8% | score 15% |

≈ +100 Elo at equal games for ~15% more compute. The mechanism (deep
re-analysis of surprise positions + counterfactual branches) most improves
the Q-head — the head its targets feed directly. Cold-start caveat: study
fires on value-head *noise* early in training and hurts before ~game 2500;
v2 fixes this by gating study on only after the value head matures.

## v2 (run100k): the headline run

Official gauntlet (`run100k/gauntlet.json`):

| vs | W-D-L | score |
|---|---|---|
| 1320 | 27-0-33 | 45.0% |
| 1500 | 10-1-49 | 17.5% |
| 1700 | 3-0-57 | 5.0% |
| 2000 | 1-0-59 | 1.7% |
| 2300 | 0-0-60 | 0.0% |
| 2600 | 0-0-60 | 0.0% |

**Benchmark Elo: 1255.** ≈ 1.3e17 FLOPs (~13% of the prophet-bench v0 budget).

### Elo-vs-games curve (quick-mode reads)

| Games | Elo |
|---|---|
| 32k | ~950 |
| 50k | ~1055 |
| 79k | ~1265 |
| 100k | 1255 (official) |

Slope ≈ 180 Elo per doubling of games until capacity-limited.

### Test-time search scaling of the final checkpoint

| Forwards/move | Elo |
|---|---|
| 64 | 1164 |
| 256 (official) | 1255 |
| 1024 | 1426 |
| 2048 | 1336 (flat vs 1024) |

The curve saturates near 1024 forwards at ~1400 Elo: the network holds ~150
Elo more than the official eval extracts, but beyond ~1024 sims its value
noise stops deeper search from helping. Search depth and model capacity must
scale together. JSON: `run100k/gauntlet_f{64,1024,2048}.json`.

### Opening theory, emerged unprompted

Sampled self-play opening (start-position policy prior in parentheses):

| Games | Favourite first move |
|---|---|
| 10k | 1.Na3 — Sodium Attack (rim-knight gibberish) |
| 32k | 1.b3 — Larsen's Opening (flank experiments) |
| 80k | 1.g3 — committed King's-fianchetto system (93.8%) |
| 100k | **1.e4** (95.6%), meeting 1.e4 e5 / 1.d4 d5 classically |

It also discovered the first-move advantage (start-position value drifted
from 0.00 toward +0.03) and plays occasional gambits. The progression
mirrors early Leela nets: tactics → development → fianchetto → center.

### Alien weirdness at ~1250

Notable non-human traits, all living where the reward landscape is flat:
- **No sense of time** — mates won positions in 40+ moves; +1.0 now and
  +1.0 later are identical to a tanh value head.
- **The "second-queen ritual"** — in a totally won position it would march a
  pawn to promotion and mate with the *new* queen rather than the army it
  already had (promotion-mate is the most common win pattern in self-play).
- **Fearless, oblivious king** — sharp where consequences differed (tactics,
  material), naive where its self-play opponents never punished it (slow
  flank play under attack).

These motivated the v3 changes (urgency discount, contempt, WDL).

## v3 (run_v3): warm-started reward-shaping — NEGATIVE RESULT

Warm-started from v2's final checkpoint via in-place surgery — input widened
18→24 features (last-2-move history + repetition flag, zero-initialized
columns), scalar value head upgraded to a WDL (win/draw/loss) head
initialized to reproduce the old value. Plus: draw **contempt** (−0.15),
per-ply **win-urgency** discount (0.997), **EMA** eval weights, and 2×
self-play search depth (32 sims). Run to ~35k games, then stopped.

**Outcome: flat on every measured axis.** No gain over the 1255 baseline,
and — tellingly — the *behavioral* targets didn't move either:

| Signal | v2 final | v3 @30-35k |
|---|---|---|
| Elo @256 forwards | 1255 | 1236 |
| Elo @1024 forwards | 1426 | 1282 |
| Avg game length | 102 plies | 102–105 |
| Decisive rate | 79% | 78–80% |

Contempt and urgency exist to make the engine win faster and draw less; game
length and decisive rate not moving means they were absorbed without effect.
Interpretation: bolted onto a confident 100k-game brain, −0.15 contempt and
a 0.997/ply discount are too gentle to repaint deeply grooved habits, and
the WDL recalibration didn't raise the extraction ceiling at this scale. The
deep-search gain even *shrank* (v2: +171 from 256→1024 forwards; v3: +46).

Conclusion: reward-shaping + calibration changes, **warm-started at fixed
size, move nothing**. The levers that work remain capacity and games — which
motivated v4. The same changes may behave differently learned from scratch
(where they shape play from move one), which v4 tests directly.

## v4 (run_v4): from scratch, bigger net + v3 architecture (in progress)

The honest follow-up to v3's negative result. From random weights, **6.47M
params** (d=256, 8 layers; 2.3× v2), carrying all of v3's architecture
(WDL head, history+repetition input, contempt, urgency, EMA) but now learned
from birth rather than bolted on, study re-gated at 2000 games, self-play
back to 16 sims (v3's 32 cost throughput for no measured benefit). Directly
comparable to v2's from-scratch 1255. Result to follow.

## Reproducing the numbers

```sh
# headline gauntlet (needs `stockfish` on PATH)
python3 scripts/gauntlet.py weights/run100k/ckpt_100000.pt --procs 6

# opening census of any checkpoint
python3 scripts/openings.py weights/run100k/ckpt_100000.pt

# watch sample games with a material/sacrifice trace
python3 scripts/watch_games.py weights/run100k/ckpt_100000.pt
```

Full 1k-granularity checkpoint sets (the complete Elo curve) are published as
GitHub Release assets; this directory's `weights/` holds milestones at 10k
granularity, enough to reproduce every curve above.
