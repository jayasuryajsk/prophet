# prophet-bench v0 — a speedrun benchmark for learning chess from scratch

**The summit: 2500 Elo, from random weights, within a fixed compute budget.**

In the spirit of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt):
fix the target and the budget, open-source everything, and let the
leaderboard drive algorithmic research. nanoGPT's speedrun took GPT-2
training from 45 minutes to under 4 by spawning real science (the Muon
optimizer, among others). prophet-bench aims the same culture at a harder
question: **how much chess can be learned per unit of compute, starting
from nothing?**

The motivating observation: AlphaZero needed ~44M self-play games; strong
humans need a few thousand games plus study. The gap is not hardware — it
is *lesson extraction per experience*. This benchmark measures exactly
that gap and rewards closing it.

---

## 1. The task

Train a chess-playing agent **from scratch** to the highest Elo possible
within the compute budget. "From scratch" means the run starts with:

- randomly initialized parameters,
- the rules of chess (a move generator / legality oracle is free),
- nothing else. See §4 (divisions) for exactly what "nothing else" means.

## 2. The budget: FLOPs, not games, not wall-clock

**Budget (v0): 10^18 FLOPs of model compute, total, for the entire run.**

Why FLOPs:
- *Not wall-clock*: wall-clock is hardware-locked; nobody can compare a
  Mac Studio to an H100 box. FLOPs divide the hardware out.
- *Not games*: a game budget is leaky — unbounded per-move search lets
  arbitrarily much compute hide inside a fixed game count. Counting
  forward passes makes search depth, study/re-analysis, and replay all
  *visible* spending from one budget.
- Games remain the **headline chart** (§6): Elo-vs-games is the
  human-comparison story; FLOPs is the rule that keeps it honest.

Counting rules (standard approximations, applied mechanically):

| Spend | FLOPs charged |
|---|---|
| Inference forward (self-play, search, study, anything) | `2 × P × T` |
| Training step (fwd + bwd), per position in the batch | `6 × P × T` |

where `P` = parameters of the model used for that pass and `T` = input
tokens per position (64 for square-token models; use the model's actual
input length). Every model invoked during the run is charged, including
auxiliary/teacher/distillation models. The run must log a cumulative FLOPs
counter and halt training when the budget is exhausted. Evaluation games
(§5) are outside the budget — they happen after the clock stops, on the
frozen checkpoint.

There is no parameter cap and no game cap: model size, sims per move,
number of games, study allocation are all free choices *priced in FLOPs*.

## 3. What is fixed

- **Environment**: standard chess, FIDE rules; draws by stalemate,
  insufficient material, fifty-move rule, threefold repetition.
- **Start**: random initialization. No pretrained weights of any kind.
- **Budget**: 10^18 FLOPs (v0).
- **Evaluation protocol**: §5, frozen per benchmark version.

Everything else — architecture, action space, search algorithm, training
targets, optimizer, replay strategy, curriculum, language — is open.
That's where the research lives.

## 4. Divisions

**Zero division** (the main event):
- No human game data, no engine-labeled data, no opening books, no
  endgame tablebases. The only source of chess knowledge is self-play
  plus the rules oracle.
- Rationale: distilling Stockfish labels is known to reach 2895 Elo
  ([DeepMind searchless chess](https://arxiv.org/abs/2402.04494)) with
  10M annotated games — a solved, different problem.

**Open division**: anything goes except pretrained weights (tablebases,
human games, engine labels all allowed; FLOPs budget still applies to all
model compute). Exists so "zero vs. priors" can itself be measured.

## 5. Evaluation protocol (frozen, outside the budget)

1. Training halts at budget exhaustion; the final checkpoint is frozen.
2. The agent plays a gauntlet against a pinned Stockfish
   (`UCI_LimitStrength` ladder; pin the SF major version in the entry):
   **rungs 1320, 1500, 1700, 2000, 2300, 2600 — 60 games per rung**,
   colors alternating, 100 ms/move for Stockfish.
3. The agent's per-move compute at evaluation is capped at
   **256 network forward passes per move** (any search configuration that
   fits). The budget buys a better brain, not a longer think at test time.
4. Games capped at 400 plies (cap = draw). The agent must be
   nondeterministic enough that games within a rung differ (sampling
   noise in search is sufficient).
5. **Benchmark Elo** = maximum-likelihood performance rating over all
   gauntlet games: the `E` solving
   `Σ_rungs n_r · 1/(1 + 10^((R_r − E)/400)) = total score`.
   Shutout rungs contribute bounds, not points; entries should report
   per-rung W-D-L so the fit is auditable.
6. Below-ladder reference (optional, for weak entries): Stockfish
   Skill Level 0 at depth 1, descriptive only — it is uncalibrated.

## 6. Leaderboard entry

An entry must include: code (open source), the FLOPs counter log, all
hyperparameters, the frozen checkpoint, per-rung gauntlet results, and
two charts: **Elo vs. games** (headline) and **Elo vs. FLOPs**.
Records should be independently re-runnable, nanoGPT-style: a record
stands when someone else can reproduce the pipeline.

Milestones: **1500** (club player), **2000** (expert), **2500** (summit).
The headline metric is simply *best Elo within budget* — beatable from
day one, unbounded at the top.

## 7. Baseline #1 — prophet v2 (this repo)

| | |
|---|---|
| Date | 2026-06-12 |
| Division | Zero |
| Model | 2,772,161-param transformer (d=192, 6 layers), 64 square tokens, from-to factorized policy + per-move Q head + V head |
| Method | Gumbel search @16 sims, study-your-losses (deep re-analysis of surprise positions + counterfactual branches, gated at 2k games), resignation, dense targets (search values, per-move Q, negamax consistency) |
| Games | 100,000 self-play games (178,251 training steps) |
| Compute | ≈1.3×10^17 FLOPs (estimated: ~2.3×10^8 self-play forwards + 4.56×10^7 trained positions; ~13% of v0 budget) — exact counter lands in the next run |
| Elo | **1255** (official gauntlet 2026-06-13: 45% vs 1320, 17.5% vs 1500, 5% vs 1700, 1.7% vs 2000, 0% vs 2300+; `runs/run100k/gauntlet.json`) |
| Hardware (informational) | Mac Studio (M-series, 14 cores, 36 GB), ~14 h wall-clock |

Checkpoint Elo trajectory (quick-mode reads at reduced eval budget):
~950 @ 32k games, ~1055 @ 50k, ~1265 @ 79k, 1255 official @ 100k.
Opening theory emerged unprompted: random → 1.Na3 (10k) → 1.b3 (32k)
→ 1.g3 system (80k) → 1.e4/1.e4-e5 classical (100k).

Test-time search scaling of the final checkpoint (diagnostic, non-official;
30 games/rung): 64 forwards → 1164, 256 → 1255, 1024 → **1426**,
2048 → 1336 (statistically flat vs 1024). The curve saturates near 1024
forwards at ~1400: the network holds ~150 Elo more than the official
256-forward eval extracts, and beyond ~1024 sims its value noise stops
deeper search from helping. Deeper self-play search remains a lever for
entry #2, but the knowledge ceiling reasserts itself — search depth and
model capacity must scale together.

Known headroom, in rough order of expected value: spend the remaining 85%
of budget; Rust move-gen/search core (~2-3× more games per second of
overhead, i.e. more FLOPs spent on the model instead of Python); deeper
study amplification; auxiliary prediction targets; optimizer work
(Muon-class); opening curriculum seeding.

## 8. Versioning

This is **v0**: numbers (budget size, gauntlet shape, eval cap) are
expected to be tuned after baseline #1 completes, then frozen as v1.
Changes after v1 create a new leaderboard, never retro-edit one.

---

*The interesting question is not whether a datacenter can learn chess —
it can — but how steep the learning curve can be made. Every entry on
this leaderboard is a measurement of how much intelligence per FLOP we
know how to extract. Climb.*
