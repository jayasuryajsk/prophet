# Prophet — 30,000-game checkpoint results

A compute-efficient, AlphaZero-style chess engine trained by **pure self-play**:
no human games, no opening book, no supervised bootstrap — from random init, on a
single Mac Studio (no GPU farm). 10M-parameter transformer.

## Headline

**~1151 Elo after 30,000 self-play games**, at the standard 256-forward search
budget. The point isn't the Elo number — it's the **games count**. From-scratch
self-play engines (AlphaZero, Leela) used **on the order of tens of millions** of
games to bootstrap; this reaches club-player strength in **30k**, on a desktop.

## Learning curve (measured)

Elo vs self-play games — each cell a separate nodes-anchored Stockfish-ladder
gauntlet on that checkpoint, at three search budgets (see `trajectory.png`):

| games | 256fw (deployed) | 512fw | 1024fw (latent) |
|---|---|---|---|
| 10k | 735 | 935 | 935 |
| 15k | 913 | 1034 | 1034 |
| 20k | 955 | 1130 | 1130 |
| 25k | 1156 | 1186 | 1294 |
| 30k | 1151 | 1304 | 1315 |

Two effects, both visible across the whole run:
- **Latent > deployed everywhere.** Deeper search (1024fw) sits ~150-180 Elo
  above the standard-budget line at every checkpoint — the network always knows
  more than its snap play shows.
- **Search saturates early, then deepens.** At 10k-20k, 1024fw = 512fw — the
  value head is too immature for deeper search to help. From ~25k, 1024fw pulls
  ahead (1293 vs 1186): the saturation point moves outward as the value head
  matures.

**Projection to 100k** (log-fit on the measured curve, wide error bars): deployed
~1630, latent (deep search) ~1760. *Caveat:* **both** curves flatten in the
25k→30k window (≈0 Elo/1k), so even these are likely optimistic without a
re-acceleration. Honest read: **~1500-1750 by 100k**, and 2000 would require the
run to break out of the current plateau — a stretch even at deep search.

## Official gauntlet (30k checkpoint, 256 forwards/move)

360 games vs a pinned Stockfish UCI_Elo ladder, MLE performance rating.

| opponent | W–D–L | score |
|---|---|---|
| SF-1320 | 10–13–37 | 27.5% |
| SF-1500 | 5–3–52 | 10.8% |
| SF-1700 | 3–1–56 | 5.8% |
| SF-2000 | 0–0–60 | 0.0% |
| SF-2300 | 0–0–60 | 0.0% |
| SF-2600 | 0–0–60 | 0.0% |

**Benchmark Elo: 1151**  (as White **1171** / as Black **1129** — a 42-pt gap;
the engine's White play is more mature, Black is still catching up.)

## Test-time search scaling (same 30k checkpoint)

The network holds more strength than its standard-budget play shows — giving the
search more thinking time cashes it out:

| forwards/move | overall Elo | White | Black |
|---|---|---|---|
| 256 (standard) | 1151 | 1171 | 1129 |
| 512 | 1268 | 1221 | 1310 |
| 1024 | **1399** | **1479** | 1310 |

So with deeper search the engine reaches **~1400 overall (White ~1480)** — and
already **drew games vs SF-1700 and SF-2000** at 27k.

## Emergent style (game outcomes)

Openings emerged from self-play with **no chess knowledge supplied**:
- **1.d4** then **2.e4** — builds the classical big pawn center.
- Develops **with tempo** (Nf3/Nc3 hitting an early ...Qxd5) — a real principle.
- Plays a **Blackmar-Diemer-Gambit-style** pawn sacrifice (f3/Nc3/Be3) for the
  initiative, and **Scandinavian-flavored** tempo lines — recognizable human
  opening *ideas*, reinvented from zero.
- Honest weaknesses: as **Black** it still plays junk edge-pawn moves (1...h5),
  and it **rarely castles** — its self-play opponent never punishes the open king
  yet. ("Alien index" tracker: 64.8/100 — high on wing-pawn storms, low on
  castling.)

## Method (what's different)

- **From–to factorized move space** (4096) instead of AlphaZero's 4672 head.
- **Dense reward** — a value estimate at every move, not just the sparse game-end.
- An explicit **per-move Q ("intuition") head** — a value for *every* move before
  search, used as the search prior.
- **Deep reflection ("study-your-losses")** — the engine re-analyzes its most
  surprising positions at ~12× the search depth and plays out counterfactual
  branches, extracting far more learning per game (the sample-efficiency lever).
- **Gumbel search** at tiny sim budgets (32) for training.

## Files in this release
- `ckpt_030000.pt` — the 30k checkpoint (10M params, PyTorch).
- `gauntlet_30000.json` — official 256fw gauntlet (the 1151 result).
- `gauntlet_30000_512fw.json`, `gauntlet_30000_1024fw.json` — search-scaling.
- `alien_26700.json`, `alien_30000.json` — emergent-style tracker.

*Numbers are nodes-anchored Stockfish (~0.1s-equivalent) MLE Elo; always quoted
with the search budget. Standard budget = 256 forwards/move.*
