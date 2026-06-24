# Prophet-JAX: the 0%-decisive (drawish-collapse) bug — handoff for Codex

**Date:** 2026-06-24
**Status:** the pgx board-bridge fix (Codex, `c2e0c30`) **and** the negamax-discount
fix (`search.py`) are both confirmed present and correct, but self-play **still
produces 0% decisive games**. New probes (§3) localize the bug to the
**search/value layer — not the env, not the ply cap, not training-time**:
prophet's search is *measurably worse than uniform-random play* at delivering
checkmate. This doc has the evidence and the next diagnostics.

**Codex update, same date:** two additional JAX-only issues were fixed after
this handoff was written:

1. mctx only masks invalid actions at the root. Interior simulations were
   selecting illegal chess actions, which `env_step` guarded by clamping to pgx
   action 0; pgx then emitted illegal-action terminal rewards. `search.py` now
   masks logits at every expanded node.
2. The Q-head was not actually participating in JAX search. `q_trust` was a dead
   config field. mctx node embeddings now carry each node's raw Q-head vector,
   and the custom qtransform completes unvisited actions with
   `q_trust * q_init`.
3. Self-play ply-cap truncations are again unknown outcomes (`NaN`) rather than
   fake draws. Policy/Q rows still enter replay, but WDL/outcome blending are
   skipped, matching the PyTorch reference.

> Original TL;DR for Codex: don't re-chase the discount sign (it's correct,
> traced below). The follow-up fix found two non-discount causes: unmasked
> interior illegal actions and dead `q_trust`/Q-head completion in mctx.

---

## 1. Symptom

Fresh-model training run (RTX PRO 4000, batch 32, sims 32, max_plies 224):

```
[32]…[480]   decisive 0% throughout (480 games)   plies ~200–210   c-loss 1.2→1.44 (RISING)
losses @480: pi 2.81   v 0.14   q 0.17   c 1.40
```

- **decisive 0%** the entire run — never a single win/loss.
- **plies pinned near the 224 cap** (mean ~205) — games shuffle to the end.
- **consistency loss RISES** (1.23→1.44): the Q-head and V-head *diverge*, not
  converge. (PyTorch reference engine: c ≈ 0.075.)
- v-loss is non-zero and rising even though every game is a draw — because the
  value target is `0.5·root_value + 0.5·z` with z=0 for draws, so it tracks the
  (noisy, ~0) search root value rather than the all-zero outcome.

For reference, the **PyTorch engine was 71–100% decisive from the first rounds**.
This is a structural JAX-only difference, not "needs more games."

---

## 2. Already fixed (confirmed present on disk — do not re-investigate)

| fix | location | what it does |
|---|---|---|
| pgx square transpose + action geometry | `env.py` (`_PGX_SQ_TO_LERF`, pgx `FROM_PLANE`) | file-major↔rank-major LERF transpose + real pgx action-plane table instead of guessed geometry. **Codex `c2e0c30`.** Verified valid bijection/involution. |
| negamax discount | `search.py:201` (`recurrent_fn`) | `discount = where(term, 0, −1)` (was `+1` → both sides cooperate). `reward = −terminal_value` (parent view), `value = 0` at terminal. |
| tracer leak | `env.py` | eager `make_chess_env()` at import (pgx lazy chess import was leaking tracers inside jit) |
| lite-state memory | `selfplay.py:_lite_state` | strips `observation` [B,8,8,119] before per-ply snapshot |
| truncated→draw | `selfplay.py:_terminal_z_white:278` | non-terminal → z=0 (was NaN-masked) |

**The negamax math is correct** (traced a 2-ply path A→B→A):
terminal: `reward=−terminal_value=+1` to the mover, `discount=0` → parent sees a
win (+1). ✓ non-terminal: `leaf = 0 + (−1)·child_v = −child_v` → flips child's
perspective to parent's. ✓ A 2-ply backup returns `+v_grandchild` in A's frame. ✓
So the bug is **not** the discount; it's a sign/perspective error *elsewhere* that
the discount fix didn't cover.

---

## 3. New evidence — two probes (the core of this report)

Both probes are on the pod at `/root/` and reproduce in <2 min.

### Probe A — prophet self-play, fresh net, resignation/study OFF (`probe_terminal.py`)
B=64, max_plies=224:

```
FINISHED (pgx terminal): 64/64      TRUNCATED: 0/64
z_white among finished:  {0.0: 64}        ← every game a draw
DECISIVE: 0/64
pgx result codes: {1: 64}                 ← all draw-coded
ply histogram [<50,50-100,100-150,150-200,200-222,≥223]: [1, 2, 1, 5, 1, 54]
```

→ **Termination fires** (this is *not* a truncation bug). 84% (54/64) shuffle to
the ply cap; the ~10 that end early end as **rule-draws** (stalemate / insufficient
material / repetition), **never checkmate**.

### Probe B — pure-random-move pgx, no net / no search / no bridge (`probe_random_pgx.py`)
B=64, 224 plies, uniform-random legal moves:

```
terminated: 6/64      DECISIVE: 6/64      draws: 0/64      still-running: 58/64
end_ply of terminated games: min 111  mean 171  max 199
```

→ **Of the random games that terminate, 100% are checkmates.** Random play
delivers mate ~9% of the time in 224 plies and produces **zero** rule-draws.

### The contrast IS the finding

| player | checkmates / 64 | rule-draws / 64 |
|---|---|---|
| pure random moves | **6** | 0 |
| prophet search (fresh net) | **0** | ~10 |

Prophet's search is **significantly worse than random at being decisive**
(P(0 mates in 64 at the 9% random rate) ≈ 0.0024) **and** it *manufactures* draws
that random never reaches. A search that adds winning value must mate **more**
than random, not less. The only thing that produces this is **symmetric
draw-seeking**: both sides' search/value declines to win → mutual non-aggression →
trade down / repeat → shuffle to the cap → draw.

---

## 4. Diagnosis — ranked hypotheses

### H1 (primary): residual value/search SIGN inversion (both players minimize their own win)
The `discount=−1` fix was necessary but not sufficient. If the quantity the search
*maximizes* is effectively negated — at the root, in the qtransform, or in the
train target — BOTH sides seek non-winning lines → the exact symmetric
draw-seeking signature above. Concrete suspects, in priority order:

1. **mctx Gumbel root × `qtransform_completed_by_mix_value` under `discount=−1`.**
   Confirm the completed-Q that the Gumbel root *argmaxes* corresponds to "good for
   the side to move at the root." Negative discount changes the sign of the backed-up
   child contributions; the qtransform's value-mixture (parent value + visited-child
   Q) may end up optimizing the **opponent's** value at the root. This is the single
   least-obvious interaction and the top thing to verify. (`search.py:247`.)
2. **root value convention.** `root_fn` returns `value=v` (network value,
   side-to-move). With interior children negated by `discount=−1`, verify the root
   value is in the frame mctx expects relative to the (negated) children — a
   root-vs-interior frame mismatch inverts the optimization. (`search.py:148–153`.)
3. **train-time value target perspective.** Outcome blend:
   `value = (1−outcome_mix)·root_value + outcome_mix·z_eff`, with z mirrored to the
   mover (`selfplay.py:391+`). Verify the mover-perspective of z matches the
   side-to-move perspective the value head predicts at that stored position. A
   half-ply perspective slip inverts the target for one color → the value head
   learns an inverted/averaged-to-zero signal → flat value → drawish.

### H2: consistency loss is mis-signed and fights the value/Q heads
`loss_cons = (q_played + v_child)²` encodes negamax `Q(s,a) = −V(child)`. The
c-loss **rises** during training → this term and the value/Q targets are mutually
unsatisfiable, and the optimizer's compromise is to push Q and V toward
cancellation → both collapse toward ~0 → flat value → drawish. **Check the stored
perspective of `child_x` / `v_child` vs `q_played`** — if `v_child` is same-side
(not the opponent's), the correct relation is `Q = +V(child)` and this loss has the
wrong sign. (`train.py` loss_fn, ~line 268; doc at line 21.)

### H3: Gumbel root (8 candidates / 32 sims) + flat value under-explores mates
Lower priority: under-exploration would make prophet ≈ random, not **worse** than
random. The draw-seeking signature points at sign, not exploration. (Worth a
sanity check at higher sims once H1/H2 are ruled out.)

---

## 5. Recommended diagnostics — ranked, ready to run

- **D1 — search-vs-random (the clincher).** Play prophet's search on one side vs
  uniform-random legal moves on the other, B games, count results. A correct search
  must crush random. If it draws or loses, **H1 confirmed**. Removes the
  search-vs-search confound in Probe A. ~30 lines on top of `batched_search`.
- **D2 — mate-in-1 search test.** Reach a position with a forced mate-in-1 (random
  play until `any(mate_in_1)`), run `batched_search`, assert it picks the mate and
  `root_value → +1`. Directly tests whether the search *values* a winning move.
- **D3 — root_value distribution dump.** Log `SearchOut.root_value` over a self-play
  batch. If it's ≈0 everywhere — including lopsided-material positions — the value
  signal is dead/flat (collapse confirmed). Cheap.
- **D4 — won-position sign micro-test.** Hand the engine a position where the side
  to move is up a queen with an obvious capture; check the search prefers the
  capture and `root_value > 0`. If `root_value < 0` there, the sign is inverted (H1).

---

## 6. Code map (file:line + conventions)

**`search.py`** — mctx wrapper
- `root_fn` (124): `value=v` (network, side-to-move), `prior_logits`=RAW policy
  logits, `embedding`=pgx State. *(H1.2 suspect.)*
- `recurrent_fn` (156): `discount=where(term,0,−1)`; `reward=where(term,−terminal_value,0)`;
  `value=where(term,0,child_v)`. Negamax lives here — **math verified correct**.
- `_make_search_fn` (236): `gumbel_muzero_policy(..., max_num_considered_actions=candidates,
  gumbel_scale=1.0)`, default `qtransform_completed_by_mix_value`. *(H1.1 suspect.)*
- `SearchOut` (302): `root_value`, `q_target` (root-perspective), `q_weight` (visits).

**`env.py`**
- `terminal_info` (679): `is_terminal=state.terminated`; `value=rewards[current_player]`
  (side-to-move: checkmate→−1, draw→0). **Confirmed firing (Probe A).**
- `env_step` (410), `legal_mask` (500), pgx bridge constants (`_PGX_SQ_TO_LERF`,
  `FROM_PLANE`) — the Codex board fix.

**`selfplay.py`**
- `_terminal_z_white` (258): non-terminal → 0.0. NOTE: `GameMeta.z_white` docstring
  says "NaN if truncated", but **empirically it's 0** — the NaN path is effectively
  dead, so `finished = ~isnan(z)` (train.py:672) treats cap-truncated games as
  finished draws. (Cosmetic, but means the decisive denominator = all games.)
- outcome blend (391): `value=(1−outcome_mix)·root_value + outcome_mix·z_eff`,
  `outcome_mix=0.5`, z mirrored to mover. *(H1.3 suspect.)*
- resignation: `resign_threshold=−0.92`, `resign_plies=8` — **never fires** (a flat
  ~0 value head never drops below −0.92, so no game is ever adjudicated decisive).

**`train.py`**
- loss (268): policy CE + value MSE + Q MSE + `consistency=(q_played+v_child)²`. *(H2.)*
- decisive metric (672): `finished=~isnan(z); decisive=(|z|>0.5)&finished;
  rate=decisive/max(1,finished.sum())`.

---

## 7. Reproduce (on the pod: `/root`, jax+pgx+mctx+flax installed, GPU visible)

```bash
python3 probe_terminal.py      # prophet self-play split   -> 0/64 decisive, 64/64 draws
python3 probe_random_pgx.py    # pure-random baseline       -> 6/64 decisive (all checkmates)

# training run that shows 0% decisive + rising c-loss:
python3 -u -m prophet_jax.train --games 6000 --batch-games 32 --sims 32 \
  --candidates 8 --max-plies 224 --buffer 50000 --d-model 320 --n-layers 8 \
  --n-heads 8 --study --schedule --gate 96 --warmup 200 \
  --eval-every 999999 --log-every 32 --out runs/validate
```

(24 GB GPUs OOM at batch ≥128 / max_plies 256 once study turns on — keep batch ≤32
for validation; the bug is identical at any batch.)
```
