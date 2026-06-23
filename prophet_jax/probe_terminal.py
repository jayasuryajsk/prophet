"""Diagnostic probe: does prophet_jax self-play TERMINATE or just TRUNCATE?

Runs ONE self-play batch from a fresh net with resignation/study OFF (gate=False),
so every "finished" game is a genuine pgx terminal (checkmate or rule-draw), and
everything else truncated at the ply cap. Prints the terminate-vs-truncate split,
the z_white outcome distribution, pgx result codes, and a ply histogram.

The single question this answers: are the 0%-decisive games drawing by *rule*
(50-move / insufficient material / stalemate -> termination fires, value 0) or
*never terminating at all* (running to the 224-ply cap -> truncation)?
"""
import numpy as np
import jax

from prophet_jax import model as model_mod
from prophet_jax.config import ModelConfig, SearchConfig, SelfPlayConfig
from prophet_jax.selfplay import generate_selfplay

key = jax.random.PRNGKey(0)
key, ik = jax.random.split(key)
cfg = ModelConfig(d_model=320, n_layers=8, n_heads=8, d_ff=4 * 320)
model, params = model_mod.build_model(cfg, ik)
print(f"fresh model: {model_mod.num_params(params):,} params", flush=True)

B = 64
scfg = SearchConfig(sims=32, root_candidates=8)
spcfg = SelfPlayConfig(max_plies=224)
key, sk = jax.random.split(key)
# gate=False -> resignation + study OFF: pure self-play, finishes only by pgx terminal.
samples, meta = generate_selfplay(params, sk, B, scfg, spcfg, False)

plies = np.asarray(meta.plies)
z = np.asarray(meta.z_white)
finished = ~np.isnan(z)
decisive = (np.abs(np.nan_to_num(z)) > 0.5) & finished

print(f"\n=== PROBE (gate=False, fresh net, B={B}, max_plies=224) ===", flush=True)
print(f"plies:  min {plies.min()}  mean {plies.mean():.1f}  max {plies.max()}")
print(f"FINISHED (pgx terminal, z!=NaN): {int(finished.sum())}/{B}")
print(f"TRUNCATED (hit ply cap, z=NaN):  {int((~finished).sum())}/{B}")
if finished.any():
    vals, cnts = np.unique(z[finished], return_counts=True)
    print("z_white among finished:", {float(v): int(c) for v, c in zip(vals, cnts)},
          "  (+1=White win, -1=Black win, 0=draw)")
print(f"DECISIVE (|z|>0.5 & finished): {int(decisive.sum())}/{B}")
res = getattr(meta, "result", None)
if res is not None:
    res = np.asarray(res)
    print("pgx result codes:", {int(v): int((res == v).sum()) for v in np.unique(res)})
h, _ = np.histogram(plies, bins=[0, 50, 100, 150, 200, 223, 1000])
print("ply hist [<50, 50-100, 100-150, 150-200, 200-222, >=223]:", h.tolist())
print("\nINTERPRETATION:")
print("  many FINISHED, z all 0  -> games draw by rule; need sharper play/value, not a bug")
print("  most TRUNCATED          -> termination not firing / games never end -> structural bug")
print("  some DECISIVE           -> engine CAN end games; 0% in training is value-collapse")
