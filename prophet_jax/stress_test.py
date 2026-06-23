"""GPU throughput stress test: how many self-play games/min does the JAX port
do on this GPU vs the Mac's ~6 g/min? Full 10M model, sims=32, batch 128->4096."""

import os, time
os.environ.setdefault("JAX_PLATFORMS", "")  # use GPU

import jax
from prophet_jax import config as cfg
from prophet_jax.model import build_model, num_params
from prophet_jax.selfplay import generate_selfplay

# the moonshot 10M architecture (d320 / 8L)
mcfg = cfg.ModelConfig(d_model=320, n_layers=8, n_heads=8, d_ff=1280,
                       head_dim=40, in_features=cfg.FEATURES)
scfg = cfg.SearchConfig(sims=32, root_candidates=16)
key = jax.random.PRNGKey(0)
model, params = build_model(mcfg, key)

print(f"device: {jax.devices()[0]}")
print(f"model params: {num_params(params):,}")

import sys
MAX_PLIES = 64
AVG_GAME = 85.0  # implied games/min assumes ~85 plies/real game
# one B per process (fresh JAX state) sidesteps the cross-call tracer leak
for B in ([int(x) for x in sys.argv[1:]] or [128]):
    spcfg = cfg.SelfPlayConfig(max_plies=MAX_PLIES)
    kk = jax.random.fold_in(key, B)
    k1, k2 = jax.random.split(kk)
    try:
        s, m = generate_selfplay(params, k1, B, scfg, spcfg, gate=True)  # warmup/compile
        jax.block_until_ready((s.x, m.plies))
        t = time.time()
        s, m = generate_selfplay(params, k2, B, scfg, spcfg, gate=True)  # timed
        jax.block_until_ready((s.x, m.plies))
        dt = time.time() - t
        mps = B * MAX_PLIES / dt
        gpm = mps * 60.0 / AVG_GAME
        print(f"  {B:>6} {dt:6.2f}s {mps:10.0f} {gpm:11.0f}   {gpm/6:.0f}x")
    except Exception as e:
        print(f"  {B:>6}  FAILED  {type(e).__name__}: {str(e)[:90]}")
        break
