"""Golden test + bench for prophet.fastlearn.

1. fast_collate must produce tensors EXACTLY equal to train.collate on
   real self-play samples (same keys, dtypes, values).
2. A train_step on identically-seeded model clones must produce identical
   losses through both collate paths.
3. Bench: serial old-collate+step vs prefetched fast path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001

import copy
import time

import numpy as np

from prophet.buffer import ReplayBuffer
from prophet.fastlearn import Prefetcher, fast_collate, fused_ema
from prophet.model import ModelConfig, PolicyQValueNet
from prophet.search import SearchConfig
from prophet.selfplay import SelfPlayConfig
from prophet.train import LossWeights, collate, train_step
from prophet.fastplay import play_game_fast

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    model = PolicyQValueNet(ModelConfig(d_model=320, n_layers=8, n_heads=8, d_ff=1280))
    model = model.to(DEV)

    @torch.no_grad()
    def eval_fn(x):
        l, a, v = model(torch.from_numpy(np.ascontiguousarray(x)).to(DEV))
        return torch.cat([l, a, v[:, None]], dim=1).float().cpu().numpy()

    print("generating real samples...")
    rng = np.random.default_rng(3)
    buf = ReplayBuffer(50_000)
    scfg = SearchConfig(sims=32, root_candidates=8, contempt=0.15)
    spcfg = SelfPlayConfig(max_plies=120, pcr_prob=0.25, pcr_cheap_sims=12)
    while len(buf) < 6000:
        rec = play_game_fast(scfg, spcfg, rng, eval_fn)
        buf.add(rec.samples)
    print(f"buffer: {len(buf)} samples")

    # --- 1. golden equality ---
    samples = buf.sample(256, np.random.default_rng(9))
    old = collate(samples, "cpu")
    new = fast_collate(samples, "cpu")
    assert set(old) == set(new), (set(old) ^ set(new))
    for k in old:
        o, n = old[k], new[k]
        assert o.shape == n.shape, (k, o.shape, n.shape)
        # stock collate leaves some fields at default dtypes; values must match
        assert torch.equal(o.to(n.dtype), n), f"MISMATCH in {k}"
    print("golden: fast_collate == collate on all 12 keys")

    # --- 2. identical losses through both paths ---
    m1 = copy.deepcopy(model)
    m2 = copy.deepcopy(model)
    o1 = torch.optim.AdamW(m1.parameters(), lr=3e-4)
    o2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
    l1 = train_step(m1, o1, collate(samples, DEV), weights=LossWeights())
    l2 = train_step(m2, o2, fast_collate(samples, DEV), weights=LossWeights())
    for k in l1:
        assert abs(l1[k] - l2[k]) < 1e-4, (k, l1[k], l2[k])
    print(f"losses identical through both paths (loss {l1['loss']:.4f})")

    # --- 3. bench ---
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ema_model = copy.deepcopy(model)
    ema_p = list(ema_model.parameters())
    mod_p = list(model.parameters())
    rng_b = np.random.default_rng(1)

    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        batch = collate(buf.sample(256, rng_b), DEV)
        train_step(model, opt, batch)
        with torch.no_grad():
            for pe, p in zip(ema_p, mod_p):
                pe.lerp_(p, 1 - 0.999)
    old_ms = (time.perf_counter() - t0) / n * 1000

    pre = Prefetcher(buf, 256, np.random.default_rng(2), DEV)
    t0 = time.perf_counter()
    for _ in range(n):
        batch = pre.next()
        train_step(model, opt, batch)
        fused_ema(ema_p, mod_p, 0.999)
    new_ms = (time.perf_counter() - t0) / n * 1000
    pre.stop()

    print(f"bench ({DEV}, batch 256): old {old_ms:.1f} ms/step -> fast {new_ms:.1f} ms/step "
          f"({old_ms / new_ms:.2f}x)")


if __name__ == "__main__":
    main()
