"""High-sample strength evaluation of saved checkpoints vs a random opponent.

Usage:
    python3 scripts/final_eval.py runs/run10k/ckpt_010000.pt --greedy 100 --search 32
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import time

import numpy as np

from prophet.evaluate import play_vs_random
from prophet.model import load_checkpoint
from prophet.search import SearchConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--greedy", type=int, default=100)
    ap.add_argument("--search", type=int, default=32)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opponent", choices=["random", "material"], default="random")
    args = ap.parse_args()

    device = torch.device("cpu")
    scfg = SearchConfig(sims=args.sims, root_candidates=8)
    for ckpt in args.ckpts:
        model = load_checkpoint(ckpt)
        print(f"== {ckpt}")
        for mode, n in [("q", args.greedy), ("policy", args.greedy), ("search", args.search)]:
            if n <= 0:
                continue
            rng = np.random.default_rng(args.seed)
            t0 = time.perf_counter()
            w, d, l = play_vs_random(
                model, device, n, mode, rng, search_cfg=scfg, opponent=args.opponent
            )
            dt = time.perf_counter() - t0
            score = (w + 0.5 * d) / n
            print(
                f"  {mode:8s} {w}-{d}-{l}  win {w/n:.0%}  score {score:.0%}  ({dt:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
