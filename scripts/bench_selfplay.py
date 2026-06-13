"""Benchmark self-play throughput for one worker config, with optional
profiling to find where the time actually goes (the Rust/C++ verdict).

Usage:
    python3 scripts/bench_selfplay.py --batch-games 48 --episodes 24
    python3 scripts/bench_selfplay.py --batch-games 1 --episodes 8   # v1-style
    python3 scripts/bench_selfplay.py --profile --episodes 12
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import cProfile
import pstats
import time

import numpy as np

from prophet.model import ModelConfig, PolicyQValueNet
from prophet.search import SearchConfig
from prophet.selfplay import SelfPlayConfig
from prophet.worker import run_vector_selfplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--batch-games", type=int, default=48)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = PolicyQValueNet(
        ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=4 * args.d_model,
        )
    ).to(device)
    model.eval()
    print(
        f"net {model.num_params():,} params | batch_games={args.batch_games} "
        f"threads={args.threads} device={args.device} sims={args.sims}"
    )

    scfg = SearchConfig(sims=args.sims, root_candidates=args.candidates)
    spcfg = SelfPlayConfig(max_plies=args.max_plies)
    stats = {"games": 0, "plies": 0, "samples": 0}

    def on_record(rec):
        stats["games"] += 1
        stats["plies"] += rec.plies
        stats["samples"] += len(rec.samples)
        return stats["games"] < args.episodes

    def run():
        run_vector_selfplay(
            model,
            device,
            scfg,
            spcfg,
            None,
            gate_fn=lambda: False,
            batch_games=args.batch_games,
            master_rng=np.random.default_rng(0),
            on_record=on_record,
            should_stop=lambda: stats["games"] >= args.episodes,
        )

    t0 = time.perf_counter()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        run()
        pr.disable()
    else:
        run()
    dt = time.perf_counter() - t0

    gpm = stats["games"] / dt * 60
    print(
        f"{stats['games']} games, {stats['plies']} plies, {stats['samples']} samples "
        f"in {dt:.1f}s -> {gpm:.1f} games/min/worker, "
        f"{stats['plies']/dt:.0f} plies/s"
    )
    if args.profile:
        st = pstats.Stats(pr)
        st.sort_stats("cumulative")
        print("\n== top functions by cumulative time ==")
        st.print_stats(18)


if __name__ == "__main__":
    main()
