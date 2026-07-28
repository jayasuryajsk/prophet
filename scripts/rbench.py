"""Rounds-based self-play throughput: counts leaf-evals/sec over a fixed
wall-time. Unlike bench_selfplay (which only counts COMPLETED games and is
therefore dominated by fill latency when batch_games is large), this measures
the continuous network-eval rate = the real GPU forward throughput. Run one
process or N concurrent to test whether MPS scales across processes.

  python3 scripts/rbench.py --device mps --batch-games 64 --secs 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001
import argparse
import time

import numpy as np

from prophet.model import ModelConfig, PolicyQValueNet
from prophet.search import SearchConfig
from prophet.selfplay import SelfPlayConfig
from prophet.worker import run_vector_selfplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--warmup-secs", type=float, default=4.0)
    ap.add_argument("--batch-games", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = PolicyQValueNet(
        ModelConfig(d_model=args.d_model, n_layers=args.n_layers,
                    n_heads=args.n_heads, d_ff=4 * args.d_model)
    ).to(device)
    model.eval()

    scfg = SearchConfig(sims=args.sims, root_candidates=args.candidates)
    spcfg = SelfPlayConfig(max_plies=160)
    # on_round(r) fires every RELOAD_EVERY_ROUNDS rounds with the actual round
    # index r, so (r_last - r_start) is the true number of rounds in the window.
    state = {"t_start": None, "r_start": None, "r_last": None, "t_last": None}

    def on_round(r):
        now = time.perf_counter()
        if state["t_start"] is None and now - t0 >= args.warmup_secs:
            state["t_start"], state["r_start"] = now, r
        state["r_last"], state["t_last"] = r, now

    def should_stop():
        return time.perf_counter() - t0 >= args.secs

    t0 = time.perf_counter()
    run_vector_selfplay(
        model, device, scfg, spcfg, None,
        gate_fn=lambda: False, batch_games=args.batch_games,
        master_rng=np.random.default_rng(0),
        on_record=lambda rec: True,
        should_stop=should_stop,
        on_round=on_round,
    )
    if state["r_start"] is None or state["r_last"] == state["r_start"]:
        print("NO DATA (window too short)")
        return
    tw = state["t_last"] - state["t_start"]
    rounds = state["r_last"] - state["r_start"]
    pos_per_s = rounds * args.batch_games / tw
    print(
        f"batch={args.batch_games} threads={args.threads} dev={args.device}: "
        f"{rounds} rounds in {tw:.1f}s -> {rounds/tw:.1f} rounds/s, "
        f"{pos_per_s:.0f} leaf-evals/s"
    )


if __name__ == "__main__":
    main()
