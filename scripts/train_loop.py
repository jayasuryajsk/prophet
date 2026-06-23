"""Self-play training loop (v2: vectorized workers, gated study/resign).

N worker processes each run a batch of concurrent games, evaluating leaves
in batched forward passes. The learner trains on MPS (or CPU), syncs
weights via an atomically-replaced checkpoint file, creates the gate file
once enough games have been played (which switches on study + resignation
in the workers), and periodically evaluates vs a uniform-random opponent.

Usage:
    python3 scripts/train_loop.py --games 100000 --study --out runs/run100k
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import time
from collections import deque

import numpy as np

from prophet.buffer import ReplayBuffer
from prophet.encoding import FEATURES
from prophet.evaluate import play_vs_random
from prophet.model import (
    ModelConfig,
    PolicyQValueNet,
    load_checkpoint,
    save_checkpoint,
    widen_input,
)
from prophet.accel import setup_perf
from prophet.schedule import loss_weights_at
from prophet.search import SearchConfig
from prophet.train import collate, train_step
from prophet.worker import vector_worker


def run_eval(model, search_cfg, rng, n_greedy, n_search):
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    dev = torch.device("cpu")
    out = {}
    for mode, n in [("q", n_greedy), ("policy", n_greedy), ("search", n_search)]:
        t0 = time.perf_counter()
        w, d, l = play_vs_random(cpu_model, dev, n, mode, rng, search_cfg=search_cfg)
        out[mode] = (w, d, l, time.perf_counter() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-games", type=int, default=48, help="concurrent games per worker")
    ap.add_argument("--worker-threads", type=int, default=2)
    ap.add_argument("--worker-device", default="cpu")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-ratio", type=float, default=4.0)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sync-every", type=int, default=25, help="games between weight syncs")
    ap.add_argument("--gate", type=int, default=2000, help="games before study/resign turn on")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-greedy-games", type=int, default=24)
    ap.add_argument("--eval-search-games", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="runs/run1")
    ap.add_argument("--study", action="store_true", help="enable study-your-losses (after gate)")
    ap.add_argument("--study-topk", type=int, default=2)
    ap.add_argument("--deep-sims", type=int, default=128)
    ap.add_argument("--branch-plies", type=int, default=16)
    ap.add_argument("--study-weight", type=float, default=2.0)
    ap.add_argument("--n-lines", type=int, default=1, help="alternate lines explored per surprise position (multi-line reflection)")
    ap.add_argument("--init-from", default=None, help="warm-start checkpoint (upgraded in place)")
    ap.add_argument("--contempt", type=float, default=0.15)
    ap.add_argument("--win-discount", type=float, default=0.997)
    ap.add_argument("--ema", type=float, default=0.999, help="weight EMA decay; checkpoints/evals use the EMA")
    ap.add_argument("--schedule", action="store_true", help="game-count curricula for study/q-trust/q-loss (moonshot)")
    ap.add_argument("--start-game", type=int, default=0, help="resume schedule/gate/counter at this game # (warm restart after a crash)")
    ap.add_argument("--compile", action="store_true", help="torch.compile worker inference (CUDA only)")
    ap.add_argument("--no-eval", action="store_true", help="skip in-loop vs-random eval (still saves milestone checkpoints); use the gauntlet instead")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "latest.pt"
    gate_path = out / "gate_on"
    if args.start_game >= args.gate:
        gate_path.touch()  # warm restart past the gate -> study on from game 1
    else:
        gate_path.unlink(missing_ok=True)
    progress_path = out / "progress.json"
    metrics_path = out / "metrics.csv"

    def write_progress(games):
        tmp = progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"games": games}))
        os.replace(tmp, progress_path)

    write_progress(args.start_game)  # workers read the resumed game # -> correct schedule band

    setup_perf(args.device)
    device = torch.device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    if args.init_from:
        model = widen_input(load_checkpoint(args.init_from), FEATURES)
        print(f"warm start from {args.init_from} ({model.num_params():,} params)")
    else:
        model = PolicyQValueNet(
            ModelConfig(
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_ff=4 * args.d_model,
            )
        )
    from dataclasses import asdict

    model_kwargs = asdict(model.cfg)
    model = model.to(device)
    ema_model = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # learner speedup comes from bf16 autocast (train uses forward_wdl, which
    # is off torch.compile's forward() path); workers compile their forward().
    save_checkpoint(ema_model, ckpt_path)

    search_cfg = SearchConfig(sims=args.sims, root_candidates=args.candidates)
    search_kwargs = {"sims": args.sims, "root_candidates": args.candidates}
    selfplay_kwargs = {
        "max_plies": args.max_plies,
        "contempt": args.contempt,
        "win_discount": args.win_discount,
    }
    study_kwargs = (
        {
            "top_k": args.study_topk,
            "deep_sims": args.deep_sims,
            "branch_plies": args.branch_plies,
            "study_weight": args.study_weight,
            "n_lines": args.n_lines,
        }
        if args.study
        else None
    )

    ctx = mp.get_context("spawn")
    game_q = ctx.Queue(maxsize=64)
    stop = ctx.Event()
    workers = [
        ctx.Process(
            target=vector_worker,
            args=(
                i, str(ckpt_path), str(gate_path), game_q, stop,
                search_kwargs, selfplay_kwargs, study_kwargs, model_kwargs,
                args.batch_games, args.worker_threads, args.worker_device,
                str(progress_path) if args.schedule else None,
                args.compile,
            ),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for w in workers:
        w.start()
    print(
        f"learner on {device}, {args.workers} workers x {args.batch_games} games "
        f"({args.worker_device}, {args.worker_threads} threads), sims={args.sims}, "
        f"net {model.num_params():,} params, study={'on' if args.study else 'off'}, "
        f"gate at {args.gate} games",
        flush=True,
    )

    buffer = ReplayBuffer(args.buffer)
    ema = {}
    recent_plies = deque(maxlen=200)
    recent_decisive = deque(maxlen=200)
    games_done = args.start_game
    total_steps = 0
    gated = args.start_game >= args.gate
    t0 = time.time()

    new_metrics = not metrics_path.exists()
    mf = open(metrics_path, "a", newline="")
    mw = csv.writer(mf)
    if new_metrics:
        mw.writerow(
            [
                "games", "steps", "buffer", "avg_plies", "decisive_rate",
                "loss", "loss_pi", "loss_v", "loss_q", "loss_cons",
                "q_w", "q_d", "q_l", "pol_w", "pol_d", "pol_l",
                "search_w", "search_d", "search_l", "games_per_min",
            ]
        )

    try:
        while games_done < args.games:
            game = game_q.get()
            buffer.add(game.samples)
            games_done += 1
            recent_plies.append(game.plies)
            recent_decisive.append(0 if game.result in ("1/2-1/2", "*") else 1)

            if not gated and games_done >= args.gate:
                gate_path.touch()
                gated = True
                print(f"  GATE OPEN @{games_done}: study/resignation enabled", flush=True)

            weights = loss_weights_at(games_done) if args.schedule else None
            if len(buffer) >= args.warmup:
                steps = max(1, round(len(game.samples) * args.train_ratio / args.batch))
                for _ in range(steps):
                    batch = collate(buffer.sample(args.batch, rng), device)
                    losses = train_step(model, opt, batch, weights=weights)
                    total_steps += 1
                    for k, v in losses.items():
                        ema[k] = v if k not in ema else 0.99 * ema[k] + 0.01 * v
                    with torch.no_grad():
                        for pe, p in zip(ema_model.parameters(), model.parameters()):
                            pe.lerp_(p, 1 - args.ema)
                        for be, b in zip(ema_model.buffers(), model.buffers()):
                            be.copy_(b)

            if games_done % args.sync_every == 0:
                save_checkpoint(ema_model, ckpt_path)
                write_progress(games_done)

            if games_done % args.log_every == 0:
                gpm = games_done / max(1e-9, (time.time() - t0) / 60)
                eta_h = (args.games - games_done) / max(gpm, 1e-9) / 60
                loss_str = (
                    f"loss {ema['loss']:.3f} (pi {ema['policy']:.3f} v {ema['value']:.3f} "
                    f"q {ema['q']:.3f} c {ema['consistency']:.3f})"
                    if ema
                    else "warmup"
                )
                print(
                    f"[{games_done}/{args.games}] {loss_str} | "
                    f"plies {np.mean(recent_plies):.0f} decisive {np.mean(recent_decisive):.0%} | "
                    f"buffer {len(buffer)} steps {total_steps} | "
                    f"{gpm:.1f} g/min eta {eta_h:.1f}h",
                    flush=True,
                )

            if games_done % args.eval_every == 0 or games_done == args.games:
                save_checkpoint(ema_model, ckpt_path)
                save_checkpoint(ema_model, out / f"ckpt_{games_done:06d}.pt")
                if args.no_eval:
                    # in-loop vs-random eval stalls the run (single-core eval
                    # starved by the workers) and is uninformative; use the
                    # cores-free gauntlet on the saved checkpoints instead
                    mw.writerow(
                        [games_done, total_steps, len(buffer),
                         round(float(np.mean(recent_plies)), 1),
                         round(float(np.mean(recent_decisive)), 3)]
                        + [round(ema.get(k, 0.0), 4) for k in ("loss", "policy", "value", "q", "consistency")]
                        + [0, 0, 0, 0, 0, 0, 0, 0, 0, round(games_done / max(1e-9, (time.time() - t0) / 60), 2)]
                    )
                    mf.flush()
                    print(f"  CKPT @{games_done} (no in-loop eval)", flush=True)
                else:
                    ev = run_eval(
                        ema_model, search_cfg, rng,
                        args.eval_greedy_games, args.eval_search_games,
                    )
                    qw, qd, ql, _ = ev["q"]
                    pw, pd, pl, _ = ev["policy"]
                    sw, sd, sl, st = ev["search"]
                    print(
                        f"  EVAL @{games_done}: q-greedy {qw}-{qd}-{ql} | "
                        f"policy-greedy {pw}-{pd}-{pl} | search {sw}-{sd}-{sl} "
                        f"(eval {st:.0f}s)",
                        flush=True,
                    )
                    gpm = games_done / max(1e-9, (time.time() - t0) / 60)
                    mw.writerow(
                        [
                            games_done, total_steps, len(buffer),
                            round(float(np.mean(recent_plies)), 1),
                            round(float(np.mean(recent_decisive)), 3),
                        ]
                        + [round(ema.get(k, 0.0), 4) for k in ("loss", "policy", "value", "q", "consistency")]
                        + [qw, qd, ql, pw, pd, pl, sw, sd, sl, round(gpm, 2)]
                    )
                    mf.flush()
    finally:
        stop.set()
        # drain so workers blocked on put() can see the stop event
        try:
            while True:
                game_q.get_nowait()
        except Exception:
            pass
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
        mf.close()
        save_checkpoint(ema_model, ckpt_path)

    print(f"done: {games_done} games, {total_steps} steps in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
