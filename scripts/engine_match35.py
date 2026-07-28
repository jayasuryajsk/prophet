"""v3.5 rematch harness: Searcher35 (tree reuse + adaptive budget + MLH +
kappa-dial) vs a UCI engine. One persistent searcher per game."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001

import argparse
import json
import math
import multiprocessing as mp
import time

import chess
import chess.engine
import numpy as np

from prophet.encoding import move_to_index
from prophet.model import load_checkpoint
from prophet.search import _terminal_value
from prophet.search35 import Searcher35

MAX_PLIES = 300
_worker = {}


def _init_worker(ckpt, args_d):
    torch.set_num_threads(1)
    _worker["model"] = load_checkpoint(ckpt)
    _worker["args"] = args_d
    _worker["engine"] = None


def _engine():
    if _worker["engine"] is None:
        _worker["engine"] = chess.engine.SimpleEngine.popen_uci(
            _worker["args"]["engine_cmd"].split("|")
        )
    return _worker["engine"]


def _kill_engine():
    e = _worker.get("engine")
    if e is not None:
        try:
            e.close()
        except Exception:
            try:
                e.transport.kill()
            except Exception:
                pass
        _worker["engine"] = None


def _play_game(job):
    game_idx, seed = job
    a = _worker["args"]
    model_is_white = game_idx % 2 == 0
    s = Searcher35(
        _worker["model"],
        budget=a["budget"],
        candidates=a["candidates"],
        kappa=a["kappa"],
        mlh_lambda=a["mlh_lambda"],
        contempt=a["search_contempt"],
        seed=seed,
    )
    board = chess.Board()
    sans = []
    spent_total = 0
    while _terminal_value(board) is None and len(sans) < MAX_PLIES:
        flipped = board.turn == chess.BLACK
        if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
            mv, spent = s.play(board)
            spent_total += spent
        else:
            mv = _engine().play(board, chess.engine.Limit(time=a["movetime"])).move
        act = move_to_index(mv, flipped)
        sans.append(board.san(mv))
        board.push(mv)
        s.advance(act)
    _kill_engine()
    if board.is_checkmate():
        winner_is_white = board.turn == chess.BLACK
        sc = 1.0 if winner_is_white == model_is_white else 0.0
    else:
        sc = 0.5
    own_moves = max(1, (len(sans) + (1 if model_is_white else 0)) // 2)
    return (sc, model_is_white, sans, spent_total / own_moves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--engine-cmd", required=True)
    ap.add_argument("--nominal", type=float, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--mlh-lambda", type=float, default=0.05)
    ap.add_argument("--search-contempt", type=float, default=0.0)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jobs = [(i, args.seed * 1_000_003 + i) for i in range(args.games)]
    args_d = {
        "engine_cmd": args.engine_cmd,
        "movetime": args.movetime,
        "budget": args.budget,
        "candidates": args.candidates,
        "kappa": args.kappa,
        "mlh_lambda": args.mlh_lambda,
        "search_contempt": args.search_contempt,
    }
    print(
        f"v3.5 match: {args.ckpt} (avg budget {args.budget}, kappa {args.kappa}) "
        f"vs `{args.engine_cmd}` @{args.movetime}s (nominal {args.nominal:.0f}) "
        f"— {args.games} games",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.procs, initializer=_init_worker, initargs=(args.ckpt, args_d)) as pool:
        for r in pool.imap_unordered(_play_game, jobs):
            results.append(r)
            print(f"  {len(results)}/{args.games} games ({time.perf_counter()-t0:.0f}s)", flush=True)

    w = sum(1 for s, *_ in results if s == 1.0)
    d = sum(1 for s, *_ in results if s == 0.5)
    l = len(results) - w - d
    n = len(results)
    score = (w + 0.5 * d) / n
    eps = min(max(score, 0.25 / n), 1 - 0.25 / n)
    perf = args.nominal - 400.0 * math.log10(1.0 / eps - 1.0)
    avg_b = sum(r[3] for r in results) / n
    print(f"\n== v3.5 result ({time.perf_counter()-t0:.0f}s) ==")
    print(f"  {w}-{d}-{l}  score {score:.1%}   (avg forwards/move actually spent: {avg_b:.0f})")
    print(f"  performance vs nominal {args.nominal:.0f}: {perf:.0f}"
          + (" (bound)" if score in (0.0, 1.0) else ""))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt, "engine": args.engine_cmd, "nominal": args.nominal,
            "movetime": args.movetime, "budget": args.budget, "kappa": args.kappa,
            "wdl": [w, d, l], "score": round(score, 4), "perf": round(perf, 1),
            "avg_forwards": round(avg_b, 1),
            "games": [{"model_white": mw, "score": s, "moves": " ".join(sans)}
                      for s, mw, sans, _ in results],
        }, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
