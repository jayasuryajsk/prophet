"""prophet-bench official evaluation gauntlet (SPEC.md §5).

Plays the frozen checkpoint against a pinned Stockfish UCI_Elo ladder and
reports per-rung W-D-L plus the maximum-likelihood performance rating.

Vectorized: each worker process runs `--concurrent` games of one rung at a
time, batching their search evals into single forward passes (exact same
per-game search semantics; batching is across games only).

Usage (official):
    python3 scripts/gauntlet.py runs/run100k/ckpt_100000.pt --procs 6
Quick look (non-official):
    python3 scripts/gauntlet.py CKPT --games-per-rung 12 --forwards 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import json
import multiprocessing as mp
import time

import chess
import chess.engine
import numpy as np

from prophet.fastboard import PyChessBoard
from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, run_search_gen

MAX_PLIES = 400

_worker = {}


def _init_worker(ckpt, forwards, candidates, sf_movetime, threads):
    torch.set_num_threads(threads)
    _worker["model"] = load_checkpoint(_worker.get("ckpt", ckpt))
    # eval-compute cap: root eval + sims, one forward each -> sims = cap - 1
    _worker["scfg"] = SearchConfig(sims=forwards - 1, root_candidates=candidates)
    _worker["limit"] = chess.engine.Limit(time=sf_movetime)
    _worker["engine"] = chess.engine.SimpleEngine.popen_uci("stockfish")


def _game_gen(game_idx, seed):
    """One gauntlet game as a generator: yields eval requests for the model's
    searches; calls Stockfish synchronously (its time is negligible).
    Returns score from the model's perspective."""
    rng = np.random.default_rng(seed)
    model_is_white = game_idx % 2 == 0
    board = chess.Board()
    pb = PyChessBoard(board)
    plies = 0
    while _terminal_value(board) is None and plies < MAX_PLIES:
        if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
            res = yield from run_search_gen(pb, _worker["scfg"], rng)
            board.push(pb.move_for(res.move_index))
        else:
            board.push(_worker["engine"].play(board, _worker["limit"]).move)
        plies += 1
    if board.is_checkmate():
        winner_is_white = board.turn == chess.BLACK
        return 1.0 if winner_is_white == model_is_white else 0.0
    return 0.5


def _play_chunk(chunk):
    """chunk: (rung, [(game_idx, seed), ...]) — all one rung, played
    concurrently with batched evals."""
    rung, games = chunk
    _worker["engine"].configure({"UCI_LimitStrength": True, "UCI_Elo": rung})
    model = _worker["model"]
    gens, pending, scores = [], [], []
    for idx, seed in games:
        g = _game_gen(idx, seed)
        gens.append(g)
        pending.append(g.send(None))
    active = list(range(len(gens)))
    while active:
        xb = torch.from_numpy(np.stack([pending[i] for i in active]))
        with torch.no_grad():
            logits, q, v = model(xb)
        logits, q, v = logits.numpy(), q.numpy(), v.numpy()
        still = []
        for j, i in enumerate(active):
            try:
                pending[i] = gens[i].send((logits[j], q[j], v[j]))
                still.append(i)
            except StopIteration as e:
                scores.append((rung, e.value))
        active = still
    return scores


def expected(e, r):
    return 1.0 / (1.0 + 10 ** ((r - e) / 400.0))


def mle_elo(games):
    """Performance rating: E solving sum(expected(E, r)) = sum(score).
    games: list of (rung, score). Returns (elo, is_bound)."""
    total = sum(s for _, s in games)
    n = len(games)
    bound = False
    if total <= 0:
        total, bound = 0.25, True  # continuity correction -> upper bound
    elif total >= n:
        total, bound = n - 0.25, True  # lower bound
    lo, hi = -500.0, 4000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(expected(mid, r) for r, _ in games) < total:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--rungs", default="1320,1500,1700,2000,2300,2600")
    ap.add_argument("--games-per-rung", type=int, default=60)
    ap.add_argument("--forwards", type=int, default=256, help="eval-compute cap per move")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--sf-movetime", type=float, default=0.1)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--concurrent", type=int, default=10, help="games batched per worker")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    rungs = [int(r) for r in args.rungs.split(",")]
    chunks = []
    for rung in rungs:
        games = [
            (i, args.seed * 1_000_003 + rung * 1_000 + i)
            for i in range(args.games_per_rung)
        ]
        for k in range(0, len(games), args.concurrent):
            chunks.append((rung, games[k : k + args.concurrent]))
    n_games = len(rungs) * args.games_per_rung
    official = args.games_per_rung >= 60 and args.forwards == 256
    print(
        f"gauntlet: {args.ckpt} | {n_games} games over rungs {rungs} | "
        f"cap {args.forwards} forwards/move | {args.procs}x{args.concurrent} vectorized | "
        f"{'OFFICIAL (spec v0)' if official else 'non-official (quick mode)'}",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        args.procs,
        initializer=_init_worker,
        initargs=(args.ckpt, args.forwards, args.candidates, args.sf_movetime, args.threads),
    ) as pool:
        done = 0
        for chunk_scores in pool.imap_unordered(_play_chunk, chunks):
            results.extend(chunk_scores)
            done += len(chunk_scores)
            print(f"  {done}/{n_games} games ({time.perf_counter()-t0:.0f}s)", flush=True)

    per_rung = {}
    for rung, score in results:
        w, d, l = per_rung.get(rung, (0, 0, 0))
        per_rung[rung] = (
            w + (score == 1.0),
            d + (score == 0.5),
            l + (score == 0.0),
        )
    print(f"\n== results ({time.perf_counter()-t0:.0f}s) ==")
    for rung in rungs:
        w, d, l = per_rung[rung]
        n = w + d + l
        print(f"  vs {rung}: {w}-{d}-{l}  score {(w + 0.5*d)/n:.1%}")

    elo, bound = mle_elo(results)
    tag = "<=" if bound and elo < rungs[0] else (">=" if bound else "")
    print(f"\n  benchmark Elo: {tag} {elo:.0f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "rungs": {str(r): per_rung[r] for r in rungs},
                    "elo": round(elo, 1),
                    "bound": bound,
                    "forwards_cap": args.forwards,
                    "games_per_rung": args.games_per_rung,
                    "official": official,
                },
                indent=2,
            )
        )
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
