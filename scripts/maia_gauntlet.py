"""Model vs Maia — human-calibrated opponents (CSSLab nets, Lichess-rating
anchored, played at nodes=1 exactly as calibrated). The externally-honest
rating test: scores here map to real Lichess ratings, not a Stockfish dial.
"""

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

from prophet.fastboard import PyChessBoard
from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, run_search_gen

MAX_PLIES = 300
_worker = {}


def _init_worker(ckpt, forwards, candidates, maia_dir):
    torch.set_num_threads(1)
    _worker["model"] = load_checkpoint(ckpt)
    _worker["scfg"] = SearchConfig(sims=forwards - 1, root_candidates=candidates)
    _worker["maia_dir"] = maia_dir
    _worker["engines"] = {}


def _engine(level):
    e = _worker["engines"].get(level)
    if e is None:
        e = chess.engine.SimpleEngine.popen_uci(
            ["lc0", f"--weights={_worker['maia_dir']}/maia-{level}.pb.gz", "--threads=1"]
        )
        _worker["engines"][level] = e
    return e


def _game_gen(level, game_idx, seed):
    rng = np.random.default_rng(seed)
    model_is_white = game_idx % 2 == 0
    board = chess.Board()
    pb = PyChessBoard(board)
    plies = 0
    sans = []
    while _terminal_value(board) is None and plies < MAX_PLIES:
        if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
            res = yield from run_search_gen(pb, _worker["scfg"], rng)
            mv = pb.move_for(res.move_index)
        else:
            mv = _engine(level).play(board, chess.engine.Limit(nodes=1)).move
        sans.append(board.san(mv))
        board.push(mv)
        plies += 1
    if board.is_checkmate():
        winner_is_white = board.turn == chess.BLACK
        return (1.0 if winner_is_white == model_is_white else 0.0, sans)
    return (0.5, sans)


def _play_chunk(chunk):
    level, games = chunk
    model = _worker["model"]
    gens, pending, scores = [], [], []
    for idx, seed in games:
        g = _game_gen(level, idx, seed)
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
                sc, sans = e.value
                scores.append((level, sc, games[i][0] % 2 == 0, sans))
        active = still
    for e in _worker["engines"].values():
        e.quit()
    _worker["engines"] = {}
    return scores


def expected(e, r):
    return 1.0 / (1.0 + 10 ** ((r - e) / 400.0))


def mle_elo(games):
    total = sum(g[1] for g in games)
    n = len(games)
    bound = False
    if total <= 0:
        total, bound = 0.25, True
    elif total >= n:
        total, bound = n - 0.25, True
    lo, hi = -500.0, 4000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(expected(mid, g[0]) for g in games) < total:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--levels", default="1100,1500,1900")
    ap.add_argument("--games-per-level", type=int, default=16)
    ap.add_argument("--forwards", type=int, default=1024)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--maia-dir", default="/Users/macstudio/prophet/maia")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--concurrent", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    chunks = []
    for lvl in levels:
        games = [
            (i, args.seed * 1_000_003 + lvl * 1_000 + i)
            for i in range(args.games_per_level)
        ]
        for k in range(0, len(games), args.concurrent):
            chunks.append((lvl, games[k : k + args.concurrent]))
    n_games = len(levels) * args.games_per_level
    print(
        f"maia gauntlet: {args.ckpt} | {n_games} games vs maia-{levels} | "
        f"{args.forwards} forwards/move | nodes=1 human-calibrated",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        args.procs,
        initializer=_init_worker,
        initargs=(args.ckpt, args.forwards, args.candidates, args.maia_dir),
    ) as pool:
        done = 0
        for chunk_scores in pool.imap_unordered(_play_chunk, chunks):
            results.extend(chunk_scores)
            done += len(chunk_scores)
            print(f"  {done}/{n_games} games ({time.perf_counter()-t0:.0f}s)", flush=True)

    print(f"\n== results ({time.perf_counter()-t0:.0f}s) ==")
    for lvl in levels:
        sub = [g for g in results if g[0] == lvl]
        w = sum(1 for g in sub if g[1] == 1.0)
        d = sum(1 for g in sub if g[1] == 0.5)
        l = len(sub) - w - d
        print(f"  vs maia-{lvl}: {w}-{d}-{l}  score {(w + 0.5*d)/len(sub):.1%}")

    elo, bound = mle_elo(results)
    tag = ">=" if bound else ""
    print(f"\n  Lichess-anchored Elo: {tag} {elo:.0f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "levels": {
                        str(lvl): [
                            sum(1 for g in results if g[0] == lvl and g[1] == 1.0),
                            sum(1 for g in results if g[0] == lvl and g[1] == 0.5),
                            sum(1 for g in results if g[0] == lvl and g[1] == 0.0),
                        ]
                        for lvl in levels
                    },
                    "elo": round(elo, 1),
                    "lower_bound": bound,
                    "forwards_cap": args.forwards,
                    "games": [
                        {"level": r, "model_white": w, "score": s, "moves": " ".join(sans)}
                        for r, s, w, sans in results
                    ],
                },
                indent=2,
            )
        )
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
