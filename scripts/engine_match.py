"""Head-to-head vs a UCI engine (e.g. GNU Chess 6, CCRL ~2700-class).
Score -> direct performance rating vs a real, externally-rated engine.
Opponent gets wall-clock time per move; hard-kills engines that hang on quit.
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


def _init_worker(ckpt, forwards, candidates, engine_cmd, movetime):
    torch.set_num_threads(1)
    _worker["model"] = load_checkpoint(ckpt)
    _worker["scfg"] = SearchConfig(sims=forwards - 1, root_candidates=candidates)
    _worker["cmd"] = engine_cmd.split("|")
    _worker["movetime"] = movetime
    _worker["engine"] = None


def _engine():
    if _worker["engine"] is None:
        _worker["engine"] = chess.engine.SimpleEngine.popen_uci(_worker["cmd"])
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


def _game_gen(game_idx, seed):
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
            mv = _engine().play(board, chess.engine.Limit(time=_worker["movetime"])).move
        sans.append(board.san(mv))
        board.push(mv)
        plies += 1
    if board.is_checkmate():
        winner_is_white = board.turn == chess.BLACK
        return (1.0 if winner_is_white == model_is_white else 0.0, sans)
    return (0.5, sans)


def _play_chunk(chunk):
    games = chunk
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
                sc, sans = e.value
                scores.append((sc, games[i][0] % 2 == 0, sans))
        active = still
    _kill_engine()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--engine-cmd", default="gnuchess|--uci",
                    help="opponent command, '|'-separated argv")
    ap.add_argument("--nominal", type=float, default=2700.0,
                    help="opponent's external (CCRL) rating for the perf calc")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=1.0)
    ap.add_argument("--forwards", type=int, default=1024)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--concurrent", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    games = [(i, args.seed * 1_000_003 + i) for i in range(args.games)]
    chunks = [games[k : k + args.concurrent] for k in range(0, len(games), args.concurrent)]
    print(
        f"match: {args.ckpt} ({args.forwards}fw) vs `{args.engine_cmd}` "
        f"@{args.movetime}s/move (nominal {args.nominal:.0f}) — {args.games} games",
        flush=True,
    )

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(
        args.procs,
        initializer=_init_worker,
        initargs=(args.ckpt, args.forwards, args.candidates, args.engine_cmd, args.movetime),
    ) as pool:
        done = 0
        for chunk_scores in pool.imap_unordered(_play_chunk, chunks):
            results.extend(chunk_scores)
            done += len(chunk_scores)
            print(f"  {done}/{args.games} games ({time.perf_counter()-t0:.0f}s)", flush=True)

    w = sum(1 for s, _, _ in results if s == 1.0)
    d = sum(1 for s, _, _ in results if s == 0.5)
    l = len(results) - w - d
    n = len(results)
    score = (w + 0.5 * d) / n
    eps = min(max(score, 0.25 / n), 1 - 0.25 / n)
    diff = -400.0 * math.log10(1.0 / eps - 1.0)
    perf = args.nominal + diff
    print(f"\n== result ({time.perf_counter()-t0:.0f}s) ==")
    print(f"  {w}-{d}-{l}  score {score:.1%}")
    print(f"  performance vs nominal {args.nominal:.0f}: {perf:.0f}"
          + (" (bound)" if score in (0.0, 1.0) else ""))

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpt": args.ckpt, "engine": args.engine_cmd, "nominal": args.nominal,
            "movetime": args.movetime, "forwards": args.forwards,
            "wdl": [w, d, l], "score": round(score, 4), "perf": round(perf, 1),
            "games": [{"model_white": mw, "score": s, "moves": " ".join(sans)}
                      for s, mw, sans in results],
        }, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
