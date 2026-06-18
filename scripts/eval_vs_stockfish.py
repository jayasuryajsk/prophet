"""Benchmark a checkpoint against Stockfish at weakened settings and
estimate Elo from match scores.

Settings ladder (weakest first):
- skill0d1: Skill Level 0, depth 1 — well below Stockfish's calibrated range
- elo1320:  UCI_LimitStrength at Elo 1320 — Stockfish's weakest *calibrated* anchor

Elo estimate: ours = anchor + 400*log10(s/(1-s)), with a continuity
correction when the score is 0 or 100% (then it's only a bound).

Usage:
    python3 scripts/eval_vs_stockfish.py runs/run10k/ckpt_010000.pt --games 20 --sims 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import math
import time

import chess
import chess.engine
import numpy as np

from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, search_move

CONFIGS = {
    "skill0d1": {
        "options": {"Skill Level": 0},
        "limit": chess.engine.Limit(depth=1),
        "anchor_elo": None,  # below calibrated range; descriptive only
    },
    "elo1320": {
        "options": {"UCI_LimitStrength": True, "UCI_Elo": 1320},
        "limit": chess.engine.Limit(time=0.05),
        "anchor_elo": 1320,
    },
}


def elo_from_score(score: float, n: int, anchor: float):
    s = min(max(score, 0.5 / n), 1 - 0.5 / n)  # continuity correction
    diff = 400 * math.log10(s / (1 - s))
    bound = "<=" if score <= 0.5 / n else (">=" if score >= 1 - 0.5 / n else "~")
    return anchor + diff, bound


def play_match(model, engine, sf_limit, n_games, scfg, rng, max_plies=300):
    w = d = l = 0
    for g in range(n_games):
        model_is_white = g % 2 == 0
        board = chess.Board()
        plies = 0
        while _terminal_value(board) is None and plies < max_plies:
            if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
                mv = search_move(model, board, scfg, torch.device("cpu"), rng)
            else:
                mv = engine.play(board, sf_limit).move
            board.push(mv)
            plies += 1
        if board.is_checkmate():
            winner_is_white = board.turn == chess.BLACK
            if winner_is_white == model_is_white:
                w += 1
            else:
                l += 1
        else:
            d += 1
        mark = "W" if board.is_checkmate() and winner_is_white == model_is_white else (
            "L" if board.is_checkmate() else "D"
        )
        print(f"    game {g+1}/{n_games}: {mark} ({plies} plies)", flush=True)
    return w, d, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt)
    scfg = SearchConfig(sims=args.sims, root_candidates=args.candidates)
    print(f"model: {args.ckpt}  search: {args.sims} sims, {args.candidates} candidates")

    for name in args.configs:
        cfg = CONFIGS[name]
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure(cfg["options"])
        rng = np.random.default_rng(args.seed)
        print(f"== vs stockfish[{name}] {cfg['options']}")
        t0 = time.perf_counter()
        w, d, l = play_match(model, engine, cfg["limit"], args.games, scfg, rng)
        engine.quit()
        n = args.games
        score = (w + 0.5 * d) / n
        line = f"  result {w}-{d}-{l}  score {score:.0%}  ({time.perf_counter()-t0:.0f}s)"
        if cfg["anchor_elo"] is not None:
            est, bound = elo_from_score(score, n, cfg["anchor_elo"])
            line += f"  -> Elo {bound} {est:.0f} (anchor {cfg['anchor_elo']})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
