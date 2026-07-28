"""Direct match between two checkpoints — far more sensitive than the
Stockfish ladder when both nets are weak. Reports W-D-L and the Elo
difference from A's perspective.

Both sides use the Phase C Rust batched searcher (history-aware, MPS/CUDA
accelerated when available) — identical machinery, so it cancels out.

Usage:
    python3 scripts/head2head.py A.pt B.pt --games 40 --forwards 256
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
import math

import chess
import numpy as np

from prophet.model import load_checkpoint
from prophet.search import _terminal_value
from prophet.searchC import RustBatchedSearcher

_DEV = ("mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")


class _DevEval:
    def __init__(self, m, dev):
        self.m = m.to(dev)
        self.dev = dev

    def __call__(self, x):
        with torch.no_grad():
            l, a, v = self.m(x.to(self.dev))
        return l.cpu(), a.cpu(), v.cpu()


def play(model_a, model_b, forwards, candidates, n_games, rng, max_plies=300):
    """A vs B, alternating colors. Returns (a_wins, draws, b_wins)."""
    aw = d = bw = 0
    batch = 64 if _DEV != "cpu" else 16
    for g in range(n_games):
        a_white = g % 2 == 0
        board = chess.Board()
        plies = 0
        while _terminal_value(board) is None and plies < max_plies:
            a_to_move = board.turn == (chess.WHITE if a_white else chess.BLACK)
            model = model_a if a_to_move else model_b
            s = RustBatchedSearcher(model, budget=forwards, batch=batch,
                                    candidates=candidates,
                                    seed=int(rng.integers(1, 1 << 62)))
            mv, _ = s.search(board)
            board.push(mv)
            plies += 1
        if board.is_checkmate():
            white_won = board.turn == chess.BLACK
            if white_won == a_white:
                aw += 1
            else:
                bw += 1
        else:
            d += 1
        print(f"  game {g + 1}/{n_games}: {aw}-{d}-{bw}", flush=True)
    return aw, d, bw


def elo_diff(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a")
    ap.add_argument("ckpt_b")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--forwards", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    torch.set_num_threads(6)
    wrap = (lambda m: _DevEval(m, _DEV)) if _DEV != "cpu" else (lambda m: m)
    model_a = wrap(load_checkpoint(args.ckpt_a).eval())
    model_b = wrap(load_checkpoint(args.ckpt_b).eval())
    rng = np.random.default_rng(args.seed)
    print(f"device {_DEV} | {args.games} games @ {args.forwards} forwards")

    aw, d, bw = play(model_a, model_b, args.forwards, args.candidates,
                     args.games, rng)
    n = aw + d + bw
    score = (aw + 0.5 * d) / n
    print(f"A: {args.ckpt_a}")
    print(f"B: {args.ckpt_b}")
    print(f"  A-draw-B: {aw}-{d}-{bw}  over {n} games")
    print(f"  A score: {score:.1%}  ->  Elo(A) - Elo(B) = {elo_diff(score):+.0f}")


if __name__ == "__main__":
    main()
