"""Direct match between two checkpoints — far more sensitive than the
Stockfish ladder when both nets are weak (e.g. comparing two 10k-game
checkpoints). Reports W-D-L and the Elo difference from A's perspective.

Usage:
    python3 scripts/head2head.py A.pt B.pt --games 100 --forwards 256
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
from prophet.search import SearchConfig, _terminal_value, run_search


def play(model_a, model_b, scfg, n_games, rng, max_plies=300):
    """A vs B, alternating colors. Returns (a_wins, draws, b_wins)."""
    aw = d = bw = 0
    dev = torch.device("cpu")
    for g in range(n_games):
        a_white = g % 2 == 0
        board = chess.Board()
        plies = 0
        while _terminal_value(board) is None and plies < max_plies:
            a_to_move = board.turn == (chess.WHITE if a_white else chess.BLACK)
            model = model_a if a_to_move else model_b
            board.push(run_search(model, board, scfg, dev, rng).move)
            plies += 1
        if board.is_checkmate():
            white_won = board.turn == chess.BLACK
            a_won = white_won == a_white
            if a_won:
                aw += 1
            else:
                bw += 1
        else:
            d += 1
    return aw, d, bw


def elo_diff(score):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / score - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a")
    ap.add_argument("ckpt_b")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--forwards", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    torch.set_num_threads(6)
    model_a = load_checkpoint(args.ckpt_a)
    model_b = load_checkpoint(args.ckpt_b)
    scfg = SearchConfig(sims=args.forwards - 1, root_candidates=args.candidates)
    rng = np.random.default_rng(args.seed)

    aw, d, bw = play(model_a, model_b, scfg, args.games, rng)
    n = aw + d + bw
    score = (aw + 0.5 * d) / n
    print(f"A: {args.ckpt_a}")
    print(f"B: {args.ckpt_b}")
    print(f"  A-draw-B: {aw}-{d}-{bw}  over {n} games")
    print(f"  A score: {score:.1%}  ->  Elo(A) - Elo(B) = {elo_diff(score):+.0f}")


if __name__ == "__main__":
    main()
