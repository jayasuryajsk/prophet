"""Inspect the opening repertoire of a checkpoint.

Samples self-play games (search has Gumbel randomness, so lines vary),
tallies the most common opening sequences, and prints the raw network's
top policy/Q choices at the start position and after common first moves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse
from collections import Counter

import chess
import numpy as np

from prophet.encoding import encode_board, legal_move_map
from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, search_move


@torch.no_grad()
def top_moves(model, board, k=5):
    x, flipped = encode_board(board)
    logits, q, v = model(torch.from_numpy(x).unsqueeze(0))
    logits, q = logits[0].numpy(), q[0].numpy()
    legal = legal_move_map(board, flipped)
    idx = np.fromiter(legal.keys(), dtype=np.int64)
    lg = logits[idx] - logits[idx].max()
    p = np.exp(lg) / np.exp(lg).sum()
    # dueling composition (q from the net is the raw advantage table)
    vf = float(v[0])
    a = q[idx]
    qc = np.tanh(np.arctanh(np.clip(vf, -0.997, 0.997)) + a - a.max())
    order = np.argsort(-p)[:k]
    return [
        (board.san(legal[int(idx[j])]), float(p[j]), float(qc[j]))
        for j in order
    ], vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--plies", type=int, default=10)
    ap.add_argument("--sims", type=int, default=64)
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt)
    scfg = SearchConfig(sims=args.sims, root_candidates=16)

    print("== network priors at start position (policy %, Q) ==")
    moves, v = top_moves(model, chess.Board())
    print(f"  V = {v:+.3f}")
    for san, p, q in moves:
        print(f"  {san:6s} p={p:5.1%}  q={q:+.3f}")

    for first in ["e4", "d4", "Nf3", "c4"]:
        board = chess.Board()
        try:
            board.push_san(first)
        except ValueError:
            continue
        moves, v = top_moves(model, board, k=3)
        line = ", ".join(f"{san} {p:.0%}" for san, p, _ in moves)
        print(f"  after 1.{first}: {line}")

    print(f"\n== {args.games} sampled self-play games, first {args.plies} plies ==")
    lines = []
    for g in range(args.games):
        rng = np.random.default_rng(g)
        board = chess.Board()
        sans = []
        while _terminal_value(board) is None and len(sans) < args.plies:
            move = search_move(model, board, scfg, torch.device("cpu"), rng)
            sans.append(board.san(move))
            board.push(move)
        lines.append(sans)

    for depth, label in [(1, "first move"), (2, "first 2 plies"), (6, "first 6 plies")]:
        c = Counter(" ".join(l[:depth]) for l in lines)
        print(f"\n  -- {label} --")
        for seq, n in c.most_common(6):
            print(f"  {n:3d}x  {seq}")

    print("\n  -- 5 full sample lines --")
    for l in lines[:5]:
        print("  " + " ".join(l))


if __name__ == "__main__":
    main()
