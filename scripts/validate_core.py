"""Validate prophet_core (Rust) against python-chess bit-for-bit.

Drives a python-chess board and a prophet_core.Board through identical random
move sequences and checks, at every position: legal action sets match, the
24-feature encoding is identical, and terminal value agrees. Any mismatch
means the Rust core would corrupt training, so this must be clean before use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: F401,I001  (torch before numpy; see README)

import chess
import numpy as np

import prophet_core
from prophet.encoding import FEATURES, encode_board, index_to_move, legal_move_map
from prophet.search import _terminal_value


def main():
    rng = np.random.default_rng(0)
    checked = 0
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for g in range(games):
        pb = chess.Board()
        rb = prophet_core.Board()
        for ply in range(220):
            x_p, flipped = encode_board(pb)
            legal_p = legal_move_map(pb, flipped)
            idx_p = set(legal_p.keys())
            term_p = _terminal_value(pb)

            x_r = np.asarray(rb.encode(), dtype=np.float32).reshape(64, FEATURES)
            idx_r = set(rb.legal_actions())
            term_r = rb.terminal_value()

            if idx_p != idx_r:
                print(f"LEGAL MISMATCH g{g} ply{ply}\n  fen={pb.fen()}")
                print(f"  py-only={sorted(idx_p-idx_r)[:6]} rust-only={sorted(idx_r-idx_p)[:6]}")
                return 1
            if not np.array_equal(x_p, x_r):
                d = np.argwhere(x_p != x_r)
                print(f"ENCODE MISMATCH g{g} ply{ply} ndiff={len(d)}\n  fen={pb.fen()}")
                print(f"  first diffs (sq,plane)={d[:5].tolist()}")
                for sq, pl in d[:5]:
                    print(f"    sq{sq} plane{pl}: py={x_p[sq,pl]} rust={x_r[sq,pl]}")
                return 1
            same_term = (term_p is None) == (term_r is None) and (
                term_p is None or abs(term_p - term_r) < 1e-6
            )
            if not same_term:
                print(f"TERMINAL MISMATCH g{g} ply{ply} py={term_p} rust={term_r}\n  fen={pb.fen()}")
                return 1

            checked += 1
            if term_p is not None or not idx_p or pb.is_game_over(claim_draw=True):
                break
            a = int(rng.choice(sorted(idx_p)))
            pb.push(index_to_move(a, pb, flipped))
            rb.push_action(a)

    print(f"OK — {checked} positions over {games} games, bit-identical to python-chess")
    return 0


if __name__ == "__main__":
    sys.exit(main())
