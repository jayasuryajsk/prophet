"""Alien-ness tracker: how far has the engine's self-play style diverged from
human chess?

Samples self-play games and measures the traits that separate a from-scratch
self-play mind from human/book theory — castling habit, wing-pawn storms (the
AlphaZero h-pawn signature), early-queen sorties, and how often it answers with
a move no human opening book would. Rolls them into a single `alien_index`
(0 = textbook human, 100 = full alien) so we can chart the mind diverging from
humanity over the course of training, alongside the Elo curve.

Intrinsic metrics need no external database; the "book" check uses a small
hardcoded set of mainstream opening moves (first move + reply) — approximate,
but enough to flag genuinely off-book choices like 1...h5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README (Homebrew libomp clash)
import torch  # noqa: I001

import argparse
import json

import chess
import numpy as np

from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, search_move

# --- a tiny "human theory" reference: mainstream first move + reply ----------
WHITE_BOOK = {"e4", "d4", "c4", "Nf3", "g3", "b3", "f4", "Nc3", "b4"}
BLACK_BOOK = {
    "e4": {"e5", "c5", "e6", "c6", "d5", "Nf6", "d6", "g6", "Nc6"},
    "d4": {"d5", "Nf6", "e6", "f5", "g6", "d6", "c5", "Nc6"},
    "Nf3": {"d5", "Nf6", "c5", "g6", "d6", "e6", "Nc6"},
    "c4": {"e5", "c5", "Nf6", "e6", "g6", "c6", "Nc6"},
    "g3": {"d5", "Nf6", "e5", "g6", "c5"},
}
WING_FILES = {0, 1, 6, 7}  # a, b, g, h — rook/knight pawns
CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}


def play_game(model, scfg, device, rng, max_plies):
    """Play one self-play game, returning per-side style observations."""
    board = chess.Board()
    obs = {
        chess.WHITE: {"castled": None, "first_q": None, "wing": 0},
        chess.BLACK: {"castled": None, "first_q": None, "wing": 0},
    }
    captures = 0
    sans = []
    ply = 0
    while _terminal_value(board) is None and ply < max_plies:
        mv = search_move(model, board, scfg, device, rng)
        mover = board.turn
        piece = board.piece_at(mv.from_square)
        san = board.san(mv)
        if board.is_capture(mv):
            captures += 1
        if san.startswith("O-O") and obs[mover]["castled"] is None:
            obs[mover]["castled"] = ply
        if piece is not None and piece.piece_type == chess.QUEEN and obs[mover]["first_q"] is None:
            obs[mover]["first_q"] = ply
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_file(mv.from_square) in WING_FILES
            and ply < 24
        ):
            obs[mover]["wing"] += 1
        sans.append(san)
        board.push(mv)
        ply += 1

    # center occupation at end of opening (~ply 12) read from the live board
    center = sum(
        1
        for sq in CENTER_SQUARES
        if (p := board.piece_at(sq)) is not None and p.piece_type == chess.PAWN
    )
    n_moves = max(1, ply)
    return {
        "obs": obs,
        "capture_rate": captures / n_moves,
        "center": center,
        "plies": ply,
        "first": sans[0] if sans else None,
        "reply": sans[1] if len(sans) > 1 else None,
    }


def summarize(games, max_plies):
    """Aggregate per-game observations into style metrics + the alien index."""

    def castle_rate(color):
        return np.mean([g["obs"][color]["castled"] is not None for g in games])

    def first_q(color):
        # un-moved queen counts as max_plies (very late / never)
        vals = [
            g["obs"][color]["first_q"] if g["obs"][color]["first_q"] is not None else max_plies
            for g in games
        ]
        return float(np.mean(vals))

    def wing(color):
        return float(np.mean([g["obs"][color]["wing"] for g in games]))

    castle_all = float(np.mean([castle_rate(c) for c in (chess.WHITE, chess.BLACK)]))
    wing_all = (wing(chess.WHITE) + wing(chess.BLACK)) / 2
    fq_all = (first_q(chess.WHITE) + first_q(chess.BLACK)) / 2

    first_book = np.mean([g["first"] in WHITE_BOOK for g in games if g["first"]])
    reply_book = np.mean(
        [g["reply"] in BLACK_BOOK.get(g["first"], set()) for g in games if g["reply"]]
    )

    # alien components, each in [0,1], higher = more alien
    a_castle = 1.0 - castle_all
    a_wing = min(wing_all / 3.0, 1.0)
    a_queen = float(np.clip((12 - fq_all) / 12.0, 0.0, 1.0))
    a_book = 1.0 - float(reply_book)
    alien_index = 100.0 * float(np.mean([a_castle, a_wing, a_queen, a_book]))

    return {
        "alien_index": round(alien_index, 1),
        "components": {
            "uncastled": round(a_castle, 3),
            "wing_storm": round(a_wing, 3),
            "early_queen": round(a_queen, 3),
            "off_book_reply": round(a_book, 3),
        },
        "castle_rate_white": round(float(castle_rate(chess.WHITE)), 3),
        "castle_rate_black": round(float(castle_rate(chess.BLACK)), 3),
        "first_queen_ply": round(fq_all, 1),
        "wing_pawn_pushes": round(wing_all, 2),
        "capture_rate": round(float(np.mean([g["capture_rate"] for g in games])), 3),
        "center_pawns": round(float(np.mean([g["center"] for g in games])), 2),
        "book_first_move": round(float(first_book), 3),
        "book_reply": round(float(reply_book), 3),
        "n_games": len(games),
        "avg_plies": round(float(np.mean([g["plies"] for g in games])), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--out", default=None, help="write JSON metrics here")
    args = ap.parse_args()

    model = load_checkpoint(args.ckpt)
    scfg = SearchConfig(sims=args.sims, root_candidates=16)
    device = torch.device("cpu")

    games = [
        play_game(model, scfg, device, np.random.default_rng(g), args.plies)
        for g in range(args.games)
    ]
    m = summarize(games, args.plies)

    print(f"\n== alien-ness of {args.ckpt} ({m['n_games']} games, {m['avg_plies']} avg plies) ==")
    print(f"\n  ALIEN INDEX: {m['alien_index']} / 100   (0 = textbook human, 100 = full alien)")
    c = m["components"]
    print("    components (higher = more alien):")
    print(f"      uncastled king   {c['uncastled']:.2f}   (castle rate W {m['castle_rate_white']:.0%} / B {m['castle_rate_black']:.0%})")
    print(f"      wing-pawn storm  {c['wing_storm']:.2f}   ({m['wing_pawn_pushes']:.2f} a/b/g/h pushes / game, first 24 plies)")
    print(f"      early queen      {c['early_queen']:.2f}   (first queen move ~ply {m['first_queen_ply']:.0f})")
    print(f"      off-book reply   {c['off_book_reply']:.2f}   (book first move {m['book_first_move']:.0%}, book reply {m['book_reply']:.0%})")
    print(f"\n    other: capture rate {m['capture_rate']:.0%}, center pawns {m['center_pawns']:.2f}/4")

    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, **m}, indent=2))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
