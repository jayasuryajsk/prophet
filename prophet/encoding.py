"""Board and move encoding.

The model always sees the position from the side-to-move's perspective:
when black is to move the board is mirrored (vertical flip + color swap),
so square indices in "model space" differ from real-board squares by ^56.

Moves are encoded as from*64 + to (4096 actions). Promotions default to
queen; underpromotions are out of the v0 action space (a knight/rook/bishop
promotion is legal iff the queen promotion is, so no legal position becomes
unplayable).
"""

import chess
import numpy as np

NUM_ACTIONS = 64 * 64

# 12 piece planes + en-passant flag + 4 castling rights + halfmove clock
# + last 2 moves (from/to square flags, 4) + repetition flag + side parity
FEATURES = 24


def encode_board(board: chess.Board):
    """Return (features[64, FEATURES] float32, flipped: bool).

    Equivalent to encoding board.mirror() when black is to move, but reads
    bitboards directly and flips square indices with ^56 — no board copy
    (mirror() was ~25% of self-play wall-clock).
    """
    flipped = board.turn == chess.BLACK
    flip = 56 if flipped else 0
    us = board.turn
    x = np.zeros((64, FEATURES), dtype=np.float32)
    bitboards = (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    )
    for color, base in ((us, 0), (not us, 6)):
        occ = board.occupied_co[color]
        for pt, bb_all in enumerate(bitboards):
            bb = bb_all & occ
            while bb:
                sq = (bb & -bb).bit_length() - 1
                x[sq ^ flip, base + pt] = 1.0
                bb &= bb - 1
    if board.ep_square is not None:
        x[board.ep_square ^ flip, 12] = 1.0
    x[:, 13] = float(board.has_kingside_castling_rights(us))
    x[:, 14] = float(board.has_queenside_castling_rights(us))
    x[:, 15] = float(board.has_kingside_castling_rights(not us))
    x[:, 16] = float(board.has_queenside_castling_rights(not us))
    x[:, 17] = min(board.halfmove_clock, 100) / 100.0
    # history: last two moves as from/to square flags (raw experience, not
    # derived knowledge). Positions reconstructed from bare FENs (study)
    # simply have these zeroed.
    stack = board.move_stack
    if stack:
        mv = stack[-1]
        x[mv.from_square ^ flip, 18] = 1.0
        x[mv.to_square ^ flip, 19] = 1.0
    if len(stack) >= 2:
        mv = stack[-2]
        x[mv.from_square ^ flip, 20] = 1.0
        x[mv.to_square ^ flip, 21] = 1.0
    # repetition: this position has occurred before (threefold is a draw —
    # part of the true game state, invisible to a single-frame encoding).
    # A repeat needs >=4 reversible halfmoves; skip the stack walk otherwise.
    if board.halfmove_clock >= 4:
        x[:, 22] = float(board.is_repetition(2))
    x[:, 23] = float(len(stack) % 2)
    return x, flipped


def move_to_index(move: chess.Move, flipped: bool) -> int:
    f, t = move.from_square, move.to_square
    if flipped:
        f ^= 56
        t ^= 56
    return f * 64 + t


def index_to_move(index: int, board: chess.Board, flipped: bool) -> chess.Move:
    f, t = divmod(index, 64)
    if flipped:
        f ^= 56
        t ^= 56
    piece = board.piece_at(f)
    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(t) in (0, 7)
    ):
        return chess.Move(f, t, promotion=chess.QUEEN)
    return chess.Move(f, t)


def legal_move_map(board: chess.Board, flipped: bool) -> dict[int, chess.Move]:
    """Action index -> move for all legal moves (queen-only promotions)."""
    out = {}
    for mv in board.legal_moves:
        if mv.promotion is not None and mv.promotion != chess.QUEEN:
            continue
        out[move_to_index(mv, flipped)] = mv
    return out
