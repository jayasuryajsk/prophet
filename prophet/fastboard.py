"""Board adapters implementing the protocol the search uses:

    encode() -> np.ndarray [64, FEATURES]
    legal_actions() -> list[int]   (from*64+to, side-to-move flipped)
    push_action(a) / pop()
    terminal_value() -> float | None
    turn -> bool (True = white)
    fen() -> str   ;   from_fen(fen) -> Board

FastBoard wraps the Rust prophet_core.Board (self-play hot path, ~11x faster).
PyChessBoard wraps python-chess (eval / Stockfish interop, off the hot path).
Both behave identically — prophet_core is validated bit-identical to
python-chess (scripts/validate_core.py).
"""

import chess
import numpy as np

from .encoding import FEATURES, encode_board, index_to_move, legal_move_map

try:
    import prophet_core

    # guard against the source dir shadowing as a namespace package (dev box)
    _HAVE_RUST = hasattr(prophet_core, "Board")
except ImportError:  # pragma: no cover - rust core optional on dev machines
    _HAVE_RUST = False


class FastBoard:
    __slots__ = ("_b",)

    def __init__(self, _inner=None):
        self._b = _inner if _inner is not None else prophet_core.Board()

    @staticmethod
    def from_fen(fen: str) -> "FastBoard":
        return FastBoard(prophet_core.Board.from_fen(fen))

    def encode(self):
        return np.asarray(self._b.encode(), dtype=np.float32).reshape(64, FEATURES)

    def legal_actions(self):
        return self._b.legal_actions()

    def push_action(self, a):
        self._b.push_action(int(a))

    def pop(self):
        self._b.pop()

    def terminal_value(self):
        return self._b.terminal_value()

    @property
    def turn(self) -> bool:
        return self._b.turn

    def fen(self) -> str:
        return self._b.fen()


class PyChessBoard:
    """python-chess adapter — for eval/Stockfish, where moves cross to an
    external engine. Exposes the underlying chess.Board as `.board`."""

    __slots__ = ("board",)

    def __init__(self, board: chess.Board | None = None):
        self.board = board if board is not None else chess.Board()

    @staticmethod
    def from_fen(fen: str) -> "PyChessBoard":
        return PyChessBoard(chess.Board(fen))

    def encode(self):
        return encode_board(self.board)[0]

    def legal_actions(self):
        _, flipped = encode_board(self.board)
        return list(legal_move_map(self.board, flipped).keys())

    def move_for(self, a) -> chess.Move:
        _, flipped = encode_board(self.board)
        return index_to_move(int(a), self.board, flipped)

    def push_action(self, a):
        mv = self.move_for(a)
        if not self.board.is_legal(mv):  # match the Rust board's strictness
            raise ValueError(f"illegal action {a}")
        self.board.push(mv)

    def pop(self):
        self.board.pop()

    def terminal_value(self):
        from .search import _terminal_value

        return _terminal_value(self.board)

    @property
    def turn(self) -> bool:
        return self.board.turn

    def fen(self) -> str:
        return self.board.fen()


def new_board(fast: bool = True):
    """A fresh start-position board: Rust-backed when available, else python."""
    return FastBoard() if (fast and _HAVE_RUST) else PyChessBoard()


def board_from_fen(fen: str, fast: bool = True):
    cls = FastBoard if (fast and _HAVE_RUST) else PyChessBoard
    return cls.from_fen(fen)
