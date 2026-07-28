"""Phase C driver: the search tree lives in Rust (prophet_core.BatchSearch);
Python only evaluates position batches with the net. Same recipe semantics
as search.py / searchB.py.

The GAME HISTORY is replayed into the Rust tree (base fen + action list),
not just the root FEN: repetition detection, the last-two-moves planes and
the parity plane all live in the history. A bare-FEN tree is blind to
threefold draws (it cost two won live games) and feeds the net skewed
features.
"""

import chess
import numpy as np
import torch

import prophet_core

from .encoding import move_to_index
from .fastboard import PyChessBoard

F = 24


def board_history(board: chess.Board):
    """(base_fen, action list) reconstructing `board` from its move stack.
    Underpromotions can't round-trip through the 4096 action space, so the
    history restarts just after the most recent one (repetition can't span
    an irreversible move anyway — every promotion resets the fifty-move
    clock, so no legal repetition reaches back past it)."""
    stack = list(board.move_stack)
    tmp = board.root()
    base_fen = tmp.fen()
    actions = []
    for mv in stack:
        if mv.promotion is not None and mv.promotion != chess.QUEEN:
            tmp.push(mv)
            base_fen = tmp.fen()
            actions = []
            continue
        actions.append(move_to_index(mv, tmp.turn == chess.BLACK))
        tmp.push(mv)
    return base_fen, actions


class RustBatchedSearcher:
    def __init__(self, model, budget=1024, batch=16, candidates=16,
                 q_trust=1.0, c_puct=1.5, c_visit=50.0, c_scale=1.0,
                 contempt=0.0, seed=0):
        self.model = model
        self.args = (budget, batch, candidates, c_puct, c_visit, c_scale,
                     q_trust, contempt, seed)

    @torch.no_grad()
    def _eval(self, x_np):
        logits, adv, v = self.model(torch.from_numpy(x_np))
        return (logits.numpy().astype("<f4").tobytes(),
                adv.numpy().astype("<f4").tobytes(),
                v.numpy().astype("<f4").tobytes())

    def search(self, board):
        """board: python-chess Board. Returns (chess.Move, spent)."""
        budget, batch, candidates, cp, cv, cs, qt, ct, seed = self.args
        base_fen, hist = board_history(board)
        t = prophet_core.BatchSearch(base_fen, budget, batch, candidates,
                                     cp, cv, cs, qt, ct, seed, history=hist)
        x = np.asarray(t.root_features(), dtype=np.float32).reshape(1, 64, F)
        lb, ab, vb = self._eval(x)
        t.set_root(lb, ab, np.frombuffer(vb, dtype="<f4")[0].item())
        guard = 0
        while not t.done():
            fb = t.collect()
            n = t.n_pending()
            if n == 0:
                guard += 1
                if guard > 4 * budget:
                    break
                continue
            guard = 0
            x = np.frombuffer(fb, dtype="<f4").reshape(n, 64, F).copy()
            lb, ab, vb = self._eval(x)
            t.apply(lb, ab, vb)
        action = t.best()
        pb = PyChessBoard(board)
        return pb.move_for(int(action)), int(t.spent_forwards())
