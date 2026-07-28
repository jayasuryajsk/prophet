"""Phase C driver: the search tree lives in Rust (prophet_core.BatchSearch);
Python only evaluates position batches with the net. Same recipe semantics
as search.py / searchB.py.
"""

import numpy as np
import torch

import prophet_core

from .fastboard import PyChessBoard

F = 24


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
        t = prophet_core.BatchSearch(board.fen(), budget, batch, candidates,
                                     cp, cv, cs, qt, ct, seed)
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
