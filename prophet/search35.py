"""v3.5 inference search — same recipe as search.py's Gumbel root, plus:

1. TREE REUSE: the search tree persists across moves within a game; after
   our move and the opponent's reply, the matching grandchild subtree
   becomes the new root (its visits/values carry over — effectively a
   2-4x budget multiplier for free).
2. ADAPTIVE BUDGET: obvious moves (one dominant candidate) get B/4,
   critical moves (top candidates inseparable) get up to 2B, banked so the
   long-run average stays ~B.
3. MLH TIEBREAK: the trained moves-left head steers — when clearly
   winning, prefer lines that END sooner; when losing, later.
4. WDL kappa-DIAL: leaf value = P(win) - P(loss) - kappa*P(draw); kappa>0
   plays for blood, kappa=0 honest, kappa<0 favors solidity.

Play-path only. Training continues to use search.py unchanged.
"""

import math

import numpy as np
import torch

from .fastboard import PyChessBoard
from .search import _terminal_value


class N35:
    __slots__ = ("prior", "q_init", "visits", "total", "expanded", "children", "mlh")

    def __init__(self, prior=0.0, q_init=0.0):
        self.prior = prior
        self.q_init = q_init
        self.visits = 0
        self.total = 0.0
        self.expanded = False
        self.children = None
        self.mlh = None


class Searcher35:
    def __init__(self, model, budget=1024, candidates=16, q_trust=1.0,
                 c_puct=1.5, c_visit=50.0, c_scale=1.0, kappa=0.0,
                 mlh_lambda=0.05, contempt=0.0, seed=0):
        self.model = model
        self.budget = budget
        self.candidates = candidates
        self.q_trust = q_trust
        self.c_puct = c_puct
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.kappa = kappa
        self.mlh_lambda = mlh_lambda
        self.contempt = contempt
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.bank = 0  # saved forwards from cheap moves, spendable on crises

    # -- model eval ---------------------------------------------------------
    @torch.no_grad()
    def _eval(self, pb):
        dev = next(self.model.parameters()).device
        x = torch.from_numpy(pb.encode()).unsqueeze(0).to(dev)
        logits, adv, v, wdl, mlh = self.model.forward_wdl(x)
        logits, adv, v, wdl, mlh = (
            logits.cpu(), adv.cpu(), v.cpu(), wdl.cpu(), mlh.cpu()
        )
        logits = logits[0].numpy()
        adv = adv[0].numpy()
        w = wdl[0].numpy()
        v_eff = float(w[2] - w[0] - self.kappa * w[1])
        idx = np.asarray(pb.legal_actions(), dtype=np.int64)
        a = adv[idx]
        qv = np.tanh(np.arctanh(np.clip(v_eff, -0.997, 0.997)) + a - a.max())
        return idx, logits, qv, v_eff, float(mlh[0])

    # -- tree ops -----------------------------------------------------------
    def _expand(self, node, pb):
        idx, logits, qv, v, mlh = self._eval(pb)
        lg = logits[idx] - logits[idx].max()
        p = np.exp(lg)
        p /= p.sum()
        node.children = {
            int(i): N35(prior=float(pr), q_init=self.q_trust * float(q))
            for i, pr, q in zip(idx, p, qv)
        }
        node.expanded = True
        node.mlh = mlh
        return v

    def _simulate(self, pb, node, depth):
        if node.expanded:
            sqrt_n = math.sqrt(max(1, node.visits))
            best_i, best_s = -1, -math.inf
            for i, c in node.children.items():
                q = -(c.total / c.visits) if c.visits else c.q_init
                s = q + self.c_puct * c.prior * sqrt_n / (1 + c.visits)
                if s > best_s:
                    best_i, best_s = i, s
            child = node.children[best_i]
            pb.push_action(best_i)
            v = -self._simulate(pb, child, depth + 1)
            pb.pop()
            node.visits += 1
            node.total += v
            return v
        term = pb.terminal_value()
        if term is not None:
            if term == 0.0 and self.contempt:
                term = -self.contempt if depth % 2 == 0 else self.contempt
            node.visits += 1
            node.total += term
            return term
        v = self._expand(node, pb)
        node.visits += 1
        node.total += v
        return v

    # -- budget heuristic ---------------------------------------------------
    def _move_budget(self, rc, logits_idx):
        """Cheap for obvious moves, deep for crises; banked around self.budget."""
        pri = sorted((c.prior for c in rc.values()), reverse=True)
        qs = sorted(
            ((-(c.total / c.visits) if c.visits else c.q_init) for c in rc.values()),
            reverse=True,
        )
        gap = qs[0] - qs[1] if len(qs) > 1 else 1.0
        if len(pri) == 1 or (pri[0] >= 0.6 and gap >= 0.10):
            b = self.budget // 4
        elif gap <= 0.04:
            b = self.budget + min(self.bank, self.budget)  # crisis: spend the bank
        else:
            b = self.budget
        self.bank = int(np.clip(self.bank + (self.budget - b), 0, 4 * self.budget))
        return max(32, b)

    # -- main entry ---------------------------------------------------------
    def play(self, board):
        """Choose a move for the current position; reuses any carried tree."""
        pb = PyChessBoard(board)
        if self.root is None or not self.root.expanded:
            self.root = N35()
            v_root = self._expand(self.root, pb)
            self.root.visits += 1
            self.root.total += v_root
        rc = self.root.children

        idx = np.array(sorted(rc.keys()), dtype=np.int64)
        logits_by_i = {i: math.log(max(rc[i].prior, 1e-9)) for i in idx}
        g = self.rng.gumbel(size=len(idx))
        base = {int(i): logits_by_i[int(i)] + g[j] for j, i in enumerate(idx)}
        m = min(self.candidates, len(idx))
        remaining = sorted(idx.tolist(), key=lambda i: -base[i])[:m]

        budget = self._move_budget(rc, None)
        spent = 0
        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1

        def completed_q(i):
            c = rc[i]
            return -(c.total / c.visits) if c.visits else c.q_init

        while spent < budget:
            per = max(1, budget // (phases * max(1, len(remaining))))
            for i in remaining:
                for _ in range(per):
                    if spent >= budget:
                        break
                    pb.push_action(i)
                    val = -self._simulate(pb, rc[i], 1)
                    pb.pop()
                    self.root.visits += 1
                    self.root.total += val
                    spent += 1
            if len(remaining) > 1:
                sig = (self.c_visit + max(c.visits for c in rc.values())) * self.c_scale
                remaining.sort(key=lambda i: base[i] + sig * completed_q(i), reverse=True)
                remaining = remaining[: max(1, len(remaining) // 2)]
            elif spent >= budget:
                break

        # MLH steering among the finalists: when clearly winning prefer the
        # child whose subtree predicts a SOONER end; when losing, later.
        root_v = self.root.total / max(1, self.root.visits)
        sig = (self.c_visit + max(c.visits for c in rc.values())) * self.c_scale

        def util(i):
            u = base[i] + sig * completed_q(i)
            c = rc[i]
            if self.mlh_lambda and c.expanded and c.mlh is not None:
                if root_v > 0.3:
                    u -= self.mlh_lambda * sig * (c.mlh / 100.0)
                elif root_v < -0.3:
                    u += self.mlh_lambda * sig * (c.mlh / 100.0)
            return u

        best = max(remaining, key=util)
        return pb.move_for(best), spent

    def advance(self, action):
        """Shift the persistent root down one ply (our move or theirs)."""
        if self.root is not None and self.root.children is not None:
            self.root = self.root.children.get(int(action))
        else:
            self.root = None
