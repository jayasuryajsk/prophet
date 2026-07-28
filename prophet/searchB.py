"""Phase B: batched-leaf Gumbel search (serving path).

Same recipe as search.py — Gumbel top-k root, sequential halving, PUCT
descent, q_trust warm-start, parity contempt — but leaf evaluations are
COLLECTED IN BATCHES via virtual loss and evaluated in one forward pass,
amortizing the per-call dispatch overhead that dominates a 10M net.

Invariants:
- a path is [(parent, action, child), ...] from the root to a leaf
- every CHILD on a pending path carries +1 vloss (removed at backup)
- boards are pushed during selection and popped before returning
- leaves store their legal-action set at selection time, so expansion
  after the batched eval needs no board replay

Training continues to use search.py unchanged.
"""

import math

import numpy as np
import torch


class NodeB:
    __slots__ = ("prior", "q_init", "visits", "total", "vloss", "expanded", "children")

    def __init__(self, prior=0.0, q_init=0.0):
        self.prior = prior
        self.q_init = q_init
        self.visits = 0
        self.total = 0.0
        self.vloss = 0
        self.expanded = False
        self.children = None


class BatchedSearcher:
    def __init__(self, model, budget=1024, candidates=16, batch=16,
                 q_trust=1.0, c_puct=1.5, c_visit=50.0, c_scale=1.0,
                 contempt=0.0, seed=0):
        self.model = model
        self.budget = budget
        self.candidates = candidates
        self.batch = batch
        self.q_trust = q_trust
        self.c_puct = c_puct
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.contempt = contempt
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def _eval_batch(self, xs):
        x = torch.from_numpy(np.stack(xs))
        logits, adv, v = self.model(x)
        return logits.numpy(), adv.numpy(), v.numpy()

    def _table(self, legal_idx, logits_row, adv_row, v):
        idx = legal_idx
        lg = logits_row[idx] - logits_row[idx].max()
        p = np.exp(lg)
        p /= p.sum()
        a = adv_row[idx]
        qv = np.tanh(np.arctanh(np.clip(float(v), -0.997, 0.997)) + a - a.max())
        return {
            int(i): NodeB(prior=float(pr), q_init=self.q_trust * float(q))
            for i, pr, q in zip(idx, p, qv)
        }

    def _pick(self, node):
        best_i, best_s = -1, -math.inf
        sqrt_n = math.sqrt(max(1, node.visits + node.vloss))
        for i, c in node.children.items():
            vis = c.visits + c.vloss
            q = (-(c.total - c.vloss) / vis) if vis else c.q_init
            s = q + self.c_puct * c.prior * sqrt_n / (1 + vis)
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    def _backup(self, path, leaf_value):
        """leaf_value is from the LAST child's side-to-move perspective."""
        v = leaf_value
        for parent, action, child in reversed(path):
            child.visits += 1
            child.total += v
            if child.vloss > 0:
                child.vloss -= 1
            v = -v
        if path:
            path[0][0].visits += 1  # root visit count (cosmetic)

    def _collect(self, pb, root, first_action):
        """One pending path starting with first_action from the root.
        Returns ('eval', path, x, legal) with pb restored, or None if the
        path hit a terminal (already backed up)."""
        path = []
        node = root
        action = first_action
        while True:
            child = node.children[action]
            pb.push_action(action)
            path.append((node, action, child))
            child.vloss += 1
            if not child.expanded:
                term = pb.terminal_value()
                if term is not None:
                    if term == 0.0 and self.contempt:
                        d = len(path)  # child is at depth d; root player at even d
                        term = -self.contempt if d % 2 == 0 else self.contempt
                    self._backup(path, term)
                    for _ in path:
                        pb.pop()
                    return None
                x = pb.encode()
                legal = np.asarray(pb.legal_actions(), dtype=np.int64)
                for _ in path:
                    pb.pop()
                return ("eval", path, x, legal)
            node = child
            action = self._pick(node)

    def search(self, pb):
        """Full budgeted search from pb's position -> (best_action, spent)."""
        logits, adv, v = self._eval_batch([pb.encode()])
        legal0 = np.asarray(pb.legal_actions(), dtype=np.int64)
        root = NodeB()
        root.children = self._table(legal0, logits[0], adv[0], float(v[0]))
        root.expanded = True
        root.visits = 1

        g = self.rng.gumbel(size=len(legal0))
        base = {int(i): float(logits[0][i]) + g[j] for j, i in enumerate(legal0)}
        m = min(self.candidates, len(legal0))
        remaining = sorted((int(i) for i in legal0), key=lambda i: -base[i])[:m]

        def completed_q(i):
            c = root.children[i]
            return -(c.total / c.visits) if c.visits else c.q_init

        spent = 1
        phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        per_phase = max(1, (self.budget - 1) // phases)

        while spent < self.budget and remaining:
            phase_spent = 0
            while phase_spent < per_phase and spent < self.budget:
                jobs = []
                want = min(self.batch, self.budget - spent, per_phase - phase_spent)
                paths_per = max(1, want // len(remaining))
                for i in remaining:
                    for _ in range(paths_per):
                        if len(jobs) >= want:
                            break
                        got = self._collect(pb, root, i)
                        if got is None:  # terminal, already backed up
                            spent += 1
                            phase_spent += 1
                        else:
                            jobs.append(got)
                    if len(jobs) >= want:
                        break
                if not jobs:
                    if phase_spent == 0:
                        break  # nothing collectable (all-terminal subtrees)
                    continue
                L, A, V = self._eval_batch([x for _, _, x, _ in jobs])
                for j, (_, path, x, legal) in enumerate(jobs):
                    leaf = path[-1][2]
                    if not leaf.expanded:
                        leaf.children = self._table(legal, L[j], A[j], float(V[j]))
                        leaf.expanded = True
                    self._backup(path, float(V[j]))
                spent += len(jobs)
                phase_spent += len(jobs)
            if len(remaining) > 1:
                sig = (self.c_visit + max(c.visits for c in root.children.values())) * self.c_scale
                remaining.sort(key=lambda i: base[i] + sig * completed_q(i), reverse=True)
                remaining = remaining[: max(1, len(remaining) // 2)]
            else:
                break

        sig = (self.c_visit + max(c.visits for c in root.children.values())) * self.c_scale
        pool = remaining or list(root.children.keys())
        best = max(pool, key=lambda i: base.get(i, -1e9) + sig * completed_q(i))
        return best, spent
