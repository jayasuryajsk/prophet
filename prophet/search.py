"""Gumbel-style search at tiny simulation budgets.

Root: Gumbel top-k candidate selection + sequential halving, acting on
completed Q-values (visited children use empirical search Q, unvisited
children are completed with the network's Q-head — the model's per-move
intuition fills in what search didn't verify).

Interior: PUCT, with unvisited children initialized from the Q-head
instead of zero, so calculation starts where intuition already points.

All values are in [-1, 1] from the perspective of the side to move at the
node's own position; a parent reads a child's value negated (negamax).

The core is written as GENERATORS that yield feature arrays [64, F] and
receive (policy_logits[4096], q[4096], v) numpy triples via .send(). This
lets a driver multiplex hundreds of concurrent games into batched network
evaluations (see worker.run_vector_selfplay). Batching happens across
games, never within one search, so search semantics are exact.
run_search() is the synchronous single-eval wrapper.
"""

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from .encoding import encode_board, legal_move_map

DRAW_HALFMOVE_CAP = 100  # fifty-move rule


@dataclass
class SearchConfig:
    sims: int = 32
    root_candidates: int = 8
    c_puct: float = 1.5
    c_visit: float = 50.0
    c_scale: float = 1.0
    q_trust: float = 1.0  # how much search trusts the Q-head for unvisited
    # children (0 early when Q is noise, ramp to 1 as the Q-head matures)


@dataclass
class Node:
    prior: float = 0.0
    q_init: float = 0.0  # network Q from the PARENT's perspective
    visits: int = 0
    total: float = 0.0  # sum of values from THIS node's perspective
    expanded: bool = False
    children: dict = field(default_factory=dict)  # action index -> Node

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    move_index: int  # chosen action (from*64 + to, side-to-move flipped)
    root_value: float
    legal_indices: np.ndarray  # [L] action indices
    policy_target: np.ndarray  # [L], sums to 1, aligned with legal_indices
    q_indices: np.ndarray  # [K] visited children
    q_values: np.ndarray  # [K] empirical search Q (root perspective)
    q_visits: np.ndarray  # [K]
    q_head_played: float = 0.0  # Q-head's raw value for the played move
    # (compare to -V(child) to detect where intuition was surprised)


def _evaluate_gen(board):
    """Generator: yields features, receives (logits, q, v); returns
    (legal action indices, priors, q-by-index, v, raw logits). `board` is any
    object implementing the fastboard protocol (encode / legal_actions / ...)."""
    x = board.encode()
    logits, q, v = yield x
    idx = np.asarray(board.legal_actions(), dtype=np.int64)
    lg = logits[idx]
    lg = lg - lg.max()
    p = np.exp(lg)
    p /= p.sum()
    priors = dict(zip(idx.tolist(), p.tolist()))
    qs = {i: float(q[i]) for i in idx.tolist()}
    return idx, priors, qs, float(v), logits


def _terminal_value(board: chess.Board) -> float | None:
    """Value for the side to move, or None if not terminal (python-chess)."""
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.halfmove_clock >= DRAW_HALFMOVE_CAP
        or board.is_repetition(3)
    ):
        return 0.0
    return None


def _expand_gen(node: Node, board):
    idx, priors, qs, v, _ = yield from _evaluate_gen(board)
    for i in idx.tolist():
        node.children[i] = Node(prior=priors[i], q_init=qs[i])
    node.expanded = True
    return v


def _simulate_gen(board, node: Node, cfg: SearchConfig):
    """One playout; returns value from this node's side-to-move perspective."""
    term = board.terminal_value()
    if term is not None:
        node.visits += 1
        node.total += term
        return term
    if not node.expanded:
        v = yield from _expand_gen(node, board)
        node.visits += 1
        node.total += v
        return v
    sqrt_n = math.sqrt(node.visits)
    best_i, best_score = -1, -math.inf
    for i, child in node.children.items():
        q = -child.mean if child.visits else cfg.q_trust * child.q_init
        score = q + cfg.c_puct * child.prior * sqrt_n / (1 + child.visits)
        if score > best_score:
            best_i, best_score = i, score
    child = node.children[best_i]
    board.push_action(best_i)
    v = -(yield from _simulate_gen(board, child, cfg))
    board.pop()
    node.visits += 1
    node.total += v
    return v


def run_search_gen(board, cfg: SearchConfig, rng: np.random.Generator):
    """Generator form of the full search; returns a SearchResult. `board` is
    any object implementing the fastboard protocol."""
    idx, priors, qs, v_root, logits = yield from _evaluate_gen(board)
    if len(idx) == 0:
        raise ValueError(f"no legal moves to search: {board.fen()}")

    root = Node()
    for i in idx.tolist():
        root.children[i] = Node(prior=priors[i], q_init=qs[i])
    root.expanded = True
    root.visits = 1
    root.total = v_root

    # Gumbel top-k candidates by g + logit
    g = rng.gumbel(size=len(idx))
    base = logits[idx] + g
    m = min(cfg.root_candidates, len(idx))
    order = np.argsort(-base)
    candidates = [int(idx[j]) for j in order[:m]]
    base_by_idx = {int(idx[j]): float(base[j]) for j in range(len(idx))}

    def completed_q(i: int) -> float:
        child = root.children[i]
        return -child.mean if child.visits else cfg.q_trust * child.q_init

    def sigma(q: float) -> float:
        max_visits = max((c.visits for c in root.children.values()), default=0)
        return (cfg.c_visit + max_visits) * cfg.c_scale * q

    # Sequential halving over the candidate set
    sims_used = 0
    remaining = list(candidates)
    phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
    while sims_used < cfg.sims:
        per = max(1, cfg.sims // (phases * max(1, len(remaining))))
        for i in remaining:
            for _ in range(per):
                if sims_used >= cfg.sims:
                    break
                child = root.children[i]
                board.push_action(i)
                val = -(yield from _simulate_gen(board, child, cfg))
                board.pop()
                root.visits += 1
                root.total += val
                sims_used += 1
        if len(remaining) > 1:
            remaining.sort(key=lambda i: base_by_idx[i] + sigma(completed_q(i)), reverse=True)
            remaining = remaining[: max(1, len(remaining) // 2)]
        elif sims_used >= cfg.sims:
            break

    best = max(remaining, key=lambda i: base_by_idx[i] + sigma(completed_q(i)))

    # Improved policy over ALL legal moves: softmax(logits + sigma(completedQ))
    comp = np.array([completed_q(int(i)) for i in idx])
    pi_logits = logits[idx] + np.array([sigma(float(c)) for c in comp])
    pi_logits -= pi_logits.max()
    pi = np.exp(pi_logits)
    pi /= pi.sum()

    # Root value: network value blended with visit-weighted child Q
    n_sum = sum(c.visits for c in root.children.values())
    if n_sum:
        q_avg = (
            sum(c.visits * -c.mean for c in root.children.values() if c.visits) / n_sum
        )
        root_value = (v_root + n_sum * q_avg) / (1 + n_sum)
    else:
        root_value = v_root

    visited = [(i, c) for i, c in root.children.items() if c.visits]
    return SearchResult(
        move_index=best,
        root_value=float(root_value),
        legal_indices=idx,
        policy_target=pi.astype(np.float32),
        q_indices=np.array([i for i, _ in visited], dtype=np.int64),
        q_values=np.array([-c.mean for _, c in visited], dtype=np.float32),
        q_visits=np.array([c.visits for _, c in visited], dtype=np.float32),
        q_head_played=float(qs.get(best, 0.0)),
    )


@torch.no_grad()
def _eval_single(model, x: np.ndarray, device):
    from .accel import autocast, to_np

    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with autocast(device):
        logits, q, v = model(xt)
    return to_np(logits[0]), to_np(q[0]), float(v[0])


def drive(gen, model, device):
    """Run a search/selfplay generator to completion with one-at-a-time
    network evals. Returns the generator's return value."""
    try:
        x = gen.send(None)
        while True:
            x = gen.send(_eval_single(model, x, device))
    except StopIteration as e:
        return e.value


def run_search(model, board, cfg: SearchConfig, device, rng: np.random.Generator) -> SearchResult:
    return drive(run_search_gen(board, cfg, rng), model, device)


def search_move(model, chess_board, cfg: SearchConfig, device, rng) -> chess.Move:
    """Run search on a python-chess board and return the chosen chess.Move
    (for eval / Stockfish interop). Leaves the board unchanged."""
    from .fastboard import PyChessBoard

    pb = PyChessBoard(chess_board)
    res = run_search(model, pb, cfg, device, rng)
    return pb.move_for(res.move_index)
