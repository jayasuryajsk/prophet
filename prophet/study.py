"""Study-your-losses: extract extra lessons from each self-play game.

A human reviews a lost game, finds the move where it went wrong, and works
out what should have been played. Mechanized here:

1. Surprise detection — score each ply by (a) the value swing the mover
   suffered (root value before the move vs. negated root value after; a
   blunder makes this large) and (b) disagreement between the search value
   and the actual game outcome.
2. Deep re-analysis — re-search the top-K surprise positions with a much
   larger sim budget, producing high-weight training samples with sharper
   policy/Q/value targets.
3. Counterfactual branch — when the deep search disagrees with the move
   that was actually played, play the better move out for a few plies at
   the normal budget and train on that line too: the lesson the game
   itself never showed.

Generator core (yields features, receives evals) so the vectorized worker
can batch study evals alongside game evals; study_game() is the
synchronous wrapper.
"""

from dataclasses import dataclass

import numpy as np

from .fastboard import board_from_fen
from .search import SearchConfig, drive, run_search_gen
from .selfplay import GameRecord, Sample


@dataclass
class StudyConfig:
    top_k: int = 2
    min_surprise: float = 0.15
    deep_sims: int = 128
    deep_candidates: int = 16
    branch_plies: int = 16
    study_weight: float = 2.0
    branch_weight: float = 1.0
    outcome_mix: float = 0.5
    n_lines: int = 1  # alternate lines explored per surprise position
    q_surprise_weight: float = 1.0  # weight of Q-head surprise in detection


def find_surprises(record: GameRecord, cfg: StudyConfig) -> list[int]:
    """Indices of the most surprising plies, by:
      - blunder swing: value flipped against the mover after the move
      - outcome miss: search value disagreed with the eventual result
      - Q-surprise: the Q-head's value for the move it played diverged from
        what the move actually led to (negamax: Q(s,a) should = -V(child)).
    Q-surprise is the move-level "my intuition was wrong here" signal —
    exactly the positions worth re-studying along multiple lines.
    """
    v = record.root_values
    qhp = record.q_head_played
    z_white = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(record.result)
    mover_white = [fen.split()[1] == "w" for fen in record.fens]
    scores = []
    for t in range(len(v)):
        swing = max(0.0, v[t] + v[t + 1]) if t + 1 < len(v) else 0.0
        outcome = 0.0
        if z_white is not None:
            z_t = z_white if mover_white[t] else -z_white
            outcome = abs(v[t] - z_t)
        # Q-head predicted qhp[t] for the played move; it actually led to a
        # position worth -v[t+1] to the mover. The gap is the Q-surprise.
        q_surprise = 0.0
        if qhp is not None and t + 1 < len(v):
            q_surprise = abs(qhp[t] + v[t + 1])
        scores.append(swing + 0.5 * outcome + cfg.q_surprise_weight * q_surprise)
    order = sorted(range(len(v)), key=lambda t: -scores[t])
    return [t for t in order[: cfg.top_k] if scores[t] >= cfg.min_surprise]


def _top_line_indices(res, n: int) -> list[int]:
    """The n moves the deep search rated highest (by empirical search Q) —
    the alternate lines worth playing out from a surprise position."""
    if n <= 0 or len(res.q_indices) == 0:
        return []
    order = np.argsort(-res.q_values)[:n]
    return [int(res.q_indices[j]) for j in order]


def _sample_from_search(board, res, value_target, weight) -> Sample:
    x = board.encode()
    board.push_action(res.move_index)
    child_x = board.encode()
    board.pop()
    return Sample(
        x=x,
        legal_indices=res.legal_indices,
        policy_target=res.policy_target,
        value_target=float(value_target),
        q_indices=res.q_indices,
        q_values=res.q_values,
        q_visits=res.q_visits,
        played_index=res.move_index,
        child_x=child_x,
        weight=weight,
    )


def _play_branch_gen(board, scfg, cfg, rng):
    """Continue self-play from a counterfactual position for a few plies."""
    raw = []  # (sample, mover_was_white)
    while board.terminal_value() is None and len(raw) < cfg.branch_plies:
        res = yield from run_search_gen(board, scfg, rng)
        raw.append(
            (
                _sample_from_search(board, res, res.root_value, cfg.branch_weight),
                board.turn,
            )
        )
        board.push_action(res.move_index)
    term = board.terminal_value()
    if term is not None:
        if term == -1.0:  # side to move is checkmated
            z_white = -1.0 if board.turn else 1.0
        else:
            z_white = 0.0
        for s, mover_was_white in raw:
            z = z_white if mover_was_white else -z_white
            s.value_target = (1 - cfg.outcome_mix) * s.value_target + cfg.outcome_mix * z
    return [s for s, _ in raw]


def study_game_gen(
    record: GameRecord,
    scfg: SearchConfig,
    cfg: StudyConfig,
    rng: np.random.Generator,
):
    if not record.fens:
        return []
    deep_cfg = SearchConfig(
        sims=cfg.deep_sims,
        root_candidates=cfg.deep_candidates,
        q_trust=scfg.q_trust,
    )
    out = []
    for t in find_surprises(record, cfg):
        board = board_from_fen(record.fens[t])
        res = yield from run_search_gen(board, deep_cfg, rng)
        out.append(_sample_from_search(board, res, res.root_value, cfg.study_weight))
        # multi-line reflection: play out each of the deep search's top moves
        # as its own branch, so a surprising position teaches a whole tree of
        # lines (the move played, plus the alternates the deep search preferred)
        for mv_idx in _top_line_indices(res, cfg.n_lines):
            board.push_action(int(mv_idx))
            out.extend((yield from _play_branch_gen(board, scfg, cfg, rng)))
            board.pop()
    return out


def study_game(
    model,
    record: GameRecord,
    scfg: SearchConfig,
    cfg: StudyConfig,
    device,
    rng: np.random.Generator,
) -> list[Sample]:
    return drive(study_game_gen(record, scfg, cfg, rng), model, device)
