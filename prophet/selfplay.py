"""Self-play game generation (generator core + synchronous wrapper).

Every position yields a training sample with dense targets:
- improved policy from the Gumbel root (all legal moves)
- value target = search root value, blended with the real outcome when the
  game actually finished (truncated games fall back to pure search value)
- per-move Q targets for every child the search visited
- the child position of the move actually played, for the negamax
  consistency loss Q(s, a) ~= -V(s')

Resignation (gated; see worker gate file): when the side to move has seen
its own root value below resign_threshold for resign_plies consecutive
turns, the game is adjudicated. A fraction of games (resign_off_prob)
ignores resignation so miscalibration would show up as losses there.
"""

from dataclasses import dataclass, replace

import numpy as np

from .fastboard import new_board
from .search import SearchConfig, drive, run_search_gen


@dataclass
class SelfPlayConfig:
    max_plies: int = 200
    outcome_mix: float = 0.5  # weight of real outcome vs search value
    resign_threshold: float = -0.92
    resign_plies: int = 8
    resign_off_prob: float = 0.1
    contempt: float = 0.15  # draws train as -contempt for both sides
    win_discount: float = 0.997  # per-ply discount: faster wins are worth more
    td_steps: int = 8  # n-step value horizon: bootstrap from the search value
    # n plies ahead instead of smearing the final outcome over every position
    td_outcome_leak: float = 0.25  # fraction of true outcome kept inside the
    # horizon as the grounding anchor (prevents pure bootstrap echo)
    # playout-cap randomization (KataGo): most moves searched cheaply (more
    # games per FLOP -> more outcome bits for the value head), a random
    # fraction at full budget (quality policy targets — policy trains only
    # on those, via Sample.policy_ok).
    pcr_prob: float = 0.0  # fraction of FULL-budget moves; 0 disables PCR
    pcr_cheap_sims: int = 12


@dataclass
class Sample:
    x: np.ndarray  # [64, F] position, side-to-move perspective
    legal_indices: np.ndarray
    policy_target: np.ndarray  # aligned with legal_indices
    value_target: float
    q_indices: np.ndarray
    q_values: np.ndarray
    q_visits: np.ndarray
    played_index: int
    child_x: np.ndarray  # [64, F] position after the played move
    weight: float = 1.0  # loss weight (study/branch samples get > 1)
    wdl: int = -1  # outcome class for side to move: 0=loss 1=draw 2=win, -1=unknown
    moves_left: float = -1.0  # plies until the game actually ended; -1 unknown
    policy_ok: bool = True  # False for cheap-search moves (PCR): skip policy CE


@dataclass
class GameRecord:
    samples: list
    result: str  # "1-0", "0-1", "1/2-1/2", or "*" if truncated
    plies: int
    fens: list | None = None  # position before each ply, aligned with samples
    root_values: list | None = None  # search root value per ply
    q_head_played: list | None = None  # Q-head value for the played move per ply


def play_game_gen(
    search_cfg: SearchConfig,
    cfg: SelfPlayConfig,
    rng: np.random.Generator,
    board=None,
    resign_enabled: bool = False,
):
    board = board or new_board()
    raw = []  # (x, search result, child features, mover_was_white, full_search)
    fens = []
    resign_active = resign_enabled and rng.random() >= cfg.resign_off_prob
    low_streak = {True: 0, False: 0}  # keyed by mover_was_white
    resigned_winner_white = None
    cheap_cfg = (
        replace(
            search_cfg,
            sims=cfg.pcr_cheap_sims,
            root_candidates=min(search_cfg.root_candidates, 4),
        )
        if cfg.pcr_prob > 0
        else None
    )

    while board.terminal_value() is None and len(raw) < cfg.max_plies:
        mover_white = board.turn
        fens.append(board.fen())
        x = board.encode()
        full = cheap_cfg is None or rng.random() < cfg.pcr_prob
        res = yield from run_search_gen(board, search_cfg if full else cheap_cfg, rng)
        board.push_action(res.move_index)
        child_x = board.encode()
        raw.append((x, res, child_x, mover_white, full))

        if res.root_value < cfg.resign_threshold:
            low_streak[mover_white] += 1
        else:
            low_streak[mover_white] = 0
        if resign_active and low_streak[mover_white] >= cfg.resign_plies:
            resigned_winner_white = not mover_white  # mover resigns; opponent wins
            break

    end_known = True  # true game length known (for the moves-left target)
    if resigned_winner_white is not None:
        result = "1-0" if resigned_winner_white else "0-1"
        z_white = 1.0 if resigned_winner_white else -1.0
    else:
        term = board.terminal_value()
        if term is None:
            # ply-cap truncation: score as a draw (contempt applies) so the
            # outcome signal reaches exactly the games where conversion
            # failed — otherwise these samples train on pure search value
            # (the net's own opinion) and the value head never learns here.
            result = "*"
            z_white = 0.0
            end_known = False  # cap hit; true game length unknowable
        elif term == -1.0:  # side to move is checkmated
            result = "0-1" if board.turn else "1-0"
            z_white = -1.0 if board.turn else 1.0
        else:
            result = "1/2-1/2"
            z_white = 0.0

    samples = []
    total = len(raw)
    for t, (x, res, child_x, mover_was_white, full_search) in enumerate(raw):
        v = res.root_value
        wdl = -1
        if z_white is not None:
            z = z_white if mover_was_white else -z_white
            wdl = int(z) + 1  # -1/0/1 -> 0/1/2
            if z == 0.0:
                z_eff = -cfg.contempt  # a draw should taste bad to both sides
            else:
                z_eff = z * cfg.win_discount ** (total - t)  # urgency
            # n-step TD: beyond the horizon, bootstrap from the search's root
            # value n plies ahead (sign flips with parity) instead of smearing
            # the final outcome over the whole game — localized credit, with
            # td_outcome_leak of the true outcome kept as the anchor.
            n = cfg.td_steps
            if n > 0 and t + n < total:
                v_ahead = raw[t + n][1].root_value
                boot = (v_ahead if n % 2 == 0 else -v_ahead) * cfg.win_discount**n
                z_eff = (1 - cfg.td_outcome_leak) * boot + cfg.td_outcome_leak * z_eff
            v = (1 - cfg.outcome_mix) * v + cfg.outcome_mix * z_eff
        samples.append(
            Sample(
                x=x,
                legal_indices=res.legal_indices,
                policy_target=res.policy_target,
                value_target=float(v),
                q_indices=res.q_indices,
                q_values=res.q_values,
                q_visits=res.q_visits,
                played_index=res.move_index,
                child_x=child_x,
                wdl=wdl,
                moves_left=float(total - t) if end_known else -1.0,
                policy_ok=full_search,
            )
        )
    return GameRecord(
        samples=samples,
        result=result,
        plies=len(raw),
        fens=fens,
        root_values=[res.root_value for _, res, *_ in raw],
        q_head_played=[res.q_head_played for _, res, *_ in raw],
    )


def play_game(
    model,
    search_cfg: SearchConfig,
    cfg: SelfPlayConfig,
    device,
    rng: np.random.Generator,
    board=None,
    resign_enabled: bool = False,
) -> GameRecord:
    return drive(
        play_game_gen(search_cfg, cfg, rng, board=board, resign_enabled=resign_enabled),
        model,
        device,
    )
