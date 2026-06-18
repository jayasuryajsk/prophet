"""Strength evaluation against a uniform-random opponent.

Modes:
- "q":      argmax over the Q-head, no search — pure intuition play.
            This curve over training IS the model's intuition forming.
- "policy": argmax over policy logits, no search.
- "search": full Gumbel search at the training sim budget.
"""

import torch  # noqa: I001  (torch before numpy; see README)

import chess
import numpy as np

from .encoding import encode_board, legal_move_map
from .search import SearchConfig, _terminal_value, search_move


PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def material_greedy_move(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    """Take the most valuable capture available (promotions count as gain);
    random among ties. No lookahead — it will recapture but never foresee."""
    best_score = -1
    best: list[chess.Move] = []
    for mv in board.legal_moves:
        s = 0
        if board.is_capture(mv):
            if board.is_en_passant(mv):
                s += 1
            else:
                s += PIECE_VALUE.get(board.piece_at(mv.to_square).piece_type, 0)
        if mv.promotion:
            s += PIECE_VALUE.get(mv.promotion, 0) - 1
        if s > best_score:
            best_score, best = s, [mv]
        elif s == best_score:
            best.append(mv)
    return best[rng.integers(len(best))]


@torch.no_grad()
def greedy_move(model, board: chess.Board, device, mode: str) -> chess.Move:
    x, flipped = encode_board(board)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    logits, q, _ = model(xt)
    scores = (q if mode == "q" else logits)[0].cpu().numpy()
    legal = legal_move_map(board, flipped)
    idx = np.fromiter(legal.keys(), dtype=np.int64)
    best = int(idx[np.argmax(scores[idx])])
    return legal[best]


def play_vs_random(
    model,
    device,
    n_games: int,
    mode: str,
    rng: np.random.Generator,
    max_plies: int = 300,
    search_cfg: SearchConfig | None = None,
    opponent: str = "random",
):
    """Returns (wins, draws, losses) from the model's perspective.

    Model alternates colors; unfinished games count as draws.
    opponent: "random" (uniform legal) or "material" (greedy capture-taker).
    """
    w = d = l = 0
    for g in range(n_games):
        model_is_white = g % 2 == 0
        board = chess.Board()
        plies = 0
        while _terminal_value(board) is None and plies < max_plies:
            if board.turn == (chess.WHITE if model_is_white else chess.BLACK):
                if mode == "search":
                    mv = search_move(model, board, search_cfg, device, rng)
                else:
                    mv = greedy_move(model, board, device, mode)
            elif opponent == "material":
                mv = material_greedy_move(board, rng)
            else:
                moves = list(board.legal_moves)
                mv = moves[rng.integers(len(moves))]
            board.push(mv)
            plies += 1
        if board.is_checkmate():
            winner_is_white = board.turn == chess.BLACK
            if winner_is_white == model_is_white:
                w += 1
            else:
                l += 1
        else:
            d += 1
    return w, d, l
