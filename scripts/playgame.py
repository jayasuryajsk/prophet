"""Play the checkpoint vs Stockfish at a fixed Elo — one game as White, one as
Black — printing the moves so you can see how it actually plays."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: I001

import chess
import chess.engine
import numpy as np

from prophet.model import load_checkpoint
from prophet.search import SearchConfig, search_move, _terminal_value

ckpt = sys.argv[1]
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ELO = int(sys.argv[3]) if len(sys.argv) > 3 else 1300
MAXPLIES = 200

model = load_checkpoint(ckpt)
cfg = SearchConfig(sims=SIMS, root_candidates=8)
dev = torch.device("cpu")


def play(model_white, seed):
    rng = np.random.default_rng(seed)
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"UCI_LimitStrength": True, "UCI_Elo": ELO})
    limit = chess.engine.Limit(time=0.1)
    board = chess.Board()
    sans = []
    while _terminal_value(board) is None and len(sans) < MAXPLIES:
        if board.turn == (chess.WHITE if model_white else chess.BLACK):
            mv = search_move(model, board, cfg, dev, rng)
        else:
            mv = eng.play(board, limit).move
        sans.append(board.san(mv))
        board.push(mv)
    eng.quit()
    return board, sans


def fmt(sans):
    out = []
    for i, s in enumerate(sans):
        out.append(f"{i//2+1}.{s}" if i % 2 == 0 else s)
    return " ".join(out)


def outcome(board, model_white):
    if board.is_checkmate():
        winner_white = board.turn == chess.BLACK
        return "MODEL WINS (checkmate)" if winner_white == model_white else "MODEL LOSES (checkmated)"
    if _terminal_value(board) is not None:
        return "draw"
    return "draw (move cap)"


print(f"ckpt={ckpt}  sims={SIMS}  vs Stockfish Elo {ELO}\n", flush=True)
for model_white, seed in [(True, 1), (False, 2)]:
    board, sans = play(model_white, seed)
    print(f"=== MODEL as {'WHITE' if model_white else 'BLACK'} ===", flush=True)
    print(fmt(sans), flush=True)
    print(f"plies={len(sans)}  result={board.result(claim_draw=True)}  -> {outcome(board, model_white)}\n", flush=True)
