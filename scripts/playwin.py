"""Play a batch vs Stockfish, report each result, and show a WON game in full."""
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
ELO = int(sys.argv[3]) if len(sys.argv) > 3 else 1320
NGAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 18
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
    return " ".join(f"{i//2+1}.{s}" if i % 2 == 0 else s for i, s in enumerate(sans))


def outcome(board, mw):
    if board.is_checkmate():
        return "WIN" if (board.turn == chess.BLACK) == mw else "LOSS"
    return "draw"


print(f"ckpt={ckpt} sims={SIMS} vs SF{ELO} — up to {NGAMES} games\n", flush=True)
wins = []
tally = {"WIN": 0, "draw": 0, "LOSS": 0}
for g in range(NGAMES):
    mw = g % 2 == 0
    board, sans = play(mw, 100 + g)
    r = outcome(board, mw)
    tally[r] += 1
    print(f"  game {g+1:>2}: model {'W' if mw else 'B'} -> {r:>4} ({len(sans)} plies)", flush=True)
    if r == "WIN":
        wins.append((mw, sans))
        if len(wins) >= 2:
            break

print(f"\ntally so far: {tally['WIN']}W {tally['draw']}D {tally['LOSS']}L", flush=True)
if wins:
    for i, (mw, sans) in enumerate(wins):
        print(f"\n=== WON GAME #{i+1} (model as {'WHITE' if mw else 'BLACK'}) ===", flush=True)
        print(fmt(sans), flush=True)
else:
    print("\n(no win in this batch — it's below SF1320 at these sims)", flush=True)
