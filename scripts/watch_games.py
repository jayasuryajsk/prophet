"""Generate annotated sample games from a checkpoint: PGN movetext plus a
material trace, flagging sacrifice-like events (model gives up >=2 points
of material for 3+ plies) and other notable moments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch before numpy — see README
import torch  # noqa: I001

import argparse

import chess
import chess.engine
import numpy as np

from prophet.model import load_checkpoint
from prophet.search import SearchConfig, _terminal_value, search_move

VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board, color):
    return sum(v * len(board.pieces(p, color)) for p, v in VALS.items())


def play(model, scfg, rng, opponent=None, model_is_white=True, max_plies=300):
    board = chess.Board()
    sans, balance = [], []  # balance from model's perspective
    while _terminal_value(board) is None and len(sans) < max_plies:
        model_turn = board.turn == (chess.WHITE if model_is_white else chess.BLACK)
        if model_turn or opponent is None:
            mv = search_move(model, board, scfg, torch.device("cpu"), rng)
        else:
            mv = opponent.play(board, chess.engine.Limit(time=0.05)).move
        sans.append(board.san(mv))
        board.push(mv)
        us = chess.WHITE if model_is_white else chess.BLACK
        balance.append(material(board, us) - material(board, not us))
    return board, sans, balance


def sac_events(balance):
    """Plies where the trace drops >=2 from its previous local max and stays
    down >=3 plies (a sacrifice or a blunder; the game result disambiguates)."""
    events, peak = [], 0
    for i, b in enumerate(balance):
        peak = max(peak, b)
        if peak - b >= 2 and all(peak - x >= 2 for x in balance[i : i + 3]):
            events.append((i, peak - b))
            peak = b  # reset so one sac isn't reported repeatedly
    return events


def pgn(sans):
    out = []
    for i, s in enumerate(sans):
        if i % 2 == 0:
            out.append(f"{i//2+1}.")
        out.append(s)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--forwards", type=int, default=256)
    ap.add_argument("--self-games", type=int, default=3)
    ap.add_argument("--sf-games", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(4)
    model = load_checkpoint(args.ckpt)
    scfg = SearchConfig(sims=args.forwards - 1, root_candidates=16)
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})

    for g in range(args.self_games):
        rng = np.random.default_rng(100 + g)
        board, sans, bal = play(model, scfg, rng)
        res = board.result(claim_draw=True)
        print(f"\n=== SELF-PLAY game {g+1}: {res} ({len(sans)} plies) ===")
        print(pgn(sans))
        evs = sac_events(bal)
        if evs:
            print(f"  material dips (ply, depth): {evs}  final balance {bal[-1]:+d}")

    for g in range(args.sf_games):
        rng = np.random.default_rng(200 + g)
        white = g % 2 == 0
        board, sans, bal = play(model, scfg, rng, opponent=engine, model_is_white=white)
        res = board.result(claim_draw=True)
        print(f"\n=== vs SF-1320 game {g+1} (model={'White' if white else 'Black'}): {res} ({len(sans)} plies) ===")
        print(pgn(sans))
        evs = sac_events(bal)
        if evs:
            print(f"  material dips (ply, depth): {evs}  final balance {bal[-1]:+d}")
    engine.quit()


if __name__ == "__main__":
    main()
