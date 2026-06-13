"""End-to-end smoke test: encoding -> model -> search -> self-play -> training."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch must be imported before numpy: Homebrew numpy and torch each bundle
# libomp, and only torch-first initialization is stable on this setup.
import torch  # noqa: I001

import chess
import numpy as np

from prophet.encoding import (
    FEATURES,
    NUM_ACTIONS,
    encode_board,
    index_to_move,
    legal_move_map,
    move_to_index,
)
from prophet.model import ModelConfig, PolicyQValueNet
from prophet.search import SearchConfig, run_search
from prophet.selfplay import SelfPlayConfig, play_game
from prophet.train import collate, train_step

PASS = "  \033[32mPASS\033[0m"


def check(cond, msg):
    if not cond:
        print(f"  \033[31mFAIL\033[0m {msg}")
        sys.exit(1)


def test_encoding(rng):
    print("[1/6] encoding round-trips")
    board = chess.Board()
    for ply in range(120):
        x, flipped = encode_board(board)
        check(x.shape == (64, FEATURES), f"feature shape {x.shape}")
        check(flipped == (board.turn == chess.BLACK), "flip flag")
        legal = legal_move_map(board, flipped)
        check(len(legal) > 0, f"no mapped legal moves at {board.fen()}")
        for idx, mv in legal.items():
            back = index_to_move(idx, board, flipped)
            check(back == mv, f"round-trip {mv} -> {idx} -> {back}")
            check(move_to_index(mv, flipped) == idx, "index round-trip")
            check(board.is_legal(back), f"decoded move illegal: {back}")
        mv = list(legal.values())[rng.integers(len(legal))]
        board.push(mv)
        if board.is_game_over(claim_draw=True):
            break
    # spot-check perspective flip: black-to-move start mirrors white-to-move start
    b = chess.Board()
    x_w, _ = encode_board(b)
    b.push(chess.Move.null())
    x_b, flipped = encode_board(b)
    check(flipped and np.allclose(x_w[:, :12], x_b[:, :12]), "mirror symmetry at start")
    print(PASS)


def test_model(device):
    print("[2/6] model forward + gradients")
    model = PolicyQValueNet(ModelConfig()).to(device)
    print(f"  params: {model.num_params():,}")
    x = torch.randn(3, 64, FEATURES, device=device)
    logits, q, v = model(x)
    check(logits.shape == (3, NUM_ACTIONS), f"policy shape {logits.shape}")
    check(q.shape == (3, NUM_ACTIONS), f"q shape {q.shape}")
    check(v.shape == (3,), f"v shape {v.shape}")
    check(bool((q.abs() <= 1).all() and (v.abs() <= 1).all()), "q/v range")
    (logits.sum() + q.sum() + v.sum()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check(len(grads) > 0 and all(g.isfinite().all() for g in grads), "finite grads")
    print(PASS)
    return model


def test_search(model, device, rng):
    print("[3/6] search returns legal moves with sane targets")
    cfg = SearchConfig(sims=16, root_candidates=4)
    for fen in [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "8/5k2/8/8/8/3K4/6R1/8 w - - 0 1",
        "7k/6pp/8/8/8/8/5PPP/6K1 b - - 0 1",
    ]:
        board = chess.Board(fen)
        t0 = time.perf_counter()
        res = run_search(model, board, cfg, device, rng)
        dt = time.perf_counter() - t0
        check(board.is_legal(res.move), f"illegal move {res.move} from {fen}")
        check(abs(res.policy_target.sum() - 1.0) < 1e-4, "policy target sums to 1")
        check(-1.0 <= res.root_value <= 1.0, "root value range")
        check(len(res.q_indices) > 0, "search produced q targets")
        check(bool((res.q_visits > 0).all()), "q targets have visits")
        print(f"  {fen.split()[0][:20]:22s} -> {res.move.uci()}  ({dt*1000:.0f} ms)")
    print(PASS)


def test_selfplay(model, device, rng):
    print("[4/6] self-play game")
    t0 = time.perf_counter()
    game = play_game(
        model,
        SearchConfig(sims=12, root_candidates=4),
        SelfPlayConfig(max_plies=30),
        device,
        rng,
    )
    dt = time.perf_counter() - t0
    check(len(game.samples) == game.plies > 0, "samples match plies")
    for s in game.samples:
        check(abs(s.policy_target.sum() - 1.0) < 1e-4, "sample policy sums to 1")
        check(-1.0 <= s.value_target <= 1.0, "sample value range")
        check(s.played_index in s.legal_indices, "played move is in legal set")
    print(
        f"  {game.plies} plies, result {game.result}, "
        f"{dt:.1f}s ({dt/game.plies*1000:.0f} ms/move)"
    )
    print(PASS)
    return game


def test_training(model, game, device):
    print("[5/6] training step + overfit a fixed batch")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = collate(game.samples[:16], device)
    first = train_step(model, opt, batch)
    check(np.isfinite(first["loss"]), f"non-finite loss {first}")
    losses = [first["loss"]]
    for _ in range(80):
        losses.append(train_step(model, opt, batch)["loss"])
    print(
        f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
        f"(pi {first['policy']:.3f}, v {first['value']:.3f}, "
        f"q {first['q']:.3f}, cons {first['consistency']:.3f})"
    )
    # policy CE is floor-limited by the targets' own entropy (draw-heavy
    # smoke games produce flat targets), so test a meaningful decrease
    # rather than a deep one
    check(losses[-1] < losses[0] * 0.85, "loss did not decrease on fixed batch")
    print(PASS)


def test_mps(game):
    print("[6/6] MPS forward/backward")
    if not torch.backends.mps.is_available():
        print("  skipped (no MPS)")
        return
    device = torch.device("mps")
    model = PolicyQValueNet(ModelConfig()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = collate(game.samples[:8], device)
    out = train_step(model, opt, batch)
    check(np.isfinite(out["loss"]), f"non-finite loss on MPS: {out}")
    print(f"  loss {out['loss']:.4f} on mps")
    print(PASS)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cpu")
    t0 = time.perf_counter()
    test_encoding(rng)
    model = test_model(device)
    test_search(model, device, rng)
    game = test_selfplay(model, device, rng)
    test_training(model, game, device)
    test_mps(game)
    print(f"\nall smoke tests passed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
