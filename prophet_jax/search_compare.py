"""Compare moonshot PyTorch search targets against JAX search targets.

This diagnostic exports one freshly initialized JAX model to the PyTorch
checkpoint format, then runs both engines on the same python-chess positions.
It is intentionally small and slow; its purpose is correctness, not throughput.

Example:

    python -m prophet_jax.search_compare --sims 32 --candidates 8
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np

from prophet.encoding import encode_board, legal_move_map
from prophet.fastboard import PyChessBoard
from prophet.model import load_checkpoint as load_torch_checkpoint
from prophet.search import SearchConfig as TorchSearchConfig
from prophet.search import run_search as run_torch_search

from . import env as env_mod
from .config import ModelConfig, SearchConfig
from .model import build_model, export_torch_checkpoint
from .search import batched_search


DEFAULT_POSITIONS = {
    "start": (),
    "fools_mate_minus_one": ("f2f3", "e7e5", "g2g4"),
    "italian": ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"),
    "queen_trade": ("d2d4", "d7d5", "c2c4", "d5c4", "d1a4"),
}


def _action_for_move(board: chess.Board, move: chess.Move) -> int:
    _, flipped = encode_board(board)
    for action, legal in legal_move_map(board, flipped).items():
        if legal == move:
            return int(action)
    raise ValueError(f"move {move.uci()} is not legal in {board.fen()}")


def _move_for_action(board: chess.Board, action: int) -> chess.Move | None:
    _, flipped = encode_board(board)
    return legal_move_map(board, flipped).get(int(action))


def _build_state(moves: tuple[str, ...], seed: int):
    state = env_mod.env_init(env_mod.start_keys(jax.random.PRNGKey(seed), 1))
    history = env_mod.empty_history(state)
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        action = _action_for_move(board, move)
        action_arr = jnp.asarray([action], dtype=jnp.int32)
        state = env_mod.env_step(state, action_arr)
        history = env_mod.update_history(history, action_arr)
        board.push(move)
    return board, state, history


def _torch_stats(res) -> dict[str, float | int | str | None]:
    if len(res.q_visits):
        total = float(np.asarray(res.q_visits).sum())
        qabs = float((np.abs(res.q_values) * res.q_visits).sum() / max(total, 1.0))
        maxq = float(np.max(res.q_values))
        minq = float(np.min(res.q_values))
    else:
        total = 0.0
        qabs = maxq = minq = 0.0
    return {
        "root": float(res.root_value),
        "qabs": qabs,
        "visits": total,
        "maxq": maxq,
        "minq": minq,
        "move": int(res.move_index),
    }


def _jax_stats(out) -> dict[str, float | int]:
    q_target = np.asarray(out.q_target)[0]
    q_weight = np.asarray(out.q_weight)[0]
    total = float(q_weight.sum())
    qabs = float((np.abs(q_target) * q_weight).sum() / max(total, 1.0))
    visited = q_weight > 0
    if np.any(visited):
        maxq = float(q_target[visited].max())
        minq = float(q_target[visited].min())
    else:
        maxq = minq = 0.0
    return {
        "root": float(np.asarray(out.root_value)[0]),
        "qabs": qabs,
        "visits": total,
        "maxq": maxq,
        "minq": minq,
        "move": int(np.asarray(out.move_index)[0]),
    }


def _fmt_stats(board: chess.Board, stats: dict[str, float | int]) -> str:
    move = _move_for_action(board, int(stats["move"]))
    return (
        f"move={move.uci() if move else stats['move']} "
        f"root={float(stats['root']):+.4f} "
        f"qabs={float(stats['qabs']):.4f} "
        f"visits={float(stats['visits']):.0f} "
        f"maxq={float(stats['maxq']):+.4f} "
        f"minq={float(stats['minq']):+.4f}"
    )


def run_compare(args: argparse.Namespace) -> int:
    cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        head_dim=args.head_dim,
    )
    _model, params = build_model(cfg, jax.random.PRNGKey(args.seed))
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "fresh.pt"
        export_torch_checkpoint(params, cfg, str(ckpt))
        torch_model = load_torch_checkpoint(ckpt)
        torch_model.eval()

        jax_cfg = SearchConfig(
            sims=args.sims,
            root_candidates=args.candidates,
            q_trust=args.q_trust,
        )
        torch_cfg = TorchSearchConfig(
            sims=args.sims,
            root_candidates=args.candidates,
            q_trust=args.q_trust,
        )
        rng = np.random.default_rng(args.seed + 101)

        for name, moves in DEFAULT_POSITIONS.items():
            board, state, history = _build_state(moves, args.seed + 7)
            legal_count = len(legal_move_map(board, encode_board(board)[1]))
            jc = SearchConfig(
                sims=jax_cfg.sims,
                root_candidates=min(jax_cfg.root_candidates, legal_count),
                q_trust=jax_cfg.q_trust,
            )
            tc = TorchSearchConfig(
                sims=torch_cfg.sims,
                root_candidates=min(torch_cfg.root_candidates, legal_count),
                q_trust=torch_cfg.q_trust,
            )
            jax_out = batched_search(
                params, jax.random.PRNGKey(args.seed + 200 + len(name)), state, jc, history
            )
            torch_res = run_torch_search(
                torch_model, PyChessBoard(board.copy(stack=True)), tc, "cpu", rng
            )
            js = _jax_stats(jax_out)
            ts = _torch_stats(torch_res)
            print(f"== {name} ==")
            print(f"fen: {board.fen()}")
            print(f"jax   {_fmt_stats(board, js)}")
            print(f"torch {_fmt_stats(board, ts)}")
            print(f"delta qabs={float(js['qabs']) - float(ts['qabs']):+.4f}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--q-trust", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    raise SystemExit(run_compare(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
