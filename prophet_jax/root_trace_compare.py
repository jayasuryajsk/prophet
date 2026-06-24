"""Compare moonshot root-search traces against JAX/mctx with fixed Gumbels.

The normal search comparison intentionally lets each engine use its own RNG.
This diagnostic removes that source of noise: for each position it draws one
host-side Gumbel vector, feeds the legal entries to moonshot/PyTorch, feeds the
same full 4096-vector to JAX/mctx, and then prints root visit distributions.

It is meant to answer one narrow question: when loss/env/terminal semantics are
already aligned, how much remaining mismatch comes from mctx's root scheduler?

Example:

    python -m prophet_jax.root_trace_compare --sims 32 --candidates 8
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

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
from .config import NUM_ACTIONS, ModelConfig, SearchConfig
from .model import build_model, export_torch_checkpoint
from .search import (
    _gumbel_root_puct_policy,
    _invalid_actions,
    recurrent_fn,
    root_fn,
    search_result,
)


DEFAULT_POSITIONS = {
    "start": (),
    "fools_mate_minus_one": ("f2f3", "e7e5", "g2g4"),
    "italian": ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"),
    "queen_trade": ("d2d4", "d7d5", "c2c4", "d5c4", "d1a4"),
}


class _FixedRootGumbel:
    """Tiny RNG stub for ``prophet.search.run_search`` root Gumbel draw."""

    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.float32)
        self.calls = 0

    def gumbel(self, size=None):
        self.calls += 1
        expected = None
        if isinstance(size, int):
            expected = (int(size),)
        elif size is not None:
            expected = tuple(size)
        if expected is not None and expected != tuple(self.values.shape):
            raise ValueError(f"expected gumbel size {self.values.shape}, got {size}")
        return self.values.copy()


def _action_for_move(board: chess.Board, move: chess.Move) -> int:
    _, flipped = encode_board(board)
    for action, legal in legal_move_map(board, flipped).items():
        if legal == move:
            return int(action)
    raise ValueError(f"move {move.uci()} is not legal in {board.fen()}")


def _move_for_action(board: chess.Board, action: int) -> str:
    _, flipped = encode_board(board)
    move = legal_move_map(board, flipped).get(int(action))
    return move.uci() if move is not None else str(int(action))


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


def _run_jax_fixed_gumbel(
    params: Any,
    key: Any,
    state: Any,
    history: jnp.ndarray,
    cfg: SearchConfig,
    gumbel_full: np.ndarray,
):
    """Run current JAX/mctx search with a caller-supplied root Gumbel vector."""
    root = root_fn(params, state, history)
    invalid = _invalid_actions(state)
    gumbel = jnp.asarray(gumbel_full, dtype=root.prior_logits.dtype)[None, :]
    policy_output = _gumbel_root_puct_policy(
        params=params,
        rng_key=key,
        root=root,
        recurrent_fn_=recurrent_fn,
        num_simulations=cfg.sims,
        invalid_actions=invalid,
        max_num_considered_actions=cfg.root_candidates,
        q_trust=cfg.q_trust,
        c_visit=cfg.c_visit,
        c_scale=cfg.c_scale,
        c_puct=cfg.c_puct,
        root_gumbel=gumbel,
    )
    return search_result(policy_output, state, params, history), root, policy_output.search_tree


def _run_jax_stock(
    params: Any,
    key: Any,
    state: Any,
    history: jnp.ndarray,
    cfg: SearchConfig,
):
    """Run current production JAX search, useful as a control."""
    root = root_fn(params, state, history)
    invalid = _invalid_actions(state)
    return _gumbel_root_puct_policy(
        params=params,
        rng_key=key,
        root=root,
        recurrent_fn_=recurrent_fn,
        num_simulations=cfg.sims,
        invalid_actions=invalid,
        max_num_considered_actions=cfg.root_candidates,
        q_trust=cfg.q_trust,
        c_visit=cfg.c_visit,
        c_scale=cfg.c_scale,
        c_puct=cfg.c_puct,
    )


def _torch_trace(res) -> dict[str, np.ndarray | float | int]:
    q_values = np.asarray(res.q_values, dtype=np.float32)
    q_visits = np.asarray(res.q_visits, dtype=np.float32)
    q_indices = np.asarray(res.q_indices, dtype=np.int32)
    qabs = float((np.abs(q_values) * q_visits).sum() / max(float(q_visits.sum()), 1.0))
    return {
        "move": int(res.move_index),
        "root": float(res.root_value),
        "indices": q_indices,
        "visits": q_visits,
        "q": q_values,
        "qabs": qabs,
    }


def _jax_trace(out) -> dict[str, np.ndarray | float | int]:
    q_target = np.asarray(out.q_target)[0]
    q_weight = np.asarray(out.q_weight)[0]
    indices = np.flatnonzero(q_weight > 0).astype(np.int32)
    visits = q_weight[indices].astype(np.float32)
    q_values = q_target[indices].astype(np.float32)
    qabs = float((np.abs(q_values) * visits).sum() / max(float(visits.sum()), 1.0))
    return {
        "move": int(np.asarray(out.move_index)[0]),
        "root": float(np.asarray(out.root_value)[0]),
        "indices": indices,
        "visits": visits,
        "q": q_values,
        "qabs": qabs,
    }


def _format_visit_list(board: chess.Board, trace: dict[str, Any], limit: int) -> str:
    rows = []
    data = sorted(
        zip(trace["indices"], trace["visits"], trace["q"]),
        key=lambda x: (-float(x[1]), int(x[0])),
    )
    for action, visits, q in data[:limit]:
        rows.append(f"{_move_for_action(board, int(action))}:{int(visits)}:{float(q):+.3f}")
    return " ".join(rows)


def _top_initial(
    board: chess.Board,
    legal: np.ndarray,
    logits: np.ndarray,
    q: np.ndarray,
    gumbel: np.ndarray,
    cfg: SearchConfig,
    limit: int,
):
    base = logits[legal] + gumbel[legal]
    moon_order = np.argsort(-base)[:limit]
    mctx_order = np.argsort(-(base + cfg.c_visit * cfg.c_scale * q[legal]))[:limit]
    moon = [f"{_move_for_action(board, legal[i])}:{base[i]:+.2f}" for i in moon_order]
    mctx = [
        f"{_move_for_action(board, legal[i])}:{(base + cfg.c_visit * cfg.c_scale * q[legal])[i]:+.2f}"
        for i in mctx_order
    ]
    return moon, mctx


def _overlap(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    sa = set(int(x) for x in a.tolist())
    sb = set(int(x) for x in b.tolist())
    return len(sa & sb), len(sa | sb)


def run_compare(args: argparse.Namespace) -> int:
    cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        head_dim=args.head_dim,
    )
    _model, params = build_model(cfg, jax.random.PRNGKey(args.seed))
    scfg = SearchConfig(
        sims=args.sims,
        root_candidates=args.candidates,
        q_trust=args.q_trust,
    )
    rng = np.random.default_rng(args.seed + 101)

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "fresh.pt"
        export_torch_checkpoint(params, cfg, str(ckpt))
        torch_model = load_torch_checkpoint(ckpt).eval()

        for name, moves in DEFAULT_POSITIONS.items():
            if args.position != "all" and name != args.position:
                continue
            board, state, history = _build_state(moves, args.seed + 7)
            pb = PyChessBoard(board.copy(stack=True))
            legal = np.asarray(pb.legal_actions(), dtype=np.int32)
            g_legal = rng.gumbel(size=len(legal)).astype(np.float32)
            gumbel = np.zeros((NUM_ACTIONS,), dtype=np.float32)
            gumbel[legal] = g_legal

            torch_res = run_torch_search(
                torch_model,
                PyChessBoard(board.copy(stack=True)),
                TorchSearchConfig(
                    sims=scfg.sims,
                    root_candidates=min(scfg.root_candidates, len(legal)),
                    q_trust=scfg.q_trust,
                ),
                "cpu",
                _FixedRootGumbel(g_legal),
            )
            jax_out, root, _tree = _run_jax_fixed_gumbel(
                params,
                jax.random.PRNGKey(args.seed + 300 + len(name)),
                state,
                history,
                SearchConfig(
                    sims=scfg.sims,
                    root_candidates=min(scfg.root_candidates, len(legal)),
                    q_trust=scfg.q_trust,
                ),
                gumbel,
            )
            torch_trace = _torch_trace(torch_res)
            jax_trace = _jax_trace(jax_out)

            root_logits = np.asarray(root.prior_logits)[0]
            root_q = np.asarray(root.embedding.q_init)[0]
            moon_top, mctx_top = _top_initial(
                board, legal, root_logits, root_q, gumbel, scfg, args.list
            )
            ov, union = _overlap(torch_trace["indices"], jax_trace["indices"])

            print(f"== {name} ==")
            print(f"fen: {board.fen()}")
            print(
                "moonshot "
                f"move={_move_for_action(board, torch_trace['move'])} "
                f"root={torch_trace['root']:+.4f} qabs={torch_trace['qabs']:.4f} "
                f"visited={len(torch_trace['indices'])}"
            )
            print(
                "jax-exact "
                f"move={_move_for_action(board, jax_trace['move'])} "
                f"root={jax_trace['root']:+.4f} qabs={jax_trace['qabs']:.4f} "
                f"visited={len(jax_trace['indices'])}"
            )
            print(f"visited overlap: {ov}/{union}")
            print(f"initial top moonshot logits+g: {' '.join(moon_top)}")
            print(f"initial top q-biased old-mctx: {' '.join(mctx_top)}")
            print(f"moonshot visits: {_format_visit_list(board, torch_trace, args.list)}")
            print(f"jax visits:      {_format_visit_list(board, jax_trace, args.list)}")

            if args.stock_control:
                stock = _run_jax_stock(
                    params,
                    jax.random.PRNGKey(args.seed + 400 + len(name)),
                    state,
                    history,
                    scfg,
                )
                stock_trace = _jax_trace(search_result(stock, state, params, history))
                print(
                    "jax-stock "
                    f"move={_move_for_action(board, stock_trace['move'])} "
                    f"root={stock_trace['root']:+.4f} "
                    f"qabs={stock_trace['qabs']:.4f}"
                )
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
    parser.add_argument("--position", choices=["all", *DEFAULT_POSITIONS.keys()], default="all")
    parser.add_argument("--list", type=int, default=8)
    parser.add_argument("--stock-control", action="store_true")
    raise SystemExit(run_compare(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
