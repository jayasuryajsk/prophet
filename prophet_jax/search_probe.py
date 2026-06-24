"""Fast search/value probes for the JAX chess engine.

Run the default mate-in-1 probe:

    python -m prophet_jax.search_probe --sims 128 --candidates 64

The default position is Fool's Mate after ``1. f3 e5 2. g4``. Black has
``Qh4#``. With full legal candidate coverage, search should visit the mating
move, assign it root-perspective Q close to +1, and select it.
"""

from __future__ import annotations

import argparse

import chess
import jax
import jax.numpy as jnp
import numpy as np

from prophet.encoding import encode_board, legal_move_map

from . import env as env_mod
from .config import ModelConfig, SearchConfig
from .model import build_model
from .search import batched_search


DEFAULT_MOVES = ("f2f3", "e7e5", "g2g4")


def _action_for_move(board: chess.Board, move: chess.Move) -> int:
    """Return prophet action index for ``move`` in ``board``."""
    _, flipped = encode_board(board)
    for action, legal in legal_move_map(board, flipped).items():
        if legal == move:
            return int(action)
    raise ValueError(f"move {move.uci()} is not legal in {board.fen()}")


def _move_for_action(board: chess.Board, action: int) -> chess.Move | None:
    """Return python-chess move for a prophet action in ``board``."""
    _, flipped = encode_board(board)
    return legal_move_map(board, flipped).get(int(action))


def _build_state_from_uci(moves: tuple[str, ...], seed: int):
    """Replay UCI moves through python-chess and the JAX pgx bridge."""
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


def _mate_actions(board: chess.Board) -> dict[int, chess.Move]:
    """All legal actions that checkmate immediately."""
    out: dict[int, chess.Move] = {}
    _, flipped = encode_board(board)
    for action, move in legal_move_map(board, flipped).items():
        child = board.copy(stack=True)
        child.push(move)
        if child.is_checkmate():
            out[int(action)] = move
    return out


def run_mate_probe(args: argparse.Namespace) -> int:
    moves = tuple(args.moves.split()) if isinstance(args.moves, str) else DEFAULT_MOVES
    board, state, history = _build_state_from_uci(moves, args.seed)
    mates = _mate_actions(board)
    if not mates:
        raise AssertionError(f"no mate-in-1 move in position: {board.fen()}")

    cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        head_dim=args.head_dim,
    )
    _model, params = build_model(cfg, jax.random.PRNGKey(args.seed + 17))
    legal_count = len(legal_move_map(board, encode_board(board)[1]))
    scfg = SearchConfig(
        sims=args.sims,
        root_candidates=min(args.candidates, legal_count),
        q_trust=args.q_trust,
    )
    key = jax.random.PRNGKey(args.seed + 23)
    out = batched_search(params, key, state, scfg, history)

    move_index = int(np.asarray(out.move_index)[0])
    selected = _move_for_action(board, move_index)
    root_value = float(np.asarray(out.root_value)[0])
    q_target = np.asarray(out.q_target)[0]
    q_weight = np.asarray(out.q_weight)[0]
    policy = np.asarray(out.policy_target)[0]

    mate_rows = []
    for action, move in mates.items():
        mate_rows.append(
            (
                action,
                move.uci(),
                float(q_target[action]),
                float(q_weight[action]),
                float(policy[action]),
            )
        )
    best_mate_q = max(row[2] for row in mate_rows)
    best_mate_visits = max(row[3] for row in mate_rows)
    mate_policy = sum(row[4] for row in mate_rows)
    selected_is_mate = move_index in mates

    print(f"fen: {board.fen()}")
    print(f"legal moves: {legal_count}")
    print(f"mate actions: {mate_rows}")
    print(
        f"selected: action={move_index} move={selected.uci() if selected else None} "
        f"mate={selected_is_mate}"
    )
    print(
        f"root_value={root_value:.4f} best_mate_q={best_mate_q:.4f} "
        f"best_mate_visits={best_mate_visits:.0f} mate_policy={mate_policy:.4f}"
    )

    ok = (
        selected_is_mate
        and best_mate_q >= args.min_mate_q
        and best_mate_visits >= 1.0
        and mate_policy >= args.min_mate_policy
    )
    if not ok:
        print("MATE PROBE FAILED")
        return 1
    print("MATE PROBE PASSED")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moves",
        default=" ".join(DEFAULT_MOVES),
        help="space-separated UCI moves from the initial position",
    )
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--q-trust", type=float, default=1.0)
    parser.add_argument("--min-mate-q", type=float, default=0.90)
    parser.add_argument("--min-mate-policy", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    raise SystemExit(run_mate_probe(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
