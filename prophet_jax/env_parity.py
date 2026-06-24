"""Moonshot env-bridge parity check for ``prophet_jax.env``.

Run with:

    python -m prophet_jax.env_parity --plies 160 --check-all-actions-every 16

The reference is the working moonshot Python path:

* ``prophet.encoding.encode_board`` for the [64, 24] tensor, including the last
  two move-history planes.
* ``prophet.encoding.legal_move_map`` for the 4096 ``from*64 + to`` action set.
* ``python-chess`` stepping for legal move semantics.

This intentionally does not import ``prophet.search`` or torch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import chess
import jax
import jax.numpy as jnp
import numpy as np

from prophet.encoding import encode_board, legal_move_map

from . import env as env_mod


DRAW_HALFMOVE_CAP = 100


@dataclass
class ParityStats:
    plies_checked: int = 0
    action_children_checked: int = 0


def _terminal_value(board: chess.Board) -> float | None:
    """Moonshot terminal value for side to move, without importing torch."""
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.halfmove_clock >= DRAW_HALFMOVE_CAP
        or board.is_repetition(3)
    ):
        return 0.0
    return None


def _reference_legal(board: chess.Board) -> set[int]:
    _, flipped = encode_board(board)
    return set(int(a) for a in legal_move_map(board, flipped).keys())


def _jax_legal(state) -> set[int]:
    mask = np.asarray(env_mod.legal_mask(state))[0]
    return set(int(a) for a in np.flatnonzero(mask))


def _fail(message: str) -> None:
    raise AssertionError(message)


def _check_position(state, history, board: chess.Board, *, label: str) -> None:
    x_jax = np.asarray(env_mod.encode_state(state, history))[0]
    x_ref, _ = encode_board(board)

    if not np.allclose(x_jax, x_ref, rtol=1e-6, atol=1e-7):
        nz = np.argwhere(~np.isclose(x_jax, x_ref, rtol=1e-6, atol=1e-7))
        details = []
        for r, c in nz[:20]:
            details.append(
                f"({int(r)},{int(c)}): jax={x_jax[r, c]!r} ref={x_ref[r, c]!r}"
            )
        _fail(
            f"{label}: encode_state mismatch at {len(nz)} cells; "
            f"fen={board.fen()} history={np.asarray(history)[0].tolist()} "
            f"first={details}"
        )

    legal_jax = _jax_legal(state)
    legal_ref = _reference_legal(board)
    if legal_jax != legal_ref:
        _fail(
            f"{label}: legal action mismatch; fen={board.fen()} "
            f"jax_only={sorted(legal_jax - legal_ref)[:20]} "
            f"ref_only={sorted(legal_ref - legal_jax)[:20]}"
        )

    is_term, term_value = env_mod.terminal_info(state)
    is_term = bool(np.asarray(is_term)[0])
    term_value = float(np.asarray(term_value)[0])
    ref_term = _terminal_value(board)
    if is_term != (ref_term is not None):
        _fail(
            f"{label}: terminal flag mismatch; fen={board.fen()} "
            f"jax={is_term} ref={ref_term}"
        )
    if ref_term is not None and term_value != float(ref_term):
        _fail(
            f"{label}: terminal value mismatch; fen={board.fen()} "
            f"jax={term_value} ref={ref_term}"
        )


def _step_jax(state, history, action: int):
    action_arr = jnp.asarray([action], dtype=jnp.int32)
    next_state = env_mod.env_step(state, action_arr)
    next_history = env_mod.update_history(history, action_arr)
    return next_state, next_history


def _check_all_children(state, history, board: chess.Board, *, label: str) -> int:
    """Verify every legal action from one position against python-chess."""
    legal = sorted(_reference_legal(board))
    checked = 0
    for action in legal:
        child_board = board.copy(stack=True)
        _, flipped = encode_board(child_board)
        move = legal_move_map(child_board, flipped)[action]
        child_board.push(move)

        child_state, child_history = _step_jax(state, history, action)
        _check_position(child_state, child_history, child_board, label=f"{label}/a={action}")
        checked += 1
    return checked


def run_parity(
    *,
    plies: int,
    seed: int,
    check_all_actions_every: int,
    verbose: bool,
) -> ParityStats:
    rng = np.random.default_rng(seed)
    state = env_mod.env_init(env_mod.start_keys(jax.random.PRNGKey(seed), 1))
    history = env_mod.empty_history(state)
    board = chess.Board()
    stats = ParityStats()

    for ply in range(plies + 1):
        label = f"ply={ply}"
        _check_position(state, history, board, label=label)
        stats.plies_checked += 1

        if check_all_actions_every > 0 and ply % check_all_actions_every == 0:
            checked = _check_all_children(state, history, board, label=label)
            stats.action_children_checked += checked
            if verbose:
                print(f"{label}: checked {checked} legal children")

        if _terminal_value(board) is not None or ply == plies:
            break

        legal = sorted(_reference_legal(board))
        if not legal:
            break
        action = int(legal[rng.integers(len(legal))])
        _, flipped = encode_board(board)
        board.push(legal_move_map(board, flipped)[action])
        state, history = _step_jax(state, history, action)

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--check-all-actions-every",
        type=int,
        default=16,
        help="also verify every legal child every N plies; 0 disables",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    stats = run_parity(
        plies=args.plies,
        seed=args.seed,
        check_all_actions_every=args.check_all_actions_every,
        verbose=args.verbose,
    )
    print(
        "env parity ok: "
        f"plies={stats.plies_checked} "
        f"checked_children={stats.action_children_checked} "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
