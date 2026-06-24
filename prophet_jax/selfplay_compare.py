"""Compare JAX and PyTorch self-play batches from identical fresh weights.

This is a diagnostic for the JAX port. It exports a freshly initialized JAX
model into the PyTorch checkpoint format, then runs:

* ``prophet_jax.selfplay.generate_selfplay`` for one batched JAX rollout.
* ``prophet.worker.run_vector_selfplay`` for the same number of PyTorch games.

It reports outcome, plies, root Q-target sharpness, and played Q-head magnitude.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import chess
import jax
import numpy as np
import torch

from prophet.encoding import encode_board, legal_move_map
from prophet.model import load_checkpoint as load_torch_checkpoint
from prophet.search import SearchConfig as TorchSearchConfig
from prophet.selfplay import SelfPlayConfig as TorchSelfPlayConfig
from prophet.worker import run_vector_selfplay

from .config import ModelConfig, SearchConfig, SelfPlayConfig
from .model import build_model, export_torch_checkpoint
from .selfplay import generate_selfplay
from .train import q_head_played_abs, q_target_stats


def _terminal_result(board: chess.Board) -> str:
    if board.is_checkmate():
        return "0-1" if board.turn == chess.WHITE else "1-0"
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.halfmove_clock >= 100
        or board.is_repetition(3)
    ):
        return "1/2-1/2"
    return "*"


def _jax_python_replay_counts(samples, meta) -> dict[str, float]:
    """Replay JAX action indices through python-chess to classify endings."""
    valid = np.asarray(meta.valid_ply).astype(bool)
    B, T = valid.shape
    played = np.asarray(samples.played).reshape(B, T)
    counts = {"black_win": 0.0, "draw": 0.0, "white_win": 0.0, "unfinished": 0.0}
    illegal = 0.0
    for g in range(B):
        board = chess.Board()
        for t in range(T):
            if not valid[g, t]:
                break
            action = int(played[g, t])
            _, flipped = encode_board(board)
            move = legal_move_map(board, flipped).get(action)
            if move is None or not board.is_legal(move):
                illegal += 1.0
                break
            board.push(move)
            if _terminal_result(board) != "*":
                break
        result = _terminal_result(board)
        if result == "0-1":
            counts["black_win"] += 1.0
        elif result == "1/2-1/2":
            counts["draw"] += 1.0
        elif result == "1-0":
            counts["white_win"] += 1.0
        else:
            counts["unfinished"] += 1.0
    counts["illegal"] = illegal
    return counts


def _jax_summary(samples, meta) -> dict[str, float]:
    qabs, qvis = q_target_stats(samples)
    z = np.asarray(meta.z_white)
    finished = ~np.isnan(z)
    decisive = (np.abs(np.nan_to_num(z)) > 0.5) & finished
    denom = max(1, int(finished.sum()))
    result = np.asarray(meta.result)
    replay = _jax_python_replay_counts(samples, meta)
    return {
        "games": float(len(z)),
        "plies": float(np.asarray(meta.plies).mean()),
        "finished": float(finished.mean()),
        "decisive": float(decisive.sum() / denom),
        "qabs": qabs,
        "qvis": qvis,
        "qhp": q_head_played_abs(meta),
        "rows": float(np.asarray(samples.valid).sum()),
        "black_win": float((result == 0).sum()),
        "draw": float((result == 1).sum()),
        "white_win": float((result == 2).sum()),
        "unfinished": float((result == -1).sum()),
        "py_black_win": replay["black_win"],
        "py_draw": replay["draw"],
        "py_white_win": replay["white_win"],
        "py_unfinished": replay["unfinished"],
        "py_illegal": replay["illegal"],
    }


def _torch_summary(records) -> dict[str, float]:
    q_num = 0.0
    q_den = 0.0
    qhp = []
    rows = 0
    decisive = 0
    finished = 0
    black_win = 0
    draw = 0
    white_win = 0
    unfinished = 0
    plies = []
    for record in records:
        plies.append(record.plies)
        if record.result != "*":
            finished += 1
        if record.result not in ("*", "1/2-1/2"):
            decisive += 1
        if record.result == "0-1":
            black_win += 1
        elif record.result == "1/2-1/2":
            draw += 1
        elif record.result == "1-0":
            white_win += 1
        else:
            unfinished += 1
        for sample in record.samples:
            if len(sample.q_visits):
                q_num += float((np.abs(sample.q_values) * sample.q_visits).sum())
                q_den += float(sample.q_visits.sum())
            rows += 1
        if record.q_head_played:
            qhp.extend(abs(float(v)) for v in record.q_head_played)
    denom = max(1, finished)
    return {
        "games": float(len(records)),
        "plies": float(np.mean(plies)) if plies else 0.0,
        "finished": float(finished / max(1, len(records))),
        "decisive": float(decisive / denom),
        "qabs": float(q_num / max(q_den, 1.0)),
        "qvis": float(q_den / max(rows, 1)),
        "qhp": float(np.mean(qhp)) if qhp else 0.0,
        "rows": float(rows),
        "black_win": float(black_win),
        "draw": float(draw),
        "white_win": float(white_win),
        "unfinished": float(unfinished),
    }


def _print_summary(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name}: games={stats['games']:.0f} plies={stats['plies']:.1f} "
        f"finished={stats['finished']:.0%} decisive={stats['decisive']:.0%} "
        f"qabs={stats['qabs']:.4f} qvis={stats['qvis']:.1f} "
        f"qhp={stats['qhp']:.4f} rows={stats['rows']:.0f} "
        f"results B/D/W/*={stats['black_win']:.0f}/"
        f"{stats['draw']:.0f}/{stats['white_win']:.0f}/{stats['unfinished']:.0f}"
    )
    if "py_black_win" in stats:
        print(
            f"{name}-python-replay: B/D/W/*="
            f"{stats['py_black_win']:.0f}/{stats['py_draw']:.0f}/"
            f"{stats['py_white_win']:.0f}/{stats['py_unfinished']:.0f} "
            f"illegal={stats['py_illegal']:.0f}"
        )


def run_compare(args: argparse.Namespace) -> int:
    cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        head_dim=args.head_dim,
    )
    _jax_model, params = build_model(cfg, jax.random.PRNGKey(args.seed))
    jax_scfg = SearchConfig(
        sims=args.sims,
        root_candidates=args.candidates,
        q_trust=args.q_trust,
    )
    jax_spcfg = SelfPlayConfig(max_plies=args.max_plies)
    jax_samples, jax_meta = generate_selfplay(
        params,
        jax.random.PRNGKey(args.seed + 1),
        args.games,
        jax_scfg,
        jax_spcfg,
        False,
    )
    jax_stats = _jax_summary(jax_samples, jax_meta)
    _print_summary("jax", jax_stats)
    replay_ok = (
        jax_stats["py_illegal"] == 0
        and jax_stats["black_win"] == jax_stats["py_black_win"]
        and jax_stats["draw"] == jax_stats["py_draw"]
        and jax_stats["white_win"] == jax_stats["py_white_win"]
        and jax_stats["unfinished"] == jax_stats["py_unfinished"]
    )

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "fresh.pt"
        export_torch_checkpoint(params, cfg, str(ckpt))
        torch_model = load_torch_checkpoint(ckpt)
        device = torch.device(args.torch_device)
        torch_model = torch_model.to(device).eval()

        records = []

        def on_record(record) -> bool:
            records.append(record)
            return len(records) < args.games

        run_vector_selfplay(
            torch_model,
            device,
            TorchSearchConfig(
                sims=args.sims,
                root_candidates=args.candidates,
                q_trust=args.q_trust,
            ),
            TorchSelfPlayConfig(max_plies=args.max_plies),
            None,
            gate_fn=lambda: False,
            batch_games=args.games,
            master_rng=np.random.default_rng(args.seed + 2),
            on_record=on_record,
            should_stop=lambda: False,
        )
        _print_summary("torch", _torch_summary(records))
    if not replay_ok:
        print("SELFPLAY REPLAY CHECK FAILED")
        return 1
    print("SELFPLAY REPLAY CHECK PASSED")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--q-trust", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    raise SystemExit(run_compare(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
