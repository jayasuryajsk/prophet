"""Deterministic self-play digest for validating byte-identical refactors.

Runs run_vector_selfplay on CPU with a fixed seed and small net, then prints
a hash digest of the resulting GameRecords (move indices, policy targets, q
values, value targets). Run BEFORE and AFTER a refactor; the digest must be
identical for a byte-identical change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: I001
import hashlib

import numpy as np

from prophet.model import ModelConfig, PolicyQValueNet
from prophet.search import SearchConfig
from prophet.selfplay import SelfPlayConfig
from prophet.worker import run_vector_selfplay


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = PolicyQValueNet(
        ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=256)
    ).to(device)
    model.eval()

    scfg = SearchConfig(sims=32, root_candidates=8)
    spcfg = SelfPlayConfig(max_plies=60)
    h = hashlib.sha256()
    recs = []

    def on_record(rec):
        recs.append(rec)
        h.update(rec.result.encode())
        h.update(np.int64(rec.plies).tobytes())
        for s in rec.samples:
            h.update(np.int64(s.played_index).tobytes())
            h.update(np.round(s.policy_target, 5).astype(np.float64).tobytes())
            h.update(np.round(s.value_target, 5).astype(np.float64).tobytes())
            h.update(s.q_indices.astype(np.int64).tobytes())
            h.update(np.round(s.q_values, 5).astype(np.float64).tobytes())
            h.update(s.legal_indices.astype(np.int64).tobytes())
        return len(recs) < 6

    run_vector_selfplay(
        model, device, scfg, spcfg, None,
        gate_fn=lambda: False, batch_games=4,
        master_rng=np.random.default_rng(0),
        on_record=on_record,
        should_stop=lambda: len(recs) >= 6,
    )
    tot_plies = sum(r.plies for r in recs)
    tot_samples = sum(len(r.samples) for r in recs)
    print(f"games={len(recs)} plies={tot_plies} samples={tot_samples}")
    print(f"DIGEST {h.hexdigest()}")


if __name__ == "__main__":
    main()
