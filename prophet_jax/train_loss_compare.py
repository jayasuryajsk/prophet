"""Compare JAX and PyTorch loss terms on the same dense batch.

This is a diagnostic for the JAX port. It builds a small JAX model, exports the
same weights to the PyTorch model, generates one JAX self-play batch, and then
computes the five training losses in both frameworks on identical tensors.

Example:

    python -m prophet_jax.train_loss_compare
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from prophet.model import load_checkpoint as load_torch_checkpoint

from .config import LossWeights, ModelConfig, SearchConfig, SelfPlayConfig
from .model import build_model, export_torch_checkpoint
from .selfplay import SamplesBatch, generate_selfplay
from .train import LossWeightArr, TrainState, _train_step_impl, make_train_state


def _valid_prefix(samples: SamplesBatch, n: int) -> SamplesBatch:
    valid = np.asarray(samples.valid).astype(bool)
    idx = np.flatnonzero(valid)[:n]
    if len(idx) < n:
        raise ValueError(f"only {len(idx)} valid rows available, need {n}")
    fields = {
        name: jnp.asarray(np.asarray(getattr(samples, name))[idx])
        for name in samples.__dataclass_fields__
    }
    fields["valid"] = jnp.ones((n,), dtype=jnp.bool_)
    return SamplesBatch(**fields)


def _torch_metrics(model, batch: SamplesBatch, weights: LossWeights) -> dict[str, float]:
    device = torch.device("cpu")
    model = model.to(device).eval()

    x = torch.from_numpy(np.asarray(batch.x)).float()
    child_x = torch.from_numpy(np.asarray(batch.child_x)).float()
    played = torch.from_numpy(np.asarray(batch.played)).long()
    value = torch.from_numpy(np.asarray(batch.value)).float()
    weight = torch.from_numpy(np.asarray(batch.weight)).float()
    wdl = torch.from_numpy(np.asarray(batch.wdl)).long()
    mask = torch.from_numpy(np.asarray(batch.mask)).bool()
    policy = torch.from_numpy(np.asarray(batch.policy)).float()
    q_target = torch.from_numpy(np.asarray(batch.q_target)).float()
    q_weight = torch.from_numpy(np.asarray(batch.q_weight)).float()

    logits, q, v, wdl_probs = model.forward_wdl(x)
    wn = weight / weight.mean().clamp_min(1e-8)

    masked = logits.masked_fill(~mask, float("-inf"))
    logp = F.log_softmax(masked, dim=-1)
    logp = torch.where(mask, logp, torch.zeros_like(logp))
    loss_pi = (wn * -(policy * logp).sum(dim=-1)).mean()

    loss_v = (wn * (v - value).pow(2)).mean()

    per_q = ((q - q_target).pow(2) * q_weight).sum(dim=-1) / q_weight.sum(dim=-1).clamp_min(1.0)
    q_scale = 0.5 + value.abs()
    loss_q = (wn * q_scale * per_q).mean()

    has_wdl = wdl >= 0
    if has_wdl.any():
        logp_wdl = torch.log(wdl_probs.clamp_min(1e-8))
        nll = -logp_wdl.gather(1, wdl.clamp_min(0).unsqueeze(1)).squeeze(1)
        loss_wdl = (wn * nll * has_wdl).sum() / has_wdl.sum().clamp_min(1)
    else:
        loss_wdl = torch.zeros(())

    with torch.no_grad():
        _, _, v_child = model(child_x)
    q_played = q.gather(1, played.unsqueeze(1)).squeeze(1)
    loss_cons = (wn * (q_played + v_child).pow(2)).mean()

    loss = (
        weights.policy * loss_pi
        + weights.value * loss_v
        + weights.q * loss_q
        + weights.consistency * loss_cons
        + weights.wdl * loss_wdl
    )
    return {
        "loss": float(loss.detach()),
        "policy": float(loss_pi.detach()),
        "value": float(loss_v.detach()),
        "q": float(loss_q.detach()),
        "consistency": float(loss_cons.detach()),
        "wdl": float(loss_wdl.detach()),
    }


def _jax_metrics(state: TrainState, batch: SamplesBatch, weights: LossWeights) -> dict[str, float]:
    warr = LossWeightArr(
        policy=jnp.asarray(weights.policy, jnp.float32),
        value=jnp.asarray(weights.value, jnp.float32),
        q=jnp.asarray(weights.q, jnp.float32),
        consistency=jnp.asarray(weights.consistency, jnp.float32),
        wdl=jnp.asarray(weights.wdl, jnp.float32),
    )
    _new_state, metrics = _train_step_impl(state, batch, warr)
    return {k: float(v) for k, v in metrics.items()}


def run_compare(args: argparse.Namespace) -> int:
    cfg = ModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        head_dim=args.head_dim,
    )
    model, params = build_model(cfg, jax.random.PRNGKey(args.seed))
    scfg = SearchConfig(sims=args.sims, root_candidates=args.candidates, q_trust=args.q_trust)
    spcfg = SelfPlayConfig(max_plies=args.max_plies)
    samples, _meta = generate_selfplay(
        params, jax.random.PRNGKey(args.seed + 1), args.batch_games, scfg, spcfg, False
    )
    batch = _valid_prefix(samples, args.batch)
    weights = LossWeights()

    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "fresh.pt"
        export_torch_checkpoint(params, cfg, str(ckpt))
        torch_model = load_torch_checkpoint(ckpt)

        state = make_train_state(model, params, lr=0.0, wd=0.0)
        jm = _jax_metrics(state, batch, weights)
        tm = _torch_metrics(torch_model, batch, weights)

    print("metric,jax,torch,delta")
    ok = True
    for key in ("loss", "policy", "value", "q", "consistency", "wdl"):
        delta = jm[key] - tm[key]
        print(f"{key},{jm[key]:.8f},{tm[key]:.8f},{delta:+.8f}")
        ok = ok and abs(delta) <= args.tol
    if not ok:
        print("TRAIN LOSS COMPARE FAILED")
        return 1
    print("TRAIN LOSS COMPARE PASSED")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--q-trust", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=16)
    parser.add_argument("--batch-games", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-3)
    raise SystemExit(run_compare(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
