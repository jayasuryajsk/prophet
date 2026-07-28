"""Training step: policy CE + value MSE + per-move Q regression +
negamax consistency Q(s, a_played) ~= -stopgrad(V(child))."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .accel import autocast
from .encoding import NUM_ACTIONS


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    q: float = 1.0
    consistency: float = 0.5
    wdl: float = 0.5
    mlh: float = 0.1


def collate(samples, device):
    b = len(samples)
    x = torch.from_numpy(np.stack([s.x for s in samples])).to(device)
    child_x = torch.from_numpy(np.stack([s.child_x for s in samples])).to(device)
    played = torch.tensor([s.played_index for s in samples], device=device)
    value = torch.tensor([s.value_target for s in samples], device=device)
    weight = torch.tensor([s.weight for s in samples], device=device)
    wdl = torch.tensor([getattr(s, "wdl", -1) for s in samples], device=device)
    moves_left = torch.tensor(
        [getattr(s, "moves_left", -1.0) for s in samples], device=device
    )
    policy_ok = torch.tensor(
        [getattr(s, "policy_ok", True) for s in samples], device=device
    )

    mask = torch.zeros(b, NUM_ACTIONS, dtype=torch.bool)
    policy = torch.zeros(b, NUM_ACTIONS)
    q_target = torch.zeros(b, NUM_ACTIONS)
    q_weight = torch.zeros(b, NUM_ACTIONS)
    for i, s in enumerate(samples):
        mask[i, s.legal_indices] = True
        policy[i, s.legal_indices] = torch.from_numpy(s.policy_target)
        if len(s.q_indices):
            q_target[i, s.q_indices] = torch.from_numpy(s.q_values)
            q_weight[i, s.q_indices] = torch.from_numpy(s.q_visits)
    return {
        "x": x,
        "child_x": child_x,
        "played": played,
        "value": value,
        "weight": weight,
        "wdl": wdl,
        "moves_left": moves_left,
        "policy_ok": policy_ok,
        "mask": mask.to(device),
        "policy": policy.to(device),
        "q_target": q_target.to(device),
        "q_weight": q_weight.to(device),
    }


def train_step(model, optimizer, batch, weights: LossWeights | None = None):
    w = weights or LossWeights()
    model.train()
    with autocast(batch["x"].device):
        logits, adv, v, wdl_probs, mlh = model.forward_wdl(batch["x"])

        wn = batch["weight"] / batch["weight"].mean().clamp_min(1e-8)

        # dueling composition under the LEGAL mask: the best legal move's Q
        # equals v, so every Q-target gradient also trains the value stream —
        # the value head eats the Q head's dense per-move signal.
        a_masked = adv.masked_fill(~batch["mask"], float("-inf"))
        a_max = a_masked.max(dim=-1, keepdim=True).values
        q = torch.tanh(
            torch.atanh(v.clamp(-0.997, 0.997)).unsqueeze(1) + adv - a_max
        )

        masked = logits.masked_fill(~batch["mask"], float("-inf"))
        logp = F.log_softmax(masked, dim=-1)
        logp = torch.where(batch["mask"], logp, torch.zeros_like(logp))
        # PCR: cheap-search moves carry no usable policy target — the policy
        # trains only on full-budget searches (value/Q/WDL/MLH train on all).
        pok = batch["policy_ok"]
        per_pi = wn * -(batch["policy"] * logp).sum(dim=-1)
        loss_pi = (per_pi * pok).sum() / pok.sum().clamp_min(1)

        loss_v = (wn * (v - batch["value"]).pow(2)).mean()

        qw = batch["q_weight"]
        per_q = ((q - batch["q_target"]).pow(2) * qw).sum(dim=-1) / qw.sum(dim=-1).clamp_min(1.0)
        # weight Q regression toward decisive positions so late-training drawish
        # targets don't flatten the Q-head (the v1 Q-regression)
        q_scale = 0.5 + batch["value"].abs()
        loss_q = (wn * q_scale * per_q).mean()

        has_wdl = batch["wdl"] >= 0
        if has_wdl.any():
            logp_wdl = torch.log(wdl_probs.clamp_min(1e-8))
            nll = -logp_wdl.gather(1, batch["wdl"].clamp_min(0).unsqueeze(1)).squeeze(1)
            loss_wdl = (wn * nll * has_wdl).sum() / has_wdl.sum().clamp_min(1)
        else:
            loss_wdl = torch.zeros((), device=v.device)

        with torch.no_grad():
            _, _, v_child = model(batch["child_x"])
        q_played = q.gather(1, batch["played"].unsqueeze(1)).squeeze(1)
        loss_cons = (wn * (q_played + v_child).pow(2)).mean()

        # moves-left: predict plies until the game ends (anti-shuffle signal;
        # scaled Huber so long-game errors don't dominate)
        ml = batch["moves_left"]
        has_ml = ml >= 0
        if has_ml.any():
            per_ml = F.smooth_l1_loss(mlh / 20.0, ml / 20.0, reduction="none")
            loss_mlh = (wn * per_ml * has_ml).sum() / has_ml.sum().clamp_min(1)
        else:
            loss_mlh = torch.zeros((), device=v.device)

        loss = (
            w.policy * loss_pi + w.value * loss_v + w.q * loss_q
            + w.consistency * loss_cons + w.wdl * loss_wdl
            + w.mlh * loss_mlh
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": loss.item(),
        "policy": loss_pi.item(),
        "value": loss_v.item(),
        "q": loss_q.item(),
        "consistency": loss_cons.item(),
        "wdl": loss_wdl.item(),
        "mlh": loss_mlh.item(),
    }
