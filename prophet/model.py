"""Transformer over 64 square tokens with three coupled from-to heads.

- policy: bilinear from/to scores -> 4096 logits ("which move to consider")
- Q:      bilinear from/to scores, tanh -> [-1, 1] per move ("how good is
          the position after this move, for the side to move") — the
          explicit per-move intuition head
- V:      pooled scalar in [-1, 1] ("how the game stands for side to move")
"""

import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .encoding import FEATURES, NUM_ACTIONS


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    head_dim: int = 64
    dropout: float = 0.0
    in_features: int = FEATURES


class PolicyQValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.embed = nn.Linear(cfg.in_features, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer, cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        d, h = cfg.d_model, cfg.head_dim
        self.p_from = nn.Linear(d, h)
        self.p_to = nn.Linear(d, h)
        self.q_from = nn.Linear(d, h)
        self.q_to = nn.Linear(d, h)
        # WDL value head: 3 logits (loss, draw, win) from the side to move's
        # perspective; the scalar v consumed by search is P(win) - P(loss)
        self.v_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward_wdl(self, x: torch.Tensor):
        """x: [B, 64, F] -> (policy [B,4096], q [B,4096], v [B], wdl [B,3])."""
        b = x.shape[0]
        h = self.norm(self.trunk(self.embed(x) + self.pos))
        scale = 1.0 / math.sqrt(self.cfg.head_dim)
        policy = torch.einsum("bid,bjd->bij", self.p_from(h), self.p_to(h)) * scale
        q = torch.tanh(
            torch.einsum("bid,bjd->bij", self.q_from(h), self.q_to(h)) * scale
        )
        wdl = torch.softmax(self.v_head(h.mean(dim=1)), dim=-1)  # [B, 3] L/D/W
        v = wdl[:, 2] - wdl[:, 0]
        return policy.reshape(b, NUM_ACTIONS), q.reshape(b, NUM_ACTIONS), v, wdl

    def forward(self, x: torch.Tensor):
        policy, q, v, _ = self.forward_wdl(x)
        return policy, q, v

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_checkpoint(model: PolicyQValueNet, path):
    """Atomic, config-aware checkpoint: {'config': ..., 'state': ...}."""
    obj = {
        "config": asdict(model.cfg),
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


def extract_state(path) -> dict:
    """State dict from a checkpoint (config-aware or bare v1 format)."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["state"] if isinstance(obj, dict) and "state" in obj else obj


def load_checkpoint(path) -> PolicyQValueNet:
    """Build the right-sized model from a checkpoint and load its weights.
    Older checkpoints (18-feature input, scalar value head) are upgraded
    in place: zero-padded input columns and a WDL head initialized so that
    P(win) - P(loss) reproduces the old scalar value."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state" in obj:
        cfg_kwargs, sd = dict(obj["config"]), obj["state"]
    else:
        cfg_kwargs, sd = {}, obj  # bare v1 state dict
    cfg_kwargs["in_features"] = sd["embed.weight"].shape[1]
    cfg = ModelConfig(**cfg_kwargs)
    sd = upgrade_state(sd, cfg)
    model = PolicyQValueNet(cfg)
    model.load_state_dict(sd)
    model.eval()
    return model


def upgrade_state(sd: dict, cfg: ModelConfig) -> dict:
    """Upgrade a scalar-value state dict to the WDL head in place.
    Old head ends in Linear(d, 1) with weight [1, d]; new head ends in
    Linear(d, 3) ordered (loss, draw, win). Setting win-row = w, loss-row
    = -w, draw-row = 0 makes P(win)-P(loss) track tanh(w.h), preserving
    the learned value function at warm start."""
    w = sd.get("v_head.2.weight")
    if w is not None and w.shape[0] == 1:
        sd = dict(sd)
        # 1.5x: softmax(a,0,-a) gives pw-pl with 2/3 the slope of tanh(a)
        # near zero; rescaling preserves the learned value magnitudes.
        w = 1.5 * w
        b = 1.5 * sd["v_head.2.bias"]
        sd["v_head.2.weight"] = torch.cat([-w, torch.zeros_like(w), w], dim=0)
        sd["v_head.2.bias"] = torch.cat([-b, torch.zeros_like(b), b], dim=0)
    return sd


def widen_input(model: PolicyQValueNet, new_features: int) -> PolicyQValueNet:
    """Grow the input feature dimension with zero-initialized columns —
    the network's function is unchanged until training uses the new senses."""
    old = model.embed
    if old.in_features == new_features:
        return model
    cfg = ModelConfig(**{**asdict(model.cfg), "in_features": new_features})
    grown = PolicyQValueNet(cfg)
    sd = model.state_dict()
    w = torch.zeros(old.out_features, new_features)
    w[:, : old.in_features] = sd["embed.weight"]
    sd["embed.weight"] = w
    grown.load_state_dict(sd)
    return grown
