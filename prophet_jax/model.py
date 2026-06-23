"""Flax linen port of Prophet's PolicyQValueNet (transformer + 3 coupled heads).

This is a *bit-for-functional-equivalence* reimplementation of the PyTorch
``prophet/model.py`` so that torch checkpoints transfer directly:

- Same trunk: per-token input embedding (Linear, with bias) + a learned
  positional parameter [1, 64, d_model], then ``n_layers`` PRE-LN transformer
  encoder blocks (PyTorch ``TransformerEncoderLayer(norm_first=True)`` math),
  then a final LayerNorm.
- Same three coupled heads reading the shared trunk output h:[B, 64, d_model]:
    * policy: bilinear from/to scores -> 4096 raw logits.
    * Q:      bilinear from/to scores -> tanh -> 4096 values in [-1, 1].
    * V/WDL:  mean-pool over the 64 tokens -> Linear -> GELU -> Linear(3) ->
              softmax (loss, draw, win); scalar v = P(win) - P(loss).

Numerics matched to the torch reference:
- LayerNorm epsilon = 1e-5 (torch default), with scale + bias.
- GELU is the *exact* erf gelu (``jax.nn.gelu(x, approximate=False)``), matching
  torch ``activation="gelu"`` and ``nn.GELU()`` defaults.
- Self-attention is full bidirectional attention over the 64 tokens, NO mask,
  scale = 1/sqrt(d_model / n_heads).
- Policy/Q reshape is C-order (index = from*64 + to). DO NOT transpose.

The hard part of the port is the *weight loader*. Torch packs Q/K/V into a
single ``in_proj_weight`` [3d, d] (+ ``in_proj_bias`` [3d]) and a separate
``out_proj`` Linear(d->d). Flax's ``MultiHeadDotProductAttention`` instead has
four separate Dense sub-modules ``query``/``key``/``value``/``out``, each with a
kernel of shape ``[in, ..., out]`` (= weight.T) reshaped over the head axis to
``[d, n_heads, head]`` (or ``[n_heads, head, d]`` for ``out``). The loader splits
and reshapes accordingly; the exporter reverses it so JAX-trained nets can be
gauntleted by the existing torch scripts.

NOTE: jax / flax / optax are not installed in this repo (only torch + numpy).
This module is written against the verified Flax linen API but has not been
executed here; ``torch.load`` is used *outside* JAX to read the checkpoint, and
numpy is the interchange format between the two frameworks.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

# ---------------------------------------------------------------------------
# Constants + ModelConfig.
#
# The canonical definitions live in ``prophet_jax.config`` (per the module
# plan). To keep this file standalone-importable *before* config.py exists, we
# import from there when available and fall back to identical local definitions
# otherwise. Either way the values match prophet/encoding.py + prophet/model.py.
# ---------------------------------------------------------------------------

# LayerNorm epsilon: torch.nn.LayerNorm default is 1e-5. Flax defaults to 1e-6,
# so we MUST pass this explicitly everywhere to match torch numerics.
LN_EPS = 1e-5

try:  # prefer the shared config module once it exists (single source of truth)
    from .config import ModelConfig, NUM_ACTIONS, FEATURES  # type: ignore
except Exception:  # noqa: BLE001 - config.py not present yet -> local fallback
    NUM_ACTIONS = 64 * 64  # 4096 = from*64 + to
    FEATURES = 24  # current production input width (24-feature encoding)

    @dataclass
    class ModelConfig:
        """Architecture config. Defaults match the torch ``ModelConfig`` exactly.

        The port is config-driven: production runs override
        ``d_model``/``n_layers`` (e.g. 192/6 for the 2.77M "100k" run, 320/8 for
        the 10M "moonshot" run). ``load_torch_checkpoint`` reads these from the
        checkpoint's ``config`` dict and reads ``in_features`` off
        ``embed.weight``; nothing here is hardcoded into the forward pass.
        """

        d_model: int = 128
        n_layers: int = 4
        n_heads: int = 4
        d_ff: int = 512
        head_dim: int = 64
        dropout: float = 0.0
        in_features: int = FEATURES


# ---------------------------------------------------------------------------
# Flax modules.
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    """One PRE-LN transformer encoder block (torch ``norm_first=True`` math).

    Block formula (exactly the torch TransformerEncoderLayer with
    ``norm_first=True``)::

        y = x + SelfAttn(LayerNorm1(x))          # full self-attn, no mask
        z = y + FF(LayerNorm2(y))                # Dense(d_ff)->gelu->Dense(d)

    Self-attention scale is 1/sqrt(d_model/n_heads); Flax's
    ``MultiHeadDotProductAttention`` applies exactly this scale by default
    (1/sqrt(head_dim) with head_dim = qkv_features/num_heads), so we do not
    override it. Dropout is 0.0 in production; ``deterministic`` is wired so no
    rng is needed at eval.
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        det = not train

        # (a) pre-LN self-attention. Single-arg call == self attention in Flax
        # (inputs_k / inputs_v copy inputs_q). qkv_features = d_model so the
        # per-head dim is d_model / n_heads and the default scale is
        # 1/sqrt(d_model/n_heads) -- matching torch MultiheadAttention.
        h = nn.LayerNorm(epsilon=LN_EPS, name="norm1")(x)
        # VERIFY: MultiHeadDotProductAttention creates four sub-modules named
        # "query"/"key"/"value"/"out" (the loader/exporter pytree depends on
        # exactly these names). If a future flax renames them, update the keys
        # in _state_dict_to_params / _params_to_state_dict to match
        # `model.tabulate(...)` or `jax.tree_util.tree_map(jnp.shape, params)`.
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=det,
            name="self_attn",
        )(h)  # full bidirectional attention over 64 tokens, NO mask
        x = x + h

        # (b) pre-LN feed-forward: Dense(d_ff) -> exact-erf GELU -> Dense(d).
        h = nn.LayerNorm(epsilon=LN_EPS, name="norm2")(x)
        h = nn.Dense(self.d_ff, name="linear1")(h)
        h = jax.nn.gelu(h, approximate=False)  # exact erf gelu (torch "gelu")
        h = nn.Dense(self.d_model, name="linear2")(h)
        x = x + h
        return x


class PolicyQValueNet(nn.Module):
    """Transformer over 64 square tokens with three coupled from-to heads.

    Mirrors ``prophet/model.py``::

        h = norm(trunk(embed(x) + pos))          # [B, 64, d_model]
        policy = einsum("bid,bjd->bij", p_from(h), p_to(h)) * scale  -> [B,4096]
        q      = tanh(einsum("bid,bjd->bij", q_from(h), q_to(h)) * scale) -> [B,4096]
        wdl    = softmax(v_head(h.mean(1)))      # [B,3] = (loss, draw, win)
        v      = wdl[:,2] - wdl[:,0]

    ``scale = 1/sqrt(head_dim)``. The policy/q reshape is C-order so that the
    flat index equals ``from*64 + to`` (token i is the FROM square, token j is
    the TO square) -- matching ``encoding.move_to_index``. DO NOT transpose.
    """

    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        cfg = self.cfg
        b = x.shape[0]

        # 1) per-token input embedding (Linear with bias). nn.Dense applies over
        #    the last axis only, so [B,64,F] -> [B,64,d_model] directly.
        h = nn.Dense(cfg.d_model, name="embed")(x)

        # 2) learned positional parameter [1, 64, d_model], init randn * 0.02,
        #    broadcast-added over the batch.
        pos = self.param(
            "pos",
            lambda key, shape: jax.random.normal(key, shape) * 0.02,
            (1, 64, cfg.d_model),
        )
        h = h + pos

        # 3) n_layers PRE-LN encoder blocks.
        # VERIFY: explicit name=f"layers_{i}" fixes the param-tree keys so the
        # torch<->flax loader/exporter can address each block deterministically.
        # The whole weight mapping assumes the resulting params pytree is exactly
        # {embed, pos, layers_0..N-1{norm1,norm2,self_attn{query,key,value,out},
        # linear1,linear2}, norm, p_from,p_to,q_from,q_to, v_head_0,v_head_2}.
        # Confirm once with build_model(...) -> jax.tree_util.tree_map(jnp.shape).
        for i in range(cfg.n_layers):
            h = EncoderBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                name=f"layers_{i}",
            )(h, train=train)

        # 4) final LayerNorm after the trunk.
        h = nn.LayerNorm(epsilon=LN_EPS, name="norm")(h)  # [B, 64, d_model]

        scale = 1.0 / math.sqrt(cfg.head_dim)

        # HEAD 1 -- policy: bilinear from/to scores -> 4096 raw logits.
        pf = nn.Dense(cfg.head_dim, name="p_from")(h)  # [B,64,head_dim]
        pt = nn.Dense(cfg.head_dim, name="p_to")(h)
        policy = jnp.einsum("bid,bjd->bij", pf, pt) * scale  # [B,64,64]
        policy = policy.reshape(b, NUM_ACTIONS)  # C-order: idx = from*64 + to

        # HEAD 2 -- Q: bilinear from/to scores -> tanh -> [-1,1]. Distinct
        # projections from policy (q_from/q_to are their own Dense layers).
        qf = nn.Dense(cfg.head_dim, name="q_from")(h)
        qt = nn.Dense(cfg.head_dim, name="q_to")(h)
        q = jnp.tanh(jnp.einsum("bid,bjd->bij", qf, qt) * scale)
        q = q.reshape(b, NUM_ACTIONS)

        # HEAD 3 -- V / WDL: mean-pool over the 64 tokens -> Dense(d) -> exact
        # GELU -> Dense(3) -> softmax (loss, draw, win). v = P(win) - P(loss).
        pooled = h.mean(axis=1)  # [B, d_model]
        vh = nn.Dense(cfg.d_model, name="v_head_0")(pooled)
        vh = jax.nn.gelu(vh, approximate=False)
        v_logits = nn.Dense(3, name="v_head_2")(vh)  # [B, 3]
        wdl = jax.nn.softmax(v_logits, axis=-1)  # [B, 3] = (loss, draw, win)
        v = wdl[:, 2] - wdl[:, 0]  # P(win) - P(loss) in [-1, 1]

        return policy, q, v, wdl


# ---------------------------------------------------------------------------
# Build / forward helpers (the interface other modules import).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Process-default model registry.
#
# CROSS-MODULE CONTRACT: ``search.py`` (and the mctx root/recurrent fns it
# builds) only ever threads ``params`` -- never the Flax ``model`` object --
# because mctx's recurrent_fn signature is ``(params, rng, action, embedding)``.
# But Flax needs the bound module (architecture) to run ``model.apply``. So we
# register the most-recently-built model here and let the 2-arg ``forward(params,
# x)`` / ``forward_wdl(params, x)`` calls (used by search) fall back to it, while
# the 3-arg ``forward(model, params, x)`` calls (used by train.py) pass the model
# explicitly. ``build_model`` registers automatically; callers may also pin a
# specific model with :func:`set_default_model`.
#
# There is exactly one network per learner process, so a process default is
# safe; warm-start (load_torch_checkpoint -> build_model) re-registers the
# correctly-sized model before any search runs.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL: "PolicyQValueNet | None" = None


def set_default_model(model: "PolicyQValueNet") -> None:
    """Register ``model`` as the process default used by the 2-arg ``forward``."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model


def get_default_model() -> "PolicyQValueNet":
    """Return the registered process-default model (or raise if unset)."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError(
            "prophet_jax.model: no default model registered. Call build_model("
            "cfg) (which registers it) or set_default_model(model) before using "
            "the 2-arg forward(params, x) / forward_wdl(params, x) form used by "
            "search.py."
        )
    return _DEFAULT_MODEL


def build_model(cfg: ModelConfig, key=None):
    """Construct the model and initialize params with a shape-inference pass.

    Returns ``(model, params)`` where ``params`` is the Flax param pytree (the
    ``'params'`` collection, NOT the wrapping ``{'params': ...}`` dict).
    Init uses ``train=False`` so dropout never runs and only the ``'params'``
    rng is required. Also registers ``model`` as the process default (see
    :func:`set_default_model`) so the 2-arg ``forward(params, x)`` used by
    search.py works without threading the model object.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    model = PolicyQValueNet(cfg)
    dummy = jnp.zeros((1, 64, cfg.in_features), dtype=jnp.float32)
    variables = model.init(key, dummy, train=False)
    set_default_model(model)
    return model, variables["params"]


def _resolve_model_params(a, b):
    """Disambiguate the (model, params) vs (params,) calling conventions.

    Returns ``(model, params)``. Two supported shapes:
      * ``forward(model, params, x)`` -> ``a`` is a PolicyQValueNet, ``b`` is
        params (the explicit form used by train.py).
      * ``forward(params, x)``        -> ``a`` is params, ``b`` is None, and the
        process-default model (from build_model/set_default_model) is used (the
        params-only form used by search.py).
    """
    if isinstance(a, PolicyQValueNet):
        return a, b
    # 2-arg form: `a` is params, the model comes from the registry.
    return get_default_model(), a


def _apply(model, params, x, *, dtype=None):
    """Internal: run the model in eval mode, optionally casting to ``dtype``.

    fp32 is the reference path. Passing ``dtype=jnp.bfloat16`` casts params and
    inputs to bf16 for an (optional) reduced-precision forward on TPU/GPU; the
    project treats fp32 as the reference and bf16 as an approximation, so exact
    bit-match to the torch CUDA bf16 path is NOT required.
    """
    if dtype is not None and dtype != jnp.float32:
        params = jax.tree_util.tree_map(lambda p: p.astype(dtype), params)
        x = x.astype(dtype)
    # train=False => deterministic everywhere => no rng needed.
    return model.apply({"params": params}, x, train=False)


def forward_wdl(a, b, x=None, *, dtype=None):
    """Eval forward returning all four outputs.

    Polymorphic on arity:
      * ``forward_wdl(model, params, x)`` (train.py) -- explicit model.
      * ``forward_wdl(params, x)``        (search.py) -- uses the registered
        process-default model (see :func:`build_model`).

    Returns ``(policy[B,4096] raw logits, q[B,4096] in [-1,1], v[B], wdl[B,3])``.
    Training consumes ``wdl`` for the WDL cross-entropy term.
    """
    if x is None:  # 2-arg form: a=params, b=x
        model, params, x = get_default_model(), a, b
    else:  # 3-arg form: a=model, b=params
        model, params = _resolve_model_params(a, b)
    return _apply(model, params, x, dtype=dtype)


def forward(a, b, x=None, *, dtype=None):
    """Eval forward returning the 3-tuple search consumes.

    Polymorphic on arity (see :func:`forward_wdl`):
      * ``forward(model, params, x)`` (train.py) -- explicit model.
      * ``forward(params, x)``        (search.py) -- registered default model.

    Returns ``(policy[B,4096] raw logits, q[B,4096] in [-1,1], v[B])``.
    """
    if x is None:  # 2-arg form: a=params, b=x
        model, params, x = get_default_model(), a, b
    else:  # 3-arg form: a=model, b=params
        model, params = _resolve_model_params(a, b)
    policy, q, v, _ = _apply(model, params, x, dtype=dtype)
    return policy, q, v


def num_params(params) -> int:
    """Total number of scalar parameters (sum of leaf sizes)."""
    return int(sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params)))


# ---------------------------------------------------------------------------
# Torch <-> Flax weight mapping.
#
# Flax MultiHeadDotProductAttention sub-module param shapes (verified API):
#   query/key/value: kernel [in_features, num_heads, head_dim], bias [num_heads, head_dim]
#   out:             kernel [num_heads, head_dim, out_features], bias [out_features]
# where head_dim = qkv_features / num_heads.  (DenseGeneral kernels are
# weight.T relative to torch's [out, in] convention, with the output axis split
# over the head dimension.)
#
# Torch MultiheadAttention:
#   in_proj_weight [3d, d] = rows 0:d=q, d:2d=k, 2d:3d=v (each [out=d, in=d]).
#   in_proj_bias   [3d]    = same q/k/v split.
#   out_proj.weight [d, d] = [out=d, in=d]; out_proj.bias [d].
# ---------------------------------------------------------------------------


def _dense_kernel(weight_oi: np.ndarray) -> np.ndarray:
    """torch Linear weight [out, in] -> flax Dense kernel [in, out] = weight.T."""
    return np.asarray(weight_oi, dtype=np.float32).T.copy()


def _attn_qkv_kernel(w_oi: np.ndarray, n_heads: int) -> np.ndarray:
    """torch q/k/v sub-weight [d, d]=[out,in] -> flax [in, n_heads, head_dim]."""
    w_oi = np.asarray(w_oi, dtype=np.float32)
    d_out, d_in = w_oi.shape
    head_dim = d_out // n_heads
    # kernel [in, out] then split the out axis into (n_heads, head_dim).
    return w_oi.T.reshape(d_in, n_heads, head_dim).copy()


def _attn_qkv_bias(b_o: np.ndarray, n_heads: int) -> np.ndarray:
    """torch q/k/v sub-bias [d] -> flax [n_heads, head_dim]."""
    b_o = np.asarray(b_o, dtype=np.float32)
    d_out = b_o.shape[0]
    head_dim = d_out // n_heads
    return b_o.reshape(n_heads, head_dim).copy()


def _attn_out_kernel(w_oi: np.ndarray, n_heads: int) -> np.ndarray:
    """torch out_proj.weight [d, d]=[out,in] -> flax out kernel [n_heads, head_dim, out].

    The flax `out` DenseGeneral contracts over (num_heads, head_dim) which are
    the *input* axes here. torch weight is [out, in=d]; transpose to [in, out]
    then split the input axis into (n_heads, head_dim).
    """
    w_oi = np.asarray(w_oi, dtype=np.float32)
    d_out, d_in = w_oi.shape  # both == d
    head_dim = d_in // n_heads
    return w_oi.T.reshape(n_heads, head_dim, d_out).copy()


def _upgrade_scalar_value_head(sd: dict) -> dict:
    """Upgrade a legacy scalar value head ([1, d]) to the WDL head ([3, d]).

    Mirrors ``prophet/model.py:upgrade_state``: old head ends in Linear(d, 1);
    the WDL head is Linear(d, 3) ordered (loss, draw, win). Setting
    win-row = 1.5*w, loss-row = -1.5*w, draw-row = 0 makes P(win)-P(loss) track
    the old scalar value at warm start. No-op if the head is already 3-wide.
    """
    w = sd.get("v_head.2.weight")
    if w is not None and np.asarray(w).shape[0] == 1:
        sd = dict(sd)
        w = 1.5 * np.asarray(w, dtype=np.float32)
        b = 1.5 * np.asarray(sd["v_head.2.bias"], dtype=np.float32)
        zero_w = np.zeros_like(w)
        zero_b = np.zeros_like(b)
        sd["v_head.2.weight"] = np.concatenate([-w, zero_w, w], axis=0)
        sd["v_head.2.bias"] = np.concatenate([-b, zero_b, b], axis=0)
    return sd


def _widen_input(sd: dict, new_features: int) -> dict:
    """Zero-pad ``embed.weight`` [d, in] up to ``new_features`` input columns.

    Mirrors ``prophet/model.py:widen_input`` so pre-history (18-feature)
    checkpoints map onto the current 24-feature encoder with the extra senses
    zero-initialized (network function unchanged until trained).
    """
    w = np.asarray(sd["embed.weight"], dtype=np.float32)
    d_out, in_features = w.shape
    if in_features == new_features:
        return sd
    sd = dict(sd)
    grown = np.zeros((d_out, new_features), dtype=np.float32)
    grown[:, :in_features] = w
    sd["embed.weight"] = grown
    return sd


def _state_dict_to_params(sd: dict, cfg: ModelConfig) -> dict:
    """Map a torch state_dict (numpy arrays) -> a Flax param pytree.

    The returned pytree's structure EXACTLY matches what ``build_model`` /
    ``model.init`` produce (module names ``embed``, ``layers_{i}``, ``norm``,
    ``p_from`` ... ``v_head_0`` / ``v_head_2``, and the MHDPA sub-modules
    ``self_attn/{query,key,value,out}``).
    """
    h = cfg.n_heads

    def kv(name):
        return np.asarray(sd[name], dtype=np.float32)

    params: dict = {}

    # input embedding (Linear with bias).
    params["embed"] = {"kernel": _dense_kernel(kv("embed.weight")), "bias": kv("embed.bias")}

    # learned positional parameter [1, 64, d_model] (stored as-is).
    params["pos"] = kv("pos")

    # encoder blocks.
    for i in range(cfg.n_layers):
        pre = f"trunk.layers.{i}."
        in_w = kv(pre + "self_attn.in_proj_weight")  # [3d, d]
        in_b = kv(pre + "self_attn.in_proj_bias")  # [3d]
        d = in_w.shape[1]
        qw, kw, vw = in_w[0:d], in_w[d : 2 * d], in_w[2 * d : 3 * d]
        qb, kb, vb = in_b[0:d], in_b[d : 2 * d], in_b[2 * d : 3 * d]
        out_w = kv(pre + "self_attn.out_proj.weight")  # [d, d]
        out_b = kv(pre + "self_attn.out_proj.bias")  # [d]

        block = {
            "norm1": {"scale": kv(pre + "norm1.weight"), "bias": kv(pre + "norm1.bias")},
            "norm2": {"scale": kv(pre + "norm2.weight"), "bias": kv(pre + "norm2.bias")},
            "self_attn": {
                "query": {"kernel": _attn_qkv_kernel(qw, h), "bias": _attn_qkv_bias(qb, h)},
                "key": {"kernel": _attn_qkv_kernel(kw, h), "bias": _attn_qkv_bias(kb, h)},
                "value": {"kernel": _attn_qkv_kernel(vw, h), "bias": _attn_qkv_bias(vb, h)},
                "out": {"kernel": _attn_out_kernel(out_w, h), "bias": out_b},
            },
            "linear1": {"kernel": _dense_kernel(kv(pre + "linear1.weight")), "bias": kv(pre + "linear1.bias")},
            "linear2": {"kernel": _dense_kernel(kv(pre + "linear2.weight")), "bias": kv(pre + "linear2.bias")},
        }
        params[f"layers_{i}"] = block

    # final LayerNorm.
    params["norm"] = {"scale": kv("norm.weight"), "bias": kv("norm.bias")}

    # policy / Q from-to projections (each Linear(d -> head_dim)).
    for name in ("p_from", "p_to", "q_from", "q_to"):
        params[name] = {
            "kernel": _dense_kernel(kv(name + ".weight")),
            "bias": kv(name + ".bias"),
        }

    # V / WDL head: Linear(d, d) [index 0] -> GELU -> Linear(d, 3) [index 2].
    params["v_head_0"] = {
        "kernel": _dense_kernel(kv("v_head.0.weight")),
        "bias": kv("v_head.0.bias"),
    }
    params["v_head_2"] = {
        "kernel": _dense_kernel(kv("v_head.2.weight")),
        "bias": kv("v_head.2.bias"),
    }

    return jax.tree_util.tree_map(lambda a: jnp.asarray(a), params)


def load_torch_checkpoint(path: str):
    """Read a torch ``.pt`` checkpoint and return ``(cfg, params)``.

    Reads the checkpoint OUTSIDE JAX with ``torch.load(weights_only=True,
    map_location='cpu')`` and converts every tensor to numpy. The config is
    pulled from ``obj['config']`` (falling back to ``ModelConfig`` defaults for
    a bare v1 state dict). ``in_features`` is read off ``embed.weight`` so the
    port is fully config-driven and never hardcodes the input width.

    Legacy migrations (matching the torch loader) are applied so older
    checkpoints transfer: scalar value head -> WDL head, and 18-feature input
    -> 24-feature zero-padded embedding.
    """
    import torch  # imported lazily; torch is the checkpoint reader, not a JAX dep

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state" in obj:
        cfg_kwargs = dict(obj.get("config", {}))
        raw_sd = obj["state"]
    else:
        cfg_kwargs, raw_sd = {}, obj  # bare v1 state dict

    # tensors -> numpy (fp32) for framework-agnostic handling.
    sd = {k: v.detach().cpu().float().numpy() for k, v in raw_sd.items()}

    # config is checkpoint-driven; in_features comes from the embedding matrix.
    cfg_kwargs.pop("in_features", None)
    cfg = ModelConfig(in_features=int(sd["embed.weight"].shape[1]), **cfg_kwargs)

    # legacy upgrades (no-ops on current-format checkpoints).
    sd = _upgrade_scalar_value_head(sd)
    sd = _widen_input(sd, FEATURES)
    cfg = ModelConfig(**{**asdict(cfg), "in_features": int(sd["embed.weight"].shape[1])})

    params = _state_dict_to_params(sd, cfg)
    return cfg, params


# ---------------------------------------------------------------------------
# Native (JAX-side) checkpoint save/load.
# ---------------------------------------------------------------------------


def _flatten_params(params, prefix=""):
    """Flatten a nested param pytree of arrays into a flat ``{path: array}``."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key + "/"))
        else:
            flat[key] = np.asarray(v)
    return flat


def _unflatten_params(flat):
    """Inverse of ``_flatten_params``: ``{path: array}`` -> nested pytree."""
    out: dict = {}
    for key, v in flat.items():
        parts = key.split("/")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(v)
    return out


def save_checkpoint(params, cfg: ModelConfig, path: str) -> None:
    """Write a native JAX checkpoint as ``{'config': ..., 'params': ...}``.

    Stored via ``np.savez`` with the flax param pytree flattened to numpy
    arrays under ``param/<path>`` keys, plus a JSON-encoded config. Atomic
    (write to a temp file then ``os.replace``). Use ``load_checkpoint`` to read
    it back.
    """
    import json

    flat = _flatten_params(params)
    arrays = {f"param/{k}": v for k, v in flat.items()}
    arrays["__config__"] = np.frombuffer(json.dumps(asdict(cfg)).encode("utf-8"), dtype=np.uint8)
    tmp = str(path) + ".tmp.npz"
    final = str(path) if str(path).endswith(".npz") else str(path) + ".npz"
    np.savez(tmp, **arrays)
    # np.savez may append .npz to the temp name; normalize before replacing.
    written = tmp if os.path.exists(tmp) else tmp + ".npz"
    os.replace(written, final)


def load_checkpoint(path: str):
    """Read a native JAX checkpoint written by ``save_checkpoint``.

    Returns ``(cfg, params)``.
    """
    import json

    p = str(path)
    if not os.path.exists(p) and os.path.exists(p + ".npz"):
        p = p + ".npz"
    data = np.load(p, allow_pickle=False)
    cfg = ModelConfig(**json.loads(bytes(data["__config__"]).decode("utf-8")))
    flat = {k[len("param/") :]: data[k] for k in data.files if k.startswith("param/")}
    params = _unflatten_params(flat)
    return cfg, params


# ---------------------------------------------------------------------------
# Flax -> torch exporter (reverse the key map) so JAX-trained nets can be
# gauntleted by the existing torch scripts.
# ---------------------------------------------------------------------------


def _kernel_to_dense_weight(kernel: np.ndarray) -> np.ndarray:
    """flax Dense kernel [in, out] -> torch Linear weight [out, in] = kernel.T."""
    return np.asarray(kernel, dtype=np.float32).T.copy()


def _params_to_state_dict(params, cfg: ModelConfig) -> dict:
    """Map a Flax param pytree -> a torch-compatible state_dict (numpy arrays).

    Exact inverse of ``_state_dict_to_params``; re-packs the four MHDPA Dense
    sub-modules back into torch's stacked ``in_proj_weight``/``in_proj_bias`` +
    ``out_proj``.
    """
    g = jax.tree_util.tree_map(lambda a: np.asarray(a, dtype=np.float32), params)
    sd: dict = {}

    sd["embed.weight"] = _kernel_to_dense_weight(g["embed"]["kernel"])
    sd["embed.bias"] = g["embed"]["bias"]
    sd["pos"] = g["pos"]

    for i in range(cfg.n_layers):
        blk = g[f"layers_{i}"]
        pre = f"trunk.layers.{i}."
        attn = blk["self_attn"]

        # query/key/value kernels [in, n_heads, head_dim] -> torch [out=d, in=d].
        def qkv_w(sub):
            k = attn[sub]["kernel"]  # [in, n_heads, head_dim]
            d_in, n_heads, head_dim = k.shape
            return k.reshape(d_in, n_heads * head_dim).T.copy()  # [d, d]

        def qkv_b(sub):
            b = attn[sub]["bias"]  # [n_heads, head_dim]
            return b.reshape(-1).copy()  # [d]

        sd[pre + "self_attn.in_proj_weight"] = np.concatenate(
            [qkv_w("query"), qkv_w("key"), qkv_w("value")], axis=0
        )  # [3d, d]
        sd[pre + "self_attn.in_proj_bias"] = np.concatenate(
            [qkv_b("query"), qkv_b("key"), qkv_b("value")], axis=0
        )  # [3d]

        # out kernel [n_heads, head_dim, out=d] -> torch out_proj.weight [out=d, in=d].
        ok = attn["out"]["kernel"]
        n_heads, head_dim, d_out = ok.shape
        sd[pre + "self_attn.out_proj.weight"] = ok.reshape(n_heads * head_dim, d_out).T.copy()
        sd[pre + "self_attn.out_proj.bias"] = attn["out"]["bias"]

        sd[pre + "linear1.weight"] = _kernel_to_dense_weight(blk["linear1"]["kernel"])
        sd[pre + "linear1.bias"] = blk["linear1"]["bias"]
        sd[pre + "linear2.weight"] = _kernel_to_dense_weight(blk["linear2"]["kernel"])
        sd[pre + "linear2.bias"] = blk["linear2"]["bias"]
        sd[pre + "norm1.weight"] = blk["norm1"]["scale"]
        sd[pre + "norm1.bias"] = blk["norm1"]["bias"]
        sd[pre + "norm2.weight"] = blk["norm2"]["scale"]
        sd[pre + "norm2.bias"] = blk["norm2"]["bias"]

    sd["norm.weight"] = g["norm"]["scale"]
    sd["norm.bias"] = g["norm"]["bias"]

    for name in ("p_from", "p_to", "q_from", "q_to"):
        sd[name + ".weight"] = _kernel_to_dense_weight(g[name]["kernel"])
        sd[name + ".bias"] = g[name]["bias"]

    sd["v_head.0.weight"] = _kernel_to_dense_weight(g["v_head_0"]["kernel"])
    sd["v_head.0.bias"] = g["v_head_0"]["bias"]
    sd["v_head.2.weight"] = _kernel_to_dense_weight(g["v_head_2"]["kernel"])
    sd["v_head.2.bias"] = g["v_head_2"]["bias"]

    return sd


def export_torch_checkpoint(params, cfg: ModelConfig, path: str) -> None:
    """Write a torch ``.pt`` checkpoint from Flax params for the torch gauntlet.

    Produces ``{'config': asdict(cfg), 'state': <torch state_dict>}`` matching
    ``prophet/model.py:save_checkpoint`` (atomic tmp + ``os.replace``), so the
    existing torch ``load_checkpoint`` reads it with no changes. Note ``config``
    omits ``in_features`` (the torch loader derives it from ``embed.weight``),
    matching the torch checkpoint format.
    """
    import torch

    sd_np = _params_to_state_dict(params, cfg)
    state = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in sd_np.items()}
    cfg_dict = {k: val for k, val in asdict(cfg).items() if k != "in_features"}
    obj = {"config": cfg_dict, "state": state}
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))
