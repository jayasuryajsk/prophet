"""JAX learner: the five training losses, an optax update step, and the
single-process self-play -> reflect -> train -> checkpoint main loop.

This is the JAX port of prophet/train.py (the losses) plus the orchestration
that scripts/train_loop.py performs in the PyTorch implementation. Because
self-play, reflection, and training all run on-accelerator in JAX, there are
NO worker processes: the learner IS the whole loop. The multi-process
worker fan-out (N workers x batch_games concurrent games) is replaced by a
single large batch ``B`` in :func:`prophet_jax.selfplay.generate_selfplay` —
thousands of parallel games == a bigger ``B``, evaluated in one batched
forward via pgx's vmapped env + mctx's batched search.

The five loss terms are ported VERBATIM from prophet/train.py.train_step:

  1. POLICY cross-entropy over legal moves (illegal -> -inf before softmax).
  2. VALUE MSE on the blended scalar v = P(win) - P(loss).
  3. Per-move Q regression (visit-weighted), with the load-bearing
     ``clamp_min(1.0)`` denominator and ``q_scale = 0.5 + |value|``
     anti-draw-flattening weighting.
  4. WDL cross-entropy on known terminal outcome classes (-1 excluded).
  5. Negamax consistency: Q(s, a_played) ~= -stopgrad(V(child)), where the
     child forward uses the SAME params with a stop-gradient.

Sample weights are mean-normalized per batch (``wn``) and applied to all five
terms, exactly as in the reference. The optimizer matches train_loop.py:
optax.adamw(lr=3e-4, weight_decay=1e-4) with global-norm gradient clipping at
1.0 (matching torch ``clip_grad_norm_(.., 1.0)``), and an EMA of the params at
decay 0.999 that is what gets checkpointed / exported / used for eval.

NOTE: jax / flax / optax / pgx / mctx are not installed in this repo; this
module is written against the verified library docs (see the project memory
and the module plan). It imports its sibling JAX modules lazily inside
:func:`main` so that the loss / train_step path can be imported and tested in
isolation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
import optax

# Sibling JAX modules. config.py is pure-python (dataclasses + schedule
# helpers) and safe to import at module load; the heavier modules (model /
# env / search / selfplay / reflection) are imported lazily inside main() so
# that importing this module to use train_step / the losses does not require
# the full pgx+mctx stack to be present.
from .config import (  # noqa: E402  (kept with the JAX imports above)
    NUM_ACTIONS,
    FEATURES,
    LossWeights,
    SearchConfig,
    SelfPlayConfig,
    StudyConfig,
    ModelConfig,
    loss_weights_at,
    q_trust_at,
    study_config_at,
)

A = NUM_ACTIONS  # 4096
F = FEATURES     # 24


# ---------------------------------------------------------------------------
# Dense batch / sample container (the training contract shared with selfplay
# and reflection). The CANONICAL definition lives in selfplay.py as a
# flax.struct.dataclass; we import it here (single source of truth) so the
# replay buffer and train_step consume exactly the same pytree that
# generate_selfplay / reflect_batch produce -- no type drift. (Importing
# selfplay needs only jax + flax, both already imported above; the heavier pgx/
# mctx-using helpers in selfplay are imported lazily there, so this stays cheap.)
#
# Shapes (N = number of samples in the batch):
#   x:        f32[N, 64, F]   position, side-to-move perspective
#   child_x:  f32[N, 64, F]   position after the played move
#   played:   i32[N]          played action index (from*64 + to)
#   value:    f32[N]          blended search/outcome value target
#   weight:   f32[N]          per-sample loss weight (study/branch > 1)
#   wdl:      i32[N]          outcome class 0/1/2, or -1 = unknown
#   mask:     bool[N, A]      legal-move mask
#   policy:   f32[N, A]       improved policy target (dense, 0 off-legal)
#   q_target: f32[N, A]       empirical search Q scattered to visited moves
#   q_weight: f32[N, A]       visit counts scattered to visited moves
#   valid:    bool[N]         True for real samples, False for padding
# ---------------------------------------------------------------------------
from .selfplay import SamplesBatch  # noqa: E402  canonical flax.struct dataclass


# Field dtypes used by the host-side replay ring to allocate its arrays. Kept
# next to SamplesBatch so the two never drift.
_SAMPLE_SPEC = {
    "x": (np.float32, (64, F)),
    "child_x": (np.float32, (64, F)),
    "played": (np.int32, ()),
    "value": (np.float32, ()),
    "weight": (np.float32, ()),
    "wdl": (np.int32, ()),
    "mask": (np.bool_, (A,)),
    "policy": (np.float32, (A,)),
    "q_target": (np.float32, (A,)),
    "q_weight": (np.float32, (A,)),
    "valid": (np.bool_, ()),
}


# ---------------------------------------------------------------------------
# Loss weights as a tiny JAX pytree of scalars. We pass loss weights INTO the
# jitted train_step as arrays (not Python floats) so changing the schedule
# band does NOT retrace the step — only the scalar values change. The
# schedule's LossWeights dataclass (Python floats) is converted with
# loss_weights_to_array() at the call site.
# ---------------------------------------------------------------------------
class LossWeightArr(NamedTuple):
    """The five loss weights as device scalars (a JAX pytree)."""

    policy: jnp.ndarray
    value: jnp.ndarray
    q: jnp.ndarray
    consistency: jnp.ndarray
    wdl: jnp.ndarray


def loss_weights_to_array(w: LossWeights) -> LossWeightArr:
    """Convert the (Python-float) schedule LossWeights into device scalars so
    the jitted train_step can take them as a traced pytree argument (no
    retrace on band change)."""
    return LossWeightArr(
        policy=jnp.asarray(w.policy, jnp.float32),
        value=jnp.asarray(w.value, jnp.float32),
        q=jnp.asarray(w.q, jnp.float32),
        consistency=jnp.asarray(w.consistency, jnp.float32),
        wdl=jnp.asarray(w.wdl, jnp.float32),
    )


# ---------------------------------------------------------------------------
# Train state: params, EMA params, optimizer state, a step counter, and the
# (static) optax transform + a reference to the model.
#
# IMPORTANT pytree split: the array leaves (params / ema_params / opt_state /
# step) are TRACED children; the model module, the optax transform, and the
# ema_decay float are STATIC aux. ``step`` is a TRACED jnp scalar (not a Python
# int) and is incremented INSIDE the jitted step — putting an ever-changing
# value in the static aux instead would re-key the jit cache and recompile the
# step every single iteration, so it deliberately lives in ``children``.
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class TrainState:
    """Mutable training state threaded through the (jitted) train_step.

    Array leaves (traced): params, ema_params, opt_state, step.
    Static aux (not traced, part of the jit cache key): model, tx, ema_decay.
    """

    params: Any
    ema_params: Any
    opt_state: Any
    step: Any = field(default=None)           # traced jnp.int32 scalar leaf
    model: Any = field(default=None)          # flax linen module (static)
    tx: optax.GradientTransformation = field(default=None)  # static
    ema_decay: float = 0.999                   # static float

    def tree_flatten(self):
        # step is a traced leaf so it can increment without recompiling.
        children = (self.params, self.ema_params, self.opt_state, self.step)
        aux = (self.model, self.tx, self.ema_decay)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        params, ema_params, opt_state, step = children
        model, tx, ema_decay = aux
        return cls(
            params=params,
            ema_params=ema_params,
            opt_state=opt_state,
            step=step,
            model=model,
            tx=tx,
            ema_decay=ema_decay,
        )


def make_optimizer(lr: float, wd: float) -> optax.GradientTransformation:
    """optax chain matching train_loop.py: clip global grad norm to 1.0
    (== torch clip_grad_norm_(params, 1.0)) THEN adamw(lr, weight_decay).

    Clipping is chained BEFORE adamw so the clipped gradient is what adamw's
    moments and the decoupled weight-decay update see — the same order as
    ``clip_grad_norm_`` followed by ``optimizer.step()`` in PyTorch.
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=wd),
    )


def make_train_state(
    model,
    params,
    lr: float = 3e-4,
    wd: float = 1e-4,
    ema_decay: float = 0.999,
) -> TrainState:
    """Build a TrainState: optimizer (adamw + grad clip), an EMA copy of the
    params (used for eval/checkpoint/export), and the optimizer state.

    Matches train_loop.py defaults: lr=3e-4, weight_decay=1e-4, ema=0.999.
    """
    tx = make_optimizer(lr, wd)
    opt_state = tx.init(params)
    # EMA starts as an exact copy of the initial params (== ema_model =
    # copy.deepcopy(model) in train_loop.py).
    ema_params = jax.tree_util.tree_map(lambda a: a, params)
    return TrainState(
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
        step=jnp.asarray(0, jnp.int32),  # traced leaf, incremented in the step
        model=model,
        tx=tx,
        ema_decay=ema_decay,
    )


# ---------------------------------------------------------------------------
# THE LOSS / TRAIN STEP. ``model`` and ``tx`` ride inside ``state`` as static
# aux, so the whole thing is jittable with a single positional jit. We expose
# both an un-jitted ``_train_step_impl`` (for testing) and the jitted
# ``train_step``.
# ---------------------------------------------------------------------------
def _train_step_impl(state: TrainState, batch: SamplesBatch, weights: LossWeightArr):
    """One optimizer step. Returns (new_state, metrics).

    Ports prophet/train.py.train_step verbatim:
      logits, q, v, wdl_probs = model.forward_wdl(params, batch.x)
      wn = weight / weight.mean().clamp_min(1e-8)
    then the five terms below. The model's forward is invoked through the
    sibling model module's functional ``forward_wdl(model, params, x)``.
    """
    from .model import forward_wdl, forward  # lazy: keeps loss path importable

    model = state.model

    # ``wn`` mean-normalizes the per-sample weights across the batch, exactly
    # as prophet/train.py: wn = weight / weight.mean().clamp_min(1e-8). Batches
    # drawn from the replay ring are all-valid (padding is dropped at add()
    # time), so the .mean() denominator is the true batch size — matching the
    # reference's padding-free collate. Computed outside loss_fn since it does
    # not depend on params.
    weight = batch.weight
    wn = weight / jnp.maximum(weight.mean(), 1e-8)

    def loss_fn(params):
        logits, q, v, wdl_probs = forward_wdl(model, params, batch.x)

        # (1) POLICY CROSS-ENTROPY — softmax over LEGAL moves only.
        masked = jnp.where(batch.mask, logits, -jnp.inf)
        logp = jax.nn.log_softmax(masked, axis=-1)
        # zero out illegal logp so the -inf positions never enter the sum
        # (jnp.where avoids 0 * -inf = nan).
        logp = jnp.where(batch.mask, logp, 0.0)
        loss_pi = jnp.mean(wn * -(batch.policy * logp).sum(axis=-1))

        # (2) VALUE MSE.
        loss_v = jnp.mean(wn * (v - batch.value) ** 2)

        # (3) PER-MOVE Q REGRESSION (visit-weighted over visited children).
        qw = batch.q_weight
        per_q = ((q - batch.q_target) ** 2 * qw).sum(axis=-1) / jnp.maximum(
            qw.sum(axis=-1), 1.0
        )
        # weight Q toward DECISIVE positions so late-training drawish targets
        # don't flatten the Q-head (the v1 Q-regression-to-draw fix). BOTH the
        # clamp_min(1.0) above and this q_scale are load-bearing.
        q_scale = 0.5 + jnp.abs(batch.value)
        loss_q = jnp.mean(wn * q_scale * per_q)

        # (4) WDL CROSS-ENTROPY — only samples with a known terminal class.
        has_wdl = batch.wdl >= 0
        has_wdl_f = has_wdl.astype(jnp.float32)  # 0/1 weights, mirrors torch bool*float
        logp_wdl = jnp.log(jnp.maximum(wdl_probs, 1e-8))
        # gather the log-prob of the true class; clamp the index lower bound to
        # 0 so the -1 "unknown" rows stay in range for the gather (they are
        # masked out by has_wdl anyway, exactly as torch's wdl.clamp_min(0)
        # .gather does). jnp.maximum(.., 0) is used instead of jnp.clip(.., a_min=
        # 0) because recent JAX (>=0.4.34, the requirements floor) deprecated/
        # removed the a_min/a_max kwargs in favour of min/max; jnp.maximum is
        # unambiguous across all versions.
        wdl_idx = jnp.maximum(batch.wdl, 0)
        nll = -jnp.take_along_axis(logp_wdl, wdl_idx[:, None], axis=1)[:, 0]
        loss_wdl = jnp.where(
            has_wdl.any(),
            (wn * nll * has_wdl_f).sum() / jnp.maximum(has_wdl_f.sum(), 1.0),
            0.0,
        )

        # (5) NEGAMAX CONSISTENCY — Q(s, a_played) ~= -stopgrad(V(child)).
        # The child forward shares the SAME params; stop-grad on V(child).
        v_child = jax.lax.stop_gradient(forward(model, params, batch.child_x)[2])
        q_played = jnp.take_along_axis(q, batch.played[:, None], axis=1)[:, 0]
        loss_cons = jnp.mean(wn * (q_played + v_child) ** 2)

        total = (
            weights.policy * loss_pi
            + weights.value * loss_v
            + weights.q * loss_q
            + weights.consistency * loss_cons
            + weights.wdl * loss_wdl
        )
        aux = {
            "loss": total,
            "policy": loss_pi,
            "value": loss_v,
            "q": loss_q,
            "consistency": loss_cons,
            "wdl": loss_wdl,
        }
        return total, aux

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Clip (global norm 1.0) + adamw live inside state.tx; updates are the
    # already-negated steps to add to params.
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    # EMA AFTER the optimizer step (decay 0.999 -> step_size 0.001). The EMA
    # params are what checkpoints / export / eval use.
    ema_params = optax.incremental_update(
        params, state.ema_params, step_size=1.0 - state.ema_decay
    )

    new_state = TrainState(
        params=params,
        ema_params=ema_params,
        opt_state=opt_state,
        step=state.step + 1,  # traced increment; no recompile per step
        model=state.model,
        tx=state.tx,
        ema_decay=state.ema_decay,
    )
    return new_state, metrics


# jit the step. ``state`` carries the static model/tx as pytree aux, so a
# plain jit (no static_argnums) specializes correctly; passing loss weights as
# a traced array pytree means schedule-band changes do NOT retrace.
train_step = jax.jit(_train_step_impl)


# ---------------------------------------------------------------------------
# Host-side replay buffer: a ring of dense numpy arrays (capacity ~200k
# samples). add() appends the dense fields of a SamplesBatch (numpy);
# sample_batch() draws ``batch_size`` rows uniformly and returns a dense
# SamplesBatch of jnp arrays. ``collate`` is unnecessary — selfplay /
# reflection already emit dense batched arrays, so we just slice the ring.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity ring buffer of dense training samples, on the host.

    Stores each field of :class:`SamplesBatch` in a pre-allocated numpy array
    of shape (capacity, *field_shape). Insertion is O(rows) memcopy; sampling
    is a fancy-index gather. Only rows that were actually written are eligible
    to be drawn (``self.size`` grows to capacity then stays).
    """

    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0
        self._buf: dict[str, np.ndarray] = {
            name: np.zeros((self.capacity, *shape), dtype=dtype)
            for name, (dtype, shape) in _SAMPLE_SPEC.items()
        }

    def __len__(self) -> int:
        return self.size

    def add(self, samples: SamplesBatch) -> int:
        """Append the valid rows of a dense SamplesBatch (jnp or numpy). Rows
        with valid==False (padding) are dropped before insertion. Returns the
        number of rows actually added."""
        valid = np.asarray(samples.valid)
        keep = np.flatnonzero(valid)
        n = int(keep.size)
        if n == 0:
            return 0
        # A large A100 self-play round can emit more valid rows than the replay
        # ring can retain. Keep only the suffix that would survive the wrap
        # anyway, before pulling the dense policy/Q tensors back to host.
        if n > self.capacity:
            keep = keep[-self.capacity:]
            n = self.capacity
        # Pull each field to host numpy once, keep only valid rows.
        fields = {
            name: np.asarray(getattr(samples, name))[keep]
            for name in _SAMPLE_SPEC
        }
        # Write into the ring, wrapping at capacity.
        start = self.pos
        idx = (start + np.arange(n)) % self.capacity
        for name, arr in fields.items():
            self._buf[name][idx] = arr.astype(self._buf[name].dtype, copy=False)
        self.pos = int((start + n) % self.capacity)
        self.size = int(min(self.capacity, self.size + n))
        return n

    def sample(self, rng: np.random.Generator, batch_size: int) -> SamplesBatch:
        """Draw ``batch_size`` rows uniformly with replacement -> dense
        SamplesBatch of jnp arrays. Stored rows are all real, so valid is all
        True for a sampled batch."""
        idx = rng.integers(self.size, size=batch_size)
        out = {name: jnp.asarray(self._buf[name][idx]) for name in _SAMPLE_SPEC}
        # everything drawn from the ring is a real sample
        out["valid"] = jnp.ones((batch_size,), dtype=jnp.bool_)
        return SamplesBatch(**out)


def sample_batch(buffer: ReplayBuffer, key, batch_size: int) -> SamplesBatch:
    """Draw a dense replay batch. ``key`` is a JAX PRNGKey; we fold it into a
    numpy Generator so the host-side ring gather is reproducible alongside the
    jitted train_step (the gather itself is host code, not jit-traceable)."""
    # VERIFY: bridging a JAX PRNGKey -> a host numpy Generator. We materialize
    # one uniform int32 from the key (jax.random.randint(key, shape=(), low,
    # high)) and seed default_rng with it; this keeps replay sampling driven by
    # the master key while the ring gather stays on the host.
    seed = int(jax.random.randint(key, (), 0, np.int32(2**31 - 1)))
    rng = np.random.default_rng(seed)
    return buffer.sample(rng, batch_size)


def q_target_stats(samples: SamplesBatch) -> tuple[float, float]:
    """Host-side Q-target diagnostics for a dense sample batch.

    Returns ``(mean_abs_visited_q, mean_root_visit_count)`` over valid rows.
    These numbers are for logging only; they help distinguish a genuinely
    learned small Q loss from a collapsed target distribution.
    """
    valid = np.asarray(samples.valid).astype(bool)
    if not np.any(valid):
        return 0.0, 0.0
    q_weight = np.asarray(samples.q_weight)[valid]
    q_target = np.asarray(samples.q_target)[valid]
    visit_sum = q_weight.sum(axis=1)
    total_weight = float(q_weight.sum())
    mean_abs_q = (
        float((np.abs(q_target) * q_weight).sum() / max(total_weight, 1.0))
        if total_weight > 0.0
        else 0.0
    )
    mean_visits = float(visit_sum.mean()) if visit_sum.size else 0.0
    return mean_abs_q, mean_visits


def q_head_played_abs(meta) -> float:
    """Mean ``abs(q_head_played)`` over real self-play plies in ``GameMeta``."""
    valid = np.asarray(meta.valid_ply).astype(bool)
    if not np.any(valid):
        return 0.0
    qhp = np.asarray(meta.q_head_played)[valid]
    return float(np.abs(qhp).mean()) if qhp.size else 0.0


# ---------------------------------------------------------------------------
# Checkpoint IO helpers (thin wrappers around the model module so the loop
# reads naturally). The native save/load and the torch exporter all live in
# model.py; we re-expose them here for the loop and degrade gracefully if a
# given entry point is missing.
# ---------------------------------------------------------------------------
def save_checkpoint(params, cfg: ModelConfig, path) -> None:
    """Native (npz/msgpack) checkpoint of (EMA) params + config via model.py."""
    from . import model as model_mod

    model_mod.save_checkpoint(params, cfg, str(path))


def export_torch_checkpoint(params, cfg: ModelConfig, path) -> None:
    """Write a PyTorch-format ``{'config': asdict(cfg), 'state': state_dict}``
    checkpoint so the existing scripts/gauntlet.py (which calls
    prophet.model.load_checkpoint) can rate the JAX net. Delegates to
    model.export_torch_checkpoint."""
    from . import model as model_mod

    model_mod.export_torch_checkpoint(params, cfg, str(path))


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic text write (tmp + os.replace) — matches train_loop's progress
    file semantics so a concurrent reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# THE LEARNER MAIN LOOP. Self-play, reflection, and training all run in this
# single process (no workers). Mirrors train_loop.py's defaults and ordering.
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prophet JAX learner (single-process).")
    ap.add_argument("--games", type=int, default=100_000)
    # B = parallel self-play games per round (replaces workers x batch_games).
    ap.add_argument("--batch-games", type=int, default=1024,
                    help="parallel self-play games per round (B); thousands of "
                         "games == a bigger B, replacing the worker fan-out")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=160)
    # Model size: default to the production "moonshot" 10M band; the port is
    # config-driven and reads cfg back from a loaded checkpoint.
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-ratio", type=float, default=4.0)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ema", type=float, default=0.999,
                    help="weight EMA decay; checkpoints/exports/eval use the EMA")
    ap.add_argument("--sync-every", type=int, default=25, help="games between checkpoint syncs")
    ap.add_argument("--gate", type=int, default=2000, help="games before study/resign turn on")
    ap.add_argument("--eval-every", type=int, default=500,
                    help="games between milestone checkpoints + metrics rows")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/jax_run")
    ap.add_argument("--study", action="store_true", help="enable study-your-losses (after gate)")
    ap.add_argument("--schedule", action="store_true",
                    help="game-count curricula for study/q-trust/q-loss (moonshot)")
    ap.add_argument("--start-game", type=int, default=0,
                    help="resume schedule/gate/counter at this game # (warm restart)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a (torch or native) checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-export-torch", action="store_true",
                    help="skip the torch-format export at each milestone "
                         "(export lets scripts/gauntlet.py rate the JAX net)")
    return ap


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """Run the JAX learner loop.

    Per round:
      (a) games_done -> pick the STATIC schedule bands: scfg via
          replace(SearchConfig, q_trust=q_trust_at(games_done)), study cfg via
          study_config_at, and loss weights via loss_weights_at. Because these
          select discrete bands, the jitted search/study re-specialize only at
          band boundaries.
      (b) gate = games_done >= gate_games. generate_selfplay(EMA-or-live
          params, key, B, scfg, spcfg, gate) -> (samples, meta); add to the
          replay buffer; games_done += B.
      (c) if gate: reflect_batch(...) -> extra study/branch samples; add them.
      (d) if len(buffer) >= warmup: run round(new_samples * train_ratio /
          batch) train_steps, each drawing a dense batch and updating
          params + EMA, with EMA-tracked metrics.
      (e) every sync_every games: save the EMA checkpoint (native, "latest")
          and write progress.json. Every eval_every games: save a milestone
          ckpt_{games:06d}, export a torch checkpoint for the gauntlet, and
          append a metrics.csv row.
    """
    # ---- lazy imports of the heavy JAX siblings (need pgx + mctx + flax) ----
    from . import model as model_mod
    from .env import make_chess_env  # noqa: F401  (constructed for parity / warmup)
    from .selfplay import generate_selfplay
    from .reflection import reflect_batch

    if args is None or isinstance(args, list):
        args = _build_argparser().parse_args(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest.npz"           # native EMA checkpoint
    latest_torch = out / "latest.pt"           # torch export for the gauntlet
    gate_path = out / "gate_on"
    progress_path = out / "progress.json"
    metrics_path = out / "metrics.csv"

    def write_progress(games: int) -> None:
        # mirror train_loop.write_progress: atomic {"games": N}
        _atomic_write_text(progress_path, json.dumps({"games": int(games)}))

    # ---- build or load the model (config-driven) --------------------------
    master_key = jax.random.PRNGKey(args.seed)
    master_key, init_key = jax.random.split(master_key)

    if args.init_from:
        # load_torch_checkpoint reads cfg from the checkpoint's "config" key,
        # so the JAX net is sized to match the loaded weights (NOT hardcoded).
        cfg, params = model_mod.load_torch_checkpoint(args.init_from)
        model = model_mod.build_model(cfg, init_key)[0]
        print(f"warm start from {args.init_from} "
              f"({model_mod.num_params(params):,} params)", flush=True)
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=4 * args.d_model,
        )
        model, params = model_mod.build_model(cfg, init_key)
        print(f"fresh model ({model_mod.num_params(params):,} params): "
              f"d_model={cfg.d_model} n_layers={cfg.n_layers}", flush=True)

    state = make_train_state(model, params, lr=args.lr, wd=args.wd, ema_decay=args.ema)

    # ---- replay + counters ------------------------------------------------
    buffer = ReplayBuffer(args.buffer)
    games_done = args.start_game
    total_steps = 0

    # gate semantics: warm restart past the gate -> study on immediately.
    gated = games_done >= args.gate
    if args.study and gated:
        gate_path.touch()
    elif not gated:
        gate_path.unlink(missing_ok=True)

    write_progress(games_done)

    # EMA-tracked scalar loss metrics (host-side dict), like train_loop's ema.
    ema_metrics: dict[str, float] = {}
    # rolling decisive-rate / plies windows (last ~200 *rounds* worth).
    from collections import deque
    recent_plies: deque[float] = deque(maxlen=200)
    recent_decisive: deque[float] = deque(maxlen=200)
    recent_q_abs: deque[float] = deque(maxlen=200)
    recent_q_visits: deque[float] = deque(maxlen=200)
    recent_qhp_abs: deque[float] = deque(maxlen=200)
    recent_study_q_abs: deque[float] = deque(maxlen=200)
    recent_study_q_visits: deque[float] = deque(maxlen=200)
    recent_study_rows: deque[float] = deque(maxlen=200)

    base_scfg = SearchConfig(sims=args.sims, root_candidates=args.candidates)
    base_stcfg = StudyConfig()
    spcfg = SelfPlayConfig(max_plies=args.max_plies)

    # metrics.csv header (games, steps, buffer, plies, decisive_rate, losses).
    new_metrics = not metrics_path.exists()
    mf = open(metrics_path, "a", newline="")
    mw = csv.writer(mf)
    if new_metrics:
        mw.writerow([
            "games", "steps", "buffer", "avg_plies", "decisive_rate",
            "loss", "loss_pi", "loss_v", "loss_q", "loss_cons", "loss_wdl",
            "q_target_abs", "q_root_visits", "q_head_played_abs",
            "study_q_target_abs", "study_q_root_visits", "study_rows",
            "games_per_min",
        ])
        mf.flush()

    # save an initial EMA checkpoint (so a gauntlet can read it at game 0).
    save_checkpoint(state.ema_params, cfg, latest_path)
    if not args.no_export_torch:
        export_torch_checkpoint(state.ema_params, cfg, latest_torch)

    t0 = time.time()
    last_sync = games_done
    last_eval = games_done

    print(f"learner (single-process JAX): B={args.batch_games} games/round, "
          f"sims={args.sims}, gate@{args.gate}, study={'on' if args.study else 'off'}, "
          f"target {args.games} games", flush=True)

    try:
        while games_done < args.games:
            # ---------- (a) schedule bands (STATIC -> jit re-specializes only
            #               at band changes) ----------
            if args.schedule:
                scfg = replace(base_scfg, q_trust=q_trust_at(games_done))
                stcfg = study_config_at(games_done, base_stcfg)
                lw = loss_weights_at(games_done)
            else:
                scfg = base_scfg
                stcfg = base_stcfg
                lw = LossWeights()
            weights_arr = loss_weights_to_array(lw)

            # ---------- (b) gate + self-play ----------
            gate = games_done >= args.gate
            if gate and not gated:
                gated = True
                if args.study:
                    gate_path.touch()
                print(f"  GATE OPEN @{games_done}: study/resignation enabled", flush=True)

            B = args.batch_games
            master_key, sp_key = jax.random.split(master_key)
            # Self-play uses the EMA params (the checkpointed/eval net), matching
            # train_loop syncing the EMA model to the workers.
            samples, meta = generate_selfplay(
                state.ema_params, sp_key, B, scfg, spcfg, gate
            )
            q_abs, q_visits = q_target_stats(samples)
            recent_q_abs.append(q_abs)
            recent_q_visits.append(q_visits)
            recent_qhp_abs.append(q_head_played_abs(meta))
            added = buffer.add(samples)
            games_done += B

            # round-level stats from meta (plies per game; decisive = result
            # not a draw/unfinished). meta.z_white is nan for truncated games.
            plies = np.asarray(meta.plies)
            z_white = np.asarray(meta.z_white)
            recent_plies.append(float(plies.mean()))
            finished = ~np.isnan(z_white)
            decisive = (np.abs(np.nan_to_num(z_white)) > 0.5) & finished
            denom = max(1, int(finished.sum()))
            recent_decisive.append(float(decisive.sum()) / denom)
            new_samples = added

            # ---------- (c) reflection (gated) ----------
            if gate and args.study:
                master_key, rf_key = jax.random.split(master_key)
                study_samples = reflect_batch(
                    state.ema_params, rf_key, meta, meta.states_per_ply, stcfg, scfg
                )
                study_q_abs, study_q_visits = q_target_stats(study_samples)
                recent_study_q_abs.append(study_q_abs)
                recent_study_q_visits.append(study_q_visits)
                recent_study_rows.append(float(np.asarray(study_samples.valid).sum()))
                new_samples += buffer.add(study_samples)

            # ---------- (d) training ----------
            if len(buffer) >= args.warmup:
                steps = max(1, round(new_samples * args.train_ratio / args.batch))
                for _ in range(steps):
                    master_key, sk = jax.random.split(master_key)
                    batch = sample_batch(buffer, sk, args.batch)
                    state, metrics = train_step(state, batch, weights_arr)
                    total_steps += 1  # host-side mirror of state.step
                    # EMA-track the scalar metrics (0.99 like train_loop).
                    for k, v in metrics.items():
                        fv = float(v)
                        ema_metrics[k] = fv if k not in ema_metrics else (
                            0.99 * ema_metrics[k] + 0.01 * fv
                        )

            # ---------- (e) checkpoint / progress / metrics ----------
            if games_done - last_sync >= args.sync_every:
                save_checkpoint(state.ema_params, cfg, latest_path)
                write_progress(games_done)
                last_sync = games_done

            if games_done - last_eval >= args.eval_every or games_done >= args.games:
                save_checkpoint(state.ema_params, cfg, latest_path)
                save_checkpoint(state.ema_params, cfg, out / f"ckpt_{games_done:06d}.npz")
                # export a torch checkpoint so scripts/gauntlet.py can rate it.
                if not args.no_export_torch:
                    export_torch_checkpoint(state.ema_params, cfg, latest_torch)
                    export_torch_checkpoint(
                        state.ema_params, cfg, out / f"ckpt_{games_done:06d}.pt"
                    )
                gpm = games_done / max(1e-9, (time.time() - t0) / 60)
                mw.writerow(
                    [games_done, total_steps, len(buffer),
                     round(float(np.mean(recent_plies)) if recent_plies else 0.0, 1),
                     round(float(np.mean(recent_decisive)) if recent_decisive else 0.0, 3)]
                    + [round(ema_metrics.get(k, 0.0), 4)
                       for k in ("loss", "policy", "value", "q", "consistency", "wdl")]
                    + [
                        round(float(np.mean(recent_q_abs)) if recent_q_abs else 0.0, 4),
                        round(float(np.mean(recent_q_visits)) if recent_q_visits else 0.0, 2),
                        round(float(np.mean(recent_qhp_abs)) if recent_qhp_abs else 0.0, 4),
                        round(float(np.mean(recent_study_q_abs)) if recent_study_q_abs else 0.0, 4),
                        round(float(np.mean(recent_study_q_visits)) if recent_study_q_visits else 0.0, 2),
                        round(float(np.mean(recent_study_rows)) if recent_study_rows else 0.0, 1),
                    ]
                    + [round(gpm, 2)]
                )
                mf.flush()
                last_eval = games_done
                print(f"  CKPT @{games_done} (steps {total_steps}, buffer {len(buffer)})",
                      flush=True)

            if games_done % max(1, args.log_every) < B:  # ~once per log_every games
                gpm = games_done / max(1e-9, (time.time() - t0) / 60)
                eta_h = (args.games - games_done) / max(gpm, 1e-9) / 60
                loss_str = (
                    f"loss {ema_metrics['loss']:.3f} (pi {ema_metrics['policy']:.3f} "
                    f"v {ema_metrics['value']:.3f} q {ema_metrics['q']:.3f} "
                    f"c {ema_metrics['consistency']:.3f})"
                    if ema_metrics else "warmup"
                )
                print(
                    f"[{games_done}/{args.games}] {loss_str} | "
                    f"plies {np.mean(recent_plies):.0f} "
                    f"decisive {np.mean(recent_decisive):.0%} | "
                    f"qabs {np.mean(recent_q_abs):.3f} "
                    f"qvis {np.mean(recent_q_visits):.1f} "
                    f"qhp {np.mean(recent_qhp_abs):.3f} | "
                    f"sqabs {(np.mean(recent_study_q_abs) if recent_study_q_abs else 0.0):.3f} "
                    f"srows {(np.mean(recent_study_rows) if recent_study_rows else 0.0):.0f} | "
                    f"buffer {len(buffer)} steps {total_steps} | "
                    f"{gpm:.1f} g/min eta {eta_h:.1f}h",
                    flush=True,
                )
    finally:
        # final checkpoint + export (always, even on Ctrl-C / crash).
        save_checkpoint(state.ema_params, cfg, latest_path)
        if not args.no_export_torch:
            export_torch_checkpoint(state.ema_params, cfg, latest_torch)
        write_progress(games_done)
        mf.close()

    print(f"done: {games_done} games, {total_steps} steps in "
          f"{(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
