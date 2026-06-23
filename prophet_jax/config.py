"""Static configuration + game-count curricula for the JAX/Flax port.

This module is pure Python — no JAX, no NumPy, no tensor math. Everything here
is *orchestration*: frozen dataclasses whose field values are ported
value-for-value from the reference PyTorch implementation (``prophet/``), and
the schedule functions that pick which *static* config band the next jitted
call should use.

Why frozen, and why value-for-value:
- The dataclasses are ``frozen=True`` so a config can be hashed and safely used
  as a ``static_argnum`` to ``jax.jit`` (mutable dataclasses are unhashable and
  would also let a stray write desync learner/worker). Treat every config as an
  immutable description of one compiled program.
- The defaults MUST match the reference exactly. ``ModelConfig`` defaults, the
  search/self-play/study knobs, the loss weights, and the schedule thresholds
  are all load-bearing for interop: trained checkpoints, replay targets, and the
  learner<->worker curriculum only line up if these numbers are identical to
  ``prophet/model.py``, ``prophet/search.py``, ``prophet/selfplay.py``,
  ``prophet/study.py``, ``prophet/train.py`` and ``prophet/schedule.py``.

CRITICAL — model size is read from the checkpoint, never hardcoded:
``ModelConfig`` here carries the *reference defaults* (d_model=128, n_layers=4)
purely so a fresh smoke-test model can be built. Production "moonshot" runs use
d_model=320 / n_layers=8, and the 100k run used d_model=192 / n_layers=6. The
checkpoint stores ``asdict(cfg)`` under the key ``"config"``; the model loader
(``prophet_jax/model.py:load_torch_checkpoint``) MUST reconstruct ``ModelConfig``
from that dict (and read ``in_features`` off ``embed.weight.shape[1]``). Do NOT
hardcode any size in the builder/loader — see ``ModelConfig`` for the contract.

The schedule functions (``q_trust_at``, ``study_config_at``, ``loss_weights_at``)
are piecewise-constant in the total self-play game count. They select a *static*
band for the *next* jitted call (e.g. ``q_trust`` baked into a search program,
the loss weights baked into a train step). They are orchestration only and never
appear inside a traced computation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "ModelConfig",
    "SearchConfig",
    "SelfPlayConfig",
    "StudyConfig",
    "LossWeights",
    "NUM_ACTIONS",
    "FEATURES",
    "PGX_NUM_ACTIONS",
    "DRAW_HALFMOVE_CAP",
    "INF",
    "q_trust_at",
    "study_config_at",
    "loss_weights_at",
]


# ---------------------------------------------------------------------------
# Module-level constants (ported value-for-value).
# ---------------------------------------------------------------------------

#: Prophet action space: ``from_square * 64 + to_square`` (both in [0, 64)).
#: Promotions default to queen; under-promotions live outside this space.
#: Matches ``prophet.encoding.NUM_ACTIONS`` and the model's flattened
#: ``[B, 4096]`` policy/Q heads.
NUM_ACTIONS: int = 4096

#: Per-token input feature width of the 24-plane board encoding. Matches
#: ``prophet.encoding.FEATURES`` and ``ModelConfig.in_features``. The JAX
#: encoder (``prophet_jax/env.py:encode_state``) must produce exactly this
#: layout (12 piece planes + ep + 4 castling + halfmove + 4 history + repetition
#: + side parity), or trained weights will not transfer.
FEATURES: int = 24

#: pgx chess action space (= 64 from-squares * 73 move-types). The pgx env uses
#: this; the prophet 4096-action space is mapped onto it in
#: ``prophet_jax/env.py`` (queen-promotion moves only). NOT the model's output
#: width — the model emits ``NUM_ACTIONS`` (4096) logits.
PGX_NUM_ACTIONS: int = 4672

#: Fifty-move rule: a position with ``halfmove_clock >= DRAW_HALFMOVE_CAP`` is a
#: draw (value 0.0 for the side to move). Matches ``prophet.search`` /
#: ``prophet.encoding`` (the no-progress counter is also stored, scaled by this
#: cap, in feature column 17).
DRAW_HALFMOVE_CAP: int = 100

#: "Infinity" sentinel used as the final (open-ended) upper threshold in the
#: schedule stage tables. Matches ``prophet.schedule.INF``. Kept as a Python int
#: (these comparisons are pure-Python orchestration, never traced).
INF: int = 10 ** 9


# ---------------------------------------------------------------------------
# Frozen config dataclasses (defaults ported value-for-value from the reference).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Transformer hyper-parameters for ``PolicyQValueNet``.

    Defaults reproduce the reference ``prophet/model.py:ModelConfig`` (the v0
    smoke-test size). The forward path is: per-token ``Linear(in_features ->
    d_model)`` + learned positional ``[1, 64, d_model]``, then ``n_layers``
    pre-LN transformer encoder layers (``n_heads`` heads, ``d_ff`` feed-forward,
    exact-erf GELU), a final ``LayerNorm``, and three coupled from/to heads of
    width ``head_dim``.

    PRODUCTION OVERRIDES — DO NOT HARDCODE A SIZE IN THE LOADER/BUILDER:
    the "moonshot" run uses ``d_model=320, n_layers=8``; the 100k run used
    ``d_model=192, n_layers=6`` (2.77M params). The checkpoint stores
    ``asdict(cfg)`` under key ``"config"``; the JAX loader must rebuild this
    dataclass from that dict and set ``in_features`` from
    ``state["embed.weight"].shape[1]``. These defaults exist only for building a
    fresh (untrained) model.
    """

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    head_dim: int = 64
    dropout: float = 0.0
    in_features: int = FEATURES


@dataclass(frozen=True)
class SearchConfig:
    """Gumbel-root + PUCT-interior search knobs (reference defaults).

    Mirrors ``prophet/search.py:SearchConfig``. ``q_trust`` controls how much
    the search trusts the Q-head's value for *unvisited* children (seeded as
    ``q_trust * q_init`` instead of 0); it is ramped per game by
    :func:`q_trust_at` (0.1 early, up to 1.0 once the Q-head matures). The other
    fields parameterize sequential halving and the completed-Q sigma transform.
    """

    sims: int = 32
    root_candidates: int = 8
    c_puct: float = 1.5
    c_visit: float = 50.0
    c_scale: float = 1.0
    q_trust: float = 1.0


@dataclass(frozen=True)
class SelfPlayConfig:
    """Self-play / outcome-mixing / resignation knobs (reference defaults).

    Mirrors ``prophet/selfplay.py:SelfPlayConfig``. The value target blends the
    search root value with the real outcome by ``outcome_mix``; draws train as
    ``-contempt`` for both sides; decisive results are scaled by
    ``win_discount ** (plies_remaining)`` so faster wins are worth more.
    Resignation (gated on by the learner once the value head matures) adjudicates
    a game after ``resign_plies`` consecutive root values below
    ``resign_threshold``, with ``resign_off_prob`` of games never resigning so
    miscalibration surfaces as real losses.
    """

    max_plies: int = 200
    outcome_mix: float = 0.5
    resign_threshold: float = -0.92
    resign_plies: int = 8
    resign_off_prob: float = 0.1
    contempt: float = 0.15
    win_discount: float = 0.997


@dataclass(frozen=True)
class StudyConfig:
    """Deep-reflection ("study your losses") knobs (reference defaults).

    Mirrors ``prophet/study.py:StudyConfig``. Surprising plies (top
    ``top_k`` by the swing/outcome/Q-surprise score, filtered to ``>=
    min_surprise``) are re-searched deeply (``deep_sims`` / ``deep_candidates``)
    into high-weight samples (``study_weight``), and the deep search's best
    ``n_lines`` alternates are played out for up to ``branch_plies`` plies at the
    normal budget (``branch_weight``), with terminal branches mixing the outcome
    in by ``outcome_mix``. ``top_k`` / ``deep_sims`` / ``branch_plies`` /
    ``n_lines`` are ramped per game by :func:`study_config_at`; the rest stay at
    these defaults.
    """

    top_k: int = 2
    min_surprise: float = 0.15
    deep_sims: int = 128
    deep_candidates: int = 16
    branch_plies: int = 16
    study_weight: float = 2.0
    branch_weight: float = 1.0
    outcome_mix: float = 0.5
    n_lines: int = 1
    q_surprise_weight: float = 1.0


@dataclass(frozen=True)
class LossWeights:
    """Per-term loss weights (reference defaults).

    Mirrors ``prophet/train.py:LossWeights``. Total loss is
    ``policy*loss_pi + value*loss_v + q*loss_q + consistency*loss_cons +
    wdl*loss_wdl``. The ``q`` and ``consistency`` weights are ramped up per game
    by :func:`loss_weights_at` (they start low while the Q-head is noise);
    ``policy``, ``value`` and ``wdl`` are fixed across the schedule.
    """

    policy: float = 1.0
    value: float = 1.0
    q: float = 1.0
    consistency: float = 0.5
    wdl: float = 0.5


# ---------------------------------------------------------------------------
# Game-count curricula (ported verbatim from prophet/schedule.py).
#
# Pure-Python piecewise selectors. They pick the *static* config band for the
# *next* jitted call (search program / train step) — orchestration only, never
# part of a traced computation. The learner writes the running game count to a
# progress file; workers poll it, so both sides compute identical bands.
# ---------------------------------------------------------------------------


def _stage(games, stages):
    """Return the value for the first stage whose threshold the game count is
    below.

    ``stages`` is ``[(upper_threshold, value), ...]`` sorted ascending. If
    ``games`` is below ``thr``, that stage's ``value`` is returned; otherwise the
    last stage's value is used (the final threshold is conventionally ``INF``).
    Verbatim from ``prophet.schedule._stage``.
    """
    for thr, val in stages:
        if games < thr:
            return val
    return stages[-1][1]


def q_trust_at(games: int) -> float:
    """How much search should trust the Q-head for *unvisited* children.

    Q is noise early, so search must not lean on ``q_init`` until the Q-head
    matures; this ramps the trust from 0.1 up to 1.0. Baked into the
    ``SearchConfig.q_trust`` field of the next search program (see
    ``cfg_fn`` in the worker/learner loop). Verbatim from
    ``prophet.schedule.q_trust_at``.
    """
    return _stage(
        games,
        [(2000, 0.1), (10000, 0.25), (40000, 0.5), (80000, 0.75), (INF, 1.0)],
    )


def study_config_at(games: int, base: StudyConfig) -> StudyConfig:
    """Ramp study intensity: ``(top_k, deep_sims, branch_plies, n_lines)``.

    Returns a copy of ``base`` with just those four fields replaced for the
    current game count (deeper re-search, more lines, higher top_k as the net
    gets good enough for deep reflection to pay off). All other ``StudyConfig``
    fields (``min_surprise``, ``study_weight``, ``branch_weight``,
    ``outcome_mix``, ``q_surprise_weight``, ``deep_candidates``) are left at
    ``base``'s values — they are not scheduled. Verbatim from
    ``prophet.schedule.study_config_at`` (``dataclasses.replace`` works on a
    frozen dataclass, returning a new frozen instance).
    """
    tk, ds, bp, nl = _stage(
        games,
        [
            (10000, (4, 256, 24, 2)),
            (40000, (6, 384, 32, 3)),
            (80000, (8, 512, 40, 3)),
            (INF, (8, 768, 48, 4)),
        ],
    )
    return replace(base, top_k=tk, deep_sims=ds, branch_plies=bp, n_lines=nl)


def loss_weights_at(games: int) -> LossWeights:
    """Ramp the Q and consistency loss weights up as the Q-head matures.

    ``policy`` / ``value`` / ``wdl`` stay fixed (1.0 / 1.0 / 0.5); only ``q`` and
    ``consistency`` change with game count. Applied learner-side per train step
    (the returned weights are baked into the next jitted ``train_step``).
    Verbatim from ``prophet.schedule.loss_weights_at``.
    """
    q, c = _stage(
        games,
        [(2000, (0.25, 0.1)), (10000, (0.5, 0.25)), (INF, (1.0, 0.6))],
    )
    return LossWeights(policy=1.0, value=1.0, q=q, consistency=c, wdl=0.5)
