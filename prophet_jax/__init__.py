# prophet_jax/__init__.py
#
# CRITICAL IMPORT-ORDER GUARD (keep these as the VERY FIRST executable lines):
# On the Mac dev box the project's historical "import torch before numpy"
# libomp/OpenMP clash (Homebrew libomp double-load -> segfault) reappears as a
# JAX/numpy clash: if numpy initializes its OpenMP/BLAS runtime before JAX's,
# importing jax afterwards can segfault or hang the process. The fix is to make
# `jax` the first heavy import in the whole package, BEFORE numpy is imported
# anywhere. Therefore:
#   1. import os and pin JAX_PLATFORMS (empty string -> let JAX auto-select the
#      best available backend: TPU/GPU/Metal/CPU; setdefault so an explicit
#      env override from the caller still wins).
#   2. import jax (and jax.numpy) HERE, at package import time, before any
#      submodule (and thus any numpy) is pulled in.
# Every submodule in this package MUST also keep its own `import jax` above any
# `import numpy`. This file enforces the ordering for the whole package by
# importing jax before re-exporting the submodules (which is where numpy lands).
import os

os.environ.setdefault("JAX_PLATFORMS", "")  # "" => JAX picks the best backend

import jax  # noqa: E402,F401  -- MUST precede any numpy import in the package
import jax.numpy as jnp  # noqa: E402,F401  -- still before numpy; safe (jnp != numpy)

# NOTE: Do NOT `import numpy` in this file. numpy is only imported *inside* the
# submodules below, all of which import jax first. Pulling jax in above
# guarantees jax's runtime is initialized before any numpy import executes.

"""prophet_jax — JAX/Flax port of the prophet chess engine.

A compute-efficient, AlphaZero-style self-play chess engine with an explicit
per-move Q ("intuition") head, ported from the reference PyTorch implementation
in ``prophet/`` to a JAX-native stack:

* environment : pgx (``pgx.make("chess")``) for batched, jittable chess
* search      : mctx Gumbel MuZero policy (``mctx.gumbel_muzero_policy``)
* model       : Flax linen transformer encoder over 64 square-tokens with three
                coupled heads (policy 4096 / per-move Q 4096 / WDL value)
* training    : optax Adam(W) with EMA of params, value_and_grad + jit

This package is a thin aggregation layer. All real logic lives in the
submodules; this ``__init__`` only (a) enforces the jax-before-numpy import
order described above and (b) re-exports the public surface so callers can do::

    from prophet_jax import (
        ModelConfig, PolicyQValueNet, build_model,
        ChessEnv, make_chess_env,
        SearchConfig, run_search, batched_search,
        SelfPlayConfig, generate_selfplay,
        StudyConfig, reflect_batch,
        LossWeights, train_step, make_train_state,
    )

Reference invariants the port preserves (see the prophet spec):
  * action space = 64*64 = 4096, action = from*64 + to (queen-promo only)
  * 24-feature board encoding, side-to-move perspective, ^56 (rank) flip
  * value convention: scalar in [-1, 1] = P(win) - P(loss), negamax across plies
  * config-driven model loading (read dims from the checkpoint, never hardcode)
"""

# ---------------------------------------------------------------------------
# Public re-exports.
#
# Importing these submodules triggers their (jax-first, then numpy) imports.
# Because `import jax` above has already run, jax's runtime is live before any
# submodule numpy import — preserving the guard for the whole package.
#
# Only the names listed in __all__ are part of the supported surface; this file
# is pure aggregation (no logic, no new symbols beyond the import guard).
# ---------------------------------------------------------------------------

# -- config.py : dataclasses + curricula + global constants ----------------
from .config import (
    ModelConfig,
    SearchConfig,
    SelfPlayConfig,
    StudyConfig,
    LossWeights,
    NUM_ACTIONS,      # 4096  (prophet from-to action space)
    FEATURES,         # 24    (board encoding feature columns)
    PGX_NUM_ACTIONS,  # 4672  (pgx chess action space)
    DRAW_HALFMOVE_CAP,  # 100
    INF,              # 10**9
    q_trust_at,
    study_config_at,
    loss_weights_at,
)

# -- model.py : Flax linen PolicyQValueNet + checkpoint interop ------------
from .model import (
    PolicyQValueNet,
    build_model,
    forward,
    forward_wdl,
    load_torch_checkpoint,
    save_checkpoint,
    export_torch_checkpoint,
    num_params,
)

# -- env.py : pgx chess wrapper + prophet<->pgx adapters -------------------
from .env import (
    ChessEnv,
    make_chess_env,
    start_keys,
    env_init,
    env_step,
    encode_state,
    legal_mask,
    prophet_to_pgx,
    terminal_info,
)

# -- search.py : mctx Gumbel root search + result extraction ---------------
from .search import (
    SearchOut,
    root_fn,
    recurrent_fn,
    run_search,
    batched_search,
    search_result,
)

# -- selfplay.py : vectorized self-play generation -------------------------
from .selfplay import (
    SamplesBatch,
    GameMeta,
    generate_selfplay,
)

# -- reflection.py : deep "study-your-losses" re-analysis ------------------
from .reflection import (
    find_surprises,
    reflect_batch,
)

# -- train.py : optax train state + jitted train step ----------------------
from .train import (
    TrainState,
    make_train_state,
    train_step,
    sample_batch,
)

# Optional metadata; harmless if absent.
try:  # pragma: no cover - version is best-effort only
    from ._version import __version__  # type: ignore
except Exception:  # noqa: BLE001 - any import failure -> fall back to default
    __version__ = "0.0.0"

__all__ = [
    # version / meta
    "__version__",
    # config
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
    # model
    "PolicyQValueNet",
    "build_model",
    "forward",
    "forward_wdl",
    "load_torch_checkpoint",
    "save_checkpoint",
    "export_torch_checkpoint",
    "num_params",
    # env
    "ChessEnv",
    "make_chess_env",
    "start_keys",
    "env_init",
    "env_step",
    "encode_state",
    "legal_mask",
    "prophet_to_pgx",
    "terminal_info",
    # search
    "SearchOut",
    "root_fn",
    "recurrent_fn",
    "run_search",
    "batched_search",
    "search_result",
    # selfplay
    "SamplesBatch",
    "GameMeta",
    "generate_selfplay",
    # reflection
    "find_surprises",
    "reflect_batch",
    # train
    "TrainState",
    "make_train_state",
    "train_step",
    "sample_batch",
]
