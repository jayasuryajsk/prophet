"""CUDA acceleration helpers.

All of this is a no-op off CUDA, so MPS/CPU behavior is byte-for-byte
unchanged — the only code path that changes is the GPU one.

- setup_perf: enable TF32 + cuDNN autotuning (free throughput on Ampere+).
- autocast: bf16 autocast context on CUDA (no GradScaler needed — bf16 has
  fp32 dynamic range), nullcontext elsewhere.
- to_np: cast a (possibly bf16) tensor to fp32 numpy (numpy has no bf16).
- maybe_compile: torch.compile on CUDA when requested. Self-play and training
  both run constant batch sizes (batch_games / --batch), so the compiled graph
  does not recompile per step.
"""

import contextlib

import torch


def setup_perf(device_str: str) -> None:
    if str(device_str).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _dev_type(device) -> str:
    return device.type if isinstance(device, torch.device) else str(device).split(":")[0]


def autocast(device):
    """bf16 autocast on CUDA; nullcontext on MPS/CPU (no behavior change)."""
    if _dev_type(device) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_np(t: torch.Tensor):
    """Tensor -> fp32 numpy (handles bf16 autocast outputs)."""
    return t.float().cpu().numpy()


def maybe_compile(model, device, enabled: bool):
    if enabled and _dev_type(device) == "cuda":
        return torch.compile(model)
    return model
