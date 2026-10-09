"""Mixed precision, one way everywhere: bf16 autocast on a card, nothing
off it. Building a disabled `torch.autocast(..., dtype=bfloat16)` context
on a CPU run still asks the CUDA runtime whether bf16 is supported, which
raises on a CUDA build with no visible device (CUDA_VISIBLE_DEVICES="")
— the CPU toy path and the CPU smokes must never touch a card."""
from __future__ import annotations

import contextlib

import torch


PRECISIONS = ("bf16", "fp32")


def autocast(device: str, precision: str = "bf16"):
    """bf16 (every mission so far): bf16 autocast on a card, nothing off it.
    fp32 (the fp32 twin, 2026-10-09): no autocast anywhere — the trainer
    turns TF32 off under TrainConfig.precision 'fp32' as well, so every
    matmul is true fp32."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "fp32" or not str(device).startswith("cuda"):
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)
