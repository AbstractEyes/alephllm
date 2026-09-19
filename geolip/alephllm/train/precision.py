"""Mixed precision, one way everywhere: bf16 autocast on a card, nothing
off it. Building a disabled `torch.autocast(..., dtype=bfloat16)` context
on a CPU run still asks the CUDA runtime whether bf16 is supported, which
raises on a CUDA build with no visible device (CUDA_VISIBLE_DEVICES="")
— the CPU toy path and the CPU smokes must never touch a card."""
from __future__ import annotations

import contextlib

import torch


def autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
