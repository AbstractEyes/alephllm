"""CanonConv — a Canon layer (0.10.18 rc8, 2026-10-14): lightweight mixing of each position with its few predecessors.

WHY: a lookup by a multi-byte key composes two steps: gather the key's bytes into one position, then match it across the
context and copy what follows. Attention-only stacks learn the gathering in an earlier layer; hybrids with few attention
layers may fail to. The gathering is what a short causal convolution does directly.

THE FORM (Allen-Zhu, Physics of Language Models Part 4.1, arXiv 2512.17351, Eq. 4.1, the residual form): for a hidden state h
(B, T, d), h'_t = h_t + w_0 * h_t + w_1 * h_(t-1) + w_2 * h_(t-2) + w_3 * h_(t-3), per-channel weight vectors w_i (d), zeros
before the first position (causal). Here every weight is born ZERO, so h' = h exactly at birth and a craft carrying the layer is
its Canon-free form at step 0 (the weights learn from the first step: dh'/dw_i = h_(t-i)). The paper inserts it before the
attention (its site A), after the q/k/v projections (B), before the MLP (C) and inside the MLP (D); this library offers A (the
mixer's input, after the block's first norm) and C (the bank's input, after the second norm).

Forward only: the decode paths (prefill / step) refuse a craft with the layer (a three-position cache is owed with its promotion).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonConv(nn.Module):
    def __init__(self, d: int, k: int = 4):
        super().__init__()
        self.k = int(k)
        self.weight = nn.Parameter(torch.zeros(d, self.k))      # weight[:, i] multiplies h_(t-i); zero-born

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        T = h.shape[-2]
        out = h
        for i in range(self.k):
            if i == 0:
                hi = h
            elif i >= T:
                continue                                         # no predecessor that far back in this window
            else:
                hi = F.pad(h[..., : T - i, :], (0, 0, i, 0))     # h_(t-i), zeros before the first position
            out = out + hi * self.weight[:, i].to(h.dtype)
        return out
