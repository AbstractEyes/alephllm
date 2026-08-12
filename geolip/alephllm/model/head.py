"""DualHead — the standard readout plus the L-012 aleph read, born null.

    logits = W_h h + gamma * W_s s(h),        gamma == 0 at init

s(h) is the signed address of a learned low-D projection of h through a
K_h-anchor head codebook. New structure enters at zero and earns its way
in by gradient: gamma's trajectory is the election instrument. With
gamma=0 the head is exactly the standard linear readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .address import AlephAddress


class DualHead(nn.Module):
    def __init__(self, d: int, vocab: int, K: int = 512, D: int = 32,
                 tau: float = 0.1, tied_weight: nn.Parameter | None = None):
        super().__init__()
        self.tied = tied_weight is not None
        if self.tied:
            # tuple wrapper: reference the embedding table WITHOUT registering
            # it as a duplicate parameter (state_dict must stay alias-free
            # for safetensors export)
            self._tied_ref = (tied_weight,)
            self.bias = nn.Parameter(torch.zeros(vocab))
        else:
            self.w_h = nn.Linear(d, vocab, bias=True)
        self.proj = nn.Linear(d, D, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(K, D, tau)
        self.w_s = nn.Linear(K, vocab, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h, disable_aleph: bool = False):
        if self.tied:
            base = h @ self._tied_ref[0].T + self.bias
        else:
            base = self.w_h(h)
        if disable_aleph:
            return base
        return base + self.gamma * self.w_s(self.addr.signed(self.proj(h)))
