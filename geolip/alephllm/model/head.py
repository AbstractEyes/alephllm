"""DualHead — the standard readout plus the L-012 aleph read, born null.

    logits = W_h h + W_s s(h),        W_s == 0 at init

s(h) is the signed address of a learned low-D projection of h through a
K_h-anchor head codebook. New structure enters at zero and earns its way
in by gradient — via WEIGHT-zero init, never gate-zero: mini-beatrix-1
measured (8.4B tokens) that a gamma-gated branch DEADLOCKS (W_s grad is
scaled by gamma=0, gamma grad through a random frozen readout is
zero-mean noise), while the bank's E_out weight-zero self-started into
+2 bpb of function. gamma remains as a frozen scalar for checkpoint
compatibility and as the ablation knob; the election gauge is ||W_s||
and the head_aleph_off toggle.
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
        nn.init.zeros_(self.w_s.weight)          # weight-zero: self-starting
        self.gamma = nn.Parameter(torch.ones(1))  # frozen; ablation knob only
        self.gamma.requires_grad_(False)

    def forward(self, h, disable_aleph: bool = False):
        if self.tied:
            base = h @ self._tied_ref[0].T + self.bias
        else:
            base = self.w_h(h)
        if disable_aleph:
            return base
        return base + self.gamma * self.w_s(self.addr.signed(self.proj(h)))
