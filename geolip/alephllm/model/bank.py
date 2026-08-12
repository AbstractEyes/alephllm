"""AnchoredBank — the E1-form anchored FFN, born on its own null path.

    Bank(x) = trunk(x) + sum_k w_k(x) * sigmoid(g_k) * expert_k(x)

Trunk: always-on d->ff->d GELU expert. Dispatch: the signed aleph address
over a learned K x d codebook read against the layer input (K=3 fat
experts, the shape validated at parity under encoder pressure). Gates
init -3.0; expert OUTPUT projections zero-init, so at birth the dispatch
contributes exactly zero and the bank is bit-identical to its dense
control (the C6 null path). No balance machinery of any kind —
differentiation is an attractor, pressure stays out of the task gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress


class AnchoredBank(nn.Module):
    def __init__(self, d: int, n_experts: int = 3, ff: int | None = None,
                 tau: float = 0.1, gate_init: float = -3.0):
        super().__init__()
        ff = ff or d
        self.n_experts = n_experts
        self.t_in = nn.Linear(d, ff, bias=False)
        self.t_out = nn.Linear(ff, d, bias=False)
        nn.init.orthogonal_(self.t_in.weight)
        nn.init.orthogonal_(self.t_out.weight)
        self.addr = AlephAddress(n_experts, d, tau)
        w_in = torch.empty(n_experts, d, ff)
        for k in range(n_experts):
            nn.init.orthogonal_(w_in[k])
        self.w_in = nn.Parameter(w_in)
        self.w_out = nn.Parameter(torch.zeros(n_experts, ff, d))  # null path
        self.gates = nn.Parameter(torch.full((n_experts,), gate_init))
        self.last_dispatch = None  # (mean|w| per expert, w sample) for instruments

    def forward(self, x, disable_dispatch: bool = False):
        trunk = self.t_out(F.gelu(self.t_in(x)))
        if disable_dispatch:
            return trunk
        w = self.addr.signed(x)                                   # (B, n, K)
        with torch.no_grad():
            self.last_dispatch = w.detach()
        h = F.gelu(torch.einsum("bnd,kdf->bnkf", x, self.w_in))
        e = torch.einsum("bnkf,kfd->bnkd", h, self.w_out)
        return trunk + torch.einsum("bnk,bnkd->bnd",
                                    w * torch.sigmoid(self.gates), e)
