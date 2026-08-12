"""Attention blocks: CausalSDPA (the workhorse) and CausalSplatHUB (the
instrumented aleph read).

CausalSplatHUB is causal linear attention through the oriented address:
prefix-sum memories over the two K-wide halves of the 2K softmax, read by
the query's halves and normalized by the scalar agreement mass. O(n·K·d)
compute, no softmax over positions, no selection event anywhere.

The naive cumsum form materializes (B, n, K, d) — fine on probe beds,
fatal at mission scale. forward() therefore uses an exact chunked scan:
within-chunk causal affinity (B, C, C) + cross-chunk carried states
(B, K, d). `forward_naive()` is kept verbatim as the equivalence oracle
for the test array.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress, dtype_floor


class CausalSDPA(nn.Module):
    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        assert d % heads == 0
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.qkv.weight)
        nn.init.orthogonal_(self.o.weight)

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d))


class CausalSplatHUB(nn.Module):
    def __init__(self, d: int, K: int = 512, D: int = 32, tau: float = 0.1,
                 chunk: int = 128):
        super().__init__()
        self.addr = AlephAddress(K, D, tau)
        self.chunk = chunk
        self.q = nn.Linear(d, D, bias=False)
        self.k = nn.Linear(d, D, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        for m in (self.q, self.k, self.v, self.o):
            nn.init.orthogonal_(m.weight)
        self.last_den_stats = None  # populated each forward for instruments

    def _halves(self, x):
        qp, qn = self.addr.oriented(self.q(x))
        kp, kn = self.addr.oriented(self.k(x))
        return qp, qn, kp, kn, self.v(x)

    def forward(self, x):
        B, n, d = x.shape
        qp, qn, kp, kn, v = self._halves(x)
        C = min(self.chunk, n)
        pad = (-n) % C
        if pad:
            z = lambda t: F.pad(t, (0, 0, 0, pad))
            qp, qn, kp, kn, v = z(qp), z(qn), z(kp), z(kn), z(v)
        nc = (n + pad) // C
        K = qp.shape[-1]
        qp, qn, kp, kn = (t.view(B, nc, C, K) for t in (qp, qn, kp, kn))
        v = v.view(B, nc, C, d)
        mask = torch.tril(torch.ones(C, C, device=x.device, dtype=v.dtype))

        num = torch.zeros(B, nc, C, d, device=x.device, dtype=v.dtype)
        den = torch.zeros(B, nc, C, 1, device=x.device, dtype=v.dtype)
        for kh, qh in ((kp, qp), (kn, qn)):
            S = torch.einsum("bick,bicd->bikd", kh, v)      # per-chunk KxD sums
            P = torch.cumsum(S, dim=1) - S                   # exclusive prefix
            zS = kh.sum(dim=2)                               # (B, nc, K)
            zP = torch.cumsum(zS, dim=1) - zS
            att = torch.einsum("bick,bijk->bicj", qh, kh) * mask   # (B,nc,C,C)
            num = num + torch.einsum("bick,bikd->bicd", qh, P) + att @ v
            den = den + torch.einsum("bick,bik->bic", qh, zP).unsqueeze(-1) \
                      + att.sum(dim=-1, keepdim=True)
        num = num.view(B, nc * C, d)[:, :n]
        den = den.view(B, nc * C, 1)[:, :n]
        cl = dtype_floor(den)
        with torch.no_grad():
            self.last_den_stats = (den.min().item(), den.mean().item(),
                                   (den <= cl).float().mean().item())
        return self.o(num / den.clamp_min(cl))

    def forward_naive(self, x):
        """Reference cumsum form (the validated probe-bed implementation).
        O(n·K·d) memory — test oracle only."""
        qp, qn, kp, kn, v = self._halves(x)
        Sp = torch.cumsum(torch.einsum("bnk,bnd->bnkd", kp, v), dim=1)
        Sn = torch.cumsum(torch.einsum("bnk,bnd->bnkd", kn, v), dim=1)
        zp = torch.cumsum(kp, dim=1)
        zn = torch.cumsum(kn, dim=1)
        num = torch.einsum("bnk,bnkd->bnd", qp, Sp) \
            + torch.einsum("bnk,bnkd->bnd", qn, Sn)
        den = (qp * zp).sum(-1, keepdim=True) + (qn * zn).sum(-1, keepdim=True)
        return self.o(num / den.clamp_min(dtype_floor(den)))
