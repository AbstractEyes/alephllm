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

    # ---------------------------------------------------- incremental decode
    def prefill(self, x):
        """Full causal pass that also returns the decode cache (K/V)."""
        B, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d)), {"k": k, "v": v}

    def step(self, x_t, cache):
        """One new position attending over everything cached (KV cache)."""
        B, _, d = x_t.shape
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)
        q, k, v = (t.view(B, 1, self.h, d // self.h).transpose(1, 2)
                   for t in (q, k, v))
        cache["k"] = torch.cat([cache["k"], k], dim=2)
        cache["v"] = torch.cat([cache["v"], v], dim=2)
        y = F.scaled_dot_product_attention(q, cache["k"], cache["v"])
        return self.o(y.transpose(1, 2).reshape(B, 1, d))


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

    # ---------------------------------------------------- incremental decode
    def prefill(self, x):
        """Full causal pass plus the decode cache. The hub's cache is the
        CONSTANT-SIZE prefix state (Sp, Sn, zp, zn) — O(K·d) regardless of
        sequence length; this is the linear-attention decode advantage."""
        out = self.forward(x)
        qp, qn, kp, kn, v = self._halves(x)
        cache = {"Sp": torch.einsum("bnk,bnd->bkd", kp, v),
                 "Sn": torch.einsum("bnk,bnd->bkd", kn, v),
                 "zp": kp.sum(dim=1), "zn": kn.sum(dim=1)}
        return out, cache

    def step(self, x_t, cache):
        """One new position: fold it into the prefix state, read once."""
        qp, qn, kp, kn, v = self._halves(x_t)          # (B,1,K)/(B,1,d)
        kp1, kn1, v1 = kp.squeeze(1), kn.squeeze(1), v.squeeze(1)
        cache["Sp"] = cache["Sp"] + kp1.unsqueeze(-1) * v1.unsqueeze(1)
        cache["Sn"] = cache["Sn"] + kn1.unsqueeze(-1) * v1.unsqueeze(1)
        cache["zp"] = cache["zp"] + kp1
        cache["zn"] = cache["zn"] + kn1
        qp1, qn1 = qp.squeeze(1), qn.squeeze(1)
        num = torch.einsum("bk,bkd->bd", qp1, cache["Sp"]) \
            + torch.einsum("bk,bkd->bd", qn1, cache["Sn"])
        den = ((qp1 * cache["zp"]).sum(-1)
               + (qn1 * cache["zn"]).sum(-1)).unsqueeze(-1)
        return self.o((num / den.clamp_min(dtype_floor(den))).unsqueeze(1))

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
