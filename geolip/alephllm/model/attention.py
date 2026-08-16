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
                 chunk: int = 256):
        super().__init__()
        self.addr = AlephAddress(K, D, tau)
        self.chunk = chunk          # 256 measured best at ctx 2048 (bench)
        self.q = nn.Linear(d, D, bias=False)
        self.k = nn.Linear(d, D, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        for m in (self.q, self.k, self.v, self.o):
            nn.init.orthogonal_(m.weight)
        self._mask_cache: dict = {}
        self._den_raw = None        # (den tensor, floor) until read
        self._den_stats = None      # cached floats after first read

    # den stats are LAZY: the reference forward paid three .item() GPU
    # syncs per call just to keep this attribute warm; instruments read
    # it at most once per health interval. Property keeps the tuple API.
    @property
    def last_den_stats(self):
        if self._den_stats is None and self._den_raw is not None:
            den, cl = self._den_raw
            with torch.no_grad():
                self._den_stats = (den.min().item(), den.mean().item(),
                                   (den <= cl).float().mean().item())
        return self._den_stats

    @last_den_stats.setter
    def last_den_stats(self, value):
        self._den_stats = value
        self._den_raw = None

    def _mask(self, C: int, device, dtype):
        key = (C, device, dtype)
        m = self._mask_cache.get(key)
        if m is None:
            m = torch.tril(torch.ones(C, C, device=device, dtype=dtype))
            self._mask_cache[key] = m
        return m

    def _halves(self, x):
        qp, qn = self.addr.oriented(self.q(x))
        kp, kn = self.addr.oriented(self.k(x))
        return qp, qn, kp, kn, self.v(x)

    def forward(self, x):
        """Fast path: the two oriented halves run as ONE 2K-wide pass —
        every term is a sum of bilinear forms over the halves, so one
        pass over cat(p, n) is the same arithmetic in half the kernels
        (equal to forward_naive to fp reorder, ~1.5e-06; speed-harness
        verdict 2026-08-15: 1.7x eager, 4.0x under torch.compile)."""
        B, n, d = x.shape
        qc = self.addr.oriented_cat(self.q(x))            # (B, n, 2K)
        kc = self.addr.oriented_cat(self.k(x))
        v = self.v(x)
        C = min(self.chunk, n)
        pad = (-n) % C
        if pad:
            qc = F.pad(qc, (0, 0, 0, pad))
            kc = F.pad(kc, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        nc = (n + pad) // C
        K2 = qc.shape[-1]
        qc = qc.view(B, nc, C, K2)
        kc = kc.view(B, nc, C, K2)
        v = v.view(B, nc, C, d)
        mask = self._mask(C, x.device, v.dtype)

        S = torch.einsum("bick,bicd->bikd", kc, v)     # per-chunk 2KxD sums
        P = torch.cumsum(S, dim=1) - S                  # exclusive prefix
        zS = kc.sum(dim=2)                              # (B, nc, 2K)
        zP = torch.cumsum(zS, dim=1) - zS
        att = torch.einsum("bick,bijk->bicj", qc, kc) * mask    # (B,nc,C,C)
        num = torch.einsum("bick,bikd->bicd", qc, P) + att @ v
        den = torch.einsum("bick,bik->bic", qc, zP).unsqueeze(-1) \
            + att.sum(dim=-1, keepdim=True)

        num = num.reshape(B, nc * C, d)[:, :n]
        den = den.reshape(B, nc * C, 1)[:, :n]
        cl = dtype_floor(den)
        self._den_raw = (den.detach(), cl)
        self._den_stats = None
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
