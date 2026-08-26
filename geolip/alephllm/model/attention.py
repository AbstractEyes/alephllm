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


class _Constellation(nn.Module):
    """One codebook with its own routing-owned q/k frames (v2 form).

    The multi-constellation hub is the PRODUCT-CODE form (B2: independent
    frames compose, .859 -> .955 monotone in members) at lawful supply
    (ROUND 5e: K <= 2*D per address space — v1's single 512-anchor book in
    32 dims ran 16x and crowded into 333-646 duplicate pairs)."""

    def __init__(self, d: int, K: int, D: int, tau: float):
        super().__init__()
        self.addr = AlephAddress(K, D, tau)
        self.q = nn.Linear(d, D, bias=False)
        self.k = nn.Linear(d, D, bias=False)
        nn.init.orthogonal_(self.q.weight)
        nn.init.orthogonal_(self.k.weight)


class CausalSplatHUB(nn.Module):
    def __init__(self, d: int, K: int = 512, D: int = 32, tau: float = 0.1,
                 chunk: int = 256, n_const: int = 1):
        super().__init__()
        if K > 2 * D:
            import warnings
            warnings.warn(
                f"CausalSplatHUB supply K={K} exceeds 2*D={2*D}: anchors on "
                f"a {D}-dim sphere past ~2x supply CROWD (measured — ROUND "
                "5e shape ladder + the mini-beatrix-1 hub census: duplicate "
                "pairs by the hundreds, consumed erank collapse). Provision "
                "K <= 2*D or raise D.", stacklevel=2)
        self.n_const = n_const
        if n_const == 1:
            # v1 layout, bit-for-bit: state-dict keys addr/q/k unchanged so
            # every shipped checkpoint and the HF automodel mirror load.
            self.addr = AlephAddress(K, D, tau)
            self.q = nn.Linear(d, D, bias=False)
            self.k = nn.Linear(d, D, bias=False)
            nn.init.orthogonal_(self.q.weight)
            nn.init.orthogonal_(self.k.weight)
        else:
            self.consts = nn.ModuleList(
                _Constellation(d, K, D, tau) for _ in range(n_const))
        self.chunk = chunk          # 256 measured best at ctx 2048 (bench)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        for m in (self.v, self.o):
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

    # ------------------------------------------------ constellation access
    def _books_cat(self, x, which: str):
        """All books' oriented codes in one batched pass: stack the frame
        weights (H, D, d) and codebooks (H, K, D) — a few-MB copy — then a
        single projection einsum, one address einsum, and the per-book 2K
        softmax batched over the H axis. Returns (B, n, H*2K)."""
        # AUTOCAST TRAP (measured, Blackwell 2026-08-26): torch.einsum is in
        # autocast's PROMOTE category — one fp32 operand (F.normalize output,
        # raw fp32 params) drags every einsum here AND the whole downstream
        # scan to fp32 (half the TC rate, double the traffic; the 4-6%-util
        # smoking gun). Cast the operands to the autocast dtype explicitly.
        dt = (torch.get_autocast_dtype("cuda")
              if torch.is_autocast_enabled() and x.is_cuda else x.dtype)
        W = torch.stack([(c.q if which == "q" else c.k).weight
                         for c in self.consts]).to(dt)   # (H, D, d)
        A = torch.stack([F.normalize(c.addr.codebook, dim=-1)
                         for c in self.consts]).to(dt)   # (H, K, D)
        tau = self.consts[0].addr.tau
        xh = torch.einsum("bnd,hkd->bnhk", x.to(dt), W)  # (B, n, H, D)
        u = torch.einsum("bnhd,hkd->bnhk",
                         F.normalize(xh, dim=-1).to(dt), A) / tau
        m = u.abs().amax(dim=-1, keepdim=True)
        e = torch.exp(torch.cat([u - m, -u - m], dim=-1))  # (B, n, H, 2K)
        e = e / e.sum(dim=-1, keepdim=True)              # per-book softmax
        B, n = x.shape[:2]
        # torch.exp is ALSO on autocast's fp32 list (bf16 in -> fp32 out), so
        # the softmax tail runs fp32 regardless — standard attention practice
        # (fp32 softmax, low-precision output). Cast the CODE to dt so the
        # whole downstream scan + backward runs bf16.
        return e.reshape(B, n, -1).to(dt)

    def _units(self):
        """Uniform view: [(addr, q, k)] whether single- or multi-book."""
        if self.n_const == 1:
            return [(self.addr, self.q, self.k)]
        return [(c.addr, c.q, c.k) for c in self.consts]

    def _halves(self, x):
        """Per-constellation oriented halves + shared values."""
        outs = []
        for addr, q, k in self._units():
            qp, qn = addr.oriented(q(x))
            kp, kn = addr.oriented(k(x))
            outs.append((qp, qn, kp, kn))
        return outs, self.v(x)

    def _scan_cat(self, qc, kc, v, mask, B, n, nc, C, d):
        """The exact chunked scan for one 2K-wide constellation."""
        K2 = qc.shape[-1]
        qc = qc.view(B, nc, C, K2)
        kc = kc.view(B, nc, C, K2)
        S = torch.einsum("bick,bicd->bikd", kc, v)     # per-chunk 2KxD sums
        P = torch.cumsum(S, dim=1) - S                  # exclusive prefix
        zS = kc.sum(dim=2)                              # (B, nc, 2K)
        zP = torch.cumsum(zS, dim=1) - zS
        att = torch.einsum("bick,bijk->bicj", qc, kc) * mask    # (B,nc,C,C)
        num = torch.einsum("bick,bikd->bicd", qc, P) + att @ v
        den = torch.einsum("bick,bik->bic", qc, zP).unsqueeze(-1) \
            + att.sum(dim=-1, keepdim=True)
        return num.reshape(B, nc * C, d)[:, :n], den.reshape(B, nc * C, 1)[:, :n]

    def forward(self, x):
        """Fast path: the two oriented halves run as ONE 2K-wide pass —
        every term is a sum of bilinear forms over the halves, so one
        pass over cat(p, n) is the same arithmetic in half the kernels
        (equal to forward_naive to fp reorder, ~1.5e-06; speed-harness
        verdict 2026-08-15: 1.7x eager, 4.0x under torch.compile).
        Multi-constellation (n_const > 1): each book scans independently
        and the reads compose BY BUDGET — numerators and agreement masses
        sum across books before the single divide (never softmax over
        books; B4 measured comparative composition at -.10)."""
        B, n, d = x.shape
        v = self.v(x)
        C = min(self.chunk, n)
        pad = (-n) % C
        vp = F.pad(v, (0, 0, 0, pad)) if pad else v
        nc = (n + pad) // C
        vc = vp.view(B, nc, C, d)
        mask = self._mask(C, x.device, v.dtype)
        if self.n_const == 1:
            qc = self.addr.oriented_cat(self.q(x))      # (B, n, 2K)
            kc = self.addr.oriented_cat(self.k(x))
        else:
            # BATCHED multi-book path (2026-08-26 Blackwell verdict: the
            # per-book Python loop was 512 sequential little scans per
            # forward — launch-bound). Budget composition is algebraically
            # ONE scan over the concatenated code: num and den are sums of
            # per-book bilinear forms, so scanning cat_h(qc_h) against
            # cat_h(kc_h) equals summing 16 separate scans (fp reorder).
            qc = self._books_cat(x, "q")                # (B, n, H*2K)
            kc = self._books_cat(x, "k")
        if qc.dtype != v.dtype:      # the einsum-promote trap, single-book
            qc = qc.to(v.dtype)      # path included (oriented_cat's exp
            kc = kc.to(v.dtype)      # chain runs fp32 under autocast)
        if pad:
            qc = F.pad(qc, (0, 0, 0, pad))
            kc = F.pad(kc, (0, 0, 0, pad))
        num, den = self._scan_cat(qc, kc, vc, mask, B, n, nc, C, d)
        cl = dtype_floor(den)
        self._den_raw = (den.detach(), cl)
        self._den_stats = None
        return self.o(num / den.clamp_min(cl))

    # ---------------------------------------------------- incremental decode
    def prefill(self, x):
        """Full causal pass plus the decode cache. The hub's cache is the
        CONSTANT-SIZE prefix state (Sp, Sn, zp, zn) per constellation —
        O(n_const·K·d) regardless of sequence length; this is the
        linear-attention decode advantage. n_const == 1 keeps the exact
        v1 cache shape (arms and the Space depend on it)."""
        out = self.forward(x)
        halves, v = self._halves(x)
        caches = [{"Sp": torch.einsum("bnk,bnd->bkd", kp, v),
                   "Sn": torch.einsum("bnk,bnd->bkd", kn, v),
                   "zp": kp.sum(dim=1), "zn": kn.sum(dim=1)}
                  for (qp, qn, kp, kn) in halves]
        return out, (caches[0] if self.n_const == 1 else {"consts": caches})

    def step(self, x_t, cache):
        """One new position: fold it into each prefix state, read once,
        compose by budget across constellations."""
        halves, v = self._halves(x_t)                  # (B,1,K)/(B,1,d)
        caches = [cache] if self.n_const == 1 else cache["consts"]
        v1 = v.squeeze(1)
        num = den = None
        for (qp, qn, kp, kn), c in zip(halves, caches):
            kp1, kn1 = kp.squeeze(1), kn.squeeze(1)
            c["Sp"] = c["Sp"] + kp1.unsqueeze(-1) * v1.unsqueeze(1)
            c["Sn"] = c["Sn"] + kn1.unsqueeze(-1) * v1.unsqueeze(1)
            c["zp"] = c["zp"] + kp1
            c["zn"] = c["zn"] + kn1
            qp1, qn1 = qp.squeeze(1), qn.squeeze(1)
            nu = torch.einsum("bk,bkd->bd", qp1, c["Sp"]) \
                + torch.einsum("bk,bkd->bd", qn1, c["Sn"])
            de = ((qp1 * c["zp"]).sum(-1)
                  + (qn1 * c["zn"]).sum(-1)).unsqueeze(-1)
            num = nu if num is None else num + nu
            den = de if den is None else den + de
        return self.o((num / den.clamp_min(dtype_floor(den))).unsqueeze(1))

    def forward_naive(self, x):
        """Reference cumsum form (the validated probe-bed implementation).
        O(n·K·d) memory — test oracle only. Sums constellations by budget,
        matching forward()."""
        halves, v = self._halves(x)
        num = den = None
        for qp, qn, kp, kn in halves:
            Sp = torch.cumsum(torch.einsum("bnk,bnd->bnkd", kp, v), dim=1)
            Sn = torch.cumsum(torch.einsum("bnk,bnd->bnkd", kn, v), dim=1)
            zp = torch.cumsum(kp, dim=1)
            zn = torch.cumsum(kn, dim=1)
            nu = torch.einsum("bnk,bnkd->bnd", qp, Sp) \
                + torch.einsum("bnk,bnkd->bnd", qn, Sn)
            de = (qp * zp).sum(-1, keepdim=True) + (qn * zn).sum(-1, keepdim=True)
            num = nu if num is None else num + nu
            den = de if den is None else den + de
        return self.o(num / den.clamp_min(dtype_floor(den)))
