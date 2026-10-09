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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress, dtype_floor


def _rms_normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize the last dim; the statistic in fp32, the dtype kept."""
    inv = torch.rsqrt(t.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return t * inv.to(t.dtype)


class CausalSDPA(nn.Module):
    """Softmax attention: fused qkv, F.scaled_dot_product_attention with
    is_causal, the output projection. Two optional guards (2026-10-09; the
    control twins continued from their collapse point):
      qk_norm="rms"  per-head RMSNorm over head_dim on q and k with learned
                     per-head per-channel gains (init 1). `install_gains`
                     sets them from a batch's trained q/k scales so the norm
                     enters near-identity mid-run (a boundary write).
      attn_fp32      under autocast the whole block (projections, logits,
                     softmax, PV, output) runs with autocast disabled, in
                     fp32 (TF32 per the global flag); a no-op otherwise.
    With both off this is the 2s form bit for bit (the same code path)."""

    def __init__(self, d: int, heads: int = 8, qk_norm: str = "",
                 attn_fp32: bool = False):
        super().__init__()
        assert d % heads == 0
        assert qk_norm in ("", "rms"), f"qk_norm: '' or 'rms', got {qk_norm!r}"
        self.h = heads
        self.qk_norm, self.attn_fp32 = qk_norm, bool(attn_fp32)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.qkv.weight)
        nn.init.orthogonal_(self.o.weight)
        if qk_norm:
            hd = d // heads
            self.q_gain = nn.Parameter(torch.ones(heads, 1, hd))
            self.k_gain = nn.Parameter(torch.ones(heads, 1, hd))

    def extra_repr(self) -> str:
        return f"heads={self.h}, qk_norm={self.qk_norm!r}, attn_fp32={self.attn_fp32}"

    # ------------------------------------------------------------ pieces
    def _split(self, x, n):
        B, _, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return tuple(t.view(B, n, self.h, d // self.h).transpose(1, 2)
                     for t in (q, k, v))

    def _guard(self, q, k):
        if not self.qk_norm:
            return q, k
        return (_rms_normalize(q) * self.q_gain.to(q.dtype),
                _rms_normalize(k) * self.k_gain.to(k.dtype))

    def _fp32_active(self, x) -> bool:
        return self.attn_fp32 and x.is_cuda and torch.is_autocast_enabled()

    def _run(self, x, n):
        B, d = x.shape[0], x.shape[-1]
        q, k, v = self._split(x, n)
        q, k = self._guard(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d))

    def forward(self, x):
        n = x.shape[1]
        if self._fp32_active(x):
            with torch.autocast("cuda", enabled=False):
                return self._run(x.float(), n)
        return self._run(x, n)

    # ---------------------------------------------------- incremental decode
    def prefill(self, x):
        """Full causal pass that also returns the decode cache (K/V; the
        guarded k when qk_norm is on)."""
        B, n, d = x.shape
        q, k, v = self._split(x, n)
        q, k = self._guard(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d)), {"k": k, "v": v}

    def step(self, x_t, cache):
        """One new position attending over everything cached (KV cache)."""
        B, _, d = x_t.shape
        q, k, v = self._split(x_t, 1)
        q, k = self._guard(q, k)
        cache["k"] = torch.cat([cache["k"], k], dim=2)
        cache["v"] = torch.cat([cache["v"], v], dim=2)
        y = F.scaled_dot_product_attention(q, cache["k"], cache["v"])
        return self.o(y.transpose(1, 2).reshape(B, 1, d))

    # ------------------------------------------------------ boundary write
    @torch.no_grad()
    def install_gains(self, x) -> dict:
        """Set the QK-norm gains from the RAW q and k on this batch: per
        head, the MEAN per-position RMS times the per-channel RMS of the
        unit-RMS directions. q_normed * gain then carries the trained
        channel pattern at the trained mean position scale, so the mean
        attention logit keeps its scale in expectation and the norm removes
        only the per-position scale. (An RMS-over-all-positions gain would
        overshoot by the position scale's spread: +9% at a CV of .23 on
        the test bed.) Returns the provenance: the gains' means and the mean
        |logit| (with the 1/sqrt(head_dim) scale) before and after on a
        64-position block of the batch."""
        assert self.qk_norm, "install_gains needs qk_norm"
        n = x.shape[1]
        q, k, _ = self._split(x.float(), n)
        hd = q.shape[-1]
        m = min(64, n)

        def logit_scale(a, b):
            return float((a[..., :m, :] @ b[..., :m, :].transpose(-1, -2)).abs().mean()
                         / math.sqrt(hd))

        before = logit_scale(q, k)

        def gains(t):
            rho = t.pow(2).mean(dim=-1, keepdim=True).sqrt()      # the per-position RMS (B, h, n, 1)
            u = t / rho.clamp_min(1e-6)                            # unit-RMS directions
            # (h, 1) * (h, hd): the mean position scale times the channel pattern
            return rho.mean(dim=(0, 2)) * u.pow(2).mean(dim=(0, 2)).sqrt()

        gq, gk = gains(q), gains(k)                    # (h, hd) each
        self.q_gain.copy_(gq.unsqueeze(1))
        self.k_gain.copy_(gk.unsqueeze(1))
        q2, k2 = self._guard(q, k)
        return {"q_gain_mean": float(gq.mean()), "k_gain_mean": float(gk.mean()),
                "logit_scale_before": before, "logit_scale_after": logit_scale(q2, k2)}


class _Constellation(nn.Module):
    """One codebook with its own routing-owned q/k frames (v2 form).

    The multi-constellation hub is the PRODUCT-CODE form (B2: independent
    frames compose, .859 -> .955 monotone in members) at lawful supply
    (aleph-splat-0 TECHNICAL_ROUND5.md, round 5e: K <= 2*D per address space — v1's single 512-anchor book in
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

    def _prefix(self, nc: int, device, dtype):
        """Strictly-lower-triangular ones (nc, nc): the exclusive prefix sum
        as ONE tensor-core GEMM. The cumsum scan kernel ran ~6x off its
        memory roofline on the (B, nc, 2K·H, d) layout (C2d, Blackwell
        2026-08-26) and its backward is flip+cumsum+flip; matmul accumulates
        fp32 inside the GEMM — strictly MORE precise than a bf16 cumsum."""
        key = ("prefix", nc, device, dtype)
        m = self._mask_cache.get(key)
        if m is None:
            m = torch.tril(torch.ones(nc, nc, device=device, dtype=dtype),
                           diagonal=-1)
            self._mask_cache[key] = m
        return m

    # ------------------------------------------------ constellation access
    def _code_cat_qk(self, x):
        """BOTH oriented codes (q and k, every book) in one batched pass.

        Stack all 2H frame weights, one projection einsum, one address
        einsum, ONE fused softmax. The oriented address IS softmax over the
        2K half-axes — exp(cat[u−m, −u−m])/Σ with m = max|u| is bit-the-same
        quantity as F.softmax(cat[u, −u]) (softmax subtracts its own max,
        which is exactly m). This is a KERNEL substitution, not a mechanism
        change: no softmax over positions, no softmax across books —
        composition stays budget. The old chain was ~8 unfused GB-scale
        elementwise passes per call, twice per forward (C2d: 24.6 ms).

        AUTOCAST TRAP (measured, Blackwell 2026-08-26): torch.einsum is in
        autocast's PROMOTE category, and F.normalize / exp / softmax are on
        its fp32 list — one fp32 operand drags the whole downstream scan to
        fp32. Operands are cast to the autocast dtype explicitly; the
        softmax accumulates fp32 inside the kernel (standard attention
        practice) and the returned CODE is in the compute dtype so the
        num/S/P scan and its backward run bf16. DELIBERATE exception: den's
        reductions (kc.sum, att.sum) stay fp32 by autocast policy — the
        agreement mass keeps v1's fp32 dtype_floor semantics at ~5% of
        scan traffic (dtype audit 2026-08-26).

        Returns (qc, kc), each (B, n, H*2K), per-book layout [K pos | K neg]
        matching oriented()/forward_naive."""
        units = self._units()
        dt = (torch.get_autocast_dtype("cuda")
              if torch.is_autocast_enabled() and x.is_cuda else x.dtype)
        W = torch.stack([q.weight for _, q, _ in units]
                        + [k.weight for _, _, k in units]).to(dt)  # (2H, D, d)
        tau = units[0][0].tau
        # tau folds into the codebook (a few-MB fp32 tensor op, MORE precise
        # than dividing bf16 u afterwards), and the query-side row
        # normalization folds into ONE post-GEMM scale: a per-row scalar
        # commutes through the linear map, so (xh/||xh||) @ A^T / tau ==
        # (xh @ (A/tau)^T) * (1/||xh||) exactly (fp reorder). Kills the
        # fp32 normalize-div + cast + separate tau-div passes (perf audit
        # 2026-08-26). Same 1e-12 floor as F.normalize.
        A = (F.normalize(torch.stack([a.codebook for a, _, _ in units]),
                         dim=-1) / tau).to(dt)                     # (H, K, D)
        A = torch.cat([A, A])                                      # (2H, K, D)
        xh = torch.einsum("bnd,hkd->bnhk", x.to(dt), W)            # (B,n,2H,D)
        inv = torch.linalg.vector_norm(                # fp32 by autocast
            xh, dim=-1, keepdim=True).clamp_min(1e-12) \
            .reciprocal().to(dt)                       # policy; cast back
        u = torch.einsum("bnhd,hkd->bnhk", xh, A) * inv            # (B,n,2H,K)
        B, n = x.shape[:2]
        H = len(units)
        # Split-axis-first WITHOUT a copy: the permute is a view, and the
        # cat (which must write a fresh tensor anyway) absorbs it — so the
        # q/k split below is a pure view instead of two GB-scale reshape
        # copies. Layout per book stays [K pos | K neg], book-major.
        u = u.view(B, n, 2, H, -1).permute(2, 0, 1, 3, 4)
        # softmax: the explicit dtype arg opts out of autocast's fp32
        # override (fp32_set_opt_dtype policy) while the CUDA kernel still
        # accumulates fp32 internally — bf16-in/bf16-out, no fp32 e pass,
        # and the softmax BACKWARD chain halves too.
        e = F.softmax(torch.cat([u, -u], dim=-1), dim=-1, dtype=dt)
        return e[0].reshape(B, n, -1), e[1].reshape(B, n, -1)

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
        L = self._prefix(nc, qc.device, qc.dtype)
        P = torch.matmul(L, S.reshape(B, nc, -1)).view_as(S)  # excl. prefix
        zS = kc.sum(dim=2)                              # (B, nc, 2K)
        zP = torch.matmul(L, zS)
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
        # BATCHED path, both n_const cases (2026-08-26 Blackwell verdicts:
        # the per-book Python loop was 512 sequential little scans —
        # launch-bound; then the split q/k exp chains were ~8 unfused
        # GB-scale passes each). Budget composition is algebraically ONE
        # scan over the concatenated code: num and den are sums of per-book
        # bilinear forms, so scanning cat_h(qc_h) against cat_h(kc_h)
        # equals summing H separate scans (fp reorder).
        qc, kc = self._code_cat_qk(x)                   # (B, n, H*2K)
        if qc.dtype != v.dtype:      # einsum-promote guard (belt-and-braces;
            qc = qc.to(v.dtype)      # _code_cat_qk already returns the
            kc = kc.to(v.dtype)      # compute dtype)
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
