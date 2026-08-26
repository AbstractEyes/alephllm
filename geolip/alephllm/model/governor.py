"""The anchor governor — min-separation projection (ROUND 5f, 2026-08-25).

Measured law (claude-mind, canon/splat_aleph_battery.md ROUND 5e/5f): anchors
crowded past the separation floor flutter under the positional sweep (the
static logit gap is quadratic in separation, the oscillation linear), and the
crowding tail tracks seed fragility. The governor holds the space open FROM
BIRTH — the projection is preventive, not curative (exact lemma: separating an
already-crowded book leaves its flip pattern bit-identical for content already
associated with it).

Form: a post-optimizer-step PROJECTION, never an auxiliary loss — this repo's
standing law ("no balance machinery; pressure stays out of the task gradient",
model/bank.py) is respected because the governor never touches the gradient.
Hub and head codebooks live natively in their own D-dim address space, so the
plain projective form applies (|cos| on RP^(D-1): the read is exactly
invariant under a -> -a). Identity when slack: the batched check is one
matmul per codebook and nothing is written unless a pair violates.

Verified properties (2026-08-25, gov25m rung): fires only when needed
(bit-identical runs when slack), census tail cut 5.4x, fragile-seed function
+.007, stable-seed +.002 — zero cost, no parameters.
"""
from __future__ import annotations

import math

import torch

from .address import AlephAddress

__all__ = ["minsep_project_", "govern_model"]


@torch.no_grad()
def minsep_project_(codebook: torch.nn.Parameter, theta_min_deg: float,
                    max_iters: int = 8) -> int:
    """Projective min-separation on one (K, D) codebook, in place.

    Bound: |cos(a_i, a_j)| <= cos(theta_min) for every pair. Exact symmetric
    geodesic push-apart, largest violation first, projected to theta_min +
    0.5 deg (the exact-bound push converges asymptotically under fp32 and
    can land ~1e-4 over). Returns the number of pair-pushes performed —
    0 means the call was a pure read (identity, zero writes).
    """
    c_max = math.cos(math.radians(theta_min_deg))
    A = torch.nn.functional.normalize(codebook.data.float(), dim=-1)
    hits = 0
    changed = False
    for _ in range(max_iters):
        G = A @ A.T
        G.fill_diagonal_(0)
        viol = (G.abs() > c_max).nonzero()
        viol = viol[viol[:, 0] < viol[:, 1]]
        if not len(viol):
            break
        order = G.abs()[viol[:, 0], viol[:, 1]].argsort(descending=True)
        for i, j in viol[order].tolist():
            g = float(A[i] @ A[j])
            if abs(g) <= c_max:
                continue
            s = 1.0 if g >= 0 else -1.0
            b = s * A[j]
            gam = math.acos(max(-1.0, min(1.0, float(A[i] @ b))))
            if gam < 1e-6:                      # exact duplicate: nudge first
                b = torch.nn.functional.normalize(
                    b + 1e-3 * torch.randn_like(b), dim=-1)
                gam = math.acos(max(-1.0, min(1.0, float(A[i] @ b))))
            delta = (math.radians(theta_min_deg + 0.5) - gam) / 2
            ai = A[i].clone()
            ti = (b - math.cos(gam) * ai) / math.sin(gam)
            tj = (ai - math.cos(gam) * b) / math.sin(gam)
            A[i] = torch.nn.functional.normalize(
                math.cos(delta) * ai - math.sin(delta) * ti, dim=-1)
            A[j] = s * torch.nn.functional.normalize(
                math.cos(delta) * b - math.sin(delta) * tj, dim=-1)
            hits += 1
            changed = True
    if changed:
        codebook.data.copy_(A.to(codebook.dtype))
    return hits


def _hub_addresses(attn):
    """Every AlephAddress a hub carries (single- or multi-constellation)."""
    if hasattr(attn, "consts"):
        return [c.addr for c in attn.consts]
    addr = getattr(attn, "addr", None)
    return [addr] if isinstance(addr, AlephAddress) else []


@torch.no_grad()
def govern_model(model, theta_min_deg: float,
                 include: tuple = ("hub", "head")) -> int:
    """Batched slack check + projection across a model's governed codebooks.

    Slack path costs one small matmul per codebook and writes nothing.
    Wrapper-aware (amoe BlockWithAdapter moves the block to .block).
    `include`: "hub" = attention codebooks, "head" = DualHead, "bank" =
    dispatch anchors (K=experts in d_model — never crowds in practice, off
    by default).
    """
    c_max = math.cos(math.radians(theta_min_deg))
    books: list[torch.nn.Parameter] = []
    if "hub" in include:
        for wrap in getattr(model, "blocks", []):
            blk = getattr(wrap, "block", wrap)
            if getattr(blk, "is_hub", False):
                books += [a.codebook for a in _hub_addresses(blk.attn)]
    if "head" in include and hasattr(model, "head"):
        addr = getattr(model.head, "addr", None)
        if isinstance(addr, AlephAddress):
            books.append(addr.codebook)
    if "bank" in include:
        for wrap in getattr(model, "blocks", []):
            blk = getattr(wrap, "block", wrap)
            addr = getattr(getattr(blk, "bank", None), "addr", None)
            if isinstance(addr, AlephAddress):
                books.append(addr.codebook)
    # Batched slack check (2026-08-26): group same-shape books into ONE bmm
    # and ONE device sync — the per-book loop was 500+ tiny matmuls each
    # with a .max() sync on the v2 craft. Only violating books ever enter
    # the Python projection.
    hits = 0
    by_shape: dict = {}
    for cb in books:
        by_shape.setdefault(tuple(cb.shape), []).append(cb)
    for shape, group in by_shape.items():
        A = torch.nn.functional.normalize(
            torch.stack([cb.data for cb in group]).float(), dim=-1)
        G = torch.bmm(A, A.transpose(1, 2)).abs()        # (n_books, K, K)
        K = shape[0]
        eye = torch.eye(K, device=G.device, dtype=torch.bool)
        G = G.masked_fill(eye, 0)
        worst = G.amax(dim=(1, 2))                       # one sync for all
        for idx in (worst > c_max).nonzero().flatten().tolist():
            hits += minsep_project_(group[idx], theta_min_deg)
    return hits
