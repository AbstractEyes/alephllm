"""Head revival — the BOUNDARY-WRITE op for a buried aleph head.

MEASURED BASIS (claude-mind repos/alephllm.md RIDERS 11-12, 2026-08-27):
a born-null aleph head racing a strong same-target linear base self-buries
within ~2k steps (2/2 crafts: v1 shipped with W_s == 0.0 exactly; run1's
proj rotated to 60x-below-chance codebook orthogonality). LS-on-buried-
reads is refuted (0.0006 bpb); revival must reconstruct the ADDRESS.

THE OP (deterministic — every step a solve, no gradient hoping):
  proj     <- whitening-PCA of the head inputs, fit UNCENTERED (the proj
              has no bias: fit distribution == deploy distribution),
              relative singular-value floor sv[0]*1e-3
  codebook <- fresh-birth normalize(randn) + min-sep 45 deg (Form-10
              discharged: k-means seeding buys nothing measurable at
              birth over constructed init — and stays off the k-means
              failure class)
  W_s      <- fp64 ridge on the base head's CE residual (onehot - p),
              best-alpha from an extended sweep folded into the install
Afterwards the caller FREEZES proj + codebook (TrainConfig
head_addr_frozen) so the burial channel is structurally closed; only W_s
trains. Trial verdicts: liveness restored to ~3.3x chance, w_s/w_h grad
ratio 1.4e-5 -> 6.9e-3, crater +0.0000 (safe non-atomic), instant
extraction ~millibits (the aleph head can only EARN function nonlinearly
over training — the toggle is the experiment from here).

LAW: BOUNDARY-WRITE — task-signal-bearing closed-form writes are
permitted only at run boundaries (phase boundaries or halted-session
resume points), never inside a running session, and always logged with
provenance (what was solved, on which data, installed norms).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..model.governor import minsep_project_

__all__ = ["revive_head"]


@torch.no_grad()
def revive_head(model, H: torch.Tensor, base_logits: torch.Tensor,
                targets: torch.Tensor, theta_deg: float = 45.0,
                seed: int = 13, sv_floor_rel: float = 1e-3,
                ridge_rel: float = 1e-3) -> dict:
    """Revive model.head in place. H: (N, d) head inputs (nf outputs);
    base_logits: (N, vocab) with the aleph path OFF; targets: (N,) ids.
    First half fits, second half evaluates. CPU/GPU agnostic; fp32/64
    math. Returns the provenance dict (log it — the law requires it)."""
    head = model.head
    dev = H.device
    K, D = head.addr.codebook.shape
    n = H.shape[0] // 2

    # -- proj: uncentered whitening-PCA (relative sv floor)
    Hf = H[:min(n, 16384)].float()
    _, sv, Vh = torch.linalg.svd(Hf, full_matrices=False)
    proj_w = Vh[:D] / sv[:D, None].clamp_min(float(sv[0]) * sv_floor_rel)

    # -- codebook: fresh-birth + min-sep (governed from birth)
    g = torch.Generator().manual_seed(seed)
    book = torch.nn.Parameter(
        F.normalize(torch.randn(K, D, generator=g), dim=-1).to(dev))
    pushes = minsep_project_(book, theta_deg)

    head.proj.weight.data.copy_(proj_w.to(head.proj.weight.dtype))
    head.addr.codebook.data.copy_(book.data.to(head.addr.codebook.dtype))

    # -- live reads (fp32 — the bf16 read floor is 18% of a buried signal)
    S = head.addr.signed(head.proj(H.float())).float()
    s_norm = float(S.norm(dim=1).mean())
    sm = S.mean(0)
    relvar = float((S - sm).norm() / S.norm().clamp_min(1e-12))

    # -- W_s: fp64 ridge on the CE residual, best alpha folded in
    Sf = S[:n].double()
    P = F.softmax(base_logits[:n].double(), dim=-1)
    R = -P
    R[torch.arange(n, device=dev), targets[:n]] += 1.0
    G = Sf.T @ Sf
    lam = ridge_rel * float(G.diagonal().mean().clamp_min(1e-12))
    W = torch.linalg.solve(
        G + lam * torch.eye(G.shape[0], dtype=torch.float64, device=dev),
        Sf.T @ R).T.float()
    base_bpb = F.cross_entropy(base_logits[n:], targets[n:]).item() / math.log(2)
    best = (0.0, base_bpb)
    for a in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        bpb = F.cross_entropy(base_logits[n:] + a * (S[n:] @ W.T),
                              targets[n:]).item() / math.log(2)
        if bpb < best[1]:
            best = (a, bpb)
    alpha, bpb = best
    head.w_s.weight.data.copy_((alpha * W).to(head.w_s.weight.dtype))

    return {"op": "head_revival", "K": K, "D": D, "theta_deg": theta_deg,
            "seed": seed, "minsep_pushes": pushes,
            "sv0": float(sv[0]), "svD": float(sv[D - 1]),
            "s_norm": s_norm, "s_relvar": relvar,
            "alpha": alpha, "w_s_norm": float(head.w_s.weight.norm()),
            "fit_positions": n, "base_bpb": base_bpb, "revived_bpb": bpb,
            "delta_bpb": bpb - base_bpb,
            "alpha_at_boundary": alpha in (0.25, 32.0)}
