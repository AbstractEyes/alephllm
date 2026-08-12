"""Optimizers — the measured split.

Muon (Newton-Schulz orthogonalized momentum) on the 2D transformer weights,
pure Adam (weight decay 0 — never AdamW) on embeddings, norms, gates,
gamma, positions and every other non-2D parameter. Measured on the aleph
mechanism: momentum-geometric beats Adam by ~.09 and the mechanism is ~20x
more optimizer-sensitive than sdpa — the optimizer is part of the
architecture. Address codebooks are 2D and ride with Muon (the winning
arm's split).

LR policy: optional short linear warmup, then FLAT (schedules measured
no-op-to-harmful on the probe beds; warmup is scale insurance only).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _ns_orth(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization (Muon core), fp32 internally."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tall else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 2e-2, momentum: float = 0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                st = self.state.setdefault(p, {})
                m = st.setdefault("m", torch.zeros_like(p))
                m.mul_(g["momentum"]).add_(p.grad)
                if p.ndim == 2:
                    p.add_(_ns_orth(m), alpha=-g["lr"])
                else:  # stacked expert matrices (K, d, ff): per-slice orth
                    upd = torch.stack([_ns_orth(m[k]) for k in range(m.shape[0])])
                    p.add_(upd, alpha=-g["lr"])


def split_params(model: nn.Module):
    """Muon set: 2D weights of Linear/attention/codebooks + 3D stacked
    expert matrices. Adam set: everything else + all embedding tables (2D
    but excluded — per-row geometry, not transport maps). Covers every
    param exactly once."""
    emb_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            emb_ids.add(id(m.weight))
    muon, adam, seen = [], [], set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim in (2, 3) and id(p) not in emb_ids:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


def build_optimizers(model: nn.Module, muon_lr: float = 2e-2,
                     muon_momentum: float = 0.95, adam_lr: float = 3e-4):
    muon_p, adam_p = split_params(model)
    return [Muon(muon_p, lr=muon_lr, momentum=muon_momentum),
            torch.optim.Adam(adam_p, lr=adam_lr, weight_decay=0.0)]


def lr_scale(step: int, warmup: int) -> float:
    if warmup <= 0 or step >= warmup:
        return 1.0
    return (step + 1) / warmup


def apply_lr(optimizers, base_lrs, step: int, warmup: int):
    s = lr_scale(step, warmup)
    for opt, base in zip(optimizers, base_lrs):
        for g in opt.param_groups:
            g["lr"] = base * s
    return s
