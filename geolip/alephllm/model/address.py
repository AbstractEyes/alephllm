"""AlephAddress — the closed-form signed address over 2K oriented half-axes.

The mechanism is reconstructive, never comparative: no softmax-over-choices,
no argmax, no top-k anywhere. Dispatch weights are
    w_k = sinh(u_k) / sum_j cosh(u_j),   u_k = cos(x_hat, a_hat_k) / tau
which is exactly the signed difference of the two halves of a 2K-softmax
over oriented axes (+a_k, -a_k). Inhibition (negative w) is first-class.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dtype_floor(t: torch.Tensor) -> float:
    """Dtype-aware clamp floor. Half dtypes flush 1e-12 to zero — the
    measured fp16 landmine; use a floor the dtype can actually represent."""
    if t.dtype in (torch.float32, torch.float64):
        return 1e-12
    return float(torch.finfo(t.dtype).tiny) * 8


class AlephAddress(nn.Module):
    """K unit anchors in D dims, cosine-read at temperature tau.

    signed(x)  -> (..., K)          w_k = sinh(u_k)/sum_j cosh(u_j)
    oriented(x)-> ((..., K), (..., K))  the two positive halves of the
                                    2K-softmax (ep/Z, en/Z); HUB feature map.
    """

    def __init__(self, K: int, D: int, tau: float = 0.1):
        super().__init__()
        self.K, self.D, self.tau = K, D, tau
        self.codebook = nn.Parameter(F.normalize(torch.randn(K, D), dim=-1))
        self.register_buffer("home", self.codebook.detach().clone())

    def _u(self, x: torch.Tensor) -> torch.Tensor:
        A = F.normalize(self.codebook, dim=-1)
        return (F.normalize(x, dim=-1) @ A.transpose(-1, -2)) / self.tau

    def oriented(self, x: torch.Tensor):
        u = self._u(x)
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        Z = (ep + en).sum(dim=-1, keepdim=True)
        return ep / Z, en / Z

    def signed(self, x: torch.Tensor) -> torch.Tensor:
        u = self._u(x)
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        return (ep - en) / (ep + en).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def health(self, x_sample: torch.Tensor) -> dict:
        """Codebook + consumption vitals for the instrument suite."""
        A = F.normalize(self.codebook.float(), dim=-1)
        gram = A @ A.T
        off = gram - torch.eye(self.K, device=gram.device)
        drift = 1.0 - F.cosine_similarity(
            A, F.normalize(self.home.float(), dim=-1), dim=-1)
        s = torch.linalg.svdvals(A)
        ps = (s * s) / (s * s).sum().clamp_min(1e-12)
        out = {
            "anchor_max_abs_cos": off.abs().max().item(),
            "anchor_merge_pairs": int((off.abs() > 0.99).sum().item() // 2),
            "drift_mean": drift.mean().item(),
            "drift_max": drift.max().item(),
            # frame health: effective rank of the codebook itself — a
            # collapsing frame (anchors folding into a subspace) shows here
            "codebook_erank": float(
                torch.exp(-(ps * ps.clamp_min(1e-12).log()).sum()).item()),
        }
        if x_sample is not None:
            p, n = self.oriented(x_sample.reshape(-1, x_sample.shape[-1]).float())
            mass = torch.cat([p, n], dim=-1).mean(0)          # (2K,) mean usage
            mass = mass / mass.sum().clamp_min(1e-12)
            ent = -(mass * mass.clamp_min(1e-12).log()).sum()
            out["usage_ppl"] = float(ent.exp().item())         # of 2K half-axes
            out["usage_cv"] = float((mass.std() / mass.mean().clamp_min(1e-12)).item())
            w = self.signed(x_sample.reshape(-1, x_sample.shape[-1]).float())
            out["sign_frac_neg"] = float((w < 0).float().mean().item())
        return out
