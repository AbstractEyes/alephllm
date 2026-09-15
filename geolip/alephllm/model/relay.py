"""RelayPatchwork + RelayEMA — a gated residual patch module over a signed
address read, and a variant with causal-EMA memory taps.

RelayPatchwork (ported from the amoe-lora adapter of the same name,
github.com/AbstractEyes/amoe-lora, as a native model component):

    slots_t = proj(x_t)                      (B, n, n_slots, D)  orthogonal, no bias
    f_t     = m_hat(slots_t) flattened       the reconstructive signed read
              m_hat(x) = sum_k sinh(u_k) a_hat_k / sum_j cosh(u_j)
                       = signed(x) @ A_hat   (composed from AlephAddress)
    y_t     = x_t + sigmoid(gate) * consume(f_t)

consume = Linear(nD -> hidden), ReLU^2, LayerNorm, Linear(hidden -> d) with the
output layer zero-initialized (weight AND bias), so a fresh module is exactly
inert: y == x. Gate initialized -3. Reconstructive, never comparative: no
softmax-over-choices, no argmax, no top-k.

RelayEMA adds one mechanism: two fixed-decay causal EMAs of the module's own
read, fed to the head through input columns that start at zero —

    F1_t = (1 - r1) F1_{t-1} + r1 f_t        r1 = 1/16
    F2_t = (1 - r2) F2_{t-1} + r2 f_t        r2 = 1/64
    y_t  = x_t + sigmoid(gate) * consume(cat(f_t, F1_t, F2_t))

so at birth RelayEMA's forward equals the memory-free RelayPatchwork. Training
uses the chunked closed-form scan (ema_chunked, renormalized so q^-t never
overflows); decode carries (F1, F2) state one step at a time — exact, because
the head is position-wise.

Defaults are the validated geometry (n_slots 32, K 64, D 4, hidden 256; ~428k
params at d = 1024). Validation record and trained weights:
huggingface.co/AbstractPhil/mini-beatrix-2s, arms/btx_e003.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress


@dataclass
class RelaySpec:
    n_slots: int = 32
    K: int = 64
    D: int = 4
    tau: float = 0.1
    hidden: int = 256
    rho1: float = 1.0 / 16.0
    rho2: float = 1.0 / 64.0
    gate_init: float = -3.0
    zero_init_head: bool = True


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def ema_chunked(f: torch.Tensor, rho: float, s0: torch.Tensor, chunk: int = 256):
    """Causal EMA over dim 1, closed form per chunk (renormalized so q^-t
    never overflows; autograd-safe). F_t = q^t (s0 + rho * sum_{s<=t} q^-s f_s),
    q = 1 - rho; state carried between chunks. -> (F (B,T,C), F_last (B,C))."""
    B, T, C = f.shape
    q = 1.0 - rho
    outs = []
    s = s0
    for a in range(0, T, chunk):
        fb = f[:, a:a + chunk]
        t = fb.shape[1]
        idx = torch.arange(1, t + 1, device=f.device, dtype=f.dtype)
        acc = torch.cumsum(fb * torch.pow(q, -idx).view(1, t, 1), dim=1)
        Fb = torch.pow(q, idx).view(1, t, 1) * (s.unsqueeze(1) + rho * acc)
        s = Fb[:, -1]
        outs.append(Fb)
    return torch.cat(outs, dim=1), s


def _feats(m, x: torch.Tensor) -> torch.Tensor:
    """The reconstructive read, flattened: m_hat = signed(slots) @ A_hat."""
    B, n, _ = x.shape
    slots = m.proj(x).view(B, n, m.n_slots, m.spec.D)
    A = F.normalize(m.addr.codebook, dim=-1)
    return (m.addr.signed(slots) @ A).reshape(B, n, -1)


class RelayPatchwork(nn.Module):
    """The memory-free form: proj -> reconstructive address read ->
    zero-initialized squared-ReLU head, residual write behind a sigmoid gate."""

    def __init__(self, d: int, spec: RelaySpec | None = None):
        super().__init__()
        s = spec or RelaySpec()
        self.spec = s
        self.n_slots = s.n_slots
        self.nD = s.n_slots * s.D
        self.proj = nn.Linear(d, self.nD, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(s.K, s.D, s.tau)
        self.consume = nn.Sequential(
            nn.Linear(self.nD, s.hidden), SquaredReLU(),
            nn.LayerNorm(s.hidden), nn.Linear(s.hidden, d))
        if s.zero_init_head:
            nn.init.zeros_(self.consume[-1].weight)
            nn.init.zeros_(self.consume[-1].bias)
        self.gate = nn.Parameter(torch.tensor(float(s.gate_init)))

    def feats(self, x: torch.Tensor) -> torch.Tensor:
        return _feats(self, x)

    def forward(self, x):
        return x + torch.sigmoid(self.gate) * self.consume(self.feats(x))


class RelayEMA(nn.Module):
    """RelayPatchwork plus the EMA memory taps. Built by widening a patchwork's
    head to 3nD input columns with the added columns zeroed, so at birth the
    forward equals the patchwork it came from (shared weights)."""

    def __init__(self, d: int, spec: RelaySpec | None = None):
        super().__init__()
        self._widen(RelayPatchwork(d, spec))

    @classmethod
    def from_patchwork(cls, base: RelayPatchwork) -> "RelayEMA":
        """Upgrade a (possibly trained) RelayPatchwork in place: shares its
        proj/addr/gate and head tail, widens the head's first layer with
        zeroed columns."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._widen(base)
        return self

    def _widen(self, base: RelayPatchwork):
        self.spec = base.spec
        self.nD = base.nD
        self.proj, self.addr, self.gate = base.proj, base.addr, base.gate
        self.n_slots = base.n_slots
        wide = nn.Linear(3 * self.nD, base.spec.hidden)
        with torch.no_grad():
            wide.weight[:, :self.nD] = base.consume[0].weight
            wide.weight[:, self.nD:] = 0.0
            wide.bias.copy_(base.consume[0].bias)
        self.consume = nn.Sequential(
            wide, base.consume[1], base.consume[2], base.consume[3])

    def feats(self, x: torch.Tensor) -> torch.Tensor:
        return _feats(self, x)

    def run(self, x, state=None):
        """-> (y = x + write, (F1_last, F2_last)). state None starts both EMAs
        at zero (fresh context). Decode: call on the single new position with
        the carried state — exact, the head is position-wise."""
        f = self.feats(x)
        B = f.shape[0]
        s1 = state[0] if state is not None else f.new_zeros(B, self.nD)
        s2 = state[1] if state is not None else f.new_zeros(B, self.nD)
        F1, s1 = ema_chunked(f, self.spec.rho1, s1)
        F2, s2 = ema_chunked(f, self.spec.rho2, s2)
        y = x + torch.sigmoid(self.gate) * self.consume(
            torch.cat([f, F1, F2], dim=-1))
        return y, (s1, s2)

    def forward(self, x, state=None):
        return self.run(x, state)[0]
