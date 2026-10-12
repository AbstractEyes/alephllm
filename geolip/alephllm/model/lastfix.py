"""LastWriteFix — the last block's always-on write under a bound (0.10.18, 2026-10-12; the big final-layer direction).

THE FINDING THIS ANSWERS: the last block's bank writes one fixed direction into every byte (94-99% of each byte's energy
in v3, 98.6-99.0% in the 2s crafts, formed by 20,219 of 61,422 steps), the per-byte size grows ~2.7x over 25,000 steps
without a ceiling, and the final LayerNorm fights it with an anti-aligned bias and falling gains (journal 2026-10-12
00:2x). The design of record: plans/2026-10-03_v4_fractal_attractor_discussion.md row A (the amplitude register) and
its companions; this module is their first implementation. Nothing here normalizes over positions except the one
comparison arm built to be refuted ("normwrite").

The fix wraps the LAST block's bank output w_t (B, n, d). The carrier direction b is NOT learned: it is the slowly
tracked mean write direction (a detached EMA across optimizer steps), so the model cannot route the growth around it
by moving the direction. a_t = <w_t, b> is the carrier amplitude per byte, c_t = w_t - a_t b the content off it.
  register     w'_t = c_t + gamma * a_t * b,  gamma = clamp(a_ref / a_bar, lo, hi), a_bar the detached EMA of mean_t a_t
               per optimizer step, a_ref = a_bar frozen at the end of the warm-in; the SAME gamma for every byte of a
               step (per-byte variation preserved; the bound acts across steps). The head reads the carrier through its
               own small path: logits += g(a_t / a_bar), g = W2 gelu(W1 s + c), W2 zero-born.
  temperature  w'_t = c_t (the carrier removed from the stream) and the head's logits are divided per byte by
               tau_t = clamp(softplus(theta0 + theta1 * a_t / a_bar), lo, hi), theta0 = softplus^-1(1), theta1 = 0 at
               birth: the carrier becomes an explicit per-byte temperature.
  gain         w'_t = gamma * w_t, gamma = clamp(n_ref / n_bar, lo, hi), n_bar the detached EMA of the mean per-byte
               norm of w (the S-305 growth form on the whole write; no decomposition).
  normwrite    the comparison arm: w' = (w - mu) / sigma * s + m with CAUSAL cumulative per-channel statistics over the
               positions of each row (the illegal bound of the falsifier; decode owed).
Eval mode freezes every statistic. The flag off builds no module: the model is the 0.10.17 form bit for bit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("", "register", "temperature", "gain", "normwrite")


class LastWriteFix(nn.Module):
    def __init__(self, d: int, vocab: int, mode: str, rho: float = 1 / 16, warm: int = 200,
                 clamp: tuple = (0.25, 4.0), head_hidden: int = 16):
        super().__init__()
        assert mode in MODES[1:], f"bank_last_fix: one of {MODES[1:]}, got {mode!r}"
        self.mode, self.rho, self.warm = mode, float(rho), int(warm)
        self.lo, self.hi = float(clamp[0]), float(clamp[1])
        self.register_buffer("b_bar", torch.zeros(d))          # the tracked mean write direction (unit at read)
        self.register_buffer("a_bar", torch.zeros(()))         # the carrier's (or the norm's) slow mean
        self.register_buffer("a_ref", torch.zeros(()))         # the reference, frozen at the end of the warm-in
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.last_a = None                                     # (B, n) the carrier of the latest forward (fp32)
        if mode == "register":
            self.g1 = nn.Linear(1, head_hidden)
            self.g2 = nn.Linear(head_hidden, vocab, bias=False)
            nn.init.zeros_(self.g2.weight)                     # zero-born: the head path earns its way in
        elif mode == "temperature":
            self.theta = nn.Parameter(torch.tensor([math.log(math.e - 1.0), 0.0]))   # softplus(theta0) = 1
        elif mode == "normwrite":
            self.s = nn.Parameter(torch.ones(d))
            self.m = nn.Parameter(torch.zeros(d))

    # ------------------------------------------------------------------ the slow statistics
    @torch.no_grad()
    def _track(self, w32, a=None):
        """One training forward: the EMA of the mean direction and of the carrier mean (or the norm mean)."""
        if self.mode in ("register", "temperature"):
            mean_dir = F.normalize(w32.mean(dim=(0, 1)), dim=0)
            stat = a.mean()
        else:
            mean_dir = F.normalize(w32.mean(dim=(0, 1)), dim=0)
            stat = w32.norm(dim=-1).mean()
        if int(self.n_updates) == 0:
            self.b_bar.copy_(mean_dir)
            self.a_bar.copy_(stat)
        else:
            self.b_bar.mul_(1 - self.rho).add_(self.rho * mean_dir)
            self.a_bar.mul_(1 - self.rho).add_(self.rho * stat)
        self.n_updates += 1
        if int(self.n_updates) == self.warm:
            self.a_ref.copy_(self.a_bar)

    def gamma(self) -> torch.Tensor:
        """The bounded detached gain; exactly 1 through the warm-in."""
        if int(self.n_updates) < self.warm or float(self.a_ref) <= 0:
            return torch.ones((), device=self.a_bar.device)
        return (self.a_ref / self.a_bar.clamp_min(1e-12)).clamp(self.lo, self.hi)

    def direction(self) -> torch.Tensor:
        return F.normalize(self.b_bar, dim=0)

    # ------------------------------------------------------------------ the write
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        dt = w.dtype
        w32 = w.float()
        if self.mode == "normwrite":
            # causal cumulative per-channel standardization over the row's positions (position t sees 0..t)
            n = w32.shape[1]
            cnt = torch.arange(1, n + 1, device=w.device, dtype=torch.float32)[None, :, None]
            mu = w32.cumsum(1) / cnt
            var = (w32 * w32).cumsum(1) / cnt - mu * mu
            out = (w32 - mu) / (var.clamp_min(0) + 1e-6).sqrt() * self.s + self.m
            if self.training:
                self._track(w32.detach())
            self.last_a = None
            return out.to(dt)
        if self.mode == "gain":
            if self.training:
                self._track(w32.detach())
            self.last_a = w32.detach().norm(dim=-1)
            return (w32 * self.gamma()).to(dt)
        if self.training:                                     # one slow update per training forward
            wd = w32.detach()
            seed_dir = (F.normalize(wd.mean(dim=(0, 1)), dim=0) if int(self.n_updates) == 0 else self.direction())
            self._track(wd, wd @ seed_dir)
        b = self.direction()
        a = w32 @ b                                           # (B, n) the carrier per byte
        c = w32 - a.unsqueeze(-1) * b                         # the content off the carrier
        self.last_a = a.detach()
        if self.mode == "temperature":
            return c.to(dt)
        return (c + (self.gamma() * a).unsqueeze(-1) * b).to(dt)

    # ------------------------------------------------------------------ the head
    def head_adjust(self, logits: torch.Tensor, a: torch.Tensor | None = None) -> torch.Tensor:
        """register: logits + g(a / a_bar); temperature: logits / tau(a); gain / normwrite: unchanged.
        a defaults to the latest forward's carrier; pass a slice for a prefill's last position."""
        if self.mode in ("gain", "normwrite"):
            return logits
        a = self.last_a if a is None else a
        assert a is not None and a.shape == logits.shape[:2], (None if a is None else tuple(a.shape), tuple(logits.shape))
        s = (a / self.a_bar.clamp_min(1e-12)).unsqueeze(-1).to(logits.dtype)     # O(1) input, a_bar detached
        if self.mode == "register":
            return logits + self.g2(F.gelu(self.g1(s)))
        tau = F.softplus(self.theta[0] + self.theta[1] * s).clamp(self.lo, self.hi)
        return logits / tau

    @torch.no_grad()
    def stats(self) -> dict:
        out = dict(mode=self.mode, n_updates=int(self.n_updates), a_bar=float(self.a_bar), a_ref=float(self.a_ref),
                   gamma=float(self.gamma()))
        if self.mode == "temperature":
            out["tau_at_mean"] = float(F.softplus(self.theta[0] + self.theta[1]).clamp(self.lo, self.hi))
        return out
