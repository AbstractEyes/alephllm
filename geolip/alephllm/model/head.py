"""DualHead — the standard readout plus an aleph read, born null.

    logits = W_h h + W_s s(h),        W_s == 0 at init

s(h) is the signed address of a learned low-D projection of h through a
K_h-anchor head codebook. New structure enters at zero and earns its way
in by gradient — via WEIGHT-zero init, never gate-zero: mini-beatrix-1
measured (8.4B tokens) that a gamma-gated branch DEADLOCKS (W_s grad is
scaled by gamma=0, gamma grad through a random frozen readout is
zero-mean noise), while the bank's E_out weight-zero self-started into
+2 bpb of function. gamma remains as a frozen scalar for checkpoint
compatibility and as the ablation knob; the election gauge is ||W_s||
and the head_aleph_off toggle.

THE DIAL FORMS (0.10.18 rc11; AlephLMConfig.head_mode). The aleph read MODULATES the readout's logits instead of adding to them.
After the final LayerNorm every byte's state has one size, so the readout's logit scale per byte (the prediction's confidence) is
the one function a linear readout cannot write from h; a plain craft builds it by spending a share of each byte on one direction
the readout reads weakly (the big final-layer direction), at the cost of the content's room. The dial gives that function to the
aleph read:

    dial      logits = (W_h h + b) * exp(beta * tanh(a - a_bar)),   a = w_t . s(h),  w_t in R^K born zero (the dial at exactly 1);
              beta = ln(dial_band): the scale stays within [1/band, band]; a_bar = the detached running mean of a over the training
              forwards (a buffer, frozen in eval): the dial carries the per-byte variation, the readout the overall scale
    dialraw   the same without the centering (a_bar stays 0)
    dialplus  the dial over the readout PLUS the carrying read:  (W_h h + b + W_s s(h)) * exp(beta * tanh(a - a_bar))
    pergain   a per-byte gain:  (W_h h + b) * exp(beta * tanh(G s(h) - g_bar)),  G in R^{V x K} born zero, g_bar the running mean per byte

In dial / dialraw / pergain the carrying read W_s stays in the module at zero and frozen (the read modulates, never carries). A
plain head's state loads into any dial form with the dial's keys missing (strict=False); dial / dialraw / dialplus and pergain do not
load into each other (the running mean's shape differs), and a trained plain head's W_s is not zero (a dial form's forward then drops
it unless the form is dialplus). At birth every form is the plain readout bit for bit (exp(0) = 1). disable_aleph returns the plain
readout W_h h + b in every form: for a dial form that is the dial held at 1, so the head_aleph_off toggle reads the dial's function in
bits. readout() is the prediction with the dial held at 1 (the carrying read kept where it trains); dial_hold (a runtime attribute,
not state) holds the dial's log scale at a given value, e.g. its own mean, so a read can separate the dial's per-byte part from an
overall temperature. The dial is computed in fp32 outside autocast and the dialed logits are fp32, as the plain path's are.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress

HEAD_MODES = ("", "dial", "dialraw", "dialplus", "pergain")


class DualHead(nn.Module):
    def __init__(self, d: int, vocab: int, K: int = 512, D: int = 32,
                 tau: float = 0.1, tied_weight: nn.Parameter | None = None,
                 mode: str = "", dial_band: float = 4.0, dial_rho: float = 1 / 64):
        super().__init__()
        assert mode in HEAD_MODES, f"head_mode: one of {HEAD_MODES}, got {mode!r}"
        assert dial_band > 1.0 and 0.0 < dial_rho <= 1.0, (dial_band, dial_rho)
        self.mode = mode
        self.tied = tied_weight is not None
        if self.tied:
            # tuple wrapper: reference the embedding table WITHOUT registering
            # it as a duplicate parameter (state_dict must stay alias-free
            # for safetensors export)
            self._tied_ref = (tied_weight,)
            self.bias = nn.Parameter(torch.zeros(vocab))
        else:
            self.w_h = nn.Linear(d, vocab, bias=True)
        self.proj = nn.Linear(d, D, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(K, D, tau)
        self.w_s = nn.Linear(K, vocab, bias=False)
        nn.init.zeros_(self.w_s.weight)          # weight-zero: self-starting
        self.gamma = nn.Parameter(torch.ones(1))  # frozen; ablation knob only
        self.gamma.requires_grad_(False)
        self.beta, self.rho = math.log(float(dial_band)), float(dial_rho)
        if mode in ("dial", "dialraw", "dialplus"):
            self.w_t = nn.Parameter(torch.zeros(K))              # the dial's read of the address: born at zero -> the dial at 1
            self.register_buffer("a_bar", torch.zeros(()))
        elif mode == "pergain":
            self.w_g = nn.Linear(K, vocab, bias=False)
            nn.init.zeros_(self.w_g.weight)
            self.register_buffer("a_bar", torch.zeros(vocab))
        if mode in ("dial", "dialraw", "pergain"):
            self.w_s.weight.requires_grad_(False)                # the carrying read: at zero, frozen (the read modulates)
        self.dial_hold = None                                    # a runtime knob, not state: the dial's log scale held at this value

    @property
    def dial_weight(self) -> torch.Tensor | None:
        """The dial's learned read of the address (w_t, or G's weight for pergain); None for the plain head."""
        if self.mode == "":
            return None
        return self.w_g.weight if self.mode == "pergain" else self.w_t

    def _base(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return h @ self._tied_ref[0].T + self.bias
        return self.w_h(h)

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """The prediction with the dial held at 1: W_h h + b, plus the carrying read where it trains (the plain head, dialplus)."""
        base = self._base(h)
        if self.mode in ("", "dialplus"):
            base = base + self.gamma * self.w_s(self.addr.signed(self.proj(h)))
        return base

    def dial_log(self, s: torch.Tensor) -> torch.Tensor:
        """The dial's log scale (fp32, computed outside autocast) from the signed address s: beta * tanh(a - a_bar); (..., 1) for the
        scalar dial, (..., V) for pergain. In a training forward that builds a graph the running mean a_bar takes one detached step
        toward this forward's mean before the read (never in eval, never under no_grad)."""
        with torch.autocast(device_type=s.device.type, enabled=False):
            s32 = s.float()
            if self.mode == "pergain":
                a = F.linear(s32, self.w_g.weight.float())                   # (..., V)
                m = a.detach().reshape(-1, a.shape[-1]).mean(0)               # (V,)
            else:
                a = (s32 @ self.w_t.float()).unsqueeze(-1)                    # (..., 1)
                m = a.detach().mean()                                         # ()
            if self.mode != "dialraw":
                if self.training and torch.is_grad_enabled():
                    with torch.no_grad():
                        self.a_bar.mul_(1.0 - self.rho).add_(self.rho * m)
                a = a - self.a_bar.float()
            return self.beta * torch.tanh(a)

    def forward(self, h, disable_aleph: bool = False):
        base = self._base(h)
        if disable_aleph:
            return base
        s = self.addr.signed(self.proj(h))
        if self.mode == "":
            return base + self.gamma * self.w_s(s)
        if self.mode == "dialplus":
            base = base + self.gamma * self.w_s(s)
        if self.dial_hold is not None:
            held = torch.as_tensor(self.dial_hold, dtype=torch.float32, device=base.device)
            return base.float() * torch.exp(held)
        return base.float() * torch.exp(self.dial_log(s))

    @torch.no_grad()
    def dial_stats(self, h: torch.Tensor) -> dict:
        """The dial on the states h, read in eval (a_bar frozen): the mean and spread of the log scale, the saturated fraction
        (|tanh| > 0.95), the running mean's size, the dial weights' size. {} for the plain head."""
        if self.mode == "":
            return {}
        was = self.training
        self.eval()
        try:
            u = self.dial_log(self.addr.signed(self.proj(h))) / self.beta       # tanh(a - a_bar) in (-1, 1)
        finally:
            self.train(was)
        return {"dial_log_mean": float(self.beta * u.mean()), "dial_log_std": float(self.beta * u.std()),
                "dial_sat": float((u.abs() > 0.95).float().mean()), "dial_center": float(self.a_bar.float().abs().mean()),
                "dial_w_norm": float(self.dial_weight.detach().float().norm())}
