"""AnchoredBank — the E1-form anchored FFN, born on its own null path.

    Bank(x) = trunk(x) + sum_k w_k(x) * sigmoid(g_k) * expert_k(x)

Trunk: always-on d->ff->d GELU expert. Dispatch: the signed aleph address
over a learned K x d codebook read against the layer input (K=3 fat
experts, the shape validated at parity under encoder pressure). Gates
init -3.0; expert OUTPUT projections zero-init, so at birth the dispatch
contributes exactly zero and the bank is bit-identical to its dense
control (the C6 null path). No balance machinery of any kind —
differentiation is an attractor, pressure stays out of the task gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .address import AlephAddress

# 0.10.18 (experiment 3, the byte experts): the seven byte classes of the CURRENT input byte, read by the bank's
# byte-class table. The order is the instruments' (duet_read / byte_exams): lower, upper, digit, space, newline,
# punct (every other ASCII byte, the controls and the specials included), high (0x80 and above).
BYTE_CLASSES = ("lower", "upper", "digit", "space", "newline", "punct", "high")
BYTE_TABLE_MODES = ("", "fixed", "add", "add0")


def byte_class_table() -> torch.Tensor:
    """(256,) long: the class id of every byte value."""
    t = torch.full((256,), 5, dtype=torch.long)
    t[97:123] = 0
    t[65:91] = 1
    t[48:58] = 2
    t[32] = 3
    t[10] = 4
    t[128:] = 6
    return t


def byte_classes(idx: torch.Tensor) -> torch.Tensor:
    """idx (B, n) byte ids -> (B, n) class ids (the table built on idx's device)."""
    return byte_class_table().to(idx.device)[idx]


class AnchoredBank(nn.Module):
    def __init__(self, d: int, n_experts: int = 3, ff: int | None = None,
                 tau: float = 0.1, gate_init: float = -3.0, byte_table: str = ""):
        super().__init__()
        ff = ff or d
        self.n_experts = n_experts
        self.t_in = nn.Linear(d, ff, bias=False)
        self.t_out = nn.Linear(ff, d, bias=False)
        nn.init.orthogonal_(self.t_in.weight)
        nn.init.orthogonal_(self.t_out.weight)
        self.addr = AlephAddress(n_experts, d, tau)
        w_in = torch.empty(n_experts, d, ff)
        for k in range(n_experts):
            nn.init.orthogonal_(w_in[k])
        self.w_in = nn.Parameter(w_in)
        self.w_out = nn.Parameter(torch.zeros(n_experts, ff, d))  # null path
        self.gates = nn.Parameter(torch.full((n_experts,), gate_init))
        self.last_dispatch = None  # (mean|w| per expert, w sample) for instruments
        # 0.10.18: the byte-class table over the dispatch amplitudes. "" = the address alone (the form above, no new
        # key); "fixed" = the amplitudes REPLACED by an installed table (class c -> expert c mod K at amplitude 1; a
        # buffer, not learned); "add" = the address PLUS that table, learned; "add0" = the address plus a zero-born
        # learned table (bit-identical to "" at birth). byte_cls arrives from AlephLM._trunk (the input bytes' classes).
        assert byte_table in BYTE_TABLE_MODES, f"bank byte_table: one of {BYTE_TABLE_MODES}, got {byte_table!r}"
        self.byte_table = byte_table
        if byte_table:
            tab = torch.zeros(len(BYTE_CLASSES), n_experts)
            if byte_table != "add0":
                for c in range(len(BYTE_CLASSES)):
                    tab[c, c % n_experts] = 1.0
            if byte_table == "fixed":
                self.register_buffer("byte_amp", tab)
            else:
                self.byte_amp = nn.Parameter(tab)

    def dispatch(self, x, byte_cls=None):
        """The dispatch amplitudes w (B, n, K) under the bank's table mode."""
        if not getattr(self, "byte_table", ""):
            return self.addr.signed(x)
        assert byte_cls is not None, "the byte-class table needs the input bytes' classes (AlephLM._trunk passes them; decode is owed)"
        tab = self.byte_amp[byte_cls]                             # (B, n, K) fp32
        if self.byte_table == "fixed":
            return tab
        return self.addr.signed(x) + tab

    def forward(self, x, disable_dispatch: bool = False, byte_cls=None):
        trunk = self.t_out(F.gelu(self.t_in(x)))
        if disable_dispatch:
            return trunk
        w = self.dispatch(x, byte_cls)                            # (B, n, K)
        with torch.no_grad():
            self.last_dispatch = w.detach()
        h = F.gelu(torch.einsum("bnd,kdf->bnkf", x, self.w_in))
        e = torch.einsum("bnkf,kfd->bnkd", h, self.w_out)
        return trunk + torch.einsum("bnk,bnkd->bnd",
                                    w * torch.sigmoid(self.gates), e)
