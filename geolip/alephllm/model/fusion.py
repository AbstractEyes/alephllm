"""Weak-token fusion at the input plane (v3).

A byte STARTS A UNIT when the atlas table says it was a choice point — the
next-byte entropy given the three previous bytes is at or above `theta`
bits — or when its trigram cell is unwitnessed, or when it is a special
byte or follows one, or when it is a newline or follows one (so a turn-end
newline pair is two units and nothing fuses across it). Every other byte
is WEAK and fuses into the open unit. Every decision reads bytes at or
before the byte itself, so the rule is causal and the decode path can
apply it one byte at a time.

The trunk keeps its front and back blocks at byte resolution; the middle
blocks run over units (the hourglass form). Pooling is a position read —
a unit's vector is the front's residual state at the unit's last byte —
never a mean over positions. Unpooling is shifted by one unit: a byte
reads the middle's output for the last unit completed before it, so no
byte sees a unit that contains it.

The zero-parameter comparator is the SPACELIKE rule: a word byte after a
non-word byte starts a unit (SpaceByte, arXiv 2404.14408), with the same
forced starts.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from ..data.special_tokens import SPECIALS

UNWITNESSED = 255          # table code for a cell below the witness floor
NEWLINE = 10
_TABLE_CACHE: dict = {}


def load_code(path: str, witness_floor: float) -> np.ndarray:
    """The 256^3 uint8 table: entropy in 1/16-bit steps (clipped to 254),
    UNWITNESSED where the raw witness count is below the floor."""
    key = (os.path.abspath(path), float(witness_floor))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    z = np.load(path)
    code = np.minimum(z["entropy_x16"].astype(np.uint8), 254)
    code[z["witness"] < witness_floor] = UNWITNESSED
    _TABLE_CACHE[key] = code
    return code


class FusionPlan:
    """Per-batch unit structure derived from the start mask."""

    def __init__(self, starts: torch.Tensor):
        B, T = starts.shape
        self.starts = starts
        self.uid = starts.long().cumsum(1) - 1                  # (B, T) 0-based unit index
        self.n_units = self.uid[:, -1] + 1                        # (B,)
        self.J = int(self.n_units.max())
        pos = torch.arange(T, device=starts.device).expand(B, T)
        self.last = torch.zeros(B, self.J, dtype=torch.long, device=starts.device) \
            .scatter_reduce(1, self.uid, pos, reduce="amax", include_self=False)

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d) -> (B, J, d): each unit's vector is the state at its last byte."""
        return h.gather(1, self.last.unsqueeze(-1).expand(-1, -1, h.shape[-1]))

    def unpool(self, v: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
        """(B, J, d) -> (B, T, d): byte t receives the output of the unit
        completed before its own (the null vector before the first)."""
        B, J, d = v.shape
        shifted = torch.cat([null.expand(B, 1, d).to(v.dtype), v[:, :-1]], dim=1)
        return shifted.gather(1, self.uid.unsqueeze(-1).expand(-1, -1, d))


class Fusion(nn.Module):
    def __init__(self, spec: dict, d_model: int):
        super().__init__()
        self.rule = spec.get("rule", "entropy")
        self.theta = float(spec.get("theta", 1.0))
        self.theta_code = int(min(254, round(self.theta * 16)))
        self.k_lo = int(spec.get("k_lo", 2))
        self.k_hi = int(spec.get("k_hi", 2))
        self.witness_floor = float(spec.get("witness_floor", 8))
        self.table = spec.get("table")
        special = torch.zeros(256, dtype=torch.bool)
        for s in SPECIALS:
            special[s] = True
        alnum = torch.zeros(256, dtype=torch.bool)
        for b in range(256):
            alnum[b] = (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or (0x80 <= b < 0xF5)
        self.register_buffer("special", special, persistent=False)
        self.register_buffer("alnum", alnum, persistent=False)
        if self.rule in ("entropy", "hybrid"):
            assert self.table, f"the {self.rule} rule needs an atlas table (npz with entropy_x16 + witness)"
            code = torch.from_numpy(load_code(self.table, self.witness_floor))
            self.register_buffer("code", code, persistent=False)
        elif self.rule != "spacelike":
            raise ValueError(f"unknown fusion rule {self.rule!r}")
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))

    # ------------------------------------------------------------ the rule
    def forced(self, idx: torch.Tensor) -> torch.Tensor:
        sp = self.special[idx]
        nl = idx == NEWLINE
        f = sp | nl
        f[:, 1:] |= sp[:, :-1] | nl[:, :-1]
        return f

    def starts(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) bool: True where a unit starts (a boundary before the byte)."""
        B, T = idx.shape
        st = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        st[:, 0] = True
        if self.rule == "entropy":
            st[:, :min(3, T)] = True
            if T > 3:
                c = idx[:, :-3] * 65536 + idx[:, 1:-2] * 256 + idx[:, 2:-1]
                code = self.code[c]
                st[:, 3:] |= (code >= self.theta_code) | (code == UNWITNESSED)
        else:
            al = self.alnum[idx]
            st[:, 1:] |= al[:, 1:] & ~al[:, :-1]
            if self.rule == "hybrid":
                # a word whose first byte was predictable fuses into the unit before it
                keep = torch.ones(B, T, dtype=torch.bool, device=idx.device)
                if T > 3:
                    c = idx[:, :-3] * 65536 + idx[:, 1:-2] * 256 + idx[:, 2:-1]
                    code = self.code[c]
                    keep[:, 3:] = (code >= self.theta_code) | (code == UNWITNESSED)
                st = st & keep
                st[:, 0] = True
        return st | self.forced(idx)

    def starts_step(self, next_id, prev1, prev2, prev3, pad_row: int) -> torch.Tensor:
        """The same rule for one new byte per row; prev* are the last three
        bytes (pad_row where the history is shorter)."""
        sp = self.special[next_id] | self.special[prev1.clamp(max=255)] & (prev1 != pad_row)
        nl = (next_id == NEWLINE) | (prev1 == NEWLINE)
        st = sp | nl
        if self.rule == "entropy":
            short = (prev1 == pad_row) | (prev2 == pad_row) | (prev3 == pad_row)
            c = prev3.clamp(max=255) * 65536 + prev2.clamp(max=255) * 256 + prev1.clamp(max=255)
            code = self.code[c]
            st |= short | (code >= self.theta_code) | (code == UNWITNESSED)
        else:
            word = self.alnum[next_id] & ~(self.alnum[prev1.clamp(max=255)] & (prev1 != pad_row))
            if self.rule == "hybrid":
                short = (prev1 == pad_row) | (prev2 == pad_row) | (prev3 == pad_row)
                c = prev3.clamp(max=255) * 65536 + prev2.clamp(max=255) * 256 + prev1.clamp(max=255)
                code = self.code[c]
                word = word & (short | (code >= self.theta_code) | (code == UNWITNESSED))
            st |= word
        return st

    def plan(self, idx: torch.Tensor) -> FusionPlan:
        return FusionPlan(self.starts(idx))


# ------------------------------------------------------------ cache helpers
def cat_caches(rows: list):
    """Merge per-row decode caches (dicts / lists of tensors) along the batch dim."""
    first = rows[0]
    if isinstance(first, dict):
        return {k: cat_caches([r[k] for r in rows]) for k in first}
    if isinstance(first, (list, tuple)):
        return [cat_caches([r[i] for r in rows]) for i in range(len(first))]
    return torch.cat(list(rows), dim=0)


def zero_cache_rows(cache, mask: torch.Tensor):
    """Zero every cache tensor for the rows where mask is True (an empty prefix state)."""
    if isinstance(cache, dict):
        for k in cache:
            cache[k] = zero_cache_rows(cache[k], mask)
        return cache
    if isinstance(cache, (list, tuple)):
        return [zero_cache_rows(c, mask) for c in cache]
    m = mask.view(-1, *([1] * (cache.dim() - 1))).to(cache.device)
    return torch.where(m, torch.zeros_like(cache), cache)


def snapshot(cache):
    if isinstance(cache, dict):
        return {k: snapshot(v) for k, v in cache.items()}
    if isinstance(cache, (list, tuple)):
        return [snapshot(c) for c in cache]
    return cache.clone()


def restore_rows(cache, old, keep_new: torch.Tensor):
    """Keep the stepped cache for rows where keep_new is True; restore the snapshot elsewhere."""
    if isinstance(cache, dict):
        for k in cache:
            cache[k] = restore_rows(cache[k], old[k], keep_new)
        return cache
    if isinstance(cache, (list, tuple)):
        return [restore_rows(c, o, keep_new) for c, o in zip(cache, old)]
    m = keep_new.view(-1, *([1] * (cache.dim() - 1))).to(cache.device)
    return torch.where(m, cache, old)
