"""Bits by position (0.10.18, 2026-10-12): the held-out bits a byte within POSITION BANDS of the window, from the same per-byte
losses as the val bpb, so a craft's use of depth reads directly beside its mean bits.

Two frames, both from one tensor of losses:
  row  - the target's distance from the ROW start (does the craft read cheaper deeper into its window?)
  doc  - the target's distance since the last DOC byte in packed rows (the same question per document; a target with no DOC
         before it in the row is excluded, the DOC targets themselves are excluded, as the DOC-boundary gauge excludes them)
Bands are [lo, hi) in bytes, powers of two by default: 0-64, 64-256, 256-1024, 1024-2048, 2048-4096, 4096+. A band with no
target in it is reported as None with count 0. Everything is in BITS (nats / ln 2).

    bits_by_position(nll, doc_mask=None, bands=BANDS) -> {"row": {band: bits}, "row_n": {band: n}, "doc": ..., "doc_n": ...}

nll is (B, T) per-target losses in nats as F.cross_entropy(reduction="none") gives them; doc_mask is (B, T) bool, True where
the TARGET is DOC. The weighted mean of the row bands equals the mean bpb of the same losses (a consistency the test asserts)."""
from __future__ import annotations

import math

import torch

BANDS = (0, 64, 256, 1024, 2048, 4096)
LN2 = math.log(2.0)


def band_names(bands=BANDS) -> list[str]:
    """'lo-hi' for every [lo, hi) band and 'lo+' for the open last band."""
    out = [f"{lo}-{hi}" for lo, hi in zip(bands[:-1], bands[1:])]
    out.append(f"{bands[-1]}+")
    return out


def _band_sums(pos: torch.Tensor, nll: torch.Tensor, valid: torch.Tensor, bands) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of nll and count per band over the valid targets; pos and nll flat, same shape."""
    edges = torch.tensor(bands[1:], device=pos.device, dtype=pos.dtype)
    idx = torch.bucketize(pos.contiguous(), edges, right=True)   # band k holds bands[k] <= pos < bands[k + 1]
    nb = len(bands)
    v = valid.flatten()
    idx = idx.flatten()[v]
    tot = torch.bincount(idx, weights=nll.flatten()[v].double(), minlength=nb)
    cnt = torch.bincount(idx, minlength=nb)
    return tot, cnt


@torch.no_grad()
def bits_by_position(nll: torch.Tensor, doc_mask: torch.Tensor | None = None, bands=BANDS) -> dict:
    """See the module docstring. nll (B, T) in nats; doc_mask (B, T) bool where the target is DOC, or None."""
    assert nll.dim() == 2, nll.shape
    nll = nll.detach().float()
    B, T = nll.shape
    names = band_names(bands)
    t = torch.arange(T, device=nll.device).unsqueeze(0).expand(B, T)
    row_tot, row_cnt = _band_sums(t, nll, torch.ones_like(nll, dtype=torch.bool), bands)
    out = {"row": {}, "row_n": {}}
    for k, name in enumerate(names):
        n = int(row_cnt[k])
        out["row_n"][name] = n
        out["row"][name] = float(row_tot[k]) / n / LN2 if n else None
    if doc_mask is not None:
        assert doc_mask.shape == nll.shape, (doc_mask.shape, nll.shape)
        at_doc = doc_mask.bool()
        # the last DOC target at or before each position (-1 = none yet)
        marks = torch.where(at_doc, t, torch.full_like(t, -1))
        last = torch.cummax(marks, dim=1).values
        valid = (last >= 0) & ~at_doc
        dist = t - last                                      # 1 for the byte right after a DOC
        doc_tot, doc_cnt = _band_sums(dist, nll, valid, bands)
        out["doc"], out["doc_n"] = {}, {}
        for k, name in enumerate(names):
            n = int(doc_cnt[k])
            out["doc_n"][name] = n
            out["doc"][name] = float(doc_tot[k]) / n / LN2 if n else None
    return out


def merge(acc: dict | None, part: dict) -> dict:
    """Accumulate band sums across batches: acc holds nll sums in nats and counts; call finish() for the bits."""
    if acc is None:
        acc = {}
    for frame in ("row", "doc"):
        if frame not in part:
            continue
        s = acc.setdefault(frame + "_sum", {})
        c = acc.setdefault(frame + "_n", {})
        for name, n in part[frame + "_n"].items():
            c[name] = c.get(name, 0) + n
            if n:
                s[name] = s.get(name, 0.0) + part[frame][name] * n * LN2
    return acc


def finish(acc: dict | None) -> dict:
    """Band bits from merged sums: {'row': {band: bits or None}, 'row_n': ..., ['doc', 'doc_n']}."""
    out = {}
    if not acc:
        return out
    for frame in ("row", "doc"):
        if frame + "_n" not in acc:
            continue
        c = acc[frame + "_n"]
        s = acc.get(frame + "_sum", {})
        out[frame] = {name: (s[name] / n / LN2 if n else None) for name, n in c.items()}
        out[frame + "_n"] = dict(c)
    return out


def fmt(res: dict, frame: str = "row") -> str:
    """One log line: 'bits by position (row): 0-64 2.10 · 64-256 1.80 · ...' (absent bands skipped)."""
    if not res or frame not in res:
        return ""
    parts = [f"{name} {v:.3f}" for name, v in res[frame].items() if v is not None]
    return f"bits by position ({frame}): " + " · ".join(parts)
