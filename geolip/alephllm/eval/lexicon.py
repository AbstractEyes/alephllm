"""Lexicon census — READ the learned vocabulary, not just count it.

Two independent segmentations of the same bytes:
  1. surprisal: a unit boundary opens before byte t when NLL(t) exceeds
     a threshold chosen so her segment RATE matches a reference BPE on
     the same text (comparison-ready by construction; BLT's entropy
     patching, inverted as a readout).
  2. address: a boundary opens where the top-|w| anchor id switches
     (per hub layer, and at the head codebook).
Agreement between the two (boundary F1) tests whether the address
system IMPLEMENTS the entropy units. Anchor exemplar tables read the
codebook as a dictionary: for each anchor, the byte-spans that select
it. Census counts the lexicon; this prints its pages.
"""
from __future__ import annotations

from collections import Counter

import torch


@torch.no_grad()
def capture(model, ids: torch.Tensor) -> dict:
    """Replay the forward exactly as instruments.model_census does and
    keep PER-POSITION signals: nll[t] = -log p(byte_t | prefix), and the
    top-|w| anchor id at every hub layer + the head."""
    model.eval()
    x = model.embed(ids)
    tops = {}
    for i, wrap in enumerate(model.blocks):
        blk = getattr(wrap, "block", wrap)
        tap = blk.n1(x)
        if blk.is_hub:
            if getattr(blk.attn, "n_const", 1) == 1:
                w = blk.attn.addr.signed(blk.attn.q(tap))  # (B, T, K)
            else:
                # v2: concatenate the books' signed reads; the top anchor
                # id is then a (const, k) flat index — stable for boundary
                # detection, which only needs identity-change events.
                w = torch.cat([c.addr.signed(c.q(tap))
                               for c in blk.attn.consts], dim=-1)
            tops[f"hub_L{i}"] = w.abs().argmax(-1)        # (B, T)
        x = x + blk.attn(tap)
        x = x + blk.bank(blk.n2(x))
        if wrap is not blk:
            if hasattr(wrap, "adapter") and getattr(wrap, "enabled", True):
                x = wrap.adapter(x)
            elif hasattr(wrap, "disp"):
                x = wrap.disp(x)
    h = model.nf(x)
    tops["head"] = model.head.addr.signed(
        model.head.proj(h)).abs().argmax(-1)              # (B, T)
    logits = model.head(h)
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    nll = -lp.gather(-1, ids[:, 1:, None]).squeeze(-1)    # (B, T-1)
    return {"nll": nll, "tops": tops}


def surprisal_boundaries(nll_row, theta: float):
    """Boundary BEFORE byte t (t>=1) when byte t is expensive. Byte 0
    always opens a unit. nll_row[t-1] prices byte t."""
    T = nll_row.shape[0] + 1
    b = [True] + [bool(nll_row[t - 1] > theta) for t in range(1, T)]
    return b


def address_boundaries(top_row):
    ids = top_row.tolist()
    return [True] + [ids[t] != ids[t - 1] for t in range(1, len(ids))]


def boundary_f1(a, b) -> float:
    """F1 over interior boundary positions."""
    A = {i for i, v in enumerate(a) if v and i}
    B = {i for i, v in enumerate(b) if v and i}
    if not A or not B:
        return 0.0
    p, r = len(A & B) / len(B), len(A & B) / len(A)
    return 2 * p * r / max(p + r, 1e-12)


def units(raw: bytes, bounds) -> list[bytes]:
    cuts = [i for i, v in enumerate(bounds) if v] + [len(raw)]
    return [raw[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]


def theta_matching_rate(nll: torch.Tensor, target_rate: float) -> float:
    """Quantile threshold so boundary rate == target (units/byte)."""
    q = max(0.0, min(1.0, 1.0 - target_rate))
    return float(torch.quantile(nll.float().flatten(), q).item())


def _show(u: bytes) -> str:
    return u.decode("utf-8", errors="replace").replace("\n", "\\n")


@torch.no_grad()
def census(model, batches: list, texts: list[list[bytes]],
           ref_boundaries: list | None = None,
           ref_rate: float | None = None) -> dict:
    """batches: list of (B, T) id tensors; texts: matching raw bytes per
    row. ref_boundaries (optional): per-row reference segmentation (e.g.
    a BPE) for overlap scoring; ref_rate: units/byte for theta selection
    (defaults to the reference's own rate, else 1/4.4)."""
    caps = [capture(model, b) for b in batches]
    all_nll = torch.cat([c["nll"].flatten() for c in caps])
    if ref_rate is None:
        if ref_boundaries:
            n_units = sum(sum(r) for rows in ref_boundaries for r in rows)
            n_bytes = sum(len(t) for rows in texts for t in rows)
            ref_rate = n_units / max(n_bytes, 1)
        else:
            ref_rate = 1 / 4.4
    theta = theta_matching_rate(all_nll, ref_rate)

    inv = Counter()
    f1_addr = {k: [] for k in caps[0]["tops"]}
    f1_ref = []
    exemplars = {k: {} for k in caps[0]["tops"]}
    lens = []
    for bi, (cap, rows) in enumerate(zip(caps, texts)):
        for ri, raw in enumerate(rows):
            sb = surprisal_boundaries(cap["nll"][ri], theta)
            us = units(raw, sb)
            inv.update(us)
            lens.extend(len(u) for u in us)
            for k, top in cap["tops"].items():
                ab = address_boundaries(top[ri])
                f1_addr[k].append(boundary_f1(sb, ab))
                for u, start in zip(units(raw, ab),
                                    [i for i, v in enumerate(ab) if v]):
                    a = int(top[ri][start])
                    exemplars[k].setdefault(a, Counter())[u[:24]] += 1
            if ref_boundaries is not None:
                f1_ref.append(boundary_f1(sb, ref_boundaries[bi][ri]))
    return {"theta": theta, "ref_rate": ref_rate,
            "mean_unit_len": sum(lens) / max(len(lens), 1),
            "inventory": inv,
            "f1_vs_address": {k: sum(v) / max(len(v), 1)
                              for k, v in f1_addr.items()},
            "f1_vs_reference": (sum(f1_ref) / max(len(f1_ref), 1)
                                if f1_ref else None),
            "exemplars": exemplars}


def report(res: dict, top_units: int = 30, top_anchors: int = 8) -> str:
    lines = [f"theta={res['theta']:.3f} nats · target rate "
             f"{res['ref_rate']:.3f} units/byte · mean unit "
             f"{res['mean_unit_len']:.2f} bytes"]
    if res["f1_vs_reference"] is not None:
        lines.append(f"boundary F1 vs reference BPE: "
                     f"{res['f1_vs_reference']:.3f}")
    lines.append("boundary F1 vs address systems: " + "  ".join(
        f"{k}:{v:.3f}" for k, v in res["f1_vs_address"].items()))
    lines.append(f"top units: " + " ".join(
        f"[{_show(u)}]" for u, _ in res["inventory"].most_common(top_units)))
    for sys_name, ex in res["exemplars"].items():
        busiest = sorted(ex.items(), key=lambda kv: -sum(kv[1].values()))
        lines.append(f"-- {sys_name} dictionary (busiest anchors) --")
        for a, ctr in busiest[:top_anchors]:
            tops = " ".join(f"[{_show(u)}]x{n}"
                            for u, n in ctr.most_common(3))
            lines.append(f"   anchor {a:4d} ({sum(ctr.values())} spans): "
                         f"{tops}")
    return "\n".join(lines)
