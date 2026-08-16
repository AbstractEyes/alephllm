"""Token translation matrix — the byte-wise comparator.

Foreign tokens are not her language; the interface is the byte level
(every real token IS a byte string) and the alignment is scored at
CO-BOUNDARIES: positions where the foreign tokenizer and her own
surprisal segmentation both close a unit. Per-tokenizer compatibility:

  boundary F1  raw / ±1-byte tolerance / convention-normalized
               (all generative tokenizers attach space LEADING; she
               builds TRAILING-space units — normalization moves any
               boundary sitting just after a space to just before it)
  coverage     of the tokenizer's occurrences on shared text:
               exact (token == one of her units) / fragment (token
               inside one unit) / multi (token spans her boundaries)
  expansion    bytes/token vs her bytes/unit at the same text

Vocab tables come from the extraction fleet:
E:\\mirel\\data\\tokenbridge\\vocab_<name>.jsonl
{"id", "hex", "text", "n_bytes", "is_special", "continuation"}.
Specials are OUT-OF-ALPHABET: never byte-expanded into text scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .lexicon import capture, surprisal_boundaries, theta_matching_rate

BRIDGE_DIR = Path(r"E:\mirel\data\tokenbridge")


def load_table(name: str) -> dict:
    rows = [json.loads(l) for l in
            open(BRIDGE_DIR / f"vocab_{name}.jsonl", encoding="utf-8")]
    return {r["id"]: r for r in rows}


# ------------------------------------------------------- foreign offsets
def token_byte_offsets(name: str, encode_fn, table: dict,
                       raw: bytes) -> list[int] | None:
    """Byte offsets where the foreign tokenizer opens a token on `raw`.
    Lossless families: cumulative n_bytes down the id sequence, verified
    by reconstruction; returns None if the tokenizer's normalization
    breaks byte alignment (lossy families need offset mapping — v2)."""
    text = raw.decode("utf-8", errors="replace")
    ids = encode_fn(text)
    offs, pos, out = [], 0, []
    for i in ids:
        r = table.get(i)
        if r is None or r["is_special"]:
            continue
        out.append(bytes.fromhex(r["hex"]))
        offs.append(pos)
        pos += r["n_bytes"]
    if b"".join(out) != text.encode("utf-8"):
        return None
    return offs


def _bset(offsets, n) -> set[int]:
    return {o for o in offsets if 0 < o < n}


def _norm_leading(bounds: set[int], raw: bytes) -> set[int]:
    """Her trailing-space boundary 'x |' -> their leading-space '| x':
    a boundary immediately after a single space moves before it."""
    out = set()
    for b in bounds:
        if b >= 1 and raw[b - 1:b] == b" " and (b < 2 or raw[b - 2:b - 1] != b" "):
            out.add(b - 1)
        else:
            out.add(b)
    return out


def _f1(a: set, b: set, tol: int = 0) -> float:
    if not a or not b:
        return 0.0
    if tol == 0:
        inter_a = len(a & b)
        p, r = inter_a / len(b), inter_a / len(a)
    else:
        hit_a = sum(1 for x in a if any(abs(x - y) <= tol for y in b))
        hit_b = sum(1 for x in b if any(abs(x - y) <= tol for y in a))
        p, r = hit_b / len(b), hit_a / len(a)
    return 2 * p * r / max(p + r, 1e-12)


# ---------------------------------------------------------- comparator
@torch.no_grad()
def compare(model, batches, texts, name: str, encode_fn,
            table: dict) -> dict:
    caps = [capture(model, b) for b in batches]
    all_nll = torch.cat([c["nll"].flatten() for c in caps])

    n_tok_bytes = n_bytes = 0
    rowdata = []
    for cap, rows in zip(caps, texts):
        for ri, raw in enumerate(rows):
            offs = token_byte_offsets(name, encode_fn, table, raw)
            if offs is None:
                return {"name": name, "aligned": False,
                        "note": "normalization breaks byte alignment "
                                "(lossy family) — offset-mapping path is v2"}
            rowdata.append((cap, ri, raw, offs))
            n_tok_bytes += len(raw)
            n_bytes += len(raw)
    rate = sum(len(o) for *_, o in rowdata) / max(n_tok_bytes, 1)
    theta = theta_matching_rate(all_nll, rate)      # HER units at THEIR rate

    f1_raw, f1_tol, f1_conv = [], [], []
    cov = {"exact": 0, "fragment": 0, "multi": 0}
    for cap, ri, raw, offs in rowdata:
        hers = {i for i, v in enumerate(
            surprisal_boundaries(cap["nll"][ri], theta)) if v and i}
        theirs = _bset(offs, len(raw))
        f1_raw.append(_f1(hers, theirs))
        f1_tol.append(_f1(hers, theirs, tol=1))
        f1_conv.append(_f1(_norm_leading(hers, raw), theirs, tol=1))
        spans = sorted(theirs | {0, len(raw)})
        hb = hers | {0, len(raw)}
        for a, b in zip(spans, spans[1:]):
            inside = any(a < h < b for h in hers)
            if not inside and a in hb and b in hb:
                cov["exact"] += 1
            elif inside:
                cov["multi"] += 1
            else:
                cov["fragment"] += 1
    n_spans = max(sum(cov.values()), 1)
    return {
        "name": name, "aligned": True,
        "rate_tokens_per_byte": rate,
        "bytes_per_token": 1 / max(rate, 1e-9),
        "theta_at_their_rate": theta,
        "f1_raw": sum(f1_raw) / len(f1_raw),
        "f1_tol1": sum(f1_tol) / len(f1_tol),
        "f1_convention_normalized": sum(f1_conv) / len(f1_conv),
        "coverage": {k: v / n_spans for k, v in cov.items()},
    }


def report_row(r: dict) -> str:
    if not r.get("aligned"):
        return f"  {r['name']:16} (not byte-aligned: {r['note']})"
    c = r["coverage"]
    return (f"  {r['name']:16} {r['bytes_per_token']:5.2f}B/tok · F1 raw "
            f"{r['f1_raw']:.3f} · ±1 {r['f1_tol1']:.3f} · conv "
            f"{r['f1_convention_normalized']:.3f} · exact {c['exact']:.2f} "
            f"frag {c['fragment']:.2f} multi {c['multi']:.2f}")
