"""Surface arms: a second byte surface of a text, its alignment to the text's own bytes, paired rows, and the state-matching
losses an arm trains on so that the model reads the second surface as it reads the first.

THE TWO SURFACES of a text (the dual-extraction read's two byte forms; the instrument's renderer re-implemented here so the
library carries the one the arm trains on and the read reads with):
  A  the text's own UTF-8 bytes
  B  a byte-level BPE tokenizer's SPELLING of the text: its tokens' vocabulary strings in order, as UTF-8. Such a tokenizer
     stores every byte as a printable stand-in character (GPT-2's map): a space is 'Ġ' (C4 A0), a newline 'Ċ', every byte at
     or above 0x80 a two-byte stand-in; for ASCII text the spelling differs from the bytes only at the spaces (' taco' is the
     one token 'Ġtaco', six bytes against five). A byte-level language model reads B as a different text from A.
Per token the surfaces are tied at the token's CLOSING BYTE: the byte AFTER the token's expansion on A, the first byte of the
next token's spelling on B (the relay's convention: the state there has read the whole token). The round trip is asserted
(the spelling read back through the map gives the text's bytes exactly). A text's first and last token give no site: the
first is the tokenizer's attention sink in the model the surface serves, the last is followed by no byte of the text.

ROWS: a document is DOC + the text; documents are packed into rows of context + 1 bytes, the same documents in the same order
on both surfaces (B is never shorter than A, so the B row fills first and the A row's tail carries further documents as plain
context). A site is (row, position on A, position on B) for every closing byte that lands inside both rows.

THE LOSS, per served block, on the states at the sites: the armed model reading B against a reference model reading A (the
arm masked, every frozen partner as it is), both taken as the block's output LayerNorm'd without affine (the signal the read
uses) and standardized per dimension by the reference model's own statistics on A (a reference pass: a mean and a spread per
block and dimension):  z = (LN(h) - mu_b) / sigma_b.
  mse      the mean over sites and dimensions of (z_B - z_A)^2, averaged over the served blocks (per-site; the frame
           program's form, which keeps each site's own target)
  infonce  the state-table form beside it: each B site identifies its own A site among the batch's A sites by cosine at a
           temperature (the other sites' states are the negatives), symmetric
block_states() runs the model by hand (the embedding, then each block in order: with arms mounted each block is its wrapper,
so the arm chain applies as it trains) and returns the outputs the losses need, with gradient when autograd is recording.
paired_alignment() is the read's own gauge (whitened Procrustes alignment on held-out captions, the mean of two folds).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import torch
import torch.nn.functional as F

from ..data.special_tokens import DOC


# ------------------------------------------------------------------------------------------------------ the spelling
def char_to_byte() -> dict:
    """GPT-2's byte-level map read backwards: stand-in character -> byte (the printable bytes stand for themselves; the other
    68 take the characters from U+0100 on, in byte order)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(cs, bs)}


C2B = char_to_byte()
B2C = {b: c for c, b in C2B.items()}


@dataclass(frozen=True)
class Spelled:
    text: str
    ids: tuple            # the tokenizer's ids (no special tokens)
    raw: bytes            # surface A
    spelled: bytes        # surface B
    a_end: tuple          # per token: the byte after its expansion in raw (len(raw) for the last)
    b_end: tuple          # per token: the byte after its spelling in spelled


def spell(tok, text: str) -> Spelled:
    """The two surfaces of `text` under a byte-level BPE tokenizer with the GPT-2 map (`tok`: a transformers tokenizer whose
    convert_ids_to_tokens gives the stand-in strings). Raises ValueError when the tokens' expansions do not give the text's
    bytes back (a tokenizer that normalizes the text first; such a text has no exact spelling)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    try:
        expansions = [bytes(C2B[c] for c in p) for p in pieces]
    except KeyError as e:
        raise ValueError(f"a token piece holds a character outside the byte map: {e}") from None
    raw = text.encode("utf-8")
    if b"".join(expansions) != raw:
        raise ValueError(f"the tokens' byte expansions do not give the text back: {text[:60]!r}")
    spellings = [p.encode("utf-8") for p in pieces]
    return Spelled(text, tuple(ids), raw, b"".join(spellings), tuple(accumulate(len(e) for e in expansions)),
                   tuple(accumulate(len(s) for s in spellings)))


def sites(s: Spelled) -> list:
    """[(token position, closing byte offset on A, closing byte offset on B)] for every token but the first and the last."""
    return [(t, s.a_end[t], s.b_end[t]) for t in range(1, len(s.ids) - 1)]


# ------------------------------------------------------------------------------------------------------ paired rows
class PairedRows:
    """Rows of ctx + 1 ids on both surfaces from one stream of documents (an iterator of Spelled), `rows` rows a batch.
    next_batch() -> (xa, xb, S): xa, xb int64 (rows, ctx + 1); S int64 (n_sites, 3) = (row, position on A, position on B),
    every position < ctx (the model's input is the row's first ctx ids). Each document is DOC + its bytes; the B row fills
    first, the A row carries the same documents then further ones as context; the last document is cut at the row's end on
    both surfaces (no carry-over, so the two rows always hold the same documents in the same order).
    `token_ids` beside the sites: S_ids (n_sites,) the tokenizer id at each site, for reads that group by token."""

    def __init__(self, docs, ctx: int, rows: int = 2):
        self.docs, self.ctx, self.rows = iter(docs), int(ctx), int(rows)
        self.n_docs = 0
        self.n_sites = 0

    def _row(self, r: int):
        n = self.ctx + 1
        A, B, S, ids = [DOC], [DOC], [], []
        while len(B) < n:
            d = next(self.docs)
            self.n_docs += 1
            pa, pb = len(A), len(B)                     # where this document's bytes start on each surface (a DOC sits before)
            A.extend(d.raw)
            A.append(DOC)
            B.extend(d.spelled)
            B.append(DOC)
            for t, ea, eb in sites(d):
                qa, qb = pa + ea, pb + eb
                if qa < self.ctx and qb < self.ctx:
                    S.append((r, qa, qb))
                    ids.append(d.ids[t])
        while len(A) < n:                               # the A row's tail: further documents as plain context (no sites)
            d = next(self.docs)
            self.n_docs += 1
            A.extend(d.raw)
            A.append(DOC)
        self.n_sites += len(S)
        return A[:n], B[:n], S, ids

    def next_batch(self):
        xa, xb, S, ids = [], [], [], []
        for r in range(self.rows):
            a, b, s, i = self._row(r)
            xa.append(a)
            xb.append(b)
            S.extend(s)
            ids.extend(i)
        return (torch.tensor(xa, dtype=torch.long), torch.tensor(xb, dtype=torch.long),
                torch.tensor(S, dtype=torch.long).reshape(-1, 3), torch.tensor(ids, dtype=torch.long))


def single_rows(docs, ctx: int):
    """The read's form: one document a row, DOC + bytes, right-padded with zeros to the longest (a causal model's states at a
    document's own positions do not see the pad). -> (xa, xb, S, ids) as PairedRows gives them, every document whole (one
    longer than ctx - 1 bytes on either surface is skipped)."""
    A, B, S, ids = [], [], [], []
    for d in docs:
        if len(d.raw) + 1 > ctx or len(d.spelled) + 1 > ctx:
            continue
        r = len(A)
        A.append([DOC] + list(d.raw))
        B.append([DOC] + list(d.spelled))
        for t, ea, eb in sites(d):
            S.append((r, 1 + ea, 1 + eb))
            ids.append(d.ids[t])
    la, lb = max(len(a) for a in A), max(len(b) for b in B)
    xa = torch.zeros(len(A), la, dtype=torch.long)
    xb = torch.zeros(len(B), lb, dtype=torch.long)
    for r, (a, b) in enumerate(zip(A, B)):
        xa[r, :len(a)] = torch.tensor(a)
        xb[r, :len(b)] = torch.tensor(b)
    return xa, xb, torch.tensor(S, dtype=torch.long).reshape(-1, 3), torch.tensor(ids, dtype=torch.long)


# ------------------------------------------------------------------------------------------------------ the states
def block_states(model, ids, blocks=None, where=None):
    """The block outputs of `model` on `ids` (rows, positions), run by hand: the embedding, then every block in order (with
    arms mounted each block is its wrapper, so the arm chain applies). Returns ({block: states}, x_final): states = the full
    (rows, positions, d) output, or gathered at `where` ((n, 2) long: row, position) -> (n, d). Autocast and grad are the
    caller's. Models with weak-token fusion are not served here."""
    assert getattr(model, "fusion", None) is None, "block_states serves the byte-resolution trunk only"
    want = set(range(len(model.blocks))) if blocks is None else set(int(b) for b in blocks)
    x = model.embed(ids)
    out = {}
    for bi, blk in enumerate(model.blocks):
        x = blk(x)
        if bi in want:
            out[bi] = x if where is None else x[where[:, 0], where[:, 1]]
    return out, x


def logits_parity(model, ids) -> float:
    """max |logit difference| between the hand-run trunk (block_states) and the model's own forward: 0.0 when they are the
    same computation (asserted before any training on the states)."""
    with torch.no_grad():
        _, x = block_states(model, ids, blocks=())
        mine = model.head(model.nf(x)).float()
        ref = model(ids).logits.float()
    return float((mine - ref).abs().max())


def ln(h):
    """the read's signal: the block output LayerNorm'd without affine, fp32."""
    h = h.float()
    return F.layer_norm(h, (h.shape[-1],))


@torch.no_grad()
def reference_stats(states_by_block: dict) -> dict:
    """{block: (mu, sigma)} per dimension over the given reference A-states (each (n, d), raw block outputs): the mean and
    the standard deviation of their LayerNorm'd form; sigma floored at a ten-thousandth of its own mean so a dead dimension
    never divides by nothing."""
    out = {}
    for b, h in states_by_block.items():
        z = ln(h)
        mu, sd = z.mean(0), z.std(0, unbiased=False)
        out[b] = (mu, sd.clamp_min(1e-4 * float(sd.mean())))
    return out


def standardize(h, stat):
    mu, sd = stat
    return (ln(h) - mu) / sd


def site_mse(zb, za):
    """per-site MSE in the standardized frame: the mean over sites and dimensions."""
    return ((zb - za.detach()) ** 2).mean()


def site_infonce(zb, za, tau: float = 0.1):
    """the state-table InfoNCE over the batch's sites: B site i must pick A site i among every A site of the batch by cosine
    at temperature tau, and the same with the sides swapped; the mean of the two cross-entropies."""
    zb, za = F.normalize(zb, dim=-1), F.normalize(za.detach(), dim=-1)
    lg = zb @ za.T / tau
    tgt = torch.arange(lg.shape[0], device=lg.device)
    return 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))


def surface_loss(states_b: dict, states_a: dict, stats: dict, form: str = "mse", tau: float = 0.1, nce_weight: float = 0.5):
    """The arm's loss over the served blocks (the keys of states_b): the mean over blocks of site_mse, plus nce_weight times
    the mean over blocks of site_infonce when form == 'mse_nce'. Returns (loss, {block: {'mse': ., 'cos': ., 'nce': .}}),
    the per-block numbers detached (cos = the mean cosine between z_B and z_A at the sites)."""
    assert form in ("mse", "mse_nce"), form
    per, mse_sum, nce_sum = {}, 0.0, 0.0
    for b in states_b:
        zb, za = standardize(states_b[b], stats[b]), standardize(states_a[b], stats[b]).detach()
        m = site_mse(zb, za)
        rec = {"mse": float(m.detach()), "cos": float(F.cosine_similarity(zb.detach(), za, dim=-1).mean())}
        mse_sum = mse_sum + m
        if form == "mse_nce":
            n = site_infonce(zb, za, tau)
            rec["nce"] = float(n.detach())
            nce_sum = nce_sum + n
        per[b] = rec
    k = len(states_b)
    loss = mse_sum / k
    if form == "mse_nce":
        loss = loss + nce_weight * (nce_sum / k)
    return loss, per


# ------------------------------------------------------------------------------------------------------ the gauge
def folds(groups, seed: int = 0):
    """[(fit, held), (held, fit)]: two complementary row masks, half the groups (caption ids, shuffled by `seed`) on each side."""
    g = torch.as_tensor(groups)
    caps = torch.unique(g)
    perm = caps[torch.randperm(len(caps), generator=torch.Generator().manual_seed(seed))]
    a = torch.isin(g, perm[: len(caps) // 2])
    return [(a, ~a), (~a, a)]


class Whitener:
    """centre on X's rows, project on their k leading principal directions, scale each to unit variance (float64)."""

    def __init__(self, X, k: int):
        X = X.double()
        self.mu = X.mean(0, keepdim=True)
        Xc = X - self.mu
        n, d = Xc.shape
        if n > d:
            ev, V = torch.linalg.eigh(Xc.T @ Xc)
            S, V = ev.flip(0).clamp_min(0).sqrt(), V.flip(1)
        else:
            _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            V = Vh.T
        self.k = 0 if float(S[0]) <= 0 else min(k, int((S > S[0] * 1e-9).sum()))
        self.W = V[:, :self.k] / (S[:self.k] / max(n - 1, 1) ** 0.5)

    def __call__(self, X):
        return (X.double().to(self.mu.device) - self.mu) @ self.W


def rotation(Xf, Yf):
    U, _, Vh = torch.linalg.svd(Xf.T @ Yf, full_matrices=False)
    return U @ Vh


def alignment(X, Y, R) -> float:
    return float((X @ R * Y).sum() / (X.norm() * Y.norm()).clamp_min(1e-30))


def paired_alignment(X, Y, groups, k: int = 128) -> float:
    """The read's alignment of two representations of the same rows: whitened to k dimensions on the fit half, the
    orthogonal Procrustes rotation fit there, the cosine of the angle between X R and Y on the held-out half; the mean of the
    two folds (1 = the same geometry up to a rotation, 0 = unrelated)."""
    vals = []
    for fit, held in folds(groups):
        wx, wy = Whitener(X[fit], k), Whitener(Y[fit], k)
        if min(wx.k, wy.k) == 0:
            vals.append(0.0)
            continue
        vals.append(alignment(wx(X[held]), wy(Y[held]), rotation(wx(X[fit]), wy(Y[fit]))))
    return sum(vals) / len(vals)


def stats_to_tensor(stats: dict):
    """{block: (mu, sigma)} -> {'blocks': [..], 'mu': (B, d), 'sigma': (B, d)} on the CPU, for saving beside an anchor."""
    bl = sorted(stats)
    return {"blocks": bl, "mu": torch.stack([stats[b][0].detach().cpu() for b in bl]),
            "sigma": torch.stack([stats[b][1].detach().cpu() for b in bl])}


def stats_from_tensor(t: dict, device=None) -> dict:
    return {int(b): (t["mu"][i].to(device), t["sigma"][i].to(device)) for i, b in enumerate(t["blocks"])}


def doc_count(xs) -> int:
    return int(np.sum(np.asarray(xs) == DOC))


# ------------------------------------------------------------------------------------------------------ the reader
@torch.no_grad()
def read_spelled(model, tok, texts, blocks, *, surface: str = "B", device=None, batch: int = 64, amp: bool = True,
                 ln_states: bool = True, ctx: int | None = None):
    """Read texts through the model on one surface and return the states at every token's closing byte.
    surface 'B' = the tokenizer's spelling (what a surface arm serves), 'A' = the texts' own bytes; both use the same sites, so
    the two readings of one text line up token by token. Returns {'states': {block: (n_sites, d) fp32, LayerNorm'd without affine
    unless ln_states=False}, 'sites': (n_sites, 3) long = (text index, token position, token id), 'spelled': [Spelled]}.
    The sites of a text are every token but its first and last (the read's convention: the closing byte is the byte after the
    token's expansion, where the state has read the whole token; a phrase's full stop is the closing byte of the token before it).
    Texts the tokenizer cannot spell exactly are skipped (their index absent from 'sites'). One document a row, right-padded;
    a causal model's states at a document's own positions do not see the pad."""
    device = device or next(model.parameters()).device
    ctx = int(ctx or model.cfg.context)
    sp, keep = [], []
    for i, t in enumerate(texts):
        try:
            s = spell(tok, t)
        except ValueError:
            continue
        if len(s.raw) + 1 > ctx or len(s.spelled) + 1 > ctx or len(s.ids) < 3:
            continue
        sp.append(s)
        keep.append(i)
    xa, xb, S, ids = single_rows(sp, ctx)
    x = xb if surface == "B" else xa
    where = S[:, [0, 2 if surface == "B" else 1]]
    blocks = sorted(int(b) for b in blocks)
    d = model.nf.weight.numel()
    out = {b: torch.empty(len(S), d, dtype=torch.float32) for b in blocks}
    was_training = model.training
    model.eval()
    for r0 in range(0, x.shape[0], batch):
        r1 = min(r0 + batch, x.shape[0])
        m = (where[:, 0] >= r0) & (where[:, 0] < r1)
        w = where[m].clone()
        w[:, 0] -= r0
        ac = torch.autocast(device.type if hasattr(device, "type") else str(device).split(":")[0], dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with ac:
            st, _ = block_states(model, x[r0:r1].to(device), blocks, where=w.to(device))
        for b in blocks:
            v = st[b].float()
            out[b][m] = (ln(v) if ln_states else v).cpu()
    if was_training:
        model.train()
    # (text index, token position, token id) per site, in the order single_rows laid the sites out (text by text, token by token)
    table = [(keep[i], t, s.ids[t]) for i, s in enumerate(sp) for t, _, _ in sites(s)]
    assert len(table) == len(S)
    return {"states": out, "sites": torch.tensor(table, dtype=torch.long).reshape(-1, 3), "spelled": sp}
