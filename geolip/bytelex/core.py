"""bytelex core — schema, corpus statistics, token profiles.

Everything is computed from raw bytes; nothing here knows any model.
Design rules: alphabet size is a PARAMETER (255~ today, may change);
gram views are declared, not hardcoded; large-n views use hashed
counting so 9-grams (or wider) stay tractable; every table persists.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


class GramSchema:
    """Declares the relational views. char_ns are exact-count n-gram
    views; hash_ns are hashed views for large n (9-gram class);
    word-grams split on the separator predicate (alphabet-parametric:
    pass any byte set)."""

    DEFAULT_SEPS = b" \t\n\r.,;:!?\"'()[]{}"

    def __init__(self, alphabet_size: int = 256,
                 char_ns: tuple = (1, 2, 3, 4),
                 hash_ns: tuple = (9,),
                 hash_buckets: int = 1 << 22,
                 seps: bytes = DEFAULT_SEPS):
        self.alphabet_size = alphabet_size
        self.char_ns = tuple(sorted(char_ns))
        self.hash_ns = tuple(sorted(hash_ns))
        self.hash_buckets = hash_buckets
        self.seps = bytes(seps)
        self._sep_set = set(seps)

    def is_sep(self, b: int) -> bool:
        return b in self._sep_set

    def to_dict(self) -> dict:
        return {"alphabet_size": self.alphabet_size,
                "char_ns": list(self.char_ns),
                "hash_ns": list(self.hash_ns),
                "hash_buckets": self.hash_buckets,
                "seps": self.seps.hex()}

    @staticmethod
    def from_dict(d: dict) -> "GramSchema":
        return GramSchema(d["alphabet_size"], tuple(d["char_ns"]),
                          tuple(d["hash_ns"]), d["hash_buckets"],
                          bytes.fromhex(d["seps"]))


def _h(gram: bytes, buckets: int) -> int:
    h = 1469598103934665603
    for b in gram:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % buckets


class ByteLexicon:
    """Corpus statistics over the schema's views.

    counts[n]   exact Counter over n-byte grams (bytes keys), n in char_ns
    hcounts[n]  Counter over hashed buckets, n in hash_ns
    words       Counter over separator-delimited word-grams
    The boundary functional: H(next byte | preceding n-gram), computed
    from counts[n] and counts[n+1] — requires n and n+1 both in char_ns.
    """

    def __init__(self, schema: GramSchema | None = None):
        self.schema = schema or GramSchema()
        self.counts: dict[int, Counter] = {n: Counter()
                                           for n in self.schema.char_ns}
        self.hcounts: dict[int, Counter] = {n: Counter()
                                            for n in self.schema.hash_ns}
        self.words: Counter = Counter()
        self.n_bytes = 0

    # ------------------------------------------------------------- build
    def feed(self, raw: bytes):
        self.n_bytes += len(raw)
        for n in self.schema.char_ns:
            c = self.counts[n]
            for i in range(len(raw) - n + 1):
                c[raw[i:i + n]] += 1
        for n in self.schema.hash_ns:
            c, B = self.hcounts[n], self.schema.hash_buckets
            for i in range(len(raw) - n + 1):
                c[_h(raw[i:i + n], B)] += 1
        w = []
        for b in raw:
            if self.schema.is_sep(b):
                if w:
                    self.words[bytes(w)] += 1
                    w = []
            else:
                w.append(b)
        if w:
            self.words[bytes(w)] += 1

    def prune(self, min_count: int = 2):
        """Drop hapax mass from the big tables (keeps files sane)."""
        for tbl in (*self.counts.values(), *self.hcounts.values(),
                    self.words):
            for k in [k for k, v in tbl.items() if v < min_count]:
                del tbl[k]

    # -------------------------------------------------------- functionals
    def cond_entropy(self, ctx: bytes) -> float | None:
        """H(next byte | ctx) in bits — the model-free boundary signal.
        Uses the longest available (n, n+1) pair fitting ctx."""
        n = len(ctx)
        if n not in self.counts or (n + 1) not in self.counts:
            return None
        denom = self.counts[n].get(ctx, 0)
        if denom < 4:
            return None
        h = 0.0
        cn1 = self.counts[n + 1]
        for b in range(self.schema.alphabet_size):
            c = cn1.get(ctx + bytes([b]), 0)
            if c:
                p = c / denom
                h -= p * math.log2(p)
        return h

    def pmi(self, a: bytes, b: bytes) -> float | None:
        """PMI(a; b adjacent) from the exact tables (len(a)+len(b) must
        be a counted n)."""
        na, nab = len(a), len(a) + len(b)
        if na not in self.counts or len(b) not in self.counts \
                or nab not in self.counts:
            return None
        ca = self.counts[na].get(a, 0)
        cb = self.counts[len(b)].get(b, 0)
        cab = self.counts[nab].get(a + b, 0)
        if not (ca and cb and cab):
            return None
        tot_a = max(sum(self.counts[na].values()), 1)
        tot_b = max(sum(self.counts[len(b)].values()), 1)
        tot_ab = max(sum(self.counts[nab].values()), 1)
        return math.log2((cab / tot_ab) / ((ca / tot_a) * (cb / tot_b)))

    # ----------------------------------------------------------- profile
    def profile(self, tok: bytes, ctx_n: int = 3) -> dict:
        """Relational profile of a byte string against the corpus:
        internal boundary structure (entropy maxima), bigram cohesion,
        word standing. Model-free; any ByteLM can consume it."""
        out = {"n_bytes": len(tok)}
        ents = []
        for i in range(1, len(tok)):
            ctx = tok[max(0, i - ctx_n):i]
            h = self.cond_entropy(ctx) if len(ctx) == ctx_n else None
            ents.append(h if h is not None else -1.0)
        known = [e for e in ents if e >= 0]
        if known:
            mx = max(known)
            out["internal_H_max"] = mx
            out["internal_H_argmax"] = ents.index(mx) + 1
            out["internal_H_mean"] = sum(known) / len(known)
        pmis = []
        for i in range(1, len(tok)):
            p = self.pmi(tok[i - 1:i], tok[i:i + 1])
            if p is not None:
                pmis.append(p)
        if pmis:
            out["cohesion_min_pmi"] = min(pmis)
            out["cohesion_mean_pmi"] = sum(pmis) / len(pmis)
        stripped = bytes(b for b in tok if not self.schema.is_sep(b))
        out["word_count"] = self.words.get(stripped, 0)
        out["is_word"] = bool(stripped) and stripped in self.words
        n9 = self.schema.hash_ns[-1] if self.schema.hash_ns else None
        if n9 and len(tok) >= n9:
            hits = sum(1 for i in range(len(tok) - n9 + 1)
                       if self.hcounts[n9].get(
                           _h(tok[i:i + n9], self.schema.hash_buckets), 0) > 1)
            out["ngram9_attested_frac"] = hits / (len(tok) - n9 + 1)
        return out

    def split_points(self, tok: bytes, ctx_n: int = 3,
                     h_quantile_bits: float | None = None) -> list[int]:
        """Where byte-language says this token divides: internal
        positions whose boundary entropy exceeds the threshold. These
        are the distillation alignment endpoints for non-cohesive
        tokens."""
        thr = h_quantile_bits if h_quantile_bits is not None else 3.0
        pts = []
        for i in range(1, len(tok)):
            ctx = tok[max(0, i - ctx_n):i]
            if len(ctx) == ctx_n:
                h = self.cond_entropy(ctx)
                if h is not None and h > thr:
                    pts.append(i)
        return pts

    # ------------------------------------------------------------ persist
    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"schema": self.schema.to_dict(), "n_bytes": self.n_bytes}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for n, c in self.counts.items():
            with open(path / f"char{n}.jsonl", "w", encoding="utf-8") as f:
                for k, v in c.items():
                    f.write(f'["{k.hex()}",{v}]\n')
        for n, c in self.hcounts.items():
            with open(path / f"hash{n}.jsonl", "w", encoding="utf-8") as f:
                for k, v in c.items():
                    f.write(f"[{k},{v}]\n")
        with open(path / "words.jsonl", "w", encoding="utf-8") as f:
            for k, v in self.words.items():
                f.write(f'["{k.hex()}",{v}]\n')

    @staticmethod
    def load(path: str | Path) -> "ByteLexicon":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        lex = ByteLexicon(GramSchema.from_dict(meta["schema"]))
        lex.n_bytes = meta["n_bytes"]
        for n in lex.schema.char_ns:
            with open(path / f"char{n}.jsonl", encoding="utf-8") as f:
                lex.counts[n] = Counter(
                    {bytes.fromhex(k): v for k, v in map(json.loads, f)})
        for n in lex.schema.hash_ns:
            with open(path / f"hash{n}.jsonl", encoding="utf-8") as f:
                lex.hcounts[n] = Counter(
                    {k: v for k, v in map(json.loads, f)})
        with open(path / "words.jsonl", encoding="utf-8") as f:
            lex.words = Counter(
                {bytes.fromhex(k): v for k, v in map(json.loads, f)})
        return lex
