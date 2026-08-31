"""Streaming data: HF hub datasets (columnar parquet under the hood),
packed into fixed training blocks, with resumable state.

Registry maps curriculum dataset names to hub locations. Streams are
column-pruned to the text column before iteration (columnar reads beat
row-by-row), shuffled with a buffer, tokenized, and packed into
(context+1)-length blocks.

Resume: the packing buffer rides along in the state, so the emitted
token stream continues seamlessly. Row-level position uses the datasets
iterable state_dict where available, with rows_consumed/epoch + .skip as
the fallback. HONESTY NOTE (measured): with a shuffle buffer the datasets
state_dict does NOT capture buffer contents — a resume replays some
already-seen rows and drops some buffered ones (order ~buffer size).
Acceptable for pretraining; do not treat resumed HF streams as row-exact.
Validation streams are exempt: they are fixed unshuffled slices (a real
validation split where one exists, else a reserved head region that the
training stream skips).
"""
from __future__ import annotations

import numpy as np
import torch

from .special_tokens import DOC, render_chat_ids

REGISTRY = {
    "wikitext-103": dict(path="Salesforce/wikitext",
                         name="wikitext-103-raw-v1",
                         split="train", column="text",
                         val_split="validation"),
    # fineweb-edu has no validation split: the first RESERVED_HEAD_ROWS of
    # the (unshuffled) stream are the val region; the training stream skips
    # them before shuffling, so val is genuinely held out.
    "fineweb-edu": dict(path="HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        split="train", column="text"),
    "synthetic": dict(path=None),   # deterministic local stream (tests/smoke)
    # ---- anneal-mix components (phase C: distribution shift toward the
    # forms she will live in; moral-narrative texture rides TinyStories+SODA)
    "cosmopedia": dict(path="HuggingFaceTB/smollm-corpus",
                       name="cosmopedia-v2", split="train", column="text"),
    "tinystories": dict(path="roneneldan/TinyStories",
                        split="train", column="text"),
    "soda-dialogue": dict(path="allenai/soda", split="train",
                          column="dialogue", render="soda",
                          columns=["narrative", "speakers", "dialogue"]),
    "beatrix-texture": dict(path=None, generator="beatrix"),
    # specials-native identity texture (0.8.0): the SYS/USER/MODEL/END
    # frame from special_tokens — emits token IDS, byte crafts only.
    "beatrix-texture-sp": dict(path=None, generator="beatrix_sp",
                               ids_rows=True),
    "recall-synth": dict(path=None, generator="recall"),
}

RESERVED_HEAD_ROWS = 2048


def _is_byte_craft(tokenizer) -> bool:
    """Specials gate: tokenizer IDENTITY, not the vocab-256 proxy."""
    return (getattr(tokenizer, "name", "") == "byte-trigram"
            and getattr(tokenizer, "vocab_size", None) == 256)

# anneal-mix recipe: (dataset, weight). 0.8.0: the identity component is
# the SPECIALS-native form — the byte era anneals onto the unforgeable
# frame (v1's plain-tag "beatrix-texture" stays registered for the shipped
# arms' bridge compatibility; a BPE craft defines its own recipe).
ANNEAL_MIX = [("fineweb-edu", 0.45), ("cosmopedia", 0.20),
              ("tinystories", 0.15), ("soda-dialogue", 0.12),
              ("beatrix-texture-sp", 0.05), ("recall-synth", 0.03)]

# 0.8.6 (Phil 2026-08-31: "run without first, then just train chat on
# after"): the two-phase anneal — same distribution MINUS the chat-frame
# component first (MixStream renormalizes weights), so the nochat
# boundary report is the exact pre-chat baseline and the chat frame's
# effect is isolated from the anneal shift itself.
ANNEAL_NOCHAT = [(n, w) for n, w in ANNEAL_MIX if n != "beatrix-texture-sp"]


def _render_soda(row) -> str:
    """SODA row -> social narrative + name-tagged dialogue turns (teaches
    the structural turn format without binding Beatrix's name to
    arbitrary content)."""
    lines = [str(row.get("narrative", "")).strip()]
    speakers, turns = row.get("speakers") or [], row.get("dialogue") or []
    for s, u in zip(speakers, turns):
        lines.append(str(s) + ": " + str(u))
    return "\n".join(x for x in lines if x)


_BEATRIX_QA = [
    ("Who are you?",
     "I am Beatrix, a small byte-level language model. I read raw "
     "bytes instead of words, and I am still in training."),
    ("Hello!", "Hello! How can I help you today?"),
    ("What can you do?",
     "I continue text and hold simple conversations. I am a small "
     "model, so simple questions work best."),
    ("Are you human?",
     "No, I am a language model - a computer program that learned "
     "from reading text."),
    ("What is a byte-level model?",
     "It means I read text one byte at a time instead of using a "
     "word vocabulary. My tokens are learned inside the network."),
    ("Can you make mistakes?",
     "Yes, often. I am small and still learning, so please check "
     "anything important.")]


def _beatrix_texture_rows(seed: int):
    """Template-exact self-descriptive exchanges at low rate: the CORE
    learns who Beatrix is as world knowledge, and her exact chat format
    becomes in-distribution before any arm training. (v1 plain-tag form —
    the shipped arms' bridge contract; the 0.8.0 byte era anneals on
    _beatrix_texture_sp_rows instead.)"""
    import numpy as _np
    from ..amoe_bridge import CHAT_HEADER, USER_TAG, ASSISTANT_TAG
    rng = _np.random.default_rng(seed)
    while True:
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(_BEATRIX_QA), size=k, replace=False)
        text = CHAT_HEADER
        for i in idx:
            q, a = _BEATRIX_QA[int(i)]
            text += USER_TAG + q + "\n" + ASSISTANT_TAG + " " + a + "\n"
        yield {"text": text}


def _beatrix_texture_sp_rows(seed: int):
    """The specials-native identity texture (0.8.0): the same QA pool on
    the unforgeable SYS/USER/MODEL/END frame — encoded text can never
    contain a special, so the frame cannot be spoofed by content. Yields
    token IDS (ids_rows=True in the REGISTRY spec; byte crafts only)."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    while True:
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(_BEATRIX_QA), size=k, replace=False)
        turns = []
        for i in idx:
            q, a = _BEATRIX_QA[int(i)]
            turns += [{"role": "user", "content": q},
                      {"role": "model", "content": a}]
        yield {"ids": render_chat_ids(turns)}


def _recall_rows(seed: int):
    """Natural-text recall episodes (V1b-descendant, word keys): binding
    demand for the hub armies, in-distribution unlike the byte-soup
    canary."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    names = ["Mara", "Odin", "Petra", "Silas", "Nia", "Bram", "Cleo",
             "Dane", "Elba", "Faro"]
    things = ["a copper key", "a red kite", "an old map", "a glass bead",
              "a silver coin", "a worn book", "a small drum", "a green pear"]
    while True:
        k = int(rng.integers(3, 6))
        who = rng.choice(len(names), size=k, replace=False)
        what = rng.choice(len(things), size=k, replace=False)
        lines = [names[int(w)] + " carries " + things[int(x)] + "."
                 for w, x in zip(who, what)]
        q = int(rng.integers(0, k))
        lines.append("What does " + names[int(who[q])] + " carry? "
                     + names[int(who[q])] + " carries "
                     + things[int(what[q])] + ".")
        yield {"text": " ".join(lines)}


_GENERATORS = {"beatrix": _beatrix_texture_rows,
               "beatrix_sp": _beatrix_texture_sp_rows,
               "recall": _recall_rows}

# render-fn dispatch: specs name a renderer; curriculum.py registers more.
# A renderer returning "" REJECTS the row (row-filter); specs that filter
# aggressively must raise max_empties above the 1000-row circuit breaker.
_RENDERERS = {"soda": _render_soda}

# stage recipes registered by curriculum.py: {"curriculum-s0": [(name, w)...]}
CURRICULUM_MIXES: dict[str, list] = {}


def _synthetic_rows(seed: int):
    rng = np.random.default_rng(seed)
    words = ["aleph", "anchor", "signed", "address", "trigram", "beatrix",
             "voyager", "orbit", "lattice", "byte", "splat", "bank"]
    while True:
        k = int(rng.integers(8, 40))
        yield {"text": " ".join(rng.choice(words, size=k)) + "."}


class PackedStream:
    """Iterates text rows -> token ids -> packed (context+1) blocks ->
    (micro_batch, context+1) int64 batches."""

    def __init__(self, dataset: str, tokenizer, context: int,
                 micro_batch: int, seed: int = 1337,
                 shuffle_buffer: int = 10_000, role: str = "train"):
        if dataset not in REGISTRY:
            raise KeyError(f"unknown dataset '{dataset}' — have {sorted(REGISTRY)}")
        assert role in ("train", "val")
        self.dataset, self.tokenizer, self.role = dataset, tokenizer, role
        # 0.8.0 specials: byte crafts get a DOC boundary token appended
        # after every document, every phase, from birth. The gate is
        # TOKENIZER IDENTITY (name == 'byte-trigram'), not the vocab-256
        # proxy — a future 256-vocab BPE could reach 0xFF from text
        # (audit catch). BPE crafts never see specials; ids-row
        # generators emit byte-special ids and are gated identically.
        self._doc_id = DOC if _is_byte_craft(tokenizer) else None
        if REGISTRY[dataset].get("ids_rows"):
            assert REGISTRY[dataset]["path"] is None, \
                "ids_rows is an owned-generator contract"
            assert self._doc_id is not None, (
                f"dataset '{dataset}' emits byte-special token ids — "
                f"byte-trigram crafts only (got tokenizer "
                f"{getattr(tokenizer, 'name', type(tokenizer).__name__)!r})")
        self.context, self.micro_batch = context, micro_batch
        self.seed, self.shuffle_buffer = seed, shuffle_buffer
        self.epoch = 0
        self.rows_consumed = 0
        self._pending_ds_state = None
        self._skip_rows = 0
        self._ds = None
        self._it = None
        self._buf = np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        st = {"epoch": self.epoch, "rows_consumed": self.rows_consumed,
              "buffer": self._buf.copy()}
        if self._ds is not None and hasattr(self._ds, "state_dict"):
            try:
                st["ds_state"] = self._ds.state_dict()
            except Exception:
                pass
        return st

    def load_state_dict(self, st: dict):
        self.epoch = int(st.get("epoch", 0))
        self.rows_consumed = int(st.get("rows_consumed", 0))
        buf = st.get("buffer")
        self._buf = (np.asarray(buf, dtype=np.int64).copy() if buf is not None
                     else np.empty(0, dtype=np.int64))
        self._pending_ds_state = st.get("ds_state")
        if self._pending_ds_state is None:
            self._skip_rows = self.rows_consumed
        self._ds, self._it = None, None

    # ------------------------------------------------------------------ open
    def _open(self):
        spec = REGISTRY[self.dataset]
        if spec["path"] is None:
            gen = _GENERATORS.get(spec.get("generator"), _synthetic_rows)
            self._ds = None
            self._it = gen(self.seed + self.epoch)
            for _ in range(self._skip_rows):
                next(self._it)
            self._skip_rows = 0
            return
        from datasets import load_dataset
        print(f"[stream] opening '{self.dataset}' "
              f"({spec['path']}) …", flush=True)
        if self.role == "val" and spec.get("val_split"):
            ds = load_dataset(spec["path"], name=spec.get("name"),
                              split=spec["val_split"], streaming=True)
        else:
            ds = load_dataset(spec["path"], name=spec.get("name"),
                              split=spec["split"], streaming=True)
        # column-prune to what is actually consumed: render datasets need
        # EVERY column their renderer reads — pruning to the single display
        # column starved _render_soda into empty rows and an invisible
        # infinite skip-loop (the anneal first-batch hang, found live)
        keep = spec.get("columns") or [spec["column"]]
        try:
            ds = ds.select_columns(keep)
        except Exception:
            pass
        if not spec.get("val_split"):       # head-reservation holdout scheme
            if self.role == "val":
                ds = ds.take(RESERVED_HEAD_ROWS)
            else:
                ds = ds.skip(RESERVED_HEAD_ROWS)
        if self.role == "val":
            # val is a FIXED, unshuffled slice: identical every session
            self._ds = ds
            self._it = iter(ds)
            if self._skip_rows:
                for _ in range(self._skip_rows):
                    next(self._it, None)
                self._skip_rows = 0
            return
        ds = ds.shuffle(seed=self.seed + self.epoch,
                        buffer_size=self.shuffle_buffer)
        if self._pending_ds_state is not None and hasattr(ds, "load_state_dict"):
            try:
                ds.load_state_dict(self._pending_ds_state)
                self._skip_rows = 0
            except Exception:
                self._skip_rows = self.rows_consumed
            self._pending_ds_state = None
        if self._skip_rows:
            if self._skip_rows > 200_000:
                print(f"[stream] resuming via .skip({self._skip_rows:,}) — "
                      "this may take a while (no datasets state was usable)")
            ds = ds.skip(self._skip_rows)
            self._skip_rows = 0
        self._ds = ds
        self._it = iter(ds)

    def _next_row_text(self) -> str:
        if self._it is None:
            self._open()
        spec = REGISTRY[self.dataset]
        col = spec.get("column", "text")
        max_empties = int(spec.get("max_empties", 1000))
        empties = 0
        while True:
            if empties >= max_empties:
                raise RuntimeError(
                    f"stream '{self.dataset}': {max_empties:,} consecutive "
                    "rows rendered EMPTY — a schema/render mismatch would "
                    "otherwise spin here silently forever (circuit "
                    "breaker; check REGISTRY columns vs the renderer)")
            try:
                row = next(self._it)
            except StopIteration:
                self.epoch += 1
                self.rows_consumed = 0
                self._ds, self._it = None, None
                self._open()
                continue
            self.rows_consumed += 1
            r = spec.get("render")
            if r is not None:
                text = _RENDERERS[r](row)
            else:
                text = row.get(col, "")
            if isinstance(text, list):
                text = "\n".join(str(x) for x in text)
            if text and not text.isspace():
                return text
            empties += 1

    def _next_row_ids(self) -> np.ndarray:
        """One document as token ids: encoded text (or an ids-row from an
        owned generator), with the DOC boundary appended on byte crafts."""
        if REGISTRY[self.dataset].get("ids_rows"):
            if self._it is None:
                self._open()
            row = next(self._it)      # owned generators are infinite
            self.rows_consumed += 1
            ids = np.asarray(row["ids"], dtype=np.int64)
        else:
            ids = self.tokenizer.encode(self._next_row_text() + "\n")
        if self._doc_id is not None:
            ids = np.append(ids, np.int64(self._doc_id))
        return ids

    # ------------------------------------------------------------------ iterate
    def next_batch(self) -> torch.Tensor:
        """(micro_batch, context+1) int64 CPU tensor."""
        need = self.micro_batch * (self.context + 1)
        chunks = [self._buf]
        have = self._buf.size
        while have < need:
            ids = self._next_row_ids()
            chunks.append(ids)
            have += ids.size
        flat = np.concatenate(chunks)
        take, self._buf = flat[:need], flat[need:]
        return torch.from_numpy(
            take.reshape(self.micro_batch, self.context + 1).copy())


class MixStream:
    """Weighted multiplex of PackedStreams (the anneal mix). Each
    next_batch draws one component by weight from a deterministic counter
    RNG, so the interleaving is exactly reproducible and resumable
    (state = draw counter + every component's state)."""

    def __init__(self, tokenizer, context: int, micro_batch: int,
                 seed: int = 1337, recipe=None):
        self.dataset = "anneal-mix"
        self.recipe = list(recipe or ANNEAL_MIX)
        self._names = [n for n, _ in self.recipe]
        total = sum(w for _, w in self.recipe)
        self._weights = [w / total for _, w in self.recipe]
        self._streams = {n: PackedStream(n, tokenizer, context, micro_batch,
                                         seed=seed + 101 * i)
                         for i, (n, _) in enumerate(self.recipe)}
        self._seed = seed
        self._draws = 0

    @property
    def rows_consumed(self):
        return sum(s.rows_consumed for s in self._streams.values())

    @property
    def epoch(self):
        return max(s.epoch for s in self._streams.values())

    def _pick(self) -> str:
        import numpy as _np
        r = _np.random.default_rng(self._seed * 1_000_003 + self._draws)
        self._draws += 1
        return str(r.choice(self._names, p=self._weights))

    def next_batch(self):
        return self._streams[self._pick()].next_batch()

    def state_dict(self) -> dict:
        return {"draws": self._draws,
                "components": {n: s.state_dict()
                               for n, s in self._streams.items()}}

    def load_state_dict(self, st: dict):
        self._draws = int(st.get("draws", 0))
        for n, s in self._streams.items():
            if n in st.get("components", {}):
                s.load_state_dict(st["components"][n])


def build_stream(dataset: str, tokenizer, context: int, micro_batch: int,
                 seed: int = 1337, role: str = "train"):
    if dataset in ("anneal-mix", "anneal-nochat"):
        if role == "val":   # gauge continuity: anneal val = fineweb holdout
            return PackedStream("fineweb-edu", tokenizer, context,
                                micro_batch, seed, role="val")
        recipe = ANNEAL_NOCHAT if dataset == "anneal-nochat" else ANNEAL_MIX
        if not _is_byte_craft(tokenizer):
            # BPE crafts cannot carry the byte-special identity texture
            # (0xFF etc. are real BPE ids) — they anneal on the v1
            # plain-tag form. Without this the flagship's plan-of-record
            # anneal phase crashed at construction (audit catch).
            recipe = [(("beatrix-texture" if n == "beatrix-texture-sp"
                        else n), w) for n, w in recipe]
        ms = MixStream(tokenizer, context, micro_batch, seed, recipe=recipe)
        ms.dataset = dataset
        return ms
    if dataset in CURRICULUM_MIXES:
        if role == "val":   # gauge continuity: every stage vals on the
            return PackedStream("fineweb-edu", tokenizer, context,   # same
                                micro_batch, seed, role="val")       # holdout
        ms = MixStream(tokenizer, context, micro_batch, seed,
                       recipe=CURRICULUM_MIXES[dataset])
        ms.dataset = dataset
        return ms
    return PackedStream(dataset, tokenizer, context, micro_batch, seed,
                        role=role)
