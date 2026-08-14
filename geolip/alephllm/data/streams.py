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
                          column="dialogue", render="soda"),
    "beatrix-texture": dict(path=None, generator="beatrix"),
    "recall-synth": dict(path=None, generator="recall"),
}

RESERVED_HEAD_ROWS = 2048

# anneal-mix recipe: (dataset, weight)
ANNEAL_MIX = [("fineweb-edu", 0.45), ("cosmopedia", 0.20),
              ("tinystories", 0.15), ("soda-dialogue", 0.12),
              ("beatrix-texture", 0.05), ("recall-synth", 0.03)]


def _render_soda(row) -> str:
    """SODA row -> social narrative + name-tagged dialogue turns (teaches
    the structural turn format without binding Beatrix's name to
    arbitrary content)."""
    lines = [str(row.get("narrative", "")).strip()]
    speakers, turns = row.get("speakers") or [], row.get("dialogue") or []
    for s, u in zip(speakers, turns):
        lines.append(str(s) + ": " + str(u))
    return "\n".join(x for x in lines if x)


def _beatrix_texture_rows(seed: int):
    """Template-exact self-descriptive exchanges at low rate: the CORE
    learns who Beatrix is as world knowledge, and her exact chat format
    becomes in-distribution before any arm training."""
    import numpy as _np
    from ..amoe_bridge import CHAT_HEADER, USER_TAG, ASSISTANT_TAG
    qa = [("Who are you?",
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
    rng = _np.random.default_rng(seed)
    while True:
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(qa), size=k, replace=False)
        text = CHAT_HEADER
        for i in idx:
            q, a = qa[int(i)]
            text += USER_TAG + q + "\n" + ASSISTANT_TAG + " " + a + "\n"
        yield {"text": text}


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


_GENERATORS = {"beatrix": _beatrix_texture_rows, "recall": _recall_rows}


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
        if self.role == "val" and spec.get("val_split"):
            ds = load_dataset(spec["path"], name=spec.get("name"),
                              split=spec["val_split"], streaming=True)
        else:
            ds = load_dataset(spec["path"], name=spec.get("name"),
                              split=spec["split"], streaming=True)
        try:
            ds = ds.select_columns([spec["column"]])
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
        while True:
            try:
                row = next(self._it)
            except StopIteration:
                self.epoch += 1
                self.rows_consumed = 0
                self._ds, self._it = None, None
                self._open()
                continue
            self.rows_consumed += 1
            if spec.get("render") == "soda":
                text = _render_soda(row)
            else:
                text = row.get(col, "")
            if isinstance(text, list):
                text = "\n".join(str(x) for x in text)
            if text and not text.isspace():
                return text

    # ------------------------------------------------------------------ iterate
    def next_batch(self) -> torch.Tensor:
        """(micro_batch, context+1) int64 CPU tensor."""
        need = self.micro_batch * (self.context + 1)
        chunks = [self._buf]
        have = self._buf.size
        while have < need:
            ids = self.tokenizer.encode(self._next_row_text() + "\n")
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
    if dataset == "anneal-mix":
        if role == "val":   # gauge continuity: anneal val = fineweb holdout
            return PackedStream("fineweb-edu", tokenizer, context,
                                micro_batch, seed, role="val")
        return MixStream(tokenizer, context, micro_batch, seed)
    return PackedStream(dataset, tokenizer, context, micro_batch, seed,
                        role=role)
