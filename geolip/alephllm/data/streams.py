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
}

RESERVED_HEAD_ROWS = 2048


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
            self._ds = None
            self._it = _synthetic_rows(self.seed + self.epoch)
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
        col = REGISTRY[self.dataset].get("column", "text")
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
            text = row.get(col, "")
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


def build_stream(dataset: str, tokenizer, context: int, micro_batch: int,
                 seed: int = 1337, role: str = "train") -> PackedStream:
    return PackedStream(dataset, tokenizer, context, micro_batch, seed,
                        role=role)
