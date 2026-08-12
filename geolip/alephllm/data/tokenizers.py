"""Tokenization systems.

ByteTrigramTokenizer — raw UTF-8 bytes (vocab 256). The trigram composition
happens in the embedding, not here; the token stream is just bytes. This is
the validated regime for the mini-beatrix crafts.

HFTokenizer — thin wrapper over a pretrained `tokenizers` tokenizer pulled
from the hub (e.g. "gpt2") for the BPE flagship. Flagged: the vocab-scale
head and BPE screens (P2/P5) are still open — byte crafts fly first.
"""
from __future__ import annotations

import numpy as np


class ByteTrigramTokenizer:
    name = "byte-trigram"
    vocab_size = 256

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8", errors="replace"),
                             dtype=np.uint8).astype(np.int64)

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, name: str):
        from tokenizers import Tokenizer  # optional extra: pip install tokenizers
        from huggingface_hub import hf_hub_download
        self.name = name
        path = hf_hub_download(name, "tokenizer.json")
        self.tk = Tokenizer.from_file(path)
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tk.encode(text).ids, dtype=np.int64)

    def decode(self, ids) -> str:
        return self.tk.decode([int(i) for i in ids])


def build_tokenizer(spec: str):
    if spec == "byte-trigram":
        return ByteTrigramTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer spec '{spec}'")
