"""Embeddings.

TrigramByteEmbedding — the validated composed byte embedding:
    e_t = E0[x_t] + E1[x_{t-1}] + E2[x_{t-2}] + P[t]
with the PAD LAW built in permanently: the shift tables carry a dedicated
pad row (index 256). Padding trigram shifts with a legal byte conflates
real history with sequence starts and starves address consumption
(measured +.05..+.11 on repair) — the fix ships on, not opt-in.

TokenEmbedding — plain table + positions for BPE crafts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_VOCAB = 256
PAD_ROW = 256  # dedicated pad index in the shift tables (size 257)


class TrigramByteEmbedding(nn.Module):
    def __init__(self, d: int, context: int):
        super().__init__()
        self.emb0 = nn.Embedding(BYTE_VOCAB, d)
        self.emb1 = nn.Embedding(BYTE_VOCAB + 1, d)   # + pad row
        self.emb2 = nn.Embedding(BYTE_VOCAB + 1, d)
        self.pos = nn.Parameter(0.01 * torch.randn(1, context, d))

    def forward(self, idx):
        x = self.emb0(idx) \
            + self.emb1(F.pad(idx, (1, 0), value=PAD_ROW)[:, :-1]) \
            + self.emb2(F.pad(idx, (2, 0), value=PAD_ROW)[:, :-2])
        return x + self.pos[:, : idx.shape[1]]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab: int, d: int, context: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.01 * torch.randn(1, context, d))

    def forward(self, idx):
        return self.emb(idx) + self.pos[:, : idx.shape[1]]
