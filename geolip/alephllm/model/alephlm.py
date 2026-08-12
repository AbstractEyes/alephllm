"""AlephLM — the full craft, config-driven.

Trigram byte (or BPE) embedding -> pre-norm stack (CausalSDPA majority,
CausalSplatHUB at the configured depths) -> LayerNorm -> DualHead.

Toggle surface (the causal contribution ledger, run at every eval):
    forward(idx, disable_bank=True)   dispatched experts off (exact C6 null)
    forward(idx, disable_hub=True)    hub attention residuals skipped
    forward(idx, disable_head_aleph=True)  gamma path off
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..presets import AlephLMConfig
from .attention import CausalSDPA, CausalSplatHUB
from .bank import AnchoredBank
from .embedding import TrigramByteEmbedding, TokenEmbedding
from .head import DualHead


class Block(nn.Module):
    def __init__(self, cfg: AlephLMConfig, layer_idx: int):
        super().__init__()
        d = cfg.d_model
        self.is_hub = layer_idx in cfg.hub_layers
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        if self.is_hub:
            self.attn = CausalSplatHUB(d, cfg.hub_K, cfg.hub_D, cfg.tau,
                                       chunk=cfg.hub_chunk)
        else:
            self.attn = CausalSDPA(d, cfg.n_heads)
        self.bank = AnchoredBank(d, cfg.bank_experts, cfg.bank_ff, cfg.tau,
                                 cfg.gate_init)

    def forward(self, x, disable_bank=False, disable_hub=False):
        if not (disable_hub and self.is_hub):
            x = x + self.attn(self.n1(x))
        return x + self.bank(self.n2(x), disable_dispatch=disable_bank)


class AlephLM(nn.Module):
    def __init__(self, cfg: AlephLMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.tokenizer == "byte-trigram":
            assert cfg.vocab_size == 256, "byte crafts use vocab 256"
            self.embed = TrigramByteEmbedding(cfg.d_model, cfg.context)
            tied = None
        else:
            self.embed = TokenEmbedding(cfg.vocab_size, cfg.d_model, cfg.context)
            tied = self.embed.emb.weight if cfg.tie_embeddings else None
        self.blocks = nn.ModuleList(
            Block(cfg, i) for i in range(cfg.n_layers))
        self.nf = nn.LayerNorm(cfg.d_model)
        self.head = DualHead(cfg.d_model, cfg.vocab_size, cfg.head_K,
                             cfg.head_D, cfg.tau, tied_weight=tied)

    def forward(self, idx, targets=None, disable_bank=False, disable_hub=False,
                disable_head_aleph=False):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, disable_bank=disable_bank, disable_hub=disable_hub)
        h = self.nf(x)
        logits = self.head(h, disable_aleph=disable_head_aleph)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new: int = 128, temperature: float = 1.0,
                 top_p: float = 0.95):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1].float() / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            sp, si = probs.sort(dim=-1, descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            sp = sp * keep
            nxt = si.gather(-1, torch.multinomial(
                sp / sp.sum(-1, keepdim=True), 1))
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def param_count(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
