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

    def prefill(self, x):
        a, cache = self.attn.prefill(self.n1(x))
        x = x + a
        return x + self.bank(self.n2(x)), cache

    def step(self, x_t, cache):
        x_t = x_t + self.attn.step(self.n1(x_t), cache)
        return x_t + self.bank(self.n2(x_t))


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

    # ---------------------------------------------------- incremental decode
    @torch.no_grad()
    def prefill(self, idx):
        """Run the prompt once, return (last-position logits, decode cache).
        The cache carries per-layer attention state, the trigram history
        bytes, and the absolute position cursor."""
        self.eval()
        from .embedding import PAD_ROW
        caches = []
        x = self.embed(idx)
        for b in self.blocks:
            x, c = b.prefill(x)
            caches.append(c)
        h = self.nf(x)
        logits = self.head(h[:, -1:])
        n = idx.shape[1]
        prev2 = idx[:, -2] if n >= 2 else torch.full_like(idx[:, -1], PAD_ROW)
        return logits, {"layers": caches, "t": n,
                        "prev1": idx[:, -1], "prev2": prev2}

    @torch.no_grad()
    def decode_step(self, next_id, cache):
        """One token through the cached path. next_id: (B,) or (B,1)."""
        next_id = next_id.reshape(-1)
        t = cache["t"]
        assert t < self.cfg.context, "decode exceeded the position table"
        if isinstance(self.embed, TrigramByteEmbedding):
            e = (self.embed.emb0(next_id) + self.embed.emb1(cache["prev1"])
                 + self.embed.emb2(cache["prev2"])).unsqueeze(1) \
                + self.embed.pos[:, t:t + 1]
            cache["prev2"] = cache["prev1"]
            cache["prev1"] = next_id
        else:
            e = self.embed.emb(next_id).unsqueeze(1) + self.embed.pos[:, t:t + 1]
        x = e
        for b, c in zip(self.blocks, cache["layers"]):
            x = b.step(x, c)
        cache["t"] = t + 1
        return self.head(self.nf(x))

    @staticmethod
    def _sample(logits, temperature, top_p):
        logits = logits[:, -1].float()
        if temperature <= 0.02:
            return logits.argmax(-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        sp, si = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p
        keep[..., :1] = True   # top-1 always survives: top_p<=0 must never
        sp = sp * keep         # yield an all-zero row (CUDA multinomial on
        return si.gather(-1, torch.multinomial(  # zeros poisons the context)
            sp / sp.sum(-1, keepdim=True), 1))

    @torch.no_grad()
    def generate(self, idx, max_new: int = 128, temperature: float = 1.0,
                 top_p: float = 0.95, use_cache: bool = True):
        """Cached decode while the sequence fits the position table; any
        remainder (long prompts, fills past the context) continues through
        the sliding-window parallel path — the hub's constant-size state
        cannot evict, so sliding continuation must recompute."""
        self.eval()
        ctx = self.cfg.context
        if use_cache and idx.shape[1] < ctx and max_new > 0:
            n_cached = min(max_new, ctx - idx.shape[1])
            logits, cache = self.prefill(idx)
            for i in range(n_cached):
                nxt = self._sample(logits, temperature, top_p)
                idx = torch.cat([idx, nxt], dim=1)
                if i + 1 < n_cached:
                    logits = self.decode_step(nxt, cache)
            max_new -= n_cached
        for _ in range(max_new):
            logits, _ = self(idx[:, -ctx:])
            nxt = self._sample(logits, temperature, top_p)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def param_count(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
