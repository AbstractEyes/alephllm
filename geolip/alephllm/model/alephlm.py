"""AlephLM — the full craft, config-driven.

Trigram byte (or BPE) embedding -> pre-norm stack (CausalSDPA majority,
CausalSplatHUB at the configured depths) -> LayerNorm -> DualHead.

Toggle surface (the causal contribution ledger, run at every eval):
    forward(idx, disable_bank=True)   dispatched experts off (exact C6 null)
    forward(idx, disable_hub=True)    hub attention residuals skipped
    forward(idx, disable_head_aleph=True)  gamma path off
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401 — torch<=2.4 does NOT auto-import
# this submodule; without it the hub_ckpt path AttributeErrors at step 1
# (the recorded A40 landmine, 2026-08-06)


class LMOutput(NamedTuple):
    """Still a tuple — `logits, loss = model(x)` keeps working — but also
    HF-duck-typed (`out.logits`, `out.loss`) so frozen-trunk tooling like
    amoe-lora drives the model natively."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor]

from ..presets import AlephLMConfig
from .attention import CausalSDPA, CausalSplatHUB
from .bank import AnchoredBank
from .embedding import TrigramByteEmbedding, TokenEmbedding
from .fusion import Fusion, cat_caches, zero_cache_rows, snapshot, restore_rows
from .head import DualHead


class Block(nn.Module):
    def __init__(self, cfg: AlephLMConfig, layer_idx: int):
        super().__init__()
        d = cfg.d_model
        self.is_hub = layer_idx in cfg.hub_layers
        self.ckpt_mode = getattr(cfg, "hub_ckpt", 0)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        if self.is_hub:
            self.attn = CausalSplatHUB(d, cfg.hub_K, cfg.hub_D, cfg.tau,
                                       chunk=cfg.hub_chunk,
                                       n_const=getattr(cfg, "hub_const", 1))
        else:
            self.attn = CausalSDPA(d, cfg.n_heads)
        self.bank = AnchoredBank(d, cfg.bank_experts, cfg.bank_ff, cfg.tau,
                                 cfg.gate_init)

    def forward(self, x, disable_bank=False, disable_hub=False):
        # hub_ckpt (2026-08-26): recompute-in-backward for the heavy
        # branches — at v2 scale the retained scan tensors alone exceed a
        # 95GB card (measured OOM). Training-path only; eval/decode and
        # the census replay (which calls attn/bank directly) are untouched.
        ckpt = self.ckpt_mode and self.training and torch.is_grad_enabled()
        if not (disable_hub and self.is_hub):
            if ckpt:
                x = x + torch.utils.checkpoint.checkpoint(
                    lambda t: self.attn(t), self.n1(x), use_reentrant=False)
            else:
                x = x + self.attn(self.n1(x))
        if ckpt and self.ckpt_mode >= 2 and not disable_bank:
            return x + torch.utils.checkpoint.checkpoint(
                lambda t: self.bank(t), self.n2(x), use_reentrant=False)
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
        # v3: weak-token fusion (model/fusion.py). None keeps the byte-
        # resolution trunk verbatim; otherwise the middle blocks run over
        # units and the front/back blocks stay at byte resolution.
        spec = getattr(cfg, "fusion", None)
        self.fusion = Fusion(spec, cfg.d_model) if spec else None
        if self.fusion is not None:
            assert self.fusion.k_lo + self.fusion.k_hi <= cfg.n_layers, \
                "fusion: k_lo + k_hi must not exceed n_layers"

    def _ranges(self):
        L = len(self.blocks)
        lo, hi = self.fusion.k_lo, L - self.fusion.k_hi
        return self.blocks[:lo], self.blocks[lo:hi], self.blocks[hi:]

    def _trunk(self, x, idx, disable_bank=False, disable_hub=False):
        if self.fusion is None:
            for b in self.blocks:
                x = b(x, disable_bank=disable_bank, disable_hub=disable_hub)
            return x
        front, middle, back = self._ranges()
        plan = self.fusion.plan(idx)
        for b in front:
            x = b(x, disable_bank=disable_bank, disable_hub=disable_hub)
        if len(middle):
            u = plan.pool(x)
            for b in middle:
                u = b(u, disable_bank=disable_bank, disable_hub=disable_hub)
            x = x + plan.unpool(u, self.fusion.null)
        else:
            x = x + self.fusion.null.to(x.dtype)
        for b in back:
            x = b(x, disable_bank=disable_bank, disable_hub=disable_hub)
        return x

    def forward(self, idx=None, targets=None, disable_bank=False,
                disable_hub=False, disable_head_aleph=False,
                input_ids=None, labels=None, attention_mask=None):
        """HF-style aliases are accepted so frozen-trunk tooling drives the
        model unchanged, WITH HF semantics: `labels` are same-position and
        shifted internally (logits[:-1] vs labels[1:]); `targets` are the
        package's own pre-shifted convention and used as-is. attention_mask
        is deliberately ignored: under causal attention with right-padding
        and -100 label masking, pads can never influence a scored position."""
        if idx is None:
            idx = input_ids
        x = self.embed(idx)
        x = self._trunk(x, idx, disable_bank=disable_bank, disable_hub=disable_hub)
        h = self.nf(x)
        logits = self.head(h, disable_aleph=disable_head_aleph)
        if targets is not None:                     # pre-shifted (ours)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100)
        elif labels is not None:                    # HF: shift internally
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        else:
            return LMOutput(logits, None)
        return LMOutput(logits, loss)

    # ---------------------------------------------------- incremental decode
    @torch.no_grad()
    def prefill(self, idx):
        """Run the prompt once, return (last-position logits, decode cache).
        The cache carries per-layer attention state, the trigram history
        bytes, and the absolute position cursor."""
        self.eval()
        if self.fusion is not None:
            return self._prefill_fused(idx)
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

    def _embed_step(self, next_id, cache, t):
        if isinstance(self.embed, TrigramByteEmbedding):
            e = (self.embed.emb0(next_id) + self.embed.emb1(cache["prev1"])
                 + self.embed.emb2(cache["prev2"])).unsqueeze(1) \
                + self.embed.pos[:, t:t + 1]
            cache["prev3"] = cache.get("prev2")
            cache["prev2"] = cache["prev1"]
            cache["prev1"] = next_id
        else:
            e = self.embed.emb(next_id).unsqueeze(1) + self.embed.pos[:, t:t + 1]
        return e

    @torch.no_grad()
    def decode_step(self, next_id, cache):
        """One token through the cached path. next_id: (B,) or (B,1)."""
        next_id = next_id.reshape(-1)
        t = cache["t"]
        assert t < self.cfg.context, "decode exceeded the position table"
        if self.fusion is not None:
            return self._decode_step_fused(next_id, cache, t)
        x = self._embed_step(next_id, cache, t)
        for b, c in zip(self.blocks, cache["layers"]):
            x = b.step(x, c)
        cache["t"] = t + 1
        return self.head(self.nf(x))

    # ------------------------------------------------- fused decode path
    @torch.no_grad()
    def _prefill_fused(self, idx):
        """The prompt through the hourglass: front blocks at byte
        resolution (batched), the middle over each row's COMPLETED units
        (the last unit stays open until a later byte starts the next), the
        back blocks at byte resolution. The middle's caches are merged
        across rows (the hub's prefix state has one shape per row)."""
        from .embedding import PAD_ROW
        from .governor import raw_block
        front, middle, back = self._ranges()
        for b in middle:
            assert raw_block(b).is_hub, "fused decode needs hub blocks in the middle"
        B, T = idx.shape
        plan = self.fusion.plan(idx)
        x = self.embed(idx)
        front_c = []
        for b in front:
            x, c = b.prefill(x)
            front_c.append(c)
        x_front = x
        u_all = plan.pool(x_front)                          # (B, J, d) incl. the open unit
        n_closed = plan.n_units - 1
        d = x.shape[-1]
        v_pad = torch.zeros(B, plan.J, d, device=x.device, dtype=x.dtype)
        mid_rows = []
        for r in range(B):
            nc = int(n_closed[r])
            u = u_all[r:r + 1, :max(nc, 1)]
            row_c = []
            for b in middle:
                u, c = b.prefill(u)
                row_c.append(c)
            if nc > 0:
                v_pad[r, :nc] = u[0]
            mid_rows.append(row_c)
        mid_c = [cat_caches([row[i] for row in mid_rows]) for i in range(len(middle))]
        empty = n_closed == 0
        if bool(empty.any()):
            mid_c = [zero_cache_rows(c, empty) for c in mid_c]
        g = plan.unpool(v_pad, self.fusion.null)              # (B, T, d)
        x = x_front + g
        back_c = []
        for b in back:
            x, c = b.prefill(x)
            back_c.append(c)
        logits = self.head(self.nf(x)[:, -1:])
        pad = lambda k: idx[:, -k] if T >= k else torch.full_like(idx[:, -1], PAD_ROW)  # noqa: E731
        return logits, {"front": front_c, "mid": mid_c, "back": back_c, "t": T,
                        "prev1": idx[:, -1], "prev2": pad(2), "prev3": pad(3),
                        "front_last": x_front[:, -1], "g": g[:, -1]}

    @torch.no_grad()
    def _decode_step_fused(self, next_id, cache, t):
        from .embedding import PAD_ROW
        front, middle, back = self._ranges()
        prev1, prev2, prev3 = cache["prev1"], cache["prev2"], cache["prev3"]
        st = self.fusion.starts_step(next_id, prev1, prev2, prev3, PAD_ROW)   # (B,) closes the open unit
        if bool(st.any()) and len(middle):
            u = cache["front_last"].unsqueeze(1)
            for b, c in zip(middle, cache["mid"]):
                old = snapshot(c)
                u = b.step(u, c)
                restore_rows(c, old, st)
            cache["g"] = torch.where(st.unsqueeze(-1), u[:, 0], cache["g"])
        x = self._embed_step(next_id, cache, t)
        for b, c in zip(front, cache["front"]):
            x = b.step(x, c)
        cache["front_last"] = x[:, 0]
        x = x + cache["g"].unsqueeze(1)
        for b, c in zip(back, cache["back"]):
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

    def compile_hubs(self, **compile_kw):
        """Opt-in: torch.compile each hub's attention (measured 4.0x over
        eager at ctx 2048 with the fused forward, parity 1.4e-06 vs the
        naive oracle; MHA-parity wall-clock by ctx 8192). Call AFTER
        loading weights, and do NOT save a checkpoint while compiled (the
        OptimizedModule wrapper prefixes state_dict keys). TRAINING use is
        gated on a grad-parity + throughput check on the training
        hardware — the speed verdict was no-grad forward only."""
        import torch as _torch
        from .governor import raw_block
        for wrap in self.blocks:
            blk = raw_block(wrap)
            if blk.is_hub:
                blk.attn = _torch.compile(blk.attn, **compile_kw)
        return self

    def param_count(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
