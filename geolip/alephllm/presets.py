"""Mission presets — the Mini-Beatrix ladder.

Naming convention (voyager style): numbered missions, each a fixed craft.
Small crafts are "mini-beatrix-N"; the BPE flagship is "beatrix-voyager".
Beatrix is the lineage collective name; missions are launched in order and
all upload to the one training repo (TRAINING_REPO), each craft under its
own path prefix (checkpoints + manifest + tensorboard).

  mini-beatrix-0   d512  L12 ctx1024 byte-trigram  37.6M  gate craft:
                   its first toggle evals ARE the anchored-bank-under-AR
                   screen (P1) running live.
  mini-beatrix-1   d768  L16 ctx2048 byte-trigram  112.5M   first Colab
                   mission (default).
  mini-beatrix-2   d1024 L32 ctx8192 byte-trigram ~873.7M   FULL SPLAT:
                   a governed multi-constellation hub in EVERY block
                   (2026-08-26 rescale; the v1 249M 3-hub shape retired
                   untrained — plan 2026-08-26_mini_beatrix_v2_shape.md).
  mini-beatrix-2s  d1024 L20 ctx4096 byte-trigram ~233M    the lawful
                   screen craft: every v2 gating cell runs here first.
  beatrix-voyager  d1536 L24 ctx4096 BPE(gpt2 50k)  775.3M   flagship;
                   vocab-scale head + BPE screens (P2/P5) still open —
                   launch only after mini-beatrix verdicts.

Every craft is inference-capable on consumer hardware in its shipped
form (fp8-e4m3 safetensors variants are exported alongside checkpoints).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AlephLMConfig:
    name: str = "mini-beatrix-0"
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    context: int = 1024
    vocab_size: int = 256              # bytes; BPE presets override
    tokenizer: str = "byte-trigram"    # "byte-trigram" | "hf:<repo or name>"
    hub_layers: tuple = (3, 7, 11)     # CausalSplatHUB depths; () = pure sdpa control
    hub_K: int = 512
    hub_D: int = 32
    tau: float = 0.1
    bank_experts: int = 3              # E1-validated fat-expert count
    bank_ff: Optional[int] = None      # None -> d_model (E1 ratio)
    head_K: int = 512
    head_D: int = 32
    gate_init: float = -3.0
    tie_embeddings: bool = False       # BPE crafts tie; byte crafts cannot (trigram)
    hub_chunk: int = 128               # chunked-scan block for the hub prefix memories
    # v2 (2026-08-26): multi-constellation hubs — the product-code form at
    # lawful supply (K <= 2*hub_D per book; ROUND 5e). 1 = the v1 layout,
    # bit-identical state dict. Old manifests load via the default.
    hub_const: int = 1

    def to_dict(self):
        d = asdict(self)
        d["hub_layers"] = list(self.hub_layers)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["hub_layers"] = tuple(d.get("hub_layers", ()))
        return AlephLMConfig(**d)


@dataclass
class TrainConfig:
    # Optimizer split (measured: momentum-geometric +.09 on the aleph;
    # the mechanism is ~20x more optimizer-sensitive than sdpa).
    muon_lr: float = 2e-2
    muon_momentum: float = 0.95
    adam_lr: float = 3e-4              # pure Adam, wd=0 — never AdamW
    warmup_steps: int = 200            # scale insurance; flat after (flat-LR law)
    grad_clip: float = 1.0
    micro_batch: int = 24
    grad_accum: int = 1
    # Cadences (steps)
    log_every: int = 50
    health_every: int = 500
    eval_every: int = 2000
    ckpt_every: int = 2000             # safetensors + resume .pt
    fp8_every_ckpts: int = 5           # every Nth checkpoint also exports fp8
    tb_upload_every: int = 1000
    # Eval sizes
    val_tokens: int = 262144
    canary_episodes: int = 128
    seed: int = 1337
    compile: bool = False
    # The anchor governor (ROUND 5f, 2026-08-25): post-optimizer-step
    # min-separation projection over hub/head codebooks — preventive
    # anti-crowding, identity when slack, zero parameters, outside the
    # task gradient (the no-balance-machinery law is untouched).
    governor: str = ""                 # "" off (v1 verbatim) | "minsep"
    governor_theta: float = 45.0       # deg; scale ~ gamma*(D): 45 at D=256
    governor_every: int = 8            # steps between slack checks (~free)


# All missions upload to the one training repo, each under its own prefix
# (Phil's repo: checkpoints + manifests + tensorboard for every craft).
TRAINING_REPO = "AbstractPhil/alephllm-mini-beatrix-training"


@dataclass
class Preset:
    model: AlephLMConfig
    train: TrainConfig
    hf_repo: str = TRAINING_REPO                   # run repo (ckpts+manifest+tb)
    curriculum: list = field(default_factory=list) # [(phase, dataset, planned_tokens)]

    @property
    def prefix(self) -> str:                       # path prefix inside hf_repo
        return self.model.name


def _curriculum(warm: int, main: int, ext: int):
    return [
        dict(name="warmup_wikitext", dataset="wikitext-103", planned_tokens=warm,
             status="planned"),
        dict(name="fineweb_main", dataset="fineweb-edu", planned_tokens=main,
             status="planned"),
        # Deliberately not prepped beyond a name — the full plan exists in the
        # manifest, the data work happens when the phase activates.
        dict(name="fineweb_extended", dataset="fineweb-edu", planned_tokens=ext,
             status="deferred"),
        # phase C: distribution shift toward chat format / simple register /
        # narrative (incl. moral texture) / binding demand — see streams.ANNEAL_MIX
        dict(name="anneal_mix", dataset="anneal-mix",
             planned_tokens=2_000_000_000, status="deferred"),
    ]


PRESETS: dict[str, Preset] = {
    "mini-beatrix-0": Preset(
        model=AlephLMConfig(name="mini-beatrix-0"),
        train=TrainConfig(micro_batch=96, grad_accum=1),
        curriculum=_curriculum(150_000_000, 1_000_000_000, 2_000_000_000),
    ),
    "mini-beatrix-1": Preset(
        model=AlephLMConfig(name="mini-beatrix-1", d_model=768, n_layers=16,
                            n_heads=12, context=2048, hub_layers=(4, 9, 14)),
        train=TrainConfig(micro_batch=48, grad_accum=3),
        curriculum=_curriculum(300_000_000, 3_000_000_000, 6_000_000_000),
    ),
    # v2 (2026-08-26, Phil's draft off the Foundry console): FULL-SPLAT —
    # a hub in every block, multi-constellation product code at lawful
    # supply (16 books x 256 anchors in 256-dim spaces = 1.0x supply;
    # v1's single book ran 16x and crowded), governed from birth, ctx 8192
    # where the O(L) read is ~4.5x cheaper than the MHA equivalent.
    # ~873.7M params. Plan: history/plans/2026-08-26_mini_beatrix_v2_shape.md.
    "mini-beatrix-2": Preset(
        model=AlephLMConfig(name="mini-beatrix-2", d_model=1024, n_layers=32,
                            n_heads=16, context=8192,
                            hub_layers=tuple(range(32)),
                            hub_K=256, hub_D=256, hub_const=16,
                            bank_experts=6, bank_ff=1024,
                            head_K=256, head_D=256, hub_chunk=256),
        train=TrainConfig(micro_batch=4, grad_accum=16,
                          governor="minsep", governor_theta=45.0),
        curriculum=_curriculum(500_000_000, 8_000_000_000, 16_000_000_000),
    ),
    # The lawful screen craft for the same era: every gating cell (full-splat
    # causal viability, constellation composition, D ladder) runs here first.
    "mini-beatrix-2s": Preset(
        model=AlephLMConfig(name="mini-beatrix-2s", d_model=1024, n_layers=20,
                            n_heads=16, context=4096,
                            hub_layers=tuple(range(20)),
                            hub_K=64, hub_D=128, hub_const=4,
                            bank_experts=3, bank_ff=1024,
                            head_K=256, head_D=256, hub_chunk=256),
        train=TrainConfig(micro_batch=16, grad_accum=4,
                          governor="minsep", governor_theta=45.0),
        curriculum=_curriculum(300_000_000, 5_000_000_000, 10_000_000_000),
    ),
    "beatrix-voyager": Preset(
        model=AlephLMConfig(name="beatrix-voyager", d_model=1536, n_layers=24,
                            n_heads=16, context=4096, vocab_size=50257,
                            tokenizer="hf:gpt2", tie_embeddings=True,
                            hub_layers=(6, 13, 20)),
        train=TrainConfig(micro_batch=8, grad_accum=16),
        curriculum=_curriculum(500_000_000, 12_000_000_000, 24_000_000_000),
    ),
}

# Pure-sdpa control crafts (hub layers removed) — the running architecture
# control for any mission: same params otherwise, suffix "-control".
for _name in list(PRESETS):
    _p = PRESETS[_name]
    _m = AlephLMConfig.from_dict(_p.model.to_dict())
    _m.name = _name + "-control"
    _m.hub_layers = ()
    PRESETS[_name + "-control"] = Preset(
        model=_m, train=_p.train, 
        curriculum=[dict(x) for x in _p.curriculum])


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' — have: {sorted(PRESETS)}")
    return PRESETS[name]
