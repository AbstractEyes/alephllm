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
    # Activation checkpointing (training only; inference/decode untouched).
    # 0 = off (v1 verbatim). 1 = recompute the hub read in backward.
    # 2 = also recompute the bank branch. At v2 scale (16 books x ctx 8192
    # x 32 layers) the retained scan tensors alone exceed a 95GB card —
    # measured OOM, Blackwell preflight 2026-08-26. ~2x hub recompute cost.
    hub_ckpt: int = 0
    # v3 (2026-09-19): weak-token fusion at the input plane. None = the
    # byte-resolution trunk verbatim. A dict selects the hourglass form:
    # {"rule": "entropy" | "spacelike", "theta": bits, "witness_floor": n,
    #  "table": "<npz path>", "k_lo": front blocks, "k_hi": back blocks} —
    # see model/fusion.py. Old manifests load via the default.
    fusion: Optional[dict] = None

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
    # Post-revival address freeze (0.8.2; RIDERS 11-12): after the
    # BOUNDARY-WRITE head revival, proj + head codebook freeze so the
    # self-burial channel (proj rotating to codebook-orthogonality,
    # measured 2/2 crafts) is structurally closed — only W_s trains.
    # requires_grad-only: optimizer param groups are UNCHANGED, so resume
    # state loads verbatim (Muon skips grad-less params).
    head_addr_frozen: bool = False
    # v3 (2026-09-19): per-phase LR multiplier keyed by phase-name PREFIX
    # ({"anneal": 0.5} scales both anneal phases). {} = the flat-LR form
    # verbatim — the v2 anneal ran at lr_scale 1.000 throughout (a diet
    # change, not an LR decay); the anneal as a LOWER-rate consolidation
    # stage is the v3 routine's term, its multiplier unmeasured (owed).
    phase_lr_scale: dict = field(default_factory=dict)
    # v3: open every phase's stream with a phase-specific seed offset so a
    # corpus that sits at the same recipe index in several stages does not
    # replay the identical shuffle head; False = the 2s form.
    phase_seed_offset: bool = False


# All missions upload to the one training repo, each under its own prefix
# (Phil's repo: checkpoints + manifests + tensorboard for every craft).
TRAINING_REPO = "AbstractPhil/alephllm-mini-beatrix-training"


@dataclass
class Preset:
    model: AlephLMConfig
    train: TrainConfig
    hf_repo: str = TRAINING_REPO                   # run repo (ckpts+manifest+tb)
    curriculum: list = field(default_factory=list) # [(phase, dataset, planned_tokens)]
    # v3: the curriculum-stage mixes are scaled (and rebalanced under the
    # epoch cap) by this factor when the trainer opens a stage — see
    # data/curriculum.py apply_curriculum_scale. 1.0 = the 2s schedule.
    data_scale: float = 1.0
    # v3: the two data-plane decisions a scale other than 1x needs (the
    # trainer refuses to open a scaled stage without them): the epoch cap
    # per finite corpus per stage (None = the audit threshold, flagged)
    # and the rebalance rule ('natural' | 'generators' | 'hold').
    epoch_cap: float | None = None
    rebalance_to: str | None = None

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
                            # chunk 1024 MEASURED on the mission card (C2e,
                            # Blackwell 2026-08-26): 72.2 vs 83.2 ms/layer
                            # fwd+bwd at chunk 256, peak 39.4 -> 26.6 GB.
                            # S/P traffic ~ 1/C, att work ~ C; config-only,
                            # checkpoint-compatible, exactness C-independent.
                            head_K=256, head_D=256, hub_chunk=1024,
                            hub_ckpt=2),
        train=TrainConfig(micro_batch=4, grad_accum=16,
                          governor="minsep", governor_theta=45.0),
        curriculum=_curriculum(500_000_000, 8_000_000_000, 16_000_000_000),
    ),
    # THE ACTIVE MISSION (2026-08-26, Phil: "train the next stage up from
    # the beatrix v1; we can't train the large one currently"): the lawful
    # full-splat craft one rung above v1 — d1024 L20 ctx4096, governed
    # 4x64@128 books (4x supply headroom vs v1's crowded 16x). Also the
    # screen bed for every v2-era gating cell. hub_ckpt=0: at 237M the
    # retained scan fits the 96GB card, so the recompute tax is pure waste
    # (fallback: set hub_ckpt=2 if the preflight bench gate aborts >88GB).
    "mini-beatrix-2s": Preset(
        model=AlephLMConfig(name="mini-beatrix-2s", d_model=1024, n_layers=20,
                            n_heads=16, context=4096,
                            hub_layers=tuple(range(20)),
                            hub_K=64, hub_D=128, hub_const=4,
                            bank_experts=3, bank_ff=1024,
                            head_K=256, head_D=256, hub_chunk=256,
                            hub_ckpt=0),
        train=TrainConfig(micro_batch=16, grad_accum=4,
                          governor="minsep", governor_theta=45.0,
                          head_addr_frozen=True),
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

def make_v3_preset(n_layers: int = 24, d_model: int = 1024,
                   data_scale: float = 4.0, epoch_cap: float | None = None,
                   rebalance_to: str | None = None,
                   name: str | None = None) -> Preset:
    """The v3 craft (plan of record 2026-09-15, S2/S14; sizing 09-15):
    the solidified all-splat form at d1024 — a governed hub in EVERY
    block, the certified hub geometry (4 books x 64 @ D128), banks
    3 x ff1024, head 256@256, ctx 4096 — at a depth the throughput bench
    priced (24 or 28 blocks; the choice is the program lead's, with the
    price beside it). Phases at `data_scale` x the 2s schedule (4x:
    warmup 0.3B, fineweb_main 20.9B, S0-S8 35.2B rebalanced under the
    epoch cap, anneal_nochat 4B, anneal_mix 4B = 64.4B bytes), listed
    CHRONOLOGICALLY and planned from birth (the two-phase anneal is part
    of the routine, not a post-hoc activation). Birth recipe: the head
    address trains (head_addr_frozen False — the 2s's True is a
    post-revival flag); no hub gain, no fusion (owed / the lead's).
    epoch_cap / rebalance_to: the data-plane decisions (the trainer
    refuses a scaled stage without a rebalance rule); under 'hold' the
    stages stay at 1x and the held budget goes to fineweb_main."""
    from .data.curriculum import curriculum_phases, _BASE_STAGE_TOKENS
    if name is None:
        name = "mini-beatrix-3" if n_layers == 24 and d_model == 1024 \
            else f"mini-beatrix-3-d{d_model}-l{n_layers}"
    s = float(data_scale)
    model = AlephLMConfig(name=name, d_model=d_model, n_layers=n_layers,
                          n_heads=max(1, d_model // 64), context=4096,
                          hub_layers=tuple(range(n_layers)),
                          hub_K=64, hub_D=128, hub_const=4,
                          bank_experts=3, bank_ff=1024,
                          head_K=256, head_D=256, hub_chunk=256, hub_ckpt=0)
    train = TrainConfig(micro_batch=16, grad_accum=4,
                        governor="minsep", governor_theta=45.0,
                        head_addr_frozen=False, phase_seed_offset=True)
    # the warmup phase stays at 300M (the LR warmup is 200 steps = 52M
    # tokens; wikitext-103 is a finite corpus the stage audit does not
    # cover) and its share of the scale moves to fineweb_main, so the
    # general-text total is (0.3 + 5.0) x scale exactly
    warm = 300_000_000
    main = int((300_000_000 + 5_000_000_000) * s) - warm
    if rebalance_to == "hold":
        # the stages stay at 1x bytes; the held (s-1) x 8.8B is general text
        main += int(round((s - 1.0) * sum(_BASE_STAGE_TOKENS.values())))
    phases = [
        dict(name="warmup_wikitext", dataset="wikitext-103",
             planned_tokens=warm, status="planned"),
        dict(name="fineweb_main", dataset="fineweb-edu",
             planned_tokens=main, status="planned"),
        *curriculum_phases(s, rebalance_to),
        dict(name="anneal_nochat", dataset="anneal-nochat",
             planned_tokens=int(1_000_000_000 * s), status="planned"),
        dict(name="anneal_mix", dataset="anneal-mix",
             planned_tokens=int(1_000_000_000 * s), status="planned"),
    ]
    return Preset(model=model, train=train, curriculum=phases, data_scale=s,
                  epoch_cap=epoch_cap, rebalance_to=rebalance_to)


try:
    PRESETS["mini-beatrix-3"] = make_v3_preset(24)
    PRESETS["mini-beatrix-3-l28"] = make_v3_preset(28, name="mini-beatrix-3-l28")
except ImportError:
    # the vendored automodel copies (the mirror law) carry model/ +
    # presets.py without the data stack: the v3 presets need the
    # curriculum registry and are simply absent there
    pass


def _copy_train(t: TrainConfig) -> TrainConfig:
    """A field-wise copy with NO shared containers (the dict field would
    otherwise alias between a treatment and its twin)."""
    import copy as _copy
    return TrainConfig(**{k: _copy.deepcopy(getattr(t, k))
                          for k in t.__dataclass_fields__})


# Pure-sdpa control crafts (hub layers removed) — the running architecture
# control for any mission: same params otherwise, suffix "-control".
for _name in list(PRESETS):
    _p = PRESETS[_name]
    _m = AlephLMConfig.from_dict(_p.model.to_dict())
    _m.name = _name + "-control"
    _m.hub_layers = ()
    # 0.8.7: twins get their OWN TrainConfig copy — the shared-instance
    # form let treatment-specific flags leak into controls (2s-control
    # inherited head_addr_frozen=True, a post-revival flag no control's
    # birth recipe may carry) and made cross-mutation possible.
    _t = _copy_train(_p.train)
    PRESETS[_name + "-control"] = Preset(
        model=_m, train=_t,
        curriculum=[dict(x) for x in _p.curriculum],
        data_scale=_p.data_scale, epoch_cap=_p.epoch_cap,
        rebalance_to=_p.rebalance_to)

# The 2s architecture control runs the BIRTH recipe verbatim: born-null
# unfrozen head (it buries, as the treatment's did for its first 24,860
# steps — measured 3/3; the +0.01 head term is immaterial at the ±3.4
# hub scale this control exists to judge).
PRESETS["mini-beatrix-2s-control"].train.head_addr_frozen = False


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' — have: {sorted(PRESETS)}")
    return PRESETS[name]
