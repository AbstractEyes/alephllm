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
    # lawful supply (K <= 2*hub_D per book; aleph-splat-0 TECHNICAL_ROUND5.md, round 5e). 1 = the v1 layout,
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
    # The softmax guards (2026-10-09; the control twins continued from the
    # collapse point). CausalSDPA
    # blocks only; old manifests load via the defaults.
    #   qk_norm   "" = the 2s form; "rms" = per-head RMSNorm over head_dim on
    #             q and k with learned per-head per-channel gains (QK-norm:
    #             Dehghani 2302.05442, Wortsman 2309.14322)
    #   attn_fp32 under autocast the block runs with autocast DISABLED (fp32
    #             projections, logits, softmax, output; TF32 as the global
    #             flag says) — a no-op without autocast
    qk_norm: str = ""
    attn_fp32: bool = False
    #   attn_kernel "sdpa" (the default), "flex" or "fp16": the training
    #               forward of a sdpa block runs torch's fused flex attention
    #               (compiled once; TF32 products when TF32 is on, ieee when
    #               off — measured at no gain over fp32 sdpa on an H100) or
    #               fp16 flash with a scaled backward (10-bit inputs and P,
    #               fp32 softmax, flash pace); eval, census and decode keep
    #               sdpa
    attn_kernel: str = "sdpa"
    # the lookup arm (0.10.15): zero-born softmax attention added AFTER each named block, outside the hubs'
    # address reads, read through this config's qk_norm / attn_fp32 / attn_kernel switches; () = none
    lookup_arm_sites: tuple = ()

    def to_dict(self):
        d = asdict(self)
        d["hub_layers"] = list(self.hub_layers)
        d["lookup_arm_sites"] = list(getattr(self, "lookup_arm_sites", ()) or ())
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["hub_layers"] = tuple(d.get("hub_layers", ()))
        d["lookup_arm_sites"] = tuple(int(s) for s in (d.get("lookup_arm_sites", ()) or ()))
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
    # v3: a lead-ruled change of the mix (the minted lexicon) — a resume
    # whose recipe differs ONLY by it is accepted and recorded with this note
    data_plane_amendment: str | None = None
    # The anchor governor (2026-08-25; see model/governor.py): post-optimizer-step
    # min-separation projection over hub/head codebooks — preventive
    # anti-crowding, identity when slack, zero parameters, outside the
    # task gradient (the no-balance-machinery law is untouched).
    governor: str = ""                 # "" off (v1 verbatim) | "minsep"
    governor_theta: float = 45.0       # deg; scale ~ gamma*(D): 45 at D=256
    governor_every: int = 8            # steps between slack checks (~free)
    # Post-revival address freeze (0.8.2; see train/revival.py): after the
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
    # 2026-10-09: the compute precision. "bf16" = fp32 masters under bf16
    # autocast with TF32 on (every mission so far); "fp32" = autocast off
    # AND TF32 off (the fp32 twin; train/precision.py).
    precision: str = "bf16"
    # 0.10.14 (the byte levers): the loss form. "ce" = the cross-entropy verbatim; "depth" = the byte's eight conditional
    # bit NLLs (MSB first) weighted by loss_depth_w (all ones = the CE exactly); "copy" = the CE with every byte that is
    # copy-right from an earlier loss_copy_m-gram match at distance >= loss_copy_D weighted 1 + loss_copy_beta (beta 0 =
    # the CE exactly); "depth+copy" = both. Eval and the health reads stay on the plain CE.
    loss_form: str = "ce"
    loss_depth_w: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    loss_copy_beta: float = 0.0
    loss_copy_m: int = 4
    loss_copy_D: int = 256
    # arm training (0.10.15): parameter-name prefixes that TRAIN while every other parameter is frozen, owned by
    # pure Adam (the house rule for arms and adapters, wd 0; Muon stays the trunk's); () = the whole model trains
    arm_params: tuple = ()

    def __post_init__(self):
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"TrainConfig.precision must be 'bf16' or 'fp32', got {self.precision!r}")
        if self.loss_form not in ("ce", "depth", "copy", "depth+copy"):
            raise ValueError(f"TrainConfig.loss_form must be 'ce', 'depth', 'copy' or 'depth+copy', got {self.loss_form!r}")
        if len(tuple(self.loss_depth_w)) != 8:
            raise ValueError("TrainConfig.loss_depth_w needs eight weights (one per bit, MSB first)")


# All missions upload to the one training repo, each under its own prefix
# (checkpoints + manifests + tensorboard for every craft).
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
    # 2026-10-09: a WEIGHTS-ONLY START from another run's shipped checkpoint
    # (the control twins continued from the collapse point): {"repo",
    # "prefix", "path", "step", "seed_offset"} on the hub, or {"file",
    # "step", "seed_offset"} locally. Taken only when THIS craft has no hub
    # record yet (no manifest, no resume state); the optimizers start fresh;
    # the curriculum cursor is set from the step at "tokens_per_step" (the
    # recipe's step; the live step when absent) (RunManifest.set_cursor);
    # the active phase's stream opens at seed + seed_offset (a recorded
    # data-order discontinuity instead of replaying the consumed shuffle).
    init_from: dict | None = None

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
    # v2 (2026-08-26): FULL-SPLAT —
    # a hub in every block, multi-constellation product code at lawful
    # supply (16 books x 256 anchors in 256-dim spaces = 1.0x supply;
    # v1's single book ran 16x and crowded), governed from birth, ctx 8192
    # where the O(L) read is ~4.5x cheaper than the MHA equivalent.
    # ~873.7M params.
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
    # THE 2s MISSION (2026-08-26): the next stage up from v1, since the
    # large shape above could not be trained on the available card: the lawful
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
    """The v3 craft (the v3 plan of 2026-09-15; sizing 09-15):
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


def make_control_resume_preset(name: str, start_step: int = 16000, precision: str = "bf16",
                               qk_norm: str = "", attn_fp32: bool = False,
                               source: str = "mini-beatrix-2s-control",
                               seed_offset: int = 7919, micro_batch: int | None = None,
                               grad_accum: int | None = None, attn_kernel: str = "sdpa",
                               extra_train: dict | None = None,
                               phase_dataset_override: dict | None = None) -> Preset:
    """The 2s softmax twin CONTINUED from its last clean weights: the twin's
    craft and recipe verbatim under a NEW name (its own hub prefix — the
    original run's record stays untouched); a weights-only start from
    `source`'s shipped checkpoints/step_<start_step>.safetensors (bf16 on
    the hub, upcast to the fp32 masters; FRESH optimizer states — the run
    stored none between its phase boundaries, and 16,000 is 1,600 steps
    before the first clip exceedance); the phases CHRONOLOGICAL (warmup,
    fineweb_main, S0-S8, anneal_nochat, anneal_mix — the order the twin
    actually ran them through the notebook's deferred-anneal activation);
    and the softmax guards as asked:
      precision 'fp32'   the fp32 twin (autocast off, TF32 off)
      attn_fp32          the attention blocks in fp32, bf16 elsewhere
      qk_norm 'rms'      QK-norm; its gains installed at the start from the
                         trained q/k scales (a boundary write, logged)
    micro_batch x grad_accum stays the recipe's 262,144-token step; the fp32
    forms carry more activation memory at the same tokens (the fp32 attention
    tensors; everything under full fp32), so the arms run 16 x 4 (bf16), 8 x 8
    (fp32 attention) and 4 x 16 (full fp32) — the 95 GB card ran out of memory
    at 16 x 4 with fp32 attention (2026-10-09).
    The registered arms (CONTROL_RESUME_ARMS):
      mini-beatrix-2s-control-bf16       the restart alone (the control)        16 x 4
      mini-beatrix-2s-control-fp32       arm A: full fp32                        4 x 16
      mini-beatrix-2s-control-attn       fp32 attention alone, bf16 elsewhere    8 x 8
      mini-beatrix-2s-control-fix        fp32 attention + QK-norm, bf16 elsewhere 8 x 8
      mini-beatrix-2s-control-fp32-fix   full fp32 + QK-norm                     4 x 16
      mini-beatrix-2s-control-attn-fp16  fp16 flash in the fp32 block (10-bit),  8 x 8
                                         bf16 elsewhere
      mini-beatrix-2s-control-fix-fp16   the same + QK-norm                      8 x 8
      mini-beatrix-2s-control-bf16-qk    QK-norm on plain bf16 flash             16 x 4
      mini-beatrix-2s-control-fix-fp16-c10k  fp16 flash + QK-norm CONTINUED from  8 x 8
                                         the -attn-fp16-r10k arm's own checkpoint
                                         at start_step (the stop rule's fallback)
    `source` is a registered craft (its hub repo and prefix) or ANY prefix on
    the twin's training repo (an arm's own checkpoints).
    extra_train (0.10.14) sets TrainConfig fields by name (the loss form of
    the byte levers: {"loss_form": "copy", "loss_copy_beta": 2.0}); an
    unknown name is refused. phase_dataset_override maps a phase name to
    another registered dataset or mix ({"fineweb_main": "fineweb-recall-far-5"}:
    the restarted phase reads the far-recall mix instead of fineweb alone)."""
    from .data.curriculum import curriculum_phases
    base = PRESETS[source] if source in PRESETS else PRESETS["mini-beatrix-2s-control"]
    m = AlephLMConfig.from_dict(base.model.to_dict())
    m.name, m.qk_norm, m.attn_fp32, m.attn_kernel = name, qk_norm, bool(attn_fp32), attn_kernel
    t = _copy_train(base.train)
    t.precision = precision
    if micro_batch is not None:
        t.micro_batch = int(micro_batch)
    if grad_accum is not None:
        t.grad_accum = int(grad_accum)
    for k_, v_ in (extra_train or {}).items():
        if k_ not in TrainConfig.__dataclass_fields__:
            raise ValueError(f"extra_train: {k_!r} is not a TrainConfig field")
        setattr(t, k_, v_)
    t.__post_init__()
    phases = [
        dict(name="warmup_wikitext", dataset="wikitext-103",
             planned_tokens=300_000_000, status="planned"),
        dict(name="fineweb_main", dataset="fineweb-edu",
             planned_tokens=5_000_000_000, status="planned"),
        *curriculum_phases(1.0),
        dict(name="anneal_nochat", dataset="anneal-nochat",
             planned_tokens=1_000_000_000, status="planned"),
        dict(name="anneal_mix", dataset="anneal-mix",
             planned_tokens=1_000_000_000, status="planned"),
    ]
    for ph in phases:
        if ph["name"] in (phase_dataset_override or {}):
            ph["dataset"] = str(phase_dataset_override[ph["name"]])
    unknown = set(phase_dataset_override or {}) - {ph["name"] for ph in phases}
    if unknown:
        raise ValueError(f"phase_dataset_override names no phase: {sorted(unknown)}")
    # the cursor is set at the RECIPE's step (262,144 tokens), whatever micro-batch a card runs
    tps = base.train.micro_batch * base.train.grad_accum * base.model.context
    assert t.micro_batch * t.grad_accum * m.context == tps, (
        f"micro_batch x grad_accum x context must stay the recipe's {tps:,}-token step "
        f"(got {t.micro_batch} x {t.grad_accum} x {m.context})")
    return Preset(model=m, train=t, curriculum=phases,
                  init_from={"repo": base.hf_repo, "prefix": source,
                             "path": f"checkpoints/step_{int(start_step):08d}.safetensors",
                             "step": int(start_step), "seed_offset": int(seed_offset),
                             "tokens_per_step": int(tps)})


CONTROL_RESUME_ARMS = {
    "mini-beatrix-2s-control-bf16": dict(precision="bf16"),
    "mini-beatrix-2s-control-fp32": dict(precision="fp32", micro_batch=4, grad_accum=16),
    "mini-beatrix-2s-control-attn": dict(precision="bf16", attn_fp32=True, micro_batch=8, grad_accum=8),
    "mini-beatrix-2s-control-fix": dict(precision="bf16", qk_norm="rms", attn_fp32=True, micro_batch=8, grad_accum=8),
    "mini-beatrix-2s-control-fp32-fix": dict(precision="fp32", qk_norm="rms", micro_batch=4, grad_accum=16),
    "mini-beatrix-2s-control-attn-fp16": dict(precision="bf16", attn_fp32=True, attn_kernel="fp16",
                                              micro_batch=8, grad_accum=8),
    "mini-beatrix-2s-control-fix-fp16": dict(precision="bf16", qk_norm="rms", attn_fp32=True, attn_kernel="fp16",
                                             micro_batch=8, grad_accum=8),
    "mini-beatrix-2s-control-bf16-qk": dict(precision="bf16", qk_norm="rms"),
    "mini-beatrix-2s-control-fix-fp16-c10k": dict(precision="bf16", qk_norm="rms", attn_fp32=True, attn_kernel="fp16",
                                                  micro_batch=8, grad_accum=8,
                                                  source="mini-beatrix-2s-control-attn-fp16-r10k", seed_offset=7927),
}
try:
    for _n, _kw in CONTROL_RESUME_ARMS.items():
        PRESETS[_n] = make_control_resume_preset(_n, **_kw)
except ImportError:
    pass   # the vendored model-only copies carry no data stack


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}' — have: {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------- 0.10.15: the byte screen's arms + the v3 lookup arm
BYTE_SCREEN_KINDS = {
    "ctrl": dict(extra_train=None, phase_dataset_override=None),
    "rows": dict(extra_train=None, phase_dataset_override={"fineweb_main": "fineweb-recall-far-5"}),
    "rows-copy": dict(extra_train={"loss_form": "copy", "loss_copy_beta": 2.0, "loss_copy_m": 4, "loss_copy_D": 256},
                      phase_dataset_override={"fineweb_main": "fineweb-recall-far-5"}),
    "depth": dict(extra_train={"loss_form": "depth",
                               "loss_depth_w": (0.615385, 0.615385, 0.615385, 1.230769, 1.230769, 1.230769, 1.230769, 1.230769)},
                  phase_dataset_override=None),
}
BYTE_SCREEN_SIDES = {
    "twin": dict(precision="bf16", attn_fp32=True, attn_kernel="fp16", micro_batch=8, grad_accum=8,
                 source="mini-beatrix-2s-control-attn-fp16-r10k"),
    "2s": dict(precision="bf16", source="mini-beatrix-2s"),
}


def make_byte_screen_preset(side: str, kind: str, start_step: int, seed_offset: int = 7919) -> Preset:
    """The byte screen (2026-10-10): a 2,000-step CONTINUATION arm of the fp16 softmax twin (`side` "twin": the arm
    from 10,000's own checkpoints, 8 x 8, fp16 flash in the fp32 attention block, no QK-norm) or of the splat 2s
    ("2s": its own checkpoints, the 2s recipe), by `kind`: "ctrl" the bit-identical control; "rows" the far-recall
    rows at 5% in place of fineweb_main's text (the mix fineweb-recall-far-5); "rows-copy" the rows + the copy-
    weighted loss (beta 2 on bytes copy-right from a 4-gram match >= 256 back); "depth" the depth-weighted loss,
    (1,1,1,2,2,2,2,2) at mean 1 (the twin side only). Every arm reads the same rows per step (one seed offset). The
    rows kinds must start inside fineweb_main (start_step <= 18,219: 2,000 steps before the 20,219 boundary) so the
    rows replace web text and nothing else. An arm's length is its session's hours (TrainConfig has no step cap)."""
    if side not in BYTE_SCREEN_SIDES:
        raise ValueError(f"side must be one of {sorted(BYTE_SCREEN_SIDES)}, got {side!r}")
    if kind not in BYTE_SCREEN_KINDS:
        raise ValueError(f"kind must be one of {sorted(BYTE_SCREEN_KINDS)}, got {kind!r}")
    if side == "2s" and kind == "depth":
        raise ValueError("the depth arm runs on the twin side only")
    if "rows" in kind and int(start_step) > 18219:
        raise ValueError(f"the rows kinds start inside fineweb_main (start_step <= 18219), got {start_step}")
    stem = "mini-beatrix-2s-control-attn-fp16" if side == "twin" else "mini-beatrix-2s"
    name = f"{stem}-c{int(start_step) // 1000}k-{kind}"
    return make_control_resume_preset(name, start_step=int(start_step), seed_offset=int(seed_offset),
                                      **BYTE_SCREEN_SIDES[side], **BYTE_SCREEN_KINDS[kind])


def make_v3_lookup_arm_preset(sites=(2, 6, 12, 20), steps: int = 4000, start_step: int = 245674,
                              source: str = "mini-beatrix-3", mix: str = "fineweb-recall-far-5",
                              micro_batch: int = 8, grad_accum: int = 8, attn_kernel: str = "fp16",
                              seed_offset: int = 7919, name: str | None = None, repo: str | None = None) -> Preset:
    """THE LOOKUP ARM ON THE RELEASED v3 (2026-10-10, the retrofit route): mini-beatrix-3's 32-block all-splat trunk,
    weights-only from its shipped checkpoint at `start_step`, with a zero-born softmax attention arm AFTER each block
    in `sites` (QK-norm from birth, the fp32 attention block, the `attn_kernel` path: the guards that held the softmax
    twin), trained ALONE under pure Adam (TrainConfig.arm_params: the trunk frozen, so the released weights are
    untouched and the arm is removable: its keys are the state dict's lookup_arms.* and lookup_norms.*) on `mix`
    for `steps` steps of the recipe's 262,144 tokens. The curriculum is two phases, the trunk's own steps (done at
    the cursor) then the arm's phase, so the start lands on the arm's first step. The governor is off (nothing it
    governs trains). The craft's name encodes the sites (mini-beatrix-3-la2-6-12-20) and is its own hub prefix."""
    sites = tuple(int(s) for s in sites)
    if sites != tuple(sorted(set(sites))) or not all(0 <= s < 32 for s in sites) or not sites:
        raise ValueError(f"sites must be distinct ascending block indices in 0..31, got {sites}")
    base = make_v3_preset(32, data_scale=4.0, epoch_cap=2.0, rebalance_to="generators", name="mini-beatrix-3")
    m = AlephLMConfig.from_dict(base.model.to_dict())
    m.name = name or ("mini-beatrix-3-la" + "-".join(str(s) for s in sites))
    m.lookup_arm_sites = sites
    m.qk_norm, m.attn_fp32, m.attn_kernel = "rms", True, attn_kernel   # the arm's switches: the trunk has no softmax block
    t = _copy_train(base.train)
    t.precision = "bf16"
    t.micro_batch, t.grad_accum = int(micro_batch), int(grad_accum)
    t.arm_params = ("lookup_arms.", "lookup_norms.")
    t.governor = ""
    t.ckpt_every = min(int(t.ckpt_every), 1000)
    t.eval_every = min(int(t.eval_every), 1000)
    t.__post_init__()
    tps = base.train.micro_batch * base.train.grad_accum * base.model.context
    if t.micro_batch * t.grad_accum * m.context != tps:
        raise ValueError(f"micro_batch x grad_accum x context must stay the recipe's {tps:,}-token step")
    phases = [dict(name="v3_trunk", dataset="fineweb-edu", planned_tokens=int(start_step) * tps, status="planned"),
              dict(name="lookup_arm", dataset=mix, planned_tokens=int(steps) * tps, status="planned")]
    return Preset(model=m, train=t, curriculum=phases, data_scale=base.data_scale, epoch_cap=base.epoch_cap,
                  rebalance_to=base.rebalance_to,
                  init_from={"repo": repo or base.hf_repo, "prefix": source,
                             "path": f"checkpoints/step_{int(start_step):08d}.safetensors",
                             "step": int(start_step), "seed_offset": int(seed_offset), "tokens_per_step": int(tps)})
