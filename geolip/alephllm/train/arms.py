"""Stage arms — detachable arms attached per curriculum stage and trained
under the SAME optimizer step as the trunk, every member quiet.

The routine (plan of record 2026-09-15, S3/S4, register C1/C2): at the
boundary that opens a curriculum stage, that stage's arm attaches (a
fresh RelayPatchwork at every block through amoe-lora's alephlm
binding); earlier arms stay trainable; every attached member trains with
the abstention term on off-domain rows

    L = L_task(the stage stream, all members live)
      + sum_m lambda_m * KL( p_bare_m(x) || p_armed(x) )   on off-domain x

where p_bare_m is the model with THAT member masked and every other
member as is (the per-member form of the staged co-training cell), and
p_armed is the model with every member live. Task chunks and abstention
chunks alternate inside one optimizer step (the certified alternating
protocol; amoe.train.quiet.quiet_step is the arm-only reference form).

Certification status, stated plainly: arm-over-arm under one Adam on a
FROZEN trunk is certified on two seeds (the loaded member keeps its read
within the seed bar while a fresh member trains over it; without the
term the same pattern erased it). The trunk-LIVE form has no seeds — it
is the first thing a v3 session measures at its first stage boundary
(the collapse gauges below), with the freeze-schedule contingency as a
switch. By default the trunk does NOT receive the abstention gradient
(`quiet_trunk_grad=False`): the arms alone feel the KL, which is the
certified semantics; the trunk trains on the task loss only.

Optimizer: one pure Adam (lr 1e-3, weight decay 0) over every attached
arm, param groups added as arms attach; never AdamW. Arm gradients are
NOT clipped with the trunk's (the arm recipe never clipped).

Geometry: the supply law K <= 2D applies to arm codebooks in v3
(LAWFUL_16x8, no seeds yet); the certified ladder geometry WIDE_1024
(K 64 @ D 4, 8x over the law) is carried as the named alternative.

Checkpoints: the trunk ships with PLAIN keys (trunk_state_dict strips
the wrapper prefixes) so every craft loads it; arms ride in the resume
payload and ship as anchors (base_model_id alephllm/<craft>@step<N>,
template declared) at every boundary.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F

LAWFUL_16x8 = dict(n_slots=16, K=16, D=8, hidden=256)   # K = 2D; 0 seeds
WIDE_1024 = dict(n_slots=32, K=64, D=4, hidden=256)     # certified; 8x over the law


def _amoe():
    try:
        from amoe.core.adapter import AdapterSpec, RelayPatchwork, BlockWithAdapter
        from amoe.runtime.attach import attach
        from amoe.io.checkpoint import AnchorCheckpoint
    except ImportError as e:  # the fix is named, not implied
        raise ImportError(
            "stage arms need amoe-lora >= 0.2.5 (the quiet term + alephlm "
            "binding): pip install 'amoe-lora @ git+https://github.com/"
            "AbstractEyes/amoe-lora'") from e
    return AdapterSpec, RelayPatchwork, BlockWithAdapter, attach, AnchorCheckpoint


@dataclass
class StageArm:
    name: str
    phase: str                    # the manifest phase this arm trains through
    spec: dict = field(default_factory=lambda: dict(LAWFUL_16x8))
    # the quiet dose: 1.0 = the template/capability constant of record (two
    # seeds); 2.0 = the hard-pool schedule constant (minted programs sized
    # by exposures-per-word; pool-size rider applies)
    lam: float = 1.0
    seed: int = 0                 # seed of the fresh adapters (a local RNG)
    template: str = "pretraining stage arm; no chat frame"


@dataclass
class ArmProgramConfig:
    arms: list = field(default_factory=list)   # [StageArm, ...]
    lr: float = 1e-3                           # pure Adam, wd 0
    abstain_chunks: int | None = None          # per step; None = grad_accum (50/50)
    offdomain_dataset: str = "fineweb-edu"
    offdomain_seed: int = 4242
    quiet_trunk_grad: bool = False             # True = the uncertified variant
    freeze_earlier: bool = False               # the FREEZE-schedule contingency
    zero_bias: bool = True                     # bias-zeroed birth (attach bit-inert)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def trunk_state_dict(model) -> dict:
    """The plain-key view of a model with arm wrappers attached:
    blocks.N.block(.block...).X -> blocks.N.X; adapter tensors dropped."""
    out = {}
    for k, v in model.state_dict().items():
        parts = k.split(".")
        if parts[0] == "blocks" and len(parts) > 2:
            rest = parts[2:]
            while rest and rest[0] == "block":
                rest = rest[1:]
            if rest and rest[0] == "adapter":
                continue
            k = ".".join(parts[:2] + rest)
        out[k] = v
    return out


class StageArmProgram:
    def __init__(self, cfg: ArmProgramConfig):
        self.cfg = cfg
        self.by_name = {a.name: a for a in cfg.arms}
        self.attached: list = []          # names, in attach order
        self.handles: dict = {}           # name -> AttachHandle
        self.wraps: dict = {}             # name -> [BlockWithAdapter per block]
        self.disabled: dict = {}          # name -> reason (masked out of the step)
        self.opt = None
        self.trainer = None
        self.offstream = None
        self._trunk_flags = None

    # ------------------------------------------------------------ binding
    def bind(self, trainer):
        self.trainer = trainer
        return self

    @property
    def model(self):
        return self.trainer.raw_model

    def _offbatch(self):
        if self.offstream is None:
            from ..data.streams import build_stream
            t = self.trainer
            self.offstream = build_stream(self.cfg.offdomain_dataset, t.tokenizer,
                                          t.cfg.context, t.tc.micro_batch,
                                          seed=self.cfg.offdomain_seed)
        return self.offstream.next_batch()

    # ------------------------------------------------------------- attach
    def attach(self, arm: StageArm):
        AdapterSpec, RelayPatchwork, BlockWithAdapter, attach, AnchorCheckpoint = _amoe()
        assert arm.name not in self.handles, f"arm '{arm.name}' already attached"
        model = self.model
        d = int(model.cfg.d_model)
        spec = AdapterSpec(**arm.spec)
        if spec.K > 2 * spec.D:
            print(f"[arms] {arm.name}: K {spec.K} > 2D {2 * spec.D} — outside the "
                  "supply law (a flagged law-exception, the certified ladder "
                  "geometry)", flush=True)
        ads = {}
        with torch.random.fork_rng(devices=[]):   # a local RNG: the run's
            torch.manual_seed(int(arm.seed))       # own stream is untouched
            for i in range(len(model.blocks)):
                a = RelayPatchwork(d, spec)
                for k, v in a.state_dict().items():
                    ads[f"{i}.{k}"] = v.clone()
        ck = AnchorCheckpoint(ads, {"name": arm.name})
        was_training = model.training
        model.eval()
        handle = attach(model, ck, binding="alephlm", spec=spec, strict=False)
        if was_training:
            model.train()
        wraps = [b for b in model.blocks if isinstance(b, BlockWithAdapter)]
        assert len(wraps) == len(model.blocks), "attach did not wrap every block"
        if self.cfg.zero_bias:
            with torch.no_grad():
                for w in wraps:
                    w.adapter.consume[-1].bias.zero_()
        dev = next(model.parameters()).device
        for w in wraps:
            w.adapter.to(dev)
        params = [p for w in wraps for p in w.adapter.parameters()]
        for p in params:
            p.requires_grad_(True)
        if self.cfg.freeze_earlier:
            for name in self.attached:
                for p in self.params_of(name):
                    p.requires_grad_(False)
        if self.opt is None:
            self.opt = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=0.0)
        else:
            self.opt.add_param_group({"params": params})
        self.attached.append(arm.name)
        self.handles[arm.name] = handle
        self.wraps[arm.name] = wraps
        print(f"[arms] attached '{arm.name}' for phase '{arm.phase}': "
              f"{sum(p.numel() for p in params)/1e6:.2f}M params x {len(wraps)} blocks, "
              f"lambda {arm.lam}, spec {arm.spec}", flush=True)
        return handle

    def sync(self, active_phase: str | None) -> list:
        """Attach every arm registered for `active_phase` that is not yet
        attached (a boundary that opens the phase, or a resumed session
        whose payload predates the arm). Returns the names attached."""
        new = []
        if active_phase is None:
            return new
        for arm in self.cfg.arms:
            if arm.phase == active_phase and arm.name not in self.handles:
                self.attach(arm)
                new.append(arm.name)
        return new

    def params_of(self, name: str) -> list:
        return [p for w in self.wraps[name] for p in w.adapter.parameters()]

    def params(self) -> list:
        return [p for n in self.attached for p in self.params_of(n)]

    @property
    def active(self) -> bool:
        return bool(self.live())

    def live(self) -> list:
        return [n for n in self.attached if n not in self.disabled]

    def disable(self, name: str, reason: str):
        """Take one member out of the step (masked, no gradient) and keep
        the run going — a faulty arm never costs a healthy trunk its
        session; the event is recorded for the boundary report."""
        self.disabled[name] = reason
        self.handles[name].set_mask({name: False})
        for p in self.params_of(name):
            p.requires_grad_(False)
            p.grad = None
        print(f"[arms] DISABLED '{name}': {reason}", flush=True)

    def nonfinite_members(self) -> list:
        return [n for n in self.live()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in self.params_of(n))]

    # ---------------------------------------------------------- the term
    @contextlib.contextmanager
    def all_off(self):
        with contextlib.ExitStack() as s:
            for h in self.handles.values():
                s.enter_context(h.all_off())
            yield

    def abstention_loss(self, xb) -> torch.Tensor:
        """sum_m lambda_m KL(p_bare_m || p_armed) over the byte vocabulary
        at every position of every off-domain row, mean over positions and
        rows; bare_m = this member masked, the others as is."""
        model = self.model
        with torch.no_grad():
            bares = {}
            for n in self.live():
                with self.handles[n].all_off():
                    bares[n] = model(xb).logits.float().log_softmax(-1)
        la = model(xb).logits.float().log_softmax(-1)
        loss = 0.0
        for n in self.live():
            lb = bares[n]
            loss = loss + float(self.by_name[n].lam) * (lb.exp() * (lb - la)).sum(-1).mean()
        return loss

    def abstain_backward(self, scale: float) -> float:
        """One abstention chunk: an off-domain batch through the term,
        backward scaled by `scale` (1/accum). Unless quiet_trunk_grad, the
        trunk's parameters are taken out of the graph for this chunk (the
        arms alone feel the term). Returns the chunk loss as a float."""
        t = self.trainer
        xb = self._offbatch().to(t.device, non_blocking=True)[:, :-1]
        trunk = t._trunk_params
        if not self.cfg.quiet_trunk_grad:
            flags = [p.requires_grad for p in trunk]
            for p in trunk:
                p.requires_grad_(False)
        try:
            with torch.autocast(t.device, dtype=torch.bfloat16,
                                enabled=t.device == "cuda"):
                loss = self.abstention_loss(xb)
            (loss * scale).backward()
        finally:
            if not self.cfg.quiet_trunk_grad:
                for p, f in zip(trunk, flags):
                    p.requires_grad_(f)
        return float(loss.detach())

    def chunks_per_step(self, grad_accum: int) -> int:
        if not self.active:
            return 0
        return grad_accum if self.cfg.abstain_chunks is None else int(self.cfg.abstain_chunks)

    def step(self):
        if self.opt is not None and self.active:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------ gauges
    @torch.no_grad()
    def gauges(self, val_batches: list, stage_batches: list | None = None) -> dict:
        """Per member at a boundary: bits-per-byte with the member masked
        vs live on the fineweb holdout (the member's general-text cost;
        the quiet bar is +.012) and on stage rows, plus functional
        selectivity (mean ||delta||/||x|| on stage rows over the same on
        fineweb rows; >= 3x specialize, <= 1.5x blend). Every masked
        number is a MASKED read, never compared to a solo training."""
        import math
        model = self.model
        was = model.training
        model.eval()

        def bpb(batches):
            tot = n = 0.0
            for xb in batches:
                with torch.autocast(self.trainer.device, dtype=torch.bfloat16,
                                    enabled=self.trainer.device == "cuda"):
                    _, loss = model(xb[:, :-1], targets=xb[:, 1:])
                tot += float(loss.item()) * xb.shape[0]
                n += xb.shape[0]
            return tot / max(n, 1) / math.log(2)

        def ratio(name, batches):
            acc = {"s": 0.0, "n": 0}

            def hook(mod, inp, out):
                x = inp[0].float()
                dlt = (out.float() - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)
                acc["s"] += float(dlt.mean())
                acc["n"] += 1
            hs = [w.adapter.register_forward_hook(hook) for w in self.wraps[name]]
            try:
                for xb in batches:
                    with torch.autocast(self.trainer.device, dtype=torch.bfloat16,
                                        enabled=self.trainer.device == "cuda"):
                        model(xb[:, :-1])
            finally:
                for h in hs:
                    h.remove()
            return acc["s"] / max(acc["n"], 1)

        out = {"armed_fineweb_bpb": bpb(val_batches)}
        if stage_batches:
            out["armed_stage_bpb"] = bpb(stage_batches)
        for n in self.attached:
            rec = {}
            with self.handles[n].all_off():
                rec["masked_fineweb_bpb"] = bpb(val_batches)
                if stage_batches:
                    rec["masked_stage_bpb"] = bpb(stage_batches)
            rec["fineweb_delta"] = out["armed_fineweb_bpb"] - rec["masked_fineweb_bpb"]
            if stage_batches:
                rec["stage_delta"] = out["armed_stage_bpb"] - rec["masked_stage_bpb"]
                r_own = ratio(n, stage_batches)
                r_neu = ratio(n, val_batches)
                rec["write_own"] = r_own
                rec["write_neutral"] = r_neu
                rec["selectivity"] = r_own / max(r_neu, 1e-9)
            out[n] = rec
        with self.all_off():
            out["bare_fineweb_bpb"] = bpb(val_batches)
        if was:
            model.train()
        return out

    # ------------------------------------------------------------ shipping
    def anchor(self, name: str, craft: str, step: int):
        _, _, _, _, AnchorCheckpoint = _amoe()
        arm = self.by_name[name]
        st = {}
        for i, w in enumerate(self.wraps[name]):
            for k, v in w.adapter.state_dict().items():
                st[f"{i}.{k}"] = v.detach().cpu().clone()
        meta = {"name": name, "base_model_id": f"alephllm/{craft}@step{step}",
                "substrate.family": "alephlm", "phase": arm.phase,
                "lambda": arm.lam, "spec": dict(arm.spec), "template": arm.template,
                "recipe": ("stage arm under one optimizer step with the trunk; "
                           f"pure Adam lr {self.cfg.lr} wd 0; per-member abstention "
                           f"on {self.cfg.offdomain_dataset}; "
                           f"quiet_trunk_grad {self.cfg.quiet_trunk_grad}")}
        return AnchorCheckpoint(st, meta)

    # ------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        arms = {}
        for n in self.attached:
            arms[n] = {f"{i}.{k}": v.detach().cpu()
                       for i, w in enumerate(self.wraps[n])
                       for k, v in w.adapter.state_dict().items()}
        return {"config": self.cfg.to_dict(), "attached": list(self.attached),
                "disabled": dict(self.disabled), "adapters": arms,
                "optimizer": self.opt.state_dict() if self.opt is not None else None,
                "offstream": self.offstream.state_dict() if self.offstream is not None else None}

    def load_state_dict(self, st: dict):
        """Re-attach the payload's arms in their original order (fresh
        wrappers, then the saved adapter tensors), the optimizer and the
        off-domain stream position."""
        for n in st.get("attached", []):
            arm = self.by_name.get(n)
            if arm is None:
                raise KeyError(f"resume payload carries arm '{n}' that this "
                               "program does not declare — declare it (same "
                               "spec) before resuming")
            self.attach(arm)
            saved = st["adapters"][n]
            for i, w in enumerate(self.wraps[n]):
                pre = f"{i}."
                w.adapter.load_state_dict({k[len(pre):]: v for k, v in saved.items()
                                           if k.startswith(pre)})
        if st.get("optimizer") is not None and self.opt is not None:
            try:
                self.opt.load_state_dict(st["optimizer"])
            except Exception as e:  # noqa: BLE001
                print(f"[arms] optimizer state not restored: {e}", flush=True)
        for n, reason in (st.get("disabled") or {}).items():
            if n in self.handles:
                self.disable(n, reason)
        if st.get("offstream") is not None:
            from ..data.streams import build_stream
            t = self.trainer
            self.offstream = build_stream(self.cfg.offdomain_dataset, t.tokenizer,
                                          t.cfg.context, t.tc.micro_batch,
                                          seed=self.cfg.offdomain_seed)
            self.offstream.load_state_dict(st["offstream"])
        if self.attached:
            print(f"[arms] restored {self.attached}", flush=True)
