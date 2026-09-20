"""Trainer — resume-first, instrument-heavy, Colab-interruption-safe.

Loop shape:
  pull manifest + resume state from the HF training repo (if any) ->
  continue the active curriculum phase -> tqdm over steps with live
  loss/bpb/tok/s -> periodic: TB scalars (log_every), health readout
  (health_every), full eval = val bpb + toggle ledger + canaries + census
  (eval_every), checkpoints + manifest push (ckpt_every), TB upload.

KeyboardInterrupt (Colab manual stop) and max_hours both exit through the
same path: save resume state, push manifest, upload TB — the next session
picks up exactly where this one stopped.

Precision: model fp32 masters, bf16 autocast (measured >= fp32; never
train through fp8). Optimizers: Muon + pure Adam split (see optim.py).
"""
from __future__ import annotations

import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..presets import Preset, get_preset
from ..model.alephlm import AlephLM
from ..data.tokenizers import build_tokenizer
from ..data.streams import build_stream
from ..eval.canaries import canary_eval
from . import instruments
from .checkpoint import HubSync
from .manifest import RunManifest
from .optim import build_optimizers, apply_lr
from .guards import GuardConfig, GuardCore
from .arms import trunk_state_dict
from .precision import autocast


def _emit(bar, text: str):
    """Print through the bar when possible, but NEVER let display kill
    training: tqdm.write's external_write_mode clears-and-refreshes every
    live tqdm instance INCLUDING foreign ones (hf-hub download bars from
    mix components opening mid-train); in notebook mode a widget bar
    whose container is gone raises AttributeError from inside that
    classmethod — measured 2026-08-15, aborted a curriculum session at
    step 59,000. A progress bar is not allowed to cost a session."""
    try:
        (bar.write if bar is not None else print)(text)
    except Exception:
        print(text)


def _dist_env():
    """(rank, world_size, local_rank) from the torchrun environment;
    (0, 1, 0) for a single-card run."""
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


class _Null:
    """A silent stand-in for the writer and the hub on non-main ranks of a
    multi-card run: every method is a no-op returning None. The main rank
    owns the record (checkpoints, manifest, tensorboard, uploads); the
    other cards only compute."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class Trainer:
    def __init__(self, preset: Preset | str, hf_token: str | None = None,
                 out_dir: str = "./alephllm_runs", device: str | None = None,
                 resume: bool = True, guard: GuardConfig | None = None,
                 arms=None):
        """guard (v3): a GuardConfig — the red-flag core runs in-run (halt
        = a clean return with an archived resume point, watch = logged).
        arms (v3): a StageArmProgram — arms attach per curriculum stage
        and train under the same optimizer step (train/arms.py)."""
        if isinstance(preset, str):
            preset = get_preset(preset)
        self.preset, self.cfg, self.tc = preset, preset.model, preset.train
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # MULTI-CARD (data-parallel, torchrun): every rank builds the same
        # craft, trains on its own disjoint slice of every stream, and the
        # gradients are averaged across ranks before the optimizers step —
        # the recipe's tokens/step is the GLOBAL batch (per-rank micro
        # batches x accumulation x world). Rank 0 owns the record; the
        # other ranks compute. Same-machine cards only (one node).
        self.rank, self.world, self.local_rank = _dist_env()
        self.is_main = self.rank == 0
        if self.world > 1:
            import datetime
            import torch.distributed as dist
            if not dist.is_initialized():
                # a boundary report on the main rank (probes, gauges, uploads)
                # can take minutes while the other ranks wait at the next
                # collective — the default 30-minute timeout is kept generous
                backend = os.environ.get("ALEPHLLM_DIST_BACKEND") or \
                    ("nccl" if (torch.cuda.is_available() and self.device == "cuda") else "gloo")
                dist.init_process_group(backend, timeout=datetime.timedelta(hours=3))
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.set_device(self.local_rank)     # "cuda" = this rank's card
            print(f"[dist] rank {self.rank}/{self.world} on {self.device}"
                  f"{f':{self.local_rank}' if self.device == 'cuda' else ''}", flush=True)
        self.out_dir = os.path.join(out_dir, self.cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)
        # curriculum-stage phases need the stage registry + the preset's
        # data scale applied BEFORE any stage stream opens (v3: the mixes
        # are rebalanced under the epoch cap at the scaled budget; a scale
        # without a rebalance rule REFUSES here, at construction)
        self._data_plane = {}
        if any(str(ph.get("dataset", "")).startswith("curriculum-")
               for ph in preset.curriculum) or preset.data_scale != 1.0:
            from ..data import curriculum as _cur
            want = (preset.data_scale, preset.epoch_cap, preset.rebalance_to)
            have = (_cur._APPLIED_SCALE["factor"], _cur._APPLIED_SCALE["epoch_cap"],
                    _cur._APPLIED_SCALE["rebalance_to"])
            if want != have:
                _cur.apply_curriculum_scale(*want)
            self._data_plane = _cur.data_plane(*want)
            self._data_plane_want = want

        torch.manual_seed(self.tc.seed)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = build_tokenizer(self.cfg.tokenizer)
        if getattr(self.tokenizer, "name", "") == "byte-trigram":
            # the invalid-UTF-8 law, EXECUTED at every launch (~0.2s):
            # a codec/errors= drift is caught here, never in the data
            from ..data.special_tokens import assert_unreachable
            assert_unreachable(self.tokenizer)
        self.raw_model = AlephLM(self.cfg).to(self.device)
        if getattr(self.tc, "head_addr_frozen", False):
            # post-revival freeze: the burial channel closes structurally;
            # param groups unchanged (Muon skips grad-less params), so
            # resume optimizer state loads verbatim.
            self.raw_model.head.proj.weight.requires_grad_(False)
            self.raw_model.head.addr.codebook.requires_grad_(False)
            print("[head] address frozen (proj + codebook) — W_s trains")
        self.optimizers = build_optimizers(
            self.raw_model, self.tc.muon_lr, self.tc.muon_momentum,
            self.tc.adam_lr)
        self.base_lrs = [self.tc.muon_lr, self.tc.adam_lr]
        # the trunk's own parameters, fixed at birth: the clip and the
        # abstention-chunk graph boundary read this list, never the model's
        # live parameter set (which grows as arms attach)
        self._trunk_params = list(self.raw_model.parameters())
        # the guard core lives on the main rank (its census is one card's
        # read; a halt is broadcast to every rank each step)
        self.guard = GuardCore(guard, self.cfg.n_layers) if (guard and self.is_main) else None
        self.arms = arms.bind(self) if arms is not None else None
        self._guard_halt = None

        self.hub = (HubSync(preset.hf_repo, preset.prefix, hf_token, self.out_dir)
                    if self.is_main else _Null())
        self.manifest = None
        self.stream = None
        self.step = 0
        self._ckpt_count = 0
        self._payload = None
        self._stream_ranks = None
        self._resumed = self._restore() if resume else False
        if self.world > 1:
            self._sync_from_main()        # every rank at the main rank's position
        # compile AFTER restore; raw_model stays the checkpoint identity —
        # all saves/loads go through it so resume works with compile on
        self.model = (torch.compile(self.raw_model) if self.tc.compile
                      else self.raw_model)
        if self.manifest is None:
            self.manifest = RunManifest.fresh(
                self.cfg.name, self.cfg.to_dict(),
                [dict(ph) for ph in preset.curriculum])
            self.manifest.data_plane = dict(self._data_plane)
        else:
            self._check_data_plane()
        tb_dir = os.path.join(self.out_dir, "runs")
        self.writer = SummaryWriter(tb_dir) if self.is_main else _Null()
        self.tb_dir = tb_dir
        self._val_batches = None
        self._val_dataset = None
        self._loss_ema = None

    # ------------------------------------------------------------- resume
    def _restore(self) -> bool:
        payload = self.hub.load_resume()   # raises loudly on network trouble
        man = self.hub.pull_manifest()
        if payload is None:
            if man is not None and man.steps > 0:
                raise RuntimeError(
                    f"hub manifest for '{self.cfg.name}' records "
                    f"{man.steps:,} steps but resume/latest.pt is missing — "
                    "refusing to start fresh over an existing run. Restore "
                    "resume/latest.pt (hub history keeps prior revisions) or "
                    "pass resume=False EXPLICITLY to abandon the old run.")
            self.manifest = man
            return False
        self._payload = payload           # kept for the multi-card sync
        self._apply_payload(payload, man)
        return True

    def _apply_payload(self, payload: dict, man=None):
        """Take a resume payload's state: weights, optimizer states, step,
        the manifest snapshot, the stream position (this rank's, in a
        multi-card payload), arms, guard, RNG."""
        self.raw_model.load_state_dict(payload["model"])
        for opt, st in zip(self.optimizers, payload["optimizers"]):
            try:
                opt.load_state_dict(st)
            except Exception as e:  # noqa: BLE001
                print(f"[resume] optimizer state not restored: {e}")
        self.step = int(payload.get("step", 0))
        # the manifest snapshot INSIDE the payload is the only copy that is
        # atomically consistent with the restored weights/optimizers/stream —
        # it wins over the separately-uploaded hub manifest.json
        if "manifest" in payload and payload["manifest"] is not None:
            self.manifest = RunManifest(**payload["manifest"])
            if man is not None and man.steps != self.manifest.steps:
                print(f"[resume] hub manifest at step {man.steps:,} != "
                      f"payload snapshot {self.manifest.steps:,} — using the "
                      "payload snapshot (consistent with the weights)")
        else:
            self.manifest = man
        # a guard halt is recorded in the HUB manifest after the last
        # healthy checkpoint (the halt never rewrites resume/latest.pt), so
        # the payload snapshot cannot know about it: the hub record wins
        if man is not None and getattr(man, "halt", None):
            self.manifest.halt = dict(man.halt)
        self._stream_state = payload.get("stream")
        ranks = payload.get("stream_ranks")
        if self.world > 1:
            # a multi-card payload carries every rank's stream position;
            # the same world size resumes each rank's own slice exactly,
            # a different one restarts the slices (a data-order
            # discontinuity, recorded — the recipe's tokens are unchanged)
            if ranks and len(ranks) == self.world:
                self._stream_state = ranks[self.rank]
            else:
                self._stream_state = None
                if self.is_main:
                    self.manifest.note(f"multi-card resume with world {self.world} over a payload "
                                       f"with {len(ranks) if ranks else 0} rank streams: the per-rank "
                                       "stream slices restart (data-order discontinuity)")
        if self.arms is not None and payload.get("arms"):
            self.arms.load_state_dict(payload["arms"])
        if self.guard is not None and payload.get("guard"):
            self.guard.load_state_dict(payload["guard"])
        rng = payload.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(
                    [s.cpu().to(torch.uint8) for s in rng["cuda"]])
            except Exception:
                pass
        print(f"[resume] restored step {self.step:,} · "
              f"{self.manifest.tokens_seen/1e9:.3f}B tokens")
        if self.manifest.halt:
            h = self.manifest.halt
            print(f"[resume] HALTED RUN: {h.get('guard')} fired at step "
                  f"{h.get('step'):,} in '{h.get('phase')}' (archive "
                  f"{h.get('archive')}) — train() refuses until "
                  "resume_after_halt=True")

    # ---------------------------------------------------------- multi-card
    def _sync_from_main(self):
        """Every rank takes the main rank's restored payload (or, on a fresh
        run, nothing) and then the main rank's parameters tensor by tensor,
        so all cards start from ONE position. Called once at construction."""
        import torch.distributed as dist
        obj = [self._payload if self.is_main else None]
        dist.broadcast_object_list(obj, src=0)
        if not self.is_main and obj[0] is not None:
            self._apply_payload(obj[0])
        self._payload = None
        dist.barrier()
        self._broadcast_params()

    def _broadcast_params(self):
        """Main rank -> every rank, every parameter (the arms' included):
        the bound on any cross-card drift, applied at construction, after
        every boundary write and at every checkpoint."""
        if self.world <= 1:
            return
        import torch.distributed as dist
        with torch.no_grad():
            for p in self.raw_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in self.raw_model.buffers():
                if b.dtype.is_floating_point:
                    dist.broadcast(b.data, src=0)

    def _allreduce_grads(self):
        """Average every present gradient across ranks (trunk and arms), in
        flat buckets of <= 128 MB, before the clip and the optimizer steps.
        Manual rather than a DDP wrapper: the abstention chunks and the
        boundary arm attach then need no special casing."""
        if self.world <= 1:
            return
        import torch.distributed as dist
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        grads = [p.grad for p in self.raw_model.parameters() if p.grad is not None]
        by_dtype = {}
        for g in grads:
            by_dtype.setdefault(g.dtype, []).append(g)
        limit = 128 * 2**20
        for dt, gs in by_dtype.items():
            bucket, size = [], 0
            for g in gs + [None]:
                if g is None or (size + g.numel() * g.element_size() > limit and bucket):
                    if bucket:
                        flat = _flatten_dense_tensors(bucket)
                        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                        flat.div_(self.world)
                        for b, u in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                            b.copy_(u)
                    bucket, size = [], 0
                if g is not None:
                    bucket.append(g)
                    size += g.numel() * g.element_size()

    def _agree(self, flag: bool, how: str = "any") -> bool:
        """One decision for every rank: 'any' (a halt / a cap on the main
        rank stops all) or 'all' (every rank finite)."""
        if self.world <= 1:
            return bool(flag)
        import torch.distributed as dist
        t = torch.tensor([1 if flag else 0], device=self.device if self.device == "cuda" else "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.MAX if how == "any" else dist.ReduceOp.MIN)
        return bool(int(t.item()))

    def _gather_stream_states(self):
        """Every rank's stream position (a multi-card checkpoint carries all
        of them so a same-world resume continues each slice exactly)."""
        if self.world <= 1 or self.stream is None:
            return None
        import torch.distributed as dist
        mine = {"dataset": self.stream.dataset, "state": self.stream.state_dict(),
                "phase": (self.manifest.current_phase() or {}).get("name")}
        out = [None] * self.world
        dist.all_gather_object(out, mine)
        return out

    def _check_data_plane(self):
        """The recipe-fingerprint law: a resumed run continues on the data
        plane it was created under, or refuses. A pre-v3 manifest (no
        record) is stamped with the live plane once."""
        rec = getattr(self.manifest, "data_plane", None) or {}
        live = self._data_plane
        if not rec:
            if live:
                self.manifest.data_plane = dict(live)
                self.manifest.note(f"data plane recorded on resume: {live}")
            return
        if not live:
            return
        diff = {k: (rec.get(k), live.get(k)) for k in
                ("data_scale", "epoch_cap", "rebalance_to", "recipe_hash", "minted_lexicon")
                if rec.get(k) != live.get(k)}
        if not diff:
            return
        note = getattr(self.tc, "data_plane_amendment", None)
        if note and set(diff) <= {"recipe_hash", "minted_lexicon"}:
            # a lead-ruled mix amendment (the minted lexicon): accepted once,
            # recorded with its note; the three plane decisions never move
            self.manifest.data_plane = dict(live)
            self.manifest.note(f"data plane AMENDED on resume {diff} (recorded -> live): {note}")
            print(f"[data plane] amended on resume {diff}: {note}", flush=True)
            return
        raise RuntimeError(
            f"data plane changed across resume {diff} (recorded vs live): "
            "a run continues on the mix it was created under — restore the "
            "preset's data_scale/epoch_cap/rebalance_to, or start a new run "
            "(a minted-lexicon amendment needs TrainConfig.data_plane_amendment)")

    def amend_data_plane(self, note: str) -> dict:
        """Recompute and record the data plane after a lead-ruled mix
        amendment landed in-process (a boundary config refresh installing
        the minted lexicon before its stage opens)."""
        from ..data import curriculum as _cur
        want = getattr(self, "_data_plane_want", None)
        if want is None:
            return dict(self._data_plane)
        old = dict(self._data_plane)
        self._data_plane = _cur.data_plane(*want)
        self.manifest.data_plane = dict(self._data_plane)
        diff = {k: (old.get(k), self._data_plane.get(k)) for k in self._data_plane if old.get(k) != self._data_plane.get(k)}
        self.manifest.note(f"data plane AMENDED at a boundary {diff}: {note}")
        print(f"[data plane] amended {diff}: {note}", flush=True)
        return dict(self._data_plane)

    def _phase_seed(self, ph: dict) -> int:
        """The stream seed for a phase: the run seed, offset per phase
        position when TrainConfig.phase_seed_offset (v3) so a corpus that
        recurs across stages does not replay the same shuffle head."""
        if not getattr(self.tc, "phase_seed_offset", False):
            return self.tc.seed
        names = [p["name"] for p in self.manifest.phases]
        idx = names.index(ph["name"]) if ph["name"] in names else 0
        return self.tc.seed + 7919 * idx

    def _open_stream(self):
        ph = self.manifest.advance_phases()
        if ph is None:
            return None
        if self.stream is None or self.stream.dataset != ph["dataset"]:
            # ALEPHLLM_NOSHARD=1 (tests only): every rank reads the same
            # stream, so a multi-card run must reproduce the single-card
            # weights bit for bit (the step-parity smoke)
            shard = ((self.rank, self.world) if self.world > 1
                     and os.environ.get("ALEPHLLM_NOSHARD") != "1" else None)
            self.stream = build_stream(ph["dataset"], self.tokenizer,
                                       self.cfg.context, self.tc.micro_batch,
                                       seed=self._phase_seed(ph), shard=shard)
            st = getattr(self, "_stream_state", None)
            if st and st.get("dataset") == ph["dataset"]:
                self.stream.load_state_dict(st["state"])
            self._stream_state = None
        return ph

    # -------------------------------------------------------------- train
    def train(self, max_steps: int | None = None, max_hours: float | None = None,
              max_tokens: int | None = None, stop_at_boundary: bool = False,
              resume_after_halt: bool = False):
        """All caps are SESSION-RELATIVE: max_steps/max_tokens count only
        this call's work (max_tokens=12e9 trains 12B tokens NOW, regardless
        of lifetime total). Cap stops announce themselves as session caps —
        only the curriculum itself prints 'curriculum complete'.

        stop_at_boundary=True returns after each phase's boundary
        checkpoint instead of rolling into the next phase — the driver
        contract (0.8.1): the caller reads self._last_boundary (the phase
        that just completed; None on a cap stop) and self._interrupted
        (True on KeyboardInterrupt — a manual stop; drivers must NOT
        auto-resume it; the 2s run0 driver silently resumed one).

        resume_after_halt (v3): a run whose manifest records a certified
        guard halt REFUSES to train until a session says so explicitly;
        clearing it is recorded in the manifest and the run continues from
        resume/latest.pt (the last healthy checkpoint — the halt archive
        is the forensic object, never the continuation point)."""
        model, tc = self.model, self.tc
        self._interrupted = False
        self._last_boundary = None
        self._guard_halt = None
        if self.manifest.halt:
            h = self.manifest.halt
            if not resume_after_halt:
                raise RuntimeError(
                    f"HALTED RUN: guard {h.get('guard')} fired at step "
                    f"{h.get('step'):,} in '{h.get('phase')}' (archive "
                    f"{h.get('archive')}; {h.get('info')}) — a red-flag halt "
                    "is never auto-resumed; read the archive, decide, then "
                    "call train(resume_after_halt=True) to continue from the "
                    "last healthy checkpoint")
            self.manifest.note(f"HALT CLEARED by the session at step {self.step:,}: "
                               f"{h}")
            self.manifest.halt = None
            print(f"[train] halt {h.get('guard')}@{h.get('step'):,} cleared — "
                  f"continuing from step {self.step:,}")
        # the GLOBAL batch: per-rank micro batches x accumulation x cards
        tokens_per_step = tc.micro_batch * tc.grad_accum * self.cfg.context * self.world
        t0 = time.time()
        start_step = self.step
        start_tokens = self.manifest.tokens_seen
        if self.is_main:
            print(self.manifest.summary())
        phase = self._open_stream()
        if phase is None:
            print("[train] all phases complete — nothing to do")
            return
        print(f"[train] phase '{phase['name']}' on {phase['dataset']} · "
              f"{tokens_per_step:,} tokens/step · device {self.device}"
              + (f" · {self.world} cards (rank {self.rank})" if self.world > 1 else ""))
        mult = self._phase_mult(phase["name"])
        if self.arms is not None:
            self.arms.sync(phase["name"])
        n_abst = (self.arms.chunks_per_step(tc.grad_accum)
                  if self.arms is not None else 0)
        if n_abst:
            print(f"[train] stage arms {self.arms.attached}: {n_abst} abstention "
                  f"chunks per step beside {tc.grad_accum} task chunks")
        # bar total = whichever bound ends this session first: remaining
        # curriculum budget, max_steps, or max_tokens (max_hours just stops
        # the bar early) — without a total tqdm shows "n/?" and no ETA
        remaining = sum(
            max(0, ph["planned_tokens"] - ph.get("tokens_done", 0))
            for ph in self.manifest.phases
            if ph["status"] in ("planned", "active"))
        total = self.step + max(1, math.ceil(remaining / tokens_per_step))
        if max_steps is not None:
            total = min(total, start_step + max_steps)
        if max_tokens is not None:
            total = min(total, self.step + math.ceil(max_tokens / tokens_per_step))
        bar = (tqdm(unit="step", initial=self.step, total=total, dynamic_ncols=True)
               if self.is_main else None)
        self._bar = bar
        model.train()
        save_on_exit = True
        try:
            while True:
                # the session caps: one decision for every card (the main
                # rank's clock and counters), so all ranks stop together
                cap = None
                if max_steps is not None and self.step - start_step >= max_steps:
                    cap = "[train] SESSION CAP: max_steps this call — a pause, not completion"
                elif max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
                    cap = "[train] SESSION CAP: max_hours this call — a pause, not completion"
                elif max_tokens is not None and \
                        self.manifest.tokens_seen - start_tokens >= max_tokens:
                    cap = (f"[train] SESSION CAP: {max_tokens/1e9:.2f}B tokens "
                           "this call — a pause, not completion")
                if self._agree(cap is not None):
                    if self.is_main:
                        print(cap or "[train] SESSION CAP reached on another card — a pause, not completion")
                    break

                step_t0 = time.time()
                lr_s = apply_lr(self.optimizers, self.base_lrs, self.step,
                                tc.warmup_steps, mult=mult)
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                for _ in range(tc.grad_accum):
                    xb = self.stream.next_batch().to(self.device,
                                                     non_blocking=True)
                    with autocast(self.device):
                        _, loss = model(xb[:, :-1], targets=xb[:, 1:])
                    (loss / tc.grad_accum).backward()
                    loss_acc += float(loss.item()) / tc.grad_accum
                abst_acc = 0.0
                for _ in range(n_abst):        # the quiet term (arms only)
                    abst_acc += self.arms.abstain_backward(1.0 / n_abst) / n_abst
                # multi-card: the gradients (trunk and arms) averaged
                # across ranks BEFORE the clip reads them
                self._allreduce_grads()
                # the clip reads the TRUNK's gradient (arm gradients are
                # never clipped — the arm recipe of record)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self._trunk_params, tc.grad_clip)
                # EVERY step, BEFORE the optimizers touch the weights: a
                # non-finite loss/grad must never enter the parameters (and
                # therefore never reach a resume checkpoint). Multi-card:
                # every rank raises when any rank's loss is non-finite.
                finite = math.isfinite(loss_acc) and bool(torch.isfinite(gnorm))
                if not self._agree(finite, how="all"):
                    raise FloatingPointError(
                        f"non-finite loss/grad at step {self.step + 1} "
                        f"(loss={loss_acc}, gnorm={float(gnorm)}, rank {self.rank}) "
                        "— weights untouched; resume state NOT overwritten")
                if n_abst:
                    # an arm's fault is the arm's: the member is masked out
                    # of the step and the run continues on a healthy trunk
                    # (after the all-reduce the arm gradients are identical
                    # on every rank; the chunk loss is agreed across ranks)
                    bad = self.arms.nonfinite_members()
                    if self._agree(not math.isfinite(abst_acc)) and not bad:
                        bad = list(self.arms.live())
                    for n in bad:
                        reason = (f"non-finite arm gradient at step {self.step + 1} "
                                  f"(abstention={abst_acc})")
                        self.arms.disable(n, reason)
                        self.manifest.note(f"ARM DISABLED '{n}': {reason}")
                        _emit(bar, f"[arms] DISABLED '{n}': {reason}")
                    if bad:
                        abst_acc = 0.0
                        n_abst = self.arms.chunks_per_step(tc.grad_accum)
                for opt in self.optimizers:
                    opt.step()
                if self.arms is not None and self.arms.active:
                    self.arms.step()

                # ANCHOR GOVERNOR (ROUND 5f): post-step min-sep projection,
                # every governor_every steps. Identity when slack (one small
                # matmul per codebook, zero writes); fires only on crowding.
                # Outside the task gradient — the no-balance-machinery law
                # (bank.py) is untouched. Measured on the gov25m rung:
                # census tail 5.4x cut, fragile seed +.007, zero cost.
                if tc.governor == "minsep" and \
                        (self.step % tc.governor_every) == 0:
                    from ..model.governor import govern_model
                    self._gov_hits = getattr(self, "_gov_hits", 0) + \
                        govern_model(self.raw_model, tc.governor_theta)

                self.step += 1
                self.manifest.steps = self.step
                self.manifest.add_tokens(tokens_per_step)
                bpb = loss_acc / math.log(2)
                self._loss_ema = loss_acc if self._loss_ema is None else \
                    0.98 * self._loss_ema + 0.02 * loss_acc
                spike = loss_acc > 2.0 * self._loss_ema + 0.5
                dt = time.time() - step_t0
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(bpb=f"{bpb:.3f}",
                                    toks=f"{self.manifest.tokens_seen/1e9:.3f}B",
                                    tps=f"{tokens_per_step/dt/1e3:.0f}k/s",
                                    phase=phase["name"][:12])

                if self.step % tc.log_every == 0 and self.is_main:
                    w = self.writer
                    w.add_scalar("train/loss", loss_acc, self.step)
                    w.add_scalar("train/bpb", bpb, self.step)
                    w.add_scalar("train/grad_norm", float(gnorm), self.step)
                    w.add_scalar("train/lr_scale", lr_s, self.step)
                    w.add_scalar("train/tokens", self.manifest.tokens_seen,
                                 self.step)
                    w.add_scalar("train/tokens_per_sec",
                                 tokens_per_step / dt, self.step)
                    w.add_scalar("collapse/loss_spike", float(spike), self.step)
                    if n_abst:
                        w.add_scalar("arms/abstention", abst_acc, self.step)
                    if tc.governor:
                        w.add_scalar("governor/hits_cum",
                                     getattr(self, "_gov_hits", 0), self.step)
                    if self.device == "cuda":
                        w.add_scalar("sys/vram_gb",
                                     torch.cuda.max_memory_allocated() / 2**30,
                                     self.step)
                    if self.guard is not None:
                        self.guard.observe_gnorm(self.step, float(gnorm))

                halt = None
                if self.step % tc.health_every == 0 and self.is_main:
                    self._health(bar)
                g_census = (self.guard is not None
                            and self.step % self.guard.cfg.census_every == 0)
                if g_census:
                    # the guard's OWN series: its cadence, one pinned batch,
                    # every arm masked — the certification's conditions
                    self.guard.observe_census(self.step, self._guard_census())
                if self.guard is not None and (
                        self.step % tc.log_every == 0 or g_census):
                    for g, info in self.guard.check(self.step):
                        self.writer.add_scalar(f"guard/{g}", float(info["step"]),
                                               self.step)
                        _emit(bar, f"[guard] {g} FIRED ({info['mode']}) at step "
                                   f"{info['step']:,}: {info}")
                        if info["mode"] == "halt" and halt is None:
                            halt = (g, info)
                if self.step % tc.eval_every == 0 and self.is_main:
                    self._full_eval()
                if self.step % tc.ckpt_every == 0:
                    self._checkpoint()          # every rank (a collective inside)
                if self.step % tc.tb_upload_every == 0:
                    self.writer.flush()
                    self.hub.upload_tensorboard(self.tb_dir)
                # a halt is the main rank's read; every rank hears it
                if self._agree(halt is not None):
                    # a certified red flag: archive the exact position AS
                    # ITS OWN FILE (resume/latest.pt stays at the last
                    # healthy checkpoint), record the halt in the manifest
                    # and RETURN — a diagnostic never crashes the run, and
                    # a halted run is never auto-resumed
                    stream_ranks = self._gather_stream_states()
                    if halt is not None:
                        g, info = halt
                        archive = self._halt_checkpoint(g, info, phase["name"],
                                                        stream_ranks=stream_ranks)
                        self._guard_halt = {"guard": g, "info": info,
                                            "phase": phase["name"], "step": self.step,
                                            "archive": archive}
                        print(f"[train] GUARD HALT: {g} fired at step {info['step']:,} "
                              f"— position archived at {archive}; latest.pt untouched; "
                              "the driver decides")
                    else:
                        self._guard_halt = {"guard": "main-rank halt", "step": self.step,
                                            "phase": phase["name"]}
                    save_on_exit = False      # the faulted state never becomes latest.pt
                    break

                done_name = phase["name"]
                newph = self.manifest.advance_phases()
                if newph is None:
                    # final boundary: archive before the curriculum ends
                    self._checkpoint(final=True, boundary=done_name)
                    self._last_boundary = done_name
                    print("[train] curriculum complete")
                    break
                if newph is not phase and newph["name"] != phase["name"]:
                    # STAGE BOUNDARY: archive an immutable resume point
                    # (weights + optimizer + stream state) so this exact
                    # position can be returned to. resume/latest.pt is
                    # overwritten constantly and cannot serve as a rewind
                    # target — measured need, 2026-08-15.
                    self._checkpoint(boundary=done_name)
                    self._last_boundary = done_name
                    if stop_at_boundary:
                        print(f"[train] BOUNDARY: '{done_name}' complete — "
                              "returning to the driver")
                        break
                    phase = self._open_stream()
                    self.manifest.note(f"switched to phase '{phase['name']}'")
                    mult = self._phase_mult(phase["name"])
                    if self.arms is not None:
                        self.arms.sync(phase["name"])
                        n_abst = self.arms.chunks_per_step(tc.grad_accum)
                        self._broadcast_params()   # the fresh arm identical on every card
        except KeyboardInterrupt:
            self._interrupted = True
            print("\n[train] interrupted — saving resume state")
        except BaseException:
            # a crash (divergence, OOM, bug) must NEVER overwrite the last
            # good resume state on the hub
            save_on_exit = False
            print("\n[train] aborting on exception — resume state on the hub "
                  "left untouched")
            raise
        finally:
            if bar is not None:
                bar.close()
            self._bar = None
            self.manifest.wall_hours += (time.time() - t0) / 3600
            if save_on_exit and self.step > start_step:
                if self._agree(self._weights_finite(), how="all"):
                    self._checkpoint(final=True)
                else:
                    print("[train] REFUSING final checkpoint: non-finite "
                          "weights detected — hub resume state left untouched")
            self.writer.flush()
            self.hub.upload_tensorboard(self.tb_dir)
            if self.is_main:
                print(self.manifest.summary())
            left = sum(max(0, p["planned_tokens"] - p.get("tokens_done", 0))
                       for p in self.manifest.phases
                       if p["status"] in ("planned", "active"))
            if left > 0 and self.is_main:
                print(f"[train] curriculum NOT complete: {left/1e9:.2f}B "
                      "tokens remain in active/planned phases — call "
                      "train() again to continue")

    @torch.no_grad()
    def _weights_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all())
                   for p in self.raw_model.parameters())

    def _phase_mult(self, phase_name: str) -> float:
        """The per-phase LR multiplier (TrainConfig.phase_lr_scale, keyed
        by phase-name prefix; the LONGEST matching prefix wins, so
        'anneal_mix' can differ from 'anneal'); 1.0 = the flat-LR form."""
        best, best_len = 1.0, -1
        for prefix, m in (self.tc.phase_lr_scale or {}).items():
            if str(phase_name).startswith(prefix) and len(prefix) > best_len:
                best, best_len = float(m), len(prefix)
        return best

    def _guard_census(self) -> dict:
        """The guard core's census: the trunk alone (arms masked) on ONE
        batch pinned at first use from the guard's census dataset (val
        head rows) and persisted with the guard state, so the series a
        guard reads is the series it was certified on."""
        g = self.guard
        if g.census_batch is None:
            vs = build_stream(g.cfg.census_dataset, self.tokenizer, self.cfg.context,
                              self.tc.micro_batch, seed=self.tc.seed + 9999,
                              role="val")
            g.census_batch = vs.next_batch()[:2, :-1].cpu()
        xb = g.census_batch.to(self.device)
        if self.arms is not None and self.arms.active:
            with self.arms.all_off():
                return instruments.model_census(self.raw_model, xb)
        return instruments.model_census(self.raw_model, xb)

    def _halt_checkpoint(self, g: str, info: dict, phase_name: str,
                         stream_ranks=None) -> str:
        """The halt archive: resume state at the exact halt position under
        its own name (never latest.pt, no shipping weights, no fp8 — the
        faulted state is a forensic object), the halt recorded in the
        manifest, the manifest pushed. Main rank only."""
        sd, extra = self._side_state()
        if stream_ranks is not None:
            extra["stream_ranks"] = stream_ranks
        stream_state = None
        if self.stream is not None:
            stream_state = {"dataset": self.stream.dataset,
                            "state": self.stream.state_dict(),
                            "phase": phase_name}
        name = f"guard_{g}_step{self.step}.pt"
        self.manifest.record_checkpoint(self.step, "halt", f"resume/{name}")
        self.manifest.halt = {"guard": g, "step": self.step, "phase": phase_name,
                              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "archive": f"resume/{name}", "info": dict(info)}
        self.manifest.note(f"GUARD HALT {g} at step {self.step:,} in '{phase_name}': "
                           f"{info} — archived as resume/{name}")
        path = self.hub.save_resume(self.raw_model, self.optimizers, stream_state,
                                    self.manifest, self.step, archive_as=name,
                                    state_dict=sd, extra=extra, skip_latest=True)
        self.hub.push_manifest(self.manifest)
        return path

    def _side_state(self):
        """(trunk state dict or None, extra payload) for the checkpoint
        writers: with arms attached the trunk ships with plain keys and the
        arms + guard state ride in the resume payload."""
        sd = None
        extra = {}
        if self.arms is not None and self.arms.active:
            sd = trunk_state_dict(self.raw_model)
            extra["arms"] = self.arms.state_dict()
        if self.guard is not None:
            extra["guard"] = self.guard.state_dict()
        return sd, extra

    # ------------------------------------------------------------- health
    def _sample_batch(self):
        """Census sample comes from the VAL set — the census must never
        consume (and silently discard) training-stream data."""
        return self._val()[0][:2, :-1]

    def _health(self, bar=None):
        census = instruments.model_census(self.raw_model, self._sample_batch())
        instruments.log_census_tb(self.writer, census, None, self.step)
        text = instruments.readout(self.step, self.manifest.tokens_seen,
                                   census, None)
        _emit(bar, text)
        return census

    def _val(self):
        """Held-out validation set for the current phase's dataset: the
        dataset's validation split where one exists, else a reserved head
        region the training stream skips (see streams.py). Rebuilt whenever
        the active phase's dataset changes; deterministic per dataset, so
        the gauge is comparable across sessions (within one dataset)."""
        ph = self.manifest.current_phase()
        if ph:
            ds = ph["dataset"]
        else:
            # terminal boundary: current_phase() is None once everything is
            # done/deferred — fall back to the LAST completed phase's
            # dataset so end-of-curriculum ledgers stay on a comparable
            # gauge (the s8 final report silently landed on 'synthetic':
            # doc_count 70 -> 1612, hub_off 5.16 — incomparable artifact,
            # 2026-08-31). 'synthetic' remains the no-phases fallback.
            done = [p for p in self.manifest.phases if p["status"] == "done"]
            ds = done[-1]["dataset"] if done else "synthetic"
        if self._val_batches is None or self._val_dataset != ds:
            vs = build_stream(ds, self.tokenizer, self.cfg.context,
                              self.tc.micro_batch, seed=self.tc.seed + 9999,
                              role="val")
            n = max(1, self.tc.val_tokens //
                    (self.tc.micro_batch * self.cfg.context))
            self._val_batches = [vs.next_batch().to(self.device)
                                 for _ in range(n)]
            self._val_dataset = ds
        return self._val_batches

    def _full_eval(self) -> dict:
        model = self.raw_model
        census = instruments.model_census(model, self._sample_batch())
        ledger = instruments.toggle_ledger(model, self._val())
        if self.arms is not None and self.arms.active:
            # the ledger above is the ARMED read; the bare trunk beside it
            with self.arms.all_off():
                bare = instruments.toggle_ledger(model, self._val())
            ledger["bpb_arms_off"] = bare["bpb_full"]
            ledger["toggle_arms_off"] = bare["bpb_full"] - ledger["bpb_full"]
        can = canary_eval(model, self.tokenizer, self.device,
                          episodes=self.tc.canary_episodes,
                          context=self.cfg.context)
        instruments.log_census_tb(self.writer, census, ledger, self.step)
        for k, v in can.items():
            self.writer.add_scalar(f"canary/{k}", v, self.step)
        sp = instruments.special_token_gauge(self.raw_model, self._val())
        if sp is not None:
            for k, v in sp.items():
                self.writer.add_scalar(f"special/{k}", v, self.step)
        extra = {"canary_acc": f"{can['recall_acc']:.3f}"}
        if sp is not None:
            extra["doc"] = (f"n={sp['doc_count']} bpb={sp['doc_bpb']:.2f} "
                            f"reset={sp['reset_bpb']:.2f}")
        self._say(instruments.readout(
            self.step, self.manifest.tokens_seen, census, ledger,
            extra=extra))
        self._last_eval = {"census": census, "ledger": ledger, "canary": can,
                           "special": sp}
        return self._last_eval

    # -------------------------------------------------------- checkpoints
    def _say(self, msg: str):
        _emit(getattr(self, "_bar", None), msg)

    def _checkpoint(self, final: bool = False, boundary: str | None = None):
        """Every rank calls this (the stream-position gather and the
        parameter broadcast are collectives); the main rank writes and
        uploads."""
        t0 = time.time()
        stream_ranks = self._gather_stream_states()
        self._broadcast_params()          # the cards re-pinned to the record
        if not self.is_main:
            return
        sd, extra = self._side_state()
        if stream_ranks is not None:
            extra["stream_ranks"] = stream_ranks
        st_name = self.hub.save_safetensors(self.raw_model, self.step,
                                            state_dict=sd)
        val_bpb = getattr(self, "_last_eval", {}).get(
            "ledger", {}).get("bpb_full") if hasattr(self, "_last_eval") else None
        self.manifest.record_checkpoint(self.step, "safetensors", st_name,
                                        val_bpb)
        self._ckpt_count += 1
        fp8_note = ""
        if self._ckpt_count % self.tc.fp8_every_ckpts == 0 or final:
            fp8_name = self.hub.save_fp8(self.raw_model, self.step,
                                         state_dict=sd)
            self.manifest.record_checkpoint(self.step, "fp8", fp8_name)
            fp8_note = " + fp8"
        stream_state = None
        if self.stream is not None:
            ph = self.manifest.current_phase()
            stream_state = {"dataset": self.stream.dataset,
                            "state": self.stream.state_dict(),
                            "phase": ph["name"] if ph else None}
            self.manifest.data_state = {"dataset": self.stream.dataset,
                                        "rows_consumed": self.stream.rows_consumed,
                                        "epoch": self.stream.epoch}
        # record BEFORE saving so the snapshot embedded in resume/latest.pt
        # includes its own entry (the summary reads the index tail)
        self.manifest.record_checkpoint(self.step, "resume", "resume/latest.pt")
        self.hub.save_resume(self.raw_model, self.optimizers, stream_state,
                             self.manifest, self.step,
                             archive_as=(f"boundary_{boundary}_step{self.step}.pt"
                                         if boundary else None),
                             state_dict=sd, extra=extra)
        self.hub.push_manifest(self.manifest)
        try:
            st_mb = os.path.getsize(os.path.join(self.out_dir, st_name)) / 2**20
            rs_gb = os.path.getsize(os.path.join(
                self.out_dir, "resume", "latest.pt")) / 2**30
            size_note = f" ({st_mb:.0f}MB{fp8_note} + resume {rs_gb:.2f}GB)"
        except OSError:
            size_note = fp8_note
        nxt = ("session end" if final
               else f"next at step {self.step + self.tc.ckpt_every:,}")
        self._say(f"[ckpt] step {self.step:,}: weights + optimizer + stream "
                  f"state saved & uploaded{size_note} in "
                  f"{time.time() - t0:.0f}s · {nxt}")

    # -------------------------------------------------------------- eval
    def evaluate(self) -> dict:
        if self.stream is None and self._open_stream() is None:
            print("[eval] no active phase; using synthetic stream")
            self.stream = build_stream("synthetic", self.tokenizer,
                                       self.cfg.context, self.tc.micro_batch)
        return self._full_eval()

    def status(self) -> str:
        return self.manifest.summary()


def prepare(preset: str = "mini-beatrix-1", hf_token: str | None = None,
            out_dir: str = "./alephllm_runs", resume: bool = True,
            guard: GuardConfig | None = None, arms=None,
            device: str | None = None) -> Trainer:
    """Notebook entrypoint: build (or resume) a run and report its state.
    device (v3): an explicit target ('cpu' for the toy path) — the single
    GPU writer rule: a toy run must never land on a card that is busy."""
    t = Trainer(preset, hf_token=hf_token, out_dir=out_dir, resume=resume,
                guard=guard, arms=arms, device=device)
    n = t.raw_model.param_count()
    print(f"craft '{t.cfg.name}': {n/1e6:.1f}M params · "
          f"ctx {t.cfg.context} · vocab {t.cfg.vocab_size} · "
          f"hub layers {list(t.cfg.hub_layers)} · device {t.device}")
    print(t.status())
    return t
