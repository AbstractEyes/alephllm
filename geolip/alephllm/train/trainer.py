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


class Trainer:
    def __init__(self, preset: Preset | str, hf_token: str | None = None,
                 out_dir: str = "./alephllm_runs", device: str | None = None,
                 resume: bool = True):
        if isinstance(preset, str):
            preset = get_preset(preset)
        self.preset, self.cfg, self.tc = preset, preset.model, preset.train
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = os.path.join(out_dir, self.cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(self.tc.seed)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = build_tokenizer(self.cfg.tokenizer)
        self.raw_model = AlephLM(self.cfg).to(self.device)
        self.optimizers = build_optimizers(
            self.raw_model, self.tc.muon_lr, self.tc.muon_momentum,
            self.tc.adam_lr)
        self.base_lrs = [self.tc.muon_lr, self.tc.adam_lr]

        self.hub = HubSync(preset.hf_repo, preset.prefix, hf_token,
                           self.out_dir)
        self.manifest = None
        self.stream = None
        self.step = 0
        self._ckpt_count = 0
        self._resumed = self._restore() if resume else False
        # compile AFTER restore; raw_model stays the checkpoint identity —
        # all saves/loads go through it so resume works with compile on
        self.model = (torch.compile(self.raw_model) if self.tc.compile
                      else self.raw_model)
        if self.manifest is None:
            self.manifest = RunManifest.fresh(
                self.cfg.name, self.cfg.to_dict(),
                [dict(ph) for ph in preset.curriculum])
        tb_dir = os.path.join(self.out_dir, "runs")
        self.writer = SummaryWriter(tb_dir)
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
        self._stream_state = payload.get("stream")
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
        return True

    def _open_stream(self):
        ph = self.manifest.advance_phases()
        if ph is None:
            return None
        if self.stream is None or self.stream.dataset != ph["dataset"]:
            self.stream = build_stream(ph["dataset"], self.tokenizer,
                                       self.cfg.context, self.tc.micro_batch,
                                       seed=self.tc.seed)
            st = getattr(self, "_stream_state", None)
            if st and st.get("dataset") == ph["dataset"]:
                self.stream.load_state_dict(st["state"])
            self._stream_state = None
        return ph

    # -------------------------------------------------------------- train
    def train(self, max_steps: int | None = None, max_hours: float | None = None,
              max_tokens: int | None = None):
        """All caps are SESSION-RELATIVE: max_steps/max_tokens count only
        this call's work (max_tokens=12e9 trains 12B tokens NOW, regardless
        of lifetime total). Cap stops announce themselves as session caps —
        only the curriculum itself prints 'curriculum complete'."""
        model, tc = self.model, self.tc
        tokens_per_step = tc.micro_batch * tc.grad_accum * self.cfg.context
        t0 = time.time()
        start_step = self.step
        start_tokens = self.manifest.tokens_seen
        print(self.manifest.summary())
        phase = self._open_stream()
        if phase is None:
            print("[train] all phases complete — nothing to do")
            return
        print(f"[train] phase '{phase['name']}' on {phase['dataset']} · "
              f"{tokens_per_step:,} tokens/step · device {self.device}")
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
        bar = tqdm(unit="step", initial=self.step, total=total,
                   dynamic_ncols=True)
        self._bar = bar
        model.train()
        save_on_exit = True
        try:
            while True:
                if max_steps is not None and self.step - start_step >= max_steps:
                    print("[train] SESSION CAP: max_steps this call — "
                          "a pause, not completion")
                    break
                if max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
                    print("[train] SESSION CAP: max_hours this call — "
                          "a pause, not completion")
                    break
                if max_tokens is not None and \
                        self.manifest.tokens_seen - start_tokens >= max_tokens:
                    print(f"[train] SESSION CAP: {max_tokens/1e9:.2f}B tokens "
                          "this call — a pause, not completion")
                    break

                step_t0 = time.time()
                lr_s = apply_lr(self.optimizers, self.base_lrs, self.step,
                                tc.warmup_steps)
                for opt in self.optimizers:
                    opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                for _ in range(tc.grad_accum):
                    xb = self.stream.next_batch().to(self.device,
                                                     non_blocking=True)
                    with torch.autocast(self.device, dtype=torch.bfloat16,
                                        enabled=self.device == "cuda"):
                        _, loss = model(xb[:, :-1], targets=xb[:, 1:])
                    (loss / tc.grad_accum).backward()
                    loss_acc += float(loss.item()) / tc.grad_accum
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tc.grad_clip)
                # EVERY step, BEFORE the optimizers touch the weights: a
                # non-finite loss/grad must never enter the parameters (and
                # therefore never reach a resume checkpoint)
                if not (math.isfinite(loss_acc) and bool(torch.isfinite(gnorm))):
                    raise FloatingPointError(
                        f"non-finite loss/grad at step {self.step + 1} "
                        f"(loss={loss_acc}, gnorm={float(gnorm)}) — weights "
                        "untouched; resume state NOT overwritten")
                for opt in self.optimizers:
                    opt.step()

                self.step += 1
                self.manifest.steps = self.step
                self.manifest.add_tokens(tokens_per_step)
                bpb = loss_acc / math.log(2)
                self._loss_ema = loss_acc if self._loss_ema is None else \
                    0.98 * self._loss_ema + 0.02 * loss_acc
                spike = loss_acc > 2.0 * self._loss_ema + 0.5
                dt = time.time() - step_t0
                bar.update(1)
                bar.set_postfix(bpb=f"{bpb:.3f}",
                                toks=f"{self.manifest.tokens_seen/1e9:.3f}B",
                                tps=f"{tokens_per_step/dt/1e3:.0f}k/s",
                                phase=phase["name"][:12])

                if self.step % tc.log_every == 0:
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
                    if self.device == "cuda":
                        w.add_scalar("sys/vram_gb",
                                     torch.cuda.max_memory_allocated() / 2**30,
                                     self.step)

                if self.step % tc.health_every == 0:
                    self._health(bar)
                if self.step % tc.eval_every == 0:
                    self._full_eval()
                if self.step % tc.ckpt_every == 0:
                    self._checkpoint()
                if self.step % tc.tb_upload_every == 0:
                    self.writer.flush()
                    self.hub.upload_tensorboard(self.tb_dir)

                done_name = phase["name"]
                newph = self.manifest.advance_phases()
                if newph is None:
                    # final boundary: archive before the curriculum ends
                    self._checkpoint(final=True, boundary=done_name)
                    print("[train] curriculum complete")
                    break
                if newph is not phase and newph["name"] != phase["name"]:
                    # STAGE BOUNDARY: archive an immutable resume point
                    # (weights + optimizer + stream state) so this exact
                    # position can be returned to. resume/latest.pt is
                    # overwritten constantly and cannot serve as a rewind
                    # target — measured need, 2026-08-15.
                    self._checkpoint(boundary=done_name)
                    phase = self._open_stream()
                    self.manifest.note(f"switched to phase '{phase['name']}'")
        except KeyboardInterrupt:
            print("\n[train] interrupted — saving resume state")
        except BaseException:
            # a crash (divergence, OOM, bug) must NEVER overwrite the last
            # good resume state on the hub
            save_on_exit = False
            print("\n[train] aborting on exception — resume state on the hub "
                  "left untouched")
            raise
        finally:
            bar.close()
            self._bar = None
            self.manifest.wall_hours += (time.time() - t0) / 3600
            if save_on_exit and self.step > start_step:
                if self._weights_finite():
                    self._checkpoint(final=True)
                else:
                    print("[train] REFUSING final checkpoint: non-finite "
                          "weights detected — hub resume state left untouched")
            self.writer.flush()
            self.hub.upload_tensorboard(self.tb_dir)
            print(self.manifest.summary())
            left = sum(max(0, p["planned_tokens"] - p.get("tokens_done", 0))
                       for p in self.manifest.phases
                       if p["status"] in ("planned", "active"))
            if left > 0:
                print(f"[train] curriculum NOT complete: {left/1e9:.2f}B "
                      "tokens remain in active/planned phases — call "
                      "train() again to continue")

    @torch.no_grad()
    def _weights_finite(self) -> bool:
        return all(bool(torch.isfinite(p).all())
                   for p in self.raw_model.parameters())

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
        ds = ph["dataset"] if ph else "synthetic"
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
        can = canary_eval(model, self.tokenizer, self.device,
                          episodes=self.tc.canary_episodes,
                          context=self.cfg.context)
        instruments.log_census_tb(self.writer, census, ledger, self.step)
        for k, v in can.items():
            self.writer.add_scalar(f"canary/{k}", v, self.step)
        self._say(instruments.readout(
            self.step, self.manifest.tokens_seen, census, ledger,
            extra={"canary_acc": f"{can['recall_acc']:.3f}"}))
        self._last_eval = {"census": census, "ledger": ledger, "canary": can}
        return self._last_eval

    # -------------------------------------------------------- checkpoints
    def _say(self, msg: str):
        _emit(getattr(self, "_bar", None), msg)

    def _checkpoint(self, final: bool = False, boundary: str | None = None):
        t0 = time.time()
        st_name = self.hub.save_safetensors(self.raw_model, self.step)
        val_bpb = getattr(self, "_last_eval", {}).get(
            "ledger", {}).get("bpb_full") if hasattr(self, "_last_eval") else None
        self.manifest.record_checkpoint(self.step, "safetensors", st_name,
                                        val_bpb)
        self._ckpt_count += 1
        fp8_note = ""
        if self._ckpt_count % self.tc.fp8_every_ckpts == 0 or final:
            fp8_name = self.hub.save_fp8(self.raw_model, self.step)
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
                                         if boundary else None))
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
            out_dir: str = "./alephllm_runs", resume: bool = True) -> Trainer:
    """Notebook entrypoint: build (or resume) a run and report its state."""
    t = Trainer(preset, hf_token=hf_token, out_dir=out_dir, resume=resume)
    n = t.raw_model.param_count()
    print(f"craft '{t.cfg.name}': {n/1e6:.1f}M params · "
          f"ctx {t.cfg.context} · vocab {t.cfg.vocab_size} · "
          f"hub layers {list(t.cfg.hub_layers)} · device {t.device}")
    print(t.status())
    return t
