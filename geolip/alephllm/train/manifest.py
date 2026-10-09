"""RunManifest — what is trained, what is planned, where the run stands.

Lives as `<prefix>/manifest.json` in the HF training repo, updated at
every checkpoint. Pull -> resume: the manifest carries token/step
accounting, per-phase status (the full plan exists here even for phases
that are deliberately not prepped yet), and the checkpoint index.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict


@dataclass
class RunManifest:
    preset: str
    model_config: dict
    created_utc: str = ""
    updated_utc: str = ""
    steps: int = 0
    tokens_seen: int = 0
    wall_hours: float = 0.0
    phases: list = field(default_factory=list)
    # phase: {name, dataset, planned_tokens, tokens_done, status:
    #         planned|active|done|deferred}
    checkpoints: list = field(default_factory=list)
    # checkpoint: {step, tokens, kind: safetensors|resume|fp8, path, val_bpb}
    data_state: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # v3: a certified red-flag halt ({guard, step, phase, archive, info});
    # a halted run refuses to continue until a session clears it
    # explicitly (Trainer.train(resume_after_halt=True))
    halt: dict | None = None
    # v3: the data plane the run was created under ({data_scale, epoch_cap,
    # rebalance_to, recipe_hash}) — asserted on resume so a session cannot
    # silently continue on a different mix (the recipe-fingerprint law)
    data_plane: dict = field(default_factory=dict)
    # 2026-10-09: a weights-only start's record ({source, step, phase,
    # seed_offset, weights, optimizers, guards_added, qk_gains}); None for
    # a run born from scratch. Read by Trainer._phase_seed (the restarted
    # phase's stream seed) and by the record.
    init_from: dict | None = None

    # ------------------------------------------------------------- phases
    def current_phase(self) -> dict | None:
        for ph in self.phases:
            if ph["status"] == "active":
                return ph
        for ph in self.phases:
            if ph["status"] == "planned":
                ph["status"] = "active"
                return ph
        return None

    def advance_phases(self) -> dict | None:
        """Mark the active phase done when its budget is met; activate the
        next planned phase. Returns the (possibly new) active phase."""
        ph = self.current_phase()
        while ph is not None and ph.get("tokens_done", 0) >= ph["planned_tokens"]:
            ph["status"] = "done"
            self.note(f"phase '{ph['name']}' complete at "
                      f"{ph['tokens_done']:,} tokens")
            ph = self.current_phase()
        return ph

    def set_cursor(self, step: int, tokens_per_step: int) -> dict | None:
        """A weights-only start (2026-10-09): place the run at `step` of its
        chronological plan — the earlier phases done at their rounded-up
        step counts, the phase holding the step active with its tokens so
        far, the rest planned; steps and tokens_seen follow. Returns the
        active phase. The rule reproduces the record: at 262,144 tokens a
        step the 2s twins' boundaries fall at 1,145 / 20,219 / 22,890 /
        25,561 / ... / 61,422 exactly (tests/test_control_fixes.py)."""
        self.phases = cursor_at_step(self.phases, step, tokens_per_step)
        self.steps = int(step)
        self.tokens_seen = sum(int(p.get("tokens_done", 0)) for p in self.phases)
        return next((p for p in self.phases if p["status"] == "active"), None)

    def add_tokens(self, n: int):
        self.tokens_seen += n
        ph = self.current_phase()
        if ph is not None:
            ph["tokens_done"] = ph.get("tokens_done", 0) + n

    def note(self, msg: str):
        self.notes.append({"utc": _now(), "msg": msg})

    def record_checkpoint(self, step: int, kind: str, path: str,
                          val_bpb: float | None = None):
        self.checkpoints.append({"step": step, "tokens": self.tokens_seen,
                                 "kind": kind, "path": path,
                                 "val_bpb": val_bpb, "utc": _now()})

    # ---------------------------------------------------------------- io
    def to_json(self) -> str:
        self.updated_utc = _now()
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(text: str) -> "RunManifest":
        return RunManifest(**json.loads(text))

    @staticmethod
    def fresh(preset_name: str, model_config: dict,
              curriculum: list) -> "RunManifest":
        m = RunManifest(preset=preset_name, model_config=model_config,
                        created_utc=_now())
        m.phases = [dict(ph, tokens_done=0) for ph in curriculum]
        m.note("manifest created")
        return m

    def summary(self) -> str:
        lines = [f"run '{self.preset}' — {self.steps:,} steps · "
                 f"{self.tokens_seen/1e9:.3f}B tokens · "
                 f"{self.wall_hours:.1f}h wall"]
        for ph in self.phases:
            done, plan = ph.get("tokens_done", 0), ph["planned_tokens"]
            lines.append(f"  [{ph['status']:^8}] {ph['name']:<18} "
                         f"{ph['dataset']:<14} {done/1e9:.3f}/{plan/1e9:.2f}B")
        if self.checkpoints:
            w = next((c for c in reversed(self.checkpoints)
                      if c["kind"] == "safetensors"), None)
            r = next((c for c in reversed(self.checkpoints)
                      if c["kind"] == "resume"), None)
            if w is not None:
                v = w.get("val_bpb")
                lines.append(f"  last weights: step {w['step']:,}"
                             + (f" · val_bpb {v:.3f}" if v else ""))
            if r is not None:
                lines.append(f"  resume state: step {r['step']:,}")
        return "\n".join(lines)


def cursor_at_step(phases: list, step: int, tokens_per_step: int) -> list:
    """The phases as they stand after `step` steps of the plan taken in
    order: a phase takes ceil(planned_tokens / tokens_per_step) steps (the
    trainer runs whole steps and closes a phase once its budget is met);
    deferred phases are skipped. Pure — returns new dicts. Raises when the
    step lies past the plan's end."""
    out, start, placed = [], 0, False
    for ph in phases:
        ph = dict(ph, tokens_done=0)
        if ph.get("status") == "deferred" or placed:
            out.append(ph)
            continue
        n = math.ceil(ph["planned_tokens"] / tokens_per_step)
        if step >= start + n:
            ph.update(status="done", tokens_done=n * tokens_per_step)
            start += n
        else:
            ph.update(status="active", tokens_done=(step - start) * tokens_per_step)
            placed = True
        out.append(ph)
    if not placed:
        raise ValueError(f"step {step:,} lies past the plan's end ({start:,} steps)")
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
