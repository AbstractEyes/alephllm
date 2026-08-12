"""RunManifest — what is trained, what is planned, where the run stands.

Lives as `<prefix>/manifest.json` in the HF training repo, updated at
every checkpoint. Pull -> resume: the manifest carries token/step
accounting, per-phase status (the full plan exists here even for phases
that are deliberately not prepped yet), and the checkpoint index.
"""
from __future__ import annotations

import json
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
            c = self.checkpoints[-1]
            lines.append(f"  last checkpoint: step {c['step']:,} ({c['kind']}) "
                         f"val_bpb={c.get('val_bpb')}")
        return "\n".join(lines)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
