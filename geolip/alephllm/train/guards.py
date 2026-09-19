"""Red-flag guards — the in-run interrupt core (v3 routine, pre-flight L6).

Three detectors over series the trainer already logs, each a pure function
over (step, value) samples so a stored run can be replayed offline:

  G1  PRE-CLIP GRADIENT NORM, rolling and RELATIVE: the rolling median of
      the last `g1_window` logged norms >= `g1_ratio` x the reference
      median AND the rolling clip fraction (share of samples above the
      clip) >= the reference clip fraction + `g1_clip_points`.
  G2  DISPATCH ENTROPY, any layer, sustained: a bank's dispatch-entropy
      fraction <= (1 - `g2_drop`) x that layer's reference median for
      `g2_sustain` consecutive census points. `g2_exempt_last` blocks at
      the tail are skipped (0 = the registered any-layer form; 1 = the
      labeled variant that exempts the byte funnel, like G3).
  G3  HIDDEN ERANK, literal collapse: for a non-exempt block, v(t) <
      `g3_collapse` x v(t-1) between consecutive census points; the last
      `g3_exempt_last` blocks (the byte-head funnel) are exempt.

The reference is a PINNED step window (lo, hi] per craft — a fresh craft
has no earlier phase to look back on. Ratios and windows are the
re-registered constants of the certification run; the window is a
per-craft setting (after warmup, before the first curriculum shift).

Modes per guard: "halt" (the trainer archives the halt position as its
own resume file, leaves resume/latest.pt at the last healthy checkpoint,
records the halt in the manifest and RETURNS cleanly — never raises; a
halted run refuses to continue until a session says so explicitly),
"watch" (logged + reported only — a diagnostic BESIDE the core, not a
core member; the only way into the core is a certification pass), "off".
Library defaults are "watch" for every guard: a guard halts only once its
certification (fires on its induced fault within the window AND stays
silent on healthy seeds) is on record and copied into the config.

Cadence transfers with the form: the certification ran the census every
100 steps and the norm sample every 50, so the core keeps its OWN census
cadence (`census_every`) on ONE fixed batch (`census_dataset` head rows,
pinned at first use and persisted) with every arm masked — the series is
the trunk's, as in the certification.

G4 (the block-14 identity drift, a gradual read needing the frame-map
instrument), G5 (arm family-exam collapse) and G6 (termination
regression at anneal checkpoints) are boundary reads the notebook's
report records through `record_boundary_read`; they carry a mode here
(watch by default) but no in-run evaluator — their instruments live in
the arm ladder's tools, not in this module.

Born-in census flags (instruments.model_census) are recorded beside the
guards but are NOT interrupts: they fire on healthy births (the erank
floor at the byte funnel, the head-buried tripwire before revival).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

GUARDS = ("G1", "G2", "G3")                 # in-run evaluators
BOUNDARY_GUARDS = ("G4", "G5", "G6")        # boundary reads (recorded)


@dataclass
class GuardConfig:
    ref_window: tuple = (1000, 2000)      # steps (lo, hi], the reference
    log_every: int = 50                   # gnorm sample cadence (trainer.log_every)
    census_every: int = 100               # the guard's own census cadence
    census_dataset: str = "fineweb-edu"   # the fixed census batch's source (val head rows)
    clip: float = 1.0                     # the clip the norms are compared to
    g1_ratio: float = 10.0
    g1_clip_points: float = 0.40
    g1_window: int = 20                   # samples (20 x log_every steps)
    g2_drop: float = 0.25
    g2_sustain: int = 3                   # consecutive census points
    g2_exempt_last: int = 0               # 0 = registered any-layer form
    g3_collapse: float = 0.10
    g3_exempt_last: int = 2               # re-registration width (replay used 1)
    modes: dict = field(default_factory=lambda: {
        g: "watch" for g in GUARDS + BOUNDARY_GUARDS})
    certification: dict = field(default_factory=dict)   # copied from the L6 ledger

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ref_window"] = list(self.ref_window)
        return d


# ------------------------------------------------------------- pure forms
def g1_eval(steps, gnorms, cfg: GuardConfig):
    """First step at which G1 fires over logged (step, pre-clip norm)
    samples, or None; plus the reference statistics."""
    s = np.asarray(steps)
    g = np.asarray(gnorms, dtype=float)
    lo, hi = cfg.ref_window
    ref = (s > lo) & (s <= hi)
    if ref.sum() < 5:
        return None, {"note": "reference window incomplete"}
    ref_med = float(np.median(g[ref]))
    ref_clip = float((g[ref] > cfg.clip).mean())
    R = {"ref_median": round(ref_med, 5), "ref_clip_frac": round(ref_clip, 3),
         "n_ref": int(ref.sum())}
    if ref_clip + cfg.g1_clip_points > 1.0:
        # the clip-fraction clause cannot be met from this reference: G1
        # is structurally unreachable on this craft — reported, not hidden
        R["unreachable"] = (f"reference clip fraction {ref_clip:.2f} + "
                            f"{cfg.g1_clip_points} > 1: G1 cannot fire")
    for i in np.where(s > hi)[0]:
        if i + 1 < cfg.g1_window:
            continue
        w = g[i + 1 - cfg.g1_window:i + 1]
        med, clip = float(np.median(w)), float((w > cfg.clip).mean())
        if med >= cfg.g1_ratio * ref_med and clip >= ref_clip + cfg.g1_clip_points:
            R.update({"fire_median": round(med, 4), "fire_clip_frac": round(clip, 3)})
            return int(s[i]), R
    return None, R


def g2_eval(steps, ent_by_layer, n_layers: int, cfg: GuardConfig):
    """ent_by_layer: {layer: [entropy fraction per census]} aligned with
    steps. Any non-exempt layer <= (1 - drop) x its reference for
    `g2_sustain` consecutive census points."""
    s = np.asarray(steps)
    lo, hi = cfg.ref_window
    ref = (s > lo) & (s <= hi)
    if ref.sum() < 3:
        return None, {"note": "reference window incomplete"}
    R = {"ref_median_by_layer": {}}
    first = None
    for L, series in ent_by_layer.items():
        if int(L) >= n_layers - cfg.g2_exempt_last:
            continue
        e = np.asarray(series, dtype=float)
        rm = float(np.nanmedian(e[ref]))
        R["ref_median_by_layer"][str(L)] = round(rm, 4)
        run = 0
        for i in np.where(s > hi)[0]:
            run = run + 1 if e[i] <= (1 - cfg.g2_drop) * rm else 0
            if run >= cfg.g2_sustain:
                if first is None or s[i] < first[0]:
                    first = (int(s[i]), int(L), round(float(e[i]), 4))
                break
    if first:
        R.update({"fire_layer": first[1], "fire_value": first[2]})
        return first[0], R
    return None, R


def g3_eval(steps, erank_by_layer, n_layers: int, cfg: GuardConfig,
            exempt_last: int | None = None):
    """Literal collapse v(t) < collapse x v(t-1) between consecutive
    census points on any non-exempt block."""
    s = np.asarray(steps)
    ex = cfg.g3_exempt_last if exempt_last is None else exempt_last
    first = None
    for L, series in erank_by_layer.items():
        if int(L) >= n_layers - ex:
            continue
        v = np.asarray(series, dtype=float)
        for i in range(1, len(v)):
            if v[i] < cfg.g3_collapse * v[i - 1] and (first is None or s[i] < first[0]):
                first = (int(s[i]), int(L), round(float(v[i - 1]), 2),
                         round(float(v[i]), 2))
                break
    if first:
        return first[0], {"fire_layer": first[1], "prev": first[2],
                          "now": first[3], "exempt_last": ex}
    return None, {"exempt_last": ex}


# --------------------------------------------------------------- the core
class GuardCore:
    """Accumulates the trainer's samples and evaluates the three forms.

    observe_gnorm(step, gnorm) at the log cadence; observe_census(step,
    census) at the health cadence (census = instruments.model_census
    output). check(step) returns the NEW firings since the last call as
    [(guard, info)]; a guard fires once. Mode "off" guards are never
    evaluated; the state round-trips through state_dict for resume."""

    def __init__(self, cfg: GuardConfig, n_layers: int):
        self.cfg, self.n_layers = cfg, n_layers
        self.g_steps, self.gnorms = [], []
        self.c_steps = []
        self.ent = {str(L): [] for L in range(n_layers)}
        self.erank = {str(L): [] for L in range(n_layers)}
        self.flags_first = {}
        self.fired = {}          # guard -> {"step": .., **info}
        self.census_batch = None  # pinned at first use (trainer), persisted
        self.boundary_reads = {}  # G4/G5/G6 records by boundary tag

    # ---- feeding
    def record_boundary_read(self, guard: str, tag: str, record: dict):
        """G4/G5/G6: a boundary instrument's reading, recorded (never an
        in-run interrupt here); `record` carries its own 'fired' flag."""
        self.boundary_reads.setdefault(guard, {})[str(tag)] = dict(record)
        if record.get("fired") and guard not in self.fired:
            self.fired[guard] = {"step": record.get("step"), "tag": str(tag),
                                 "mode": self.mode(guard), **record}
    def observe_gnorm(self, step: int, gnorm: float):
        self.g_steps.append(int(step))
        self.gnorms.append(float(gnorm))

    def observe_census(self, step: int, census: dict):
        self.c_steps.append(int(step))
        for L in range(self.n_layers):
            li = census["layers"].get(L, {})
            self.ent[str(L)].append(float(li.get("bank_dispatch_entropy_frac",
                                                 float("nan"))))
            self.erank[str(L)].append(float(li.get("hidden_erank", float("nan"))))
        for k, v in census.get("flags", {}).items():
            if v and k not in self.flags_first:
                self.flags_first[k] = {"step": int(step),
                                       "detail": census.get("flag_detail", {}).get(k)}

    # ---- evaluation
    def mode(self, g: str) -> str:
        return self.cfg.modes.get(g, "watch")

    def evaluate(self) -> dict:
        out = {}
        if self.mode("G1") != "off":
            f, r = g1_eval(self.g_steps, self.gnorms, self.cfg)
            out["G1"] = {"first_fire": f, **r}
        if self.mode("G2") != "off":
            f, r = g2_eval(self.c_steps, self.ent, self.n_layers, self.cfg)
            out["G2"] = {"first_fire": f, **r}
        if self.mode("G3") != "off":
            f, r = g3_eval(self.c_steps, self.erank, self.n_layers, self.cfg)
            out["G3"] = {"first_fire": f, **r}
        return out

    def check(self, step: int) -> list:
        new = []
        for g, rec in self.evaluate().items():
            if rec["first_fire"] is not None and g not in self.fired:
                info = dict(rec)
                info["step"] = int(rec["first_fire"])
                info["seen_at"] = int(step)
                info["mode"] = self.mode(g)
                self.fired[g] = info
                new.append((g, info))
        return new

    def halting(self, new_firings) -> list:
        return [(g, i) for g, i in new_firings if self.mode(g) == "halt"]

    def summary(self) -> dict:
        return {"config": self.cfg.to_dict(), "fired": self.fired,
                "evaluation": self.evaluate(),
                "library_flags_first": self.flags_first,
                "boundary_reads": self.boundary_reads,
                "n_gnorm_samples": len(self.gnorms),
                "n_census_points": len(self.c_steps)}

    # ---- resume
    def state_dict(self) -> dict:
        return {"g_steps": list(self.g_steps), "gnorms": list(self.gnorms),
                "c_steps": list(self.c_steps), "ent": {k: list(v) for k, v in self.ent.items()},
                "erank": {k: list(v) for k, v in self.erank.items()},
                "flags_first": dict(self.flags_first), "fired": dict(self.fired),
                "census_batch": (self.census_batch.cpu() if self.census_batch is not None
                                 else None),
                "boundary_reads": dict(self.boundary_reads)}

    def load_state_dict(self, st: dict):
        self.g_steps = list(st.get("g_steps", []))
        self.gnorms = list(st.get("gnorms", []))
        self.c_steps = list(st.get("c_steps", []))
        for k in self.ent:
            self.ent[k] = list(st.get("ent", {}).get(k, []))
            self.erank[k] = list(st.get("erank", {}).get(k, []))
        self.flags_first = dict(st.get("flags_first", {}))
        self.fired = dict(st.get("fired", {}))
        self.census_batch = st.get("census_batch")
        self.boundary_reads = dict(st.get("boundary_reads", {}))


def certification_from_ledger(ledger: dict) -> dict:
    """Read a certification verdict (the pre-flight L6 ledger's 'verdict'
    -> 'certification' block) into per-guard modes: CERTIFIED -> "halt",
    otherwise "watch". The G3 verdict is taken at the configured width."""
    cert = (ledger.get("verdict") or {}).get("certification") or {}
    modes = {}
    for g, keys in (("G1", ("G1",)), ("G2", ("G2",)),
                    ("G3", ("G3_exempt2", "G3_exempt1"))):
        rec = next((cert[k] for k in keys if k in cert), None)
        modes[g] = "halt" if rec and rec.get("CERTIFIED") else "watch"
    return {"modes": modes, "certification": cert}
