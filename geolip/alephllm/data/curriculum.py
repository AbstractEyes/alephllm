"""Early-life curriculum — stages S0..S8 over the locked core.

Plan of record: claude-mind history/plans/2026-08-15_early_life_curriculum.md
Laws in force here:
- NO chat template, NO identity rows, NO boilerplate headers. Dialogue
  enters only as narrative quotation. (chat-in-anneal law)
- Causal-test holdouts are enforced AT SOURCE: held-out families never
  reach a training row, so stage-boundary arm groups train on genuinely
  unseen structure. Holdout registry: HOLDOUTS below.
- Gauge continuity: every stage vals on the fineweb-edu 2013 holdout
  (build_stream dispatch); stage-specific measurement is the probe
  suite's job (train/probes.py).

Importing this module registers everything into streams (REGISTRY,
_RENDERERS, _GENERATORS, CURRICULUM_MIXES).
"""
from __future__ import annotations

import numpy as np

from .streams import (REGISTRY, _GENERATORS, _RENDERERS, CURRICULUM_MIXES)

# ---------------------------------------------------------------- holdouts
# Families excluded from ALL curriculum training rows; the stage-boundary
# arm collectives train on exactly these. (plan: "Arm collectives")
HOLDOUTS = {
    "babi_tasks": {"16", "19"},        # basic induction, path finding
    "proofwriter_depth": 5,            # depth-5 rule chains (QDep >= 5)
    "rulechain_depth": 5,              # owned generator honors the same cut
    "arith_family": "sub3",            # 3-digit subtraction
}

# ---------------------------------------------------------------- renderers
def _render_cosmo_young(row) -> str:
    """cosmopedia-v2 filtered to young audiences; '' rejects the row."""
    aud = str(row.get("audience", ""))
    if aud not in ("young_children", "middle_school_students"):
        return ""
    return str(row.get("text", ""))


def _render_fineweb_good(row) -> str:
    """fineweb-edu, int_score >= 4 only — the good 15% of her old diet."""
    try:
        if int(row.get("int_score") or 0) < 4:
            return ""
    except (TypeError, ValueError):
        return ""
    return str(row.get("text", ""))


def _render_siqa(row) -> str:
    """SIQA row -> short social narrative with the correct answer folded
    in as the outcome sentence (no Q/A scaffolding survives)."""
    ctx = str(row.get("context", "")).strip()
    q = str(row.get("question", "")).strip().rstrip("?")
    try:
        label = int(row.get("label"))
    except (TypeError, ValueError):
        return ""
    ans = str(row.get(f"answer{'ABC'[label - 1]}", "")).strip()
    if not (ctx and q and ans):
        return ""
    return f"{ctx} Asked {q.lower()}? The answer was: {ans}."


def _render_babi(row) -> str:
    """bAbI row -> passage + question + answer as plain prose. Held-out
    task families are rejected at source."""
    if str(row.get("task", "")) in HOLDOUTS["babi_tasks"]:
        return ""
    p = str(row.get("passage", "")).strip()
    q = str(row.get("question", "")).strip()
    a = str(row.get("answer", "")).strip()
    if not (p and q and a):
        return ""
    return f"{p}\n{q} {a.capitalize()}."


def _render_proofwriter(row) -> str:
    """ProofWriter -> theory + question + verdict prose; depth holdout
    rejected at source."""
    try:
        if int(row.get("QDep") or 0) >= HOLDOUTS["proofwriter_depth"]:
            return ""
    except (TypeError, ValueError):
        return ""
    t = str(row.get("theory", "")).strip()
    q = str(row.get("question", "")).strip().rstrip(".")
    a = str(row.get("answer", "")).strip()
    if not (t and q and a):
        return ""
    verdict = {"True": "So it is true that",
               "False": "So it is false that"}.get(a, "It is unknown whether")
    return f"{t} {verdict} {q[0].lower() + q[1:]}."


_ATOMIC_TEMPLATES = {
    "xEffect": "After {e}, {x} {t}.",
    "oEffect": "After {e}, the others {t}.",
    "xReact": "After {e}, {x} feels {t}.",
    "oReact": "After {e}, the others feel {t}.",
    "xWant": "After {e}, {x} wants {t}.",
    "oWant": "After {e}, the others want {t}.",
    "xNeed": "Before {e}, {x} needs {t}.",
    "xIntent": "{x} does this because {t}.",
    "xAttr": "Doing this shows that {x} is {t}.",
    "isBefore": "{e} happens before {t}.",
    "isAfter": "{e} happens after {t}.",
    "Causes": "{e} causes {t}.",
    "HinderedBy": "{e} can be blocked when {t}.",
    "xReason": "{x} does this because {t}.",
    "HasSubEvent": "While {e}, {t}.",
}
_NAMES = ["Ada", "Bern", "Cato", "Dima", "Eli", "Fern", "Gil", "Hana"]


def _render_atomic(row) -> str:
    """ATOMIC-2020 triple -> one causal flicker sentence. PersonX/Y get
    stable names hashed from the event so co-references line up."""
    rel = str(row.get("relation", ""))
    tmpl = _ATOMIC_TEMPLATES.get(rel)
    ev = str(row.get("event", "")).strip()
    tails = row.get("tail") or []
    if isinstance(tails, str):
        tails = [tails]
    tails = [str(t).strip() for t in tails
             if str(t).strip().lower() not in ("", "none")]
    if not (tmpl and ev and tails):
        return ""
    h = abs(hash(ev))
    x = _NAMES[h % len(_NAMES)]
    y = _NAMES[(h // 7 + 1) % len(_NAMES)]
    ev = ev.replace("PersonX", x).replace("PersonY", y).replace("___", "it")
    out = []
    for t in tails[:3]:
        t = t.replace("PersonX", x).replace("PersonY", y)
        out.append(tmpl.format(e=ev[0].lower() + ev[1:], x=x, t=t))
    return " ".join(out)


# ---------------------------------------------------------------- generators
def _counting_rows(seed: int):
    rng = np.random.default_rng(seed)
    things = ["apple", "stone", "bird", "cup", "leaf", "coin", "star",
              "shell", "bead", "drum"]
    while True:
        n = int(rng.integers(2, 9))
        t = things[int(rng.integers(0, len(things)))]
        seq = ", ".join(str(i) for i in range(1, n + 1))
        yield {"text": f"Mira counts the {t}s: {seq}. There are {n} {t}s. "
                       f"One more {t} arrives. Now there are {n + 1} {t}s."}


def _perspective_rows(seed: int):
    """Same event in three persons + Sally-Anne class false-belief
    stories. Teaches perspective as CONCEPT; no identity, no template."""
    rng = np.random.default_rng(seed)
    names = ["Sana", "Tom", "Lea", "Rui", "Maya", "Kofi"]
    objects = ["marble", "ribbon", "acorn", "spoon", "crayon"]
    places = ["basket", "box", "drawer", "jar", "bag"]
    while True:
        a, b = rng.choice(len(names), size=2, replace=False)
        a, b = names[int(a)], names[int(b)]
        o = objects[int(rng.integers(0, len(objects)))]
        p1, p2 = rng.choice(len(places), size=2, replace=False)
        p1, p2 = places[int(p1)], places[int(p2)]
        if rng.random() < 0.5:
            yield {"text":
                   f"{a} put the {o} in the {p1} and went outside. While "
                   f"{a} was away, {b} moved the {o} to the {p2}. When {a} "
                   f"came back, {a} looked for the {o} in the {p1}, because "
                   f"{a} did not see it move. The {o} was really in the "
                   f"{p2}."}
        else:
            ev = f"dropped the {o} near the {p1}"
            yield {"text":
                   f"{a} {ev}. Seen from {a}: I {ev}. Said to {a}: you "
                   f"{ev}. Told about {a}: {a} {ev}. Three ways of saying, "
                   f"one thing that happened."}


def _concept_flicker_rows(seed: int):
    rng = np.random.default_rng(seed)
    kinds = [("robin", "bird", "fly"), ("trout", "fish", "swim"),
             ("oak", "tree", "grow tall"), ("beetle", "insect", "crawl"),
             ("whale", "mammal", "swim"), ("rose", "flower", "bloom"),
             ("granite", "rock", "stay hard"), ("maple", "tree", "grow")]
    pairs = [("big", "small"), ("hot", "cold"), ("wet", "dry"),
             ("fast", "slow"), ("heavy", "light"), ("open", "shut")]
    while True:
        r = rng.random()
        if r < 0.45:
            a, k, v = kinds[int(rng.integers(0, len(kinds)))]
            b = kinds[int(rng.integers(0, len(kinds)))][0]
            yield {"text": f"A {a} is a kind of {k}. Most {k}s can {v}. "
                           f"A {a} is not a {b}; they differ in kind."}
        elif r < 0.8:
            x, y = pairs[int(rng.integers(0, len(pairs)))]
            yield {"text": f"{x.capitalize()} is the opposite of {y}. If a "
                           f"thing is {x}, it is not {y}."}
        else:
            a, k, _ = kinds[int(rng.integers(0, len(kinds)))]
            c, k2, _ = kinds[int(rng.integers(0, len(kinds)))]
            same = "the same kind" if k == k2 else "different kinds"
            yield {"text": f"A {a} and a {c}: are they alike? A {a} is a "
                           f"{k} and a {c} is a {k2}, so they are {same}."}


_PRED = ["blim", "torv", "quen", "harl", "sook", "vell", "mund", "prin"]


def _rulechain_rows(seed: int, max_depth: int | None = None):
    """Prose modus-ponens chains over nonsense predicates (no world
    knowledge shortcut). Training stream stays BELOW the held-out depth."""
    rng = np.random.default_rng(seed)
    cap = (max_depth if max_depth is not None
           else HOLDOUTS["rulechain_depth"] - 1)
    names = ["Pia", "Ezo", "Kel", "Vin", "Osa", "Tam"]
    while True:
        d = int(rng.integers(1, cap + 1))
        ps = [(_PRED[int(i)]) for i in
              rng.choice(len(_PRED), size=d + 1, replace=False)]
        who = names[int(rng.integers(0, len(names)))]
        rules = [f"If someone is {ps[i]}, then they are {ps[i + 1]}."
                 for i in range(d)]
        neg = rng.random() < 0.25
        if neg:
            chain = (f"{who} is not {ps[-1]}. "
                     + " ".join(reversed([f"So {who} is not {ps[i]}."
                                          for i in range(d)])))
            yield {"text": " ".join(rules) + f" {chain}"}
        else:
            steps = " ".join(f"So {who} is {ps[i + 1]}." for i in range(d))
            yield {"text": " ".join(rules) + f" {who} is {ps[0]}. {steps}"}


def _arith_rows(seed: int, include_holdout: bool = False):
    """Digit-level arithmetic in prose with carry work shown. The held-out
    family (3-digit subtraction) never appears in training rows."""
    rng = np.random.default_rng(seed)
    while True:
        fam = str(rng.choice(["add2", "add3", "sub2", "mul1", "sub3"]))
        if fam == "sub3" and not include_holdout:
            continue
        if fam == "add2":
            a, b = int(rng.integers(10, 99)), int(rng.integers(10, 99))
            yield {"text": f"Take {a} and add {b}. {a} + {b} = {a + b}."}
        elif fam == "add3":
            a, b = int(rng.integers(100, 999)), int(rng.integers(100, 999))
            yield {"text": f"Take {a} and add {b}. {a} + {b} = {a + b}."}
        elif fam == "sub2":
            a, b = sorted([int(rng.integers(10, 99)),
                           int(rng.integers(10, 99))], reverse=True)
            yield {"text": f"Start at {a} and take away {b}. "
                           f"{a} - {b} = {a - b}."}
        elif fam == "mul1":
            a, b = int(rng.integers(2, 9)), int(rng.integers(2, 9))
            rows = "; ".join(f"{i} {a}s make {a * i}"
                             for i in range(1, b + 1))
            yield {"text": f"Count by {a}s: {rows}. So {a} times {b} "
                           f"is {a * b}."}
        else:
            a, b = sorted([int(rng.integers(100, 999)),
                           int(rng.integers(100, 999))], reverse=True)
            yield {"text": f"Start at {a} and take away {b}. "
                           f"{a} - {b} = {a - b}."}


def _causal_arc_rows(seed: int):
    rng = np.random.default_rng(seed)
    arcs = [("left the gate open", "the goat walked out",
             "the garden rows were eaten", "closed the gate first",
             "the garden stayed whole"),
            ("watered the seeds each day", "green shoots came up",
             "flowers opened in spring", "forgot to water them",
             "the pots stayed bare"),
            ("stacked the cups too high", "the tower leaned",
             "the cups crashed down", "stacked only three",
             "the tower stood"),
            ("put the bread out too long", "the crust went hard",
             "the birds got the loaf", "wrapped the bread up",
             "it stayed soft for morning")]
    names = ["Noor", "Ivo", "Sela", "Bram"]
    while True:
        c, e1, e2, alt, alte = arcs[int(rng.integers(0, len(arcs)))]
        n = names[int(rng.integers(0, len(names)))]
        if rng.random() < 0.6:
            yield {"text": f"{n} {c}. Because of that, {e1}. Because of "
                           f"that, {e2}."}
        else:
            yield {"text": f"{n} {c}, so {e1}, and then {e2}. If {n} had "
                           f"{alt}, then {alte}."}


def _try_fail_rows(seed: int):
    """Attempt -> check -> find the error -> correct -> succeed, in
    neutral third person. Reward rendered as textual consequence."""
    rng = np.random.default_rng(seed)
    names = ["Rina", "Josef", "Talia", "Emre"]
    while True:
        n = names[int(rng.integers(0, len(names)))]
        if rng.random() < 0.5:
            a, b = int(rng.integers(20, 90)), int(rng.integers(10, 60))
            wrong = a + b + int(rng.integers(1, 10))
            yield {"text":
                   f"{n} adds {a} and {b} and writes {wrong}. {n} checks "
                   f"by counting back: {wrong} is too big. {n} tries "
                   f"again carefully: {a} + {b} = {a + b}. The check "
                   f"works now, and {n} keeps the good method."}
        else:
            yield {"text":
                   f"{n} builds a paper bridge and it sags. {n} looks at "
                   f"where it bends: the middle has no fold. {n} folds a "
                   f"ridge down the middle and builds again. The bridge "
                   f"holds. Folding made it strong, so {n} folds first "
                   f"every time after."}


_GLOSSARY = [
    ("river", "a wide stream of water that flows across land"),
    ("island", "land with water all the way around it"),
    ("promise", "words that say you will surely do a thing"),
    ("repair", "to make a broken thing work again"),
    ("gather", "to bring things together into one place"),
    ("ancient", "from a very long time ago"),
    ("fragile", "easy to break and needing gentle hands"),
    ("observe", "to watch something closely to learn about it"),
    ("compare", "to look at two things to see how they differ"),
    ("cause", "the thing that makes another thing happen"),
]


def _register_rows(seed: int):
    """Definitions + the same content said three ways: THAT registers
    exist is the lesson; no register is marked as 'hers'."""
    rng = np.random.default_rng(seed)
    facts = [("the rain filled the barrel by morning",
              "precipitation had filled the barrel before morning",
              "the rain filled the barrel up while everyone slept"),
             ("the bridge is closed and walkers must use the ferry",
              "the bridge is closed; pedestrians are directed to the ferry",
              "the bridge is shut, so people walk onto the ferry instead")]
    while True:
        if rng.random() < 0.55:
            w, g = _GLOSSARY[int(rng.integers(0, len(_GLOSSARY)))]
            yield {"text": f"The word '{w}' means {g}. Used in a "
                           f"sentence: a {w} is easy to point at once "
                           f"you know the word."}
        else:
            p, f, c = facts[int(rng.integers(0, len(facts)))]
            yield {"text": f"Plainly: {p}. Formally: {f}. For a child: "
                           f"{c}. Three sayings, one fact."}


# ---------------------------------------------------------------- registry
REGISTRY.update({
    "aochildes": dict(path="deven367/babylm-100M-aochildes",
                      split="train", column="text"),
    "simple-wiki": dict(path="wikimedia/wikipedia", name="20231101.simple",
                        split="train", column="text", columns=["text"]),
    "cosmo-young": dict(path="HuggingFaceTB/smollm-corpus",
                        name="cosmopedia-v2", split="train", column="text",
                        columns=["text", "audience"], render="cosmo_young",
                        max_empties=50_000),
    "fineweb-good": dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                         split="train", column="text",
                         columns=["text", "int_score"],
                         render="fineweb_good", max_empties=50_000),
    "siqa-narrative": dict(path="lighteval/siqa", split="train",
                           column="context", render="siqa",
                           columns=["context", "question", "answerA",
                                    "answerB", "answerC", "label"]),
    "babi-prose": dict(path="Muennighoff/babi", split="train",
                       column="passage", render="babi",
                       columns=["passage", "question", "answer", "task"],
                       max_empties=20_000),
    "proofwriter-prose": dict(path="tasksource/proofwriter", split="train",
                              column="theory", render="proofwriter",
                              columns=["theory", "question", "answer",
                                       "QDep"], max_empties=20_000),
    "atomic-flicker": dict(path="Estwld/atomic2020-origin", split="train",
                           column="event", render="atomic",
                           columns=["knowledge_type", "event", "relation",
                                    "relation_description", "tail"],
                           max_empties=20_000),
    # sciq spec present but OUT of default mixes: CC-BY-NC license pending
    # Phil's call (plan: Vetting results / FLAGS).
    "sciq-prose": dict(path="allenai/sciq", split="train",
                       column="support"),
    "counting-synth": dict(path=None, generator="counting"),
    "perspective-synth": dict(path=None, generator="perspective"),
    "concept-synth": dict(path=None, generator="concept"),
    "rulechain-synth": dict(path=None, generator="rulechain"),
    "arith-synth": dict(path=None, generator="arith"),
    "causal-synth": dict(path=None, generator="causal"),
    "tryfail-synth": dict(path=None, generator="tryfail"),
    "register-synth": dict(path=None, generator="register"),
})

_RENDERERS.update({
    "cosmo_young": _render_cosmo_young,
    "fineweb_good": _render_fineweb_good,
    "siqa": _render_siqa,
    "babi": _render_babi,
    "proofwriter": _render_proofwriter,
    "atomic": _render_atomic,
})

_GENERATORS.update({
    "counting": _counting_rows,
    "perspective": _perspective_rows,
    "concept": _concept_flicker_rows,
    "rulechain": _rulechain_rows,
    "arith": _arith_rows,
    "causal": _causal_arc_rows,
    "tryfail": _try_fail_rows,
    "register": _register_rows,
})

# ---------------------------------------------------------- corpus sizes
# Usable bytes AFTER render+holdout filtering (measured 2026-08-15:
# rows x mean rendered bytes over a 300-row sample). float('inf') =
# procedural generator or a corpus far larger than any stage budget.
# These exist so weights can be checked against reality — see
# audit_epochs(). A finite corpus asked for more than ~2 epochs is a
# memorization risk, and the guard in amoe/train phrases it exactly so:
# "the memorization becomes the arm".
INF = float("inf")
CORPUS_BYTES = {
    "babi-prose": 6.6e6,          # 18,013 rows, 10% held out (tasks 16/19)
    "siqa-narrative": 5.1e6,      # 33,410 rows x ~152 B
    "proofwriter-prose": 243.3e6,  # 585,552 rows x ~416 B
    "atomic-flicker": 187.7e6,    # ~1.3M triples x ~144 B
    "tinystories": 1911.7e6,
    "simple-wiki": 694.6e6,
    "aochildes": 3.2e6,           # ~11MB of very short utterances
    "cosmo-young": INF, "fineweb-good": INF, "fineweb-edu": INF,
}
MAX_EPOCHS = 4.0                  # audit threshold
MAX_GENERATOR_SHARE = 0.35        # cap per procedural generator
MIN_BALLAST = 0.30                # natural-text floor per stage
NATURAL = {"tinystories", "simple-wiki", "cosmo-young", "fineweb-good",
           "aochildes"}


def audit_mix(warn=True) -> dict:
    """Three constraints per stage, all learned by breaking them on
    2026-08-15: epoch cap (memorization), natural-text ballast
    (catastrophic forgetting — fineweb 1.14 -> 1.70 inside S3), and a
    per-generator share cap. The last is why S3's 54% rulechain-synth
    was dangerous even though it never repeats a row: her free
    generation started emitting the TEMPLATE — "Water boils when they
    are boiling. If someone is boiling then they are wet." A generator
    with few frames colonizes the model's prose at high share."""
    out = {}
    for stage, recipe in CURRICULUM_MIXES.items():
        ballast = sum(w for n, w in recipe if n in NATURAL)
        top_gen = max([w for n, w in recipe if n.endswith("-synth")],
                      default=0.0)
        out[stage] = {"ballast": ballast, "top_generator": top_gen}
        if warn:
            if ballast < MIN_BALLAST:
                print(f"[curriculum] WARNING {stage} natural-text ballast "
                      f"{ballast:.0%} < {MIN_BALLAST:.0%} — forgetting risk",
                      flush=True)
            if top_gen > MAX_GENERATOR_SHARE:
                print(f"[curriculum] WARNING {stage} single generator at "
                      f"{top_gen:.0%} > {MAX_GENERATOR_SHARE:.0%} — template "
                      "colonization risk", flush=True)
    return out


def audit_epochs(warn=True) -> list:
    """How many times each stage re-reads each finite corpus. Rows:
    (stage, component, weight, need_bytes, corpus_bytes, epochs)."""
    rows = []
    for stage, recipe in CURRICULUM_MIXES.items():
        budget = STAGE_TOKENS.get(stage, 0)
        for name, w in recipe:
            corpus = CORPUS_BYTES.get(name, INF)
            if corpus == INF or name.endswith("-synth"):
                continue
            need = budget * w
            rows.append((stage, name, w, need, corpus, need / corpus))
    if warn:
        for r in rows:
            if r[5] > MAX_EPOCHS:
                print(f"[curriculum] WARNING {r[0]} re-reads {r[1]} "
                      f"{r[5]:.0f}x (weight {r[2]}) — memorization risk",
                      flush=True)
    return rows


# ------------------------------------------------------------- stage mixes
# TWO constraints, both learned the hard way on 2026-08-15:
# (1) EPOCH CAP — no FINITE corpus re-read more than ~2x within its
#     stage (audit_epochs()); procedural generators carry the volume.
#     The first weights asked bAbI for 45 epochs and SIQA for 41.
# (2) NATURAL-LANGUAGE BALLAST >= ~25% in EVERY stage. S3 shipped with
#     6% and her fineweb holdout went 1.134 -> 2.059 bpb in ~1B tokens:
#     a narrow synthetic diet makes her forget how to read English
#     while she learns to follow rules. Replay is not optional in
#     continued pretraining; the S7 spiral is consolidation, not
#     rescue.
CURRICULUM_MIXES.update({
    "curriculum-s0": [("tinystories", 0.74), ("simple-wiki", 0.20),
                      ("aochildes", 0.01), ("recall-synth", 0.03),
                      ("counting-synth", 0.02)],
    "curriculum-s1": [("perspective-synth", 0.30), ("tinystories", 0.28),
                      ("simple-wiki", 0.20), ("cosmo-young", 0.12),
                      ("concept-synth", 0.04), ("recall-synth", 0.03),
                      ("siqa-narrative", 0.02), ("aochildes", 0.01)],
    "curriculum-s2": [("concept-synth", 0.35), ("cosmo-young", 0.35),
                      ("simple-wiki", 0.15), ("tinystories", 0.12),
                      ("recall-synth", 0.03)],
    "curriculum-s3": [("rulechain-synth", 0.32), ("proofwriter-prose", 0.22),
                      ("tinystories", 0.18), ("simple-wiki", 0.12),
                      ("cosmo-young", 0.08), ("concept-synth", 0.04),
                      ("recall-synth", 0.03), ("babi-prose", 0.01)],
    "curriculum-s4": [("arith-synth", 0.35), ("cosmo-young", 0.32),
                      ("simple-wiki", 0.12), ("tinystories", 0.08),
                      ("rulechain-synth", 0.06), ("counting-synth", 0.04),
                      ("recall-synth", 0.03)],
    "curriculum-s5": [("causal-synth", 0.32), ("atomic-flicker", 0.25),
                      ("tinystories", 0.20), ("simple-wiki", 0.10),
                      ("cosmo-young", 0.07), ("recall-synth", 0.05),
                      ("siqa-narrative", 0.01)],
    "curriculum-s6": [("tryfail-synth", 0.32), ("tinystories", 0.22),
                      ("arith-synth", 0.13), ("cosmo-young", 0.10),
                      ("simple-wiki", 0.10), ("causal-synth", 0.08),
                      ("recall-synth", 0.05)],
    "curriculum-s7": [("tinystories", 0.10), ("simple-wiki", 0.08),
                      ("cosmo-young", 0.10), ("rulechain-synth", 0.10),
                      ("arith-synth", 0.10), ("atomic-flicker", 0.08),
                      ("causal-synth", 0.08), ("tryfail-synth", 0.08),
                      ("proofwriter-prose", 0.08), ("fineweb-good", 0.10),
                      ("recall-synth", 0.05), ("perspective-synth", 0.04),
                      ("babi-prose", 0.01)],
    "curriculum-s8": [("register-synth", 0.32), ("simple-wiki", 0.25),
                      ("cosmo-young", 0.20), ("tinystories", 0.15),
                      ("recall-synth", 0.04), ("concept-synth", 0.03),
                      ("siqa-narrative", 0.01)],
})

# stage name -> planned tokens (bytes); plan-of-record budgets
STAGE_TOKENS = {
    "curriculum-s0": 700_000_000, "curriculum-s1": 700_000_000,
    "curriculum-s2": 1_000_000_000, "curriculum-s3": 1_200_000_000,
    "curriculum-s4": 1_200_000_000, "curriculum-s5": 1_000_000_000,
    "curriculum-s6": 800_000_000, "curriculum-s7": 1_400_000_000,
    "curriculum-s8": 800_000_000,
}


def append_curriculum_phases(manifest) -> int:
    """Idempotently append S0..S8 as manifest phases. Returns how many
    were added (0 on re-run). Status vocabulary is the manifest's:
    planned|active|done|deferred — current_phase() activates 'planned'
    phases; anything else is invisible to the scheduler (a 'pending'
    typo here once made the whole curriculum read as complete)."""
    audit_epochs(warn=True)
    audit_mix(warn=True)
    have = {p["name"] for p in manifest.phases}
    added = 0
    for ds, toks in STAGE_TOKENS.items():
        name = ds.replace("-", "_")
        if name in have:
            # repair pass: normalize a curriculum phase left unschedulable
            # by the pre-fix status string
            for p in manifest.phases:
                if p["name"] == name and p.get("status") == "pending":
                    p["status"] = "planned"
            continue
        manifest.phases.append(dict(name=name, dataset=ds,
                                    planned_tokens=toks, tokens_done=0,
                                    status="planned"))
        added += 1
    return added


def fold_head_gate(model) -> dict:
    """The verified head surgery: fold the fossilized gamma gate into W_s
    (semantic no-op, measured max|diff| 2.4e-07) so the aleph head is
    weight-zero-small WITH its inherited direction — the bank recipe.
    Safe to call twice: gamma==1.0 makes it the identity."""
    import torch
    g = float(model.head.gamma.item())
    with torch.no_grad():
        model.head.w_s.weight.mul_(model.head.gamma)
        model.head.gamma.fill_(1.0)
    return {"gamma_before": g,
            "w_s_norm": float(model.head.w_s.weight.norm().item())}


def rollback_to(run, step: int, first_stage: str, repo: str | None = None):
    """Rewind a live run to a stage boundary and re-plan from there.

    Prefers resume/boundary_<stage>_step<N>.pt (exact: weights +
    optimizer + stream state). Those archives only exist from geolip
    0.6.4 onward, so older boundaries fall back to safetensors weights
    with a FRESH optimizer — a real law-exception that must be flagged
    on anything trained afterwards, because Muon/Adam moment estimates
    are discarded.

    Every phase from first_stage onward is reset to planned/0 so the
    corrected mixes run it again; earlier phases keep their history.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    craft = run.manifest.preset
    repo = repo or getattr(run.hub, "repo_id", None)
    man = run.manifest
    exact = False
    for ph in man.phases:                      # find the archive by stage
        if ph["name"] == first_stage:
            break
    prev = None
    for ph in man.phases:
        if ph["name"] == first_stage:
            break
        prev = ph["name"]
    if prev:
        try:
            arch = hf_hub_download(
                repo, f"{craft}/resume/boundary_{prev}_step{step}.pt")
            payload = torch.load(arch, map_location=run.device,
                                 weights_only=False)
            run.raw_model.load_state_dict(payload["model"])
            for opt, st in zip(run.optimizers, payload["optimizers"]):
                opt.load_state_dict(st)
            exact = True
        except Exception as e:                 # noqa: BLE001
            print(f"[rollback] no exact archive for {prev}@{step} "
                  f"({type(e).__name__}) — weights-only rewind", flush=True)
    if not exact:
        sd = load_file(hf_hub_download(
            repo, f"{craft}/checkpoints/step_{step:08d}.safetensors"))
        run.raw_model.load_state_dict(sd)

    ck = next((c for c in man.checkpoints
               if c["step"] == step and c["kind"] == "safetensors"), None)
    man.steps = run.step = int(step)
    if ck and ck.get("tokens"):
        man.tokens_seen = int(ck["tokens"])
    hit = False
    for ph in man.phases:
        if ph["name"] == first_stage:
            hit = True
        if hit:
            ph["status"] = "planned"
            ph["tokens_done"] = 0
    run.stream = None                          # force a fresh stream open
    man.note(f"ROLLBACK to step {step:,}; re-planning from {first_stage} "
             f"({'exact resume archive' if exact else 'weights only — '
                 'FRESH OPTIMIZER (law exception)'})")
    print(f"[rollback] step {step:,} · {man.tokens_seen/1e9:.3f}B · "
          f"re-planning from {first_stage} · "
          f"{'EXACT (optimizer+stream restored)' if exact else 'FRESH OPTIMIZER — flag any arm trained after this'}",
          flush=True)
    return {"step": step, "exact": exact, "first_stage": first_stage}
