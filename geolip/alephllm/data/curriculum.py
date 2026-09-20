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


def _render_definition(row) -> str:
    """WordNet-class gloss -> definitional prose in several surface forms,
    so she learns the ANSWER SHAPE for "what does X mean?" rather than one
    template. Synonym/antonym/hypernym fields become the follow-on
    sentences a dictionary entry implies."""
    w = str(row.get("word", "")).strip()
    d = str(row.get("definition", "")).strip().rstrip(".")
    if not (w and d) or len(w) > 40:
        return ""
    pos = str(row.get("part_of_speech", "")).strip()

    def _lst(k, n=2):
        v = row.get(k) or []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v[:n] if str(x).strip()]

    syn, ant, hyp = _lst("synonyms"), _lst("antonyms", 1), _lst("hypernyms", 1)
    ex = _lst("examples", 1)
    art = "An" if w[:1].lower() in "aeiou" else "A"
    h = abs(hash(w)) % 3
    if h == 0:
        s = f"The word '{w}'"
        if pos:
            s += f", used as a {pos},"
        s += f" means {d}."
    elif h == 1:
        s = f"{w.capitalize()}: {d}."
    else:
        s = f"What does '{w}' mean? It means {d}."
    if ex:
        s += " For example: " + ex[0].rstrip(".") + "."
    if syn:
        s += (" Words close in meaning are " + " and ".join(syn) + "."
              if len(syn) > 1 else " A close word is " + syn[0] + ".")
    if ant:
        s += " The opposite is " + ant[0] + "."
    if hyp:
        s += f" {art} {w} is a kind of {hyp[0]}."
    return s


_PG_START = "*** START OF"
_PG_END = "*** END OF"


def _render_gutenberg(row) -> str:
    """Public-domain book text with Project Gutenberg boilerplate cut.
    Markers vary (THIS/THE), so cut on the first line CONTAINING the
    marker, then drop table/index/page-number lines."""
    txt = str(row.get("TEXT") or row.get("text") or "")
    if not txt:
        return ""
    i = txt.upper().find(_PG_START)
    if i >= 0:
        nl = txt.find("\n", i)
        txt = txt[nl + 1:] if nl > 0 else txt[i:]
    j = txt.upper().find(_PG_END)
    if j >= 0:
        txt = txt[:j]
    keep = []
    for ln in txt.replace("\r\n", "\n").split("\n"):
        s = ln.strip()
        if not s:
            keep.append("")
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in s)
        if alpha / max(len(s), 1) < 0.6 or s.isdigit():
            continue
        keep.append(s)
    out = "\n".join(keep).strip()
    return out if len(out) > 400 else ""


def _render_wikipedia(row) -> str:
    """Full-English Wikipedia article, trailing apparatus removed."""
    txt = str(row.get("text", ""))
    for cut in ("\nSee also", "\nReferences", "\nExternal links",
                "\nFurther reading", "\nNotes"):
        k = txt.find(cut)
        if k > 0:
            txt = txt[:k]
    txt = txt.strip()
    return txt if len(txt) > 800 else ""


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


_RULECHAIN_NAMES = ["Pia", "Ezo", "Kel", "Vin", "Osa", "Tam"]


def _rulechain_rows(seed: int, max_depth: int | None = None, *,
                    vocab: list | None = None, weights: list | None = None,
                    pairs: list | None = None, pair_rate: float = 0.0):
    """Prose modus-ponens chains over nonsense predicates (no world
    knowledge shortcut). Training stream stays BELOW the held-out depth.

    The default draws the legacy eight predicates uniformly and is
    bit-identical to earlier releases (a stream's rows never change
    under a resume). `vocab` swaps the predicate lexicon — the graded,
    atlas-minted lexicon is the difficulty dial of the curriculum;
    `weights` skews the per-word draw (band mixes); `pairs` (word pairs
    sharing an onset) with `pair_rate` puts BOTH members of one or two
    pairs into that share of rows, so commitment between look-alike
    surfaces is trained at source. Every row's predicates are distinct
    and the prose frame is the same in every mode."""
    rng = np.random.default_rng(seed)
    cap = (max_depth if max_depth is not None
           else HOLDOUTS["rulechain_depth"] - 1)
    names = _RULECHAIN_NAMES
    legacy = vocab is None and not pairs
    words = list(_PRED) if vocab is None else [str(w) for w in vocab]
    assert len(set(words)) == len(words) >= 2, "predicates must be distinct"
    p = None
    if weights is not None:
        p = np.asarray(weights, dtype=np.float64)
        assert p.shape == (len(words),) and (p >= 0).all() and p.sum() > 0, "weights: one per word, >= 0"
        p = p / p.sum()
    pairs = [[str(a), str(b)] for a, b in (pairs or [])]
    idx = {w: i for i, w in enumerate(words)}
    for a, b in pairs:
        assert a in idx and b in idx and a != b, f"pair {(a, b)} must be two distinct vocabulary words"
    while True:
        d = int(rng.integers(1, cap + 1))
        k = d + 1
        if legacy:
            ps = [(_PRED[int(i)]) for i in
                  rng.choice(len(_PRED), size=k, replace=False)]
        elif pairs and pair_rate > 0 and rng.random() < pair_rate:
            n_pairs = min(2, k // 2, len(pairs))
            chosen = rng.choice(len(pairs), size=n_pairs, replace=False)
            ps = [w for i in chosen for w in pairs[int(i)]]
            taken = set(ps)
            fill = [i for i, w in enumerate(words) if w not in taken]
            need = k - len(ps)
            if need > 0:
                fp = None if p is None else p[fill] / p[fill].sum()
                ps += [words[int(i)] for i in rng.choice(fill, size=need, replace=False, p=fp)]
            rng.shuffle(ps)
        else:
            ps = [words[int(i)] for i in rng.choice(len(words), size=k, replace=False, p=p)]
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
    # WordNet-class glosses: the definitional ANSWER SHAPE (Phil ask,
    # 2026-08-15) — "teach the model how to respond to definitions".
    "definitions": dict(path="mjbommar/opengloss-v1.3-definitions",
                        split="train", column="definition",
                        render="definition",
                        columns=["word", "part_of_speech", "definition",
                                 "synonyms", "antonyms", "hypernyms",
                                 "examples"],
                        max_empties=20_000),
    # large permissive ballast — the structural answer to the S3 crash
    "gutenberg": dict(path="sedthh/gutenberg_english", split="train",
                      column="TEXT", render="gutenberg",
                      columns=["TEXT"], max_empties=20_000),
    "wikipedia-en": dict(path="wikimedia/wikipedia", name="20231101.en",
                         split="train", column="text", render="wikipedia",
                         columns=["text"], max_empties=20_000),
    "counting-synth": dict(path=None, generator="counting"),
    "perspective-synth": dict(path=None, generator="perspective"),
    "concept-synth": dict(path=None, generator="concept"),
    "rulechain-synth": dict(path=None, generator="rulechain"),
    # the same prose frame over the atlas-minted lexicon (set_minted_lexicon)
    "rulechain-minted": dict(path=None, generator="rulechain_minted"),
    "arith-synth": dict(path=None, generator="arith"),
    "causal-synth": dict(path=None, generator="causal"),
    "tryfail-synth": dict(path=None, generator="tryfail"),
    "register-synth": dict(path=None, generator="register"),
})

_RENDERERS.update({
    "definition": _render_definition,
    "gutenberg": _render_gutenberg,
    "wikipedia": _render_wikipedia,
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
    "rulechain_minted": lambda seed, max_depth=None: _rulechain_minted_rows(seed, max_depth),
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
    "definitions": 111e6,         # 565,604 glosses x ~197 B rendered
    "wikitext-103": 540e6,        # the warmup corpus (~1.8M lines of prose)
    "cosmo-young": INF, "fineweb-good": INF, "fineweb-edu": INF,
    "gutenberg": INF,             # ~18GB raw, ~13GB after the cut
    "wikipedia-en": INF,          # ~19GB of article prose
}
MAX_EPOCHS = 4.0                  # audit threshold
MAX_GENERATOR_SHARE = 0.35        # cap per procedural generator


def _is_gen(name: str) -> bool:
    """A procedural generator: the -synth suffix by convention, or a member
    of a shared prose frame (_FRAMES) whatever its name."""
    return name.endswith("-synth") or name in _FRAMES

MIN_BALLAST = 0.30                # natural-text floor per stage
NATURAL = {"tinystories", "simple-wiki", "cosmo-young", "fineweb-good",
           "aochildes", "gutenberg", "wikipedia-en"}


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
        # sources sharing ONE prose frame (_FRAMES) count as one generator:
        # the colonization risk is the frame's share, not the name's
        frames: dict = {}
        for n, w in recipe:
            if _is_gen(n):
                key = _FRAMES.get(n, n)
                frames[key] = frames.get(key, 0.0) + w
        top_gen = max(frames.values(), default=0.0)
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
            if corpus == INF or _is_gen(name):
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
    "curriculum-s2": [("concept-synth", 0.30), ("cosmo-young", 0.30),
                      ("simple-wiki", 0.12), ("tinystories", 0.10),
                      ("definitions", 0.08), ("wikipedia-en", 0.07),
                      ("recall-synth", 0.03)],
    "curriculum-s3": [("rulechain-synth", 0.32), ("proofwriter-prose", 0.22),
                      ("tinystories", 0.10), ("wikipedia-en", 0.08),
                      ("simple-wiki", 0.12), ("cosmo-young", 0.08),
                      ("concept-synth", 0.04), ("recall-synth", 0.03),
                      ("babi-prose", 0.01)],
    "curriculum-s4": [("arith-synth", 0.35), ("cosmo-young", 0.32),
                      ("simple-wiki", 0.12), ("tinystories", 0.08),
                      ("rulechain-synth", 0.06), ("counting-synth", 0.04),
                      ("recall-synth", 0.03)],
    "curriculum-s5": [("causal-synth", 0.32), ("atomic-flicker", 0.25),
                      ("tinystories", 0.12), ("gutenberg", 0.08),
                      ("simple-wiki", 0.10), ("cosmo-young", 0.07),
                      ("recall-synth", 0.05), ("siqa-narrative", 0.01)],
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
    "curriculum-s8": [("definitions", 0.25), ("register-synth", 0.25),
                      ("simple-wiki", 0.15), ("cosmo-young", 0.12),
                      ("wikipedia-en", 0.10), ("tinystories", 0.09),
                      ("recall-synth", 0.03), ("siqa-narrative", 0.01)],
})

# stage name -> planned tokens (bytes); plan-of-record budgets (the 1x
# schedule of the 2s mission, 8.8B in total)
STAGE_TOKENS = {
    "curriculum-s0": 700_000_000, "curriculum-s1": 700_000_000,
    "curriculum-s2": 1_000_000_000, "curriculum-s3": 1_200_000_000,
    "curriculum-s4": 1_200_000_000, "curriculum-s5": 1_000_000_000,
    "curriculum-s6": 800_000_000, "curriculum-s7": 1_400_000_000,
    "curriculum-s8": 800_000_000,
}

# ------------------------------------------------------- scaling (v3)
# The 1x forms are kept verbatim so a scale can be applied, re-applied
# or undone without drift; apply_curriculum_scale() rewrites the live
# STAGE_TOKENS / CURRICULUM_MIXES from these.
#
# TWO numbers here are the program lead's, not the library's: the epoch
# cap (the law text above says ~2x per finite corpus; the audit warns at
# MAX_EPOCHS = 4) and WHERE the freed weight goes (the record's
# precedent when a finite corpus hit its cap: procedural generators up
# to their share cap, then natural text). A scale other than 1x
# therefore REFUSES to apply until both are supplied.
_BASE_STAGE_TOKENS = dict(STAGE_TOKENS)
_BASE_MIXES = {k: [tuple(x) for x in v] for k, v in CURRICULUM_MIXES.items()
               if k in STAGE_TOKENS}
_APPLIED_SCALE = {"factor": 1.0, "epoch_cap": None, "rebalance_to": None}
REBALANCE_RULES = ("natural", "generators", "hold")


def scaled_curriculum(factor: float, epoch_cap: float | None = None,
                      rebalance_to: str | None = None,
                      fallback: str = "fineweb-good") -> dict:
    """The stage budgets at `factor` x the 1x schedule and the mixes
    REBALANCED so no finite corpus is re-read past `epoch_cap` within its
    stage (the epoch-cap law: every finite corpus's epochs-within-stage is
    computed BEFORE a mix ships). Pure: nothing is mutated.

    epoch_cap: None reads as the audit threshold (MAX_EPOCHS) and is
    flagged — the law text says ~2; the number is the lead's.
    rebalance_to (where the freed weight goes):
      'natural'    pro rata to the mix's NATURAL members with epoch
                   headroom (infinite ones unbounded), never a generator;
                   `fallback` is added if no natural member can absorb.
      'generators' the record's precedent: the mix's procedural
                   generators first, up to MAX_GENERATOR_SHARE each, then
                   the natural rule for the remainder.
      'hold'       the stages stay at 1x bytes (no component moves); the
                   extra (factor-1) x 8.8B is returned as `held_tokens`
                   for the preset to spend on general text / the anneals.
      None         refuses (ValueError) whenever a component would have to
                   move — the tables are printed so the lead can rule.

    Returns {"factor", "epoch_cap", "rebalance_to", "stage_tokens",
    "mixes", "table", "cross_stage", "flags", "held_tokens"}; table rows
    are (stage, component, weight_before, weight_after, epochs_before,
    epochs_after) for every finite non-generator component; cross_stage
    maps each finite corpus to its total epochs over ALL stages after the
    rebalance (a corpus can sit under its per-stage cap in every stage
    and still be re-read many times over the curriculum).
    """
    assert factor > 0
    flags = {}
    cap = float(MAX_EPOCHS if epoch_cap is None else epoch_cap)
    if epoch_cap is None and factor != 1.0:
        flags["epoch_cap"] = (f"epoch cap not supplied: the audit threshold "
                              f"{MAX_EPOCHS:g} was used (the law text says ~2)")
    if rebalance_to is not None and rebalance_to not in REBALANCE_RULES:
        raise ValueError(f"rebalance_to must be one of {REBALANCE_RULES} or None")
    hold = rebalance_to == "hold"
    eff = 1.0 if hold else factor
    stage_tokens = {k: int(round(v * eff)) for k, v in _BASE_STAGE_TOKENS.items()}
    held = int(round((factor - eff) * sum(_BASE_STAGE_TOKENS.values())))
    mixes, table = {}, []
    cross = {}
    for stage, recipe in _BASE_MIXES.items():
        budget = stage_tokens[stage]
        w = {n: float(x) for n, x in recipe}
        before = dict(w)

        def cap_of(n, budget=budget):
            if _is_gen(n):
                # a generator's cap is its share; sources sharing one prose
                # frame are capped by the FRAME's sum (the live weights)
                fr = _FRAMES.get(n)
                others = (sum(x for m, x in w.items() if m != n and _FRAMES.get(m) == fr)
                          if fr else 0.0)
                return max(0.0, MAX_GENERATOR_SHARE - others)
            corpus = CORPUS_BYTES.get(n, INF)
            if corpus == INF:
                return INF
            return cap * corpus / budget

        freed = 0.0
        for n in list(w):
            c = cap_of(n)
            if not _is_gen(n) and w[n] > c:
                freed += w[n] - c
                w[n] = c
        if freed > 1e-9 and rebalance_to is None:
            over = [(n, round(before[n], 4), round(cap_of(n), 4)) for n in before
                    if not _is_gen(n) and before[n] > cap_of(n) + 1e-12]
            raise ValueError(
                f"curriculum scale x{factor:g}: {stage} re-reads finite corpora "
                f"past {cap:g} epochs {over} and no rebalance rule was supplied "
                f"(rebalance_to in {REBALANCE_RULES}) — the lead's decision; run "
                "scaled_curriculum(factor, epoch_cap, rule) for each rule to see "
                "the tables")
        guard = 0
        while freed > 1e-9 and guard < 50:
            guard += 1
            recips = []
            if rebalance_to == "generators":
                recips = [n for n in w if _is_gen(n)
                          and cap_of(n) - w[n] > 1e-12]
            if not recips:
                recips = [n for n in w if n in NATURAL and cap_of(n) - w[n] > 1e-12]
            if not recips:
                if fallback not in w:
                    w[fallback] = 0.0
                    flags[stage] = f"no member could absorb {freed:.3f}: added {fallback}"
                recips = [fallback]
            # recipients sharing one prose frame receive as ONE unit (the
            # frame's headroom under the cap), split within the frame pro
            # rata — so a lexicon split keeps its ratio under any scale
            groups: dict = {}
            for n in recips:
                groups.setdefault(_FRAMES.get(n, n), []).append(n)
            gw = {g: sum(w[n] for n in ms) for g, ms in groups.items()}
            total = sum(gw.values())
            moved = 0.0
            for g, ms in groups.items():
                share = freed * (gw[g] / total if total > 0 else 1.0 / len(groups))
                room = (max(0.0, MAX_GENERATOR_SHARE - gw[g]) if g in _FRAMES.values() and len(ms) > 1
                        else cap_of(ms[0]) - w[ms[0]])
                give = min(share, room)
                for n in ms:
                    w[n] += give * (w[n] / gw[g] if gw[g] > 0 else 1.0 / len(ms))
                moved += give
            freed -= moved
            if moved <= 1e-12:
                break
        s = sum(w.values())
        w = {n: x / s for n, x in w.items()}
        mixes[stage] = [(n, round(x, 5)) for n, x in w.items() if x > 0]
        for n in before:
            corpus = CORPUS_BYTES.get(n, INF)
            if corpus == INF or _is_gen(n):
                continue
            table.append((stage, n, round(before[n], 4), round(w[n], 4),
                          round(budget * before[n] / corpus, 2),
                          round(budget * w[n] / corpus, 2)))
            cross[n] = cross.get(n, 0.0) + budget * w[n] / corpus
    cross = {n: round(v, 2) for n, v in cross.items()}
    return {"factor": factor, "epoch_cap": cap, "rebalance_to": rebalance_to,
            "stage_tokens": stage_tokens, "mixes": mixes, "table": table,
            "cross_stage": cross, "flags": flags, "held_tokens": held}


def data_plane(factor: float = 1.0, epoch_cap: float | None = None,
               rebalance_to: str | None = None) -> dict:
    """The data-plane record a run is created under: the three decisions
    + a fingerprint of the resulting stage recipes, so a resume can assert
    it continues on the SAME mix (the recipe-fingerprint law)."""
    import hashlib
    import json
    sc = scaled_curriculum(factor, epoch_cap, rebalance_to) if (
        factor != 1.0 or rebalance_to is not None) else {
        "stage_tokens": dict(_BASE_STAGE_TOKENS), "mixes": _BASE_MIXES,
        "epoch_cap": float(MAX_EPOCHS if epoch_cap is None else epoch_cap)}
    blob = json.dumps({"tokens": sc["stage_tokens"],
                       "mixes": {k: [list(x) for x in v] for k, v in sc["mixes"].items()}},
                      sort_keys=True)
    return {"data_scale": float(factor), "epoch_cap": sc["epoch_cap"],
            "rebalance_to": rebalance_to,
            "recipe_hash": hashlib.sha1(blob.encode()).hexdigest()[:12],
            # the minted lexicon's identity rides with the plane (None = unset)
            "minted_lexicon": MINTED["sha"]}


def apply_curriculum_scale(factor: float, epoch_cap: float | None = None,
                           rebalance_to: str | None = None,
                           verbose: bool = True) -> dict:
    """Rewrite the live STAGE_TOKENS + CURRICULUM_MIXES for `factor`
    (idempotent; factor 1.0 restores the 1x forms bit-exact). The streams
    read CURRICULUM_MIXES at build time, so this must run BEFORE
    append_curriculum_phases / prepare() opens a stage. Refuses (via
    scaled_curriculum) when a scale would move a component and no
    rebalance rule was supplied."""
    sc = scaled_curriculum(factor, epoch_cap, rebalance_to)
    STAGE_TOKENS.clear()
    STAGE_TOKENS.update(sc["stage_tokens"])
    for k, v in sc["mixes"].items():
        CURRICULUM_MIXES[k] = list(v)
    _APPLIED_SCALE.update(factor=factor, epoch_cap=epoch_cap,
                          rebalance_to=rebalance_to)
    if verbose:
        moved = [r for r in sc["table"] if r[2] != r[3]]
        print(f"[curriculum] scale x{factor:g} ({rebalance_to or 'no rule'}, cap "
              f"{sc['epoch_cap']:g} ep): {sum(sc['stage_tokens'].values())/1e9:.2f}B "
              f"over {len(sc['stage_tokens'])} stages; {len(moved)} finite components "
              f"moved" + (f"; held {sc['held_tokens']/1e9:.1f}B" if sc["held_tokens"] else "")
              + (f"; FLAGS {sc['flags']}" if sc["flags"] else ""), flush=True)
        for st, n, wb, wa, eb, ea in moved:
            print(f"[curriculum]   {st} {n:<18} w {wb:.3f} -> {wa:.3f}  epochs {eb:.1f} -> {ea:.1f}",
                  flush=True)
        hot = {n: e for n, e in sc["cross_stage"].items() if e > sc["epoch_cap"]}
        if hot:
            print(f"[curriculum]   cross-stage epochs above the cap: {hot}", flush=True)
    return sc


def curriculum_phases(factor: float = 1.0, rebalance_to: str | None = None) -> list:
    """The S0..S8 manifest phase dicts at `factor` (planned; chronological
    order) — for presets that carry the curriculum from birth. Budgets
    only (no mix is touched here): 'hold' keeps the stages at 1x."""
    eff = 1.0 if rebalance_to == "hold" else float(factor)
    return [dict(name=ds.replace("-", "_"), dataset=ds,
                 planned_tokens=int(round(toks * eff)), status="planned")
            for ds, toks in _BASE_STAGE_TOKENS.items()]


def append_curriculum_phases(manifest, scale: float | None = None,
                             epoch_cap: float | None = None,
                             rebalance_to: str | None = None) -> int:
    """Idempotently append S0..S8 as manifest phases. Returns how many
    were added (0 on re-run). Status vocabulary is the manifest's:
    planned|active|done|deferred — current_phase() activates 'planned'
    phases; anything else is invisible to the scheduler (a 'pending'
    typo here once made the whole curriculum read as complete).
    `scale` (v3): apply the data scale (rebalanced mixes) first; phases
    that are still 'planned' take the scaled budget."""
    if scale is not None and (scale, epoch_cap, rebalance_to) != (
            _APPLIED_SCALE["factor"], _APPLIED_SCALE["epoch_cap"],
            _APPLIED_SCALE["rebalance_to"]):
        apply_curriculum_scale(scale, epoch_cap, rebalance_to)
    audit_epochs(warn=True)
    audit_mix(warn=True)
    have = {p["name"] for p in manifest.phases}
    added = 0
    for ds, toks in STAGE_TOKENS.items():
        name = ds.replace("-", "_")
        if name in have:
            for p in manifest.phases:
                if p["name"] != name:
                    continue
                # repair pass: normalize a curriculum phase left
                # unschedulable by the pre-fix status string
                if p.get("status") == "pending":
                    p["status"] = "planned"
                # a still-planned phase follows the applied scale
                if p.get("status") == "planned" and p["planned_tokens"] != toks:
                    p["planned_tokens"] = toks
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


# ------------------------------------------------- the minted lexicon (v3)
# The curriculum's difficulty dial: a graded lexicon minted by the trigram
# atlas (geolip-bytelex) — an easy-close band, a trap band and pairs of
# words sharing an onset — swapped into the rules generator's UNCHANGED
# prose frame as a second registry source beside the legacy eight
# predicates. Nothing here is active until set_minted_lexicon() runs; the
# source refuses to open without a lexicon (no silent fallback). The
# lexicon's identity rides with the data plane (data_plane()), so a resume
# on a different lexicon is caught like any other mix change.
MINTED: dict = {"path": None, "lexicon": None, "pair_rate": 0.0, "splits": {}, "sha": None}
_FRAMES = {"rulechain-synth": "rulechain", "rulechain-minted": "rulechain"}   # sources sharing one prose frame
_PRISTINE_MIXES = {k: [tuple(x) for x in v] for k, v in _BASE_MIXES.items()}


def minted_vocab(lex: dict) -> tuple:
    """(words, weights, pairs) from a minted-lexicon artifact:
    {"easy": [...], "trap": [...], "pairs": [[a, b], ...],
     "band_weights": {"easy": w, "trap": w, "pairs": w}}  (weights default 1:
    a uniform draw over the union, so equal band sizes give a 1:1 mix).
    Pair words are vocabulary members in their own right; `pairs` adds the
    forced co-occurrence."""
    bw = dict(lex.get("band_weights") or {})
    words, weights = [], []
    for band in ("easy", "trap"):
        for w in lex.get(band) or []:
            words.append(str(w)); weights.append(float(bw.get(band, 1.0)))
    pairs = [[str(a), str(b)] for a, b in (lex.get("pairs") or [])]
    for a, b in pairs:
        for w in (a, b):
            if w not in words:
                words.append(w); weights.append(float(bw.get("pairs", 1.0)))
    assert len(set(words)) == len(words), "minted lexicon: duplicate words"
    assert not (set(words) & set(_PRED)), "minted lexicon: a legacy predicate leaked in"
    for w in words:
        assert w.isalpha() and w.islower(), f"minted lexicon: {w!r} is not a lowercase word"
    return words, weights, pairs


def _rulechain_minted_rows(seed: int, max_depth: int | None = None):
    lex = MINTED["lexicon"]
    if lex is None:
        raise RuntimeError("rulechain-minted: no minted lexicon installed — call "
                           "set_minted_lexicon(path, splits) before the stage opens "
                           "(this source never falls back to the legacy predicates)")
    words, weights, pairs = minted_vocab(lex)
    return _rulechain_rows(seed, max_depth, vocab=words, weights=weights,
                           pairs=pairs, pair_rate=float(MINTED["pair_rate"]))


def _reapply_mixes(stages) -> None:
    """Re-derive the live mixes for `stages` from the (amended) 1x forms
    under whatever scale is applied — the streams read CURRICULUM_MIXES
    at build time, so this runs before the amended stage opens."""
    a = _APPLIED_SCALE
    if a["factor"] != 1.0 or a["rebalance_to"] is not None:
        apply_curriculum_scale(a["factor"], a["epoch_cap"], a["rebalance_to"], verbose=False)
    else:
        for st in stages:
            CURRICULUM_MIXES[st] = list(_BASE_MIXES[st])


def set_minted_lexicon(path, splits: dict | None = None, pair_rate: float = 0.0,
                       verbose: bool = True) -> dict:
    """Install the minted lexicon (a JSON file path or the loaded dict) and
    the per-stage share splits, e.g.
        {"curriculum-s3": {"rulechain-synth": 0.08, "rulechain-minted": 0.24}}
    A split re-divides the stage's frame members' 1x weights: the split's
    sum must equal the weight it replaces (the dial re-divides the frame's
    share; moving the frame's total is a separate mix decision), and the
    frame's sum stays under MAX_GENERATOR_SHARE. The live mixes are then
    re-derived under the applied scale. `pair_rate` is the share of minted
    rows that carry both members of one or two onset pairs. Returns the
    record (path, sha, counts, splits)."""
    import hashlib
    import json
    if isinstance(path, dict):
        lex, src = path, "<dict>"
    else:
        with open(path, encoding="utf-8") as f:
            lex = json.load(f)
        src = str(path)
    words, weights, pairs = minted_vocab(lex)
    sha = hashlib.sha1(json.dumps({"w": sorted(words), "p": sorted(map(sorted, pairs)),
                                   "bw": lex.get("band_weights") or {}}, sort_keys=True).encode()).hexdigest()[:12]
    splits = {st: {n: float(x) for n, x in sp.items()} for st, sp in (splits or {}).items()}
    for st, sp in splits.items():
        assert st in _BASE_MIXES, f"unknown stage {st!r}"
        assert all(n in _FRAMES for n in sp), f"a split names only frame members {sorted(_FRAMES)}: {sp}"
        base = list(_PRISTINE_MIXES[st])
        replaced = sum(w for n, w in base if n in _FRAMES)
        assert abs(sum(sp.values()) - replaced) < 1e-6, (
            f"{st}: the split sums to {sum(sp.values()):.4f} but the frame's 1x weight is {replaced:.4f} — "
            "a split re-divides the frame's share; changing the total is a mix decision")
        assert sum(sp.values()) <= MAX_GENERATOR_SHARE + 1e-9, f"{st}: the frame would exceed the share cap"
        new = [(n, w) for n, w in base if n not in _FRAMES] + [(n, w) for n, w in sp.items() if w > 0]
        _BASE_MIXES[st] = [tuple(x) for x in new]
    for st in _PRISTINE_MIXES:
        if st not in splits:
            _BASE_MIXES[st] = list(_PRISTINE_MIXES[st])
    MINTED.update(path=src, lexicon=lex, pair_rate=float(pair_rate), splits=splits, sha=sha)
    _reapply_mixes(list(_BASE_MIXES))
    rec = {"path": src, "sha": sha, "words": len(words), "easy": len(lex.get("easy") or []),
           "trap": len(lex.get("trap") or []), "pairs": len(pairs), "pair_rate": float(pair_rate), "splits": splits,
           "live": {st: [(n, w) for n, w in CURRICULUM_MIXES[st] if n in _FRAMES] for st in splits}}
    if verbose:
        print(f"[curriculum] minted lexicon {sha}: {rec['words']} words ({rec['easy']} easy / {rec['trap']} trap / "
              f"{rec['pairs']} pairs, pair rows {pair_rate:.0%}); live frame shares {rec['live']}", flush=True)
    return rec


def clear_minted_lexicon() -> None:
    """Remove the lexicon and restore every stage's 1x recipe (tests)."""
    for st in _PRISTINE_MIXES:
        _BASE_MIXES[st] = list(_PRISTINE_MIXES[st])
    MINTED.update(path=None, lexicon=None, pair_rate=0.0, splits={}, sha=None)
    _reapply_mixes(list(_BASE_MIXES))
