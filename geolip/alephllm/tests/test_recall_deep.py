"""The deep far-recall renderer (0.10.16; plans 2026-10-11): window-aligned recall episodes at distances to 3,600 bytes.
Run: python -m geolip.alephllm.tests.test_recall_deep   (CPU, seconds; no download)
Cases:
  1 the renderer: a short row renders empty; a long row renders to EXACTLY 4,095 UTF-8 bytes; deterministic; the plant and
    the question carry the same value (the updated value in the update form); the distance from the plant's start to the
    question's start lies in 64..3,700; the question ends inside the text; all four forms occur; at least a third of the
    distances pass 1,024 and a sixth pass 2,048
  2 the fit: a cut inside a multi-byte character backs off to the character and pads to the exact length
  3 the registry, the mix and the corpus-size entry
  4 alignment through the real PackedStream with the byte tokenizer: every block of a deep-recall stream ends in "\\n" + DOC
    and begins where a rendered document begins (eight micro-batches of two)
  5 the lookup-arm preset: arm_lr sets the arm's Adam rate (None keeps the recipe's 3e-4); the deep mix is accepted at the
    4,096 context and the phase names it
"""
from __future__ import annotations

import random
import re

import numpy as np

from ..data import streams as S
from ..data.curriculum import CORPUS_BYTES
from ..data.special_tokens import DOC
from ..data.streams import (CURRICULUM_MIXES, REGISTRY, RF_DEEP_TEXT_BYTES, PackedStream, _render_recall_far_deep,
                            _rf_fit)
from ..data.tokenizers import ByteTrigramTokenizer
from ..presets import make_v3_lookup_arm_preset

RESULTS: list = []
CUE = r"(?:code|word|key|number|token)"
PAT = re.compile(rf"(?:The secret {CUE} is (?:now )?([0-9a-z]+)\. |Locker \d{{4}} holds {CUE} ([0-9a-z]+)\. "
                 rf"|The secret {CUE} for locker \d{{4}} is ([0-9a-z]+)\. )")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _doc(rng, n_sent):
    words = ["the river", "a long road", "the old mill", "her garden", "the market", "a small town", "the harbor",
             "the library", "a quiet street", "the station", "the café", "a naïve plan"]
    verbs = ["was busy", "stood empty", "opened early", "closed at dusk", "filled with people", "waited for rain"]
    return " ".join(f"{rng.choice(words).capitalize()} {rng.choice(verbs)} on {rng.choice(['Monday', 'Tuesday', 'Friday'])}."
                    for _ in range(n_sent))


def _parse(text):
    """(form, plant start, question start, question end, plant value, question value) in UTF-8 byte offsets."""
    ms = list(PAT.finditer(text))
    if len(ms) < 2:
        return None
    first, last = ms[0], ms[-1]
    g = first.group(0)
    form = "lockers" if g.startswith("Locker") else ("lockers2" if "for locker" in g else
                                                     ("update" if (any(" is now " in m.group(0) for m in ms) or len(ms) >= 3) else "single"))
    bs = lambda i: len(text[:i].encode("utf-8"))  # noqa: E731
    vals = [next(v for v in m.groups() if v) for m in ms]
    planted = vals[1] if form == "update" else vals[0]        # the update's "now" value is the one asked
    if form in ("lockers", "lockers2"):
        planted = None                                         # four values; the question's must be among the plant's
        asked = vals[-1]
        ok = asked in vals[:-1]
        return form, bs(first.start()), bs(last.start()), bs(last.end()), ok, asked
    return form, bs(first.start()), bs(last.start()), bs(last.end()), planted == vals[-1], vals[-1]


def case_1():
    rng = random.Random(7)
    check("1 a short row renders empty", _render_recall_far_deep({"text": _doc(rng, 20)}) == "")
    forms, dists, n_ok, n_len, n_det, n_end = {}, [], 0, 0, 0, 0
    rows = 0
    for _ in range(600):
        text = _doc(rng, rng.randint(95, 160))
        out = _render_recall_far_deep({"text": text})
        if not out:
            continue
        rows += 1
        n_len += len(out.encode("utf-8")) == RF_DEEP_TEXT_BYTES
        n_det += _render_recall_far_deep({"text": text}) == out
        parsed = _parse(out)
        if parsed is None:
            continue
        form, p, q, q_end, same, _v = parsed
        forms[form] = forms.get(form, 0) + 1
        dists.append(q - p)
        n_ok += bool(same) and 64 <= q - p <= 3700
        n_end += q_end <= RF_DEEP_TEXT_BYTES
    d = np.array(dists)
    check("1 rows rendered", rows >= 300, f"{rows} of 600")
    check("1 every rendered row is exactly 4,095 UTF-8 bytes", n_len == rows, f"{n_len}/{rows}")
    check("1 deterministic per row", n_det == rows)
    check("1 every parsed episode asks the planted value at a distance in 64..3,700", n_ok == len(dists), f"{n_ok}/{len(dists)}")
    check("1 the question ends inside the text", n_end == len(dists))
    check("1 the four forms occur", all(f in forms for f in ("single", "lockers", "lockers2", "update")), str(forms))
    check("1 at least a third of the distances pass 1,024", (d > 1024).mean() >= 0.33, f"{(d > 1024).mean():.2f}")
    check("1 at least a sixth of the distances pass 2,048", (d > 2048).mean() >= 0.16, f"{(d > 2048).mean():.2f}")
    check("1 the median distance is past 512", float(np.median(d)) > 512, f"median {np.median(d):.0f}")


def case_2():
    doc = ("abc " * 10 + "é" * 50).encode("utf-8")      # the cut falls inside a 2-byte character for odd n
    for n in (41, 42, 43, 60):
        t = _rf_fit(doc, n)
        check(f"2 fit n={n}: exact length and valid text", len(t.encode("utf-8")) == n and t.encode("utf-8").decode("utf-8") == t
              and (t.endswith(" ") or len(doc[:n].decode("utf-8", errors="strict")) == len(t)))


def case_3():
    check("3 the registry entry", REGISTRY["recall-far-deep-fineweb"]["render"] == "recall_far_deep"
          and REGISTRY["recall-far-deep-fineweb"]["max_empties"] >= 10000 and "recall_far_deep" in S._RENDERERS)
    check("3 the mix", CURRICULUM_MIXES["fineweb-recall-deep-25"] == [("fineweb-edu", 0.75), ("recall-far-deep-fineweb", 0.25)])
    check("3 the corpus-size entry", CORPUS_BYTES["recall-far-deep-fineweb"] == float("inf"))


def case_4():
    rng = random.Random(11)
    rows = [{"text": _doc(rng, rng.randint(100, 150))} for _ in range(400)]
    tok = ByteTrigramTokenizer()
    ps = PackedStream("recall-far-deep-fineweb", tok, context=RF_DEEP_TEXT_BYTES + 1, micro_batch=2, seed=0)
    ps._it = iter(rows)
    ps._ds = object()
    rendered = [r for r in (_render_recall_far_deep(x) for x in rows) if r]
    heads = [r.encode("utf-8")[:24] for r in rendered]
    n_blocks, n_end, n_head = 0, 0, 0
    for _ in range(8):
        b = ps.next_batch().numpy()
        for row in b:
            n_blocks += 1
            n_end += int(row[-1]) == DOC and int(row[-2]) == ord("\n")
            n_head += bytes(int(v) for v in row[:24]) in heads
    check("4 every block ends in newline + DOC (one document per block)", n_end == n_blocks, f"{n_end}/{n_blocks}")
    check("4 every block begins where a rendered document begins", n_head == n_blocks, f"{n_head}/{n_blocks}")
    check("4 the block is context + 1 ids", b.shape[1] == RF_DEEP_TEXT_BYTES + 2)


def case_5():
    p = make_v3_lookup_arm_preset(mix="fineweb-recall-deep-25", arm_lr=1e-3, steps=1000, start_step=253674,
                                  source="mini-beatrix-3-la2-6-12-20-c249k", micro_batch=1, grad_accum=64)
    check("5 arm_lr sets the Adam rate", abs(p.train.adam_lr - 1e-3) < 1e-12)
    check("5 the deep mix is the arm phase's dataset", p.curriculum[1]["dataset"] == "fineweb-recall-deep-25"
          and p.curriculum[1]["planned_tokens"] == 1000 * 262144 and p.init_from["prefix"] == "mini-beatrix-3-la2-6-12-20-c249k")
    check("5 the recipe's step is kept", p.train.micro_batch * p.train.grad_accum * p.model.context == 262144)
    q = make_v3_lookup_arm_preset()
    check("5 arm_lr None keeps the recipe's 3e-4", abs(q.train.adam_lr - 3e-4) < 1e-12)


def main():
    for fn in (case_1, case_2, case_3, case_4, case_5):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised", False, repr(e))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
