"""The byte levers (0.10.14; plans 2026-10-10): the loss forms and the far-recall data piece.
Run: python -m geolip.alephllm.tests.test_byte_levers   (CPU, seconds; no download)
Cases:
  1 the far-recall renderer: deterministic per row; the plant and the question carry the same value at a gap in range;
    short rows and rows without sentence starts render empty; the three forms occur; the update form asks the later value;
    the dataset, the mix and the corpus-size entry are registered
  2 the loss forms on a tiny model: "ce" is the cross-entropy; "depth" with eight ones equals it; "copy" with beta 0 equals
    it; the depth NLLs sum to the byte's NLL; the copy mask marks exactly the repeat's bytes past the m-gram at distance >= D
    and nothing at a nearer repeat; the weighted forms differ from the CE when their weights bite; eval mode is the CE
  3 the resume factory's extra_train and phase_dataset_override (set, refused when unknown)
"""
from __future__ import annotations

import math
import random
import string

import torch
import torch.nn.functional as F

from ..data import streams as S
from ..data.curriculum import CORPUS_BYTES
from ..data.streams import CURRICULUM_MIXES, REGISTRY, _render_recall_far
from ..model.alephlm import AlephLM, copy_mask, depth_nll
from ..presets import AlephLMConfig, TrainConfig, make_control_resume_preset

RESULTS: list = []
TINY = AlephLMConfig(name="tiny-levers", d_model=64, n_layers=2, n_heads=4, context=128, hub_layers=(),
                     hub_K=32, hub_D=8, head_K=32, head_D=8, hub_chunk=16)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _doc(rng, n_sent):
    words = ["the river", "a long road", "the old mill", "her garden", "the market", "a small town", "the harbor",
             "the library", "a quiet street", "the station"]
    verbs = ["was busy", "stood empty", "opened early", "closed at dusk", "filled with people", "waited for rain"]
    return " ".join(f"{rng.choice(words).capitalize()} {rng.choice(verbs)} on {rng.choice(['Monday', 'Tuesday', 'Friday'])}."
                    for _ in range(n_sent))


def case_1():
    rng = random.Random(1)
    check("1 the dataset, the mix and the corpus entry are registered",
          REGISTRY["recall-far-fineweb"]["render"] == "recall_far" and "recall_far" in S._RENDERERS
          and CURRICULUM_MIXES["fineweb-recall-far-5"] == [("fineweb-edu", 0.95), ("recall-far-fineweb", 0.05)]
          and CORPUS_BYTES["recall-far-fineweb"] == float("inf"))
    check("1 a short row renders empty", _render_recall_far({"text": "Too short. Really."}) == "")
    check("1 a row without a sentence start near byte 200 renders empty",
          _render_recall_far({"text": "x" * 3000}) == "")
    forms, gaps, n_ok = {"single": 0, "lockers": 0, "update": 0}, [], 0
    for i in range(120):
        text = _doc(rng, rng.randint(12, 90))
        out = _render_recall_far({"text": text})
        again = _render_recall_far({"text": text})
        if out == "":
            continue
        assert out == again, "the renderer is not deterministic"
        b = out.encode()
        if b" is now " in b:
            forms["update"] += 1
            v2 = b.split(b" is now ")[1].split(b". ")[0]
            q = b.rfind(b"The secret ")
            n_ok += int(b[q:].split(b" is ")[1].startswith(v2))
            plant, question = b.find(b"The secret "), q
        elif b"Locker " in b:
            forms["lockers"] += 1
            q = b.rfind(b"Locker ")
            line = b[q:].split(b". ")[0]
            plant = b.find(line)
            n_ok += int(plant < q)
            question = q
        else:
            forms["single"] += 1
            plant, question = b.find(b"The secret "), b.rfind(b"The secret ")
            n_ok += int(b[plant:].split(b". ")[0] == b[question:].split(b". ")[0] and plant < question)
        gaps.append(question - plant)
    n = sum(forms.values())
    check("1 the three forms occur over the rendered rows", n >= 60 and all(v > 0 for v in forms.values()), str(forms))
    check("1 every rendered row's question matches its record (the update form: the later value)", n_ok == n, f"{n_ok}/{n}")
    check("1 the gaps sit in range", gaps and min(gaps) >= 64 and max(gaps) <= 2400 + 400, f"{min(gaps)}..{max(gaps)}")


def case_2():
    torch.manual_seed(2)
    m = AlephLM(AlephLMConfig.from_dict({**TINY.to_dict(), "name": "tiny-levers-512", "context": 512}))
    m.train()
    x = torch.randint(0, 255, (2, 64))
    y = torch.randint(0, 255, (2, 64))
    y[0, :5] = -100
    ce = m(x, targets=y).loss
    m.loss_form = "depth"
    m.loss_depth_w = (1.0,) * 8
    d1 = m(x, targets=y).loss
    m.loss_form = "copy"
    m.loss_copy_beta = 0.0
    c0 = m(x, targets=y).loss
    check("2 depth with eight ones and copy with beta 0 are the CE exactly",
          abs(float(d1 - ce)) < 1e-4 and abs(float(c0 - ce)) < 1e-6, f"{float(d1):.6f} {float(c0):.6f} vs {float(ce):.6f}")
    lg = torch.randn(7, 256)
    tg = torch.randint(0, 256, (7,))
    dn = depth_nll(lg, tg)
    check("2 the eight conditional NLLs sum to the byte's NLL", torch.allclose(dn.sum(-1), F.cross_entropy(lg, tg, reduction="none"), atol=1e-5))
    # the copy mask: a random 16-byte string, repeated at distance 300 (marked past the m-gram) and at distance 40 (unmarked)
    L = 400
    seq = torch.randint(0, 256, (L + 1,))
    s = torch.randint(0, 256, (16,))
    seq[10:26] = s
    seq[310:326] = s          # distance 300 >= D
    seq2 = seq.clone()
    seq2[310:326] = torch.randint(0, 256, (16,))
    seq2[60:76] = s           # distance 50 < D
    idx, tgt = seq[None, :-1], seq[None, 1:]
    cm = copy_mask(idx, tgt, m=4, D=256)[0]
    marked = cm.nonzero()[:, 0].tolist()
    want = list(range(313, 325))   # target positions 313..324 = bytes 314..325 of the repeat (the first m bytes' m-grams include filler)
    cm2 = copy_mask(seq2[None, :-1], seq2[None, 1:], m=4, D=256)[0]
    check("2 the copy mask marks the far repeat past its first m bytes and nothing else", marked == want, str(marked[:20]))
    check("2 a repeat nearer than D is not marked", int(cm2.sum()) == 0, str(int(cm2.sum())))
    m.loss_form = "copy"
    m.loss_copy_beta = 2.0
    xb, yb = seq[None, :-1], seq[None, 1:]
    c2 = m(xb, targets=yb).loss
    m.loss_form = "ce"
    ceb = m(xb, targets=yb).loss
    m.loss_form = "depth"
    m.loss_depth_w = (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0)
    d2 = m(xb, targets=yb).loss
    check("2 the weighted forms differ from the CE when their weights bite", abs(float(c2 - ceb)) > 1e-4 and abs(float(d2 - ceb)) > 1e-3)
    m.eval()
    with torch.no_grad():
        ev = m(xb, targets=yb).loss
    check("2 eval mode is the plain CE whatever the form", abs(float(ev - ceb)) < 1e-6)
    # the forms refuse bad settings on the config
    bad = False
    try:
        TrainConfig(loss_form="bits")
    except ValueError:
        bad = True
    check("2 an unknown loss form is refused by TrainConfig", bad)


def case_3():
    p = make_control_resume_preset("x-lever", extra_train={"loss_form": "copy", "loss_copy_beta": 2.0},
                                   phase_dataset_override={"fineweb_main": "fineweb-recall-far-5"})
    ph = {x["name"]: x["dataset"] for x in p.curriculum}
    check("3 extra_train sets the loss form; the override moves one phase's dataset",
          p.train.loss_form == "copy" and p.train.loss_copy_beta == 2.0 and ph["fineweb_main"] == "fineweb-recall-far-5"
          and ph["warmup_wikitext"] == "wikitext-103" and ph["anneal_mix"] == "anneal-mix")
    q = make_control_resume_preset("x-plain")
    check("3 the plain factory keeps the CE and fineweb", q.train.loss_form == "ce" and
          {x["name"]: x["dataset"] for x in q.curriculum}["fineweb_main"] == "fineweb-edu")
    refused = 0
    for kw in (dict(extra_train={"no_such_field": 1}), dict(phase_dataset_override={"no_phase": "fineweb-edu"}),
               dict(extra_train={"loss_form": "bits"})):
        try:
            make_control_resume_preset("x-bad", **kw)
        except ValueError:
            refused += 1
    check("3 unknown fields, phases and forms are refused", refused == 3, str(refused))


def main():
    for fn in (case_1, case_2, case_3):
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
