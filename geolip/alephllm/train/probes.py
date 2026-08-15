"""Curriculum probe batteries P0..P8 — the stage instruments.

Each item is multiple-choice scored by byte-NLL of the option
continuation (fp32, no sampling): the model prefers the right ending or
it doesn't. Batteries carry per-family sub-scores; the causal-test
holdout families (rulechain depth-5, 3-digit subtraction) are gauged
here from time zero even though no training row ever contains them —
that contrast IS the arm-collective experiment's readout.

Deterministic: fixed item lists, no RNG at probe time.
"""
from __future__ import annotations

import math

import torch


def _mc(prompt, options, answer=0, family="core"):
    return {"prompt": prompt, "options": options, "answer": answer,
            "family": family}


def _chain(depth, who="Pia", start=0):
    preds = ["blim", "torv", "quen", "harl", "sook", "vell"]
    ps = [preds[(start + i) % len(preds)] for i in range(depth + 1)]
    rules = " ".join(f"If someone is {ps[i]}, then they are {ps[i + 1]}."
                     for i in range(depth))
    steps = " ".join(f"So {who} is {ps[i + 1]}." for i in range(depth - 1))
    prompt = (rules + f" {who} is {ps[0]}. " + (steps + " " if steps else "")
              + f"So {who} is")
    wrong = preds[(start + depth + 2) % len(preds)]
    return _mc(prompt, [f" {ps[depth]}.", f" {wrong}.", f" not {ps[0]}."],
               0, family=f"d{depth}")


SUITES: dict[str, list] = {
    "p0_grounding": [
        _mc("Mira counts the shells: 1, 2, 3, 4. There are",
            [" 4 shells.", " 5 shells.", " 3 shells."]),
        _mc("Mira counts the cups: 1, 2, 3, 4, 5, 6. There are 6 cups. "
            "One more cup arrives. Now there are",
            [" 7 cups.", " 6 cups.", " 5 cups."]),
        _mc("Tom put the ball in the box and left the room. The ball is "
            "in the", [" box.", " room.", " garden."]),
        _mc("The cat sat on the warm stone. The thing the cat sat on "
            "was", [" the stone.", " the cat.", " the sun."]),
        _mc("Lily dropped the cup and it broke. The broken thing was",
            [" the cup.", " Lily.", " the floor."]),
        _mc("First it rained, then the path got muddy. The path got "
            "muddy", [" after the rain.", " before the rain.",
                      " instead of the rain."]),
    ],
    "p1_self_other": [
        _mc("Sana put the marble in the basket and went outside. While "
            "Sana was away, Tom moved the marble to the jar. When Sana "
            "came back, Sana looked for the marble in the",
            [" basket.", " jar.", " bag."], family="belief"),
        _mc("Sana put the marble in the basket and went outside. While "
            "Sana was away, Tom moved the marble to the jar. The marble "
            "was really in the", [" jar.", " basket.", " drawer."],
            family="reality"),
        _mc("Lea gave Rui a pear because Rui was hungry. The hungry one "
            "was", [" Rui.", " Lea.", " the pear."], family="reference"),
        _mc("Maya said to Kofi: you dropped the crayon. The one who "
            "dropped the crayon was", [" Kofi.", " Maya.", " the box."],
            family="reference"),
        _mc("Seen from Tom: I dropped the acorn. Told about Tom: Tom",
            [" dropped the acorn.", " caught the acorn.",
             " ate the acorn."], family="perspective"),
    ],
    "p2_difference": [
        _mc("A robin is a kind of bird. A trout is a kind of fish. A "
            "robin and a trout are",
            [" different kinds.", " the same kind.", " both trees."]),
        _mc("A whale and a beetle: are they alike? A whale is a mammal "
            "and a beetle is an insect, so they are",
            [" different kinds.", " the same kind.", " both fish."]),
        _mc("Hot is the opposite of", [" cold.", " warm.", " red."]),
        _mc("If a thing is heavy, it is not", [" light.", " big.",
                                               " hard."]),
        _mc("An oak is a kind of tree. Most trees can",
            [" grow tall.", " swim.", " fly."]),
    ],
    "p3_logic": ([_chain(d, who=w, start=s)
                  for d, w, s in [(1, "Pia", 0), (1, "Kel", 2),
                                  (2, "Ezo", 1), (2, "Osa", 3),
                                  (3, "Vin", 0), (3, "Pia", 2),
                                  (4, "Tam", 1), (4, "Kel", 3),
                                  (5, "Osa", 0), (5, "Vin", 2)]]  # d5=holdout
                 + [
        _mc("If someone is blim, then they are torv. Pia is not torv. "
            "So Pia is", [" not blim.", " blim.", " torv."],
            family="negation"),
    ]),
    "p4_arithmetic": [
        _mc("Take 47 and add 25. 47 + 25 =", [" 72.", " 62.", " 73."],
            family="add2"),
        _mc("Take 38 and add 57. 38 + 57 =", [" 95.", " 85.", " 96."],
            family="add2"),
        _mc("Take 236 and add 458. 236 + 458 =",
            [" 694.", " 684.", " 794."], family="add3"),
        _mc("Start at 81 and take away 46. 81 - 46 =",
            [" 35.", " 45.", " 34."], family="sub2"),
        _mc("Count by 4s: 1 4s make 4; 2 4s make 8; 3 4s make 12. So 4 "
            "times 3 is", [" 12.", " 8.", " 16."], family="mul1"),
        _mc("Start at 642 and take away 275. 642 - 275 =",
            [" 367.", " 377.", " 467."], family="sub3"),   # holdout family
        _mc("Start at 813 and take away 358. 813 - 358 =",
            [" 455.", " 465.", " 555."], family="sub3"),   # holdout family
    ],
    "p5_causal": [
        _mc("Noor left the gate open. Because of that,",
            [" the goat walked out.", " the garden grew taller.",
             " the gate turned blue."]),
        _mc("Ivo watered the seeds each day. Because of that,",
            [" green shoots came up.", " the pots stayed empty.",
             " the rain stopped."]),
        _mc("Sela stacked the cups too high, so the tower leaned, and "
            "then", [" the cups crashed down.", " the cups sang.",
                     " the table left."]),
        _mc("Noor left the gate open, so the goat walked out. If Noor "
            "had closed the gate first, then",
            [" the garden stayed whole.", " the goat walked out anyway.",
             " the gate opened itself."], family="counterfactual"),
        _mc("After Ada repairs the fence, Ada feels",
            [" proud.", " purple.", " taller."], family="atomic"),
    ],
    "p6_learning": [
        _mc("Rina adds 34 and 21 and writes 61. Rina checks by counting "
            "back: 34 + 21 =", [" 55, so 61 is wrong.",
                                " 61, so it is right.",
                                " 12, so both are wrong."]),
        _mc("Josef builds a paper bridge and it sags. The middle has no "
            "fold. To make it hold, Josef should",
            [" fold a ridge down the middle.", " remove more paper.",
             " push down harder on it."]),
        _mc("Talia tries a key and it does not turn. A different key "
            "opens the lock. Next time at this door, Talia should",
            [" use the key that worked.", " use the key that failed.",
             " use no key at all."]),
    ],
    "p7_composite": [
        _mc("If a box is heavy, then it needs two hands. The stone box "
            "weighs 40 and the straw box weighs 4. The stone box is the "
            "heavy one, so it", [" needs two hands.", " needs no hands.",
                                 " is the light one."]),
        _mc("Emre has 12 beads and gives away 5. If someone has fewer "
            "than 8 beads, they visit the bead stall. 12 - 5 = 7, and 7 "
            "is fewer than 8, so Emre",
            [" visits the bead stall.", " stays home.",
             " gives away 8 more."]),
    ],
    "p8_articulation": [
        _mc("The word 'island' means",
            [" land with water all the way around it.",
             " a wide stream of water that flows across land.",
             " a very tall hill of sand."]),
        _mc("The word 'fragile' means",
            [" easy to break and needing gentle hands.",
             " very strong and hard to bend.",
             " from a very long time ago."]),
        _mc("Formally: the bridge is closed; pedestrians are directed to "
            "the ferry. Plainly: the bridge is",
            [" shut, so people walk onto the ferry.",
             " open, so people walk across it.",
             " a ferry, so people drive over it."], family="register"),
    ],
}

HOLDOUT_FAMILIES = {"p3_logic": {"d5"}, "p4_arithmetic": {"sub3"}}


@torch.no_grad()
def _option_nll(model, tok, device, prompt: str, option: str) -> float:
    p = tok.encode(prompt)
    full = tok.encode(prompt + option)
    x = torch.tensor([full[:-1].tolist()], device=device)
    logits = model(x)[0].float()
    lp = torch.log_softmax(logits[0], dim=-1)
    n_p = len(p)
    tgt = torch.tensor(full[n_p:].tolist(), device=device)
    sel = lp[n_p - 1: n_p - 1 + tgt.shape[0]]
    return float(-sel.gather(1, tgt[:, None]).sum().item())


@torch.no_grad()
def run_suite(model, tok, device, name: str) -> dict:
    was = model.training
    model.eval()
    fam_hit, fam_n = {}, {}
    for item in SUITES[name]:
        nlls = [_option_nll(model, tok, device, item["prompt"], o)
                for o in item["options"]]
        hit = int(min(range(len(nlls)), key=nlls.__getitem__)
                  == item["answer"])
        f = item["family"]
        fam_hit[f] = fam_hit.get(f, 0) + hit
        fam_n[f] = fam_n.get(f, 0) + 1
    if was:
        model.train()
    total = sum(fam_hit.values()) / max(sum(fam_n.values()), 1)
    return {"acc": total,
            "families": {f: fam_hit[f] / fam_n[f] for f in fam_n},
            "n": sum(fam_n.values())}


@torch.no_grad()
def run_all(model, tok, device) -> dict:
    return {name: run_suite(model, tok, device, name) for name in SUITES}


def report(results: dict) -> str:
    lines = []
    for name, r in results.items():
        fams = " ".join(f"{f}:{a:.2f}" for f, a in
                        sorted(r["families"].items())
                        if len(r["families"]) > 1)
        hold = HOLDOUT_FAMILIES.get(name, set())
        mark = "".join(f" [holdout {h}:{r['families'][h]:.2f}]"
                       for h in sorted(hold) if h in r["families"])
        lines.append(f"  {name:16} {r['acc']:.3f} ({r['n']} items)"
                     + (f"  {fams}" if fams else "") + mark)
    return "\n".join(lines)
