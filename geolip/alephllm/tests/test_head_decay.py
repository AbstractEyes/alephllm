"""The decayed group (0.10.18 rc10): TrainConfig.decay_params moves the named weights from Muon to AdamW with decoupled weight decay.
Run: python -m geolip.alephllm.tests.test_head_decay   (CPU, seconds; no download)
Cases:
  1 off by default: build_optimizers is [Muon, Adam] over exactly split_params' two sets
  2 the split with the output layer named: its two readout weights form the AdamW group at the given decay; its bias and every
    other Adam-set parameter the decay-0 group; Muon keeps every other matrix; every parameter sits in exactly one group
  3 the decay is decoupled: a step on zero gradients scales the named weights by (1 - lr * decay) and moves nothing else
  4 decay 0 is the Adam update: on the same gradients the decay-0 group follows torch's Adam exactly, and Muon's remaining
    matrices move exactly as in the default split
  5 TrainConfig checks the settings: a negative decay or a decay without named weights is refused; lists become tuples; a prefix
    that names no weight is refused
"""
from __future__ import annotations

import copy

import torch

from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig, TrainConfig
from ..train.optim import Muon, build_optimizers, split_params

RESULTS: list = []
TINY = dict(name="tiny-decay", d_model=64, n_layers=4, n_heads=4, context=128, hub_layers=(0, 2), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)
HEAD = ("head.w_h.", "head.w_s.")


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _model(seed=0):
    torch.manual_seed(seed)
    return AlephLM(AlephLMConfig(**TINY))


def _ids(params):
    return {id(p) for p in params}


def case_1():
    m = _model()
    muon_p, adam_p = split_params(m)
    opts = build_optimizers(m)
    check("1a off by default: Muon then torch's Adam", isinstance(opts[0], Muon) and type(opts[1]) is torch.optim.Adam,
          f"{type(opts[0]).__name__}, {type(opts[1]).__name__}")
    check("1b the default sets are split_params' sets",
          _ids(opts[0].param_groups[0]["params"]) == _ids(muon_p) and _ids(opts[1].param_groups[0]["params"]) == _ids(adam_p)
          and len(opts[1].param_groups) == 1 and opts[1].param_groups[0]["weight_decay"] == 0.0)


def case_2():
    m = _model()
    named = dict(m.named_parameters())
    muon_p, adam_p = split_params(m)
    opts = build_optimizers(m, decay_prefixes=HEAD, decay_wd=0.1)
    g0, g1 = opts[1].param_groups
    want = {id(named["head.w_h.weight"]), id(named["head.w_s.weight"])}
    check("2a the AdamW group holds exactly the output layer's two readout weights at decay 0.1",
          type(opts[1]) is torch.optim.AdamW and _ids(g1["params"]) == want and g1["weight_decay"] == 0.1)
    check("2b the output bias and the rest of the Adam set sit at decay 0",
          id(named["head.w_h.bias"]) in _ids(g0["params"]) and _ids(g0["params"]) == _ids(adam_p) and g0["weight_decay"] == 0.0)
    check("2c Muon keeps every other matrix", _ids(opts[0].param_groups[0]["params"]) == _ids(muon_p) - want)
    every = [p for o in opts for g in o.param_groups for p in g["params"]]
    check("2d every parameter in exactly one group", len(every) == len(_ids(every)) == len(list(m.parameters())),
          f"{len(every)} placed, {len(list(m.parameters()))} parameters")


def case_3():
    m = _model()
    named = dict(m.named_parameters())
    lr, wd = 3e-4, 1.2
    opts = build_optimizers(m, adam_lr=lr, decay_prefixes=HEAD, decay_wd=wd)
    before = {n: p.detach().clone() for n, p in named.items()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    for o in opts:
        o.step()
    dec = ("head.w_h.weight", "head.w_s.weight")
    scaled = all(torch.allclose(named[n].detach(), before[n] * (1 - lr * wd), rtol=1e-7, atol=0) for n in dec)
    still = all(torch.equal(named[n].detach(), before[n]) for n in named if n not in dec)
    ratio = float(named["head.w_h.weight"].detach().norm() / before["head.w_h.weight"].norm())
    check("3a the named weights shrink by exactly (1 - lr * decay)", scaled, f"norm ratio {ratio:.8f} vs {1 - lr * wd:.8f}")
    check("3b nothing else moves on zero gradients", still)


def case_4():
    a, b = _model(), _model()
    na, nb = dict(a.named_parameters()), dict(b.named_parameters())
    oa = build_optimizers(a)
    ob = build_optimizers(b, decay_prefixes=HEAD, decay_wd=0.0)
    gen = torch.Generator().manual_seed(5)
    for _ in range(3):
        for n in na:
            g = torch.randn(na[n].shape, generator=gen)
            na[n].grad, nb[n].grad = g.clone(), g.clone()
        for o in oa + ob:
            o.step()
    dec = ("head.w_h.weight", "head.w_s.weight")
    adam_names = [n for n, p in a.named_parameters() if id(p) in _ids(oa[1].param_groups[0]["params"])]
    muon_names = [n for n, p in b.named_parameters() if id(p) in _ids(ob[0].param_groups[0]["params"])]
    check("4a the decay-0 group follows torch's Adam bit for bit", all(torch.equal(na[n], nb[n]) for n in adam_names),
          f"{len(adam_names)} tensors")
    check("4b Muon's remaining matrices move exactly as in the default split", all(torch.equal(na[n], nb[n]) for n in muon_names),
          f"{len(muon_names)} tensors")
    ref = copy.deepcopy(a)
    w = dict(ref.named_parameters())["head.w_h.weight"]
    w.data.copy_(dict(_model().named_parameters())["head.w_h.weight"].data)
    opt = torch.optim.Adam([w], lr=3e-4)
    gen = torch.Generator().manual_seed(5)
    grads = []
    for _ in range(3):
        step_g = {n: torch.randn(na[n].shape, generator=gen) for n in na}
        grads.append(step_g["head.w_h.weight"])
    for g in grads:
        w.grad = g.clone()
        opt.step()
    check("4c at decay 0 the output layer's weight takes torch's Adam steps", torch.equal(w.detach(), nb["head.w_h.weight"].detach()))
    check("4d and differs from its Muon path in the default split", not torch.equal(na["head.w_h.weight"], nb["head.w_h.weight"]))
    return dec


def case_5():
    try:
        TrainConfig(decay_wd=-0.1, decay_params=HEAD)
        neg = False
    except ValueError:
        neg = True
    try:
        TrainConfig(decay_wd=0.1)
        bare = False
    except ValueError:
        bare = True
    tc = TrainConfig(decay_params=list(HEAD), decay_wd=0.1)
    check("5a a negative decay is refused", neg)
    check("5b a decay without named weights is refused", bare)
    check("5c the prefixes become a tuple", tc.decay_params == HEAD)
    try:
        build_optimizers(_model(), decay_prefixes=("head.nothing.",), decay_wd=0.1)
        none = False
    except ValueError:
        none = True
    check("5d a prefix that names no weight is refused", none)


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3, case_4, case_5):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
