"""The lookup arm's depth (0.10.17): extra attention layers per site, keyed <site>_l<k> under lookup_arms / lookup_norms.
Run: python -m geolip.alephllm.tests.test_lookup_arm_depth   (CPU, seconds; no download)
Cases:
  1 keys: a depth-2 craft's state dict holds the depth-1 craft's keys unchanged plus six _l2 keys per site; depth 1 adds none
  2 birth identity: a depth-2 model loads a TRAINED depth-1 arm's state dict with exactly the _l2 keys missing, no unexpected
    keys, and gives the same logits (the new layers' output projections are zero)
  3 training: under the arm prefixes every arm parameter sits in Adam and none in Muon; after one step the new layer's output
    projection has moved and every trunk parameter is bit-identical
  4 the config round trip keeps the depth (and defaults to 1 for an old dict); the preset's depth: lookup_arm_depth, the -d2
    name suffix, arm_params unchanged, a depth below 1 refused
"""
from __future__ import annotations

import torch

from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig, make_v3_lookup_arm_preset
from ..train.optim import build_optimizers, freeze_except

TINY = AlephLMConfig(name="tiny-la", d_model=64, n_layers=4, n_heads=4, context=128, hub_layers=(0, 1, 2, 3),
                     hub_K=8, hub_D=16, hub_const=1, bank_experts=2, bank_ff=64, head_K=8, head_D=16, hub_chunk=32,
                     qk_norm="rms", attn_fp32=True, attn_kernel="fp16")
ARMED = AlephLMConfig.from_dict({**TINY.to_dict(), "name": "tiny-la-armed", "lookup_arm_sites": (1, 3)})
DEEP = AlephLMConfig.from_dict({**ARMED.to_dict(), "name": "tiny-la-deep", "lookup_arm_depth": 2})
RESULTS: list = []
ARM = ("lookup_arms.", "lookup_norms.")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def case_1():
    k1 = set(AlephLM(ARMED).state_dict())
    k2 = set(AlephLM(DEEP).state_dict())
    extra = k2 - k1
    check("1 the depth-1 keys are a subset of the depth-2 keys", k1 <= k2)
    check("1 six _l2 keys per site, all under the arm prefixes", len(extra) == 12 and all(k.startswith(ARM) and "_l2." in k for k in extra), str(sorted(extra)[:4]))
    check("1 depth 1 adds nothing", set(AlephLM(AlephLMConfig.from_dict({**ARMED.to_dict(), "lookup_arm_depth": 1})).state_dict()) == k1)


def case_2():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (2, 64))
    m1 = AlephLM(ARMED).eval()
    with torch.no_grad():                                   # a TRAINED arm: non-zero output projections, moved gains
        for arm in m1.lookup_arms.values():
            arm.o.weight.normal_(0, 0.02)
            arm.q_gain.mul_(1.1)
    m2 = AlephLM(DEEP).eval()
    res = m2.load_state_dict(m1.state_dict(), strict=False)
    check("2 exactly the _l2 keys missing, none unexpected", not res.unexpected_keys and len(res.missing_keys) == 12 and all("_l2." in k for k in res.missing_keys),
          f"missing {len(res.missing_keys)} unexpected {len(res.unexpected_keys)}")
    with torch.no_grad():
        l1 = m1(x).logits
        l2 = m2(x).logits
    check("2 the same logits at birth", torch.allclose(l1, l2, atol=1e-6, rtol=1e-5), f"max |diff| {(l1 - l2).abs().max().item():.2e}")
    with torch.no_grad():
        l0 = m1.__class__(TINY).eval()(x).logits
    check("2 the trained arm is not null on this batch (the identity means the new layer, not a dead arm)", not torch.allclose(l0, l1, atol=1e-4))


def case_3():
    torch.manual_seed(1)
    m = AlephLM(DEEP)
    trunk_before = {n: p.detach().clone() for n, p in m.named_parameters() if not n.startswith(ARM)}
    first_before = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith(ARM) and "_l2." not in n}
    freeze_except(m, ARM)
    muon, adam = build_optimizers(m, 0.02, 0.95, 1e-3, adam_prefixes=ARM)
    adam_ids = {id(p) for g in adam.param_groups for p in g["params"]}
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]} if muon is not None else set()
    arm_params = [p for n, p in m.named_parameters() if n.startswith(ARM)]
    check("3 every arm parameter under Adam, none under Muon", all(id(p) in adam_ids for p in arm_params) and not any(id(p) in muon_ids for p in arm_params))
    x = torch.randint(0, 255, (2, 64))
    out = m(x, targets=x)
    out.loss.backward()
    adam.step()
    moved = [n for n, p in m.named_parameters() if "_l2." in n and n.endswith("o.weight") and p.abs().sum().item() > 0]
    check("3 the new layers' output projections moved off zero", len(moved) == 2, str(moved))
    same_trunk = all(torch.equal(p.detach(), trunk_before[n]) for n, p in m.named_parameters() if n in trunk_before)
    check("3 the trunk is bit-identical", same_trunk)
    first_moved = any(not torch.equal(p.detach(), first_before[n]) for n, p in m.named_parameters() if n in first_before)
    check("3 the first layer trains too under the arm prefixes", first_moved)


def case_4():
    check("4 round trip keeps the depth", AlephLMConfig.from_dict(DEEP.to_dict()).lookup_arm_depth == 2)
    d = ARMED.to_dict()
    d.pop("lookup_arm_depth", None)
    check("4 an old dict defaults to depth 1", AlephLMConfig.from_dict(d).lookup_arm_depth == 1)
    p2 = make_v3_lookup_arm_preset(depth=2)
    p1 = make_v3_lookup_arm_preset()
    check("4 the preset's depth and name", p2.model.lookup_arm_depth == 2 and p2.model.name == "mini-beatrix-3-la2-6-12-20-d2"
          and p1.model.lookup_arm_depth == 1 and p1.model.name == "mini-beatrix-3-la2-6-12-20")
    check("4 arm_params unchanged", p2.train.arm_params == ("lookup_arms.", "lookup_norms."))
    try:
        make_v3_lookup_arm_preset(depth=0)
        check("4 depth 0 refused", False)
    except ValueError:
        check("4 depth 0 refused", True)
    p3 = make_v3_lookup_arm_preset(depth=2, steps=4000, start_step=254000, source="mini-beatrix-3-la2-6-12-20-deep1k",
                                   mix="fineweb-recall-deep-25", arm_lr=3e-4, name="mini-beatrix-3-la2-6-12-20-deep-d2")
    check("4 the pair's structure arm builds", p3.model.name == "mini-beatrix-3-la2-6-12-20-deep-d2" and p3.model.lookup_arm_depth == 2
          and abs(p3.train.adam_lr - 3e-4) < 1e-12 and p3.init_from["step"] == 254000 and p3.curriculum[1]["dataset"] == "fineweb-recall-deep-25")


def main():
    for fn in (case_1, case_2, case_3, case_4):
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
