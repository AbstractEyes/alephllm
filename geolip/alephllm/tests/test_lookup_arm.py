# -*- coding: utf-8 -*-
"""The lookup arm (0.10.15) and the byte screen's arm factory, on CPU with a tiny all-hub craft.

  1. birth identity: a model with arms loads a no-arm state dict with exactly the arm keys missing and gives the SAME
     logits (the output projection is zero at birth; QK-norm gains at one);
  2. the arm trains alone: freeze_except + build_optimizers(adam_prefixes) put every arm parameter under Adam and none
     under Muon; after one Adam step the arm is alive (logits differ) and every trunk parameter is bit-identical;
  3. config round trip: lookup_arm_sites survives to_dict / from_dict as a tuple;
  4. decode with arms is refused (owed);
  5. the v3 lookup-arm preset: the name, the sites, the switches, arm_params, the 8 x 8 step, the two-phase curriculum
     whose cursor at the trunk's step lands on the arm's first step, the init_from path;
  6. the byte screen's factory: the seven arms build with their names, mixes and losses; the refusals.
Run: python -m geolip.alephllm.tests.test_lookup_arm"""
from __future__ import annotations

import torch

from ..model.alephlm import AlephLM
from ..presets import (AlephLMConfig, make_byte_screen_preset, make_v3_lookup_arm_preset)
from ..train.manifest import RunManifest
from ..train.optim import build_optimizers, freeze_except

TINY = AlephLMConfig(name="tiny-la", d_model=64, n_layers=4, n_heads=4, context=128, hub_layers=(0, 1, 2, 3),
                     hub_K=8, hub_D=16, hub_const=1, bank_experts=2, bank_ff=64, head_K=8, head_D=16, hub_chunk=32,
                     qk_norm="rms", attn_fp32=True, attn_kernel="fp16")
ARMED = AlephLMConfig.from_dict({**TINY.to_dict(), "name": "tiny-la-armed", "lookup_arm_sites": (1, 3)})
PASSED = []
FAILED = []


def ok(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def main():
    torch.manual_seed(0)
    x = torch.randint(0, 255, (2, 64))
    # ---- 1. birth identity
    m0 = AlephLM(TINY).eval()
    m1 = AlephLM(ARMED).eval()
    res = m1.load_state_dict(m0.state_dict(), strict=False)
    arm_keys = {k for k in m1.state_dict() if k.startswith(("lookup_arms.", "lookup_norms."))}
    ok("1a the arm keys are exactly the missing keys", set(res.missing_keys) == arm_keys and not res.unexpected_keys,
       f"missing {sorted(res.missing_keys)[:3]} unexpected {list(res.unexpected_keys)[:3]}")
    ok("1b two arms at sites 1 and 3", set(m1.lookup_arms.keys()) == {"1", "3"} and m1.lookup_sites == (1, 3))
    ok("1c the output projections are zero at birth", all(float(a.o.weight.abs().max()) == 0.0 for a in m1.lookup_arms.values()))
    ok("1d the QK-norm gains are one at birth", all(float((a.q_gain - 1).abs().max()) == 0.0 and float((a.k_gain - 1).abs().max()) == 0.0
                                                  for a in m1.lookup_arms.values()))
    with torch.no_grad():
        l0 = m0(x).logits
        l1 = m1(x).logits
    ok("1e identical logits at birth", torch.equal(l0, l1), f"max |diff| {float((l0 - l1).abs().max()):.3e}")
    # ---- 2. the arm trains alone
    m1.train()
    n_train, n_all = freeze_except(m1, ("lookup_arms.", "lookup_norms."))
    arm_params = [p for n, p in m1.named_parameters() if n.startswith(("lookup_arms.", "lookup_norms."))]
    ok("2a the trainable count is the arm's", n_train == sum(p.numel() for p in arm_params) and n_train < n_all,
       f"{n_train} of {n_all}")
    muon, adam = build_optimizers(m1, 2e-2, 0.95, 3e-4, adam_prefixes=("lookup_arms.", "lookup_norms."))
    ids_adam = {id(p) for g in adam.param_groups for p in g["params"]}
    ids_muon = {id(p) for g in muon.param_groups for p in g["params"]}
    ok("2b every arm parameter under Adam, none under Muon",
       all(id(p) in ids_adam for p in arm_params) and not any(id(p) in ids_muon for p in arm_params))
    trunk_before = {n: p.detach().clone() for n, p in m1.named_parameters() if not n.startswith(("lookup_arms.", "lookup_norms."))}
    tg = torch.randint(0, 255, (2, 64))
    for _ in range(2):
        loss = m1(x, targets=tg).loss
        loss.backward()
        ok_grad = all(p.grad is None for n, p in m1.named_parameters() if n in trunk_before)
        muon.step(); adam.step()
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
    ok("2c the trunk receives no gradients", ok_grad)
    ok("2d the trunk is bit-identical after two steps",
       all(torch.equal(trunk_before[n], p.detach()) for n, p in m1.named_parameters() if n in trunk_before))
    ok("2e the arm is alive after two steps (output projection moved)",
       all(float(a.o.weight.abs().max()) > 0.0 for a in m1.lookup_arms.values()))
    m1.eval()
    with torch.no_grad():
        l2 = m1(x).logits
    ok("2f the logits moved", not torch.equal(l0, l2), f"max |diff| {float((l0 - l2).abs().max()):.3e}")
    # ---- 3. config round trip
    d = ARMED.to_dict()
    back = AlephLMConfig.from_dict(d)
    ok("3 lookup_arm_sites round-trips (list in the dict, tuple back)", d["lookup_arm_sites"] == [1, 3] and back.lookup_arm_sites == (1, 3))
    # ---- 4. decode refused
    try:
        m1.prefill(x[:, :8])
        ok("4 decode with arms is refused", False, "no assertion")
    except AssertionError as e:
        ok("4 decode with arms is refused", "owed" in str(e))
    # ---- 5. the v3 lookup-arm preset
    p = make_v3_lookup_arm_preset((2, 6, 12, 20), steps=4000)
    cfg, t = p.model, p.train
    ok("5a name and sites", cfg.name == "mini-beatrix-3-la2-6-12-20" and tuple(cfg.lookup_arm_sites) == (2, 6, 12, 20))
    ok("5b the v3 trunk (32 hubs) with the arm's switches",
       cfg.n_layers == 32 and tuple(cfg.hub_layers) == tuple(range(32)) and cfg.qk_norm == "rms" and cfg.attn_fp32 and cfg.attn_kernel == "fp16")
    ok("5c arm_params, 8 x 8, the governor off, bf16", t.arm_params == ("lookup_arms.", "lookup_norms.") and (t.micro_batch, t.grad_accum) == (8, 8)
       and t.governor == "" and t.precision == "bf16" and t.ckpt_every == 1000)
    man = RunManifest.fresh(cfg.name, cfg.to_dict(), [dict(ph) for ph in p.curriculum])
    ph = man.set_cursor(245674, 262144)
    ok("5d the cursor at the trunk's step lands on the arm's first step",
       ph is not None and ph["name"] == "lookup_arm" and int(ph.get("tokens_done", 0)) == 0 and man.steps == 245674,
       f"{ph and ph['name']} tokens_done {ph and ph.get('tokens_done')}")
    ok("5e init_from = the released v3 checkpoint", p.init_from["prefix"] == "mini-beatrix-3" and p.init_from["path"] == "checkpoints/step_00245674.safetensors"
       and p.init_from["step"] == 245674 and p.init_from["tokens_per_step"] == 262144)
    ok("5f the arm phase is the far-recall mix for 4,000 steps", p.curriculum[1]["dataset"] == "fineweb-recall-far-5"
       and p.curriculum[1]["planned_tokens"] == 4000 * 262144)
    try:
        make_v3_lookup_arm_preset((3, 1))
        ok("5g unsorted sites refused", False)
    except ValueError:
        ok("5g unsorted sites refused", True)
    # ---- 6. the byte screen's factory
    arms = {}
    for side, kinds in (("twin", ("ctrl", "rows", "rows-copy", "depth")), ("2s", ("ctrl", "rows", "rows-copy"))):
        for kind in kinds:
            arms[(side, kind)] = make_byte_screen_preset(side, kind, 18000)
    fw = lambda pr: [ph for ph in pr.curriculum if ph["name"] == "fineweb_main"][0]["dataset"]  # noqa: E731
    ok("6a seven arms, their names", [a.model.name for a in arms.values()] == [
        "mini-beatrix-2s-control-attn-fp16-c18k-ctrl", "mini-beatrix-2s-control-attn-fp16-c18k-rows",
        "mini-beatrix-2s-control-attn-fp16-c18k-rows-copy", "mini-beatrix-2s-control-attn-fp16-c18k-depth",
        "mini-beatrix-2s-c18k-ctrl", "mini-beatrix-2s-c18k-rows", "mini-beatrix-2s-c18k-rows-copy"])
    ok("6b the rows kinds read the far-recall mix, the others fineweb",
       all(fw(a) == ("fineweb-recall-far-5" if "rows" in k else "fineweb-edu") for (s, k), a in arms.items()))
    ok("6c the losses", arms[("twin", "rows-copy")].train.loss_form == "copy" and arms[("twin", "rows-copy")].train.loss_copy_beta == 2.0
       and arms[("twin", "depth")].train.loss_form == "depth" and abs(sum(arms[("twin", "depth")].train.loss_depth_w) - 8.0) < 1e-3
       and arms[("twin", "ctrl")].train.loss_form == "ce")
    ok("6d the twin side: fp16 flash in the fp32 block, 8 x 8, from the r10k arm", all(
        a.model.attn_kernel == "fp16" and a.model.attn_fp32 and (a.train.micro_batch, a.train.grad_accum) == (8, 8)
        and a.init_from["prefix"] == "mini-beatrix-2s-control-attn-fp16-r10k" for (s, k), a in arms.items() if s == "twin"))
    ok("6e the 2s side: its own checkpoint, 16 x 4", all(a.init_from["prefix"] == "mini-beatrix-2s" and (a.train.micro_batch, a.train.grad_accum) == (16, 4)
                                                       for (s, k), a in arms.items() if s == "2s"))
    refused = 0
    for args in (("2s", "depth", 18000), ("twin", "rows", 20000), ("v3", "ctrl", 18000), ("twin", "x", 18000)):
        try:
            make_byte_screen_preset(*args)
        except ValueError:
            refused += 1
    ok("6f the refusals (2s depth; a late rows start; an unknown side; an unknown kind)", refused == 4, str(refused))
    print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} checks passed")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
