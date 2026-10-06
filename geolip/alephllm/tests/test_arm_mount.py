"""The arm mount on a stand-in. Run: python -m geolip.alephllm.tests.test_arm_mount   (CPU, seconds; no download)

A small random trunk of the same family (3 blocks, d 64) and a three-member group written the training route's way (fresh arms
attached in order through the stage-arm program, perturbed so they write, saved as anchors with the step's base_model_id). Cases:
  1 the mount reproduces the program that wrote the anchors: logits bit-equal on a random batch, every member on;
  2 masks: only([third]) equals a trunk with only the third mounted; all masked equals the bare trunk; masks restore;
  3 detach_all with verify: the bare logits bit-equal to the pre-attach logits;
  4 the carried-member check: the same tensors re-saved (other header bytes) pass; a tampered tensor fails, named alone;
  5 the trunk-bound check: an anchor from another step is refused, named;
  6 mount_anchors in another order attaches in that order.
Needs the amoe package with its alephlm binding on the path.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

import torch

from ..presets import AlephLMConfig
from .. import arm_mount as AM

TINY = AlephLMConfig(name="tiny-mount-test", d_model=64, n_layers=3, n_heads=2, context=64, vocab_size=256, tokenizer="byte-trigram",
                     hub_layers=(0, 1, 2), hub_K=8, hub_D=16, tau=0.1, bank_experts=3, bank_ff=64, head_K=16, head_D=16, gate_init=-3.0,
                     tie_embeddings=False, hub_chunk=16, hub_const=2, hub_ckpt=0)
MEMBERS = ["s1_perspective", "s2_concept", "s9_caption"]
RESULTS: list = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def write_group(model, members, out_dir, run="gTA", close=3, step=AM.STEP):
    """Fresh arms through the program, perturbed, saved as anchors -> (program, {member: path})."""
    from ..train.arms import StageArmProgram, ArmProgramConfig, StageArm
    arms = [StageArm(name=m, phase="test", spec=dict(n_slots=4, K=4, D=4, hidden=16), lam=2.0, seed=10 + i) for i, m in enumerate(members)]
    prog = StageArmProgram(ArmProgramConfig(arms=arms)).bind(AM.inference_binding(model))
    g = torch.Generator().manual_seed(7)
    for a in arms:
        prog.attach(a)
        with torch.no_grad():
            for w in prog.wraps[a.name]:
                w.adapter.consume[-1].weight.add_(torch.randn(w.adapter.consume[-1].weight.shape, generator=g) * 0.2)
                w.adapter.consume[-1].bias.add_(torch.randn(w.adapter.consume[-1].bias.shape, generator=g) * 0.2)
                w.adapter.gate.fill_(1.0)
    files = {}
    for m in members:
        p = os.path.join(out_dir, AM.anchor_name(run, close, m))
        prog.anchor(m, AM.CRAFT, step).save(p)
        files[m] = p
    model.eval()
    return prog, files


def run_all() -> bool:
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="arm_mount_test_")
    base = AM.load_trunk(step=None, cfg=TINY)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    x = torch.randint(0, 256, (2, 40), generator=torch.Generator().manual_seed(1))
    bare = AM.logits(base, x)
    _, files = write_group(base, MEMBERS, tmp)
    with torch.no_grad():
        ref = AM.logits(base, x)
    check("the stand-in arms write on the trunk", float((ref - bare).abs().max()) > 1e-4, f"max abs write {float((ref - bare).abs().max()):.3f}")

    # 1 the mount reproduces the writing program
    m1 = AM.load_trunk(step=None, cfg=TINY, state=state)
    prog = AM.mount_group(m1, "gTA", close=3, members=MEMBERS, files=files, check_base=False)
    with torch.no_grad():
        got = AM.logits(m1, x)
    check("1 mount reproduces the program (bit-equal logits)", torch.equal(got, ref), f"max abs {float((got - ref).abs().max())}")
    check("1 members attached in order", list(prog.attached) == MEMBERS)
    check("1 anchors recorded", prog.mounted["anchors"]["s9_caption"]["base_model_id"] == f"alephllm/{AM.CRAFT}@step{AM.STEP}")
    check("1 adapters frozen", not any(p.requires_grad for p in prog.params()))

    # 2 masks
    m2 = AM.load_trunk(step=None, cfg=TINY, state=state)
    AM.mount_group(m2, "gTA", close=3, members=["s9_caption"], files={"s9_caption": files["s9_caption"]}, check_base=False)
    with torch.no_grad():
        third_alone = AM.logits(m2, x)
        with AM.only(prog, ["s9_caption"]):
            got = AM.logits(m1, x)
        check("2 only(third) == the third mounted alone", torch.equal(got, third_alone), f"max abs {float((got - third_alone).abs().max())}")
        with AM.masked(prog, MEMBERS):
            got = AM.logits(m1, x)
        check("2 all masked == the bare trunk", torch.equal(got, bare))
        got = AM.logits(m1, x)
    check("2 masks restored", torch.equal(got, ref))

    # 3 detach
    AM.detach_all(prog, verify=True)
    with torch.no_grad():
        got = AM.logits(m1, x)
    check("3 detach_all restores the bare trunk bit-exact", torch.equal(got, bare) and not prog.attached)

    # 4 the carried-member check
    from amoe.io.checkpoint import load_anchor
    base_files = {}
    for m in MEMBERS[:2]:
        ck = load_anchor(files[m])
        ck.meta["name"] = m + "-resaved"
        p = os.path.join(tmp, AM.anchor_name("gTX", 2, m))
        ck.save(p)
        base_files[m] = p
    hashes = AM.check_frozen_members(files, base_files, MEMBERS[:2])
    check("4 re-saved tensors pass the carried-member check", all(hashes.values()))
    ck = load_anchor(files["s2_concept"])
    k0 = sorted(ck.adapters)[0]
    ck.adapters[k0] = ck.adapters[k0] + 1e-3
    bad = os.path.join(tmp, AM.anchor_name("gTX", 2, "s2_concept"))
    ck.save(bad)
    try:
        AM.check_frozen_members(files, {"s1_perspective": base_files["s1_perspective"], "s2_concept": bad}, MEMBERS[:2])
        check("4 a tampered tensor fails the carried-member check", False, "no error")
    except ValueError as e:
        check("4 a tampered tensor fails the carried-member check", "s2_concept" in str(e) and "s1_perspective" not in str(e))

    # 5 the trunk-bound check
    m5 = AM.load_trunk(step=None, cfg=TINY, state=state)
    try:
        AM.mount_group(m5, "gTA", close=3, members=MEMBERS, files=files, check_base=False, require_step=AM.STEP + 1)
        check("5 an anchor from another step is refused", False, "no error")
    except ValueError as e:
        check("5 an anchor from another step is refused", "trunk-bound" in str(e) and MEMBERS[0] in str(e))

    # 6 another order
    m6 = AM.load_trunk(step=None, cfg=TINY, state=state)
    order = [MEMBERS[2], MEMBERS[0], MEMBERS[1]]
    prog6 = AM.mount_anchors(m6, {m: files[m] for m in order})
    check("6 mount_anchors keeps the given order", list(prog6.attached) == order)
    AM.detach_all(prog6, verify=True)

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASSED' if not failed else 'FAILED: ' + ', '.join(failed)} ({len(RESULTS)} checks; {tmp})", flush=True)
    return not failed


def test_arm_mount():
    assert run_all()


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
