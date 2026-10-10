"""The control twins' continuation (2026-10-09): the softmax guards, the precision switch, the weights-only start.
Run: python -m geolip.alephllm.tests.test_control_fixes   (CPU, seconds; no download: the synthetic stream)
Cases:
  1 CausalSDPA with the guards off is the 2s form bit for bit (attn_fp32 is a no-op off a card)
  2 qk_norm 'rms': finite, causal, unit-RMS q/k under gains 1, decode parity (prefill + step == the full pass)
  3 install_gains: the gains equal the per-head per-channel RMS of q; the installed block sits closer to the raw block than
    gains 1 do; the logit scale is kept within 25%
  4 precision: the autocast contexts; TrainConfig and autocast refuse an unknown precision
  5 cursor_at_step reproduces every boundary of the 2s twins' schedule (1,145 ... 61,422) and the step-16,000 cursor
  6 the registered arms: the four presets and their switches, the craft and recipe verbatim otherwise, chronological phases,
    a 14,000 build, no cross-mutation of the twin
  7 the weights-only start on a tiny craft (CPU, tokenless): the cursor, the weights, the installed gains, the restarted
    phase's seed, fresh optimizers, two steps, the final checkpoint, the hand-off to the normal resume, one more step, a
    kernel change on a later resume recorded in the manifest
  8 attn_fp32 on a card (skipped without one): the fp32-attention output under autocast is fp32 and sits closer to the
    plain fp32 pass than the bf16 pass does
  9 attn_kernel: the default is sdpa, an unknown kernel is refused, the config's kernel reaches every block and survives the
    dict round trip, the two fp32-attention arms under bf16 carry "flex" and the others "sdpa", off a card the flex block
    IS the sdpa block, a dead kernel falls back to sdpa; on a card (skipped without one, or when the kernel will not build):
    the fused kernel's forward and gradients agree with fp32 sdpa and the eval path stays on sdpa bit for bit
  10 attn_kernel "fp16": accepted, off a card the block is the sdpa block, the two fp16 arms carry it, the trainer's kernel
    label names it; on a card: forward and gradient within 1e-2 of fp32 sdpa, a loss scaled down by 1e-6 (gradients under
    fp16's floor) still yields gradients within 1e-2 of fp32's, a sink key attended by every query yields finite gradients
    within 2e-2 of fp32's, and the eval path stays on sdpa bit for bit; the counters: an overflowing sink backward fires a
    retry and stays finite, a forward overflow falls back to fp32 sdpa for that call (output and gradients the fp32 path's)
  11 the continuation source: an arm whose source is another arm's prefix (init_from on that prefix, the twin's repo, its own
    seed offset; the derived name at another step keeps the prefix); the bf16 + QK-norm arm; on a card QK-norm under bf16
    autocast gives a finite bf16 output with gradients to the gains
"""
from __future__ import annotations

import contextlib
import math
import os
import tempfile

import torch

from ..model.attention import CausalSDPA
from ..model.alephlm import AlephLM
from ..presets import (AlephLMConfig, TrainConfig, Preset, PRESETS,
                       make_control_resume_preset, CONTROL_RESUME_ARMS)
from ..train.manifest import RunManifest, cursor_at_step
from ..train.precision import autocast

RESULTS: list = []
TINY = AlephLMConfig(name="tiny-control-src", d_model=64, n_layers=3, n_heads=4, context=128, hub_layers=(),
                     hub_K=32, hub_D=8, head_K=32, head_D=8, hub_chunk=16)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _same_weights(a: CausalSDPA, b: CausalSDPA):
    with torch.no_grad():
        b.qkv.weight.copy_(a.qkv.weight)
        b.o.weight.copy_(a.o.weight)


def case_1():
    torch.manual_seed(0)
    a, b = CausalSDPA(64, 4), CausalSDPA(64, 4, attn_fp32=True)
    _same_weights(a, b)
    x = torch.randn(2, 24, 64)
    with torch.no_grad():
        check("1 guards off == the 2s form (bit for bit)", torch.equal(a(x), b(x)))
    check("1 no guard params when off", not any(k.endswith(("q_gain", "k_gain")) for k in a.state_dict()))


def case_2():
    torch.manual_seed(1)
    m = CausalSDPA(64, 4, qk_norm="rms")
    x = torch.randn(2, 40, 64)
    with torch.no_grad():
        y = m(x)
        check("2 qk_norm forward finite + shape", bool(torch.isfinite(y).all()) and y.shape == x.shape)
        z = x.clone()
        z[:, 30:] = torch.randn(2, 10, 64)
        check("2 causal", float((m(x)[:, :29] - m(z)[:, :29]).abs().max()) < 1e-5)
        q, k, _ = m._split(x, 40)
        q2, k2 = m._guard(q, k)
        dev = max(float((q2.pow(2).mean(-1).sqrt() - 1).abs().max()), float((k2.pow(2).mean(-1).sqrt() - 1).abs().max()))
        check("2 unit-RMS q and k under gains 1", dev < 1e-3, f"max deviation {dev:.2e}")
        m.q_gain.mul_(1.7)
        m.k_gain.mul_(0.6)                      # non-trivial gains: the decode path must carry them
        full = m(x)
        out, cache = m.prefill(x[:, :-1])
        last = m.step(x[:, -1:], cache)
        d = max(float((last[:, 0] - full[:, -1]).abs().max()), float((out - full[:, :-1]).abs().max()))
        check("2 decode parity (prefill + step == the full pass)", d < 1e-5, f"max |d| {d:.2e}")
    sd = m.state_dict()
    check("2 guard params in the state dict", "q_gain" in sd and "k_gain" in sd and tuple(sd["q_gain"].shape) == (4, 1, 16))


def case_3():
    torch.manual_seed(2)
    raw, g = CausalSDPA(64, 4), CausalSDPA(64, 4, qk_norm="rms")
    with torch.no_grad():                          # a trained-looking scale: q and k 2.5x the init's (the init sits near unit RMS)
        raw.qkv.weight[:128].mul_(2.5)
    _same_weights(raw, g)
    scale = torch.linspace(0.2, 3.0, 64)          # anisotropic channels: the per-position scale varies, the gains matter
    x = torch.randn(4, 48, 64) * scale
    with torch.no_grad():
        q, k, _ = g._split(x, 48)
        q1, k1 = g._guard(q, k)                    # gains 1: unit-RMS q and k
        hd = q.shape[-1]
        s_raw = float((q[..., :48, :] @ k[..., :48, :].transpose(-1, -2)).abs().mean() / math.sqrt(hd))
        s_ones = float((q1[..., :48, :] @ k1[..., :48, :].transpose(-1, -2)).abs().mean() / math.sqrt(hd))
        y_raw = raw(x)
        rec = g.install_gains(x)
        y_inst = g(x)
        rho = q.pow(2).mean(-1).sqrt()
        med = rho.transpose(0, 1).reshape(4, -1).median(dim=-1).values.unsqueeze(-1)
        gq = med.expand(-1, 16)
    check("3 gains == the typical (median) position scale of q, one scalar per head",
          torch.allclose(g.q_gain.squeeze(1), gq, atol=1e-5) and float(gq.mean()) > 2.0, f"mean gain {float(gq.mean()):.3f}")
    r_inst, r_ones = rec["logit_scale_after"] / rec["logit_scale_before"], s_ones / s_raw
    check("3 the install keeps the logit scale (within 10%) where gains 1 do not",
          0.9 < r_inst < 1.1 and 0.9 < rec["logit_ratio_median"] < 1.1 and r_ones < 0.5,
          f"after/before: installed mean {r_inst:.3f}, typical entry {rec['logit_ratio_median']:.3f}, gains 1 {r_ones:.3f}")
    xs = x.clone()
    xs[:, 0] *= 30.0                                 # a sink: one position 30x the rest
    g2 = CausalSDPA(64, 4, qk_norm="rms")
    _same_weights(raw, g2)
    with torch.no_grad():
        rec2 = g2.install_gains(xs)
    check("3 under a sink the typical entry keeps its scale while the mean |logit| falls (the sink normalized down)",
          0.9 < rec2["logit_ratio_median"] < 1.1 and rec2["logit_scale_after"] / rec2["logit_scale_before"] < 0.9,
          f"typical {rec2['logit_ratio_median']:.3f}, mean {rec2['logit_scale_after'] / rec2['logit_scale_before']:.3f}")
    check("3 the raw scale on record matches an independent read", abs(rec["logit_scale_before"] - s_raw) < 1e-5)
    check("3 the installed block runs finite and differs from the raw block only mildly",
          bool(torch.isfinite(y_inst).all()) and float((y_inst - y_raw).abs().mean()) < 0.5 * float(y_raw.abs().mean()))


def case_4():
    check("4 autocast: cpu -> null", isinstance(autocast("cpu"), contextlib.nullcontext))
    check("4 autocast: cuda + fp32 -> null", isinstance(autocast("cuda", "fp32"), contextlib.nullcontext))
    bad = False
    try:
        autocast("cuda", "fp16")
    except ValueError:
        bad = True
    check("4 autocast refuses an unknown precision", bad)
    bad = False
    try:
        TrainConfig(precision="fp16")
    except ValueError:
        bad = True
    check("4 TrainConfig refuses an unknown precision; the default is bf16", bad and TrainConfig().precision == "bf16")


def case_5():
    p = make_control_resume_preset("x-cursor-test")
    tps = p.train.micro_batch * p.train.grad_accum * p.model.context
    check("5 tokens a step", tps == 262_144, f"{tps:,}")
    bounds, start = [], 0
    for ph in p.curriculum:
        start += math.ceil(ph["planned_tokens"] / tps)
        bounds.append(start)
    want = [1145, 20219, 22890, 25561, 29376, 33954, 38532, 42347, 45399, 50740, 53792, 57607, 61422]
    check("5 every boundary of the 2s twins' schedule", bounds == want, f"{bounds}")
    cur = cursor_at_step(p.curriculum, 16000, tps)
    st = {c["name"]: (c["status"], c["tokens_done"]) for c in cur}
    check("5 the step-16,000 cursor",
          st["warmup_wikitext"] == ("done", 300_154_880) and st["fineweb_main"] == ("active", 3_894_149_120)
          and st["curriculum_s0"] == ("planned", 0) and sum(c["tokens_done"] for c in cur) == 4_194_304_000)
    m = RunManifest.fresh("x", {}, p.curriculum)
    act = m.set_cursor(16000, tps)
    check("5 set_cursor: steps, tokens_seen, the active phase",
          m.steps == 16000 and m.tokens_seen == 4_194_304_000 and act["name"] == "fineweb_main")
    bad = False
    try:
        cursor_at_step(p.curriculum, 61422, tps)
    except ValueError:
        bad = True
    last = cursor_at_step(p.curriculum, 61421, tps)
    check("5 the end of the plan", bad and last[-1]["status"] == "active" and last[-1]["name"] == "anneal_mix")
    c0 = cursor_at_step(p.curriculum, 0, tps)
    check("5 step 0 = the plan's start", c0[0]["status"] == "active" and c0[0]["tokens_done"] == 0)
    c20219 = cursor_at_step(p.curriculum, 20219, tps)
    check("5 the fineweb boundary step opens s0", c20219[1]["status"] == "done" and c20219[2]["status"] == "active"
          and c20219[2]["tokens_done"] == 0)


def case_6():
    check("6 the nine arms registered", len(CONTROL_RESUME_ARMS) == 9 and all(n in PRESETS for n in CONTROL_RESUME_ARMS),
          ", ".join(CONTROL_RESUME_ARMS))
    fx, f32 = PRESETS["mini-beatrix-2s-control-fix"], PRESETS["mini-beatrix-2s-control-fp32"]
    b16, ff = PRESETS["mini-beatrix-2s-control-bf16"], PRESETS["mini-beatrix-2s-control-fp32-fix"]
    at = PRESETS["mini-beatrix-2s-control-attn"]
    base = PRESETS["mini-beatrix-2s-control"]
    check("6 the switches per arm",
          fx.model.qk_norm == "rms" and fx.model.attn_fp32 and fx.train.precision == "bf16"
          and f32.train.precision == "fp32" and not f32.model.qk_norm and not f32.model.attn_fp32
          and b16.train.precision == "bf16" and not b16.model.qk_norm and not b16.model.attn_fp32
          and at.train.precision == "bf16" and not at.model.qk_norm and at.model.attn_fp32
          and ff.train.precision == "fp32" and ff.model.qk_norm == "rms")
    same = all(getattr(fx.model, k) == getattr(base.model, k) for k in
               ("d_model", "n_layers", "n_heads", "context", "hub_layers", "bank_experts", "bank_ff", "head_K", "head_D",
                "tokenizer", "vocab_size")) and fx.model.hub_layers == ()
    check("6 the twin's craft otherwise verbatim (hub layers empty)", same)
    recipe = all(getattr(fx.train, k) == getattr(base.train, k) for k in
                 ("muon_lr", "adam_lr", "muon_momentum", "warmup_steps", "grad_clip",
                  "seed", "head_addr_frozen", "governor", "governor_theta", "ckpt_every", "phase_seed_offset"))
    check("6 the twin's recipe verbatim (born-null head unfrozen)", recipe and fx.train.head_addr_frozen is False)
    steps = {n: PRESETS[n].train.micro_batch * PRESETS[n].train.grad_accum * PRESETS[n].model.context for n in CONTROL_RESUME_ARMS}
    mbs = {n: (PRESETS[n].train.micro_batch, PRESETS[n].train.grad_accum) for n in CONTROL_RESUME_ARMS}
    check("6 every arm trains the recipe's 262,144-token step at its own micro-batch",
          all(v == 262_144 for v in steps.values()) and mbs["mini-beatrix-2s-control-bf16"] == (16, 4)
          and mbs["mini-beatrix-2s-control-attn"] == (8, 8) and mbs["mini-beatrix-2s-control-fix"] == (8, 8)
          and mbs["mini-beatrix-2s-control-fp32"] == (4, 16) and mbs["mini-beatrix-2s-control-fp32-fix"] == (4, 16), f"{mbs}")
    bad = False
    try:
        make_control_resume_preset("x-bad-step", micro_batch=8)      # 8 x 4 x 4096 != 262,144
    except AssertionError:
        bad = True
    check("6 a micro-batch that breaks the step is refused", bad)
    names = [ph["name"] for ph in fx.curriculum]
    check("6 chronological phases, all planned",
          names[:2] == ["warmup_wikitext", "fineweb_main"] and names[2] == "curriculum_s0" and names[10] == "curriculum_s8"
          and names[-2:] == ["anneal_nochat", "anneal_mix"] and "fineweb_extended" not in names
          and all(ph["status"] == "planned" for ph in fx.curriculum) and len(names) == 13)
    check("6 init_from (the cursor at the recipe's 262,144-token step)", fx.init_from["path"] == "checkpoints/step_00016000.safetensors"
          and fx.init_from["prefix"] == "mini-beatrix-2s-control" and fx.init_from["step"] == 16000
          and fx.init_from["seed_offset"] == 7919 and fx.init_from["repo"] == base.hf_repo
          and fx.init_from["tokens_per_step"] == 262_144)
    p14 = make_control_resume_preset("x-r14k", start_step=14000, precision="fp32", qk_norm="rms")
    check("6 a 14,000 build", p14.init_from["path"] == "checkpoints/step_00014000.safetensors"
          and p14.model.name == "x-r14k" and p14.train.precision == "fp32" and p14.model.qk_norm == "rms")
    check("6 no cross-mutation of the twin", base.model.qk_norm == "" and not base.model.attn_fp32
          and base.train.precision == "bf16" and not base.init_from)


def case_7():
    from ..train.trainer import Trainer
    from safetensors.torch import save_file
    tmp = tempfile.mkdtemp(prefix="alephllm_ctl_")
    torch.manual_seed(7)
    src = AlephLM(TINY)
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in src.state_dict().items()}
    f = os.path.join(tmp, "src_step16.safetensors")
    save_file(sd, f, metadata={"step": "16"})
    cfg = AlephLMConfig.from_dict(TINY.to_dict())
    cfg.name, cfg.qk_norm = "tiny-control-fix", "rms"
    tc = TrainConfig(micro_batch=2, grad_accum=1, warmup_steps=0, log_every=1, health_every=10**6, eval_every=10**6,
                     ckpt_every=10**6, tb_upload_every=10**6, val_tokens=256, canary_episodes=1)
    phases = [dict(name="p0", dataset="synthetic", planned_tokens=2_560, status="planned"),
              dict(name="p1", dataset="synthetic", planned_tokens=5_120, status="planned"),
              dict(name="p2", dataset="synthetic", planned_tokens=2_560, status="planned")]
    p = Preset(model=cfg, train=tc, curriculum=phases, init_from={"file": f, "step": 16, "seed_offset": 7919})
    tps = 2 * 128
    t = Trainer(p, hf_token=None, out_dir=tmp, device="cpu")
    st = {ph["name"]: (ph["status"], ph["tokens_done"]) for ph in t.manifest.phases}
    check("7 the cursor at step 16", t.step == 16 and st["p0"] == ("done", 2_560) and st["p1"] == ("active", 6 * tps)
          and st["p2"] == ("planned", 0) and t.manifest.tokens_seen == 16 * tps, f"{st}")
    msd = t.raw_model.state_dict()
    check("7 the source weights in the fp32 masters (bf16 upcast)",
          all(torch.equal(msd[k], v.to(torch.float32)) for k, v in sd.items()))
    gains = [q for n, q in t.raw_model.named_parameters() if n.endswith(("q_gain", "k_gain"))]
    init = t.manifest.init_from or {}
    check("7 the guards fresh + installed (a boundary write on record)",
          len(gains) == 6 and all(not torch.allclose(g, torch.ones_like(g)) for g in gains)
          and "qk_gains" in init and len(init["qk_gains"]["blocks"]) == 3 and init["phase"] == "p1"
          and any("BOUNDARY WRITE" in n["msg"] for n in t.manifest.notes))
    check("7 the restarted phase's seed", t._phase_seed({"name": "p1"}) == tc.seed + 7919
          and t._phase_seed({"name": "p2"}) == tc.seed)
    check("7 fresh optimizers", all(len(o.state) == 0 for o in t.optimizers))
    t.train(max_steps=2)
    st = {ph["name"]: (ph["status"], ph["tokens_done"]) for ph in t.manifest.phases}
    check("7 two steps trained", t.step == 18 and st["p1"] == ("active", 8 * tps) and t.manifest.tokens_seen == 18 * tps)
    latest = os.path.join(tmp, cfg.name, "resume", "latest.pt")
    check("7 the final checkpoint on disk", os.path.exists(latest)
          and os.path.exists(os.path.join(tmp, cfg.name, "checkpoints", "step_00000018.safetensors")))
    t2 = Trainer(p, hf_token=None, out_dir=tmp, device="cpu")
    g1, g2 = t.raw_model.state_dict()["blocks.0.attn.q_gain"], t2.raw_model.state_dict()["blocks.0.attn.q_gain"]
    check("7 the hand-off to the normal resume", t2.step == 18 and (t2.manifest.init_from or {}).get("step") == 16
          and t2._phase_seed({"name": "p1"}) == tc.seed + 7919 and torch.equal(g1, g2)
          and len(t2.optimizers[1].state) > 0)
    t2.train(max_steps=1)
    check("7 the resumed run steps on", t2.step == 19 and t2.manifest.tokens_seen == 19 * tps)
    cfg.attn_kernel = "flex"            # a kernel change on a resume: recorded, never refused (0.10.11)
    t3 = Trainer(p, hf_token=None, out_dir=tmp, device="cpu")
    check("7 a kernel change on resume is recorded (model config + note)",
          t3.manifest.model_config.get("attn_kernel") == "flex" and t3.step == 19
          and any("attention kernel AMENDED on resume at step 19" in n["msg"] for n in t3.manifest.notes))
    cfg.attn_kernel = "sdpa"


def case_8():
    if not torch.cuda.is_available():
        print("SKIP 8 attn_fp32 on a card (no CUDA here)", flush=True)
        return
    torch.manual_seed(8)
    a, b = CausalSDPA(64, 4).cuda(), CausalSDPA(64, 4, attn_fp32=True).cuda()
    _same_weights(a, b)
    x = torch.randn(2, 64, 64, device="cuda")
    with torch.no_grad():
        ref = a(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_bf16, y_fp32attn = a(x), b(x)
    d16, d32 = float((y_bf16.float() - ref).abs().max()), float((y_fp32attn - ref).abs().max())
    check("8 attn_fp32: fp32 output under autocast; the plain path bf16",
          y_fp32attn.dtype == torch.float32 and y_bf16.dtype == torch.bfloat16)
    check("8 fp32 attention closer to the fp32 pass than bf16", d32 < d16, f"{d32:.2e} vs {d16:.2e}")


def case_9():
    from ..model import attention as A
    torch.manual_seed(9)
    a, b = CausalSDPA(64, 4), CausalSDPA(64, 4, attn_kernel="flex")
    _same_weights(a, b)
    a.train(); b.train()
    x = torch.randn(2, 64, 64)
    check("9 the default kernel is sdpa; flex accepted", a.attn_kernel == "sdpa" and b.attn_kernel == "flex")
    refused = False
    try:
        CausalSDPA(64, 4, attn_kernel="flash")
    except AssertionError:
        refused = True
    check("9 an unknown kernel is refused", refused)
    check("9 off a card the flex block is the sdpa block (train mode)", torch.equal(a(x), b(x)))
    cfg = AlephLMConfig.from_dict({**TINY.to_dict(), "name": "tiny-flex", "attn_kernel": "flex"})
    kinds = {m.attn_kernel for m in AlephLM(cfg).modules() if isinstance(m, CausalSDPA)}
    kinds0 = {m.attn_kernel for m in AlephLM(TINY).modules() if isinstance(m, CausalSDPA)}
    check("9 the config's kernel reaches every sdpa block; the default builds sdpa", kinds == {"flex"} and kinds0 == {"sdpa"})
    check("9 the kernel survives the dict round trip", AlephLMConfig.from_dict(cfg.to_dict()).attn_kernel == "flex")
    arms = {k: make_control_resume_preset(k, **v).model.attn_kernel for k, v in CONTROL_RESUME_ARMS.items()}
    want = {"mini-beatrix-2s-control-bf16": "sdpa", "mini-beatrix-2s-control-fp32": "sdpa",
            "mini-beatrix-2s-control-attn": "sdpa", "mini-beatrix-2s-control-fix": "sdpa",
            "mini-beatrix-2s-control-fp32-fix": "sdpa", "mini-beatrix-2s-control-attn-fp16": "fp16",
            "mini-beatrix-2s-control-fix-fp16": "fp16", "mini-beatrix-2s-control-bf16-qk": "sdpa",
            "mini-beatrix-2s-control-fix-fp16-c10k": "fp16"}
    check("9 the fp16 arms carry fp16 flash, every other arm sdpa", arms == want, str(arms))
    q = torch.randn(1, 4, 32, 16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    saved = A._FLEX["dead"]
    A._FLEX["dead"] = "forced by case 9"
    try:
        y_dead = A._flex(q, k, v)
    finally:
        A._FLEX["dead"] = saved
    check("9 a dead kernel falls back to sdpa", torch.equal(y_dead, torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)))
    if not torch.cuda.is_available():
        print("SKIP 9 the fused kernel on a card (no CUDA here)", flush=True)
        return
    ref, flx = CausalSDPA(64, 4, attn_fp32=True).cuda(), CausalSDPA(64, 4, attn_fp32=True, attn_kernel="flex").cuda()
    _same_weights(ref, flx)
    ref.train(); flx.train()
    xr = torch.randn(2, 256, 64, device="cuda", requires_grad=True)
    xf = xr.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y1, y2 = ref(xr), flx(xf)
    if A._FLEX["dead"]:
        print(f"SKIP 9 the fused kernel on a card (it would not build: {A._FLEX['dead']})", flush=True)
        return
    y1.square().mean().backward(); y2.square().mean().backward()
    dy = float((y1 - y2).abs().max() / y1.abs().max())
    dg = float((ref.qkv.weight.grad - flx.qkv.weight.grad).abs().max() / ref.qkv.weight.grad.abs().max())
    prec = "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"
    check(f"9 fused kernel forward within 1e-2 of fp32 sdpa ({prec} products in this process)", dy < 1e-2, f"{dy:.2e}")
    check(f"9 fused kernel qkv gradient within 2e-2 of fp32 sdpa ({prec} products)", dg < 2e-2, f"{dg:.2e}")
    ref.eval(); flx.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        check("9 the eval path stays on sdpa bit for bit", torch.equal(ref(xr), flx(xr)))


def case_10():
    from ..model import attention as A
    torch.manual_seed(10)
    a, b = CausalSDPA(64, 4), CausalSDPA(64, 4, attn_kernel="fp16")
    _same_weights(a, b)
    a.train(); b.train()
    x = torch.randn(2, 64, 64)
    check("10 fp16 accepted; off a card the fp16 block is the sdpa block", b.attn_kernel == "fp16" and torch.equal(a(x), b(x)))
    cfg = AlephLMConfig.from_dict({**TINY.to_dict(), "name": "tiny-fp16", "attn_fp32": True, "attn_kernel": "fp16"})
    m = AlephLM(cfg)
    lab = A.attention_kernel_label(m, "bf16")
    check("10 the trainer's kernel label names fp16 flash", lab is not None and lab.startswith("fp16 flash"), str(lab))
    check("10 the label for the plain twin names bf16 flash",
          A.attention_kernel_label(AlephLM(TINY), "bf16").startswith("bf16 sdpa under autocast"))
    if not torch.cuda.is_available():
        print("SKIP 10 fp16 flash on a card (no CUDA here)", flush=True)
        return
    ref, f16 = CausalSDPA(64, 4, attn_fp32=True).cuda(), CausalSDPA(64, 4, attn_fp32=True, attn_kernel="fp16").cuda()
    _same_weights(ref, f16)
    ref.train(); f16.train()

    def grads(blk, xin, loss_scale=1.0):
        xx = xin.detach().clone().requires_grad_(True)
        blk.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(xx)
        (y.square().mean() * loss_scale).backward()
        return y.detach(), xx.grad.detach().clone(), blk.qkv.weight.grad.detach().clone()

    def rel(u, v):
        return float((u - v).abs().max() / v.abs().max().clamp_min(1e-30))

    xr = torch.randn(2, 256, 64, device="cuda")
    y1, gx1, gw1 = grads(ref, xr)
    y2, gx2, gw2 = grads(f16, xr)
    check("10 fp16 flash forward within 1e-2 of fp32 sdpa", rel(y2, y1) < 1e-2, f"{rel(y2, y1):.2e}")
    check("10 fp16 flash gradients within 1e-2 of fp32 sdpa (input, qkv)", rel(gx2, gx1) < 1e-2 and rel(gw2, gw1) < 1e-2,
          f"{rel(gx2, gx1):.2e} {rel(gw2, gw1):.2e}")
    _, gx1s, gw1s = grads(ref, xr, 1e-6)
    _, gx2s, gw2s = grads(f16, xr, 1e-6)
    check("10 a loss scaled by 1e-6 (gradients under fp16's floor) still matches fp32 within 1e-2",
          rel(gx2s, gx1s) < 1e-2 and rel(gw2s, gw1s) < 1e-2 and float(gx2s.abs().max()) > 0,
          f"{rel(gx2s, gx1s):.2e} {rel(gw2s, gw1s):.2e} max {float(gx2s.abs().max()):.2e}")
    # a sink: the first position's key aligned with every query (the first run's twin relies on such keys); every query's
    # gradient then lands on one key's v — the overflow case for an unscaled fp16 backward
    with torch.no_grad():
        for blk in (ref, f16):
            blk.qkv.weight.zero_()
            blk.qkv.weight[:64, :64] = torch.eye(64, device="cuda") * 4.0      # q = 4 x
            blk.qkv.weight[64:128, :64] = torch.eye(64, device="cuda") * 4.0   # k = 4 x
            blk.qkv.weight[128:, :64] = torch.eye(64, device="cuda")           # v = x
    xs = torch.randn(1, 256, 64, device="cuda") * 0.1
    xs[:, 0] = 3.0                                                              # one key every query agrees with
    y1, gx1, gw1 = grads(ref, xs)
    y2, gx2, gw2 = grads(f16, xs)
    check("10 the sink case: finite gradients within 2e-2 of fp32's",
          bool(torch.isfinite(gx2).all() & torch.isfinite(gw2).all()) and rel(gx2, gx1) < 2e-2 and rel(gw2, gw1) < 2e-2,
          f"{rel(gx2, gx1):.2e} {rel(gw2, gw1):.2e}")
    ref.eval(); f16.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        check("10 the eval path stays on sdpa bit for bit", torch.equal(ref(xs), f16(xs)))
    ref.train(); f16.train()
    # the counters. A true sink: every query carries a positive component along the sink key, so every query's gradient lands
    # on one key's v; with identical upstream rows (a sum loss) the unscaled dv reaches 256 x 2^10 > 65,504 -> a retry
    xt = torch.randn(1, 256, 64, device="cuda") * 0.1 + 0.5
    xt[:, 0] = 3.0
    st0 = dict(A._FP16)
    xx = xt.detach().clone().requires_grad_(True)
    f16.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f16(xx).sum().backward()
    st1 = dict(A._FP16)
    xr_ = xt.detach().clone().requires_grad_(True)
    ref.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref(xr_).sum().backward()
    check("10 an overflowing sink backward fires a retry; the gradient finite and within 2e-2 of fp32's",
          st1["retries"] > st0["retries"] and st1["calls"] == st0["calls"] + 1 and bool(torch.isfinite(xx.grad).all())
          and rel(xx.grad, xr_.grad) < 2e-2,
          f"retries +{st1['retries'] - st0['retries']}, dev {rel(xx.grad, xr_.grad):.2e} "
          f"(max |x-grad| fp16 {float(xx.grad.abs().max()):.3e}, fp32 {float(xr_.grad.abs().max()):.3e})")
    # a forward overflow: an input past fp16's range -> that call runs in fp32 sdpa, forward and backward
    xo = xt.detach().clone()
    xo[:, 5] = 3.0e4                                   # q = 4 x = 1.2e5 > 65,504
    y_o, gx_o, gw_o = grads(f16, xo)
    y_r, gx_r, gw_r = grads(ref, xo)
    st2 = dict(A._FP16)

    def same(a, b):
        # the fp32 path's numbers, NaN in the same places counted as equal: an input this far past fp16's range sends fp32
        # attention's own backward non-finite (the trainer's non-finite guard then refuses the step, as it should)
        return (bool(torch.equal(torch.isnan(a), torch.isnan(b)))
                and torch.allclose(torch.nan_to_num(a), torch.nan_to_num(b), atol=1e-4, rtol=1e-3))
    check("10 a forward overflow falls back to fp32 sdpa for that call (counted; output and gradients the fp32 path's, NaN-aware)",
          st2["fallbacks"] == st1["fallbacks"] + 1 and bool(torch.isfinite(y_o).all())
          and torch.allclose(y_o, y_r, atol=1e-4, rtol=1e-4) and same(gx_o, gx_r) and same(gw_o, gw_r),
          f"fallbacks +{st2['fallbacks'] - st1['fallbacks']}; the fp32 path's x-grad finite: {bool(torch.isfinite(gx_r).all())}")


def case_11():
    p = PRESETS["mini-beatrix-2s-control-fix-fp16-c10k"]
    base = PRESETS["mini-beatrix-2s-control"]
    check("11 a continuation arm takes another arm's prefix as its source (the twin's repo, its own seed offset)",
          p.init_from["prefix"] == "mini-beatrix-2s-control-attn-fp16-r10k" and p.init_from["repo"] == base.hf_repo
          and p.init_from["path"] == "checkpoints/step_00016000.safetensors" and p.init_from["seed_offset"] == 7927
          and p.model.qk_norm == "rms" and p.model.attn_kernel == "fp16" and p.model.attn_fp32 is True
          and p.train.micro_batch == 8 and p.train.grad_accum == 8, str(p.init_from))
    q = make_control_resume_preset("x-c14k", start_step=14000, **CONTROL_RESUME_ARMS["mini-beatrix-2s-control-fix-fp16-c10k"])
    check("11 the derived name at another step keeps the source prefix",
          q.init_from["path"] == "checkpoints/step_00014000.safetensors"
          and q.init_from["prefix"] == "mini-beatrix-2s-control-attn-fp16-r10k" and q.init_from["step"] == 14000)
    b = PRESETS["mini-beatrix-2s-control-bf16-qk"]
    check("11 the bf16 + QK-norm arm: bf16 autocast, plain flash, QK-norm, 16 x 4",
          b.train.precision == "bf16" and b.model.qk_norm == "rms" and b.model.attn_fp32 is False
          and b.model.attn_kernel == "sdpa" and b.train.micro_batch == 16 and b.train.grad_accum == 4)
    if not torch.cuda.is_available():
        print("SKIP 11 QK-norm under bf16 autocast on a card (no CUDA here)", flush=True)
        return
    blk = CausalSDPA(64, 4, qk_norm="rms").cuda()
    blk.train()
    x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = blk(x)
    y.float().square().mean().backward()
    check("11 QK-norm under bf16 autocast: a bf16 output, finite, gradients to the gains",
          y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all()) and blk.q_gain.grad is not None
          and bool(torch.isfinite(blk.q_gain.grad).all()) and bool(torch.isfinite(x.grad).all()))


def main():
    for fn in (case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8, case_9, case_10, case_11):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — a case's crash is a FAIL row, the rest still run
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised", False, repr(e))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
