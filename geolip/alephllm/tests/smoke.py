"""Full smoke + test array. Run: python -m geolip.alephllm.tests.smoke

Every major subsystem gets a mechanical correctness case (shapes,
finiteness, exactness of null paths, causality, resume roundtrips).
No accuracy claims are made here — accuracy verdicts belong to GPU runs.
"""
from __future__ import annotations

import math
import os
import tempfile
import traceback

import torch

from ..presets import AlephLMConfig, get_preset, PRESETS
from ..model.address import AlephAddress, dtype_floor
from ..model.attention import CausalSplatHUB
from ..model.bank import AnchoredBank
from ..model.alephlm import AlephLM
from ..data.tokenizers import ByteTrigramTokenizer
from ..data.streams import build_stream
from ..train.optim import build_optimizers, split_params, Muon
from ..train.manifest import RunManifest
from ..train import instruments
from ..eval.canaries import make_episodes, canary_eval

TINY = AlephLMConfig(name="tiny-test", d_model=64, n_layers=3, n_heads=4,
                     context=128, hub_layers=(1,), hub_K=32, hub_D=8,
                     head_K=32, head_D=8, hub_chunk=16)
RESULTS = []


def case(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


@case("address math: signed/oriented identities + fp16 floor")
def t_address():
    a = AlephAddress(16, 8)
    x = torch.randn(4, 10, 8)
    w = a.signed(x)
    p, n = a.oriented(x)
    assert torch.isfinite(w).all() and torch.isfinite(p).all()
    assert torch.allclose(w, p - n, atol=1e-6), "signed == ep/Z - en/Z"
    assert ((p + n).sum(-1) - 1).abs().max() < 1e-5, "2K softmax sums to 1"
    assert dtype_floor(torch.zeros(1, dtype=torch.float16)) > 0
    assert torch.tensor(dtype_floor(torch.zeros(1, dtype=torch.float16)),
                        dtype=torch.float16).item() > 0, "floor representable"


@case("bank C6 null path: dispatch contributes exactly zero at init")
def t_bank_null():
    b = AnchoredBank(32, 3)
    x = torch.randn(2, 8, 32)
    full, trunk = b(x), b(x, disable_dispatch=True)
    assert torch.equal(full, trunk), "zero-init expert out must be bit-exact"


@case("head born-null: gamma=0 -> exact standard head")
def t_head_null():
    m = AlephLM(TINY)
    x = torch.randint(0, 256, (2, 32))
    a, _ = m(x)
    b, _ = m(x, disable_head_aleph=True)
    assert torch.equal(a, b), "gamma=0 must be bit-exact to base head"


@case("hub chunked scan == naive cumsum oracle")
def t_hub_equiv():
    torch.manual_seed(0)
    for n in (16, 33, 128):
        hub = CausalSplatHUB(48, K=24, D=8, chunk=16)
        x = torch.randn(2, n, 48)
        y1, y2 = hub(x), hub.forward_naive(x)
        err = (y1 - y2).abs().max().item()
        assert err < 1e-4, f"chunk/naive mismatch {err:.2e} at n={n}"


@case("causality: future perturbation never touches the past")
def t_causal():
    torch.manual_seed(1)
    m = AlephLM(TINY)
    m.eval()
    x = torch.randint(0, 256, (1, 64))
    y = x.clone()
    y[0, 40:] = torch.randint(0, 256, (24,))
    with torch.no_grad():
        la, _ = m(x)
        lb, _ = m(y)
    d = (la[0, :39] - lb[0, :39]).abs().max().item()
    assert d < 1e-4, f"causality leak {d:.2e}"


@case("model forward/backward finite; grads reach every trainable param")
def t_fwd_bwd():
    m = AlephLM(TINY)
    x = torch.randint(0, 256, (2, 64))
    _, loss = m(x[:, :-1], targets=x[:, 1:])
    assert math.isfinite(loss.item())
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    # gamma=0 blocks w_s/addr grads? No: gamma itself gets grad through the
    # product; w_s/addr get grad scaled by gamma=0 -> zero grads exist but
    # are not None. Only truly disconnected params may appear here.
    assert not missing, f"no grad for: {missing}"


@case("optimizer split covers every param exactly once; steps change both")
def t_optim():
    m = AlephLM(TINY)
    muon_p, adam_p = split_params(m)
    ids = [id(p) for p in muon_p + adam_p]
    assert len(ids) == len(set(ids)), "param counted twice"
    uniq = {id(p) for p in m.parameters()}
    assert set(ids) == uniq, "split misses params"
    emb_w = m.embed.emb0.weight if hasattr(m.embed, "emb0") else m.embed.emb.weight
    assert id(emb_w) not in {id(p) for p in muon_p}, "embeddings must be Adam"
    assert any(p.ndim == 3 for p in muon_p), "stacked experts ride Muon"
    opts = build_optimizers(m)
    x = torch.randint(0, 256, (2, 33))
    _, loss = m(x[:, :-1], targets=x[:, 1:])
    loss.backward()
    before = [p.detach().clone() for p in m.parameters()]
    for o in opts:
        o.step()
    changed = sum((not torch.equal(b, p.detach()))
                  for b, p in zip(before, m.parameters()))
    assert changed > 10, "optimizer steps changed almost nothing"


@case("muon state survives a state_dict roundtrip")
def t_muon_state():
    m = AlephLM(TINY)
    opts = build_optimizers(m)
    x = torch.randint(0, 256, (2, 33))
    _, loss = m(x[:, :-1], targets=x[:, 1:])
    loss.backward()
    for o in opts:
        o.step()
    st = opts[0].state_dict()
    opts2 = build_optimizers(m)
    opts2[0].load_state_dict(st)
    k0 = list(opts[0].state.values())[0]["m"]
    k1 = list(opts2[0].state.values())[0]["m"]
    assert torch.equal(k0, k1), "momentum buffer lost"


@case("checkpoint roundtrip: resume.pt restores exact logits + opt state")
def t_ckpt_roundtrip():
    from ..train.checkpoint import HubSync, load_fp8_state
    from safetensors.torch import load_file
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        hub = HubSync("none/none", "tiny", token=None, local_dir=td)
        m = AlephLM(TINY)
        opts = build_optimizers(m)
        x = torch.randint(0, 256, (2, 65))
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        loss.backward()
        for o in opts:
            o.step()
        man = RunManifest.fresh("tiny-test", TINY.to_dict(), [])
        hub.save_resume(m, opts, {"dataset": "synthetic",
                                  "state": {"epoch": 0, "rows_consumed": 5}},
                        man, step=1)
        with torch.no_grad():
            ref, _ = m(x[:, :-1])
        m2 = AlephLM(TINY)
        opts2 = build_optimizers(m2)
        payload = torch.load(os.path.join(td, "resume", "latest.pt"),
                             map_location="cpu", weights_only=False)
        m2.load_state_dict(payload["model"])
        for o, st in zip(opts2, payload["optimizers"]):
            o.load_state_dict(st)
        with torch.no_grad():
            got, _ = m2(x[:, :-1])
        assert torch.equal(ref, got), "resume logits differ"
        # safetensors bf16 + fp8 exports load and run
        st_name = hub.save_safetensors(m, step=1)
        sd = load_file(os.path.join(td, st_name))
        assert all(v.dtype == torch.bfloat16 for v in sd.values())
        fp8_name = hub.save_fp8(m, step=1)
        sd8 = load_fp8_state(os.path.join(td, fp8_name))
        m3 = AlephLM(TINY)
        m3.load_state_dict({k: v.float() for k, v in sd8.items()})
        with torch.no_grad():
            l8, _ = m3(x[:, :-1])
        assert torch.isfinite(l8).all(), "fp8 rehydrated forward not finite"


@case("stream packing + stateful resume (synthetic)")
def t_stream():
    tok = ByteTrigramTokenizer()
    s = build_stream("synthetic", tok, context=64, micro_batch=4, seed=3)
    b1 = s.next_batch()
    assert b1.shape == (4, 65) and b1.dtype == torch.int64
    assert int(b1.max()) < 256 and int(b1.min()) >= 0
    st = s.state_dict()
    b2 = s.next_batch()
    s2 = build_stream("synthetic", tok, context=64, micro_batch=4, seed=3)
    s2.load_state_dict(st)
    b3 = s2.next_batch()
    assert torch.equal(b2, b3), "stream resume diverged"


@case("manifest: phase accounting + advance + json roundtrip")
def t_manifest():
    cur = [dict(name="a", dataset="synthetic", planned_tokens=100,
                status="planned"),
           dict(name="b", dataset="synthetic", planned_tokens=200,
                status="planned")]
    m = RunManifest.fresh("tiny-test", {}, cur)
    ph = m.advance_phases()
    assert ph["name"] == "a" and ph["status"] == "active"
    m.add_tokens(150)
    ph = m.advance_phases()
    assert ph["name"] == "b", "phase should advance when budget met"
    assert m.tokens_seen == 150
    m2 = RunManifest.from_json(m.to_json())
    assert m2.phases[0]["status"] == "done"


@case("canaries: episodes well-formed, in-window, keys distinct")
def t_canary_gen():
    toks, apos, ans, allv = make_episodes(32, 256, pairs=4, seed=1)
    assert int(toks.max()) < 256
    L = toks.shape[1]
    assert int(apos.max()) < L, "answer must sit inside the episode"
    got = torch.gather(toks, 1, apos)
    assert torch.equal(got, ans), "answer tokens must match values"
    m = AlephLM(TINY)
    r = canary_eval(m, None, "cpu", episodes=16, context=TINY.context)
    assert 0.0 <= r["recall_acc"] <= 1.0
    s = r["wrongpair_frac"] + r["noretrieval_frac"]
    assert s <= 1.0 - r["recall_acc"] + 1e-6


@case("instruments: erank sane, census + readout render, flags fire")
def t_instruments():
    er = instruments.effective_rank(torch.randn(256, 5) @ torch.randn(5, 64))
    assert er < 8, f"rank-5 matrix read as erank {er:.1f}"
    er_full = instruments.effective_rank(torch.randn(512, 64))
    assert er_full > 40
    m = AlephLM(TINY)
    x = torch.randint(0, 256, (2, 64))
    census = instruments.model_census(m, x)
    assert set(census["layers"]) == {0, 1, 2}
    assert "hub_consumed_erank" in census["layers"][1]
    txt = instruments.readout(0, 0, census, None)
    assert "collapse flags" in txt
    led = instruments.toggle_ledger(m, [torch.randint(0, 256, (2, 65))])
    assert abs(led["toggle_bank_off"]) < 1e-6, \
        "bank toggle must be ~0 at birth (null path)"
    assert abs(led["toggle_head_aleph_off"]) < 1e-6


@case("presets: all constructible, param counts in expected bands")
def t_presets():
    bands = {"mini-beatrix-0": (30, 55), "mini-beatrix-1": (95, 155),
             "mini-beatrix-2": (210, 330)}
    for name, (lo, hi) in bands.items():
        p = get_preset(name)
        n = AlephLM(p.model).param_count() / 1e6
        assert lo < n < hi, f"{name}: {n:.1f}M outside [{lo},{hi}]M"
    assert get_preset("mini-beatrix-1-control").model.hub_layers == ()
    assert "beatrix-voyager" in PRESETS


@case("mini train loop: 8 steps on synthetic, finite, resume-safe exit")
def t_train_loop():
    from ..train.trainer import Trainer
    from ..presets import Preset, TrainConfig
    with tempfile.TemporaryDirectory() as td:
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=4, grad_accum=1,
                                     warmup_steps=2, log_every=4,
                                     health_every=100, eval_every=100,
                                     ckpt_every=6, tb_upload_every=100,
                                     val_tokens=256, canary_episodes=8),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000,
                                    status="planned")])
        t = Trainer(p, hf_token=None, out_dir=td, resume=False)
        t.train(max_steps=8)
        assert t.step == 8
        assert t.manifest.tokens_seen == 8 * 4 * TINY.context
        assert any(c["kind"] == "resume" for c in t.manifest.checkpoints)
        t2 = Trainer(p, hf_token=None, out_dir=td, resume=True)
        assert t2.step == 8, "local resume should restore the step count"


@case("crash safety: divergence NEVER overwrites the resume checkpoint")
def t_crash_no_clobber():
    from ..train.trainer import Trainer
    from ..presets import Preset, TrainConfig
    import hashlib
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=4, grad_accum=1,
                                     warmup_steps=0, log_every=4,
                                     health_every=1000, eval_every=1000,
                                     ckpt_every=4, tb_upload_every=1000,
                                     val_tokens=256, canary_episodes=8),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000,
                                    status="planned")])
        t = Trainer(p, hf_token=None, out_dir=td, resume=False)
        t.train(max_steps=4)                       # writes a good resume ckpt
        rp = os.path.join(t.out_dir, "resume", "latest.pt")
        good = hashlib.sha256(open(rp, "rb").read()).hexdigest()
        with torch.no_grad():                      # poison -> NaN loss next step
            t.raw_model.head.gamma.fill_(float("nan"))
        try:
            t.train(max_steps=4)
            raise AssertionError("poisoned run should raise FloatingPointError")
        except FloatingPointError:
            pass
        now = hashlib.sha256(open(rp, "rb").read()).hexdigest()
        assert now == good, "crash overwrote the good resume checkpoint"


@case("refuses fresh start over an existing run (manifest w/o resume state)")
def t_refuse_clobber():
    from ..train.trainer import Trainer
    from ..presets import Preset, TrainConfig
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=4, grad_accum=1,
                                     val_tokens=256, canary_episodes=8),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000,
                                    status="planned")])
        man = RunManifest.fresh(TINY.name, TINY.to_dict(), p.curriculum)
        man.steps = 4000                           # run exists; no resume file
        mdir = os.path.join(td, TINY.name)
        os.makedirs(mdir, exist_ok=True)
        open(os.path.join(mdir, "manifest.json"), "w").write(man.to_json())
        try:
            Trainer(p, hf_token=None, out_dir=td, resume=True)
            raise AssertionError("should refuse to start over an existing run")
        except RuntimeError as e:
            assert "refusing" in str(e)


def main():
    passed = failed = 0
    for name, fn in RESULTS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\nsmoke: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
