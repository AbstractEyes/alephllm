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
import torch.nn as nn

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
    # mini-beatrix-2 rescaled 2026-08-26: FULL SPLAT (hub every block,
    # 16 constellations x 256 anchors @ D=256), ~849M measured at build.
    bands = {"mini-beatrix-0": (30, 55), "mini-beatrix-1": (95, 155),
             "mini-beatrix-2": (780, 920), "mini-beatrix-2s": (200, 280)}
    for name, (lo, hi) in bands.items():
        p = get_preset(name)
        n = AlephLM(p.model).param_count() / 1e6
        assert lo < n < hi, f"{name}: {n:.1f}M outside [{lo},{hi}]M"
    assert get_preset("mini-beatrix-1-control").model.hub_layers == ()
    assert "beatrix-voyager" in PRESETS
    assert get_preset("mini-beatrix-2").model.hub_const == 16
    assert get_preset("mini-beatrix-2").train.governor == "minsep"


    ctl = get_preset("mini-beatrix-2s-control")
    trt = get_preset("mini-beatrix-2s")
    assert ctl.train is not trt.train, "control shares treatment TrainConfig"
    assert ctl.train.head_addr_frozen is False and         trt.train.head_addr_frozen is True
    assert ctl.model.hub_layers == ()


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


@case("kv-cache decode == full forward (byte trigram + hub + sdpa)")
def t_kv_cache():
    torch.manual_seed(5)
    m = AlephLM(TINY)
    m.eval()
    seq = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        full, _ = m(seq)                       # oracle: one parallel pass
        logits, cache = m.prefill(seq[:, :34])
        err0 = (logits[:, -1] - full[:, 33]).abs().max().item()
        assert err0 < 2e-3, f"prefill logits mismatch {err0:.2e}"
        for j in range(34, 40):                # teacher-forced cached steps
            step_logits = m.decode_step(seq[:, j], cache)
            err = (step_logits[:, -1] - full[:, j]).abs().max().item()
            assert err < 2e-3, f"cached step {j} mismatch {err:.2e}"
    assert cache["t"] == 40
    # single-byte prompt: trigram history must fall back to the pad row
    lg1, c1 = m.prefill(seq[:, :1])
    with torch.no_grad():
        f1, _ = m(seq[:, :1])
    assert (lg1[:, -1] - f1[:, 0]).abs().max().item() < 2e-3
    # cached vs uncached generate agree greedily
    with torch.no_grad():
        a = m.generate(seq[:, :8], max_new=12, temperature=0.0, use_cache=True)
        b = m.generate(seq[:, :8], max_new=12, temperature=0.0, use_cache=False)
    agree = (a == b).float().mean().item()
    assert agree > 0.95, f"greedy cached/uncached agreement only {agree:.2f}"
    # context boundary: prompts at/over the context still produce all
    # requested tokens (cached fill + sliding continuation), never 0/crash
    ctx = TINY.context
    with torch.no_grad():
        for plen in (ctx - 1, ctx, ctx + 8):
            p = torch.randint(0, 256, (1, plen))
            g = m.generate(p, max_new=10, temperature=0.0, use_cache=True)
            assert g.shape[1] == plen + 10, f"plen={plen}: got {g.shape[1]}"
        # exact-fill: cached path may run all the way to the position wall
        p = torch.randint(0, 256, (1, ctx - 4))
        g = m.generate(p, max_new=4, temperature=0.0, use_cache=True)
        assert g.shape[1] == ctx
    # top_p=0 must not produce a zero-mass row (CUDA-context poison guard)
    nxt = m._sample(torch.randn(3, 1, 256), temperature=1.0, top_p=0.0)
    assert nxt.shape == (3, 1) and torch.isfinite(nxt.float()).all()


@case("kv-cache decode == full forward (BPE tied-embedding craft)")
def t_kv_cache_bpe():
    torch.manual_seed(6)
    cfg = AlephLMConfig(name="tiny-bpe-test", d_model=64, n_layers=2,
                        n_heads=4, context=96, vocab_size=500,
                        tokenizer="hf:test", tie_embeddings=True,
                        hub_layers=(1,), hub_K=32, hub_D=8,
                        head_K=32, head_D=8, hub_chunk=16)
    m = AlephLM(cfg)
    m.eval()
    seq = torch.randint(0, 500, (2, 24))
    with torch.no_grad():
        full, _ = m(seq)
        logits, cache = m.prefill(seq[:, :20])
        assert (logits[:, -1] - full[:, 19]).abs().max().item() < 2e-3
        for j in range(20, 24):
            sl = m.decode_step(seq[:, j], cache)
            assert (sl[:, -1] - full[:, j]).abs().max().item() < 2e-3


@case("forward is HF-duck-typed: input_ids/labels aliases + .logits/.loss")
def t_hf_compat():
    m = AlephLM(TINY)
    x = torch.randint(0, 256, (2, 33))
    out = m(input_ids=x, labels=x,
            attention_mask=torch.ones_like(x))
    assert hasattr(out, "logits") and hasattr(out, "loss")
    assert out.loss is not None and math.isfinite(out.loss.item())
    logits, loss = m(x[:, :-1], targets=x[:, 1:])   # tuple unpack unchanged
    # HF labels (same-position, shifted internally) must equal our
    # pre-shifted convention exactly
    assert abs(loss.item() - out.loss.item()) < 1e-4, "label-shift semantics"
    assert torch.allclose(logits, out.logits[:, :-1], atol=1e-4)


@case("census + toggles survive frozen-trunk adapter wrappers")
def t_census_wrapped():
    class _Wrap(torch.nn.Module):          # mirrors amoe BlockWithAdapter
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.enabled = True
            self.adapter = torch.nn.Identity()

        def forward(self, *a, **k):
            return self.adapter(self.block(*a, **k))

        def prefill(self, *a, **k):
            out, cache = self.block.prefill(*a, **k)
            return self.adapter(out), cache

        def step(self, *a, **k):
            return self.adapter(self.block.step(*a, **k))

    m = AlephLM(TINY)
    m.blocks = torch.nn.ModuleList(_Wrap(b) for b in m.blocks)
    x = torch.randint(0, 256, (2, 64))
    census = instruments.model_census(m, x)
    assert "hub_consumed_erank" in census["layers"][1]
    led = instruments.toggle_ledger(m, [torch.randint(0, 256, (2, 65))])
    assert "toggle_hub_off" in led
    with torch.no_grad():
        a = m.generate(x[:, :8], max_new=6, temperature=0.0, use_cache=True)
        b = m.generate(x[:, :8], max_new=6, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


@case("amoe bridge: chat rows byte-exact prefix, loss lands on replies")
def t_bridge_rows():
    from ..amoe_bridge import render_chat_rows, ASSISTANT_TAG
    conv = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am Beatrix."}]
    rows = render_chat_rows([conv], context=512)
    assert len(rows) == 2, "one row per assistant turn"
    for r, reply in zip(rows, ["Hello!", "I am Beatrix."]):
        ids, n = r["ids"], r["n_prefix"]
        assert bytes(ids[:n]).decode().endswith(ASSISTANT_TAG)
        assert bytes(ids[n:]).decode() == " " + reply + "\n"
    assert not render_chat_rows([conv], context=10), "over-budget dropped"


@case("anneal mix: generators well-formed, MixStream resumes batch-exact")
def t_anneal_mix():
    from ..data.streams import (MixStream, _beatrix_texture_rows,
                                _recall_rows, _render_soda)
    r = next(_beatrix_texture_rows(0))["text"]
    assert "Beatrix:" in r and "User:" in r
    rr = next(_recall_rows(0))["text"]
    assert "carries" in rr and "What does" in rr
    s = _render_soda({"narrative": "Two friends met.",
                      "speakers": ["Ana", "Ben"],
                      "dialogue": ["Hi Ben.", "Hi Ana."]})
    assert "Ana: Hi Ben." in s and s.startswith("Two friends met.")
    tok = ByteTrigramTokenizer()
    recipe = [("beatrix-texture", 0.5), ("recall-synth", 0.3),
              ("synthetic", 0.2)]
    m1 = MixStream(tok, context=64, micro_batch=2, seed=9, recipe=recipe)
    a = [m1.next_batch() for _ in range(6)]
    st = m1.state_dict()
    b = [m1.next_batch() for _ in range(6)]
    m2 = MixStream(tok, context=64, micro_batch=2, seed=9, recipe=recipe)
    m2.load_state_dict(st)
    c = [m2.next_batch() for _ in range(6)]
    for x, y in zip(b, c):
        assert torch.equal(x, y), "mix resume diverged"
    assert m1.rows_consumed > 0


@case("render datasets keep their renderer's columns; breaker trips on empties")
def t_render_columns():
    from ..data.streams import REGISTRY, _render_soda, PackedStream
    # every render dataset must declare the columns its renderer reads,
    # and the renderer must produce non-empty text from EXACTLY those
    # columns (the anneal hang: pruning to one column starved the render)
    for name, spec in REGISTRY.items():
        if spec.get("render"):
            assert spec.get("columns"), f"{name}: render without columns"
    pruned = {"narrative": "Two friends met.", "speakers": ["Ana", "Ben"],
              "dialogue": ["Hi.", "Hello."]}
    keep = REGISTRY["soda-dialogue"]["columns"]
    assert set(pruned) == set(keep)
    assert _render_soda({k: pruned[k] for k in keep}).strip()
    # circuit breaker: a stream of empty rows must RAISE, never spin
    s = PackedStream("synthetic", ByteTrigramTokenizer(), context=32,
                     micro_batch=1, seed=0)
    s._it = iter([{"text": ""}] * 5000)
    s._ds = object()   # pretend open so _open() is not re-entered
    try:
        s._next_row_text()
        raise AssertionError("empty-row loop did not trip the breaker")
    except RuntimeError as e:
        assert "circuit breaker" in str(e)


@case("chat-sft plumbing: persona rows render, samples run KV-cached")
def t_chat_sft_plumbing():
    from ..chat_sft import persona_conversations, _samples, FIXED_PROMPTS
    from ..amoe_bridge import render_chat_rows
    convs = persona_conversations()
    assert len(convs) > 100
    rows = render_chat_rows(convs, context=TINY.context * 8)
    assert len(rows) > 100
    assert all(r["n_prefix"] < len(r["ids"]) for r in rows)
    m = AlephLM(TINY)
    s = _samples(m, "cpu", max_new=12)   # exercises prefill+decode_step
    assert set(s) == {c[0]["content"] for c in FIXED_PROMPTS}


@case("session caps are relative: max_tokens counts THIS call only")
def t_session_caps():
    from ..train.trainer import Trainer
    from ..presets import Preset, TrainConfig
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=4, grad_accum=1,
                                     warmup_steps=0, log_every=100,
                                     health_every=1000, eval_every=1000,
                                     ckpt_every=1000, tb_upload_every=1000,
                                     val_tokens=256, canary_episodes=8),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000_000,
                                    status="planned")])
        t = Trainer(p, hf_token=None, out_dir=td, resume=False)
        per_step = 4 * TINY.context
        t.train(max_tokens=3 * per_step)          # session 1: exactly 3 steps
        assert t.step == 3, f"expected 3 steps, got {t.step}"
        t.train(max_tokens=2 * per_step)          # session 2 must run AGAIN
        assert t.step == 5, ("second session's cap must count only its own "
                             f"tokens (lifetime-cap bug): step={t.step}")


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


@case("v2 hub: multi-constellation fused == naive; v1 layout untouched")
def t_hub_multiconst():
    import warnings
    from ..model.attention import CausalSplatHUB
    torch.manual_seed(3)
    h = CausalSplatHUB(48, K=16, D=16, chunk=16, n_const=3).eval()
    x = torch.randn(2, 50, 48)
    with torch.no_grad():
        d = (h.forward(x) - h.forward_naive(x)).abs().max()
        assert d < 1e-4, f"fused vs naive {float(d)}"
        out, cache = h.prefill(x)
        assert "consts" in cache and len(cache["consts"]) == 3
        y = h.step(x[:, :1], cache)
        assert y.shape == (2, 1, 48)
    h1 = CausalSplatHUB(48, K=16, D=8, chunk=16)      # v1 form
    ks = set(h1.state_dict())
    assert "addr.codebook" in ks and "q.weight" in ks, "v1 keys must survive"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CausalSplatHUB(48, K=64, D=8)
        assert any("supply" in str(x.message) for x in w), "K>2D must warn"


@case("governor: identity when slack (mission D), projects crowded books")
def t_governor():
    from ..model.governor import govern_model, minsep_project_
    from ..model.address import AlephAddress
    torch.manual_seed(4)
    # mission-scale geometry: K=64 @ D=128 is born slack at theta=45
    a = AlephAddress(64, 128)
    before = a.codebook.data.clone()
    assert minsep_project_(a.codebook, 45.0) == 0
    assert torch.equal(a.codebook.data, before), "slack must be zero-write"
    with torch.no_grad():
        a.codebook[1] = torch.nn.functional.normalize(
            a.codebook[0] * 0.999 + 1e-3 * torch.randn(128), dim=-1)
    hits = minsep_project_(a.codebook, 45.0)
    A = torch.nn.functional.normalize(a.codebook.data, dim=-1)
    G = (A @ A.T).abs()
    G.fill_diagonal_(0)
    import math
    assert hits > 0 and float(G.max()) <= math.cos(math.radians(45)) + 5e-3
    assert minsep_project_(a.codebook, 45.0) == 0, "idempotent"
    m = AlephLM(TINY)
    govern_model(m, 45.0)                             # runs wrapper-aware


@case("hub_ckpt: checkpointed training path is grad-exact vs plain")
def t_hub_ckpt():
    torch.manual_seed(6)
    cfg = AlephLMConfig(name="ck-test", d_model=64, n_layers=3, n_heads=4,
                        context=128, hub_layers=(0, 1, 2), hub_K=16, hub_D=16,
                        hub_const=2, head_K=32, head_D=16, hub_chunk=16,
                        hub_ckpt=2)
    m = AlephLM(cfg)
    m.train()
    x = torch.randint(0, 255, (2, 64))
    out = m(x, targets=x)
    out.loss.backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters()
          if p.grad is not None}
    m.zero_grad()
    for b in m.blocks:
        b.ckpt_mode = 0
    out2 = m(x, targets=x)
    out2.loss.backward()
    assert float((out.loss - out2.loss).abs()) < 1e-6
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.allclose(g1[n], p.grad, atol=1e-5), n



@case("special tokens: invalid-UTF-8 law, DOC packing, chat frame, BPE guard")
def t_special_tokens():
    import numpy as np
    from ..data import special_tokens as sp
    # registry invariants: 13 ids, all provably outside UTF-8
    assert sp.SPECIALS <= sp.INVALID_UTF8 and len(sp.SPECIALS) == 13
    tok = ByteTrigramTokenizer()
    sp.assert_unreachable(tok)          # the law, executed
    # DOC packing from birth: synthetic rows are short, so a batch spans
    # many documents -> DOC must appear, exactly at row boundaries
    s = build_stream("synthetic", tok, context=64, micro_batch=4, seed=5)
    bs = [s.next_batch() for _ in range(6)]
    b = bs[0]
    n_doc = sum(int((x == sp.DOC).sum()) for x in bs)
    assert n_doc >= 4, f"DOC missing from packed byte stream ({n_doc})"
    others = sp.SPECIALS - {sp.DOC}
    assert not any(int((x == t).sum()) for x in bs for t in others),         "non-DOC special leaked into a plain-text stream"
    # resume determinism still holds with DOC in the pack
    st = s.state_dict()
    b2 = s.next_batch()
    s2 = build_stream("synthetic", tok, context=64, micro_batch=4, seed=5)
    s2.load_state_dict(st)
    assert torch.equal(b2, s2.next_batch()), "DOC packing broke resume"
    # the specials-native chat frame: unforgeable + round-trip visible
    ids = sp.render_chat_ids(
        [{"role": "user", "content": "hi ÿ😀"},
         {"role": "model", "content": "hello"}])
    assert int((ids == sp.USER).sum()) == 1 and int((ids == sp.MODEL).sum()) == 1
    assert int((ids == sp.END).sum()) == 3 and int((ids == sp.SYS).sum()) == 1
    vis = sp.decode_visible(ids)
    for tag in ("⟦SYS⟧", "⟦USER⟧", "⟦MODEL⟧", "⟦END⟧"):
        assert tag in vis
    # ids-row generator packs, carries the frame, and appends DOC
    sb = build_stream("beatrix-texture-sp", tok, context=96, micro_batch=2,
                      seed=7)
    bbs = [sb.next_batch() for _ in range(4)]
    assert sum(int((x == sp.DOC).sum()) for x in bbs) >= 1
    assert sum(int((x == sp.END).sum()) for x in bbs) >= 2
    # BPE guard: no DOC on a non-256 vocab; ids-row datasets refuse outright
    class FakeBPE:
        vocab_size = 50257
        def encode(self, text):
            import numpy as _np
            return _np.frombuffer(text.encode("utf-8", errors="replace"),
                                  dtype=_np.uint8).astype(_np.int64)
    fb = build_stream("synthetic", FakeBPE(), context=64, micro_batch=2,
                      seed=5)
    assert fb._doc_id is None
    ns = build_stream("anneal-nochat", tok, context=64, micro_batch=2,
                      seed=5)
    assert ns.dataset == "anneal-nochat"
    assert "beatrix-texture-sp" not in ns._streams, list(ns._streams)
    assert "fineweb-edu" in ns._streams
    try:
        build_stream("beatrix-texture-sp", FakeBPE(), context=64,
                     micro_batch=2, seed=5)
        raise RuntimeError("ids-row dataset accepted a BPE tokenizer")
    except AssertionError:
        pass
    # the eval gauge runs and returns the three numbers
    from ..train.instruments import special_token_gauge
    cfg = AlephLMConfig(name="sp", d_model=64, n_layers=2, n_heads=2,
                        context=64, hub_layers=(1,), hub_K=16, hub_D=16,
                        head_K=16, head_D=16, hub_chunk=16)
    m = AlephLM(cfg)
    g = special_token_gauge(m, [x[:, :65] for x in bs])
    assert g and g["doc_count"] >= 1 and g["doc_bpb"] > 0 and g["reset_bpb"] > 0
    # audit hardening: strict roles, esc guards
    try:
        sp.render_chat_ids([{"role": "system", "content": "x"}])
        raise RuntimeError("unknown role accepted")
    except AssertionError:
        pass
    for bad in (0x00, 0xFC, 0xC0, True, 0.5):
        try:
            sp.esc(bad)
            raise RuntimeError(f"esc accepted {bad!r}")
        except AssertionError:
            pass


@case("head revival: liveness gauge, burial flag, the BOUNDARY-WRITE op")
def t_head_revival():
    import torch.nn.functional as Fn
    from ..train.revival import revive_head
    from ..train.instruments import model_census
    torch.manual_seed(3)
    cfg = AlephLMConfig(name="rev", d_model=64, n_layers=2, n_heads=2,
                        context=64, hub_layers=(1,), hub_K=16, hub_D=16,
                        head_K=16, head_D=16, hub_chunk=16)
    m = AlephLM(cfg).eval()
    x = torch.randint(0, 256, (4, 64))
    # bury the head the way the real crafts did (RIDER 12): burial needs
    # a RANK-COLLAPSED book (a full-rank book spans everything and
    # normalize rescues any residual) — collapse the book to one line,
    # then rotate proj into its orthogonal complement
    with torch.no_grad():
        v0 = Fn.normalize(torch.randn(16), dim=0)
        m.head.addr.codebook.copy_(
            Fn.normalize(v0.expand(16, 16) + 1e-3 * torch.randn(16, 16),
                         dim=-1))
        P = torch.eye(16) - v0.outer(v0)
        m.head.proj.weight.copy_(P @ m.head.proj.weight)
    c = model_census(m, x)
    assert c["head"]["liveness_ratio"] < 0.1 and c["flags"]["head_buried"],         c["head"]["liveness_ratio"]
    # collect head inputs + base logits, run the op
    hs = []
    hook = m.nf.register_forward_hook(
        lambda mod, i, o: hs.append(o.detach().float()
                                    .reshape(-1, o.shape[-1])))
    with torch.no_grad():
        out = m(x[:, :-1], disable_head_aleph=True)
    hook.remove()
    H = torch.cat(hs)
    B = out.logits.float().reshape(-1, 256)
    Y = x[:, 1:].reshape(-1)
    prov = revive_head(m, H, B, Y, theta_deg=45.0)
    assert prov["s_norm"] > 0.01 and prov["delta_bpb"] <= 1e-6, prov
    c2 = model_census(m, x)
    assert c2["head"]["liveness_ratio"] > 0.5 and         not c2["flags"]["head_buried"]
    # the freeze flag: requires_grad off, optimizer groups unchanged
    m.head.proj.weight.requires_grad_(False)
    m.head.addr.codebook.requires_grad_(False)
    muon_p, adam_p = split_params(m)
    assert any(p is m.head.proj.weight for p in muon_p)  # membership kept
    out2 = m(x[:, :-1], targets=x[:, 1:])
    out2.loss.backward()
    assert m.head.proj.weight.grad is None
    assert m.head.addr.codebook.grad is None
    assert m.head.w_s.weight.grad is not None

@case("relay: exact zero write at birth + EMA birth parity == patchwork")
def t_relay_birth():
    from ..model.relay import RelayPatchwork, RelayEMA, RelaySpec
    torch.manual_seed(5)
    sp = RelaySpec(n_slots=8, K=16, D=4, hidden=32)
    base = RelayPatchwork(48, sp)
    x = torch.randn(2, 20, 48)
    assert torch.equal(base(x), x), "zero head weight+bias must be exactly inert"
    ema = RelayEMA(48, sp)
    assert torch.equal(ema(x), x), "EMA form inherits the inert birth"
    # shared-weight parity with a LIVE head: widen a trained-looking patchwork
    with torch.no_grad():
        nn.init.orthogonal_(base.consume[-1].weight)
        base.consume[-1].bias.normal_()
        base.gate.fill_(0.5)
    ema2 = RelayEMA.from_patchwork(base)
    d = (ema2(x) - base(x)).abs().max().item()
    assert d < 1e-5, f"birth parity vs patchwork: {d} (fp reorder only)"


@case("relay: chunked closed-form EMA == naive recurrence")
def t_relay_ema_scan():
    from ..model.relay import ema_chunked
    torch.manual_seed(6)
    f = torch.randn(3, 70, 12)
    s0 = torch.randn(3, 12)
    for rho in (1 / 16, 1 / 64):
        F_c, s_c = ema_chunked(f, rho, s0, chunk=32)
        s = s0.clone()
        outs = []
        for t in range(70):
            s = (1 - rho) * s + rho * f[:, t]
            outs.append(s.clone())
        F_n = torch.stack(outs, dim=1)
        d = (F_c - F_n).abs().max().item()
        assert d < 1e-5, f"rho {rho}: chunked vs loop {d}"
        assert torch.allclose(s_c, F_n[:, -1], atol=1e-5)


@case("relay: one-step decode with carried state == full-sequence scan")
def t_relay_decode():
    from ..model.relay import RelayEMA, RelaySpec
    torch.manual_seed(7)
    ema = RelayEMA(48, RelaySpec(n_slots=8, K=16, D=4, hidden=32))
    with torch.no_grad():
        nn.init.orthogonal_(ema.consume[-1].weight)
        ema.consume[0].weight.normal_(std=0.05)
        ema.gate.fill_(0.5)
    x = torch.randn(2, 24, 48)
    y_full, _ = ema.run(x, None)
    y_pre, st = ema.run(x[:, :16], None)          # prefill
    steps = []
    for t in range(16, 24):                        # cached decode, 1 pos/call
        y_t, st = ema.run(x[:, t:t + 1], st)
        steps.append(y_t)
    y_inc = torch.cat([y_pre] + steps, dim=1)
    d = (y_full - y_inc).abs().max().item()
    assert d < 1e-5, f"incremental vs full: {d}"


@case("relay: composed read == direct sinh/cosh reconstruction")
def t_relay_mhat():
    from ..model.relay import RelayPatchwork, RelaySpec
    torch.manual_seed(8)
    rp = RelayPatchwork(48, RelaySpec(n_slots=8, K=16, D=4, hidden=32))
    x = torch.randn(2, 10, 48)
    got = rp.feats(x)
    import torch.nn.functional as Fn
    slots = rp.proj(x).view(2, 10, 8, 4)
    A = Fn.normalize(rp.addr.codebook, dim=-1)
    u = (Fn.normalize(slots, dim=-1) @ A.transpose(-1, -2)) / rp.addr.tau
    m = u.abs().amax(dim=-1, keepdim=True)
    ep, en = torch.exp(u - m), torch.exp(-u - m)
    ref = (((ep - en) @ A) / (ep + en).sum(-1, keepdim=True)).reshape(2, 10, -1)
    d = (got - ref).abs().max().item()
    assert d < 1e-6, f"signed@A vs m_hat closed form: {d}"


@case("v3 preset: mini-beatrix-3 builds; 64.4B chronological planned phases; twins isolated")
def t_v3_presets():
    from ..presets import make_v3_preset
    p = get_preset("mini-beatrix-3")
    assert p.model.n_layers == 24 and p.model.hub_layers == tuple(range(24))
    assert p.model.hub_K <= 2 * p.model.hub_D, "supply law"
    assert p.train.head_addr_frozen is False and p.train.compile is False
    names = [ph["name"] for ph in p.curriculum]
    assert names[:2] == ["warmup_wikitext", "fineweb_main"]
    assert names[-2:] == ["anneal_nochat", "anneal_mix"]
    assert names.index("curriculum_s8") < names.index("anneal_nochat")
    assert all(ph["status"] == "planned" for ph in p.curriculum)
    tot = sum(ph["planned_tokens"] for ph in p.curriculum)
    assert abs(tot - 64.4e9) < 1e6, tot
    assert p.data_scale == 4.0
    c = get_preset("mini-beatrix-3-control")
    assert c.model.hub_layers == () and c.data_scale == 4.0
    assert c.train.phase_lr_scale is not p.train.phase_lr_scale, "dict aliased"
    q = make_v3_preset(28, name="x", data_scale=1.0)
    assert q.model.n_layers == 28 and abs(sum(
        ph["planned_tokens"] for ph in q.curriculum) - 16.1e9) < 1e6
    n = AlephLM(get_preset("mini-beatrix-3-l28").model).param_count() / 1e6
    assert 300 < n < 360, f"L28 {n:.1f}M"


@case("curriculum scale: 4x caps every finite corpus; natural/generators/hold rules; refusal; 1x bit-exact")
def t_curriculum_scale():
    from ..data import curriculum as C
    sc = C.scaled_curriculum(4.0, epoch_cap=4.0, rebalance_to="natural")
    assert abs(sum(sc["stage_tokens"].values())
               - 4 * sum(C._BASE_STAGE_TOKENS.values())) < 1
    assert not sc["flags"], sc["flags"]
    for st, n, wb, wa, eb, ea in sc["table"]:
        assert ea <= 4.0 + 1e-6, (st, n, ea)
    moved = [r for r in sc["table"] if r[2] != r[3]]
    assert moved, "4x must cap something (siqa/aochildes/babi...)"
    for st, mix in sc["mixes"].items():
        assert abs(sum(w for _, w in mix) - 1.0) < 1e-3
        ballast = sum(w for n, w in mix if n in C.NATURAL)
        assert ballast >= C.MIN_BALLAST - 1e-9, (st, ballast)
        top = max([w for n, w in mix if n.endswith("-synth")], default=0.0)
        assert top <= C.MAX_GENERATOR_SHARE + 1e-9, (st, top)
        base = dict(C._BASE_MIXES[st])
        for n, w in mix:
            if n.endswith("-synth"):
                assert abs(w - base[n]) < 2e-5, "generator share must not move"
    assert sc["cross_stage"]["siqa-narrative"] > 4.0, "cross-stage column"
    # the record's precedent: generators absorb first, capped at their share
    sg = C.scaled_curriculum(4.0, epoch_cap=4.0, rebalance_to="generators")
    s1 = dict(sg["mixes"]["curriculum-s1"])
    assert s1["perspective-synth"] > 0.30 - 1e-9 and s1["perspective-synth"] <= C.MAX_GENERATOR_SHARE + 1e-9
    for st, mix in sg["mixes"].items():
        for n, w in mix:
            if n.endswith("-synth"):
                assert w <= C.MAX_GENERATOR_SHARE + 1e-9, (st, n, w)
    # the law's ~2 cap moves more weight than the audit's 4
    s2 = C.scaled_curriculum(4.0, epoch_cap=2.0, rebalance_to="natural")
    assert len([r for r in s2["table"] if r[2] != r[3]]) >= len(moved)
    # hold: stages at 1x, the extra returned for general text
    sh = C.scaled_curriculum(4.0, epoch_cap=4.0, rebalance_to="hold")
    assert sh["stage_tokens"] == C._BASE_STAGE_TOKENS
    assert sh["held_tokens"] == 3 * sum(C._BASE_STAGE_TOKENS.values())
    assert not [r for r in sh["table"] if r[2] != r[3]]
    # no rule = a refusal, with the offending stage named
    try:
        C.scaled_curriculum(4.0)
        raise AssertionError("4x without a rebalance rule must refuse")
    except ValueError as e:
        assert "curriculum-s0" in str(e) and "rebalance_to" in str(e)
    assert "epoch_cap" in C.scaled_curriculum(4.0, rebalance_to="natural")["flags"]
    one = C.scaled_curriculum(1.0)
    for st, mix in one["mixes"].items():
        base, got = dict(C._BASE_MIXES[st]), dict(mix)
        assert set(base) == set(got)
        assert all(abs(base[k] - got[k]) < 1e-9 for k in base), st
    dp1, dp2 = C.data_plane(4.0, 4.0, "natural"), C.data_plane(4.0, 2.0, "natural")
    assert dp1["recipe_hash"] != dp2["recipe_hash"] and dp1 == C.data_plane(4.0, 4.0, "natural")
    assert C.data_plane(1.0)["recipe_hash"] == C.data_plane(1.0)["recipe_hash"]
    before = {k: [tuple(x) for x in v] for k, v in C.CURRICULUM_MIXES.items()
              if k in C.STAGE_TOKENS}
    C.apply_curriculum_scale(4.0, 4.0, "natural", verbose=False)
    assert C.STAGE_TOKENS["curriculum-s0"] == 2_800_000_000
    assert dict(C.CURRICULUM_MIXES["curriculum-s1"])["siqa-narrative"] < 0.02
    C.apply_curriculum_scale(1.0, verbose=False)
    after = {k: [tuple(x) for x in v] for k, v in C.CURRICULUM_MIXES.items()
             if k in C.STAGE_TOKENS}
    assert after == before, "1x restore is not bit-exact"
    assert len(C.curriculum_phases(4.0)) == 9
    assert C.curriculum_phases(4.0, "hold")[0]["planned_tokens"] == 700_000_000
    # a still-planned manifest phase follows the applied scale
    from ..train.manifest import RunManifest
    m = RunManifest.fresh("x", {}, C.curriculum_phases(1.0))
    m.phases[0]["status"] = "done"
    assert C.append_curriculum_phases(m, 4.0, 4.0, "natural") == 0
    assert m.phases[0]["planned_tokens"] == 700_000_000
    assert m.phases[1]["planned_tokens"] == 2_800_000_000
    C.apply_curriculum_scale(1.0, verbose=False)


@case("guards: G1/G2/G3 replay a synthetic series at the expected steps; modes; state roundtrip")
def t_guards():
    from ..train.guards import (GuardConfig, GuardCore, g1_eval, g2_eval,
                                g3_eval, certification_from_ledger)
    cfg = GuardConfig(ref_window=(100, 200), log_every=10, census_every=10,
                      g1_window=5, g2_sustain=2, g3_exempt_last=1,
                      modes={"G1": "halt", "G2": "watch", "G3": "halt"})
    steps = list(range(10, 401, 10))
    g = [8.0 if s > 300 else 0.5 for s in steps]
    f1, r1 = g1_eval(steps, g, cfg)
    assert f1 == 330, (f1, r1)
    ent = {"0": [0.8] * 40, "1": [0.8 if s <= 250 else 0.4 for s in steps],
           "2": [0.8] * 40}
    f2, r2 = g2_eval(steps, ent, 3, cfg)
    assert f2 == 270 and r2["fire_layer"] == 1, (f2, r2)
    er = {"0": [100.0 if s < 300 else 5.0 for s in steps], "1": [100.0] * 40,
          "2": [3.0 if s >= 200 else 100.0 for s in steps]}
    f3, r3 = g3_eval(steps, er, 3, cfg)
    assert f3 == 300 and r3["fire_layer"] == 0, (f3, r3)
    assert g3_eval(steps, er, 3, cfg, exempt_last=0)[0] == 200, "funnel exempt"
    core = GuardCore(cfg, 3)
    fired = []
    for i, s in enumerate(steps):
        core.observe_gnorm(s, g[i])
        census = {"layers": {L: {"bank_dispatch_entropy_frac": ent[str(L)][i],
                                 "hidden_erank": er[str(L)][i]}
                             for L in range(3)},
                  "flags": {"den_floor": s == 50}, "flag_detail": {}}
        core.observe_census(s, census)
        fired += core.check(s)
    assert [n for n, _ in fired] == ["G2", "G3", "G1"], fired
    assert core.halting(fired[:1]) == [] and len(core.halting(fired)) == 2
    assert core.flags_first["den_floor"]["step"] == 50
    core2 = GuardCore(cfg, 3)
    core2.load_state_dict(core.state_dict())
    assert core2.fired == core.fired and core2.check(400) == []
    led = {"verdict": {"certification": {
        "G1": {"CERTIFIED": True}, "G2": {"CERTIFIED": False},
        "G3_exempt2": {"CERTIFIED": True}}}}
    assert certification_from_ledger(led)["modes"] == {"G1": "halt", "G2": "watch",
                                                         "G3": "halt"}


@case("phase LR: multiplier after warmup; {} = the flat form; a guard halt returns cleanly + resumes")
def t_phase_lr_and_guard_halt():
    from ..train.optim import apply_lr, lr_scale
    from ..train.trainer import Trainer
    from ..train.guards import GuardConfig
    from ..presets import Preset, TrainConfig
    m = AlephLM(TINY)
    opts = build_optimizers(m, 2e-2, 0.95, 3e-4)
    s = apply_lr(opts, [2e-2, 3e-4], 500, 200, mult=0.3)
    assert abs(s - 0.3) < 1e-12 and abs(opts[0].param_groups[0]["lr"] - 6e-3) < 1e-12
    assert abs(apply_lr(opts, [2e-2, 3e-4], 100, 200) - lr_scale(100, 200)) < 1e-12

    class _T:
        tc = TrainConfig(phase_lr_scale={"anneal": 0.25, "anneal_mix": 0.5})
    assert Trainer._phase_mult(_T(), "anneal_nochat") == 0.25
    assert Trainer._phase_mult(_T(), "anneal_mix") == 0.5, "longest prefix wins"
    assert Trainer._phase_mult(_T(), "curriculum_s0") == 1.0
    # a guard that fires at the first sample after its reference window
    # (five samples at steps 1..5; the halt at step 6, after the step-4
    # checkpoint — so latest.pt must stay at step 4)
    gc = GuardConfig(ref_window=(0, 5), log_every=1, census_every=2,
                     census_dataset="synthetic", g1_window=1, g1_ratio=0.0,
                     g1_clip_points=-1.0,
                     modes={"G1": "halt", "G2": "off", "G3": "off"})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=2, grad_accum=1, warmup_steps=2,
                                     log_every=1, health_every=100, eval_every=100,
                                     ckpt_every=4, tb_upload_every=100,
                                     val_tokens=256, canary_episodes=4),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000, status="planned")])
        t = Trainer(p, hf_token=None, out_dir=td, resume=False, guard=gc,
                    device="cpu")
        t.train(max_steps=20)
        assert t._guard_halt is not None and t._guard_halt["guard"] == "G1"
        assert t.step == 6, t.step
        rdir = os.path.join(td, "tiny-test", "resume")
        assert os.path.exists(os.path.join(rdir, "guard_G1_step6.pt"))
        latest = torch.load(os.path.join(rdir, "latest.pt"), map_location="cpu",
                            weights_only=False)
        assert latest["step"] == 4, "the halt must not rewrite latest.pt"
        arch = torch.load(os.path.join(rdir, "guard_G1_step6.pt"), map_location="cpu",
                          weights_only=False)
        assert arch["step"] == 6 and "G1" in arch["guard"]["fired"]
        assert t.manifest.halt["guard"] == "G1" and t.manifest.halt["step"] == 6
        assert t.manifest.halt["archive"] == "resume/guard_G1_step6.pt"
        assert any("GUARD HALT" in n["msg"] for n in t.manifest.notes)
        assert not any(c["kind"] == "fp8" and c["step"] == 6 for c in t.manifest.checkpoints)
        t2 = Trainer(p, hf_token=None, out_dir=td, resume=True, guard=gc,
                     device="cpu")
        assert t2.step == 4 and t2.manifest.halt["guard"] == "G1"
        assert torch.equal(t2.guard.census_batch, t.guard.census_batch), "pinned batch"
        try:
            t2.train(max_steps=1)
            raise AssertionError("a halted run must refuse to train")
        except RuntimeError as e:
            assert "HALTED RUN" in str(e)
        t2.train(max_steps=1, resume_after_halt=True)
        assert t2.step == 5 and t2.manifest.halt is None
        assert any("HALT CLEARED" in n["msg"] for n in t2.manifest.notes)
        t2.guard.record_boundary_read("G5", "syn", {"fired": False, "step": 5})
        assert "G5" in t2.guard.summary()["boundary_reads"]


@case("stage arms: attach bit-inert; plain trunk keys; one step moves arms+trunk; gauges; resume")
def t_arms():
    try:
        import amoe  # noqa: F401
    except ImportError:
        raise AssertionError("amoe-lora is not installed in this env — the "
                             "stage-arm program needs it: pip install "
                             "'amoe-lora @ git+https://github.com/AbstractEyes/amoe-lora'")
    from ..train.arms import (StageArm, ArmProgramConfig, StageArmProgram,
                              trunk_state_dict)
    from ..train.trainer import Trainer
    from ..model.governor import govern_model
    from ..presets import Preset, TrainConfig
    from safetensors.torch import load_file
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        acfg = ArmProgramConfig(
            arms=[StageArm("a0", phase="syn", spec=dict(n_slots=4, K=8, D=4, hidden=16),
                           lam=2.0, seed=1)],
            lr=1e-3, abstain_chunks=1, offdomain_dataset="synthetic", offdomain_seed=9)
        p = Preset(model=TINY,
                   train=TrainConfig(micro_batch=2, grad_accum=1, warmup_steps=2,
                                     log_every=2, health_every=4, eval_every=100,
                                     ckpt_every=100, tb_upload_every=100,
                                     val_tokens=256, canary_episodes=4),
                   curriculum=[dict(name="syn", dataset="synthetic",
                                    planned_tokens=10_000_000, status="planned")])
        t = Trainer(p, hf_token=None, out_dir=td, resume=False,
                    arms=StageArmProgram(acfg), device="cpu")
        x = torch.randint(0, 256, (2, 33))
        t.raw_model.eval()
        with torch.no_grad():
            ref = t.raw_model(x).logits.clone()
        t.arms.sync("syn")
        assert t.arms.attached == ["a0"]
        with torch.no_grad():
            armed = t.raw_model(x).logits
        assert torch.equal(ref, armed), "a fresh bias-zeroed arm must be bit-inert"
        assert set(trunk_state_dict(t.raw_model)) == set(AlephLM(TINY).state_dict())
        w_arm = [q.detach().clone() for q in t.arms.params()]
        w_trunk = t.raw_model.embed.emb0.weight.detach().clone()
        t.train(max_steps=2)
        assert any(not torch.equal(a, b) for a, b in zip(w_arm, t.arms.params()))
        assert not torch.equal(w_trunk, t.raw_model.embed.emb0.weight)
        t.raw_model.eval()
        with torch.no_grad():        # the armed read AT the checkpoint (the
            a1 = t.raw_model(x).logits.clone()   # governor below may move weights)
        census = instruments.model_census(t.raw_model, x[:, :-1])
        assert set(census["layers"]) == {0, 1, 2} and "hub_consumed_erank" in census["layers"][1]
        govern_model(t.raw_model, 45.0)
        ev = t._full_eval()
        assert "toggle_arms_off" in ev["ledger"]
        g = t.arms.gauges(t._val(), t._val())
        assert "a0" in g and math.isfinite(g["a0"]["fineweb_delta"])
        assert math.isfinite(g["a0"]["selectivity"])
        sd = load_file(os.path.join(td, "tiny-test",
                                    f"checkpoints/step_{t.step:08d}.safetensors"))
        assert "blocks.0.n1.weight" in sd and not any(".adapter." in k for k in sd)
        t2 = Trainer(p, hf_token=None, out_dir=td, resume=True,
                     arms=StageArmProgram(acfg), device="cpu")
        assert t2.step == 2 and t2.arms.attached == ["a0"]
        t2.raw_model.eval()
        with torch.no_grad():
            a2 = t2.raw_model(x).logits
        assert torch.equal(a1, a2), "resumed armed logits differ"
        ck = t2.arms.anchor("a0", "tiny-test", t2.step)
        assert any(k.endswith("addr.home") for k in ck.adapters), "anchor format"
        assert ck.meta["base_model_id"] == "alephllm/tiny-test@step2"
        m3 = AlephLM(TINY)
        m3.load_state_dict({k: v.float() for k, v in
                            trunk_state_dict(t2.raw_model).items()})
        m3.eval()
        with t2.arms.handles["a0"].all_off():
            with torch.no_grad():
                off = t2.raw_model(x).logits
        with torch.no_grad():
            bare = m3(x).logits
        assert torch.equal(off, bare), "masked arm != the plain-key trunk"
        # a faulty member leaves the step, the program stays resumable
        t2.arms.disable("a0", "smoke")
        assert not t2.arms.active and t2.arms.chunks_per_step(1) == 0
        assert t2.arms.state_dict()["disabled"] == {"a0": "smoke"}
        with torch.no_grad():
            assert torch.equal(t2.raw_model(x).logits, bare), "disabled = masked"
        muon_p, _ = split_params(t2.raw_model)
        assert not any(id(p) in {id(q) for q in t2.arms.params()} for p in muon_p), \
            "arm parameters must never enter the trunk's optimizer split"


# ------------------------------------------------------------ weak-token fusion (v3)
def _fusion_cfg(hub_layers=(0, 1, 2), **spec):
    import dataclasses
    base = dict(rule="spacelike", k_lo=1, k_hi=1)
    base.update(spec)
    return dataclasses.replace(TINY, name="tiny-fused", hub_layers=hub_layers, fusion=base)


def _atlas_table(path):
    """A synthetic atlas table: every cell 'a'-led is a choice point (3 bits), everything else closed (0 bits),
    cells with the first byte 'z' unwitnessed."""
    import numpy as np
    ent = np.zeros(256 ** 3, dtype=np.uint8)
    wit = np.full(256 ** 3, 100, dtype=np.uint16)
    idx = np.arange(256 ** 3)
    b0 = idx // 65536
    ent[b0 == ord("a")] = 3 * 16
    wit[b0 == ord("z")] = 0
    np.savez_compressed(path, entropy_x16=ent, witness=wit)
    return path


@case("fusion: no middle (k_lo = n_layers) reproduces the unfused logits exactly")
def t_fusion_identity():
    torch.manual_seed(3)
    m0 = AlephLM(TINY)
    torch.manual_seed(3)
    m1 = AlephLM(_fusion_cfg(hub_layers=TINY.hub_layers, k_lo=3, k_hi=0))   # the same layout; no middle
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert missing == ["fusion.null"] and not unexpected, (missing, unexpected)
    m0.eval(); m1.eval()
    x = torch.randint(0, 256, (2, 48))
    with torch.no_grad():
        d = (m0(x).logits - m1(x).logits).abs().max().item()
    assert d == 0.0 or d < 1e-6, f"identity broken {d:.2e}"


@case("fusion: forced starts at specials and newlines (both sides); units never cross them")
def t_fusion_forced():
    from ..model.fusion import Fusion
    from ..data.special_tokens import DOC, END
    f = Fusion(dict(rule="spacelike"), 8)
    x = torch.tensor([[ord("a"), ord("b"), 10, 10, ord("c"), DOC, ord("d"), ord("e"), END, ord("f")]])
    st = f.starts(x)[0].tolist()
    assert st[2] and st[3] and st[4], f"newline pair must be two units and release the next byte: {st}"
    assert st[5] and st[6] and st[8] and st[9], f"specials are their own units: {st}"
    assert not st[1], "b fuses into a"
    assert not st[7], "e fuses into d"


@case("fusion: the parallel path is causal (prefix logits invariant to the suffix)")
def t_fusion_causal():
    torch.manual_seed(4)
    m = AlephLM(_fusion_cfg())
    m.eval()
    x = torch.randint(0, 256, (1, 64))
    y = x.clone()
    y[0, 40:] = torch.randint(0, 256, (24,))
    with torch.no_grad():
        la, _ = m(x)
        lb, _ = m(y)
    d = (la[0, :39] - lb[0, :39]).abs().max().item()
    assert d < 1e-4, f"fusion causality leak {d:.2e}"


@case("fusion: cached decode matches the parallel path, rows with different unit structures")
def t_fusion_decode():
    torch.manual_seed(5)
    m = AlephLM(_fusion_cfg())
    m.eval()
    text = [b"the quick brown fox jumps over the lazy dog\n\nand then some more words here",
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
    x = torch.tensor([list(t[:70]) for t in text])
    with torch.no_grad():
        full = m(x).logits
        logits, cache = m.prefill(x[:, :30])
        assert (logits[:, 0] - full[:, 29]).abs().max().item() < 1e-4, "prefill last-position mismatch"
        worst = 0.0
        for t in range(30, 70):
            lt = m.decode_step(x[:, t], cache)
            if t + 1 < 70:
                worst = max(worst, (lt[:, 0] - full[:, t]).abs().max().item())
    assert worst < 1e-3, f"fused decode drifts from the parallel path: {worst:.2e}"


@case("fusion: the entropy rule reads the atlas table (choice points, unwitnessed cells) and decodes")
def t_fusion_entropy():
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        path = _atlas_table(os.path.join(td, "atlas.npz"))
        torch.manual_seed(6)
        m = AlephLM(_fusion_cfg(rule="entropy", theta=1.0, table=path, witness_floor=8))
        m.eval()
        x = torch.tensor([[ord("q"), ord("a"), ord("b"), ord("c"), ord("d"), ord("z"), ord("b"), ord("c"), ord("d"), ord("e")]])
        st = m.fusion.starts(x)[0].tolist()
        # cell for target t is (x[t-3], x[t-2], x[t-1]): t=4 -> ('a','b','c') a-led = choice point; t=8 -> ('z','b','c') unwitnessed
        assert st[4] and st[8], f"choice point / unwitnessed cell must start units: {st}"
        assert not st[5] and not st[7], f"closed cells fuse: {st}"
        plan = m.fusion.plan(x)
        assert int(plan.n_units[0]) == int(sum(st))
        y = torch.randint(0, 256, (2, 40))
        with torch.no_grad():
            full = m(y).logits
            _, cache = m.prefill(y[:, :20])
            worst = 0.0
            for t in range(20, 40):
                lt = m.decode_step(y[:, t], cache)
                if t + 1 < 40:
                    worst = max(worst, (lt[:, 0] - full[:, t]).abs().max().item())
        assert worst < 1e-3, f"entropy-rule decode drifts: {worst:.2e}"


@case("fusion: the hybrid rule fuses a predictable word into the unit before it, and decodes")
def t_fusion_hybrid():
    with tempfile.TemporaryDirectory() as td:
        path = _atlas_table(os.path.join(td, "atlas.npz"))
        torch.manual_seed(8)
        m = AlephLM(_fusion_cfg(rule="hybrid", theta=1.0, table=path, witness_floor=8))
        m.eval()
        # "xa b" -> the word 'b' starts at t=3 with cell ('x','a',' '): x-led = closed (0 bits) -> 'b' FUSES;
        # "ba c" -> the word 'c' starts at t=7 with cell ('b','a',' ')... use an a-led cell: "ma c": ('a',' ','c')? cells are
        # (x[t-3], x[t-2], x[t-1]); for t=7 in "xa bba c": (x[4],x[5],x[6]) = ('b','a',' ') -> closed too. Build explicitly:
        s = list(b"xa b") + list(b"aa c")            # t=3: cell ('x','a',' ') closed -> fuse; t=7: cell ('a','a',' ') a-led -> start
        x = torch.tensor([s])
        st = m.fusion.starts(x)[0].tolist()
        assert not st[3], f"a predictable word must fuse: {st}"
        assert st[7], f"a choice-point word must start a unit: {st}"
        y = torch.tensor([list(b"the cat sat on the mat and the dog ran off to the barn again")])
        with torch.no_grad():
            full = m(y).logits
            _, cache = m.prefill(y[:, :25])
            worst = 0.0
            for t in range(25, y.shape[1]):
                lt = m.decode_step(y[:, t], cache)
                if t + 1 < y.shape[1]:
                    worst = max(worst, (lt[:, 0] - full[:, t]).abs().max().item())
        assert worst < 1e-3, f"hybrid-rule decode drifts: {worst:.2e}"


@case("fusion: forward/backward finite; grads reach the null vector and the middle blocks")
def t_fusion_grads():
    torch.manual_seed(7)
    m = AlephLM(_fusion_cfg())
    x = torch.randint(0, 256, (2, 48))
    _, loss = m(x[:, :-1], targets=x[:, 1:])
    assert math.isfinite(loss.item())
    loss.backward()
    assert m.fusion.null.grad is not None and m.fusion.null.grad.abs().sum() > 0, "the null vector never trained"
    mid = [p for p in m.blocks[1].parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mid), "no gradient reached the middle block"


@case("minted lexicon: the legacy rules stream bit-identical; minted rows draw only minted words and carry onset pairs at the rate; "
      "a split keeps its ratio under a scale and the frame cap; the plane records the lexicon; refusal without one; an amendment passes the resume check")
def t_minted_lexicon():
    import hashlib
    import itertools
    import re
    import types
    from ..data import curriculum as C
    from ..data.streams import CURRICULUM_MIXES
    from ..train.trainer import Trainer
    rows = [r["text"] for r in itertools.islice(C._rulechain_rows(7), 500)]
    assert hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16] == "c8b74cce6c7a3476", "the legacy rules stream changed"
    easy = [f"{'abcdefghijk'[i]}anes" for i in range(11)]
    trap = [f"{'abcdefghijk'[i]}ortu" for i in range(11)]
    lex = {"easy": easy, "trap": trap, "pairs": [["skarn", "skelt"], ["thrum", "thane"], ["plinx", "plost"], ["grend", "grulp"]]}
    words, weights, pairs = C.minted_vocab(lex)
    assert len(words) == 30 and len(pairs) == 4
    pred = re.compile(r"If someone is (\w+), then they are (\w+)\.")
    seen, pair_rows = set(), 0
    mrows = list(itertools.islice(C._rulechain_rows(3, vocab=words, weights=weights, pairs=pairs, pair_rate=0.5), 600))
    for r in mrows:
        ws = {w for m in pred.finditer(r["text"]) for w in m.groups()}
        assert ws and not (ws & set(C._PRED)), r
        seen |= ws
        pair_rows += any(a in ws and b in ws for a, b in pairs)
    assert seen == set(words), sorted(set(words) - seen)
    assert 0.4 <= pair_rows / len(mrows) <= 0.8, pair_rows / len(mrows)
    C.apply_curriculum_scale(4.0, 2.0, "generators", verbose=False)
    h0 = C.data_plane(4.0, 2.0, "generators")
    rec = C.set_minted_lexicon(lex, {"curriculum-s3": {"rulechain-synth": 0.08, "rulechain-minted": 0.24}}, 0.3, verbose=False)
    fr = {n: w for n, w in CURRICULUM_MIXES["curriculum-s3"] if n in C._FRAMES}
    assert sum(fr.values()) <= C.MAX_GENERATOR_SHARE + 1e-6, fr
    assert abs(fr["rulechain-minted"] / sum(fr.values()) - 0.75) < 1e-3, fr
    assert C.audit_mix(warn=False)["curriculum-s3"]["top_generator"] <= C.MAX_GENERATOR_SHARE + 1e-6
    h1 = C.data_plane(4.0, 2.0, "generators")
    assert h1["recipe_hash"] != h0["recipe_hash"] and h1["minted_lexicon"] == rec["sha"]
    tok = ByteTrigramTokenizer()
    b = build_stream("curriculum-s3", tok, 128, 2, seed=5).next_batch()
    assert tuple(b.shape) == (2, 129), b.shape
    try:
        C.set_minted_lexicon(lex, {"curriculum-s3": {"rulechain-synth": 0.10, "rulechain-minted": 0.30}}, 0.3, verbose=False)
        raise RuntimeError("a split that moves the frame's total was accepted")
    except AssertionError:
        pass
    C.clear_minted_lexicon()
    assert C.data_plane(4.0, 2.0, "generators")["recipe_hash"] == h0["recipe_hash"]
    try:
        build_stream("rulechain-minted", tok, 128, 2, seed=1).next_batch()
        raise AssertionError("the minted source opened without a lexicon")
    except RuntimeError as e:
        assert "no minted lexicon" in str(e), e
    notes = []
    fake = types.SimpleNamespace(tc=types.SimpleNamespace(data_plane_amendment=None),
                                 manifest=types.SimpleNamespace(data_plane=dict(h0), note=notes.append),
                                 _data_plane=dict(h1))
    try:
        Trainer._check_data_plane(fake)
        raise AssertionError("a recipe change was accepted without an amendment note")
    except RuntimeError as e:
        assert "data plane changed" in str(e), e
    fake.tc.data_plane_amendment = "test: the lead's ruling"
    Trainer._check_data_plane(fake)
    assert fake.manifest.data_plane == h1 and notes and "AMENDED" in notes[0]
    C.apply_curriculum_scale(1.0, verbose=False)


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
