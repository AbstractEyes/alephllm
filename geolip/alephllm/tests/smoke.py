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
