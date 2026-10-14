"""The Canon layers (0.10.18 rc8, 2026-10-14; model/canon.py; Allen-Zhu 2512.17351 Eq. 4.1).
Run: python -m geolip.alephllm.tests.test_canon   (CPU, seconds; no download)
Cases:
  1 the flag off = rc7 bit for bit: the config's key at its default; no Canon module and no new state-dict key
  2 the layer: zero-born = the identity bit for bit; with weights it equals the depthwise causal convolution written with
    conv1d; causal (a later position never moves an earlier output); windows shorter than the kernel
  3 a craft with the layer at birth equals the craft without it bit for bit (same seed: the logits and the loss), and its state
    dict carries exactly the Canon weights more; the config round trip keeps the sites; a bad site refuses
  4 it trains: one Adam step, finite loss, every Canon weight takes gradient, the craft then differs from its Canon-free twin;
    the hub_ckpt=2 path under grad equals the plain path
  5 the readers that replay a block (instruments.model_census, eval.lexicon.capture) read the Canon layer's output; decode refuses
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.alephlm import AlephLM
from ..model.canon import CanonConv
from ..presets import AlephLMConfig

RESULTS: list = []
TINY = dict(name="tiny-canon", d_model=64, n_layers=3, n_heads=4, context=128, hub_layers=(1,), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def conv_ref(h, w):
    """The depthwise causal convolution with conv1d: out_t = h_t + sum_i w[:, i] * h_(t-i)."""
    d, k = w.shape
    x = F.pad(h.transpose(1, 2), (k - 1, 0))                      # (B, d, T + k - 1), zeros before the first position
    kern = w.flip(1).unsqueeze(1)                                 # (d, 1, k): conv1d correlates, so the newest tap goes last
    return h + F.conv1d(x, kern, groups=d).transpose(1, 2)


# ------------------------------------------------------------------ 1 the flag off
def case_1():
    cfg = AlephLMConfig(**TINY)
    check("1a the config's key at its default", cfg.to_dict()["canon"] == "")
    m = AlephLM(cfg)
    keys = [k for k in m.state_dict() if "canon" in k]
    check("1b no Canon module and no new state-dict key with the flag off",
          not keys and all(b.canon_a is None and b.canon_c is None for b in m.blocks), str(keys))


# ------------------------------------------------------------------ 2 the layer
def case_2():
    torch.manual_seed(0)
    c = CanonConv(32)
    h = torch.randn(2, 20, 32)
    with torch.no_grad():
        check("2a zero-born: the identity bit for bit", torch.equal(c(h), h))
        c.weight.normal_()
        y = c(h)
        r = conv_ref(h, c.weight)
        check("2b with weights: the depthwise causal convolution (conv1d reference)", float((y - r).abs().max()) <= 1e-5,
              f"max |diff| {float((y - r).abs().max()):.2e}")
        h2 = h.clone()
        h2[:, 12] += 5.0
        y2 = c(h2)
        check("2c causal: a change at position 12 leaves positions 0-11 exactly", torch.equal(y2[:, :12], y[:, :12])
              and float((y2[:, 12:16] - y[:, 12:16]).abs().max()) > 0 and torch.equal(y2[:, 16:], y[:, 16:]))
        hs = h[:, :2]
        check("2d a window shorter than the kernel", torch.equal(c(hs), conv_ref(hs, c.weight)) or
              float((c(hs) - conv_ref(hs, c.weight)).abs().max()) <= 1e-6)


# ------------------------------------------------------------------ 3 at birth
def case_3():
    torch.manual_seed(0)
    m0 = AlephLM(AlephLMConfig(**TINY))
    torch.manual_seed(0)
    m1 = AlephLM(AlephLMConfig(**dict(TINY, canon="AC")))
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    extra = sorted(set(sd1) - set(sd0))
    want = sorted(f"blocks.{i}.canon_{s}.weight" for i in range(TINY["n_layers"]) for s in "ac")
    check("3a the state dict carries exactly the Canon weights more", extra == want and not (set(sd0) - set(sd1)), str(extra))
    check("3b every shared tensor equal (the layer draws no random numbers)", all(torch.equal(sd0[k], sd1[k]) for k in sd0))
    idx = torch.randint(0, 256, (2, 33))
    m0.eval(), m1.eval()
    with torch.no_grad():
        o0, o1 = m0(idx[:, :-1], idx[:, 1:]), m1(idx[:, :-1], idx[:, 1:])
    check("3c at birth the logits and the loss equal the Canon-free craft's bit for bit",
          torch.equal(o0.logits, o1.logits) and torch.equal(o0.loss, o1.loss))
    d = AlephLMConfig(**dict(TINY, canon="A")).to_dict()
    check("3d the config round trip keeps the sites", AlephLMConfig.from_dict(d).canon == "A")
    try:
        AlephLM(AlephLMConfig(**dict(TINY, canon="AB")))
        bad = False
    except AssertionError:
        bad = True
    check("3e a site other than A / C refuses", bad)


# ------------------------------------------------------------------ 4 it trains
def case_4():
    torch.manual_seed(1)
    m = AlephLM(AlephLMConfig(**dict(TINY, canon="AC")))
    torch.manual_seed(1)
    m0 = AlephLM(AlephLMConfig(**TINY))
    m.train()
    idx = torch.randint(0, 256, (2, 33))
    loss = m(idx[:, :-1], idx[:, 1:]).loss
    loss.backward()
    gr = [b.canon_a.weight.grad for b in m.blocks] + [b.canon_c.weight.grad for b in m.blocks]
    check("4a finite loss; every Canon weight takes gradient", bool(torch.isfinite(loss)) and all(g is not None and float(g.abs().sum()) > 0 for g in gr))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    opt.step()
    m.eval(), m0.eval()
    with torch.no_grad():
        moved = float((m(idx[:, :-1]).logits - m0(idx[:, :-1]).logits).abs().max())
    check("4b after one step the craft differs from its Canon-free twin (the layer is live)", moved > 0, f"max |dlogit| {moved:.2e}")
    torch.manual_seed(2)
    mc = AlephLM(AlephLMConfig(**dict(TINY, canon="AC", hub_ckpt=2)))
    torch.manual_seed(2)
    mp = AlephLM(AlephLMConfig(**dict(TINY, canon="AC")))
    with torch.no_grad():
        for b in list(mc.blocks) + list(mp.blocks):
            b.canon_a.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(3))
            b.canon_c.weight.normal_(0, 0.1, generator=torch.Generator().manual_seed(4))
    mc.train(), mp.train()
    lc, lp = mc(idx[:, :-1], idx[:, 1:]).loss, mp(idx[:, :-1], idx[:, 1:]).loss
    lc.backward(), lp.backward()
    gc, gp = mc.blocks[1].canon_a.weight.grad, mp.blocks[1].canon_a.weight.grad
    check("4c the hub_ckpt=2 path under grad equals the plain path (loss and the Canon gradient)",
          float((lc - lp).detach().abs()) <= 1e-6 and float((gc - gp).abs().max()) <= 1e-5, f"{float((lc - lp).detach().abs()):.2e}")


# ------------------------------------------------------------------ 5 the readers and decode
def case_5():
    torch.manual_seed(5)
    m = AlephLM(AlephLMConfig(**dict(TINY, canon="AC")))
    with torch.no_grad():
        for b in m.blocks:
            b.canon_a.weight.normal_(0, 0.2)
            b.canon_c.weight.normal_(0, 0.2)
    m.eval()
    idx = torch.randint(0, 256, (2, 32))
    from ..eval.lexicon import capture
    with torch.no_grad():
        ref = m(idx).logits
        cap = capture(m, idx)
    nll = cap.get("nll")
    if nll is not None:
        lp = torch.log_softmax(ref.float(), -1)[:, :-1].gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
        ok = float((nll[:, 1:] - (-lp)).abs().max()) <= 1e-4 if nll.shape[-1] == idx.shape[-1] else float((nll - (-lp)).abs().max()) <= 1e-4
    else:
        ok = False
    check("5a the lexicon replay reads the Canon layer: its per-position loss equals the forward's", ok)
    from ..train.instruments import model_census
    try:
        cen = model_census(m, idx)
        ok = isinstance(cen, dict) and len(cen) > 0
    except Exception as e:  # noqa: BLE001
        ok = False
        print("   census:", type(e).__name__, e)
    check("5b the model census runs on a craft with the layer", ok)
    refused = 0
    for call in (lambda: m.prefill(idx[:, :8]), lambda: m.generate(idx[:, :8], max_new=2)):
        try:
            call()
        except NotImplementedError:
            refused += 1
    check("5c decode (prefill, generate) refuses a craft with the layer", refused == 2)


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3, case_4, case_5):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
