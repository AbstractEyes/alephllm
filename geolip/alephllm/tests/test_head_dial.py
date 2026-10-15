"""The head's dial forms (0.10.18 rc11): AlephLMConfig.head_mode = "" | dial | dialraw | dialplus | pergain.
Run: python -m geolip.alephllm.tests.test_head_dial   (CPU, seconds; no download)
Cases:
  1 at birth every form is the plain readout bit for bit (train and eval, full forward and the cached decode), and the plain
    head's own birth logits too (its carrying read is zero); the state dict = the plain head's keys plus the dial's
  2 the toggle: disable_aleph returns the plain readout in every form; with the dial moved, the full forward differs from it
  3 the bound: however large the dial's weights, the per-position scale stays within [1/band, band]; the band is the config's
  4 the centering: in training the running mean follows the forward's mean (detached, one step a forward) and in eval it is
    frozen; dialraw keeps it at zero; pergain keeps one mean per byte
  5 the split: w_t is one-dimensional (Adam); proj, the codebook, W_h and W_s stay Muon; the carrying read is frozen in dial /
    dialraw / pergain and trains in dialplus; with the readout named for decay the AdamW group holds exactly the named weights
    (pergain: G among them when named)
  6 the gradient: a cross-entropy loss reaches w_t at birth (the dial can move from zero); none reaches the frozen W_s
  7 a plain head's state loads into a dial head (the dial's keys missing, the rest strict): the logits are the plain readout
  8 dial_stats reads: a zero dial = mean 0, spread 0, nothing saturated; a huge dial saturates
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..model.alephlm import AlephLM
from ..model.head import HEAD_MODES, DualHead
from ..presets import AlephLMConfig
from ..train.optim import Muon, build_optimizers, split_params

RESULTS: list = []
TINY = dict(name="tiny-dial", d_model=64, n_layers=2, n_heads=4, context=64, hub_layers=(), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)
FORMS = [m for m in HEAD_MODES if m]


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _model(mode="", seed=0, **over):
    torch.manual_seed(seed)
    return AlephLM(AlephLMConfig(**{**TINY, **over}, head_mode=mode))


def _ids(seed=1, n=2, t=48):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(32, 127, (n, t), generator=g)


def _plain_readout(m, x):
    with torch.no_grad():
        return m(x, disable_head_aleph=True)[0]


def _move(m, scale=1.0, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = m.head.dial_weight
        w.copy_(scale * torch.randn(w.shape, generator=g))


def case_1():
    x = _ids()
    for mode in HEAD_MODES:
        m = _model(mode)
        ref = _plain_readout(m, x)
        m.train()
        with torch.no_grad():
            tr = m(x)[0]
        m.eval()
        with torch.no_grad():
            ev = m(x)[0]
            lg, cache = m.prefill(x[:, :-1])
            st = m.decode_step(x[:, -1], cache)
        check(f"1a {mode or 'plain'}: at birth the full forward is the plain readout (train and eval)",
              torch.equal(tr, ref) and torch.equal(ev, ref))
        check(f"1b {mode or 'plain'}: the cached decode agrees with the full forward at birth",
              torch.allclose(lg[:, 0], ev[:, -2], atol=1e-5) and torch.allclose(st[:, 0], ev[:, -1], atol=1e-5))
    plain = set(_model("").state_dict())
    want = {"dial": {"head.w_t", "head.a_bar"}, "dialraw": {"head.w_t", "head.a_bar"}, "dialplus": {"head.w_t", "head.a_bar"},
            "pergain": {"head.w_g.weight", "head.a_bar"}}
    for mode in FORMS:
        keys = set(_model(mode).state_dict())
        check(f"1c {mode}: the state dict is the plain head's keys plus the dial's", keys == plain | want[mode], str(sorted(keys ^ plain)))


def case_2():
    x = _ids()
    for mode in FORMS:
        m = _model(mode)
        _move(m, 1.0)
        m.eval()
        ref = _plain_readout(m, x)
        with torch.no_grad():
            full = m(x)[0]
            off = m(x, disable_head_aleph=True)[0]
        check(f"2 {mode}: the toggle is the plain readout and the moved dial differs from it",
              torch.equal(off, ref) and not torch.allclose(full, ref), f"max|diff| {float((full - ref).abs().max()):.3g}")


def case_3():
    x = _ids()
    for band in (4.0, 2.0):
        for mode in ("dial", "pergain"):
            m = _model(mode, head_dial_band=band)
            _move(m, 50.0)
            m.eval()
            with torch.no_grad():
                h = m.nf(m._trunk(m.embed(x), x))
                lg = m.head.dial_log(m.head.addr.signed(m.head.proj(h)))
                full, base = m.head(h), m.head(h, disable_aleph=True)
            scale = torch.exp(lg)
            ratio = (full / base.where(base.abs() > 1e-6, torch.full_like(base, float("nan")))).nan_to_num(1.0)
            check(f"3 {mode} band {band:g}: the scale stays within [1/band, band] and multiplies the readout",
                  float(scale.max()) <= band + 1e-5 and float(scale.min()) >= 1 / band - 1e-6 and float(scale.max()) > band * 0.9
                  and torch.allclose(full, base * scale.to(base.dtype), atol=1e-4), f"scale [{float(scale.min()):.3f}, {float(scale.max()):.3f}]")


def case_4():
    x = _ids()
    for mode in ("dial", "dialraw", "pergain"):
        m = _model(mode)
        _move(m, 1.0)
        m.train()
        with torch.no_grad():
            h = m.nf(m._trunk(m.embed(x), x))
            s = m.head.addr.signed(m.head.proj(h))
            w = m.head.dial_weight
            a = F.linear(s.float(), w.float()) if mode == "pergain" else (s.float() @ w.float()).unsqueeze(-1)
            mean = a.reshape(-1, a.shape[-1]).mean(0) if mode == "pergain" else a.mean()
            before = m.head.a_bar.clone()
            m.head(h)
            after1 = m.head.a_bar.clone()
            m.head(h)
            after2 = m.head.a_bar.clone()
            m.eval()
            m.head(h)
            frozen = m.head.a_bar.clone()
        rho = m.head.rho
        if mode == "dialraw":
            check("4 dialraw: the running mean stays at zero", float(before.abs().max()) == 0 and float(after2.abs().max()) == 0)
        else:
            check(f"4a {mode}: one detached step toward the forward's mean a training forward",
                  torch.allclose(after1, rho * mean, atol=1e-6) and torch.allclose(after2, (1 - rho) * after1 + rho * mean, atol=1e-6)
                  and tuple(after1.shape) == ((256,) if mode == "pergain" else ()))
            check(f"4b {mode}: frozen in eval", torch.equal(frozen, after2))


def case_5():
    for mode in HEAD_MODES:
        m = _model(mode)
        named = dict(m.named_parameters())
        muon, adam = split_params(m)
        mi, ai = {id(p) for p in muon}, {id(p) for p in adam}
        on_muon = all(id(named[k]) in mi for k in ("head.proj.weight", "head.addr.codebook", "head.w_h.weight", "head.w_s.weight"))
        dial_adam = (mode == "" or mode == "pergain" or id(named["head.w_t"]) in ai)
        g_muon = (mode != "pergain" or id(named["head.w_g.weight"]) in mi)
        frozen = named["head.w_s.weight"].requires_grad == (mode in ("", "dialplus"))
        check(f"5a {mode or 'plain'}: the split (proj / codebook / W_h / W_s Muon; w_t Adam; G Muon; W_s frozen in the modulating forms)",
              on_muon and dial_adam and g_muon and frozen)
    m = _model("pergain")
    named = dict(m.named_parameters())
    opts = build_optimizers(m, decay_prefixes=("head.w_h.", "head.w_s.", "head.w_g."), decay_wd=0.1)
    g1 = opts[1].param_groups[1]
    want = {id(named[k]) for k in ("head.w_h.weight", "head.w_s.weight", "head.w_g.weight")}
    check("5b pergain: the named readout weights (G among them) form the AdamW group; Muon keeps the rest",
          {id(p) for p in g1["params"]} == want and g1["weight_decay"] == 0.1 and isinstance(opts[0], Muon)
          and not want & {id(p) for p in opts[0].param_groups[0]["params"]})
    m = _model("dial")
    named = dict(m.named_parameters())
    opts = build_optimizers(m, decay_prefixes=("head.w_h.", "head.w_s."), decay_wd=0.1)
    g1 = opts[1].param_groups[1]
    check("5c dial: the readout's two matrices form the AdamW group and w_t sits in the decay-0 group",
          {id(p) for p in g1["params"]} == {id(named["head.w_h.weight"]), id(named["head.w_s.weight"])}
          and id(named["head.w_t"]) in {id(p) for p in opts[1].param_groups[0]["params"]})


def case_6():
    x = _ids()
    for mode in FORMS:
        m = _model(mode)
        m.train()
        _, loss = m(x[:, :-1], targets=x[:, 1:])
        loss.backward()
        w = m.head.dial_weight
        g = w.grad
        ws = m.head.w_s.weight.grad
        check(f"6 {mode}: the loss reaches the dial at birth" + ("" if mode == "dialplus" else "; none reaches the frozen W_s"),
              g is not None and float(g.abs().max()) > 0 and torch.isfinite(g).all() and (mode == "dialplus" or ws is None),
              f"|grad| max {float(g.abs().max()):.3g}")


def case_7():
    x = _ids()
    plain = _model("", seed=5)
    with torch.no_grad():
        plain.head.w_s.weight.normal_()                      # a carrying read that learned something: it must not leak into the dial forms
    sd = plain.state_dict()
    for mode in FORMS:
        m = _model(mode, seed=9)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        m.eval()
        with torch.no_grad():
            out = m(x)[0]
            ref = m(x, disable_head_aleph=True)[0]
            ref_plain = plain.eval()(x, disable_head_aleph=True)[0]
        ok = not unexpected and set(missing) <= {"head.w_t", "head.a_bar", "head.w_g.weight"} and torch.equal(ref, ref_plain)
        if mode == "dialplus":
            ok = ok and not torch.equal(out, ref)          # the carrying read loaded: dialplus carries it (the dial at 1)
        else:
            ok = ok and torch.equal(out, ref)              # the carrying read is loaded but frozen and unused: the plain readout
        check(f"7 {mode}: a plain state loads (the dial's keys missing) and the head reads as the plain readout" + (" plus W_s" if mode == "dialplus" else ""), ok, f"missing {sorted(missing)}")


def case_8():
    x = _ids()
    for mode in ("dial", "pergain"):
        m = _model(mode)
        with torch.no_grad():
            h = m.nf(m._trunk(m.embed(x), x))
        z = m.head.dial_stats(h)
        _move(m, 50.0)
        big = m.head.dial_stats(h)
        check(f"8 {mode}: dial_stats (zero: mean 0, spread 0, nothing saturated, weights 0; huge: saturated)",
              z["dial_log_mean"] == 0 and z["dial_log_std"] == 0 and z["dial_sat"] == 0 and z["dial_w_norm"] == 0
              and big["dial_sat"] > 0.5 and big["dial_w_norm"] > 0 and abs(big["dial_log_mean"]) <= math.log(4.0) + 1e-6,
              f"huge: sat {big['dial_sat']:.2f}, log mean {big['dial_log_mean']:.3f}, std {big['dial_log_std']:.3f}")
    check("8c the plain head reads no dial", _model("").head.dial_stats(torch.zeros(2, 3, TINY["d_model"])) == {})


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
