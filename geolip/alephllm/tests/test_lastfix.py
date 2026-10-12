"""The last block's write fix (0.10.18, 2026-10-12; model/lastfix.py: the amplitude register and its comparison forms).
Run: python -m geolip.alephllm.tests.test_lastfix   (CPU, seconds)
Cases:
  1 flag off: no module, no state-dict key, the model is the plain form
  2 the register on a stream of writes: the direction tracks the mean write; the gain is exactly 1 through the warm-in; after it a
    carrier grown 3x is held near the reference (the gain clamps); the per-byte ratios survive the bound (no per-position
    normalization); eval freezes the statistics; the head path is zero-born and takes gradient
  3 register at birth in a model: logits within 1e-4 of the plain model (gamma 1, the decomposition's rounding only)
  4 gain at birth: bit for bit the plain model; after a warm-in the whole write is scaled by one bounded gain
  5 temperature: the carrier is removed from the stream; tau = 1 at birth (the head untouched); the parameters move under a step
  6 normwrite: causal (later positions never change earlier outputs); finite; decode refuses
  7 decode parity in eval for register / gain / temperature after training forwards (prefill + steps == the full pass)
  8 a training step in every mode: finite loss; hub_ckpt=2 keeps one slow update per step; the config round trip
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.lastfix import LastWriteFix
from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig

RESULTS: list = []
TINY = dict(name="tiny-fix", d_model=64, n_layers=3, n_heads=4, context=128, hub_layers=(1,), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def close(a, b, tol):
    return bool(torch.isfinite(a).all()) and float((a - b).abs().max()) <= tol


def same_model(**flags):
    torch.manual_seed(11)
    plain = AlephLM(AlephLMConfig(**TINY))
    torch.manual_seed(11)
    fixed = AlephLM(AlephLMConfig(**dict(TINY, **flags)))
    sd = plain.state_dict()
    missing = fixed.load_state_dict(sd, strict=False)
    return plain, fixed, missing


def case_1():
    m = AlephLM(AlephLMConfig(**TINY))
    check("1a flag off: no module", m.last_fix is None and all(b.fix is None for b in m.blocks))
    check("1b flag off: no fix key in the state dict", not [k for k in m.state_dict() if ".fix." in k])


def case_2():
    torch.manual_seed(1)
    d = 32
    fx = LastWriteFix(d, 256, "register", rho=0.5, warm=4, clamp=(0.25, 4.0))
    u = F.normalize(torch.randn(d), dim=0)
    fx.train()
    gam = []
    for step in range(12):
        scale = 1.0 if step < 6 else 3.0                      # the carrier grows 3x after the warm-in
        a_true = scale * (10.0 + torch.rand(2, 8))            # per-byte amplitudes, all positive
        w = a_true.unsqueeze(-1) * u + 0.3 * torch.randn(2, 8, d)
        out = fx(w)
        gam.append(float(fx.gamma()))
        if step == 0:
            check("2a the direction tracks the mean write", float(fx.direction() @ u) > 0.99, f"cos {float(fx.direction() @ u):.4f}")
    check("2b the gain is exactly 1 through the warm-in", all(g == 1.0 for g in gam[:3]), str(gam[:5]))
    a_eff = (out @ fx.direction())
    held = float(a_eff.mean()) / float(fx.a_ref)
    check("2c after the warm-in a 3x carrier is held near the reference", 0.8 < held < 1.6 and gam[-1] < 0.6, f"effective/ref {held:.2f}, gamma {gam[-1]:.3f}")
    a_raw = w @ fx.direction()                               # the carrier the forward saw (the noise included)
    r_in = (a_raw[0, 1] / a_raw[0, 0]).item()
    r_out = (a_eff[0, 1] / a_eff[0, 0]).item()
    check("2d the per-byte ratios survive the bound (one gain for every byte of the step)", abs(r_in - r_out) < 1e-4 * abs(r_in), f"{r_in:.5f} vs {r_out:.5f}")
    fx.eval()
    n0 = int(fx.n_updates)
    fx(w)
    check("2e eval freezes the statistics", int(fx.n_updates) == n0)
    logits = torch.zeros(2, 8, 256, requires_grad=True)
    adj = fx.head_adjust(logits)
    check("2f the head path is zero-born", torch.equal(adj, logits))
    adj.sum().backward()
    fx.train()
    w = (10 + torch.rand(2, 8)).unsqueeze(-1) * u
    fx(w)
    lg = torch.randn(2, 8, 256)
    loss = fx.head_adjust(lg).logsumexp(-1).mean()
    loss.backward()
    check("2g the head path takes gradient", fx.g2.weight.grad is not None and float(fx.g2.weight.grad.abs().sum()) > 0)
    st = fx.stats()
    check("2h stats", st["mode"] == "register" and st["n_updates"] > 0 and st["a_ref"] > 0, str(st))


def case_3():
    plain, fixed, miss = same_model(bank_last_fix="register", fix_warm=2)
    idx = torch.randint(0, 256, (2, 48))
    plain.eval()
    fixed.eval()
    with torch.no_grad():
        a = plain(idx)[0]
        fixed.train()
        fixed(idx)                               # one training forward seeds the direction
        fixed.eval()
        b = fixed(idx)[0]
    check("3a register at birth: within 1e-4 of the plain model", close(a, b, 1e-4), f"max dev {float((a - b).abs().max()):.2e}; missing keys {len(miss.missing_keys)}")
    check("3b the fix keys are the only new keys", all(".fix." in k for k in miss.missing_keys) and not miss.unexpected_keys, str(miss.missing_keys[:3]))


def case_4():
    plain, fixed, _ = same_model(bank_last_fix="gain", fix_warm=3)
    idx = torch.randint(0, 256, (2, 48))
    plain.eval()
    fixed.eval()
    with torch.no_grad():
        check("4a gain at birth is the plain model bit for bit", torch.equal(plain(idx)[0], fixed(idx)[0]))
    fx = fixed.last_fix
    fixed.train()
    with torch.no_grad():
        for _ in range(3):
            fixed(idx)
        with torch.no_grad():
            for p in fixed.blocks[-1].bank.parameters():
                p.mul_(2.0)                       # the write doubles
        fixed(idx)
        g = float(fx.gamma())
    check("4b after the warm-in a doubled write meets a gain below 1", 0.3 < g < 0.95, f"gamma {g:.3f}")


def case_5():
    torch.manual_seed(3)
    fx = LastWriteFix(32, 256, "temperature", warm=1)
    fx.train()
    u = F.normalize(torch.randn(32), dim=0)
    w = (5 + torch.rand(2, 8)).unsqueeze(-1) * u + 0.1 * torch.randn(2, 8, 32)
    out = fx(w)
    check("5a the carrier is removed from the stream", float((out @ fx.direction()).abs().max()) < 1e-3)
    lg = torch.randn(2, 8, 256)
    check("5b tau = 1 at birth: the head untouched", close(fx.head_adjust(lg), lg, 1e-6))
    loss = (fx.head_adjust(lg) ** 2).mean()
    loss.backward()
    check("5c theta takes gradient", fx.theta.grad is not None and float(fx.theta.grad.abs().sum()) > 0)


def case_6():
    torch.manual_seed(4)
    fx = LastWriteFix(32, 256, "normwrite")
    fx.eval()
    w = torch.randn(1, 20, 32)
    out = fx(w)
    w2 = w.clone()
    w2[:, 12:] += 3.0
    out2 = fx(w2)
    check("6a normwrite is causal", torch.equal(out[:, :12], out2[:, :12]) and bool(torch.isfinite(out).all()))
    m = AlephLM(AlephLMConfig(**dict(TINY, bank_last_fix="normwrite")))
    try:
        m.prefill(torch.randint(0, 256, (1, 8)))
        refused = False
    except AssertionError:
        refused = True
    check("6b decode with normwrite refuses (owed)", refused)


def case_7():
    idx = torch.randint(0, 256, (1, 40))
    for mode in ("register", "gain", "temperature"):
        torch.manual_seed(5)
        m = AlephLM(AlephLMConfig(**dict(TINY, bank_last_fix=mode, fix_warm=2)))
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(3):
            loss = m(idx[:, :-1], idx[:, 1:]).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            full = m(idx)[0]
            logits, cache = m.prefill(idx[:, :30])
            outs = [logits]
            for t in range(30, 40):
                outs.append(m.decode_step(idx[:, t], cache))
            dec = torch.cat(outs, dim=1)
        check(f"7 decode parity ({mode})", close(dec, full[:, 29:], 1e-4), f"max dev {float((dec - full[:, 29:]).abs().max()):.2e}")


def case_8():
    idx = torch.randint(0, 256, (2, 32))
    for mode in ("register", "temperature", "gain", "normwrite"):
        torch.manual_seed(6)
        m = AlephLM(AlephLMConfig(**dict(TINY, bank_last_fix=mode, fix_warm=1)))
        m.train()
        loss = m(idx[:, :-1], idx[:, 1:]).loss
        loss.backward()
        check(f"8a a training step ({mode}) is finite", bool(torch.isfinite(loss)), f"{loss.item():.3f}")
    m = AlephLM(AlephLMConfig(**dict(TINY, bank_last_fix="register", hub_ckpt=2)))
    m.train()
    loss = m(idx[:, :-1], idx[:, 1:]).loss
    loss.backward()
    check("8b hub_ckpt=2 keeps one slow update per step", int(m.last_fix.n_updates) == 1, str(int(m.last_fix.n_updates)))
    d = AlephLMConfig(**dict(TINY, bank_last_fix="register")).to_dict()
    cfg2 = AlephLMConfig(**{k: (tuple(v) if k in ("hub_layers", "lookup_arm_sites", "fix_clamp") else v) for k, v in d.items()})
    check("8c the config round trip", cfg2.bank_last_fix == "register" and tuple(cfg2.fix_clamp) == (0.25, 4.0))


def case_9():
    from ..presets import PRESETS, LASTFIX_BED, LASTFIX_MODES
    base = PRESETS["mini-beatrix-2s"]
    ok_all = True
    for name, kw in LASTFIX_BED.items():
        p = PRESETS[name]
        same_recipe = (p.train.micro_batch == base.train.micro_batch and p.train.grad_accum == base.train.grad_accum
                       and p.train.governor == base.train.governor and p.train.head_addr_frozen == base.train.head_addr_frozen
                       and p.curriculum == base.curriculum and p.model.d_model == base.model.d_model
                       and p.model.hub_layers == base.model.hub_layers and p.hf_repo == base.hf_repo)
        ok = same_recipe and p.model.name == name and p.model.bank_last_fix == kw["mode"] and p.train.seed == kw.get("seed", base.train.seed)
        ok_all &= ok
        check(f"9a {name}: the 2s recipe verbatim, the fix {kw['mode']!r}, seed {p.train.seed}", ok)
    check("9b the base 2s preset is untouched", base.model.bank_last_fix == "" and base.train.seed == 1337 and base.model.name == "mini-beatrix-2s")
    check("9c every fix mode has a preset", all(any(kw["mode"] == m for kw in LASTFIX_BED.values()) for m in LASTFIX_MODES))
    cfg = PRESETS["mini-beatrix-2s-fix-register"].model
    tiny = AlephLMConfig(**dict(TINY, bank_last_fix=cfg.bank_last_fix, fix_rho=cfg.fix_rho, fix_warm=cfg.fix_warm, fix_clamp=cfg.fix_clamp))
    m = AlephLM(tiny)
    check("9d the preset's fix fields build the module at the last block", m.last_fix is not None and m.last_fix.mode == "register" and m.blocks[0].fix is None)


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8, case_9):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
