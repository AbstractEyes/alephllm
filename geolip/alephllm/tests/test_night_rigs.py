"""The night rigs (0.10.18, 2026-10-12): rotary in attention and in the address, the position-table switch, the bank's
byte-class table (experiment 3's small bed).
Run: python -m geolip.alephllm.tests.test_night_rigs   (CPU, seconds; no download)
Cases:
  1 flags off = 0.10.17 bit for bit: the config's new keys at their defaults; no new state-dict key; CausalSDPA equals the
    inline sdpa arithmetic; the hub's fast path equals its cumsum oracle; the bank equals its written formula
  2 rotary attention: the q.k logit of two positions depends on their distance alone; causal; finite; decode parity (prefill +
    steps == the full pass); a model with the flag carries it in every block and arm
  3 rotary address: the chunked hub equals the cumsum oracle (the pad path exercised); the agreement mass equals the plain
    hub's bit for bit (den unrotated); the rotated code product is relative; prefill refuses (decode owed)
  4 pos_table False: the table zero and frozen; the forward equals the trigram sum alone; the state-dict key stays
  5 the byte-class table: the class of every byte kind; "fixed" dispatch = the one-hot rows; "add0" at birth = "" bit for
    bit; "add" = address + table; the plumbing through AlephLM (classes from idx reach every bank; disable_bank; the
    hub_ckpt=2 path under grad); decode refuses with the table on
  6 every flag on: a tiny Adam step, finite loss, gradients reach the table and the rotary blocks, none reach the frozen table
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..model.attention import CausalSDPA, CausalSplatHUB, _Rope, rope_half
from ..model.bank import AnchoredBank, BYTE_CLASSES, byte_class_table, byte_classes
from ..model.embedding import TrigramByteEmbedding, PAD_ROW
from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig

RESULTS: list = []
TINY = dict(name="tiny-night", d_model=64, n_layers=3, n_heads=4, context=128, hub_layers=(1,), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def close(a, b, tol=1e-5):
    return bool(torch.isfinite(a).all()) and float((a - b).abs().max()) <= tol


# ------------------------------------------------------------------ 1 flags off
def case_1():
    torch.manual_seed(0)
    cfg = AlephLMConfig(**TINY)
    d = cfg.to_dict()
    defaults = dict(attn_rotary=False, hub_rotary=False, rotary_base=10000.0, pos_table=True, bank_byte_table="",
                    bank_last_fix="", fix_rho=0.0625)
    check("1a the config's new keys at their defaults", all(d[k] == v for k, v in defaults.items()), str({k: d[k] for k in defaults}))
    m = AlephLM(cfg)
    keys = list(m.state_dict().keys())
    bad = [k for k in keys if "byte_amp" in k or "rope" in k or "lastfix" in k or "use_pos" in k]
    check("1b no new state-dict key with the flags off", not bad, str(bad))
    # CausalSDPA == inline sdpa
    attn = CausalSDPA(64, 4)
    x = torch.randn(2, 20, 64)
    with torch.no_grad():
        y = attn(x)
        q, k, v = attn.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(2, 20, 4, 16).transpose(1, 2) for t in (q, k, v))
        ref = attn.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 20, 64))
    check("1c CausalSDPA (flags off) is the inline sdpa bit for bit", torch.equal(y, ref))
    hub = CausalSplatHUB(64, 16, 8, 0.1, chunk=16)
    with torch.no_grad():
        check("1d the plain hub's fast path equals its cumsum oracle", close(hub(x), hub.forward_naive(x), 1e-5))
    bank = AnchoredBank(64, 3)
    with torch.no_grad():
        yb = bank(x)
        w = bank.addr.signed(x)
        h = F.gelu(torch.einsum("bnd,kdf->bnkf", x, bank.w_in))
        e = torch.einsum("bnkf,kfd->bnkd", h, bank.w_out)
        ref = bank.t_out(F.gelu(bank.t_in(x))) + torch.einsum("bnk,bnkd->bnd", w * torch.sigmoid(bank.gates), e)
    check("1e the plain bank is its written formula bit for bit", torch.equal(yb, ref))


# ------------------------------------------------------------------ 2 rotary attention
def case_2():
    torch.manual_seed(1)
    attn = CausalSDPA(64, 4, rotary=True)
    rope = attn._rope
    q, k = torch.randn(16), torch.randn(16)
    vals = []
    for m in (3, 10, 50, 100):
        n = m - 3
        cq, sq = rope.tables(1, m, q.device, q.dtype)
        ck, sk = rope.tables(1, n, k.device, k.dtype)
        vals.append(float(rope_half(q[None], cq, sq)[0] @ rope_half(k[None], ck, sk)[0]))
    check("2a the rotated q.k logit depends on the distance alone", max(vals) - min(vals) < 1e-4, f"{[round(v, 6) for v in vals]}")
    x = torch.randn(1, 24, 64)
    with torch.no_grad():
        full = attn(x)
        x2 = x.clone()
        x2[:, 20:] += 1.0
        full2 = attn(x2)
        check("2b causal and finite", torch.isfinite(full).all() and torch.equal(full[:, :20], full2[:, :20]))
        logits, cache = attn.prefill(x[:, :16])
        outs = [logits]
        for t in range(16, 24):
            outs.append(attn.step(x[:, t:t + 1], cache))
        dec = torch.cat(outs, dim=1)
    check("2c decode parity (prefill + steps == the full pass)", close(dec, full, 1e-5), f"max dev {float((dec - full).abs().max()):.2e}")
    cfg = AlephLMConfig(**dict(TINY, attn_rotary=True, lookup_arm_sites=(0,)))
    m = AlephLM(cfg)
    flags = [b.attn.rotary for b in m.blocks if not b.is_hub] + [a.rotary for a in m.lookup_arms.values()]
    check("2d the flag reaches every softmax block and arm", all(flags) and len(flags) == 3, str(flags))
    with torch.no_grad():
        out = m(torch.randint(0, 256, (2, 32)))[0]
    check("2e a rotary craft's forward is finite", bool(torch.isfinite(out).all()))


# ------------------------------------------------------------------ 3 rotary address
def case_3():
    torch.manual_seed(2)
    hub = CausalSplatHUB(64, 16, 8, 0.1, chunk=16, rotary=True)
    plain = CausalSplatHUB(64, 16, 8, 0.1, chunk=16)
    plain.load_state_dict(hub.state_dict())
    x = torch.randn(2, 40, 64)                 # 40 = 2.5 chunks: the pad path
    with torch.no_grad():
        y = hub(x)
        den_r = hub._den_raw[0].clone()
        yn = hub.forward_naive(x)
        yp = plain(x)
        den_p = plain._den_raw[0].clone()
    check("3a the chunked rotary hub equals its cumsum oracle", close(y, yn, 2e-5), f"max dev {float((y - yn).abs().max()):.2e}")
    check("3b the agreement mass is the plain hub's bit for bit", torch.equal(den_r, den_p))
    check("3c the rotary changes the read", float((y - yp).abs().max()) > 1e-4)
    c1, c2 = torch.rand(1, 1, 32), torch.rand(1, 1, 32)
    dots = []
    for s in (0, 7, 100):
        a = hub._rotate_codes(c1, offset=s + 5)
        b = hub._rotate_codes(c2, offset=s)
        dots.append(float((a * b).sum()))
    check("3d the rotated code product depends on the distance alone", max(dots) - min(dots) < 1e-5, str([round(v, 6) for v in dots]))
    try:
        hub.prefill(x)
        refused = False
    except AssertionError:
        refused = True
    check("3e prefill with the address rotary refuses (decode owed)", refused)
    cfg = AlephLMConfig(**dict(TINY, hub_rotary=True))
    m = AlephLM(cfg)
    check("3f the flag reaches the hub blocks only", m.blocks[1].attn.rotary and not getattr(m.blocks[0].attn, "rotary", False))
    half = CausalSplatHUB(64, 16, 8, 0.1, chunk=16, rotary=True, rotary_frac=0.5)
    half.load_state_dict(hub.state_dict())
    with torch.no_grad():
        yh = half(x)
        yhn = half.forward_naive(x)
    check("3g a half-rotated address: the chunked form equals its oracle and differs from both the full and the plain read",
          close(yh, yhn, 2e-5) and float((yh - y).abs().max()) > 1e-4 and float((yh - yp).abs().max()) > 1e-4 and half.n_rot == 8)
    cfg2 = AlephLMConfig(**dict(TINY, hub_rotary=True, hub_rotary_frac=0.25))
    check("3h the fraction reaches the hub and the config round trip keeps it", AlephLM(cfg2).blocks[1].attn.n_rot == 4 and cfg2.to_dict()["hub_rotary_frac"] == 0.25)


# ------------------------------------------------------------------ 4 the position table switch
def case_4():
    torch.manual_seed(3)
    e = TrigramByteEmbedding(32, 64, pos_table=False)
    idx = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        y = e(idx)
        ref = e.emb0(idx) + e.emb1(F.pad(idx, (1, 0), value=PAD_ROW)[:, :-1]) + e.emb2(F.pad(idx, (2, 0), value=PAD_ROW)[:, :-2])
    check("4a the table is zero and frozen", bool((e.pos == 0).all()) and not e.pos.requires_grad)
    check("4b the forward is the trigram sum alone", torch.equal(y, ref))
    check("4c the state dict keeps the key", "pos" in e.state_dict())
    torch.manual_seed(3)
    e2 = TrigramByteEmbedding(32, 64, pos_table=True)
    with torch.no_grad():
        check("4d pos_table True is the form before (table added)", torch.equal(e2(idx), ref + e2.pos[:, :16]))


# ------------------------------------------------------------------ 5 the byte-class table
def case_5():
    t = byte_class_table()
    exp = {ord("a"): 0, ord("Z"): 1, ord("7"): 2, 32: 3, 10: 4, ord("!"): 5, 0: 5, 0xC3: 6, 0xFF: 6}
    check("5a the class of every byte kind", all(int(t[b]) == c for b, c in exp.items()), str({b: int(t[b]) for b in exp}))
    cls = torch.arange(7)[None]
    bank = AnchoredBank(16, 3, byte_table="fixed")
    x = torch.randn(1, 7, 16)
    with torch.no_grad():
        w = bank.dispatch(x, cls)
    want = torch.zeros(7, 3)
    for c in range(7):
        want[c, c % 3] = 1.0
    check("5b fixed dispatch = the one-hot rows", torch.equal(w[0], want) and "byte_amp" in bank.state_dict() and not isinstance(bank.byte_amp, torch.nn.Parameter))
    torch.manual_seed(4)
    b0 = AnchoredBank(16, 3)
    torch.manual_seed(4)
    b1 = AnchoredBank(16, 3, byte_table="add0")
    x = torch.randn(2, 7, 16)
    with torch.no_grad():
        check("5c add0 at birth is the plain bank bit for bit", torch.equal(b0(x), b1(x, byte_cls=cls.expand(2, 7))))
        b2 = AnchoredBank(16, 3, byte_table="add")
        check("5d add = address + table", torch.equal(b2.dispatch(x, cls.expand(2, 7)), b2.addr.signed(x) + want[None].expand(2, 7, 3)) and isinstance(b2.byte_amp, torch.nn.Parameter))
    cfg = AlephLMConfig(**dict(TINY, bank_byte_table="fixed"))
    m = AlephLM(cfg)
    idx = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        out = m(idx)[0]
        seen = [torch.equal(b.bank.last_dispatch, b.bank.byte_amp[byte_classes(idx)]) for b in m.blocks]
    check("5e every bank saw the input bytes' classes", all(seen) and bool(torch.isfinite(out).all()), str(seen))
    with torch.no_grad():
        off = m(idx, disable_bank=True)[0]
    check("5f disable_bank path runs with the table on", bool(torch.isfinite(off).all()))
    cfg2 = AlephLMConfig(**dict(TINY, bank_byte_table="add", hub_ckpt=2))
    m2 = AlephLM(cfg2)
    m2.train()
    loss = m2(idx[:, :-1], idx[:, 1:]).loss
    loss.backward()
    check("5g the hub_ckpt=2 path carries the classes under grad", bool(torch.isfinite(loss)) and m2.blocks[0].bank.byte_amp.grad is not None)
    try:
        m.prefill(idx)
        refused = False
    except AssertionError:
        refused = True
    check("5h decode with the table on refuses (owed)", refused)


# ------------------------------------------------------------------ 6 every flag on
def case_6():
    torch.manual_seed(5)
    cfg = AlephLMConfig(**dict(TINY, attn_rotary=True, hub_rotary=True, pos_table=False, bank_byte_table="add"))
    m = AlephLM(cfg)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    idx = torch.randint(0, 256, (2, 64))
    m.train()
    loss = m(idx[:, :-1], idx[:, 1:]).loss
    loss.backward()
    # at birth the dispatch has NO gradient (the experts' output projections are zero: the null path), so the table's
    # gradient is exactly zero on the first step and the expert outputs' is not; after one step the table moves
    g_tab0 = m.blocks[0].bank.byte_amp.grad
    g_out = m.blocks[0].bank.w_out.grad
    g_qkv = m.blocks[0].attn.qkv.weight.grad
    check("6a the first step with every flag on: finite loss, the null path (table grad exactly zero), gradients at the experts and the rotary block",
          bool(torch.isfinite(loss)) and g_tab0 is not None and float(g_tab0.abs().sum()) == 0.0
          and g_out is not None and float(g_out.abs().sum()) > 0 and g_qkv is not None and float(g_qkv.abs().sum()) > 0,
          f"loss {loss.item():.3f}")
    check("6b the frozen position table takes no gradient", m.embed.pos.grad is None and not m.embed.pos.requires_grad)
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss2 = m(idx[:, :-1], idx[:, 1:]).loss
    loss2.backward()
    g_tab1 = m.blocks[0].bank.byte_amp.grad
    check("6c the second step: the table takes gradient once the experts write", g_tab1 is not None and float(g_tab1.abs().sum()) > 0)
    check("6d the param count counts the table once", m.param_count() == sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3, case_4, case_5, case_6):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
