"""The census of the night rigs' bodies (0.10.18 rc9, 2026-10-14): instruments.model_census replays a block through every form a craft of
the nano bed carries, so the library trainer's health readout and full eval run on those crafts.
Run: python -m geolip.alephllm.tests.test_census_rc9   (CPU, seconds; no download)
Cases:
  1 the dual hub: the census reads both hubs book by book (the multi-constellation aggregation: the merge pairs summed, the max keys
    the worst book, the rest the mean), the consumed erank over both hubs' oriented reads; it equals the aggregation written out
  2 the bank's byte-class table: the census replay passes the input bytes' classes (as AlephLM._trunk does), so a craft with the
    table ("add") is read; the replay's last block equals the model's own trunk output (the hidden RMS)
  3 every instrument of the trainer's full eval (the census, the toggle ledger, the canaries, the special-token gauge, the bits by
    position) runs on the nano v1 bodies' forms: dual hubs with Canon layers, Pollux with Canon layers, dual hubs, Castor; all with the
    byte-class table added
"""
from __future__ import annotations

import torch

from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig
from ..train import instruments

RESULTS: list = []
TINY = dict(name="tiny-census", d_model=64, n_layers=4, n_heads=4, context=128, hub_layers=(0, 2), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _idx(seed=3, B=2, T=128):
    return torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(seed))


def case_1():
    torch.manual_seed(0)
    m = AlephLM(AlephLMConfig(**dict(TINY, hub_dual=True))).eval()
    idx = _idx()
    torch.manual_seed(1)
    cen = instruments.model_census(m, idx)
    li = cen["layers"][0]
    books = instruments._hub_books(m.blocks[0].attn)
    check("1a a dual hub's books are its rotary hub's then its plain hub's",
          len(books) == 2 and books[0] is m.blocks[0].attn.rot and books[1] is m.blocks[0].attn.plain)
    with torch.no_grad():
        x = m.embed(idx)
        tap = m.blocks[0].n1(x)
        hs = [b.addr.health(b.q(tap)) for b in books]
    want = {}
    for k in hs[0]:
        vals = [h[k] for h in hs]
        want[k] = sum(vals) if k == "anchor_merge_pairs" else (max(vals) if "max" in k else sum(vals) / len(vals))
    check("1b the dual hub's address health is the books' aggregation written out", li["hub_addr"] == want,
          f"{sorted(want)[:3]}...")
    check("1c the consumed erank reads both hubs", isinstance(li.get("hub_consumed_erank"), float) and li["hub_consumed_erank"] > 0)


def case_2():
    torch.manual_seed(0)
    m = AlephLM(AlephLMConfig(**dict(TINY, bank_byte_table="add"))).eval()
    idx = _idx()
    try:
        cen = instruments.model_census(m, idx)
        ok = True
    except AssertionError as e:
        ok, cen = False, None
        print("   census:", e)
    check("2a a craft with the byte-class table is read by the census", ok)
    if cen is None:
        return
    with torch.no_grad():
        h = m._trunk(m.embed(idx), idx)
    rms = float(h[:2].float().pow(2).mean().sqrt())
    last = cen["layers"][len(m.blocks) - 1]["hidden_rms"]
    check("2b the replay's last block equals the model's own trunk (hidden RMS)", abs(last - rms) <= 1e-5 * max(1.0, rms),
          f"{last:.6f} vs {rms:.6f}")


def case_3():
    from ..data.tokenizers import build_tokenizer
    from ..eval.canaries import canary_eval
    tok = build_tokenizer("byte-trigram")
    L = 6
    forms = {"dual hubs + Canon": dict(hub_layers=(0, 2, 4), attn_rotary=True, hub_dual=True, canon="AC"),
             "Pollux + Canon": dict(hub_layers=(0, 2, 4), attn_rotary=True, hub_rotary=True, canon="AC"),
             "dual hubs": dict(hub_layers=(0, 2, 4), attn_rotary=True, hub_dual=True),
             "Castor": dict(attn_rotary=True)}
    val = [torch.randint(0, 256, (2, 257), generator=torch.Generator().manual_seed(7))]
    for label, extra in forms.items():
        torch.manual_seed(0)
        cfg = dict(TINY, n_layers=L, context=256, hub_layers=(), bank_experts=7, bank_byte_table="add")
        m = AlephLM(AlephLMConfig(**dict(cfg, **extra)))
        try:
            cen = instruments.model_census(m, val[0][:2, :-1])
            led = instruments.toggle_ledger(m, val)
            can = canary_eval(m, tok, "cpu", episodes=2, context=256)
            sp = instruments.special_token_gauge(m, val)
            pb = instruments.position_bits_gauge(m, val)
            instruments.readout(1, 0, cen, led)
            ok = len(cen["layers"]) == L and "bpb_full" in led and "recall_acc" in can and pb is not None and sp is not None
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"   {label}:", type(e).__name__, e)
        check(f"3 the full eval's instruments run on {label} (with the byte-class table)", ok)


if __name__ == "__main__":
    for fn in (case_1, case_2, case_3):
        fn()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks pass")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
