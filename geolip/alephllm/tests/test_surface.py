"""Surface arms: the spelling, the sites, the paired rows, the losses, the hand-run states and the gauge.
Run: python -m geolip.alephllm.tests.test_surface   (CPU, seconds; no download; no transformers: a stand-in tokenizer)
Cases:
  1 the byte map: a bijection over 256 bytes; ' ' -> 'Ġ' (C4 A0 in UTF-8), '\\n' -> 'Ċ'
  2 spell + sites under a stand-in byte-level tokenizer (a token = a run of non-space bytes with its leading space, every
    further space its own token; pieces as GPT-2 spells bytes): the round trip; B longer than A by the spaces and the high
    bytes; the closing byte on A is the first byte of the next token's expansion, on B the first byte of its spelling; the
    first and the last token give no site; the instrument's example ' taco' -> 'Ġtaco' (six bytes against five)
  3 PairedRows: both rows ctx + 1 long, DOC first, the same documents in the same order; every site inside both rows, the
    byte before each closing byte = the token's last byte on each surface; single_rows the same per document
  4 the losses: identical states -> mse 0 and cosine 1; the InfoNCE of the true pairing is below a shuffled pairing's;
    surface_loss's per-block record
  5 block_states equals the model's own forward bit for bit on a tiny trunk, bare and with two arms attached (one masked),
    with and without gathering at sites; logits_parity 0.0
  6 paired_alignment: a rotated copy with small noise reads near 1; an unrelated matrix near 0
"""
from __future__ import annotations

import types

import torch

from ..data.special_tokens import DOC
from ..presets import AlephLMConfig
from ..train import surface as S

RESULTS: list = []
TINY = AlephLMConfig(name="tiny-surface-test", d_model=64, n_layers=3, n_heads=2, context=96, vocab_size=256, tokenizer="byte-trigram",
                     hub_layers=(0, 1, 2), hub_K=8, hub_D=16, tau=0.1, bank_experts=3, bank_ff=64, head_K=16, head_D=16, gate_init=-3.0,
                     tie_embeddings=False, hub_chunk=16, hub_const=2, hub_ckpt=0)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


class StandInTokenizer:
    """a byte-level tokenizer for the tests: a token is a run of non-space bytes with its one leading space (if any); every
    further space is a token of its own; the vocabulary grows as texts arrive; pieces are GPT-2's spellings of the bytes."""

    def __init__(self):
        self.vocab, self.pieces = {}, []

    def _tokens(self, raw: bytes):
        out, i = [], 0
        while i < len(raw):
            j = i
            if raw[j] == 0x20:
                j += 1
                if j < len(raw) and raw[j] == 0x20:   # a second space stands alone
                    out.append(raw[i:i + 1])
                    i += 1
                    continue
            while j < len(raw) and raw[j] != 0x20:
                j += 1
            out.append(raw[i:j])
            i = j
        return out

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for tb in self._tokens(text.encode("utf-8")):
            p = "".join(S.B2C[b] for b in tb)
            if p not in self.vocab:
                self.vocab[p] = len(self.pieces)
                self.pieces.append(p)
            ids.append(self.vocab[p])
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]


TEXTS = ["a taco truck parked by the sea", "two dogs  run", "café au lait on a tray", "snow over the harbour at dusk, boats moored",
         "x", "the quick brown fox jumps over the lazy dog again and again", "naïve résumé", "tags: 1girl, red hair, smile"]


def run_all() -> bool:
    torch.manual_seed(0)
    # 1 the map
    c2b = S.char_to_byte()
    check("map bijection", len(c2b) == 256 and sorted(c2b.values()) == list(range(256)))
    check("map space/newline", c2b["Ġ"] == 0x20 and "Ġ".encode("utf-8") == b"\xc4\xa0" and c2b["Ċ"] == 0x0a)
    # 2 spell + sites
    tok = StandInTokenizer()
    ok_round, ok_len, ok_close, ok_ends = True, True, True, True
    for text in TEXTS:
        sp = S.spell(tok, text)
        raw = text.encode("utf-8")
        ok_round &= sp.raw == raw and b"".join(bytes(c2b[c] for c in p) for p in tok.convert_ids_to_tokens(sp.ids)) == raw
        extra = raw.count(b" ") + sum(1 for b in raw if b >= 0x80)
        ok_len &= len(sp.spelled) - len(raw) == extra
        st = S.sites(sp)
        ok_ends &= all(0 < t < len(sp.ids) - 1 for t, _, _ in st) and len(st) == max(len(sp.ids) - 2, 0)
        pieces = tok.convert_ids_to_tokens(sp.ids)
        for t, ea, eb in st:
            nxt_a = bytes(c2b[c] for c in pieces[t + 1])
            nxt_b = pieces[t + 1].encode("utf-8")
            ok_close &= raw[ea] == nxt_a[0] and sp.spelled[eb] == nxt_b[0]
            ok_close &= raw[ea - 1] == bytes(c2b[c] for c in pieces[t])[-1] and sp.spelled[eb - 1] == pieces[t].encode("utf-8")[-1]
    check("spell round trip", ok_round)
    check("spelling length = bytes + spaces + high bytes", ok_len)
    check("closing bytes on both surfaces", ok_close)
    check("first and last token give no site", ok_ends)
    sp = S.spell(tok, "a taco")
    check("' taco' -> 'Ġtaco' six bytes against five", tok.convert_ids_to_tokens(sp.ids)[1] == "Ġtaco" and len(sp.spelled) == 7 and len(sp.raw) == 6)
    # 3 paired rows
    def docs():
        i = 0
        while True:
            yield S.spell(tok, TEXTS[i % len(TEXTS)] + f" {i}")
            i += 1
    ctx = 64
    pr = S.PairedRows(docs(), ctx, rows=2)
    xa, xb, st, ids = pr.next_batch()
    ok_shape = xa.shape == (2, ctx + 1) and xb.shape == (2, ctx + 1) and bool((xa[:, 0] == DOC).all()) and bool((xb[:, 0] == DOC).all())
    ok_sites = st.shape[1] == 3 and len(st) > 4 and bool((st[:, 1] < ctx).all()) and bool((st[:, 2] < ctx).all()) and len(ids) == len(st)

    def docs_of(row, back):
        out, cur = [], []
        for v in row.tolist()[1:]:
            if v == DOC:
                out.append(bytes(cur))
                cur = []
            else:
                cur.append(v)
        return out
    same = True
    for r in range(2):
        da, db = docs_of(xa[r], None), docs_of(xb[r], None)
        db_back = [bytes(c2b[c] for c in d.decode("utf-8")) for d in db]
        same &= da[:len(db_back)] == db_back and len(da) >= len(db_back)
    check("paired rows: shape, DOC first", ok_shape)
    check("paired rows: sites inside both rows", ok_sites)
    check("paired rows: the same documents in order on both surfaces", same)
    # the byte before each closing byte is the token's last byte, on each surface; the token ids match
    ok_tok = True
    for (r, qa, qb), tid in zip(st.tolist(), ids.tolist()):
        piece = tok.pieces[tid]
        ok_tok &= int(xa[r, qa - 1]) == bytes(c2b[c] for c in piece)[-1] and int(xb[r, qb - 1]) == piece.encode("utf-8")[-1]
    check("paired rows: each site closes its own token", ok_tok)
    spelled = [S.spell(tok, t) for t in TEXTS]
    xa1, xb1, st1, ids1 = S.single_rows(spelled, ctx)
    n_fit = sum(1 for d in spelled if len(d.raw) + 1 <= ctx and len(d.spelled) + 1 <= ctx)
    ok_single = 0 < n_fit < len(TEXTS) and xa1.shape[0] == n_fit and bool((xa1[:, 0] == DOC).all()) and len(st1) == len(ids1) > 0
    for (r, qa, qb), tid in zip(st1.tolist(), ids1.tolist()):
        piece = tok.pieces[tid]
        ok_single &= int(xa1[r, qa - 1]) == bytes(c2b[c] for c in piece)[-1] and int(xb1[r, qb - 1]) == piece.encode("utf-8")[-1]
    check("single rows: one document a row, sites close their tokens", ok_single)
    # 4 the losses
    A = {0: torch.randn(60, 16) * 3 + 1, 1: torch.randn(60, 16) * 0.5 - 2}
    stats = S.reference_stats(A)
    loss0, per0 = S.surface_loss(A, A, stats, form="mse")
    check("identical states: mse 0, cos 1", float(loss0) == 0.0 and all(abs(v["cos"] - 1) < 1e-5 for v in per0.values()))
    perm = torch.randperm(60)
    Ash = {b: v[perm] for b, v in A.items()}
    l_true, p_true = S.surface_loss(A, A, stats, form="mse_nce")
    l_sh, p_sh = S.surface_loss(A, Ash, stats, form="mse_nce")
    check("InfoNCE: the true pairing below a shuffled one", float(l_true) < float(l_sh) and all(p_true[b]["nce"] < p_sh[b]["nce"] for b in A)
          and set(p_true[0]) == {"mse", "cos", "nce"})
    z = S.standardize(A[0], stats[0])
    check("standardize: zero mean, unit spread", float(z.mean(0).abs().max()) < 1e-4 and float((z.std(0, unbiased=False) - 1).abs().max()) < 1e-3)
    # 5 the hand-run states against the model's forward
    from ..model.alephlm import AlephLM
    from .. import arm_mount as AM
    from ..train.arms import StageArmProgram, ArmProgramConfig, StageArm
    model = AlephLM(TINY).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, 256, (2, 40))
    where = torch.tensor([[0, 5], [1, 17], [0, 39]])
    check("logits parity, bare trunk", S.logits_parity(model, ids) == 0.0)
    full, _ = S.block_states(model, ids)
    gath, _ = S.block_states(model, ids, where=where)
    check("gathering at sites = indexing the full outputs",
          all(torch.equal(gath[b], full[b][where[:, 0], where[:, 1]]) for b in full) and set(full) == {0, 1, 2})
    arms = [StageArm(name=n, phase="test", spec=dict(n_slots=4, K=4, D=4, hidden=16), lam=2.0, seed=10 + i) for i, n in enumerate(["m1", "m2"])]
    prog = StageArmProgram(ArmProgramConfig(arms=arms)).bind(AM.inference_binding(model))
    g = torch.Generator().manual_seed(7)
    for a in arms:
        prog.attach(a)
        with torch.no_grad():
            for w in prog.wraps[a.name]:
                w.adapter.consume[-1].weight.add_(torch.randn(w.adapter.consume[-1].weight.shape, generator=g) * 0.2)
                w.adapter.gate.fill_(1.0)
    model.eval()
    armed = S.logits_parity(model, ids)
    with prog.handles["m2"].all_off():
        masked = S.logits_parity(model, ids)
        st_m, _ = S.block_states(model, ids)
    st_a, _ = S.block_states(model, ids)
    writes = any(not torch.equal(st_m[b], st_a[b]) for b in st_a)
    check("logits parity with two arms, and with one masked", armed == 0.0 and masked == 0.0 and writes)
    # a gradient reaches the training arm through the hand-run states
    for p in prog.params_of("m2"):
        p.requires_grad_(True)
    model.train()
    st_b, _ = S.block_states(model, ids, where=where)
    with torch.no_grad():
        st_t, _ = S.block_states(model, ids[:, torch.randperm(40)], where=where)
    stats3 = S.reference_stats({b: v.detach() for b, v in st_t.items()})
    loss, _ = S.surface_loss(st_b, st_t, stats3, form="mse")
    loss.backward()
    check("the loss reaches the arm's parameters", any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in prog.params_of("m2")))
    # 5b the reader: both surfaces, the same sites; the states equal the hand-run gather
    model.eval()
    for p in prog.params_of("m2"):
        p.requires_grad_(False)
    texts_r = TEXTS[:5]
    ra = S.read_spelled(model, tok, texts_r, blocks=[0, 2], surface="A", amp=False)
    rb = S.read_spelled(model, tok, texts_r, blocks=[0, 2], surface="B", amp=False)
    n_sites = sum(len(S.sites(S.spell(tok, t))) for t in texts_r)
    ok_read = (ra["sites"].shape == (n_sites, 3) and torch.equal(ra["sites"], rb["sites"]) and ra["states"][2].shape == (n_sites, 64)
               and set(ra["states"]) == {0, 2} and bool((ra["sites"][:, 0] < 5).all()))
    # the first text's first site, by hand: DOC + raw, the state at the closing byte, LayerNorm'd
    sp0 = S.spell(tok, texts_r[0])
    t0, ea0, _ = S.sites(sp0)[0]
    x0 = torch.tensor([[DOC] + list(sp0.raw)])
    with torch.no_grad():
        st0, _ = S.block_states(model, x0, [2], where=torch.tensor([[0, 1 + ea0]]))
    ok_read &= torch.allclose(S.ln(st0[2])[0], ra["states"][2][0], atol=1e-5) and int(ra["sites"][0, 2]) == sp0.ids[t0] and int(ra["sites"][0, 1]) == t0
    check("read_spelled: the same sites on both surfaces; states equal the hand-run gather", ok_read)
    # 6 the gauge
    X = torch.randn(600, 64, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    Y = X @ Q + 0.01 * torch.randn(600, 64, dtype=torch.float64)
    grp = torch.arange(600) // 3
    a_rot, a_unr = S.paired_alignment(X, Y, grp, k=32), S.paired_alignment(X, torch.randn(600, 64, dtype=torch.float64), grp, k=32)
    check("alignment: rotated copy near 1, unrelated near 0", a_rot > 0.98 and abs(a_unr) < 0.3, f"{a_rot:.3f} / {a_unr:.3f}")
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
