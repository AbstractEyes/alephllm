"""Canary suite — synthetic in-context binding probes run against
checkpoints throughout training (clean-protocol descendant of the V1b
recall battery: every answer fully in-window by construction, keys drawn
directly at random with no modular folding, all keys distinct within an
episode).

Episode layout (token ids):
    k1 v1 SEP k2 v2 SEP ... kP vP SEP QRY k_i  ->  v_i (3 answer positions)

The eval decomposes errors selection-gauge style: a wrong answer that
matches some OTHER pair's value is wrong-pair retrieval; anything else is
no-retrieval.
"""
from __future__ import annotations

import torch


def _special_ids(vocab: int):
    """Byte crafts (vocab 256) use the 0.8.0 specials layout: CUE (0xF6)
    marks the query, SEP is the frame separator (0xF8), and the payload
    alphabet stops below 0xC0 so NO special can appear as content. The
    old frame used QRY=255/SEP=254 — under 0.8.0 those are DOC/USER:
    QRY==DOC planted a document-reset token at exactly the retrieval
    point (audit catch, 2026-08-26). recall_acc re-baselines here.
    BPE crafts keep the old top-of-vocab convention."""
    if vocab == 256:
        from ..data.special_tokens import CUE, SEP as SEP_ID
        return 0xC0, SEP_ID, CUE       # payload alphabet: 1..0xBF
    hi = vocab
    return hi - 6, hi - 2, hi - 1


def make_episodes(n: int, vocab: int, pairs: int = 4, klen: int = 3,
                  vlen: int = 3, seed: int = 0):
    """Returns (tokens (n, L), answer_pos (n, vlen), answers (n, vlen),
    all_values (n, pairs, vlen))."""
    alpha_hi, SEP, QRY = _special_ids(vocab)
    g = torch.Generator().manual_seed(seed)
    toks, apos, ans, allv = [], [], [], []
    for _ in range(n):
        while True:
            keys = torch.randint(1, alpha_hi, (pairs, klen), generator=g)
            if len({tuple(k.tolist()) for k in keys}) == pairs:
                break
        vals = torch.randint(1, alpha_hi, (pairs, vlen), generator=g)
        qi = int(torch.randint(0, pairs, (1,), generator=g).item())
        seq = []
        for p in range(pairs):
            seq += keys[p].tolist() + vals[p].tolist() + [SEP]
        seq += [QRY] + keys[qi].tolist()
        base = len(seq)
        seq += vals[qi].tolist()
        toks.append(seq)
        apos.append(list(range(base, base + vlen)))
        ans.append(vals[qi].tolist())
        allv.append(vals.tolist())
    return (torch.tensor(toks), torch.tensor(apos), torch.tensor(ans),
            torch.tensor(allv))


@torch.no_grad()
def canary_eval(model, tokenizer, device, episodes: int = 128,
                context: int = 1024, pairs: int = 4, seed: int = 7) -> dict:
    vocab = model.cfg.vocab_size
    toks, apos, ans, allv = make_episodes(episodes, vocab, pairs=pairs,
                                          seed=seed)
    assert toks.shape[1] <= context, "canary episode exceeds context"
    model.eval()
    toks = toks.to(device)
    logits, _ = model(toks[:, :-1])
    # answer position j is PREDICTED at input position j-1
    pred = logits.argmax(-1)                                  # (n, L-1)
    idx = (apos - 1).to(device)                               # (n, vlen)
    got = torch.gather(pred, 1, idx).cpu()                    # (n, vlen)
    ok = (got == ans).all(dim=1)
    acc = ok.float().mean().item()
    wrong = ~ok
    wrongpair = torch.zeros_like(ok)
    if wrong.any():
        eq = (allv == got.unsqueeze(1)).all(dim=-1).any(dim=-1)
        wrongpair = wrong & eq
    model.train()
    return {"recall_acc": acc,
            "wrongpair_frac": float(wrongpair.float().mean().item()),
            "noretrieval_frac": float((wrong & ~wrongpair).float().mean().item())}


def make_val_batches(stream, device, n_batches: int = 4):
    return [stream.next_batch().to(device) for _ in range(n_batches)]
