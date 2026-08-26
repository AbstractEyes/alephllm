"""Born-in instruments — the full gauge suite, logged to TensorBoard and
rendered in the runtime health readout. Instruments before spend.

Covers, with no exceptions:
  - effective-rank census: consumed address erank (hub layers, per-layer),
    hidden-state erank per layer (structural collapse tell), codebook gram
    erank (frame health), head address erank
  - CV analysis: coefficient of variation of expert load (banks) and
    anchor usage (hubs/head) + usage perplexity — routing balance without
    ever adding balance pressure
  - structural collapse detectors: anchor merging (|cos|>0.99 pairs),
    dispatch entropy collapse, hidden-erank floor, hub denominator floor
    rate, loss spike / NaN guards
  - sign census (inhibition fraction per bank/head)
  - trajectories: gates, gamma, anchor drift from home
  - toggle ledger: bank-off / hub-off / head-aleph-off deltas (bpb), the
    causal contribution instrument
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(M: torch.Tensor) -> float:
    """exp(entropy) of normalized squared singular values."""
    M = M.float()
    if M.ndim > 2:
        M = M.reshape(-1, M.shape[-1])
    if M.shape[0] > 4096:
        M = M[torch.randperm(M.shape[0], device=M.device)[:4096]]
    s = torch.linalg.svdvals(M - M.mean(0, keepdim=True))
    p = (s * s) / (s * s).sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())


@torch.no_grad()
def bank_stats(bank) -> dict:
    """Dispatch health for one AnchoredBank (uses cached last_dispatch)."""
    out = {"gates": [float(torch.sigmoid(g).item()) for g in bank.gates]}
    w = bank.last_dispatch
    if w is None:
        return out
    w = w.reshape(-1, w.shape[-1]).float()
    load = w.abs().mean(0)                                   # (K,)
    out["load"] = [round(float(v), 5) for v in load]
    out["load_cv"] = float((load.std() / load.mean().clamp_min(1e-12)).item())
    p = load / load.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum()
    out["dispatch_entropy_frac"] = float(
        (ent / math.log(max(len(load), 2))).item())
    out["sign_frac_neg"] = float((w < 0).float().mean().item())
    return out


@torch.no_grad()
def model_census(model, sample_idx: torch.Tensor) -> dict:
    """Full structural census on a small batch. Replays the forward pass
    block by block so every gauge reads the EXACT tensor its mechanism
    consumes: attention reads n1(x), the bank reads n2(x), the head reads
    nf(x). Hidden-state erank is measured on block OUTPUTS (the collapse
    tell). The replay is the real forward — den stats and dispatch caches
    are populated by the same calls the model makes."""
    was_training = model.training
    model.eval()
    census = {"layers": {}, "head": {}, "flags": {}}
    hidden_eranks = []
    x = model.embed(sample_idx)
    for i, wrap in enumerate(model.blocks):
        # frozen-trunk adapter wrappers (amoe-lora) hold the real block at
        # .block — census taps read the inner block, then the wrapper's
        # patch is applied to keep the replayed stream faithful
        blk = getattr(wrap, "block", wrap)
        li = {}
        tap_attn = blk.n1(x)
        if blk.is_hub:
            if getattr(blk.attn, "n_const", 1) == 1:
                q = blk.attn.q(tap_attn)
                li["hub_addr"] = blk.attn.addr.health(q)
                p, n = blk.attn.addr.oriented(
                    q.reshape(-1, q.shape[-1]).float())
                li["hub_consumed_erank"] = effective_rank(
                    torch.cat([p, n], -1))
            else:
                # v2 multi-constellation: per-book health, worst-case
                # aggregates surfaced (the collapse flags key on these),
                # consumed erank over the concatenated oriented reads.
                healths, cats = [], []
                for c in blk.attn.consts:
                    q = c.q(tap_attn)
                    healths.append(c.addr.health(q))
                    p, n = c.addr.oriented(
                        q.reshape(-1, q.shape[-1]).float())
                    cats.append(torch.cat([p, n], -1))
                agg = {}
                for k in healths[0]:
                    vals = [h[k] for h in healths]
                    if k == "anchor_merge_pairs":
                        agg[k] = sum(vals)            # total across books
                    elif "max" in k:
                        agg[k] = max(vals)            # worst book
                    else:
                        agg[k] = sum(vals) / len(vals)
                li["hub_addr"] = agg
                li["hub_consumed_erank"] = effective_rank(
                    torch.cat(cats, -1))
        x = x + blk.attn(tap_attn)
        if blk.is_hub and blk.attn.last_den_stats is not None:
            dmin, dmean, dfloor = blk.attn.last_den_stats
            li["hub_den"] = {"min": dmin, "mean": dmean, "floor_frac": dfloor}
        tap_bank = blk.n2(x)
        li["bank_addr"] = blk.bank.addr.health(tap_bank)
        x = x + blk.bank(tap_bank)
        if wrap is not blk:
            if hasattr(wrap, "adapter") and getattr(wrap, "enabled", True):
                x = wrap.adapter(x)
            elif hasattr(wrap, "disp"):
                x = wrap.disp(x)
        li.update({f"bank_{k}": v for k, v in bank_stats(blk.bank).items()})
        li["hidden_erank"] = effective_rank(x[:2])
        hidden_eranks.append(li["hidden_erank"])
        li["hidden_rms"] = float(x[:2].float().pow(2).mean().sqrt().item())
        census["layers"][i] = li

    h_final = model.nf(x)
    head = model.head
    ph = head.proj(h_final)
    census["head"]["gamma"] = float(head.gamma.item())
    census["head"]["w_s_norm"] = float(head.w_s.weight.float().norm().item())
    census["head"]["addr"] = head.addr.health(ph)
    w = head.addr.signed(ph.reshape(-1, ph.shape[-1]).float())
    census["head"]["consumed_erank"] = effective_rank(w)

    # ------------------------------------------------------ collapse flags
    flags = {}
    detail = {}
    d = model.cfg.d_model
    low = [f"L{i}:{e:.0f}" for i, e in enumerate(hidden_eranks)
           if e < 0.05 * d]
    flags["hidden_erank_floor"] = bool(low)
    if low:
        detail["hidden_erank_floor"] = ",".join(low)
    merged = []
    for i, li in census["layers"].items():
        bp = li.get("bank_addr", {}).get("anchor_merge_pairs", 0)
        hp = li.get("hub_addr", {}).get("anchor_merge_pairs", 0)
        if bp:
            merged.append(f"bankL{i}:{bp}p")
        if hp:
            merged.append(f"hubL{i}:{hp}p")
    head_p = census["head"]["addr"].get("anchor_merge_pairs", 0)
    if head_p:
        merged.append(f"head:{head_p}p")
    flags["anchor_merge"] = bool(merged)
    if merged:
        detail["anchor_merge"] = ",".join(merged)
    census["flag_detail"] = detail
    flags["dispatch_collapse"] = any(
        li.get("bank_dispatch_entropy_frac", 1.0) < 0.10
        for li in census["layers"].values())
    flags["den_floor"] = any(
        li.get("hub_den", {}).get("floor_frac", 0.0) > 1e-3
        for li in census["layers"].values())
    flags["codebook_erank_floor"] = any(
        li["bank_addr"]["codebook_erank"] < 1.5
        for li in census["layers"].values())
    census["flags"] = flags
    if was_training:
        model.train()
    return census


@torch.no_grad()
def toggle_ledger(model, val_batches: list) -> dict:
    """Causal contribution ledger: bpb with each mechanism toggled off.
    Positive delta = the mechanism was carrying function."""
    was_training = model.training
    model.eval()

    def bpb(**kw):
        tot, n = 0.0, 0
        for xb in val_batches:
            _, loss = model(xb[:, :-1], targets=xb[:, 1:], **kw)
            tot += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        return (tot / max(n, 1)) / math.log(2)

    full = bpb()
    led = {"bpb_full": full,
           "bpb_bank_off": bpb(disable_bank=True),
           "bpb_head_aleph_off": bpb(disable_head_aleph=True)}
    if any(getattr(b, "block", b).is_hub for b in model.blocks):
        led["bpb_hub_off"] = bpb(disable_hub=True)
    for k in list(led):
        if k != "bpb_full":
            led[k.replace("bpb_", "toggle_")] = led[k] - full
    if was_training:
        model.train()
    return led


def log_census_tb(writer, census: dict, ledger: dict | None, step: int):
    for i, li in census["layers"].items():
        writer.add_scalar(f"erank/hidden_L{i}", li["hidden_erank"], step)
        if "hub_consumed_erank" in li:
            writer.add_scalar(f"erank/hub_consumed_L{i}",
                              li["hub_consumed_erank"], step)
        if "bank_load_cv" in li:
            writer.add_scalar(f"cv/bank_load_L{i}", li["bank_load_cv"], step)
        if "bank_dispatch_entropy_frac" in li:
            writer.add_scalar(f"collapse/dispatch_entropy_L{i}",
                              li["bank_dispatch_entropy_frac"], step)
        if "bank_sign_frac_neg" in li:
            writer.add_scalar(f"sign/bank_frac_neg_L{i}",
                              li["bank_sign_frac_neg"], step)
        if "bank_addr" in li:
            writer.add_scalar(f"drift/bank_L{i}",
                              li["bank_addr"]["drift_mean"], step)
            writer.add_scalar(f"erank/bank_codebook_L{i}",
                              li["bank_addr"]["codebook_erank"], step)
        if "hub_addr" in li:
            writer.add_scalar(f"drift/hub_L{i}",
                              li["hub_addr"]["drift_mean"], step)
            writer.add_scalar(f"erank/hub_codebook_L{i}",
                              li["hub_addr"]["codebook_erank"], step)
            writer.add_scalar(f"cv/hub_usage_L{i}",
                              li["hub_addr"].get("usage_cv", 0.0), step)
        if "hub_den" in li:
            writer.add_scalar(f"collapse/den_floor_L{i}",
                              li["hub_den"]["floor_frac"], step)
        for k, g in enumerate(li.get("bank_gates", [])):
            writer.add_scalar(f"gates/L{i}_e{k}", g, step)
    writer.add_scalar("head/gamma", census["head"]["gamma"], step)
    writer.add_scalar("head/w_s_norm", census["head"]["w_s_norm"], step)
    writer.add_scalar("head/consumed_erank",
                      census["head"]["consumed_erank"], step)
    writer.add_scalar("erank/head_codebook",
                      census["head"]["addr"]["codebook_erank"], step)
    for name, val in census["flags"].items():
        writer.add_scalar(f"collapse/flag_{name}", float(val), step)
    if ledger:
        for k, v in ledger.items():
            writer.add_scalar(f"toggles/{k}", v, step)


def readout(step: int, tokens: int, census: dict, ledger: dict | None,
            extra: dict | None = None) -> str:
    """The structured runtime health block printed between tqdm segments."""
    L = census["layers"]
    hid = " ".join(f"{li['hidden_erank']:.0f}" for li in L.values())
    hubs = {i: li for i, li in L.items() if "hub_consumed_erank" in li}
    lines = [
        f"== health @ step {step:,} · {tokens/1e9:.3f}B tokens ==",
        f"hidden erank/layer : {hid}",
    ]
    if hubs:
        lines.append("hub consumed erank : " + " ".join(
            f"L{i}:{li['hub_consumed_erank']:.0f}" for i, li in hubs.items()))
        lines.append("hub den floor      : " + " ".join(
            f"L{i}:{li.get('hub_den', {}).get('floor_frac', 0):.1e}"
            for i, li in hubs.items()))
    cvs = [li.get("bank_load_cv") for li in L.values()
           if li.get("bank_load_cv") is not None]
    if cvs:
        lines.append(f"bank load CV       : mean {sum(cvs)/len(cvs):.3f} "
                     f"max {max(cvs):.3f}")
    sf = [li.get("bank_sign_frac_neg") for li in L.values()
          if li.get("bank_sign_frac_neg") is not None]
    if sf:
        lines.append(f"sign census (bank) : frac_neg mean {sum(sf)/len(sf):.3f}")
    lines.append(f"head aleph         : gamma {census['head']['gamma']:+.3f} · "
                 f"||W_s|| {census['head']['w_s_norm']:.4f} · "
                 f"consumed erank {census['head']['consumed_erank']:.1f}")
    if ledger:
        tog = " ".join(f"{k.replace('toggle_', '')}:{v:+.4f}"
                       for k, v in ledger.items() if k.startswith("toggle_"))
        lines.append(f"toggle ledger (bpb): full {ledger['bpb_full']:.4f} · {tog}")
    det = census.get("flag_detail", {})
    fired = [k + ("(" + det[k] + ")" if k in det else "")
             for k, v in census["flags"].items() if v]
    lines.append("collapse flags     : " +
                 (" !! " + ", ".join(fired) if fired else "none"))
    if extra:
        lines.append(" · ".join(f"{k} {v}" for k, v in extra.items()))
    return "\n".join(lines)


@torch.no_grad()
def special_token_gauge(model, val_batches: list, doc_id: int | None = None,
                        reset_window: int = 16) -> dict | None:
    """The DOC-boundary instrument (0.8.0 specials era, byte crafts only).

    Three numbers per eval, from the same fixed val batches as the ledger,
    IN BITS so they read directly against val bpb (audit catch: the first
    draft reported nats beside a bits gauge):
      doc_count   DOC targets seen (frequency sanity — 0 means the
                  formatter is not running; the zeros then logged are
                  ABSENCE, not perfection)
      doc_bpb     mean bits at positions whose TARGET is DOC — can the
                  craft see documents ending?
      reset_bpb   mean bits over the first `reset_window` targets AFTER
                  each DOC — the context-reset cost vs val bpb. (Windows
                  clip at packed-row edges; bias ~ W/ctx, negligible.)
    """
    if doc_id is None:
        from ..data.special_tokens import DOC as doc_id  # single source
    if getattr(model.cfg, "vocab_size", None) != 256:
        return None
    was_training = model.training
    model.eval()
    doc_tot = doc_n = reset_tot = reset_n = 0.0
    for xb in val_batches:
        x, y = xb[:, :-1], xb[:, 1:]
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1), reduction="none").view_as(y)
        at_doc = y == doc_id
        doc_tot += float(nll[at_doc].sum())
        doc_n += float(at_doc.sum())
        after = torch.zeros_like(at_doc)
        for k in range(1, reset_window + 1):
            after[:, k:] |= at_doc[:, :-k]
        after &= ~at_doc
        reset_tot += float(nll[after].sum())
        reset_n += float(after.sum())
    if was_training:
        model.train()
    return {"doc_count": int(doc_n),
            "doc_bpb": doc_tot / max(doc_n, 1.0) / math.log(2),
            "reset_bpb": reset_tot / max(reset_n, 1.0) / math.log(2)}
