"""Bridge to amoe-lora — AMOE arms on a locked AlephLM core.

amoe-lora (github.com/AbstractEyes/amoe-lora >= 0.2.3) resolves AlephLM
natively (binding "alephlm") and its BlockWithAdapter passes the cached
decode path through, so the five verbs work unchanged:

    import amoe
    from geolip.alephllm import prepare
    from geolip.alephllm.amoe_bridge import render_chat_rows

    run = prepare("mini-beatrix-1", hf_token=..., resume=True)
    rows = render_chat_rows(conversations, context=run.cfg.context)
    ck = amoe.train(run.raw_model, rows,
                    amoe.TrainConfig(name="chat", precision="fp32"),
                    tokenizer=None)          # rows are pre-rendered bytes
    ck.save("chat.anchor.pt")
    h = amoe.attach(run.raw_model, "chat.anchor.pt")   # toggleable arm
    ...
    amoe.detach(h, verify=True)              # bit-exact core restore

This module supplies the byte-side pieces: the chat template (identical
to the serving space's format) rendered into amoe's pre-tokenized row
form {"ids": [...], "n_prefix": int}, one row per assistant turn, with
the label mask beginning exactly at the assistant span (the collator
law — the prefix is byte-exact by construction, no template re-tokenize
drift is possible at byte level).
"""
from __future__ import annotations

CHAT_HEADER = ("A conversation with Beatrix, a small byte-level "
               "language model still in pretraining.\n")
USER_TAG = "User: "
ASSISTANT_TAG = "Beatrix:"
BINDING = "alephlm"


def render_conversation(turns: list[dict], upto: int | None = None) -> str:
    """Transcript text for turns[:upto] plus an open assistant tag."""
    upto = len(turns) if upto is None else upto
    out = CHAT_HEADER
    for t in turns[:upto]:
        tag = USER_TAG if t["role"] == "user" else ASSISTANT_TAG + " "
        out += f"{tag}{t['content']}\n"
    return out + ASSISTANT_TAG


def render_chat_rows(conversations: list[list[dict]],
                     context: int = 2048) -> list[dict]:
    """amoe pre-tokenized rows from role/content conversations.

    One row per assistant turn: ids = UTF-8 bytes of the transcript up to
    and including that reply (newline-terminated), n_prefix = bytes up to
    and including the open assistant tag — loss lands only on the reply.
    Rows longer than `context` are dropped (byte budgets are real)."""
    rows = []
    for conv in conversations:
        for j, turn in enumerate(conv):
            if turn["role"] != "assistant":
                continue
            prefix = render_conversation(conv, upto=j)
            full = prefix + " " + turn["content"] + "\n"
            ids = list(full.encode("utf-8", errors="replace"))
            if len(ids) > context:
                continue
            rows.append({"ids": ids,
                         "n_prefix": len(prefix.encode("utf-8",
                                                       errors="replace"))})
    return rows


def arm_meta(name: str, craft: str, checkpoint_step: int) -> dict:
    """Provenance fields for AnchorCheckpoint.meta — which locked core
    this arm belongs to (amoe's strict attach check reads base_model_id)."""
    return {"name": name,
            "base_model_id": f"alephllm/{craft}@step{checkpoint_step}",
            "substrate.family": "alephlm"}
