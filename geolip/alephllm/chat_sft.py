"""First chat conditioning — the arm route, one trigger.

run_chat_conditioning() takes the newest (or a named) locked core
checkpoint and produces the first chat arm: smoltalk everyday
conversations + a synthesized Beatrix persona set, rendered to byte rows
(exact-prefix masks), trained with amoe's certified frozen-trunk recipe,
gauged (core-preservation bpb, toggle-law bit-exactness, fixed-prompt
samples arm-off vs arm-on), and pushed to the training repo under
<craft>/arms/.

The full-SFT control (mode="both") finetunes a COPY of the core on the
same rows — the pre-registered gate comparison for arm integration.

Requires amoe-lora:  pip install git+https://github.com/AbstractEyes/amoe-lora
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import torch

from .presets import AlephLMConfig, TRAINING_REPO
from .model.alephlm import AlephLM
from .amoe_bridge import render_chat_rows, render_conversation, arm_meta

FIXED_PROMPTS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "What can you do?"}],
    [{"role": "user", "content": "Tell me about the moon."}],
]

NEUTRAL = ("The industrial revolution transformed manufacturing through "
           "mechanization, and cities grew rapidly as workers moved from "
           "farms to factories. Steam power reshaped both industry and "
           "transport across the century. ") * 4


def persona_conversations() -> list[list[dict]]:
    """~150 deterministic persona/limits rows: who she is, what she is,
    what she cannot do — short replies sized to her byte budget."""
    name = ("I am Beatrix, a small byte-level language model. I read raw "
            "bytes instead of words, and I am still in training.")
    convs = []
    for q in ["Who are you?", "What are you?", "What is your name?",
              "Tell me about yourself.", "Are you an AI?", "Are you human?",
              "Introduce yourself.", "Who am I talking to?"]:
        convs.append([{"role": "user", "content": q},
                      {"role": "assistant", "content": name}])
    for g, r in [("Hello!", "Hello! How can I help you today?"),
                 ("Hi", "Hi there! What would you like to talk about?"),
                 ("Hey", "Hey! I am listening."),
                 ("Good morning", "Good morning! I hope your day is going well."),
                 ("How are you?", "I am doing well, thank you for asking. How are you?"),
                 ("Thank you", "You are welcome!"),
                 ("Goodbye", "Goodbye! Thank you for talking with me.")]:
        convs.append([{"role": "user", "content": g},
                      {"role": "assistant", "content": r}])
    limits = ("I am a small model, so I make mistakes and my knowledge is "
              "limited. Simple questions work best.")
    for q in ["Can you write code for me?", "What is 847 times 293?",
              "Predict the stock market.", "Are you always right?",
              "How smart are you?"]:
        convs.append([{"role": "user", "content": q},
                      {"role": "assistant", "content": limits}])
    for q, r in [("What model are you?",
                  "I am an AlephLLM craft called mini-beatrix. My routing "
                  "uses signed geometric addresses instead of softmax."),
                 ("Where do you run?",
                  "I am small enough to run on ordinary computers, even "
                  "without a graphics card."),
                 ("Is this conversation private?",
                  "No — conversations in my public space are stored openly "
                  "for research, so please do not share personal details.")]:
        convs.append([{"role": "user", "content": q},
                      {"role": "assistant", "content": r}])
    return convs * 6   # ~140 rows after rendering


def load_chat_corpus(context: int, max_rows: int = 20_000,
                     max_row_bytes: int = 1024) -> list[dict]:
    """smoltalk everyday-conversations + persona set -> byte rows.

    max_row_bytes caps training-row length (NOT the model context): the
    amoe recipe is fp32 without activation checkpointing, and its max_len
    filter does not run on the pre-tokenized path — long rows multiply
    training memory ~linearly, so the cap is enforced HERE."""
    from datasets import load_dataset
    convs = []
    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations",
                      split="train")
    for r in ds:
        msgs = [{"role": m["role"], "content": m["content"]}
                for m in r["messages"] if m["role"] in ("user", "assistant")]
        if msgs:
            convs.append(msgs)
    convs += persona_conversations()
    rows = render_chat_rows(convs, context=min(context, max_row_bytes))
    return rows[:max_rows]


def _bpb(model, device) -> float:
    ids = torch.tensor([list(NEUTRAL.encode())[:768]], device=device)
    with torch.no_grad():
        _, loss = model(ids[:, :-1], targets=ids[:, 1:])
    return loss.item() / math.log(2)


def _samples(model, device, max_new=100) -> dict:
    out = {}
    for conv in FIXED_PROMPTS:
        prompt = render_conversation(conv)
        ids = torch.tensor([list(prompt.encode())], device=device)
        with torch.no_grad():
            g = model.generate(ids, max_new=max_new, temperature=0.0)
        txt = bytes(g[0, ids.shape[1]:].tolist()).decode(
            "utf-8", errors="replace")
        out[conv[0]["content"]] = txt.split("\n")[0].strip()[:200]
    return out


def run_chat_conditioning(hf_token: str | None = None,
                          craft: str = "mini-beatrix-1",
                          core_step: int | None = None,
                          steps: int = 1200, batch_size: int = 4,
                          max_row_bytes: int = 1024,
                          mode: str = "arm", push: bool = True,
                          device: str | None = None) -> dict:
    import amoe
    from huggingface_hub import HfApi, hf_hub_download
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    api = HfApi(token=hf_token)

    # ---- locked core
    files = api.list_repo_files(TRAINING_REPO)
    cks = sorted(f for f in files
                 if f.startswith(f"{craft}/checkpoints/step_")
                 and f.endswith(".safetensors") and "/fp8/" not in f)
    ck = (f"{craft}/checkpoints/step_{core_step:08d}.safetensors"
          if core_step else cks[-1])
    core_step = int(ck.split("step_")[1][:8])
    from safetensors.torch import load_file
    man = json.load(open(hf_hub_download(
        TRAINING_REPO, f"{craft}/manifest.json", token=hf_token),
        encoding="utf-8"))
    model = AlephLM(AlephLMConfig.from_dict(man["model_config"]))
    model.load_state_dict(load_file(hf_hub_download(
        TRAINING_REPO, ck, token=hf_token)))
    model = model.to(device).eval()
    print(f"[chat-sft] core: {craft} step {core_step:,} on {device}")

    # ---- data
    rows = load_chat_corpus(model.cfg.context, max_row_bytes=max_row_bytes)
    print(f"[chat-sft] {len(rows)} rows "
          f"(median {sorted(len(r['ids']) for r in rows)[len(rows)//2]} bytes)")

    # ---- baseline gauges
    report = {"craft": craft, "core_step": core_step, "steps": steps,
              "n_rows": len(rows), "base_bpb": _bpb(model, device),
              "samples_core": _samples(model, device)}
    print(f"[chat-sft] core neutral bpb {report['base_bpb']:.4f}")

    # ---- arm train (frozen core, certified recipe)
    t0 = time.time()
    cfg = amoe.TrainConfig(name=f"chat-{craft}", steps=steps,
                           batch_size=batch_size, grad_checkpointing=False)
    ckpt = amoe.train(model, rows, cfg, tokenizer=None, device=device)
    report["train_seconds"] = round(time.time() - t0, 1)
    ckpt.meta.update(arm_meta("chat", craft, core_step))
    os.makedirs("arms", exist_ok=True)
    arm_name = f"chat-s{steps}@step{core_step}.anchor.pt"
    local = os.path.join("arms", arm_name)
    ckpt.save(local)
    print(f"[chat-sft] arm trained in {report['train_seconds']}s -> {local}")

    # ---- attach + gauges
    handle = amoe.attach(model, local)
    wrappers = [b for b in model.blocks if hasattr(b, "adapter")]
    model.eval()
    report["arm_on_bpb"] = _bpb(model, device)
    report["samples_arm"] = _samples(model, device)
    for w in wrappers:
        w.enabled = False
    report["toggle_bit_exact"] = abs(_bpb(model, device)
                                     - report["base_bpb"]) < 1e-9
    for w in wrappers:
        w.enabled = True
    print(f"[chat-sft] core tax {report['arm_on_bpb']-report['base_bpb']:+.4f} "
          f"bpb · arm-off bit-exact: {report['toggle_bit_exact']}")
    for q in report["samples_core"]:
        print(f"  Q: {q}\n    core: {report['samples_core'][q]!r}"
              f"\n    arm : {report['samples_arm'][q]!r}")

    # ---- optional full-SFT control (the gate comparison)
    if mode == "both":
        amoe.detach(handle, verify=True)
        control = copy.deepcopy(model).to(device)
        opt = torch.optim.Adam(control.parameters(), lr=1e-4,
                               weight_decay=0.0)
        g = torch.Generator().manual_seed(0)
        control.train()
        from amoe.train.data import pad_batch
        for step in range(1, steps // 2 + 1):
            ix = torch.randint(0, len(rows), (batch_size,), generator=g)
            x, y, _ = pad_batch([rows[i] for i in ix], 0, device)
            out = control(input_ids=x, labels=y)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
        control.eval()
        report["control_bpb"] = _bpb(control, device)
        report["samples_control"] = _samples(control, device)
        print(f"[chat-sft] CONTROL (full SFT) neutral bpb "
              f"{report['control_bpb']:.4f} "
              f"(forgetting {report['control_bpb']-report['base_bpb']:+.4f})")

    # ---- ship (retry; on final failure keep the local file and say how
    # to recover — a trained arm must never die on a network hiccup)
    if push:
        for attempt in range(4):
            try:
                api.upload_file(path_or_fileobj=local,
                                path_in_repo=f"{craft}/arms/{arm_name}",
                                repo_id=TRAINING_REPO)
                api.upload_file(
                    path_or_fileobj=json.dumps(report, indent=1).encode(),
                    path_in_repo=f"{craft}/arms/{arm_name}.report.json",
                    repo_id=TRAINING_REPO)
                print(f"[chat-sft] shipped {craft}/arms/{arm_name} (+report)")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    print(f"[chat-sft] UPLOAD FAILED after 4 tries ({e}) — "
                          f"the arm is safe at {local}; re-push with:")
                    print(f"  HfApi(token=...).upload_file(path_or_fileobj="
                          f"{local!r}, path_in_repo="
                          f"'{craft}/arms/{arm_name}', "
                          f"repo_id='{TRAINING_REPO}')")
                else:
                    time.sleep(10 * (attempt + 1))
    return report
