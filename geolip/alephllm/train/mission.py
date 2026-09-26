"""The mission driver: the v3 notebook's session cells (the decision block,
the preset build, the guard core, the push probe, the boundary-stopping
session loop) as one module that torchrun launches on one or many cards:

    torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.train.mission

Configuration is a JSON file named by ALEPHLLM_MISSION_CONFIG (the
notebook's decision-block names, lower-cased; see DEFAULTS). No argparse:
environment and JSON only. Rank 0 owns the record (checkpoints, manifest,
reports, uploads, the progress bar); the other ranks compute and write
their console to <out_dir>/<craft>/rank<r>.log.

The recipe's step is the GLOBAL batch: micro_batch x grad_accum x cards
x context must equal the preset's tokens per step (262,144 for the v3
craft) — the schedule and every settled number assume it.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

DEFAULTS = {
    "craft": "mini-beatrix-3",
    "depth": 24, "data_scale": 4.0, "epoch_cap": 2.0, "rebalance_to": "generators",
    "anneal_lr_scale": "from_screen",           # a number once the anneal screen sets it
    "arm_spec": "LAWFUL_16x8", "arm_spec_certified": False,
    "stage_arms": [["s1_perspective", "curriculum_s1", 1.0, 1], ["s2_concept", "curriculum_s2", 1.0, 2],
                   ["s3_rules", "curriculum_s3", 1.0, 3], ["s4_arith", "curriculum_s4", 1.0, 4],
                   ["s5_causal", "curriculum_s5", 1.0, 5], ["s6_tryfail", "curriculum_s6", 1.0, 6],
                   ["s7_mixed", "curriculum_s7", 1.0, 7], ["s8_register", "curriculum_s8", 1.0, 8]],
    "fusion": None,                             # the screened rule as a dict; None = the byte trunk unfused
    "coverage_audit": None,                     # the audit report's path (gates the start)
    "head_birth": "revival_birth",
    "quiet_trunk_grad": False,
    "abstain_chunks": None,                     # None = the certified 50/50 form (grad_accum abstention chunks per step); an int = that many
    # 0.10.4: recompute the stage adapters' intermediates in the backward instead of keeping them
    # (amoe-lora >= 0.2.7; exact, one extra adapter forward per backward). The memory fix for
    # stacked arms: kept, they cost ~3 GB per attached arm at micro_batch 2 on the v3 craft.
    # Refreshable at any boundary.
    "adapter_recompute": False,
    "resume_after_halt": False,
    "micro_batch": None, "grad_accum": None,    # PER RANK; required on a card
    "max_hours": 1000.0,
    "out_dir": "./alephllm_runs",
    "l6_ledger": "v3_preflight/l6_guards/v3_l6_guards_ledger.json",   # hub path (training repo) or a local file
    "local": False,                             # the CPU toy path (tests)
    "fresh": False,                             # abandon any hub resume state (explicit)
    # the curriculum's difficulty dial (0.10.2): the atlas-minted lexicon for the rules
    # generator — {"path": "hf://<repo>/<file>" | local, "splits": {"curriculum-s3":
    # {"rulechain-synth": 0.08, "rulechain-minted": 0.24}}, "pair_rate": 0.3,
    # "amendment": "<the lead's ruling, dated>"}; None = the legacy predicates only.
    # Lands at launch, on resume, or at any boundary before its stage opens.
    "minted_lexicon": None,
}


def _load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = os.environ.get("ALEPHLLM_MISSION_CONFIG")
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def _dist():
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)))


REFRESHABLE = ("arm_spec_certified", "anneal_lr_scale", "minted_lexicon", "stage_arms", "abstain_chunks",
               "adapter_recompute")


def _resolve_path(path: str, token: str | None) -> str:
    """A local path, or hf://<repo_id>/<path in repo> fetched to the cache."""
    if str(path).startswith("hf://"):
        from huggingface_hub import hf_hub_download
        rest = str(path)[len("hf://"):]
        parts = rest.split("/")
        repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
        return hf_hub_download(repo, sub, token=token)
    return str(path)


def apply_minted_lexicon(C: dict, run=None) -> dict | None:
    """Install the minted lexicon named by the config (the splits, the pair
    rate) into the curriculum; at a boundary (run given) the data plane is
    recomputed and recorded with the amendment note. Returns the record."""
    spec = C.get("minted_lexicon")
    if not spec:
        return None
    from ..data import curriculum as _cur
    token = os.environ.get("HF_TOKEN")
    path = _resolve_path(spec["path"], token)
    rec = _cur.set_minted_lexicon(path, spec.get("splits") or {}, float(spec.get("pair_rate", 0.0)))
    note = spec.get("amendment") or "minted lexicon installed by config (no note given)"
    if run is not None:
        run.amend_data_plane(note)
    return rec


def refresh_stage_arms(C_old: dict, C_new: dict, run) -> None:
    """A boundary edit of the stage arms' quiet constants lands on the live
    arms; names, phases, seeds and the arm count are frozen at launch."""
    old, new = C_old.get("stage_arms") or [], C_new.get("stage_arms") or []
    if old == new or run is None or getattr(run, "arms", None) is None:
        return
    assert len(old) == len(new) and all(o[0] == n[0] and o[1] == n[1] and o[3] == n[3] for o, n in zip(old, new)), \
        "stage_arms: only the quiet constant (lambda) may change after launch"
    for o, n in zip(old, new):
        if float(o[2]) != float(n[2]):
            run.arms.by_name[n[0]].lam = float(n[2])
            print(f"[mission] config: arm {n[0]} lambda {float(o[2])} -> {float(n[2])}", flush=True)


def _refresh_config(C: dict, run) -> dict:
    """Re-read the config file at every boundary: the later gates (the arm
    certification, the anneal multiplier) are settled by screens that run
    while the mission trains, so their values land in the file, not in a
    relaunch. Every rank reads the same file, so the gate decision agrees.
    A multiplier that arrives is applied to the trainer's per-phase LR
    scale before the anneal phase opens."""
    try:
        new = _load_config()
    except Exception as e:  # noqa: BLE001
        print(f"[mission] config re-read failed ({e!r}); keeping the launch values")
        return C
    old = dict(C)
    for k in REFRESHABLE:
        if new.get(k) != C.get(k):
            print(f"[mission] config: {k} {C.get(k)!r} -> {new.get(k)!r}")
            C[k] = new.get(k)
    if isinstance(C["anneal_lr_scale"], (int, float)):
        run.tc.phase_lr_scale = {"anneal": float(C["anneal_lr_scale"])}
    if C.get("minted_lexicon") != old.get("minted_lexicon"):
        try:
            if C.get("minted_lexicon"):
                apply_minted_lexicon(C, run)
            else:
                from ..data import curriculum as _cur
                _cur.clear_minted_lexicon()
                run.amend_data_plane("minted lexicon removed by config")
        except Exception as e:  # noqa: BLE001
            print(f"[mission] minted lexicon NOT installed ({e!r}); the config value is kept for the next boundary")
            C["minted_lexicon"] = old.get("minted_lexicon")
    try:
        refresh_stage_arms(old, C, run)
    except AssertionError as e:
        print(f"[mission] stage_arms edit refused: {e}")
        C["stage_arms"] = old.get("stage_arms")
    if C.get("abstain_chunks") != old.get("abstain_chunks") and getattr(run, "arms", None) is not None:
        v = C.get("abstain_chunks")
        run.arms.cfg.abstain_chunks = (int(v) if v is not None else None)
        print(f"[mission] config: abstention chunks per step -> {run.arms.cfg.abstain_chunks!r} (None = every task chunk pairs with one)", flush=True)
    if bool(C.get("adapter_recompute")) != bool(old.get("adapter_recompute")) and getattr(run, "arms", None) is not None:
        try:
            run.arms.set_recompute(bool(C.get("adapter_recompute")))
        except ImportError as e:
            print(f"[mission] adapter_recompute NOT applied ({e}); the launch value is kept", flush=True)
            C["adapter_recompute"] = old.get("adapter_recompute")
    return C


def build_preset(C: dict, token: str | None):
    """The notebook's preflight cell, minus the bench: the preset for the
    chosen depth, the anneal multiplier, the head-birth flag, the fusion
    spec (an atlas table on the hub is fetched once), the toy override."""
    from ..presets import make_v3_preset, PRESETS, AlephLMConfig
    craft = C["craft"] if not C["local"] else "v3-local-toy"
    p = make_v3_preset(int(C["depth"]), data_scale=float(C["data_scale"]), epoch_cap=float(C["epoch_cap"]),
                       rebalance_to=C["rebalance_to"], name=craft)
    if isinstance(C["anneal_lr_scale"], (int, float)):
        p.train.phase_lr_scale = {"anneal": float(C["anneal_lr_scale"])}
    if C["head_birth"] == "revival_birth":
        p.train.head_addr_frozen = True
    if C["local"]:
        d = int(C["depth"])
        p.model = AlephLMConfig(name=craft, d_model=64, n_layers=d, n_heads=1, context=512,
                                hub_layers=tuple(range(d)), hub_K=4, hub_D=8, hub_const=2,
                                bank_experts=3, bank_ff=64, head_K=8, head_D=16, hub_chunk=32, hub_ckpt=0)
        p.train.micro_batch, p.train.grad_accum, p.train.warmup_steps = 2, 1, 2
        p.train.log_every, p.train.health_every, p.train.eval_every = 2, 4, 8
        p.train.ckpt_every, p.train.tb_upload_every, p.train.val_tokens, p.train.canary_episodes = 8, 1000, 256, 4
        p.curriculum = [dict(ph, dataset="synthetic", planned_tokens=4096) for ph in p.curriculum]
    fusion = C.get("fusion")
    if isinstance(fusion, dict):
        spec = dict(fusion)
        if str(spec.get("table", "")).startswith("hf://"):
            from huggingface_hub import hf_hub_download
            owner, repo, path = spec["table"][5:].split("/", 2)
            spec["table"] = hf_hub_download(f"{owner}/{repo}", path, token=token)
        p.model.fusion = spec
    if C["micro_batch"] is not None:
        p.train.micro_batch = int(C["micro_batch"])
    if C["grad_accum"] is not None:
        p.train.grad_accum = int(C["grad_accum"])
    PRESETS[craft] = p
    return p


def build_guard(C: dict, token: str | None, local: bool):
    """The notebook's guard cell: modes from the shipped certification ledger."""
    from .guards import GuardConfig, certification_from_ledger
    from ..presets import TRAINING_REPO
    led = None
    src = C["l6_ledger"]
    if src and os.path.exists(src):
        led = json.load(open(src, encoding="utf-8"))
        print("guard ledger (local):", src)
    elif src and not local:
        try:
            from huggingface_hub import hf_hub_download
            led = json.load(open(hf_hub_download(TRAINING_REPO, src, token=token), encoding="utf-8"))
            print("guard ledger (hub):", src)
        except Exception as e:  # noqa: BLE001
            print("guard ledger unavailable:", repr(e)[:200])
    cert = certification_from_ledger(led or {})
    print("guard modes:", cert["modes"])
    return GuardConfig(modes=cert["modes"], certification=cert["certification"],
                       census_dataset="synthetic" if local else "fineweb-edu",
                       ref_window=(4, 16) if local else (1000, 2000),
                       census_every=4 if local else 100)


def build_arms(C: dict, local: bool):
    from .arms import StageArm, ArmProgramConfig, StageArmProgram, LAWFUL_16x8, WIDE_1024
    arms = C.get("stage_arms") or []
    if not arms:
        return None
    spec = {"LAWFUL_16x8": LAWFUL_16x8, "WIDE_1024": WIDE_1024}[C["arm_spec"]]
    if spec["K"] > 2 * spec["D"]:
        print(f"ARM_SPEC {C['arm_spec']}: K {spec['K']} > 2D — a flagged exception to the supply law")
    return StageArmProgram(ArmProgramConfig(
        arms=[StageArm(n, ph, spec=dict(spec), lam=float(lam), seed=int(seed)) for n, ph, lam, seed in arms],
        offdomain_dataset="synthetic" if local else "fineweb-edu",
        quiet_trunk_grad=bool(C["quiet_trunk_grad"]),
        abstain_chunks=(int(C["abstain_chunks"]) if C.get("abstain_chunks") is not None else None),
        adapter_recompute=bool(C.get("adapter_recompute", False))))


def push_probe(craft: str, token: str | None, extra: dict) -> bool:
    """The ship-complete law: a write to the training repo and a read-back
    before any training (tokenless = disarmed)."""
    if not token:
        return False
    from huggingface_hub import HfApi
    from ..presets import TRAINING_REPO
    api = HfApi(token=token)
    path = f"{craft}/v3_preflight/push_probe.json"
    body = json.dumps(dict(extra, craft=craft, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))).encode()
    api.upload_file(path_or_fileobj=body, path_in_repo=path, repo_id=TRAINING_REPO, commit_message="v3 push probe")
    ok = path in set(api.list_repo_files(TRAINING_REPO))
    print("push probe", "READ BACK OK" if ok else "NOT VISIBLE after upload", path)
    return ok


def revive_at_birth(run):
    """The born-with-function solve at step 0 (the head screen's arm B):
    the head address fitted in closed form on a birth sample, then frozen;
    provenance in the manifest. Main rank solves; every card takes it."""
    import torch
    if run.is_main:
        from .revival import revive_head
        m = run.raw_model.float()
        hs, bl, ys = [], [], []
        hook = m.nf.register_forward_hook(lambda mod, i, o: hs.append(o.detach().float().reshape(-1, o.shape[-1])))
        with torch.no_grad():
            for xb in run._val():
                out = m(xb[:, :-1], disable_head_aleph=True)
                bl.append(out.logits.float().reshape(-1, m.cfg.vocab_size))
                ys.append(xb[:, 1:].reshape(-1))
        hook.remove()
        prov = revive_head(m, torch.cat(hs), torch.cat(bl), torch.cat(ys), theta_deg=45.0)
        m.head.proj.weight.requires_grad_(False)
        m.head.addr.codebook.requires_grad_(False)
        run.manifest.note(f"HEAD_BIRTH revival_birth at step 0: {json.dumps(prov, default=float)[:400]}")
        print("head solved at birth (provenance in the manifest); address frozen; W_s trains")
    else:
        m = run.raw_model
        m.head.proj.weight.requires_grad_(False)
        m.head.addr.codebook.requires_grad_(False)
    run._broadcast_params()


def boundary_report(run, tag: str, craft: str, report_dir: str, prev_anneal: dict):
    """Main rank only: probes, census, the toggle ledger, the arms' gauges,
    the specials gauge, the anchors shipped, the report JSON uploaded."""
    import torch
    from . import probes
    from .instruments import model_census, toggle_ledger, special_token_gauge
    from ..model.governor import govern_model
    m, dev = run.raw_model, run.device
    m.eval()
    with torch.no_grad():
        pr = probes.run_all(m, run.tokenizer, dev)
        census = model_census(m, run._sample_batch())
        ledger = toggle_ledger(m, run._val())
        arms = None
        if run.arms is not None and run.arms.attached:
            with run.arms.all_off():
                bare = toggle_ledger(m, run._val())
            ledger["bpb_arms_off"] = bare["bpb_full"]
            ledger["toggle_arms_off"] = bare["bpb_full"] - ledger["bpb_full"]
            arms = run.arms.gauges(run._val(), None)
        sp = special_token_gauge(m, run._val())
    erank = {str(L): census["layers"][L].get("hidden_erank") for L in census["layers"]}
    rep = {"tag": tag, "step": run.step, "tokens": run.manifest.tokens_seen, "phase": tag,
           "next_phase": (run.manifest.current_phase() or {}).get("name"), "lr_mult": run._phase_mult(tag),
           "probes": pr, "census_flags": census.get("flags"), "erank_profile": erank, "ledger": ledger, "special": sp,
           "governor_hits_cum": getattr(run, "_gov_hits", 0),
           "crowd_check_extra_hits": govern_model(m, run.tc.governor_theta) if run.tc.governor else None,
           "guard": run.guard.summary() if run.guard is not None else None, "arms": arms,
           "arms_disabled": dict(run.arms.disabled) if run.arms is not None else None,
           "halt": run.manifest.halt, "cards": run.world}
    if run.guard is not None:
        if arms:
            bad = {n: r for n, r in arms.items() if isinstance(r, dict) and (r.get("fineweb_delta", 0) > 0.012)}
            run.guard.record_boundary_read("G5", tag, {"fired": False, "step": run.step, "past_quiet_bar": list(bad)})
        if tag.startswith("anneal") and sp:
            prev = prev_anneal.get("sp")
            delta = {k: sp[k] - prev[k] for k in ("doc_bpb", "reset_bpb") if prev and k in sp and k in prev}
            run.guard.record_boundary_read("G6", tag, {"fired": False, "step": run.step, "delta_vs_previous": delta})
            prev_anneal["sp"] = sp
    if run.arms is not None:
        for n in run.arms.attached:
            apath = os.path.join(run.out_dir, "arms", f"{n}_step{run.step}.safetensors")
            if os.path.exists(apath):
                continue          # the checkpoint routine shipped it already (0.10.3)
            ck = run.arms.anchor(n, craft, run.step)
            os.makedirs(os.path.dirname(apath), exist_ok=True)
            ck.save(apath)
            run.hub._up(apath, f"arms/{n}_step{run.step}.safetensors")
    body = json.dumps(rep, default=float)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f"{tag}_step{run.step}.json"), "w", encoding="utf-8") as f:
        f.write(body)
    run.hub.upload_bytes(body.encode(), f"reports/v3/{tag}_step{run.step}.json")
    print(probes.report(pr))
    m.train()
    return rep


def main():
    C = _load_config()
    rank, world = _dist()
    local = bool(C["local"])
    craft = C["craft"] if not local else "v3-local-toy"
    out_dir = C["out_dir"]
    if rank != 0:
        os.makedirs(os.path.join(out_dir, craft), exist_ok=True)
        log = open(os.path.join(out_dir, craft, f"rank{rank}.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from .. import __version__
    from ..presets import PRESETS
    from . import prepare
    token = os.environ.get("HF_TOKEN") if rank == 0 else None
    device = "cpu" if (local or not torch.cuda.is_available()) else "cuda"
    print(f"[mission] alephllm {__version__} · craft {craft} · rank {rank}/{world} · device {device} · "
          f"token {'present' if token else ('n/a (non-main rank)' if rank else 'ABSENT')}", flush=True)

    p = build_preset(C, token)
    tc, cfg = p.train, p.model
    if C.get("minted_lexicon"):
        # the difficulty dial at launch / resume: the lexicon + splits into the
        # curriculum before the trainer applies the scale; a resume whose
        # recipe differs only by it is accepted under the amendment note
        rec = apply_minted_lexicon(C)
        tc.data_plane_amendment = C["minted_lexicon"].get("amendment") or "minted lexicon installed by config"
        print(f"[mission] minted lexicon {rec['sha']}: {rec['words']} words, splits {rec['splits']}, pair rows "
              f"{rec['pair_rate']:.0%}", flush=True)
    assert cfg.hub_K <= 2 * cfg.hub_D, "supply law violated — this preset should not exist"
    assert tc.compile is False, "eager on Blackwell (the 2026-08-26 law)"
    tokens_step = tc.micro_batch * tc.grad_accum * cfg.context * world
    if not local:
        assert tc.micro_batch and tc.grad_accum, "micro_batch / grad_accum (per rank) are required on a card"
        assert tokens_step == 262_144, (f"the recipe's step is 262,144 tokens: micro_batch {tc.micro_batch} x "
                                        f"grad_accum {tc.grad_accum} x cards {world} x ctx {cfg.context} = {tokens_step:,}")
    print(f"[mission] {cfg.name}: depth {cfg.n_layers} · d {cfg.d_model} · ctx {cfg.context} · "
          f"{tc.micro_batch} x {tc.grad_accum} per card x {world} cards = {tokens_step:,} tokens/step · "
          f"fusion {getattr(cfg, 'fusion', None)} · anneal {C['anneal_lr_scale']} · head {C['head_birth']}")
    for ph in p.curriculum:
        print(f"   {ph['name']:<18} {ph['dataset']:<16} {ph['planned_tokens']/1e9:8.3f}B")

    # the start gates (the design's step 1): the data plane assembled
    gates = [("coverage audit over the pretraining mix", C["coverage_audit"] is not None or local),
             ("weak-token fusion: the rule screened (a dict) or the unfused trunk chosen (null)",
              C.get("fusion") is None or isinstance(C.get("fusion"), dict))]
    for name, ok in gates:
        print(f"[gate] {name}: {'done' if ok else 'OPEN'}")
    if not all(ok for _, ok in gates):
        raise SystemExit("the pre-flight is incomplete — the OPEN gates above are builds/tests on the queue")

    guard = build_guard(C, token, local) if rank == 0 else None
    arms = build_arms(C, local)
    if rank == 0 and not local:
        assert push_probe(craft, token, {"depth": cfg.n_layers, "cards": world, "tokens_per_step": tokens_step}), \
            "the push probe did not pass"

    run = prepare(PRESETS[craft], hf_token=token, out_dir=out_dir, guard=guard, arms=arms,
                  resume=not bool(C["fresh"]), device=device)
    if run.step == 0 and C["head_birth"] == "revival_birth":
        revive_at_birth(run)
    elif run.step == 0 and C["head_birth"] == "born_null" and run.is_main:
        run.manifest.note("HEAD_BIRTH born_null: the 2s birth form (buried 3/3 on record) — a flagged exception")
    report_dir = os.path.join(run.out_dir, "reports", "v3")
    prev_anneal: dict = {}
    resume_after_halt = bool(C["resume_after_halt"])

    t_end = time.time() + float(C["max_hours"]) * 3600
    while time.time() < t_end:
        hours_left = (t_end - time.time()) / 3600
        if hours_left < (0.2 if not local else 0.0):
            break
        C = _refresh_config(C, run)
        nxt = run.manifest.current_phase()
        if nxt is not None and not local:
            if any(a.phase == nxt["name"] for a in (arms.cfg.arms if arms else [])) and not C["arm_spec_certified"]:
                print(f"BOUNDARY GATE: '{nxt['name']}' attaches a stage arm and the lawful geometry is not yet certified "
                      "(2 seeds) — the session stops here; the certification is on the card queue")
                break
            if nxt["name"].startswith("anneal") and not isinstance(C["anneal_lr_scale"], (int, float)):
                print(f"BOUNDARY GATE: '{nxt['name']}' needs the anneal multiplier from the screen — the session stops here")
                break
        run.train(max_hours=hours_left, stop_at_boundary=True, resume_after_halt=resume_after_halt)
        resume_after_halt = False
        if getattr(run, "_interrupted", False):
            print("manual stop - no auto-resume; resume state is on the hub")
            break
        halt = getattr(run, "_guard_halt", None)
        if halt:
            if run.is_main:
                boundary_report(run, f"guard_{halt['guard']}", craft, report_dir, prev_anneal)
                print(f"GUARD HALT {halt['guard']} at step {halt.get('step')} — the session ends here; a human decides "
                      "(read the archive + the report; resume_after_halt=true continues from the last healthy checkpoint)")
            break
        tag = getattr(run, "_last_boundary", None) or "session_cap"
        if run.is_main:
            boundary_report(run, tag, craft, report_dir, prev_anneal)
        if run.manifest.current_phase() is None:
            print("curriculum complete — the two-phase anneal was planned from birth and has run")
            break
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print("session over — resume state on the hub" if token else "session over — resume state local", flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
