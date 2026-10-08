"""Mount published arm groups on a trained AlephLM trunk.

An *arm* is a detachable adapter trained on one exact frozen trunk (amoe-lora's
RelayPatchwork, one module after every block). The stage-arm program
(train/arms.py) trains groups of them; this module brings a trained group back
onto a trunk by the same route, for evaluation and downstream use:

    from geolip.alephllm.arm_mount import load_trunk, mount_group, only, detach_all
    model = load_trunk(245674, device="cuda")        # the trunk at a stored step
    prog = mount_group(model, "gCA")                  # the eight stage arms + the caption arm
    with only(prog, ["s9_caption"]):                  # one member, the others masked
        ...
    detach_all(prog, verify=True)                     # the bare trunk again, bit-exact

Anchor sets live on the training repository (AbstractPhil/alephllm-mini-beatrix-
training, mini-beatrix-3/arm_refit/group/anchors/<run>_close<k>_<member>
.safetensors) in the amoe anchor format, whose header metadata carries
content_hash_v2, a hash of the tensors. GROUPS names the published runs; any run
mounts with its close and member list. Two checks guard a mount: every anchor's
base_model_id must name the trunk step (arms are trunk-bound: the same anchors on
another trunk lose most of their effect), and a group that extends an earlier one
(gCA over gXA, gCB over gXB) must carry that group's members unchanged, tensor
hash for tensor hash.

The mount wraps every block in amoe's BlockWithAdapter (amoe-lora >= 0.2.11), whose
forward, prefill and step all apply the adapter, so block-level taps taken after
the mount see the armed stream. Masking a member removes its write without
renormalizing the others (masked is not solo); in a stack each member sees the
stream after the earlier ones, so a later member with every earlier one masked
equals that member mounted alone.
"""
from __future__ import annotations

import contextlib
import json
import os
import struct
import types

import torch

ANCHOR_REPO = "AbstractPhil/alephllm-mini-beatrix-training"
ANCHOR_DIR = "mini-beatrix-3/arm_refit/group/anchors"
CHECKPOINT_DIR = "mini-beatrix-3/checkpoints"
CRAFT = "mini-beatrix-3"
STEP = 245674
STAGE_ARMS = ["s1_perspective", "s2_concept", "s3_rules", "s4_arith", "s5_causal",
              "s6_tryfail", "s7_mixed", "s8_register"]
CAPTION_ARM = "s9_caption"
PHASES = {s: f"curriculum_s{i}" for i, s in enumerate(STAGE_ARMS, 1)}
PHASES[CAPTION_ARM] = "caption_pack"
# run -> (close, members): the published group anchor sets
GROUPS = {
    "gRA": (4, STAGE_ARMS[:4]), "gRB": (4, STAGE_ARMS[:4]),          # stages 1-4
    "gXA": (8, STAGE_ARMS), "gXB": (8, STAGE_ARMS),                  # stages 1-8 (the four held fixed, four more over them)
    "gEA": (8, STAGE_ARMS), "gEB": (8, STAGE_ARMS),                  # stages 1-8 from scratch
    "gCA": (9, STAGE_ARMS + [CAPTION_ARM]), "gCB": (9, STAGE_ARMS + [CAPTION_ARM]),   # the eight held fixed + the caption arm
}
# a group that extends an earlier one carries that group's members unchanged
EXTENDS = {"gCA": ("gXA", 8, STAGE_ARMS), "gCB": ("gXB", 8, STAGE_ARMS)}
DEFAULT_SPEC = dict(n_slots=16, K=16, D=8, hidden=256)


# ----------------------------------------------------------------- the trunk
def v3_config(craft: str = CRAFT):
    from .presets import make_v3_preset
    return make_v3_preset(32, data_scale=4.0, epoch_cap=2.0, rebalance_to="generators", name=craft).model


def checkpoint_file(step: int = STEP, local_dir: str | None = None, repo: str = ANCHOR_REPO) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f"{CHECKPOINT_DIR}/step_{step:08d}.safetensors", local_dir=local_dir, token=False)


def load_trunk(step: int | None = STEP, device: str = "cpu", *, ckpt: str | None = None, cfg=None, state: dict | None = None,
               local_dir: str | None = None, repo: str = ANCHOR_REPO):
    """The trunk at a stored step: the v3 preset's AlephLM, the weights loaded strictly (every key present, none unexpected),
    frozen, in eval mode. `ckpt` names a local weight file; `cfg` and `state` override the preset and the weights."""
    from safetensors.torch import load_file
    from .model.alephlm import AlephLM
    cfg = cfg or v3_config()
    model = AlephLM(cfg)
    if state is None and (ckpt or step is not None):
        state = load_file(ckpt or checkpoint_file(step, local_dir, repo))
    if state is not None:
        res = model.load_state_dict(state, strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise ValueError(f"the weights do not match the trunk: missing {res.missing_keys[:4]}, unexpected {res.unexpected_keys[:4]}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


# --------------------------------------------------------------- the anchors
def header_metadata(path: str) -> dict:
    """The safetensors header's metadata (content_hash_v2, base_model_id, ...) without reading any tensor."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header.get("__metadata__") or {}


def anchor_name(run: str, close: int, member: str) -> str:
    return f"{run}_close{close}_{member}.safetensors"


def anchor_files(run: str, close: int, members, local_dir: str | None = None, repo: str = ANCHOR_REPO) -> dict:
    """member -> local path of the run's anchor at that close, downloaded unless already present."""
    from huggingface_hub import hf_hub_download
    return {m: hf_hub_download(repo, f"{ANCHOR_DIR}/{anchor_name(run, close, m)}", local_dir=local_dir, token=False)
            for m in members}


def check_frozen_members(files: dict, base_files: dict, members) -> dict:
    """Members carried unchanged from an earlier group must have the same tensors: content_hash_v2 equal member by member
    (the hash of the tensors, so header differences do not matter). Returns {member: hash}; raises naming the offenders."""
    hashes, bad = {}, []
    for m in members:
        a = header_metadata(files[m]).get("content_hash_v2")
        b = header_metadata(base_files[m]).get("content_hash_v2")
        hashes[m] = a
        if not a or a != b:
            bad.append(f"{m}: {a} vs {b}")
    if bad:
        raise ValueError("members carried from the earlier group differ from it (content_hash_v2): " + "; ".join(bad))
    return hashes


def _spec_of(meta: dict) -> dict:
    spec = meta.get("spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            import ast
            spec = ast.literal_eval(spec)
    return dict(spec) if isinstance(spec, dict) else dict(DEFAULT_SPEC)


def inference_binding(model, device=None):
    """The trainer stand-in the arm program binds to outside training: the model, its device, the trunk's parameters, a
    single-process rank layout and the context length. Nothing here streams data or steps an optimizer."""
    from .data.tokenizers import build_tokenizer
    device = device or next(model.parameters()).device
    return types.SimpleNamespace(raw_model=model, device=device, _trunk_params=list(model.parameters()), rank=0, world=1,
                                 tokenizer=build_tokenizer(model.cfg.tokenizer),
                                 cfg=types.SimpleNamespace(context=int(model.cfg.context)),
                                 tc=types.SimpleNamespace(micro_batch=2))


def mount_anchors(model, anchors: dict, *, device=None, compile_chain: bool = False, recompute: bool = False,
                  require_step: int | None = STEP, phases: dict | None = None):
    """Mount anchor files on a trunk in the given order (a dict keeps insertion order: member -> path) by the arm program's own
    route: one StageArm per anchor from its metadata, the program bound to an inference stand-in, then load_state_dict attaches
    every member in order (fresh wrappers, then the saved tensors). Every adapter parameter is frozen. Returns the program;
    `prog.mounted` records the files, their base_model_id and tensor hashes."""
    from amoe.io.checkpoint import load_anchor
    from .train.arms import StageArmProgram, ArmProgramConfig, StageArm
    phases = phases or PHASES
    arms, adapters, info = [], {}, {}
    for name, path in anchors.items():
        ck = load_anchor(path)
        base_id = str(ck.meta.get("base_model_id", ""))
        if require_step is not None and not base_id.endswith(f"@step{require_step}"):
            raise ValueError(f"{name}: trained on {base_id!r}, this trunk is step {require_step}; arms are trunk-bound")
        n_blocks = len({k.split(".", 1)[0] for k in ck.adapters})
        if n_blocks != len(model.blocks):
            raise ValueError(f"{name}: the anchor carries {n_blocks} blocks, the trunk has {len(model.blocks)}")
        arms.append(StageArm(name=name, phase=str(ck.meta.get("phase", phases.get(name, "unknown"))), spec=_spec_of(ck.meta),
                             lam=float(ck.meta.get("lambda", 1.0)), seed=0))
        adapters[name] = ck.adapters
        info[name] = {"file": os.path.basename(path), "base_model_id": base_id, "spec": _spec_of(ck.meta),
                      "content_hash_v2": header_metadata(path).get("content_hash_v2")}
    prog = StageArmProgram(ArmProgramConfig(arms=arms, adapter_compile=bool(compile_chain), adapter_recompute=bool(recompute)))
    prog.bind(inference_binding(model, device))
    was_training = model.training
    prog.load_state_dict({"attached": list(anchors), "adapters": adapters, "disabled": {}})
    for p in prog.params():
        p.requires_grad_(False)
    if not was_training:
        model.eval()
    prog.mounted = {"members": list(anchors), "anchors": info}
    return prog


def mount_group(model, run: str = "gCA", close: int | None = None, members=None, *, repo: str = ANCHOR_REPO,
                local_dir: str | None = None, device=None, files: dict | None = None, base_files: dict | None = None,
                check_base: bool = True, compile_chain: bool = False, recompute: bool = False, require_step: int | None = STEP):
    """Mount a published group (GROUPS, or any run with `close` and `members`) on an already-built trunk. Downloads the anchors
    unless `files` (member -> path) is given; a group that extends an earlier one is checked against it (content_hash_v2 of the
    carried members) unless check_base=False. `compile_chain` runs the arm chain through torch.compile as training did.
    Returns the arm program: masks through masked()/only() or prog.handles[name], the exact detach through detach_all()."""
    if run in GROUPS:
        close = GROUPS[run][0] if close is None else int(close)
        members = list(GROUPS[run][1]) if members is None else list(members)
    if close is None or not members:
        raise ValueError("a run outside GROUPS needs close= and members=")
    files = files or anchor_files(run, close, members, local_dir, repo)
    checked = None
    if check_base and run in EXTENDS:
        base_run, base_close, base_members = EXTENDS[run]
        base_files = base_files or anchor_files(base_run, base_close, base_members, local_dir, repo)
        checked = check_frozen_members(files, base_files, [m for m in base_members if m in members])
    prog = mount_anchors(model, {m: files[m] for m in members}, device=device, compile_chain=compile_chain, recompute=recompute,
                         require_step=require_step)
    prog.mounted.update({"run": run, "close": close, "checked_against": EXTENDS[run][0] if checked else None})
    return prog


# ------------------------------------------------------------------- masks
@contextlib.contextmanager
def masked(prog, names):
    """The program with the named members switched off; their writes are removed, the others are not renormalized."""
    with contextlib.ExitStack() as stack:
        for n in names:
            stack.enter_context(prog.handles[n].all_off())
        yield prog


@contextlib.contextmanager
def only(prog, names):
    """The program with every attached member outside `names` switched off."""
    with masked(prog, [n for n in prog.attached if n not in names]):
        yield prog


def detach_all(prog, verify: bool = True):
    """Every member off the trunk, the last attached first. verify=True asserts the bare logits equal the pre-attach fingerprint
    bit for bit. Returns the model."""
    for n in reversed(list(prog.attached)):
        prog.handles[n].detach(verify=verify)
    prog.attached.clear()
    prog.handles.clear()
    prog.wraps.clear()
    return prog.model


def logits(model, x):
    out = model(x)
    return (out.logits if hasattr(out, "logits") else out[0]).float()


# ------------------------------------------------------------ surface arms
# A surface arm makes the trunk read a tokenizer's SPELLING of a text (train/surface.py) as it reads the text's own bytes. The
# published arms live on SURFACE_REPO under surface/<convention>/<run>/: the member's anchor (amoe format, base_model_id
# ...@step245674), the standardization statistics the loss used (stats.pt, for the record) and the readout. A surface arm trained
# over a frozen stage-arm group mounts over that group, in training order (the group from the training repo, the member from
# SURFACE_REPO); one trained on the bare trunk mounts alone.
SURFACE_REPO = "AbstractPhil/beatrix-tokenizers"
SURFACE_ARMS = {
    # name -> the registry row: convention (the spelling family of geolip.bytelex.extract), the reference tokenizer, the member,
    # the group it was trained over (run, close; None = the bare trunk), the anchor file on SURFACE_REPO, the seed label
    "qwen3": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s10_qwen", "base": ("gXA", 8),
              "file": "surface/qwen3/mse_gXA_o0/s10_qwen.safetensors", "seed": "A"},
    "qwen3-B": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s10_qwen", "base": ("gXB", 8),
                "file": "surface/qwen3/mse_gXB_o1/s10_qwen.safetensors", "seed": "B"},
    "qwen3-solo": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s10_qwen", "base": None,
                   "file": "surface/qwen3/mse_solo_o0/s10_qwen.safetensors", "seed": "A"},
    # the controls (mountable like the arms; what they are is in the registry): the shuffled-pairing control was trained against
    # the wrong targets; the untrained-copy control was trained on a random-initialization copy of the trunk (seed 0) and mounts
    # on such a copy, never on the trunk
    "qwen3-shuf": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s10_qwen", "base": ("gXA", 8),
                   "file": "surface/qwen3/mse_gXA_o0_shuf/s10_qwen.safetensors", "seed": "A", "control": "shuffled pairs"},
    "qwen3-untrained": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s10_qwen", "base": None,
                        "file": "surface/qwen3/mse_untrained_o0/s10_qwen.safetensors", "seed": "A", "control": "untrained copy (seed 0)",
                        "trunk": "untrained copy (seed 0)"},   # its anchor names step 0: the mount skips the step check for this row
    # the family (the same recipe; the lossy conventions read A as the text as written; sites = the token closings unless noted).
    # Placement: the solo read settled it (the Qwen arm alone on the bare trunk reads at least as well as the arm over the eight on
    # both bars, at two-thirds of the time a step, with nothing to mount underneath), so the family trains ALONE on the trunk
    # (base None) except the T5 seed-A arm, which had started over the eight before the read and stays as that convention's
    # placement datum. Populated as the arms land.
    "qwen3-bytes": {"convention": "gpt2", "tokenizer": "Qwen/Qwen3-0.6B", "member": "s14_qwen_bytes", "base": None,
                    "file": "surface/qwen3/mse_solo_o0_bytes/s14_qwen_bytes.safetensors", "seed": "A", "sites": "bytes"},
    "t5": {"convention": "sentencepiece", "tokenizer": "google/t5-v1_1-xxl", "member": "s11_t5", "base": ("gXA", 8),
           "file": "surface/t5/mse_gXA_o0/s11_t5.safetensors", "seed": "A"},
    "t5-solo": {"convention": "sentencepiece", "tokenizer": "google/t5-v1_1-xxl", "member": "s11_t5", "base": None,
                "file": "surface/t5/mse_solo_o0/s11_t5.safetensors", "seed": "A"},
    "t5-B": {"convention": "sentencepiece", "tokenizer": "google/t5-v1_1-xxl", "member": "s11_t5", "base": None,
             "file": "surface/t5/mse_solo_o1/s11_t5.safetensors", "seed": "B"},
    "clip": {"convention": "clip", "tokenizer": "openai/clip-vit-large-patch14", "member": "s12_clip", "base": None,
             "file": "surface/clip/mse_solo_o0/s12_clip.safetensors", "seed": "A"},
    "clip-B": {"convention": "clip", "tokenizer": "openai/clip-vit-large-patch14", "member": "s12_clip", "base": None,
               "file": "surface/clip/mse_solo_o1/s12_clip.safetensors", "seed": "B"},
    "bert": {"convention": "wordpiece", "tokenizer": "bert-base-uncased", "member": "s13_bert", "base": None,
             "file": "surface/bert/mse_solo_o0/s13_bert.safetensors", "seed": "A"},
    "bert-B": {"convention": "wordpiece", "tokenizer": "bert-base-uncased", "member": "s13_bert", "base": None,
               "file": "surface/bert/mse_solo_o1/s13_bert.safetensors", "seed": "B"},
}


def reader_kwargs(row: dict) -> dict:
    """The train.surface.read_spelled keyword arguments a registry row implies (its convention, its A text rule, its site set),
    so a mounted arm is read as it was trained: read_spelled(model, tok, texts, blocks, **reader_kwargs(prog.surface))."""
    from .train import surface as SF
    return {"convention": row["convention"], "a_text": row.get("a_text", "written"),
            "site_fn": SF.sites_every_byte if row.get("sites") == "bytes" else SF.sites}


def surface_files(name: str, local_dir: str | None = None, repo: str = SURFACE_REPO) -> dict:
    """member -> local path for a surface arm and, when it was trained over a group, that group's anchors first (in training
    order), downloaded unless present. The registry row is SURFACE_ARMS[name]."""
    from huggingface_hub import hf_hub_download
    row = SURFACE_ARMS[name]
    files = {}
    if row["base"] is not None:
        run, close = row["base"]
        files.update(anchor_files(run, close, GROUPS[run][1] if run in GROUPS else STAGE_ARMS, local_dir))
    files[row["member"]] = hf_hub_download(repo, row["file"], local_dir=local_dir, token=False)
    return files


def mount_surface(model, name: str = "qwen3", *, local_dir: str | None = None, repo: str = SURFACE_REPO, device=None,
                  compile_chain: bool = False, recompute: bool = False, require_step: int | None = STEP, files: dict | None = None):
    """Mount a published surface arm by name: its group (if any) then the member, by the training route (mount_anchors). Returns
    the arm program; prog.surface carries the registry row. The arm serves the tokenizer's spelling: read text through it with
    train.surface.read_spelled(model, tok, texts); mask it with masked(prog, [row['member']]) for the plain reading."""
    row = SURFACE_ARMS[name]
    if row.get("trunk", "").startswith("untrained"):
        require_step = None            # the control's arm was trained on a random-initialization copy (its anchor names step 0)
    files = files or surface_files(name, local_dir, repo)
    order = ([m for m in (GROUPS[row["base"][0]][1] if row["base"] and row["base"][0] in GROUPS else STAGE_ARMS)] if row["base"] else []) + [row["member"]]
    phases = dict(PHASES)
    phases[row["member"]] = f"surface:{row['convention']}"
    prog = mount_anchors(model, {m: files[m] for m in order}, device=device, compile_chain=compile_chain, recompute=recompute,
                         require_step=require_step, phases=phases)
    prog.surface = dict(row, name=name)
    return prog
