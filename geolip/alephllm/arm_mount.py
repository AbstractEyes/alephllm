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
