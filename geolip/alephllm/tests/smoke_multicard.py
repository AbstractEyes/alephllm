"""The multi-card smoke (CPU, gloo; or cards, nccl). Two checks:

  parity  — every rank reads the SAME stream (ALEPHLLM_NOSHARD=1); after N
            steps the weights must equal a single-process run's bit for bit
            (the gradient average of identical gradients is that gradient:
            the step-parity law applied to the data-parallel path)
  full    — sharded streams; the run crosses three phase boundaries, a stage
            arm attaches on every rank, the abstention chunks run, a
            checkpoint carries every rank's stream position, a resume with
            the same world continues, and the weights agree across ranks

Run:  torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.tests.smoke_multicard parity
      python -m geolip.alephllm.tests.smoke_multicard parity          (the single-process reference)
      torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.tests.smoke_multicard full
The parity hashes are printed; the driver script compares them.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import torch


def _hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _preset(name: str, arms: bool):
    from ..presets import make_v3_preset, AlephLMConfig, PRESETS
    p = make_v3_preset(2, data_scale=4.0, epoch_cap=2.0, rebalance_to="generators", name=name)
    p.model = AlephLMConfig(name=name, d_model=64, n_layers=2, n_heads=1, context=128,
                            hub_layers=(0, 1), hub_K=4, hub_D=8, hub_const=2,
                            bank_experts=3, bank_ff=64, head_K=8, head_D=16, hub_chunk=32, hub_ckpt=0)
    p.train.micro_batch, p.train.grad_accum, p.train.warmup_steps = 2, 1, 2
    p.train.log_every, p.train.health_every, p.train.eval_every = 2, 1000, 1000
    p.train.ckpt_every, p.train.tb_upload_every, p.train.val_tokens, p.train.canary_episodes = 1000, 1000, 256, 1
    p.train.head_addr_frozen = False
    p.curriculum = [dict(ph, dataset="synthetic", planned_tokens=1024) for ph in p.curriculum]
    PRESETS[name] = p
    return p


def _arms():
    from ..train.arms import StageArm, ArmProgramConfig, StageArmProgram, LAWFUL_16x8
    return StageArmProgram(ArmProgramConfig(
        arms=[StageArm("s1_toy", "curriculum_s1", spec=dict(LAWFUL_16x8), lam=1.0, seed=1)],
        offdomain_dataset="synthetic"))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    from ..train.trainer import Trainer
    root = os.environ.get("ALEPHLLM_SMOKE_DIR") or os.path.join(tempfile.gettempdir(), f"alephllm_smoke_{mode}_w{world}")
    if rank == 0 and os.environ.get("ALEPHLLM_SMOKE_KEEP") != "1":
        shutil.rmtree(root, ignore_errors=True)
    dev = "cpu" if os.environ.get("ALEPHLLM_SMOKE_CPU") == "1" or not torch.cuda.is_available() else "cuda"
    if world > 1:
        import torch.distributed as dist
        backend = os.environ.get("ALEPHLLM_DIST_BACKEND") or ("nccl" if dev == "cuda" else "gloo")
        dist.init_process_group(backend)
        if dev == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.barrier()
    if mode == "parity":
        os.environ["ALEPHLLM_NOSHARD"] = "1"
        p = _preset("smoke-parity", arms=False)
        t = Trainer(p, hf_token=None, out_dir=root, device=dev, resume=False, guard=None, arms=None)
        t.train(max_steps=3)
        h = _hash(t.raw_model)
        print(f"[smoke parity] world {world} rank {rank}: step {t.step} tokens {t.manifest.tokens_seen} hash {h}", flush=True)
        if world > 1:
            out = [None] * world
            dist.all_gather_object(out, h)
            assert len(set(out)) == 1, f"ranks disagree after the step: {out}"
            if rank == 0:
                print(f"[smoke parity] all {world} ranks agree: {out[0]}", flush=True)
        if rank == 0:
            with open(os.path.join(root, f"hash_w{world}.txt"), "w") as f:
                f.write(h)
    elif mode == "full":
        p = _preset("smoke-full", arms=True)
        t = Trainer(p, hf_token=None, out_dir=root, device=dev, resume=False, guard=None, arms=_arms())
        # 1,024 tokens per phase; 2 rows x 128 ctx x world tokens/step -> the 4th phase (curriculum_s1) opens
        # after 3 x ceil(1024 / (256 * world)) steps; the arm attaches there and trains under the abstention chunks
        per_phase = -(-1024 // (256 * world))
        t.train(max_steps=3 * per_phase + 2)
        assert t.arms.attached == ["s1_toy"], f"rank {rank}: arms {t.arms.attached} at step {t.step}"
        h = _hash(t.raw_model)
        print(f"[smoke full] world {world} rank {rank}: step {t.step} phase {t.manifest.current_phase()['name']} "
              f"arms {t.arms.attached} hash {h}", flush=True)
        if world > 1:
            out = [None] * world
            dist.all_gather_object(out, h)
            assert len(set(out)) == 1, f"ranks disagree: {out}"
            dist.barrier()
        step0 = t.step
        # the resume: the final checkpoint of train() wrote resume/latest.pt (tokenless = local) with every
        # rank's stream position; a same-world resume continues each slice
        t2 = Trainer(p, hf_token=None, out_dir=root, device=t.device, resume=True, guard=None, arms=_arms())
        assert t2.step == step0, f"rank {rank}: resumed step {t2.step} != {step0}"
        assert t2.arms.attached == ["s1_toy"], f"rank {rank}: resumed arms {t2.arms.attached}"
        assert _hash(t2.raw_model) == h, f"rank {rank}: resumed weights differ"
        t2.train(max_steps=2)
        assert t2.step == step0 + 2
        h2 = _hash(t2.raw_model)
        if world > 1:
            out = [None] * world
            dist.all_gather_object(out, h2)
            assert len(set(out)) == 1, f"ranks disagree after the resume: {out}"
        print(f"[smoke full] world {world} rank {rank}: resumed {step0} -> {t2.step}, arms {t2.arms.attached}, "
              f"stream_ranks {'carried' if world > 1 else 'n/a'}, hash {h2} — OK", flush=True)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
