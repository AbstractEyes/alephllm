"""Checkpointing + HF upload.

Three artifact kinds, all under the craft's prefix in the training repo:
  checkpoints/step_XXXXXXXX.safetensors   bf16 weights (step cadence)
  checkpoints/fp8/step_XXXXXXXX.safetensors  fp8-e4m3 2D weights + scales
      (every Nth checkpoint — the SHIPPING/inference-test format; never
      train from these)
  resume/latest.pt                        full resume state: fp32 model,
      both optimizers, data-stream state, manifest snapshot, RNG
  runs/...                                tensorboard event files

Uploads retry with backoff and never crash training — a failed upload is
logged and retried at the next cadence.
"""
from __future__ import annotations

import io
import json
import os
import time

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file, load_file

from .manifest import RunManifest


def _retry(fn, tries: int = 3, wait: float = 5.0, label: str = "op"):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — uploads must not kill training
            if i == tries - 1:
                print(f"[upload] {label} FAILED after {tries} tries: {e}")
                return None
            time.sleep(wait * (i + 1))


class HubSync:
    def __init__(self, repo: str, prefix: str, token: str | None,
                 local_dir: str):
        self.repo, self.prefix, self.token = repo, prefix, token
        self.local = local_dir
        os.makedirs(os.path.join(local_dir, "checkpoints", "fp8"), exist_ok=True)
        os.makedirs(os.path.join(local_dir, "resume"), exist_ok=True)
        self.api = HfApi(token=token) if token else None
        if self.api:
            _retry(lambda: self.api.create_repo(
                self.repo, exist_ok=True, private=False), label="create_repo")

    # ------------------------------------------------------------ helpers
    def _up(self, local_path: str, repo_path: str):
        if not self.api:
            return
        _retry(lambda: self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{self.prefix}/{repo_path}",
            repo_id=self.repo), label=repo_path)

    def upload_bytes(self, data: bytes, repo_path: str):
        if not self.api:
            return
        _retry(lambda: self.api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=f"{self.prefix}/{repo_path}",
            repo_id=self.repo), label=repo_path)

    def download(self, repo_path: str) -> str | None:
        """None means THE FILE DOES NOT EXIST on the hub. Transient errors
        retry and then RAISE — a flaky network must never be mistaken for
        'no resume state exists' (that mistake overwrites real training
        state with a fresh run)."""
        if not self.api:                      # tokenless: local artifacts only
            local = os.path.join(self.local, repo_path)
            return local if os.path.exists(local) else None
        from huggingface_hub.errors import EntryNotFoundError, \
            RepositoryNotFoundError, RevisionNotFoundError
        last = None
        for i in range(4):
            try:
                return hf_hub_download(self.repo, f"{self.prefix}/{repo_path}",
                                       token=self.token,
                                       local_dir=os.path.join(self.local, "_pull"),
                                       force_download=True)
            except (EntryNotFoundError, RepositoryNotFoundError,
                    RevisionNotFoundError):
                return None                   # genuine absence
            except Exception as e:  # noqa: BLE001 — transient: retry then raise
                last = e
                time.sleep(5.0 * (i + 1))
        raise RuntimeError(
            f"could not download {self.prefix}/{repo_path} after 4 tries "
            f"(NOT treating this as 'no resume exists'): {last}")

    # ----------------------------------------------------------- weights
    def save_safetensors(self, model, step: int) -> str:
        sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
              for k, v in model.state_dict().items()}
        name = f"checkpoints/step_{step:08d}.safetensors"
        path = os.path.join(self.local, name)
        save_file(sd, path, metadata={"step": str(step), "format": "bf16"})
        self._up(path, name)
        return name

    def save_fp8(self, model, step: int) -> str:
        """fp8-e4m3 shipping variant: 2D+ weights stored fp8 with a
        per-tensor fp32 scale (amax -> 448); small params kept bf16."""
        sd, out = model.state_dict(), {}
        for k, v in sd.items():
            v = v.detach().float().cpu()
            if v.ndim >= 2 and v.numel() > 4096:
                scale = v.abs().amax().clamp_min(1e-12) / 448.0
                out[k] = (v / scale).to(torch.float8_e4m3fn)
                out[k + ".scale"] = scale.reshape(1)
            else:
                out[k] = v.to(torch.bfloat16)
        name = f"checkpoints/fp8/step_{step:08d}.safetensors"
        path = os.path.join(self.local, name)
        save_file(out, path, metadata={"step": str(step), "format": "fp8-e4m3",
                                       "note": "inference/testing only"})
        self._up(path, name)
        return name

    # ------------------------------------------------------------ resume
    def save_resume(self, model, optimizers, stream_state: dict,
                    manifest: RunManifest, step: int) -> str:
        payload = {
            "model": model.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "stream": stream_state,
            "manifest": json.loads(manifest.to_json()),
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)},
            "step": step,
        }
        path = os.path.join(self.local, "resume", "latest.pt")
        torch.save(payload, path)
        self._up(path, "resume/latest.pt")
        return "resume/latest.pt"

    def load_resume(self) -> dict | None:
        """Hub is authoritative when a token is present — a stale local
        file must never shadow (or be shadowed by) the hub state silently.
        Tokenless runs use local state only."""
        path = self.download("resume/latest.pt")
        if path is None:
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    # ---------------------------------------------------------- manifest
    def push_manifest(self, manifest: RunManifest):
        self.upload_bytes(manifest.to_json().encode(), "manifest.json")

    def pull_manifest(self) -> RunManifest | None:
        path = self.download("manifest.json")
        if path is None:
            return None
        return RunManifest.from_json(open(path, encoding="utf-8").read())

    # ------------------------------------------------------- tensorboard
    def upload_tensorboard(self, tb_dir: str):
        if not self.api or not os.path.isdir(tb_dir):
            return
        _retry(lambda: self.api.upload_folder(
            folder_path=tb_dir,
            path_in_repo=f"{self.prefix}/runs",
            repo_id=self.repo,
            allow_patterns=["*tfevents*"]), label="tensorboard")


def load_fp8_state(path: str) -> dict:
    """Rehydrate an fp8 shipping checkpoint to bf16 tensors."""
    raw = load_file(path)
    out = {}
    for k, v in raw.items():
        if k.endswith(".scale"):
            continue
        if v.dtype == torch.float8_e4m3fn:
            out[k] = (v.float() * raw[k + ".scale"]).to(torch.bfloat16)
        else:
            out[k] = v
    return out
