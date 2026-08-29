"""geolip.alephllm — AlephLLM training and inference stack.

Signed-address (aleph) language models: trigram byte embeddings with the
dedicated pad row, a pre-norm stack of SDPA + CausalSplatHUB attention,
E1-form anchored FFN banks born on their own null path, and a dual output
head whose aleph read enters at gamma=0 and must earn its way in by
gradient.

Notebook API:
    from geolip.alephllm import prepare, PRESETS
    run = prepare("mini-beatrix-1", hf_token=...)   # pulls manifest, resumes
    run.train(max_hours=8.0)                        # tqdm + health readouts
    run.evaluate()                                  # bpb, toggles, canaries
"""

__version__ = "0.8.3"

from .presets import PRESETS, AlephLMConfig, TrainConfig, Preset, get_preset
from .model.alephlm import AlephLM
from .train.trainer import Trainer, prepare
from .train.manifest import RunManifest

__all__ = [
    "PRESETS", "AlephLMConfig", "TrainConfig", "Preset", "get_preset",
    "AlephLM", "Trainer", "prepare", "RunManifest", "__version__",
]
