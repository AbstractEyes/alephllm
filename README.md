# AlephLLM

Signed-address (aleph) language models, trained end to end. The mechanism
family: dispatch weights are the closed-form signed address

    w_k = sinh(u_k) / Σ_j cosh(u_j),    u_k = cos(x̂, â_k) / τ

— reconstructive, never comparative (no argmax, no top-k, no
softmax-over-choices anywhere in a routing role). Inhibition is
first-class. New structure is always born on its own null path and must
earn its way in by gradient.

## Architecture (every choice measured on the probe-bed record)

```
trigram byte embedding (dedicated pad row — the pad law)
   ↓
N pre-norm layers:
   x + Attn(LN x)      CausalSDPA majority + 3 CausalSplatHUB layers
   x + Bank(LN x)      E1-form anchored FFN: trunk + 3 dispatched experts,
                       expert outputs ZERO-INIT (exact null path), gates
                       σ(-3), no balance machinery of any kind
   ↓
LayerNorm → DualHead:  logits = W_h h + γ·W_s·s(h),  γ = 0 at birth
```

CausalSplatHUB is causal **linear** attention through the oriented
address (prefix-sum memories over 2K half-axes), implemented as an exact
chunked scan; every hub layer is config-swappable to SDPA, so the pure-
SDPA craft is the running architecture control (`*-control` presets).

Training: **Muon** (Newton-Schulz orthogonalized momentum) on 2D/3D
transport weights + **pure Adam** (wd=0) on embeddings/1D — the measured
split; flat LR after a short warmup; bf16 autocast over fp32 masters;
never trains through fp8 (fp8-e4m3 is the shipping format only).

## Missions ("Mini-Beatrix" ladder, voyager-style numbering)

| craft | d / L / ctx | params | tokenizer |
|---|---|---|---|
| mini-beatrix-0 | 512 / 12 / 1024 | 37.6M | byte-trigram |
| mini-beatrix-1 | 768 / 16 / 2048 | 112.5M | byte-trigram |
| mini-beatrix-2 | 1024 / 20 / 2048 | 249.1M | byte-trigram |
| beatrix-voyager | 1536 / 24 / 4096 | 775.3M | BPE (gpt2) |

Each craft also has a `*-control` twin (hub layers removed). Training
runs, checkpoints, manifests and TensorBoard live in
[alephllm-mini-beatrix-training](https://huggingface.co/AbstractPhil/alephllm-mini-beatrix-training),
one prefix per craft. Every craft is inference-capable on consumer
hardware in its shipped form.

## Install & run

```
pip install git+https://github.com/AbstractEyes/alephllm
```

```python
from geolip.alephllm import prepare
run = prepare("mini-beatrix-1", hf_token=...)  # pulls manifest, resumes
run.train(max_hours=8)                          # tqdm + health readouts
run.evaluate()                                  # bpb, toggles, canaries
```

The Colab entrypoint notebook is
[notebooks/alephllm_colab.ipynb](notebooks/alephllm_colab.ipynb) —
install cell + prep/train/eval. Training is resume-first: stop any time,
rerun the notebook, it continues from the uploaded state.

## Instrumentation (born-in, no exceptions)

Effective-rank census (hidden states per layer, consumed address per hub,
codebooks, head), coefficient-of-variation load analysis, structural
collapse detectors (anchor merging, dispatch-entropy collapse, hidden-
erank floor, denominator floor rate, loss spikes), sign census, gate/γ
trajectories, anchor drift, and the **toggle ledger** — bank-off /
hub-off / head-aleph-off bpb deltas, the causal contribution instrument —
all on TensorBoard and in the runtime health readout.

## Tests

```
python -m geolip.alephllm.tests.smoke
```

17 mechanical cases: address identities, exact null paths, chunked-scan
vs naive-oracle equivalence, causality, optimizer-split coverage,
checkpoint/stream/manifest resume roundtrips, canary well-formedness,
a live 8-step train loop.

## Package layout

```
geolip/alephllm/
  presets.py        mission ladder + train configs
  model/            address · attention · bank · head · embedding · alephlm
  data/             tokenizers · streams (HF columnar streaming, resumable)
  train/            optim (Muon+Adam) · trainer · instruments · checkpoint · manifest
  eval/canaries.py  clean-protocol in-context binding probes
  tests/smoke.py    the full test array
```


**Technical companion:** [TECHNICAL.md](TECHNICAL.md) — architecture, training semantics, instruments, and the Beatrix-era numbers spine.
