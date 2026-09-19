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
   x + Attn(LN x)      CausalSDPA or CausalSplatHUB per layer (v1: 3
                       hub layers; the v2 era is FULL SPLAT — a governed
                       multi-constellation hub in EVERY block)
   x + Bank(LN x)      E1-form anchored FFN: trunk + 3 dispatched experts,
                       expert outputs ZERO-INIT (exact null path), gates
                       σ(-3), no balance machinery of any kind
   ↓
LayerNorm → DualHead:  logits = W_h h + W_s·s(h),  W_s = 0 at birth
                       (weight-zero, never gate-zero — the measured law;
                       γ survives frozen at 1 as the ablation knob)
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

| craft | d / L / ctx | params | tokenizer | status |
|---|---|---|---|---|
| mini-beatrix-0 | 512 / 12 / 1024 | 37.6M | byte-trigram | gate craft |
| mini-beatrix-1 | 768 / 16 / 2048 | 112.5M | byte-trigram | COMPLETE — [automodel](https://huggingface.co/AbstractPhil/mini-beatrix-1) |
| **mini-beatrix-2s** | 1024 / 20 / 4096 | 237.1M | byte-trigram | **COMPLETE 2026-08-31, 16.101B tokens, full splat — [automodel](https://huggingface.co/AbstractPhil/mini-beatrix-2s)** |
| mini-beatrix-3 | 1024 / 24–28 / 4096 | 283–330M | byte-trigram | the v3 routine craft (`make_v3_preset`); designed, decisions pending — [notebook](notebooks/beatrix_v3_colab.ipynb) |
| mini-beatrix-2 | 1024 / 32 / 8192 | 849.0M | byte-trigram | full splat; shelved pending logistics |
| beatrix-voyager | 1536 / 24 / 4096 | 775.3M | BPE (gpt2) | awaits BPE screens |

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

The v3 routine notebook is
[notebooks/beatrix_v3_colab.ipynb](notebooks/beatrix_v3_colab.ipynb):
a decision block (depth, data scale, epoch cap, rebalance rule, anneal
multiplier, arm geometry, stage arms, waivers) that refuses to run while
unfilled, a preflight with a micro-batch ladder and a price line, the data
plane with its epoch tables, the guard core from the certification ledger,
a push probe, the boundary-exact session driver (reports, arm anchors,
clean halts), samples, the anneal watch and the growth table. `LOCAL = True`
runs the whole notebook on a CPU toy craft.

### Multi-card (data-parallel, one machine)

The same session driver runs on several cards through `torchrun`; the
notebook's decision block becomes a JSON file:

```
ALEPHLLM_MISSION_CONFIG=v3.json torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.train.mission
```

Every rank builds the same craft, reads a disjoint slice of every stream
(so a finite corpus keeps the epoch count the recipe planned), and the
gradients — trunk and arms — are averaged across ranks before the clip and
the optimizer steps; the step is the GLOBAL batch (`micro_batch x
grad_accum x cards x context` must equal the recipe's tokens per step),
so the schedule and every settled number carry over unchanged. Rank 0
owns the record (checkpoints, manifest, reports, uploads); a checkpoint
carries every rank's stream position, a same-world resume continues each
slice exactly, and the parameters are re-broadcast from rank 0 at every
checkpoint and boundary write. A two-rank run reproduces a single-card run
bit for bit when both ranks read the same data (the parity smoke below).

## Instrumentation (born-in, no exceptions)

Effective-rank census (hidden states per layer, consumed address per hub,
codebooks, head), coefficient-of-variation load analysis, structural
collapse detectors (anchor merging, dispatch-entropy collapse, hidden-
erank floor, denominator floor rate, loss spikes), sign census, gate/γ
trajectories, anchor drift, and the **toggle ledger** — bank-off /
hub-off / head-aleph-off bpb deltas, the causal contribution instrument —
all on TensorBoard and in the runtime health readout.

## Relay adapters (`model/relay.py`)

`RelayPatchwork` is a gated residual patch module over the signed address
read: project into slots → reconstructive read against a learned codebook →
squared-ReLU head, output layer zero-initialized (weight and bias) so a
fresh module is exactly inert (y == x) until gradient earns it in.
`RelayEMA` widens the same head with two fixed-decay causal EMAs of the
module's own read (r = 1/16, 1/64); the added input columns start at zero,
so at birth it equals the plain patchwork. Training uses an exact chunked
closed-form scan; decode carries (F1, F2) state one position at a time.
Ported from [amoe-lora](https://github.com/AbstractEyes/amoe-lora);
validation record and trained weights:
[mini-beatrix-2s](https://huggingface.co/AbstractPhil/mini-beatrix-2s),
`arms/btx_e003`.

## Tests

```
python -m geolip.alephllm.tests.smoke
```

47 mechanical cases: address identities, exact null paths, chunked-scan
vs naive-oracle equivalence, causality, optimizer-split coverage,
checkpoint/stream/manifest resume roundtrips, crash safety (divergence
never overwrites resume state), multi-constellation hub equivalence,
governor projection, special-token laws, head revival, relay birth /
scan / decode parity, canary well-formedness, a live 8-step train loop;
and the v3 set — the v3 preset and its twins, the curriculum scaler
(epoch cap, three rebalance rules, refusal without a rule, 1x restored
bit-exact), the guard core (G1/G2/G3 replay; a halt that archives its
position, never rewrites the last healthy resume point, and refuses to
continue until a session clears it), per-phase LR multipliers, and the
stage-arm program (bit-inert attach, plain trunk keys, one shared step,
gauges, resume, a disabled member), and weak-token fusion (identity with
no middle, forced starts, causality, cached decode against the parallel
path for rows with different unit structures, the entropy and hybrid rules
over an atlas table, gradients reaching the null vector and the middle).
The stage-arm case needs the `amoe` package with its `alephlm` binding on
the path (the repo source, not a stale installed copy). To run the suite
off-card on a CUDA build, hide the card with `CUDA_VISIBLE_DEVICES=-1`
(an empty value makes torch report a card that the runtime cannot open).

The multi-card path has its own smoke (CPU over gloo, or cards over nccl):

```
python -m geolip.alephllm.tests.smoke_multicard parity            # the single-process reference hash
torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.tests.smoke_multicard parity   # must print the same hash
torchrun --standalone --nproc_per_node=2 -m geolip.alephllm.tests.smoke_multicard full     # boundaries, arm attach, resume
```

(`ALEPHLLM_SMOKE_CPU=1 ALEPHLLM_DIST_BACKEND=gloo` for a CPU run; on
Windows launch the ranks by hand with `MASTER_ADDR/MASTER_PORT/RANK/
WORLD_SIZE` and `USE_LIBUV=0`, since torchrun's rendezvous ignores the
libuv switch there.)

## Package layout — every code piece, briefly

`presets.py` — the mission ladder: model + train configs per craft,
including the `*-control` twins; `make_v3_preset` builds the v3 craft at
a chosen depth with its data plane (scale, epoch cap, rebalance rule) and
the two-phase anneal planned from birth.

**model/**

| file | what it is |
|---|---|
| `address.py` | `AlephAddress` — the closed-form signed address over 2K oriented half-axes; `signed` / `oriented` reads plus a codebook health census (drift, merging, effective rank, usage) |
| `attention.py` | `CausalSDPA`, the workhorse block, and `CausalSplatHUB` — causal linear attention through the oriented address, an exact chunked scan with a fused fast path and a naive oracle for parity |
| `bank.py` | `AnchoredBank` — the anchored FFN: always-on trunk + 3 dispatched experts, expert outputs zero-initialized (exact null path), no balance machinery |
| `head.py` | `DualHead` — standard readout plus an aleph read whose weights are zero at birth (weight-zero, never gate-zero) |
| `embedding.py` | `TrigramByteEmbedding` — composed byte embedding e_t = E0[x_t] + E1[x_{t-1}] + E2[x_{t-2}] + P[t], with a dedicated pad row; `TokenEmbedding` for BPE crafts |
| `fusion.py` | weak-token fusion at the input plane (0.9.0, opt-in via `AlephLMConfig.fusion`): a causal unit-start rule — the atlas entropy table over the three previous bytes (`entropy`), word starts (`spacelike`), or word starts kept only at choice points (`hybrid`), with forced starts at specials and newlines — and the hourglass wiring: front blocks at byte resolution, the middle blocks over units (a position read at each unit's last byte, never a mean pool), an unpool shifted by one unit, back blocks and the head at byte resolution; cached decode keeps a unit-level cache |
| `governor.py` | the anchor governor — a min-separation projection that relaxes crowded codebooks; exact identity when anchors have room |
| `relay.py` | `RelayPatchwork` / `RelayEMA` — the relay adapters above |
| `alephlm.py` | `AlephLM` — the full craft: embedding → pre-norm stack → `DualHead`, with cached prefill/step decode and per-mechanism toggle switches for ablation |

**data/**

| file | what it is |
|---|---|
| `tokenizers.py` | the byte tokenizer (vocab 256; trigram composition lives in the embedding) and an HF BPE wrapper |
| `streams.py` | resumable packed streaming from HF hub datasets; stream state rides in checkpoints |
| `curriculum.py` | staged training mixes S0–S8 with procedural generators, the epoch-cap and ballast audits that guard every mix, and the scaler that rebalances the mixes under an epoch cap at a larger data budget (three rebalance rules; refuses without one) |
| `special_tokens.py` | control tokens placed in invalid-UTF-8 byte space (cannot collide with any real text), document packing, and the chat frame |

**train/**

| file | what it is |
|---|---|
| `optim.py` | the measured optimizer split: Muon (Newton-Schulz orthogonalized momentum) on 2D transport weights, pure Adam (wd 0, never AdamW) on the rest |
| `trainer.py` | the resume-first training loop: pulls manifest + state from the hub, session caps, crash-safe checkpointing, structured health readouts; per-phase LR multipliers, the guard core and the stage-arm program ride in the same step; a data-plane fingerprint is asserted on resume; multi-card (0.10.0): per-rank stream shards, gradients averaged across ranks before the clip, agreed stop decisions, rank-0 record, per-rank stream positions in every checkpoint |
| `mission.py` | the session driver as a module (the notebook's decision block, preset, guard core, push probe and boundary loop) for `torchrun` on one or many cards; configuration by JSON, no argparse; the JSON is re-read at every boundary (0.10.1), so the later gates — the arm certification flag and the anneal multiplier — land in the file while the mission trains, without a relaunch |
| `precision.py` | one autocast policy: bf16 on a card, no autocast context off it (a disabled CUDA autocast still queries the card) |
| `guards.py` | the red-flag guard core: three in-run evaluators (norm surge, dispatch-entropy collapse, rank collapse) over a pinned reference window, per-guard modes (halt / watch / off) filled from a certification ledger; a halt archives the position under its own name and the run refuses to continue until cleared |
| `arms.py` | the stage-arm program: fresh relay arms attached per curriculum stage (bias-zeroed, inert at birth), trained under one pure-Adam group beside the trunk with a per-member abstention term on off-domain rows; masked-detachability gauges, anchors, resume, and member disabling on a fault |
| `instruments.py` | the born-in gauge suite: effective-rank census, collapse detectors, sign census, gate trajectories, and the toggle ledger (per-mechanism causal contribution) |
| `checkpoint.py` | bf16 checkpoints, fp8 shipping copies, resume state, and their HF uploads |
| `manifest.py` | `RunManifest` — the run's state of record on the hub; pull → resume |
| `probes.py` | stage probe batteries P0–P8, multiple-choice scored by byte-NLL of the option continuations |
| `revival.py` | head revival — a boundary write that restores a buried aleph head's contribution without moving the loss |

**eval/**

| file | what it is |
|---|---|
| `canaries.py` | synthetic in-context binding probes, run against checkpoints throughout training |
| `lexicon.py` | lexicon census — reads the learned vocabulary via two independent byte segmentations (surprisal boundaries vs address-switch boundaries) and scores their agreement |
| `tokenbridge.py` | byte-level token translation matrix for comparing against foreign tokenizers |
| `exams/*.jsonl` | the 270-item surface-disjoint exam battery (9 suites, difficulty tiers + holdouts) |

**bridges & conditioning**

| file | what it is |
|---|---|
| `amoe_bridge.py` | attach/train amoe-lora arms on a locked core: byte chat rows, provenance-stamped anchors, exact-prefix masking |
| `chat_sft.py` | the first chat-conditioning recipe — produces a detachable chat arm from a locked core |

`tests/smoke.py` — the full 47-case test array above; `tests/smoke_multicard.py` — the multi-card parity and full-path smokes.


**Technical companion:** [TECHNICAL.md](TECHNICAL.md) — architecture, training semantics, instruments, and the Beatrix-era numbers spine.
