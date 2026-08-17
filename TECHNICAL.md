# AlephLLM — Technical Companion (Beatrix era)

Companion to the article *Raising Beatrix: A Byte-Level Model's Measured
Childhood* (AbstractPhil with Claude Fable 5 & Claude Opus 5, August 2026).
This document carries the stack-side technical detail: architecture,
training semantics, data machinery, and the instrument suite. The run
record (checkpoints, reports, arms) lives in the training repository's
companion: [alephllm-mini-beatrix-training](https://huggingface.co/AbstractPhil/alephllm-mini-beatrix-training).

Package: `pip install geolip-alephllm @ git+https://github.com/AbstractEyes/alephllm`
(namespace package `geolip.alephllm`). Python ≥3.10, torch ≥2.4,
datasets ≥5.0.0 (the `Json` feature type used by curriculum sources).

## 1. Model: `AlephLM` (preset `mini-beatrix-1`)

112.5M parameters · d=768 · 16 pre-norm layers · context 2048 bytes ·
vocab 256 (raw UTF-8; no tokenizer).

**Input — TrigramByteEmbedding.** `e_t = E0[x_t] + E1[x_{t-1}] + E2[x_{t-2}] + P[t]`
with E0: 256×d and E1,E2: 257×d — the 257th row is a dedicated PAD
symbol for sequence starts (input-only; unemittable). The pad row is a
law, not an option: padding trigram history with a legal byte conflates
real history with beginnings and measurably starves address consumption
(+.05–.11 on repair tasks). Composed input space ≈ 256×257×257 ≈ 16.9M
states from ~591k parameters.

**Attention.** Layers {4, 9, 14} carry `CausalSplatHUB`: a linear-cost
causal read where queries are scored against a learned anchor codebook
by the signed address rule (below) and values accumulate through an
exact chunked prefix scan (verified equal to a naive oracle; the 0.6.1
fused path is fp-reorder-equal at ~1e-6). The other 13 layers use
standard SDPA. Removing the hubs costs +3.7 to +4.9 bits/byte at the
graduate (toggle ledger) — the largest single-mechanism contribution in
the model.

**Feed-forward.** Every layer's FFN is an anchored expert **bank**:
experts dispatched per-token by signed addresses
`w_k = sinh(u_k) / Σ_j cosh(u_j)` over learned unit anchors — no
softmax-over-choices, no top-k, inhibition (negative dispatch) as a
first-class citizen (lifetime sign census: ~46–49% negative). Expert
outputs are zero-initialized (**born null, weight-zero**): at step 0
the bank contributes exactly nothing, bit for bit. At the graduate,
removing the banks costs +2.2 to +2.6 bits/byte.

**Head — DualHead.** `logits = W_h·h + W_s·s(h)` where `s(h)` is the
signed address of a low-D projection of h through a 512-anchor head
codebook, `W_s` zero-initialized. A scalar γ exists only as a frozen
ablation knob. History that became law: the original γ-gated birth
DEADLOCKS (W_s's gradient is scaled by γ=0, and γ's gradient through a
random readout is zero-mean noise) — the **GATE-ZERO DEADLOCK law**:
born-null must be weight-zero, never gate-zero. A mid-run migration
fossilized γ at its checkpoint value (0.0015); the repair folded γ into
W_s (`W_s *= γ; γ := 1`), a measured semantic no-op (max logit diff
2.4e-07). Post-fold the head grew ‖W_s‖ 0.027 → 24.8 across the
curriculum while its causal contribution stayed exactly 0.0000 — the
program's standing open mechanism question. Fresh constructions are
immune (γ initializes to 1.0, frozen).

**Optimizer split (measured, house law).** Muon for 2D/3D non-embedding
transport weights; pure Adam (wd=0) for embeddings and 1-D parameters;
flat LR after short warmup; bf16 autocast over fp32 masters; fp8-e4m3
is a shipping format only, never trained through.

## 2. Trainer semantics

`prepare("mini-beatrix-1", hf_token=…)` pulls `manifest.json` +
`resume/latest.pt` from the training repo and continues — training is
**resume-first**; sessions are Colab-shaped and interrupt-safe
(KeyboardInterrupt checkpoints and uploads; crashes never overwrite the
hub resume; non-finite loss/grad raises before the optimizer step).

- **Session-relative caps** (0.5.1): `max_tokens`/`max_steps` count the
  current call. A cap prints "SESSION CAP — a pause, not completion"
  and the summary names remaining curriculum tokens. (The original
  lifetime-total semantics once dressed a mid-phase halt as completion;
  the fix is regression-tested.)
- **Manifest phases**: the full curriculum lives in the manifest with
  per-phase datasets/budgets/status (`planned|active|done|deferred`);
  `current_phase()` activates planned phases in order. The notebook's
  school loop caps each `train()` call at the active phase's remaining
  tokens so sessions stop exactly at stage boundaries, where the
  boundary instruments fire.
- **Display is never allowed to kill training**: every bar print goes
  through a guarded emitter (tqdm's class-level write refreshes foreign
  bars and can raise from notebook widgets — it aborted a session at
  step 59,000 before the guard).

## 3. Data machinery (`data/streams.py`, `data/curriculum.py`)

`PackedStream`: HF streaming → column-pruned rows → renderer → byte
packing into (context+1) blocks, with resumable state (the packing
buffer rides in the state; shuffled HF streams resume approximately —
measured and documented, val streams are fixed unshuffled slices).
Registry entries declare `path/name/split/column`, optional `render`
(dispatched via a renderer registry), optional `max_empties` (the
circuit breaker for row-filtering renderers), and val provenance
(explicit split, or reserved head rows the training stream skips).

`MixStream`: weighted multiplex of PackedStreams with a deterministic
draw counter — exactly reproducible interleaving, per-component
resumable state. The anneal recipe and all nine curriculum stages are
recipes over this machinery. **Gauge continuity law**: every curriculum
stage vals on the same fineweb-edu holdout as pretraining, so one gauge
spans the whole life. (Census fact recorded with the gauge: that
holdout is 100% crawl-year 2013.)

**Curriculum (`curriculum.py`)**: stage mixes S0–S8 (~8.8B bytes) over
vetted sources (TinyStories, simple Wikipedia, AO-CHILDES,
cosmopedia-v2 filtered by its per-row `audience` column, SIQA/bAbI/
ProofWriter/ATOMIC parquet mirrors, opengloss definitions) plus eight
owned template-free generators (counting, perspective/false-belief,
concept flickers, rule-chains, arithmetic, causal arcs,
try-fail-adjust, register parallax). **Holdouts are enforced at the
renderer/generator level** — bAbI families 16/19, ProofWriter and
rule-chain depth ≥5, three-digit subtraction never reach a training
row; they exist for the arm program's causal experiments. Laws
embedded: no chat template, no identity rows, no boilerplate headers in
any core corpus (the chat-in-anneal law); dialogue enters only as
narrative quotation.

## 4. Instruments (`train/instruments.py`, `train/probes.py`, `eval/lexicon.py`)

- **Census**: replay-based (reads the exact pre-norm tensors each
  mechanism consumes): per-layer hidden effective rank, hub/bank/head
  anchor health (usage perplexity, drift, |cos|>0.99 merge pairs),
  dispatch entropy, sign census, hub denominator floors, collapse
  flags. Census probes are sample-size sensitive — cross-protocol
  comparisons are invalid (recorded after a synthetic-stream census
  misread as an erank explosion).
- **Toggle ledger**: bits-per-byte with each mechanism disabled — the
  causal contribution gauge, run at every eval and every curriculum
  boundary.
- **Exams** (`probes.py` + `eval/exams/*.jsonl`): 270 surface-disjoint
  items (9 suites × 30, difficulty 1–3, family labels, holdout families
  as exam sections), scored by byte-NLL over options. Evaluation laws,
  each purchased by a measured failure: items surface-disjoint from all
  training generators (the probe-validity law); MC-likelihood exams are
  blind to arms that reshape output format — capability verdicts need a
  generative tier; answer matching must be order-aware (substring
  matching inflated 1.00 vs 0.70 strict).
- **Lexicon census** (`eval/lexicon.py`): reads the learned vocabulary
  — surprisal-boundary segmentation at a BPE-matched rate vs
  address-switch segmentation per hub, boundary-F1 agreement, unit
  inventories, per-anchor exemplar dictionaries. Findings: her units
  are word+trailing-space (BPE's convention is leading-space);
  address/surprisal agreement rises monotonically with hub depth (L4
  0.07 → L14 0.36–0.39); a deep-hub anchor is a de-facto whitespace
  anchor (88% of its spans); the head codebook became word-aligned
  during the curriculum while contributing zero bpb.

## 5. The numbers spine (details in the training-repo companion)

| era | steps | bytes | fineweb val bpb |
|---|---|---|---|
| pretrain lock | 51,882 | 15.30B | 1.033 |
| anneal end | 58,664 | 17.30B | 1.045 (retro same-gauge: 1.0479) |
| S3 conviction (discarded path) | ~— | ~20B era | spike to 1.71–2.05 |
| rewind point | 66,803 | — | — |
| spiral (S7) end | 85,795 | 25.30B | 1.111 |
| graduation | 88,508 | 26.10B | 1.121 |
| benchmark-best (exam mean) | 83,903 | — | 1.118 |

Toggle ledger trend (same gauge, dated ledgers): banks +0.30 → +1.06
across steps 4k–10k (2.9B; ~25–50× the encoder-era toggle class) →
+2.08 @8.4B → +2.29 @lock → +2.61 (curriculum-era high); hubs +0.73 →
+1.61 (10k) → +2.54 @8.4B → +3.00 by 13.4–14.0B → +3.25 @lock → +3.73
post-anneal → 4.3–4.9 (curriculum era). Head: 0.0000 at every
measurement in her life.

## 6. Reuse map

Smoke: `python -m geolip.alephllm.tests.smoke` (26 cases). Notebooks:
`notebooks/alephllm_colab.ipynb` (pretrain era, archived) and
`notebooks/beatrix_curriculum.ipynb` (the school notebook: one-time
instrument gate, boundary-exact session driver, growth table, one-shot
rewind cell). The arm system is external:
[amoe-lora](https://github.com/AbstractEyes/amoe-lora). The byte
information system is external:
[geolip-bytelex](https://github.com/AbstractEyes/geolip-bytelex).
