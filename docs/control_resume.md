# The control twin, continued: the softmax guards and the weights-only start (0.10.9; the fused kernel 0.10.11; fp16 flash in the fp32 block 0.10.12; continuation arms and the kernel's counters 0.10.13)

`mini-beatrix-2s-control` is the pure-softmax twin of the full-splat `mini-beatrix-2s` craft (d1024, 20 blocks, 16 heads,
ctx 4096, byte-trigram; the same anchored banks, dual head, Muon + Adam split, the same curriculum). Its first run destabilized
under the shared recipe: the pre-clip gradient norm crossed the clip (1.0) at step 17,600 inside the fineweb phase and stayed
above it through the curriculum stages; the held-out loss rose from 1.08 at step 16,000 to 1.57 at step 22,000. The literature's
reading of that signature is softmax logit growth (Dehghani et al. 2023, arXiv 2302.05442; Wortsman et al. 2023,
arXiv 2309.14322; Rybakov et al. 2024, arXiv 2410.16682), and the first run computed its attention in bf16.

0.10.9 lets the twin be continued from its last clean checkpoint under the candidate guards, as a new craft each time, with the
original run's record untouched.

## The switches

| field | values | effect |
|---|---|---|
| `AlephLMConfig.qk_norm` | `""` (default) or `"rms"` | per-head RMS normalization of q and k over the head dimension, with learned per-head per-channel gains (init 1); applies to `CausalSDPA` blocks |
| `AlephLMConfig.attn_fp32` | `False` (default) or `True` | under bf16 autocast a `CausalSDPA` block runs with autocast disabled: fp32 projections, logits, softmax, PV product and output (TF32 as the global flag says); a no-op without autocast |
| `AlephLMConfig.attn_kernel` | `"sdpa"` (default), `"flex"` or `"fp16"` | the training forward of a `CausalSDPA` block on a card runs torch's fused flex attention (compiled once per process; measured at no gain on an H100) or fp16 flash with a scaled backward (the sections below); eval, census and decode stay on sdpa; a flex kernel that will not build falls back to sdpa with one printed line |
| `TrainConfig.precision` | `"bf16"` (default) or `"fp32"` | `bf16` = fp32 masters under bf16 autocast with TF32 on (every mission so far); `fp32` = autocast off and TF32 off everywhere |
| `Preset.init_from` | `None` (default) or a spec | a weights-only start from another run's shipped checkpoint (below) |

With every switch at its default the model and the trainer are bit-for-bit the previous release: the guards are additional
code paths, not rewrites. `CausalSDPA` with `qk_norm=""`, `attn_fp32=False` and `attn_kernel="sdpa"` runs the old forward
exactly.

## The registered arms

`geolip.alephllm.presets.CONTROL_RESUME_ARMS` registers five presets, each the twin's craft and recipe verbatim (hub layers
empty, the born-null head unfrozen, Muon 2e-2 / Adam 3e-4, clip 1.0, micro-batch 16 x accumulation 4 = 262,144 tokens a
step) under a new name and a chronological phase list (warmup, fineweb_main, S0-S8, anneal_nochat, anneal_mix), starting from
`mini-beatrix-2s-control/checkpoints/step_00016000.safetensors`:

| preset | precision | attn_fp32 | qk_norm | micro-batch × accumulation | role |
|---|---|---|---|---|---|
| `mini-beatrix-2s-control-bf16` | bf16 | off | off | 16 × 4 | the control: the restart alone must be able to fail as the first run did |
| `mini-beatrix-2s-control-fp32` | fp32 | (everything fp32) | off | 4 × 16 | precision as the single variable |
| `mini-beatrix-2s-control-attn` | bf16 | on | off | 8 × 8 | fp32 attention alone: no insertion cost, the first run's precision elsewhere |
| `mini-beatrix-2s-control-fix` | bf16 | on | rms | 8 × 8 | the guards under the first run's precision elsewhere |
| `mini-beatrix-2s-control-fp32-fix` | fp32 | (everything fp32) | rms | 4 × 16 | both |
| `mini-beatrix-2s-control-attn-fp16` | bf16 | on, fp16 flash | off | 8 × 8 | 10-bit attention inputs and P with the fp32 softmax, at flash pace (0.10.12) |
| `mini-beatrix-2s-control-fix-fp16` | bf16 | on, fp16 flash | rms | 8 × 8 | the same with QK-norm |
| `mini-beatrix-2s-control-bf16-qk` | bf16 | off (bf16 flash) | rms | 16 × 4 | QK-norm on the first run's own attention, the field's guard at the control's pace (0.10.13) |
| `mini-beatrix-2s-control-fix-fp16-c10k` | bf16 | on, fp16 flash | rms | 8 × 8 | fp16 flash + QK-norm CONTINUED from the `-attn-fp16-r10k` arm's own checkpoint at `start_step` (the stop rule's fallback, 0.10.13) |

Every arm trains the recipe's 262,144-token step; the factory refuses a micro-batch that breaks it. The fp32 forms hold more
activation memory at the same tokens (the fp32 attention tensors, or everything under full fp32), so they run smaller
micro-batches: the 95 GB card ran out of memory at 16 × 4 with fp32 attention (2026-10-09). The micro-batch changes only the
accumulation order, not the step.

`make_control_resume_preset(name, start_step, precision, qk_norm, attn_fp32, source, seed_offset, micro_batch, grad_accum,
attn_kernel)` builds any other combination; `source` is a registered craft (its hub repo and prefix) or, from 0.10.13, ANY prefix
on the twin's training repo, so a running arm continues from its own last clean checkpoint under a new guard (the QK-norm gains
installed as the boundary write; a fresh seed offset so the restarted phase reads a new shuffle) (for instance a start from `step_00014000`, the other intact checkpoint).

Measured on the twin at step 16,000, one real fineweb batch, the same weights: bf16 costs +0.0018 in loss over fp32 and fp32
attention under bf16 recovers about 60% of that; the QK-norm insertion costs +0.91 (the 95 GB card) and +0.95 (a 4090) even
with the exact scalar install, because the trained model relies on its sink positions and the norm removes their scale. The
QK-norm arms therefore start from a perturbed function whose recovery confounds their early gradient norms; the arms without
QK-norm start from the checkpoint's exact function.

## The weights-only start

A craft whose preset carries `init_from` and which has no record on the hub yet (no manifest, no `resume/latest.pt` under its
prefix) starts this way in `Trainer.__init__` (`_init_from_checkpoint`):

1. the source checkpoint is fetched (`HubSync.fetch`; a `{"file": ...}` spec reads a local file) and loaded into the fp32
   masters with `strict=False`; the only keys allowed to be missing are the guards this craft adds (the QK-norm gains);
   anything else refuses;
2. the optimizer states are fresh — the source run stored none between its phase boundaries (Muon's momentum and Adam's
   moments start at zero; the 200-step warmup is long past, so the learning rate is at its flat value from the first step);
3. `self.step = spec["step"]` and the manifest's cursor is set from it (`RunManifest.set_cursor`): a phase takes
   `ceil(planned_tokens / tokens_per_step)` steps, earlier phases are done at their rounded-up counts, the phase holding the
   step is active with its tokens so far, the rest planned. At 262,144 tokens a step this reproduces every boundary of the
   first run (1,145 / 20,219 / 22,890 / 25,561 / 29,376 / 33,954 / 38,532 / 42,347 / 45,399 / 50,740 / 53,792 / 57,607 /
   61,422); at step 16,000 fineweb_main is active with 3,894,149,120 of 5,000,000,000 tokens counted;
4. the restarted phase's stream opens at `seed + seed_offset` (7919 by default): a fresh shuffle instead of replaying the
   shuffle head the source run consumed — a recorded data-order discontinuity (the manifest notes it; later phases keep the
   recipe's seed);
5. with `qk_norm` on, the gains are installed from one micro-batch of the active phase's training corpus drawn from a throwaway
   stream (seed + 4242: never the training stream's rows, never the validation gauge), block by block, each block run with its
   gains in place so the next block reads the guarded stream;
6. `manifest.init_from` records the source, the step, the phase, the seed offset, the upcast, the fresh optimizers, the guard
   keys added and the gain install (dataset, seed, rows, per-block gains and logit-scale ratios); a manifest note says the same.

Every later session resumes from the craft's own `resume/latest.pt` as before; `init_from` rides inside the manifest snapshot
so the restarted phase keeps its seed on resume.

## The gain install (a boundary write)

Inserting a normalization into a trained network changes the attention logits' scale at once; the gains make the insertion
exact for the ordinary logit. For a head with raw queries `q_i` (position `i`) and per-position scale
`rho_i = sqrt(mean_c q_ic^2)`, the installed gain is ONE scalar per head, broadcast over the channels:

    gain = median_i(rho_i)

so that `q_normed_i * gain = q_i * median(rho) / rho_i`: the raw query with its per-position scale replaced by the typical
one. Every logit between two ordinary positions is then preserved exactly (its scale factor is 1 at the medians), and the norm
changes only the outlier positions: a trained attention carries a few positions (the sequence start above all) whose scale is
10-50x the rest. Two forms that look natural are wrong, both measured on the twin at step 16,000: a MEAN position scale lifts
every ordinary logit by the outliers' share (1.3-3.3x per block), and a per-channel gain pattern (the RMS of the unit
directions per channel) squares the heads' shared-channel anisotropy inside the dot product (the typical logit 1.3-4.5x, the
loss on one fineweb batch 0.77 → 2.11). The gains remain per-head per-channel parameters for training. The same for k. What
no gain can preserve is the sink mechanism itself (a sink key's logit drops to the typical scale), so the insertion still
costs some loss that training must recover; the notebook's preflight prints that cost on a real batch (guard off vs gains
installed), and the arms without QK-norm start from the checkpoint's exact function. `CausalSDPA.install_gains` returns the
gains' means, the mean |logit| (with the 1/sqrt(head_dim) scale) before and after on a 64-position block, and the median over
the block's entries of |logit after| / |logit before| (the typical entry's ratio, 1 by construction away from the outliers);
`AlephLM.install_qk_gains` walks the trunk.

## The fused kernel for fp32 attention (0.10.11)

The fp32 attention of 0.10.9 ran through `scaled_dot_product_attention` with fp32 inputs, which has no fused tensor-core
kernel: flash is bf16/fp16 only, and the memory-efficient kernel computes fp32 on the plain cores. On an H100 that cost 3x
(2026-10-09): 7.2 s a step for the `-attn` arm against 2.3 s for the first run's bf16 form, so the attention alone was about
two thirds of the step.

`attn_kernel="flex"` routes the training forward of a `CausalSDPA` block through `torch.nn.attention.flex_attention`,
compiled once per process (the only compiled region; the model itself stays eager), with a cached causal block mask. The
products q·k and p·v run on TF32 tensor cores when TF32 is on (the bf16-precision arms: 10 mantissa bits in, fp32
accumulation) and in ieee fp32 when it is off (the fp32-precision arms); the softmax, its normalization and the output
accumulation are fp32 either way; no n × n matrix is materialized. The kernel is compiled per fp32-matmul precision, so the
two settings never share one. The first call prints the kernel's deviation from sdpa on that batch
(`[attn] flex kernel (... products) vs sdpa on the first batch: max rel deviation ...`). When the kernel cannot be built on a
stack (no Triton, a compile error) the module prints one line and every block falls back to fp32 sdpa for the rest of the
process: the run continues at the old pace with the old numbers. Eval, the census and decode never use it, so held-out
losses are computed by the same kernel in every arm. A run resumed under a different kernel than its record continues; the
manifest gets the updated model config and a note naming the step (the steps before that session ran the former).

Measured on a 4090 (torch 2.11; one row of ctx 4096, 16 heads, head 64; forward + backward through `CausalSDPA` with
`attn_fp32=True`, the same weights, against fp32 sdpa):

| mode | forward, max rel deviation | gradient, input / qkv weight | pace per row (forward + backward) |
|---|---|---|---|
| ieee products (TF32 off) | 2.4e-7 | 1.9e-6 / 2.7e-6 | — |
| TF32 products (TF32 on) | 4.8e-4 | 1.3e-3 / 1.5e-3 | 8.3 ms against 12.7 ms for fp32 sdpa (1.5x) |

The 4090 is the floor for the pace ratio: its TF32 tensor rate equals its plain fp32 rate (82 TFLOPS both), so the 1.5x there
is the fusion alone; an H100's TF32 tensor rate is about seven times its plain fp32 rate (495 against 67 TFLOPS), and the
products are where the fp32 kernel spends its time. The notebook's C2 bench is the number to book by.

Under the bf16-precision arms the attention products are therefore TF32, not full fp32: between bf16's 7 mantissa bits and
fp32's 23, with the softmax in fp32. The `-fp32` and `-fp32-fix` arms keep ieee products throughout (TF32 off) and stay on the
default sdpa kernel.

**Measured on an H100 (2026-10-09, the `-attn` arm at 8 × 8): 7.28 s a step with the fused kernel against 7.2 s with fp32
sdpa — no gain.** On the 4090 at 8 rows the fused kernel's default tiles gave 1.29x over fp32 sdpa (72.7 against 94.0 ms for
one block's forward + backward) and torch's autotune of the tiles (`mode="max-autotune-no-cudagraphs"`) 1.0x (95.6 ms): the
fp32 flex backward is tile-bound at every setting tried. The fused kernel stays available through `attn_kernel="flex"` but
no registered arm carries it from 0.10.12 on; `-attn` and `-fix` are back on fp32 sdpa, the full-precision form at its price.

## fp16 flash inside the fp32 block (0.10.12)

`attn_kernel="fp16"` keeps the block's projections in fp32 (TF32 per the global flag) and runs the attention itself through
fp16 flash: q, k and v are cast to fp16 (10 mantissa bits, against bf16's 7 and fp32's 23), the fused kernel forms the
logits and the softmax in fp32 and takes the P·V product with P in fp16, and the output returns in fp32. The precision class
is the fused kernel's TF32 class (10 bits in, fp32 softmax) at flash pace: the kernel the first run used, so the step costs
about the control's plus the fp32 projections and one extra fp16 forward per block.

The backward is where fp16 differs from TF32: fp16 keeps full precision only above 6e-5 and nothing below 6e-8, and a
mean-reduced loss sends per-element gradients around 1e-5 and lower. The block therefore runs flash's fp16 backward on a
SCALED upstream gradient: a power of two chosen per call from the gradient's own maximum (the largest element lands near
2^10), undone in fp32 afterwards. A sink key attended by every query gathers the gradient of every query into one dv and can
overflow fp16 even so; the result is checked and recomputed at a smaller scale when it does (eight halvings by 16 at most).
The recomputation of the forward inside the backward costs one fp16 flash forward per block. Eval, the census and decode stay
on fp32 sdpa, as for the fused kernel. The trainer prints the kernel it runs (`[attn] ...`) at the start of every session.

From 0.10.13 the kernel keeps three counters for the process (`fp16_kernel_stats()`): calls, backward retries (the scale
lowered after an overflow) and fp32 fallbacks. A forward overflow (an input past fp16's 65,504, which the first run's
checkpoints sit 100x under, or a non-finite input) no longer reaches the trainer's non-finite guard: that call runs in fp32
sdpa, forward and backward, and is counted. The trainer prints the three counters at the end of every session and writes them
into the manifest's notes.

## The tests

`python -m geolip.alephllm.tests.test_control_fixes` (CPU, seconds, no download): the guards-off identity; QK-norm shapes,
causality, unit RMS, decode parity; the gain install's formula and its logit-scale bookkeeping; the precision contexts and the
config refusals; the cursor rule against every boundary of the first run's schedule; the four arms and the twin's isolation;
the weights-only start on a tiny craft end to end (cursor, weights, gains, seed, fresh optimizers, two steps, the final
checkpoint, the hand-off to a normal resume); the kernel switch's plumbing (the default, the refusal, every block, the dict
round trip, the arms, the dead-kernel fallback); the fp16 form's plumbing and the trainer's kernel label; and, on a card,
the fp32-attention path under autocast, the fused kernel's forward and gradient parity, the fp16 form's parity (plain, under
a 1e-6 loss scale, and with a sink key), its counters (an overflowing sink backward fires a retry; a forward overflow falls back
to fp32 sdpa for that call) with the eval path on sdpa bit for bit; the continuation source and the bf16 + QK-norm arm, with
QK-norm under bf16 autocast on a card. The full smoke array (`tests.smoke`) passes
unchanged.

## The notebook

`notebooks/beatrix_2s_control_resume_colab.ipynb`: the install pinned at the release tag, a preflight (the switches, the
step-1 read of the same weights under the three precisions on a real fineweb batch, the gain install rehearsed, a bench under
the arm's precision and kernel with the peak-memory gate), the boundary-stopping session (the first session performs the weights-only
start; every boundary ships a checkpoint and a report under `reports/resume/`), and a growth table beside the first run's
reports.
