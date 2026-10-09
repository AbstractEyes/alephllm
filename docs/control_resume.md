# The control twin, continued: the softmax guards and the weights-only start (0.10.9)

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
| `TrainConfig.precision` | `"bf16"` (default) or `"fp32"` | `bf16` = fp32 masters under bf16 autocast with TF32 on (every mission so far); `fp32` = autocast off and TF32 off everywhere |
| `Preset.init_from` | `None` (default) or a spec | a weights-only start from another run's shipped checkpoint (below) |

With every switch at its default the model and the trainer are bit-for-bit the previous release: the guards are additional
code paths, not rewrites. `CausalSDPA` with `qk_norm=""` and `attn_fp32=False` runs the old forward exactly.

## The registered arms

`geolip.alephllm.presets.CONTROL_RESUME_ARMS` registers five presets, each the twin's craft and recipe verbatim (hub layers
empty, the born-null head unfrozen, Muon 2e-2 / Adam 3e-4, clip 1.0, micro-batch 16 x accumulation 4 = 262,144 tokens a
step) under a new name and a chronological phase list (warmup, fineweb_main, S0-S8, anneal_nochat, anneal_mix), starting from
`mini-beatrix-2s-control/checkpoints/step_00016000.safetensors`:

| preset | precision | attn_fp32 | qk_norm | role |
|---|---|---|---|---|
| `mini-beatrix-2s-control-bf16` | bf16 | off | off | the control: the restart alone must be able to fail as the first run did |
| `mini-beatrix-2s-control-fp32` | fp32 | (everything fp32) | off | precision as the single variable |
| `mini-beatrix-2s-control-attn` | bf16 | on | off | fp32 attention alone: no insertion cost, the first run's precision elsewhere |
| `mini-beatrix-2s-control-fix` | bf16 | on | rms | the guards under the first run's precision elsewhere |
| `mini-beatrix-2s-control-fp32-fix` | fp32 | (everything fp32) | rms | both |

`make_control_resume_preset(name, start_step, precision, qk_norm, attn_fp32, source, seed_offset)` builds any other
combination (for instance a start from `step_00014000`, the other intact checkpoint).

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

## The tests

`python -m geolip.alephllm.tests.test_control_fixes` (CPU, seconds, no download): the guards-off identity; QK-norm shapes,
causality, unit RMS, decode parity; the gain install's formula and its logit-scale bookkeeping; the precision contexts and the
config refusals; the cursor rule against every boundary of the first run's schedule; the four arms and the twin's isolation;
the weights-only start on a tiny craft end to end (cursor, weights, gains, seed, fresh optimizers, two steps, the final
checkpoint, the hand-off to a normal resume); and, on a card, the fp32-attention path under autocast. The full smoke array
(`tests.smoke`) passes unchanged.

## The notebook

`notebooks/beatrix_2s_control_resume_colab.ipynb`: the install pinned at the release tag, a preflight (the switches, the
step-1 read of the same weights under the three precisions on a real fineweb batch, the gain install rehearsed, a bench under
the arm's precision with the peak-memory gate), the boundary-stopping session (the first session performs the weights-only
start; every boundary ships a checkpoint and a report under `reports/resume/`), and a growth table beside the first run's
reports.
