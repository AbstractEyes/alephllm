# The lookup arm (0.10.15)

A detachable softmax attention arm added to a trained all-splat trunk, so the trunk gains content-addressed lookup
without a retrain. Built for mini-beatrix-3 after the byte battery showed what the splat memory lacks: a planted
value is not fetched at its cue behind text (the first byte at .00-.06), while softmax attention fetches it to the end
of the context.

## What it is

`AlephLMConfig.lookup_arm_sites` names blocks. After each named block the stream receives one more causal attention
module (the same `CausalSDPA` the softmax crafts use, with this config's `qk_norm`, `attn_fp32` and `attn_kernel`
switches) behind its own LayerNorm:

    x = block_i(x)
    x = x + arm_i(norm_i(x))        # only for i in lookup_arm_sites

The arm is born null by weight: its output projection starts at zero, so a model with arms gives the trunk's exact
logits at step 0 and loads a no-arm checkpoint with exactly the arm keys missing. Its parameters are the state dict's
`lookup_arms.*` and `lookup_norms.*`, so the arm can be saved, shipped and removed on its own.

## Training it alone

`TrainConfig.arm_params` names parameter prefixes. When set, every other parameter is frozen (`freeze_except`) and the
named parameters are owned by pure Adam whatever their shape (Muon keeps the trunk; the trunk receives no gradient and
stays bit-identical). The weights-only start (`Preset.init_from`) accepts the arm keys as missing, like the QK-norm
guards.

`make_v3_lookup_arm_preset(sites, steps, start_step=245674)` builds the mini-beatrix-3 craft with arms at `sites`,
started from the released checkpoint, trained on the far-recall mix (`fineweb-recall-far-5`) for `steps` steps of the
recipe's 262,144 tokens, 8 x 8 micro-batches, QK-norm from birth and fp16 flash inside the fp32 attention block. The
curriculum is two phases, the trunk's own steps (done at the cursor) and the arm's phase, so the start lands on the
arm's first step. The craft's name encodes the sites (`mini-beatrix-3-la2-6-12-20`).

Decode with arms is not implemented yet (`prefill` refuses); training and scoring use the forward path.

## The byte screen's arms

`make_byte_screen_preset(side, kind, start_step)` builds the 2,000-step continuation arms of the byte screen: `side`
"twin" (the fp16 softmax twin from its own checkpoints) or "2s" (the splat 2s); `kind` "ctrl", "rows" (the far-recall
rows at 5% in place of fineweb text), "rows-copy" (the rows plus the copy-weighted loss) or "depth" (the
depth-weighted loss, twin side only). The rows kinds must start inside the fineweb phase (start_step <= 18,219).

## Continuing a trained arm

A second run of `steps` more continues from the arm craft's own checkpoint under a new name, a weights-only start whose source
already carries the arm (no missing keys; fresh optimizer moments): `make_v3_lookup_arm_preset(sites, steps, start_step=<the arm's
step>, source="mini-beatrix-3-la2-6-12-20", name="mini-beatrix-3-la2-6-12-20-c<step // 1000>k")`. The notebook's `CONTINUE_FROM =
(step, craft)` does exactly this; its preflight then expects no missing keys and reports the arm's effect on a batch instead of the
birth identity.
