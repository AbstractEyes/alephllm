# The byte levers (0.10.14): the loss forms and the far-recall data piece

Two training levers for exact, distant byte prediction, built as minimal deltas in the trainer's grammar. Neither changes a
model's architecture; both are off by default (the cross-entropy and the plain fineweb phase verbatim).

## The loss forms (`TrainConfig.loss_form`; set on the model by the trainer; eval always the plain CE)

| form | definition | the control inside it |
|---|---|---|
| `ce` | the cross-entropy, verbatim | — |
| `depth` | the target byte's eight conditional bit NLLs along its binary path, most significant bit first, weighted by `loss_depth_w` and summed | all eight weights at 1 give the CE exactly (the eight terms telescope to the byte's NLL) |
| `copy` | the CE per byte, weighted `1 + loss_copy_beta` on every byte that is COPY-RIGHT: the `loss_copy_m`-gram of inputs ending at its position occurred at least `loss_copy_D` positions earlier in the same row followed by the same byte (exact: a hash of the m-gram, then equality of hash and follower; one L x L comparison per row, computed on the card from the batch alone) | beta 0 gives the CE exactly |
| `depth+copy` | both | — |

The weighting is normalized by the sum of the weights, so a form changes the balance of the gradient, not its scale.

## The far-recall data piece (`recall-far-fineweb`; the mix `fineweb-recall-far-5`)

A fineweb-edu row rendered with a planted record near its start and the record's question at a gap sampled log-uniformly
between 64 and 2,400 bytes later: high-entropy values (random digits, letters or hex of 5-8 symbols), five cue nouns, three
forms (a single record; four locker lines with one asked; a record updated once in between, the later value asked).
Deterministic per row (the seed is the row's own bytes). The held-out head of fineweb-edu stays reserved. The mix keeps
fineweb at 95%: the natural-language ballast rule holds trivially. The gap is capped at 2,400 because a packed stream cuts
documents into context+1 blocks: a longer span rarely lands in one window (a block-aligned form is the next engineering step
if the lever earns it).

Why a new piece beside the registered `recall-synth` rows (3-5% of every stage since the 2s mission): those rows are short
closed-vocabulary episodes (a name carries one of eight things, asked within a few lines); they teach keyed lookup over a
fixed set at a few dozen bytes, not exact copying of unpredictable content at distance.

## The screen (plans/2026-10-10_byte_accuracy_and_distant_behavior.md section 5)

`make_control_resume_preset(name, ..., extra_train={"loss_form": "copy", "loss_copy_beta": 2.0},
phase_dataset_override={"fineweb_main": "fineweb-recall-far-5"})` builds an arm; the byte battery
(history tools `byte_battery.py`) reads the arm's checkpoints at bit depth.

Tests: `python -m geolip.alephllm.tests.test_byte_levers`.
