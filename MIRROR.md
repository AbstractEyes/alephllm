# The automodel mirror law (unguarded invariant — read before touching model/)

huggingface.co/**AbstractPhil/mini-beatrix-1**,
**AbstractPhil/mini-beatrix-2s** (added 2026-08-31, mission final
weights), **AbstractPhil/mini-beatrix-2.5s** (added 2026-09-22, the 2s core
with its arms) AND **AbstractPhil/mini-beatrix-3** (added 2026-10-05, mission
final weights, with its stage arms since the same day) are HF remote-code packages
(`MiniBeatrixConfig` / `MiniBeatrixForCausalLM` in `modeling_minibeatrix.py`)
that carry **vendored copies of `geolip/alephllm/model/*.py`** plus
`presets.py`. Nothing in this repo enforces the link.

**LAW (claude-mind, repos/alephllm.md): any change to `model/*.py` or
`presets.py` must be mirrored to EVERY one of these HF repos in the same
session** — the vendored copies drift silently otherwise. The machine
HF_TOKEN can write it.

Two of the vendored files are NOT verbatim. The transformers remote-code
loader requires every relative import in a package file to exist as a
sibling file, so the hub copies of `alephlm.py` and `presets.py` defer their
imports of modules that are not vendored (`fusion`, the data stack). Each
such site is marked `# VENDORED`. Copying the library's text over those two
files makes a package unloadable while a logits comparison still passes.

Mirror procedure:
1. Copy the changed files over the vendored copies (same relative imports),
   keeping every `# VENDORED` site in `alephlm.py` and `presets.py`.
2. Verify locally BEFORE push: load the shipped weights through the vendored
   code path and compare logits vs `geolip` (`AlephLM`) — parity must be 0.
3. Prove the load BEFORE push: `AutoModelForCausalLM.from_pretrained(<folder>,
   trust_remote_code=True)` in a process where `geolip` cannot be imported,
   with fresh `HF_HOME` and `HF_MODULES_CACHE`. Step 2 passes on a package
   that cannot load.
4. Push; note the mirror in the session digest.

A package builds its model from its own `config.json` fields, never from a
preset name (the library preset named `mini-beatrix-3` is the 24-block shape;
the trained craft is the 32-block one in its run manifest).

Compatibility contract the mirror relies on (also the arms + checkpoint
contract): `n_const == 1` constructs the exact v1 module layout — state-dict
keys `addr.*`, `q.weight`, `k.weight` unchanged; `Block.prefill/step` arity
unchanged; `model.blocks` / `cfg.d_model` / `cfg.name` names unchanged.
The 2026-08-26 v2 surgery (multi-constellation + governor + supply warning)
was designed to keep all of it: v1 checkpoints, the 29 shipped arms, and the
live Space load bit-identically.
The group anchor sets on the training repo (`mini-beatrix-3/arm_refit/group/anchors`)
mount on a trunk through `geolip.alephllm.arm_mount.mount_group` (0.10.6); the public
package's `arms/` files hold the same tensors in the package's own layout and mount
through the package's runtime.
