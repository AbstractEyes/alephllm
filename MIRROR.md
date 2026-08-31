# The automodel mirror law (unguarded invariant — read before touching model/)

huggingface.co/**AbstractPhil/mini-beatrix-1** AND
**AbstractPhil/mini-beatrix-2s** (added 2026-08-31, mission final
weights) are HF remote-code packages
(`MiniBeatrixConfig` / `MiniBeatrixForCausalLM` in `modeling_minibeatrix.py`)
that carries **vendored verbatim copies of `geolip/alephllm/model/*.py`** plus
`presets.py`. Nothing in this repo enforces the link.

**LAW (claude-mind, repos/alephllm.md): any change to `model/*.py` or
`presets.py` must be mirrored to BOTH HF repos in the same session** — the
vendored copies drift silently otherwise. The machine HF_TOKEN can write it.

Mirror procedure:
1. Copy the changed files over the vendored copies (same relative imports).
2. Verify locally BEFORE push: load the shipped weights through the vendored
   code path and compare logits vs `geolip` (`AlephLM`) — parity must be 0.
3. Push; note the mirror in the session digest.

Compatibility contract the mirror relies on (also the arms + checkpoint
contract): `n_const == 1` constructs the exact v1 module layout — state-dict
keys `addr.*`, `q.weight`, `k.weight` unchanged; `Block.prefill/step` arity
unchanged; `model.blocks` / `cfg.d_model` / `cfg.name` names unchanged.
The 2026-08-26 v2 surgery (multi-constellation + governor + supply warning)
was designed to keep all of it: v1 checkpoints, the 29 shipped arms, and the
live Space load bit-identically.
