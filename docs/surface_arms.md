# Surface arms: reading a tokenizer's spelling as the plain bytes

A byte-level model reads text as its UTF-8 bytes. A tokenizer-based model hands text around in its own **spelling**: the
vocabulary strings of its tokens in order. For a byte-level BPE tokenizer with GPT-2's byte map (Qwen 2.5 and 3, Llama 3,
GPT-2 and kin) every byte is stored as a printable stand-in character, so a space becomes the two bytes of 'Ġ' and ' taco'
becomes the one token 'Ġtaco'. The byte model reads that spelling as a different text from the plain bytes, and the
difference grows with depth: before any training, the alignment between her reading of Qwen's spelling and her reading of the
plain bytes falls from .73 at block 8 to .32 at block 31 (whitened Procrustes alignment on held-out captions).

A **surface arm** is a small detachable adapter (13.7M parameters, one module after each of the 32 blocks, gate born closed,
head born at zero) trained so that the model reads one spelling convention the way it reads the plain bytes. It is tied to a
convention, not to a model: every tokenizer that spells bytes the same way hands the model the same bytes for the same text.

This page is the library's documentation of the mechanism: the two surfaces, the sites, the loss, the quiet terms, the bars,
how to train an arm for a new tokenizer, and how to read through a published one. The runs, their numbers and the arms live on
`AbstractPhil/beatrix-tokenizers`; the code is `geolip/alephllm/train/surface.py` (the surfaces, the rows, the losses, the
reader) and `geolip/alephllm/arm_mount.py` (the registry and the mount).

## The two surfaces and the closing byte

| surface | what it is | example ('a taco truck' under Qwen3) |
|---|---|---|
| A | the text's own UTF-8 bytes | `a taco truck` (12 bytes) |
| B | the tokenizer's spelling: its tokens' vocabulary strings in order, as UTF-8 | `aĠtacoĠtruck` (14 bytes: each space is two) |

Per token the two surfaces are tied at the token's **closing byte**: the byte after the token's expansion on A, the byte after
its spelling on B. The state at a closing byte has read the whole token on either surface, so the two states describe the same
prefix of the text. A text's first and last tokens give no site (the first is the attention sink of the model the surface
serves, the last is followed by no byte of the text). A phrase that ends in a full stop has its last site at the token before
the stop, whose closing byte is the stop.

`spell(tok, text, convention)` renders both surfaces and the per-token closings (`Spelled`: `raw`, `spelled`, `a_end`, `b_end`,
`ids`); `sites(spelled)` lists the sites.

### The conventions

| convention | spelling | tokenizers | exactness |
|---|---|---|---|
| `gpt2` | each byte through GPT-2's byte-to-character map; a space is 'Ġ', a newline 'Ċ' | Qwen 2.5 / 3, Llama 3, GPT-2, DeepSeek, o200k kin | exact: the pieces' characters through the map give the text's bytes back, or the text is refused |
| `sentencepiece` | the pieces in order; '▁' (three bytes) before each word; `<0xNN>` byte fallbacks | T5, Gemma, Llama 2, Mistral | lossy (normalization); a lone '▁' piece stands for the space before the word it precedes |
| `wordpiece` | the pieces in order joined by single spaces; '##' continuations; lower-cased | BERT kin | lossy (case, re-spacing) |
| `clip` | the pieces in order; the byte map with '</w>' after each word; lower-cased | CLIP ViT-L/14 kin | lossy (case, spacing) |

Under the exact convention a text whose tokens do not reproduce its bytes (a tokenizer that normalized it) raises `ValueError`
and the callers skip and count it. Under the lossy conventions each token's closing byte on A comes from the tokenizer's offset
mapping: the byte after the token's span in the text as written (`a_text="written"`, the default; `a_text="normalized"` reads
A as the tokenizer's own normalization of the text instead). A text whose tokens do not tie to it (an empty span, a span out
of order, a text not covered to its end) raises and is counted. On the 1,024 held-out captions of the Qwen arm's read, every
convention spells every caption: Qwen3 at 9.5 sites a caption with the spelling 1.18 times the bytes, T5 (`google/t5-v1_1-xxl`,
the same tokenizer as t5-base) at 11.1 sites and 1.41 times, bert-base-uncased at 9.6 and 1.04, CLIP ViT-L/14 at 9.4 and 1.68.

### The every-byte site set

`sites_every_byte(spelled)` gives one site per byte of the text instead of one per token (the gpt2 convention only, where B is
the byte-wise map of A): the segmentation-free variant of the arm. The token sites are the subset at the tokens' last bytes.
`PairedRows`, `single_rows` and `read_spelled` take the site set as `site_fn`.

## The rows

A document is the model's document byte followed by the text. Documents are packed into rows of context + 1 bytes, the same
documents in the same order on both surfaces: the B row fills first (B is never shorter than A) and the A row's tail carries
further documents as plain context. A site is (row, position on A, position on B) for every closing byte that lands inside
both rows; the training step reads two such rows per surface. The read's form (`single_rows`) is one document a row,
right-padded, the sites at the same closings.

## The loss

Per served block (all 32 by default) and per site, the armed model reading B is pulled toward a reference model reading A:
the reference is the same model with the arm masked and every frozen partner as it is. Both states are the block's output
LayerNorm'd without affine (the signal the reads use) and standardized per dimension by the reference model's own statistics on
A, measured once at the start over the reference rows:

    z = (LN(h) - mu_b) / sigma_b        per block b and dimension, sigma floored at 1e-4 of its mean

    mse      the mean over sites and dimensions of (z_B - z_A)^2, averaged over the served blocks
    mse_nce  the same plus 0.5 x a symmetric state-table InfoNCE at temperature 0.1 (each B site identifies its own A site
             among the batch's A sites by cosine; the other sites are the negatives)

The reference statistics ship beside each arm (`stats.pt`) for the record; nothing at read time needs them.

## The quiet terms

The arm must not change what the model reads elsewhere. Two quiet terms, weight lambda 2 each:

| term | rows | what it keeps |
|---|---|---|
| quiet on A | the same A rows the loss used | KL(reference || armed) on the next-byte distributions: the plain surface read as before |
| quiet off-domain | one web-text row and one partner's row (the frozen stage arms' own text) | KL(this member masked || armed): the partners' subjects read as before |

The optimizer is pure Adam (lr 1e-3, no weight decay) on the arm alone; the trunk and the partners are frozen; the arm chain
runs compiled with recompute-in-backward.

## The bars (fixed before any run)

| bar | reads | threshold |
|---|---|---|
| B1 the gap closes | armed B against masked A, whitened Procrustes alignment on 512 held-out captions of each of two draws, per block | >= .85 at every served block >= 12 |
| B2 silence on A | armed A against masked A, the same gauge; beside it the cost in bits per byte on web text and on each partner's text | >= .995 at every block; +<= .012 bits per byte |
| controls | a shuffled pairing (the arm trained against the wrong targets) must fail B1; an untrained-copy arm and a bare-trunk solo read beside; the second draw repeats the first; two seeds | |

The alignment is whitened (K = 128 directions, two folds by caption, the map fitted on one fold and scored on the other, the
mean of the two), so that 1 means the same geometry up to a rotation and 0 means unrelated.

## Training an arm for a new tokenizer

The driver that trained the published arms (`v3_qwen_arm_refit.py` in the research record) takes the convention and the
tokenizer as arguments; the library pieces it uses are the ones above:

1. Render documents with `spell(tok, text, convention, a_text)`, skipping and counting the texts that raise.
2. Feed them to `PairedRows(docs, ctx, rows=2, site_fn=sites)`; each batch gives the two surfaces' rows and the sites.
3. Mount the arm over its group with `StageArmProgram` (the group frozen, the member trainable), take the block outputs by
   `block_states(model, rows, blocks, where=sites)` on both surfaces (the reference with the member masked), standardize by
   `reference_stats` / `standardize`, and apply `surface_loss`.
4. Add the quiet terms, step pure Adam, and read `paired_alignment` on held-out captions at every close.
5. Publish the final anchor, the statistics, the results and the readout under `surface/<convention>/<run>/` and add a row to
   `arm_mount.SURFACE_ARMS` (convention, tokenizer, member, the group trained over, the file, the seed, the site set).

## Reading through a published arm

```python
from geolip.alephllm import load_trunk
from geolip.alephllm.arm_mount import mount_surface, masked, reader_kwargs
from geolip.alephllm.train.surface import read_spelled
from transformers import AutoTokenizer

model = load_trunk(245674, device="cuda")
prog = mount_surface(model, "qwen3")                 # the group first, then the member, by the training route
tok = AutoTokenizer.from_pretrained(prog.surface["tokenizer"])
kw = reader_kwargs(prog.surface)                     # the arm's convention, A-text rule and site set
r = read_spelled(model, tok, texts, blocks=[12, 16, 20, 24], **kw)
r["states"][20]                                      # (n_sites, d) her states on the spelling at every closing byte, LayerNorm'd
r["sites"]                                           # (text index, token position, token id) per row
with masked(prog, [prog.surface["member"]]):         # the plain reading at the same sites, the arm off
    plain = read_spelled(model, tok, texts, blocks=[20], surface="A", **kw)
```

The rows of `states` line up token by token with the tokenizer's ids, so a reading of the same text by the tokenizer's own
model can be compared site by site.

## Results

The Qwen arm (convention `gpt2`, `Qwen/Qwen3-0.6B`), two seeds over the frozen eight, 4,000 steps each, read on 512 held-out
captions of each of two draws; the readouts and results files on `AbstractPhil/beatrix-tokenizers` carry every block.

| run | the gap (armed spelling vs plain bytes), lowest served block / the plateau | silence on plain bytes | bits per byte |
|---|---|---|---|
| seed A (mse_gXA_o0) | .895 at block 31 / .962-.971 at blocks 16-27 (before training .32 / .45-.53) | .995-.998 at most blocks, .984 at 31 | +.0003 web, +.0005 worst partner |
| seed B (mse_gXB_o1) | .909 at block 31 / .953-.964 at blocks 8-28 | .994-.997 at most blocks, .981 at 31 | +.0003, +.0003 |
| MSE + InfoNCE form (mse_nce_gXA_o0) | .884 at block 31; behind the MSE form at every block | .977 at 31 | +.0002 |
| the shuffled-pairing control (mse_gXA_o0_shuf) | .201 at block 31; BELOW the untrained reading at every block (.484 at block 12 against .640 before) | .969 at 31 | +.0003 |

The gap bar (.85 at every served block) is met on both seeds and both draws; the bits bar (+.012) by a wide margin; the strict
silence bar (.995 at every block) is missed at the last block on both seeds (.98) and, on seed B, by one to two thousandths at a
few middle blocks. The control trained against the wrong pairings makes the model read the spelling worse than with no arm,
so the closure is not an artifact of the alignment map. An independent read on the mounted seed-A arm found that the spelling
read through the arm reaches the tokenizer's own model's final states as far as the plain bytes do (.546 against .562 at block
24 before any arm) and carries the same mood axis at blocks 12-20. The MSE form is the one carried forward; the InfoNCE term
bought nothing on this bed. The family (T5, CLIP and BERT spellings; an every-byte Qwen arm) trains on the same recipe.
