"""bytelex — universal, model-free relational system over byte information.

NOT a model instrument: no weights, no Beatrix. The statistics live in
the byte corpus itself, so every ByteLM derivative — present or future,
whatever its internal structure becomes — consumes the same matrix.
Alphabet-parametric (256 today; can grow or shrink), gram-modular
(char n-grams, word-grams, hashed large-n; add views as required).

Core objects:
  GramSchema   — declares the views (alphabet size, char ns, hashed ns,
                 separator predicate for word-grams). Serializable.
  ByteLexicon  — corpus statistics per view: gram counts, successor
                 branching entropy (the model-free boundary functional),
                 adjacent-gram PMI (cohesion). Persistable.
  profile()    — token bytes -> relational profile: cohesion, internal
                 entropy maxima (where byte-language says the token
                 SPLITS), edge sharpness, frequency standing.
  The full lexicon translation matrix = profile() over every token of
  every tokenizer, persisted per vocabulary.

Distillation loss primitives (consumed by any ByteLM):
  alignment endpoints = corpus-entropy boundaries (not any model's);
  per-token weight = cohesion; non-cohesive tokens split at their
  internal maxima before matching.
"""
from .core import GramSchema, ByteLexicon

__all__ = ["GramSchema", "ByteLexicon"]
