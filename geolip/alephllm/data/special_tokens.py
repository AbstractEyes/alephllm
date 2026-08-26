"""Special tokens for the byte crafts — the invalid-UTF-8 control plane.

THE LAW THIS RESTS ON (verifiable, not conventional): the stream tokenizer
encodes with str.encode("utf-8", errors="replace") (tokenizers.py), and a
UTF-8 encoder can NEVER emit the byte values that are structurally invalid
in UTF-8 — 0xC0, 0xC1 (overlong-encoding leads) and 0xF5–0xFF (leads past
U+10FFFF). Those THIRTEEN ids are therefore provably unreachable by any
encoded text, forever, with zero vocabulary or architecture change: the
embedding and head rows already exist (vocab 256), they simply never
receive data-driven gradient until a formatter emits them deliberately.
This is STRONGER than BPE-style specials, which are collision-free only by
tokenizer discipline. (PAD_ROW = 256 lives outside the byte range — no
collision with the trigram shift tables.)

Byte-craft bonus: the trigram embedding (emb0/emb1/emb2) feeds each special
directly into the input representation of the NEXT TWO positions, so a
special colors a local window by construction before attention acts.

Layout (single-byte, hot structural set):
    0xFF DOC    document boundary — appended by the packer after every
                document, in EVERY phase, from birth
    0xFE USER   user-turn open
    0xFD MODEL  model-turn open (Beatrix's voice)
    0xFC END    universal block close (turn / sys / think / data)
    0xFB SYS    system-instruction block open
    0xFA THINK  private-reasoning block open        (reserved for arm era)
    0xF9 DATA   structured/tool payload block open  (reserved for arm era)
    0xF8 SEP    in-block field separator
    0xF7 MODE   register tag: MODE + one ASCII byte payload
    0xF6 CUE    recall/binding cue marker (canary + binding instruments)
    0xF5 ESC    escape prefix: ESC + <byte> = 256 extended slots (table
                below; append-only allocation, record every grant)
    0xC0 RES0   hard-reserved single-byte slot
    0xC1 RES1   hard-reserved single-byte slot

USAGE DOCTRINE (the guarantee is data discipline, not the id):
- Structure from birth: DOC rides every phase (the packer owns it).
- Behavior tokens appear ONLY paired with their behavior, introduced at
  phase boundaries so boundary reports bracket them (instrument-first).
- The curriculum stages stay prose (chat-in-anneal law): chat specials
  enter through the anneal formats, never s0–s8 rows.
- amoe_bridge's v1 plain-text tags are a shipped-arm contract — untouched;
  the specials-native chat format below is the v2s-era form.
- BPE crafts never see these: every emitter is gated on vocab_size == 256.
"""
from __future__ import annotations

import numpy as np

DOC = 0xFF
USER = 0xFE
MODEL = 0xFD
END = 0xFC
SYS = 0xFB
THINK = 0xFA
DATA = 0xF9
SEP = 0xF8
MODE = 0xF7
CUE = 0xF6
ESC = 0xF5
RES0 = 0xC0
RES1 = 0xC1

NAMES = {DOC: "DOC", USER: "USER", MODEL: "MODEL", END: "END", SYS: "SYS",
         THINK: "THINK", DATA: "DATA", SEP: "SEP", MODE: "MODE", CUE: "CUE",
         ESC: "ESC", RES0: "RES0", RES1: "RES1"}

# The full invalid-UTF-8 set — every special MUST live here.
INVALID_UTF8 = frozenset({0xC0, 0xC1} | set(range(0xF5, 0x100)))
SPECIALS = frozenset(NAMES)
assert SPECIALS <= INVALID_UTF8 and len(SPECIALS) == 13

# ESC-space allocation table (ESC + payload byte). Append-only; every
# grant is recorded here and in claude-mind. Unlisted payloads are free —
# EXCEPT the special-range values (0xC0, 0xC1, 0xF5–0xFF), which are
# permanently unallocatable: an ESC payload equal to END/USER/DOC would
# false-fire every stateless raw-id scanner forever (esc() enforces it).
ESC_SLOTS = {
    0x00: "RESERVED_NULL",   # never allocate — the all-zero guard
    0x01: "CANARY",          # canary-episode marker (instruments only)
}

_DEFAULT_SYS = ("You are Beatrix, a small byte-level language model "
                "still in training.")


def is_special(b: int) -> bool:
    return int(b) in SPECIALS


def mode_tag(register: str) -> np.ndarray:
    """MODE + one ASCII payload byte, e.g. mode_tag('s') for simple."""
    p = register.encode("ascii")
    assert len(p) == 1, "MODE payload is exactly one ASCII byte"
    return np.array([MODE, p[0]], dtype=np.int64)


def esc(slot: int) -> np.ndarray:
    """ESC + payload byte -> one extended special (2-byte sequence)."""
    assert isinstance(slot, int) and not isinstance(slot, bool), slot
    assert 0 < slot <= 0xFF and slot not in INVALID_UTF8, (
        f"ESC slot {slot:#04x} is unallocatable: 0x00 is the null guard "
        "and special-range payloads would false-fire raw-id scanners")
    return np.array([ESC, slot], dtype=np.int64)


def _bytes_ids(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"),
                         dtype=np.uint8).astype(np.int64)


def render_chat_ids(turns: list, sys_text: str | None = _DEFAULT_SYS
                    ) -> np.ndarray:
    """Specials-native chat transcript as token ids (the v2s-era format):

        [SYS] sys bytes [END] { [USER]|[MODEL] content bytes [END] }*

    Text content is UTF-8 encoded, so it can never contain a special —
    the frame is unforgeable by construction. turns: [{'role': 'user' |
    'model', 'content': str}, ...]. Pass sys_text=None to omit the block.
    """
    parts = []
    if sys_text:
        parts += [np.array([SYS], dtype=np.int64), _bytes_ids(sys_text),
                  np.array([END], dtype=np.int64)]
    for t in turns:
        role = t["role"]
        # strict: an unknown role silently supervised as the model's own
        # speech would corrupt the very turn structure the frame teaches
        # (audit catch). 'assistant' is a deliberate alias for converted
        # datasets; everything else is rejected loudly.
        assert role in ("user", "model", "assistant"), f"unknown role {role!r}"
        opener = USER if role == "user" else MODEL
        parts += [np.array([opener], dtype=np.int64),
                  _bytes_ids(t["content"]),
                  np.array([END], dtype=np.int64)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def decode_visible(ids) -> str:
    """Human-readable decode: text spans via UTF-8, specials as ⟦NAME⟧.
    For sample printouts and probes — inference stop logic uses the raw
    ids, never this rendering. SPOOFABLE BY DESIGN: content containing
    literal ⟦NAME⟧ glyphs renders identically to a real special, so
    nothing downstream may PARSE this string — the ids are the truth."""
    out, run = [], []

    def flush():
        if run:
            out.append(bytes(run).decode("utf-8", errors="replace"))
            run.clear()

    seq = [int(i) for i in ids]
    i = 0
    while i < len(seq):
        b = seq[i]
        if b == ESC and i + 1 < len(seq):
            flush()
            name = ESC_SLOTS.get(seq[i + 1], f"{seq[i + 1]:02X}")
            out.append(f"⟦ESC:{name}⟧")
            i += 2
            continue
        if b in SPECIALS:
            flush()
            out.append(f"⟦{NAMES[b]}⟧")
        else:
            run.append(b)
        i += 1
    flush()
    return "".join(out)


def assert_unreachable(tokenizer, extra_texts: tuple = ()) -> None:
    """Runtime gate: PROVE the tokenizer cannot emit any special id, by
    exhaustion - every Unicode codepoint (surrogates included; the
    errors='replace' path maps them to '?') is encoded in one pass and
    the emitted byte set is intersected with the specials. Measured
    ~0.2s on CPU; the Trainer runs it once at init for byte crafts, so
    a future codec/errors= change is caught at launch, not in the data.
    (Audit 2026-08-26: the sweep's never-emitted set is EXACTLY
    {0xC0, 0xC1, 0xF5-0xFF} - the 13 specials, no more, no less.)"""
    everything = "".join(map(chr, range(0x110000)))
    for text in (everything,) + tuple(extra_texts):
        ids = tokenizer.encode(text)
        bad = SPECIALS.intersection(int(i) for i in np.unique(ids))
        assert not bad, (
            f"tokenizer emitted special ids {sorted(bad)} from plain text — "
            "the invalid-UTF-8 law is violated; specials are NOT safe here")
