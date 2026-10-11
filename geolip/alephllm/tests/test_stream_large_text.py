"""Whole-book sources through the datasets shuffle buffer (0.10.13.1).
Run: python -m geolip.alephllm.tests.test_stream_large_text   (CPU, tokenless, no download: a local parquet source)
     ALEPHLLM_TEST_BIG=1 adds case 3 (about 2.2 GB of text held in memory; a minute or two)
Cases:
  1 the spec flag: gutenberg carries large_text; a flagged spec loads with its columns and large_string features, an
    unflagged spec with no extra arguments
  2 equivalence: a local parquet source of long rows (three columns, as the book source has) read through PackedStream with
    the flag and without, at a buffer that fits in 2 GiB: the same rows in the same order (the training role: the head
    skip, the shuffle; the validation role: the fixed head slice); the flagged stream's column is large_string; a flagged
    stream packs a batch
  3 the overflow (ALEPHLLM_TEST_BIG=1): a shuffle buffer holding more than 2 GiB of text: without the flag, datasets >= 5.1
    raises ArrowInvalid (offset overflow) [an earlier datasets never joins the buffer: reported, not failed]; with the
    flag the stream yields every row once
"""
from __future__ import annotations

import gc
import os
import tempfile

from ..data import curriculum as _cur  # noqa: F401 (registers the curriculum sources)
from ..data import streams
from ..data.streams import REGISTRY, PackedStream, _load_kwargs
from ..data.tokenizers import build_tokenizer

RESULTS: list = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def _datasets_version() -> tuple:
    import datasets
    return tuple(int(x) for x in datasets.__version__.split(".")[:2])


def _write_source(root: str, n_rows: int, row_bytes: int, rows_per_group: int = 64, n_files: int = 2) -> str:
    """A local parquet source shaped like the book source: TEXT (long, carries its row id), SOURCE, METADATA."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = os.path.join(root, "data")
    os.makedirs(d, exist_ok=True)
    schema = pa.schema([("TEXT", pa.string()), ("SOURCE", pa.string()), ("METADATA", pa.string())])
    per_file = -(-n_rows // n_files)
    for f in range(n_files):
        lo, hi = f * per_file, min(n_rows, (f + 1) * per_file)
        with pq.ParquetWriter(os.path.join(d, f"train-{f:05d}-of-{n_files:05d}.parquet"), schema,
                              compression="zstd") as w:
            for g in range(lo, hi, rows_per_group):
                ids = range(g, min(hi, g + rows_per_group))
                text = []
                for i in ids:
                    line = f"row {i:07d}: the long book goes on, page after page, line after line.\n"
                    text.append(f"<{i:07d}>\n" + line * (row_bytes // len(line)))
                w.write_table(pa.table({"TEXT": text, "SOURCE": ["synthetic"] * len(text),
                                        "METADATA": [f'{{"id": {i}}}' for i in ids]}, schema=schema))
    return root


def _row_id(row) -> int:
    t = row["TEXT"]
    return int(t[1:8])


def _stream(name: str, role: str, buffer: int) -> PackedStream:
    st = PackedStream(name, build_tokenizer("byte-trigram"), context=256, micro_batch=2, seed=1337,
                      shuffle_buffer=buffer, role=role)
    st._open()
    return st


def case_1():
    g = REGISTRY.get("gutenberg", {})
    check("1 gutenberg carries large_text", bool(g.get("large_text")), f"spec {sorted(g)}")
    kw = _load_kwargs(g)
    from datasets import Value
    ok = (kw.get("columns") == ["TEXT"] and list(kw.get("features", {})) == ["TEXT"]
          and kw["features"]["TEXT"] == Value("large_string"))
    check("1 a flagged spec loads TEXT as large_string", ok, repr(kw))
    check("1 an unflagged spec loads with no extra arguments", _load_kwargs(REGISTRY["tinystories"]) == {})


def case_2(root: str):
    n_rows, row_bytes, buffer = streams.RESERVED_HEAD_ROWS + 1024, 64 * 1024, 512
    src = _write_source(os.path.join(root, "c2"), n_rows, row_bytes)
    REGISTRY["_t2_plain"] = dict(path=src, split="train", column="TEXT", columns=["TEXT"])
    REGISTRY["_t2_large"] = dict(path=src, split="train", column="TEXT", columns=["TEXT"], large_text=True)
    try:
        a = _stream("_t2_plain", "train", buffer)
        b = _stream("_t2_large", "train", buffer)
        ids_a = [_row_id(r) for r in a._it]
        ids_b = [_row_id(r) for r in b._it]
        check("2 train: the same rows in the same order", ids_a == ids_b and len(ids_a) == 1024,
              f"{len(ids_a)} / {len(ids_b)} rows, first {ids_a[:4]} / {ids_b[:4]}")
        check("2 train: the head stays held out, every other row once",
              sorted(ids_b) == list(range(streams.RESERVED_HEAD_ROWS, n_rows)))
        ft = b._ds.features["TEXT"] if b._ds.features else None
        check("2 the flagged stream's column is large_string", getattr(ft, "dtype", None) == "large_string", repr(ft))
        va = _stream("_t2_plain", "val", buffer)
        vb = _stream("_t2_large", "val", buffer)
        ids_va = [_row_id(next(va._it)) for _ in range(64)]
        ids_vb = [_row_id(next(vb._it)) for _ in range(64)]
        check("2 val: the same fixed head slice", ids_va == ids_vb == list(range(64)), f"first {ids_vb[:4]}")
        x = _stream("_t2_large", "train", buffer).next_batch()
        check("2 a flagged stream packs a batch", tuple(x.shape) == (2, 257), f"shape {tuple(x.shape)}")
    finally:
        REGISTRY.pop("_t2_plain", None)
        REGISTRY.pop("_t2_large", None)


def case_3(root: str):
    head, buffer, row_bytes = 8, 2100, 1024 * 1024
    n_rows = head + buffer + 92
    src = _write_source(os.path.join(root, "c3"), n_rows, row_bytes, rows_per_group=32)
    REGISTRY["_t3_plain"] = dict(path=src, split="train", column="TEXT", columns=["TEXT"])
    REGISTRY["_t3_large"] = dict(path=src, split="train", column="TEXT", columns=["TEXT"], large_text=True)
    keep_head = streams.RESERVED_HEAD_ROWS
    streams.RESERVED_HEAD_ROWS = head
    try:
        import pyarrow as pa
        joined = _datasets_version() >= (5, 1)
        err = None
        try:
            _row_id(next(_stream("_t3_plain", "train", buffer)._it))
        except pa.ArrowInvalid as e:
            err = e
        gc.collect()
        if joined:
            check("3 without the flag: the buffer past 2 GiB overflows (datasets >= 5.1)",
                  err is not None and "offset overflow" in str(err), repr(err)[:160])
        else:
            check("3 without the flag: no overflow on this datasets (the buffer is never joined)", err is None,
                  repr(err)[:160])
        ids = [_row_id(r) for r in _stream("_t3_large", "train", buffer)._it]
        check("3 with the flag: every row once", sorted(ids) == list(range(head, n_rows)),
              f"{len(ids)} rows of {n_rows - head}")
    finally:
        streams.RESERVED_HEAD_ROWS = keep_head
        REGISTRY.pop("_t3_plain", None)
        REGISTRY.pop("_t3_large", None)
        gc.collect()


def main():
    import datasets
    import pyarrow
    print(f"datasets {datasets.__version__} · pyarrow {pyarrow.__version__}", flush=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        cases = [("case_1", case_1, ()), ("case_2", case_2, (root,))]
        if os.environ.get("ALEPHLLM_TEST_BIG") == "1":
            cases.append(("case_3", case_3, (root,)))
        else:
            print("SKIP case_3 (set ALEPHLLM_TEST_BIG=1: about 2.2 GB of text in memory)", flush=True)
        for name, fn, args in cases:
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 — a case's crash is a FAIL row, the rest still run
                import traceback
                traceback.print_exc()
                check(f"{name} raised", False, repr(e))
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
