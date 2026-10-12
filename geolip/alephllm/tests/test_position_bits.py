"""Bits by position (0.10.18, 2026-10-12; eval/position_bits.py + train/instruments.position_bits_gauge).
Run: python -m geolip.alephllm.tests.test_position_bits   (CPU, seconds)
Cases:
  1 exact band means on synthetic losses (a loss equal to the position: every band's mean is the mean of its positions, in bits);
    the counts cover every target once; the weighted mean of the bands equals the plain mean
  2 the doc frame: targets before the first DOC and the DOC targets are excluded; the distance counts from the DOC (1 = the byte after)
  3 merge / finish across batches equal one call on the concatenation
  4 the model gauge on a tiny craft: finite bits in every populated band, empty bands None with count 0, the row bands' weighted
    mean equals the val bpb of the same batches to 1e-5, train mode restored
  5 band names and the log line
"""
from __future__ import annotations

import math

import torch

from ..eval import position_bits as pb
from ..model.alephlm import AlephLM
from ..presets import AlephLMConfig
from ..train import instruments

RESULTS: list = []
LN2 = math.log(2.0)
TINY = dict(name="tiny-pos", d_model=64, n_layers=2, n_heads=4, context=160, hub_layers=(1,), hub_K=16, hub_D=8,
            head_K=32, head_D=8, hub_chunk=16, bank_experts=3)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip(), flush=True)


def wmean(res, frame="row"):
    tot = sum(v * n for (b, v), n in zip(res[frame].items(), res[frame + "_n"].values()) if v is not None)
    cnt = sum(res[frame + "_n"].values())
    return tot / cnt


# 1 exact band means
B, T = 3, 300
nll = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, T).clone()      # loss == position (nats)
r = pb.bits_by_position(nll)
names = pb.band_names()
exp = {"0-64": (0 + 63) / 2, "64-256": (64 + 255) / 2, "256-1024": (256 + 299) / 2}
ok = all(abs(r["row"][k] - v / LN2) < 1e-4 for k, v in exp.items())
check("1a band means exact", ok, {k: round(r["row"][k], 3) for k in exp})
check("1b counts cover every target once", sum(r["row_n"].values()) == B * T and r["row_n"]["0-64"] == B * 64 and r["row_n"]["1024-2048"] == 0, r["row_n"])
check("1c empty bands None", r["row"]["1024-2048"] is None and r["row"]["4096+"] is None)
check("1d weighted mean equals the plain mean", abs(wmean(r) - float(nll.mean()) / LN2) < 1e-6, f"{wmean(r):.5f} vs {float(nll.mean()) / LN2:.5f}")

# 2 the doc frame
nll = torch.ones(1, 40)
doc = torch.zeros(1, 40, dtype=torch.bool)
doc[0, 5] = True
doc[0, 20] = True
nll[0, 6] = 10.0          # distance 1 after the first DOC
nll[0, 35] = 10.0         # distance 15 after the second DOC
r = pb.bits_by_position(nll, doc)
n_doc = sum(r["doc_n"].values())
# targets 0..4 (before the first DOC) and the two DOC targets excluded: 40 - 5 - 2 = 33 in the doc frame, all in the 0-64 band
check("2a doc frame excludes the pre-DOC targets and the DOC targets", n_doc == 33 and r["doc_n"]["0-64"] == 33, r["doc_n"])
expect = (33 - 2 + 2 * 10.0) / 33 / LN2
check("2b doc frame mean", abs(r["doc"]["0-64"] - expect) < 1e-6, f"{r['doc']['0-64']:.5f} vs {expect:.5f}")
check("2c row frame unchanged by the mask", sum(r["row_n"].values()) == 40 and abs(wmean(r) - float(nll.mean()) / LN2) < 1e-6)
# the distance is counted from the DOC: with a band edge at 8, a DOC at 5 puts targets 6..13 in the first band
r8 = pb.bits_by_position(nll, doc, bands=(0, 8, 16, 64))
check("2d distances count from the DOC", r8["doc_n"]["0-8"] == 7 + 7 and r8["doc_n"]["8-16"] == 7 + 8 and r8["doc_n"]["16-64"] == 0 + 4, r8["doc_n"])

# 3 merge / finish
torch.manual_seed(0)
a, b = torch.rand(2, 50) * 3, torch.rand(3, 50) * 3
da, db = torch.rand(2, 50) < 0.05, torch.rand(3, 50) < 0.05
whole = pb.bits_by_position(torch.cat([a, b]), torch.cat([da, db]))
acc = pb.merge(None, pb.bits_by_position(a, da))
acc = pb.merge(acc, pb.bits_by_position(b, db))
m = pb.finish(acc)
ok = all((m["row"][k] is None and whole["row"][k] is None) or abs(m["row"][k] - whole["row"][k]) < 1e-6 for k in names) and m["row_n"] == whole["row_n"] \
    and all((m["doc"][k] is None and whole["doc"][k] is None) or abs(m["doc"][k] - whole["doc"][k]) < 1e-6 for k in names) and m["doc_n"] == whole["doc_n"]
check("3 merge / finish across batches equals one call", ok)

# 4 the model gauge
torch.manual_seed(1)
model = AlephLM(AlephLMConfig(**TINY))
model.train()
from ..data.special_tokens import DOC
batches = []
for i in range(2):
    xb = torch.randint(0, 256, (2, 161))
    xb[:, 40 + i * 7] = DOC
    batches.append(xb)
res = instruments.position_bits_gauge(model, batches)
populated = [k for k, v in res["row"].items() if v is not None]
check("4a populated bands finite, empty bands None", populated == ["0-64", "64-256"] and all(math.isfinite(res["row"][k]) for k in populated)
      and res["row"]["256-1024"] is None and res["row_n"]["256-1024"] == 0, res["row"])
with torch.no_grad():
    model.eval()
    tot = n = 0.0
    for xb in batches:
        lg, _ = model(xb[:, :-1])
        nl = torch.nn.functional.cross_entropy(lg.reshape(-1, 256).float(), xb[:, 1:].reshape(-1), reduction="none")
        tot += float(nl.sum())
        n += nl.numel()
    model.train()
check("4b the row bands' weighted mean equals the val bpb", abs(wmean(res) - tot / n / LN2) < 1e-5, f"{wmean(res):.5f} vs {tot / n / LN2:.5f}")
expected_doc = 0
for xb in batches:
    y = xb[:, 1:]
    for rr in range(y.shape[0]):
        hits = (y[rr] == DOC).nonzero().flatten()
        if len(hits):
            expected_doc += int(y.shape[1] - int(hits[0]) - len(hits))   # after the first DOC, the DOC targets themselves excluded
check("4c doc frame present for the byte craft; counts exclude the DOC targets and the pre-DOC targets",
      "doc" in res and sum(res["doc_n"].values()) == expected_doc, (sum(res["doc_n"].values()) if "doc" in res else None, expected_doc))
check("4d train mode restored", model.training)
check("4e no batches -> None", instruments.position_bits_gauge(model, []) is None)

# 5 names and the log line
check("5a band names", names == ["0-64", "64-256", "256-1024", "1024-2048", "2048-4096", "4096+"], names)
line = pb.fmt(res)
check("5b log line", line.startswith("bits by position (row): 0-64 ") and "64-256" in line and "256-1024" not in line, line)

n_pass = sum(ok for _, ok in RESULTS)
print(f"\n{n_pass}/{len(RESULTS)} checks passed")
if n_pass != len(RESULTS):
    raise SystemExit(1)
