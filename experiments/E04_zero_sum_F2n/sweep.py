"""W5 (E04) -- systematic zero-sum structure sweep for DuX(2^n) within the
workstation budget (data <= 2^26).

Wraps experiments/E04_zero_sum_F2n/run.py's `run_structure` and runs a list of
(active words, per-word F_2-dimension) structures, aggregating everything into
one JSON plus a printed table.  A word counts as balanced only if its XOR over
the structure is 0 for EVERY key tried (--keys, default 3).

Modes
  --spec "3;3,7;3,7,11"            explicit structures (comma = words, ; = list)
  --spec-pairs                     all pairs (i,j), i in 0..3, i<j<=15
  --dims 4,5,6,7,8                 sweep the per-word dimension for each structure

Usage
  python sweep.py --instance dux-2^8  --spec-pairs --dims 8 --layers 8
  python sweep.py --instance dux-2^8  --spec "3" --dims 4,5,6,7,8 --layers 8
  python sweep.py --instance dux-2^16 --spec "3" --dims 8,10,12,14,16 --layers 12
  python sweep.py --instance dux-2^16 --spec "3,7" --dims 12,13 --layers 12 --chunk 22
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dux import DuX                                     # noqa: E402
from run import run_structure                           # noqa: E402


def one(c, active, dim, layers, keys, seed, chunk):
    rng = np.random.default_rng(seed)
    agg = None
    t0 = time.time()
    for _ in range(keys):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        b = run_structure(c, rks, active, dim, layers, rng, chunk)
        agg = b if agg is None else [[x and y for x, y in zip(r1, r2)]
                                     for r1, r2 in zip(agg, b)]
    counts = [sum(r) for r in agg]
    max_full = max((l + 1 for l in range(layers) if counts[l] == 16), default=0)
    last = max((l for l in range(layers) if counts[l] > 0), default=-1)
    return {"active": active, "dim": dim, "data_log2": dim * len(active),
            "layers": layers, "keys": keys, "seed": seed,
            "balanced_count_per_layer": counts,
            "balanced_matrix": agg,
            "max_layers_all16": max_full,
            "last_nontrivial_layer": last + 1,
            "pattern_last": "".join("1" if v else "0" for v in agg[last]) if last >= 0 else "-",
            "patterns_per_layer": ["".join("1" if v else "0" for v in agg[l])
                                   for l in range(layers)],
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^8")
    ap.add_argument("--spec", default="3")
    ap.add_argument("--spec-pairs", action="store_true")
    ap.add_argument("--dims", default=None, help="comma-separated per-word dimensions (default n)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--max-data-log2", type=int, default=26)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = DuX(a.instance)
    n = c.F.n
    dims = [n] if a.dims is None else [int(v) for v in a.dims.split(",")]
    if a.spec_pairs:
        structures = [[i, j] for i in range(4) for j in range(i + 1, 16)]
    else:
        structures = [[int(v) for v in part.split(",")] for part in a.spec.split(";")]

    rows = []
    print(f"{a.instance}  keys={a.keys}  layers={a.layers}  seed={a.seed}")
    print("structure                 dim  data   all16  last layer : pattern            s")
    for st in structures:
        for d in dims:
            if d > n:
                print(f"{str(st):24s} {d:4d}  SKIPPED (dimension > n = {n})")
                continue
            if d * len(st) > a.max_data_log2:
                print(f"{str(st):24s} {d:4d}  2^{d * len(st):<4d} SKIPPED (> 2^{a.max_data_log2})")
                continue
            r = one(c, st, d, a.layers, a.keys, a.seed, a.chunk)
            rows.append(r)
            print(f"{str(st):24s} {d:4d}  2^{r['data_log2']:<4d} {r['max_layers_all16']:5d}  "
                  f"L{r['last_nontrivial_layer']:<2d}: {r['pattern_last']:17s} {r['elapsed_s']:6.1f}")
    out = {"instance": a.instance, "n": n, "keys": a.keys, "layers": a.layers,
           "seed": a.seed, "rows": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
