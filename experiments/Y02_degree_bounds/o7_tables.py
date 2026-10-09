"""Y02 / W15-B -- the O7 prediction table for every YuX structure R5 uses.

One row per (instance, structure, direction): the threshold T, the number of
full-zero-sum layers, the pattern of the first non-full layer and the four
per-position margins T - D there.  These rows are the PREDICTIONS that W17
(Y03) measures on toys and small instances and that S11/S12 measure at full
scale.

Data is log2 of the number of chosen ciphertexts (chosen plaintexts in the
`enc` direction) PER KEY.

Usage:
    python experiments/Y02_degree_bounds/o7_tables.py \
        --out results/Y02_degree_bounds
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from zero_sum_criterion import patterns  # noqa: E402

# instance -> (label, q for the criterion, layers to run)
INSTANCES = [
    ("yu2x-8", "Yu2X-8", "2^8", 9),
    ("yu2x-16", "Yu2X-16", "2^16", 12),
    ("yupx-65537", "YupX-65537", 65537, 12),
    ("yuxtoy-2^4", "yuxtoy-2^4", "2^4", 8),
    ("yuxtoy-193", "yuxtoy-193", 193, 8),
    ("yuxtoy-257", "yuxtoy-257", 257, 8),
]

# (label, active words, free blocks, direction)
STRUCTURES = [
    ("1 word, position 0", [0], [], "dec"),
    ("1 word, position 3", [3], [], "dec"),
    ("2 words (0,4)", [0, 4], [], "dec"),
    ("3 words (0,4,8)", [0, 4, 8], [], "dec"),
    ("4 words (0,4,8,12)", [0, 4, 8, 12], [], "dec"),
    ("5 words (0,1,4,8,12)", [0, 1, 4, 8, 12], [], "dec"),
    ("6 words (0,1,3,4,8,12)", [0, 1, 3, 4, 8, 12], [], "dec"),
    ("7 words (0,1,3,4,6,8,12)", [0, 1, 3, 4, 6, 8, 12], [], "dec"),
    ("full block 0", [], [0], "dec"),
    ("full block 0 + word 4", [4], [0], "dec"),
    ("full block 0 + words (4,8,12)", [4, 8, 12], [0], "dec"),
    ("full blocks 0,1", [], [0, 1], "dec"),
    ("CPA 1 word, position 0", [0], [], "enc"),
    ("CPA 1 word, position 1", [1], [], "enc"),
    ("CPA 2 words (1,5)", [1, 5], [], "enc"),
    ("CPA full block 0", [], [0], "enc"),
]


def rows(cipher="yux", layers_default=12):
    out = []
    for inst, label, q, nl in INSTANCES:
        for sname, active, free, direction in STRUCTURES:
            r = patterns(q, active, nl, None, None, direction, cipher, free)
            l = r["l_full"]
            out.append({
                "instance": inst, "instance_label": label, "q": str(q),
                "structure": sname, "active": active, "free_blocks": free,
                "direction": direction, "s": r["s"], "threshold": r["threshold"],
                "log2_data": r["log2_data"], "l_full": l, "next": r["next"],
                "margins_next": r["margin_per_position"][l] if l < nl else None,
                "patterns": r["patterns"],
            })
    return out


def coset_rows(ks=range(8, 17), layers=12):
    """YupX: a single order-2^k subgroup coset per active word (O12 Sect. 5.5
    makes one coset enough as soon as the weight a is not a multiple of 2^k)."""
    out = []
    for k in ks:
        for active in ([0], [3]):
            r = patterns(65537, active, layers, None, k, "dec", "yux", [])
            l = r["l_full"]
            out.append({"instance": "yupx-65537", "structure":
                        f"1 word position {active[0]}, 2^{k} coset",
                        "active": active, "coset_k": k, "s": 1,
                        "threshold": r["threshold"], "log2_data": k,
                        "l_full": l, "next": r["next"],
                        "margins_next": r["margin_per_position"][l] if l < layers else None})
    return out


def markdown(rs, cs):
    L = ["# Y02 . O7 predictions for the YuX structures used in R5\n",
         "Threshold `T = s (2^n - 1)` over `F_{2^n}` and `T = s (p - 1)` over "
         "`F_p` (full field), `T = 2^k` for one order-`2^k` subgroup coset. "
         "`l_full` is the number of layers on which all sixteen words are "
         "balanced; `next` is the 4-bit pattern of block positions still "
         "balanced on layer `l_full + 1`, with the margins `T - D` beside it "
         "(a negative margin means the criterion does not apply; a margin of "
         "exactly 0 or -4 marks the Frobenius boundary `D = 2^k`, where "
         "balance is possible but O7 cannot prove it).\n",
         "Data is log2 of the chosen ciphertexts (chosen plaintexts for the "
         "CPA rows) **per key**.\n"]
    for inst, label, q, _nl in INSTANCES:
        sub = [r for r in rs if r["instance"] == inst]
        L.append(f"## {label}  (`{inst}`, F_{q})\n")
        L.append("| structure | s | data log2 | T | full layers | next layer | margins (pos 0..3) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sub:
            L.append(f"| {r['structure']} | {r['s']} | {r['log2_data']} | "
                     f"{r['threshold']} | {r['l_full']} | {r['next']} | "
                     f"{r['margins_next']} |")
        L.append("")
    L.append("## YupX-65537, single subgroup coset (O12 Sect. 5.5)\n")
    L.append("| structure | data log2 | T | full layers | next layer | margins (pos 0..3) |")
    L.append("|---|---|---|---|---|---|")
    for r in cs:
        L.append(f"| {r['structure']} | {r['log2_data']} | {r['threshold']} | "
                 f"{r['l_full']} | {r['next']} | {r['margins_next']} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rs, cs = rows(), coset_rows()
    for r in rs:
        if r["instance"] in ("yu2x-16", "yupx-65537", "yu2x-8"):
            print(f"{r['instance_label']:11s} {r['structure']:32s} s={r['s']:2d} "
                  f"data 2^{r['log2_data']:<6} full={r['l_full']:2d} "
                  f"next={r['next']} margins {r['margins_next']}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({"structures": rs, "cosets": cs},
                  open(os.path.join(a.out, "o7_yux_tables.json"), "w"), indent=1)
        open(os.path.join(a.out, "o7_yux_tables.md"), "w").write(markdown(rs, cs))
        print("wrote o7_yux_tables.{json,md}")


if __name__ == "__main__":
    main()
