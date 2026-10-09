"""E14 / W28 (T3) step 4 -- the S21 data plan: what every multi-structure row
of the ledger costs once the weights are multi-index.

For each row: the margin and the usable weights at dims = 1 (the published
accounting) and at dims = s (T3), the rows per structure, the number of
structures the rank ceiling forces, and the resulting data per key.

The rank ceilings are the measured ones of the ledger (`rank_max`, and the
number of rows a structure yields is `dim K` or 4 per weight), so the only
NEW quantity is the weight count.  Whether one structure's multi-index rows
actually reach the ceiling is the empirical question `rank_curves.py` measures
on the toys and S21 has to settle on the real instances; this table is what
S21 should expect if they do.

Usage
  python data_plan.py --out ../../results/E14_multi_weights/data_plan.md \
      --json ../../results/E14_multi_weights/data_plan.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
for p_ in (ROOT, os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round")):
    sys.path.insert(0, p_)

import weighted as WT                                            # noqa: E402
from dux.field import make_field                                 # noqa: E402

# (label, q, active, layers, cipher, combine, free_blocks, subspace_dims,
#  dims, rank_max (measured, ledger), published structures, published log2 data)
ROWS = [
    ("DuX(2^8) 8/12, four full words", "2^8", [3, 7, 11, 15], 6, "dux", None,
     (), None, 4, 4756, 5, 34.32),
    ("DuX(2^8) 8/12, dims (8,8,8,5)", "2^8", [3, 7, 11, 15], 6, "dux", None,
     (), [8, 8, 8, 5], 4, 4756, 75, 35.2),
    ("Yu2X-8 8/12, full block", "2^8", [], 6, "yux", "1110", (0,), None,
     4, 3822, 11, 35.46),
    ("Yu2X-8 7/12, two words", "2^8", [0, 4], 5, "yux", None, (), None,
     2, 4940, 40, 21.32),
    ("DuX(65537) 13 (dagger), full block", "65537", [], 11, "dux", "0001",
     (0,), None, 4, 8608, 308, 72.3),
]


def row(label, q, active, layers, cipher, combine, fb, sd, dims, rank_max,
        pub_structs, pub_bits):
    F = make_field(q)
    out = {"row": label, "q": q, "active": active, "layers": layers,
           "cipher": cipher, "combine": combine,
           "free_blocks": list(fb), "subspace_dims": sd,
           "rank_max": rank_max, "published_structures": pub_structs,
           "published_log2_data": pub_bits}
    if sd:
        pts = 1
        for m in sd:
            pts *= 2 ** m
    else:
        pts = (F.q ** len(active)) * ((F.q ** 4) ** len(fb))
    out["points"] = pts
    out["log2_points"] = round(math.log2(pts), 2)
    for d in (1, dims):
        pl = WT.plan(F, active, layers, "dec", cipher, combine, dims=d,
                     nweights=1, free_blocks=fb, subspace_dims=sd)
        nrows_per_weight = (len(WT.combine_vectors(cipher, F, "dec", combine))
                            if combine else 4)
        nw_needed = math.ceil(rank_max / nrows_per_weight)
        nw = min(pl.usable_weights, nw_needed)
        rows_per_structure = nw * nrows_per_weight
        nstruct = max(1, math.ceil(rank_max / rows_per_structure))
        out[f"dims{d}"] = {
            "margin": pl.margin, "usable_weights": pl.usable_weights,
            "axis_cap": pl.axis_cap, "rows_per_weight": nrows_per_weight,
            "N_w_needed": nw_needed, "N_w_used": nw,
            "rows_per_structure": rows_per_structure, "structures": nstruct,
            "log2_data": round(math.log2(nstruct * pts), 2),
            "rule": pl.weight_rule}
    out["dims"] = dims
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [row(*r) for r in ROWS]
    L = ["# E14 / T3 — the S21 data plan (multi-index weights)", "",
         "`rank_max` is the MEASURED ceiling of the ledger row; a structure",
         "yields `dim K` (or 4) rows per weight, so it needs",
         "`N_w = ceil(rank_max / rows_per_weight)` weights and",
         "`ceil(rank_max / rows_per_structure)` structures.  The only new",
         "quantity is the weight count; whether one structure's multi-index",
         "rows actually reach `rank_max` is what S21 has to measure.", "",
         "| row | points/structure | rank_max | dims = 1: weights / structures / data | dims = s: weights / structures / **data** | published |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        o, m = r["dims1"], r[f"dims{r['dims']}"]
        L.append(f"| {r['row']} | 2^{r['log2_points']} | {r['rank_max']} | "
                 f"{o['usable_weights']} / {o['structures']} / "
                 f"2^{o['log2_data']} | {m['usable_weights']} / "
                 f"{m['structures']} / **2^{m['log2_data']}** | "
                 f"{r['published_structures']} structures, "
                 f"2^{r['published_log2_data']} |")
    L += ["", "Notes.", "",
          "* The dims = 1 column is the repository's own accounting, not the",
          "  published one: the published rows were run BEFORE the axis cap of",
          "  W25 and before `plan` existed, and some of them used fewer weights",
          "  than the margin allows (the ledger records what was run).",
          "* The memo's targets were: DuX(2^8) 8/12 one structure 2^32 or",
          "  (8,8,8,5) 2^29; Yu2X-8 8/12 one structure 2^32; Yu2X-8 7/12 three",
          "  structures 2^17.6; DuX(65537) 13(dagger) three structures 2^65.6.",
          "* The 13(dagger) row is THEORY only (prime field, dim K = 1 combined",
          "  row); it moves the ledger's J section, not A.", ""]
    for r in rows:
        L.append(f"* **{r['row']}** — dims = {r['dims']}: "
                 + r["dims" + str(r["dims"])]["rule"])
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
    if a.json:
        json.dump({"rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
