"""E13 / W27 step 1 -- the O10 kernel table for the cheap set {1, 2}, with the
characteristic-2 tau-valuations that decide whether a combined row collapses.

Two readings of "does this row collapse", both reported:

 * the memo's: the kernel vector y, restricted to the rows of ONE balanced
   POSITION, is an element of F_2[Z_4] (one coefficient per outer block); a
   valuation below 3 means it is not in the ideal (tau^3) that the orbit-sum
   theorem leaves;
 * the assembly's: the combined row is sum_{b,c} (yL)_{4b+c} Phi_{b,c}, so what
   multiplies each equivariant family Phi_{0,c} is yhat_c = ((yL)_{4b+c})_b, and
   min_c v(yhat_c) < 3 is the statement that survives into `combine_rows_b`.

Both are computed; `usable` is min_c v(yhat_c) < 3.  Reference:
`experiments/E11_cheap_rows/char2_collapse.py` (the theorem and `tau_valuation`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cheap_rows import ALL_FIELDS, PATTERNS, describe                # noqa: E402


def tau_valuation(y):
    """The tau-adic valuation of sum_j y_j t^j in F_2[Z_4], tau = t + 1.

    Copied from `experiments/E11_cheap_rows/char2_collapse.py` (three
    experiment directories carry a `run.py`, so that module is imported by path
    elsewhere; this is four lines and the test locks the two against each
    other)."""
    poly = [int(v) & 1 for v in y]
    v = 0
    while v < 4:
        if sum(poly) % 2:
            return v
        q, carry = [0] * 4, 0
        for j in range(3, -1, -1):
            carry ^= poly[j]
            q[j] = carry
        poly = [q[(j + 1) % 4] for j in range(4)]
        v += 1
    return 4


def kernel_entry(cipher, field, pattern, cheap, direction="dec"):
    r = describe(cipher, field, direction, pattern, list(cheap))
    char2 = str(field).startswith("2^")
    bal = r["balanced_positions"]
    R = r["rows"]
    out = {"cipher": cipher, "field": field, "direction": direction,
           "pattern": pattern, "cheap": list(cheap), "dim_K": r["dim_K"],
           "rows": R, "vectors": []}
    for e in r["kernel"]:
        yL = e["yL"]
        item = {"y": e["y"], "yL": yL}
        if char2:
            hats = {str(c): [yL[4 * b + c] & 1 for b in range(4)] for c in cheap}
            item["yhat"] = hats
            item["tau_yhat"] = {c: tau_valuation(v) for c, v in hats.items()}
            item["tau_min"] = min(item["tau_yhat"].values())
            item["per_position"] = {
                str(p): [e["y"][R.index(4 * b + p)] & 1 for b in range(4)]
                for p in bal}
            item["tau_per_position"] = {
                p: tau_valuation(v) for p, v in item["per_position"].items()}
            item["usable"] = item["tau_min"] < 3
        out["vectors"].append(item)
    if char2:
        out["usable_vectors"] = sum(1 for v in out["vectors"] if v["usable"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="dux", choices=("dux", "yux"))
    ap.add_argument("--direction", default="dec", choices=("dec", "enc"))
    ap.add_argument("--cheap", default=None,
                    help="comma list; default {1,2} for DuX, {2,3} for YuX")
    ap.add_argument("--fields", default=",".join(ALL_FIELDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cheap = ([int(v) for v in a.cheap.split(",")] if a.cheap
             else ([1, 2] if a.cipher == "dux" else [2, 3]))
    fields = a.fields.split(",")
    rows, table = [], {}
    for field in fields:
        line = []
        for pat in PATTERNS:
            e = kernel_entry(a.cipher, field, pat, cheap, a.direction)
            rows.append(e)
            line.append(e["dim_K"])
        table[field] = line
        print(f"{a.cipher} {a.direction} F_{field:6s} cheap {cheap}: "
              + " | ".join(f"{v}" for v in line))
    print("patterns                       : " + " | ".join(PATTERNS))
    # the reference rows: the cheap set the paper uses
    base = [1] if a.cipher == "dux" else [3]
    base_table = {}
    for field in fields:
        base_table[field] = [describe(a.cipher, field, a.direction, pat,
                                      [2] if a.cipher == "dux" else [3])["dim_K"]
                             for pat in PATTERNS]
    print(f"{a.cipher} {a.direction} (paper's cheap set): "
          + " | ".join(str(v) for v in base_table[fields[0]]))
    del base
    char2 = [f for f in fields if f.startswith("2^")]
    if char2:
        print("\ncharacteristic-2 tau-valuations (assembly reading: "
              "min_c v(yhat_c); < 3 means no orbit-sum collapse)")
        for pat in PATTERNS:
            e = kernel_entry(a.cipher, char2[0], pat, cheap, a.direction)
            if not e["dim_K"]:
                continue
            print(f"  {pat}: dim K = {e['dim_K']}, tau_min = "
                  f"{[v['tau_min'] for v in e['vectors']]}, "
                  f"usable vectors = {e['usable_vectors']}")
    res = {"cipher": a.cipher, "direction": a.direction, "cheap": cheap,
           "fields": fields, "patterns": PATTERNS,
           "dim_K_table": table, "dim_K_table_paper_cheap_set": base_table,
           "entries": rows}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
