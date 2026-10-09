"""E13 / W27 step 5 -- which cells the new usable class "x1x1" opens.

With the T1 cheap set {1, 2} the characteristic-2 combined row of every class
whose positions 1 AND 3 are balanced (0101, 0111, 1101, 1111 -- the "x1x1"
family) has dim K = 4 or 8 and does NOT collapse (`kernel_table.py`), so those
cells become key-recovery cells instead of distinguishers.  This re-runs
`min_data_v3.search_all` with the two cheap sets and lists the cells whose
verdict changes, per instance and layer.

Usage
  python min_data_x1x1.py --out ../../results/E13_b_coordinate/min_data_x1x1.md \
      --json ../../results/E13_b_coordinate/min_data_x1x1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
for p_ in (ROOT, os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments", "E09_unified_criterion"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round")):
    sys.path.insert(0, p_)

import min_data_v3 as MD                                          # noqa: E402
from min_data_v2 import INSTANCES                                 # noqa: E402
import weighted as WT                                             # noqa: E402
from dux.field import make_field                                  # noqa: E402

X1X1 = ("0101", "0111", "1101", "1111")


LAYERS = {"DuX(2^16)": 12, "DuX(2^8)": 12}
# both characteristic-2 DuX instances have twelve rounds, and the r_KR = 2
# attack costs two rounds on top of the distinguisher, so only layers <= 10
# describe an attack on the cipher itself.
MAX_ATTACK_LAYER = {"DuX(2^16)": 10, "DuX(2^8)": 10}


def instances_char2_dux():
    return [(name, cipher, q, LAYERS[name]) for name, cipher, q in INSTANCES
            if cipher == "dux" and str(q).startswith("2^")]


def compare(name, cipher, q, layers):
    base = MD.search_all(name, cipher, q, "dec", layers)
    ext = MD.search_all(name, cipher, q, "dec", layers, cheap=[1, 2])
    idx = {(r["layer"], r["class"]): r for r in base}
    out = []
    for r in ext:
        b = idx[(r["layer"], r["class"])]
        if not r["reachable"]:
            continue
        opened = (r["class"] in X1X1 and r["dim_K"] >= 2
                  and b["usable"].startswith("char-2 unusable")
                  or (r["class"] in X1X1 and r["dim_K"] >= 2
                      and b["dim_K"] <= 1))
        out.append({"instance": name, "q": q, "layer": r["layer"],
                    "class": r["class"], "log2_data": r["log2_data"],
                    "active": r.get("active"), "dims": r.get("dims"),
                    "free_blocks": r.get("free_blocks"),
                    "dim_K_cheap2": b["dim_K"], "usable_cheap2": b["usable"],
                    "dim_K_cheap12": r["dim_K"], "usable_cheap12": r["usable"],
                    "newly_usable": bool(opened)})
    return out


def weight_budget(q, active, layers, combine, dims=1, subspace_dims=None,
                  free_blocks=(), rank_max=12000):
    """The weights the T1 combined row of `combine` can use at that cell, and
    the resulting TOTAL data: dim K = 4 rows per weight, so one structure gives
    4 N_w rows and the attack needs ceil(rank_max / (4 N_w)) of them."""
    import math
    F = make_field(q)
    try:
        pl = WT.plan(F, active, layers, "dec", "dux", combine, dims=dims,
                     nweights=1, cheap_set=(1, 2),
                     subspace_dims=subspace_dims, free_blocks=tuple(free_blocks))
    except AssertionError as e:
        return {"error": str(e).splitlines()[0]}
    dimK = len(WT.combine_vectors("dux", F, "dec", combine, cheap=[1, 2]))
    rows_per = dimK * pl.usable_weights
    nstruct = max(1, math.ceil(rank_max / rows_per))
    if subspace_dims:
        pts = 1
        for m in subspace_dims:
            pts *= 2 ** m
    else:
        pts = F.q ** len(active) * (F.q ** 4) ** len(free_blocks)
    return {"margin": pl.margin, "usable_weights": pl.usable_weights,
            "axis_cap": pl.axis_cap, "dim_K": dimK,
            "rows_per_structure": rows_per, "structures": nstruct,
            "log2_points": round(math.log2(pts), 2),
            "log2_total_data": round(math.log2(nstruct * pts), 2),
            "rule": pl.weight_rule}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--from-json", default=None,
                    help="reuse the (expensive) min-data search of an earlier "
                         "JSON and only recompute the weight budgets")
    ap.add_argument("--rank-max", type=int, default=12000,
                    help="the number of equation rows at which the extended "
                         "template determines the sixteen key words (step 3); "
                         "NOT the rank there -- the number of structures "
                         "scales with it")
    a = ap.parse_args()
    if a.from_json and os.path.exists(a.from_json):
        rows = json.load(open(a.from_json))["rows"]
        print(f"(reusing the search of {a.from_json}: {len(rows)} cells)")
    else:
        rows = []
        for name, cipher, q, layers in instances_char2_dux():
            rows += compare(name, cipher, q, layers)
    opened = [r for r in rows if r["newly_usable"]]
    for r in opened:
        r["weights"] = weight_budget(r["q"], r["active"] or [], r["layer"],
                                     r["class"], subspace_dims=r.get("dims"),
                                     free_blocks=r.get("free_blocks") or (),
                                     rank_max=a.rank_max)
    # the deepest newly-usable cell per instance that actually has a weight
    # budget (a cell whose position-1 margin is negative is a distinguisher the
    # combined row cannot read)
    usable = [r for r in opened if "error" not in r["weights"]
              and r["layer"] <= MAX_ATTACK_LAYER[r["instance"]]]
    best = {}
    for r in usable:
        k = r["instance"]
        cur = (r["layer"], -r["weights"]["log2_total_data"])
        if k not in best or cur > (best[k]["layer"],
                                   -best[k]["weights"]["log2_total_data"]):
            best[k] = r

    L = ["# E13 / T1 — the cells the class x1x1 opens (characteristic 2, DuX)",
         "",
         "`min_data_v3.search_all` run twice, with the paper's cheap set {2} and",
         "with T1's {1, 2}.  A cell counts as NEWLY USABLE when its class has",
         "positions 1 and 3 balanced (0101, 0111, 1101, 1111) and its dim K goes",
         "from <= 1 (where the characteristic-2 orbit-sum theorem kills the",
         "combined row, W19-B) to >= 2.", "",
         "## 1. The deepest newly-usable cell per instance", "",
         f"With `rank_max_b = {a.rank_max}` -- the ROW COUNT at which the "
         "extended template determines the sixteen key words (step 3), not "
         "the rank there -- a cell needs `ceil(rank_max_b / (dim K x N_w))` "
         "structures.", "",
         "| instance | layer | class | structure | points | usable weights | rows/structure | structures | **total data** | dim K {2} -> {1,2} |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(best):
        r = best[name]
        st = (f"free block {r['free_blocks']}, " if r["free_blocks"] else "")
        st += f"words {r['active']}" + (f", dims {r['dims']}" if r["dims"] else "")
        w = r["weights"]
        L.append(f"| {name} | {r['layer']} | `{r['class']}` | {st} | "
                 f"2^{w['log2_points']} | {w['usable_weights']} | "
                 f"{w['rows_per_structure']} | {w['structures']} | "
                 f"**2^{w['log2_total_data']}** | {r['dim_K_cheap2']} -> "
                 f"{r['dim_K_cheap12']} |")
    L += ["", "## 2. Every newly-usable cell with a weight budget", "",
          "| instance | layer | class | structure | points | usable weights | structures | total data |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(usable, key=lambda r: (r["instance"], -r["layer"],
                                           r["weights"]["log2_total_data"])):
        w = r["weights"]
        st = (f"free block {r['free_blocks']}, " if r["free_blocks"] else "")
        st += f"words {r['active']}" + (f", dims {r['dims']}" if r["dims"] else "")
        L.append(f"| {r['instance']} | {r['layer']} | `{r['class']}` | {st} | "
                 f"2^{w['log2_points']} | {w['usable_weights']} | "
                 f"{w['structures']} | **2^{w['log2_total_data']}** |")
    L += ["", "Cells whose class is newly usable but whose POSITION-1 margin is "
          "negative are distinguishers the combined row cannot read, and layers "
          "above 10 would need more rounds than the cipher has (r = l + 2, and "
          "both instances have twelve); together they are "
          f"{len(opened) - len(usable)} of the {len(opened)} newly-usable cells "
          "and are listed in the JSON only."]
    L += ["", "## 3. YuX, for the record", "",
          "The same trick gives YuX the cheap set {2, 3}; `kernel_table.py"
          " --cipher yux` shows `1110` going from dim K = 3 to 4 and `1111`",
          "from 4 to 8.  The classes that BECOME nonzero (xx11, 11x1) all",
          "require position 3 balanced, which is what `1111` already requires,",
          "so no new route opens -- only the existing twelve-round route's",
          "weight budget drops by about a quarter (4 rows per weight instead",
          "of 3).", ""]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
    if a.json:
        json.dump({"rank_max_b": a.rank_max, "rows": rows,
                   "newly_usable": opened, "with_weight_budget": usable,
                   "deepest_per_instance": best}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
