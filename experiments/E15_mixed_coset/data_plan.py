"""E15 / W29 step 4 -- the S20 data plan for the mixed structures of T2.

Every row is recomputed with the repository's own tools: the degrees come from
`tools/zero_sum_criterion.py`, the usable weights from `weighted.plan`, and the
cost estimate scales the S16 run of the 12-round DuX(65537) attack
(`results/R8_server/s10_R8optA_dux65537_r12_pset39614_seed2026.json`) phase by phase:

    slice moments  ~ number of points
    weight dgemm   ~ N_w x (number of slices)
    row assembly   ~ N_w                       (independent of the points)
    elimination    ~ unchanged                 (same rows, same rank)

Usage
  python data_plan.py --out ../../results/E15_mixed_coset/data_plan.md \
      --json ../../results/E15_mixed_coset/data_plan.json
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
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round"), HERE):
    sys.path.insert(0, p_)

from dux.field import make_field                                 # noqa: E402
import mixed as MX                                      # noqa: E402
import weighted as WT                                            # noqa: E402
import zero_sum_criterion as zsc                                 # noqa: E402

# the S16 reference run: 12-round DuX(65537), two O15 point sets of 39 614
# points, N_w = 9 000, seed 2026 (results/R8_server/s10_R8optA_dux65537_r12_pset39614_seed2026.json)
REF = {"points": 39614 ** 2, "nslice": 39614, "nw": 9000,
       "slice_s": 2059, "dgemm_s": 658, "rows_s": 588, "elim_s": 639,
       "core_hours": 20.8, "wall": "1:09:31", "peak_gb": 19.3, "procs": 32}

# (label, q, active words, layers, combine, coset spec, N_w needed, dims)
ROWS = [
    ("DuX(65537) 12/12, k = 14", "65537", [3, 7], 10, "0001", [14, None], 8608, 1),
    ("DuX(65537) 12/12, k = 13", "65537", [3, 7], 10, "0001", [13, None], 8608, 2),
    ("DuX(65537) 11/12, k = 15", "65537", [3], 9, "0001", [15], 8608, 1),
    ("DuX(65537) 12/12, full x full (current)", "65537", [3, 7], 10, "0001",
     None, 8608, 1),
]


def cost(points, nslice, nw):
    s = REF["slice_s"] * points / REF["points"]
    d = REF["dgemm_s"] * (nslice / REF["nslice"]) * (nw / REF["nw"])
    r = REF["rows_s"] * nw / REF["nw"]
    e = REF["elim_s"]
    tot = s + d + r + e
    ref_tot = REF["slice_s"] + REF["dgemm_s"] + REF["rows_s"] + REF["elim_s"]
    return {"slice_s": round(s), "dgemm_s": round(d), "rows_s": round(r),
            "elim_s": e, "wall_s": round(tot),
            "core_hours": round(REF["core_hours"] * tot / ref_tot, 1)}


def row(label, q, active, layers, combine, coset, need, dims):
    F = make_field(q)
    pl = WT.plan(F, active, layers, "dec", "dux", combine, dims=dims,
                 nweights=1, coset=coset)
    crit = pl.crit
    D = crit["max_degree_per_position"][layers - 1]
    used = WT.used_positions("dec", "dux", combine)
    if coset is None:
        npts = F.q ** len(active)
        nslice = F.q
        kinds = ["full"] * len(active)
    else:
        npts, nslice = 1, None
        for j, k in enumerate(coset):
            n = F.q if k is None else (1 << k)
            npts *= n
            if j == 0:
                nslice = n
        kinds = ["full" if k is None else f"2^{k}" for k in coset]
    c = cost(npts, nslice, min(need, pl.usable_weights))
    return {"row": label, "q": q, "active": active, "layers": layers,
            "combine": combine, "structure": kinds, "dims": dims,
            "threshold": crit["threshold"],
            "threshold_plain": crit.get("threshold_plain"),
            "D": D, "used_positions": used,
            "margin": pl.margin, "usable_weights": pl.usable_weights,
            "axis_cap": pl.axis_cap, "needed_weights": need,
            # W29: on a coset of order 2^k two weights whose exponents agree
            # modulo 2^k give PROPORTIONAL rows, so `usable_weights` is only
            # attained while the margin stays below every coset order.  The
            # toy measures this exactly; all the targets
            # below are clear of it.
            "residues_collide": (False if coset is None
                                 else MX.residues_collide(pl.margin, coset)),
            "enough": pl.usable_weights >= need,
            "points": npts, "log2_data": round(math.log2(npts), 2),
            "slices": nslice, "cost": c, "weight_rule": pl.weight_rule}


def variants(q="65537", active=(3, 7), layers=10, combine="0001", need=8608):
    """The 'coset x point set' candidate: T = 2^k + (n - 1) (S16 already folds
    an inner-axis mask into the panel product, so the assembly is available)."""
    F = make_field(q)
    r = zsc.patterns(q, list(active), layers, None, None, "dec", "dux")
    D3 = r["max_degree_per_position"][layers - 1][3]
    out = []
    for k in (13, 14, 15, 16):
        # T = 2^k + (n - 1) > D3 + N_w  ->  n >= D3 + need + 1 - 2^k + 1
        n = D3 + need + 2 - (1 << k)
        if n < 2 or n > F.q:
            continue
        out.append({"k": k, "coset_points": 1 << k, "pointset_n": n,
                    "threshold": (1 << k) + n - 1, "D3": D3,
                    "margin": (1 << k) + n - 1 - D3,
                    "points": (1 << k) * n,
                    "log2_data": round(math.log2((1 << k) * n), 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [row(*r) for r in ROWS]
    var = variants()
    L = ["# E15 / T2 — the S20 data plan for mixed structures over F_p", "",
         "Every number below is recomputed by `tools/zero_sum_criterion.py` and",
         "`experiments/E07_key_recovery_2round/weighted.py::plan`; the cost model",
         "scales the S16 run `results/R8_server/s10_R8optA_dux65537_r12_pset39614_seed2026.json` phase by",
         "phase (slice moments ~ points, weight dgemm ~ N_w x slices, row",
         "assembly ~ N_w, elimination unchanged).", "",
         "## 1. The rows", "",
         "| row | structure | T | D (layer) | margin | usable weights | residues collide? | N_w needed | data | slices | est. core-hours |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['row']} | {' x '.join(r['structure'])} | {r['threshold']} "
                 f"| {r['D']} | {r['margin']} | {r['usable_weights']} "
                 f"({'dims ' + str(r['dims'])}) "
                 f"| {'**YES**' if r['residues_collide'] else 'no'} "
                 f"| {r['needed_weights']} "
                 f"| **2^{r['log2_data']}** | {r['slices']} "
                 f"| {r['cost']['core_hours']} |")
    L += ["", "The weight rules, verbatim from `plan`:", ""]
    for r in rows:
        L.append(f"* **{r['row']}** — {r['weight_rule']}")
    L += ["", "## 2. The commands", ""]
    for r in rows:
        if all(k == "full" for k in r["structure"]):
            continue
        spec = ",".join(r["structure"]).replace("2^", "")
        single = len(r["structure"]) == 1
        L.append("```bash")
        L.append(f"# {r['row']}: {r['needed_weights']} weights, "
                 f"2^{r['log2_data']} chosen ciphertexts per key")
        L.append(f"python experiments/E07_key_recovery_2round/attack_12round.py \\")
        L.append(f"    --instance dux-{r['q']} --layers {r['layers']} "
                 f"--active {','.join(map(str, r['active']))} \\")
        L.append((f"    --coset {spec} " if single else f"    --mixed {spec} ")
                 + f"--weights {r['needed_weights']} "
                 + (f"--weight-dims {r['dims']} " if r['dims'] > 1 else "")
                 + (f"--combine {r['combine']} " if r['combine'] else "--combine none ")
                 + "\\")
        L.append("    --normalise --structures 1 --procs 32 --seed 2026")
        L.append("```")
        L.append("")
    L += ["> `attack_12round.py --mixed` is the streamed driver; the in-memory",
          "> `attack_2round_fast.py --mixed 14,full` is the same rows for a toy",
          "> size.  Both default to the old behaviour without the switch.", "",
          "## 3. The 'coset x point set' variant (T = 2^k + (n - 1))", "",
          "| k | coset points | point-set size n | T | D_3 | margin | data |",
          "|---|---|---|---|---|---|---|"]
    for v in var:
        L.append(f"| {v['k']} | {v['coset_points']} | {v['pointset_n']} | "
                 f"{v['threshold']} | {v['D3']} | {v['margin']} | "
                 f"2^{v['log2_data']} |")
    L += ["", "Computed only, not run: it needs the divided-difference mask of the",
          "point-set axis AND the coset algebra on the other, which the criterion",
          "supports (the thresholds add, Theorem 1 being per word) but no driver",
          "combines yet.", ""]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        print("saved", a.out, file=sys.stderr)
    if a.json:
        json.dump({"reference_run": REF, "rows": rows, "variants": var},
                  open(a.json, "w"), indent=1)
        print("saved", a.json, file=sys.stderr)


if __name__ == "__main__":
    main()
