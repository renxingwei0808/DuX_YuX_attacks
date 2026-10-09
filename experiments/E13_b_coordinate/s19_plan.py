"""E13 / W27 step 6 -- the S19 data and cost account for T1.

Two real-instance targets:

  * **DuX(2^8) 8/12** -- the cheap试验场: the six-layer distinguisher reaches
    `1101` already at three words, so the extended row is testable at 2^23/2^24.
  * **DuX(2^16) 12/12** -- the headline: two words (3, 7) over all of F_{2^16},
    2^32 chosen ciphertexts, layer 10, class `1101`; the extended combined row
    reads positions 1 and 3, so the weight range is the position-1 margin.

Everything is recomputed here: the degrees from `tools/cipher_degree.py` (via
`tools/zero_sum_criterion.py`), the weights from `weighted.plan(cheap_set=(1,2))`,
the columns from `bcoord.column_count`, and the cost from the measured
characteristic-2 throughput of the 8-round Yu2X-8 runs (`results/Y06_kr2/y08/`).

COST MODEL.  The characteristic-2 assembly is one log-domain gather + XOR per
element of every block-pair moment matrix, i.e. `npairs * nP^2 * N` of them per
structure, plus `N_w * nslice * width` for the weight accumulation (O12's slice
decomposition makes the weights almost free -- Y08 measured 1.6 % and 0.004 %).
The paper's row has `npairs = 10` (the unordered pairs, Mom being symmetric);
the extended row adds `Mom2`, which is NOT symmetric, so it needs all 16 ordered
pairs: **26 instead of 10**.  Y08 measured 43.8 core-hours per 2^32 structure on
a quiet machine at `10 * 63^2 * 2^32` gathers and 114.3 on a loaded one, which
is the 2.6x spread that section documents.

Usage
  python s19_plan.py --rank-max 16000 \
      --out ../../results/E13_b_coordinate/s19_plan.md \
      --json ../../results/E13_b_coordinate/s19_plan.json
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

import bcoord as BC                                              # noqa: E402
import weighted as WT                                            # noqa: E402
from assemble_fast import MomentLayout, Precomp                  # noqa: E402
from dux.registry import get_cipher                              # noqa: E402

# measured on the 8-round Yu2X-8 runs, results/Y06_kr2/y08/ (YuX, nP = 63, npairs = 10)
Y08 = {"nP": 63, "npairs": 10, "N": 2 ** 32,
       "core_hours_quiet": 43.8, "core_hours_loaded": 114.3}
GATHERS_REF = Y08["npairs"] * Y08["nP"] ** 2 * Y08["N"]

TARGETS = [
    # label, instance, active words, layers, pattern, rounds, weight axes,
    # subspace dimensions (None = whole words)
    ("DuX(2^16) 12/12", "dux-2^16", [3, 7], 10, "1101", 12, 1, None),
    ("DuX(2^8) 8/12, three whole words", "dux-2^8", [3, 7, 11], 6, "1101", 8,
     1, None),
    ("DuX(2^8) 8/12, T1 + T3 (three weight axes)", "dux-2^8", [3, 7, 11], 6,
     "1101", 8, 3, None),
    # the memo also lists the subspace variants; they are cheaper PER STRUCTURE
    # and dearer in TOTAL, because the margin -- hence the rows one structure
    # yields -- shrinks much faster than the point count.
    ("DuX(2^8) 8/12, subspace (8,8,7)", "dux-2^8", [3, 7, 11], 6, "1101", 8,
     1, [8, 8, 7]),
    ("DuX(2^8) 8/12, subspace (8,8,6)", "dux-2^8", [3, 7, 11], 6, "1101", 8,
     1, [8, 8, 6]),
]


def account(label, instance, active, layers, combine, rounds, dims, sub,
            rank_max):
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    preb = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lay, layb = MomentLayout(pre), BC.LayoutB(preb)
    pl = WT.plan(F, active, layers, "dec", "dux", combine, dims=dims, nweights=1,
                 cheap_set=(1, 2), subspace_dims=sub)
    crit = pl.crit
    D = crit["max_degree_per_position"][layers - 1]
    marg = crit["margin_per_position"][layers - 1]
    dimK = len(WT.combine_vectors("dux", F, "dec", combine, cheap=[1, 2]))
    nw = min(pl.usable_weights, math.ceil(rank_max / dimK))
    nstruct = max(1, math.ceil(rank_max / (dimK * nw)))
    N = F.q ** len(active)
    if sub:
        N = 1
        for m in sub:
            N *= 2 ** m
    nslice = F.q
    npairs_ext = len(layb.pairs)                 # 16 ordered, Mom2 is not symmetric
    gathers = nstruct * (len(lay.pairs) + npairs_ext) * layb.nP ** 2 * N
    wacc = nstruct * nw * nslice * layb.width
    ch_q = Y08["core_hours_quiet"] * gathers / GATHERS_REF
    ch_l = Y08["core_hours_loaded"] * gathers / GATHERS_REF
    elim_gb = rank_max * len(preb.mons) * 2 / 2 ** 30      # uint16, dense
    rows_gb = nw * dimK * nstruct * len(preb.mons) * 2 / 2 ** 30
    return {"row": label, "instance": instance, "field": F.name,
            "active": active, "subspace_dims": sub,
            "layers": layers, "rounds": rounds,
            "pattern": crit["patterns"][layers - 1], "combine": combine,
            "threshold": crit["threshold"], "D": D, "margin_per_position": marg,
            "used_positions": WT.used_positions("dec", "dux", combine),
            "weight_dims": dims,
            "weight_margin": pl.margin, "usable_weights": pl.usable_weights,
            "axis_cap": pl.axis_cap, "dim_K": dimK,
            "rank_max_b": rank_max, "N_w": nw, "structures": nstruct,
            "rows": nw * dimK * nstruct, "rows_per_structure": nw * dimK,
            "M_b": len(preb.mons), "M": len(pre.mons),
            "nP": layb.nP, "width_plain": lay.width, "width_ext": layb.width,
            "pairs_plain": len(lay.pairs), "pairs_ext": npairs_ext,
            "points": N, "log2_points": round(math.log2(N), 2),
            "log2_data": round(math.log2(nstruct * N), 2),
            "gathers_log2": round(math.log2(gathers), 2),
            "weight_acc_log2": round(math.log2(wacc), 2),
            "core_hours_quiet": round(ch_q, 1),
            "core_hours_loaded": round(ch_l, 1),
            "elimination_matrix_gb": round(elim_gb, 2),
            "row_stack_gb": round(rows_gb, 2),
            "weight_rule": pl.weight_rule}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-max", type=int, required=True,
                    help="the number of equation rows at which the extended "
                         "template determines the sixteen key words (step 3: "
                         "`rank_b.py`); NOT the rank at that point")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [account(*t, a.rank_max) for t in TARGETS]
    L = ["# E13 / T1 — the S19 data and cost account", "",
         f"All numbers recomputed here; `rank_max_b = {a.rank_max}` is the "
         "ROW COUNT at which the extended template determines the sixteen "
         "key words (step 3).  It is not a rank: the system is already "
         "rank-deficient there (12056 of 16000 rows on toy-2^4), and it is "
         "rows, not rank, that the data account has to buy.", "",
         "## 1. The cells", "",
         "| row | structure | T | D (layer) | margins | rows read | weight margin | weight axes | usable weights | dim K |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['row']} | {len(r['active'])} words {r['active']}"
                 + (f", dims {r['subspace_dims']}" if r['subspace_dims'] else "")
                 + f", 2^{r['log2_points']} | {r['threshold']} | {r['D']} "
                 f"| {r['margin_per_position']} | positions "
                 f"{r['used_positions']} | {r['weight_margin']} "
                 f"| {r['weight_dims']} | {r['usable_weights']} "
                 f"| {r['dim_K']} |")
    L += ["", "## 2. The system", "",
          "| row | M (paper) | M_b (extended) | N_w | rows/structure | structures | total rows | **total data** | row stack | elimination matrix |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['row']} | {r['M']} | **{r['M_b']}** | {r['N_w']} | "
                 f"{r['rows_per_structure']} | {r['structures']} | "
                 f"{r['rows']} | **2^{r['log2_data']}** | "
                 f"{r['row_stack_gb']} GB | "
                 f"**{r['elimination_matrix_gb']} GB** |")
    L += ["", "## 3. The cost", "",
          "The pair-moment count goes from 10 (unordered, `Mom` symmetric) to",
          "10 + 16 = 26, because `Mom2[A][B] = sum_P P_A^e (P_B^f)^2` has no",
          "symmetry.  Scaling Y08 Sect. 4.1's measured characteristic-2",
          "throughput by the gather count:", "",
          "| row | gathers/structure | weight accumulation | core-hours (quiet machine) | core-hours (loaded) |",
          "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['row']} | 2^{r['gathers_log2']} | "
                 f"2^{r['weight_acc_log2']} | **{r['core_hours_quiet']}** | "
                 f"{r['core_hours_loaded']} |")
    L += ["",
          "> Y08 Sect. 4.1 measured 43.8 core-hours per 2^32 structure on a",
          "> quiet machine and 114.3 on a loaded one for `10 * 63^2 * 2^32`",
          "> gathers; the two columns are that spread carried over.  The R9",
          "> memo's estimate was 250-500 core-hours, from scaling the F_p",
          "> 12-round run (45 core-hours) by 2.6 for the characteristic-2",
          "> kernel and by 2 for the extra moment family.", "",
          "## 4. The commands", ""]
    for r in rows:
        L.append("```bash")
        L.append(f"# {r['row']}: {r['N_w']} weights x dim K {r['dim_K']} = "
                 f"{r['rows_per_structure']} rows from each of "
                 f"{r['structures']} structure(s) of 2^{r['log2_points']} "
                 f"chosen ciphertexts (2^{r['log2_data']} in total)")
        L.append("python experiments/E13_b_coordinate/run_toy.py \\")
        L.append(f"    --instance {r['instance']} --layers {r['layers']} "
                 f"--active {','.join(map(str, r['active']))} \\")
        L.append(f"    --combine {r['combine']} --structures {r['structures']} "
                 + (f"--weight-dims {r['weight_dims']} "
                    if r['weight_dims'] > 1 else "")
                 + "--seeds 2026,7,11")
        L.append("```")
        L.append("")
    L += ["> `run_toy.py` is the in-memory driver; at 2^32 the structure does",
          "> not fit and S19 needs the streamed variant (the same slice",
          "> decomposition `bcoord.weighted_moments_b` implements, moved into",
          "> `attack_12round.py`'s worker layout).  That port is the one piece",
          "> of T1 this round leaves to the server.", ""]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
    if a.json:
        json.dump({"rank_max_b": a.rank_max, "reference": Y08, "rows": rows},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
