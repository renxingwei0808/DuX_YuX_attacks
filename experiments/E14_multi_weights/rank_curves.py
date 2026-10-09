"""E14 / W28 (T3) step 2 -- do multi-index weights carry rank like extra
structures?

The template bound cannot answer this.  A weighted moment vector of a real
structure is `sum_x x^a delta_x` with `sum_x x^a = 0`, i.e. an element of the
same span V_0 of signed measures that `rank_phi.realizable_moments` generates,
so the pseudo-structure ceiling is the SAME for weighted and unweighted rows --
adding a `--weight-dims` switch there would be a no-op.  (That is also why the
W25 axis cap had to be introduced separately: it bounds how many of the N_w
weighted rows of ONE structure can be independent, which the ceiling does not
see.)  The question is therefore empirical, and this script measures it:

    (a)  N structures x every admissible ONE-dimensional weight
    (b)  ONE structure x N_w TWO-dimensional weights

on the same cell, at matched row counts, on >= 2 random master keys, reporting
the rank, the number of pinned monomials and how many of the sixteen inner key
words come out.

Usage
  python rank_curves.py --instance toy-2^4 --layers 3 --active 3,7 \
      --grid 64,256,512,1024,2048,4756 --seeds 2026,7 \
      --out ../../results/E14_multi_weights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for p_ in (ROOT, os.path.join(ROOT, "tools"), E07, os.path.join(E07, "fast")):
    sys.path.insert(0, p_)

import weighted as WT                                            # noqa: E402
from assemble_fast import Precomp                                # noqa: E402
from attack_2round_toy import active_words, linear_row           # noqa: E402
from dux.registry import cipher_family, get_cipher               # noqa: E402
from nmin_scan import pinned_columns                             # noqa: E402
import gf2n_solve                                                # noqa: E402
import modp_solve                                                # noqa: E402


def solve_prefix(pre, F, R, nrows):
    known = pinned_columns(pre, normalise=True)
    kc = np.array(sorted(known), dtype=np.int64)
    kv = np.array([known[int(i)] for i in kc], dtype=np.int64)
    sub = R[:nrows]
    if F.char == 2:
        B = np.zeros(nrows, dtype=np.int64)
        for t, ci in enumerate(kc):
            B ^= F.vmul(sub[:, ci], np.int64(kv[t]))
    else:
        B = (-(sub[:, kc] @ kv)) % F.p
    keep = [i for i in range(len(pre.mons)) if i not in known]
    A = np.ascontiguousarray(sub[:, keep])
    if F.char == 2:
        det, rank, nfree = (gf2n_solve.solve(F, A, B) if gf2n_solve.available()
                            else modp_solve.solve_gf2n(F, A, B))
    else:
        det, rank, nfree = modp_solve.solve(F.p, A, B)
    back = {keep[j]: j for j in range(len(keep))}
    return det, int(rank), int(nfree), back


def key_words(pre, det, back, rks):
    good, got = 0, {}
    for b in range(4):
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            ci = pre.idx[(None, ((b, e),))]
            j = back.get(ci)
            if j in det:
                got[f"{b},{i}"] = int(det[j])
                good += int(det[j]) == int(rks[0][4 * b + i])
    return good, got


def build(instance, layers, active, dims, nstruct, nweights, seed, combine=None):
    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    fam = cipher_family(instance)
    act = active_words(3, active)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher=fam)
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(seed)))
    pl = WT.plan(F, act, layers, "dec", fam, combine, dims=dims,
                 nweights=nweights)
    ycomb = WT.combine_vectors(fam, F, "dec", combine) if combine else None
    rows, t0 = [], time.time()
    for st in range(nstruct):
        rr, _o, npts = WT.weighted_rows(pre, c, rks, rounds, 3,
                                        seed + 5000 + st, active, pl.weights,
                                        lrow=lrow, ycomb=ycomb)
        rows.append(rr)
    R = np.concatenate(rows, axis=0)
    return {"pre": pre, "F": F, "rks": rks, "R": R, "plan": pl,
            "points": npts, "structures": nstruct,
            "assemble_s": round(time.time() - t0, 1)}


def curve(b, grid):
    out = []
    for n in grid:
        if n > b["R"].shape[0]:
            continue
        det, rank, nfree, back = solve_prefix(b["pre"], b["F"], b["R"], n)
        good, _got = key_words(b["pre"], det, back, b["rks"])
        out.append({"rows": int(n), "rank": rank, "free": nfree,
                    "pinned": len(det), "key_words": good})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-2^4")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--combine", default=None)
    ap.add_argument("--grid", default="64,256,512,1024,2048,4756")
    ap.add_argument("--structures-1d", type=int, default=75)
    ap.add_argument("--structures-2d", type=int, default=10)
    ap.add_argument("--weights-2d", type=int, default=None)
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid = [int(v) for v in a.grid.split(",")]
    combine = None if a.combine in (None, "none") else a.combine
    runs = []
    for seed in [int(v) for v in a.seeds.split(",")]:
        one = build(a.instance, a.layers, a.active, 1, a.structures_1d, None,
                    seed, combine)
        two = build(a.instance, a.layers, a.active, 2, a.structures_2d,
                    a.weights_2d, seed, combine)
        c1, c2 = curve(one, grid), curve(two, grid)
        print(f"  seed {seed}: 1-axis {one['plan'].usable_weights} weights "
              f"x {a.structures_1d} structures = {one['R'].shape[0]} rows "
              f"(2^{np.log2(a.structures_1d * one['points']):.2f} data); "
              f"2-axis {len(two['plan'].weights)} weights x "
              f"{a.structures_2d} structures = {two['R'].shape[0]} rows "
              f"(2^{np.log2(a.structures_2d * two['points']):.2f} data)",
              flush=True)
        for r1, r2 in zip(c1, c2):
            print(f"    rows {r1['rows']:6d}: 1-axis rank {r1['rank']:6d} "
                  f"pinned {r1['pinned']:6d} keys {r1['key_words']:2d}/16  |  "
                  f"2-axis rank {r2['rank']:6d} pinned {r2['pinned']:6d} "
                  f"keys {r2['key_words']:2d}/16", flush=True)
        runs.append({"seed": seed,
                     "one_axis": {"weights": one["plan"].usable_weights,
                                  "structures": a.structures_1d,
                                  "rows": int(one["R"].shape[0]),
                                  "points_per_structure": one["points"],
                                  "log2_data": round(float(np.log2(
                                      a.structures_1d * one["points"])), 2),
                                  "axis_cap": one["plan"].axis_cap,
                                  "rule": one["plan"].weight_rule,
                                  "assemble_s": one["assemble_s"],
                                  "curve": c1},
                     "two_axis": {"weights": len(two["plan"].weights),
                                  "usable": two["plan"].usable_weights,
                                  "structures": a.structures_2d,
                                  "rows": int(two["R"].shape[0]),
                                  "points_per_structure": two["points"],
                                  "log2_data": round(float(np.log2(
                                      a.structures_2d * two["points"])), 2),
                                  "axis_cap": two["plan"].axis_cap,
                                  "rule": two["plan"].weight_rule,
                                  "assemble_s": two["assemble_s"],
                                  "curve": c2}})
    res = {"instance": a.instance, "layers": a.layers, "active": a.active,
           "combine": combine, "grid": grid, "runs": runs}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"rank_curves_{a.instance}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
