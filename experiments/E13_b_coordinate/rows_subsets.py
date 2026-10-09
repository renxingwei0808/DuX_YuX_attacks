"""S19 (R9 / T1) -- solve WEIGHT SUBSETS of the rows a streamed run saved.

`attack_12round.py --cheap-set 1,2 --rows-dir D` writes one structure's rows
as `rows_<tag>_st<k>.npy`, weight-major: for a = 0, 1, ... the dim K combined
rows of that weight.  This script re-solves selected weight subsets of those
rows at EQUAL row counts -- the memo's "odd weights vs all weights" control
(the T1 "risk": an even weight gives sum x^{2a} Y^2 = (sum x^a Y)^2, so
those rows might be Frobenius-dependent on others) on the real instance, and
the row-count curve of the all-weights system -- without re-running the 2^32
assembly.  Same pinning and the same solver as the driver; the sixteen
degree-1 inner columns are checked against the same master key (`--seed`).

  python rows_subsets.py --instance dux-2^16 --layers 10 --active 3,7 \
      --rows ~/dux-data/S19_dux2p16_seed2026/rows_<tag>_st0.npy --seed 2026 \
      --subsets all:3600,odd:3600,even:3600,all:7200 --out results/R9_server/x.json
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
for p_ in (ROOT, os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round", "fast"),
           HERE):
    sys.path.insert(0, p_)

import bcoord as BC                                              # noqa: E402
from dux.registry import get_cipher                              # noqa: E402
import gf2n_solve                                                # noqa: E402
import modp_solve                                                # noqa: E402


def parse_subset(spec):
    """'odd:3600' -> (predicate on a, number of weights); 'all:3600' takes
    a = 0..3599, 'odd:3600' the first 3600 odd a, 'even:3600' the first 3600
    even a, 'a<7200' every a below 7200."""
    kind, _, n = spec.partition(":")
    if kind.startswith("a<"):
        top = int(kind[2:])
        return (lambda a: a < top), None, spec
    n = int(n)
    pred = {"all": (lambda a: True), "odd": (lambda a: a % 2 == 1),
            "even": (lambda a: a % 2 == 0)}[kind]
    return pred, n, spec


def solve_subset(pre, F, R, dimK, nw_total, pred, count, truth):
    """Rows of the weights a in [0, nw_total) with pred(a), the first `count` of
    them; returns the rank / pinned / 16-of-16 record."""
    sel = [a for a in range(nw_total) if pred(a)]
    if count is not None:
        sel = sel[:count]
    idx = np.concatenate([np.arange(a * dimK, (a + 1) * dimK) for a in sel])
    sub = R[idx]
    zero = (0, 0, 0, 0)
    known = {pre.col_const: 1}
    for m, i in pre.idx.items():
        tag, inner = m
        if tag is None and inner and all(e == zero for _b, e in inner):
            known[i] = 1
    kc = sorted(known)
    keep = [i for i in range(len(pre.mons)) if i not in known]
    B = np.zeros(sub.shape[0], dtype=np.int64)
    for ci in kc:
        B ^= F.vmul(sub[:, ci].astype(np.int64), np.int64(known[ci]))
    A = np.ascontiguousarray(sub[:, keep])
    t0 = time.time()
    det, rank, nfree = (gf2n_solve.solve(F, A, B) if gf2n_solve.available(None)
                        else modp_solve.solve_gf2n(F, A, B))
    back = {keep[j]: j for j in range(len(keep))}
    good = 0
    for b in range(4):
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            ci = pre.idx[(None, ((b, e),))]
            if back.get(ci) in det and int(det[back[ci]]) == truth[4 * b + i]:
                good += 1
    return {"weights": len(sel), "first_weight": int(sel[0]),
            "last_weight": int(sel[-1]), "rows": int(sub.shape[0]),
            "rank": int(rank), "free": int(nfree), "pinned": int(len(det)),
            "inner_correct": good, "solve_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^16")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--combine", default="1101")
    ap.add_argument("--rows", required=True, help="the rows_*.npy file(s), comma list")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--subsets", default="all:3600,odd:3600,even:3600")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rounds = a.layers + 2
    c = get_cipher(a.instance, rounds=rounds)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    from cheap_rows import block_coeffs
    dimK = len(block_coeffs("dux", f"2^{F.n}", "dec", a.combine, list(BC.CHEAP_SET)))
    rks = c.key_schedule(c.random_key(np.random.default_rng(a.seed)))
    truth = [int(v) for v in rks[0]]
    R = np.concatenate([np.load(fn) for fn in a.rows.split(",")], axis=0)
    assert R.shape[1] == len(pre.mons), (R.shape, len(pre.mons))
    assert R.shape[0] % dimK == 0
    nw_total = R.shape[0] // dimK
    print(f"{a.instance}: {R.shape[0]} rows = {nw_total} weights x dim K {dimK}, "
          f"M_b = {len(pre.mons)}", flush=True)
    out = []
    for spec in a.subsets.split(","):
        pred, count, label = parse_subset(spec)
        r = solve_subset(pre, F, R, dimK, nw_total, pred, count, truth)
        r["subset"] = label
        out.append(r)
        print(f"  {label:>10}: weights {r['weights']} (a = {r['first_weight']}.."
              f"{r['last_weight']}), rows {r['rows']}, rank {r['rank']}, "
              f"pinned {r['pinned']}, {r['inner_correct']}/16 ({r['solve_s']} s)",
              flush=True)
    res = {"instance": a.instance, "layers": a.layers, "active": a.active,
           "combine": a.combine, "cheap_set": list(BC.CHEAP_SET), "dim_K": dimK,
           "rows_files": a.rows.split(","), "seed": a.seed,
           "weights_available": nw_total, "subsets": out}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
