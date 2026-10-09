"""E13 / W27 step 3 -- the TEMPLATE bound of the extended (cheap set {1, 2})
linearisation, and whether it determines the sixteen inner key words.

Same question and same method as `E07/rank_phi.py` and
`E11/char2_collapse.py --ceiling`: every equation row is a fixed linear image
of the structure's point sums,

    row_{s,(j,c)}  =  Phi_{j,c} . m_s ,

so the rank of any real system is bounded by rank(Phi_stack) and, on the
realizable generator, that bound is SHARP (m ranges exactly over the span V_0
of the signed measures of total weight 0 that real structures produce).  A
negative answer here -- the 16 degree-1 key columns not determined at the
ceiling -- would kill T1 before any decryption.

The only change from `rank_phi.py` is the wider moment vector: the cubic row
also reads pm2[b] (the squared single moments) and Mom2[A][B] (the squared-
second-factor pair moments), so the pseudo-structure generators are the ones
in `bcoord.py`.

Usage
  python rank_b.py --instance toy-2^4 --rounds 5 --combine 1101 \
      --structures 3000 --checkpoints 500,1000,2000,3000 \
      --out ../../results/E13_b_coordinate/rank_toy-2^4_1101.json
  python rank_b.py --instance dux-2^8 --rounds 8 --combine 1101 --moments uniform
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
for p in (ROOT, os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "experiments", "E07_key_recovery_2round"),
          os.path.join(ROOT, "experiments", "E07_key_recovery_2round", "fast"),
          HERE):
    sys.path.insert(0, p)

import bcoord as BC                                             # noqa: E402
from dux.registry import get_cipher                             # noqa: E402
from attack_2round_toy import linear_row                        # noqa: E402
from cheap_rows import block_coeffs                             # noqa: E402
from nmin_scan import pinned_columns                            # noqa: E402
import gf2n_solve                                               # noqa: E402
import modp_solve                                               # noqa: E402


def degree_one_columns(pre):
    """The 16 columns carrying a single inner key word rk^0_{4b+c}."""
    cols = {}
    for b in range(4):
        for c in range(4):
            u = tuple(1 if t == c else 0 for t in range(4))
            m = (None, ((b, u),))
            if m in pre.idx:
                cols[f"{4 * b + c}"] = int(pre.idx[m])
    return cols


def outer_key_columns(pre):
    """The columns carrying a single outer key word k'_i (and k'^2_3) with an
    EMPTY inner part.

    These sixteen columns are identically zero in every row, combined or not:
    a pure outer-key term carries the factor sum_P 1 = q^s = 0 (see the (1)-(6)
    expansion in `bcoord`), so the count below is a sanity check on the
    expansion -- it must stay 0 -- and NOT a statement about whether the outer
    key is recoverable.  The outer key words do appear, but always multiplied
    by an inner monomial (the columns k'_i * k^m), and the gate for T1 is the
    sixteen degree-1 INNER columns of `degree_one_columns`."""
    return {f"{j},{i}": int(pre.idx[((j, i), ())])
            for j in pre.outer for i in pre.outer_coords}


def build_rows(pre, lrow, gen, nstruct, ycomb, progress=0):
    """`nstruct` pseudo-structures' worth of rows, as a uint16 stack."""
    nrows = len(ycomb) if ycomb else 2 * len(pre.outer)
    acc = np.zeros((nrows * nstruct, len(pre.mons)), dtype=np.uint16)
    t0 = time.time()
    for s in range(nstruct):
        rows = BC.rows_from_moments_b(pre, gen(), lrow)
        if ycomb:
            out = [BC.combine_rows_b(rows, y, pre.F) for y in ycomb]
        else:
            out = [rows[k] for k in sorted(rows)]
        acc[nrows * s:nrows * (s + 1)] = np.asarray(out, dtype=np.uint16)
        if progress and (s + 1) % progress == 0:
            print(f"    .. {s + 1}/{nstruct} pseudo-structures "
                  f"({time.time() - t0:.0f} s)", flush=True)
    return acc, nrows


def determinacy(pre, F, acc, normalise=True):
    known = pinned_columns(pre, normalise=normalise)
    kc = np.array(sorted(known), dtype=np.int64)
    kv = np.array([known[int(i)] for i in kc], dtype=np.int64)
    B = np.zeros(acc.shape[0], dtype=np.int64)
    for t, ci in enumerate(kc):
        B ^= F.vmul(acc[:, ci].astype(np.int64), np.int64(kv[t]))
    keep = np.array([i for i in range(len(pre.mons)) if i not in known],
                    dtype=np.int64)
    sub = acc[:, keep]
    det, rank, nfree = gf2n_solve.solve(F, sub, B) if gf2n_solve.available() \
        else modp_solve.solve_gf2n(F, sub, B)
    back = {int(keep[j]): j for j in range(len(keep))}
    d1 = degree_one_columns(pre)
    ok = {w: (back.get(ci) in det) for w, ci in d1.items()}
    ko = {w: (back.get(ci) in det) for w, ci in outer_key_columns(pre).items()}
    return {"rows": int(acc.shape[0]), "rank": int(rank), "free": int(nfree),
            "monomials": int(len(pre.mons)),
            "pinned_by_normalisation": int(len(known)),
            "determined_monomials": int(len(det)),
            "degree_one_key_columns_determined": sum(1 for v in ok.values() if v),
            "degree_one_detail": ok,
            "outer_key_columns_determined": sum(1 for v in ko.values() if v),
            "outer_key_detail": ko}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-2^4")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--combine", default=None,
                    help="O10 pattern, e.g. 1101 (cheap set {1,2} -> dim K = 4)")
    ap.add_argument("--structures", type=int, default=2000)
    ap.add_argument("--checkpoints", default=None,
                    help="comma list; default: only the final count")
    ap.add_argument("--moments", choices=("uniform", "realizable"),
                    default="realizable")
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = get_cipher(a.instance, rounds=a.rounds)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    fkey = f"2^{F.n}"
    ycomb = (block_coeffs("dux", fkey, "dec", a.combine, list(BC.CHEAP_SET))
             if a.combine else None)
    rng = np.random.default_rng(a.seed)
    gen = ((lambda: BC.uniform_moments_b(pre, rng)) if a.moments == "uniform"
           else (lambda: BC.realizable_moments_b(pre, rng, a.points)))
    nrows = len(ycomb) if ycomb else 2 * len(pre.outer)
    print(f"{a.instance} ({F.name}): extended M_b = {len(pre.mons)} "
          f"(paper M = {BC.column_count(F.q - 1, pre.U, pre.S)['M']}), "
          f"{a.moments} pseudo-structures, {nrows} rows each"
          + (f", combine {a.combine} (dim K = {len(ycomb)})" if ycomb else ""),
          flush=True)

    cps = sorted({int(v) for v in a.checkpoints.split(",")}) if a.checkpoints \
        else [a.structures]
    acc, _ = build_rows(pre, lrow, gen, max(cps), ycomb, a.progress)
    curve = []
    for n in cps:
        t0 = time.time()
        r = determinacy(pre, F, acc[:nrows * n])
        r["structures"] = n
        r["seconds"] = round(time.time() - t0, 1)
        curve.append(r)
        print(f"  N = {n:6d}  rows = {r['rows']:6d}  rank = {r['rank']:6d}  "
              f"{r['determined_monomials']} monomials pinned, "
              f"{r['degree_one_key_columns_determined']}/16 inner key words, "
              f"{r['outer_key_columns_determined']}/{4 * len(pre.outer_coords)} "
              f"outer key words  ({r['seconds']} s)", flush=True)
    res = {"instance": a.instance, "field": F.name, "rounds": a.rounds,
           "cheap_set": list(BC.CHEAP_SET), "combine": a.combine,
           "dim_K": len(ycomb) if ycomb else None,
           "rows_per_structure": nrows, "moments": a.moments,
           "points": a.points if a.moments == "realizable" else None,
           "seed": a.seed, "monomials": len(pre.mons),
           "counts": BC.column_count(F.q - 1, pre.U, pre.S),
           "curve": curve,
           "saturated_rank": max(r["rank"] for r in curve)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
