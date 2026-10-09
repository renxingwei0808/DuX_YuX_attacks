"""E07 / S4 step 3 -- cost per point of the r_KR = 2 assembly.

The 12-round attack needs N_min structures of q^3 = 2^48 chosen ciphertexts
each.  That cannot be run, but its cost can be measured: every quantity the
assembly extracts from a structure is a point sum (assemble_fast.Moments), so
the cost is strictly linear in the number of points and the constant can be
measured on truncated structures.

This script times the four phases separately on structures truncated to
`--points` points (`--limit-points` of attack_2round_fast), checks that the
streamed assembly reproduces the in-memory one on a full small structure, and
prints core-seconds per point and per (point x monomial pair).

    python throughput.py --instance dux-65537 --active 3,7 --points 2**18,2**20,2**22
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from dux import DuX                                             # noqa: E402
from attack_2round_toy import (structure_data, structure_stream,  # noqa: E402
                               linear_row, active_words)
from assemble_fast import (Precomp, Moments, structure_rows,     # noqa: E402
                           rows_from_moments, _pow_matrix, moment_matrix,
                           exact_chunk)


def check_streaming_equals_direct(instance, layers, active, unknown, outer, seed):
    """The streamed assembly must give exactly the rows of structure_rows."""
    rounds = layers + 2
    c = DuX(instance, rounds=rounds)
    rng = np.random.default_rng(seed)
    rks = c.key_schedule(c.random_key(rng))
    pre = Precomp(c.F, c.alpha, unknown, outer)
    lrow = linear_row(c, 0)
    P = structure_data(c, rks, rounds, 3, seed + 5000, active)
    direct = structure_rows(pre, P, rks[0], lrow)
    mom = Moments(pre)
    for chunkP in structure_stream(c, rks, rounds, 3, seed + 5000, active,
                                   chunk=max(1, len(P[0]) // 3 + 1)):
        mom.add(chunkP)
    streamed = rows_from_moments(pre, mom, lrow)
    for a, b in zip(direct, streamed):
        assert np.array_equal(a, b), "streamed assembly differs from structure_rows"
    return len(direct), len(P[0])


def time_phases(c, rks, rounds, pre, lrow, active, npoints, chunk, seed,
                assemble_rows=True):
    F = c.F
    t_dec = t_pw = t_mom = 0.0
    mom = Moments(pre)
    p = pre.p
    ec = exact_chunk(p)
    t0 = time.perf_counter()
    for P in structure_stream(c, rks, rounds, 3, seed, active, chunk=chunk,
                              limit=npoints):
        t1 = time.perf_counter()
        t_dec += t1 - t0
        PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
        t2 = time.perf_counter()
        t_pw += t2 - t1
        PWf = {b: PW[b].astype(np.float64) for b in pre.unknown}
        for b in pre.unknown:
            mom.pm[b] = (mom.pm[b] + PW[b].sum(axis=1)) % p
        for A in pre.unknown:
            for B in pre.unknown:
                if B < A:
                    continue
                m = moment_matrix(PWf, A, B, p, ec)
                mom.Mom[(A, B)] = (mom.Mom[(A, B)] + m) % p
                if A != B:
                    mom.Mom[(B, A)] = (mom.Mom[(B, A)] + m.T) % p
        mom.npoints += len(P[0])
        t3 = time.perf_counter()
        t_mom += t3 - t2
        t0 = t3
    # The equations can only be ASSEMBLED on a complete structure: the
    # monomial set of attack_2round_toy.monomial_set drops the monomials whose
    # coefficient vanishes by a full-structure identity, and _scatter refuses
    # to silently discard a nonzero coefficient.  On a truncated structure the
    # three data-dependent phases above are still exactly the ones that would
    # run, and the row assembly itself costs the same for every structure size
    # (it only sees the nP x nP moment matrices), so it is timed separately on
    # a genuine full structure.
    t_rows, nrows = 0.0, 0
    if assemble_rows:
        t4 = time.perf_counter()
        rows = rows_from_moments(pre, mom, lrow)
        t_rows = time.perf_counter() - t4
        nrows = len(rows)
    return {"decrypt_s": t_dec, "powmatrix_s": t_pw, "moments_s": t_mom,
            "rows_s": t_rows, "total_s": t_dec + t_pw + t_mom + t_rows,
            "points": mom.npoints, "rows": nrows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--active", default="3,7,11")
    ap.add_argument("--unknown-blocks", default="0,1,2,3")
    ap.add_argument("--outer-blocks", default="0,1,2,3")
    ap.add_argument("--points", default="2**18,2**20,2**22")
    ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--target-log2", type=float, default=48.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    unknown = [int(v) for v in a.unknown_blocks.split(",")]
    outer = [int(v) for v in a.outer_blocks.split(",")]
    active = a.active
    rounds = a.layers + 2

    neq, npts = check_streaming_equals_direct("toy-257", 5, "3,7", unknown, outer, 11)
    print(f"streaming check: OK ({neq} rows on a {npts}-point toy-257 structure)")

    c = DuX(a.instance, rounds=rounds)
    rng = np.random.default_rng(a.seed)
    rks = c.key_schedule(c.random_key(rng))
    pre = Precomp(c.F, c.alpha, unknown, outer)
    lrow = linear_row(c, 0)
    nP = pre.pe_arr.shape[0]
    npairs = len(unknown) ** 2 * nP * nP
    print(f"{a.instance}: r = {rounds}, active {active_words(3, active)}, "
          f"unknown {unknown}, nP = {nP}, monomial pairs per point = {npairs}")

    # cost of the (point-count independent) row assembly, on a real structure
    full = time_phases(c, rks, rounds, pre, lrow, "3", None, a.chunk,
                       a.seed + 5000, assemble_rows=True)
    print(f"  row assembly on a full single-word structure "
          f"({full['points']} points): {full['rows_s']:.2f}s for {full['rows']} rows")

    runs = []
    for expr in a.points.split(","):
        n = int(eval(expr, {"__builtins__": {}}))
        r = time_phases(c, rks, rounds, pre, lrow, active, n,
                        min(a.chunk, n), a.seed + 5000, assemble_rows=False)
        r["rows_s"] = full["rows_s"]
        r["total_s"] += full["rows_s"]
        r["points_log2"] = round(float(np.log2(r["points"])), 2)
        for k in ("decrypt", "powmatrix", "moments"):
            r[f"{k}_ns_per_point"] = round(r[f"{k}_s"] / r["points"] * 1e9, 1)
        r["total_ns_per_point"] = round(r["total_s"] / r["points"] * 1e9, 1)
        runs.append(r)
        print(f"  2^{r['points_log2']:<5} points: decrypt {r['decrypt_s']:8.2f}s  "
              f"pow {r['powmatrix_s']:8.2f}s  moments {r['moments_s']:8.2f}s  "
              f"rows {r['rows_s']:.2f}s  => {r['total_ns_per_point']:.1f} ns/point",
              flush=True)

    big = runs[-1]
    per_point = big["total_s"] / big["points"]
    target = 2.0 ** a.target_log2
    est = per_point * target
    print(f"\nlinear extrapolation to 2^{a.target_log2:g} points per structure:")
    print(f"  {per_point * 1e9:.1f} ns/point  =>  {est:.3e} core-seconds per structure "
          f"= 2^{np.log2(est):.1f} core-s")
    print(f"  decrypt share {big['decrypt_s'] / big['total_s'] * 100:.0f} %, "
          f"power table {big['powmatrix_s'] / big['total_s'] * 100:.0f} %, "
          f"moments {big['moments_s'] / big['total_s'] * 100:.0f} %")
    res = {"instance": a.instance, "rounds": rounds, "active": active,
           "row_assembly_s": round(full["rows_s"], 3),
           "unknown_blocks": unknown, "nP": nP, "pairs_per_point": npairs,
           "runs": runs, "ns_per_point": round(per_point * 1e9, 2),
           "target_points_log2": a.target_log2,
           "core_seconds_per_structure": est,
           "core_seconds_per_structure_log2": round(float(np.log2(est)), 2)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
