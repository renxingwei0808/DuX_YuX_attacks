"""E07 / S4 -- how many structures does the r_KR = 2 system actually need?

`attack_2round_fast.py` answers "does it work" for one structure count.  The
12-round complexity is N_min x (points per structure), so the number that
matters is the SMALLEST N for which the system pins down all 16 words of rk^0.
Re-running the whole attack per N would re-assemble the same rows every time,
so this script assembles the largest N once, keeps the rows in memory, and
solves nested prefixes of them.

    python nmin_scan.py --instance dux-65537 --layers 9 --unknown-blocks 0,1,2,3 \
        --structures 3500 --grid 1800,2000,2200,2400,2600,2800,3000,3500 \
        --normalise --procs 24 --seed 2026 --out results/E07_key_recovery_2round/s4

Every prefix uses the SAME key and the same structures, so the rank column is a
monotone curve in N and the "rank saturation point" is well defined.  Different
--seed = different master key (and different structure constants).

W16: with `--weights` the same scan runs over WEIGHTS (O12) instead of, or in
addition to, structures.  `--weight-grid` solves nested prefixes of the weight
list with all structures present, which is the rank-vs-weight curve S9/S10 need
in order to size N_w; `--combine PAT` folds the four per-outer-block rows into
the dim K cheap rows of O10 first.

    python nmin_scan.py --instance toy-257 --layers 5 --pos 3 \
        --unknown-blocks 0,1,2,3 --structures 55 --weights 47 --normalise \
        --weight-grid 1,2,4,8,16,32,47 --grid 10,20,30,40,55
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from dux import DuX                                     # noqa: E402
from dux.registry import cipher_family, get_cipher  # noqa: E402
from attack_2round_toy import active_words, linear_row  # noqa: E402
from assemble_fast import Precomp                       # noqa: E402
import attack_2round_fast as A                          # noqa: E402
import weighted as WT                                   # noqa: E402
import modp_solve                                       # noqa: E402
sys.path.insert(0, os.path.join(HERE, "fast"))
import gf2n_solve                                       # noqa: E402


def pinned_columns(pre, normalise):
    """The columns whose value is known a priori: the constant monomial and,
    when the system is homogeneous (all four inner blocks unknown), every
    monomial with all-zero inner exponents."""
    zero = (0, 0, 0, 0)
    known = {pre.col_const: 1}
    if normalise:
        for m, i in pre.idx.items():
            tag, inner = m
            if tag is None and inner and all(e == zero for _b, e in inner):
                known[i] = 1
    return known


def solve_prefix(F, R, keep, kc, kv, N, neq, progress=None):
    """Solve the system formed by the first N structures (N * neq rows)."""
    sub = R[:N * neq]
    if F.char == 2:
        B = np.zeros(sub.shape[0], dtype=np.int64)
        for t, ci in enumerate(kc):
            B ^= F.vmul(sub[:, ci], np.int64(kv[t]))
        if gf2n_solve.available():
            return gf2n_solve.solve(F, sub[:, keep], B, progress=progress)
        return modp_solve.solve_gf2n(F, sub[:, keep], B, progress=progress)
    B = (-(sub[:, kc] @ kv)) % F.p
    return modp_solve.solve(F.p, sub[:, keep], B, progress=progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--pos", type=int, default=3)
    ap.add_argument("--active", default=None)
    ap.add_argument("--unknown-blocks", default="0,1,2,3")
    ap.add_argument("--outer-blocks", default="0,1,2,3")
    ap.add_argument("--structures", type=int, default=3500)
    ap.add_argument("--grid", default="1800,2000,2200,2400,2600,2800,3000,3500")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--normalise", action="store_true")
    ap.add_argument("--assume-L1", action="store_true")
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--weights", type=int, default=None,
                    help="O12: weights per structure (default: the whole margin)")
    ap.add_argument("--weight-dims", type=int, default=1)
    ap.add_argument("--combine", default=None,
                    help="O10: cheap-row pattern, e.g. 0001")
    ap.add_argument("--weight-grid", default=None,
                    help="solve nested prefixes of the WEIGHT list (all "
                         "structures present) and print the rank-weight curve")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rounds = a.layers + 2
    unknown = [int(v) for v in a.unknown_blocks.split(",")] if a.unknown_blocks else []
    outer = [int(v) for v in a.outer_blocks.split(",")]
    grid = sorted({int(v) for v in a.grid.split(",")} | {a.structures})
    assert grid[-1] <= a.structures

    c = get_cipher(a.instance, rounds=rounds)
    F = c.F
    fam = cipher_family(a.instance)
    pre = Precomp(F, c.alpha, unknown, outer, cipher=fam)
    rng = np.random.default_rng(a.seed)
    rks = c.key_schedule(c.random_key(rng))
    act = active_words(a.pos, a.active)
    npts = F.q ** len(act)
    M, neq = len(pre.mons), len(outer)
    print(f"{a.instance} ({F.name}): r = {rounds}, unknown {unknown}, active {act}, "
          f"M = {M}, {neq} eqs/structure, {npts} points/structure, seed {a.seed}, "
          f"L = {'t(rk^1)' if a.assume_L1 else 'L0 (fixed)'}", flush=True)

    t0 = time.time()
    weighted = bool(a.weights or a.combine or a.weight_dims > 1 or a.weight_grid)
    ws, margin, ycomb = None, None, None
    if weighted:
        assert sorted(unknown) == [0, 1, 2, 3], \
            "the weighted / combined assembly needs all four inner blocks unknown"
        pl = WT.plan(F, act, a.layers, "dec", fam, a.combine,
                     a.weight_dims, a.weights)
        ws, margin, crit = pl.weights, pl.margin, pl.crit
        ycomb = WT.combine_vectors(fam, F, "dec", a.combine) if a.combine else None
        neq = len(ycomb) if ycomb else len(outer)
        print(f"  O12 weights: dims {a.weight_dims}, margin {margin}, "
              f"{len(ws)} weights/structure"
              + (f"; O10 combine {a.combine}: dim K = {len(ycomb)}" if ycomb else "")
              + f" -> {len(ws) * neq} rows/structure", flush=True)
        print(f"  O12 weight rule: {pl.weight_rule} "
              f"(usable_weights = {pl.usable_weights})", flush=True)
        # row order: structure-major, then weight, then equation
        rows = np.empty((a.structures * len(ws) * neq, M), dtype=np.int64)
        lrow0 = linear_row(c, 0)
        for st in range(a.structures):
            rr, _o, _n = WT.weighted_rows(pre, c, rks, rounds, a.pos,
                                          a.seed + 5000 + st, a.active, ws,
                                          lrow0, ycomb=ycomb)
            rows[st * len(ws) * neq:(st + 1) * len(ws) * neq] = rr
            print(f"  assembled {st + 1}/{a.structures} ({time.time() - t0:.1f}s)",
                  flush=True)
    else:
        rows = np.empty((a.structures * neq, M), dtype=np.int64)
        with Pool(a.procs, initializer=A._init,
                  initargs=(a.instance, rounds, unknown, outer, a.pos, a.seed,
                            a.active, a.assume_L1, None)) as pool:
            for i, rr in enumerate(pool.imap(A._one, range(a.structures), chunksize=1)):
                rows[i * neq:(i + 1) * neq] = np.asarray(rr, dtype=np.int64)
                if (i + 1) % 250 == 0 or i + 1 == a.structures:
                    print(f"  assembled {i + 1}/{a.structures} ({time.time() - t0:.1f}s)",
                          flush=True)
    t_asm = time.time() - t0

    known = pinned_columns(pre, a.normalise)
    kc = sorted(known)
    kv = np.array([known[i] for i in kc], dtype=np.int64)
    keep = [i for i in range(M) if i not in known]
    cols = [pre.mons[i] for i in keep]
    # Drop the pinned columns ONCE (they are the same for every prefix): at the
    # full-scale row count the per-prefix fancy index would double the peak
    # memory of an already 3 GB matrix.
    if F.char == 2:
        Ball = np.zeros(rows.shape[0], dtype=np.int64)
        for t, ci in enumerate(kc):
            Ball ^= F.vmul(rows[:, ci], np.int64(kv[t]))
    else:
        Ball = (-(rows[:, kc] @ kv)) % F.p
    rows = np.delete(rows, kc, axis=1)
    print(f"  system {rows.shape[0]} x {rows.shape[1]} over {F.name} "
          f"({len(kc)} columns pinned, {rows.nbytes / 2**30:.2f} GiB)", flush=True)
    cpos = {m: i for i, m in enumerate(cols)}
    truth, want = {}, {}
    for b in unknown:
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            truth[f"{b},{i}"] = int(rks[0][4 * b + i])
            want[f"{b},{i}"] = cpos.get((None, ((b, e),)))

    def solve_rows(sub, B, tag):
        t1 = time.time()
        if F.char == 2:
            if gf2n_solve.available():
                det, rank, nfree = gf2n_solve.solve(F, sub, B, progress=a.progress)
            else:
                det, rank, nfree = modp_solve.solve_gf2n(F, sub, B,
                                                         progress=a.progress)
        else:
            det, rank, nfree = modp_solve.solve(F.p, sub, B, progress=a.progress)
        good = sum(1 for k, ci in want.items()
                   if ci is not None and det.get(ci) == truth[k])
        return {"rank": rank, "free": nfree, "pinned": len(det),
                "inner_correct": good, "inner_total": len(truth),
                "equations": int(sub.shape[0]), "unknowns": len(keep),
                "solve_s": round(time.time() - t1, 1)}, tag

    wscan = []
    if a.weight_grid:
        wgrid = sorted({int(v) for v in a.weight_grid.split(",")} | {len(ws)})
        per = len(ws) * neq
        for W in wgrid:
            if W > len(ws):
                continue
            keepr = np.concatenate([np.arange(st * per, st * per + W * neq)
                                    for st in range(a.structures)])
            rec, _ = solve_rows(rows[keepr], Ball[keepr], W)
            rec["weights"] = W
            rec["structures"] = a.structures
            rec["data_log2"] = round(float(np.log2(a.structures) + np.log2(npts)), 2)
            wscan.append(rec)
            print(f"  weights={W:5d}  rows {rec['equations']:6d}  rank {rec['rank']:6d}"
                  f"  pinned {rec['pinned']:6d}  words {rec['inner_correct']}/"
                  f"{rec['inner_total']}  ({rec['solve_s']:.1f}s)", flush=True)

    scan = []
    for N in grid:
        per = len(ws) * neq if weighted else neq
        sub, Bn = rows[:N * per], Ball[:N * per]
        rec, _ = solve_rows(sub, Bn, N)
        rec["N"] = N
        rec["data_log2"] = round(float(np.log2(N) + np.log2(npts)), 2)
        rank, nfree, good, dt = rec["rank"], rec["free"], rec["inner_correct"], rec["solve_s"]
        scan.append(rec)
        print(f"  N={N:5d}  rank {rank:6d}  free {nfree:6d}  pinned {rec['pinned']:6d}  "
              f"words {good}/{len(truth)}  data 2^{rec['data_log2']}  ({dt:.1f}s)",
              flush=True)

    ok = [r["N"] for r in scan if r["inner_correct"] == r["inner_total"]]
    rmax = max(r["rank"] for r in scan)
    sat = min([r["N"] for r in scan if r["rank"] == rmax], default=None)
    res = {"instance": a.instance, "field": F.name, "rounds": rounds,
           "layers": a.layers, "unknown_blocks": unknown, "outer_blocks": outer,
           "active_words": act, "points_per_structure": npts, "monomials": M,
           "weights": ({"n": len(ws), "dims": a.weight_dims, "margin": margin,
                        "weight_word": act[0], "range": f"|a| < {margin}"}
                       if weighted else None),
           "equation_rows": ({"combine": a.combine, "dim_K": len(ycomb),
                              "y": [list(map(int, y)) for y in ycomb]}
                             if weighted and ycomb else
                             {"combine": None, "rows": "one per outer block"}),
           "weight_scan": wscan,
           "normalise": a.normalise, "assume_L1": a.assume_L1, "seed": a.seed,
           "assembled": a.structures, "assemble_s": round(t_asm, 1),
           "procs": a.procs, "scan": scan,
           "N_min": min(ok) if ok else None, "rank_max": rmax,
           "rank_saturation_N": sat,
           "N_min_rule_rank_over_neq": -(-rmax // neq)}
    print(f"  => N_min = {res['N_min']}, rank_max = {rmax} at N >= {sat}, "
          f"rule rank/neq = {res['N_min_rule_rank_over_neq']}", flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"nmin_{a.instance}_u{''.join(map(str, unknown))}_s{a.seed}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
