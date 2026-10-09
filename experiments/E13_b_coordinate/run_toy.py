"""E13 / W27 step 4 -- the toy end-to-end of T1: r_KR = 2 key recovery from the
EXTENDED (cheap set {1,2}) combined rows, in characteristic 2.

The cell: `tools/zero_sum_criterion.py --q 2^4 --active 3` puts toy-2^4's
layer 3 at `1101` -- positions 0, 1 and 3 balanced, position 2 not -- with
margins (7, 4, 0, 8).  The combined row of `1101` with the cheap set {1,2}
reads positions 1 and 3, so the weight range is min(4, 8) = 4 and dim K = 4:
16 rows per 16-point structure.  With the paper's cheap set {2} the same cell
has dim K = 1 and the row collapses (W19-B), i.e. nothing at all.

Success = all sixteen inner key words right AND the re-encryption check: rk^0
is DuX's master key, so the recovered words are re-run through the key schedule
and a fresh random plaintext is encrypted with them and compared.

Usage
  python run_toy.py --instance toy-2^4 --layers 3 --active 3 --combine 1101 \
      --structures 800 --seeds 2026,7 --out ../../results/E13_b_coordinate
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
from assemble_fast import _pow_matrix                            # noqa: E402
from attack_2round_toy import (active_words, linear_row,         # noqa: E402
                               structure_data)
from cheap_rows import block_coeffs                              # noqa: E402
from nmin_scan import pinned_columns                             # noqa: E402
import weighted as WT                                            # noqa: E402
import gf2n_solve                                                # noqa: E402
import modp_solve                                                # noqa: E402


structure_rows_b = BC.structure_rows_b


def run_key(instance, layers, active, combine, nstruct, seed, progress=0,
            weight_subset="all"):
    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rng = np.random.default_rng(seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    act = active_words(3, active)
    pl = WT.plan(F, act, layers, "dec", "dux", combine, dims=1,
                 cheap_set=(1, 2))
    ws = pl.weights
    # the memo's "risk" control: at an EVEN weight sum_x x^{2a} Y^2 =
    # (sum_x x^a Y)^2, so those rows are Frobenius images of others and may
    # carry no new rank.  `--weight-subset odd` drops the even ones.
    if weight_subset == "odd":
        ws = [w for w in ws if int(w[0]) % 2 == 1]
    elif weight_subset == "even":
        ws = [w for w in ws if int(w[0]) % 2 == 0]
    assert ws, f"no weight left in the subset {weight_subset}"
    ycoeff = block_coeffs("dux", f"2^{F.n}", "dec", combine, list(BC.CHEAP_SET))
    print(f"  seed {seed}: M_b = {len(pre.mons)}, dim K = {len(ycoeff)}, "
          f"{len(ws)} weights ({pl.weight_rule}) -> "
          f"{len(ws) * len(ycoeff)} rows/structure", flush=True)
    # store the stack as uint16: the extended system has 47 749 columns, so
    # 1 500 structures x 16 rows is 9.2 GB at int64 and 2.3 GB here, and the
    # solver takes uint16 anyway (`rank_b.py` does the same).  Safe because
    # this driver is characteristic-2 only and n <= 16.
    assert F.char == 2 and F.q <= 1 << 16, "the T1 rows are characteristic 2"
    per = len(ws) * len(ycoeff)
    R = np.zeros((per * nstruct, len(pre.mons)), dtype=np.uint16)
    t0 = time.time()
    for st in range(nstruct):
        rr, _o, npts = structure_rows_b(pre, c, rks, rounds, seed + 5000 + st,
                                        active, ws, lrow, ycoeff)
        assert rr.shape == (per, len(pre.mons)), rr.shape
        R[per * st:per * (st + 1)] = rr
        if progress and (st + 1) % progress == 0:
            print(f"    assembled {st + 1}/{nstruct} structures "
                  f"({time.time() - t0:.0f} s)", flush=True)
    t_asm = time.time() - t0

    known = pinned_columns(pre, normalise=True)
    kc = np.array(sorted(known), dtype=np.int64)
    kv = np.array([known[int(i)] for i in kc], dtype=np.int64)
    B = np.zeros(R.shape[0], dtype=np.int64)
    for t, ci in enumerate(kc):
        B ^= F.vmul(R[:, ci].astype(np.int64), np.int64(kv[t]))
    keep = [i for i in range(len(pre.mons)) if i not in known]
    sub = np.ascontiguousarray(R[:, keep])
    del R
    t1 = time.time()
    det, rank, nfree = (gf2n_solve.solve(F, sub, B) if gf2n_solve.available()
                        else modp_solve.solve_gf2n(F, sub, B))
    t_solve = time.time() - t1
    back = {keep[j]: j for j in range(len(keep))}
    got, truth = {}, {}
    for b in range(4):
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            ci = pre.idx[(None, ((b, e),))]
            truth[f"{b},{i}"] = int(rks[0][4 * b + i])
            if back.get(ci) in det:
                got[f"{b},{i}"] = int(det[back[ci]])
    good = sum(1 for k in truth if got.get(k) == truth[k])
    # re-encryption check: rk^0 IS DuX's master key, so re-derive the schedule
    recheck = None
    if good == 16:
        Krec = [got[f"{b},{i}"] for b in range(4) for i in range(4)]
        rks2 = c.key_schedule(tuple(Krec))
        pt = np.random.default_rng(seed + 99).integers(0, F.q, size=(16, 8))
        X = tuple(pt[i].astype(np.int64) for i in range(16))
        ct1 = c.encrypt(X, rks, rounds=rounds, vec=True)
        ct2 = c.encrypt(X, rks2, rounds=rounds, vec=True)
        recheck = all(np.array_equal(ct1[i], ct2[i]) for i in range(16))
    npts_struct = F.q ** len(act)
    return {"seed": seed, "instance": instance, "layers": layers,
            "active": act, "combine": combine, "cheap_set": list(BC.CHEAP_SET),
            "dim_K": len(ycoeff), "weights": len(ws),
            "weight_subset": weight_subset, "weight_list": [list(w) for w in ws],
            "weight_rule": pl.weight_rule, "margin": pl.margin,
            "structures": nstruct, "points_per_structure": npts_struct,
            "rows": int(sub.shape[0]), "monomials": len(pre.mons),
            "unknowns": int(sub.shape[1]), "rank": int(rank), "free": int(nfree),
            "pinned_monomials": len(det),
            "inner_correct": good, "inner_total": 16,
            "reencryption_ok": recheck,
            "log2_data": round(float(np.log2(nstruct * npts_struct)), 2),
            "assemble_s": round(t_asm, 1), "solve_s": round(t_solve, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-2^4")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--active", default="3")
    ap.add_argument("--combine", default="1101")
    ap.add_argument("--structures", type=int, default=800)
    ap.add_argument("--seeds", default="2026,7")
    ap.add_argument("--progress", type=int, default=100)
    ap.add_argument("--weight-subset", default="all",
                    choices=("all", "odd", "even"),
                    help="the memo's risk control: even weights give rows that "
                         "are Frobenius images of others (sum x^{2a} Y^2 = "
                         "(sum x^a Y)^2), so 'odd' isolates the ones that "
                         "cannot be")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(v) for v in a.seeds.split(",")]
    print(f"{a.instance}: layer {a.layers} pattern {a.combine}, cheap set "
          f"{list(BC.CHEAP_SET)} (T1), {a.structures} structures, seeds {seeds}")
    res = []
    for seed in seeds:
        r = run_key(a.instance, a.layers, a.active, a.combine, a.structures,
                    seed, a.progress, a.weight_subset)
        res.append(r)
        print(f"  seed {seed}: rank {r['rank']}/{r['unknowns']}, "
              f"{r['pinned_monomials']} monomials pinned, "
              f"{r['inner_correct']}/16 key words, re-encryption "
              f"{r['reencryption_ok']}, data 2^{r['log2_data']} "
              f"({r['assemble_s']} s + {r['solve_s']} s)", flush=True)
    out = {"runs": res,
           "all_keys_16_of_16": all(r["inner_correct"] == 16 for r in res),
           "all_reencryption_ok": all(r["reencryption_ok"] for r in res)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = "" if a.weight_subset == "all" else f"_{a.weight_subset}"
        fn = os.path.join(a.out,
                          f"toy_{a.instance}_{a.combine}_cheap12{tag}.json")
        json.dump(out, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
