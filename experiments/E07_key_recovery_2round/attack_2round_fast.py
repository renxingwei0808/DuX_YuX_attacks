"""E07 / S3 -- r_KR = 2 structured linearisation, matrix-product assembly.

Same attack as attack_2round_toy.py (see that file for the derivation of the
equation and the monomial set), with two changes that make the multi-block
case reachable:

  * the per-structure assembly uses assemble_fast.py (matrix products instead
    of the |joint| x |joint| double loop), and
  * structures are generated and assembled in parallel processes.

The point of this script is the question we flagged as the main risk of the
toy prototype (attack_2round_toy.py): with more than one
unknown inner block, is the rank of the linearised system still enough to pin
down the degree-1 monomials (the inner round-key words)?

    python attack_2round_fast.py --instance toy-257 --unknown-blocks 0     --structures 100
    python attack_2round_fast.py --instance toy-257 --unknown-blocks 0,1   --structures 800 --procs 24
    python attack_2round_fast.py --instance toy-257 --unknown-blocks 0,1,2 --structures 4000 --procs 40

Note on normalisation: when ALL FOUR inner blocks are unknown every equation is
homogeneous (the constant monomial has coefficient 0), so the solution is only
determined up to a scalar.  `--normalise` pins the monomials whose inner
exponent vector is all-zero to their true value 1, which removes exactly that
freedom; it is a no-op when at least one inner block is known.

W16 additions (O10 + O12), see weighted.py:
  --weights N       use N weights a per structure instead of one; the weighted
                    equation sum_x x^a (...) = 0 holds for every |a| below the
                    margin T - D of the layer-l words the equation uses, so ONE
                    structure yields N x (equations per structure) rows.
  --weight-dims k   weight over the first k active words (|a| < margin).
  --combine PAT     fold the four per-outer-block rows into the dim K cheap
                    combined rows of O10 (`0001` for DuX, `1110` for YuX).
Both are only available when all four inner blocks are unknown (the moment
assembly `rows_from_moments` needs that anyway).

    python attack_2round_fast.py --instance toy-257 --layers 5 --pos 3 \
        --unknown-blocks 0,1,2,3 --structures 55 --weights 47 --normalise
    python attack_2round_fast.py --instance toy-257 --layers 6 --active 3,7 \
        --unknown-blocks 0,1,2,3 --structures 1 --weight-dims 2 --combine 0001 \
        --normalise
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
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from dux import DuX                              # noqa: E402
from dux.registry import cipher_family, get_cipher  # noqa: E402
from dux.linear import t_xor                     # noqa: E402
from attack_2round_toy import linear_row, structure_data, active_words  # noqa: E402
from assemble_fast import Precomp, structure_rows  # noqa: E402
from assemble_fast_2n import structure_rows_2n  # noqa: E402
import weighted as WT  # noqa: E402
import interp_mask  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "E15_mixed_coset"))
import mixed as MX  # noqa: E402
import modp_solve  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "fast"))
import gf2n_solve  # noqa: E402

_G = {}


def _init(instance, rounds, alpha_unknown, outer, pos, seed, active,
          assume_l1=False, limit=None, ks_literal=False):
    c = get_cipher(instance, rounds=rounds, ks_literal=ks_literal)
    rng = np.random.default_rng(seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    # O2: L1 = Rot_{-4} o L0, so the equation set is the same under either
    # choice and the attack must not need to know t(rk^1).  The default is
    # therefore the fixed L0 of the equivalent cipher; --assume-L1 restores
    # the old behaviour (reading two key bits) purely as a control.
    _G.update(c=c, rks=rks, rounds=rounds, pos=pos, active=active,
              pre=Precomp(c.F, c.alpha, alpha_unknown, outer,
                          cipher=cipher_family(instance)),
              lrow=linear_row(c, t_xor(rks[1])
                              if (assume_l1 and cipher_family(instance) == "dux")
                              else 0),
              seed=seed, limit=limit)


def _one(st):
    c, rks, pre = _G["c"], _G["rks"], _G["pre"]
    P = structure_data(c, rks, _G["rounds"], _G["pos"], _G["seed"] + 5000 + st,
                       _G["active"], limit=_G.get("limit"))
    asm = structure_rows_2n if c.F.char == 2 else structure_rows
    # Return a numpy array, not lists of Python ints: at the full-scale size
    # (4 rows x 38 051 columns per structure, 3 500 structures) the list form
    # costs ~15 GB in the parent process, the array form 4.3 GB.
    return np.asarray(asm(pre, P, rks[0], _G["lrow"]), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-257")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--pos", type=int, default=3, help="active ciphertext word (s = 1 alias)")
    ap.add_argument("--active", default=None,
                    help="comma list of active ciphertext words, each over all of F_q "
                         "(q^s points per structure); overrides --pos")
    ap.add_argument("--unknown-blocks", default="0")
    ap.add_argument("--outer-blocks", default="0,1,2,3")
    ap.add_argument("--structures", type=int, default=100)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--normalise", action="store_true")
    ap.add_argument("--ks-literal", action="store_true",
                    help="YuX only: the SLIDING-window reading of Algorithm 1 "
                         "line 2 instead of the non-overlapping Sect. VI-D "
                         "one (default).  rk^0 = the master key either way, so "
                         "this is a control, not a different attack.")
    ap.add_argument("--assume-L1", action="store_true",
                    help="control run: pick L0/L1 from t(rk^1) instead of always L0")
    ap.add_argument("--limit-points", type=int, default=None,
                    help="use only the first L points of each structure "
                         "(throughput calibration only -- the equations do NOT "
                         "hold on a truncated structure)")
    ap.add_argument("--csolver", default=None,
                    help="path to the C F_2^n eliminator (default: fast/gf2nsolve "
                         "if it has been built; 'none' forces the numpy one)")
    ap.add_argument("--block", type=int, default=256,
                    help="panel width of the C eliminator")
    ap.add_argument("--progress", type=int, default=0,
                    help="print elimination progress every N pivots")
    ap.add_argument("--weights", type=int, default=None,
                    help="O12: number of weights a per structure (default: all "
                         "admissible, i.e. the whole margin)")
    ap.add_argument("--weight-dims", type=int, default=1,
                    help="O12: weight over the first k active words (|a| < margin)")
    ap.add_argument("--weight-word", type=int, default=None,
                    help="O12: the ciphertext word carrying the weight; it must "
                         "be the FIRST active word (the slowest structure axis)")
    ap.add_argument("--combine", default=None,
                    help="O10: fold the per-outer-block rows with the cheap-row "
                         "kernel of this balanced pattern, e.g. 0001")
    ap.add_argument("--weighted", action="store_true",
                    help="use the weighted assembly even with a single weight")
    ap.add_argument("--point-set", type=int, default=None,
                    help="O15: let each active ciphertext word run over a RANDOM "
                         "set of this many points instead of all of F_q.  The "
                         "sums are masked with the divided-difference weights, "
                         "so the threshold becomes T' = sum_i (n_i - 1); the "
                         "point sets are drawn from --point-seed.")
    ap.add_argument("--point-seed", type=int, default=None,
                    help="seed of the O15 point sets (default: --seed)")
    ap.add_argument("--cheap-set", default="2",
                    help="T1 (R9), DuX / decryption / characteristic 2 only: "
                         "'1,2' uses the EXTENDED combined row of "
                         "experiments/E13_b_coordinate (the cubic coordinate b "
                         "is Frobenius-bilinear there), which turns the pattern "
                         "`1101` from dim K = 1 (collapsed, W19-B) into dim "
                         "K = 4.  The default '2' is the paper's row, unchanged.")
    ap.add_argument("--mixed", default=None,
                    help="T2 (R9), F_p only: the STRUCTURE TYPE of every active "
                         "word as a comma list, 'k' for a coset of the order-2^k "
                         "subgroup and 'full' for all of F_p, e.g. '6,full'.  The "
                         "coset words must come first (they carry the weights, "
                         "and the weight axis is the slowest one).  No mask: the "
                         "threshold is T = sum_i T_i of Theorem 1, not O15's.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rounds = a.layers + 2
    unknown = [int(v) for v in a.unknown_blocks.split(",")] if a.unknown_blocks else []
    outer = [int(v) for v in a.outer_blocks.split(",")]

    c = get_cipher(a.instance, rounds=rounds, ks_literal=a.ks_literal)
    F = c.F
    fam = cipher_family(a.instance)
    cheap_set = tuple(int(v) for v in a.cheap_set.split(","))
    if cheap_set == (1, 2):
        sys.path.insert(0, os.path.join(HERE, "..", "E13_b_coordinate"))
        import bcoord as BC
        assert a.combine in WT.X1X1_CLASSES, (
            "the cheap set {1,2} needs a pattern with positions 1 and 3 "
            f"balanced, i.e. one of {list(WT.X1X1_CLASSES)}")
        pre = BC.PrecompB(F, c.alpha, unknown, outer)
    else:
        assert cheap_set == (2,), "the only cheap sets are {2} and T1's {1,2}"
        pre = Precomp(F, c.alpha, unknown, outer, cipher=fam)
    rng = np.random.default_rng(a.seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    M = len(pre.mons)
    act = active_words(a.pos, a.active)
    lrow = linear_row(c, t_xor(rks[1]) if (a.assume_L1 and fam == "dux") else 0)
    if a.point_set:
        npts = a.point_set ** len(act)
    elif a.mixed:
        npts = 1
        for k in MX.parse_mixed(a.mixed, len(act)):
            npts *= F.q if k is None else (1 << k)
    else:
        npts = F.q ** len(act) if a.limit_points is None else a.limit_points
    print(f"{a.instance} ({F.name}): r = {rounds} ({a.layers}-layer distinguisher + 2), "
          f"unknown inner blocks {unknown}, outer blocks {outer}, active words {act}")
    print(f"  monomials M = {M} = 2^{np.log2(M):.1f}; "
          f"{a.structures} structures x {npts} chosen ciphertexts "
          f"= 2^{np.log2(a.structures * npts):.1f} data; {len(outer)} eqs/structure")

    t0 = time.time()
    # T2 (R9): a mixed structure MUST go through the weighted path -- its rows
    # are only valid for weights with a_i >= 1 on every coset axis (the plain
    # a = 0 sum sees the smaller `threshold_plain`), and the point sets only
    # reach the assembly there.
    weighted = bool(a.weights or a.combine or a.weighted or a.weight_dims > 1
                    or a.mixed or a.point_set)
    if weighted:
        assert sorted(unknown) == [0, 1, 2, 3], \
            "the weighted / combined assembly needs all four inner blocks unknown"
        assert not a.assume_L1, "O10's kernel vectors are computed for the fixed L0"
        if a.weight_word is not None:
            assert a.weight_word == act[0], \
                "the weight word must be the first active word (slowest axis)"
        pts = mask = None
        ks = None
        if a.point_set:
            pts = WT.point_sets(F, len(act), a.point_set,
                                a.seed if a.point_seed is None else a.point_seed)
            mask = [interp_mask.divided_difference_weights(F, u) for u in pts]
        elif a.mixed:
            assert F.char != 2, "T2's mixed structures are an F_p construction"
            ks = MX.parse_mixed(a.mixed, len(act))
            pts = MX.mixed_axes(F, ks, a.seed if a.point_seed is None
                                else a.point_seed)
        pl = WT.plan(F, act, a.layers, "dec", fam, a.combine,
                     a.weight_dims, a.weights, coset=ks,
                     pointset=(None if (pts is None or ks is not None)
                               else [len(u) for u in pts]),
                     cheap_set=cheap_set)
        ws, margin, crit = pl.weights, pl.margin, pl.crit
        ycomb = (WT.combine_vectors(fam, F, "dec", a.combine, cheap=list(cheap_set))
                 if a.combine else None)
        neq = len(ycomb) if ycomb else len(outer)
        print(f"  O12 weights: dims {a.weight_dims}, margin {margin} "
              f"(layer {a.layers} pattern {crit['patterns'][a.layers - 1]}), "
              f"{len(ws)} weights/structure"
              + (f"; O10 combine {a.combine} (cheap set {list(cheap_set)}): "
                 f"dim K = {len(ycomb)}, y = {ycomb}"
                 if ycomb else "")
              + f" -> {len(ws) * neq} equations/structure", flush=True)
        print(f"  O12 weight rule: {pl.weight_rule} "
              f"(usable_weights = {pl.usable_weights})", flush=True)
        rows = []
        for st in range(a.structures):
            if cheap_set == (1, 2):
                assert pts is None, "the T1 rows have no point-set variant yet"
                rr, _order, npts_real = BC.structure_rows_b(
                    pre, c, rks, rounds, a.seed + 5000 + st, a.active, ws,
                    linear_row(c, 0), ycomb)
            else:
                rr, _order, npts_real = WT.weighted_rows(
                    pre, c, rks, rounds, a.pos, a.seed + 5000 + st, a.active, ws,
                    lrow=linear_row(c, 0), ycomb=ycomb, points=pts, mask=mask)
            rows.append(rr)
            print(f"  assembled {st + 1}/{a.structures} structures "
                  f"({rr.shape[0]} rows, {time.time() - t0:.1f}s)", flush=True)
        R = np.concatenate(rows, axis=0)
    else:
        neq = len(outer)
        R = np.empty((a.structures * neq, M), dtype=np.int64)
        with Pool(a.procs, initializer=_init,
                  initargs=(a.instance, rounds, unknown, outer, a.pos, a.seed,
                            a.active, a.assume_L1, a.limit_points,
                            a.ks_literal)) as pool:
            for i, rr in enumerate(pool.imap_unordered(_one, range(a.structures), chunksize=1)):
                R[i * neq:(i + 1) * neq] = rr
                if (i + 1) % 100 == 0 or i + 1 == a.structures:
                    print(f"  assembled {i + 1}/{a.structures} structures "
                          f"({time.time() - t0:.1f}s)", flush=True)
    t_asm = time.time() - t0

    # known columns: the constant, plus (if requested) every monomial whose
    # inner exponents are all zero -- their true value is 1.
    zero = tuple([0, 0, 0, 0])
    known_cols = {pre.col_const: 1}
    if a.normalise:
        for m, i in pre.idx.items():
            tag, inner = m
            if tag is None and inner and all(e == zero for _b, e in inner):
                known_cols[i] = 1
    kc = sorted(known_cols)
    keep = [i for i in range(M) if i not in known_cols]
    kv = np.array([known_cols[i] for i in kc], dtype=np.int64)
    if F.char == 2:
        # -x = x; the pinned columns are XORed into the right-hand side
        B = np.zeros(R.shape[0], dtype=np.int64)
        for t, ci in enumerate(kc):
            B ^= F.vmul(R[:, ci], np.int64(kv[t]))
    else:
        B = (-(R[:, kc] @ kv)) % F.p
    R = R[:, keep]
    cols = [pre.mons[i] for i in keep]
    print(f"  system {R.shape[0]} x {R.shape[1]} over {F.name} "
          f"({len(kc)} columns pinned)")

    t1 = time.time()
    solver = "numpy"
    if F.char == 2:
        cbin = None if a.csolver == "none" else (a.csolver or None)
        if a.csolver != "none" and gf2n_solve.available(cbin):
            solver = "c-omp"
            det, rank, nfree = gf2n_solve.solve(F, R, B, binary=cbin, block=a.block,
                                                progress=a.progress)
        else:
            det, rank, nfree = modp_solve.solve_gf2n(F, R, B, progress=a.progress)
    else:
        det, rank, nfree = modp_solve.solve(F.p, R, B, progress=a.progress)
    t_solve = time.time() - t1
    print(f"  rank {rank} / {R.shape[1]} unknowns ({nfree} free), "
          f"{len(det)} monomials pinned down  ({t_solve:.1f}s)")

    cpos = {m: i for i, m in enumerate(cols)}
    ok, truth = {}, {}
    for b in unknown:
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            m = (None, ((b, e),))
            truth[f"{b},{i}"] = rks[0][4 * b + i]
            if m in cpos and cpos[m] in det:
                ok[f"{b},{i}"] = det[cpos[m]]
    good = sum(1 for k in truth if ok.get(k) == truth[k])
    print(f"  inner key words recovered correctly: {good}/{len(truth)}")
    if good != len(truth):
        print(f"    got   {ok}")
        print(f"    truth {truth}")
    res = {"instance": a.instance, "field": F.name, "rounds": rounds, "layers": a.layers,
           "unknown_blocks": unknown, "outer_blocks": outer,
           "active_words": act, "points_per_structure": npts, "monomials": M,
           "weights": ({"n": len(ws), "dims": a.weight_dims, "margin": margin,
                        "weight_word": act[0], "range": f"|a| < {margin}"}
                       if weighted else None),
           "equation_rows": ({"combine": a.combine, "dim_K": len(ycomb),
                              "y": [({f"{b},{c}": int(v) for (b, c), v in y.items()}
                                     if isinstance(y, dict) else list(map(int, y)))
                                    for y in ycomb],
                             "cheap_set": list(cheap_set)}
                             if weighted and ycomb else
                             {"combine": None, "rows": "one per outer block"}),
           "structures": a.structures, "equations": R.shape[0],
           "unknowns": R.shape[1], "rank": rank, "free": nfree,
           "pinned": len(det), "inner_correct": good, "inner_total": len(truth),
           "normalise": a.normalise, "assume_L1": a.assume_L1,
           "data_log2": round(float(np.log2(a.structures * npts)), 2),
           "assemble_s": round(t_asm, 1), "solve_s": round(t_solve, 1),
           "procs": a.procs, "seed": a.seed, "solver": solver}
    print(f"  data 2^{res['data_log2']}; assemble {t_asm:.1f}s, solve {t_solve:.1f}s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"{a.instance}_u{''.join(map(str, unknown))}_N{a.structures}"
        fn = os.path.join(a.out, f"fast_{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
