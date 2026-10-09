"""E10 / W14 -- r_KR = 2 in the ENCRYPTION direction (CPA): peel two rounds.

E10 section 5 peels ONE round off a CPA distinguisher and recovers the master key
of 7-round DuX.  Peeling two rounds needs the same structured linearisation as
E07, with the two S-boxes swapped:

    Z_l = state after the l-th S layer;  Z_{l+1} = S(L(Z_l) + rk^l);  C = Z_r + rk^r.

With an l = r - 2 layer full zero-sum (every one of the 16 words of Z_{r-2} is
balanced over the structure) and kappa = -rk^r, kappa' = -L^{-1}(rk^{r-1}):

    W   = S^{-1}(C + kappa)                     (inner, per block: 4 unknown kappa words)
    X   = L^{-1}(W) + kappa'                    (L^{-1} = the 5-rotation sum, both fields)
    sum_P [S^{-1}(X)]_{4j+3} = sum_P [L(Z_{r-2})]_{4j+3} = 0

and the coordinate-3 map of S^{-1} is QUADRATIC, f = x0 x1 + x3 + alpha, so

    sum_{s,t} A_s B_t Q_{s,t} + sum_s V_s T_s
        + kappa'_{4j+1} sum_s A_s T_s + kappa'_{4j+0} sum_s B_s T_s  =  0        (*)

with T_s = sum_P W_s, Q_{s,t} = sum_P W_s W_t and A/B/V the L^{-1} rows of block
positions 0/1/3.  (alpha and kappa'_{4j+3} carry the factor |structure| = 0.)
Coordinate 0, g = x1 x2 + x0 + alpha, gives a second equation per block from the
same moments -- quad (1, 2), linear 0, keys kappa'_{4j+2} / kappa'_{4j+1}.

This is E07's equation with (quad, linear) = ((0,1), 3) instead of ((0,3), 2)
and with the inner expansion of S^{-1} instead of S.  The inner S-box being the
CHEAP one is the whole point: S^{-1} has coordinate degrees (2,3,4,2) against
S's (5,3,2,8), so the key-exponent set shrinks from |U| = 75 / 50 to 19, and the
linearisation monomial count for four unknown inner blocks from

    38 051 (F_p) / 18 025 (F_{2^n})   down to   3 063 (F_p) / 3 031 (F_{2^n}),

i.e. N_min drops by an order of magnitude.

W16 additions (O10 + O12): `--weights N` turns one structure into N weighted
copies of the same equation (the weighted zero-sum sum_x x^a (...) = 0 holds
for every |a| below the margin of the layer-l words the equation uses), and
`--combine PAT` folds the per-(block, coordinate) rows into the dim K cheap
combined rows of O10 for the CPA direction (cheap columns {0, 3} for DuX,
{0, 1} for YuX).

Usage
  python kr2_cpa.py --instance toy-257 --layers 3 --active 1 --structures 500
  python kr2_cpa.py --instance dux-2^16 --layers 6 --active 1 --structures 400 --count-only
  python kr2_cpa.py --instance toy-257 --layers 3 --active 1 --structures 2 \
      --weights 128 --combine 0001
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
sys.path.insert(0, os.path.join(HERE, "..", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(HERE, "..", "E07_key_recovery_2round", "fast"))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))

from dux import DuX                                            # noqa: E402
from dux.params import ROT_INV, WORDS                          # noqa: E402
from dux.keyschedule import Rf_inv, round_constants            # noqa: E402
from keypoly import KeyPoly                                    # noqa: E402
from assemble_fast import (Moments, Precomp, _scatter,          # noqa: E402
                           MomentLayout, _pow_matrix, combine_rows,
                           weighted_moment_stream)
import weighted as WT                                          # noqa: E402
from assemble_fast_2n import _scatter2, gmm, gmv               # noqa: E402
import modp_solve                                              # noqa: E402
import gf2n_solve                                              # noqa: E402
from rank_phi import realizable_moments                        # noqa: E402


# outer coordinate -> (quadratic pair, linear position); the key word paired
# with the first member of the pair is the second, and vice versa.
OUTER = {3: ((0, 1), 3), 0: ((1, 2), 0)}


def joint_expansion_inv(F, alpha):
    """S^{-1}(C + kappa): per coordinate, [(kappa-exponents, C-exponents, coeff)].

    Same shape as attack_2round_toy.joint_expansion, but for the decryption
    S-box and WITHOUT dropping the pure-key terms (they are what O5 kills)."""
    x = [KeyPoly.var_plus_const(F, i, 0, 8) + KeyPoly.var_plus_const(F, 4 + i, 0, 8)
         for i in range(4)]
    f = (x[0] * x[1] + x[3]).add_const(alpha)
    g = (x[1] * x[2] + x[0]).add_const(alpha)
    y1 = (x[2] * f + x[1]).add_const(alpha)
    y2 = (f * g + x[2]).add_const(alpha)
    return [[(e[:4], e[4:], co) for e, co in t.d.items()] for t in (g, y1, y2, f)]


def linv_row(t=0):
    """L^{-1}(W)_i = sum_l W_{(i+l) mod 16}, l in ROT_INV -- a 0/1 row in both
    characteristics (the inverse layer is a plain sum of five rotations)."""
    return tuple(1 if l in ROT_INV[t] else 0 for l in range(WORDS))


def make_precomp(c, outer_blocks, coords):
    coords = sorted(coords)
    kwords = sorted({i for co in coords for i in OUTER[co][0]})
    return Precomp(c.F, c.alpha, [0, 1, 2, 3], outer_blocks,
                   joint=joint_expansion_inv(c.F, c.alpha), outer_coords=kwords)


def structure_moments(c, rks, rounds, active, seed, pre, chunk_log2=20):
    """Encrypt one chosen-plaintext structure and accumulate the ciphertext
    moments.  The active plaintext words run over all of F_q, the other 16 - s
    are fixed random constants."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    total = F.q ** len(active)
    csize = min(total, 1 << chunk_log2)
    mom = Moments(pre)
    for start in range(0, total, csize):
        idx = np.arange(start, start + min(csize, total - start), dtype=np.int64)
        P = [np.full(len(idx), consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            P[w] = (idx // (F.q ** t)) % F.q
        mom.add(c.encrypt(tuple(P), rks, rounds=rounds, vec=True))
    return mom


def structure_weighted(c, rks, rounds, active, seed, pre, weights):
    """Yield (a, Moments) for one chosen-plaintext structure and every weight.

    The moments are over the CIPHERTEXT words (the inner S^{-1} sees C + kappa)
    while the weight is carried by the first active PLAINTEXT word, which is
    the slowest axis of the structure -- the O12 slice decomposition applies
    verbatim (memo Sect. 5.4)."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    s = len(active)
    total = F.q ** s
    idx = np.arange(total, dtype=np.int64)
    # NOTE the axis order: `structure_moments` above uses idx // q^t for the
    # t-th active word, i.e. the FIRST active word is the FASTEST axis.  The
    # slice decomposition needs the weight word to be the slowest, so the
    # weighted path builds the product set the other way round (axis j has
    # stride q^(s-1-j)), which is the same point SET.
    P = [np.full(total, consts[i], dtype=np.int64) for i in range(16)]
    xvals = []
    for t, w in enumerate(active):
        col = (idx // (F.q ** (s - 1 - t))) % F.q
        P[w] = col
        xvals.append(col)
    C = c.encrypt(tuple(P), rks, rounds=rounds, vec=True)
    PW = {b: _pow_matrix(F, C, b, pre.pe_arr) for b in pre.unknown}
    dims = len(weights[0])
    layout = MomentLayout(pre)
    return weighted_moment_stream(pre, PW, xvals[:dims], weights, layout)


def rows_cpa(pre, mom, lrow, coords):
    """One row of (*) per outer block and per outer coordinate."""
    F, p = pre.F, pre.p
    pm, Mom = mom.pm, mom.Mom
    rows = []
    for j in pre.outer:
        for co in sorted(coords):
            (i1, i2), i3 = OUTER[co]
            sel = lambda off: np.array(                      # noqa: E731
                [lrow[(s - 4 * j - off) % WORDS] for s in range(WORDS)],
                dtype=np.int64)
            A, B, V = sel(i1), sel(i2), sel(i3)
            row = np.zeros(len(pre.mons), dtype=np.int64)
            if F.char == 2:
                def comb(co_, b):
                    acc = np.zeros(pre.C.shape[1:], dtype=np.int64)
                    for t in range(4):
                        if co_[4 * b + t]:
                            acc ^= F.vmul(pre.C[t], np.int64(co_[4 * b + t]))
                    return acc
                CA = {b: comb(A, b) for b in pre.unknown}
                CB = {b: comb(B, b) for b in pre.unknown}
                CV = {b: comb(V, b) for b in pre.unknown}
                for b in pre.unknown:
                    _scatter2(row, pre.col_single[b], gmv(F, CV[b], pm[b]))
                for bs in pre.unknown:
                    for bt in pre.unknown:
                        G = gmm(F, gmm(F, CA[bs], Mom[(bs, bt)]), CB[bt].T)
                        _scatter2(row, pre.col_pair[(bs, bt)], G)
                for b in pre.unknown:
                    _scatter2(row, pre.col_kT[(j, i2, b)], gmv(F, CA[b], pm[b]))
                    _scatter2(row, pre.col_kT[(j, i1, b)], gmv(F, CB[b], pm[b]))
            else:
                CA = {b: np.tensordot(A[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p
                      for b in pre.unknown}
                CB = {b: np.tensordot(B[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p
                      for b in pre.unknown}
                CV = {b: np.tensordot(V[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p
                      for b in pre.unknown}
                for b in pre.unknown:
                    _scatter(row, pre.col_single[b], (CV[b] @ pm[b]) % p, p)
                for bs in pre.unknown:
                    for bt in pre.unknown:
                        G = (CA[bs] @ Mom[(bs, bt)]) % p
                        G = (G @ CB[bt].T) % p
                        _scatter(row, pre.col_pair[(bs, bt)], G, p)
                for b in pre.unknown:
                    _scatter(row, pre.col_kT[(j, i2, b)], (CA[b] @ pm[b]) % p, p)
                    _scatter(row, pre.col_kT[(j, i1, b)], (CB[b] @ pm[b]) % p, p)
                row %= p
            rows.append(row)
    return rows


def _master_from_kappa(c, rounds, rks, kappa, seed):
    """kappa = -rk^r  ->  master key by inverting the key schedule; then an
    independent check on a fresh plaintext/ciphertext pair."""
    F = c.F
    kk = [None] * 16
    for key, v in kappa.items():
        b, i = (int(t) for t in key.split(","))
        kk[4 * b + i] = int(v)
    if any(v is None for v in kk):
        return False, False
    rk_last = [v if F.char == 2 else (-v) % F.p for v in kk]
    rcs = round_constants(F, c.alpha, rounds)
    stx = list(rk_last)
    for i in range(rounds, 0, -1):
        rc = rcs[i - 1]
        for j in range(3, -1, -1):
            stx = list(Rf_inv(F, c.alpha, stx, rc[4 * j:4 * j + 4]))
    master = tuple(stx)
    rng = np.random.default_rng(seed + 99991)
    Pv = tuple(int(v) for v in rng.integers(0, F.q, size=16))
    ok = (tuple(c.encrypt(Pv, c.key_schedule(master), rounds=rounds))
          == tuple(c.encrypt(Pv, rks, rounds=rounds)))
    return tuple(rk_last) == tuple(rks[rounds]), bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-257")
    ap.add_argument("--layers", type=int, default=3,
                    help="layers of the CPA zero-sum distinguisher (r = layers + 2)")
    ap.add_argument("--active", default="1", help="active plaintext words")
    ap.add_argument("--outer-blocks", default="0,1,2,3")
    ap.add_argument("--coords", default="0,3", help="outer S^-1 coordinates (0 and/or 3)")
    ap.add_argument("--structures", type=int, default=500)
    ap.add_argument("--checkpoints", default="",
                    help="also report the rank after these structure counts")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--pseudo", action="store_true",
                    help="feed W13-A realizable pseudo-moments instead of encrypting "
                         "a structure: measures rank(Phi|V_0), the template bound on "
                         "rank_max, for instances whose structures are out of reach")
    ap.add_argument("--pseudo-points", type=int, default=256)
    ap.add_argument("--weights", type=int, default=None,
                    help="O12: weights a per structure (default: the whole margin)")
    ap.add_argument("--weight-dims", type=int, default=1)
    ap.add_argument("--combine", default=None,
                    help="O10 (CPA direction, cheap columns {0,3}): fold the "
                         "per-(block, coordinate) rows with this pattern's kernel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rounds = a.layers + 2
    active = [int(v) for v in a.active.split(",")]
    outer = [int(v) for v in a.outer_blocks.split(",")]
    coords = [int(v) for v in a.coords.split(",")]
    c = DuX(a.instance, rounds=rounds)
    F = c.F
    pre = make_precomp(c, outer, coords)
    M = len(pre.mons)
    neq = len(outer) * len(coords)
    npts = F.q ** len(active)
    print(f"{a.instance} ({F.name}): r = {rounds} = {a.layers}-layer CPA zero-sum + 2, "
          f"active plaintext words {active} ({npts} = 2^{np.log2(npts):.0f} points)")
    print(f"  inner S^-1 expansion: |U| = {len(pre.U)}, |S| = {len(pre.S)}; "
          f"monomials M = {M}; outer coords {sorted(coords)} -> {neq} equations/structure; "
          f"N_min >= ceil(M/{neq}) = {-(-M // neq)}")
    if a.count_only:
        return

    rks = c.key_schedule(c.random_key(np.random.default_rng(a.seed)))
    lrow = linv_row(0)                       # O2: fix L0
    cps = sorted({int(v) for v in a.checkpoints.split(",") if v}) + [a.structures]

    t0 = time.time()
    weighted = bool(a.weights or a.combine or a.weight_dims > 1)
    ws = margin = ycomb = None
    if weighted:
        assert not a.pseudo, "pseudo-moments carry no structure to weight"
        pl = WT.plan(F, active, a.layers, "enc", "dux", a.combine,
                     a.weight_dims, a.weights)
        ws, margin, crit = pl.weights, pl.margin, pl.crit
        assert a.weights is None or a.weights <= pl.usable_weights, (
            f"--weights {a.weights} exceeds the {pl.usable_weights} weights O12 "
            f"admits here ({pl.weight_rule})")
        if a.combine:
            from cheap_rows import block_coeffs
            ycomb = block_coeffs("dux", str(WT.field_key(F)), "enc", a.combine)
            assert sorted(coords) == [0, 3], \
                "the O10 CPA combination needs both cheap coordinates (0 and 3)"
            neq = len(ycomb)
        else:
            neq = len(outer) * len(coords)
        print(f"  O12 weights: dims {a.weight_dims}, margin {margin} "
              f"(layer {a.layers} pattern {crit['patterns'][a.layers - 1]}), "
              f"{len(ws)} weights/structure"
              + (f"; O10 combine {a.combine}: dim K = {len(ycomb)}" if ycomb else "")
              + f" -> {len(ws) * neq} equations/structure", flush=True)
        print(f"  O12 weight rule: {pl.weight_rule} "
              f"(usable_weights = {pl.usable_weights})", flush=True)
    R = np.empty((a.structures * neq * (len(ws) if weighted else 1), M),
                 dtype=np.int64)
    prng = np.random.default_rng(a.seed + 1)
    row_at = 0
    for s in range(a.structures):
        if weighted:
            for _a, mom in structure_weighted(c, rks, rounds, active,
                                              a.seed + 5000 + s, pre, ws):
                rr = rows_cpa(pre, mom, lrow, coords)
                if ycomb is None:
                    R[row_at:row_at + neq] = np.asarray(rr, dtype=np.int64)
                    row_at += neq
                else:
                    # rows_cpa emits (outer block j, coordinate co) in that order
                    order = [(j, co) for j in pre.outer for co in sorted(coords)]
                    for y in ycomb:
                        R[row_at] = combine_rows(rr, [y[(j, co)] for j, co in order], F)
                        row_at += 1
        else:
            mom = (realizable_moments(pre, prng, a.pseudo_points) if a.pseudo
                   else structure_moments(c, rks, rounds, active, a.seed + 5000 + s, pre))
            R[row_at:row_at + neq] = np.asarray(rows_cpa(pre, mom, lrow, coords),
                                                dtype=np.int64)
            row_at += neq
        if (s + 1) % 100 == 0 or weighted:
            print(f"  assembled {s+1}/{a.structures} ({row_at} rows, "
                  f"{time.time()-t0:.1f}s)", flush=True)
    t_asm = time.time() - t0
    per_struct = neq * (len(ws) if weighted else 1)

    zero = (0, 0, 0, 0)
    known = {pre.col_const: 1}
    for m, i in pre.idx.items():
        tag, inner = m
        if tag is None and inner and all(e == zero for _b, e in inner):
            known[i] = 1
    kc = sorted(known)
    keep = [i for i in range(M) if i not in known]
    kv = np.array([known[i] for i in kc], dtype=np.int64)
    cols = [pre.mons[i] for i in keep]
    cpos = {m: i for i, m in enumerate(cols)}

    curve, res_last = [], None
    for n in cps:
        Rn = R[:n * per_struct][:, keep]
        if F.char == 2:
            Bn = np.zeros(Rn.shape[0], dtype=np.int64)
            for t, ci in enumerate(kc):
                Bn ^= F.vmul(R[:n * per_struct, ci], np.int64(kv[t]))
            if gf2n_solve.available():
                det, rank, nfree = gf2n_solve.solve(F, Rn, Bn)
            else:
                det, rank, nfree = modp_solve.solve_gf2n(F, Rn, Bn)
        else:
            Bn = (-(R[:n * per_struct][:, kc] @ kv)) % F.p
            det, rank, nfree = modp_solve.solve(F.p, Rn, Bn)
        ok, truth = {}, {}
        for b in range(4):
            for i in range(4):
                e = tuple(1 if t == i else 0 for t in range(4))
                m = (None, ((b, e),))
                kr = int(rks[rounds][4 * b + i])
                truth[f"{b},{i}"] = kr if F.char == 2 else (-kr) % F.p
                if m in cpos and cpos[m] in det:
                    ok[f"{b},{i}"] = det[cpos[m]]
        good = sum(1 for k in truth if ok.get(k) == truth[k])
        if a.pseudo:
            good = -1                       # pseudo-moments carry no key
        master_ok = pair_ok = False
        if good == 16 and not a.pseudo:
            master_ok, pair_ok = _master_from_kappa(c, rounds, rks, ok, a.seed)
        print(f"  N = {n:5d}: {Rn.shape[0]:5d} eqs, rank {rank}/{Rn.shape[1]} "
              f"({nfree} free) -> kappa words {good}/16"
              + (f"; master key == truth: {master_ok}; known-pair check: {pair_ok}"
                 if good == 16 else ""), flush=True)
        curve.append({"structures": n, "equations": int(Rn.shape[0]), "rank": int(rank),
                      "free": int(nfree), "kappa_correct": good,
                      "master_key_ok": bool(master_ok), "known_pair_ok": bool(pair_ok),
                      "data_log2": round(float(np.log2(n * npts)), 2)})
        res_last = (good, rank, nfree, master_ok, pair_ok)

    good, rank, nfree, master_ok, pair_ok = res_last
    res = {"instance": a.instance, "field": F.name, "rounds": rounds,
           "layers": a.layers, "active_words": active, "points_per_structure": npts,
           "coords": sorted(coords), "outer_blocks": outer, "monomials": M,
           "unknowns": len(keep), "equations_per_structure": per_struct,
           "weights": ({"n": len(ws), "dims": a.weight_dims, "margin": margin,
                        "weight_word": active[0], "range": f"|a| < {margin}"}
                       if weighted else None),
           "equation_rows": ({"combine": a.combine, "dim_K": len(ycomb)}
                             if ycomb else
                             {"combine": None, "rows": "one per (outer block, coord)"}),
           "structures": a.structures, "pseudo": bool(a.pseudo),
           "rank": int(rank), "free": int(nfree),
           "kappa_correct": good, "master_key_ok": bool(master_ok),
           "known_pair_ok": bool(pair_ok), "curve": curve, "seed": a.seed,
           "data_log2": round(float(np.log2(a.structures * npts)), 2),
           "assemble_s": round(t_asm, 1)}
    print(f"  data 2^{res['data_log2']} chosen plaintexts; assemble {t_asm:.1f} s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"{a.instance}_r{rounds}_a{'_'.join(map(str, active))}"
        fn = os.path.join(a.out, f"kr2cpa_{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
