"""W8 (E08) -- Boolean degree of the plaintext-side key-recovery rounds.

Question (docs/glossary.md O6): Ni-Wang-Li (DCC 2026, 94:123) Sect. 4.3
take the Boolean degree of the r_KR = 2 equation system to be d = 6, obtained
as log2 of the univariate F_{2^n}-degree 64.  Is that right?

We answer it exactly, with two independent methods:

  A. Moebius transform (`moebius.py`) of the真 truth table -- only reachable on
     toy-2^4, where one S-box block is 16 key bits (and 24 bits for the
     two-layer restriction).
  B. Exact multivariate key polynomial (`keypoly.py`): the S-box is a short
     chain of quadratic steps, so S(P+k) is an explicit polynomial in the four
     key words with small exponents; the Boolean degree is
     max sum_i HW(e_i) over its support.  This works for ANY n and for the
     full 16-word inner round key, which the Moebius transform cannot reach.

Measured (all of them n-independent for n in {4,8,16}):

  (1) one layer, no sum:      S(P+k) coordinate degrees      (3,2,2,4)   [DuX]
                                                             (4,3,2,2)   [Yu2X]
  (2) one layer, summed over an even structure (O5)          (2,1,1,3)   [DuX]
  (3) two layers, cheapest outer coordinate, ALL 16 inner key words
      symbolic:                                              d = 7       [both]
      (restricted to a single inner block it is only 4 -- the degree comes
       from products of S-box outputs of DIFFERENT inner blocks, where the
       exponents cannot wrap and the Hamming weights add)

Conclusion: d = 7, not 6; the degree histograms are in
results/E08_boolean_degree_extension/boolean_degrees.json.

Usage: python experiments/E08_boolean_degree_extension/run.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dux import DuX                                          # noqa: E402
from dux.field import make_field                             # noqa: E402
from dux.params import ROT_FWD_BIN, WORDS                    # noqa: E402
from dux.sbox import vS                                      # noqa: E402
from moebius import moebius, anf_degree, degree_histogram    # noqa: E402
from keypoly import (KeyPoly, sbox_enc_chain,                # noqa: E402
                     yu2x_sbox_enc_chain, YU2X_ROT_FWD)

NB, NW = 4, 4        # toy-2^4: bits per word, words per block


# ---------------------------------------------------------------- method A --
def key_index_arrays(nwords=NW, nbits=NB):
    idx = np.arange(1 << (nwords * nbits), dtype=np.int64)
    mask = (1 << nbits) - 1
    return [(idx >> (nbits * i)) & mask for i in range(nwords)]


def moebius_degrees(vals, nbits=NB):
    degs = []
    for c in range(len(vals)):
        d = []
        for b in range(nbits):
            d.append(anf_degree(moebius(((vals[c] >> b) & 1).astype(np.uint8))))
        degs.append(max(d))
    return degs


def moebius_one_layer(c, rng, trials):
    F, alpha = c.F, c.alpha
    K = key_index_arrays()
    single, summed = [], []
    for _ in range(trials):
        P = [int(v) for v in rng.integers(0, F.q, size=NW)]
        y = vS(F, tuple(F.vadd(K[i], P[i]) for i in range(NW)), alpha)
        single.append(moebius_degrees(y))
    for _ in range(trials):
        # structure of even size; a random even set (NOT a full-field structure
        # on one word -- that one is itself a zero-sum and makes everything 0)
        Pset = [[int(v) for v in rng.integers(0, F.q, size=NW)] for _ in range(6)]
        acc = [np.zeros(1 << (NW * NB), dtype=np.int64) for _ in range(4)]
        for P in Pset:
            y = vS(F, tuple(F.vadd(K[i], P[i]) for i in range(NW)), alpha)
            for i in range(4):
                acc[i] = F.vadd(acc[i], y[i])
        summed.append(moebius_degrees(acc))
    return single, summed


def moebius_two_layer_one_block(c, rng, nP=16, j=0):
    """The coordinate-2 equation with only inner block 0 and (k'_0,k'_3)
    symbolic: 16 + 8 = 24 Boolean variables, exact ANF."""
    F, alpha = c.F, c.alpha
    rots = ROT_FWD_BIN[0]
    K = key_index_arrays()
    N = 1 << (NW * NB)
    rk0 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    Pset = [[int(v) for v in rng.integers(0, F.q, size=WORDS)] for _ in range(nP)]
    S0 = np.zeros(N, dtype=np.int64)
    S2 = np.zeros(N, dtype=np.int64)
    S3 = np.zeros(N, dtype=np.int64)
    S03 = np.zeros(N, dtype=np.int64)
    for P in Pset:
        words = [None] * WORDS
        yb0 = vS(F, tuple(F.vadd(K[i], P[i]) for i in range(NW)), alpha)
        for i in range(NW):
            words[i] = yb0[i]
        for b in range(1, 4):
            xb = tuple(np.full(1, F.add(P[4 * b + i], rk0[4 * b + i]), dtype=np.int64)
                       for i in range(NW))
            yb = vS(F, xb, alpha)
            for i in range(NW):
                words[4 * b + i] = np.full(N, int(yb[i][0]), dtype=np.int64)
        u = []
        for i in range(NW):
            acc = np.zeros(N, dtype=np.int64)
            for s in rots:
                acc = F.vadd(acc, words[(4 * j + i + s) % WORDS])
            u.append(acc)
        S0 = F.vadd(S0, u[0]); S2 = F.vadd(S2, u[2]); S3 = F.vadd(S3, u[3])
        S03 = F.vadd(S03, F.vmul(u[0], u[3]))
    big = np.empty(N << (2 * NB), dtype=np.int64)
    for kp0 in range(1 << NB):
        t0 = F.vmul(S3, np.full(N, kp0, dtype=np.int64))
        for kp3 in range(1 << NB):
            t3 = F.vmul(S0, np.full(N, kp3, dtype=np.int64))
            big[((kp0 << 16) | (kp3 << 20)):((kp0 << 16) | (kp3 << 20)) + N] = \
                F.vadd(F.vadd(S03, t0), F.vadd(t3, S2))
    degs = [anf_degree(moebius(((big >> b) & 1).astype(np.uint8))) for b in range(NB)]
    return max(degs), degs


# ---------------------------------------------------------------- method B --
def keypoly_one_layer(F, alpha, chain, rng, nP=6):
    """(1) single S(P+k) and (2) sum over an even structure, exact, any n."""
    P = [int(v) for v in rng.integers(0, F.q, size=4)]
    y = chain([KeyPoly.var_plus_const(F, i, P[i]) for i in range(4)], alpha)
    single = [t.boolean_degree() for t in y]
    acc = [KeyPoly(F) for _ in range(4)]
    for _ in range(nP):
        Pp = [int(v) for v in rng.integers(0, F.q, size=4)]
        yy = chain([KeyPoly.var_plus_const(F, i, Pp[i]) for i in range(4)], alpha)
        for i in range(4):
            acc[i] = acc[i] + yy[i]
    return single, [t.boolean_degree() for t in acc]


def keypoly_two_layer(F, alpha, rots, chain, pair, lin_idx, nP=4, seed=11, blocks=4):
    """Boolean degree of the cheapest outer coordinate's zero-sum equation,
    with `blocks` inner S-box blocks symbolic (blocks = 4 -> the whole inner
    round key).  `pair` = the two u-indices that are multiplied,
    `lin_idx` = the u-indices that appear alone."""
    NV = 16
    rng = np.random.default_rng(seed)
    Ps = [[int(v) for v in rng.integers(0, F.q, size=WORDS)] for _ in range(nP)]
    rk0 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    j = 0
    S = [KeyPoly(F, None, NV) for _ in range(4)]
    prod = KeyPoly(F, None, NV)
    for P in Ps:
        words = []
        for b in range(4):
            if b < blocks:
                xb = [KeyPoly.var_plus_const(F, 4 * b + i, P[4 * b + i], NV)
                      for i in range(4)]
                words += list(chain(xb, alpha))
            else:
                xb = [KeyPoly.const(F, F.add(P[4 * b + i], rk0[4 * b + i]), NV)
                      for i in range(4)]
                words += list(chain(xb, alpha))
        u = []
        for i in range(4):
            acc = KeyPoly(F, None, NV)
            for s in rots:
                acc = acc + words[(4 * j + i + s) % WORDS]
            u.append(acc)
        for i in range(4):
            S[i] = S[i] + u[i]
        prod = prod + u[pair[0]] * u[pair[1]]
    degs = [t.boolean_degree() for t in S]
    total = max([prod.boolean_degree()] + [1 + degs[i] for i in pair]
                + [degs[i] for i in lin_idx])
    return {"deg_sum_product": prod.boolean_degree(), "deg_sum_u": degs,
            "total": total, "support_product": prod.support_size()}


# ------------------------------------------------------------------- main ---
def binom_le(n, d):
    from math import comb
    return sum(comb(n, i) for i in range(d + 1))


def reestimate(m, n_v, d, n_eq_log2):
    """Ni et al. Sect. 4.3:  T_c = 2^m * C(n_v,<=d)/n_eq ,  T_s = C(n_v,<=d)^w."""
    import math
    M = binom_le(n_v, d)
    lM = math.log2(M)
    return {"n_v": n_v, "d": d, "log2_monomials": round(lM, 1),
            "log2_Tc": round(m + lM - n_eq_log2, 1),
            "log2_Ts_bitops": round(2.37 * lM, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--skip-moebius", action="store_true")
    ap.add_argument("--out", default="results/E08_boolean_degree_extension")
    a = ap.parse_args()
    t0 = time.time()
    res = {"seed": a.seed}

    print("=== A. Moebius transform on toy-2^4 (exact ANF, ground truth) ===")
    if not a.skip_moebius:
        c = DuX("toy-2^4")
        rng = np.random.default_rng(a.seed)
        single, summed = moebius_one_layer(c, rng, a.trials)
        print(f"(1) S(P+k) coordinate degrees, {a.trials} random P: {single}")
        print(f"(2) sum_P S(P+k) over a 6-element structure:        {summed}")
        d24, per_bit = moebius_two_layer_one_block(c, rng)
        print(f"(3) coordinate-2 two-layer equation, ONE inner block "
              f"(24 variables): degree {d24} (per output bit {per_bit})")
        res["moebius"] = {"one_layer_single": single, "one_layer_summed": summed,
                          "two_layer_one_inner_block": d24}

    print("\n=== B. Exact key polynomial (any n, any number of inner blocks) ===")
    res["keypoly"] = {}
    for name in ("2^4", "2^8", "2^16"):
        F = make_field(name)
        rng = np.random.default_rng(a.seed)
        s_dux, sum_dux = keypoly_one_layer(F, 179 % F.q if F.q < 180 else 179,
                                           sbox_enc_chain, rng)
        one_blk = keypoly_two_layer(F, 179 % F.q if F.q < 180 else 179,
                                    ROT_FWD_BIN[0], sbox_enc_chain, (0, 3), (2,),
                                    blocks=1)
        all_blk = keypoly_two_layer(F, 179 % F.q if F.q < 180 else 179,
                                    ROT_FWD_BIN[0], sbox_enc_chain, (0, 3), (2,),
                                    blocks=4)
        print(f"DuX  F_{name}: (1) {s_dux}  (2) {sum_dux}  "
              f"(3) one inner block -> {one_blk['total']}, "
              f"all 16 inner words -> {all_blk['total']}")
        entry = {"one_layer_single": s_dux, "one_layer_summed": sum_dux,
                 "two_layer_one_block": one_blk, "two_layer_all_blocks": all_blk}
        if name != "2^4":
            al = 205 % F.q if F.q < 206 else 205
            s_yu, sum_yu = keypoly_one_layer(F, al, yu2x_sbox_enc_chain,
                                             np.random.default_rng(a.seed))
            yu = keypoly_two_layer(F, al, YU2X_ROT_FWD, yu2x_sbox_enc_chain,
                                   (0, 1), (2, 3), blocks=4)
            print(f"Yu2X F_{name}: (1) {s_yu}  (2) {sum_yu}  "
                  f"all 16 inner words -> {yu['total']}")
            entry["yu2x_one_layer_single"] = s_yu
            entry["yu2x_one_layer_summed"] = sum_yu
            entry["yu2x_two_layer_all_blocks"] = yu
        res["keypoly"][name] = entry

    print("\n=== C. Re-estimate of Ni et al. Sect. 4.3 (12-round Yu2X-16) ===")
    rows = []
    for n_v in (128, 256):
        for d in (6, 7, 8):
            r = reestimate(96, n_v, d, 8)     # m = 96, 16n = 256 equations
            rows.append(r)
            print(f"  n_v={n_v:3d} d={d}: monomials 2^{r['log2_monomials']}, "
                  f"T_c = 2^{r['log2_Tc']}, T_s = 2^{r['log2_Ts_bitops']} bit ops")
    res["reestimate_yu2x16_12round"] = rows

    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "boolean_degrees.json")
    json.dump(res, open(fn, "w"), indent=1)
    print(f"\nsaved {fn}  ({res['elapsed_s']} s)")


if __name__ == "__main__":
    main()
