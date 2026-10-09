"""E14 / W28 -- C1: weighted moments on the reduced configuration of
[LS26 v2, Section 7].

WHAT IS BEING CHECKED.  Our paper asserts (Sect. 1 and App. C.2) that weighted
moments alone would bring the data of [LS26] down to ONE subspace per stage.
[LS26 v2, Sect. 7] verifies their three-stage elimination on DuX(2^8) with the
six-round distinguisher, s = 3, seven rounds attacked, 2^24 ciphertexts per
subspace and **10, 4 and 4 subspaces** at the three stages (2^27.3 in total).
This script runs the same three stages on ONE subspace, with the weights
a = 0, 1, ..., 40 that the margin at W1 admits, and reports the rank of each
stage against the number of weights.

THE CONFIGURATION.  Their Table C3 gives the class degree bounds of the four
states of (13); at W1 they are (836, 1142, 1560, 724), so with T = 3(2^8 - 1)
= 765 the criterion guarantees C3 and nothing else, and the weight margin is
765 - 724 = 41.  `tools/zero_sum_criterion.py` reproduces that vector exactly
for 100 of the 560 three-word active sets (e.g. (0,2,3)); the default here is
the lexicographically first of them, and `--active` takes any other.  (Our own
best three-word set, (3,7,11), has (418, 571, 780, 362) instead, i.e. a
strictly better distinguisher.)

THE THREE STAGES ([LS26 v2] Theorem 3, weighted).  With
`v` a mask supported on the balanced class C3, `w = v M_t`, and every sum over
the subspace carrying the weight `x^a` of its first active word:

 (i)   v in V_{2}       : affine in the eight a_{4k}, a_{4k+3}; the coefficient
                          of a_{4k} is w_{4k+2} sum x^a P^{(k)}_3 and that of
                          a_{4k+3} is w_{4k+2} sum x^a P^{(k)}_0.
 (ii)  v in V_{1,2}     : after (i), affine in the four a_{4k+2}, coefficient
                          w_{4k+1} sum x^a P^{(k)}_3.
 (iii) v in V_{0,1,2,3} : after (i) and (ii), affine in the eight linearised
                          a_{4k+1}, a^2_{4k+1}, coefficients (16).

The pure-key terms still drop out under a weight because sum_{x in F_q} x^a = 0
for every 0 <= a < q - 1 (a = 0 included, the structure having q^s = 0 points),
which is Lemma 1 of our paper and Theorem 4's hypothesis.

The masks are the left kernels of `tools/cheap_rows.py` -- [LS26]'s V_C is our
O10 kernel for r_KR = 1 -- so their dimensions (1, 2, 3, 4 over F_{2^n} and
1, 1, 1, 4 over F_65537) come out of the repository's own tool.

Usage
  python c1_ls_reduced.py --keys 2026,7,11 --weights 41 \
      --out ../../results/E14_multi_weights/C1_ls_reduced.json
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
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round")):
    sys.path.insert(0, p_)

from dux.registry import get_cipher                              # noqa: E402
from dux.sbox import vS                                          # noqa: E402
from cheap_rows import cheap_rows                                # noqa: E402
import zero_sum_criterion as zsc                                 # noqa: E402


# ------------------------------------------------------------- the masks --
def masks(field, cheap, balanced="0001"):
    """The rows w = v M of the O10 left kernel = [LS26]'s V_C."""
    return [np.array(yL, dtype=np.int64) % 2
            for _y, yL in cheap_rows("dux", field, "dec", balanced, cheap)]


def new_directions(field, cheap, prev_cheap, balanced="0001"):
    """A basis of V_C modulo V_{C'} (the rows stage j adds to stage j-1)."""
    import itertools
    cur = masks(field, cheap, balanced)
    old = masks(field, prev_cheap, balanced) if prev_cheap else []
    out, span = [], [r.copy() for r in old]

    def rank(rows):
        rows = [r.copy() % 2 for r in rows]
        r = 0
        for c in range(16):
            piv = next((i for i in range(r, len(rows)) if rows[i][c]), None)
            if piv is None:
                continue
            rows[r], rows[piv] = rows[piv], rows[r]
            for i in range(len(rows)):
                if i != r and rows[i][c]:
                    rows[i] ^= rows[r]
            r += 1
        return r
    del itertools
    for v in cur:
        if rank(span + [v]) > rank(span):
            span.append(v)
            out.append(v)
    return out


# --------------------------------------------------------- the structure --
def slice_sums(F, arrs, nslice, slice_len):
    """XOR-sum of every array over each slice -> (nslice, len(arrs))."""
    return np.array([np.bitwise_xor.reduce(a.reshape(nslice, slice_len), axis=1)
                     for a in arrs], dtype=np.int64).T


def weighted(F, G, xvals, a):
    """sum_v xvals[v]^a G[v]  (one weighted moment per column of G)."""
    from assemble_fast import pow_field_vec
    coef = pow_field_vec(F, xvals, int(a))
    out = np.zeros(G.shape[1], dtype=np.int64)
    for v in range(G.shape[0]):
        if coef[v]:
            out ^= F.vmul(G[v], np.int64(coef[v]))
    return out


def subspace_slices(c, rks, rounds, active, seed):
    """Yield (v, plaintexts of slice v) for the 2^{8s} subspace, slice by slice.

    The slices are the values of the FIRST active word -- the weight axis, and
    the slowest one -- so the whole structure is never in memory at once
    (2^24 points x 16 int64 words is 2.1 GB before any temporary)."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    s = len(active)
    L = F.q ** (s - 1)
    idx = np.arange(L, dtype=np.int64)
    inner = {w: (idx // (F.q ** (s - 1 - j))) % F.q
             for j, w in enumerate(active) if j > 0}
    for v in range(F.q):
        cols = dict(inner)
        cols[active[0]] = np.full(L, v, dtype=np.int64)
        C = tuple(cols[i] if i in cols else np.full(L, consts[i], dtype=np.int64)
                  for i in range(16))
        yield v, [np.asarray(x, dtype=np.int64)
                  for x in c.decrypt(C, rks, rounds=rounds, vec=True)]


def gf2_solve(F, A, b):
    """Solve A z = b over F_{2^n}; returns (rank, solution or None, free)."""
    m, n = A.shape
    M = np.concatenate([A % F.q, b.reshape(-1, 1) % F.q], axis=1)
    piv, r = [], 0
    for col in range(n):
        p = next((i for i in range(r, m) if M[i][col]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        inv = F.inv(int(M[r][col]))
        M[r] = F.vmul(M[r], np.int64(inv))
        for i in range(m):
            if i != r and M[i][col]:
                M[i] ^= F.vmul(M[r], np.int64(int(M[i][col])))
        piv.append(col)
        r += 1
        if r == m:
            break
    if any(M[i][n] and not M[i][:n].any() for i in range(m)):
        return r, None, n - r
    z = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(piv):
        z[col] = M[i][n]
    return r, z, n - r


# ------------------------------------------------------------ the stages --
def _slice_cols_y(F, P, key, alpha):
    """The sixteen coordinates of S(P + key) on one slice."""
    cols = []
    for b in range(4):
        x = tuple(F.vadd(P[4 * b + i], np.int64(key[4 * b + i])) for i in range(4))
        y = vS(F, x, alpha)
        cols += list(y)
    return cols


def _slice_cols_plain(F, P):
    """Per block: (P_{4k+3}, P_{4k}, P_{4k+2}, P_{4k} P_{4k+3})."""
    cols = []
    for b in range(4):
        cols += [P[4 * b + 3], P[4 * b + 0], P[4 * b + 2],
                 F.vmul(P[4 * b + 0], P[4 * b + 3])]
    return cols


def slice_table(c, rks, rounds, active, seed, builder):
    """(nslice x ncols) table of the per-slice XOR sums of `builder`'s columns."""
    F = c.F
    G = None
    for v, P in subspace_slices(c, rks, rounds, active, seed):
        cols = builder(P)
        if G is None:
            G = np.zeros((F.q, len(cols)), dtype=np.int64)
        for j, col in enumerate(cols):
            G[v, j] = int(np.bitwise_xor.reduce(col))
    return G


def run_key(instance, rounds, active, seed, nweights, verbose=True):
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    rng = np.random.default_rng(seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    t0 = time.time()
    N = F.q ** len(active)
    xvals = np.arange(F.q, dtype=np.int64)
    ws = list(range(nweights))
    fkey = f"2^{F.n}"

    # ---- stage (i): v in V_{2}, eight unknowns a_{4k}, a_{4k+3} -----------
    V2 = masks(fkey, [2])
    Gp = slice_table(c, rks, rounds, active, seed,
                     lambda P: _slice_cols_plain(F, P))
    t_data = time.time() - t0
    mom = {a: weighted(F, Gp, xvals, a).reshape(4, 4) for a in ws}
    rows, rhs, curve = [], [], []
    for a in ws:
        M = mom[a]                                  # M[k] = (A_k, B_k, C_k, D_k)
        for w in V2:
            r = np.zeros(8, dtype=np.int64)
            const = np.int64(0)
            for k in range(4):
                cw = np.int64(int(w[4 * k + 2]) % F.q)
                if not cw:
                    continue
                r[k] ^= F.mul(int(cw), int(M[k][0]))       # a_{4k}   * A_k
                r[4 + k] ^= F.mul(int(cw), int(M[k][1]))   # a_{4k+3} * B_k
                const ^= F.mul(int(cw), int(M[k][2]) ^ int(M[k][3]))
            rows.append(r)
            rhs.append(const)
        rk_, _z, _f = gf2_solve(F, np.asarray(rows), np.asarray(rhs))
        curve.append({"weights": a + 1, "rows": len(rows), "rank": int(rk_)})
    rank1, z1, free1 = gf2_solve(F, np.asarray(rows), np.asarray(rhs))
    key = [0] * 16
    if z1 is not None and rank1 == 8:
        for k in range(4):
            key[4 * k] = int(z1[k])
            key[4 * k + 3] = int(z1[4 + k])
    stage1 = {"unknowns": 8, "masks": len(V2), "rows": len(rows),
              "rank": int(rank1), "free": int(free1), "curve": curve,
              "solved": rank1 == 8,
              "correct": all(key[4 * k + j] == int(K[4 * k + j])
                             for k in range(4) for j in (0, 3))}
    if verbose:
        print(f"    stage (i):   {len(V2)} mask(s) x {len(ws)} weights = "
              f"{len(rows)} rows, rank {rank1}/8, correct "
              f"{stage1['correct']}")

    # ---- stage (ii): v in V_{1,2}, four unknowns a_{4k+2} -----------------
    V12new = new_directions(fkey, [1, 2], [2])
    G2 = slice_table(c, rks, rounds, active, seed,
                     lambda P: _slice_cols_y(F, P, key, c.alpha))
    base = {a: weighted(F, G2, xvals, a) for a in ws}
    rows2, rhs2, curve2 = [], [], []
    for a in ws:
        yv = base[a].reshape(4, 4)                  # yv[k][c] = sum x^a y_c
        M = mom[a]
        for w in V12new:
            r = np.zeros(4, dtype=np.int64)
            const = np.int64(0)
            for k in range(4):
                for cc in range(4):
                    cw = int(w[4 * k + cc]) % F.q
                    if cw:
                        const ^= F.mul(cw, int(yv[k][cc]))
                cw1 = int(w[4 * k + 1]) % F.q
                if cw1:
                    r[k] ^= F.mul(cw1, int(M[k][0]))     # a_{4k+2} * A_k
            rows2.append(r)
            rhs2.append(const)
        rk_, _z, _f = gf2_solve(F, np.asarray(rows2), np.asarray(rhs2))
        curve2.append({"weights": a + 1, "rows": len(rows2), "rank": int(rk_)})
    rank2, z2, free2 = gf2_solve(F, np.asarray(rows2), np.asarray(rhs2))
    if z2 is not None and rank2 == 4:
        for k in range(4):
            key[4 * k + 2] = int(z2[k])
    stage2 = {"unknowns": 4, "masks": len(V12new), "rows": len(rows2),
              "rank": int(rank2), "free": int(free2), "curve": curve2,
              "solved": rank2 == 4,
              "correct": all(key[4 * k + 2] == int(K[4 * k + 2])
                             for k in range(4))}
    if verbose:
        print(f"    stage (ii):  {len(V12new)} mask(s) x {len(ws)} weights = "
              f"{len(rows2)} rows, rank {rank2}/4, correct {stage2['correct']}")

    # ---- stage (iii): v in V_{0,1,2,3}, a_{4k+1} and its square -----------
    Vall = masks(fkey, [0, 1, 2, 3])
    G3 = slice_table(c, rks, rounds, active, seed,
                     lambda P: _slice_cols_y(F, P, key, c.alpha))
    base3 = {a: weighted(F, G3, xvals, a) for a in ws}
    rows3, rhs3, curve3 = [], [], []
    for a in ws:
        yv = base3[a].reshape(4, 4)
        M = mom[a]
        for w in Vall:
            r = np.zeros(8, dtype=np.int64)
            const = np.int64(0)
            for k in range(4):
                for cc in range(4):
                    cw = int(w[4 * k + cc]) % F.q
                    if cw:
                        const ^= F.mul(cw, int(yv[k][cc]))
                w3 = int(w[4 * k + 3]) % F.q
                w0 = int(w[4 * k + 0]) % F.q
                sy2 = int(yv[k][2])                  # sum x^a y_2 of block k
                lin = 0
                if w3:
                    lin ^= F.mul(w3, int(M[k][1]))   # w_{4k+3} sum x^a P^{(k)}_0
                if w0:
                    lin ^= F.mul(w0, sy2)            # w_{4k}   sum x^a y_2
                r[k] = lin
                if w3:
                    r[4 + k] = F.mul(w3, sy2)        # coefficient of a^2_{4k+1}
            rows3.append(r)
            rhs3.append(const)
        rk_, _z, _f = gf2_solve(F, np.asarray(rows3), np.asarray(rhs3))
        curve3.append({"weights": a + 1, "rows": len(rows3), "rank": int(rk_)})
    rank3, z3, free3 = gf2_solve(F, np.asarray(rows3), np.asarray(rhs3))
    if z3 is not None:
        for k in range(4):
            key[4 * k + 1] = int(z3[k])
    stage3 = {"unknowns": 8, "effective_columns": 7, "masks": len(Vall),
              "rows": len(rows3), "rank": int(rank3), "free": int(free3),
              "curve": curve3, "solved": rank3 >= 7,
              "correct": all(key[4 * k + 1] == int(K[4 * k + 1])
                             for k in range(4))}
    if verbose:
        print(f"    stage (iii): {len(Vall)} masks x {len(ws)} weights = "
              f"{len(rows3)} rows, rank {rank3}/7 effective, correct "
              f"{stage3['correct']}")
    good = sum(1 for i in range(16) if key[i] == int(K[i]))
    return {"seed": seed, "instance": instance, "rounds": rounds,
            "active": list(active), "points": int(N), "weights": len(ws),
            "stage_i": stage1, "stage_ii": stage2, "stage_iii": stage3,
            "key_words_correct": good, "key_words_total": 16,
            "data_log2": round(float(np.log2(N)), 2),
            "decrypt_s": round(t_data, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^8")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--active", default="0,2,3",
                    help="the three active ciphertext words; the default is the "
                         "lexicographically first set whose layer-6 class bounds "
                         "are [LS26 v2]'s (836, 1142, 1560, 724)")
    ap.add_argument("--weights", type=int, default=41)
    ap.add_argument("--keys", default="2026,7,11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    active = [int(v) for v in a.active.split(",")]
    layers = a.rounds - 1
    _c = get_cipher(a.instance, rounds=a.rounds)
    fkey = f"2^{_c.F.n}" if _c.F.char == 2 else str(_c.F.q)
    pred = zsc.patterns(fkey, active, layers, None, None, "dec", "dux")
    D = pred["max_degree_per_position"][layers - 1]
    marg = pred["margin_per_position"][layers - 1]
    print(f"{a.instance}, {a.rounds} rounds, active {active}: layer {layers} "
          f"class bounds {D}, T = {pred['threshold']}, margins {marg}, "
          f"pattern {pred['patterns'][layers - 1]}")
    print(f"  [LS26 v2] Table C3 at W1: (836, 1142, 1560, 724), guaranteed C3, "
          f"margin 765 - 724 = 41")
    dims = {c: len(masks(fkey, cc)) for c, cc in
            (("V_{2}", [2]), ("V_{1,2}", [1, 2]), ("V_{0,1,2}", [0, 1, 2]),
             ("V_{0,1,2,3}", [0, 1, 2, 3]))}
    print(f"  mask spaces (tools/cheap_rows.py): {dims} "
          f"([LS26]: 1, 2, 3, 4)")
    res = []
    for seed in [int(v) for v in a.keys.split(",")]:
        print(f"  key seed {seed}:", flush=True)
        r = run_key(a.instance, a.rounds, active, seed, a.weights)
        print(f"    -> {r['key_words_correct']}/16 key words from ONE "
              f"subspace of 2^{r['data_log2']} ciphertexts", flush=True)
        res.append(r)
    out = {"instance": a.instance, "rounds": a.rounds, "active": active,
           "layer": layers, "class_bounds": D, "threshold": pred["threshold"],
           "margins": marg, "pattern": pred["patterns"][layers - 1],
           "ls_class_bounds_W1": [836, 1142, 1560, 724],
           "ls_margin": 41, "ls_subspaces": [10, 4, 4],
           "ls_total_log2_data": 27.32,
           "mask_space_dims": dims, "weights": a.weights, "runs": res,
           "all_keys_16_of_16": all(r["key_words_correct"] == 16 for r in res)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
