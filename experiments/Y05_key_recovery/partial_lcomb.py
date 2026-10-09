"""Y05-A.4 -- r_KR = 1 from a PARTIAL zero-sum when the linear layer is dense.

Why a new script.  `experiments/E06_key_recovery_1round/attack_1round_partial.py`
turns a partial zero-sum into equations by INVERTING the linear layer: with
W = L^{-1}(Z_l - rk^1) and DuX's L^{-1} rotations {1,4,8,9,13} (residues {0,1}
mod 4), output position p of W collects Z_l at positions {p, p+1} only, so a
pattern such as `1101` still leaves whole W words balanced.

YuX's L^{-1} = sum_{j in {0,3,4,8,9,12,14}} Rot_j has residues {0,1,2,3} mod 4:
EVERY W word collects all four positions of Z_l, so a partial pattern leaves NO
balanced W word at all, and `tools/cheap_rows.py` reports dim K = 0 for `1100`.
The (l+1)-round route is not dead, though -- it just has to be written in the
FORWARD direction (memo Sect. 5.4):

    Z_l = L(W) + rk^1 ,   W = SL(P + rk^0) ,   so for every balanced word i
    sum_{P in structure} Z_l,i = [ L( sum_P W ) ]_i + |structure| * rk^1_i
                               = [ L( sum_P W ) ]_i = 0 .

One equation per balanced WORD (not per block), and it mixes the four blocks
through L, so the unknowns are the union of all four S-box coordinates'
key monomials, in all four blocks:  4 * |mons| instead of |mons|.
With O12 weights (Sect. 5.3) the same structure gives one such equation per
admissible weight a < margin[i], so a single structure is usually enough.

Usage
  python3 experiments/Y05_key_recovery/partial_lcomb.py \
      --instance yuxtoy-257 --rounds 6 --active 0 --pattern 1100 --keys 3
  python3 experiments/Y05_key_recovery/partial_lcomb.py \
      --instance yuxtoy-257 --rounds 6 --active 0 --pattern 1100 --weights 0 --keys 3
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
sys.path.insert(0, os.path.join(HERE, "..", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from dux.registry import cipher_family, get_cipher          # noqa: E402
from attack_1round_partial import equation_template, rref    # noqa: E402
import zero_sum_criterion as zsc                             # noqa: E402


# ------------------------------------------------------------------ setup --
def field_key(F):
    return f"2^{F.n}" if F.char == 2 else F.q


def margins(F, active, layers, cipher):
    r = zsc.patterns(field_key(F), list(active), layers, None, None, "dec",
                     cipher, [])
    return r["margin_per_position"][layers - 1], r["patterns"][layers - 1]


def linear_rows(c):
    """L[i][j] of the ENCRYPTION linear layer (L(x)_i = sum_j row[j] x_{i+j})."""
    row = c.lin._rows[0]
    n = len(row)
    return [[int(row[(j - i) % n]) for j in range(n)] for i in range(n)]


def structure(c, rks, rounds, active, rng):
    """One chosen-ciphertext structure: every active word runs over all of F_q.
    Returns (P, xs) with P[w] the plaintext word w and xs the first active
    ciphertext word (the weight carrier)."""
    F = c.F
    assert len(active) == 1, "one active word per structure (weights need a single axis)"
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    vals = np.arange(F.q, dtype=np.int64)
    C = [np.full(F.q, consts[i], dtype=np.int64) for i in range(16)]
    C[active[0]] = vals
    P = c.decrypt(tuple(C), rks, rounds=rounds, vec=True)
    return P, vals


# --------------------------------------------------------------- moments ---
def moments(c, P, xs, needed, weights):
    """mom[a][b][pe] = sum_x x^a * prod_i P[4b+i]^{pe_i}."""
    F = c.F
    maxe = max(max(pe) for pe in needed)
    wcol = {}
    for a in weights:
        if a == 0:
            wcol[a] = np.ones(len(xs), dtype=np.int64)
        else:
            col = xs.copy()
            for _ in range(a - 1):
                col = F.vmul(col, xs)
            wcol[a] = col
    out = {a: [dict() for _ in range(4)] for a in weights}
    for b in range(4):
        pw = []
        for i in range(4):
            p_i = [np.ones(len(xs), dtype=np.int64)]
            for _ in range(maxe):
                p_i.append(F.vmul(p_i[-1], P[4 * b + i]))
            pw.append(p_i)
        for pe in needed:
            acc = None
            for i in range(4):
                if pe[i] == 0:
                    continue
                acc = pw[i][pe[i]] if acc is None else F.vmul(acc, pw[i][pe[i]])
            if acc is None:
                acc = np.ones(len(xs), dtype=np.int64)
            for a in weights:
                out[a][b][pe] = int(F.vsum(F.vmul(acc, wcol[a])))
    return out


# ----------------------------------------------------------------- solve ---
def build_rows(c, mom, tmpl, mons, Lrows, words, weights):
    """One row per (weight, balanced word).  Columns: block b, monomial m."""
    F = c.F
    nm = len(mons)
    idx = {m: t for t, m in enumerate(mons)}
    rows = []
    for a in weights:
        for i in words:
            row = [0] * (4 * nm)
            for jp in range(16):
                lij = Lrows[i][jp]
                if lij == 0:
                    continue
                b, co = jp // 4, jp % 4
                _, terms = tmpl[co]
                for m, lst in terms.items():
                    v = 0
                    for pe, coeff in lst:
                        v = F.add(v, F.mul(coeff, mom[a][b][pe]))
                    if v:
                        col = b * nm + idx[m]
                        row[col] = F.add(row[col], F.mul(lij, v))
            rows.append(row)
    return rows


def solve(F, rows, mons, nblocks=4):
    """Pin the constant monomial of every block to 1, then read the degree-one
    monomials.  Returns {(b, i): value} for the key words that are determined."""
    nm = len(mons)
    zero = tuple([0] * 4)
    const_cols = [b * nm + mons.index(zero) for b in range(nblocks)]
    rhs = []
    red = []
    for row in rows:
        r = list(row)
        v = 0
        for col in const_cols:
            v = F.sub(v, r[col])
            r[col] = 0
        red.append(r)
        rhs.append(v)
    A, B, piv = rref(F, red, rhs)
    got = {}
    free_cols = set()
    for t, p in enumerate(piv):
        if p < 0:
            continue
        # a pivot column is determined only if the row has no other nonzero
        if any(A[t][k] for k in range(len(A[t])) if k != p):
            continue
        got[p] = B[t]
    out, det = {}, 0
    for b in range(nblocks):
        for i in range(4):
            e = [0] * 4
            e[i] = 1
            col = b * nm + mons.index(tuple(e))
            if col in got:
                out[(b, i)] = got[col]
                det += 1
    rank = sum(1 for p in piv if p >= 0)
    return out, det, rank, len(free_cols)


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yuxtoy-257")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--active", default="0")
    ap.add_argument("--pattern", default=None,
                    help="balanced pattern of layer rounds-1 (default: from O7)")
    ap.add_argument("--weights", type=int, default=None,
                    help="number of weights a per structure (default: the whole "
                         "margin of the balanced positions; 0 = no weights)")
    ap.add_argument("--structures", type=int, default=None)
    ap.add_argument("--keys", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    c = get_cipher(a.instance)
    fam = cipher_family(a.instance)
    F = c.F
    active = [int(v) for v in a.active.split(",")]
    layers = a.rounds - 1
    marg, pred = margins(F, active, layers, fam)
    pat = a.pattern or pred
    pos = [p for p in range(4) if pat[p] == "1"]
    assert pos, f"layer {layers} has no balanced position (pattern {pat})"
    words = [4 * b + p for b in range(4) for p in pos]
    wmargin = min(marg[p] for p in pos)
    nw = wmargin if a.weights is None else a.weights
    weights = list(range(max(1, nw)))

    tmpl = {co: equation_template(F, c.alpha, co, fam) for co in range(4)}
    mons = sorted({m for co in range(4) for m in tmpl[co][0]})
    needed = sorted({pe for co in range(4) for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    unknowns = 4 * len(mons)
    per_struct = len(weights) * len(words)
    nstruct = a.structures or max(1, -(-(unknowns + 8) // per_struct))
    Lrows = linear_rows(c)

    print(f"{a.instance} ({'F_2^%d' % F.n if F.char == 2 else 'F_%d' % F.q}): "
          f"r = {a.rounds}, layer {layers} pattern {pat} "
          f"(O7 predicts {pred}), balanced words {words}")
    print(f"  L-combined equations: {len(mons)} k-monomials/block x 4 blocks = "
          f"{unknowns} unknowns; margins {marg} -> {len(weights)} weights "
          f"(a < {wmargin}); {per_struct} equations/structure; {nstruct} structures")

    res = []
    t0 = time.time()
    for kk in range(a.keys):
        rng = np.random.default_rng(a.seed + kk)
        key = c.random_key(rng)
        rks = c.key_schedule(key)
        rows = []
        for s in range(nstruct):
            P, xs = structure(c, rks, a.rounds, active, rng)
            mom = moments(c, P, xs, needed, weights)
            rows += build_rows(c, mom, tmpl, mons, Lrows, words, weights)
        got, det, rank, _ = solve(F, rows, mons)
        truth = {(b, i): int(rks[0][4 * b + i]) for b in range(4) for i in range(4)}
        ok = sum(1 for k, v in got.items() if truth[k] == v)
        res.append({"key": kk, "determined": det, "correct": ok, "rank": rank,
                    "rows": len(rows)})
        print(f"  key {kk}: rows {len(rows)}, rank {rank}/{unknowns}, "
              f"determined {det}/16, correct {ok}/16")
    data_log2 = round(np.log2(nstruct * F.q), 2)
    print(f"  data 2^{data_log2} per key; {time.time() - t0:.1f}s")
    out = {"instance": a.instance, "cipher": fam,
           "field": f"F_2^{F.n}" if F.char == 2 else f"F_{F.q}",
           "rounds": a.rounds, "layers": layers, "pattern": pat,
           "predicted_pattern": pred, "balanced_words": words,
           "margins": marg, "weights": len(weights), "weight_bound": wmargin,
           "monomials_per_block": len(mons), "unknowns": unknowns,
           "equations_per_structure": per_struct, "structures": nstruct,
           "data_log2": data_log2, "keys": a.keys, "seed": a.seed,
           "results": res, "elapsed_s": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"partial_lcomb_{a.instance}_r{a.rounds}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(out, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
