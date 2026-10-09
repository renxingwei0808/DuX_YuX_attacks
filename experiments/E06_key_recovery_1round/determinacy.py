"""E06 / W10 -- Lemma O8: which key words an r_KR = 1 equation can determine,
and why the linearised system is rank-deficient.

Background: the rank deficiencies observed by attack_1round.py and
attack_1round_partial.py (records in results/E06_key_recovery_1round/).

Setting.  W = S(P + k) is the last (encryption-direction) S layer of the
attack; a structure with sum_P 1 = |P| = 0 in F_q turns each balanced
coordinate of W into one F_q-linear equation in the monomials of k:

    sum_P S_c(P + k) = sum_{ke != 0-in-P} coeff * k^{ke} * M[pe] = 0,
    M[pe] = sum_P P0^{pe0} P1^{pe1} P2^{pe2} P3^{pe3},

because every term whose P-exponent is (0,0,0,0) carries the factor |P| = 0.

Part 1 (determinacy).  The expansion is done TWICE and cross-checked:
  * symbolically with sympy over Z[alpha] (this file), and
  * with the exact field arithmetic of experiments/E08.../keypoly.py, which is
    what the attack scripts actually use (`equation_template`).
For every coordinate we report which key words appear on their own as a
degree-1 monomial and with which moment as coefficient.  Over F_{2^n} the
integer coefficients are reduced mod 2, which is Lucas' theorem applied to the
binomial coefficients of the expansion.

Part 2 (rank).  Write T for the (monomials x moments) coefficient matrix of the
equation template, so that the linearised system assembled from n structures is

    A = M T^t,        M[s][pe] = the moment M[pe] of structure s.

Then A x = 0  <=>  T^t x in ker(M) = V^perp, with V = span of the achievable
moment vectors.  Hence

    rank(A) = rank(T) - dim( rowspace(T) INTERSECT V^perp ).

V^perp always contains one vector per ALREADY BALANCED coordinate c' of W:
that coordinate's own equation  sum_pe (sum_m T_{c'}[m][pe] * m(k)) M[pe] = 0
holds for every structure, so u_{c'}[pe] = sum_m T_{c'}[m][pe] * m(k) is
orthogonal to every achievable moment vector.  Those vectors are key-dependent
but computable here, which turns the observed rank deficiency into a
prediction:

    predicted rank = rank(T) - dim( rowspace(T) INTERSECT span{u_{c'}} )   (an
    UPPER bound: V could be smaller than the identities alone force).

Usage
    python determinacy.py                      # the O8 table for all fields
    python determinacy.py --rank --instance dux-2^8 --rounds 6 --active 0 \
        --balanced 0,1,2,3 --coords 0,1,2      # rank prediction vs measurement
    python determinacy.py --out results/E06_key_recovery_1round
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))
sys.path.insert(0, HERE)

from dux import DuX                                    # noqa: E402
from dux.field import make_field                       # noqa: E402
from attack_1round_partial import equation_template    # noqa: E402

COORD_NAME = {"dux": {0: "c", 1: "b", 2: "a", 3: "y3"},
              "yux": {0: "z3", 1: "z2", 2: "z1", 3: "z0"}}
COORD_DEG = {"dux": {0: 5, 1: 3, 2: 2, 3: 8},
             "yux": {0: 8, 1: 5, 2: 3, 3: 2}}


# ------------------------------------------------------------- symbolic ----
def sympy_sbox(cipher="dux"):
    """S(P + k) with sympy over Z[alpha].

    DuX: S = (R^{-1})^4 -> (c, b, a, y3) at positions 0..3.
    YuX: S = Pf^{-4}    -> (z3, z2, z1, z0) at positions 0..3."""
    k = sp.symbols("k0 k1 k2 k3")
    p = sp.symbols("p0 p1 p2 p3")
    al = sp.Symbol("alpha")
    x = [k[i] + p[i] for i in range(4)]
    if cipher == "yux":
        z0 = sp.expand(x[3] - x[0] * x[1] - x[2] - al)
        z1 = sp.expand(x[2] - z0 * x[0] - x[1] - al)
        z2 = sp.expand(x[1] - z1 * z0 - x[0] - al)
        z3 = sp.expand(x[0] - z2 * z1 - z0 - al)
        return k, p, al, [z3, z2, z1, z0]
    a = sp.expand(x[2] - x[0] * x[3] - al)
    b = sp.expand(x[1] - x[3] * a - al)
    c = sp.expand(x[0] - a * b - al)
    y3 = sp.expand(x[3] - b * c - al)
    return k, p, al, [c, b, a, y3]


def sympy_terms(coord, cipher="dux"):
    """{(k-exponents, p-exponents): [(alpha exponent, integer coeff), ...]}."""
    k, p, al, S = sympy_sbox(cipher)
    poly = sp.Poly(sp.expand(S[coord]), *k, *p, al)
    out = {}
    for mono, co in zip(poly.monoms(), poly.coeffs()):
        ke, pe, ae = tuple(mono[:4]), tuple(mono[4:8]), mono[8]
        out.setdefault((ke, pe), []).append((ae, int(co)))
    return out


def to_field(F, alpha, terms):
    """Evaluate the sympy coefficients in F: sum_j c_j * alpha^j."""
    out = {}
    for key, lst in terms.items():
        v = 0
        for ae, co in lst:
            t = 1
            for _ in range(ae):
                t = F.mul(t, alpha)
            c = (co % 2) if F.char == 2 else (co % F.p)
            if c:
                v = F.add(v, F.mul(F.from_int(c), t))
        if v:
            out[key] = v
    return out


def surviving(field_terms):
    """Drop the terms killed by sum_P 1 = |P| = 0 (P-exponent all zero)."""
    return {kp: v for kp, v in field_terms.items() if any(kp[1])}


# --------------------------------------------------------------- part 1 ----
def determinacy_table(field_name, alpha, cipher="dux"):
    """For each coordinate: the surviving k-monomials, and how each key word
    appears on its own (degree-1 monomial) and with which moment."""
    F = make_field(field_name)
    rows = {}
    for coord in range(4):
        terms = to_field(F, alpha, sympy_terms(coord, cipher))
        surv = surviving(terms)
        kmons = sorted({ke for ke, _ in surv})
        entry = {"coordinate": coord, "name": COORD_NAME[cipher][coord],
                 "degree": COORD_DEG[cipher][coord],
                 "monomials_before_sum": len({ke for ke, _ in terms}),
                 "monomials_after_sum": len(kmons),
                 "linear": {}, "k1": None}
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            moms = sorted({pe for (ke, pe) in surv if ke == e})
            entry["linear"][f"k{i}"] = [list(pe) for pe in moms]
        # how does k1 show up at all?
        with_k1 = sorted({ke for ke, _ in surv if ke[1] > 0})
        killed_k1 = sorted({ke for ke, _ in terms if ke[1] > 0})
        entry["k1"] = {"monomials_after_sum": [list(m) for m in with_k1],
                       "monomials_before_sum": [list(m) for m in killed_k1]}
        # which key words survive ONLY as a difference (YuX's k3 - k2)?
        pair = {}
        for i in range(4):
            for t in range(i + 1, 4):
                ei = tuple(1 if u == i else 0 for u in range(4))
                et = tuple(1 if u == t else 0 for u in range(4))
                ci = {pe: v for (ke, pe), v in surv.items() if ke == ei}
                ct = {pe: v for (ke, pe), v in surv.items() if ke == et}
                if ci and ct and set(ci) == set(ct) and \
                        all(F.add(ci[pe], ct[pe]) == 0 for pe in ci):
                    pair[f"k{t} - k{i}"] = [list(pe) for pe in sorted(ci)]
        entry["only_as_difference"] = pair
        rows[coord] = entry
        # cross-check against the field expansion used by the attack scripts
        mons_ref, terms_ref = equation_template(F, alpha, coord, cipher)
        ref = {(ke, pe): 0 for ke, lst in terms_ref.items() for pe, _ in lst}
        for ke, lst in terms_ref.items():
            for pe, co in lst:
                ref[(ke, pe)] = F.add(ref[(ke, pe)], co)
        ref = {kp: v for kp, v in ref.items() if v}
        assert ref == surv, (f"sympy and keypoly disagree on {field_name} "
                             f"coordinate {coord} ({cipher})")
        entry["cross_checked_with_keypoly"] = True
    return rows


# --------------------------------------------------------------- part 2 ----
def template_matrix(F, alpha, coord, moments):
    """T[m][pe] over the given monomial and moment orders."""
    mons, terms = equation_template(F, alpha, coord)
    mi = {m: i for i, m in enumerate(mons)}
    pi = {pe: i for i, pe in enumerate(moments)}
    T = np.zeros((len(mons), len(moments)), dtype=np.int64)
    for ke, lst in terms.items():
        for pe, co in lst:
            T[mi[ke]][pi[pe]] = F.add(int(T[mi[ke]][pi[pe]]), co)
    return mons, T


def rank_field(F, rows):
    """Rank of a matrix over F_q (Gaussian elimination, no pivoting tricks)."""
    A = [list(map(int, r)) for r in rows]
    m = len(A)
    n = len(A[0]) if m else 0
    r = 0
    for c in range(n):
        pr = next((i for i in range(r, m) if A[i][c]), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        inv = F.inv(A[r][c])
        A[r] = [F.mul(v, inv) for v in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [F.sub(u, F.mul(f, v)) for u, v in zip(A[i], A[r])]
        r += 1
        if r == m:
            break
    return r


def monomial_at_key(F, ke, kblock):
    v = 1
    for i, e in enumerate(ke):
        for _ in range(e):
            v = F.mul(v, kblock[i])
    return v


def predict_rank(F, alpha, coords, balanced, kblock, drop_const=True):
    """Predicted rank of the linearised r_KR = 1 system.

    Each usable coordinate gives its OWN equation per structure, so the map
    from the monomial vector x to what the structures see is

        Phi(x) = ( T_c^t x )_{c in coords}   in  (moment space)^{|coords|},

    and the system is  Phi(x) in (V^perp)^{|coords|}.  Hence

        rank(A) = rank(Phi) - dim( image(Phi) INTERSECT (V^perp)^{|coords|} ),

    where V^perp is spanned by one vector per already balanced coordinate c':
        u_{c'}[pe] = sum_m T_{c'}[m][pe] * m(k)                (key-dependent).
    `drop_const` removes the all-zero monomial, matching the attack scripts,
    which pin it to 1 and move it to the right-hand side.
    """
    all_c = sorted(set(coords) | set(balanced))
    moments = sorted({pe for co in all_c
                      for lst in equation_template(F, alpha, co)[1].values()
                      for pe, _ in lst})
    nM = len(moments)
    Ts, mons_all = {}, {}
    for co in all_c:
        mons_all[co], Ts[co] = template_matrix(F, alpha, co, moments)
    const = (0, 0, 0, 0)
    mons = sorted({m for co in coords for m in mons_all[co]})
    if drop_const:
        mons = [m for m in mons if m != const]
    mi = {m: i for i, m in enumerate(mons)}
    # block-diagonal stacking: one copy of the moment space per coordinate
    T = np.zeros((len(mons), len(coords) * nM), dtype=np.int64)
    for slot, co in enumerate(coords):
        for j, m in enumerate(mons_all[co]):
            if m in mi:
                T[mi[m], slot * nM:(slot + 1) * nM] = Ts[co][j]
    rT = rank_field(F, T) if len(mons) else 0
    # identities, one per (balanced coordinate, slot)
    U = []
    for co in balanced:
        u = np.zeros(nM, dtype=np.int64)
        for j, m in enumerate(mons_all[co]):
            val = monomial_at_key(F, m, kblock)
            if val:
                for t in range(nM):
                    if Ts[co][j][t]:
                        u[t] = F.add(int(u[t]), F.mul(val, int(Ts[co][j][t])))
        if not u.any():
            continue
        for slot in range(len(coords)):
            v = np.zeros(len(coords) * nM, dtype=np.int64)
            v[slot * nM:(slot + 1) * nM] = u
            U.append(v)
    rU = rank_field(F, U) if U else 0
    both = rank_field(F, list(T) + U) if U else rT
    inter = rT + rU - both
    return {"monomials": len(mons), "moments": nM, "coords": list(coords),
            "balanced": list(balanced), "drop_const": drop_const,
            "rank_Phi": rT, "identities": rU, "intersection": inter,
            "predicted_rank": rT - inter}


# ---------------------------------------------------------------- grid -----
RANK_GRID = [
    # (label, instance, rounds, active word, Z_l pattern, W balanced coords,
    #  coords used, observed rank, observed unknowns, key words the linear part fixes)
    ("2^8 1111 c=0,1,2", "dux-2^8", 6, 3, "1111", (0, 1, 2, 3), (0, 1, 2), 10, 13, (0, 2, 3)),
    ("2^8 1111 c=0", "dux-2^8", 6, 3, "1111", (0, 1, 2, 3), (0,), 9, 13, (0, 3)),
    ("2^8 1111 c=3", "dux-2^8", 6, 3, "1111", (0, 1, 2, 3), (3,), 21, 36, (0, 3)),
    ("2^8 1111 c=0,1,2,3", "dux-2^8", 6, 3, "1111", (0, 1, 2, 3), (0, 1, 2, 3), 24, 39, (0, 2, 3)),
    ("2^16 1101 c=0", "dux-2^16", 10, 0, "1101", (0, 3), (0,), 10, 13, (0, 1, 3)),
    ("257 1001 c=3", "toy-257", 6, 0, "1001", (3,), (3,), 34, 65, (0, 1, 2, 3)),
]


def run_grid(seed=2026):
    """Predicted vs observed rank for every cell the repository has measured."""
    ok = 0
    out = []
    for (label, inst, rounds, act, zpat, bal, coords, orank, ounk, ofix) in RANK_GRID:
        c = DuX(inst, rounds=rounds)
        K = c.random_key(np.random.default_rng(seed))
        pr = predict_rank(c.F, c.alpha, list(coords), list(bal),
                          list(K[0:4]))
        good = (pr["predicted_rank"] == orank and pr["monomials"] == ounk)
        ok += good
        out.append({"label": label, "instance": inst, "rounds": rounds,
                    "active": act, "Z_pattern": zpat, "W_balanced": list(bal),
                    "coords": list(coords), "unknowns": pr["monomials"],
                    "rank_Phi": pr["rank_Phi"], "identities": pr["identities"],
                    "intersection": pr["intersection"],
                    "predicted_rank": pr["predicted_rank"],
                    "observed_rank": orank, "observed_unknowns": ounk,
                    "linear_part_fixes": list(ofix), "match": bool(good)})
        print(f"{'OK' if good else 'XX'}  {label:22s} unknowns {pr['monomials']:>3d}"
              f"/{ounk:<3d} rank(Phi) {pr['rank_Phi']:>3d} - inter {pr['intersection']}"
              f"  predicted {pr['predicted_rank']:>3d}  observed {orank}")
    print(f"{ok}/{len(RANK_GRID)} measured rank cells reproduced")
    return ok == len(RANK_GRID), out


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="dux", choices=("dux", "yux"))
    ap.add_argument("--fields", default="2^8,2^16,65537,257")
    ap.add_argument("--alpha", type=int, default=None,
                    help="default: 179 for DuX, 205 for YuX")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--grid", action="store_true",
                    help="predicted vs observed rank on every measured cell")
    ap.add_argument("--instance", default="dux-2^8")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--coords", default="0,1,2")
    ap.add_argument("--balanced", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.alpha is None:
        a.alpha = 179 if a.cipher == "dux" else 205
    report = {"cipher": a.cipher, "alpha": a.alpha, "determinacy": {},
              "rank": [], "rank_grid": []}
    for fn in a.fields.split(","):
        rows = determinacy_table(fn, a.alpha, a.cipher)
        report["determinacy"][fn] = rows
        print(f"\n=== {fn} (alpha = {a.alpha}) "
              f"[sympy vs keypoly: cross-checked]")
        print("coord  name  deg  k-monomials (before/after sum_P)  "
              "key words visible on their own")
        for co in range(4):
            e = rows[co]
            solo = [w for w, moms in e["linear"].items() if moms]
            print(f"  {co}    {e['name']:<3s}  {e['degree']:<3d}  "
                  f"{e['monomials_before_sum']:>3d} / {e['monomials_after_sum']:<3d}"
                  f"                     {', '.join(solo) if solo else '(none)'}")
        for co in range(4):
            e = rows[co]
            k1m = e["k1"]["monomials_after_sum"]
            print(f"  coord {co} ({e['name']}): k1 survives in "
                  f"{len(k1m)} monomial(s): {k1m if len(k1m) <= 6 else '...'}")
            if e["only_as_difference"]:
                print(f"           key words visible only as a difference: "
                      f"{sorted(e['only_as_difference'])}")

    if a.grid:
        print("\n=== rank prediction vs measurement (O8)")
        allok, report["rank_grid"] = run_grid(a.seed)
    if a.rank:
        c = DuX(a.instance, rounds=a.rounds)
        rng = np.random.default_rng(a.seed)
        K = c.random_key(rng)
        coords = [int(v) for v in a.coords.split(",")]
        balanced = [int(v) for v in a.balanced.split(",")] if a.balanced else []
        print(f"\n=== rank prediction, {a.instance}, coords {coords}, "
              f"balanced coordinates of W {balanced}")
        for j in range(4):
            pr = predict_rank(c.F, c.alpha, coords, balanced, list(K[4 * j:4 * j + 4]))
            pr["block"] = j
            report["rank"].append(pr)
            print(f"  block {j}: {pr['monomials']} unknowns, {pr['moments']} moments, "
                  f"rank(Phi) = {pr['rank_Phi']}, identities {pr['identities']}, "
                  f"intersection {pr['intersection']} -> predicted rank "
                  f"{pr['predicted_rank']}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "determinacy.json" if a.cipher == "dux"
                          else f"determinacy_{a.cipher}.json")
        json.dump(report, open(fn, "w"), indent=1)
        print("\nsaved", fn)


if __name__ == "__main__":
    main()
