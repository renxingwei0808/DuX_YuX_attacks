"""E06b (W6) -- r_KR = 1 key recovery from a PARTIAL zero-sum.

When the l-layer distinguisher balances only some block positions of the state
Z_l, only some coordinates of the last (encryption-direction) S-box give a
usable equation.  With  W = L^{-1}(Z_l - rk^1) = SL(P + rk^0)  and L^{-1}'s
rotations {1,4,8,9,13} (or {1,5,8,12,13}), whose residues mod 4 are {0,1},
output position p of W collects Z_l at positions p and p+1, hence

    Z_l balanced at {0,1,2,3} -> W balanced everywhere -> coordinates 0,1,2,3
    Z_l balanced at {0,1,3}   -> W balanced at {0,3}   -> coordinates 0 and 3
    Z_l balanced at {0,3}     -> W balanced at {3}     -> coordinate 3 only

(the correspondence between zero-sum positions and key recovery
coordinates).  Coordinate 2 -- the one the
full-zero-sum attack of attack_1round.py uses -- is exactly the one we lose.

Method.  Expand S(P + k) once, symbolically, as a polynomial in the four key
words AND the four plaintext words (experiments/E08.../keypoly.py).  Summing
over a structure replaces every plaintext monomial by a numeric moment
    M[e] = sum_{P in structure} P0^{e0} P1^{e1} P2^{e2} P3^{e3},
so each structure turns coordinate c into ONE F_q-linear equation in the
monomials of k that survive the sum:

    coordinate 2:  3 monomials  (affine in (k0,k3))       <- the easy case
    coordinate 1:  5 (F_2^n) / 6 (F_p) monomials
    coordinate 0: 14 (F_2^n) / 20 (F_p) monomials, F_q-degree 4
    coordinate 3: 37 (F_2^n) / 66 (F_p) monomials, F_q-degree 7

so the PARTIAL case is still pure linear algebra -- it just needs ~14-20
structures instead of 2.  The key is read off the degree-one monomials and the
solution is unique as soon as the matrix has full rank (checked, and the
recovered key is compared with the real one).

Usage
  # DuX(2^8), 6 rounds: layer-5 zero-sum at {0,3} (single word, position 0)
  python attack_1round_partial.py --instance dux-2^8 --rounds 6 --active 0 --coords 3
  # DuX(65537), 10 rounds: layer-9 zero-sum at {0,1,3} (single word, position 0)
  python attack_1round_partial.py --instance dux-65537 --rounds 10 --active 0 --coords 0,3
  # DuX(2^8), 7 rounds: layer-6 {0,1,3} from the 3-word structure (2^24 each)
  python attack_1round_partial.py --instance dux-2^8 --rounds 7 --active 3,7,11 --coords 0,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "E08_boolean_degree_extension"))
from dux import DuX                                     # noqa: E402
from dux.registry import cipher_family, get_cipher      # noqa: E402
from keypoly import KeyPoly, ENC_CHAIN, sbox_enc_chain  # noqa: E402
from experiments_e04_helpers import random_basis, span_values  # noqa: E402


# --------------------------------------------------------------- symbolic --
def sbox_joint_polys(F, alpha, cipher="dux"):
    """S(P + k) with BOTH k (variables 0..3) and P (variables 4..7) symbolic.

    `cipher` selects the family's ENCRYPTION S-box chain: DuX's (R^{-1})^4 with
    coordinate degrees (5,3,2,8) or YuX's Pf^{-4} with (8,5,3,2)."""
    x = [KeyPoly.var_plus_const(F, i, 0, 8) + KeyPoly.var_plus_const(F, 4 + i, 0, 8)
         for i in range(4)]
    return ENC_CHAIN[cipher](x, alpha)


def equation_template(F, alpha, coord, cipher="dux"):
    """After summing over a structure every pure-key monomial dies (its
    coefficient is |structure| = 0), so the equation is
        sum_{(ke, pe)} coeff * M[pe] * k^{ke} = 0.
    Returns (list of k-monomials, {k-monomial: [(P-exponent, coeff), ...]})."""
    poly = sbox_joint_polys(F, alpha, cipher)[coord]
    terms = {}
    for e, co in poly.d.items():
        ke, pe = e[:4], e[4:]
        if not any(pe):
            continue                       # killed by sum_P 1 = |structure| = 0
        terms.setdefault(ke, []).append((pe, co))
    mons = sorted(terms)
    return mons, terms


# -------------------------------------------------------------- structure --
def structure_moments(c, rks, rounds, active, dim, basis_seed, needed, chunk_log2=20):
    """Decrypt one ciphertext structure and return, for every S-box block j,
    the moments {P-exponent vector -> sum over the structure}, computed in
    chunks so that 2^24-2^26 structures fit in memory."""
    F = c.F
    rng = np.random.default_rng(basis_seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    s = len(active)
    if F.char == 2:
        vals = [span_values(random_basis(F.n, dim, rng), int(rng.integers(0, F.q)))
                for _ in active]
        total_log2 = dim * s
        per = dim
    else:
        assert s == 1, "prime fields: one active word (full field) per structure"
        vals = [np.arange(F.q, dtype=np.int64)]
        total_log2 = None
        per = None
    maxe = max(max(pe) for pe in needed)
    moments = [{pe: 0 for pe in needed} for _ in range(4)]
    if F.char == 2:
        n_chunks = 1 << max(0, total_log2 - chunk_log2)
        csize = 1 << min(total_log2, chunk_log2)
        starts = [i * csize for i in range(n_chunks)]
    else:
        csize, starts = F.q, [0]
    for start in starts:
        idx = np.arange(start, start + csize, dtype=np.int64)
        C = [np.full(csize, consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            C[w] = vals[t][(idx >> (per * t)) & ((1 << per) - 1)] if F.char == 2 else vals[t]
        P = c.decrypt(tuple(C), rks, rounds=rounds, vec=True)
        for j in range(4):
            blk = [P[4 * j + i] for i in range(4)]
            pw = []
            for i in range(4):
                p_i = [np.ones(csize, dtype=np.int64)]
                for _ in range(maxe):
                    p_i.append(F.vmul(p_i[-1], blk[i]))
                pw.append(p_i)
            for pe in needed:
                acc = None
                for i in range(4):
                    if pe[i] == 0:
                        continue
                    acc = pw[i][pe[i]] if acc is None else F.vmul(acc, pw[i][pe[i]])
                v = F.vsum(acc) if acc is not None else (csize % F.q if F.char != 2 else 0)
                moments[j][pe] = F.add(moments[j][pe], v)
    return moments, (total_log2 if F.char == 2 else int(np.log2(F.q)))


# ------------------------------------------------------------------ solve --
def rref(F, rows, rhs):
    """Reduced row echelon form of [A|b] over F_q.  Returns (A, b, pivots)."""
    m, n = len(rows), len(rows[0])
    A = [list(r) for r in rows]
    B = list(rhs)
    piv, r = [], 0
    for cidx in range(n):
        p = next((i for i in range(r, m) if A[i][cidx]), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        B[r], B[p] = B[p], B[r]
        inv = F.inv(A[r][cidx])
        A[r] = [F.mul(v, inv) for v in A[r]]
        B[r] = F.mul(B[r], inv)
        for i in range(m):
            if i != r and A[i][cidx]:
                f = A[i][cidx]
                A[i] = [F.sub(u, F.mul(f, v)) for u, v in zip(A[i], A[r])]
                B[i] = F.sub(B[i], F.mul(f, B[r]))
        piv.append(cidx)
        r += 1
        if r == m:
            break
    return A, B, piv


def determined_values(F, rows, rhs, mons):
    """Which monomials are pinned down by the (generally rank-deficient)
    linearised system?  A pivot monomial is determined iff its row has no
    entry in a free column.  Returns {monomial: value}."""
    A, B, piv = rref(F, rows, rhs)
    n = len(mons)
    free = [ci for ci in range(n) if ci not in piv]
    out = {}
    for i, ci in enumerate(piv):
        if all(A[i][fc] == 0 for fc in free):
            out[mons[ci]] = B[i]
    return out, len(piv)


def monomial_columns(F, mons, fixed, var_idx):
    """For every monomial, the vector of its values as the key words in
    `var_idx` run over F_q^{len(var_idx)} (the others are `fixed`)."""
    q = F.q
    t = len(var_idx)
    N = q ** t
    grids = []
    for a in range(t):
        rep = q ** (t - 1 - a)
        g = np.repeat(np.arange(q, dtype=np.int64), rep)
        grids.append(np.tile(g, N // len(g)))
    pw = []
    for a, v in enumerate(var_idx):
        maxe = max(m[v] for m in mons)
        col = [np.ones(N, dtype=np.int64)]
        for _ in range(maxe):
            col.append(F.vmul(col[-1], grids[a]))
        pw.append(col)
    cols = []
    for m in mons:
        acc = None
        for i in range(4):
            if i in var_idx or m[i] == 0:
                continue
            c = np.full(N, _pow(F, fixed[i], m[i]), dtype=np.int64)
            acc = c if acc is None else F.vmul(acc, c)
        for a, v in enumerate(var_idx):
            if m[v]:
                acc = pw[a][m[v]] if acc is None else F.vmul(acc, pw[a][m[v]])
        cols.append(acc if acc is not None else np.ones(N, dtype=np.int64))
    return cols


def _pow(F, base, e):
    r = 1
    for _ in range(e):
        r = F.mul(r, base)
    return r


def brute_force_free_words(F, rows, mons, known, free, n_rows=6, budget_log2=24,
                           max_out=8):
    """Enumerate the undetermined key words over F_q^{|free|} and keep the
    values satisfying the first `n_rows` linearised equations."""
    t = len(free)
    if t * np.log2(F.q) > budget_log2:
        return None
    N = F.q ** t
    fixed = [known.get(i, 0) for i in range(4)]
    cols = monomial_columns(F, mons, fixed, free)
    alive = np.ones(N, dtype=bool)
    for row in rows[:n_rows]:
        acc = np.zeros(N, dtype=np.int64)
        for coeff, col in zip(row, cols):
            if coeff:
                acc = F.vadd(acc, F.vmul(col, np.full(N, coeff, dtype=np.int64)))
        alive &= (acc == 0)
        if alive.sum() <= 1:
            break
    idx = np.nonzero(alive)[0]
    out = []
    for v in idx[:max_out]:
        vals = []
        rem = int(v)
        for a in range(t):
            rep = F.q ** (t - 1 - a)
            vals.append((rem // rep) % F.q)
        out.append(vals)
    return out


# ------------------------------------------------------- sequential mode ---
SEQUENTIAL_STEPS = ((2, (0, 3)), (1, (2,)), (3, (1,)))
"""The order attack_1round.py uses when W is balanced on all four coordinates
(`1111`): coordinate 2 (degree 2) is affine in (k0, k3); coordinate 1 is then
affine in k2; coordinate 3 is then affine in k1 -- the k1^2 term carries the
factor sum_P a(P+k), which is 0 exactly because coordinate 2 is balanced.

This is the `1111` case that Lemma O8 says coordinates 0/1/2 alone cannot
solve: k1 is invisible in coordinates 1 and 2, and in coordinate 0 it appears
only through -k1 * sum_P a(P+k) = 0.  Coordinate 3 is the only one that sees
it, so the fix is simply to use it -- either by linearising it (37/66
monomials, `--coords 0,1,2,3`) or, far more cheaply, by substituting the
already recovered words first, which is what this mode does: 2 structures and
a handful of field operations instead of 41."""


def _subst(F, ke, known, target):
    """Split a key monomial into (exponent of `target`, product of the known
    words); returns None if it involves an unknown word other than `target`."""
    acc, deg = 1, 0
    for i, e in enumerate(ke):
        if not e:
            continue
        if i == target:
            deg = e
        elif i in known:
            for _ in range(e):
                acc = F.mul(acc, known[i])
        else:
            return None
    return deg, acc


def _roots(F, poly):
    """All roots in F_q of a univariate polynomial given as {degree: coeff}."""
    if not any(poly.values()):
        return None                      # identically zero: no information
    deg = max(d for d, v in poly.items() if v)
    if deg == 1:
        return [F.mul(F.sub(0, poly.get(0, 0)), F.inv(poly[1]))]
    x = np.arange(F.q, dtype=np.int64)
    acc = np.zeros(F.q, dtype=np.int64)
    for d, v in poly.items():
        if not v:
            continue
        t = np.full(F.q, v, dtype=np.int64)
        for _ in range(d):
            t = F.vmul(t, x)
        acc = F.vadd(acc, t)
    return [int(v) for v in np.nonzero(acc == 0)[0]]


def sequential_solve(F, coeffs, steps=SEQUENTIAL_STEPS):
    """Recover one block's key words by substitution instead of linearisation.

    `coeffs[co][s]` is {key-monomial: sum_pe coeff * M_s[pe]} for coordinate co
    and structure s, i.e. one assembled equation.  Returns (key list with None
    for anything undetermined, per-step diagnostics)."""
    known, info = {}, []
    for co, targets in steps:
        rows = coeffs.get(co)
        if not rows:
            info.append({"coord": co, "targets": list(targets), "status": "no data"})
            continue
        if len(targets) > 1:
            R, B = [], []
            for cf in rows:
                r, b, ok = [0] * len(targets), 0, True
                for ke, v in cf.items():
                    if not v:
                        continue
                    sup = [i for i in range(4) if ke[i]]
                    if not sup:
                        b = F.sub(b, v)
                    elif len(sup) == 1 and ke[sup[0]] == 1 and sup[0] in targets:
                        t = targets.index(sup[0])
                        r[t] = F.add(r[t], v)
                    elif all(i in known for i in sup):
                        sp = _subst(F, ke, known, -1)
                        b = F.sub(b, F.mul(v, sp[1]))
                    else:
                        ok = False
                if ok:
                    R.append(r)
                    B.append(b)
            det, rank = determined_values(
                F, R, B, [tuple(1 if t == i else 0 for t in range(4)) for i in targets])
            for i in targets:
                e = tuple(1 if t == i else 0 for t in range(4))
                if e in det:
                    known[i] = det[e]
            info.append({"coord": co, "targets": list(targets), "rank": rank,
                         "solved": [i for i in targets if i in known]})
        else:
            i = targets[0]
            cand = None
            for cf in rows:
                poly = {}
                bad = False
                for ke, v in cf.items():
                    if not v:
                        continue
                    sp = _subst(F, ke, known, i)
                    if sp is None:
                        bad = True
                        break
                    d, c = sp
                    poly[d] = F.add(poly.get(d, 0), F.mul(v, c))
                if bad:
                    continue
                r = _roots(F, poly)
                if r is None:
                    continue
                cand = r if cand is None else [v for v in cand if v in set(r)]
                if cand is not None and len(cand) == 1:
                    break
            if cand and len(cand) == 1:
                known[i] = cand[0]
            info.append({"coord": co, "targets": [i],
                         "candidates": None if cand is None else len(cand),
                         "solved": [i] if i in known else []})
    return [known.get(i) for i in range(4)], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^8",
                    help="dux-*/toy-* or yu2x-*/yupx-*/yuxtoy-* (dux.registry)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--active", default="0")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--coords", default="0,3", help="usable S-box coordinates")
    ap.add_argument("--sequential", action="store_true",
                    help="W balanced on all four coordinates (`1111`): recover "
                         "(k0,k3) from coordinate 2, then k2 from coordinate 1, "
                         "then k1 from coordinate 3, by substitution (O8). Needs "
                         "--coords 1,2,3 (or 0,1,2,3) and only 2-3 structures.")
    ap.add_argument("--structures", type=int, default=None,
                    help="default: |monomials| + 4 spare")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = get_cipher(a.instance)
    fam = cipher_family(a.instance)
    F = c.F
    dim = (F.n if F.char == 2 else None) if a.dim is None else a.dim
    active = [int(v) for v in a.active.split(",")]
    coords = [int(v) for v in a.coords.split(",")]
    rng = np.random.default_rng(a.seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)

    # one equation template per usable coordinate
    tmpl = {co: equation_template(F, c.alpha, co, fam) for co in coords}
    mons = sorted({m for co in coords for m in tmpl[co][0]})
    needed = sorted({pe for co in coords for lst in tmpl[co][1].values() for pe, _ in lst})
    n_eq_per_struct = len(coords)
    if a.sequential:
        n_struct = a.structures or 3
    else:
        n_struct = a.structures or (len(mons) // n_eq_per_struct + 4)
    print(f"{a.instance}: {a.rounds} rounds, coordinates {coords}, "
          f"{len(mons)} monomials in k, {len(needed)} plaintext moments, "
          f"{n_struct} structures x {n_eq_per_struct} equations")

    t0 = time.time()
    rows = {j: [] for j in range(4)}
    rhs = {j: [] for j in range(4)}
    seq = {j: {} for j in range(4)}
    data_log2_each = None
    for s in range(n_struct):
        moments, data_log2_each = structure_moments(
            c, rks, a.rounds, active, dim, a.seed + 1000 + s, needed, a.chunk)
        for j in range(4):
            for co in coords:
                _, terms = tmpl[co]
                row = []
                for m in mons:
                    v = 0
                    for pe, coeff in terms.get(m, []):
                        v = F.add(v, F.mul(coeff, moments[j][pe]))
                    row.append(v)
                rows[j].append(row)
                rhs[j].append(0)
                seq[j].setdefault(co, []).append(
                    {m: row[t] for t, m in enumerate(mons) if row[t]})
        if (s + 1) % 5 == 0 or s == n_struct - 1:
            print(f"  {s + 1}/{n_struct} structures  ({time.time() - t0:.1f}s)")

    # the system is homogeneous; pin the constant monomial (all-zero exponent)
    # to 1, then read the key off the degree-one monomials.  The linearised
    # system is usually rank-deficient (columns of different monomials are
    # fixed combinations of the same moments), so we report which monomials it
    # DOES pin down and enumerate the remaining key word over F_q.
    const = (0, 0, 0, 0)
    cols_all = [m for m in mons if m != const]
    rec, ranks, bruted = [], [], []
    if a.sequential:
        seq_info = []
        for j in range(4):
            kj, info = sequential_solve(F, seq[j])
            seq_info.append({"block": j, "steps": info})
            rec += kj
            print(f"  block {j} (sequential): k = {kj}  "
                  f"(true {list(K[4 * j:4 * j + 4])})")
            for st in info:
                print(f"      coord {st['coord']} -> "
                      f"k{','.join(map(str, st['targets']))}: {st}")
        success = tuple(rec) == tuple(K)
        total_log2 = data_log2_each + float(np.log2(n_struct))
        print(f"recovered rk^0 == master key: {success};  data "
              f"2^{total_log2:.1f} chosen ciphertexts;  {time.time() - t0:.1f}s")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fn = os.path.join(a.out, f"sequential_{a.instance}_r{a.rounds}_"
                                     f"{'_'.join(map(str, active))}.json")
            json.dump({"instance": a.instance, "rounds": a.rounds,
                       "active": active, "dim": dim, "coords": coords,
                       "mode": "sequential", "structures": n_struct,
                       "steps": seq_info, "success": bool(success),
                       "data_log2_per_structure": data_log2_each,
                       "data_log2_total": round(total_log2, 2),
                       "elapsed_s": round(time.time() - t0, 1)},
                      open(fn, "w"), indent=1)
            print("saved", fn)
        return
    for j in range(4):
        R, B = [], []
        for row in rows[j]:
            b = 0
            if const in mons:
                b = F.sub(0, row[mons.index(const)])
            R.append([v for t, v in enumerate(row) if mons[t] != const])
            B.append(b)
        det, rank = determined_values(F, R, B, cols_all)
        ranks.append(rank)
        kj = [det.get(tuple(1 if t == i else 0 for t in range(4))) for i in range(4)]
        free = [i for i in range(4) if kj[i] is None]
        if free:
            known = {i: kj[i] for i in range(4) if kj[i] is not None}
            cand = brute_force_free_words(F, rows[j], mons, known, free)
            bruted.append({"block": j, "free_words": free,
                           "candidates": None if cand is None else len(cand),
                           "enumeration_log2": round(len(free) * float(np.log2(F.q)), 1)})
            if cand and len(cand) == 1:
                for slot, i in enumerate(free):
                    kj[i] = cand[0][slot]
        rec += kj
        print(f"  block {j}: rank {rank}/{len(cols_all)}; linear part fixes "
              f"{[i for i in range(4) if kj[i] is not None and i not in free]}, "
              f"enumerated {free}; k = {kj}  (true {list(K[4 * j:4 * j + 4])})")
    success = tuple(rec) == tuple(K)
    total_log2 = data_log2_each + float(np.log2(n_struct))
    print(f"recovered rk^0 == master key: {success};  data 2^{total_log2:.1f} chosen "
          f"ciphertexts;  {time.time() - t0:.1f}s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"partial_{a.instance}_r{a.rounds}_"
                                 f"{'_'.join(map(str, active))}_c{a.coords.replace(',', '')}.json")
        json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                   "dim": dim, "coords": coords, "monomials": len(mons),
                   "structures": n_struct, "data_log2_per_structure": data_log2_each,
                   "data_log2_total": round(total_log2, 2), "ranks": ranks,
                   "brute_forced": bruted,
                   "success": success, "elapsed_s": round(time.time() - t0, 1)},
                  open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
