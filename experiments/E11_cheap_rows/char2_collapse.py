"""E11 / W19-B -- why the dim K = 1 combined row collapses in characteristic 2.

S10-B measured the fact: the O10 combined row of the DuX pattern `0001`
(dim K = 1) has a pseudo-structure ceiling of 2 178 over F_{2^n} -- 45.8 % of
the four-row ceiling 4 756, and INDEPENDENT of n -- while over F_p it keeps
8 198 of 8 608, i.e. 95.2 %.  The 12-round DuX(65537) run got its master key
out of the F_p system; the toy-2^4 char-2 system pinned 1 122 monomials and
0 of the 16 key words.  This script establishes the mechanism and measures the
ceiling-level consequence.

THE SYMMETRY.  `assemble_fast.OuterEquation.rows` builds outer block j's
coefficient of the inner word s as `lrow[(s - 4j - off) mod 16]`, so replacing
(j, s) by (j+1, s+4) leaves it unchanged: outer block j+1's equation is outer
block j's equation with every inner block relabelled b -> b+1 and the outer
key slot relabelled j -> j+1.  Writing

    rho   = the block rotation acting on the MOMENT vector  ((rho m)_b = m_{b+1})
    pi    = the same rotation acting on the linearisation COLUMNS
            ((b, u) -> (b+1, u) inside a monomial, and (j, i) -> (j+1, i))

that is exactly

    Phi_j  =  pi^j . Phi_0 . rho^j                                          (1)

(`--identity-trials` checks (1) numerically in whichever field is selected).

THE COMBINED MAP.  The O10 row for a kernel vector y is Psi_y = sum_j y_j Phi_j,
so by (1)

    Psi_y  =  sum_j y_j . pi^j . Phi_0 . rho^j ,                            (2)

which is a RELATIVE TRACE over Z_4 twisted by y -- and relative traces behave
completely differently depending on whether |Z_4| = 4 is invertible.

  * F_p, p odd (the DuX(65537) / toy-257 case).  The `0001` kernel vector is
    y_j = chi(j) = (-1)^j, a NONTRIVIAL character of Z_4, and 4 is invertible.
    Decomposing the moment space into isotypic components V_0^psi (all four
    characters are F_p-rational whenever p = 1 mod 4, e.g. 257 and 65537),
    rho^j acts on V_0^psi as psi(j), so

        Psi|_{V_0^psi}  =  ( sum_j (chi.psi)(j) pi^j ) . Phi_0
                        =  4 . P_{(chi.psi)^-1} . Phi_0 ,                   (3)

    with P a PROJECTOR onto one pi-eigenspace.  Nothing is annihilated: each
    source component keeps one of the four character parts of Phi_0's image,
    which is why the ceiling stays at 95 %.

  * F_{2^n}.  Over F_2 the only kernel vector is y = (1,1,1,1) and 4 = 0, so
    (2) is the plain trace.  Put tau = 1 + pi and sigma = 1 + rho; both are
    NILPOTENT of order 4 (char 2, |Z_4| = 4).  Expanding pi^j = (1+tau)^j and
    rho^j = (1+sigma)^j, Lucas' theorem gives C(j,a) = [a submask of j] mod 2
    for j <= 3, so sum_{j=0}^{3} C(j,a) C(j,b) = #{j : (a|b) submask of j} =
    2^(2 - popcount(a|b)), which is odd exactly when a|b = 3 (bitwise OR):

        Psi  =  sum_{a | b = 3, a,b <= 3}  tau^a . Phi_0 . sigma^b .        (4)

    EVERY term sits at depth a + b >= 3 of the tau/sigma filtration (a|b = 3
    forces it), and the ONLY term with b = 0 is tau^3 . Phi_0.  In particular
    on the rho-invariant part of the moment space (sigma m = 0) it degenerates
    to Psi = tau^3 . Phi_0 = (sum_j pi^j) . Phi_0, whose image is the space of
    pi-ORBIT SUMS -- dimension = the number of pi-orbits, and every one of the
    16 degree-1 key columns appears there only through the orbit sum
    k_c + k_{4+c} + k_{8+c} + k_{12+c}.  (`--identity-trials` checks (4) too.)

WHAT THIS SCRIPT MEASURES.  Two things the mechanism predicts but does not by
itself pin down:

  (a) the number of pi-orbits, i.e. how much of the ceiling the orbit-sum term
      of (4) could possibly account for;
  (b) the CEILING-level determinacy: build the combined system from pseudo
      structures (`rank_phi.realizable_moments`, the sharp template bound
      rank(Phi|V_0)), pin the normalisation columns, and ask which columns the
      row space determines -- in particular the 16 degree-1 key columns.
      A negative answer at the ceiling is stronger than S10-B's measurement on
      one real system: no real structure set can do better than V_0.

Usage
  python experiments/E11_cheap_rows/char2_collapse.py --instance toy-2^4 \
      --rounds 6 --combine 0001 --structures 6000 --points 256 \
      --out results/E11_cheap_rows/collapse_toy-2^4.json
  python experiments/E11_cheap_rows/char2_collapse.py --instance toy-257 \
      --rounds 7 --combine 0001 --structures 8700 --points 256 \
      --out results/E11_cheap_rows/collapse_toy-257.json
  python experiments/E11_cheap_rows/char2_collapse.py --instance yu2x-16 \
      --rounds 10 --cipher yux --combine 1110 --identity-only
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round", "fast"))

from dux.registry import cipher_family, get_cipher            # noqa: E402
from assemble_fast import Precomp, combine_rows, rows_from_moments  # noqa: E402
from assemble_fast_2n import rows_from_moments_2n             # noqa: E402
from attack_2round_toy import linear_row                      # noqa: E402
from cheap_rows import block_coeffs                           # noqa: E402
from nmin_scan import pinned_columns                          # noqa: E402
import gf2n_solve                                             # noqa: E402
import modp_solve                                             # noqa: E402
import rank_phi as RP                                         # noqa: E402


# ------------------------------------------------------ the two rotations --
def column_rotation(pre, k=1):
    """pi^k on the linearisation columns: inner block b -> b+k inside every
    monomial, and the outer key slot j -> j+k."""
    perm = np.empty(len(pre.mons), dtype=np.int64)
    for i, m in enumerate(pre.mons):
        slot, inner = m
        s2 = None if slot is None else ((slot[0] + k) % 4, slot[1])
        i2 = tuple(sorted(((b + k) % 4, u) for b, u in inner))
        perm[i] = pre.idx[(s2, i2)]
    return perm


def apply_perm(perm, v):
    """(pi v)[perm[i]] = v[i] -- the push-forward, matching column_rotation."""
    out = np.zeros_like(v)
    out[perm] = v
    return out


def rotate_moments(mom, k):
    """rho^k on a moment vector: (rho m)_b = m_{b+1}."""
    pm = {b: mom.pm[(b + k) % 4] for b in mom.pm}
    Mom = {(A, B): mom.Mom[((A + k) % 4, (B + k) % 4)] for (A, B) in mom.Mom}
    return RP.PseudoMoments(pm, Mom)


def tau_valuation(y):
    """The tau-adic valuation of  y^(t) = sum_j y_j t^j  in F_2[Z_4].

    Characteristic 2 with |Z_4| = 4 makes  F_2[t]/(t^4 - 1) = F_2[t]/((t+1)^4)
    a LOCAL ring with maximal ideal (tau), tau = t + 1: every element is
    tau^v times a unit.  So the combined map restricted to the rho-invariant
    moments, y^(pi) . Phi_0, has  rank(y^(pi)) = rank(tau^v), and on a pi-orbit
    of length L (a free F_2[t]/((t+1)^L)-module) tau^e has rank max(0, L - e).
    v = 3 -- which for the all-ones vector is the largest possible -- leaves
    exactly ONE coefficient per length-4 orbit, i.e. only the ORBIT SUM."""
    poly = [int(v) & 1 for v in y]
    v = 0
    while v < 4:
        # divide by (t + 1) in F_2[t]/(t^4 - 1): synthetic division works iff
        # the sum of the coefficients is 0 (i.e. t = 1 is a root).
        if sum(poly) % 2:
            return v
        q, carry = [0] * 4, 0
        for j in range(3, -1, -1):          # poly = (t+1) * q  in F_2[t]/(t^4-1)
            carry ^= poly[j]
            q[j] = carry
        poly = [q[(j + 1) % 4] for j in range(4)]
        v += 1
    return 4


def tau_rank(sizes, e):
    """rank(tau^e) on the column space, from the pi-orbit length histogram."""
    return int(sum(cnt * max(0, int(L) - e) for L, cnt in sizes.items()))


def orbit_stats(perm):
    n = len(perm)
    seen = np.zeros(n, dtype=bool)
    sizes = {}
    for i in range(n):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True
            ln += 1
            j = int(perm[j])
        sizes[ln] = sizes.get(ln, 0) + 1
    return {"orbits": int(sum(sizes.values())),
            "sizes": {str(k): v for k, v in sorted(sizes.items())}}


# ------------------------------------------------- the operator identities --
def _rows(pre, mom, lrow, char2):
    asm = rows_from_moments_2n if char2 else rows_from_moments
    return [np.asarray(r, dtype=np.int64) for r in asm(pre, mom, lrow)]


def check_identities(pre, lrow, F, ycomb, trials, seed, npts):
    """(1) Phi_j = pi^j Phi_0 rho^j, and in characteristic 2 also
    (4) Psi = sum_{a|b=3} tau^a Phi_0 sigma^b (bitwise OR, Lucas)."""
    p = F.q if F.char == 2 else F.p
    rng = np.random.default_rng(seed)
    perms = {k: column_rotation(pre, k) for k in (1, 2, 3)}
    out = {"equivariance_trials": trials, "equivariance_ok": True,
           "char2_expansion_ok": None, "symmetrised_eigenvalue_ok": True,
           "symmetrised_eigenvalue": "+1 (pi-invariant orbit sums)"
                                     if F.char == 2 else
                                     "-1 (pi-anti-invariant)"}

    def red(v):
        return v % p if F.char != 2 else v

    def psi_of(mom):
        """The combined row Psi_y m = sum_j y_j Phi_j m."""
        rs = _rows(pre, mom, lrow, F.char == 2)
        return red(np.asarray(combine_rows(rs, ycomb[0], F), dtype=np.int64))

    def symmetrise(mom):
        """sigma^3 m = sum_j rho^j m -- the rho-INVARIANT part of the moment
        space, where (4) degenerates to tau^3 Phi_0 (char 2) and (3) to the
        projector for the trivial psi (F_p)."""
        pm, Mom = {}, {}
        for x in mom.pm:
            acc = mom.pm[x].copy()
            for j in (1, 2, 3):
                r = rotate_moments(mom, j).pm[x]
                acc = (acc ^ r) if F.char == 2 else (acc + r) % p
            pm[x] = acc
        for x in mom.Mom:
            acc = mom.Mom[x].copy()
            for j in (1, 2, 3):
                r = rotate_moments(mom, j).Mom[x]
                acc = (acc ^ r) if F.char == 2 else (acc + r) % p
            Mom[x] = acc
        return RP.PseudoMoments(pm, Mom)

    for _ in range(trials):
        mom = RP.realizable_moments(pre, rng, npts)
        rows = _rows(pre, mom, lrow, F.char == 2)
        for k in (1, 2, 3):
            r0 = _rows(pre, rotate_moments(mom, k), lrow, F.char == 2)[0]
            if not np.array_equal(red(apply_perm(perms[k], r0)), red(rows[k])):
                out["equivariance_ok"] = False

        if F.char == 2:
            # Psi = XOR_j row_j  vs  sum_{a|b=3} tau^a Phi_0 sigma^b.
            psi = rows[0].copy()
            for j in (1, 2, 3):
                psi = psi ^ rows[j]
            # sigma^b m: (1 + rho)^b applied to the moment vector, char 2.
            sig = {0: mom}
            for b in (1, 2, 3):
                prev = sig[b - 1]
                sig[b] = RP.PseudoMoments(
                    {x: prev.pm[x] ^ rotate_moments(prev, 1).pm[x] for x in prev.pm},
                    {x: prev.Mom[x] ^ rotate_moments(prev, 1).Mom[x]
                     for x in prev.Mom})
            acc = np.zeros_like(psi)
            for b in range(4):
                base = _rows(pre, sig[b], lrow, True)[0]        # Phi_0 sigma^b m
                for a in range(4):
                    if (a | b) != 3:
                        continue
                    v = base.copy()                              # tau^a = (1+pi)^a
                    for _t in range(a):
                        v = v ^ apply_perm(perms[1], v)
                    acc = acc ^ v
            if not np.array_equal(acc, psi):
                out["char2_expansion_ok"] = False
            elif out["char2_expansion_ok"] is None:
                out["char2_expansion_ok"] = True

        # The decisive contrast.  On the rho-invariant part of the moment space
        # the combined row is a pi-EIGENVECTOR: eigenvalue +1 in characteristic
        # 2 (an orbit-sum vector, so the four columns of every orbit carry the
        # same coefficient), eigenvalue -1 over F_p (the alternating sum).  The
        # F_p system reaches the other two eigenvalues from the psi != 1
        # isotypic components, which is what lets it separate an orbit at all.
        v = psi_of(symmetrise(mom))
        want = v if F.char == 2 else red(-v)
        if not np.array_equal(red(apply_perm(perms[1], v)), want):
            out["symmetrised_eigenvalue_ok"] = False
    return out


# --------------------------------------------------------- the ceiling run --
def degree_one_columns(pre):
    """The 16 columns carrying a single inner key word rk^0_{4b+c}."""
    cols = {}
    for b in range(4):
        for c in range(4):
            u = tuple(1 if t == c else 0 for t in range(4))
            m = (None, ((b, u),))
            if m in pre.idx:
                cols[f"{4 * b + c}"] = int(pre.idx[m])
    return cols


def ceiling_determinacy(pre, F, lrow, ycomb, structures, npts, seed, progress):
    p = F.q if F.char == 2 else F.p
    rng = np.random.default_rng(seed)
    char2 = F.char == 2
    nrows = len(ycomb) if ycomb else 4
    dt = np.uint16 if char2 else (np.int32 if p < 2 ** 31 else np.int64)
    acc = np.zeros((nrows * structures, len(pre.mons)), dtype=dt)
    t0 = time.time()
    for s in range(structures):
        rows = _rows(pre, RP.realizable_moments(pre, rng, npts), lrow, char2)
        comb = [combine_rows(rows, y, F) for y in ycomb] if ycomb else rows
        acc[nrows * s:nrows * (s + 1)] = np.asarray(comb)
        if progress and (s + 1) % progress == 0:
            print(f"  .. {s + 1}/{structures} pseudo-structures "
                  f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    known = pinned_columns(pre, normalise=True)
    keep = np.array([i for i in range(len(pre.mons)) if i not in known],
                    dtype=np.int64)
    kc = np.array(sorted(known), dtype=np.int64)
    kv = np.array([known[int(i)] for i in kc], dtype=np.int64)
    if char2:
        B = np.zeros(acc.shape[0], dtype=np.int64)
        for t, ci in enumerate(kc):
            B ^= F.vmul(acc[:, ci].astype(np.int64), np.int64(kv[t]))
    else:
        B = (-(acc[:, kc].astype(np.int64) @ kv)) % p
    nrows_total = int(acc.shape[0])
    sub = acc[:, keep]
    del acc                                  # the fancy index above is a copy
    if char2:
        det, rank, nfree = gf2n_solve.solve(F, sub, B)
    else:
        det, rank, nfree = modp_solve.solve(p, sub, B)
    solve_s = time.time() - t1
    back = {int(keep[j]): j for j in range(len(keep))}
    d1 = degree_one_columns(pre)
    determined = {w: (back.get(ci) in det) for w, ci in d1.items()}
    return {"structures": structures, "rows": nrows_total,
            "rank": int(rank), "free": int(nfree),
            "monomials": int(len(pre.mons)), "pinned_by_normalisation": len(known),
            "determined_monomials": len(det),
            "degree_one_key_columns_determined":
                sum(1 for v in determined.values() if v),
            "degree_one_detail": determined,
            "assemble_s": round(t1 - t0, 1), "solve_s": round(solve_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-2^4")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--combine", default="0001")
    ap.add_argument("--structures", type=int, default=6000)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--identity-trials", type=int, default=2)
    ap.add_argument("--identity-only", action="store_true",
                    help="stop after the operator identities and the orbit count")
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = get_cipher(a.instance, rounds=a.rounds)
    F = c.F
    fam = cipher_family(a.instance)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher=fam)
    lrow = linear_row(c, 0)
    fkey = f"2^{F.n}" if F.char == 2 else str(F.q)
    ycomb = (None if a.combine in ("none", "", None)
             else block_coeffs(fam, fkey, "dec", a.combine))
    print(f"{a.instance} ({fam}, {F.name}): M = {len(pre.mons)}, "
          + (f"combine {a.combine} -> dim K = {len(ycomb)}, y = {ycomb}"
             if ycomb else "the four plain rows (no --combine)"))

    res = {"instance": a.instance, "cipher": fam, "q": F.q, "char": F.char,
           "rounds": a.rounds, "combine": a.combine,
           "dim_K": len(ycomb) if ycomb else 4,
           "y": [[int(v) for v in y] for y in ycomb] if ycomb else None,
           "monomials": len(pre.mons), "seed": a.seed}
    res["orbits"] = orbit_stats(column_rotation(pre, 1))
    print(f"  pi-orbits on the {len(pre.mons)} columns: "
          f"{res['orbits']['orbits']} (sizes {res['orbits']['sizes']})")
    if F.char == 2 and ycomb:
        sizes = {int(k): v for k, v in res["orbits"]["sizes"].items()}
        res["kernel_vectors"] = []
        for y in ycomb:
            v = tau_valuation(y)
            res["kernel_vectors"].append(
                {"y": [int(t) for t in y], "tau_valuation": v,
                 "rank_on_rho_invariant_moments": tau_rank(sizes, v),
                 "orbit_sums_only": v == 3})
            print(f"    y = {[int(t) for t in y]}: tau-valuation {v}, "
                  f"rank(tau^{v}) = {tau_rank(sizes, v)}"
                  + ("   <-- ORBIT SUMS ONLY" if v == 3 else ""))
        res["char2_usable"] = any(k["tau_valuation"] < 3
                                  for k in res["kernel_vectors"])
    res["identities"] = (check_identities(pre, lrow, F, ycomb,
                                          a.identity_trials, a.seed, a.points)
                         if ycomb else {"skipped": "no --combine"})
    if ycomb:
        print(f"  Phi_j = pi^j Phi_0 rho^j: {res['identities']['equivariance_ok']}; "
              f"char-2 depth-3 expansion: {res['identities']['char2_expansion_ok']}; "
              f"symmetrised row is a pi-eigenvector "
              f"{res['identities']['symmetrised_eigenvalue']}: "
              f"{res['identities']['symmetrised_eigenvalue_ok']}")
    if not a.identity_only:
        res["ceiling"] = ceiling_determinacy(pre, F, lrow, ycomb, a.structures,
                                             a.points, a.seed + 1, a.progress)
        cl = res["ceiling"]
        print(f"  ceiling: rank {cl['rank']} of {cl['monomials']} columns, "
              f"{cl['determined_monomials']} monomials determined, "
              f"{cl['degree_one_key_columns_determined']}/16 key words")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
