"""E07 / W13-A -- how much of the r_KR = 2 rank is decided by the template?

Every equation row of the structured linearisation is a FIXED linear image of
the structure's moment vector:

    row_{s,j}  =  Phi_j . m_s ,

with m_s the point sums of structure s (`assemble_fast.Moments`: the single
block moments pm[b], nP entries each, and the block-pair moments Mom[A][B],
nP x nP each) and Phi_j -- the outer block j -- built once and for all from
`Precomp` (the joint expansion C[c], the linear-layer rows A/B/W2 and the
scatter maps col_pair / col_single / col_kT).  Nothing in Phi_j depends on the
structure.  Hence

    rank(A)  <=  rank([Phi_0 | Phi_1 | Phi_2 | Phi_3])  =:  rank(Phi_stack)

with equality as soon as the moment vectors of the structures used span the
moment space.  results/E07 measures rank_max = 8 608 (F_65537, M = 38 051) and
4 756 (F_{2^n}, M = 18 025) and observes N_min = rank_max / 4; O8(b) says the
rank defect of r_KR = 1 "comes mostly from the template".  This script decides
whether the same is true for r_KR = 2, by feeding PSEUDO-structures -- moment
vectors that are not the moments of any particular structure -- to the same
assembly and looking at where the rank saturates.

Two generators, `--moments`:

  uniform      every entry of pm and Mom independent uniform, subject only to
               the symmetries the assembly needs to stay well defined:
                 * Mom[B][A] = Mom[A][B]^T and Mom[A][A] symmetric;
                 * pm[b][j0] = 0 for the all-zero plaintext exponent, i.e.
                   sum_P 1 = 0.  This one is not cosmetic: it is O5 (a full
                   product-set structure has q^s == 0 mod p points) and it is
                   what makes the pure-key monomials drop out of the equation.
                   Without it the assembly raises, because those monomials are
                   not in the linearisation set at all.
               This is the loosest bound: it ignores every other identity a
               real moment vector satisfies.

  realizable   m = sum_i c_i mu(x_i) over random points x_i in F_q^16 and
               random coefficients c_i with sum_i c_i = 0, where mu(x) is the
               moment vector of the single point x.  Every real structure's
               moment vector is of this form (c_i = 1, sum = q^s = 0), so this
               generator ranges exactly over the span V_0 of realizable signed
               measures of total weight 0 -- it automatically satisfies every
               identity real moments satisfy (the Hankel collapse of Mom[A][A],
               Mom[A][B][j0][.] = pm[B][.], ...).  rank(Phi|V_0) is therefore
               the sharp template bound.

Usage
  python rank_phi.py --instance dux-65537 --structures 3000 \
      --checkpoints 1000,2000,2152,2200,2400,3000 --out ../../results/...json
  python rank_phi.py --instance dux-2^16 --structures 1600 \
      --checkpoints 800,1189,1200,1400,1600
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fast"))

from dux import DuX                                        # noqa: E402
from dux.registry import cipher_family, get_cipher          # noqa: E402
from attack_2round_toy import linear_row                    # noqa: E402
from assemble_fast import Precomp, _pow_matrix, rows_from_moments  # noqa: E402
from assemble_fast_2n import rows_from_moments_2n           # noqa: E402
import gf2n_solve                                           # noqa: E402
import modp_solve                                           # noqa: E402


# ------------------------------------------------------------ pseudo-moments
class PseudoMoments:
    """Duck-typed stand-in for `assemble_fast.Moments`."""

    def __init__(self, pm, Mom):
        self.pm, self.Mom, self.npoints = pm, Mom, 0


def _zero_pe_index(pre):
    z = np.flatnonzero((pre.pe_arr == 0).all(axis=1))
    assert z.size == 1, "the all-zero plaintext exponent must occur exactly once"
    return int(z[0])


def uniform_moments(pre, rng):
    p, nP = pre.p, pre.pe_arr.shape[0]
    j0 = _zero_pe_index(pre)
    pm, Mom = {}, {}
    for b in pre.unknown:
        v = rng.integers(0, p, size=nP).astype(np.int64)
        v[j0] = 0                      # sum_P 1 = q^s = 0   (O5)
        pm[b] = v
    for i, A in enumerate(pre.unknown):
        for B in pre.unknown[i:]:
            m = rng.integers(0, p, size=(nP, nP)).astype(np.int64)
            if A == B:
                m = np.triu(m) + np.triu(m, 1).T
                m %= p
            Mom[(A, B)] = m
            if A != B:
                Mom[(B, A)] = m.T.copy()
    return PseudoMoments(pm, Mom)


def realizable_moments(pre, rng, npts):
    """m = sum_i c_i mu(x_i), sum_i c_i = 0 -- the span of real structures."""
    F, p, nP = pre.F, pre.p, pre.pe_arr.shape[0]
    P = [rng.integers(0, F.q, size=npts).astype(np.int64) for _ in range(16)]
    c = rng.integers(0, p, size=npts).astype(np.int64)
    c[0] = (-int(c[1:].sum())) % p if p != 2 else c[0]
    if F.char == 2:
        c[0] = np.bitwise_xor.reduce(c[1:])
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    pm, Mom = {}, {}
    if F.char == 2:
        cw = {b: F.vmul(PW[b], c[None, :]) for b in pre.unknown}
        for b in pre.unknown:
            pm[b] = np.bitwise_xor.reduce(cw[b], axis=1)
        for i, A in enumerate(pre.unknown):
            for B in pre.unknown[i:]:
                m = np.zeros((nP, nP), dtype=np.int64)
                for r in range(nP):
                    m[r] = np.bitwise_xor.reduce(F.vmul(PW[B], cw[A][r][None, :]),
                                                 axis=1)
                Mom[(A, B)] = m
                if A != B:
                    Mom[(B, A)] = m.T.copy()
        return PseudoMoments(pm, Mom)
    cw = {b: (PW[b] * c[None, :]) % p for b in pre.unknown}
    for b in pre.unknown:
        pm[b] = cw[b].sum(axis=1) % p
    for i, A in enumerate(pre.unknown):
        for B in pre.unknown[i:]:
            m = (cw[A].astype(np.float64) @ PW[B].astype(np.float64).T)
            m = np.asarray(m, dtype=np.int64) % p
            Mom[(A, B)] = m
            if A != B:
                Mom[(B, A)] = m.T.copy()
    return PseudoMoments(pm, Mom)


# --------------------------------------------------------- incremental rank
class IncrementalRankFp:
    """Row-by-row rank over F_p, so every prefix rank is available for free.

    The basis is kept in reduced row echelon form; adding a batch of k rows is
    two BLAS products, `R -= R[:, pivots] @ Basis` (forward) and
    `Basis -= Basis[:, new_pivots] @ R_new` (back substitution).  Entries stay
    below p, so the accumulators (at most rank * p^2 = 2^45 for p = 65537 and
    rank <= 9 000) are exact in float64."""

    def __init__(self, p, ncols, cap, colchunk=8192):
        self.p, self.ncols, self.colchunk = p, ncols, colchunk
        self.B = np.zeros((cap, ncols), dtype=np.float64)
        self.piv = []
        self.r = 0

    def add(self, rows):
        p = self.p
        R = np.asarray(rows, dtype=np.float64) % p
        if self.r:
            coef = R[:, np.asarray(self.piv, dtype=np.int64)].copy()
            R -= coef @ self.B[:self.r]
            R %= p
        new, newpiv = [], []
        for i in range(R.shape[0]):
            row = R[i]
            nz = np.flatnonzero(row)
            if nz.size == 0:
                continue
            c = int(nz[0])
            row = (row * pow(int(row[c]), p - 2, p)) % p
            if i + 1 < R.shape[0]:
                f = R[i + 1:, c].copy()
                R[i + 1:] -= np.outer(f, row)
                R[i + 1:] %= p
            for t in range(len(new)):
                if new[t][c]:
                    new[t] = (new[t] - new[t][c] * row) % p
            new.append(row)
            newpiv.append(c)
        if new:
            Rn = np.asarray(new)
            if self.r:
                pn = np.asarray(newpiv, dtype=np.int64)
                co = self.B[:self.r][:, pn].copy()
                for c0 in range(0, self.ncols, self.colchunk):
                    c1 = min(c0 + self.colchunk, self.ncols)
                    blk = self.B[:self.r, c0:c1]
                    blk -= co @ Rn[:, c0:c1]
                    blk %= p
            if self.r + len(new) > self.B.shape[0]:
                grow = max(self.B.shape[0], self.r + len(new) - self.B.shape[0])
                self.B = np.vstack([self.B, np.zeros((grow, self.ncols))])
            self.B[self.r:self.r + len(new)] = Rn
            self.piv += newpiv
            self.r += len(new)
        return self.r


# ------------------------------------------------------------------ driver
def run(instance, structures, checkpoints, seed, moments, npts, rounds,
        progress=0, cap=None, combine=None, unknown=(0, 1, 2, 3),
        one_shot=False):
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    fam = cipher_family(instance)
    pre = Precomp(F, c.alpha, list(unknown), [0, 1, 2, 3], cipher=fam)
    lrow = linear_row(c, 0)
    ycomb = None
    if combine:
        sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
        from cheap_rows import block_coeffs
        from combine_helper import field_key
        ycomb = block_coeffs(fam, field_key(F), "dec", combine)
    rng = np.random.default_rng(seed)
    M = len(pre.mons)
    gen = (lambda: uniform_moments(pre, rng)) if moments == "uniform" else \
          (lambda: realizable_moments(pre, rng, npts))
    asm0 = rows_from_moments_2n if F.char == 2 else rows_from_moments
    if ycomb is None:
        asm = asm0
        nrows = 4
    else:
        from assemble_fast import combine_rows

        def asm(pre_, mom_, lrow_):
            rs = asm0(pre_, mom_, lrow_)
            return [combine_rows(rs, y, F) for y in ycomb]
        nrows = len(ycomb)
    curve, t0 = [], time.time()

    if one_shot:
        # S9 (1e) / S10 (1a): only the SATURATED rank is wanted, not the curve.
        # `IncrementalRankFp` pays O(N * batch * ncols) element-wise modulo to
        # keep every prefix rank available, which at the 12 000-pseudo-structure
        # scale of the combined-row ceiling costs hours; one elimination of the
        # whole stack costs one solve (minutes) and answers the same question.
        rows = []
        for _s in range(structures):
            rows.extend(asm(pre, gen(), lrow))
            if progress and (_s + 1) % progress == 0:
                print(f"    .. {_s + 1}/{structures} pseudo-structures "
                      f"({time.time() - t0:.0f} s)", flush=True)
        R = np.asarray(rows, dtype=np.int64)
        del rows
        t1 = time.time()
        rhs = np.zeros(R.shape[0], dtype=np.int64)
        if F.char == 2:
            _det, rank, _free = gf2n_solve.solve(F, R, rhs)
        else:
            _det, rank, _free = modp_solve.solve(pre.p, R, rhs)
        curve.append({"structures": structures, "rows": int(R.shape[0]),
                      "rank": rank, "seconds": round(time.time() - t0, 1),
                      "solve_s": round(time.time() - t1, 1), "one_shot": True})
        print(f"  N = {structures:5d}  rows = {R.shape[0]:6d}  rank = {rank:6d}"
              f"   ({time.time() - t0:.0f} s, one shot)", flush=True)
    elif F.char == 2:
        acc = np.zeros((nrows * structures, M), dtype=np.uint16)
        n = 0
        for s in range(structures):
            acc[nrows * n:nrows * (n + 1)] = np.asarray(asm(pre, gen(), lrow),
                                                        dtype=np.uint16)
            n += 1
            if n in checkpoints:
                _, rank, nfree = gf2n_solve.solve(
                    F, acc[:nrows * n], np.zeros(nrows * n, dtype=np.int64))
                curve.append({"structures": n, "rows": nrows * n, "rank": rank,
                              "seconds": round(time.time() - t0, 1)})
                print(f"  N = {n:5d}  rows = {nrows*n:6d}  rank = {rank:6d}"
                      f"   ({time.time()-t0:.0f} s)", flush=True)
    else:
        cap0 = cap or min(M, nrows * structures)
        inc = IncrementalRankFp(pre.p, M, cap=min(M, cap0) + 64)
        batch, n = [], 0
        for s in range(structures):
            batch += list(asm(pre, gen(), lrow))
            n += 1
            if len(batch) >= 256 or n == structures or n in checkpoints:
                inc.add(np.asarray(batch, dtype=np.int64))
                batch = []
            if n in checkpoints:
                curve.append({"structures": n, "rows": nrows * n, "rank": inc.r,
                              "seconds": round(time.time() - t0, 1)})
                print(f"  N = {n:5d}  rows = {nrows*n:6d}  rank = {inc.r:6d}"
                      f"   ({time.time()-t0:.0f} s)", flush=True)
            elif progress and n % progress == 0:
                print(f"    .. N = {n}, rank = {inc.r}, "
                      f"{time.time()-t0:.0f} s", flush=True)
    return {"instance": instance, "cipher": fam, "q": F.q, "char": F.char,
            "monomials": M, "combine": combine, "unknown_blocks": list(unknown),
            "rows_per_structure": nrows, "moments": moments, "npts": npts if moments == "realizable" else None,
            "seed": seed, "curve": curve,
            "saturated_rank": max(r["rank"] for r in curve) if curve else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--structures", type=int, default=3000)
    ap.add_argument("--checkpoints", default="1000,2000,2152,2200,2400,3000")
    ap.add_argument("--moments", choices=["uniform", "realizable"],
                    default="uniform")
    ap.add_argument("--points", type=int, default=200,
                    help="number of random points per pseudo-structure "
                         "(--moments realizable only)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--unknown-blocks", default="0,1,2,3",
                    help="inner blocks treated as unknown (fewer = a much "
                         "smaller monomial set, for calibration)")
    ap.add_argument("--combine", default=None,
                    help="O10: fold the four per-outer-block rows with this "
                         "pattern's cheap-row kernel before taking the rank")
    ap.add_argument("--cap", type=int, default=None,
                    help="initial basis capacity (rows); it grows if needed, "
                         "so this only trades memory against one copy")
    ap.add_argument("--one-shot", action="store_true",
                    help="skip the incremental curve: build all rows, then one "
                         "elimination.  Same saturated rank, minutes instead of "
                         "hours, but no intermediate checkpoints.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cps = sorted({int(v) for v in a.checkpoints.split(",")})
    print(f"{a.instance}: rank(Phi_stack) from {a.moments} pseudo-structures "
          f"(seed {a.seed})")
    unknown = [int(v) for v in a.unknown_blocks.split(",")]
    res = run(a.instance, a.structures, set(cps), a.seed, a.moments, a.points,
              a.rounds, a.progress, a.cap, a.combine, unknown, a.one_shot)
    print(f"saturated rank = {res['saturated_rank']}  "
          f"(monomials M = {res['monomials']})")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
