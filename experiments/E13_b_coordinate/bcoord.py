"""E13 / W27 -- the CUBIC coordinate b of DuX's outer S-box as a second cheap
coordinate, in characteristic 2 only (R9 memo Sect. 1, T1).

WHY IT IS CHEAP.  The encryption S-box is S(x) = (c, b, a, y3) with

    a  = x2 - x0 x3 - alpha                 (deg 2)   -- the coordinate E07 uses
    b  = x1 - x3 a - alpha                  (deg 3)
       = x1 - x2 x3 + x0 x3^2 + alpha x3 - alpha ,

so in CHARACTERISTIC 2, where t -> t^2 is the Frobenius and hence additive,

    b  =  x1 + x2 x3 + x0 x3^2 + alpha x3 + alpha ,
    x3^2 = (u3 + k'_3)^2 = u3^2 + k'^2_3 ,    u3^2 = sum_t W3[t]^2 Y_t^2 ,

i.e. x0 x3^2 is still a product of TWO sums.  Linearising sum_P b therefore
needs one extra moment family, the "squared-second-factor" pair moments

    Mom2[A][B][e][f]  =  sum_P  P_A^e  (P_B^f)^2 ,

and no triple moments.  (The degree-5 coordinate c contains x0^2 x3^3 and
x0 x1 x3, which are genuinely trilinear, so b is the ONLY extra coordinate this
trick reaches.  Over F_p, x0 x3^2 is a true cubic and the trick does not
apply -- which is fine, because characteristic 2 is where the cheap set {2}
collapses (the characteristic-2 collapse theorem of the paper,
`experiments/E11_cheap_rows/char2_collapse.py`) and F_p does not need it.)

Summing the identity over one structure, with x_i = u_i + k'_i, u_i =
sum_s W_i[s] Y_s and sum_P 1 = 0:

    sum_P b_j =   sum_s (W1 + alpha W3)[s] T_s                          (1)
                + sum_{s,t} W2[s] W3[t] Q_{s,t}                         (2)
                + k'_3 sum_s W2[s] T_s  +  k'_2 sum_t W3[t] T_t         (3)
                + sum_{s,t} W0[s] W3[t]^2 Q2_{s,t}                      (4)
                + k'^2_3 sum_s W0[s] T_s                                (5)
                + k'_0 sum_t W3[t]^2 T_t^2                              (6)

with T_s = sum_P Y_s, Q_{s,t} = sum_P Y_s Y_t and Q2_{s,t} = sum_P Y_s Y_t^2.
Line (6) needs NO new moments at all: sum_P Y_t^2 = (sum_P Y_t)^2 because the
Frobenius is additive, so it is the elementwise square of the ordinary
single-block moment.  Only (4) is new data.

NEW LINEARISATION COLUMNS.  Writing K = U for the key-exponent set of the joint
expansion of S(P + rk^0) and K_nc = S for the exponents that come with a
nonzero plaintext exponent, line (4) needs the pair monomials k^{u + 2v}
(u, v in K), line (5) a new outer unknown k'^2_3, line (3) a new outer unknown
k'_2, and line (6) the columns k'_0 k^{2m}, m in K_nc.  `extra_monomials`
below produces exactly those; `column_count` turns them into the closed form

    M_b = 1 + 4 |(K+K) u (K+2K)| + 6 |(KxK) u (Kx2K) u (2KxK)|
            + 4 (1 + 4 |K_nc u 2K_nc|)          (the k'_0 family)
            + 12 (1 + 4 |K_nc|)                 (k'_3, k'_2, k'^2_3)

which is the same bookkeeping as the paper's

    M   = 6 |K|^2 + 4 |K+K| + 8 (1 + 4 |K_nc|) + 1 .

Usage (see run.py for the drivers):
    pre = PrecompB(F, alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    mom = MomentsB(pre); mom.add(P)
    rows = rows_from_moments_b(pre, mom, lrow)      # {(j, coord): row}
"""
from __future__ import annotations

import os
import sys

# The extended row set calls the mom2n kernel with a contraction length of
# nP ~ 50, where OpenMP's fork/join plus its critical section cost far more
# than the work: measured 5.0 s/structure with four threads against 0.15 s
# with one (toy-2^4, four unknown blocks).  The solver runs in its own process
# and sets its own thread count, so pinning this one here is safe.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from assemble_fast import Precomp, _pow_matrix                      # noqa: E402
from assemble_fast_2n import (_expx, _lib, _logs, _scatter2, gmv,  # noqa: E402
                              moment_matrix_2n, rows_from_moments_2n)
from assemble_fast_2n import gmm as _gmm_numpy                      # noqa: E402
from attack_2round_toy import add_exp, joint_expansion              # noqa: E402

def gmm(F, A, B):
    """A @ B over F_{2^n}.  Same result as `assemble_fast_2n.gmm`, but routed
    through the mom2n C kernel when it has been built: that kernel computes
    XOR_k expx[la[i][k] + lb[j][k]], which IS a matrix product once the second
    operand is transposed.  The extended row set calls it ~500 times per
    structure (the paper's calls it ~150), so the numpy inner loop over k
    becomes the assembly's bottleneck without this."""
    lib = _lib()
    A = np.ascontiguousarray(np.asarray(A, dtype=np.int64))
    if lib is None or A.shape[1] == 0:
        return _gmm_numpy(F, A, B)
    Bt = np.ascontiguousarray(np.asarray(B, dtype=np.int64).T)
    la, lb = _logs(F, A), _logs(F, Bt)
    out32 = np.empty((A.shape[0], Bt.shape[0]), dtype=np.int32)
    lib.mom2n(np.ascontiguousarray(la), np.ascontiguousarray(lb),
              A.shape[0], Bt.shape[0], A.shape[1], _expx(F), out32)
    return out32.astype(np.int64)


# The cheap set {1, 2}: position 2 is the quadratic a (the paper's row), 1 the
# cubic b.  The outer key unknowns the two rows together need are k'_0, k'_2,
# k'_3 and the Frobenius image k'^2_3, which we label 4 + 3 = 7 so that the
# monomial tuples keep sorting as plain integers.
CHEAP_SET = (1, 2)
K3_SQUARED = 7
OUTER_COORDS_B = (0, 2, 3, K3_SQUARED)


def double_exp(u, order):
    """2u in the exponent ring (reduce_exp's convention: 0 -> 0)."""
    return tuple(((2 * e - 1) % order + 1) if e else 0 for e in u)


def exponent_sets(F, alpha, cipher="dux", joint=None):
    """(K, K_nc) -- the key exponents of S(P + k), all / with a P factor."""
    joint = joint if joint is not None else joint_expansion(F, alpha, cipher)
    U = set().union(*[{ke for ke, _pe, _c in joint[co]} for co in range(4)])
    S = set().union(*[{ke for ke, pe, _c in joint[co] if any(pe)} for co in range(4)])
    return U, S, joint


def extra_monomials(order, U, S, unknown, outer):
    """The columns the b row needs on top of the paper's set."""
    ub = sorted(unknown)
    extra = set()
    for A in ub:
        for B in ub:
            for u in U:
                for v in U:
                    w = double_exp(v, order)
                    if A == B:
                        extra.add((None, ((A, add_exp(u, w, order)),)))
                    else:
                        extra.add((None, tuple(sorted([(A, u), (B, w)]))))
    for j in outer:
        for b in ub:
            for m in S:
                extra.add(((j, 0), ((b, double_exp(m, order)),)))
    return extra


def column_count(order, U, S, nblocks=4, nouter=4):
    """The closed form of M_b (and of the paper's M, for the ratio)."""
    KK = {add_exp(a, b, order) for a in U for b in U}
    KK2 = {add_exp(a, double_exp(b, order), order) for a in U for b in U}
    cross = ({(a, b) for a in U for b in U}
             | {(a, double_exp(b, order)) for a in U for b in U}
             | {(double_exp(b, order), a) for a in U for b in U})
    S0 = S | {double_exp(m, order) for m in S}
    npair = nblocks * (nblocks - 1) // 2
    M = npair * len(U) ** 2 + nblocks * len(KK) + 2 * nouter * (1 + nblocks * len(S)) + 1
    M_b = (1
           + nblocks * len(KK | KK2)
           + npair * len(cross)
           + nouter * (1 + nblocks * len(S0))
           + 3 * nouter * (1 + nblocks * len(S)))
    return {"K": len(U), "K_nc": len(S), "K+K": len(KK),
            "(K+K)u(K+2K)": len(KK | KK2), "cross": len(cross),
            "K_nc u 2K_nc": len(S0), "M": M, "M_b": M_b,
            "ratio": round(M_b / M, 4)}


class PrecompB(Precomp):
    """`Precomp` with the extended column set and the two extra column maps."""

    def __init__(self, F, alpha, unknown, outer, joint=None, cipher="dux"):
        assert F.char == 2, "the b coordinate is bilinear only in characteristic 2"
        assert cipher == "dux", "T1 is about DuX's outer S-box"
        U, S, joint = exponent_sets(F, alpha, cipher, joint)
        extra = extra_monomials(F.q - 1, U, S, unknown, outer)
        super().__init__(F, alpha, unknown, outer, joint, OUTER_COORDS_B,
                         cipher, extra_mons=extra)
        order = self.order
        nU = len(self.Ul)
        self.col_pair2 = {}
        for A in self.unknown:
            for B in self.unknown:
                M = np.full((nU, nU), -1, dtype=np.int64)
                for i, u in enumerate(self.Ul):
                    for j, v in enumerate(self.Ul):
                        w = double_exp(v, order)
                        m = ((None, ((A, add_exp(u, w, order)),)) if A == B else
                             (None, tuple(sorted([(A, u), (B, w)]))))
                        M[i][j] = self.idx.get(m, -1)
                self.col_pair2[(A, B)] = M
        self.col_kT2 = {}
        for j in self.outer:
            for b in self.unknown:
                arr = np.full(nU, -1, dtype=np.int64)
                for t, u in enumerate(self.Ul):
                    m = ((j, 0), ((b, double_exp(u, order)),))
                    if m in self.idx:
                        arr[t] = self.idx[m]
                self.col_kT2[(j, 0, b)] = arr


# ------------------------------------------------------------- moments -----
class MomentsB:
    """`assemble_fast.Moments` plus the squared-second-factor pair moments."""

    def __init__(self, pre: PrecompB):
        assert not pre.known, "the b row is assembled with all inner blocks unknown"
        self.pre = pre
        nP = pre.pe_arr.shape[0]
        self.pm = {b: np.zeros(nP, dtype=np.int64) for b in pre.unknown}
        self.Mom = {(A, B): np.zeros((nP, nP), dtype=np.int64)
                    for A in pre.unknown for B in pre.unknown}
        self.Mom2 = {(A, B): np.zeros((nP, nP), dtype=np.int64)
                     for A in pre.unknown for B in pre.unknown}
        self.pm2 = {b: np.zeros(nP, dtype=np.int64) for b in pre.unknown}
        self.npoints = 0

    def add(self, P):
        pre, F = self.pre, self.pre.F
        PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
        self.npoints += len(P[0])
        add_moments(F, pre, PW, self.pm, self.pm2, self.Mom, self.Mom2)


def add_moments(F, pre, PW, pm, pm2, Mom, Mom2, w=None):
    """XOR the point sums of one chunk (optionally weighted by `w`) in.

    `pm2[b][f] = sum_P w (P_b^f)^2` is a SEPARATE accumulator, not pm[b]^2:
    the Frobenius identity sum_P Y^2 = (sum_P Y)^2 only survives while the
    weight is trivial, since sum_P x^a Y^2 = (sum_P x^{a/2} Y)^2."""
    PWw = PW if w is None else {b: F.vmul(PW[b], w[None, :]) for b in PW}
    PWsq = {b: F.vmul(PW[b], PW[b]) for b in PW}
    PWsqw = PWsq if w is None else {b: F.vmul(PWsq[b], w[None, :]) for b in PWsq}
    LGw = {b: _logs(F, PWw[b]) for b in PWw}
    LGsq = {b: _logs(F, PWsq[b]) for b in PWsq}
    for b in pre.unknown:
        pm[b] ^= np.bitwise_xor.reduce(PWw[b], axis=1)
        pm2[b] ^= np.bitwise_xor.reduce(PWsqw[b], axis=1)
    for A in pre.unknown:
        for B in pre.unknown:
            Mom[(A, B)] ^= moment_matrix_2n(F, {A: PWw[A], B: PW[B]}, A, B,
                                            LGw[A], _logs(F, PW[B]))
            Mom2[(A, B)] ^= moment_matrix_2n(F, {A: PWw[A], B: PWsq[B]}, A, B,
                                             LGw[A], LGsq[B])


# ----------------------------------------------------------------- rows ----
def _sq(F, a):
    a = np.asarray(a, dtype=np.int64)
    return F.vmul(a, a)


def b_row_from_moments(pre: PrecompB, mom, lrow, j):
    """The row of  sum_P b  for outer block j (equation (1)-(6) above)."""
    F, alpha = pre.F, pre.alpha
    pm, Mom, Mom2 = mom.pm, mom.Mom, mom.Mom2

    def sel(off):
        return np.array([lrow[(s - 4 * j - off) % 16] for s in range(16)],
                        dtype=np.int64)

    R0, R1, R2, R3 = sel(0), sel(1), sel(2), sel(3)
    Wlin = np.array([int(R1[s]) ^ F.mul(int(alpha), int(R3[s])) for s in range(16)],
                    dtype=np.int64)

    def comb(coeffs, b):
        acc = np.zeros(pre.C.shape[1:], dtype=np.int64)
        for c in range(4):
            if coeffs[4 * b + c]:
                acc ^= F.vmul(pre.C[c], np.int64(coeffs[4 * b + c]))
        return acc

    C0 = {b: comb(R0, b) for b in pre.unknown}
    C2 = {b: comb(R2, b) for b in pre.unknown}
    C3 = {b: comb(R3, b) for b in pre.unknown}
    CW = {b: comb(Wlin, b) for b in pre.unknown}
    # (XOR_c W3[4b+c] C[c])^2 = XOR_c W3[4b+c]^2 C[c]^2 -- the Frobenius is
    # additive, so the squared-coefficient combination is just C3 squared.
    C3s = {b: _sq(F, C3[b]) for b in pre.unknown}

    row = np.zeros(len(pre.mons), dtype=np.int64)
    for b in pre.unknown:                                              # (1)
        _scatter2(row, pre.col_single[b], gmv(F, CW[b], pm[b]))
    for bs in pre.unknown:                                             # (2)
        for bt in pre.unknown:
            _scatter2(row, pre.col_pair[(bs, bt)],
                      gmm(F, gmm(F, C2[bs], Mom[(bs, bt)]), C3[bt].T))
    for b in pre.unknown:                                              # (3)
        _scatter2(row, pre.col_kT[(j, 3, b)], gmv(F, C2[b], pm[b]))
        _scatter2(row, pre.col_kT[(j, 2, b)], gmv(F, C3[b], pm[b]))
    for bs in pre.unknown:                                             # (4)
        for bt in pre.unknown:
            _scatter2(row, pre.col_pair2[(bs, bt)],
                      gmm(F, gmm(F, C0[bs], Mom2[(bs, bt)]), C3s[bt].T))
    for b in pre.unknown:                                              # (5)
        _scatter2(row, pre.col_kT[(j, K3_SQUARED, b)], gmv(F, C0[b], pm[b]))
    for b in pre.unknown:                                              # (6)
        _scatter2(row, pre.col_kT2[(j, 0, b)], gmv(F, C3s[b], mom.pm2[b]))
    return row


def rows_from_moments_b(pre: PrecompB, mom, lrow):
    """{(outer block j, cheap coordinate c): row} for c in CHEAP_SET.

    c = 2 is the paper's quadratic row (`rows_from_moments_2n`, unchanged);
    c = 1 is the cubic row of this module."""
    out = {}
    for j, r in zip(pre.outer, rows_from_moments_2n(pre, mom, lrow)):
        out[(j, 2)] = r
    for j in pre.outer:
        out[(j, 1)] = b_row_from_moments(pre, mom, lrow, j)
    return out


def combine_rows_b(rows, coeff, F):
    """sum_{(b, c)} coeff[(b, c)] * rows[(b, c)] -- O10 with two cheap columns.

    `coeff` is what `tools.cheap_rows.block_coeffs` returns for a two-element
    cheap set: a dict keyed by (outer block, cheap position)."""
    acc = None
    for (b, c), v in coeff.items():
        v = int(v) % F.q
        if not v:
            continue
        r = np.asarray(rows[(b, c)], dtype=np.int64)
        acc = F.vmul(r, np.int64(v)) if acc is None else acc ^ F.vmul(r, np.int64(v))
    return np.zeros(len(next(iter(rows.values()))), dtype=np.int64) if acc is None else acc


def true_key_vector(pre: PrecompB, rk0, rk1):
    """The extended linearisation monomials evaluated at the true key.

    `attack_2round_toy.monomial_value` cannot be reused unchanged because the
    outer label K3_SQUARED = 7 means (rk1[4j+3])^2, not rk1[4j+7]."""
    from attack_2round_toy import pow_field
    F = pre.F
    out = np.zeros(len(pre.mons), dtype=np.int64)
    for i, (outer, inner) in enumerate(pre.mons):
        if outer is None:
            v = 1
        else:
            j, co = outer
            v = rk1[4 * j + (co - 4)] if co >= 4 else rk1[4 * j + co]
            if co >= 4:
                v = F.mul(v, v)
        for b, exps in inner:
            for t, e in enumerate(exps):
                if e:
                    v = F.mul(v, pow_field(F, rk0[4 * b + t], e))
        out[i] = v
    return out


# ------------------------------------------------- weighted moments (O12) --
class MomentsView:
    """Duck-typed stand-in for `MomentsB` (only the five tables are read)."""

    __slots__ = ("pm", "pm2", "Mom", "Mom2", "npoints")

    def __init__(self, pm, pm2, Mom, Mom2, npoints=0):
        self.pm, self.pm2, self.Mom, self.Mom2 = pm, pm2, Mom, Mom2
        self.npoints = npoints


class LayoutB:
    """Flat layout of one structure's point sums, extended by pm2 and Mom2.

    [pm[b]] ++ [pm2[b]] ++ [Mom[(A,B)]] ++ [Mom2[(A,B)]], all ordered pairs
    (Mom2 is NOT symmetric, so no pair is folded away)."""

    def __init__(self, pre):
        self.pre = pre
        self.nP = int(pre.pe_arr.shape[0])
        self.blocks = list(pre.unknown)
        self.pairs = [(A, B) for A in self.blocks for B in self.blocks]
        n, nb, npair = self.nP, len(self.blocks), len(self.pairs)
        self.off_pm = {b: i * n for i, b in enumerate(self.blocks)}
        self.off_pm2 = {b: (nb + i) * n for i, b in enumerate(self.blocks)}
        base = 2 * nb * n
        self.off_mom = {ab: base + k * n * n for k, ab in enumerate(self.pairs)}
        base2 = base + npair * n * n
        self.off_mom2 = {ab: base2 + k * n * n for k, ab in enumerate(self.pairs)}
        self.width = base2 + npair * n * n

    def flatten(self, mom):
        n = self.nP
        g = np.zeros(self.width, dtype=np.int64)
        for b in self.blocks:
            g[self.off_pm[b]:self.off_pm[b] + n] = mom.pm[b]
            g[self.off_pm2[b]:self.off_pm2[b] + n] = mom.pm2[b]
        for ab in self.pairs:
            o = self.off_mom[ab]
            g[o:o + n * n] = np.asarray(mom.Mom[ab]).ravel()
            o = self.off_mom2[ab]
            g[o:o + n * n] = np.asarray(mom.Mom2[ab]).ravel()
        return g

    def moments(self, g, npoints=0):
        n = self.nP
        g = np.asarray(g, dtype=np.int64)
        pm = {b: g[self.off_pm[b]:self.off_pm[b] + n] for b in self.blocks}
        pm2 = {b: g[self.off_pm2[b]:self.off_pm2[b] + n] for b in self.blocks}
        Mom, Mom2 = {}, {}
        for ab in self.pairs:
            o = self.off_mom[ab]
            Mom[ab] = g[o:o + n * n].reshape(n, n)
            o = self.off_mom2[ab]
            Mom2[ab] = g[o:o + n * n].reshape(n, n)
        return MomentsView(pm, pm2, Mom, Mom2, npoints)


def _slice_moments(F, pre, layout, PW, lo, hi, w=None):
    """The flat point sums of the points [lo, hi) of one structure."""
    n = layout.nP
    sub = {b: PW[b][:, lo:hi] for b in layout.blocks}
    pm = {b: np.zeros(n, dtype=np.int64) for b in layout.blocks}
    pm2 = {b: np.zeros(n, dtype=np.int64) for b in layout.blocks}
    Mom = {ab: np.zeros((n, n), dtype=np.int64) for ab in layout.pairs}
    Mom2 = {ab: np.zeros((n, n), dtype=np.int64) for ab in layout.pairs}
    add_moments(F, pre, sub, pm, pm2, Mom, Mom2,
                None if w is None else w[lo:hi])
    return layout.flatten(MomentsView(pm, pm2, Mom, Mom2, hi - lo))


def weighted_moments_b(pre, PW, xvals, weights, layout=None, progress=None):
    """Yield (a, MomentsView) for every weight a, by the O12 slice decomposition.

    Same mechanism as `assemble_fast.weighted_moment_stream`: the structure is
    cut into the slices of its SLOWEST active word, each slice contributes its
    plain (or inner-weighted) point sums, and the outer axis is a Vandermonde
    combination of those.  Only the layout is wider (pm2 and Mom2)."""
    F = pre.F
    layout = layout or LayoutB(pre)
    N = PW[layout.blocks[0]].shape[1]
    dims = len(weights[0])
    q = F.q
    s = 1
    while q ** s < N:
        s += 1
    assert q ** s == N, "the structure must be a full product set"
    step = q ** (s - 1)
    by_inner = {}
    for a in weights:
        by_inner.setdefault(tuple(a[1:]), []).append(int(a[0]))
    v0 = np.arange(q, dtype=np.int64)
    for t, iw in enumerate(sorted(by_inner)):
        w = None
        if any(iw):
            from assemble_fast import pow_field_vec
            w = np.ones(N, dtype=np.int64)
            for jx, e in enumerate(iw):
                if e:
                    w = F.vmul(w, pow_field_vec(F, xvals[jx + 1], e))
        G = np.zeros((q, layout.width), dtype=np.int64)
        for k in range(q):
            G[k] = _slice_moments(F, pre, layout, PW, k * step, (k + 1) * step, w)
        for a0 in sorted(by_inner[iw]):
            from assemble_fast import pow_field_vec
            coef = pow_field_vec(F, v0, a0)
            g = np.zeros(layout.width, dtype=np.int64)
            for k in range(q):
                if coef[k]:
                    g ^= F.vmul(G[k], np.int64(coef[k]))
            yield (a0,) + tuple(iw), layout.moments(g, N)
        if progress:
            progress(t + 1, len(by_inner))
    del dims


# ------------------------------------------- pseudo-structures (template) --
def uniform_moments_b(pre, rng, layout=None):
    """The loosest template bound: every point sum independent and uniform,
    subject only to sum_P 1 = 0 (O5) on pm and pm2.

    `Mom` keeps the symmetry Mom[B][A] = Mom[A][B]^T that the assembly relies
    on; `Mom2` has none (its two factors are not interchangeable)."""
    F = pre.F
    layout = layout or LayoutB(pre)
    n = layout.nP
    j0 = int(np.flatnonzero((pre.pe_arr == 0).all(axis=1))[0])
    pm, pm2, Mom, Mom2 = {}, {}, {}, {}
    for b in layout.blocks:
        v = rng.integers(0, F.q, size=n).astype(np.int64)
        v[j0] = 0
        pm[b] = v
        v2 = rng.integers(0, F.q, size=n).astype(np.int64)
        v2[j0] = 0
        pm2[b] = v2
    for i, A in enumerate(layout.blocks):
        for B in layout.blocks[i:]:
            m = rng.integers(0, F.q, size=(n, n)).astype(np.int64)
            if A == B:
                m = np.triu(m) ^ np.triu(m, 1).T
            Mom[(A, B)] = m
            if A != B:
                Mom[(B, A)] = m.T.copy()
    for A in layout.blocks:
        for B in layout.blocks:
            Mom2[(A, B)] = rng.integers(0, F.q, size=(n, n)).astype(np.int64)
    return MomentsView(pm, pm2, Mom, Mom2, 0)


def realizable_moments_b(pre, rng, npts, layout=None):
    """The SHARP template bound: m = sum_i c_i mu(x_i) with XOR_i c_i = 0, i.e.
    the span V_0 of the moment vectors of real structures."""
    F = pre.F
    layout = layout or LayoutB(pre)
    n = layout.nP
    P = [rng.integers(0, F.q, size=npts).astype(np.int64) for _ in range(16)]
    c = rng.integers(0, F.q, size=npts).astype(np.int64)
    c[0] = np.bitwise_xor.reduce(c[1:])
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in layout.blocks}
    pm = {b: np.zeros(n, dtype=np.int64) for b in layout.blocks}
    pm2 = {b: np.zeros(n, dtype=np.int64) for b in layout.blocks}
    Mom = {ab: np.zeros((n, n), dtype=np.int64) for ab in layout.pairs}
    Mom2 = {ab: np.zeros((n, n), dtype=np.int64) for ab in layout.pairs}
    add_moments(F, pre, PW, pm, pm2, Mom, Mom2, c)
    return MomentsView(pm, pm2, Mom, Mom2, 0)


# ------------------------------------------------------------- one driver --
def structure_rows_b(pre, c, rks, rounds, seed, active, weights, lrow, ycoeff):
    """The dim K combined rows of ONE structure, one batch per weight.

    `ycoeff` is `tools.cheap_rows.block_coeffs(..., cheap=[1, 2])`, i.e. one
    dict {(outer block, cheap position): coefficient} per kernel vector."""
    from attack_2round_toy import active_words, structure_data
    from assemble_fast import _pow_matrix
    F = c.F
    act = active_words(3, active)
    s = len(act)
    P = structure_data(c, rks, rounds, 3, seed, active)
    N = len(P[0])
    idx = np.arange(N, dtype=np.int64)
    xvals = [(idx // (F.q ** (s - 1 - j))) % F.q for j in range(s)]
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    out, order = [], []
    for a, mom in weighted_moments_b(pre, PW, xvals[:len(weights[0])], weights):
        rows = rows_from_moments_b(pre, mom, lrow)
        for t, y in enumerate(ycoeff):
            out.append(combine_rows_b(rows, y, F))
            order.append((a, t))
    return np.asarray(out, dtype=np.int64), order, N


# ------------------------------------------ S19: the STREAMED layout (R9) --
class LayoutBStream:
    """`LayoutB` for the slice-streamed driver (`attack_12round.py --cheap-set 1,2`).

    Same five tables, but `Mom` is stored for the UNORDERED pairs only, as in
    `assemble_fast.MomentLayout`: Mom[(B, A)] = Mom[(A, B)]^T holds slice by
    slice and the weighted accumulation is linear, so it holds for every
    weighted moment vector too and `moments()` restores the transposes.
    `Mom2` has no such symmetry (its second factor is squared) and keeps all
    ordered pairs.  Width for nP = 50: 8 * 50 + 10 * 2500 + 16 * 2500 = 65 400
    against `MomentLayout`'s 25 200 -- the per-slice kernel makes 10 + 16 = 26
    `mom2n` calls instead of 10, which is the 2.6x of the S19 cost account.

    [pm[b]] ++ [pm2[b]] ++ [Mom[(A,B)], A <= B] ++ [Mom2[(A,B)], all (A,B)]"""

    def __init__(self, pre):
        self.pre = pre
        self.nP = int(pre.pe_arr.shape[0])
        self.blocks = list(pre.unknown)
        self.pairs = [(A, B) for i, A in enumerate(self.blocks)
                      for B in self.blocks[i:]]
        self.pairs2 = [(A, B) for A in self.blocks for B in self.blocks]
        n, nb = self.nP, len(self.blocks)
        self.off_pm = {b: i * n for i, b in enumerate(self.blocks)}
        self.off_pm2 = {b: (nb + i) * n for i, b in enumerate(self.blocks)}
        base = 2 * nb * n
        self.off_pair = {ab: base + k * n * n for k, ab in enumerate(self.pairs)}
        base2 = base + len(self.pairs) * n * n
        self.off_pair2 = {ab: base2 + k * n * n
                          for k, ab in enumerate(self.pairs2)}
        self.width = base2 + len(self.pairs2) * n * n

    def flatten(self, mom):
        n = self.nP
        g = np.zeros(self.width, dtype=np.int64)
        for b in self.blocks:
            g[self.off_pm[b]:self.off_pm[b] + n] = mom.pm[b]
            g[self.off_pm2[b]:self.off_pm2[b] + n] = mom.pm2[b]
        for ab in self.pairs:
            o = self.off_pair[ab]
            g[o:o + n * n] = np.asarray(mom.Mom[ab]).ravel()
        for ab in self.pairs2:
            o = self.off_pair2[ab]
            g[o:o + n * n] = np.asarray(mom.Mom2[ab]).ravel()
        return g

    def moments(self, g, npoints=0):
        n = self.nP
        g = np.asarray(g, dtype=np.int64)
        pm = {b: g[self.off_pm[b]:self.off_pm[b] + n] for b in self.blocks}
        pm2 = {b: g[self.off_pm2[b]:self.off_pm2[b] + n] for b in self.blocks}
        Mom, Mom2 = {}, {}
        for (A, B) in self.pairs:
            o = self.off_pair[(A, B)]
            m = g[o:o + n * n].reshape(n, n)
            Mom[(A, B)] = m
            if A != B:
                Mom[(B, A)] = m.T
        for ab in self.pairs2:
            o = self.off_pair2[ab]
            Mom2[ab] = g[o:o + n * n].reshape(n, n)
        return MomentsView(pm, pm2, Mom, Mom2, npoints)


def char2_slice_moments_b(pre, layout, PW, bounds, w=None):
    """The flat point sums of every slice [lo, hi) of `bounds`, on the wide
    layout -- the streamed twin of `_slice_moments` / `add_moments`, written
    like `attack_12round.char2_slice_moments` (logs once per point block, one
    `mom2n` call per pair per slice).

    `w` is a per-point weight of length PW[b].shape[1] (T3's inner axis or
    O15's inner mask): it scales ONE operand, the first factor, in every table
    -- pm, pm2 and both pair families -- so that the weighted sums are
    sum_P w P_A^e P_B^f and sum_P w P_A^e (P_B^f)^2.  `F.vmul` rather than a
    raw log addition, so that the zero sentinel stays a sentinel (S16)."""
    F = pre.F
    assert F.char == 2, "the b coordinate is a characteristic-2 construction"
    n = layout.nP
    lib, expx = _lib(), _expx(F)
    PWsq = {b: F.vmul(PW[b], PW[b]) for b in layout.blocks}
    if w is not None:
        w = np.asarray(w, dtype=np.int64)
        assert w.shape[0] == PW[layout.blocks[0]].shape[1], "one weight per point"
        PWw = {b: F.vmul(PW[b], w[None, :]) for b in layout.blocks}
        PWsqw = {b: F.vmul(PWsq[b], w[None, :]) for b in layout.blocks}
    else:
        PWw, PWsqw = PW, PWsq
    LG = {b: _logs(F, PW[b]) for b in layout.blocks}
    LGw = LG if w is None else {b: _logs(F, PWw[b]) for b in layout.blocks}
    LGsq = {b: _logs(F, PWsq[b]) for b in layout.blocks}

    def _mom(la, lb, L):
        if lib is not None:
            out32 = np.empty((n, n), dtype=np.int32)
            lib.mom2n(la, lb, n, n, L, expx, out32)
            return out32.astype(np.int64)
        m = np.empty((n, n), dtype=np.int64)
        for i in range(n):
            m[i] = np.bitwise_xor.reduce(expx[la[i][None, :] + lb], axis=1)
        return m

    G = np.zeros((len(bounds), layout.width), dtype=np.int64)
    for k, (lo, hi) in enumerate(bounds):
        L = hi - lo
        subw = {b: np.ascontiguousarray(LGw[b][:, lo:hi]) for b in layout.blocks}
        sub = (subw if w is None else
               {b: np.ascontiguousarray(LG[b][:, lo:hi]) for b in layout.blocks})
        subsq = {b: np.ascontiguousarray(LGsq[b][:, lo:hi]) for b in layout.blocks}
        for b in layout.blocks:
            o = layout.off_pm[b]
            G[k, o:o + n] = np.bitwise_xor.reduce(PWw[b][:, lo:hi], axis=1)
            o = layout.off_pm2[b]
            G[k, o:o + n] = np.bitwise_xor.reduce(PWsqw[b][:, lo:hi], axis=1)
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            G[k, o:o + n * n] = _mom(subw[A], sub[B], L).ravel()
        for (A, B) in layout.pairs2:
            o = layout.off_pair2[(A, B)]
            G[k, o:o + n * n] = _mom(subw[A], subsq[B], L).ravel()
    return G


def rows_from_moments_b_combined(pre, mom, lrow, ycoeff):
    """The dim K combined rows of one weighted moment vector (`ycoeff` is
    `weighted.combine_vectors(..., cheap=[1, 2])`: one dict per kernel
    vector), or the eight per-(block, coordinate) rows when `ycoeff` is None
    -- in the fixed order (j, 2) for j in outer, then (j, 1)."""
    rows = rows_from_moments_b(pre, mom, lrow)
    if ycoeff is None:
        return ([rows[(j, 2)] for j in pre.outer]
                + [rows[(j, 1)] for j in pre.outer])
    return [combine_rows_b(rows, y, pre.F) for y in ycoeff]
