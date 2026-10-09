"""E07 / S3 -- fast assembly of the r_KR = 2 linearised equations over F_p.

The prototype (attack_2round_toy.py) builds one equation per (structure,
outer block) with a double loop over the joint expansions of two inner
S-boxes: |joint[c_s]| x |joint[c_t]| <= 338^2 cross moments per (s, t) pair
and 256 such pairs, i.e. ~3 x 10^7 length-q numpy reductions per structure.
That is the bottleneck of the toy prototype.

This module replaces it by matrix products.  With

    C[c][i][j]   = coefficient of  k^{U[i]} P^{PE[j]}  in coordinate c of S(P+k)
    PW[b][j]     = the length-q array of values of P_b^{PE[j]}
    Mom[b][b']   = PW[b] @ PW[b']^T          (q-fold moments, one matmul)
    CA[b]        = sum_c A_{4b+c} C[c],  CB[b] = sum_c B_{4b+c} C[c]

the whole quadratic term of the equation collapses to

    sum_{s,t} A_s B_t  sum_P Y_s Y_t  =  sum_{b,b'} CA[b] Mom[b][b'] CB[b']^T

which is 32 matmuls of size 75 per outer block instead of 3 x 10^7 reductions.
Everything is exact integer arithmetic reduced mod p.

`structure_rows` here is the prime-field assembly (DuX(65537) and the toy
Fermat primes).  `Precomp` is field-generic and is shared with
`assemble_fast_2n.structure_rows_2n`, the characteristic-2 assembly (no BLAS
there, so the moments are XOR reductions instead of dgemm).

Structures may activate several ciphertext words (`--active 3,7`), so the
number of points N per structure is q^s, not q; nothing below assumes N = q.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from attack_2round_toy import (joint_expansion, monomial_set, add_exp,  # noqa: E402
                               linear_row, structure_data)
from dux.sbox import vS as _vS_dux  # noqa: E402
from yux.sbox import vS as _vS_yux  # noqa: E402

# the INNER S layer Y = SL(P + rk^0) of the known blocks
VS = {"dux": _vS_dux, "yux": _vS_yux}


class OuterEquation:
    """Which linear-layer rows the outer S-box's CHEAP coordinate uses.

    The r_KR = 2 equation is `sum_P (cheap coordinate of S(u + k')) = 0` with
    u = block j of L(SL(P + rk^0)) and k' = block j of rk^1.  Both families'
    cheap coordinate is quadratic and bilinear, so after dropping every term
    that carries the factor |structure| = 0 the equation is

        sum_s w[s] T_s  -  sum_{s,t} A_s B_t Q_{s,t}
            -  k'_{kA} sum_s A_s T_s  -  k'_{kB} sum_s B_s T_s  =  0

    with T_s = sum_P Y_s, Q_{s,t} = sum_P Y_s Y_t, Y = SL(P + rk^0), and
    A / B / w the rows of L that produce the u-coordinates the chain uses:

      DuX   a  = u2 - u0 u3 - alpha        (position 2)
            w = row(2),  A = row(0), B = row(3),  kA = 3, kB = 0
      YuX   z0 = u3 - u0 u1 - u2 - alpha   (position 3)
            w = row(3) - row(2),  A = row(0), B = row(1),  kA = 1, kB = 0

    (`w` is a signed combination for YuX because z0 carries BOTH the linear
    u3 and the linear -u2.)"""

    def __init__(self, w_offsets, a_off, b_off, kA, kB, name):
        self.w_offsets = tuple(w_offsets)      # ((offset, sign), ...)
        self.a_off, self.b_off = a_off, b_off
        self.kA, self.kB = kA, kB
        self.name = name

    @property
    def outer_coords(self):
        return tuple(sorted({self.kA, self.kB}))

    def rows(self, lrow, j, p, char2):
        """(A, W, B) as length-16 coefficient vectors of Y_s for outer block j."""
        def sel(off):
            return np.array([lrow[(s - 4 * j - off) % 16] for s in range(16)],
                            dtype=np.int64)
        A = sel(self.a_off)
        B = sel(self.b_off)
        W = np.zeros(16, dtype=np.int64)
        for off, sign in self.w_offsets:
            v = sel(off)
            W = (W ^ v) if char2 else (W + sign * v)
        if not char2:
            A, B, W = A % p, B % p, W % p
        return A, W, B


OUTER_EQUATIONS = {
    "dux": OuterEquation(((2, +1),), 0, 3, 3, 0, "a = u2 - u0 u3 - alpha"),
    "yux": OuterEquation(((3, +1), (2, -1)), 0, 1, 1, 0,
                         "z0 = u3 - u0 u1 - u2 - alpha"),
}


class Precomp:
    """Everything that does not depend on the structure."""

    def __init__(self, F, alpha, unknown, outer, joint=None, outer_coords=None,
                 cipher="dux", extra_mons=()):
        # Precomp itself is field-generic (it only builds the monomial index and
        # the column maps); `structure_rows` below is the prime-field assembly and
        # `assemble_fast_2n.structure_rows_2n` is the characteristic-2 one.
        self.F, self.order = F, F.q - 1
        self.p = F.q if F.char == 2 else F.p
        self.alpha = alpha
        self.cipher = cipher
        self.eq = OUTER_EQUATIONS[cipher]
        self.unknown = sorted(unknown)
        self.known = [b for b in range(4) if b not in self.unknown]
        self.outer = list(outer)
        self.joint = (joint if joint is not None
                      else joint_expansion(F, alpha, cipher))
        self.outer_coords = (tuple(outer_coords) if outer_coords is not None
                             else self.eq.outer_coords)
        self.mons, _, self.U, self.S = monomial_set(F, alpha, self.unknown, self.outer,
                                                    self.joint, self.outer_coords,
                                                    cipher, extra_mons)
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.Ul = sorted(self.U)
        self.PEl = sorted({pe for co in range(4) for _, pe, _ in self.joint[co]})
        ui = {u: i for i, u in enumerate(self.Ul)}
        pi = {e: i for i, e in enumerate(self.PEl)}
        nU, nP = len(self.Ul), len(self.PEl)
        self.C = np.zeros((4, nU, nP), dtype=np.int64)
        for co in range(4):
            for ke, pe, coeff in self.joint[co]:
                self.C[co][ui[ke]][pi[pe]] = coeff % self.p

        # column maps -------------------------------------------------------
        self.col_const = self.idx[(None, ())]
        self.col_single = {b: np.full(nU, -1, dtype=np.int64) for b in self.unknown}
        for b in self.unknown:
            for i, u in enumerate(self.Ul):
                m = (None, ((b, u),))
                if m in self.idx:
                    self.col_single[b][i] = self.idx[m]
        self.col_pair = {}
        for a in range(len(self.unknown)):
            for bidx in range(len(self.unknown)):
                A, B = self.unknown[a], self.unknown[bidx]
                M = np.full((nU, nU), -1, dtype=np.int64)
                for i, u in enumerate(self.Ul):
                    for j, v in enumerate(self.Ul):
                        if A == B:
                            m = (None, ((A, add_exp(u, v, self.order)),))
                        else:
                            m = (None, tuple(sorted([(A, u), (B, v)])))
                        M[i][j] = self.idx.get(m, -1)
                self.col_pair[(A, B)] = M
        self.col_kT = {}
        self.col_kconst = {}
        for j in self.outer:
            for i in self.outer_coords:
                self.col_kconst[(j, i)] = self.idx[((j, i), ())]
                for b in self.unknown:
                    arr = np.full(nU, -1, dtype=np.int64)
                    for t, u in enumerate(self.Ul):
                        m = ((j, i), ((b, u),))
                        if m in self.idx:
                            arr[t] = self.idx[m]
                    self.col_kT[(j, i, b)] = arr
        # power exponents needed per coordinate
        self.pe_arr = np.array(self.PEl, dtype=np.int64)      # (nP, 4)


def _pow_matrix(F, P, b, pe_arr):
    """PW[j] = prod_c P[4b+c]^{pe_arr[j][c]} as an (nP, q) int64 array."""
    q = len(P[0])
    nP = pe_arr.shape[0]
    caches = []
    for c in range(4):
        mx = int(pe_arr[:, c].max())
        cache = [np.ones(q, dtype=np.int64)]
        base = P[4 * b + c]
        for _ in range(mx):
            cache.append(F.vmul(cache[-1], base))
        caches.append(cache)
    out = np.ones((nP, q), dtype=np.int64)
    for j in range(nP):
        acc = None
        for c in range(4):
            e = int(pe_arr[j][c])
            if e == 0:
                continue
            v = caches[c][e]
            acc = v if acc is None else F.vmul(acc, v)
        if acc is not None:
            out[j] = acc
    return out


def _mom_vec(PWf, y, p, chunk):
    """PWf @ y with the same exactness bound as moment_matrix."""
    N = PWf.shape[1]
    yf = np.asarray(y, dtype=np.float64)
    if N <= chunk:
        return (PWf @ yf % p).astype(np.int64)
    acc = np.zeros(PWf.shape[0], dtype=np.int64)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        acc = (acc + (PWf[:, lo:hi] @ yf[lo:hi] % p).astype(np.int64)) % p
    return acc


def _scatter(row, cols, vals, p):
    """row[cols] += vals (mod p), ignoring cols == -1; assert dropped == 0."""
    cols = cols.ravel()
    vals = np.asarray(vals).ravel() % p
    m = cols >= 0
    if not np.all(vals[~m] == 0):
        raise RuntimeError("nonzero coefficient outside the predicted monomial set")
    np.add.at(row, cols[m], vals[m])


def moment_matrix(PWf, A, B, p, chunk):
    """Mom[i][j] = sum_over_points PW[A][i] * PW[B][j]  (mod p).

    float64 dgemm is exact only while the accumulator stays below 2^53, and it
    accumulates N terms each below p^2, so the constraint is N * p^2 < 2^53.
    That holds for s = 1 and for s = 2 on the toy primes (2^16 * 2^16 = 2^32),
    but NOT for s = 2 on p = 65537 (2^32 * 2^32), so the point axis is split
    into chunks of at most `chunk` points and the partial products are reduced
    mod p before being added."""
    N = PWf[A].shape[1]
    if N <= chunk:
        return (PWf[A] @ PWf[B].T % p).astype(np.int64)
    acc = np.zeros((PWf[A].shape[0], PWf[B].shape[0]), dtype=np.int64)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        acc += (PWf[A][:, lo:hi] @ PWf[B][:, lo:hi].T % p).astype(np.int64)
        acc %= p
    return acc


def exact_chunk(p):
    """Largest number of points whose float64 moment accumulator stays exact."""
    return max(1, int(2 ** 53 // (p * p)))


# --------------------------------------------------------------- streaming --
# For the 12-round attack a structure has q^3 = 2^48 points, so the (nP x N)
# power matrix of `structure_rows` does not fit in memory (75 x 2^48 x 8 B).
# Everything `structure_rows` needs from the data is a POINT SUM, though:
# pm[b] = sum_P P^e and Mom[(A,B)] = sum_P P_A^e P_B^f.  `Moments` accumulates
# exactly those over a stream of point chunks, so the memory is O(nP^2) and the
# structure can be walked in blocks of any size.  (Only the all-inner-blocks-
# unknown case is supported, which is the case the full-scale attacks use;
# with known blocks the equation also needs per-outer-block point sums.)


class Moments:
    """Streaming accumulator for the point sums of one structure."""

    def __init__(self, pre: Precomp):
        assert not pre.known, "streaming assembly needs all inner blocks unknown"
        self.pre = pre
        self.p = pre.p
        nP = pre.pe_arr.shape[0]
        self.pm = {b: np.zeros(nP, dtype=np.int64) for b in pre.unknown}
        self.Mom = {(A, B): np.zeros((nP, nP), dtype=np.int64)
                    for A in pre.unknown for B in pre.unknown}
        self.npoints = 0

    def add(self, P):
        """Accumulate one chunk of plaintexts (16 arrays of equal length)."""
        pre, F, p = self.pre, self.pre.F, self.p
        PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
        self.npoints += len(P[0])
        if F.char == 2:
            for b in pre.unknown:
                self.pm[b] ^= np.bitwise_xor.reduce(PW[b], axis=1)
            for A in pre.unknown:
                for B in pre.unknown:
                    a, bb = PW[A], PW[B]
                    for i in range(a.shape[0]):
                        self.Mom[(A, B)][i] ^= np.bitwise_xor.reduce(
                            F.vmul(bb, a[i][None, :]), axis=1)
            return
        chunk = exact_chunk(p)
        PWf = {b: PW[b].astype(np.float64) for b in pre.unknown}
        for b in pre.unknown:
            self.pm[b] = (self.pm[b] + PW[b].sum(axis=1)) % p
        for A in pre.unknown:
            for B in pre.unknown:
                if B < A:
                    continue
                m = moment_matrix(PWf, A, B, p, chunk)
                self.Mom[(A, B)] = (self.Mom[(A, B)] + m) % p
                if A != B:
                    self.Mom[(B, A)] = (self.Mom[(B, A)] + m.T) % p


def structure_rows(pre: Precomp, P, rk0, lrow, chunk=None):
    """Return one equation row (length len(pre.mons)) per outer block."""
    F, p = pre.F, pre.p
    assert F.char != 2, "prime-field assembly; use assemble_fast_2n for F_2^n"
    q = len(P[0])
    chunk = chunk or exact_chunk(p)
    PW, PWf, pm = {}, {}, {}
    for b in pre.unknown:
        PW[b] = _pow_matrix(F, P, b, pre.pe_arr)
        # float64 lets BLAS do the N-fold moment products; entries are < p, so
        # the accumulator stays exact while N * p^2 < 2^53 -- see moment_matrix,
        # which splits the point axis into chunks when it would not.
        PWf[b] = PW[b].astype(np.float64)
        pm[b] = PW[b].sum(axis=1) % p
    Ynum = {}
    for b in pre.known:
        x = tuple(F.vadd(P[4 * b + i], rk0[4 * b + i]) for i in range(4))
        y = VS[pre.cipher](F, x, pre.alpha)
        for i in range(4):
            Ynum[4 * b + i] = y[i] % p
    Mom = {}
    for A in pre.unknown:
        for B in pre.unknown:
            if (B, A) in Mom and A != B:
                Mom[(A, B)] = Mom[(B, A)].T % p
            else:
                Mom[(A, B)] = moment_matrix(PWf, A, B, p, chunk)

    rows = []
    for j in pre.outer:
        A, W2, B = pre.eq.rows(lrow, j, p, False)
        row = np.zeros(len(pre.mons), dtype=np.int64)
        CA = {b: (np.tensordot(A[4 * b:4 * b + 4], pre.C, axes=(0, 0))) % p
              for b in range(4)}
        CB = {b: (np.tensordot(B[4 * b:4 * b + 4], pre.C, axes=(0, 0))) % p
              for b in range(4)}
        CW = {b: (np.tensordot(W2[4 * b:4 * b + 4], pre.C, axes=(0, 0))) % p
              for b in range(4)}

        # (1) + sum_s m2[s] T_s
        for b in pre.unknown:
            _scatter(row, pre.col_single[b], (CW[b] @ pm[b]) % p, p)
        for b in pre.known:
            for c in range(4):
                if W2[4 * b + c]:
                    row[pre.col_const] += int(W2[4 * b + c]) * F.vsum(Ynum[4 * b + c])

        # (2) - sum_{s,t} m0[s] m3[t] sum_P Y_s Y_t
        ycA = {b: (sum(int(A[4 * b + c]) * Ynum[4 * b + c] for c in range(4)) % p)
               for b in pre.known}
        ycB = {b: (sum(int(B[4 * b + c]) * Ynum[4 * b + c] for c in range(4)) % p)
               for b in pre.known}
        for bs in range(4):
            for bt in range(4):
                if bs in pre.unknown and bt in pre.unknown:
                    G = (CA[bs] @ Mom[(bs, bt)]) % p
                    G = (G @ CB[bt].T) % p
                    _scatter(row, pre.col_pair[(bs, bt)], (-G) % p, p)
                elif bs in pre.unknown:
                    v = (CA[bs] @ _mom_vec(PWf[bs], ycB[bt], p, chunk)) % p
                    _scatter(row, pre.col_single[bs], (-v) % p, p)
                elif bt in pre.unknown:
                    v = (CB[bt] @ _mom_vec(PWf[bt], ycA[bs], p, chunk)) % p
                    _scatter(row, pre.col_single[bt], (-v) % p, p)
                else:
                    row[pre.col_const] -= int(F.vsum(F.vmul(ycA[bs], ycB[bt])))

        # (3)/(4)  - k'_{kA} sum_s A_s T_s  and  - k'_{kB} sum_s B_s T_s
        kA, kB = pre.eq.kA, pre.eq.kB
        for b in pre.unknown:
            _scatter(row, pre.col_kT[(j, kA, b)], (-(CA[b] @ pm[b])) % p, p)
            _scatter(row, pre.col_kT[(j, kB, b)], (-(CB[b] @ pm[b])) % p, p)
        for b in pre.known:
            sA = sum(int(A[4 * b + c]) * F.vsum(Ynum[4 * b + c]) for c in range(4))
            sB = sum(int(B[4 * b + c]) * F.vsum(Ynum[4 * b + c]) for c in range(4))
            row[pre.col_kconst[(j, kA)]] -= sA
            row[pre.col_kconst[(j, kB)]] -= sB
        rows.append(row % p)
    return rows


def rows_from_moments(pre: Precomp, mom: Moments, lrow):
    """The same equations as `structure_rows`, built from streamed point sums.

    Only the all-unknown case is supported (see `Moments`), so the `known`
    branches of `structure_rows` are absent; the surviving terms are exactly
    (1), the unknown x unknown part of (2), and (3)/(4)."""
    F, p = pre.F, pre.p
    assert F.char != 2, "prime-field assembly"
    pm, Mom = mom.pm, mom.Mom
    rows = []
    kA, kB = pre.eq.kA, pre.eq.kB
    for j in pre.outer:
        A, W2, B = pre.eq.rows(lrow, j, p, False)
        row = np.zeros(len(pre.mons), dtype=np.int64)
        CA = {b: np.tensordot(A[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p for b in range(4)}
        CB = {b: np.tensordot(B[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p for b in range(4)}
        CW = {b: np.tensordot(W2[4 * b:4 * b + 4], pre.C, axes=(0, 0)) % p for b in range(4)}
        for b in pre.unknown:
            _scatter(row, pre.col_single[b], (CW[b] @ pm[b]) % p, p)
        for bs in pre.unknown:
            for bt in pre.unknown:
                G = (CA[bs] @ Mom[(bs, bt)]) % p
                G = (G @ CB[bt].T) % p
                _scatter(row, pre.col_pair[(bs, bt)], (-G) % p, p)
        for b in pre.unknown:
            _scatter(row, pre.col_kT[(j, kA, b)], (-(CA[b] @ pm[b])) % p, p)
            _scatter(row, pre.col_kT[(j, kB, b)], (-(CB[b] @ pm[b])) % p, p)
        rows.append(row % p)
    return rows


# ------------------------------------------------------- weighted moments --
# O12 (docs/glossary.md; the weighted-moments theorem of the paper).
#
# For a structure whose s active ciphertext words run over all of F_q, every
# layer-l state word Z has formal total degree <= D, so for every exponent
# vector a in N^s with |a| < T - D
#
#     sum_{x in structure} x^a Z(x) = 0 ,        x^a = prod_j x_j^{a_j},
#
# because each surviving monomial needs a_j + e_j to be a positive multiple of
# q - 1 in every coordinate, i.e. total degree >= T.  In the r_KR = 2 equation
# the pure-key terms and the rk^1 terms carry the factor sum_x x^a, which is 0
# for a = 0 (|structure| = q^s = 0 in F_q) and for a != 0 alike, so the
# linearisation monomial set does NOT change: only the moments do, from
#     pm[b][e]        = sum_x P_b^e                to  sum_x x^a P_b^e
#     Mom[A][B][e][f] = sum_x P_A^e P_B^f          to  sum_x x^a P_A^e P_B^f.
# ONE structure therefore yields (number of admissible a) x (equations per
# structure) rows instead of one batch.
#
# SLICE DECOMPOSITION (memo Sect. 5.4).  Writing the structure as the disjoint
# union of the slices {x_{j0} = v} of its first active word,
#
#     M_a = sum_v v^{a_0} G_v^{(a_1,...)} ,
#
# where G_v is the (possibly inner-weighted) PLAIN moment vector of slice v.
# So the weighted moments are a Vandermonde matrix times a table of ordinary
# per-slice moments -- no NTT anywhere, and the per-slice moments are exactly
# what `Moments.add` already computes.  The cost is one plain-moment pass per
# distinct inner exponent plus one dense product per weight.


def weight_vector_count(dims, maxnorm):
    """How many a in N^dims have |a| < maxnorm: C(maxnorm - 1 + dims, dims).

    Closed form, so a caller can size a weight set without building it -- for
    two dimensions and a five-digit margin the list itself is billions of
    tuples (the O15 planner of W25 needs exactly that number)."""
    assert dims >= 1 and maxnorm >= 1
    return math.comb(maxnorm - 1 + dims, dims)


def weight_vectors(dims, maxnorm, limit=None):
    """Every a in N^dims with |a| < maxnorm, ordered by |a| then lexicographically.

    There are `weight_vector_count(dims, maxnorm)` of them, which is `maxnorm`
    for one dimension and maxnorm*(maxnorm+1)/2 for two.  `limit` stops after
    that many vectors -- the order is by |a|, so the first `limit` of them are
    the cheapest, which is what every caller of `--weights N` wants."""
    assert dims >= 1 and maxnorm >= 1
    out = []
    for total in range(maxnorm):
        def rec(k, left, acc):
            if limit is not None and len(out) >= limit:
                return
            if k == dims - 1:
                out.append(tuple(acc + [left]))
                return
            for v in range(left + 1):
                rec(k + 1, left - v, acc + [v])
        rec(0, total, [])
        if limit is not None and len(out) >= limit:
            break
    return out[:limit] if limit is not None else out


def combine_rows(rows, y, F):
    """sum_j y_j * row_j -- the cheap-coordinate combination of O10.

    `rows` are the per-outer-block rows `structure_rows` / `rows_from_moments`
    return (outer blocks in increasing order) and `y` the per-block
    coefficients `tools.cheap_rows.block_coeffs` gives."""
    rows = [np.asarray(r, dtype=np.int64) for r in rows]
    assert len(rows) == len(y), "one coefficient per outer block"
    if F.char == 2:
        acc = np.zeros_like(rows[0])
        for r, c in zip(rows, y):
            if c % F.q:
                acc ^= F.vmul(r, np.int64(c % F.q))
        return acc
    p = F.p
    acc = np.zeros_like(rows[0])
    for r, c in zip(rows, y):
        c %= p
        if c:
            acc = (acc + c * r) % p
    return acc % p


class FlatMoments:
    """Duck-typed stand-in for `Moments` (only `pm` / `Mom` are read)."""

    __slots__ = ("pm", "Mom", "npoints")

    def __init__(self, pm, Mom, npoints=0):
        self.pm, self.Mom, self.npoints = pm, Mom, npoints


class MomentLayout:
    """Flat layout of one structure's point sums.

    [ pm[b] for b in unknown ] ++ [ Mom[(A,B)].ravel() for A <= B ]
    -- only unordered pairs are stored, since Mom[(B,A)] = Mom[(A,B)]^T."""

    def __init__(self, pre):
        self.pre = pre
        self.nP = int(pre.pe_arr.shape[0])
        self.blocks = list(pre.unknown)
        self.pairs = [(A, B) for i, A in enumerate(self.blocks)
                      for B in self.blocks[i:]]
        n = self.nP
        self.off_pm = {b: i * n for i, b in enumerate(self.blocks)}
        base = len(self.blocks) * n
        self.off_pair = {}
        for k, ab in enumerate(self.pairs):
            self.off_pair[ab] = base + k * n * n
        self.width = base + len(self.pairs) * n * n

    def flatten(self, mom):
        g = np.zeros(self.width, dtype=np.int64)
        n = self.nP
        for b in self.blocks:
            g[self.off_pm[b]:self.off_pm[b] + n] = mom.pm[b]
        for (A, B) in self.pairs:
            o = self.off_pair[(A, B)]
            g[o:o + n * n] = np.asarray(mom.Mom[(A, B)]).ravel()
        return g

    def moments(self, g, npoints=0):
        n = self.nP
        g = np.asarray(g, dtype=np.int64)
        pm = {b: g[self.off_pm[b]:self.off_pm[b] + n] for b in self.blocks}
        Mom = {}
        for (A, B) in self.pairs:
            o = self.off_pair[(A, B)]
            m = g[o:o + n * n].reshape(n, n)
            Mom[(A, B)] = m
            if A != B:
                Mom[(B, A)] = m.T
        return FlatMoments(pm, Mom, npoints)


def pow_field_vec(F, v, e):
    """v^e for a numpy array v of field elements (0^0 = 1, 0^e = 0 for e > 0)."""
    v = np.asarray(v, dtype=np.int64)
    if e == 0:
        return np.ones_like(v)
    if F.char == 2:
        r = np.ones_like(v)
        b = v.copy()
        while e:
            if e & 1:
                r = F.vmul(r, b)
            b = F.vmul(b, b)
            e >>= 1
        return r
    p = F.p
    r = np.ones_like(v)
    b = v % p
    while e:
        if e & 1:
            r = (r * b) % p
        b = (b * b) % p
        e >>= 1
    return r


class WeightedMoments:
    """N_w weighted-moment accumulators for one structure (O12).

    `weights` is a list of exponent vectors over the WEIGHT WORDS, all of the
    same length.  `add_slice(mom, v)` folds the plain moments of one slice in
    with the coefficient prod_j v_j^{a_j}; `as_moments(i)` returns a
    `Moments`-shaped view that `rows_from_moments` accepts unchanged.

    A single-word structure's "slice" is a single point, so the same class
    covers the naive per-point accumulation used by the tests."""

    def __init__(self, pre, weights, layout=None):
        self.pre = pre
        self.F = pre.F
        self.p = pre.p
        self.layout = layout or MomentLayout(pre)
        self.weights = [tuple(np.atleast_1d(np.asarray(w, dtype=np.int64)).tolist())
                        for w in weights]
        self.dims = len(self.weights[0]) if self.weights else 0
        assert all(len(w) == self.dims for w in self.weights), \
            "every weight needs one exponent per weight word"
        self.W = np.array(self.weights, dtype=np.int64)     # (N_w, dims)
        self.acc = np.zeros((len(self.weights), self.layout.width), dtype=np.int64)
        self.npoints = 0
        self._buf_g, self._buf_c = [], []
        self._chunk = 1 if self.F.char == 2 else max(1, exact_chunk(self.p))

    # -- coefficients ------------------------------------------------------
    def coeffs(self, v):
        """[prod_j v_j^{a_j} for a in weights] as an int64 array."""
        v = np.atleast_1d(np.asarray(v, dtype=np.int64))
        assert v.size == self.dims
        F = self.F
        out = np.ones(len(self.weights), dtype=np.int64)
        for j in range(self.dims):
            col = self.W[:, j]
            emax = int(col.max())
            pw = np.ones(emax + 1, dtype=np.int64)
            for e in range(1, emax + 1):
                pw[e] = F.mul(int(pw[e - 1]), int(v[j]))
            f = pw[col]
            out = F.vmul(out, f) if F.char == 2 else (out * f) % self.p
        return out

    # -- accumulation ------------------------------------------------------
    def add_slice(self, mom, v, npoints=None):
        g = mom if isinstance(mom, np.ndarray) else self.layout.flatten(mom)
        self.npoints += (npoints if npoints is not None
                         else getattr(mom, "npoints", 0))
        self._buf_g.append(np.asarray(g, dtype=np.int64))
        self._buf_c.append(self.coeffs(v))
        if len(self._buf_g) >= self._chunk:
            self.flush()

    def flush(self):
        if not self._buf_g:
            return
        G = np.array(self._buf_g, dtype=np.int64)          # (k, width)
        C = np.array(self._buf_c, dtype=np.int64)          # (k, N_w)
        self._buf_g, self._buf_c = [], []
        if self.F.char == 2:
            for t in range(G.shape[0]):
                self.acc ^= self.F.vmul(G[t][None, :], C[t][:, None])
            return
        add = (C.T.astype(np.float64) @ G.astype(np.float64)) % self.p
        self.acc = (self.acc + add.astype(np.int64)) % self.p

    def as_moments(self, i):
        self.flush()
        return self.layout.moments(self.acc[i], self.npoints)

    def __len__(self):
        return len(self.weights)


def slice_bounds(q, s, axis=0):
    """(lo, hi) of every slice of the FIRST active word in `structure_data`'s
    point order (axis j has stride q^(s-1-j), so axis 0 is the slowest).

    `q` is the size of one axis: the field size for a full-domain structure and
    the point-set size |U| for an O15 product set (all axes the same size)."""
    assert axis == 0, "the slice decomposition slices along the slowest axis"
    step = q ** (s - 1)
    return [(v * step, (v + 1) * step) for v in range(q)]


def _stride(sizes, j):
    """Stride of axis j in `structure_data`'s flattening (last axis fastest)."""
    acc = 1
    for n in sizes[j + 1:]:
        acc *= n
    return acc


def slice_moment_table(pre, layout, PW, bounds, w=None):
    """Per-slice PLAIN (or inner-weighted) moments: an (nslices, width) table.

    This is exactly `Moments.add` run once per slice; `w` is an optional
    per-point weight array (the inner factor prod_{j>=1} x_j^{a_j})."""
    F, p = pre.F, pre.p
    n = layout.nP
    G = np.zeros((len(bounds), layout.width), dtype=np.int64)
    chunk = 1 << 62 if F.char == 2 else max(1, exact_chunk(p))
    for k, (lo, hi) in enumerate(bounds):
        cols = {}
        for b in layout.blocks:
            X = PW[b][:, lo:hi]
            cols[b] = X if w is None else (F.vmul(X, w[None, lo:hi]) if F.char == 2
                                           else (X * w[None, lo:hi]) % p)
        for b in layout.blocks:
            o = layout.off_pm[b]
            if F.char == 2:
                G[k, o:o + n] = np.bitwise_xor.reduce(cols[b], axis=1)
            else:
                G[k, o:o + n] = cols[b].sum(axis=1) % p
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            if F.char == 2:
                m = np.zeros((n, n), dtype=np.int64)
                for i in range(n):
                    m[i] = np.bitwise_xor.reduce(
                        F.vmul(PW[B][:, lo:hi], cols[A][i][None, :]), axis=1)
            else:
                a64 = cols[A].astype(np.float64)
                b64 = PW[B][:, lo:hi].astype(np.float64)
                N = hi - lo
                if N <= chunk:
                    m = (a64 @ b64.T % p).astype(np.int64)
                else:
                    m = np.zeros((n, n), dtype=np.int64)
                    for c0 in range(0, N, chunk):
                        c1 = min(c0 + chunk, N)
                        m = (m + (a64[:, c0:c1] @ b64[:, c0:c1].T % p).astype(np.int64)) % p
            G[k, o:o + n * n] = m.ravel()
    return G


def weighted_moment_stream(pre, PW, xvals, weights, layout=None, progress=None,
                           points=None, mask=None):
    """Yield (a, FlatMoments) for every weight `a`, using the slice decomposition.

    `PW[b]` is the (nP x N) power matrix of inner block b over the whole
    structure (`_pow_matrix`), `xvals` the list of the weight words' values at
    every point (`xvals[0]` must be the SLOWEST active word) and `weights` a
    list of exponent vectors of that length.

    Memory is one (q x width) table plus one moment vector: the N_w weighted
    moment vectors are never materialised, so the caller can assemble a row and
    throw the moments away.  Cost is one plain-moment pass per distinct INNER
    exponent (all but the first) plus one (1 x q) x (q x width) product per
    weight.

    O15 (W25): with `points` (the per-axis point sets U_0, ..., U_{s-1}, the
    SAME ones `structure_data(values=...)` was given) and `mask` (their
    divided-difference weights w_u) the structure is the product set
    U_0 x ... x U_{s-1} instead of F_q^s and every sum carries the mask
    prod_i w_{x_i}.  Nothing else changes: the mask factorises exactly like
    x^a, so the outer axis contributes w_{U_0[k]} * U_0[k]^{a_0} to slice k and
    the inner axes fold their w_{x_j} into the per-point vector, which is the
    slice decomposition of O12 with the per-slice weight w_u u^a."""
    F, p = pre.F, pre.p
    layout = layout or MomentLayout(pre)
    q = F.q
    N = PW[layout.blocks[0]].shape[1]
    if points is None:
        s = 1
        while q ** s < N:
            s += 1
        assert q ** s == N, "the structure must be a full product set"
        nslice = q
        v0 = np.arange(q, dtype=np.int64)
        outer_mask = None
        inner_mask = None
    else:
        sizes = [len(u) for u in points]
        s = len(sizes)
        tot = 1
        for n_ in sizes:
            tot *= n_
        assert tot == N, f"the point sets give {tot} points, the structure has {N}"
        nslice = sizes[0]
        v0 = np.asarray(points[0], dtype=np.int64)
        outer_mask = None if mask is None else np.asarray(mask[0], dtype=np.int64)
        inner_mask = None
        if mask is not None and s > 1:
            inner_mask = np.ones(N, dtype=np.int64)
            for j in range(1, s):
                col = np.asarray(mask[j], dtype=np.int64)[
                    (np.arange(N, dtype=np.int64) // _stride(sizes, j)) % sizes[j]]
                inner_mask = (F.vmul(inner_mask, col) if F.char == 2
                              else (inner_mask * col) % p)
    # The slices of the SLOWEST axis.  With equal axis sizes this is exactly
    # `slice_bounds(nslice, s)`; T2 (R9) mixes a coset word with a full-field
    # word, so the step is the product of the REMAINING axis sizes.
    if points is None:
        bounds = slice_bounds(nslice, s)
    else:
        step = _stride(sizes, 0)
        bounds = [(v * step, (v + 1) * step) for v in range(nslice)]
    inner = sorted({tuple(a[1:]) for a in weights})
    by_inner = {}
    for a in weights:
        by_inner.setdefault(tuple(a[1:]), []).append(int(a[0]))
    for t, iw in enumerate(inner):
        w = None
        if iw:
            w = np.ones(N, dtype=np.int64)
            for j, e in enumerate(iw):
                if e:
                    w = (F.vmul(w, pow_field_vec(F, xvals[j + 1], e)) if F.char == 2
                         else (w * pow_field_vec(F, xvals[j + 1], e)) % p)
        if inner_mask is not None:
            w = inner_mask.copy() if w is None else (
                F.vmul(w, inner_mask) if F.char == 2 else (w * inner_mask) % p)
        G = slice_moment_table(pre, layout, PW, bounds, w)
        if F.char != 2:
            Gf = G.astype(np.float64)
        for a0 in sorted(by_inner[iw]):
            coef = pow_field_vec(F, v0, a0)
            if outer_mask is not None:
                coef = (F.vmul(coef, outer_mask) if F.char == 2
                        else (coef * outer_mask) % p)
            if F.char == 2:
                g = np.zeros(layout.width, dtype=np.int64)
                for k in range(nslice):
                    if coef[k]:
                        g ^= F.vmul(G[k], np.int64(coef[k]))
            else:
                cf = coef.astype(np.float64)
                chunk = max(1, exact_chunk(p))
                if nslice <= chunk:
                    g = ((cf @ Gf) % p).astype(np.int64)
                else:
                    g = np.zeros(layout.width, dtype=np.int64)
                    for c0 in range(0, nslice, chunk):
                        c1 = min(c0 + chunk, nslice)
                        g = (g + ((cf[c0:c1] @ Gf[c0:c1]) % p).astype(np.int64)) % p
            yield (a0,) + iw, layout.moments(g, N)
        if progress:
            progress(t + 1, len(inner))
