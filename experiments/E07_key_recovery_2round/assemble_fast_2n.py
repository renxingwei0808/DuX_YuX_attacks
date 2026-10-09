"""E07 / W12 -- characteristic-2 assembly of the r_KR = 2 linearised equations.

Same equation and same monomial set as assemble_fast.py (see that file and
attack_2round_toy.py for the derivation); only the arithmetic changes.

Over F_p the whole quadratic term collapses into dgemm calls because integer
products of field elements fit exactly in float64.  There is no BLAS for
F_{2^n}, so every product goes through the field's log/antilog tables
(dux/field.py, `F.vmul`) and every sum is an XOR:

    Mom[A][B][i][j] = XOR_over_points ( PW[A][i] * PW[B][j] )

with PW[b] the (nP x N) matrix of plaintext-monomial values of inner block b.
nP is around 50, so that is ~2 500 length-N reductions per unordered block
pair -- affordable at toy / DuX(2^8) scale, which is what W12 needs.

Matrix products over F_{2^n} use `gmm` below: one broadcast `F.vmul` plus an
XOR accumulation per inner index.  The matrices here are (nU x nP) x (nP x nP)
x (nP x nU) with nU = nP = 50, so this is ~10^5 field multiplications per
outer block and structure.

Note on the monomial count: the joint expansion of S(P + k) has binomial
coefficients, and in characteristic 2 Lucas' theorem kills most of them.  The
monomial set is therefore SMALLER than over F_p:

    unknown inner blocks |  1     2      4
    F_p (|U| = 75)       |  1082  7780   38051
    F_{2^n} (|U| = 50)   |   763  4017   18025      (identical for n = 4, 8, 16)

The counts do not depend on n because every exponent in the expansion is at
most 8 < 2^n - 1, so the mod-(q-1) reduction never fires.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from assemble_fast import Precomp, _pow_matrix  # noqa: E402
from dux.sbox import vS as _vS_dux  # noqa: E402
from yux.sbox import vS as _vS_yux  # noqa: E402

# the INNER S layer Y = SL(P + rk^0) of the known blocks
VS = {"dux": _vS_dux, "yux": _vS_yux}


def gmm(F, A, B):
    """Matrix product over F_{2^n}: out[i][j] = XOR_k A[i][k] * B[k][j]."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k in range(A.shape[1]):
        out ^= F.vmul(A[:, k:k + 1], B[k:k + 1, :])
    return out


def gmv(F, A, v):
    """Matrix-vector product over F_{2^n}."""
    return gmm(F, A, np.asarray(v, dtype=np.int64).reshape(-1, 1))[:, 0]


# --- optional C kernel for the moment matrices (fast/mom2n.c) --------------
_LIB = None
_LIB_TRIED = False


def _lib():
    """ctypes handle for fast/libmom2n.so, or None if it has not been built."""
    global _LIB, _LIB_TRIED
    if not _LIB_TRIED:
        _LIB_TRIED = True
        import ctypes
        path = os.path.join(HERE, "fast", "libmom2n.so")
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
            lib.mom2n.restype = None
            lib.mom2n.argtypes = [i32, i32, ctypes.c_int32, ctypes.c_int32,
                                  ctypes.c_int64, i32, i32]
            lib.momvec2n.restype = None
            lib.momvec2n.argtypes = [i32, ctypes.c_int32, ctypes.c_int64,
                                     i32, i32, i32]
            _LIB = lib
    return _LIB


_EXPX = {}


def _expx(F):
    """`F._exp` padded with zeros, as int32, so that a sentinel log value of
    2*order (used for the field element 0) lands in the zero region.

    `F.vmul` costs two int64 gathers, an add, a comparison per operand and a
    `np.where`; the moment matrices are ~1.6e9 element operations per structure
    at the DuX(2^16) scale, which makes those temporaries the bottleneck.  With
    this table one gather is enough and everything is int32."""
    key = (F.n, F.poly)          # NOT id(F): field objects are short-lived and
    e = _EXPX.get(key)           # CPython reuses ids, which silently mixes fields
    if e is None:
        order = F.order
        arr = np.zeros(4 * order + 2, dtype=np.int32)
        arr[:2 * order] = np.asarray(F._exp, dtype=np.int32)
        e = _EXPX[key] = arr
    return e


def _logs(F, a):
    """log of every entry, with 0 mapped to the sentinel 2*order."""
    la = np.asarray(F._log, dtype=np.int32)[a]
    la[a == 0] = 2 * F.order
    return la


def moment_matrix_2n(F, PW, A, B, la=None, lb=None):
    """Mom[i][j] = XOR_over_points PW[A][i] * PW[B][j].

    `la` / `lb` are the sentinel logarithms of PW[A] / PW[B] (see `_logs`);
    the caller passes them in so that they are computed once per structure
    rather than once per block pair."""
    a, b = PW[A], PW[B]
    expx = _expx(F)
    if la is None:
        la = _logs(F, a)
    if lb is None:
        lb = _logs(F, b)
    lib = _lib()
    if lib is not None:
        out32 = np.empty((a.shape[0], b.shape[0]), dtype=np.int32)
        lib.mom2n(la, lb, a.shape[0], b.shape[0], la.shape[1], expx, out32)
        return out32.astype(np.int64)
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)
    for i in range(a.shape[0]):
        # (nP x N) broadcast against one row, then XOR-reduce the point axis
        out[i] = np.bitwise_xor.reduce(expx[la[i][None, :] + lb], axis=1)
    return out


def moment_vector_2n(F, PWb, y, lb=None):
    """v[i] = XOR_over_points PW[b][i] * y."""
    y = np.asarray(y)
    expx = _expx(F)
    if lb is None:
        lb = _logs(F, PWb)
    ly = _logs(F, y)
    lib = _lib()
    if lib is not None:
        out32 = np.empty(PWb.shape[0], dtype=np.int32)
        lib.momvec2n(lb, PWb.shape[0], lb.shape[1],
                     np.ascontiguousarray(ly), expx, out32)
        return out32.astype(np.int64)
    return np.bitwise_xor.reduce(expx[lb + ly[None, :]], axis=1).astype(np.int64)


def _scatter2(row, cols, vals):
    """row[cols] ^= vals, ignoring cols == -1; a nonzero value that would be
    dropped means the monomial set is wrong, so raise instead of silently
    losing it."""
    cols = np.asarray(cols).ravel()
    vals = np.asarray(vals, dtype=np.int64).ravel()
    m = cols >= 0
    if not np.all(vals[~m] == 0):
        raise RuntimeError("nonzero coefficient outside the predicted monomial set")
    np.bitwise_xor.at(row, cols[m], vals[m])


def structure_rows_2n(pre: Precomp, P, rk0, lrow):
    """One equation row (length len(pre.mons)) per outer block, over F_{2^n}."""
    F = pre.F
    assert F.char == 2, "characteristic-2 assembly; use assemble_fast for F_p"
    PW, pm, LG = {}, {}, {}
    for b in pre.unknown:
        PW[b] = _pow_matrix(F, P, b, pre.pe_arr)
        pm[b] = np.bitwise_xor.reduce(PW[b], axis=1)
        LG[b] = _logs(F, PW[b])          # once per block, reused by every pair

    Ynum = {}
    for b in pre.known:
        x = tuple(F.vadd(P[4 * b + i], rk0[4 * b + i]) for i in range(4))
        y = VS[pre.cipher](F, x, pre.alpha)
        for i in range(4):
            Ynum[4 * b + i] = y[i]

    Mom = {}
    for A in pre.unknown:
        for B in pre.unknown:
            if (B, A) in Mom and A != B:
                Mom[(A, B)] = Mom[(B, A)].T
            else:
                Mom[(A, B)] = moment_matrix_2n(F, PW, A, B, LG[A], LG[B])

    rows = []
    kA, kB = pre.eq.kA, pre.eq.kB
    for j in pre.outer:
        A, W2, B = pre.eq.rows(lrow, j, pre.p, True)
        row = np.zeros(len(pre.mons), dtype=np.int64)

        def comb(coeffs, b):
            """XOR_c coeffs[4b+c] * C[c]  (an (nU x nP) matrix over F_{2^n})."""
            acc = np.zeros(pre.C.shape[1:], dtype=np.int64)
            for c in range(4):
                if coeffs[4 * b + c]:
                    acc ^= F.vmul(pre.C[c], np.int64(coeffs[4 * b + c]))
            return acc

        CA = {b: comb(A, b) for b in range(4)}
        CB = {b: comb(B, b) for b in range(4)}
        CW = {b: comb(W2, b) for b in range(4)}

        # (1)  sum_s m2[s] T_s
        for b in pre.unknown:
            _scatter2(row, pre.col_single[b], gmv(F, CW[b], pm[b]))
        for b in pre.known:
            for c in range(4):
                if W2[4 * b + c]:
                    row[pre.col_const] ^= F.mul(int(W2[4 * b + c]),
                                                F.vsum(Ynum[4 * b + c]))

        # (2)  sum_{s,t} m0[s] m3[t] sum_P Y_s Y_t   (signs vanish in char 2)
        ycA, ycB = {}, {}
        for b in pre.known:
            za = np.zeros(len(P[0]), dtype=np.int64)
            zb = np.zeros(len(P[0]), dtype=np.int64)
            for c in range(4):
                if A[4 * b + c]:
                    za ^= F.vmul(Ynum[4 * b + c], np.int64(A[4 * b + c]))
                if B[4 * b + c]:
                    zb ^= F.vmul(Ynum[4 * b + c], np.int64(B[4 * b + c]))
            ycA[b], ycB[b] = za, zb
        for bs in range(4):
            for bt in range(4):
                if bs in pre.unknown and bt in pre.unknown:
                    G = gmm(F, gmm(F, CA[bs], Mom[(bs, bt)]), CB[bt].T)
                    _scatter2(row, pre.col_pair[(bs, bt)], G)
                elif bs in pre.unknown:
                    v = gmv(F, CA[bs],
                            moment_vector_2n(F, PW[bs], ycB[bt], LG[bs]))
                    _scatter2(row, pre.col_single[bs], v)
                elif bt in pre.unknown:
                    v = gmv(F, CB[bt],
                            moment_vector_2n(F, PW[bt], ycA[bs], LG[bt]))
                    _scatter2(row, pre.col_single[bt], v)
                else:
                    row[pre.col_const] ^= F.vsum(F.vmul(ycA[bs], ycB[bt]))

        # (3)/(4)  k'_{kA} sum_s A_s T_s  and  k'_{kB} sum_s B_s T_s
        for b in pre.unknown:
            _scatter2(row, pre.col_kT[(j, kA, b)], gmv(F, CA[b], pm[b]))
            _scatter2(row, pre.col_kT[(j, kB, b)], gmv(F, CB[b], pm[b]))
        for b in pre.known:
            sA = sB = 0
            for c in range(4):
                if A[4 * b + c]:
                    sA ^= F.mul(int(A[4 * b + c]), F.vsum(Ynum[4 * b + c]))
                if B[4 * b + c]:
                    sB ^= F.mul(int(B[4 * b + c]), F.vsum(Ynum[4 * b + c]))
            row[pre.col_kconst[(j, kA)]] ^= sA
            row[pre.col_kconst[(j, kB)]] ^= sB
        rows.append(row)
    return rows


def rows_from_moments_2n(pre: Precomp, mom, lrow):
    """Characteristic-2 twin of `assemble_fast.rows_from_moments`.

    Same equations as `structure_rows_2n`, built from an object exposing the
    streamed point sums `pm[b]` and `Mom[(A, B)]`; only the all-inner-blocks-
    unknown case is supported, so the `known` branches are absent."""
    F = pre.F
    assert F.char == 2, "characteristic-2 assembly; use assemble_fast for F_p"
    assert not pre.known, "moment assembly needs all inner blocks unknown"
    pm, Mom = mom.pm, mom.Mom
    rows = []
    kA, kB = pre.eq.kA, pre.eq.kB
    for j in pre.outer:
        A, W2, B = pre.eq.rows(lrow, j, pre.p, True)
        row = np.zeros(len(pre.mons), dtype=np.int64)

        def comb(coeffs, b):
            acc = np.zeros(pre.C.shape[1:], dtype=np.int64)
            for c in range(4):
                if coeffs[4 * b + c]:
                    acc ^= F.vmul(pre.C[c], np.int64(coeffs[4 * b + c]))
            return acc

        CA = {b: comb(A, b) for b in pre.unknown}
        CB = {b: comb(B, b) for b in pre.unknown}
        CW = {b: comb(W2, b) for b in pre.unknown}
        for b in pre.unknown:
            _scatter2(row, pre.col_single[b], gmv(F, CW[b], pm[b]))
        for bs in pre.unknown:
            for bt in pre.unknown:
                G = gmm(F, gmm(F, CA[bs], Mom[(bs, bt)]), CB[bt].T)
                _scatter2(row, pre.col_pair[(bs, bt)], G)
        for b in pre.unknown:
            _scatter2(row, pre.col_kT[(j, kA, b)], gmv(F, CA[b], pm[b]))
            _scatter2(row, pre.col_kT[(j, kB, b)], gmv(F, CB[b], pm[b]))
        rows.append(row)
    return rows
