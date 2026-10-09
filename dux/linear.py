"""DuX linear layer LM (paper Sect. 3.2, 4.2).

Convention: (X <<< j)_i = X_{(i+j) mod 16}  (function `rotl`).
With this convention a circulant Circ(m0..m15) acts as
    (M X)_i = sum_j m_j * X_{(i+j) mod 16},
so M0 = Circ(1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1) is exactly the rotation set
{0,1,3,4,7,8,9,10,12,14,15} listed in the paper.

Decryption direction (both fields):
    L0^{-1}(X) = X<<<1 + X<<<4 + X<<<8 + X<<<9  + X<<<13
    L1^{-1}(X) = X<<<1 + X<<<5 + X<<<8 + X<<<12 + X<<<13
Encryption direction:
    F_{2^n}: rotation XOR-sums ROT_FWD_BIN
    F_p    : circulant inverse of L^{-1}, computed generically (checked
             against the paper's M0 for p = 65537 in tests/test_linear.py)

Key observation (docs/glossary.md, O2):  L1 = Rot_{-4} o L0 and
L1^{-1} = Rot_{+4} o L0^{-1}, i.e. the two "random" MDS layers differ by a
whole-block rotation, which commutes with the S-box layer.
"""
from __future__ import annotations

import numpy as np

from .params import ROT_INV, ROT_FWD_BIN, WORDS, TXOR_WORD


def rotl(x, j):
    """(X <<< j)_i = X_{(i+j) mod 16}; works for tuples of ints or of arrays."""
    j %= WORDS
    return tuple(x[(i + j) % WORDS] for i in range(WORDS))


def rot_sum(F, x, rots, vec=False):
    add = F.vadd if vec else F.add
    out = [None] * WORDS
    for i in range(WORDS):
        acc = x[(i + rots[0]) % WORDS]
        for j in rots[1:]:
            acc = add(acc, x[(i + j) % WORDS])
        out[i] = acc
    return tuple(out)


def t_xor(rk):
    """XOR of the two lowest bits of word TXOR_WORD (=15) of the round key."""
    w = int(rk[TXOR_WORD])
    return (w & 1) ^ ((w >> 1) & 1)


def inv_matrix_rows(F):
    """Return {t: first row of the circulant L_t} for a prime field,
    obtained by inverting the circulant of ROT_INV[t] modulo p.
    A circulant's inverse is circulant, so one row suffices."""
    p = F.p
    rows = {}
    for t, rots in ROT_INV.items():
        # Build M (L_t^{-1}) and solve M * y = e_0 ; then first row of M^{-1}
        # is obtained from the first *column* of M^{-1}: for circulants,
        # M^{-1}[0][k] = (M^{-1} e_k)[0].  Easier: invert the full matrix.
        M = np.zeros((WORDS, WORDS), dtype=np.int64)
        for i in range(WORDS):
            for j in rots:
                M[i][(i + j) % WORDS] = (M[i][(i + j) % WORDS] + 1) % p
        Minv = _mat_inv_mod_p(M, p)
        rows[t] = tuple(int(v) for v in Minv[0])
        # sanity: circulant structure
        for i in range(WORDS):
            for k in range(WORDS):
                assert Minv[i][k] == rows[t][(k - i) % WORDS], "inverse not circulant?"
    return rows


def _mat_inv_mod_p(M, p):
    n = M.shape[0]
    A = np.concatenate([M % p, np.eye(n, dtype=np.int64)], axis=1) % p
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] % p), None)
        if piv is None:
            raise ValueError("matrix singular mod p")
        A[[c, piv]] = A[[piv, c]]
        inv = pow(int(A[c][c]), p - 2, p)
        A[c] = (A[c] * inv) % p
        for r in range(n):
            if r != c and A[r][c]:
                A[r] = (A[r] - A[r][c] * A[c]) % p
    return A[:, n:]


class LinearLayer:
    def __init__(self, F):
        self.F = F
        if F.char == 2:
            self.fwd_rows = None
        else:
            self._rows = inv_matrix_rows(F)

    def L_inv(self, x, t, vec=False):
        return rot_sum(self.F, x, ROT_INV[t], vec)

    def L(self, x, t, vec=False):
        F = self.F
        if F.char == 2:
            return rot_sum(F, x, ROT_FWD_BIN[t], vec)
        row = self._rows[t]
        add = F.vadd if vec else F.add
        mul = F.vmul if vec else F.mul
        out = []
        for i in range(WORDS):
            acc = None
            for j in range(WORDS):
                m = row[j]
                if m == 0:
                    continue
                term = mul(x[(i + j) % WORDS], m)
                acc = term if acc is None else add(acc, term)
            out.append(acc)
        return tuple(out)

    def LM(self, x, rk, vec=False):
        return self.L(x, t_xor(rk), vec)

    def LM_inv(self, x, rk, vec=False):
        return self.L_inv(x, t_xor(rk), vec)
