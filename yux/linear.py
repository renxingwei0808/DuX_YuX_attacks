"""YuX linear layer (paper Sect. IV-A.3, IV-B.1, IV-B.2, footnote 2).

Convention (footnote 2): (X <<< j)_i = X_{(i+j) mod 16}, i.e. the same word
rotation dux/linear.py uses.  With it a circulant Circ(m_0..m_15) acts as
    (M X)_i = sum_j m_j X_{(i+j) mod 16}.

Decryption direction (both fields), Sect. IV-A.3:
    LP_inv(X) = sum_{j in {0,3,4,8,9,12,14}} (X <<< j)          (7 terms)
Encryption direction:
    LP = LP_inv^{-1};
    F_{2^n} (Sect. IV-B.1): the XOR of the rotations {1,2,3,5,6,7,8,12,13,14,15}
    F_p     (Sect. IV-B.2): the circulant with first row v_p
                            (fractions, and integers for p = 65537)

Both encryption forms are COMPUTED here as the circulant inverse of LP_inv
(over F_2 and over F_p respectively) and only checked against the paper's
explicit set / vector in tests/test_yux.py -- the same discipline dux/linear.py
follows for DuX's M0.

Difference to DuX worth recording:
DuX has a KEY-DEPENDENT choice L0/L1 driven by two bits of rk^i_15; YuX has
none, so every method here takes an optional `t` argument that is accepted and
IGNORED.  It exists purely so that scripts written against dux.linear's
signature keep working through dux/registry.py.
"""
from __future__ import annotations

import numpy as np

from dux.linear import _mat_inv_mod_p, rot_sum, rotl  # noqa: F401  (shared)
from .params import ROT_INV, ROT_FWD_BIN, V_P_65537, WORDS


def circulant(rots, n=WORDS):
    """The 0/1 circulant of a rotation set, as an integer matrix."""
    M = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in rots:
            M[i][(i + j) % n] += 1
    return M


def _mat_inv_gf2(M, n=WORDS):
    """Inverse over F_2 of an n x n 0/1 matrix, by Gauss-Jordan on bitmasks."""
    rows = []
    for i in range(n):
        v = 0
        for j in range(n):
            if int(M[i][j]) & 1:
                v |= 1 << j
        rows.append(v | (1 << (n + i)))          # augmented with the identity
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, n) if rows[i] >> c & 1), None)
        if piv is None:
            raise ValueError("matrix singular over F_2")
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(n):
            if i != r and (rows[i] >> c & 1):
                rows[i] ^= rows[r]
        r += 1
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            out[i][j] = (rows[i] >> (n + j)) & 1
    return out


def forward_row(F):
    """First row of the encryption circulant LP = LP_inv^{-1}.

    F_{2^n}: a 0/1 vector, which must be the indicator of ROT_FWD_BIN.
    F_p    : v_p (equal to params.V_P_65537 when p = 65537)."""
    M = circulant(ROT_INV)
    if F.char == 2:
        Minv = _mat_inv_gf2(M)
    else:
        Minv = _mat_inv_mod_p(M % F.p, F.p)
    row = tuple(int(v) for v in Minv[0])
    for i in range(WORDS):                       # a circulant's inverse is circulant
        for k in range(WORDS):
            assert int(Minv[i][k]) == row[(k - i) % WORDS], "inverse not circulant?"
    return row


def assert_invertible(F):
    """Raise if the decryption circulant is singular over the field (the toy
    primes 5/13/17 are exactly the cases dux/params.py warns about)."""
    forward_row(F)


class LinearLayer:
    """Same interface as dux.linear.LinearLayer; `t` is accepted and ignored."""

    def __init__(self, F):
        self.F = F
        self.rot_inv = ROT_INV
        self._row = forward_row(F)
        self.rot_fwd = tuple(j for j in range(WORDS) if self._row[j]) \
            if F.char == 2 else None

    # --- the two directions ------------------------------------------------
    def L_inv(self, x, t=0, vec=False):
        return rot_sum(self.F, x, ROT_INV, vec)

    def L(self, x, t=0, vec=False):
        F = self.F
        row = self._row
        if F.char == 2:
            return rot_sum(F, x, self.rot_fwd, vec)
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

    # --- dux-compatible aliases (YuX has no key-dependent linear layer) -----
    def LM(self, x, rk=None, vec=False):
        return self.L(x, 0, vec)

    def LM_inv(self, x, rk=None, vec=False):
        return self.L_inv(x, 0, vec)

    @property
    def _rows(self):
        """dux.linear.LinearLayer exposes {t: first row}; scripts index it with
        t = 0 or 1 and YuX has only one layer, so both keys give the same row."""
        return {0: self._row, 1: self._row}


def paper_v_p(F):
    """v_p from the printed FRACTIONS (Sect. IV-B.2), evaluated in F_p."""
    from .params import V_P_FRACTIONS
    p = F.p
    return tuple((a % p) * pow(b % p, p - 2, p) % p for a, b in V_P_FRACTIONS)


__all__ = ["LinearLayer", "circulant", "forward_row", "paper_v_p",
           "assert_invertible", "ROT_INV", "ROT_FWD_BIN", "V_P_65537"]
