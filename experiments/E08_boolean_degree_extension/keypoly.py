"""W8 (E08) -- EXACT multivariate polynomial of the DuX S-box in the key words.

Represents a polynomial in the four key words (k0,k1,k2,k3) of one S-box block
over F_{2^n} as {exponent vector -> coefficient}.  Because the S-box is a short
chain of quadratic steps, the exponents stay small (<= 5 per variable for one
layer, <= 10 for a product of two), so for every n >= 4 the exponent support is
the SAME; only the coefficients live in different fields.  This lets us compute
the Boolean degree

    deg_2(f) = max_{monomials with nonzero coeff} sum_i HW(e_i)

exactly for n = 4, 8, 16 and check whether it depends on n (a Moebius transform
could only reach n = 4, where 2^{4n} = 2^16).
"""
from __future__ import annotations

from itertools import product as iproduct


class KeyPoly:
    """dict {exponent vector -> coeff in F_q}; F is a dux.field field object.
    The number of variables `nv` is fixed per instance (4 for one S-box block,
    8 when the outer round key of a second layer is symbolic as well)."""

    __slots__ = ("F", "d", "nv")

    def __init__(self, F, d=None, nv=4):
        self.F = F
        self.d = {} if d is None else d
        self.nv = nv

    @classmethod
    def const(cls, F, v, nv=4):
        return cls(F, {tuple([0] * nv): int(v)} if v else {}, nv)

    @classmethod
    def var_plus_const(cls, F, i, cval, nv=4):
        e = [0] * nv
        e[i] = 1
        d = {tuple(e): 1}
        if cval:
            d[tuple([0] * nv)] = int(cval)
        return cls(F, d, nv)

    def copy(self):
        return KeyPoly(self.F, dict(self.d), self.nv)

    def __add__(self, other):
        F = self.F
        d = dict(self.d)
        for e, v in other.d.items():
            w = F.add(d.get(e, 0), v)
            if w:
                d[e] = w
            else:
                d.pop(e, None)
        return KeyPoly(F, d, self.nv)

    def __sub__(self, other):
        F = self.F
        d = dict(self.d)
        for e, v in other.d.items():
            w = F.sub(d.get(e, 0), v)
            if w:
                d[e] = w
            else:
                d.pop(e, None)
        return KeyPoly(F, d, self.nv)

    def add_const(self, v):
        return self + KeyPoly.const(self.F, v, self.nv)

    def sub_const(self, v):
        return self - KeyPoly.const(self.F, v, self.nv)

    def __mul__(self, other):
        F = self.F
        order = F.q - 1
        d = {}
        for e1, v1 in self.d.items():
            for e2, v2 in other.d.items():
                e = tuple(_reduce_exp(a + b, order) for a, b in zip(e1, e2))
                w = F.add(d.get(e, 0), F.mul(v1, v2))
                if w:
                    d[e] = w
                else:
                    d.pop(e, None)
        return KeyPoly(F, d, self.nv)

    def scalar_mul(self, s):
        F = self.F
        if s == 0:
            return KeyPoly(F, None, self.nv)
        return KeyPoly(F, {e: F.mul(v, s) for e, v in self.d.items()}, self.nv)

    def boolean_degree(self):
        if not self.d:
            return -1
        return max(sum(bin(x).count("1") for x in e) for e in self.d)

    def support_size(self):
        return len(self.d)

    def top_monomials(self):
        dmax = self.boolean_degree()
        return [e for e in self.d if sum(bin(x).count("1") for x in e) == dmax]


def _reduce_exp(e, order):
    """k^{2^n} = k, i.e. exponents live in {0} u [1, 2^n - 1]."""
    if e == 0:
        return 0
    return ((e - 1) % order) + 1


def sbox_enc_chain(x, alpha):
    """The encryption S-box S = (R^{-1})^4 on four KeyPoly inputs, valid over
    F_p and over F_{2^n} (where - is +):
        a = x2 - x0 x3 - alpha,  b = x1 - x3 a - alpha,
        c = x0 - a b  - alpha,  y3 = x3 - b c - alpha,   S = (c, b, a, y3)."""
    a = (x[2] - x[0] * x[3]).sub_const(alpha)
    b = (x[1] - x[3] * a).sub_const(alpha)
    c = (x[0] - a * b).sub_const(alpha)
    y3 = (x[3] - b * c).sub_const(alpha)
    return (c, b, a, y3)


def sbox_enc_keypoly(F, alpha, P):
    """S(P + k) as four KeyPoly in k = (k0,k1,k2,k3).  Encryption S-box, the
    chain of dux/sbox.py: a = x2 - x0x3 - alpha, b = x1 - x3 a - alpha,
    c = x0 - a b - alpha, y3 = x3 - b c - alpha.  Returns (y0,y1,y2,y3)."""
    x = [KeyPoly.var_plus_const(F, i, P[i]) for i in range(4)]
    a = (x[2] + x[0] * x[3]).add_const(alpha)      # char 2: - == +
    b = (x[1] + x[3] * a).add_const(alpha)
    c = (x[0] + a * b).add_const(alpha)
    y3 = (x[3] + b * c).add_const(alpha)
    return (c, b, a, y3)


def sbox_dec_keypoly(F, alpha, P):
    """S^{-1}(P + k) = R^4, for completeness."""
    x = [KeyPoly.var_plus_const(F, i, P[i]) for i in range(4)]
    f = (x[0] * x[1] + x[3]).add_const(alpha)
    g = (x[1] * x[2] + x[0]).add_const(alpha)
    y1 = (x[2] * f + x[1]).add_const(alpha)
    y2 = (f * g + x[2]).add_const(alpha)
    return (g, y1, y2, f)


# ---------------------------------------------------------------------------
# Yu2X (Liu et al., IEEE TIT 2024; analysed by Ni-Wang-Li, DCC 2026, 94:123)
# ---------------------------------------------------------------------------
# Pf(x0,x1,x2,x3) = (x1, x2, x3, x0 + x1 x2 + x3 + alpha),  S^{-1} = Pf^4
# Pf^{-1}(x0,x1,x2,x3) = (x3 + x0 x1 + x2 + alpha, x0, x1, x2),  S = (Pf^{-1})^4
# giving S = (D, C, B, A) with
#     A = x3 + x0 x1 + x2 + alpha          (deg 2)
#     B = x2 + A x0 + x1 + alpha           (deg 3)
#     C = x1 + B A + x0 + alpha            (deg 5)
#     D = x0 + C B + A + alpha             (deg 8)
# i.e. coordinate degrees (8,5,3,2), matching Ni et al. Sect. 4.2.
# The forward linear layer is the circulant inverse over F_2 of
# L^{-1} = sum_{j in {0,3,4,8,9,12,14}} Rot_j, namely
YU2X_ROT_FWD = (1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15)
YU2X_ROT_INV = (0, 3, 4, 8, 9, 12, 14)


def yu2x_sbox_enc_chain(x, alpha):
    """The Yu2X encryption S-box on four KeyPoly inputs (characteristic 2)."""
    A = (x[3] + x[0] * x[1] + x[2]).add_const(alpha)
    B = (x[2] + A * x[0] + x[1]).add_const(alpha)
    C = (x[1] + B * A + x[0]).add_const(alpha)
    D = (x[0] + C * B + A).add_const(alpha)
    return (D, C, B, A)


def sbox_enc_chain_yux(x, alpha):
    """The YuX ENCRYPTION S-box S = Pf^{-4} on four KeyPoly inputs, valid over
    F_p as well as over F_{2^n} (where - is +):
        z0 = x3 - x0 x1 - x2 - alpha        (deg 2)   -> position 3
        z1 = x2 - z0 x0 - x1 - alpha        (deg 3)   -> position 2
        z2 = x1 - z1 z0 - x0 - alpha        (deg 5)   -> position 1
        z3 = x0 - z2 z1 - z0 - alpha        (deg 8)   -> position 0
        S = (z3, z2, z1, z0)
    In characteristic 2 this coincides with `yu2x_sbox_enc_chain` above (which
    is the same chain written with +); over F_p the signs matter, and they are
    the ones of Construction 1's Pf^{-1}, NOT the quadratic system printed in
    Sect. VI-D (see yux/sbox.py, erratum E1)."""
    z0 = (x[3] - x[0] * x[1] - x[2]).sub_const(alpha)
    z1 = (x[2] - z0 * x[0] - x[1]).sub_const(alpha)
    z2 = (x[1] - z1 * z0 - x[0]).sub_const(alpha)
    z3 = (x[0] - z2 * z1 - z0).sub_const(alpha)
    return (z3, z2, z1, z0)


def sbox_dec_chain_yux(x, alpha):
    """The YuX DECRYPTION S-box S^{-1} = Pf^4 on four KeyPoly inputs:
        y0 = x0 + x1 x2 + x3 + alpha        (deg 2)
        y1 = x1 + x2 x3 + y0 + alpha        (deg 2)
        y2 = x2 + x3 y0 + y1 + alpha        (deg 3)
        y3 = x3 + y0 y1 + y2 + alpha        (deg 4)"""
    y0 = (x[0] + x[1] * x[2] + x[3]).add_const(alpha)
    y1 = (x[1] + x[2] * x[3] + y0).add_const(alpha)
    y2 = (x[2] + x[3] * y0 + y1).add_const(alpha)
    y3 = (x[3] + y0 * y1 + y2).add_const(alpha)
    return (y0, y1, y2, y3)


ENC_CHAIN = {"dux": sbox_enc_chain, "yux": sbox_enc_chain_yux}
DEC_CHAIN = {"dux": sbox_dec_keypoly, "yux": sbox_dec_chain_yux}
