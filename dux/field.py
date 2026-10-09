"""Finite field arithmetic for DuX: F_p (p prime) and F_{2^n}.

Elements are plain Python ints in [0, q).  For F_{2^n} an int is the
coefficient vector of the polynomial (bit i = coefficient of x^i), which is
also the convention used by the DuX paper for constants such as alpha = 179.

Every field exposes scalar ops (add/sub/mul/neg/from_int) and numpy
vectorised ops (vadd/vsub/vmul/vneg) so that the same cipher code can run
on a single state or on a batch of 2^k states (see dux/vec.py).
"""
from __future__ import annotations

import numpy as np


def _prime_factors(m: int):
    fs, d = set(), 2
    while d * d <= m:
        while m % d == 0:
            fs.add(d)
            m //= d
        d += 1
    if m > 1:
        fs.add(m)
    return fs


class PrimeField:
    """F_p with p prime."""

    def __init__(self, p: int):
        self.p = p
        self.q = p
        self.char = p
        self.name = f"F_{p}"

    # scalar
    def add(self, a, b):
        return (a + b) % self.p

    def sub(self, a, b):
        return (a - b) % self.p

    def mul(self, a, b):
        return (a * b) % self.p

    def neg(self, a):
        return (-a) % self.p

    def inv(self, a):
        return pow(a, self.p - 2, self.p)

    def from_int(self, x):
        return x % self.p

    def two(self):
        return 2 % self.p

    # vectorised (int64 arrays; p < 2^31 so products fit)
    dtype = np.int64

    def vadd(self, a, b):
        return (a + b) % self.p

    def vsub(self, a, b):
        return (a - b) % self.p

    def vmul(self, a, b):
        return (a * b) % self.p

    def vneg(self, a):
        return (-a) % self.p

    def vsum(self, a):
        """Sum of an array of field elements (as a field element)."""
        return int(np.sum(a % self.p, dtype=np.int64) % self.p)


def clmul_mod(a: int, b: int, poly: int, n: int) -> int:
    """Carry-less multiply of two n-bit polynomials modulo `poly` (degree n)."""
    r = 0
    top = 1 << n
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & top:
            a ^= poly
    return r


class BinaryField:
    """F_{2^n} = F_2[x]/(poly).  Uses log/antilog tables for multiplication."""

    dtype = np.int64

    def __init__(self, n: int, poly: int):
        self.n = n
        self.poly = poly
        self.q = 1 << n
        self.char = 2
        self.name = f"F_2^{n}"
        self.order = self.q - 1
        g = self._find_generator()
        self.generator = g
        exp = np.zeros(2 * self.order, dtype=np.int64)
        log = np.zeros(self.q, dtype=np.int64)
        x = 1
        for i in range(self.order):
            exp[i] = x
            log[x] = i
            x = clmul_mod(x, g, poly, n)
        assert x == 1, "generator order mismatch"
        exp[self.order:] = exp[: self.order]
        self._exp = exp
        self._log = log

    def _find_generator(self) -> int:
        fs = _prime_factors(self.order)
        for g in range(2, self.q):
            ok = True
            for f in fs:
                if self._pow_clmul(g, self.order // f) == 1:
                    ok = False
                    break
            if ok:
                return g
        raise RuntimeError("no generator found (is poly irreducible?)")

    def _pow_clmul(self, a: int, e: int) -> int:
        r = 1
        while e:
            if e & 1:
                r = clmul_mod(r, a, self.poly, self.n)
            a = clmul_mod(a, a, self.poly, self.n)
            e >>= 1
        return r

    # scalar
    def add(self, a, b):
        return a ^ b

    sub = add

    def neg(self, a):
        return a

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self._exp[self._log[a] + self._log[b]])

    def inv(self, a):
        assert a != 0
        return int(self._exp[(self.order - self._log[a]) % self.order])

    def from_int(self, x):
        return x % self.q

    def two(self):
        return 0  # 2 = 1 + 1 = 0 in characteristic 2

    # vectorised
    def vadd(self, a, b):
        return a ^ b

    vsub = vadd

    def vneg(self, a):
        return a

    def vmul(self, a, b):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        r = self._exp[self._log[a] + self._log[b]]
        return np.where((a == 0) | (b == 0), 0, r)

    def vsum(self, a):
        return int(np.bitwise_xor.reduce(np.asarray(a, dtype=np.int64)))


def make_field(q_or_name):
    """Factory.  Accepts 65537, 256, 65536, or strings like '2^8', '2^16', 'p65537'."""
    from .params import FIELDS
    if isinstance(q_or_name, str):
        spec = FIELDS[q_or_name]
    else:
        spec = next(v for v in FIELDS.values() if v["q"] == q_or_name)
    if spec["char"] == 2:
        return BinaryField(spec["n"], spec["poly"])
    return PrimeField(spec["q"])
