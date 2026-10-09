"""Tests for the analysis tooling added by W3 and W8.

They are cheap (< 2 s) and lock the two numbers the write-up depends on:
the exact S-box key-polynomial degrees and the exactness of the univariate
polynomial model used to recompute the zero-sum layers.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "experiments", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "experiments", "E02_degree_bounds"))

from dux import DuX                                     # noqa: E402
from dux.field import make_field                        # noqa: E402
from moebius import moebius, anf_degree                 # noqa: E402
from keypoly import (KeyPoly, sbox_enc_chain,           # noqa: E402
                     yu2x_sbox_enc_chain, YU2X_ROT_FWD, YU2X_ROT_INV)
from symbolic_layers import PolyRing, symbolic_layers   # noqa: E402


def test_moebius_roundtrip_and_degree():
    rng = np.random.default_rng(0)
    tt = rng.integers(0, 2, size=1 << 12).astype(np.uint8)
    assert np.array_equal(moebius(moebius(tt)), tt)      # involution
    m = 3
    f = np.zeros(1 << m, dtype=np.uint8)
    for i in range(1 << m):
        f[i] = ((i & 1) & ((i >> 1) & 1)) ^ ((i >> 2) & 1)
    assert anf_degree(moebius(f)) == 2


@pytest.mark.parametrize("name", ["2^4", "2^8", "2^16"])
def test_sbox_key_polynomial_degrees(name):
    """S(P+k) has Boolean degrees (3,2,2,4) and the sum over an even structure
    drops each of them by exactly one (docs/glossary.md O5/O6)."""
    F = make_field(name)
    alpha = 179 % F.q if F.q < 180 else 179
    rng = np.random.default_rng(3)
    P = [int(v) for v in rng.integers(0, F.q, size=4)]
    y = sbox_enc_chain([KeyPoly.var_plus_const(F, i, P[i]) for i in range(4)], alpha)
    assert [t.boolean_degree() for t in y] == [3, 2, 2, 4]
    acc = [KeyPoly(F) for _ in range(4)]
    for _ in range(6):                                   # even-size structure
        Pp = [int(v) for v in rng.integers(0, F.q, size=4)]
        yy = sbox_enc_chain([KeyPoly.var_plus_const(F, i, Pp[i]) for i in range(4)], alpha)
        for i in range(4):
            acc[i] = acc[i] + yy[i]
    assert [t.boolean_degree() for t in acc] == [2, 1, 1, 3]


def test_yu2x_sbox_reconstruction():
    """Our reconstruction of the Yu2X S-box from Ni et al. Sect. 2.3 has the
    F_q-degrees (8,5,3,2) they state, and its linear layer really is the
    circulant inverse of L^{-1}."""
    F = make_field("2^8")
    x = [KeyPoly.var_plus_const(F, i, 0) for i in range(4)]
    y = yu2x_sbox_enc_chain(x, 205)
    assert [max(sum(e) for e in t.d) for t in y] == [8, 5, 3, 2]
    A = np.zeros((16, 16), dtype=int)
    B = np.zeros((16, 16), dtype=int)
    for i in range(16):
        for j in YU2X_ROT_FWD:
            A[i][(i + j) % 16] ^= 1
        for j in YU2X_ROT_INV:
            B[i][(i + j) % 16] ^= 1
    assert np.array_equal((A @ B) % 2, np.eye(16, dtype=int))


@pytest.mark.parametrize("name", ["toy-257", "dux-2^8"])
def test_symbolic_layers_match_the_cipher(name):
    """The univariate polynomial model must evaluate to the real state."""
    c = DuX(name)
    rng = np.random.default_rng(5)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    consts = [int(v) for v in rng.integers(0, c.F.q, size=16)]
    pos, layers = 3, 3
    P, states = symbolic_layers(c, rks, pos, layers, consts, force_L0=False)
    for X in [0, 1, 7, c.F.q - 1]:
        C = tuple(consts[i] if i != pos else X for i in range(16))
        real = c.decrypt_layers(C, rks, layers, yield_all=True)
        for l in range(layers):
            for i in range(16):
                coeffs = states[l][i]
                # evaluate the polynomial at X (Horner over F_q)
                acc = 0
                for e in range(len(coeffs) - 1, -1, -1):
                    acc = c.F.add(c.F.mul(acc, X), int(coeffs[e]))
                assert acc == real[l][i], (name, l, i, X)
    # O2: with L0 forced the per-block-position degrees are unchanged
    _, st0 = symbolic_layers(c, rks, pos, layers, consts, force_L0=True)
    ring = PolyRing(c.F)
    for l in range(layers):
        a = sorted(ring.degree(st0[l][i]) for i in range(16))
        b = sorted(ring.degree(states[l][i]) for i in range(16))
        assert a == b
