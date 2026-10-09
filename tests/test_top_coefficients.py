"""W13-B: exact top coefficients of the encryption-direction univariate polynomial."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from dux import DuX                                       # noqa: E402
from dux.params import ROT_FWD_BIN                         # noqa: E402
from tools.exponent_sets import run as expsets             # noqa: E402
from tools.top_coefficients import propagate, verdict      # noqa: E402


def full_polynomials(instance, pos, layers, seed):
    """Reference: the complete unreduced univariate polynomial of every word."""
    c = DuX(instance, rounds=layers + 2)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(seed)))

    def pmul(a, b):
        out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
        for i in np.flatnonzero(a):
            out[i:i + len(b)] ^= F.vmul(b, np.int64(a[i]))
        return out

    def padd(a, b):
        o = np.zeros(max(len(a), len(b)), dtype=np.int64)
        o[:len(a)] ^= a
        o[:len(b)] ^= b
        return o

    def const(v):
        return np.array([int(v)], dtype=np.int64)

    def sbox(x):
        x0, x1, x2, x3 = x
        al = const(c.alpha)
        a = padd(padd(x2, pmul(x0, x3)), al)
        b = padd(padd(x1, pmul(x3, a)), al)
        cc = padd(padd(x0, pmul(a, b)), al)
        y3 = padd(padd(x3, pmul(b, cc)), al)
        return [cc, b, a, y3]

    W = [const(rks[0][i]) for i in range(16)]
    W[pos] = np.array([int(rks[0][pos]), 1], dtype=np.int64)
    out_layers = []
    for layer in range(1, layers + 1):
        outs = []
        for b in range(4):
            outs += sbox(W[4 * b:4 * b + 4])
        out_layers.append(outs)
        rots = ROT_FWD_BIN[0]
        W = []
        for i in range(16):
            acc = outs[(i + rots[0]) % 16]
            for j in rots[1:]:
                acc = padd(acc, outs[(i + j) % 16])
            W.append(padd(acc, const(rks[layer][i])))
    return c, rks, out_layers


@pytest.mark.parametrize("instance,layers", [("toy-2^4", 3), ("dux-2^8", 4)])
def test_window_matches_full_polynomial(instance, layers):
    """The tracked window is the true top of the polynomial, cancellations included."""
    c, rks, full = full_polynomials(instance, 1, layers, 2026)
    top = propagate(c, rks, 1, layers, 4)
    for l in range(layers):
        for w in range(16):
            u, ref = top[l][w], full[l][w]
            assert u.d == len(ref) - 1
            for t in range(4):
                e = u.d - t
                assert u.w[t] == (int(ref[e]) if e >= 0 else 0), (l, w, t)


def test_sum_over_field_matches_direct_summation():
    """`verdict` reproduces the actual sum over the whole field."""
    c, rks, full = full_polynomials("dux-2^8", 1, 4, 2026)
    F = c.F
    top = propagate(c, rks, 1, 4, 4)
    X = np.arange(F.q, dtype=np.int64)
    for w in range(16):
        ref = full[3][w]
        acc = np.zeros(F.q, dtype=np.int64)
        pw = np.ones(F.q, dtype=np.int64)
        for e in range(len(ref)):
            if ref[e]:
                acc ^= F.vmul(pw, np.int64(ref[e]))
            pw = F.vmul(pw, X)
        direct = int(F.vsum(acc))
        st, val = verdict(F, top[3][w])
        if st == "balanced":
            assert direct == 0, w
        elif st == "not balanced":
            assert direct == val != 0, w


def test_frobenius_cells_are_proved():
    """The two single-word cells where E10 measured better than O7."""
    for seed in (2026, 7, 99):
        c = DuX("dux-2^16", rounds=8)
        rks = c.key_schedule(c.random_key(np.random.default_rng(seed)))
        outs = propagate(c, rks, 1, 6, 4)[5]           # layer 6
        assert all(verdict(c.F, u)[0] == "balanced" for u in outs), seed

        c8 = DuX("dux-2^8", rounds=6)
        rks8 = c8.key_schedule(c8.random_key(np.random.default_rng(seed)))
        outs8 = propagate(c8, rks8, 1, 4, 4)[3]        # layer 4
        for b in range(4):                              # position 2 only
            assert verdict(c8.F, outs8[4 * b + 2])[0] == "balanced", (seed, b)


def test_prime_field_does_not_get_the_extra_layer():
    """Same formal degree 65536 at layer 6, but over F_p the multiple of p-1 is
    the LEADING coefficient, not the sub-leading one -- so DuX(65537) is not
    balanced there.  This is the whole Frobenius story in one assertion."""
    c = DuX("dux-65537", rounds=8)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    outs = propagate(c, rks, 1, 6, 4)[5]
    st = [verdict(c.F, u)[0] for u in outs]
    assert all(st[4 * b + p] == "balanced" for b in range(4) for p in (0, 1, 2))
    assert all(st[4 * b + 3] == "not balanced" for b in range(4))


def test_enc_exponent_sets_are_full_intervals():
    """Why tools/exponent_sets.py cannot prove those cells: from layer 2 on the
    exponent set is the whole interval [0, D], so it carries no information
    beyond the max-plus degree D."""
    for n, layers in ((8, 3), (16, 5)):
        for r in expsets(n, 1, layers, direction="enc")[1:]:
            for w in range(16):
                assert r["set_size"][w] == r["max_exponent"][w] + 1, (n, r["layer"], w)


def test_O9_leading_pair_recursion_is_an_identity_over_GF2():
    """The O9 lemma table, checked as a symbolic identity rather than on keys.

    Inputs u0..u3 of formal degree m with leading/sub-leading (p_i, p_i'); the
    claim is  a' = p0 p3' + p0' p3,  b' = p3^2 p0',  c' = p0^2 p3^2 p3',
    y3' = p0^2 p3^4 (p0 p3' + p0' p3).  m >= 2 is needed: at m = 1 the slot
    2m - 1 coincides with m and u2's own leading term lands in a'."""
    sp = pytest.importorskip("sympy")
    syms = sp.symbols('p0 p0d p1 p1d p2 p2d p3 p3d alpha')
    p0, p0d, p1, p1d, p2, p2d, p3, p3d, al = syms

    def red(e):
        return sp.Poly(sp.expand(e), *syms, modulus=2).as_expr()

    class T:
        def __init__(self, d, w):
            self.d, self.w = d, [red(x) for x in w]

    def add(u, v):
        d = max(u.d, v.d)
        w = [0, 0]
        for x in (u, v):
            sh = d - x.d
            for t in range(2):
                if 0 <= t - sh < 2:
                    w[t] = w[t] + x.w[t - sh]
        return T(d, w)

    def mul(u, v):
        return T(u.d + v.d,
                 [u.w[0] * v.w[0], u.w[0] * v.w[1] + u.w[1] * v.w[0]])

    for m in (2, 3, 8, 128):
        x0, x1, x2, x3 = (T(m, [p0, p0d]), T(m, [p1, p1d]),
                          T(m, [p2, p2d]), T(m, [p3, p3d]))
        k = T(0, [al, 0])
        a = add(add(x2, mul(x0, x3)), k)
        b = add(add(x1, mul(x3, a)), k)
        c = add(add(x0, mul(a, b)), k)
        y3 = add(add(x3, mul(b, c)), k)
        claim = {"a": (2 * m, p0 * p3, p0 * p3d + p0d * p3),
                 "b": (3 * m, p0 * p3 ** 2, p3 ** 2 * p0d),
                 "c": (5 * m, p0 ** 2 * p3 ** 3, p0 ** 2 * p3 ** 2 * p3d),
                 "y3": (8 * m, p0 ** 3 * p3 ** 5,
                        p0 ** 2 * p3 ** 4 * (p0 * p3d + p0d * p3))}
        for name, t in (("a", a), ("b", b), ("c", c), ("y3", y3)):
            d, lead, subl = claim[name]
            assert t.d == d, (m, name)
            assert red(t.w[0] - lead) == 0, (m, name)
            assert red(t.w[1] - subl) == 0, (m, name)
            # the induction step: p0' = p3' = 0 kills every sub-leading term
            assert red(t.w[1].subs({p0d: 0, p3d: 0})) == 0, (m, name)


def test_O9_holds_on_the_degenerate_keys_too():
    """lambda = rk0_2 + rk0_0 rk0_3 + alpha = 0 makes the layer-1 a-word
    vanish identically.  The max-plus bound D does NOT drop, so O7 alone does
    not cover these keys -- but the leading coefficients vanish with it and the
    whole tracked window is zero from layer 3 on, so c_{D-1} = 0 still holds."""
    # (instance, layer, positions that must be balanced, the position whose
    #  formal degree is exactly q -- the one O7 cannot reach)
    for instance, layers, coords, frob in (("dux-2^8", 4, (2,), 2),
                                           ("dux-2^16", 6, (0, 1, 2, 3), 3)):
        c = DuX(instance, rounds=layers + 2)
        F = c.F
        rng = np.random.default_rng(11)
        for _ in range(3):
            rks = [list(map(int, rng.integers(0, F.q, size=16)))
                   for _ in range(layers + 3)]
            rks[0][2] = F.add(c.alpha, F.mul(rks[0][0], rks[0][3]))
            assert F.add(F.add(rks[0][2], F.mul(rks[0][0], rks[0][3])),
                         c.alpha) == 0
            prof = propagate(c, [tuple(r) for r in rks], 1, layers, 4)
            outs = prof[layers - 1]
            for b in range(4):
                for p in coords:
                    assert verdict(F, outs[4 * b + p])[0] == "balanced"
            # the max-plus bound is unchanged: this is not an O7 case
            assert outs[frob].d == F.q
            # and the window is identically zero from layer 3 on
            for l in range(2, layers):
                assert all(all(v == 0 for v in u.w) for u in prof[l])
