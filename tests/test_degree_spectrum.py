"""W17 -- tools/degree_spectrum.py: exact reduced degrees from character sums.

  sum_{x in F_q} x^a Z(x) = -[X^{q-1-a}] Z(x),

so the smallest a with a nonzero weighted sum gives the exact reduced degree.
The tool is checked three ways: against the exact univariate polynomials of
experiments/E02_degree_bounds/symbolic_layers.py (DuX), against the max-plus
bound (which YuX meets exactly, DuX does not everywhere), and on the two
fields.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E02_degree_bounds"))

from dux import DuX                                   # noqa: E402
from cipher_degree import profile                     # noqa: E402
from degree_spectrum import exact_degrees             # noqa: E402
from symbolic_layers import symbolic_layers           # noqa: E402


def test_spectrum_equals_the_exact_univariate_polynomials_dux():
    """Same numbers as the dense-polynomial model of E02, word for word.

    Both sides must use the SAME linear layer.  `degree_spectrum.exact_degrees`
    decrypts through the equivalent fixed-L0 cipher (O2), because the real
    DuX picks L0/L1 from t_xor(rk^i) and the max-plus bound it is compared with
    assumes L0; so `symbolic_layers` is called with `force_L0=True` here.
    (With `force_L0=False` the two disagree from layer 2 on -- a permuted word
    order, not different degrees -- which is exactly the key-dependence that
    the E02 spectrum records in `results/E02_degree_bounds/` show.)"""
    c = DuX("toy-257")
    F = c.F
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    _P, states = symbolic_layers(c, rks, 3, 5, consts, force_L0=True)

    def deg(v):
        nz = np.nonzero(np.asarray(v) % F.p)[0]
        return int(nz[-1]) if nz.size else None

    sym = [[(deg(w) or None) if deg(w) else (None if deg(w) is None else 0)
            for w in st] for st in states]
    sym = [[None if (d is None or d == 0) else d for d in row] for row in sym]
    _c, _w, spec, _t = exact_degrees("toy-257", 3, 5, 2026)
    spec = [[None if d in (None, 0) else d for d in row] for row in spec]
    assert sym == spec


@pytest.mark.parametrize("instance,pos,layers", [("yuxtoy-257", 0, 5),
                                                 ("yuxtoy-257", 3, 5),
                                                 ("yuxtoy-193", 0, 5)])
def test_yux_max_plus_bound_is_met_exactly(instance, pos, layers):
    """Memo Sect. 2: on YuX every state word's exact reduced degree equals the
    max-plus bound until the bound saturates -- no cancellation anywhere."""
    from dux.registry import get_cipher
    q = get_cipher(instance).F.q
    bound = profile([pos], layers, cipher="yux")
    for seed in (2026, 7):
        _c, words, degs, _t = exact_degrees(instance, pos, layers, seed)
        for l in range(layers):
            for i in words:
                b, e = bound[l][i], degs[l][i]
                if b == 0:
                    assert e is None, f"layer {l+1} word {i}: bound 0 but degree {e}"
                elif b < q - 1:
                    assert e == b, f"layer {l+1} word {i}: bound {b}, exact {e}"
                else:
                    assert e is None or e <= q - 1


def test_dux_has_genuine_cancellations():
    """The same scan on DuX finds words whose exact degree is BELOW the bound,
    which is why the memo's 'no cancellation' statement is a YuX statement."""
    bound = profile([3], 5, cipher="dux")
    _c, words, degs, _t = exact_degrees("toy-257", 3, 5, 2026)
    loose = [(l + 1, i, bound[l][i], degs[l][i]) for l in range(5) for i in words
             if 0 < bound[l][i] < 256 and degs[l][i] is not None
             and degs[l][i] < bound[l][i]]
    assert loose, "expected at least one non-tight DuX cell"
    assert all(b - e == 1 for _l, _i, b, e in loose), \
        "the observed DuX cancellations all cost exactly one degree"


@pytest.mark.parametrize("instance,layers", [("yuxtoy-2^4", 4), ("yu2x-8", 4)])
def test_char2_spectrum_is_a_valid_bound(instance, layers):
    """F_{2^n}: the same identity holds (sum_x x^m = 0 unless (q-1) | m, m > 0).

    The max-plus bound stays VALID there but is no longer met everywhere: in
    characteristic 2 a product of two equal (or Frobenius-related) factors is a
    square, so degrees can collapse.  This is the difference between the F_p
    toys, where the memo's "no cancellation" statement holds word for word, and
    the binary ones."""
    from dux.registry import get_cipher
    q = get_cipher(instance).F.q
    bound = profile([0], layers, cipher="yux")
    _c, words, degs, _t = exact_degrees(instance, 0, layers, 2026)
    tight = loose = 0
    for l in range(layers):
        for i in words:
            b, e = bound[l][i], degs[l][i]
            if b == 0:
                assert e is None
                continue
            if e is None:
                loose += 1
                continue
            assert e <= max(b, q - 1), f"layer {l+1} word {i}: bound {b}, exact {e}"
            if b < q - 1:
                tight += (e == b)
                loose += (e < b)
    assert tight > 0, "the bound should still be met somewhere"


def test_zero_sum_layer_agrees_with_the_a_equals_0_column():
    """a = 0 is the ordinary zero-sum test: a word is balanced iff its exact
    reduced degree is below q - 1."""
    from dux.registry import get_cipher
    c = get_cipher("yuxtoy-257")
    q = c.F.q
    _c, words, degs, _t = exact_degrees("yuxtoy-257", 0, 5, 2026)
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, q, size=16)]
    C = [np.full(q, consts[i], dtype=np.int64) for i in range(16)]
    C[0] = np.arange(q, dtype=np.int64)
    states = c.decrypt_layers(tuple(C), rks, 5, vec=True, yield_all=True)
    for l, st in enumerate(states):
        for i in words:
            balanced = int(st[i].sum() % q) == 0
            assert balanced == (degs[l][i] is None or degs[l][i] < q - 1), \
                f"layer {l+1} word {i}"
