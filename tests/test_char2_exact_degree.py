"""W20 / O13 -- the exact-degree machinery behind the characteristic-2 result.

Two independent implementations have to agree with each other and with the
table of an earlier, independent symbolic computation (the AUDIT_LAYER*
constants below):

  * `tools/degree_spectrum.py` measures the exact REDUCED degree from character
    sums (univariate) or from the reduced coefficient array (multi-word and
    full-block, added by W20);
  * `tools/char2_exact_degree.py` computes the exact FORMAL degree by dense
    symbolic expansion, with an FFT convolution for the long polynomials.

The numbers pinned below are the ones O13 rests on: the layer-2 collapse
(3 -> 1 at block 1 position 1), the layer-3/4 profiles, the fact that the
prime field shows no gap at all, and that a full block shows none either.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux.field import make_field                                 # noqa: E402
import char2_exact_degree as CE                                  # noqa: E402
import degree_spectrum as DS                                     # noqa: E402


# ------------------------------------------------------- the polynomial ring
@pytest.mark.parametrize("q", ["2^4", "2^8", "2^16", "257", "65537"])
def test_fft_convolution_is_exact(q):
    """The FFT path and the direct path must give the same product."""
    F = make_field(q)
    rng = np.random.default_rng(5)
    for L in (600, 1500):
        a = rng.integers(0, F.q, size=L).astype(np.int64)
        b = rng.integers(0, F.q, size=L).astype(np.int64)
        A, B = CE.Poly(F, a), CE.Poly(F, b)
        old = CE.FFT_CUTOFF
        try:
            CE.FFT_CUTOFF = 10 ** 9
            direct = (A * B).c
            CE.FFT_CUTOFF = 512
            fft = (A * B).c
        finally:
            CE.FFT_CUTOFF = old
        assert np.array_equal(direct, fft)


@pytest.mark.parametrize("q", ["2^4", "257"])
def test_inverse_evaluation_matrix_recovers_the_coefficients(q):
    """W20's multi-word mode inverts the evaluation map; on one variable that
    must return exactly the coefficients it started from."""
    F = make_field(q)
    rng = np.random.default_rng(3)
    coef = rng.integers(0, F.q, size=F.q).astype(np.int64)
    xs = np.arange(F.q, dtype=np.int64)
    vals = np.zeros(F.q, dtype=np.int64)
    xp = np.ones(F.q, dtype=np.int64)
    for e in range(F.q):
        term = F.vmul(xp, np.full(F.q, int(coef[e]), dtype=np.int64))
        vals = (vals ^ term) if F.char == 2 else (vals + term) % F.p
        xp = F.vmul(xp, xs)
    W = DS.inverse_evaluation_matrix(F)
    got = DS._axis_transform(F, W, vals[:, None])[:, 0]
    mod = F.q if F.char == 2 else F.p
    assert np.array_equal(got % mod, coef % mod)


# --------------------------------------------------------- the O13 numbers --
AUDIT_LAYER2 = [2, 3, 4, 5, 3, 1, 5, 5, 3, 4, 5, 7, 2, 3, 4, 5]
AUDIT_LAYER3 = [11, 14, 18, 25, 10, 12, 17, 22, 10, 12, 17, 22, 12, 10, 19, 22]
AUDIT_LAYER4 = [44, 47, 69, 91, 47, 44, 72, 91, 47, 50, 72, 97, 44, 47, 69, 91]


@pytest.mark.parametrize("seed", [2026, 7, 11])
def test_exact_formal_degrees_reproduce_the_audit_table(seed):
    """The independent table, on three keys, from the symbolic side."""
    _c, degs, _n = CE.exact_formal_degrees("yu2x-8", 0, 4, seed)
    assert degs[1] == AUDIT_LAYER2
    assert degs[2] == AUDIT_LAYER3
    assert degs[3] == AUDIT_LAYER4


def test_layer2_collapse_is_block_1_only_and_is_a_char2_effect():
    """Lemma 2 of Y07: u_1 + u_3 is a CONSTANT for exactly one of the four
    blocks in characteristic 2, and for none of them over F_p."""
    _c, _d, notes = CE.exact_formal_degrees("yu2x-8", 0, 2, 2026, explain=True)
    assert [n["collapses"] for n in notes] == [False, True, False, False]
    _c, _d, notes_p = CE.exact_formal_degrees("yuxtoy-257", 0, 2, 2026,
                                              explain=True)
    assert not any(n["collapses"] for n in notes_p)
    assert all(n["u1_plus_u3_degree"] == 2 for n in notes_p)


def test_prime_field_yux_has_no_gap_at_all():
    """YupX's prime field: the max-plus bound is an EQUALITY, not a bound."""
    from cipher_degree import profile
    _c, degs, _n = CE.exact_formal_degrees("yuxtoy-257", 0, 4, 2026)
    bound = profile([0], 4, "dec", False, "yux")
    for l in range(4):
        assert degs[l] == [int(v) for v in bound[l]]


def test_dux_char2_gap_is_a_single_layer_and_does_not_propagate():
    """DuX in characteristic 2 loses a little at layer 3 and nothing after --
    the accumulating gap is a YuX phenomenon (Y07 Sect. 0)."""
    from cipher_degree import profile
    _c, degs, _n = CE.exact_formal_degrees("dux-2^16", 3, 6, 2026)
    bound = profile([3], 6, "dec", True, "dux")
    gaps = [sum(int(bound[l][w]) - degs[l][w] for w in range(16))
            for l in range(6)]
    assert gaps[2] > 0                       # layer 3 does lose something
    assert gaps[3:] == [0, 0, 0]             # layers 4..6 are exact


# ------------------------------------------ the measured side (reduced degs)
def test_multiword_mode_agrees_with_the_univariate_scan():
    _c, _w, degs, _t = DS.exact_degrees("yuxtoy-2^4", 0, 4, 2026)
    _c2, axes, tot = DS.exact_total_degrees("yuxtoy-2^4", 4, 2026, active=(0,))
    assert axes == [0]
    for l in range(4):
        assert [d or 0 for d in degs[l]] == [t or 0 for t in tot[l]]


def test_full_block_is_tight_in_characteristic_2():
    """O11's substitution makes the four block variables independent, so the
    layer-1 relation Y_1 = Y_0 + g disappears and the bound is met exactly."""
    from cipher_degree import profile
    _c, axes, tot = DS.exact_total_degrees("yuxtoy-2^4", 3, 2026, free_block=0)
    assert axes == [0, 1, 2, 3]
    bound = profile([], 3, "dec", True, "yux", (0,))
    for l in range(3):
        assert tot[l] == [int(v) for v in bound[l]]


def test_single_word_is_loose_but_key_independent():
    """The same two words drop at layer 2 for every key -- that is O13."""
    tots = [DS.exact_total_degrees("yu2x-8", 2, s, active=(0,))[2]
            for s in (2026, 7, 11)]
    assert all(t[1] == AUDIT_LAYER2 for t in tots)
