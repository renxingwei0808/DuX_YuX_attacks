"""W16 -- weighted moments (O12) and cheap combined rows (O10) in the r_KR = 2
assembly.

The five checks the task asks for:
  (i)   the slice identity: `weighted_moment_stream` (per-slice plain moments
        times a Vandermonde matrix) equals the direct per-point weighted sums;
  (ii)  the weighted equations vanish at the TRUE key, both as the four
        per-outer-block rows and as the O10 combined row, and `--assume-L1`
        gives the same equation set with the y vector negated;
  (iii) a weight beyond the margin gives a NON-zero residual (the criterion is
        tight, and a wrong margin cannot pass unnoticed);
  (iv)  the same two over F_{2^n};
  (v)   a single subgroup coset with a weight that is not a multiple of 2^k is
        already a zero-sum (memo Sect. 5.5) -- no signed coset difference.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux import DuX                                              # noqa: E402
from dux.field import make_field                                 # noqa: E402
from dux.registry import get_cipher                              # noqa: E402
from assemble_fast import (MomentLayout, Precomp, WeightedMoments,  # noqa: E402
                           _pow_matrix, combine_rows, pow_field_vec,
                           weight_vectors, weighted_moment_stream)
from attack_2round_toy import linear_row, monomial_value, structure_data  # noqa: E402
import weighted as WT                                            # noqa: E402


def _true_key_vector(c, pre, rks):
    """The linearisation monomials evaluated at the true key of the EQUIVALENT
    fixed-L0 cipher (O2).  The assembly always uses L0, and with t(rk^1) = 1 the
    real cipher's second round key is block-rotated relative to it, so the outer
    key words are rk'^1 = Rot_{4 t(rk^1)}(rk^1); the inner key rk^0 is
    unchanged, which is why the attack still recovers the master key."""
    F = c.F
    eq, _T = c.equivalent_fixedL0_keys(rks)
    return np.array([monomial_value(F, m, eq[0], eq[1], F.q - 1)
                     for m in pre.mons], dtype=np.int64)


def _residual(F, rows, truth):
    rows = np.asarray(rows, dtype=np.int64)
    if F.char == 2:
        out = np.zeros(rows.shape[0], dtype=np.int64)
        for j in range(rows.shape[1]):
            if truth[j]:
                out ^= F.vmul(rows[:, j], np.int64(truth[j]))
        return out
    return (rows.astype(np.float64) @ truth.astype(np.float64)) % F.p


# --- (i) the slice identity ------------------------------------------------
def test_slice_decomposition_equals_direct_weighted_sums():
    """M_a = sum_v v^{a_0} G_v^{(a_1)} with G the per-slice plain moments."""
    c = DuX("toy-193", rounds=8)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0], [0])            # one block keeps the test cheap
    P = structure_data(c, rks, 8, 3, 7000, "3,7")
    N = len(P[0])
    PW = {0: _pow_matrix(F, P, 0, pre.pe_arr)}
    idx = np.arange(N, dtype=np.int64)
    x0, x1 = (idx // F.q) % F.q, idx % F.q
    ws = weight_vectors(2, 3)                      # (0,0) (0,1) (1,0) (0,2) (1,1) (2,0)
    assert len(ws) == 6
    got = dict(weighted_moment_stream(pre, PW, [x0, x1], ws))
    for a in ws:
        w = (pow_field_vec(F, x0, a[0]) * pow_field_vec(F, x1, a[1])) % F.p
        pm = (PW[0].astype(np.float64) @ w.astype(np.float64) % F.p).astype(np.int64)
        assert np.array_equal(pm, got[a].pm[0] % F.p), f"pm at weight {a}"
        aw = ((PW[0] * w[None, :]) % F.p).astype(np.float64)
        bb = PW[0].astype(np.float64)
        chunk = int(2 ** 53 // (F.p * F.p))
        acc = np.zeros((PW[0].shape[0],) * 2, dtype=np.int64)
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            acc = (acc + (aw[:, lo:hi] @ bb[:, lo:hi].T % F.p).astype(np.int64)) % F.p
        assert np.array_equal(acc, np.asarray(got[a].Mom[(0, 0)]) % F.p), \
            f"Mom at weight {a}"


def test_weighted_moments_class_matches_the_stream():
    """`WeightedMoments.add_slice` (the rank-1 API used for small weight sets)
    accumulates the same thing as `weighted_moment_stream`."""
    c = DuX("toy-257", rounds=7)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0], [0])
    P = structure_data(c, rks, 7, 3, 7000, None)               # 257 points
    PW = {0: _pow_matrix(F, P, 0, pre.pe_arr)}
    x0 = np.arange(F.q, dtype=np.int64)
    ws = [(a,) for a in range(6)]
    layout = MomentLayout(pre)
    wm = WeightedMoments(pre, ws, layout)
    for v in range(F.q):                                        # one point per slice
        g = np.zeros(layout.width, dtype=np.int64)
        col = PW[0][:, v]
        g[layout.off_pm[0]:layout.off_pm[0] + layout.nP] = col
        g[layout.off_pair[(0, 0)]:] = np.outer(col, col).ravel() % F.p
        wm.add_slice(g, (v,), npoints=1)
    ref = dict(weighted_moment_stream(pre, PW, [x0], ws, layout))
    for i, a in enumerate(ws):
        m = wm.as_moments(i)
        assert np.array_equal(m.pm[0] % F.p, ref[a].pm[0] % F.p)
        assert np.array_equal(np.asarray(m.Mom[(0, 0)]) % F.p,
                              np.asarray(ref[a].Mom[(0, 0)]) % F.p)


# --- (ii) + (iii) the true key, over F_p -----------------------------------
@pytest.fixture(scope="module")
def toy257_setup():
    c = DuX("toy-257", rounds=7)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    return c, F, rks, pre, _true_key_vector(c, pre, rks)


def test_margins_are_the_ones_the_criterion_reports(toy257_setup):
    c, F, rks, pre, truth = toy257_setup
    assert WT.weight_margin(F, [3], 5)[0] == 47                 # positions {2,3}
    assert WT.weight_margin(F, [3], 5, combine="0001")[0] == 159  # position 3 only
    assert WT.weight_margin(make_field("257"), [3, 7], 6, combine="0001")[0] == 150
    assert WT.weight_margin(make_field("193"), [3, 7], 6, combine="0001")[0] == 22
    assert WT.weight_margin(make_field("2^4"), [3, 7, 11], 4, combine="0001")[0] == 19


def test_weighted_rows_vanish_at_the_true_key(toy257_setup):
    c, F, rks, pre, truth = toy257_setup
    lrow = linear_row(c, 0)
    ws = [(a,) for a in range(16)]
    rows, order, N = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws, lrow)
    assert rows.shape[0] == 16 * 4 and N == 257
    assert int(np.abs(_residual(F, rows, truth)).max()) == 0


def test_combined_row_vanishes_and_extends_the_weight_range(toy257_setup):
    c, F, rks, pre, truth = toy257_setup
    lrow = linear_row(c, 0)
    y = WT.combine_vectors("dux", F, "dec", "0001")
    assert len(y) == 1 and [v % F.p for v in y[0]] == [1, F.p - 1, 1, F.p - 1]
    # inside the plain margin AND in the range only the combined row reaches
    for lo, hi in ((0, 16), (47, 60), (150, 159)):
        ws = [(a,) for a in range(lo, hi)]
        rows, _, _ = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws, lrow,
                                      ycomb=y)
        assert rows.shape[0] == hi - lo
        assert int(np.abs(_residual(F, rows, truth)).max()) == 0, (lo, hi)


def test_weights_beyond_the_margin_do_not_vanish(toy257_setup):
    """The criterion is tight: a = margin already breaks the plain rows."""
    c, F, rks, pre, truth = toy257_setup
    lrow = linear_row(c, 0)
    ws = [(a,) for a in range(47, 52)]
    rows, _, _ = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws, lrow)
    assert int(np.abs(_residual(F, rows, truth)).max()) != 0
    y = WT.combine_vectors("dux", F, "dec", "0001")
    ws = [(a,) for a in range(159, 163)]
    rows, _, _ = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws, lrow, ycomb=y)
    assert int(np.abs(_residual(F, rows, truth)).max()) != 0


def test_assume_L1_gives_the_same_equation_set(toy257_setup):
    """O2: with L1 the kernel vector's cheap coefficients change sign, so the
    combined equations are the negatives of the L0 ones -- the same set."""
    c, F, rks, pre, truth = toy257_setup
    ws = [(a,) for a in range(6)]
    y0 = WT.combine_vectors("dux", F, "dec", "0001")
    from cheap_rows import block_coeffs
    y1 = block_coeffs("dux", "257", "dec", "0001", t=1)
    assert [(-v) % F.p for v in y0[0]] == [v % F.p for v in y1[0]]
    r0, _, _ = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws,
                                linear_row(c, 0), ycomb=y0)
    r1, _, _ = WT.weighted_rows(pre, c, rks, 7, 3, 7000, None, ws,
                                linear_row(c, 0), ycomb=y1)
    assert np.array_equal((-r0) % F.p, r1 % F.p)
    assert int(np.abs(_residual(F, r1, truth)).max()) == 0


# --- (iv) characteristic 2 --------------------------------------------------
def test_char2_weighted_and_combined_rows(toy257_setup):
    """toy-2^4, three words (3,7,11), layer 4 pattern 1101, combined row 0001."""
    c = DuX("toy-2^4", rounds=6)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    truth = _true_key_vector(c, pre, rks)
    lrow = linear_row(c, 0)
    margin, crit = WT.weight_margin(F, [3, 7, 11], 4, combine="0001")
    assert margin == 19 and crit["patterns"][3] == "1101"
    y = WT.combine_vectors("dux", F, "dec", "0001")
    assert [v % F.q for v in y[0]] == [1, 1, 1, 1]           # all ones in char 2
    ws = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 3), (5, 4, 6)]
    # position 2 is NOT balanced at this layer, so the four per-outer-block rows
    # do NOT vanish individually -- only their O10 combination does.  That is
    # the point of the lemma and it is what makes `1101` usable at all.
    plain, _, _ = WT.weighted_rows(pre, c, rks, 6, 3, 7000, "3,7,11",
                                   [(0, 0, 0)], lrow)
    assert int(np.abs(_residual(F, plain, truth)).max()) != 0
    rows, _, _ = WT.weighted_rows(pre, c, rks, 6, 3, 7000, "3,7,11", ws, lrow,
                                  ycomb=y)
    assert int(np.abs(_residual(F, rows, truth)).max()) == 0
    over = [(19, 0, 0), (10, 9, 1)]                          # |a| >= 19
    rows, _, _ = WT.weighted_rows(pre, c, rks, 6, 3, 7000, "3,7,11", over, lrow,
                                  ycomb=y)
    assert int(np.abs(_residual(F, rows, truth)).max()) != 0


# --- (v) a single subgroup coset -------------------------------------------
def test_single_coset_with_a_nonzero_weight_is_already_a_zero_sum():
    """Memo Sect. 5.5 on yuxtoy-257: 2^6 coset, a in {1,2,3,5,7} balances every
    word at layers 2 and 3 and fails at layer 4 (D = 104 > 64), while a = 0 and
    a = 64 (multiples of 2^k) fail already at layer 2."""
    c = get_cipher("yuxtoy-257", rounds=8)
    p = c.F.q
    k = 6
    g = 3                                    # a generator of F_257^*
    H = np.array([pow(g, (p - 1) // (1 << k) * t, p) for t in range(1 << k)],
                 dtype=np.int64)
    assert len(set(H.tolist())) == 1 << k
    coset = (H * pow(g, 5, p)) % p           # a nontrivial coset representative
    for seed in (2026, 7):
        rng = np.random.default_rng(seed)
        rks = c.key_schedule(c.random_key(rng))
        C = [np.full(len(coset), int(v), dtype=np.int64)
             for v in rng.integers(0, p, size=16)]
        C[0] = coset
        states = c.decrypt_layers(tuple(C), rks, 5, rounds=8, vec=True,
                                  yield_all=True)
        for a in (1, 2, 3, 5, 7):
            w = pow_field_vec(c.F, coset, a)
            for layer in (2, 3):
                s = [(int((w * states[layer - 1][i]).sum() % p)) for i in range(16)]
                assert max(s) == 0, f"layer {layer}, weight {a}, seed {seed}"
            s4 = [(int((w * states[3][i]).sum() % p)) for i in range(16)]
            assert max(s4) != 0, f"layer 4 should FAIL for weight {a}"
        for a in (0, 64):                    # multiples of 2^k: no cancellation
            w = pow_field_vec(c.F, coset, a)
            s2 = [(int((w * states[1][i]).sum() % p)) for i in range(16)]
            assert max(s2) != 0, f"weight {a} is a multiple of 2^{k}"
