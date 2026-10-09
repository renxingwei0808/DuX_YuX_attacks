"""E15 / W29 (T2) -- mixed structures over F_p: full-field word x coset word.

Pins:
  * the threshold T = sum_i T_i and the SEPARATE plain threshold (an unsigned
    coset factor at exponent 0 is |H| != 0, so only the WEIGHTED sums reach
    T = sum_i T_i);
  * the weight filter (a_i >= 1 and 2^{k_i} does not divide a_i on every coset
    axis, nothing on the full-field axes) and its closed-form count;
  * the toy zero-sum, measured, against both thresholds;
  * that the streamed and the in-memory assemblies produce the same rows on a
    mixed structure.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for _p in (ROOT, os.path.join(ROOT, "tools"), E07,
           os.path.join(ROOT, "experiments", "E15_mixed_coset")):
    sys.path.insert(0, _p)

import mixed as MX                                               # noqa: E402
import weighted as WT                                            # noqa: E402
import zero_sum_criterion as zsc                                 # noqa: E402
from dux.field import make_field                                 # noqa: E402
from dux.registry import get_cipher                              # noqa: E402


def test_threshold_adds_per_word():
    assert zsc.threshold(65537, [3, 7], coset=[14, None]) == 65536 + 16384
    assert zsc.threshold(65537, [3, 7], coset=[13, None]) == 65536 + 8192
    assert zsc.threshold(65537, [3], coset=[15]) == 32768
    assert zsc.threshold(65537, [3, 7]) == 2 * 65536              # unchanged
    assert zsc.threshold_plain(65537, [3, 7], [14, None]) == 65536
    assert zsc.threshold_plain(193, [3, 7], [6, None]) == 192


def test_criterion_reports_both_readings():
    r = zsc.patterns(193, [3, 7], 6, None, [6, None], "dec", "dux")
    assert r["threshold"] == 256 and r["threshold_plain"] == 192
    # layer 5: D = (112, 153, 209, 97); 209 < 256 but 209 >= 192
    assert r["patterns"][4] == "1111"
    assert r["patterns_plain"][4] == "1101"
    assert r["log2_data"] == round(6 + np.log2(193), 2)


def test_coset_sums_are_nonzero_exactly_on_the_multiples():
    c = MX.subgroup_coset(193, 6, rep=5)
    assert len(c) == 64
    for m in range(0, 193, 1):
        s = sum(pow(int(x), m, 193) for x in c) % 193
        assert (s != 0) == (m % 64 == 0), m


def test_weight_filter_and_count():
    F = make_field("65537")
    pl = WT.plan(F, [3, 7], 10, "dec", "dux", "0001", dims=1, nweights=5,
                 coset=[14, None])
    assert pl.margin == 11694 and pl.usable_weights == 11693
    assert pl.weights[:3] == [(1,), (2,), (3,)]        # a = 0 is excluded
    p2 = WT.plan(F, [3, 7], 10, "dec", "dux", "0001", dims=2, nweights=5,
                 coset=[13, None])
    assert p2.margin == 3502
    assert p2.usable_weights == 3501 * 3502 // 2 == 6130251
    assert all(a[0] >= 1 for a in p2.weights)
    # the closed form agrees with brute force on a small case
    mods = [8, 0]
    brute = sum(1 for a0 in range(40) for a1 in range(40)
                if a0 + a1 < 40 and a0 >= 1 and a0 % 8)
    assert WT.coset_weight_count(2, 40, mods) == brute


def test_mixed_zero_sum_on_toy_193_two_keys():
    """Layer 5 of toy-193 with a 2^6-coset word: position 2 (D = 209) is
    balanced for every weight a < 47 = 256 - 209 and NOT at a = 0 or a >= 47."""
    import zero_sum_mixed as ZM
    c = get_cipher("toy-193", rounds=6)
    F = c.F
    ks = MX.parse_mixed("6,full", 2)
    for seed in (2026, 7):
        rng = np.random.default_rng(seed)
        rks = c.key_schedule(c.random_key(rng))
        axes = MX.mixed_axes(F, ks, seed)
        consts = [int(v) for v in rng.integers(0, F.q, size=16)]
        X, x0 = ZM.structure(c, axes, [3, 7], consts)
        sums = ZM.layer_sums(c, rks, X, x0, 5, [1, 46, 47])
        pos2 = lambda t: all(sums[4][t][4 * b + 2] == 0 for b in range(4))
        assert not pos2(0)                    # the plain sum: 209 >= 192
        assert pos2(1) and pos2(2)            # a = 1 and a = 46
        assert not pos2(3)                    # a = 47 = T - D
        # position 3 (D = 97) is balanced in every column
        assert all(all(sums[4][t][4 * b + 3] == 0 for b in range(4))
                   for t in range(4))


def test_streamed_mixed_rows_equal_the_in_memory_rows():
    from assemble_fast import MomentLayout, Precomp, _pow_matrix, rows_from_moments
    from attack_2round_toy import linear_row, structure_data
    rounds, layers, active = 7, 5, "3,7"
    c = get_cipher("toy-193", rounds=rounds)
    F = c.F
    ks = MX.parse_mixed("6,full", 2)
    pts = MX.mixed_axes(F, ks, 2026)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    ws = [(a,) for a in (1, 2, 5)]
    rows, _order, N = WT.weighted_rows(pre, c, rks, rounds, 3, 2026 + 5000,
                                       active, ws, lrow=lrow, points=pts)
    assert N == 64 * 193
    # the reference: the definition, summed point by point
    P = structure_data(c, rks, rounds, 3, 2026 + 5000, active, values=list(pts))
    layout = MomentLayout(pre)
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    idx = np.arange(N, dtype=np.int64)
    xv = np.asarray(pts[0])[(idx // 193) % 64]
    ref = []
    for (a,) in ws:
        w = np.array([pow(int(v), a, F.p) for v in xv], dtype=np.int64)
        pm = {b: (PW[b] * w[None, :]).sum(axis=1) % F.p for b in pre.unknown}
        Mom = {}
        for A in pre.unknown:
            for B in pre.unknown:
                Mom[(A, B)] = ((PW[A] * w[None, :]) % F.p).astype(np.float64) @ \
                    PW[B].astype(np.float64).T
                Mom[(A, B)] = np.asarray(Mom[(A, B)], dtype=np.int64) % F.p
        from assemble_fast import FlatMoments
        ref.extend(rows_from_moments(pre, FlatMoments(pm, Mom, N), lrow))
    assert np.array_equal(rows, np.asarray(ref, dtype=np.int64))


def test_coset_weight_residues_bound_the_rows_one_structure_gives():
    """On a coset of order 2^k, x^{a+2^k} = x^a * g^{2^k}: two weights whose
    coset exponents agree modulo 2^k give PROPORTIONAL rows.  So the rows a
    single mixed structure contributes is the number of distinct residue
    tuples, not `plan.usable_weights` (which counts admissible exponents and
    caps them by W25's axis bound -- an upper bound, loose exactly when a
    margin exceeds a coset order).

    toy-193, layer 5, `0001` combined row, two weight axes: margin 159 > 64, so
    the first 8300 admissible vectors carry only 6174 distinct residues -- and
    6174 is exactly the rank the attack measures
    (`results/E15_mixed_coset/fast_toy-193_mixed_c6_0001_d2_s*.json`).

    At all three S20 targets the margin is BELOW the coset order, so nothing
    collides and `usable_weights` is attained."""
    F = make_field("193")
    pl = WT.plan(F, [3, 7], 5, "dec", "dux", "0001", dims=2, nweights=8300,
                 coset=[6, None])
    assert pl.margin == 159 and pl.axis_cap == 64 * 193
    mod = MX.weight_moduli(F, [6, None])
    assert mod == [64, 192]
    assert len(pl.weights) == 8300
    assert MX.independent_weights(pl.weights, mod) == 6174
    assert MX.residues_collide(pl.margin, [6, None])
    # the best any 8300-vector choice could do at this margin
    best = len({(r0, r1) for r0 in range(1, 64) for r1 in range(159)
                if r0 + r1 < 159})
    assert best == 8001 < 8198        # the F_p dim-K-1 ceiling: one structure
    for k, margin in ((14, 11694), (13, 3502), (15, 13951)):
        assert not MX.residues_collide(margin, [k, None]), k


def test_streamed_driver_mixed_moments_equal_the_definition():
    """S20 (R9): the STREAMED driver's F_p slice phase on a mixed structure --
    `attack_12round._block_moments` with `values = mixed_axes(...)`, then the
    Vandermonde of the coset axis -- must give M_a = sum_x x_3^a g(x) point by
    point, where x_3 runs over the coset (the slowest axis) and x_7 over all
    of F_p.  This is the path `s20_k14.sh` runs at 2^30; the in-memory
    `weighted_rows(points=...)` is pinned separately above."""
    from assemble_fast import MomentLayout, Precomp, _pow_matrix, pow_field_vec
    from attack_2round_toy import structure_data
    import attack_12round as A12

    rounds, layers = 7, 5
    c = get_cipher("toy-193", rounds=rounds)
    F = c.F
    ks = MX.parse_mixed("4,full", 2)
    values = MX.mixed_axes(F, ks, 2026 + 4000)
    svals = values[0]
    nslice, slice_len = len(svals), F.q
    assert (nslice, slice_len) == (16, 193)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    weights = [a for a in range(1, 40) if a % nslice != 0][:6]

    G = np.zeros((nslice, layout.width), dtype=np.int64)
    A12._W.update(c=c, rks=rks, rounds=rounds, pos=3, active=[3, 7],
                  seed=seed, slice_len=slice_len, pre=pre, layout=layout,
                  values=values, G=G, inner_mask=None)
    # two worker tasks, as the driver's task list would cut them
    A12._block_moments((0, 9, 0))
    A12._block_moments((9, nslice, 9))
    V = A12.vandermonde(F, svals, weights)
    acc = ((V.T.astype(np.float64) @ G.astype(np.float64)) % F.p).astype(np.int64)

    P = structure_data(c, rks, rounds, 3, seed, [3, 7], values=list(values))
    N = len(P[0])
    assert N == nslice * slice_len
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    xv = svals[(np.arange(N) // slice_len) % nslice]
    n = layout.nP
    for i, a in enumerate(weights):
        w = pow_field_vec(F, xv, a) % F.p
        ref = np.zeros(layout.width, dtype=np.int64)
        for b in layout.blocks:
            o = layout.off_pm[b]
            ref[o:o + n] = (PW[b] * w[None, :]).sum(axis=1) % F.p
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            ref[o:o + n * n] = ((((PW[A] * w[None, :]) % F.p).astype(np.float64)
                                 @ PW[B].astype(np.float64).T) % F.p
                                ).astype(np.int64).ravel()
        assert np.array_equal(acc[i], ref), f"weight {a}"
