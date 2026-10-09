"""E13 / W27 (T1) -- the cubic coordinate b as a second cheap coordinate.

Pins the four things the R9 memo marks [verified] or [to be tested]:
  * the {1,2} kernel table over the six fields, and the tau-valuations that say
    the `1101` combined row does NOT collapse in characteristic 2;
  * the exact column count M_b of the extended linearisation;
  * that the extended rows are the ones they claim to be (they reproduce
    sum_P W'_{4j+c} exactly, weighted and unweighted);
  * that they vanish at the true key exactly on the admissible weights.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for _p in (ROOT, os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round"),
           os.path.join(ROOT, "experiments", "E13_b_coordinate")):
    sys.path.insert(0, _p)

import bcoord as BC                                              # noqa: E402
import kernel_table as KT                                        # noqa: E402
from cheap_rows import PATTERNS, describe                        # noqa: E402
from dux.registry import get_cipher                              # noqa: E402
from dux.sbox import vS                                          # noqa: E402
from assemble_fast import _pow_matrix, pow_field_vec             # noqa: E402
from attack_2round_toy import linear_row, structure_data         # noqa: E402

# technique T1 (docs/glossary.md), the two DuX rows
MEMO_DUX_CHEAP_2 = [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4]
MEMO_DUX_CHEAP_12 = [2, 1, 5, 0, 4, 4, 8, 0, 2, 1, 5, 0, 4, 4, 8]
MEMO_YUX_CHEAP_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4]
MEMO_YUX_CHEAP_23 = [0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 8]


@pytest.mark.parametrize("field", ["2^4", "2^8", "2^16"])
def test_dux_kernel_table_cheap_12_matches_the_memo(field):
    got = [describe("dux", field, "dec", p, [1, 2])["dim_K"] for p in PATTERNS]
    base = [describe("dux", field, "dec", p, [2])["dim_K"] for p in PATTERNS]
    assert base == MEMO_DUX_CHEAP_2
    assert got == MEMO_DUX_CHEAP_12


@pytest.mark.parametrize("field", ["2^4", "2^8", "2^16"])
def test_yux_kernel_table_cheap_23_matches_the_memo(field):
    got = [describe("yux", field, "dec", p, [2, 3])["dim_K"] for p in PATTERNS]
    base = [describe("yux", field, "dec", p, [3])["dim_K"] for p in PATTERNS]
    assert base == MEMO_YUX_CHEAP_3
    assert got == MEMO_YUX_CHEAP_23


def test_prime_fields_differ_only_on_0001_and_1001():
    """The memo only tabulates characteristic 2; over F_p the same cheap set
    gives dim K = 1 instead of 2 on `0001` / `1001` and the rest is equal."""
    for field in ("193", "257", "65537"):
        got = [describe("dux", field, "dec", p, [1, 2])["dim_K"] for p in PATTERNS]
        diff = [(p, a, b) for p, a, b in zip(PATTERNS, got, MEMO_DUX_CHEAP_12)
                if a != b]
        assert [p for p, _a, _b in diff] == ["0001", "1001"]
        assert all(a == 1 and b == 2 for _p, a, b in diff)


def test_1101_kernel_does_not_collapse_in_characteristic_2():
    """The four kernel vectors of `1101` have tau-valuation < 3 on at least one
    cheap coordinate, so none of them is an orbit-sum row (W19-B).  The memo's
    generator g = rows {1, 3, 9} is in the kernel and its position-3 component
    is the unit (1,0,0,0)."""
    e = KT.kernel_entry("dux", "2^16", "1101", [1, 2])
    assert e["dim_K"] == 4
    assert [v["tau_min"] for v in e["vectors"]] == [2, 0, 0, 2]
    assert e["usable_vectors"] == 4
    # the generator: rows 1, 3 and 9 of the layer matrix
    R = e["rows"]
    g = [0] * len(R)
    for r in (1, 3, 9):
        g[R.index(r)] = 1
    span = [v["y"] for v in e["vectors"]]

    def rank(rows):
        rows = [list(r) for r in rows]
        rk, n = 0, len(rows[0])
        for c in range(n):
            piv = next((i for i in range(rk, len(rows)) if rows[i][c] & 1), None)
            if piv is None:
                continue
            rows[rk], rows[piv] = rows[piv], rows[rk]
            for i in range(len(rows)):
                if i != rk and rows[i][c] & 1:
                    rows[i] = [(a ^ b) & 1 for a, b in zip(rows[i], rows[rk])]
            rk += 1
        return rk
    assert rank(span) == 4 and rank(span + [g]) == 4        # g is in the kernel
    # its position-3 component is the unit of F_2[Z_4]
    per3 = [g[R.index(4 * b + 3)] for b in range(4)]
    assert per3 == [1, 0, 0, 0]
    assert KT.tau_valuation(per3) == 0
    # and `0001` is only PARTIALLY collapsed: tau = 2, not 0 and not 3
    e01 = KT.kernel_entry("dux", "2^16", "0001", [1, 2])
    assert e01["dim_K"] == 2
    assert [v["tau_min"] for v in e01["vectors"]] == [2, 2]
    assert all(v["per_position"]["3"] in ([1, 0, 1, 0], [0, 1, 0, 1])
               for v in e01["vectors"])


def test_extended_column_count():
    """M_b = 47 749, from the closed form AND from the built column set.

    The paper's M = 18 025 (characteristic 2) / 38 051 (F_p) is reproduced by
    the same bookkeeping, which is what makes the two comparable."""
    c = get_cipher("toy-2^4", rounds=5)
    pre = BC.PrecompB(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    counts = BC.column_count(c.F.q - 1, pre.U, pre.S)
    assert counts["M"] == 18025
    assert counts["M_b"] == 47749 == len(pre.mons)
    assert counts["K"] == 50 and counts["K_nc"] == 40 and counts["K+K"] == 434
    assert round(counts["ratio"], 2) == 2.65


def _direct_outer_words(c, P, rk0, rk1, lrow):
    F = c.F
    Y = [None] * 16
    for b in range(4):
        x = tuple(F.vadd(P[4 * b + i], rk0[4 * b + i]) for i in range(4))
        y = vS(F, x, c.alpha)
        for i in range(4):
            Y[4 * b + i] = y[i]
    U = []
    for i in range(16):
        acc = np.zeros(len(P[0]), dtype=np.int64)
        for l in range(16):
            if lrow[l]:
                acc ^= F.vmul(Y[(i + l) % 16], np.int64(lrow[l]))
        U.append(acc)
    W = [None] * 16
    for b in range(4):
        x = tuple(F.vadd(U[4 * b + i], rk1[4 * b + i]) for i in range(4))
        y = vS(F, x, c.alpha)
        for i in range(4):
            W[4 * b + i] = y[i]
    return W


@pytest.mark.parametrize("active,weights", [
    ("3", [(0,), (1,), (3,)]),
    ("3,7", [(0, 0), (1, 0), (0, 1), (2, 3)]),
])
def test_extended_rows_reproduce_the_outer_word_sums(active, weights):
    """row . (true monomial vector) == sum_P x^a W'_{4j+c} for c in {1,2}."""
    c = get_cipher("toy-2^4", rounds=5)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    eq, _T = c.equivalent_fixedL0_keys(rks)
    P = structure_data(c, rks, 5, 3, 7777, active)
    N = len(P[0])
    s = len(active.split(","))
    idx = np.arange(N, dtype=np.int64)
    xvals = [(idx // (F.q ** (s - 1 - j))) % F.q for j in range(s)]
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    truth = BC.true_key_vector(pre, eq[0], eq[1])
    W = _direct_outer_words(c, P, eq[0], eq[1], lrow)
    for a, mom in BC.weighted_moments_b(pre, PW, xvals[:len(weights[0])], weights):
        w = np.ones(N, dtype=np.int64)
        for j, e in enumerate(a):
            w = F.vmul(w, pow_field_vec(F, xvals[j], int(e)))
        rows = BC.rows_from_moments_b(pre, mom, lrow)
        for j in range(4):
            for co in (1, 2):
                want = int(F.vsum(F.vmul(w, W[4 * j + co])))
                got = int(np.bitwise_xor.reduce(
                    F.vmul(np.asarray(rows[(j, co)]), truth)))
                assert got == want, (a, j, co)


def test_combined_rows_vanish_exactly_on_the_admissible_weights():
    """toy-2^4, layer 3, pattern 1101: the margin is 4, so the combined rows of
    the cheap set {1,2} vanish at the true key for a = 0..3 and not at a = 4."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E13_b_coordinate"))
    import run_toy as RT
    import weighted as WT
    from cheap_rows import block_coeffs
    c = get_cipher("toy-2^4", rounds=5)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    eq, _T = c.equivalent_fixedL0_keys(rks)
    truth = BC.true_key_vector(pre, eq[0], eq[1])
    pl = WT.plan(F, [3], 3, "dec", "dux", "1101", dims=1, cheap_set=(1, 2))
    assert pl.margin == 4 and pl.usable_weights == 4
    ycoeff = block_coeffs("dux", "2^4", "dec", "1101", [1, 2])
    assert len(ycoeff) == 4
    R, _o, _n = RT.structure_rows_b(pre, c, rks, 5, 2026 + 5000, "3",
                                    pl.weights, lrow, ycoeff)
    assert all(int(np.bitwise_xor.reduce(F.vmul(r, truth))) == 0 for r in R)
    # and the bound is EXACTLY tight: the first weight outside the margin,
    # a = 4, already produces a row that does not vanish
    for bad in ((4,), (5,), (6,)):
        R2, _o, _n = RT.structure_rows_b(pre, c, rks, 5, 2026 + 5000, "3",
                                         [bad], lrow, ycoeff)
        assert any(int(np.bitwise_xor.reduce(F.vmul(r, truth))) for r in R2), bad


def test_default_monomial_set_is_unchanged():
    """The `extra` / `extra_mons` hooks default to empty, so the published
    column counts do not move."""
    from assemble_fast import Precomp
    for inst, M in (("toy-2^4", 18025), ("dux-2^8", 18025), ("toy-257", 38051)):
        c = get_cipher(inst, rounds=5)
        pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
        assert len(pre.mons) == M


def test_pseudo_structure_rows_are_independent_at_small_counts():
    """`rank_b.py`'s machinery, in miniature: the extended combined rows of
    realizable pseudo-structures are independent while there are few of them,
    which is what makes the saturation point meaningful.  (The full ceiling run
    is minutes, not a test -- see experiments/E13_b_coordinate/rank_b.py.)"""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E13_b_coordinate"))
    import rank_b as RB
    from cheap_rows import block_coeffs
    from attack_2round_toy import linear_row as _lrow
    c = get_cipher("toy-2^4", rounds=5)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    ycomb = block_coeffs("dux", "2^4", "dec", "1101", [1, 2])
    rng = np.random.default_rng(2026)
    acc, nrows = RB.build_rows(pre, _lrow(c, 0),
                               lambda: BC.realizable_moments_b(pre, rng, 60),
                               12, ycomb)
    assert nrows == 4
    r = RB.determinacy(pre, F, acc)
    assert r["rows"] == 48 and r["rank"] == 48
    assert r["monomials"] == 47749


def test_pure_outer_key_columns_are_identically_zero():
    """The sixteen columns k'_i with an EMPTY inner part carry the factor
    sum_P 1 = q^s = 0, so they vanish in every row -- combined or not.  This is
    why `rank_b.py`'s "outer key columns determined" count stays 0 and is a
    check on the (1)-(6) expansion rather than the rule-2 gate; the gate is the
    sixteen degree-1 INNER columns."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E13_b_coordinate"))
    import rank_b as RB
    from cheap_rows import block_coeffs
    from attack_2round_toy import linear_row as _lrow
    c = get_cipher("toy-2^4", rounds=5)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    cols = np.array(sorted(RB.outer_key_columns(pre).values()), dtype=np.int64)
    assert len(cols) == 16
    rng = np.random.default_rng(11)
    lrow = _lrow(c, 0)
    plain, _ = RB.build_rows(pre, lrow,
                             lambda: BC.realizable_moments_b(pre, rng, 60),
                             6, None)
    assert plain.shape[0] == 48                      # 8 rows per structure
    assert not plain[:, cols].any()
    ycomb = block_coeffs("dux", "2^4", "dec", "1101", [1, 2])
    comb, _ = RB.build_rows(pre, lrow,
                            lambda: BC.realizable_moments_b(pre, rng, 60),
                            6, ycomb)
    assert not comb[:, cols].any()
    # the degree-1 INNER key columns, by contrast, are all present
    d1 = np.array(sorted(RB.degree_one_columns(pre).values()), dtype=np.int64)
    assert len(d1) == 16
    assert bool((comb[:, d1] != 0).any(axis=0).all())


def test_default_cheap_rows_table_is_unchanged():
    """R9 rule 5: the new `--cheap` / `--all-fields` switches must leave the
    published O10 table alone.  `table()` with no arguments still walks the
    four TABLE_FIELDS over both ciphers and both directions (240 cells), and
    the six-field variant reproduces `results/E11_cheap_rows/cheap_rows_table.json`
    cell for cell."""
    import json
    from cheap_rows import ALL_FIELDS, PATTERNS, TABLE_FIELDS, table
    assert TABLE_FIELDS == ["2^16", "65537", "193", "257"]
    assert ALL_FIELDS == ["2^4", "2^8", "2^16", "193", "257", "65537"]
    rows = table()
    assert len(rows) == 2 * 2 * len(TABLE_FIELDS) * len(PATTERNS) == 240
    assert sorted({r["field"] for r in rows}) == sorted(TABLE_FIELDS)
    pub = os.path.join(ROOT, "results", "E11_cheap_rows",
                       "cheap_rows_table.json")
    if os.path.exists(pub):
        key = lambda r: (r["cipher"], r["direction"], r["field"], r["balanced"])
        got = sorted(table(fields=ALL_FIELDS), key=key)
        want = sorted(json.load(open(pub)), key=key)
        assert len(got) == len(want) == 360
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


# ------------------------------------------------ S19 (R9): the streamed port --
def _stream_rows_b(instance, layers, active, combine, weights, block_slices=1,
                   point_chunk=1 << 16, nparts=1, a1=0):
    """`attack_12round.py --cheap-set 1,2`'s phases 1-3 in-process (no Pool):
    the worker fills the wide slice table, the parent XORs the Vandermonde
    combination and assembles the combined rows -- the very code S19 runs on
    the 2^32 structure, driven through the module-level `_W` as the S13-A
    tests do."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round",
                                    "fast"))
    from assemble_fast_2n import _logs
    from attack_2round_toy import active_words
    import attack_12round as A12
    import weighted as WT

    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    act = active_words(3, active)
    pre, layout, slice_fn = A12.precomp_b(c)
    assert isinstance(pre, BC.PrecompB) and slice_fn is BC.char2_slice_moments_b
    q, npts = F.q, F.q ** len(act)
    L = npts // q
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    G = np.zeros((q * nparts, layout.width), dtype=np.int64)
    saved = dict(A12._W)          # `_W` is module-global: leave it as found
    try:
        # T1 + T3: a second weight axis a1 on the second active word is the
        # per-pass inner mask y^{a1} (the driver's --weight-dims 2 pass)
        values_full = [np.arange(q, dtype=np.int64)] * len(act)
        inner_mask = (A12.inner_weight_mask(F, values_full, 1, (a1,)) if a1
                      else None)
        A12._W.update(c=c, rks=rks, rounds=rounds, pos=3, active=act, seed=seed,
                      slice_len=L, pre=pre, layout=layout, slice_fn=slice_fn,
                      point_chunk=point_chunk, G=G, values=None,
                      inner_mask=inner_mask)
        if nparts == 1:
            for v in range(0, q, block_slices):
                A12._block_moments_2n((v, min(v + block_slices, q), v, 0, 1))
        else:
            for v in range(q):
                for pt in range(nparts):
                    A12._block_moments_2n((v, v + 1, v * nparts, pt, nparts))
            G = np.bitwise_xor.reduce(G.reshape(q, nparts, layout.width), axis=1)
        ACC = np.zeros((len(weights), layout.width), dtype=np.int64)
        A12._W.update(LG=np.ascontiguousarray(_logs(F, G).T),
                      LV=A12.vandermonde_logs(F, np.arange(q, dtype=np.int64),
                                              weights),
                      ACC=ACC)
        A12._weight_acc_2n((0, len(weights)))
    finally:
        A12._W.clear()
        A12._W.update(saved)
    ycomb = (WT.combine_vectors("dux", F, "dec", combine, cheap=[1, 2])
             if combine else None)
    rows = A12.assemble_rows(pre, layout, ACC, npts, weights, ycomb,
                             linear_row(c, 0))
    order = A12.row_order(pre, weights, ycomb)
    return c, pre, rks, act, rows, order, seed, ycomb


@pytest.mark.parametrize("active,combine,nw,bs,pc,nparts", [
    ("3,7", "1101", 5, 1, 1 << 16, 1),     # 16 slices of 16 points, one per task
    ("3,7", "1101", 5, 4, 1 << 16, 1),     # four slices per task
    ("3,7", None, 3, 1, 1 << 16, 1),       # the eight uncombined rows
    ("3,7,11", "1101", 4, 1, 64, 4),       # 16 slices of 256 points, split 4 ways
])
def test_streamed_t1_rows_equal_the_in_memory_extended_rows(active, combine, nw,
                                                             bs, pc, nparts):
    """S19: the streamed cheap-set-{1,2} rows must equal `bcoord.structure_rows_b`
    (the in-memory driver every W27 number came from) weight for weight and
    row for row, with and without the `1101` combination, whatever the task
    cut (`--block-slices`, `--point-chunk`, `--slice-parts`)."""
    weights = list(range(nw))
    c, pre, rks, act, rows, order, seed, ycomb = _stream_rows_b(
        "toy-2^4", 3, active, combine, weights, bs, pc, nparts)
    lrow = linear_row(c, 0)
    if ycomb is not None:
        assert len(ycomb) == 4 and isinstance(ycomb[0], dict)
        ref, ref_order, n = BC.structure_rows_b(pre, c, rks, 5, seed, active,
                                                [(a,) for a in weights], lrow, ycomb)
        assert n == c.F.q ** len(act)
        assert rows.shape == ref.shape == (nw * 4, len(pre.mons))
        assert np.array_equal(rows, ref)
        assert [o[1] for o in order] == [("cheap12", t) for _a, t in ref_order]
    else:
        # the eight per-(block, coordinate) rows, in the driver's fixed order
        from attack_2round_toy import structure_data
        P = structure_data(c, rks, 5, 3, seed, active)
        N = len(P[0])
        idx = np.arange(N, dtype=np.int64)
        xvals = [(idx // (c.F.q ** (len(act) - 1 - j))) % c.F.q
                 for j in range(len(act))]
        PW = {b: _pow_matrix(c.F, P, b, pre.pe_arr) for b in pre.unknown}
        ref = []
        for _a, mom in BC.weighted_moments_b(pre, PW, xvals[:1],
                                             [(a,) for a in weights]):
            r8 = BC.rows_from_moments_b(pre, mom, lrow)
            ref.extend([r8[(j, 2)] for j in pre.outer] + [r8[(j, 1)] for j in pre.outer])
        assert rows.shape == (nw * 8, len(pre.mons))
        assert np.array_equal(rows, np.asarray(ref, dtype=np.int64))
        assert [o[1] for o in order][:8] == [0, 1, 2, 3, (0, 1), (1, 1), (2, 1), (3, 1)]


@pytest.mark.parametrize("active,a1,nparts", [
    ("3,7", 1, 1),          # y^1 on word 7, one task per slice
    ("3,7", 3, 1),
    ("3,7,11", 2, 4),       # y^2 on word 7 (word 11 free), slices split 4 ways
])
def test_streamed_t1_rows_with_a_second_weight_axis(active, a1, nparts):
    """S19's T1 + T3 variant (`s19_dux2p8.sh t1t3`: `--cheap-set 1,2
    --weight-dims 2`): the streamed cheap-set-{1,2} rows of one pass at the
    second-axis exponent a1 -- the inner mask y^{a1} handed to
    `char2_slice_moments_b(w=...)` -- must equal `bcoord.structure_rows_b`'s
    rows at the multi-index weights (a0, a1)."""
    weights = list(range(4))
    c, pre, rks, act, rows, order, seed, ycomb = _stream_rows_b(
        "toy-2^4", 3, active, "1101", weights, 1, 1 << 16, nparts, a1=a1)
    ref, ref_order, n = BC.structure_rows_b(
        pre, c, rks, 5, seed, active, [(a, a1) for a in weights],
        linear_row(c, 0), ycomb)
    assert n == c.F.q ** len(act)
    assert rows.shape == ref.shape == (len(weights) * 4, len(pre.mons))
    assert np.array_equal(rows, ref)
    # and NOT the a1 = 0 rows: the second axis does change the system
    ref0, _o, _n = BC.structure_rows_b(pre, c, rks, 5, seed, active,
                                       [(a, 0) for a in weights],
                                       linear_row(c, 0), ycomb)
    assert not np.array_equal(rows, ref0)


def test_stream_layout_round_trips_and_matches_layout_b():
    """`LayoutBStream.moments(flatten(m))` gives back the five tables, with the
    transposed `Mom` pairs restored, and its `char2_slice_moments_b` equals
    `_slice_moments` on the wide `LayoutB` table by table -- weighted too."""
    c = get_cipher("toy-2^4", rounds=5)
    F = c.F
    pre = BC.PrecompB(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    rng = np.random.default_rng(5)
    m = BC.realizable_moments_b(pre, rng, 64)
    ls = BC.LayoutBStream(pre)
    back = ls.moments(ls.flatten(m))
    for b in ls.blocks:
        assert np.array_equal(back.pm[b], m.pm[b])
        assert np.array_equal(back.pm2[b], m.pm2[b])
    for ab in ls.pairs2:
        assert np.array_equal(back.Mom[ab], m.Mom[ab]), ab
        assert np.array_equal(back.Mom2[ab], m.Mom2[ab]), ab
    # the per-slice kernel against the in-memory reference, three slices of a
    # 64-point structure, unweighted and with a per-point weight
    rks = c.key_schedule(c.random_key(rng))
    P = structure_data(c, rks, 5, 3, 77, "3,7")
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    bounds = [(0, 16), (16, 32), (48, 64)]
    lb = BC.LayoutB(pre)
    for w in (None, rng.integers(1, F.q, size=P[0].shape[0]).astype(np.int64)):
        G = BC.char2_slice_moments_b(pre, ls, PW, bounds, w=w)
        for k, (lo, hi) in enumerate(bounds):
            ref = lb.moments(BC._slice_moments(F, pre, lb, PW, lo, hi, w))
            got = ls.moments(G[k])
            for b in ls.blocks:
                assert np.array_equal(got.pm[b], ref.pm[b])
                assert np.array_equal(got.pm2[b], ref.pm2[b])
            for ab in ls.pairs2:
                assert np.array_equal(got.Mom[ab], ref.Mom[ab]), (ab, w is None)
                assert np.array_equal(got.Mom2[ab], ref.Mom2[ab]), (ab, w is None)


def test_driver_cli_cheap_set_12_writes_its_json(tmp_path):
    """The script end to end with `--cheap-set 1,2` (S19), through the JSON:
    the T1 kernel vectors are dicts keyed by (block, coordinate) and must
    serialise (the first server run crashed exactly there, after the solve)."""
    import json
    import subprocess
    E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
    out = subprocess.run(
        [sys.executable, os.path.join(E07, "attack_12round.py"),
         "--instance", "toy-2^4", "--layers", "3", "--active", "3,7",
         "--cheap-set", "1,2", "--combine", "1101", "--weights", "3",
         "--normalise", "--structures", "2", "--procs", "2",
         "--slice-chunk", "16", "--block-slices", "4", "--seed", "2026",
         "--out", str(tmp_path), "--tag", "t1smoke"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "M_b = 47749 columns" in out.stdout
    assert "dim K = 4" in out.stdout
    res = json.load(open(tmp_path / "s10_t1smoke.json"))
    assert res["cheap_set"] == [1, 2] and res["monomials"] == 47749
    assert res["equations"] == 2 * 3 * 4 and res["equation_rows"]["dim_K"] == 4
    y0 = dict((tuple(k), v) for k, v in res["equation_rows"]["y"][0])
    assert set(y0) == {(b, c) for b in range(4) for c in (1, 2)}
