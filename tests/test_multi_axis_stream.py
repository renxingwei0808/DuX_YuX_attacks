"""S21 (R9 / T3) -- the streamed driver's JOINT multi-axis Vandermonde and
subspace structures (`attack_12round.py --vandermonde-axes m`,
`--subspace-dims`), pinned against the in-memory `weighted.weighted_rows`.

Why it exists.  `--weight-dims 2` (W28) walks the structure once per
second-axis exponent; the single-structure rows of T3 need thousands of weight
vectors, so on the (8,8,8,5) DuX(2^8) cell and on the Yu2X-8 full block that
is hundreds of passes.  Slicing the structure by its first m axes JOINTLY makes
the first m weight axes one Vandermonde over the sub-slices -- one pass for all
of them -- and leaves only the remaining axes as inner-mask passes.  The rows
must come out identical to the in-memory multi-index rows, and the m = 1
helpers must reduce to the old ones exactly.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for _p in (ROOT, E07, os.path.join(E07, "fast"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, _p)

from dux.registry import cipher_family, get_cipher               # noqa: E402
from assemble_fast import MomentLayout, Precomp, pow_field_vec   # noqa: E402
from assemble_fast_2n import _logs                               # noqa: E402
from attack_2round_toy import active_words, linear_row           # noqa: E402
import attack_12round as A12                                     # noqa: E402
import weighted as WT                                            # noqa: E402


def test_vandermonde_logs_multi_reduces_to_vandermonde_logs():
    F = get_cipher("toy-2^4", rounds=5).F
    vals = np.array([0, 1, 5, 0, 9, 15, 3], dtype=np.int64)
    weights = [0, 1, 2, 7]
    ref = A12.vandermonde_logs(F, vals, weights)
    got = A12.vandermonde_logs_multi(F, [vals], [(a,) for a in weights])
    assert np.array_equal(ref, got)
    # two axes: log(v^a w^b) = a log v + b log w, sentinel iff 0^(>0) appears
    w2 = np.array([3, 0, 1, 7, 0, 2, 4], dtype=np.int64)
    vecs = [(0, 0), (1, 0), (0, 2), (2, 3)]
    got = A12.vandermonde_logs_multi(F, [vals, w2], vecs)
    expx = np.asarray(F._exp, dtype=np.int64)
    for i, (a, b) in enumerate(vecs):
        want = F.vmul(pow_field_vec(F, vals, a), pow_field_vec(F, w2, b))
        for s in range(len(vals)):
            if want[s] == 0:
                assert got[i, s] == 2 * F.order
            else:
                assert expx[got[i, s]] == want[s]


def test_inner_weight_mask_reduces_to_the_repeat_mask():
    F = get_cipher("toy-2^4", rounds=5).F
    q = F.q
    for s in (2, 3):
        values = [np.arange(q, dtype=np.int64)] * s
        L = q ** (s - 1)
        for a1 in (1, 3):
            old = np.repeat(pow_field_vec(F, values[1], a1), L // q)
            got = A12.inner_weight_mask(F, values, 1, (a1,))
            assert np.array_equal(old, got)
    assert A12.inner_weight_mask(F, [np.arange(q)] * 3, 1, (0, 0)) is None
    # three words, joint axes m = 2, one inner axis: the mask is y^e over the
    # 16-point slice, i.e. the word's own value list
    got = A12.inner_weight_mask(F, [np.arange(q)] * 3, 2, (5,))
    assert np.array_equal(got, pow_field_vec(F, np.arange(q), 5))


def _stream_multi(instance, layers, act, values, vecs, vaxes, combine=None,
                  block_slices=1, point_chunk=1 << 16):
    """`attack_12round`'s characteristic-2 phases 1-3 in-process (no Pool)
    with the S21 joint Vandermonde: one pass per inner tuple, the module-level
    `_W` filled the way `main` fills it."""
    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    fam = cipher_family(instance)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher=fam)
    layout = MomentLayout(pre)
    sizes = [len(v) for v in values]
    npts = int(np.prod(sizes))
    nslice = int(np.prod(sizes[:vaxes]))
    L = npts // nslice
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    inner_list = sorted({tuple(w[vaxes:]) for w in vecs})
    ycomb = WT.combine_vectors(fam, F, "dec", combine) if combine else None
    rows, order = [], []
    saved = dict(A12._W)          # `_W` is module-global: leave it as found
    try:
        for inner in inner_list:
            _one_pass(c, rks, rounds, act, values, vaxes, inner, vecs, pre,
                      layout, L, nslice, npts, block_slices, point_chunk, seed,
                      ycomb, rows, order)
    finally:
        A12._W.clear()
        A12._W.update(saved)
    return c, pre, rks, np.asarray(rows, dtype=np.int64), order, seed, ycomb


def _one_pass(c, rks, rounds, act, values, vaxes, inner, vecs, pre, layout, L,
              nslice, npts, block_slices, point_chunk, seed, ycomb, rows, order):
    F = c.F
    if True:  # (kept flat: the body is the driver's per-pass sequence)
        wv = sorted(tuple(w[:vaxes]) for w in vecs if tuple(w[vaxes:]) == inner)
        mask = A12.inner_weight_mask(F, values, vaxes, inner)
        G = np.zeros((nslice, layout.width), dtype=np.int64)
        A12._W.update(c=c, rks=rks, rounds=rounds, pos=3, active=act, seed=seed,
                      slice_len=L, pre=pre, layout=layout,
                      slice_fn=A12.char2_slice_moments, point_chunk=point_chunk,
                      G=G, values=values, inner_mask=mask)
        for v in range(0, nslice, block_slices):
            A12._block_moments_2n((v, min(v + block_slices, nslice), v, 0, 1))
        ACC = np.zeros((len(wv), layout.width), dtype=np.int64)
        A12._W.update(LG=np.ascontiguousarray(_logs(F, G).T),
                      LV=A12.vandermonde_logs_multi(
                          F, A12.slice_axis_values(values, vaxes, 0, nslice), wv),
                      ACC=ACC)
        A12._weight_acc_2n((0, len(wv)))
        rows.extend(A12.assemble_rows(pre, layout, ACC, npts, wv, ycomb,
                                      linear_row(c, 0)))
        order.extend([tuple(w) + tuple(inner) for w in wv])


@pytest.mark.parametrize("instance,layers,act,vecs,vaxes,combine,bs", [
    # three words, two joint axes (256 sub-slices of 16 points), no inner axis
    ("toy-2^4", 3, [3, 7, 11], [(a, b) for a in range(4) for b in range(3)], 2, None, 1),
    ("toy-2^4", 3, [3, 7, 11], [(a, b) for a in range(3) for b in range(3)], 2, "1111", 8),
    # three words, two joint axes AND one inner axis (a pass per a2)
    ("toy-2^4", 3, [3, 7, 11],
     [(a, b, e) for a in range(3) for b in range(2) for e in range(2)], 2, None, 4),
    # three words, all three joint (4096 single-point sub-slices)
    ("toy-2^4", 3, [3, 7, 11], [(a, b, e) for a in range(2) for b in range(2)
                                for e in range(2)], 3, None, 64),
    # YuX two words, both joint
    ("yuxtoy-2^4", 3, [0, 4], [(a, b) for a in range(4) for b in range(3)], 2, None, 1),
])
def test_joint_vandermonde_rows_equal_the_in_memory_multi_index_rows(
        instance, layers, act, vecs, vaxes, combine, bs):
    q = get_cipher(instance, rounds=layers + 2).F.q
    c, pre, rks, rows, order, seed, ycomb = _stream_multi(
        instance, layers, act, [np.arange(q, dtype=np.int64)] * len(act),
        vecs, vaxes, combine, bs)
    F = c.F
    # the in-memory reference emits the vectors in the order given; the
    # streamed rows are grouped by inner tuple, so compare per vector
    ref, ref_order, n = WT.weighted_rows(pre, c, rks, layers + 2, 3, seed, act,
                                         vecs, lrow=linear_row(c, 0), ycomb=ycomb)
    assert n == F.q ** len(act)
    per = len(ycomb) if ycomb else 4
    ref_by = {}
    for i, (a, _tag) in enumerate(ref_order):
        ref_by.setdefault(tuple(a), []).append(ref[i])
    assert rows.shape == ref.shape
    for k, vec in enumerate(order):
        got = rows[k * per:(k + 1) * per]
        assert np.array_equal(got, np.asarray(ref_by[vec])), vec


def test_subspace_structure_rows_equal_the_in_memory_rows_on_the_same_points():
    """`--subspace-dims`: the value lists are the subspaces 0..2^m-1, the
    streamed rows equal `weighted_rows(points=those lists)`, two joint axes
    on a (4, 3, 2) structure of toy-2^4 (16 x 8 x 4 = 512 points)."""
    instance, layers, act = "toy-2^4", 3, [3, 7, 11]
    c0 = get_cipher(instance, rounds=layers + 2)
    values = A12.subspace_values(c0.F, [4, 3, 2])
    assert [len(v) for v in values] == [16, 8, 4]
    vecs = [(a, b) for a in range(3) for b in range(3)]
    c, pre, rks, rows, order, seed, _y = _stream_multi(instance, layers, act,
                                                       values, vecs, 2)
    ref, ref_order, n = WT.weighted_rows(pre, c, rks, layers + 2, 3, seed, act,
                                         vecs, lrow=linear_row(c, 0),
                                         points=values)
    assert n == 512
    ref_by = {}
    for i, (a, _tag) in enumerate(ref_order):
        ref_by.setdefault(tuple(a), []).append(ref[i])
    for k, vec in enumerate(order):
        assert np.array_equal(rows[k * 4:(k + 1) * 4], np.asarray(ref_by[vec])), vec


def test_full_block_reordered_axes_give_the_same_single_axis_rows():
    """T3 on a full block enumerates the block with the weight words first
    (positions 3, 2, 0, 1 for DuX; 3, 2, 1, 0 for YuX).  Same point set, so
    the single-axis rows must be byte-identical to the standard order's."""
    instance, layers = "yuxtoy-2^4", 3
    c = get_cipher(instance, rounds=layers + 2)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="yux")
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    lrow = linear_row(c, 0)
    ws = [(a,) for a in range(4)]
    std = [3, 0, 1, 2]                    # `--full-block 0` as the driver lists it
    reo = [3, 2, 1, 0]                    # `plan`'s weight words for dims = 4
    r1, o1, n1 = WT.weighted_rows(pre, c, rks, layers + 2, 3, 7026, std, ws, lrow=lrow)
    r2, o2, n2 = WT.weighted_rows(pre, c, rks, layers + 2, 3, 7026, reo, ws, lrow=lrow)
    assert n1 == n2 == F.q ** 4
    assert np.array_equal(r1, r2)
    # and the joint two-axis streamed rows on the reordered block equal the
    # in-memory two-index rows there
    vecs = [(a, b) for a in range(3) for b in range(2)]
    c, pre, rks, rows, order, seed, _y = _stream_multi(
        instance, layers, reo, [np.arange(F.q, dtype=np.int64)] * 4, vecs, 2,
        block_slices=16)
    ref, ref_order, n = WT.weighted_rows(pre, c, rks, layers + 2, 3, seed, reo,
                                         vecs, lrow=lrow)
    ref_by = {}
    for i, (a, _tag) in enumerate(ref_order):
        ref_by.setdefault(tuple(a), []).append(ref[i])
    for k, vec in enumerate(order):
        assert np.array_equal(rows[k * 4:(k + 1) * 4], np.asarray(ref_by[vec])), vec


def test_driver_cli_multi_axis_and_subspace_smoke():
    """The script itself: a subspace (3, 3, 2) structure of toy-2^4 with two
    joint axes and one inner axis, assemble-only; the banner carries the
    sub-slice count and the pass count."""
    out = subprocess.run(
        [sys.executable, os.path.join(E07, "attack_12round.py"),
         "--instance", "toy-2^4", "--layers", "3", "--active", "3,7,11",
         "--subspace-dims", "3,3,2", "--weight-dims", "3", "--vandermonde-axes", "2",
         "--inner-norm", "1", "--combine", "none", "--normalise",
         "--structures", "1", "--procs", "2", "--slice-chunk", "64",
         "--block-slices", "4", "--seed", "2026", "--assemble-only"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "subspace structure, dims [3, 3, 2]" in out.stdout
    assert "64 sub-slices of 4 points" in out.stdout
    assert "2 inner tuple(s)" in out.stdout
    assert "assemble-only" in out.stdout


def test_accumulator_checkpoints_are_pass_aware(tmp_path):
    """A second pass over the same structure (another inner exponent tuple)
    must not resume from the first pass's accumulator: the checkpoint name
    carries the pass label, and `newest_checkpoint` only matches its own."""
    acc = np.arange(6, dtype=np.int64).reshape(2, 3)
    np.save(tmp_path / "acc2n_st0_a10_4096.npy", acc)
    np.save(tmp_path / "acc2n_st0_a10_8192.npy", acc + 1)
    np.save(tmp_path / "acc2n_st0_in0-1_4096.npy", acc + 7)
    np.save(tmp_path / "acc2n_st0_4096.npy", acc + 9)
    got, n = A12.newest_checkpoint(str(tmp_path), "acc2n", "0_a10", 65536, 4096)
    assert n == 8192 and np.array_equal(got, acc + 1)
    got, n = A12.newest_checkpoint(str(tmp_path), "acc2n", "0_in0-1", 65536, 4096)
    assert n == 4096 and np.array_equal(got, acc + 7)
    got, n = A12.newest_checkpoint(str(tmp_path), "acc2n", 0, 65536, 4096)
    assert n == 4096 and np.array_equal(got, acc + 9)     # the single-axis name
    got, n = A12.newest_checkpoint(str(tmp_path), "acc2n", "0_a11", 65536, 4096)
    assert got is None and n == 0


def test_driver_cli_full_block_two_axis_smoke():
    """`--full-block 0 --weight-dims 2 --vandermonde-axes 2`: the block is
    enumerated with the two weight words (positions 3 and 2) as the slowest
    axes, and the cost rule 2 a0 + 3 a1 < margin picks the vectors."""
    out = subprocess.run(
        [sys.executable, os.path.join(E07, "attack_12round.py"),
         "--instance", "yuxtoy-2^4", "--layers", "3", "--full-block", "0",
         "--weight-dims", "2", "--vandermonde-axes", "2", "--max-vectors", "6",
         "--combine", "1110", "--normalise", "--structures", "1", "--procs", "2",
         "--slice-chunk", "64", "--block-slices", "8", "--seed", "2026",
         "--assemble-only"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "active words [3, 2, 0, 1]" in out.stdout
    assert "256 sub-slices of 256 points" in out.stdout
    assert "6 weight vectors over the first 2 active words [3, 2]" in out.stdout
    assert "assemble-only" in out.stdout


def _true_key_vector(c, pre, rks):
    """`tests/test_weighted_rows.py::_true_key_vector` (the equivalent fixed-L0
    keys, O2), repeated here for the YuX full block."""
    from attack_2round_toy import monomial_value
    F = c.F
    eq, _T = c.equivalent_fixedL0_keys(rks)
    return np.array([monomial_value(F, m, eq[0], eq[1], F.q - 1)
                     for m in pre.mons], dtype=np.int64)


def _residual2(F, rows, truth):
    out = np.zeros(rows.shape[0], dtype=np.int64)
    for j in range(rows.shape[1]):
        if truth[j]:
            out ^= F.vmul(rows[:, j], np.int64(truth[j]))
    return out


def test_full_block_two_word_weights_vanish_under_the_cost_rule():
    """T3's full-block rule (E14 counted it, nobody had summed it): with the
    block enumerated weight-words-first, the `1110` combined rows weighted by
    x_3^{a0} x_2^{a1} vanish at the true key for every vector `plan` admits
    (2 a0 + 3 a1 < margin: positions 3 / 2 have degree 2 / 3 in y after the O11
    substitution) -- the rows S21 item 2 buys its single structure with.
    yuxtoy-2^4, full block 0, layer 3, margin 48, through the streamed joint
    two-axis path.  (The bound is a sufficient condition; on this toy the rows
    beyond it happen to vanish as well -- probed up to cost 70 -- so tightness
    is not asserted here; the single-axis version was measured tight on the
    real Yu2X-8 block in S11.)"""
    instance, layers = "yuxtoy-2^4", 3
    c0 = get_cipher(instance, rounds=layers + 2)
    F = c0.F
    pl = WT.plan(F, [], layers, "dec", "yux", "1110", 2, None, free_blocks=(0,))
    assert pl.margin == 48 and pl.weight_words == (3, 2)
    act = [3, 2, 0, 1]
    ok = [(0, 0), (1, 0), (0, 1), (5, 3), (10, 9), (23, 0), (0, 15), (12, 7)]
    assert all(2 * a + 3 * b < 48 for a, b in ok)
    admitted = {tuple(w) for w in pl.weights}
    assert all(w in admitted for w in ok) and (24, 0) not in admitted
    c, pre, rks, rows, order, seed, ycomb = _stream_multi(
        instance, layers, act, [np.arange(F.q, dtype=np.int64)] * 4, ok, 2,
        combine="1110", block_slices=16)
    assert len(ycomb) == 3 and rows.shape == (3 * len(ok), len(pre.mons))
    truth = _true_key_vector(c, pre, rks)
    assert int(_residual2(F, rows, truth).max()) == 0


def test_subspace_four_axis_weighted_rows_vanish_at_the_true_key():
    """S21 item 1's rows: a (4, 4, 3, 2) subspace structure of toy-2^4 (2^13
    points; T = 15 + 15 + 7 + 3 = 40, layer 3 D = (8, 11, 15, 7), the four
    plain rows read positions 2 / 3 => |a| < 25), four-axis weights streamed
    with two joint axes and two inner ones: every admitted vector's rows
    vanish at the true key."""
    instance, layers = "toy-2^4", 3
    c0 = get_cipher(instance, rounds=layers + 2)
    F = c0.F
    act = [3, 7, 11, 15]
    dims = [4, 4, 3, 2]
    pl = WT.plan(F, act, layers, "dec", "dux", None, 4, None, subspace_dims=dims)
    assert pl.margin == 25 and pl.usable_norm == 25
    values = A12.subspace_values(F, dims)
    vecs = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
            (3, 2, 1, 1), (7, 6, 3, 2), (0, 0, 2, 1), (14, 0, 0, 3), (13, 1, 1, 3)]
    # W25's axis cap keeps the cheapest 16 * 16 * 8 * 4 = 8192 of the C(28, 4)
    # vectors below the margin: every |a| <= 18 and part of |a| = 19
    assert pl.usable_weights == 8192 and max(sum(w) for w in pl.weights) == 19
    admitted = {tuple(w) for w in pl.weights}
    assert all(w in admitted for w in vecs)
    c, pre, rks, rows, order, seed, _y = _stream_multi(instance, layers, act,
                                                       values, vecs, 2,
                                                       block_slices=16)
    assert rows.shape == (4 * len(vecs), len(pre.mons))
    truth = _true_key_vector(c, pre, rks)
    assert int(_residual2(F, rows, truth).max()) == 0
