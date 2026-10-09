"""E14 / W28 (T3) -- multi-index weights.

Pins:
  * the five weight counts of the R9 memo's T3 table, together with the
    single-axis counts the published ledger rows use (which must not move);
  * the generalised full-block cost rule (word i costs deg_i per power);
  * the STREAMED two-axis weights of `attack_12round.py` against the in-memory
    `weighted.weighted_rows`, in both characteristics.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for _p in (ROOT, os.path.join(ROOT, "tools"), E07,
           os.path.join(E07, "fast")):
    sys.path.insert(0, _p)

import weighted as WT                                            # noqa: E402
from assemble_fast import (MomentLayout, Precomp, _pow_matrix,   # noqa: E402
                           pow_field_vec, rows_from_moments,
                           slice_moment_table, weight_vector_count)
from assemble_fast_2n import rows_from_moments_2n                # noqa: E402
from attack_2round_toy import linear_row, structure_data         # noqa: E402
from dux.field import make_field                                 # noqa: E402
from dux.registry import get_cipher                              # noqa: E402
import attack_12round as A12                                     # noqa: E402

# technique T3 (docs/glossary.md): (label, field, active, layers,
# cipher, combine, free_blocks, subspace_dims, dims, single-axis count (the
# PUBLISHED one), multi-index count (the memo's))
ROWS = [
    ("DuX(2^8) 8/12, four full words", "2^8", [3, 7, 11, 15], 6, "dux", None,
     (), None, 4, 240, math.comb(243, 4)),
    ("DuX(2^8) 8/12, dims (8,8,8,5)", "2^8", [3, 7, 11, 15], 6, "dux", None,
     (), [8, 8, 8, 5], 4, 16, math.comb(19, 4)),
    ("Yu2X-8 8/12, full block", "2^8", [], 6, "yux", "1110", (0,), None,
     4, 126, 798873),
    ("Yu2X-8 7/12, two words", "2^8", [0, 4], 5, "yux", None, (), None,
     2, 34, math.comb(35, 2)),
    ("DuX(65537) 13, full block", "65537", [], 11, "dux", "0001", (0,), None,
     4, 28, 3158),
]


@pytest.mark.parametrize("label,q,active,layers,cipher,combine,fb,sd,dims,one,many",
                         [(r[0],) + r[1:] for r in ROWS])
def test_weight_counts(label, q, active, layers, cipher, combine, fb, sd,
                       dims, one, many):
    F = make_field(q)
    p1 = WT.plan(F, active, layers, "dec", cipher, combine, dims=1, nweights=1,
                 free_blocks=fb, subspace_dims=sd)
    assert p1.usable_weights == one, (label, "single axis")
    pn = WT.plan(F, active, layers, "dec", cipher, combine, dims=dims,
                 nweights=1, free_blocks=fb, subspace_dims=sd)
    assert pn.usable_weights == many, (label, "multi index")


def test_full_block_cost_rule():
    """DuX puts 2/3/5/8 on positions 2/1/0/3 and YuX on 3/2/1/0."""
    assert WT.block_weight_costs("dux", "dec") == [(2, 2), (1, 3), (0, 5), (3, 8)]
    assert WT.block_weight_costs("yux", "dec") == [(3, 2), (2, 3), (1, 5), (0, 8)]
    # the counting function degenerates to the plain one when every cost is 1
    for dims in (1, 2, 3):
        for norm in (5, 17, 40):
            assert (WT.cost_weight_count([1] * dims, norm)
                    == weight_vector_count(dims, norm))
    # and the enumerated vectors really satisfy the cost bound
    costs = [2, 3, 5, 8]
    ws = WT.cost_weight_vectors(costs, 20)
    assert len(ws) == WT.cost_weight_count(costs, 20)
    assert all(sum(a * c for a, c in zip(w, costs)) < 20 for w in ws)


def test_single_axis_full_block_rule_is_unchanged():
    """dims = 1 still uses the paper's `margin // 2`, so the published 126 and
    28 do not move (see `tests/test_usable_weights.py`)."""
    F = make_field("2^8")
    pl = WT.plan(F, [], 6, "dec", "yux", "1110", dims=1, nweights=1,
                 free_blocks=(0,))
    assert pl.usable_norm == pl.margin // 2 == 126


@pytest.mark.parametrize("instance,layers,active", [
    ("toy-193", 4, "3,7"),
    ("toy-2^4", 4, "3,7"),
])
def test_streamed_two_axis_weights_equal_the_in_memory_rows(instance, layers,
                                                            active):
    """M_{a0,a1} = sum_x x^{a0} [Pi^T diag(y^{a1}) Pi'] -- the driver folds
    y^{a1} into one operand of the panel product (the O15 inner-mask
    mechanism) and keeps the a0 axis as the Vandermonde."""
    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    act = [int(v) for v in active.split(",")]
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    q, npts = F.q, F.q ** len(act)
    L = npts // q
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    P = structure_data(c, rks, rounds, 3, seed, active)
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    bounds = [(i * L, (i + 1) * L) for i in range(q)]
    yv = np.arange(q, dtype=np.int64)
    lrow = linear_row(c, 0)
    asm = rows_from_moments_2n if F.char == 2 else rows_from_moments
    got = {}
    for a1 in (0, 1, 2):
        w = np.tile(np.repeat(pow_field_vec(F, yv, a1), L // len(yv)), q)
        G = (A12.char2_slice_moments(pre, layout, PW, bounds, w=w) if F.char == 2
             else slice_moment_table(pre, layout, PW, bounds, w=w))
        for a0 in range(3):
            coef = pow_field_vec(F, np.arange(q, dtype=np.int64), a0)
            if F.char == 2:
                g = np.zeros(layout.width, dtype=np.int64)
                for k in range(q):
                    if coef[k]:
                        g ^= F.vmul(G[k], np.int64(coef[k]))
            else:
                g = ((coef.astype(np.float64) @ G.astype(np.float64))
                     % F.p).astype(np.int64)
            got[(a0, a1)] = np.asarray(
                asm(pre, layout.moments(g, npts), lrow), dtype=np.int64)
    ws = sorted(got)
    ref, order, n = WT.weighted_rows(pre, c, rks, rounds, 3, seed, active, ws,
                                     lrow=lrow)
    assert n == npts
    by = {}
    for i, (a, _j) in enumerate(order):
        by.setdefault(tuple(a), []).append(ref[i])
    for a in ws:
        assert np.array_equal(got[a], np.asarray(by[a])), a


def test_two_axis_weighted_moments_equal_the_definition():
    """M_{a0,a1} = sum_x x^{a0} y^{a1} (moment), summed point by point."""
    c = get_cipher("toy-193", rounds=6)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    P = structure_data(c, rks, 6, 3, seed, "3,7")
    N = len(P[0])
    idx = np.arange(N, dtype=np.int64)
    xv = (idx // F.q) % F.q
    yv = idx % F.q
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    ws = [(0, 0), (1, 0), (0, 1), (2, 3)]
    from assemble_fast import weighted_moment_stream
    for a, mom in weighted_moment_stream(pre, PW, [xv, yv], ws, layout):
        w = (pow_field_vec(F, xv, a[0]) * pow_field_vec(F, yv, a[1])) % F.p
        for b in pre.unknown:
            want = (PW[b] * w[None, :]).sum(axis=1) % F.p
            assert np.array_equal(mom.pm[b], want), (a, b)
        for (A, B) in layout.pairs:
            want = np.asarray(((PW[A] * w[None, :]) % F.p).astype(np.float64)
                              @ PW[B].astype(np.float64).T, dtype=np.int64) % F.p
            assert np.array_equal(mom.Mom[(A, B)] % F.p, want), (a, A, B)


def test_t4_ten_round_single_structure_smoke():
    """T4 in miniature: one 2^16 structure and a handful of weights recovers
    the ten-round DuX(65537) master key.  The 50-key runs are in
    results/E14_multi_weights/t4_10round.md."""
    import subprocess
    E06 = os.path.join(ROOT, "experiments", "E06_key_recovery_1round")
    out = subprocess.run(
        [sys.executable, os.path.join(E06, "attack_1round.py"),
         "--instance", "dux-65537", "--rounds", "10", "--pos", "3",
         "--structures", "1", "--auto-weight", "--nweights", "8",
         "--keys", "3", "--seed", "2026"],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "success 3/3 = 100.0%" in out.stdout
    assert "structures used: min 1, max 1" in out.stdout


def test_char2_worker_honours_the_inner_mask():
    """The plumbing, not the algebra: `_block_moments_2n` must hand
    `_W["inner_mask"]` to `char2_slice_moments`, tiled to the chunk it is
    decrypting.  Before R9 the characteristic-2 worker ignored the mask (only
    the F_p twin used it), which is what `--weight-dims 2` now needs; the
    algebra itself is locked by the test above.  The mask used here is
    y^1 over the whole field, so it contains a zero -- the sentinel S16 had a
    bug with."""
    layers = 3
    c = get_cipher("toy-2^4", rounds=layers + 2)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    q = F.q
    mask = np.asarray(pow_field_vec(F, np.arange(q, dtype=np.int64), 1),
                      dtype=np.int64)
    assert (mask == 0).any() and (mask != 0).any()
    real = A12.char2_slice_moments
    seen, out = [], {}

    def spy(pre_, layout_, PW, bounds, w=None):
        seen.append(None if w is None else np.asarray(w).copy())
        return real(pre_, layout_, PW, bounds, w=w)

    for tag, im in (("masked", mask), ("plain", None)):
        seen.clear()
        G = np.zeros((1, layout.width), dtype=np.int64)
        A12._W.update(c=c, rks=rks, rounds=layers + 2, pos=3, active=[3],
                      seed=2026 + 5000, slice_len=q, pre=pre, layout=layout,
                      point_chunk=64, G=G, values=None, inner_mask=im)
        A12.char2_slice_moments = spy
        try:
            A12._block_moments_2n((0, 1, 0, 0, 1))
        finally:
            A12.char2_slice_moments = real
        assert len(seen) == 1
        if im is None:
            assert seen[0] is None
        else:
            assert np.array_equal(seen[0], mask)
        out[tag] = G.copy()
        assert G.any(), tag
    assert not np.array_equal(out["masked"], out["plain"])
