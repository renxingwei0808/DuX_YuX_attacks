"""S10: the slice-streaming 12-round driver must produce the SAME equation rows
as the in-memory weighted assembly of W16.

`attack_12round.py` exists because `weighted.structure_powers` materialises the
whole structure's power matrix (75 x 2^32 x 8 B = 2.6 TB at the 12-round size).
It walks the structure slice by slice instead and accumulates
    acc[a] = sum_v v^a G_v          (O12 Sect. 5.4)
with one dgemm per block of slices.  This test pins the refactoring: for a toy
structure small enough for both paths, the rows must agree exactly -- for a
one-word structure (a slice is a single point) and for a two-word structure (a
slice is a whole q-point line), with and without the O10 combined row.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
sys.path.insert(0, ROOT)
sys.path.insert(0, E07)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux.registry import get_cipher                              # noqa: E402
from assemble_fast import (MomentLayout, Precomp, _pow_matrix,   # noqa: E402
                           slice_moment_table)
from attack_2round_toy import active_words, linear_row, structure_stream  # noqa: E402
import weighted as WT                                            # noqa: E402


def _stream_acc(instance, layers, active, weights, procs_free=True):
    """The driver's phase 1+2, in-process (no Pool), for one structure."""
    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    act = active_words(3, active)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    q, npts = F.q, F.q ** len(act)
    L = npts // q
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    seed = 2026 + 5000
    acc = np.zeros((len(weights), layout.width), dtype=np.int64)
    blk = 8
    for v0 in range(0, q, blk):
        v1 = min(v0 + blk, q)
        P = None
        for chunk in structure_stream(c, rks, rounds, 3, seed, active,
                                      chunk=(v1 - v0) * L, start=v0 * L,
                                      stop=v1 * L):
            assert P is None
            P = chunk
        PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
        G = slice_moment_table(pre, layout, PW,
                               [(i * L, (i + 1) * L) for i in range(v1 - v0)])
        V = np.empty((v1 - v0, len(weights)), dtype=np.int64)
        from assemble_fast import pow_field_vec
        vals = np.arange(v0, v1, dtype=np.int64)
        for j, a in enumerate(weights):
            V[:, j] = pow_field_vec(F, vals, int(a))
        acc = (acc + (V.T.astype(np.float64) @ G.astype(np.float64)) % F.p) % F.p
    return c, pre, layout, rks, acc, npts


@pytest.mark.parametrize("instance,layers,active,combine,nw", [
    ("toy-257", 5, "3", None, 6),          # one word: a slice is one point
    ("toy-257", 5, "3", "0001", 6),
    ("toy-193", 4, "3,7", "0001", 5),      # two words: a slice is a q-point line
    ("toy-193", 4, "3,7", None, 4),
])
def test_streamed_rows_equal_the_in_memory_weighted_rows(instance, layers,
                                                         active, combine, nw):
    from assemble_fast import combine_rows, rows_from_moments
    c, pre, layout, rks, acc, npts = _stream_acc(instance, layers, active,
                                                 list(range(nw)))
    F = c.F
    lrow = linear_row(c, 0)
    ycomb = WT.combine_vectors("dux", F, "dec", combine) if combine else None
    streamed = []
    for i in range(nw):
        rows = rows_from_moments(pre, layout.moments(acc[i], npts), lrow)
        if ycomb is None:
            streamed.extend(rows)
        else:
            streamed.extend(combine_rows(rows, y, F) for y in ycomb)
    ws = [(a,) for a in range(nw)]
    ref, _order, n = WT.weighted_rows(pre, c, rks, layers + 2, 3, 2026 + 5000,
                                      active, ws, lrow=lrow, ycomb=ycomb)
    assert n == npts
    assert np.array_equal(np.asarray(streamed, dtype=np.int64), ref)


def test_driver_cli_runs_and_reports_the_expected_banner():
    """End-to-end smoke of the script itself (the banner carries the numbers
    the 12-round run records quote: M, margin, dim K and the y vector)."""
    out = subprocess.run(
        [sys.executable, os.path.join(E07, "attack_12round.py"),
         "--instance", "toy-257", "--layers", "6", "--active", "3,7",
         "--weights", "4", "--combine", "0001", "--normalise",
         "--structures", "1", "--procs", "2", "--slice-chunk", "257",
         "--block-slices", "16", "--seed", "2026"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "monomials M = 38051" in out.stdout
    assert "margin 150" in out.stdout
    assert "dim K = 1" in out.stdout
    assert "257 slices of 257 points" in out.stdout


@pytest.mark.parametrize("instance,layers,active,combine,nw", [
    ("toy-2^4", 4, "3", None, 5),
    ("toy-2^4", 4, "3", "0001", 5),
])
def test_char2_weighted_moments_equal_the_in_memory_path(instance, layers,
                                                         active, combine, nw):
    """The characteristic-2 branch takes the other route of O12: a weight is a
    per-point scalar, so the weighted moment is the plain `mom2n.c` moment with
    one operand scaled by x^a.  It must land on the same rows as W16's
    slice-based `weighted_rows`."""
    sys.path.insert(0, os.path.join(E07, "fast"))
    from assemble_fast import combine_rows
    from assemble_fast_2n import rows_from_moments_2n
    import attack_12round as A12

    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    assert F.char == 2
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    seed = 2026 + 5000
    weights = list(range(nw))
    acc, npts = A12.char2_weighted_acc(pre, c, rks, rounds, 3, active, seed,
                                       weights, layout)
    lrow = linear_row(c, 0)
    ycomb = WT.combine_vectors("dux", F, "dec", combine) if combine else None
    got = []
    for i in range(nw):
        rows = rows_from_moments_2n(pre, layout.moments(acc[i], npts), lrow)
        if ycomb is None:
            got.extend(rows)
        else:
            got.extend(combine_rows(rows, y, F) for y in ycomb)
    ref, _order, n = WT.weighted_rows(pre, c, rks, rounds, 3, seed, active,
                                      [(a,) for a in weights], lrow=lrow,
                                      ycomb=ycomb)
    assert n == npts
    assert np.array_equal(np.asarray(got, dtype=np.int64), ref)


@pytest.mark.parametrize("instance,layers,active", [
    ("toy-257", 5, "3,7"),
    ("toy-193", 4, "3,7"),
    ("toy-257", 5, "3"),
])
def test_fast_slice_moments_equals_the_reference_slice_table(instance, layers, active):
    """`attack_12round.fast_slice_moments` stacks the four blocks into one dgemm
    instead of ten; it must return the reference table bit for bit.

    The reference converts BOTH operands to float64 inside each of the ten
    block-pair products -- twenty casts of an (nP x L) panel per slice, which at
    the 12-round size is 786 MB of cast traffic per slice and dominated the S10
    moment phase.  This test is what allows that to be replaced."""
    from assemble_fast import slice_bounds
    import attack_12round as A12

    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    act = active_words(3, active)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    layout = MomentLayout(pre)
    rng = np.random.default_rng(5)
    rks = c.key_schedule(c.random_key(rng))
    P = None
    for chunk in structure_stream(c, rks, rounds, 3, 99, active,
                                  chunk=F.q ** len(act)):
        P = chunk
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    bounds = slice_bounds(F.q, len(act))[:6]     # a handful of slices is enough
    ref = slice_moment_table(pre, layout, PW, bounds)
    got = A12.fast_slice_moments(pre, layout, PW, bounds)
    assert np.array_equal(got, ref)


# ------------------------------------------------- S13-A: char-2 streaming --
def _char2_stream_acc(instance, layers, active, free_block, weights,
                      point_chunk, nparts=1):
    """`attack_12round`'s characteristic-2 phase 1 + 2, in-process (no Pool).

    The driver's worker functions read their state out of the module-level
    `_W`, so the test fills it directly: that way the streaming path under test
    is the very code the 11-round Yu2X-16 and 8-round Yu2X-8 runs execute, not
    a re-implementation of it."""
    sys.path.insert(0, os.path.join(E07, "fast"))
    from assemble_fast_2n import _logs
    from dux.registry import cipher_family
    import attack_12round as A12

    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    assert F.char == 2
    if free_block is not None:
        act = [4 * free_block + 3] + [4 * free_block + i for i in range(3)]
    else:
        act = active_words(3, active)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3],
                  cipher=cipher_family(instance))
    layout = MomentLayout(pre)
    q, npts = F.q, F.q ** len(act)
    L = npts // q
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    G = np.zeros((q * nparts, layout.width), dtype=np.int64)
    A12._W.update(c=c, rks=rks, rounds=rounds, pos=3, active=act, seed=seed,
                  slice_len=L, pre=pre, layout=layout,
                  point_chunk=point_chunk, G=G,
                  values=None)          # `_W` is module-global: a coset test
                                        # earlier in the file leaves its own
                                        # value sets behind otherwise
    for v in range(q):
        for pt in range(nparts):
            A12._block_moments_2n((v, v + 1, v * nparts, pt, nparts))
    if nparts > 1:
        G = np.bitwise_xor.reduce(G.reshape(q, nparts, layout.width), axis=1)
    ACC = np.zeros((len(weights), layout.width), dtype=np.int64)
    A12._W.update(LG=np.ascontiguousarray(_logs(F, G).T),
                  LV=A12.vandermonde_logs(F, np.arange(q, dtype=np.int64),
                                          weights),
                  ACC=ACC)
    A12._weight_acc_2n((0, len(weights)))
    return c, pre, layout, rks, act, ACC, npts, seed


@pytest.mark.parametrize("instance,layers,active,free_block,nw,pc", [
    ("yuxtoy-2^4", 3, "0,4", None, 5, 64),    # 2 words: 16 slices of 16 points
    ("yuxtoy-2^4", 3, None, 0, 4, 1024),      # O11 full block: 16 x 4 096
])
def test_char2_streamed_moments_equal_the_in_memory_path(instance, layers,
                                                         active, free_block,
                                                         nw, pc):
    """S13-A: the slice-streamed characteristic-2 accumulator must equal the
    in-memory one weight for weight.

    The in-memory route (`char2_weighted_acc`) materialises the whole
    structure, which is 550 GB at the 2^32 sizes of Y06-2/Y06-3 -- that is why
    those two cells were left unrun.  The streamed route slices the structure
    by the value of the weight word and XOR-accumulates v^a G_v instead, so it
    is this equality that carries the two runs."""
    import attack_12round as A12

    weights = list(range(nw))
    c, pre, layout, rks, act, ACC, npts, seed = _char2_stream_acc(
        instance, layers, active, free_block, weights, pc)
    ref, npts_ref = A12.char2_weighted_acc(pre, c, rks, layers + 2, 3, act,
                                           seed, weights, layout)
    assert npts_ref == npts
    assert np.array_equal(ACC, ref)


@pytest.mark.parametrize("instance,layers,active,free_block,combine,nw,pc", [
    ("yuxtoy-2^4", 3, "0,4", None, None, 4, 16),
    ("yuxtoy-2^4", 3, None, 0, "1110", 3, 4096),
])
def test_char2_streamed_rows_equal_weighted_rows(instance, layers, active,
                                                 free_block, combine, nw, pc):
    """...and the rows those accumulators produce equal W16's `weighted_rows`,
    with and without the O10 combined rows (`1110`, dim K = 3)."""
    from assemble_fast import combine_rows
    from assemble_fast_2n import rows_from_moments_2n

    weights = list(range(nw))
    c, pre, layout, rks, act, ACC, npts, seed = _char2_stream_acc(
        instance, layers, active, free_block, weights, pc)
    F = c.F
    lrow = linear_row(c, 0)
    ycomb = WT.combine_vectors("yux", F, "dec", combine) if combine else None
    if ycomb is not None:
        assert len(ycomb) == 3
    got = []
    for i in range(nw):
        rows = rows_from_moments_2n(pre, layout.moments(ACC[i], npts), lrow)
        if ycomb is None:
            got.extend(rows)
        else:
            got.extend(combine_rows(rows, y, F) for y in ycomb)
    ref, _order, n = WT.weighted_rows(pre, c, rks, layers + 2, 3, seed, act,
                                      [(a,) for a in weights], lrow=lrow,
                                      ycomb=ycomb)
    assert n == npts
    assert np.array_equal(np.asarray(got, dtype=np.int64), ref)


@pytest.mark.parametrize("instance,layers,active", [
    ("yuxtoy-2^4", 3, "0,4"),
    ("toy-2^4", 3, "3,7"),
])
def test_char2_slice_moments_equals_the_reference_slice_table(instance, layers,
                                                              active):
    """`char2_slice_moments` routes the per-slice plain moments through
    `fast/mom2n.c` instead of the reference's per-row `F.vmul` loop; it must
    return `slice_moment_table`'s table bit for bit."""
    from assemble_fast import slice_bounds
    from dux.registry import cipher_family
    import attack_12round as A12

    rounds = layers + 2
    c = get_cipher(instance, rounds=rounds)
    F = c.F
    act = active_words(3, active)
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3],
                  cipher=cipher_family(instance))
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(5)))
    P = None
    for chunk in structure_stream(c, rks, rounds, 3, 99, act,
                                  chunk=F.q ** len(act)):
        P = chunk
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    bounds = slice_bounds(F.q, len(act))[:5]
    ref = slice_moment_table(pre, layout, PW, bounds)
    got = A12.char2_slice_moments(pre, layout, PW, bounds)
    assert np.array_equal(got, ref)


def test_vandermonde_logs_matches_the_explicit_powers():
    """The log-domain Vandermonde is an outer product mod the multiplicative
    order; the zero column (0^0 = 1, 0^a = 0) is the only special case."""
    from assemble_fast import pow_field_vec
    from assemble_fast_2n import _expx
    import attack_12round as A12

    c = get_cipher("yuxtoy-2^4", rounds=5)
    F = c.F
    vals = np.arange(F.q, dtype=np.int64)
    weights = list(range(7))
    lV = A12.vandermonde_logs(F, vals, weights)
    expx = _expx(F)
    for i, a in enumerate(weights):
        assert np.array_equal(expx[lV[i]].astype(np.int64),
                              pow_field_vec(F, vals, a))


# ------------------------------------------------ S13-D: coset structures --
def test_subgroup_coset_matches_the_Y03_zero_sum_runner():
    """`attack_12round.subgroup_coset` repeats Y03's construction (three
    experiment directories carry a `run.py`, so it cannot be imported); the two
    must give the same coset for the same representative."""
    import importlib.util
    import attack_12round as A12

    # Y03's run.py does `from run import ...` for E04's helpers, so a stale
    # `run` module from another experiment directory has to be stepped around.
    saved = sys.modules.pop("run", None)
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E04_zero_sum_F2n"))
    try:
        spec = importlib.util.spec_from_file_location(
            "y03run", os.path.join(ROOT, "experiments", "Y03_zero_sum",
                                   "run.py"))
        y03 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(y03)
    finally:
        sys.modules.pop("run", None)
        if saved is not None:
            sys.modules["run"] = saved
    for p, k in ((257, 4), (193, 6), (65537, 15)):
        got = A12.subgroup_coset(p, k, None, rep=7)
        ref = y03.subgroup_coset(p, k, None, rep=7)
        assert np.array_equal(got, ref)
        # a coset of the order-2^k subgroup is a zero-sum for every a that
        # 2^k does not divide (O12 Sect. 5.5) -- and only for those
        for a in (0, 1, 2, 3, (1 << k) - 1, 1 << k):
            tot = int(sum(pow(int(v), a, p) for v in got) % p)
            assert (tot == 0) == (a % (1 << k) != 0), (p, k, a, tot)


def test_structure_values_default_reproduces_the_product_set():
    """The `values` argument is additive: passing the full field back must give
    the byte-identical structure every earlier result was measured on."""
    c = get_cipher("toy-257", rounds=6)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(3)))
    from attack_2round_toy import structure_data
    ref = structure_data(c, rks, 6, 3, 77, "3,7")
    got = structure_data(c, rks, 6, 3, 77, "3,7",
                         values=[np.arange(F.q, dtype=np.int64)] * 2)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    chunks = list(structure_stream(c, rks, 6, 3, 77, "3,7", chunk=1000,
                                   values=[np.arange(F.q, dtype=np.int64)] * 2))
    cat = [np.concatenate([ch[i] for ch in chunks]) for i in range(16)]
    assert all(np.array_equal(a, b) for a, b in zip(ref, cat))


def test_coset_streamed_moments_equal_the_in_memory_weighted_stream():
    """S13-D: with one coset of the order-2^k subgroup on the active word, the
    driver's F_p slice phase must give the same weighted moments as W16's
    in-memory `weighted_moment_stream` on the same point set."""
    from attack_2round_toy import structure_data
    import attack_12round as A12

    k, layers, nw = 4, 4, 5
    c = get_cipher("yuxtoy-257", rounds=layers + 2)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="yux")
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    svals = A12.subgroup_coset(F.p, k, np.random.default_rng(11))
    nslice = 1 << k
    weights = [a for a in range(1, 3 * nw) if a % nslice != 0][:nw]

    G = np.zeros((nslice, layout.width), dtype=np.int64)
    A12._W.update(c=c, rks=rks, rounds=layers + 2, pos=0, active=[0],
                  seed=seed, slice_len=1, pre=pre, layout=layout,
                  values=[svals], G=G)
    A12._block_moments((0, nslice, 0))
    V = A12.vandermonde(F, svals, weights)
    acc = (V.T.astype(np.float64) @ G.astype(np.float64)) % F.p

    # the reference is the definition, point by point: M_a = sum_x x^a g(x).
    # (`weighted_moment_stream` cannot serve here -- it asserts a full product
    # set, and a coset is exactly the structure that is not one.)
    from assemble_fast import pow_field_vec
    P = structure_data(c, rks, layers + 2, 0, seed, [0], values=[svals])
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    n = layout.nP
    for i, a in enumerate(weights):
        w = pow_field_vec(F, svals, a) % F.p
        ref = np.zeros(layout.width, dtype=np.int64)
        for b in layout.blocks:
            o = layout.off_pm[b]
            ref[o:o + n] = (PW[b] * w[None, :]).sum(axis=1) % F.p
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            ref[o:o + n * n] = (((PW[A] * w[None, :]) @ PW[B].T)
                                % F.p).ravel()
        assert np.array_equal(acc[i].astype(np.int64), ref)


@pytest.mark.parametrize("instance,layers,active,free_block,nw,pc,nparts", [
    ("yuxtoy-2^4", 3, None, 0, 4, 512, 8),     # a full-block slice split 8 ways
    ("yuxtoy-2^4", 3, "0,4", None, 4, 4, 4),   # and a two-word one split 4 ways
])
def test_char2_slice_parts_fold_to_the_same_accumulator(instance, layers,
                                                        active, free_block,
                                                        nw, pc, nparts):
    """`--slice-parts` hands one slice to several workers and XOR-folds their
    partial sums.  Without it the 8-round Yu2X-8 structure has only q = 256
    tasks, so ~20 % of a 104-worker machine idles in the last wave -- and the
    fold has to be exact, because it runs for half a day before anything is
    solved."""
    import attack_12round as A12

    weights = list(range(nw))
    *_, ACC, npts, _seed = _char2_stream_acc(instance, layers, active,
                                             free_block, weights, pc, nparts)
    *_, ref, npts_ref, _s2 = _char2_stream_acc(instance, layers, active,
                                               free_block, weights, pc, 1)
    assert npts == npts_ref
    assert np.array_equal(ACC, ref)


def test_rows_dir_checkpoint_round_trips_through_the_driver():
    """`--rows-dir` must make a re-run bit-identical: the 13-structure
    full-block job is ~14 h, so a restart may not silently change the system."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, os.path.join(E07, "attack_12round.py"),
               "--instance", "yuxtoy-2^4", "--layers", "3", "--full-block", "0",
               "--combine", "1110", "--normalise", "--structures", "2",
               "--procs", "2", "--slice-chunk", "16", "--point-chunk", "1024",
               "--seed", "2026", "--rows-dir", td, "--out", td,
               "--tag", "rd"]
        first = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert first.returncode == 0, first.stderr[-2000:]
        assert "rows assembled" in first.stdout
        files = sorted(os.listdir(td))
        assert "rows_rd_st0.npy" in files and "rows_rd_st1.npy" in files
        import json as _json
        a = _json.load(open(os.path.join(td, "s10_rd.json")))
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert second.returncode == 0, second.stderr[-2000:]
        assert "rows loaded from" in second.stdout
        b = _json.load(open(os.path.join(td, "s10_rd.json")))
        for k in ("rank", "free", "pinned", "inner_correct", "equations",
                  "unknowns", "monomials"):
            assert a[k] == b[k], k

        def _rank_line(out):                       # without the timing
            return [ln.split("(")[0] for ln in out.splitlines() if "rank " in ln]
        assert _rank_line(first.stdout) == _rank_line(second.stdout)


@pytest.mark.parametrize("extra,prefix", [
    (["--full-block", "0", "--combine", "1110"], "acc2n"),
])
def test_resume_checkpoints_reproduces_the_uninterrupted_accumulator(extra,
                                                                     prefix):
    """`--resume-checkpoints` must give bit-identical rows to an uninterrupted
    run.  On the machine these runs live on, jobs are killed every few hours;
    `--rows-dir` already saves FINISHED structures, and this saves the finished
    chunks of the one in flight, so a kill costs a chunk rather than a
    structure."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = [sys.executable, os.path.join(E07, "attack_12round.py"),
                "--instance", "yuxtoy-2^4", "--layers", "3", "--normalise",
                "--structures", "1", "--procs", "2", "--slice-chunk", "4",
                "--point-chunk", "1024", "--seed", "2026",
                "--checkpoint-every", "4", "--checkpoint-dir", td,
                "--out", td] + extra
        full = subprocess.run(base + ["--tag", "full"], capture_output=True,
                              text=True, timeout=900)
        assert full.returncode == 0, full.stderr[-2000:]
        cps = sorted(f for f in os.listdir(td) if f.startswith(prefix))
        assert cps, f"no {prefix} checkpoint was written"
        # keep only the earliest checkpoint, so the resumed run has real work
        for f in cps[1:]:
            os.remove(os.path.join(td, f))
        res = subprocess.run(base + ["--tag", "res", "--resume-checkpoints"],
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        assert "resumed at slice" in res.stdout, res.stdout[-1500:]
        import json as _json
        a = _json.load(open(os.path.join(td, "s10_full.json")))
        b = _json.load(open(os.path.join(td, "s10_res.json")))
        for k in ("rank", "free", "pinned", "inner_correct", "equations",
                  "unknowns", "monomials"):
            assert a[k] == b[k], (k, a[k], b[k])


def test_two_word_point_set_streamed_moments_equal_the_definition():
    """S16 optional A (O15 with two words): both active words run over their
    own random point set; the inner axis' divided-difference mask is folded
    into the per-slice moments in the worker, the outer axis' mask into the
    Vandermonde coefficients in the parent.  The result must be the masked
    weighted moment by its definition, point by point:
        M_a = sum_x w0(x_3) w1(x_7) x_3^a g(x)."""
    from attack_2round_toy import structure_data
    import attack_12round as A12
    import interp_mask as IM

    layers, nw, n = 4, 5, 20
    c = get_cipher("toy-257", rounds=layers + 2)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="dux")
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    rng = np.random.default_rng(11)
    U0 = np.sort(rng.choice(F.q, size=n, replace=False)).astype(np.int64)
    U1 = np.sort(rng.choice(F.q, size=n, replace=False)).astype(np.int64)
    w0 = IM.divided_difference_weights(F, U0)
    w1 = IM.divided_difference_weights(F, U1)
    weights = list(range(nw))

    G = np.zeros((n, layout.width), dtype=np.int64)
    A12._W.update(c=c, rks=rks, rounds=layers + 2, pos=3, active=[3, 7],
                  seed=seed, slice_len=n, pre=pre, layout=layout,
                  values=[U0, U1], G=G, inner_mask=np.asarray(w1, dtype=np.int64))
    A12._block_moments((0, n, 0))
    V = (A12.vandermonde(F, U0, weights) * np.asarray(w0)[:, None]) % F.p
    acc = (V.T.astype(np.float64) @ G.astype(np.float64)) % F.p

    from assemble_fast import pow_field_vec
    P = structure_data(c, rks, layers + 2, 3, seed, [3, 7], values=[U0, U1])
    x3 = np.repeat(U0, n)          # word 3 is the slowest axis
    wpt = (np.repeat(np.asarray(w0), n) * np.tile(np.asarray(w1), n)) % F.p
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    nP = layout.nP
    for i, a in enumerate(weights):
        w = (wpt * pow_field_vec(F, x3, a)) % F.p
        ref = np.zeros(layout.width, dtype=np.int64)
        for b in layout.blocks:
            o = layout.off_pm[b]
            ref[o:o + nP] = (PW[b] * w[None, :]).sum(axis=1) % F.p
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            ref[o:o + nP * nP] = (((PW[A] * w[None, :]) @ PW[B].T)
                                  % F.p).ravel()
        assert np.array_equal(acc[i].astype(np.int64), ref)


def test_char2_point_set_streamed_moments_equal_the_definition():
    """S16: the characteristic-2 streaming path with an O15 point set that
    CONTAINS 0.  The mask enters the Vandermonde logs; the v = 0 column's
    sentinel (0^a = 0 for a > 0) must survive the addition of log w_0, and the
    a = 0 row of that column must become log w_0 (0^0 w_0 = w_0).  Checked
    against the definition M_a = XOR_x w(x) x^a g(x), point by point."""
    from assemble_fast_2n import _logs
    from assemble_fast import pow_field_vec
    from attack_2round_toy import structure_data
    import attack_12round as A12
    import interp_mask as IM

    layers, nw, n = 3, 6, 12
    c = get_cipher("toy-2^4", rounds=layers + 2)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="dux")
    layout = MomentLayout(pre)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    seed = 2026 + 5000
    U = np.sort(np.random.default_rng(5).choice(F.q, size=n, replace=False)).astype(np.int64)
    if U[0] != 0:                       # the whole point of the test
        U[0] = 0
        U = np.sort(U)
    w = np.asarray(IM.divided_difference_weights(F, U), dtype=np.int64)
    weights = list(range(nw))

    G = np.zeros((n, layout.width), dtype=np.int64)
    # `inner_mask=None` explicitly: `_W` is a module-level worker dict that
    # earlier tests in this file leave an F_p mask in, and since R9 the
    # characteristic-2 block path honours it too (T3's second weight axis),
    # exactly as the F_p twin always has.  This cell is single-word, so the
    # mask belongs on the Vandermonde side only.
    A12._W.update(c=c, rks=rks, rounds=layers + 2, pos=3, active=[3],
                  seed=seed, slice_len=1, pre=pre, layout=layout,
                  point_chunk=64, G=G, values=[U], inner_mask=None)
    for i in range(n):
        A12._block_moments_2n((i, i + 1, i, 0, 1))
    ACC = np.zeros((nw, layout.width), dtype=np.int64)
    A12._W.update(LG=np.ascontiguousarray(_logs(F, G).T),
                  LV=A12.vandermonde_logs(F, U, weights, mask=w), ACC=ACC)
    A12._weight_acc_2n((0, nw))

    P = structure_data(c, rks, layers + 2, 3, seed, [3], values=[U])
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    nP = layout.nP
    for i, a in enumerate(weights):
        wa = F.vmul(w, pow_field_vec(F, U, a))
        ref = np.zeros(layout.width, dtype=np.int64)
        for b in layout.blocks:
            o = layout.off_pm[b]
            ref[o:o + nP] = np.bitwise_xor.reduce(F.vmul(PW[b], wa[None, :]), axis=1)
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            AW = F.vmul(PW[A], wa[None, :])
            M = np.zeros((nP, nP), dtype=np.int64)
            for x in range(n):
                M ^= F.vmul(AW[:, x][:, None], PW[B][:, x][None, :])
            ref[o:o + nP * nP] = M.ravel()
        assert np.array_equal(ACC[i], ref), a
