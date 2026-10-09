"""S10 -- the 12-round DuX attack: ONE 2^32 structure per key, walked slice by
slice (O12 Sect. 5.4) with the O10 combined row.

Why a separate driver.  `attack_2round_fast.py --weighted` materialises the
whole structure's power matrix `PW[b]` (nP x N) before it computes anything
(`weighted.structure_powers` -> `assemble_fast._pow_matrix`).  At the 12-round
size that is 75 x 2^32 x 8 B = 2.6 TB, so the in-memory path simply does not
exist here.  Everything the assembly needs from a structure is a point sum, so
this driver walks the structure in SLICES of the weight word and never holds
more than a few slices at a time:

    M_a  =  sum_x x_3^a g(x)  =  sum_{v in F_q} v^a G_v ,
    G_v  =  the PLAIN moments of the slice {x_3 = v}   (`assemble_fast.Moments`)

`structure_stream` emits the points of a two-word structure in exactly that
order (axis 0 is the slowest), so chunk v IS slice v when the chunk size is q.

Phase 1 (`slice_moments`) fills a (chunk x width) table of per-slice moments in
parallel; phase 2 accumulates it into the N_w weighted moment vectors with ONE
dgemm per chunk,  acc += (V^T G) mod p  with V[v][a] = v^a.  Phase 3 turns each
weighted moment vector into the four `rows_from_moments` rows and folds them
into the single cheap combined row of O10 (`--combine 0001`, y = (-1,1,-1,1)).
Phase 4 is the ordinary `modp_solve` elimination.

Exactness: the dgemm accumulates at most `exact_chunk(p)` products of two
residues before a reduction, the same bound `assemble_fast` uses everywhere.

    # the S10 smoke test: same algebra as the full run, toy field
    python attack_12round.py --instance toy-257 --layers 6 --active 3,7 \
        --weights 150 --combine 0001 --normalise --structures 1

    # 12-round DuX(65537), one key
    python attack_12round.py --instance dux-65537 --layers 10 --active 3,7 \
        --weights 9000 --combine 0001 --normalise --procs 48 --seed 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(HERE, "fast"))

from dux.registry import cipher_family, get_cipher            # noqa: E402
from attack_2round_toy import (active_words, linear_row,      # noqa: E402
                               structure_data, structure_stream)
from assemble_fast import (MomentLayout, Precomp, _pow_matrix,  # noqa: E402
                           combine_rows, exact_chunk, pow_field_vec,
                           rows_from_moments, slice_moment_table)
from assemble_fast_2n import rows_from_moments_2n              # noqa: E402
import modp_solve                                             # noqa: E402
import weighted as WT                                         # noqa: E402

_W = {}


def _omp_threads(n):
    """Pin the OpenMP runtime of `fast/mom2n.c` to `n` threads in THIS process.

    The kernel parallelises over its point axis, which is the right thing when
    it is called once on a whole structure (S5) and exactly the wrong thing
    inside a forked worker that already owns one core: 104 workers x 104 teams
    oversubscribe the machine and each team allocates its own private
    accumulator.  libgomp reads OMP_NUM_THREADS at load time, so after the fork
    only `omp_set_num_threads` can still change it."""
    import ctypes
    try:
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(int(n))
        return True
    except OSError:
        return False


def _init(instance, rounds, key_seed, struct_seed, pos, active, slice_len,
          shm_name, shape, extra=None, point_chunk=1 << 16, ks_literal=False,
          values=None, inner_mask=None, cheap_set=(2,)):
    # One BLAS thread per worker: the parent needs all of them for the
    # per-chunk (N_w x k) @ (k x width) accumulation, and the workers' own
    # products have a tiny (nP x nP) output, so threading them only fights the
    # pool.  OpenBLAS reads OPENBLAS_NUM_THREADS at load time and the workers
    # are forked, so the env var cannot separate the two -- threadpoolctl calls
    # openblas_set_num_threads at runtime, which can.
    try:
        from threadpoolctl import threadpool_limits
        _W["_tp"] = threadpool_limits(limits=1)      # keep the handle alive
    except Exception:                                # optional dependency
        pass
    _omp_threads(1)                                  # see _omp_threads
    c = get_cipher(instance, rounds=rounds, ks_literal=ks_literal)
    rng = np.random.default_rng(key_seed)
    K = c.random_key(rng)
    _W.update(c=c, rks=c.key_schedule(K), rounds=rounds, pos=pos,
              active=active, seed=struct_seed, slice_len=slice_len)
    if tuple(cheap_set) == (1, 2):
        # S19 (R9 / T1): the extended column set and the wide slice layout of
        # E13; the worker's per-slice kernel is the one that also fills pm2
        # and Mom2.  Same `pe_arr` / `unknown`, so the decrypt-and-power step
        # below is untouched.
        _W["pre"], _W["layout"], _W["slice_fn"] = precomp_b(c)
    else:
        _W["pre"] = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3],
                            cipher=cipher_family(instance))
        _W["layout"] = MomentLayout(_W["pre"])
        _W["slice_fn"] = char2_slice_moments
    _W["point_chunk"] = int(point_chunk)
    _W["values"] = values
    # O15 with TWO words (S16 optional A): the divided-difference mask of the
    # inner axis is the same vector on every slice, so it is folded into the
    # per-slice moments here; the outer axis' mask goes into the Vandermonde
    # coefficients in the parent, exactly as for one word.
    _W["inner_mask"] = (None if inner_mask is None
                        else np.asarray(inner_mask, dtype=np.int64))
    shm = shared_memory.SharedMemory(name=shm_name)
    _W["shm"] = shm
    _W["G"] = np.ndarray(shape, dtype=np.int64, buffer=shm.buf)
    # characteristic 2 shares three more buffers: the log-transpose of the
    # slice table, the Vandermonde logs and the accumulator itself (the
    # weighted sum runs in the workers, one disjoint weight range each).
    _W["_shm_extra"] = []
    for name, (shm_name2, shape2, dtype2) in (extra or {}).items():
        s2 = shared_memory.SharedMemory(name=shm_name2)
        _W["_shm_extra"].append(s2)
        _W[name] = np.ndarray(shape2, dtype=np.dtype(dtype2), buffer=s2.buf)


class _SerialPool:
    """`--procs 1`: do the block tasks in this very process, no `Pool` at all.

    S18 / docs/measurement_protocol.md rule 1 reports an attack as a run on ONE core whose thread
    count, sampled with `ps -o nlwp`, stayed at 1.  `multiprocessing.Pool(1)`
    cannot meet that: even with a single worker the parent starts three
    bookkeeping threads (`_handle_workers`, `_handle_tasks`, `_handle_results`),
    so the sample is 4.  They are idle threads and cost no CPU, but the
    protocol asks for a number, not for an argument, and the fork also buys
    nothing here -- one worker just means every block is copied through shared
    memory instead of being computed where it is used.

    Same initializer, same task function, same shared-memory blocks: the
    worker state `_W` is simply built in the parent.  `tests/test_procs1.py`
    locks the two paths to the same rows.
    """

    def __init__(self, initializer, initargs):
        initializer(*initargs)

    def imap_unordered(self, fn, tasks, chunksize=1):
        return map(fn, tasks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for s in _W.pop("_shm_extra", []):            # our own attachments;
            s.close()                                 # the creator still owns
        shm = _W.pop("shm", None)                     # unlink()
        if shm is not None:
            shm.close()
        return False


def _make_pool(procs, initializer, initargs):
    """`Pool(procs, ...)`, except that one process means no pool -- see
    `_SerialPool`.  Every other process count is untouched."""
    if int(procs) == 1:
        return _SerialPool(initializer, initargs)
    from multiprocessing import Pool
    return Pool(procs, initializer=initializer, initargs=initargs)


def _block_moments(arg):
    """Plain moments of the slices [v0, v1) -> shared rows [row0, row0+v1-v0).

    One decryption pass over the block's (v1 - v0) * slice_len points, then
    `assemble_fast.slice_moment_table` -- the very routine the in-memory
    weighted path uses -- to split those points back into their slices.  For a
    ONE-word structure a slice is a single point, so this is the "per-point
    accumulation" of O12 done `block` points at a time; for the two-word
    12-round structure a slice is a whole q-point line."""
    v0, v1, row0 = arg
    c, pre, layout = _W["c"], _W["pre"], _W["layout"]
    L = _W["slice_len"]
    t0 = time.time()
    npts = (v1 - v0) * L
    P = None
    for chunk in structure_stream(c, _W["rks"], _W["rounds"], _W["pos"],
                                  _W["seed"], _W["active"], chunk=npts,
                                  start=v0 * L, stop=v1 * L,
                                  values=_W.get("values")):
        assert P is None, "the block must come out of the stream in one chunk"
        P = chunk
    assert len(P[0]) == npts, f"block [{v0},{v1}) had {len(P[0])} points"
    PW = {b: _pow_matrix(c.F, P, b, pre.pe_arr) for b in pre.unknown}
    bounds = [(i * L, (i + 1) * L) for i in range(v1 - v0)]
    _W["G"][row0:row0 + (v1 - v0)] = fast_slice_moments(pre, layout, PW, bounds,
                                                         w=_W.get("inner_mask"))
    return time.time() - t0



# ------------------------------------------------------- characteristic 2 --
def _mom2n_pair(F, la, lb, nPa, nPb):
    """XOR_p a[i][p] * b[j][p] from the two sentinel-log panels.

    `assemble_fast_2n.moment_matrix_2n` takes a PW dict keyed by block, which
    cannot express "the same block, one side scaled" (the diagonal pair with a
    weight on one operand), so the kernel is called directly here."""
    from assemble_fast_2n import _expx, _lib
    expx = _expx(F)
    lib = _lib()
    if lib is not None:
        out32 = np.empty((nPa, nPb), dtype=np.int32)
        lib.mom2n(np.ascontiguousarray(la), np.ascontiguousarray(lb),
                  nPa, nPb, la.shape[1], expx, out32)
        return out32.astype(np.int64)
    out = np.zeros((nPa, nPb), dtype=np.int64)
    for i in range(nPa):
        out[i] = np.bitwise_xor.reduce(expx[la[i][None, :] + lb], axis=1)
    return out


def char2_weighted_acc(pre, c, rks, rounds, pos, active, seed, weights, layout,
                       progress=0):
    """The N_w weighted moment vectors of ONE structure over F_{2^n}.

    S13-A note: this is now the REFERENCE, not the driver's path.  It calls
    `structure_data`, i.e. it materialises the whole structure (16 int64 arrays
    of q^s points), which is 550 GB for the 2^32 structures of Y06-2/Y06-3.
    The driver streams instead (`_block_moments_2n` + `_weight_acc_2n`) and
    `tests/test_stream_weighted_12round.py` locks the two against each other.

    There is no BLAS in characteristic 2, so the F_p slice-plus-dgemm route
    buys nothing here: a weight is a per-point scalar, so the weighted moment
    is simply the PLAIN moment with one operand scaled by x^a, which is what
    `fast/mom2n.c` already computes (one gather per product from the log
    panels).  Cost is N_w passes of nP^2 * N gathers, i.e. the same
    3.6 s/structure kernel of S5 once per weight."""
    from assemble_fast_2n import _logs, moment_vector_2n
    F = c.F
    act = active_words(pos, active)
    s_act = len(act)
    P = structure_data(c, rks, rounds, pos, seed, active)
    N = len(P[0])
    idx = np.arange(N, dtype=np.int64)
    x0 = (idx // (F.q ** (s_act - 1))) % F.q       # the weight word's value
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    lPW = {b: _logs(F, PW[b]) for b in pre.unknown}
    nP = layout.nP
    acc = np.zeros((len(weights), layout.width), dtype=np.int64)
    t0 = time.time()
    for i, a in enumerate(weights):
        xa = pow_field_vec(F, x0, int(a))
        lS = {b: _logs(F, F.vmul(PW[b], xa[None, :])) for b in pre.unknown}
        for b in pre.unknown:
            o = layout.off_pm[b]
            acc[i, o:o + nP] = moment_vector_2n(F, PW[b], xa, lb=lPW[b])
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            acc[i, o:o + nP * nP] = _mom2n_pair(F, lS[A], lPW[B], nP, nP).ravel()
        if progress and (i + 1) % progress == 0:
            print(f"    weighted moments {i + 1}/{len(weights)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return acc, N


def char2_slice_moments(pre, layout, PW, bounds, w=None):
    """`assemble_fast.slice_moment_table`'s characteristic-2 branch, but through
    `fast/mom2n.c`.

    The reference builds every block-pair moment with a Python loop over the nP
    rows of one operand (one broadcast `F.vmul` plus an XOR-reduce per row),
    i.e. nP numpy passes over an (nP x L) panel per pair per slice.  The kernel
    does the same XOR_p expx[la[i][p] + lb[j][p]] in one call, so the logs of
    the four panels are taken once per point block and reused by all ten pairs.
    `tests/test_stream_weighted_12round.py` locks it against the reference.

    `w` (O15's inner mask, and T3's second weight axis): a per-point weight of
    length L.  As in the F_p twin it scales ONE operand -- (X w) X^T -- which
    in the log domain is log w added to that panel's logs; `F.vmul` is used
    rather than the raw addition so that the zero sentinel stays a sentinel
    (S16's bug: (2*order + log w) % order is a valid log, so 0 * w came out
    as w).
    """
    from assemble_fast_2n import _expx, _lib, _logs
    F = pre.F
    assert F.char == 2, "characteristic-2 path"
    n = layout.nP
    lib, expx = _lib(), _expx(F)
    LG = {b: _logs(F, PW[b]) for b in layout.blocks}
    if w is not None:
        w = np.asarray(w, dtype=np.int64)
        PWw = {b: F.vmul(PW[b], w[None, :]) for b in layout.blocks}
        LGw = {b: _logs(F, PWw[b]) for b in layout.blocks}
    else:
        PWw, LGw = PW, LG
    G = np.zeros((len(bounds), layout.width), dtype=np.int64)
    for k, (lo, hi) in enumerate(bounds):
        sub = {b: np.ascontiguousarray(LG[b][:, lo:hi]) for b in layout.blocks}
        subw = (sub if w is None else
                {b: np.ascontiguousarray(LGw[b][:, lo:hi]) for b in layout.blocks})
        for b in layout.blocks:
            o = layout.off_pm[b]
            G[k, o:o + n] = np.bitwise_xor.reduce(PWw[b][:, lo:hi], axis=1)
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            la, lb = subw[A], sub[B]
            if lib is not None:
                out32 = np.empty((n, n), dtype=np.int32)
                lib.mom2n(la, lb, n, n, hi - lo, expx, out32)
                G[k, o:o + n * n] = out32.astype(np.int64).ravel()
            else:
                m = np.empty((n, n), dtype=np.int64)
                for i in range(n):
                    m[i] = np.bitwise_xor.reduce(expx[la[i][None, :] + lb],
                                                 axis=1)
                G[k, o:o + n * n] = m.ravel()
    return G


def vandermonde_logs(F, vals, weights, mask=None):
    """Logs of V[v][a] = v^a over F_{2^n}, with `_logs`' sentinel for the zeros.

    v^a = g^{a log v} for v != 0, so the whole (N_w x k) matrix is one outer
    product reduced mod the multiplicative order -- no repeated squaring, and
    the only special case is the v = 0 column (0^0 = 1, 0^a = 0 for a > 0).

    `mask` (O15): a per-value weight w_v, never 0; the entry becomes
    log(w_v v^a) = log w_v + a log v.  The sentinel entries stay sentinels --
    0^a w_v is still 0.  (S16 found the unguarded version of this add in the
    driver: (2*order + log w) % order is a valid log, so 0^a w came out as w
    whenever the point set contained 0, which a random one does 2/3 of the
    time, and the char-2 assembly then rejected the rows.)"""
    order = F.order
    vals = np.asarray(vals, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    lv = np.zeros(vals.shape, dtype=np.int64)
    nz = vals != 0
    lv[nz] = np.asarray(F._log, dtype=np.int64)[vals[nz]]
    out = (w[:, None] * lv[None, :]) % order
    if mask is not None:
        mask = np.asarray(mask, dtype=np.int64)
        assert mask.shape == vals.shape and (mask != 0).all(), "mask weights are nonzero"
        out = (out + np.asarray(F._log, dtype=np.int64)[mask][None, :]) % order
    if not nz.all():
        zc = np.flatnonzero(~nz)
        out[:, zc] = 2 * order
        z0 = np.flatnonzero(w == 0)
        if z0.size:
            out[np.ix_(z0, zc)] = (0 if mask is None
                                   else np.asarray(F._log, dtype=np.int64)[mask[zc]][None, :])
    return out.astype(np.int32)


def subspace_values(F, dims):
    """S21 (R9 / T3): the value list of each active word of a SUBSPACE
    structure in characteristic 2 -- the F_2-linear subspace of F_{2^n} whose
    top n - m bits vanish, i.e. the integers 0 .. 2^m - 1 (closed under XOR).
    The criterion's threshold for it is T_i = 2^{m_i} - 1 (O7)."""
    assert F.char == 2, "subspace structures are a characteristic-2 construction"
    return [np.arange(1 << int(m), dtype=np.int64) for m in dims]


def axis_strides(sizes):
    """Strides of a product set with the given axis sizes (axis 0 slowest)."""
    strides, acc = [0] * len(sizes), 1
    for j in range(len(sizes) - 1, -1, -1):
        strides[j] = acc
        acc *= int(sizes[j])
    return strides


def slice_axis_values(values, m, lo, hi):
    """v_j(s), j < m, for the sub-slices s in [lo, hi) when the structure is
    sliced by its first m axes jointly (`--vandermonde-axes m`)."""
    sizes = [len(values[j]) for j in range(m)]
    strides = axis_strides(sizes)
    s = np.arange(lo, hi, dtype=np.int64)
    return [np.asarray(values[j], dtype=np.int64)[(s // strides[j]) % sizes[j]]
            for j in range(m)]


def inner_weight_mask(F, values, m, inner):
    """prod_{j >= m} values[j]^{inner[j - m]} laid out over ONE slice, in the
    point order of the inner axes (axis m the slowest of them).  None when
    every inner exponent is 0.  With m = 1 and one inner axis this is exactly
    the `np.repeat(y^{a1}, slice_len // len(y))` mask of --weight-dims 2."""
    sizes = [len(values[j]) for j in range(m, len(values))]
    strides = axis_strides(sizes)
    L = 1
    for n in sizes:
        L *= n
    w = None
    for t, e in enumerate(inner):
        if not e:
            continue
        col = pow_field_vec(F, np.asarray(values[m + t], dtype=np.int64), int(e))
        col = np.tile(np.repeat(col, strides[t]), L // (sizes[t] * strides[t]))
        w = col if w is None else (F.vmul(w, col) if F.char == 2
                                   else (w * col) % F.p)
    return w


def vandermonde_logs_multi(F, axes, wvecs):
    """`vandermonde_logs` for a MULTI-INDEX weight over m jointly sliced axes:
    log of prod_j v_j(s)^{a_j} for every weight vector a and sub-slice s, the
    sentinel 2*order where some v_j(s) = 0 with a_j > 0 (0^0 = 1 otherwise).
    Same convention as `vandermonde_logs` (no mask), to which it reduces at
    m = 1."""
    order = F.order
    A = np.asarray(wvecs, dtype=np.int64).reshape(len(wvecs), -1)
    k = len(axes[0])
    out = np.zeros((A.shape[0], k), dtype=np.int64)
    sentinel = np.zeros((A.shape[0], k), dtype=bool)
    logt = np.asarray(F._log, dtype=np.int64)
    for j, v in enumerate(axes):
        v = np.asarray(v, dtype=np.int64)
        nz = v != 0
        lv = np.zeros(k, dtype=np.int64)
        lv[nz] = logt[v[nz]]
        out = (out + (A[:, j:j + 1] % order) * lv[None, :]) % order
        if not nz.all():
            sentinel |= (A[:, j:j + 1] > 0) & (~nz)[None, :]
    out[sentinel] = 2 * order
    return out.astype(np.int32)


def precomp_b(c):
    """S19 (R9 / T1): (PrecompB, LayoutBStream, char2_slice_moments_b) for the
    cheap set {1, 2} -- the extended linearisation of
    `experiments/E13_b_coordinate/bcoord.py`, streamed.  Characteristic 2,
    DuX, decryption direction only (`bcoord.PrecompB` asserts the first two)."""
    sys.path.insert(0, os.path.join(HERE, "..", "E13_b_coordinate"))
    import bcoord as BC
    pre = BC.PrecompB(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    return pre, BC.LayoutBStream(pre), BC.char2_slice_moments_b


def _decrypt_range(start, stop):
    """The structure's points [start, stop) as one chunk (worker context)."""
    c = _W["c"]
    P = None
    for chunk in structure_stream(c, _W["rks"], _W["rounds"], _W["pos"],
                                  _W["seed"], _W["active"], chunk=stop - start,
                                  start=start, stop=stop,
                                  values=_W.get("values")):
        assert P is None, "the block must come out of the stream in one chunk"
        P = chunk
    assert len(P[0]) == stop - start
    return P


def _block_moments_2n(arg):
    """Characteristic-2 twin of `_block_moments`.

    Two regimes, because a slice is q^(s-1) points: for the two-word 11-round
    structure that is 65 536 points (a slice per decryption pass), for the
    full-block 8-round structure 2^24 (16 int64 arrays of 2^24 = 2.1 GB, times
    the power panels -- far too much for 100 workers), so a big slice is walked
    in `--point-chunk` sub-chunks and its moments XOR-accumulated."""
    v0, v1, row0, part, nparts = arg
    c, pre, layout = _W["c"], _W["pre"], _W["layout"]
    L, pc = _W["slice_len"], _W["point_chunk"]
    G = _W["G"]
    t0 = time.time()
    if nparts == 1 and L <= pc:
        step = max(1, pc // L)
        for w0 in range(v0, v1, step):
            w1 = min(w0 + step, v1)
            P = _decrypt_range(w0 * L, w1 * L)
            PW = {b: _pow_matrix(c.F, P, b, pre.pe_arr) for b in pre.unknown}
            im = _W.get("inner_mask")
            G[row0 + w0 - v0:row0 + w1 - v0] = _W.get("slice_fn", char2_slice_moments)(
                pre, layout, PW,
                [(i * L, (i + 1) * L) for i in range(w1 - w0)],
                w=(None if im is None else np.tile(im, w1 - w0)))
    else:
        # `nparts` > 1 splits one slice across that many tasks (the sums are
        # XOR-folded by the parent).  Without it the full-block structure has
        # only q = 256 tasks per structure, so 104 workers run three uneven
        # waves and lose ~20 % of the machine.
        sub = L // nparts
        for v in range(v0, v1):
            g = np.zeros(layout.width, dtype=np.int64)
            base = v * L + part * sub
            for lo in range(0, sub, pc):
                n = min(pc, sub - lo)
                P = _decrypt_range(base + lo, base + lo + n)
                PW = {b: _pow_matrix(c.F, P, b, pre.pe_arr) for b in pre.unknown}
                im = _W.get("inner_mask")
                g ^= _W.get("slice_fn", char2_slice_moments)(
                    pre, layout, PW, [(0, n)],
                    w=(None if im is None else im[part * sub + lo:
                                                  part * sub + lo + n]))[0]
            G[row0 + (v - v0) * nparts + part] = g
    return time.time() - t0


def _weight_acc_2n(arg):
    """ACC[a] ^= XOR_v v^a G_v for the weight range [a0, a1) of one slice block.

    Over F_p this is a dgemm; in characteristic 2 it is again `mom2n`, with the
    SLICE axis as the contraction axis: XOR_v expx[lV[a][v] + lG[w][v]] is
    exactly the kernel's moment matrix of the (N_w x k) Vandermonde logs
    against the (width x k) slice-moment logs.  The weight ranges are disjoint
    rows of the shared accumulator, so the workers never collide."""
    a0, a1 = arg
    from assemble_fast_2n import _expx, _lib
    F = _W["c"].F
    lV, lG, acc = _W["LV"], _W["LG"], _W["ACC"]
    k = lV.shape[1]
    lib, expx = _lib(), _expx(F)
    la = np.ascontiguousarray(lV[a0:a1])
    if lib is not None:
        out32 = np.empty((a1 - a0, lG.shape[0]), dtype=np.int32)
        lib.mom2n(la, lG, a1 - a0, lG.shape[0], k, expx, out32)
        acc[a0:a1] ^= out32.astype(np.int64)
    else:
        for i in range(a1 - a0):
            acc[a0 + i] ^= np.bitwise_xor.reduce(expx[la[i][None, :] + lG],
                                                 axis=1).astype(np.int64)
    return a1 - a0


def fast_slice_moments(pre, layout, PW, bounds, w=None):
    """`assemble_fast.slice_moment_table` with w = None, but one dgemm per slice.

    The reference builds the ten block-pair moments as ten separate
    (nP x L) @ (L x nP) products and converts BOTH operands to float64 inside
    every one of them -- twenty conversions of a 75 x 65 537 panel per slice,
    which at the 12-round size is 786 MB of pure cast traffic per slice and
    dominates the cost (measured ~4.2 s of CPU per slice).  Stacking the four
    blocks into one (4nP x L) panel turns that into ONE conversion and ONE
    dgemm, and the pair moments are read off the block structure of X X^T:

        (X X^T)[nP*a + i][nP*b + j] = sum_x PW[A_a][i][x] * PW[A_b][j][x] .

    Exactness is the same argument `assemble_fast` uses everywhere: the sum has
    L <= exact_chunk(p) terms, each a product of two residues.

    `w` (O15, two active words): a per-point weight of length L, the
    divided-difference mask of the INNER axis.  It scales ONE operand of the
    product -- (X w) X^T, not (X s)(X s)^T -- reduced mod p first so that every
    term is still a product of two residues and the chunk bound is unchanged.
    """
    F, p = pre.F, pre.p
    assert F.char != 2, "prime-field path"
    n = layout.nP
    blocks = layout.blocks
    nb = len(blocks)
    idx = {b: i for i, b in enumerate(blocks)}
    G = np.zeros((len(bounds), layout.width), dtype=np.int64)
    chunk = max(1, exact_chunk(p))
    for k, (lo, hi) in enumerate(bounds):
        X = np.ascontiguousarray(np.concatenate([PW[b][:, lo:hi] for b in blocks],
                                                axis=0))
        Xf = X.astype(np.float64)
        L = hi - lo
        if w is None:
            Xw, Xwf = X, Xf
        else:
            assert len(w) == L, f"inner mask has {len(w)} entries, slice has {L}"
            Xw = (X * np.asarray(w, dtype=np.int64)[None, :]) % p
            Xwf = Xw.astype(np.float64)
        if L <= chunk:
            M = (Xwf @ Xf.T) % p
        else:
            M = np.zeros((nb * n, nb * n), dtype=np.float64)
            for c0 in range(0, L, chunk):
                c1 = min(c0 + chunk, L)
                M = (M + (Xwf[:, c0:c1] @ Xf[:, c0:c1].T) % p) % p
        M = M.astype(np.int64)
        for b in blocks:
            o = layout.off_pm[b]
            G[k, o:o + n] = Xw[idx[b] * n:(idx[b] + 1) * n].sum(axis=1) % p
        for (A, B) in layout.pairs:
            o = layout.off_pair[(A, B)]
            G[k, o:o + n * n] = M[idx[A] * n:(idx[A] + 1) * n,
                                  idx[B] * n:(idx[B] + 1) * n].ravel()
    return G


def newest_checkpoint(cdir, prefix, st, nslice, ck):
    """(accumulator, slices already folded in) from the newest usable
    checkpoint of structure `st`, or (None, 0).

    A checkpoint is usable when it is a whole number of chunks, is short of the
    full structure, and loads -- a file half-written when the job was killed
    simply fails to load and is skipped."""
    import glob
    import re
    best, bestfn = 0, None
    # exactly <prefix>_st<st>_<slices>.npy: a pass label ("0_a11", "0_in0-1")
    # is part of `st`, so the single-axis lookup must not pick a pass file up
    pat = re.compile(rf"^{re.escape(prefix)}_st{re.escape(str(st))}_(\d+)\.npy$")
    for fn in glob.glob(os.path.join(cdir, f"{prefix}_st{st}_*.npy")):
        m = pat.match(os.path.basename(fn))
        if not m:
            continue
        d = int(m.group(1))
        if 0 < d < nslice and d % ck == 0 and d > best:
            best, bestfn = d, fn
    if bestfn is None:
        return None, 0
    try:
        return np.load(bestfn), best
    except Exception as e:                       # truncated by the kill
        print(f"    checkpoint {bestfn} unusable ({e}); starting from 0",
              flush=True)
        return None, 0


def _labels(weights, inner, by_inner):
    """The weight label of every row batch of one pass: the single-axis
    exponent, (a0, a1) for --weight-dims 2, the full vector for the S21
    multi-axis path."""
    if by_inner is not None:
        return [tuple(w) + tuple(inner) for w in weights]
    if inner:
        return [(w, inner[0]) for w in weights]
    return list(weights)


def row_order(pre, weights, ycomb):
    """The (weight, tag) label of every row a structure contributes, in the
    order `assemble_rows` emits them (needed to relabel rows loaded back from
    a `--rows-dir` checkpoint)."""
    out = []
    for w in weights:
        if ycomb is None:
            out.extend([(w, j) for j in pre.outer])
            if getattr(pre, "col_pair2", None) is not None:      # T1: 8 rows
                out.extend([(w, (j, 1)) for j in pre.outer])
        elif isinstance(ycomb[0], dict):                         # T1: dim K dicts
            out.extend([(w, ("cheap12", t)) for t in range(len(ycomb))])
        else:
            out.extend([(w, tuple(int(t) for t in y)) for y in ycomb])
    return out


def assemble_rows(pre, layout, acc, npts, weights, ycomb, lrow):
    """One structure's equation rows, one batch per weight.

    `rows_from_moments`/`rows_from_moments_2n` turn a weighted moment vector
    into the four per-outer-block rows; `ycomb` (O10) folds those four into
    dim K combined rows."""
    F = pre.F
    if getattr(pre, "col_pair2", None) is not None:
        # S19 (R9 / T1): the eight (block, coordinate) rows of the extended
        # system, folded with the two-coordinate kernel vectors of E13
        sys.path.insert(0, os.path.join(HERE, "..", "E13_b_coordinate"))
        import bcoord as BC
        out = []
        for i, _w in enumerate(weights):
            out.extend(BC.rows_from_moments_b_combined(
                pre, layout.moments(acc[i], npts), lrow, ycomb))
        return np.asarray(out, dtype=np.int64)
    asm = rows_from_moments_2n if F.char == 2 else rows_from_moments
    out = []
    for i, _w in enumerate(weights):
        rows = asm(pre, layout.moments(acc[i], npts), lrow)
        if ycomb is None:
            out.extend(rows)
        else:
            out.extend(combine_rows(rows, y, F) for y in ycomb)
    return np.asarray(out, dtype=np.int64)


def subgroup_coset(p, k, rng, rep=None):
    """One coset a*H of the order-2^k subgroup H of F_p^* (O12 Sect. 5.5).

    Same construction as `experiments/Y03_zero_sum/run.py::subgroup_coset`,
    which is where the coset zero-sums of Y03-6 were measured; that module is
    called `run.py` and three experiment directories carry a `run.py`, so it is
    repeated here rather than imported.  `tests/test_stream_weighted_12round.py`
    locks the two against each other."""
    assert (p - 1) % (1 << k) == 0
    from dux.field import _prime_factors
    fs = _prime_factors(p - 1)
    g = next(x for x in range(2, p)
             if all(pow(x, (p - 1) // f, p) != 1 for f in fs))
    h = pow(g, (p - 1) >> k, p)
    H = np.empty(1 << k, dtype=np.int64)
    v = 1
    for i in range(1 << k):
        H[i] = v
        v = (v * h) % p
    a = int(rng.integers(1, p)) if rep is None else rep
    return (H * a) % p


def vandermonde(F, vals, weights):
    """V[i][a] = vals[i]^weights[a] over F (int64)."""
    from assemble_fast import pow_field_vec
    V = np.empty((len(vals), len(weights)), dtype=np.int64)
    for j, a in enumerate(weights):
        V[:, j] = pow_field_vec(F, vals, int(a))
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-257")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--pos", type=int, default=3)
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--full-block", type=int, default=None,
                    help="O11: the four ciphertext words of this inner block "
                         "run over F_q^4 (the first S layer is then free); "
                         "overrides --active, and `plan` halves the usable "
                         "weights.  Same spelling as run_zsx.py.")
    ap.add_argument("--weight-min", type=int, default=0,
                    help="drop the weights below this one.  A single coset "
                         "already forces a >= 1 (the plain equation is not a "
                         "zero sum there); this makes the same restriction "
                         "available on a full-domain structure, which is what "
                         "isolates 'is the a = 0 row carrying rank?'")
    ap.add_argument("--coset", type=int, default=None,
                    help="F_p (O12 Sect. 5.5): the single active ciphertext "
                         "word runs over ONE coset of the order-2^k subgroup "
                         "of F_p^*, so the structure is 2^k points and the "
                         "admissible weights are those with 2^k not dividing a "
                         "(a = 0 is NOT one of them)")
    ap.add_argument("--mixed", default=None,
                    help="T2 (R9), F_p only: the STRUCTURE TYPE of every active "
                         "word, 'k' for a coset of the order-2^k subgroup and "
                         "'full' for all of F_p, e.g. '14,full'.  The coset "
                         "words come first (the weight axis is the slowest "
                         "one).  The threshold is T = sum_i T_i of Theorem 1, "
                         "and every coset axis needs a_i >= 1 with 2^{k_i} not "
                         "dividing a_i -- `plan` filters that.")
    ap.add_argument("--point-set", type=int, default=None,
                    help="O15 (W25): every active ciphertext word runs over "
                         "its own RANDOM set of this many points instead of "
                         "all of F_q, and every sum is masked with the "
                         "divided-difference weights, so the threshold becomes "
                         "T' = s (n - 1).  One active word (both "
                         "characteristics), or two over F_p (S16: the inner "
                         "axis' mask is folded into the per-slice panel, "
                         "M = (X w) X^T); two words over F_{2^n} still need "
                         "attack_2round_fast.py --point-set.")
    ap.add_argument("--point-seed", type=int, default=None,
                    help="seed of the O15 point sets (default: --seed)")
    ap.add_argument("--slice-parts", type=int, default=1,
                    help="characteristic 2: split each slice across this many "
                         "tasks (load balance when a slice is q^3 points)")
    ap.add_argument("--point-chunk", type=int, default=1 << 16,
                    help="characteristic 2: points decrypted at once inside a "
                         "slice (a full-block slice has q^3 points)")
    ap.add_argument("--ks-literal", action="store_true",
                    help="YuX only: the sliding-window reading of Algorithm 1 "
                         "line 2 (default: the non-overlapping Sect. VI-D one)")
    ap.add_argument("--weights", type=int, default=None,
                    help="N_w: the weights are a = 0 .. N_w-1 on the FIRST "
                         "active word (default: the whole margin)")
    ap.add_argument("--weight-dims", type=int, default=1,
                    help="T3 (R9): weight over the first TWO active words.  The "
                         "driver walks a GRID: --weights picks the first-axis "
                         "exponents (from `plan`'s vectors) and --weight-a1 the "
                         "second-axis ones, and a pass keeps the pairs with "
                         "a0 + a1 < usable_norm, so the row count is at most "
                         "|{a0}| x |{a1}| x dim K.  "
                         "M_{a0,a1} = sum_x x^{a0} [Pi_x^T diag(y^{a1}) Pi'_x]. "
                         "The structure is walked once per a1 (the y^{a1} "
                         "factor is the INNER MASK the O15 path already folds "
                         "into one operand of the panel product), and the a0 "
                         "axis stays the Vandermonde.  1 (the default) is the "
                         "single-axis path, unchanged.")
    ap.add_argument("--vandermonde-axes", type=int, default=1,
                    help="S21 (R9 / T3), characteristic 2: slice the structure "
                         "by its first m active words JOINTLY, so that the "
                         "first m weight axes are one multi-index Vandermonde "
                         "over the sub-slices (one pass over the structure for "
                         "all of them); the remaining --weight-dims - m axes "
                         "are inner masks, one pass per inner exponent tuple.  "
                         "1 (the default) is the single-axis path, unchanged.")
    ap.add_argument("--inner-norm", type=int, default=None,
                    help="--vandermonde-axes m < --weight-dims: keep the weight "
                         "vectors whose inner exponents sum to at most this "
                         "(the number of passes is the number of inner tuples)")
    ap.add_argument("--max-vectors", type=int, default=None,
                    help="--vandermonde-axes >= 2: cap the weight vectors (in "
                         "`plan`'s |a| order, after --inner-norm)")
    ap.add_argument("--subspace-dims", default=None,
                    help="S21 (R9 / T3), characteristic 2: every active word "
                         "runs over the F_2-subspace 0 .. 2^m - 1 of the given "
                         "dimension m (comma list, one per active word); the "
                         "threshold is T = sum_i (2^m_i - 1) (O7).")
    ap.add_argument("--weight-a1", default=None,
                    help="--weight-dims 2: the second-axis exponents, a comma "
                         "list (default: 0..N_a1-1 with N_a1 = --weights-a1-n)")
    ap.add_argument("--weights-a1-n", type=int, default=2,
                    help="--weight-dims 2: how many second-axis exponents to "
                         "use when --weight-a1 is not given")
    ap.add_argument("--combine", default="0001",
                    help="O10 pattern; 'none' keeps the four per-block rows")
    ap.add_argument("--cheap-set", default="2",
                    help="S19 (R9 / T1), DuX / decryption / characteristic 2 "
                         "only: '1,2' streams the EXTENDED combined rows of "
                         "experiments/E13_b_coordinate (the cubic coordinate b "
                         "as a second cheap coordinate; 47 749 columns, "
                         "`1101` at dim K = 4).  The slice kernel then also "
                         "accumulates pm2 and the 16 ordered Mom2 pairs "
                         "(2.6x the width).  The default '2' is the paper's "
                         "row set, unchanged.")
    ap.add_argument("--normalise", action="store_true")
    ap.add_argument("--structures", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--slice-chunk", type=int, default=2048,
                    help="slices per dgemm block (memory knob)")
    ap.add_argument("--block-slices", type=int, default=1,
                    help="slices one worker task decrypts at once; raise it for "
                         "single-word structures, where a slice is one point")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="dump the accumulator every N slices (0 = never)")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume-checkpoints", action="store_true",
                    help="restart a structure from the newest accumulator "
                         "checkpoint instead of from slice 0.  --rows-dir "
                         "already makes a restart skip FINISHED structures; "
                         "this makes it skip the finished CHUNKS of the one "
                         "that was in flight, which is what a kill costs on a "
                         "machine that keeps taking the job out.")
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--weight-grid", default=None,
                    help="comma list of N_w values: solve the row prefix of "
                         "each (the rows are emitted weight by weight, so the "
                         "prefix of W weights is the first W*neq rows) and "
                         "report the rank-vs-weights curve")
    ap.add_argument("--rows-dir", default=None,
                    help="save each structure's assembled rows there and reuse "
                         "them on a re-run: a 13-structure full-block job is "
                         "half a day, so it has to be restartable")
    ap.add_argument("--structure-grid", default=None,
                    help="comma list of structure counts: solve the row prefix "
                         "of each (rows are emitted structure-major) and "
                         "report the rank-vs-structures curve")
    ap.add_argument("--assemble-only", action="store_true",
                    help="stop after the weighted moments (budget calibration)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rounds = a.layers + 2
    fam = cipher_family(a.instance)
    c = get_cipher(a.instance, rounds=rounds, ks_literal=a.ks_literal)
    F = c.F
    # O11: a whole inner block of ciphertext words running over F_q^4 buys the
    # first S layer for free.  The weight word has to be the SLOWEST axis of
    # the product set, so the block's position-3 word is listed first; the O7
    # tool takes the free block itself instead of those four words, and after
    # the change of variables that coordinate has degree 2 in y, so each power
    # of the weight costs TWO degree units (Y06 Sect. 5, S11 measured).
    free_blocks = () if a.full_block is None else (int(a.full_block),)
    if free_blocks:
        b0 = free_blocks[0]
        act = [4 * b0 + 3] + [4 * b0 + i for i in range(3)]
        plain_active = []
    else:
        act = active_words(a.pos, a.active)
        plain_active = act
    assert len(act) >= 1
    q = F.q
    cheap_set = tuple(int(v) for v in a.cheap_set.split(","))
    combine = None if a.combine in (None, "none") else a.combine
    if cheap_set == (1, 2):
        assert F.char == 2 and fam == "dux", \
            "the cheap set {1,2} is DuX in characteristic 2 (E13)"
        assert not free_blocks and not a.point_set, \
            "the T1 rows take ordinary active words, no full block / point set"
        assert combine in WT.X1X1_CLASSES, (
            "the cheap set {1,2} needs a pattern with positions 1 and 3 "
            f"balanced, i.e. one of {list(WT.X1X1_CLASSES)}, not {combine}")
        pre, layout, _fn = precomp_b(c)
    else:
        assert cheap_set == (2,), "the only cheap sets are {2} and T1's {1,2}"
        pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher=fam)
        layout = MomentLayout(pre)
    M = len(pre.mons)
    coset_k = a.coset
    if coset_k:
        assert F.char != 2, "the coset structures of O12 Sect. 5.5 are F_p only"
        assert len(act) == 1 and not free_blocks, \
            "the coset structure uses one active ciphertext word"
    mixed_ks = None
    if a.mixed:
        assert F.char != 2, "T2's mixed structures are an F_p construction"
        assert not free_blocks and not coset_k and not a.point_set, \
            "--mixed replaces --coset / --point-set / --full-block"
        sys.path.insert(0, os.path.join(HERE, "..", "E15_mixed_coset"))
        import mixed as _MX
        mixed_ks = _MX.parse_mixed(a.mixed, len(act))
    sub_dims = None
    if a.subspace_dims:
        sub_dims = [int(v) for v in a.subspace_dims.split(",")]
        assert F.char == 2, "--subspace-dims is a characteristic-2 construction"
        assert not (free_blocks or coset_k or a.mixed or a.point_set), \
            "--subspace-dims replaces --full-block / --coset / --mixed / --point-set"
        assert len(sub_dims) == len(act) and all(1 <= m <= F.n for m in sub_dims), \
            f"one subspace dimension in 1..{F.n} per active word"
    pset_n = a.point_set
    if pset_n:
        assert not coset_k, "a coset already is a point set with its own threshold"
        assert not free_blocks, \
            "a full block must run over all of F_q^4 (O11); a point set cannot"
        assert len(act) == 1 or (len(act) == 2 and F.char != 2), \
            "the streaming point-set path takes one active word, or two over " \
            "F_p (S16 optional A); see --point-set"
        assert pset_n <= q, f"{pset_n} points requested from {F.name}"
        import interp_mask as _IM
    # W19-C: `plan` now applies the four O12 corrections itself (the full-block
    # halving, the coset exclusion, the subspace thresholds and the O9 a = 0
    # rows) and returns `usable_weights`.  The driver's job is only to hand it
    # `free_blocks` / `coset` and the active words the CRITERION should see --
    # empty for a full block, because those four words are the free block, not
    # four ordinary active words.
    pl = WT.plan(F, plain_active, a.layers, "dec", fam, combine,
                 max(1, a.weight_dims), a.weights,
                 free_blocks=free_blocks, coset=(mixed_ks or coset_k),
                 pointset=[pset_n] * len(plain_active) if pset_n else None,
                 cheap_set=(cheap_set if cheap_set != (2,) else None),
                 subspace_dims=sub_dims)
    if free_blocks and a.weight_dims > 1:
        # T3 on a full block: `plan` puts the d weight axes on the block's d
        # cheapest ciphertext words (positions 3, 2, 1, 0 for DuX after the
        # substitution); those must be the d SLOWEST axes of the walk, so the
        # block is enumerated in that order (same point set, same sums).
        ww = list(pl.weight_words[:a.weight_dims])
        assert ww[0] == act[0]
        act = ww + [w for w in act if w not in ww]
    ws, margin, crit = pl.weights, pl.margin, pl.crit
    if a.weight_min:
        ws = [w for w in ws if int(w[0]) >= a.weight_min]
    usable = pl.usable_weights
    assert a.weights is None or a.weights <= usable, (
        f"--weights {a.weights} exceeds the {usable} weights O12 admits here "
        f"({pl.weight_rule})")
    if free_blocks:
        assert pl.weight_word == act[0], (
            f"plan puts the full-block weight on ciphertext word "
            f"{pl.weight_word}, but the structure's slowest axis is {act[0]}")
    weights = sorted({int(w[0]) for w in ws})
    ycomb = (WT.combine_vectors(fam, F, "dec", combine,
                                cheap=(list(cheap_set) if cheap_set != (2,)
                                       else None))
             if combine else None)
    neq = len(ycomb) if ycomb else (8 if cheap_set == (1, 2) else 4)
    if mixed_ks:
        mix_rng = np.random.default_rng(a.seed + 4000)
        npts = 1
        nslice = None
        for j, k in enumerate(mixed_ks):
            n = F.q if k is None else (1 << k)
            npts *= n
            if j == 0:
                nslice = n
    elif coset_k:
        rep_rng = np.random.default_rng(a.seed + 4000)
        nslice = 1 << coset_k
        npts = nslice              # one active word over one coset
    elif pset_n:
        pset_rng = np.random.default_rng(a.point_seed if a.point_seed is not None
                                         else a.seed + 4000)
        nslice = pset_n
        npts = pset_n ** len(act)  # every active word over its own point set
    elif sub_dims:
        sub_values = subspace_values(F, sub_dims)
        npts = 1
        for v in sub_values:
            npts *= len(v)
        nslice = len(sub_values[0])
    else:
        nslice = q
        npts = q ** len(act)
    # S21: `--vandermonde-axes m` slices by the first m axes jointly
    vaxes = max(1, a.vandermonde_axes)
    assert 1 <= vaxes <= max(1, a.weight_dims) <= len(act), \
        "1 <= --vandermonde-axes <= --weight-dims <= number of active words"
    if vaxes > 1:
        assert F.char == 2, "the joint multi-axis Vandermonde is the char-2 path"
        assert not (coset_k or mixed_ks or pset_n or a.weight_grid), \
            "--vandermonde-axes >= 2 takes full-domain / subspace / full-block words"
        axis_sizes = ([len(v) for v in sub_values] if sub_dims
                      else [q] * len(act))
        nslice = 1
        for n in axis_sizes[:vaxes]:
            nslice *= n
    slice_len = npts // nslice     # points per value of the weight word

    print(f"{a.instance} ({F.name}): r = {rounds} ({a.layers}-layer distinguisher + 2), "
          f"active words {act}")
    print(f"  monomials M = {M} = 2^{np.log2(M):.1f}; {a.structures} structure(s) "
          f"x {npts} chosen ciphertexts = 2^{np.log2(a.structures * npts):.2f} data")
    if cheap_set == (1, 2):
        print(f"  T1 (R9 / S19): cheap set {{1,2}} -- the extended system of E13 "
              f"(M_b = {M} columns), slice layout width {layout.width} "
              f"(pm, pm2, {len(layout.pairs)} Mom + {len(layout.pairs2)} Mom2 "
              f"pairs), streamed", flush=True)
    if free_blocks:
        print(f"  O11: full block {free_blocks[0]} (ciphertext words {act}), "
              f"weight word {act[0]}")
    if mixed_ks:
        print(f"  T2 (R9): mixed structure {a.mixed} on words {act} -> "
              f"{npts} = 2^{np.log2(npts):.2f} chosen ciphertexts/structure; "
              f"T = {crit['threshold']} (plain "
              f"{crit.get('threshold_plain')}), {usable} admissible weights")
    if sub_dims:
        print(f"  T3 (R9 / S21): subspace structure, dims {sub_dims} on words "
              f"{act} -> {npts} = 2^{np.log2(npts):.2f} chosen ciphertexts/"
              f"structure; T = {crit['threshold']}", flush=True)
    if coset_k:
        print(f"  O12 Sect. 5.5: ONE coset of the order-2^{coset_k} subgroup "
              f"of F_{F.p}^* on word {act[0]} -> {npts} = 2^{coset_k} chosen "
              f"ciphertexts/structure; {usable} admissible weights "
              f"(1 <= a < {margin}, 2^{coset_k} does not divide a)")
    print(f"  O12: margin {margin} (layer {a.layers} pattern "
          f"{crit['patterns'][a.layers - 1]}), N_w = {len(weights)} weights "
          f"a = {weights[0]}..{weights[-1]} on word {act[0]}; "
          f"{nslice} slices of {slice_len} points"
          + (f"; O10 combine {combine}: dim K = {len(ycomb)}, y = {ycomb}"
             if ycomb else "")
          + f" -> {len(weights) * neq} equations/structure", flush=True)
    print(f"  O12 weight rule: {pl.weight_rule} "
          f"(usable_weights = {pl.usable_weights})", flush=True)

    width = layout.width
    acc_bytes = len(weights) * width * 8
    if F.char == 2 and a.slice_parts > 1:
        print(f"  char-2 slice split: {a.slice_parts} tasks per slice "
              f"({slice_len // a.slice_parts} points each)")
    print(f"  accumulator {len(weights)} x {width} x 8 B = "
          f"{acc_bytes / 1e9:.2f} GB; slice table {a.slice_chunk} x {width} x 8 B "
          f"= {a.slice_chunk * width * 8 / 1e9:.2f} GB", flush=True)

    t0 = time.time()
    all_rows, order = [], []
    per_structure = []
    rng0 = np.random.default_rng(a.seed)
    rks0 = c.key_schedule(c.random_key(rng0))
    tag0 = a.tag or f"{a.instance}_r{rounds}_Nw{len(weights)}"
    coset_reps = []
    # T3 (R9): with --weight-dims 2 the structure is walked once per second-axis
    # exponent a1; within a pass the admissible first-axis exponents are the a0
    # with |a| = a0 + a1 below the same bound, so the pass with a1 = 0 is
    # exactly the single-axis run and every later pass is strictly shorter.
    assert not (a.weight_grid and a.weight_dims > 1), \
        "--weight-grid is a prefix of the single-axis weight order"
    multi = vaxes > 1 or a.weight_dims > 2
    by_inner = None
    if a.weight_dims == 1:
        a1_list = [0]
    elif not multi:
        assert a.weight_dims == 2, "the streamed path takes one or two weight axes"
        assert len(act) >= 2, "two weight axes need two active words"
        a1_list = ([int(v) for v in a.weight_a1.split(",")] if a.weight_a1
                   else list(range(a.weights_a1_n)))
        print(f"  T3: second weight axis on word {act[1]}, a1 in {a1_list}; "
              f"the grid is {len(weights)} x {len(a1_list)} pairs filtered by "
              f"a0 + a1 < {pl.usable_norm}", flush=True)
    else:
        # S21: the plan's weight VECTORS, split into (first vaxes axes = the
        # joint Vandermonde, the rest = inner masks); one pass per inner tuple
        assert F.char == 2, "the multi-axis streamed path is characteristic 2"
        vecs = [tuple(int(t) for t in w) for w in ws]
        if a.inner_norm is not None:
            vecs = [w for w in vecs if sum(w[vaxes:]) <= a.inner_norm]
        if a.max_vectors:
            vecs = vecs[:a.max_vectors]
        assert vecs, "no weight vector left"
        inner_list = sorted({w[vaxes:] for w in vecs})
        by_inner = {inn: sorted(w[:vaxes] for w in vecs if w[vaxes:] == inn)
                    for inn in inner_list}
        a1_list = inner_list
        print(f"  T3 (S21): {len(vecs)} weight vectors over the first "
              f"{a.weight_dims} active words {act[:a.weight_dims]}; joint "
              f"Vandermonde on the first {vaxes} ({nslice} sub-slices of "
              f"{slice_len} points), {len(inner_list)} inner tuple(s) "
              f"{inner_list[:8]}{'...' if len(inner_list) > 8 else ''} on words "
              f"{act[vaxes:a.weight_dims]} -> "
              f"{len(vecs) * neq} equations/structure", flush=True)
    passes = [(st, a1) for st in range(a.structures) for a1 in a1_list]
    weights_all = list(weights) if by_inner is None else vecs
    values_full = None
    for st, a1 in passes:
        if by_inner is None:
            weights = [w for w in weights_all if w + a1 < pl.usable_norm]
            inner = (a1,) if a.weight_dims > 1 else ()
        else:
            inner = a1
            weights = by_inner[inner]
        assert weights, f"no first-axis weight left at a1 = {a1}"
        seed = a.seed + 5000 + st
        pmask = inner_mask = None
        if sub_dims:
            svals, values = sub_values[0], sub_values
        elif mixed_ks:
            values = _MX.mixed_axes(F, mixed_ks, a.seed + 4000 + st)
            svals = values[0]
        elif coset_k:
            svals = subgroup_coset(F.p, coset_k, rep_rng)
            coset_reps.append(int(svals[0]))
            values = [svals]
        elif pset_n:
            values = [np.sort(pset_rng.choice(q, size=pset_n, replace=False))
                      .astype(np.int64) for _ in act]
            svals = values[0]      # the weight word = the slowest axis
            pmask = _IM.divided_difference_weights(F, svals)
            inner_mask = (_IM.divided_difference_weights(F, values[1])
                          if len(act) == 2 else None)
        else:
            svals, values = None, None
        values_full = (values if values is not None
                       else [np.arange(q, dtype=np.int64)] * len(act))
        if any(inner):
            # the inner-axis weights over ONE slice, in the inner axes' point
            # order: y^{a1} on the second word for --weight-dims 2, the
            # product over the inner tuple for the S21 multi-axis path
            col = inner_weight_mask(F, values_full, vaxes, inner)
            assert col is not None and len(col) == slice_len, \
                "the inner axes must tile a slice"
            inner_mask = (col if inner_mask is None else
                          (F.vmul(inner_mask, col) if F.char == 2
                           else (inner_mask * col) % F.p))
        # the accumulator checkpoints carry the pass label too: a second pass
        # over the same structure must not resume from the first pass's
        # accumulator (the S16 trap, one level down)
        ck_label = (st if not any(inner) and by_inner is None and a.weight_dims == 1
                    else (f"{st}_a1{a1}" if by_inner is None
                          else f"{st}_in" + "-".join(map(str, inner))))
        rows_file = (os.path.join(a.rows_dir,
                                  f"rows_{tag0}_st{st}"
                                  + (f"_a1{a1}" if (a.weight_dims > 1
                                                    and by_inner is None)
                                     else "")
                                  + ("_in" + "-".join(map(str, inner))
                                     if by_inner is not None else "")
                                  + ".npy")
                     if a.rows_dir else None)
        if rows_file and os.path.exists(rows_file):
            srows = np.load(rows_file)
            assert srows.shape[0] == len(weights) * neq, \
                f"{rows_file} has {srows.shape[0]} rows, expected " \
                f"{len(weights) * neq}"
            all_rows.extend(srows)
            order.extend(row_order(pre, _labels(weights, inner, by_inner), ycomb))
            per_structure.append({"structure": st, "a1": a1, "seed": seed,
                                  "loaded_from": rows_file})
            print(f"  structure {st}: {srows.shape[0]} rows loaded from "
                  f"{rows_file}", flush=True)
            continue
        if F.char == 2:
            # S13-A: the same slice decomposition as the F_p path.  The old
            # in-memory route (`char2_weighted_acc`, kept for the equivalence
            # test) materialises the whole structure: 16 int64 arrays of 2^32
            # points is 550 GB, which is why Y06-2's Yu2X-16 half and Y06-3
            # were left unrun.  Here a slice never leaves a worker.
            nwt = len(weights)
            ck = min(a.slice_chunk, nslice)
            npart = max(1, a.slice_parts)
            assert slice_len % npart == 0, "--slice-parts must divide the slice"
            order2 = 2 * F.order
            shm_G = shared_memory.SharedMemory(create=True,
                                               size=ck * npart * width * 8)
            shm_LG = shared_memory.SharedMemory(create=True, size=width * ck * 4)
            shm_LV = shared_memory.SharedMemory(create=True, size=nwt * ck * 4)
            shm_A = shared_memory.SharedMemory(create=True, size=nwt * width * 8)
            G = np.ndarray((ck * npart, width), dtype=np.int64,
                           buffer=shm_G.buf)
            LG = np.ndarray((width, ck), dtype=np.int32, buffer=shm_LG.buf)
            LV = np.ndarray((nwt, ck), dtype=np.int32, buffer=shm_LV.buf)
            ACC = np.ndarray((nwt, width), dtype=np.int64, buffer=shm_A.buf)
            ACC[:] = 0
            lo0 = 0
            if a.resume_checkpoints and a.checkpoint_dir:
                prev, lo0 = newest_checkpoint(a.checkpoint_dir, "acc2n",
                                              ck_label, nslice, ck)
                if prev is not None:
                    ACC[:] = prev
                    print(f"  structure {st}: resumed at slice {lo0}/{nslice} "
                          f"from the accumulator checkpoint", flush=True)
            extra = {"LG": (shm_LG.name, (width, ck), "int32"),
                     "LV": (shm_LV.name, (nwt, ck), "int32"),
                     "ACC": (shm_A.name, (nwt, width), "int64")}
            wstep = max(1, -(-nwt // max(1, 2 * a.procs)))
            wtasks = [(i, min(i + wstep, nwt)) for i in range(0, nwt, wstep)]
            t_mom = t_dot = 0.0
            from assemble_fast_2n import _logs as _logs2n
            try:
                with _make_pool(a.procs, _init,
                                (a.instance, rounds, a.seed, seed, a.pos,
                                 act, slice_len, shm_G.name,
                                 (ck * npart, width), extra, a.point_chunk,
                                 a.ks_literal, values, inner_mask,
                                 cheap_set)) as pool:
                    done = lo0
                    for lo in range(lo0, nslice, ck):
                        hi = min(lo + ck, nslice)
                        k = hi - lo
                        tA = time.time()
                        if npart == 1:
                            tasks = [(v, min(v + a.block_slices, hi),
                                      v - lo, 0, 1)
                                     for v in range(lo, hi, a.block_slices)]
                        else:
                            tasks = [(v, v + 1, (v - lo) * npart, pt, npart)
                                     for v in range(lo, hi)
                                     for pt in range(npart)]
                        for _ in pool.imap_unordered(_block_moments_2n, tasks,
                                                     chunksize=1):
                            pass
                        t_mom += time.time() - tA
                        tB = time.time()
                        Gk = (G[:k] if npart == 1 else
                              np.bitwise_xor.reduce(
                                  G[:k * npart].reshape(k, npart, width),
                                  axis=1))
                        LG[:, :k] = _logs2n(F, Gk).T
                        if vaxes > 1:
                            LV[:, :k] = vandermonde_logs_multi(
                                F, slice_axis_values(values_full, vaxes, lo, hi),
                                weights)
                        else:
                            svals_blk = (np.arange(lo, hi, dtype=np.int64)
                                         if svals is None else svals[lo:hi])
                            LV[:, :k] = vandermonde_logs(
                                F, svals_blk, weights,
                                mask=None if pmask is None else pmask[lo:hi])  # O15
                        if k < ck:
                            # pad with the log sentinel: those products are 0,
                            # so the tail of the last block contributes nothing
                            LG[:, k:] = order2
                            LV[:, k:] = 0
                        for _ in pool.imap_unordered(_weight_acc_2n, wtasks,
                                                     chunksize=1):
                            pass
                        t_dot += time.time() - tB
                        done = hi
                        if a.progress:
                            print(f"    slices {done}/{nslice}  moments {t_mom:.0f}s"
                                  f"  wacc {t_dot:.0f}s  "
                                  f"({time.time() - t0:.0f}s)", flush=True)
                        if (a.checkpoint_every and a.checkpoint_dir
                                and done % a.checkpoint_every < ck):
                            os.makedirs(a.checkpoint_dir, exist_ok=True)
                            fn = os.path.join(a.checkpoint_dir,
                                              f"acc2n_st{ck_label}_{done}.npy")
                            np.save(fn, ACC)
                            h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
                            print(f"    checkpoint {fn} sha256 {h}", flush=True)
                acc = ACC.copy()
            finally:
                for _s in (shm_G, shm_LG, shm_LV, shm_A):
                    _s.close()
                    _s.unlink()
            per_structure.append({"structure": st,
                                  "a1": (list(a1) if by_inner is not None else a1),
                                  "seed": seed, "weights": len(weights),
                                  "moments_s": round(t_mom, 1),
                                  "dgemm_s": round(t_dot, 1)})
            print(f"  structure {st}: slice moments {t_mom:.1f}s + weighted "
                  f"accumulation {t_dot:.1f}s (char-2 streaming)", flush=True)
            if a.assemble_only:
                continue
            tC = time.time()
            srows = assemble_rows(pre, layout, acc, npts, weights, ycomb,
                                  linear_row(c, 0))
            if rows_file:
                os.makedirs(a.rows_dir, exist_ok=True)
                np.save(rows_file, srows)
            all_rows.extend(srows)
            order.extend(row_order(pre, _labels(weights, inner, by_inner), ycomb))
            print(f"  structure {st}{f' (a1 = {a1})' if a.weight_dims > 1 else ''}: "
                  f"{srows.shape[0]} rows assembled "
                  f"({time.time() - tC:.1f}s)", flush=True)
            continue
        acc = np.zeros((len(weights), width), dtype=np.int64)
        vals_all = (np.arange(q, dtype=np.int64) if svals is None else svals)
        ck = min(a.slice_chunk, nslice)
        lo0 = 0
        if a.resume_checkpoints and a.checkpoint_dir:
            prev, lo0 = newest_checkpoint(a.checkpoint_dir, "acc", ck_label,
                                          nslice, ck)
            if prev is not None:
                acc = prev
                print(f"  structure {st}: resumed at slice {lo0}/{nslice} "
                      f"from the accumulator checkpoint", flush=True)
        shm = shared_memory.SharedMemory(create=True, size=ck * width * 8)
        G = np.ndarray((ck, width), dtype=np.int64, buffer=shm.buf)
        # sum over one dgemm block: (#slices) products of two residues, each
        # < p^2; exact in float64 while #slices <= 2^53 / p^2.
        echunk = max(1, exact_chunk(F.p))
        t_mom = t_dot = 0.0
        try:
            with _make_pool(a.procs, _init,
                            (a.instance, rounds, a.seed, seed, a.pos,
                             act, slice_len, shm.name, (ck, width),
                             None, a.point_chunk, a.ks_literal,
                             values, inner_mask)) as pool:
                done = lo0
                for lo in range(lo0, nslice, ck):
                    hi = min(lo + ck, nslice)
                    tA = time.time()
                    tasks = [(v, min(v + a.block_slices, hi), v - lo)
                             for v in range(lo, hi, a.block_slices)]
                    for _ in pool.imap_unordered(_block_moments, tasks,
                                                 chunksize=1):
                        pass
                    t_mom += time.time() - tA
                    tB = time.time()
                    V = vandermonde(F, vals_all[lo:hi], weights)   # (k, N_w)
                    if pmask is not None:                          # O15 mask
                        V = (V * pmask[lo:hi, None]) % F.p
                    Gk = G[:hi - lo]
                    for c0 in range(0, hi - lo, echunk):
                        c1 = min(c0 + echunk, hi - lo)
                        add = (V[c0:c1].T.astype(np.float64)
                               @ Gk[c0:c1].astype(np.float64)) % F.p
                        acc += add.astype(np.int64)
                        acc %= F.p
                    t_dot += time.time() - tB
                    done = hi
                    if a.progress:
                        print(f"    slices {done}/{nslice}  moments {t_mom:.0f}s  "
                              f"dgemm {t_dot:.0f}s  ({time.time() - t0:.0f}s)",
                              flush=True)
                    if (a.checkpoint_every and a.checkpoint_dir
                            and done % a.checkpoint_every < ck):
                        os.makedirs(a.checkpoint_dir, exist_ok=True)
                        fn = os.path.join(a.checkpoint_dir,
                                          f"acc_st{ck_label}_{done}.npy")
                        np.save(fn, acc)
                        h = hashlib.sha256(open(fn, "rb").read()).hexdigest()
                        print(f"    checkpoint {fn} sha256 {h}", flush=True)
        finally:
            shm.close()
            shm.unlink()
        per_structure.append({"structure": st, "a1": a1, "seed": seed,
                              "moments_s": round(t_mom, 1),
                              "dgemm_s": round(t_dot, 1)})
        print(f"  structure {st}: moments {t_mom:.1f}s + dgemm {t_dot:.1f}s",
              flush=True)

        if a.assemble_only:
            continue
        tC = time.time()
        srows = assemble_rows(pre, layout, acc, npts, weights, ycomb,
                              linear_row(c, 0))
        if rows_file:
            os.makedirs(a.rows_dir, exist_ok=True)
            np.save(rows_file, srows)
        all_rows.extend(srows)
        order.extend(row_order(pre, _labels(weights, inner, by_inner), ycomb))
        print(f"  structure {st}{f' (a1 = {a1})' if a.weight_dims > 1 else ''}: "
              f"{srows.shape[0]} rows assembled "
              f"({time.time() - tC:.1f}s)", flush=True)
    weights = weights_all
    t_asm = time.time() - t0
    if a.assemble_only:
        print(f"  assemble-only: {t_asm:.1f}s")
        return

    R = np.asarray(all_rows, dtype=np.int64)
    del all_rows
    rks = rks0

    zero = (0, 0, 0, 0)
    known_cols = {pre.col_const: 1}
    if a.normalise:
        for m, i in pre.idx.items():
            tag, inner = m
            if tag is None and inner and all(e == zero for _b, e in inner):
                known_cols[i] = 1
    kc = sorted(known_cols)
    keep = [i for i in range(M) if i not in known_cols]
    kv = np.array([known_cols[i] for i in kc], dtype=np.int64)
    if F.char == 2:
        B = np.zeros(R.shape[0], dtype=np.int64)
        for t, ci in enumerate(kc):
            B ^= F.vmul(R[:, ci], np.int64(kv[t]))
    else:
        B = (-(R[:, kc] @ kv)) % F.p
    R = np.ascontiguousarray(R[:, keep])
    cols = [pre.mons[i] for i in keep]
    nrows, ncols = R.shape
    print(f"  system {nrows} x {ncols} over {F.name} ({len(kc)} columns pinned, "
          f"{nrows * ncols * 8 / 2**30:.2f} GiB)", flush=True)

    cpos = {m: i for i, m in enumerate(cols)}
    truth = {}
    for b in range(4):
        for i in range(4):
            truth[f"{b},{i}"] = rks[0][4 * b + i]

    def _solve(rows, rhs, progress=None):
        if F.char == 2:
            import gf2n_solve
            if gf2n_solve.available(None):
                return gf2n_solve.solve(F, rows, rhs, progress=progress or 0)
            return modp_solve.solve_gf2n(F, rows, rhs, progress=progress)
        return modp_solve.solve(F.p, rows, rhs, progress=progress)

    def score(det):
        ok = {}
        for b in range(4):
            for i in range(4):
                e = tuple(1 if t == i else 0 for t in range(4))
                m = (None, ((b, e),))
                if m in cpos and cpos[m] in det:
                    ok[f"{b},{i}"] = det[cpos[m]]
        return ok, sum(1 for k in truth if ok.get(k) == truth[k])

    curve = []
    if a.weight_grid:
        grid = sorted({int(v) for v in a.weight_grid.split(",")}
                      | {len(weights)})
        per = len(weights) * neq          # rows one structure contributes
        for W in grid:
            if W > len(weights):
                continue
            # rows are emitted structure-major (for each structure, for each
            # weight, neq rows), so "the first W weights of EVERY structure" is
            # a gather, not a prefix -- same convention as nmin_scan.py:210.
            keepr = np.concatenate([np.arange(st * per, st * per + W * neq)
                                    for st in range(a.structures)])
            nr = int(keepr.size)
            tG = time.time()
            dG, rG, fG = _solve(R[keepr], B[keepr])
            _okG, goodG = score(dG)
            rec = {"weights": W, "rows": nr, "rank": rG, "free": fG,
                   "pinned": len(dG), "inner_correct": goodG,
                   "solve_s": round(time.time() - tG, 1)}
            curve.append(rec)
            print(f"  weights={W:6d}  rows {nr:6d}  rank {rG:6d}  "
                  f"pinned {len(dG):6d}  words {goodG}/16  "
                  f"({rec['solve_s']:.1f}s)", flush=True)

    scurve = []
    if a.structure_grid:
        per = len(order) // a.structures        # rows every structure contributed
        for S in sorted({int(v) for v in a.structure_grid.split(",")}
                        | {a.structures}):
            if S > a.structures or S < 1:
                continue
            nr = S * per
            tS = time.time()
            dS, rS, fS = _solve(R[:nr], B[:nr])
            _okS, goodS = score(dS)
            rec = {"structures": S, "rows": nr, "rank": rS, "free": fS,
                   "pinned": len(dS), "inner_correct": goodS,
                   "solve_s": round(time.time() - tS, 1)}
            scurve.append(rec)
            print(f"  structures={S:4d}  rows {nr:6d}  rank {rS:6d}  "
                  f"pinned {len(dS):6d}  words {goodS}/16  "
                  f"({rec['solve_s']:.1f}s)", flush=True)

    t1 = time.time()
    det, rank, nfree = _solve(R, B, progress=a.progress)
    t_solve = time.time() - t1
    print(f"  rank {rank} / {ncols} unknowns ({nfree} free), "
          f"{len(det)} monomials pinned down  ({t_solve:.1f}s)", flush=True)

    ok, good = score(det)
    print(f"  inner key words recovered correctly: {good}/{len(truth)}")
    if good != len(truth):
        print(f"    got   {ok}")
        print(f"    truth {truth}")

    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "rounds": rounds, "layers": a.layers, "active_words": act,
           "unknown_blocks": [0, 1, 2, 3], "outer_blocks": [0, 1, 2, 3],
           "free_blocks": list(free_blocks), "ks_literal": bool(a.ks_literal),
           "weight_rule": pl.weight_rule,
           "points_per_structure": npts, "structures": a.structures,
           "slices": nslice, "points_per_slice": slice_len,
           "coset": coset_k, "coset_reps": coset_reps,
           "mixed": ({"spec": a.mixed, "ks": mixed_ks,
                      "axis_seed": a.seed + 4000,
                      "log2_points": round(float(np.log2(npts)), 2)}
                     if mixed_ks else None),
           "point_set": ({"n": pset_n, "words": len(act),
                          "seed": (a.point_seed if a.point_seed is not None
                                   else a.seed + 4000),
                          "threshold": len(act) * (pset_n - 1)}
                         if pset_n else None),
           "block_slices": a.block_slices, "slice_chunk": a.slice_chunk,
           "slice_parts": a.slice_parts, "point_chunk": a.point_chunk,
           "monomials": M, "cheap_set": list(cheap_set),
           "subspace_dims": sub_dims,
           "weights": {"n": len(weights), "dims": a.weight_dims, "margin": margin,
                       "usable": usable, "weight_min": a.weight_min,
                       "first": (list(weights[0]) if by_inner is not None
                                 else weights[0]),
                       "last": (list(weights[-1]) if by_inner is not None
                                else weights[-1]), "weight_word": act[0],
                       "degree_units_per_power": 2 if free_blocks else 1,
                       "range": (f"a < {len(weights)}" if by_inner is None else
                                 f"|a| < {pl.usable_norm}, inner norm <= "
                                 f"{a.inner_norm}, {len(weights)} vectors"),
                       "vandermonde_axes": vaxes,
                       "weight_words": (list(act[:a.weight_dims])
                                        if a.weight_dims > 1 else [act[0]]),
                       "inner_tuples": ([list(t) for t in a1_list]
                                        if by_inner is not None else None),
                       "usable_norm": pl.usable_norm},
           "equation_rows": ({"combine": combine, "dim_K": len(ycomb),
                              "y": [([[list(k), int(v)] for k, v in sorted(y.items())]
                                     if isinstance(y, dict)      # T1: {(block, coord): c}
                                     else list(map(int, y))) for y in ycomb]}
                             if ycomb else
                             {"combine": None, "rows": "one per outer block"}),
           "equations": int(nrows), "unknowns": int(ncols), "rank": rank,
           "free": nfree, "pinned": len(det), "inner_correct": good,
           "inner_total": len(truth), "normalise": a.normalise,
           "data_log2": round(float(np.log2(a.structures * npts)), 2),
           "assemble_s": round(t_asm, 1), "solve_s": round(t_solve, 1),
           "weight_curve": curve, "structure_curve": scurve,
           "rows_dir": a.rows_dir,
           "per_structure": per_structure, "procs": a.procs, "seed": a.seed,
           "solver": "c-omp" if F.char == 2 else "numpy-blocked"}
    print(f"  data 2^{res['data_log2']}; assemble {t_asm:.1f}s, solve {t_solve:.1f}s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"s10_{tag0}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
