"""S9 (1b/6b): `modp_solve.solve` fills its augmented buffer in place.

The old entry point built three full-size float64 temporaries on top of the
caller's int64 row block (`np.asarray(..., float64)` -> `hstack` -> `% p`), so
the elimination peaked at ~4.9x (rows x cols x 8 B); step 1b measured 13.13 GB
on the 8 800 x 38 040 system over F_193 that OOM-killed the same run inside the
15 GB container.  These tests lock the REPLACEMENT's semantics: whatever the
buffering does, `solve` must still return exactly the triple the reference
Gauss-Jordan returns, for int64 input, for plain Python lists, and for rows that
carry values outside [0, p).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))

import modp_solve  # noqa: E402


def reference_solve(p, rows, rhs):
    """Textbook Gauss-Jordan over F_p; same return triple as modp_solve.solve."""
    A = (np.asarray(rows, dtype=np.int64) % p).astype(np.int64)
    b = (np.asarray(rhs, dtype=np.int64).reshape(-1) % p).astype(np.int64)
    m, n = A.shape
    Aug = np.concatenate([A, b[:, None]], axis=1)
    piv_cols, piv_rows, r = [], [], 0
    for c in range(n):
        nz = np.flatnonzero(Aug[r:, c])
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        Aug[[r, pr]] = Aug[[pr, r]]
        Aug[r] = (Aug[r] * pow(int(Aug[r, c]), p - 2, p)) % p
        for i in range(m):
            if i != r and Aug[i, c]:
                Aug[i] = (Aug[i] - Aug[i, c] * Aug[r]) % p
        piv_cols.append(c)
        piv_rows.append(r)
        r += 1
        if r == m:
            break
    free = np.ones(n, dtype=bool)
    if piv_cols:
        free[np.array(piv_cols, dtype=np.int64)] = False
    det = {ci: int(Aug[ri, n]) for ci, ri in zip(piv_cols, piv_rows)
           if not Aug[ri, :n][free].any()}
    return det, len(piv_cols), int(free.sum())


@pytest.mark.parametrize("p,m,n,seed", [(193, 40, 25, 1), (257, 60, 30, 2),
                                        (65537, 50, 35, 3)])
def test_solve_matches_the_reference_elimination(p, m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, p, size=(m, n), dtype=np.int64)
    x = rng.integers(0, p, size=n, dtype=np.int64)
    b = (A @ x) % p
    det, rank, nfree = modp_solve.solve(p, A.copy(), b.copy(), block=8)
    rdet, rrank, rfree = reference_solve(p, A, b)
    assert (rank, nfree) == (rrank, rfree)
    assert det == rdet
    # a full-rank square-ish system must pin every column to the true solution
    if rank == n:
        assert all(det[c] == int(x[c]) for c in range(n))


def test_solve_accepts_lists_and_unreduced_entries():
    """The old entry point reduced mod p after the float64 cast; the buffered
    one must do the same for negative and out-of-range inputs, and must still
    take a plain list of lists."""
    p = 193
    rng = np.random.default_rng(7)
    A = rng.integers(-4 * p, 4 * p, size=(30, 12), dtype=np.int64)
    x = rng.integers(0, p, size=12, dtype=np.int64)
    b = (A @ x) % p
    det_arr, rank_arr, free_arr = modp_solve.solve(p, A.copy(), b.copy(), block=5)
    det_lst, rank_lst, free_lst = modp_solve.solve(p, A.tolist(), b.tolist(), block=5)
    assert (det_arr, rank_arr, free_arr) == (det_lst, rank_lst, free_lst)
    assert rank_arr == 12 and all(det_arr[c] == int(x[c]) for c in range(12))


def test_solve_does_not_modify_the_caller_rows():
    """`attack_2round_fast` reads R.shape after the solve, and `nmin_scan`
    re-slices the same row block for the next grid point, so the argument must
    come back untouched."""
    p = 257
    rng = np.random.default_rng(11)
    A = rng.integers(0, p, size=(20, 9), dtype=np.int64)
    b = rng.integers(0, p, size=20, dtype=np.int64)
    A0, b0 = A.copy(), b.copy()
    modp_solve.solve(p, A, b, block=4)
    assert np.array_equal(A, A0) and np.array_equal(b, b0)


def test_solve_chunking_is_irrelevant_to_the_answer():
    """The row-block size of the in-place fill is a memory knob only."""
    p = 65537
    rng = np.random.default_rng(13)
    A = rng.integers(0, p, size=(45, 20), dtype=np.int64)
    b = rng.integers(0, p, size=45, dtype=np.int64)
    ref = modp_solve.solve(p, A.copy(), b.copy(), block=8)
    for cb in (1, 64, 1 << 10, 1 << 30):
        assert modp_solve.solve(p, A.copy(), b.copy(), block=8,
                                chunk_bytes=cb) == ref
