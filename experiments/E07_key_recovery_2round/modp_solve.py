"""Gauss-Jordan over F_p on large dense systems, for the S3 linearisation.

`attack_2round_toy.solve_modp_numpy` is fine for the 1-unknown-block prototype
(1082 columns) but its per-pivot `A[nzr] = (A[nzr] - col[nzr,None]*A[r]) % p`
allocates a fancy-indexed copy of the whole matrix.  For the 4-unknown-block
system (38 051 columns, ~9 000 pivots) that is ~25 GB of memory traffic per
pivot, i.e. many hours.

`rref_blocked` does the same elimination in panels of `block` columns:

  * inside a panel only the m x block sub-matrix is touched, and the composed
    row transformation is accumulated in an m x k matrix D with the invariant

        row_i(current) = row_i(original) + sum_j D[i][j] * row_{prow[j]}(original)

    (every elementary operation has column support on a single pivot row, so
    the product of them is I + D S with S the pivot-row selector);
  * the trailing columns are then updated with ONE matrix product
        trail += D[:, :k] @ trail[prow]
    which numpy hands to BLAS dgemm.

Exactness: all entries stay in [0, p); the panel update produces values below
p^2 + p and the trailing update below k * p^2 (2^41 for p = 65537, k = 256),
both far inside the 2^53 exactly-representable range of float64.

Returns ({column: value} for the columns the system determines uniquely,
rank, number of free columns) -- the same triple as solve_modp_numpy.
"""
from __future__ import annotations

import numpy as np


def rref_blocked(Aug, p, ncols, block=128, progress=None):
    """RREF of the augmented matrix Aug = [A | b] (modified in place).

    `ncols` is the number of columns of A (the last column of Aug is the
    right-hand side and is never a pivot).  Returns (pivot columns, pivot rows).
    """
    m, ntot = Aug.shape
    used = np.zeros(m, dtype=bool)
    piv_cols, piv_rows = [], []
    D = np.empty((m, block), dtype=np.float64)
    c0 = 0
    while c0 < ncols and len(piv_cols) < m:
        c1 = min(c0 + block, ncols)
        w = c1 - c0
        Aug[:, c0:c1] %= p          # the trailing update left them unreduced
        D[:, :w] = 0.0
        local_rows = []
        k = 0
        for c in range(c0, c1):
            col = Aug[:, c]
            cand = np.flatnonzero((col != 0) & (~used))
            if cand.size == 0:
                continue
            pr = int(cand[0])
            inv = pow(int(col[pr]) % p, p - 2, p)
            Aug[pr, c0:c1] *= inv
            Aug[pr, c0:c1] %= p
            D[pr, :w] *= inv
            D[pr, :w] %= p
            D[pr, k] = (D[pr, k] + inv - 1) % p
            f = Aug[:, c].copy()
            f[pr] = 0.0
            Aug[:, c0:c1] -= np.outer(f, Aug[pr, c0:c1])
            Aug[:, c0:c1] %= p
            D[:, :w] -= np.outer(f, D[pr, :w])
            D[:, k] -= f
            D[:, :w] %= p
            used[pr] = True
            local_rows.append(pr)
            piv_cols.append(c)
            piv_rows.append(pr)
            k += 1
        if k and c1 < ntot:
            # Reducing only the k extracted pivot rows keeps every increment
            # below k*p^2; over all panels the trailing entries stay under
            # (ncols/block)*k*p^2 < 2^48 for p = 65537, so the expensive
            # element-wise modulo of the whole trailing block is deferred to
            # the panel that next reads those columns (and to the end).
            X = Aug[np.array(local_rows), c1:] % p
            Aug[:, c1:] += D[:, :k] @ X
        c0 = c1
        if progress:
            print(f"    rref: {len(piv_cols)} pivots after column {c1}/{ncols}",
                  flush=True)
    Aug %= p
    return piv_cols, piv_rows


def solve(p, rows, rhs, block=128, progress=None, chunk_bytes=1 << 27):
    # S9 (1b): the augmented matrix is filled IN PLACE.  The old entry point
    #   A = np.asarray(rows, float64) ; hstack([A, b]) ; % p
    # built three full-size float64 temporaries on top of the caller's int64
    # `rows`, so the peak was ~4.9x (rows x cols x 8 B) -- measured 13.13 GB on
    # the 8 800 x 38 040 F_193 system of S9 step 1b, which is what OOM-killed
    # the same run inside a 15 GB container.  Casting and reducing
    # row-block by row-block into one preallocated buffer leaves only the
    # caller's `rows` plus `Aug`, i.e. ~2.0x, which is the floor for an entry
    # point that may not free its argument.
    rows = np.asarray(rows)
    m, n = rows.shape
    Aug = np.empty((m, n + 1), dtype=np.float64)
    step = max(1, chunk_bytes // max(1, 8 * n))
    for r0 in range(0, m, step):
        r1 = min(r0 + step, m)
        Aug[r0:r1, :n] = rows[r0:r1]          # int64 -> float64, no temporary
        Aug[r0:r1, :n] %= p
    Aug[:, n] = np.asarray(rhs).reshape(-1)
    Aug[:, n] %= p
    piv_cols, piv_rows = rref_blocked(Aug, p, n, block, progress)
    free = np.ones(n, dtype=bool)
    free[np.array(piv_cols, dtype=np.int64)] = False
    det = {}
    for ci, ri in zip(piv_cols, piv_rows):
        if not Aug[ri, :n][free].any():
            det[ci] = int(Aug[ri, n]) % p
    return det, len(piv_cols), int(free.sum())


# ------------------------------------------------------- characteristic 2 --
def solve_gf2n(F, rows, rhs, progress=None):
    """Gauss-Jordan over F_{2^n}; returns ({column: value}, rank, #free).

    There is no BLAS for F_{2^n}, so this is the plain per-pivot rank-1 update
    with two savings that matter at the W12 scale (a few thousand columns):

      * only the columns from the pivot column on are touched (the ones to the
        left are already zero in every non-pivot row), and
      * the update is one broadcast `F.vmul` (a pair of table look-ups) plus an
        XOR over the whole remaining sub-matrix, not a Python loop over rows.

    Addition is XOR, so there is no modulus to defer and no exactness question:
    every intermediate value is an element of the field.
    """
    A = np.asarray(rows, dtype=np.int64)
    b = np.asarray(rhs, dtype=np.int64).reshape(-1, 1)
    Aug = np.ascontiguousarray(np.hstack([A, b]))
    del A, b
    m, ntot = Aug.shape
    n = ntot - 1
    exp, log, order = F._exp, F._log, F.order
    piv_cols, piv_rows = [], []
    r = 0
    for c in range(n):
        nz = np.flatnonzero(Aug[r:, c])
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            Aug[[r, pr]] = Aug[[pr, r]]
        inv = F.inv(int(Aug[r, c]))
        if inv != 1:
            Aug[r, c:] = F.vmul(Aug[r, c:], np.int64(inv))
        col = Aug[:, c].copy()
        col[r] = 0
        if col.any():
            # One contiguous slice update instead of a fancy-indexed copy:
            # rows with col == 0 are masked to a no-op, so the whole trailing
            # sub-matrix is touched exactly once, in place.
            piv = Aug[r, c:]
            lr = log[piv]
            mr = piv != 0
            lc = log[col]
            mc = col != 0
            prod = exp[lc[:, None] + lr[None, :]]
            prod &= -(mc[:, None] & mr[None, :]).astype(np.int64)
            Aug[:, c:] ^= prod
        piv_cols.append(c)
        piv_rows.append(r)
        r += 1
        if progress and r % progress == 0:
            print(f"    rref(2^n): {r} pivots after column {c}/{n}", flush=True)
        if r == m:
            break
    free = np.ones(n, dtype=bool)
    if piv_cols:
        free[np.array(piv_cols, dtype=np.int64)] = False
    det = {}
    for ci, ri in zip(piv_cols, piv_rows):
        if not Aug[ri, :n][free].any():
            det[ci] = int(Aug[ri, n])
    return det, len(piv_cols), int(free.sum())
