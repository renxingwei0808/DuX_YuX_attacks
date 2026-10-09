"""The C blocked eliminator over F_{2^n} must agree with the numpy one."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round", "fast"))

from dux.field import make_field  # noqa: E402
import modp_solve  # noqa: E402
import gf2n_solve  # noqa: E402

pytestmark = pytest.mark.skipif(not gf2n_solve.available(),
                                reason="experiments/E07_key_recovery_2round/fast/gf2nsolve not built")


def _random_system(F, m, n, rank, rng):
    """A consistent system whose matrix has the prescribed rank."""
    U = rng.integers(0, F.q, size=(rank, n)).astype(np.int64)
    T = rng.integers(0, F.q, size=(m, rank)).astype(np.int64)
    A = np.zeros((m, n), dtype=np.int64)
    for k in range(rank):
        A ^= F.vmul(T[:, k:k + 1], U[k:k + 1, :])
    x = rng.integers(0, F.q, size=n).astype(np.int64)
    b = np.zeros(m, dtype=np.int64)
    for j in range(n):
        b ^= F.vmul(A[:, j], np.int64(x[j]))
    return A, b, x


@pytest.mark.parametrize("field,m,n,rank,block", [
    ("2^4", 40, 25, 18, 8),
    ("2^8", 60, 45, 30, 16),
    ("2^8", 50, 50, 50, 7),
    ("2^16", 70, 55, 40, 32),
])
def test_c_matches_numpy(field, m, n, rank, block):
    F = make_field(field)
    rng = np.random.default_rng(7 + m + n)
    A, b, x = _random_system(F, m, n, rank, rng)
    d1, r1, f1 = modp_solve.solve_gf2n(F, A.copy(), b.copy())
    d2, r2, f2 = gf2n_solve.solve(F, A.copy(), b.copy(), block=block)
    assert (r1, f1) == (r2, f2)
    assert d1 == d2
    # every determined column must carry the true value
    for c, v in d2.items():
        assert v == int(x[c])


def test_full_rank_square_system_is_solved_exactly():
    F = make_field("2^16")
    rng = np.random.default_rng(11)
    n = 60
    A = rng.integers(1, F.q, size=(n, n)).astype(np.int64)
    x = rng.integers(0, F.q, size=n).astype(np.int64)
    b = np.zeros(n, dtype=np.int64)
    for j in range(n):
        b ^= F.vmul(A[:, j], np.int64(x[j]))
    det, rank, nfree = gf2n_solve.solve(F, A, b, block=16)
    assert rank == n and nfree == 0
    assert [det[c] for c in range(n)] == [int(v) for v in x]
