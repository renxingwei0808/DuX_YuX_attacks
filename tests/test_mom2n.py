"""The C moment kernel must agree with the numpy path element for element.

`assemble_fast_2n.moment_matrix_2n` has two implementations: a numpy one and
`fast/libmom2n.so` (loaded through ctypes when it has been built).  Both are
compared here against the definition written directly with `F.vmul`, so the
test is meaningful whether or not the library is present.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))

from dux.field import make_field  # noqa: E402
import assemble_fast_2n as A2  # noqa: E402


def _reference(F, PW, A, B):
    a, b = PW[A], PW[B]
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)
    for i in range(a.shape[0]):
        out[i] = np.bitwise_xor.reduce(F.vmul(b, a[i][None, :]), axis=1)
    return out


@pytest.mark.parametrize("field,nP,N", [("2^4", 9, 300), ("2^8", 12, 1000),
                                        ("2^16", 11, 2048)])
def test_moment_matrix_matches_vmul(field, nP, N):
    F = make_field(field)
    rng = np.random.default_rng(3 + nP)
    PW = {0: rng.integers(0, F.q, size=(nP, N)).astype(np.int64),
          1: rng.integers(0, F.q, size=(nP, N)).astype(np.int64)}
    PW[0][2, :17] = 0            # exercise the zero-operand sentinel
    PW[1][5, :23] = 0
    assert np.array_equal(A2.moment_matrix_2n(F, PW, 0, 1), _reference(F, PW, 0, 1))
    y = rng.integers(0, F.q, size=N).astype(np.int64)
    y[:13] = 0
    ref = np.bitwise_xor.reduce(F.vmul(PW[0], y[None, :]), axis=1)
    assert np.array_equal(A2.moment_vector_2n(F, PW[0], y), ref)


def test_the_two_implementations_agree():
    """Force the numpy path and compare it with whatever `_lib()` returns."""
    F = make_field("2^8")
    rng = np.random.default_rng(99)
    PW = {0: rng.integers(0, F.q, size=(10, 777)).astype(np.int64),
          1: rng.integers(0, F.q, size=(10, 777)).astype(np.int64)}
    with_lib = A2.moment_matrix_2n(F, PW, 0, 1)
    saved_lib, saved_tried = A2._LIB, A2._LIB_TRIED
    try:
        A2._LIB, A2._LIB_TRIED = None, True
        without_lib = A2.moment_matrix_2n(F, PW, 0, 1)
    finally:
        A2._LIB, A2._LIB_TRIED = saved_lib, saved_tried
    assert np.array_equal(with_lib, without_lib)
