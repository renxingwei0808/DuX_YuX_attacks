"""The streamed r_KR = 2 assembly must equal the in-memory one row by row.

The 12-round attack walks a 2^48-point structure in chunks, so the equations
are built from accumulated point sums (assemble_fast.Moments) instead of from
the (nP x N) power matrix.  The two paths must agree exactly.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))

from dux import DuX  # noqa: E402
from attack_2round_toy import (structure_data, structure_stream,  # noqa: E402
                               linear_row)
from assemble_fast import (Precomp, Moments, structure_rows,  # noqa: E402
                           rows_from_moments)


@pytest.mark.parametrize("instance,layers,active,chunk", [
    ("toy-257", 5, "3", 90),
    ("toy-257", 5, "3,7", 7000),
    ("toy-193", 5, "3,7", 5000),
])
def test_streaming_matches_direct(instance, layers, active, chunk):
    rounds = layers + 2
    c = DuX(instance, rounds=rounds)
    rng = np.random.default_rng(3)
    rks = c.key_schedule(c.random_key(rng))
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    P = structure_data(c, rks, rounds, 3, 55, active)
    direct = structure_rows(pre, P, rks[0], lrow)
    mom = Moments(pre)
    nchunks = 0
    for chunkP in structure_stream(c, rks, rounds, 3, 55, active, chunk=chunk):
        mom.add(chunkP)
        nchunks += 1
    assert nchunks > 1, "the test must actually exercise more than one chunk"
    assert mom.npoints == len(P[0])
    streamed = rows_from_moments(pre, mom, lrow)
    for a, b in zip(direct, streamed):
        assert np.array_equal(a, b)


def test_limit_points_is_a_prefix_of_the_full_structure():
    c = DuX("toy-193", rounds=7)
    rng = np.random.default_rng(4)
    rks = c.key_schedule(c.random_key(rng))
    full = structure_data(c, rks, 7, 3, 9, "3,7")
    part = structure_data(c, rks, 7, 3, 9, "3,7", limit=1000)
    for i in range(16):
        assert np.array_equal(full[i][:1000], part[i])
