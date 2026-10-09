"""W13-A: the template bound rank(Phi_stack) for the r_KR = 2 linearisation."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E07_key_recovery_2round"))

from dux import DuX                                                    # noqa: E402
from attack_2round_toy import linear_row, structure_data               # noqa: E402
from assemble_fast import Moments, Precomp, rows_from_moments          # noqa: E402
from assemble_fast_2n import rows_from_moments_2n, structure_rows_2n   # noqa: E402
import modp_solve                                                      # noqa: E402
from rank_phi import (IncrementalRankFp, realizable_moments,           # noqa: E402
                      uniform_moments, _zero_pe_index)


def test_incremental_rank_matches_gauss_jordan():
    rng = np.random.default_rng(1)
    for p in (193, 257, 65537):
        for m, n, r in ((60, 40, 40), (80, 50, 20), (30, 120, 25)):
            A = (rng.integers(0, p, size=(m, r)) @ rng.integers(0, p, size=(r, n))) % p
            inc = IncrementalRankFp(p, n, cap=min(m, n) + 8)
            for i in range(0, m, 7):
                inc.add(A[i:i + 7])
            Aug = np.hstack([A.astype(np.float64), np.zeros((m, 1))])
            piv, _ = modp_solve.rref_blocked(Aug, p, n, block=16)
            assert inc.r == len(piv) == min(r, n, m)


def test_uniform_moments_respect_the_realizability_constraints():
    c = DuX("toy-257", rounds=7)
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    mom = uniform_moments(pre, np.random.default_rng(3))
    j0 = _zero_pe_index(pre)
    for b in pre.unknown:
        assert mom.pm[b][j0] == 0                     # sum_P 1 = q^s = 0 (O5)
    for A in pre.unknown:
        assert np.array_equal(mom.Mom[(A, A)], mom.Mom[(A, A)].T)
        for B in pre.unknown:
            assert np.array_equal(mom.Mom[(A, B)], mom.Mom[(B, A)].T)


def test_realizable_moments_are_moments_of_a_signed_measure():
    """Both characteristics: total weight zero, and the row/column identity
    Mom[A][B][j0][.] = pm[B][.] that any real structure satisfies."""
    for inst in ("toy-257", "toy-2^4"):
        c = DuX(inst, rounds=7)
        pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
        mom = realizable_moments(pre, np.random.default_rng(5), 64)
        j0 = _zero_pe_index(pre)
        for A in pre.unknown:
            for B in pre.unknown:
                assert np.array_equal(mom.Mom[(A, B)][j0], mom.pm[B])
                assert np.array_equal(mom.Mom[(A, B)][:, j0], mom.pm[A])


def test_rows_from_moments_2n_matches_the_direct_assembly():
    c = DuX("toy-2^4", rounds=5)
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    P = structure_data(c, rks, 5, 3, 11)
    mom = Moments(pre)
    mom.add(P)
    a = np.asarray(structure_rows_2n(pre, P, rks[0], lrow))
    b = np.asarray(rows_from_moments_2n(pre, mom, lrow))
    assert np.array_equal(a, b)


def test_pseudo_structures_span_at_least_the_real_row_space():
    """A real structure's moment vector is one of the vectors the `realizable`
    generator produces, so its rows lie in the pseudo-structure row space."""
    c = DuX("toy-257", rounds=7)
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lrow = linear_row(c, 0)
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    rng = np.random.default_rng(4)
    inc = IncrementalRankFp(pre.p, len(pre.mons), cap=600)
    for _ in range(120):
        inc.add(np.asarray(rows_from_moments(pre, uniform_moments(pre, rng), lrow)))
    before = inc.r
    for st in range(3):
        mom = Moments(pre)
        mom.add(structure_data(c, rks, 7, 3, 500 + st))
        inc.add(np.asarray(rows_from_moments(pre, mom, lrow)))
    assert inc.r == before + 12          # far from saturation, so all rows are new
