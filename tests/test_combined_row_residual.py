"""R5 rule 4 / S10 step 4(b): the O10 combined row must vanish at the true key
on a WHOLE structure, and must be unusable on a truncated one.

Rule 4 of the R5 addendum asks for a residual-zero check at the true key before
any assembly.  S10 step 4(b) asks for the converse control: that the equation
really does need the whole product set.  Both are cheap on a toy and both are
locked here.

The truncated case is stronger than "the equation is false": `_scatter` refuses
to drop a nonzero coefficient outside the predicted monomial set, and on a
truncated structure exactly those coefficients (the ones the whole-structure
identities annihilate) are nonzero -- so the assembly RAISES instead of
silently producing a wrong row.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
sys.path.insert(0, ROOT)
sys.path.insert(0, E07)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))

from dux.registry import get_cipher                              # noqa: E402
from attack_2round_toy import linear_row, structure_data         # noqa: E402
from assemble_fast import Precomp, combine_rows, structure_rows  # noqa: E402
import weighted as WT                                            # noqa: E402


def _setup(instance, rounds, active, pattern):
    c = get_cipher(instance, rounds=rounds)
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    y = WT.combine_vectors("dux", c.F, "dec", pattern)[0]
    return c, rks, pre, y, WT.true_key_vector(c, pre, rks)


@pytest.mark.parametrize("instance,rounds,active,pattern", [
    ("toy-257", 8, "3,7", "0001"),      # the S10 algebra, toy-sized
    ("toy-193", 8, "3,7", "0001"),
])
def test_combined_row_vanishes_at_the_true_key(instance, rounds, active, pattern):
    c, rks, pre, y, truth = _setup(instance, rounds, active, pattern)
    P = structure_data(c, rks, rounds, 3, 7026, active)
    row = combine_rows(structure_rows(pre, P, rks[0], linear_row(c, 0)), y, c.F)
    residual = int(np.dot(np.asarray(row) % c.F.p, truth % c.F.p) % c.F.p)
    assert residual == 0


@pytest.mark.parametrize("limit", [1 << 15, 1 << 14])
def test_a_truncated_structure_cannot_even_be_assembled(limit):
    """S10 step 4(b): the equation is not merely false on a truncated structure
    -- the assembly refuses, because the coefficients the whole-structure
    identities kill are no longer zero."""
    c, rks, pre, _y, _truth = _setup("toy-257", 8, "3,7", "0001")
    P = structure_data(c, rks, 8, 3, 7026, "3,7", limit=limit)
    assert len(P[0]) == limit
    with pytest.raises(RuntimeError, match="outside the predicted monomial set"):
        structure_rows(pre, P, rks[0], linear_row(c, 0))
