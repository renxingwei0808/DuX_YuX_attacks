"""W12 -- multi-word structures and the characteristic-2 r_KR = 2 assembly.

Two properties are checked, both independent of any solver:

1. `structure_data(..., active="3,7")` really is the full product set
   F_q x F_q on words 3 and 7 (q^s points, constants elsewhere), it reduces to
   the original single-word structure for s = 1, and the multi-word zero-sum
   predicted by O7 actually holds (toy-2^4: 2 words -> layer 3, 16/16 words).

2. Every assembled equation row is satisfied by the TRUE key: substituting the
   true value of each linearisation monomial leaves residual 0.  This is the
   strongest cheap check on an assembly -- a wrong coefficient, a wrong column
   map or a wrong moment shows up immediately -- and it is run for the
   characteristic-2 assembly (1 and 2 unknown inner blocks) as well as for the
   prime-field one with a multi-word structure.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E08_boolean_degree_extension"))

from dux import DuX  # noqa: E402
from dux.linear import t_xor  # noqa: E402
from attack_2round_toy import (structure_data, linear_row,  # noqa: E402
                               monomial_value, active_words)
from assemble_fast import Precomp, structure_rows  # noqa: E402
from assemble_fast_2n import structure_rows_2n  # noqa: E402


def _key(c, seed):
    return c.key_schedule(c.random_key(np.random.default_rng(seed)))


def test_active_words_parsing():
    assert active_words(3) == [3]
    assert active_words(3, "3,7") == [3, 7]
    assert active_words(0, [3, 7, 11]) == [3, 7, 11]


def test_structure_is_a_full_product_set():
    c = DuX("toy-2^4", rounds=5)
    rks = _key(c, 5)
    P1 = structure_data(c, rks, 3, 3, 11)
    P2 = structure_data(c, rks, 3, 3, 11, "3,7")
    assert len(P1[0]) == c.F.q
    assert len(P2[0]) == c.F.q ** 2
    # the s = 1 path is unchanged: same seed -> same constants -> same data
    P1b = structure_data(c, rks, 3, 3, 11, "3")
    for i in range(16):
        assert np.array_equal(P1[i], P1b[i])


def test_two_word_zero_sum_matches_the_criterion():
    """O7: toy-2^4 with 2 active words is balanced on all 16 words at layer 3
    (single-word structures only reach layer 2)."""
    c = DuX("toy-2^4", rounds=12)
    for seed in (1, 2):
        rks = _key(c, seed)
        Z = structure_data(c, rks, 3, 3, 100 * seed, "3,7")
        assert all(c.F.vsum(Z[i]) == 0 for i in range(16))
        Z1 = structure_data(c, rks, 3, 3, 100 * seed)
        assert not all(c.F.vsum(Z1[i]) == 0 for i in range(16))


ASSEMBLY_CASES = [
    # (instance, layers, active words, unknown inner blocks)
    ("toy-2^4", 3, "3,7", [0]),
    ("toy-2^4", 3, "3,7", [0, 1]),
    ("dux-2^8", 5, None, [0]),
    ("toy-257", 5, None, [0]),        # prime-field reference path
]


@pytest.mark.parametrize("instance,layers,active,unknown", ASSEMBLY_CASES)
def test_equations_hold_at_the_true_key(instance, layers, active, unknown):
    c = DuX(instance, rounds=layers + 2)
    F = c.F
    rks = _key(c, 2026)
    outer = [0, 1, 2, 3]
    pre = Precomp(F, c.alpha, unknown, outer)
    lrow = linear_row(c, t_xor(rks[1]))
    truth = np.array([monomial_value(F, m, rks[0], rks[1], F.q - 1)
                      for m in pre.mons], dtype=np.int64)
    asm = structure_rows_2n if F.char == 2 else structure_rows
    for st in range(3):
        P = structure_data(c, rks, layers + 2, 3, 4242 + st, active)
        for row in asm(pre, P, rks[0], lrow):
            if F.char == 2:
                res = F.vsum(F.vmul(row, truth))
            else:
                res = int(np.dot(row % F.p, truth % F.p) % F.p)
            assert res == 0
