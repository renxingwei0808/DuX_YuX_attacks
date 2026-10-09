"""End-to-end checks of the key-recovery scripts (E06, E07).

Fast instances only: DuX(2^8) with a 5-layer distinguisher (2^8 data per
structure) and one toy-257 structure for the r_KR = 2 equation.  The heavy
runs (DuX(65537), DuX(2^16), the full r_KR = 2 pipeline) are reproduced by the
commands in REPRODUCE.md.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E08_boolean_degree_extension"))

from dux import DuX                                          # noqa: E402
from dux.linear import t_xor                                 # noqa: E402
from attack_1round import attack_once                        # noqa: E402
from attack_1round_partial import equation_template          # noqa: E402
from attack_2round_toy import (monomial_set, structure_data,  # noqa: E402
                               linear_row)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_one_round_key_recovery_recovers_the_master_key(seed):
    """6-round DuX(2^8) from the 5-layer zero-sum: two structures suffice."""
    c = DuX("dux-2^8")
    rng = np.random.default_rng(seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    rec, used, why = attack_once(c, rks, 6, [3], None, rng, structures=2)
    assert rec == tuple(K), (rec, K, why)
    assert used <= 4


def test_partial_zero_sum_equation_degrees():
    """The equations the partial solver uses have the degrees claimed in
    for the partial solver: coordinate 0 -> F_q-degree 4, coordinate 3 -> 7."""
    from dux.field import make_field
    for name, sizes in (("2^8", {0: 14, 1: 5, 2: 3, 3: 37}),
                        ("65537", {0: 20, 1: 6, 2: 3, 3: 66})):
        F = make_field(name)
        for co, n in sizes.items():
            mons, _ = equation_template(F, 179, co)
            assert len(mons) == n, (name, co, len(mons))
        mons0, _ = equation_template(F, 179, 0)
        mons3, _ = equation_template(F, 179, 3)
        assert max(sum(m) for m in mons0) == 4
        assert max(sum(m) for m in mons3) == 7


def test_two_round_coordinate2_equation_holds():
    """The r_KR = 2 equation (*) is exactly zero on a real 7-round toy-257
    structure -- this is what the whole E07 linearisation rests on."""
    from dux.sbox import vS
    c = DuX("toy-257", rounds=7)
    F = c.F
    rng = np.random.default_rng(2026)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    P = structure_data(c, rks, 7, 3, 4242)
    row = linear_row(c, t_xor(rks[1]))
    Y = []
    for b in range(4):
        Y += list(vS(F, tuple(F.vadd(P[4 * b + i], rks[0][4 * b + i])
                              for i in range(4)), c.alpha))
    for j in range(4):
        u = []
        for i in range(4):
            acc = np.zeros(len(P[0]), dtype=np.int64)
            for l in range(16):
                if row[l]:
                    acc = F.vadd(acc, F.vmul(Y[(4 * j + i + l) % 16],
                                             np.full(len(P[0]), row[l], dtype=np.int64)))
            u.append(acc)
        s0, s2, s3 = F.vsum(u[0]), F.vsum(u[2]), F.vsum(u[3])
        s03 = F.vsum(F.vmul(u[0], u[3]))
        kp = [rks[1][4 * j + i] for i in range(4)]
        res = F.sub(F.sub(F.sub(s2, s03), F.mul(kp[3], s0)), F.mul(kp[0], s3))
        assert res == 0, (j, res)


def test_two_round_monomial_count():
    """M_actual for the full 4-block system is 38051 = 2^15.2 and is the same
    over F_257 and F_65537 (the exponents never wrap)."""
    from dux.field import make_field
    for name in ("257", "65537"):
        F = make_field(name)
        mons, _, U, S = monomial_set(F, 179 % F.q, [0, 1, 2, 3], [0, 1, 2, 3])
        assert (len(U), len(S)) == (75, 66)
        assert len(mons) == 38051
