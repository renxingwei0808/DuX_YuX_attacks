"""W14: r_KR = 2 in the encryption direction (CPA)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E10_cpa"))

from dux import DuX                                              # noqa: E402
from dux.linear import t_xor                                     # noqa: E402
from attack_2round_toy import monomial_value                     # noqa: E402
from kr2_cpa import (OUTER, joint_expansion_inv, linv_row,       # noqa: E402
                     make_precomp, rows_cpa, structure_moments,
                     _master_from_kappa)


def _neg(F, v):
    return int(v) if F.char == 2 else (-int(v)) % F.p


@pytest.mark.parametrize("instance,layers,active", [
    ("toy-2^4", 1, [1]),
    ("toy-2^4", 2, [1, 5]),
    ("toy-257", 3, [1]),
    ("dux-2^8", 3, [0]),
])
def test_equations_hold_at_the_true_key(instance, layers, active):
    """The residual of every assembled row at the true (kappa, kappa') is 0."""
    rounds = layers + 2
    c = DuX(instance, rounds=rounds)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    coords = [0, 3]
    pre = make_precomp(c, [0, 1, 2, 3], coords)
    lrow = linv_row(0)
    kappa = [_neg(F, v) for v in rks[rounds]]
    kappap = [_neg(F, v) for v in c.lin.L_inv(tuple(int(v) for v in rks[rounds - 1]), 0)]
    truth = np.array([monomial_value(F, m, kappa, kappap, F.q - 1)
                      for m in pre.mons], dtype=np.int64)
    for st in range(2):
        mom = structure_moments(c, rks, rounds, active, 4242 + st, pre)
        for row in rows_cpa(pre, mom, lrow, coords):
            if F.char == 2:
                res = int(F.vsum(F.vmul(row, truth)))
            else:
                res = int(np.dot(row % F.p, truth % F.p) % F.p)
            assert res == 0


def test_equations_hold_when_the_peeled_round_uses_L1():
    """O2 in the CPA r_KR=2 setting: the assembly fixes L0, but the peeled
    round may use L1 = Rot_{-4} o L0.  Then L1^{-1} = Rot_{+4} o L0^{-1}, so the
    row the code writes for outer block j is the TRUE equation of block j - 1
    with the key monomial (j, i) standing for kappa'_{4(j-1)+i}.  The inner
    unknowns kappa are untouched, which is why fixing L0 costs nothing."""
    rounds = 5
    c = DuX("toy-257", rounds=rounds)
    F = c.F
    # seed 2 is a key whose round r-1 selects L1
    rks = c.key_schedule(c.random_key(np.random.default_rng(2)))
    t = t_xor(rks[rounds - 1])
    assert t == 1
    coords = [0, 3]
    pre = make_precomp(c, [0, 1, 2, 3], coords)
    kappa = [_neg(F, v) for v in rks[rounds]]
    kp = [_neg(F, v) for v in
          c.lin.L_inv(tuple(int(v) for v in rks[rounds - 1]), t)]
    # shift the outer-block label by t blocks, per the derivation above
    kp_shift = [kp[4 * ((j - t) % 4) + i] for j in range(4) for i in range(4)]
    truth = np.array([monomial_value(F, m, kappa, kp_shift, F.q - 1)
                      for m in pre.mons], dtype=np.int64)
    for st in range(2):
        mom = structure_moments(c, rks, rounds, [1], 909 + st, pre)
        for row in rows_cpa(pre, mom, linv_row(0), coords):
            assert int(np.dot(row % F.p, truth % F.p) % F.p) == 0


def test_inner_sinv_expansion_is_much_smaller_than_the_forward_one():
    """The reason the CPA r_KR=2 system is an order of magnitude cheaper."""
    c = DuX("dux-65537", rounds=8)
    pre = make_precomp(c, [0, 1, 2, 3], [0, 3])
    assert len(pre.U) == 19                       # forward S: 75
    assert len(pre.mons) == 3307                  # forward S: 38 051
    c2 = DuX("dux-2^16", rounds=8)
    pre2 = make_precomp(c2, [0, 1, 2, 3], [0, 3])
    assert len(pre2.U) == 19
    assert len(pre2.mons) == 3259                 # forward S: 18 025


def test_outer_coordinate_tables_match_the_sbox():
    """OUTER names the quadratic pair and the linear position of S^{-1}'s two
    quadratic coordinates: f = x0 x1 + x3 + alpha and g = x1 x2 + x0 + alpha."""
    assert OUTER[3] == ((0, 1), 3)
    assert OUTER[0] == ((1, 2), 0)
    c = DuX("toy-257", rounds=5)
    j = joint_expansion_inv(c.F, c.alpha)
    # coordinate 3 has 7 terms, coordinate 0 has 7 -- both quadratic
    assert len(j[3]) == 7 and len(j[0]) == 7
    # the other two are the expensive ones
    assert len(j[1]) > 7 and len(j[2]) > len(j[1])


def test_end_to_end_master_key_on_toy_257():
    """3-layer CPA zero-sum + 2 rounds = 5 rounds, all four inner blocks
    unknown; the rank saturates at 1 392 and N_min = 1392/8 = 174."""
    rounds = 5
    c = DuX("toy-257", rounds=rounds)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    coords = [0, 3]
    pre = make_precomp(c, [0, 1, 2, 3], coords)
    lrow = linv_row(0)
    import modp_solve
    neq = 4 * len(coords)
    N = 174
    R = np.empty((N * neq, len(pre.mons)), dtype=np.int64)
    for s in range(N):
        mom = structure_moments(c, rks, rounds, [1], 2026 + 5000 + s, pre)
        R[s * neq:(s + 1) * neq] = np.asarray(rows_cpa(pre, mom, lrow, coords),
                                              dtype=np.int64)
    zero = (0, 0, 0, 0)
    known = {pre.col_const: 1}
    for m, i in pre.idx.items():
        tag, inner = m
        if tag is None and inner and all(e == zero for _b, e in inner):
            known[i] = 1
    kc = sorted(known)
    keep = [i for i in range(len(pre.mons)) if i not in known]
    kv = np.array([known[i] for i in kc], dtype=np.int64)
    B = (-(R[:, kc] @ kv)) % F.p
    det, rank, _ = modp_solve.solve(F.p, R[:, keep], B)
    assert rank == 1392
    cols = {pre.mons[i]: t for t, i in enumerate(keep)}
    got = {}
    for b in range(4):
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            got[f"{b},{i}"] = det[cols[(None, ((b, e),))]]
            assert got[f"{b},{i}"] == (-int(rks[rounds][4 * b + i])) % F.p
    assert _master_from_kappa(c, rounds, rks, got, 2026) == (True, True)
