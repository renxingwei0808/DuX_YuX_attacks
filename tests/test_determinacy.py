"""W10 -- Lemma O8: which key words each r_KR = 1 coordinate can determine,
and the rank prediction for the linearised system.

The determinacy table is derived twice (sympy over Z[alpha] in determinacy.py,
exact field arithmetic in keypoly.py) and `determinacy_table` asserts the two
agree, so these tests pin the *content* of the table: which coordinates see
k1, and how.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E08_boolean_degree_extension"))

import determinacy as D  # noqa: E402
from dux import DuX  # noqa: E402


@pytest.mark.parametrize("field", ["2^8", "2^16", "65537", "257"])
def test_o8_table(field):
    """O8: coordinate 2 has no k1 at all; coordinate 1 keeps it only as
    |structure| * k1 (killed by the sum); coordinate 0 keeps it only through
    -k1 * sum_P a(P+k) (so it vanishes exactly when W position 2 is balanced);
    coordinate 3 is the only one that always sees k1."""
    t = D.determinacy_table(field, 179)
    assert t[2]["k1"]["monomials_before_sum"] == []          # a has no k1
    assert t[1]["k1"]["monomials_before_sum"] != []          # b has k1 ...
    assert t[1]["k1"]["monomials_after_sum"] == []           # ... only as |P| k1
    # coordinate 0: k1 survives, and exactly as the k-monomials of k1 * a
    assert sorted(map(tuple, t[0]["k1"]["monomials_after_sum"])) == [
        (0, 1, 0, 0), (0, 1, 0, 1), (1, 1, 0, 0)]
    assert t[3]["k1"]["monomials_after_sum"] != []
    # which words each coordinate exposes as a degree-1 monomial
    solo = {co: sorted(w for w, moms in t[co]["linear"].items() if moms)
            for co in range(4)}
    assert solo[2] == ["k0", "k3"]
    assert solo[1] == ["k0", "k2", "k3"]
    assert solo[0] == ["k0", "k1", "k2", "k3"]
    assert solo[3] == ["k0", "k1", "k2", "k3"]


@pytest.mark.parametrize("field,after", [("2^8", (14, 5, 3, 37)),
                                         ("2^16", (14, 5, 3, 37)),
                                         ("65537", (20, 6, 3, 66)),
                                         ("257", (20, 6, 3, 66))])
def test_monomial_counts(field, after):
    """The linearisation sizes quoted in the E06 records (results/E06_key_recovery_1round/)."""
    t = D.determinacy_table(field, 179)
    assert tuple(t[co]["monomials_after_sum"] for co in range(4)) == after


def test_rank_grid_reproduced():
    ok, rows = D.run_grid()
    assert ok, [r for r in rows if not r["match"]]


@pytest.mark.parametrize("seed", [2026, 7, 99])
def test_rank_prediction_is_key_independent(seed):
    """The predicted rank comes from the template and the identity vectors of
    the balanced coordinates; it should not depend on which key we pick."""
    c = DuX("dux-2^8", rounds=6)
    K = c.random_key(np.random.default_rng(seed))
    pr = D.predict_rank(c.F, c.alpha, [0, 1, 2], [0, 1, 2, 3], list(K[0:4]))
    assert pr["predicted_rank"] == 10


def test_sequential_recovers_the_1111_case():
    """The `1111` failure observed in E06 is exactly O8: coordinates 0,1,2
    cannot see k1.  Substituting through coordinate 3 recovers the whole key
    from 3 structures."""
    from attack_1round_partial import (equation_template, sequential_solve,
                                       structure_moments)
    c = DuX("dux-2^8", rounds=6)
    F = c.F
    K = c.random_key(np.random.default_rng(2026))
    rks = c.key_schedule(K)
    coords = [1, 2, 3]
    tmpl = {co: equation_template(F, c.alpha, co) for co in coords}
    needed = sorted({pe for co in coords for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    seq = {j: {} for j in range(4)}
    for st in range(3):
        moments, _ = structure_moments(c, rks, 6, [3], F.n, 1000 + st, needed)
        for j in range(4):
            for co in coords:
                mons, terms = tmpl[co]
                row = {}
                for m in mons:
                    v = 0
                    for pe, coeff in terms.get(m, []):
                        v = F.add(v, F.mul(coeff, moments[j][pe]))
                    if v:
                        row[m] = v
                seq[j].setdefault(co, []).append(row)
    rec = []
    for j in range(4):
        kj, _ = sequential_solve(F, seq[j])
        rec += kj
    assert tuple(rec) == tuple(K)
