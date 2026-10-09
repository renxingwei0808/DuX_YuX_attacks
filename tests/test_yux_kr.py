"""W18 (Y05) -- YuX key recovery: the equations, the chain, and the toys.

  * `joint_expansion(..., cipher="yux")` really is S(P + k) point by point;
  * the r_KR = 2 rows vanish at the true key of the equivalent fixed-L0 cipher,
    plain and O10-combined, plain and O12-weighted;
  * the r_KR = 1 sequential chain of `recover_block_yux` recovers the master
    key of the toys, and the YuX degeneracy it works around is real: without a
    weight, position 3's two coefficients are both zero.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E04_zero_sum_F2n"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux.registry import get_cipher                              # noqa: E402
from yux.sbox import S as S_yux                                  # noqa: E402
from keypoly import KeyPoly, sbox_enc_chain_yux                  # noqa: E402
from attack_2round_toy import joint_expansion, linear_row        # noqa: E402
from assemble_fast import OUTER_EQUATIONS, Precomp               # noqa: E402
import weighted as WT                                            # noqa: E402
import attack_1round as A1                                       # noqa: E402


def _residual(F, rows, truth):
    rows = np.asarray(rows, dtype=np.int64)
    if F.char == 2:
        out = np.zeros(rows.shape[0], dtype=np.int64)
        for j in range(rows.shape[1]):
            if truth[j]:
                out ^= F.vmul(rows[:, j], np.int64(truth[j]))
        return out
    return (rows.astype(np.float64) @ truth.astype(np.float64)) % F.p


# --------------------------------------------------------- the expansion --
@pytest.mark.parametrize("inst", ["yuxtoy-257", "yuxtoy-193", "yuxtoy-2^4"])
def test_joint_expansion_matches_the_numeric_sbox(inst):
    c = get_cipher(inst)
    F = c.F
    joint = joint_expansion(F, c.alpha, "yux")
    rng = np.random.default_rng(2026)
    for _ in range(30):
        P = [int(v) for v in rng.integers(0, F.q, size=4)]
        K = [int(v) for v in rng.integers(0, F.q, size=4)]
        want = S_yux(F, tuple(F.add(P[i], K[i]) for i in range(4)), c.alpha)
        for co in range(4):
            acc = 0
            for ke, pe, coeff in joint[co]:
                t = coeff
                for i, e in enumerate(ke):
                    for _r in range(e):
                        t = F.mul(t, K[i])
                for i, e in enumerate(pe):
                    for _r in range(e):
                        t = F.mul(t, P[i])
                acc = F.add(acc, t)
            assert acc == want[co], f"coordinate {co}"


def test_outer_equation_table():
    """DuX's cheap coordinate is a = u2 - u0 u3 - alpha at position 2, YuX's is
    z0 = u3 - u0 u1 - u2 - alpha at position 3, so the row combinations and the
    outer key words differ."""
    d, y = OUTER_EQUATIONS["dux"], OUTER_EQUATIONS["yux"]
    assert (d.w_offsets, d.a_off, d.b_off, d.kA, d.kB) == (((2, 1),), 0, 3, 3, 0)
    assert (y.w_offsets, y.a_off, y.b_off, y.kA, y.kB) == \
        (((3, 1), (2, -1)), 0, 1, 1, 0)
    assert d.outer_coords == (0, 3) and y.outer_coords == (0, 1)


@pytest.mark.parametrize("inst,exp", [("yuxtoy-257", (90, 80, 53857)),
                                      ("yuxtoy-193", (90, 80, 53857)),
                                      ("yuxtoy-2^4", (63, 50, 27895))])
def test_monomial_counts(inst, exp):
    """|U|, |S| and M for four unknown inner blocks (memo Sect. 6: 90/80 over
    F_p and 63/50 in characteristic 2, against DuX's 75/66 and 50/40)."""
    c = get_cipher(inst)
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="yux")
    assert (len(pre.U), len(pre.S), len(pre.mons)) == exp


# ------------------------------------------------------------ r_KR = 2 ----
@pytest.mark.parametrize("inst,rounds,layers", [("yuxtoy-257", 6, 4),
                                                ("yuxtoy-193", 6, 4)])
def test_rKR2_rows_vanish_at_the_true_key(inst, rounds, layers):
    c = get_cipher(inst, rounds=rounds)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="yux")
    truth = WT.true_key_vector(c, pre, rks)
    lrow = linear_row(c, 0)
    margin, crit = WT.weight_margin(F, [0], layers, "dec", "yux")
    assert crit["patterns"][layers - 1] == "1111" and margin > 0
    ws = [(a,) for a in range(min(8, margin))]
    rows, _o, _n = WT.weighted_rows(pre, c, rks, rounds, 0, 7000, None, ws, lrow)
    assert rows.shape[0] == len(ws) * 4
    assert int(np.abs(_residual(F, rows, truth)).max()) == 0
    # and a weight past the margin does not vanish
    bad, _o, _n = WT.weighted_rows(pre, c, rks, rounds, 0, 7000, None,
                                   [(margin,), (margin + 1,)], lrow)
    assert int(np.abs(_residual(F, bad, truth)).max()) != 0


def test_rKR2_combined_rows_on_a_full_block_pattern():
    """yuxtoy-2^4, full block 0 (2^16 chosen ciphertexts): layer 4 is `1110`,
    which is the pattern O10 turns into THREE combined equations.

    The measurement also shows an identity O7 does not predict: the four
    position-3 sums of the layer-4 state are (5,7,7,5) and their XOR is 0, so
    the BLOCK SUM of the unbalanced position vanishes too and the four
    per-outer-block rows are usable as they stand.  That is a measured bonus,
    not something the criterion guarantees, so the attack is built on the three
    combined rows and this test records both."""
    c = get_cipher("yuxtoy-2^4", rounds=6)
    F = c.F
    rks = c.key_schedule(c.random_key(np.random.default_rng(2026)))
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="yux")
    truth = WT.true_key_vector(c, pre, rks)
    lrow = linear_row(c, 0)
    y = WT.combine_vectors("yux", F, "dec", "1110")
    assert len(y) == 3
    # the weight rides on the cheapest ciphertext coordinate (position 3), so
    # the structure lists word 3 first -- the same point set, different axis order
    plain, _o, N = WT.weighted_rows(pre, c, rks, 6, 3, 7000, "3,0,1,2",
                                    [(0,)], lrow)
    assert N == F.q ** 4
    assert int(np.abs(_residual(F, plain, truth)).max()) == 0   # the bonus
    comb, _o, _n = WT.weighted_rows(pre, c, rks, 6, 3, 7000, "3,0,1,2",
                                    [(0,), (1,), (2,)], lrow, ycomb=y)
    assert comb.shape[0] == 9
    assert int(np.abs(_residual(F, comb, truth)).max()) == 0
    # the layer-4 state really is `1110` with a vanishing position-3 block sum
    from attack_2round_toy import structure_data
    P = structure_data(c, rks, 6, 3, 7000, "3,0,1,2")
    rng = np.random.default_rng(7000)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    idx = np.arange(F.q ** 4, dtype=np.int64)
    act = [3, 0, 1, 2]
    C = tuple((idx // (F.q ** (3 - act.index(i)))) % F.q if i in act
              else np.full(len(idx), consts[i], dtype=np.int64) for i in range(16))
    Z = c.decrypt_layers(C, rks, 4, rounds=6, vec=True)
    sums = [int(np.bitwise_xor.reduce(Z[i])) for i in range(16)]
    assert all(sums[4 * b + p] == 0 for b in range(4) for p in range(3))
    assert any(sums[4 * b + 3] for b in range(4))
    assert sums[3] ^ sums[7] ^ sums[11] ^ sums[15] == 0


# ------------------------------------------------------------ r_KR = 1 ----
def test_yux_rKR1_degeneracy_is_real():
    """Without a weight, YuX's position-3 equation has BOTH coefficients zero:
    they are sum_P P1 and sum_P P0, and the layer-(l+1) pattern is `1100`."""
    c = get_cipher("yuxtoy-257", rounds=5)
    F = c.F
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    P, xs = A1.structure_plaintexts(c, rks, 5, [0], rng)
    for j in range(4):
        Pb = tuple(P[4 * j + i] for i in range(4))
        f00 = A1.coord_sum(c, Pb, (0, 0, 0, 0), 3)
        assert F.sub(A1.coord_sum(c, Pb, (1, 0, 0, 0), 3), f00) == 0
        assert F.sub(A1.coord_sum(c, Pb, (0, 1, 0, 0), 3), f00) == 0
    k, ok, why = A1.recover_block(c, [(P, None)], 0)
    assert k is None and "k0,k1" in why


@pytest.mark.parametrize("inst,rounds,nstruct", [("yuxtoy-257", 5, 2),
                                                 ("yuxtoy-193", 5, 2),
                                                 ("yuxtoy-2^4", 3, 3),
                                                 ("yu2x-8", 5, 2)])
def test_yux_rKR1_recovers_the_master_key(inst, rounds, nstruct):
    c = get_cipher(inst)
    rng = np.random.default_rng(2026)
    weights, lo, hi = A1.auto_weights(c, [0], rounds)
    for _ in range(3):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        rec, used, why = A1.attack_once(c, rks, rounds, [0], None, rng, nstruct,
                                        weights=weights, lo=lo, hi=hi)
        assert rec == tuple(K), (inst, why)


def test_dux_rKR1_still_needs_no_weight():
    """Regression: DuX's layer-(l+1) pattern is `0000` at position 3, so the
    plain O5 attack is unchanged."""
    c = get_cipher("dux-2^8")
    rng = np.random.default_rng(2026)
    for _ in range(3):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        rec, used, why = A1.attack_once(c, rks, 6, [3], None, rng, 2)
        assert rec == tuple(K), why


# ------------------------------------------- the L-combined partial route --
sys.path.insert(0, os.path.join(ROOT, "experiments", "Y05_key_recovery"))
import partial_lcomb as PL                                       # noqa: E402


def test_L_inv_residues_explain_why_yux_needs_the_forward_form():
    """DuX's L^{-1} offsets hit only residues {0,1} mod 4, so a partial pattern
    still leaves whole balanced W words; YuX's hit all four, so it does not."""
    from cipher_degree import CIPHERS
    assert {o % 4 for o in CIPHERS["dux"]["ROT_INV"]} == {0, 1}
    assert {o % 4 for o in CIPHERS["yux"]["ROT_INV"]} == {0, 1, 2, 3}


def test_linear_rows_reproduce_the_forward_layer():
    c = get_cipher("yuxtoy-257")
    F, rows = c.F, PL.linear_rows(c)
    rng = np.random.default_rng(11)
    x = tuple(int(v) for v in rng.integers(0, F.q, size=16))
    ref = c.lin.L(x)
    for i in range(16):
        acc = 0
        for j in range(16):
            acc = F.add(acc, F.mul(rows[i][j], x[j]))
        assert acc == ref[i]


@pytest.mark.parametrize("inst,rounds,pat", [("yuxtoy-257", 6, "1100"),
                                             ("yuxtoy-2^4", 4, "1100")])
def test_L_combined_rows_vanish_at_the_true_key(inst, rounds, pat):
    """Every balanced word gives sum_P [L(W)]_i = 0, so the row built from the
    structure's moments has residual 0 on the true key's monomial vector."""
    c = get_cipher(inst)
    F, fam = c.F, "yux"
    marg, pred = PL.margins(F, [0], rounds - 1, fam)
    assert pred == pat
    pos = [p for p in range(4) if pat[p] == "1"]
    words = [4 * b + p for b in range(4) for p in pos]
    weights = list(range(min(marg[p] for p in pos)))
    tmpl = {co: PL.equation_template(F, c.alpha, co, fam) for co in range(4)}
    mons = sorted({m for co in range(4) for m in tmpl[co][0]})
    needed = sorted({pe for co in range(4) for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    P, xs = PL.structure(c, rks, rounds, [0], rng)
    mom = PL.moments(c, P, xs, needed, weights)
    rows = PL.build_rows(c, mom, tmpl, mons, PL.linear_rows(c), words, weights)
    # the true key vector: monomial m = (e0..e3) evaluates to prod k_i^{e_i}
    truth = np.zeros(4 * len(mons), dtype=np.int64)
    for b in range(4):
        k = [int(rks[0][4 * b + i]) for i in range(4)]
        for t, m in enumerate(mons):
            v = 1
            for i in range(4):
                for _ in range(m[i]):
                    v = F.mul(v, k[i])
            truth[b * len(mons) + t] = v
    assert not np.any(_residual(F, rows, truth))


def test_L_combined_route_recovers_the_master_key():
    """yuxtoy-257, 6 rounds = layer-5 `1100` + L-combined + 48 weights,
    ONE structure of 2^8 chosen ciphertexts."""
    c = get_cipher("yuxtoy-257")
    F, fam = c.F, "yux"
    marg, pat = PL.margins(F, [0], 5, fam)
    pos = [p for p in range(4) if pat[p] == "1"]
    words = [4 * b + p for b in range(4) for p in pos]
    weights = list(range(min(marg[p] for p in pos)))
    tmpl = {co: PL.equation_template(F, c.alpha, co, fam) for co in range(4)}
    mons = sorted({m for co in range(4) for m in tmpl[co][0]})
    needed = sorted({pe for co in range(4) for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    rng = np.random.default_rng(2026)
    rks = c.key_schedule(c.random_key(rng))
    P, xs = PL.structure(c, rks, 6, [0], rng)
    rows = PL.build_rows(c, PL.moments(c, P, xs, needed, weights), tmpl, mons,
                         PL.linear_rows(c), words, weights)
    got, det, _rank, _ = PL.solve(F, rows, mons)
    assert det == 16
    for b in range(4):
        for i in range(4):
            assert got[(b, i)] == int(rks[0][4 * b + i])


def test_weights_beyond_the_margin_break_the_equations():
    """The tightness check: yuxtoy-2^4's layer-3 `1100` allows a = 0 only, and
    a = 1 already leaves a nonzero residual at the true key."""
    c = get_cipher("yuxtoy-2^4")
    F, fam = c.F, "yux"
    marg, pat = PL.margins(F, [0], 3, fam)
    pos = [p for p in range(4) if pat[p] == "1"]
    assert min(marg[p] for p in pos) == 1          # only a = 0 is admissible
    words = [4 * b + p for b in range(4) for p in pos]
    tmpl = {co: PL.equation_template(F, c.alpha, co, fam) for co in range(4)}
    mons = sorted({m for co in range(4) for m in tmpl[co][0]})
    needed = sorted({pe for co in range(4) for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    rng = np.random.default_rng(5)
    rks = c.key_schedule(c.random_key(rng))
    P, xs = PL.structure(c, rks, 4, [0], rng)
    mom = PL.moments(c, P, xs, needed, [0, 1, 2, 3])
    truth = np.zeros(4 * len(mons), dtype=np.int64)
    for b in range(4):
        k = [int(rks[0][4 * b + i]) for i in range(4)]
        for t, m in enumerate(mons):
            v = 1
            for i in range(4):
                for _ in range(m[i]):
                    v = F.mul(v, k[i])
            truth[b * len(mons) + t] = v
    Lr = PL.linear_rows(c)
    ok = PL.build_rows(c, mom, tmpl, mons, Lr, words, [0])
    assert not np.any(_residual(F, ok, truth))
    bad = PL.build_rows(c, mom, tmpl, mons, Lr, words, [1, 2, 3])
    assert np.any(_residual(F, bad, truth))
