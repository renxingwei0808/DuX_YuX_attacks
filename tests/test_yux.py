"""W15-A (Y01) -- the YuX reference model against the paper's definitions.

Specification source: Liu et al., IEEE TIT 70(5), 2024 -- Sect. III
(Construction 1, Fig. 1), Sect. IV-A/B (Algorithms 1-3, Table II, footnote 2).
Every test below names the clause it locks; the full clause table is
docs/yux_specification.md.
"""
import glob
import json
import os

import numpy as np
import pytest

from dux.field import make_field
from dux.registry import cipher_family, get_cipher, instances
from yux import YuX
from yux.linear import LinearLayer, circulant, forward_row, paper_v_p
from yux.params import (INSTANCES, ROT_FWD_BIN, ROT_INV, V_P_65537, WORDS)
from yux.sbox import (Pf, Pf_inv, S, S_closed, S_inv, S_inv_closed,
                      paper_VI_D_system, vS, vS_inv)
from yux.keyschedule import (key_expand, ks_round, ks_round_inverse,
                             round_constants, rot3, seed_state)

ALL = sorted(INSTANCES)
OFFICIAL = [k for k, v in INSTANCES.items() if v["official"]]
FIELDS3 = ["2^8", "2^16", "65537"]
FIELDS_ALL = FIELDS3 + ["2^4", "193", "257"]
VEC_FILES = sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                          "vectors", "yux_*.json")))


def _rand(F, rng, n=4):
    return tuple(int(v) for v in rng.integers(0, F.q, size=n))


# --------------------------------------------------------------- S-box ----
@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_Pf_inv_is_the_inverse_of_Pf(fname):
    """Construction 1 / Fig. 1: Pf^{-1} o Pf = Pf o Pf^{-1} = id."""
    F = make_field(fname)
    rng = np.random.default_rng(2026)
    alpha = 205 % F.q if F.char != 2 else 205 % F.q
    for _ in range(200):
        x = _rand(F, rng)
        assert Pf_inv(F, Pf(F, x, alpha), alpha) == x
        assert Pf(F, Pf_inv(F, x, alpha), alpha) == x


@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_S_inv_closed_form_equals_Pf4(fname):
    """S^{-1} = Pf^4 with the closed form of yux/sbox.py (degrees 2,2,3,4)."""
    F = make_field(fname)
    rng = np.random.default_rng(7)
    alpha = 205 % F.q
    for _ in range(200):
        x = _rand(F, rng)
        assert S_inv_closed(F, x, alpha) == S_inv(F, x, alpha)


@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_S_closed_form_equals_Pf_minus_4(fname):
    """S = Pf^{-4} with the closed form of yux/sbox.py (degrees 8,5,3,2)."""
    F = make_field(fname)
    rng = np.random.default_rng(11)
    alpha = 205 % F.q
    for _ in range(200):
        x = _rand(F, rng)
        assert S_closed(F, x, alpha) == S(F, x, alpha)


@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_S_and_S_inv_are_mutually_inverse(fname):
    F = make_field(fname)
    rng = np.random.default_rng(13)
    alpha = 205 % F.q
    for _ in range(200):
        x = _rand(F, rng)
        assert S(F, S_inv(F, x, alpha), alpha) == x
        assert S_inv(F, S(F, x, alpha), alpha) == x


@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_vectorised_sbox_matches_scalar(fname):
    F = make_field(fname)
    rng = np.random.default_rng(17)
    alpha = 205 % F.q
    xs = [_rand(F, rng) for _ in range(64)]
    X = tuple(np.array([x[i] for x in xs], dtype=np.int64) for i in range(4))
    Y, Yi = vS(F, X, alpha), vS_inv(F, X, alpha)
    for k, x in enumerate(xs):
        assert tuple(int(w[k]) for w in Y) == S(F, x, alpha)
        assert tuple(int(w[k]) for w in Yi) == S_inv(F, x, alpha)


def test_paper_VI_D_system_is_a_sign_typo_over_Fp():
    """Erratum E1 (docs/yux_specification.md).  The quadratic system printed
    in Sect. VI-D equals S = Pf^{-4} in characteristic 2 and does NOT over F_p;
    the reference model follows Construction 1, so this test pins the
    discrepancy rather than the printed system."""
    rng = np.random.default_rng(23)
    for fname in ("2^8", "2^16", "2^4"):
        F = make_field(fname)
        for _ in range(100):
            x = _rand(F, rng)
            assert paper_VI_D_system(F, x, 205 % F.q) == S(F, x, 205 % F.q)
    for fname in ("65537", "257", "193"):
        F = make_field(fname)
        diff = sum(paper_VI_D_system(F, x, 205 % F.q) != S(F, x, 205 % F.q)
                   for x in (_rand(F, rng) for _ in range(100)))
        assert diff == 100, "the F_p discrepancy is systematic, not sporadic"
    # and the first line is exactly a sign flip:  y3_paper = -(z0)
    F = make_field("65537")
    for _ in range(50):
        x = _rand(F, rng)
        z0 = S(F, x, 205)[3]
        assert paper_VI_D_system(F, x, 205)[3] == F.sub(0, z0)


# -------------------------------------------------------------- linear ----
@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_linear_layers_are_mutually_inverse(fname):
    """Sect. IV-A.3 + IV-B: LP o LP_inv = LP_inv o LP = I over both fields."""
    F = make_field(fname)
    lin = LinearLayer(F)
    rng = np.random.default_rng(29)
    for _ in range(20):
        x = _rand(F, rng, WORDS)
        assert lin.L(lin.L_inv(x)) == x
        assert lin.L_inv(lin.L(x)) == x


def test_paper_rotation_sets_and_v_p_are_locked():
    """Sect. IV-A.3: LP_inv = sum over {0,3,4,8,9,12,14}.
    Sect. IV-B.1: Yu2X's LP = XOR over {1,2,3,5,6,7,8,12,13,14,15}.
    Sect. IV-B.2: YupX's LP is the circulant with first row v_p, printed both
    as fractions and (for p = 65537) as integers.  All three are COMPUTED as
    the circulant inverse of LP_inv here and only compared with the paper."""
    assert ROT_INV == (0, 3, 4, 8, 9, 12, 14)
    for fname in ("2^4", "2^8", "2^16"):
        F = make_field(fname)
        row = forward_row(F)
        assert tuple(j for j in range(WORDS) if row[j]) == ROT_FWD_BIN
        assert set(row) <= {0, 1}
    F = make_field("65537")
    assert forward_row(F) == V_P_65537
    assert paper_v_p(F) == V_P_65537            # fractions == printed integers
    M = circulant(ROT_INV) % 65537
    Minv = np.zeros((WORDS, WORDS), dtype=np.int64)
    for i in range(WORDS):
        for k in range(WORDS):
            Minv[i][k] = V_P_65537[(k - i) % WORDS]
    assert np.array_equal((M @ Minv) % 65537, np.eye(WORDS, dtype=np.int64))


def test_linear_layer_ignores_the_t_argument():
    """YuX has no key-dependent layer; `t` exists only for dux compatibility."""
    F = make_field("257")
    lin = LinearLayer(F)
    rng = np.random.default_rng(31)
    x = _rand(F, rng, WORDS)
    assert lin.L(x, 0) == lin.L(x, 1) == lin.LM(x, None)
    assert lin.L_inv(x, 0) == lin.L_inv(x, 1) == lin.LM_inv(x, None)
    assert lin._rows[0] == lin._rows[1]


@pytest.mark.parametrize("fname", FIELDS_ALL)
def test_vectorised_linear_matches_scalar(fname):
    F = make_field(fname)
    lin = LinearLayer(F)
    rng = np.random.default_rng(37)
    xs = [_rand(F, rng, WORDS) for _ in range(8)]
    X = tuple(np.array([x[i] for x in xs], dtype=np.int64) for i in range(WORDS))
    Y = lin.L(X, 0, vec=True)
    Yi = lin.L_inv(X, 0, vec=True)
    for k, x in enumerate(xs):
        assert tuple(int(w[k]) for w in Y) == lin.L(x)
        assert tuple(int(w[k]) for w in Yi) == lin.L_inv(x)


# ------------------------------------------------------- key schedule -----
@pytest.mark.parametrize("inst", ALL)
def test_key_schedule_is_invertible(inst):
    """Algorithm 1 read with X_i = key_{4i..4i+3}: the state after round i is
    rk^i, so rk^{i-1} follows from rk^i without inverting S."""
    c = get_cipher(inst)
    rng = np.random.default_rng(2026)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    assert len(rks) == c.r + 1
    assert rks[0] == tuple(K)
    for i in range(1, c.r + 1):
        assert c.key_schedule_inverse(rks[i], i) == rks[i - 1]


def test_ks_literal_reading_uses_only_seven_key_words():
    """Ambiguity A1: read verbatim, Algorithm 1 line 2 seeds the NLFSR with
    four SLIDING windows, so rk^1..rk^r depend on key_0..key_6 only."""
    c = get_cipher("yuxtoy-257", ks_literal=True)
    rng = np.random.default_rng(2026)
    K = list(c.random_key(rng))
    rks = c.key_schedule(K)
    for w in range(7, 16):                      # touching key_7..key_15
        K2 = list(K)
        K2[w] = (K2[w] + 1) % c.F.q
        rks2 = c.key_schedule(K2)
        assert rks2[1:] == rks[1:]
        assert rks2[0] != rks[0]
    for w in range(7):                          # key_0..key_6 do matter
        K2 = list(K)
        K2[w] = (K2[w] + 1) % c.F.q
        assert c.key_schedule(K2)[1] != rks[1]
    # the non-overlapping reading uses all sixteen
    c2 = get_cipher("yuxtoy-257")
    rks_n = c2.key_schedule(K)
    for w in range(16):
        K2 = list(K)
        K2[w] = (K2[w] + 1) % c2.F.q
        assert c2.key_schedule(K2)[1] != rks_n[1]
    # both readings keep the state recursion invertible
    seed = seed_state(K, ks_literal=True)
    rcs = round_constants(c.F, c.alpha, c.r)
    rk1, _ = ks_round(c.F, c.alpha, seed, rcs[0])
    assert ks_round_inverse(c.F, c.alpha, rk1, rcs[0]) == seed


def test_key_schedule_block_rotation_and_constants():
    """Algorithm 1 line 6 uses a 4-word block rotation with the state
    convention (X <<< 3)_i = X_{(i+3) mod 4}; Algorithm 2 builds rc from the
    ENCRYPTION S-box applied to (id+1, id+2, id+3, id+4)."""
    assert rot3((10, 11, 12, 13)) == (13, 10, 11, 12)
    F = make_field("65537")
    rcs = round_constants(F, 205, 3)
    assert len(rcs) == 3 and all(len(r) == 16 for r in rcs)
    for i in range(3):
        for j in range(4):
            t = 16 * i + 4 * j
            assert tuple(rcs[i][4 * j:4 * j + 4]) == \
                S(F, tuple(F.from_int(t + k) for k in (1, 2, 3, 4)), 205)


# ------------------------------------------------------------- cipher -----
@pytest.mark.parametrize("inst", ALL)
def test_encrypt_decrypt_roundtrip_scalar_and_vector(inst):
    """Algorithm 3 and its exact inverse, on all six instances."""
    c = get_cipher(inst)
    rng = np.random.default_rng(2026)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    Ps = [tuple(int(v) for v in rng.integers(0, c.F.q, size=WORDS))
          for _ in range(8)]
    Cs = [c.encrypt(P, rks) for P in Ps]
    for P, C in zip(Ps, Cs):
        assert c.decrypt(C, rks) == P
    Pv = tuple(np.array([P[i] for P in Ps], dtype=np.int64) for i in range(WORDS))
    Cv = c.encrypt(Pv, rks, vec=True)
    for k, C in enumerate(Cs):
        assert tuple(int(w[k]) for w in Cv) == C
    Pv2 = c.decrypt(Cv, rks, vec=True)
    for k, P in enumerate(Ps):
        assert tuple(int(w[k]) for w in Pv2) == P


@pytest.mark.parametrize("inst", ["yuxtoy-257", "yuxtoy-2^4", "yu2x-8"])
def test_decrypt_layers_last_layer_matches_decrypt(inst):
    """decrypt_layers(..., layers=r) must end where decrypt does (before the
    final ARK with rk^0); encrypt_layers is its mirror."""
    c = get_cipher(inst)
    rng = np.random.default_rng(5)
    rks = c.key_schedule(c.random_key(rng))
    C = tuple(int(v) for v in rng.integers(0, c.F.q, size=WORDS))
    last = c.decrypt_layers(C, rks, c.r)
    assert c.ARK_inv(last, rks[0]) == c.decrypt(C, rks)
    P = c.decrypt(C, rks)
    assert c.ARK(c.encrypt_layers(P, rks, c.r), rks[c.r]) == C
    states = c.decrypt_layers(C, rks, c.r, yield_all=True)
    assert len(states) == c.r and states[-1] == last


def test_equivalent_fixedL0_is_trivial_for_yux():
    """O2 has no content for YuX: there is a single linear layer."""
    c = get_cipher("yuxtoy-257")
    rng = np.random.default_rng(9)
    rks = c.key_schedule(c.random_key(rng))
    new, T = c.equivalent_fixedL0_keys(rks)
    assert T == 0 and new == [tuple(rk) for rk in rks]
    P = tuple(int(v) for v in rng.integers(0, c.F.q, size=WORDS))
    assert c.encrypt_fixedL0(P, rks) == c.encrypt(P, rks)


# ------------------------------------------------------------ registry ----
def test_registry_dispatches_by_prefix():
    from dux import DuX
    assert cipher_family("dux-65537") == "dux"
    assert cipher_family("toy-257") == "dux"
    for inst in ALL:
        assert cipher_family(inst) == "yux"
        assert isinstance(get_cipher(inst), YuX)
    assert isinstance(get_cipher("dux-2^8"), DuX)
    with pytest.raises(KeyError):
        cipher_family("aes-128")
    reg = instances()
    assert set(reg) >= set(ALL) | {"dux-65537", "toy-257"}
    assert reg["yu2x-16"]["family"] == "yux"


def test_table_II_parameters():
    """Table II: field, defining polynomial, alpha = 205, round numbers."""
    from dux.params import FIELDS
    assert INSTANCES["yu2x-8"]["field"] == "2^8"
    assert FIELDS["2^8"]["poly"] == 0x11B          # x^8+x^4+x^3+x+1
    assert INSTANCES["yu2x-16"]["field"] == "2^16"
    assert FIELDS["2^16"]["poly"] == 0x1100B       # x^16+x^12+x^3+x+1
    assert INSTANCES["yupx-65537"]["field"] == "65537"
    for inst in OFFICIAL:
        assert INSTANCES[inst]["alpha"] == 205
    assert INSTANCES["yu2x-8"]["rounds"] == 12
    assert (INSTANCES["yu2x-16"]["rounds"], INSTANCES["yu2x-16"]["rounds_128"]) == (12, 14)
    assert (INSTANCES["yupx-65537"]["rounds"], INSTANCES["yupx-65537"]["rounds_fhe"]) == (14, 9)


# ------------------------------------------------------------- vectors ----
@pytest.mark.parametrize("path", VEC_FILES, ids=[os.path.basename(p) for p in VEC_FILES])
def test_yux_regression_vectors(path):
    data = json.load(open(path))
    inst = data["instance"]
    spec = INSTANCES[inst]
    assert (spec["field"], spec["alpha"], spec["rounds"]) == \
           (data["field"], data["alpha"], data["rounds"])
    c = get_cipher(inst)
    assert len(data["vectors"]) >= 3
    for v in data["vectors"]:
        K, P, C = tuple(v["K"]), tuple(v["P"]), tuple(v["C"])
        rks = c.key_schedule(K)
        assert [list(rk) for rk in rks] == v["round_keys"]
        assert list(rks[1]) == v["rk1"]
        assert c.encrypt(P, rks) == C
        assert c.decrypt(C, rks) == P
        dec = c.decrypt_layers(C, rks, c.r, yield_all=True)
        assert [list(s) for s in dec] == v["dec_layer_states"]
