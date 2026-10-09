import numpy as np
import pytest

from dux import DuX, make_field, INSTANCES
from dux.params import ROT_INV, ROT_FWD_BIN, M0_ROW_65537, WORDS
from dux.sbox import S, S_inv, S_closed, S_inv_closed
from dux.linear import LinearLayer, rotl, rot_sum, t_xor
from dux.keyschedule import Rf, Rf_inv, round_constants

ALL = list(INSTANCES)


def rng_state(rng, F):
    return tuple(int(v) for v in rng.integers(0, F.q, size=WORDS))


@pytest.mark.parametrize("name", ALL)
def test_sbox_inverse_and_closed_forms(name):
    c = DuX(name)
    F, a = c.F, c.alpha
    rng = np.random.default_rng(1)
    for _ in range(300):
        x = tuple(int(v) for v in rng.integers(0, F.q, size=4))
        y = S_inv(F, x, a)
        assert S(F, y, a) == x
        assert S_inv_closed(F, x, a) == y
        assert S_closed(F, x, a) == S(F, x, a)


@pytest.mark.parametrize("name", ALL)
def test_linear_inverse(name):
    c = DuX(name)
    F = c.F
    rng = np.random.default_rng(2)
    for t in (0, 1):
        for _ in range(50):
            x = rng_state(rng, F)
            assert c.lin.L(c.lin.L_inv(x, t), t) == x
            assert c.lin.L_inv(c.lin.L(x, t), t) == x


def test_paper_M0_row_p65537():
    F = make_field("65537")
    lin = LinearLayer(F)
    assert lin._rows[0] == M0_ROW_65537
    # M1 = M0 shifted by one 4-word sub-block (paper Sect. 5.4)
    assert lin._rows[1] == tuple(M0_ROW_65537[(k + 4) % 16] for k in range(16))


def test_forward_rotation_sets_are_inverse_over_F2():
    """Circulant over F_2: ROT_FWD_BIN[t] * ROT_INV[t] = identity."""
    for t in (0, 1):
        A = np.zeros((16, 16), dtype=int)
        B = np.zeros((16, 16), dtype=int)
        for i in range(16):
            for j in ROT_FWD_BIN[t]:
                A[i][(i + j) % 16] ^= 1
            for j in ROT_INV[t]:
                B[i][(i + j) % 16] ^= 1
        assert np.array_equal((A @ B) % 2, np.eye(16, dtype=int))


@pytest.mark.parametrize("name", ALL)
def test_O2_L1_is_rotated_L0(name):
    """L1 = Rot_{-4} o L0 and L1^{-1} = Rot_{+4} o L0^{-1}."""
    c = DuX(name)
    rng = np.random.default_rng(3)
    for _ in range(30):
        x = rng_state(rng, c.F)
        assert c.lin.L(x, 1) == rotl(c.lin.L(x, 0), -4)
        assert c.lin.L_inv(x, 1) == rotl(c.lin.L_inv(x, 0), 4)


@pytest.mark.parametrize("name", ALL)
def test_rf_inverse(name):
    c = DuX(name)
    rng = np.random.default_rng(4)
    rc = tuple(int(v) for v in rng.integers(0, c.F.q, size=4))
    for _ in range(30):
        x = rng_state(rng, c.F)
        assert Rf_inv(c.F, c.alpha, Rf(c.F, c.alpha, x, rc), rc) == x


@pytest.mark.parametrize("name", ALL)
def test_encrypt_decrypt_roundtrip(name):
    c = DuX(name)
    rng = np.random.default_rng(5)
    for _ in range(10):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        assert len(rks) == c.r + 1
        P = rng_state(rng, c.F)
        for r in (1, 2, 3, c.r):
            C = c.encrypt(P, rks, rounds=r)
            assert c.decrypt(C, rks, rounds=r) == P
        # decrypt_layers consistency: after all r layers + ARK_0 = P
        C = c.encrypt(P, rks)
        last = c.decrypt_layers(C, rks, c.r)
        assert c.ARK_inv(last, rks[0]) == P


@pytest.mark.parametrize("name", ALL)
def test_O2_fixedL0_equivalence(name):
    """DuX(P; rks) == Rot_{-4T}( DuX_L0(P; rk'_i = Rot_{4 T_i}(rk_i)) )."""
    c = DuX(name)
    rng = np.random.default_rng(6)
    seen_T = set()
    for _ in range(40):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        P = rng_state(rng, c.F)
        C = c.encrypt(P, rks)
        rks2, T = c.equivalent_fixedL0_keys(rks)
        seen_T.add(T % 4)
        C2 = c.encrypt_fixedL0(P, rks2)
        assert C == rotl(C2, -4 * T)
    # the key-dependent layer choice really varies across keys
    assert len(seen_T) >= 2


@pytest.mark.parametrize("name", ["dux-2^8", "dux-2^16", "dux-65537"])
def test_vectorised_matches_scalar(name):
    c = DuX(name)
    rng = np.random.default_rng(7)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    N = 64
    Ps = [rng_state(rng, c.F) for _ in range(N)]
    Pv = tuple(np.array([p[i] for p in Ps], dtype=np.int64) for i in range(16))
    Cv = c.encrypt(Pv, rks, vec=True)
    for k, P in enumerate(Ps):
        assert tuple(int(w[k]) for w in Cv) == c.encrypt(P, rks)
    Dv = c.decrypt(Cv, rks, vec=True)
    for k, P in enumerate(Ps):
        assert tuple(int(w[k]) for w in Dv) == P
    Lv = c.decrypt_layers(Cv, rks, 5, vec=True)
    for k in range(N):
        Ck = tuple(int(w[k]) for w in Cv)
        assert tuple(int(w[k]) for w in Lv) == c.decrypt_layers(Ck, rks, 5)


def test_round_constants_shape():
    c = DuX("dux-2^8")
    rcs = round_constants(c.F, c.alpha, c.r)
    assert len(rcs) == c.r and all(len(rc) == 16 for rc in rcs)


def test_txor_uses_word15():
    rk = [0] * 16
    rk[15] = 0b10
    assert t_xor(rk) == 1
    rk[15] = 0b11
    assert t_xor(rk) == 0
