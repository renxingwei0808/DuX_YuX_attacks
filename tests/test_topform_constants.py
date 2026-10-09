"""O14 (R8): key-independent constant sums of the YuX full-block structure at
the exact-boundary cells D = T over F_p (tools/topform_constants.py)."""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import topform_constants as tfc   # noqa: E402


@pytest.mark.parametrize("p,layers", [(5, 3), (17, 4)])
def test_exact_hit_layer_and_degrees(p, layers):
    assert tfc.exact_hit_layer(p) == layers
    _, deg = tfc.constants_affine(p, layers)
    assert deg[3::4] == [4 * (p - 1)] * 4          # position 3 hits T exactly
    assert all(d < 4 * (p - 1) for i, d in enumerate(deg) if i % 4 != 3)


@pytest.mark.parametrize("p,layers", [(5, 3), (17, 4)])
def test_direct_sums_are_key_independent_and_match_top_form(p, layers):
    c_aff, _ = tfc.constants_affine(p, layers)
    c_proj, _ = tfc.constants_projective(p, layers)
    assert c_aff[3::4] == c_proj[3::4]
    sums = [tfc.direct_sums(p, layers, seed=s) for s in (1, 2, 3)]
    assert sums[0] == sums[1] == sums[2]                       # independent of keys and constants
    assert all(v == 0 for s in sums for i, v in enumerate(s) if i % 4 != 3)   # positions 0-2 balanced
    assert sums[0][3::4] == c_aff[3::4]                        # position 3 = universal constant
    assert any(v != 0 for v in sums[0][3::4])                  # and it is not a zero-sum


def test_below_threshold_is_zero_sum():
    # p = 5, layer 2: D = (2,2,3,4) < T = 16 -> all sixteen sums are zero
    assert tfc.direct_sums(5, 2, seed=1) == [0] * 16


# --------------------------------------------------------------------------
# W24 additions: the general (cipher, direction) top forms, the real key
# schedule, and the encryption-direction degeneracy.
# --------------------------------------------------------------------------
import numpy as np                                            # noqa: E402


@pytest.mark.parametrize("p,layers", [(5, 3), (17, 4)])
def test_general_reproduces_the_reference_recursion(p, layers):
    """`topforms_general` with `sinv_top` must be the untouched `topforms`."""
    M = tfc.linv_matrix(p)
    y = tuple(np.arange(p, dtype=np.int64) + k for k in range(4))
    a, da = tfc.topforms(y, p, layers, M)
    b, db = tfc.topforms_general(y, p, layers, M, tfc.sinv_top)
    assert da == db
    assert all((u == v).all() for u, v in zip(a, b))


@pytest.mark.parametrize("p,layers", [(5, 3), (17, 4)])
def test_real_key_schedule_reproduces_the_constant(p, layers):
    """W24 step 2: 3 real master keys x 2 inactive-constant sets."""
    c, _ = tfc.constants_projective(p, layers)
    T = 4 * (p - 1)
    rows = []
    for seed in (2026, 7, 11):
        rows += tfc.direct_sums_general(p, layers, seed, "yux", "dec", constant_sets=2)
    assert len(rows) == 6
    assert len({tuple(r["sums"]) for r in rows}) == 1          # key-independent
    assert rows[0]["sums"] == c
    assert any(v for v in rows[0]["sums"][3::4])               # a non-zero constant
    assert T == 4 * (p - 1)


def test_layer_matrix_agrees_with_linv_matrix():
    for p in (5, 17, 257):
        assert (tfc.layer_matrix(p, "yux", "dec") == tfc.linv_matrix(p)).all()


def test_encryption_exact_hit_layers():
    """8^{l-1} = 4(p-1) has only p = 3 / 2, 17 / 3, 65537 / 7."""
    assert tfc.exact_hit_layer(3, "enc", "dux") == 2
    assert tfc.exact_hit_layer(17, "enc", "yux") == 3
    assert tfc.exact_hit_layer(65537, "enc", "yux") == 7
    assert tfc.exact_hit_layer(65537, "enc", "dux") == 7
    assert tfc.exact_hit_layer(257, "enc", "yux") is None
    assert tfc.exact_hit_layer(5, "dec", "dux") is None        # DuX decryption never hits


@pytest.mark.parametrize("cipher,p,layers,top", [("yux", 17, 3, 0), ("dux", 3, 2, 3)])
def test_encryption_boundary_cell_is_a_zero_sum(cipher, p, layers, top):
    """W24 step 3.  The CPA exact-boundary cell sums to 0, NOT to a non-zero
    constant as the R8 memo predicted -- on the top-form recursion, on the
    brute-force sum of the top forms, and on the real cipher with the real key
    schedule (3 master keys x 2 constant sets)."""
    M = tfc.layer_matrix(p, cipher, "enc")
    fn = tfc.SBOX_TOP[(cipher, "enc")]
    c, deg = tfc.constants_projective(p, layers, M, fn)
    ca, _ = tfc.constants_affine(p, layers, M, fn)
    assert deg[top::4] == [4 * (p - 1)] * 4                    # the boundary is where we think
    assert all(d < 4 * (p - 1) for i, d in enumerate(deg) if i % 4 != top)
    # the projective sum is only the true sum at the boundary words; the
    # brute-force sum of the top forms is valid everywhere
    assert c[top::4] == ca[top::4] == [0] * 4
    assert ca == [0] * 16
    rows = []
    for seed in (2026, 7, 11):
        rows += tfc.direct_sums_general(p, layers, seed, cipher, "enc", constant_sets=2)
    assert len(rows) == 6
    assert all(r["sums"] == [0] * 16 for r in rows)


def test_encryption_degeneracy_is_the_reason_for_the_zero():
    """YuX's forward circulant restricted to the free variables has rank 3 on
    the rows the encryption top forms use -- exactly over Q, so the p = 65537
    layer-7 cell is a zero sum too, without the 2^48 projective sum.  DuX's
    rank is 4, so its layer-7 constant stays unknown."""
    from fractions import Fraction as Fr
    from yux.params import V_P_FRACTIONS
    v = [Fr(a, b) for a, b in V_P_FRACTIONS]
    rows = [[v[(j - i) % 16] for j in range(4)] for i in (0, 1, 4, 5, 8, 9, 12, 13)]
    # every such row is a.y0 + b.y1 + c.sigma, i.e. its y2 and y3 coefficients agree
    assert all(r[2] == r[3] for r in rows)
    for p in (5, 17, 257, 65537):
        M = tfc.layer_matrix(p, "yux", "enc")
        assert _rank4([[int(M[i][j]) % p for j in range(4)] for i in (0, 1, 4, 5, 8, 9, 12, 13)], p) == 3
        Mi = tfc.linv_matrix(p)
        assert _rank4([[int(Mi[4 * b + o][j]) % p for j in range(4)]
                       for b in range(4) for o in (1, 2, 3)], p) == 4
    for p in (3, 257, 65537):
        M = tfc.layer_matrix(p, "dux", "enc")
        assert _rank4([[int(M[4 * b + o][j]) % p for j in range(4)]
                       for b in range(4) for o in (0, 3)], p) == 4


def _rank4(rows, p):
    A = [r[:] for r in rows]
    r = 0
    for col in range(4):
        piv = next((i for i in range(r, len(A)) if A[i][col] % p), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][col], p - 2, p)
        A[r] = [(x * inv) % p for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][col] % p:
                f = A[i][col]
                A[i] = [(A[i][k] - f * A[r][k]) % p for k in range(4)]
        r += 1
    return r


def test_o2_fixed_L0_differs_only_by_a_word_rotation():
    """DuX's key-dependent L0/L1: the real `encrypt_layers` sums equal the
    fixed-L0 sums rotated by 4 T, with T the accumulated t_xor (O2)."""
    # by path: several experiment directories carry a module called `run`, and
    # whichever is imported first wins for the rest of the session
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "y09_run", os.path.join(_ROOT, "experiments", "Y09_topform_constants", "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    o2_rotation_check = mod.o2_rotation_check
    for seed in (2026, 7, 11):
        r = o2_rotation_check(seed=seed)
        assert r["predicted_rotation"] in r["matching_rotations"]
        assert any(v for v in r["real"])                       # past the boundary: not all zero
