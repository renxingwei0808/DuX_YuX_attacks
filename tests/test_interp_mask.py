"""O15 (R8): divided-difference masks (tools/interp_mask.py)."""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import interp_mask as im   # noqa: E402


@pytest.mark.parametrize("p,n", [(257, 7), (257, 40), (193, 100)])
def test_divided_differences_kill_low_degrees(p, n):
    rng = np.random.default_rng(n)
    U = np.sort(rng.choice(p, size=n, replace=False)).astype(np.int64)
    w = im.divided_difference_weights_prime(p, U)
    ua = np.ones(n, dtype=np.int64)
    for e in range(n - 1):
        assert int((w * ua % p).sum() % p) == 0, e      # sum w_u u^e = 0 for e <= n-2
        ua = ua * U % p
    assert int((w * ua % p).sum() % p) == 1              # e = n-1: leading coefficient 1


def test_plan_numbers_from_the_memo():
    assert im.min_points(70226, 9000, 2)[0] == 39614
    assert im.min_points(40545, 2152, 1)[0] == 42698
    assert im.usable_weights([39614, 39614], 70226) == 9000
    assert im.threshold([257]) == 256


def test_masked_moments_on_yuxtoy257_match_criterion():
    out = im.check("yuxtoy-257", words=3, n=60, layers=4, keys=2, seed=3)
    for r in out["results"]:
        for obs, pred in zip(r["observed_first_nonzero_a"], r["predicted_zero_moments"]):
            assert obs is None or obs >= pred


# --------------------------------------------------------------------------
# W25 additions: characteristic 2, the repository's own degrees, product sets.
# --------------------------------------------------------------------------
def test_divided_differences_kill_low_degrees_in_char_2():
    from dux.field import make_field
    for name, n in (("2^4", 12), ("2^8", 40)):
        F = make_field(name)
        rng = np.random.default_rng(n)
        U = np.sort(rng.choice(F.q, size=n, replace=False)).astype(np.int64)
        w = im.divided_difference_weights_2n(F, U)
        ua = np.ones(n, dtype=np.int64)
        for e in range(n - 1):
            acc = F.vmul(w.astype(F.dtype), ua.astype(F.dtype))
            assert int(np.bitwise_xor.reduce(acc)) == 0, (name, e)
            ua = F.vmul(ua.astype(F.dtype), U.astype(F.dtype)).astype(np.int64)
        acc = F.vmul(w.astype(F.dtype), ua.astype(F.dtype))
        assert int(np.bitwise_xor.reduce(acc)) == 1     # e = n-1: leading coefficient


def test_dispatch_matches_the_prime_routine():
    from dux.field import make_field
    F = make_field("257")
    rng = np.random.default_rng(7)
    U = np.sort(rng.choice(257, size=30, replace=False)).astype(np.int64)
    assert (im.divided_difference_weights(F, U)
            == im.divided_difference_weights_prime(257, U)).all()


def test_degrees_come_from_the_repository_tool():
    """`maxplus_degrees` must be `tools/cipher_degree.py`'s `profile`, which is
    also what tools/zero_sum_criterion.py uses."""
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.join(_ROOT, "tools"))
    from cipher_degree import profile
    from dux.registry import get_cipher
    for inst, fam, word, layers in (("yuxtoy-257", "yux", 3, 4),
                                    ("toy-257", "dux", 3, 5),
                                    ("toy-2^4", "dux", 0, 3)):
        c = get_cipher(inst)
        assert (im.maxplus_degrees(c, fam, {word}, layers)
                == profile([word], layers, "dec", c.F.char == 2, fam))


def test_char2_check_respects_the_criterion():
    """Characteristic 2: observed >= predicted (O13 leaves the max-plus bound
    conservative, so equality is not required)."""
    out = im.check("toy-2^4", words=3, n=12, layers=3, keys=2, seed=5)
    assert out["all_respected"]
    assert out["threshold"] == 11


def test_prime_check_is_exact_cell_by_cell():
    out = im.check("yuxtoy-257", words=3, n=100, layers=4, keys=2, seed=1)
    assert out["exact"], out["results"][0]
    assert out["predicted_zero_moments"] == [37, 37, 5, 0, 36, 35, 4, 0,
                                             39, 36, 7, 0, 40, 38, 8, 0]
