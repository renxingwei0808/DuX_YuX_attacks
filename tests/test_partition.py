"""R7 / W23 -- known-key zero-sum partitions (tools/zero_sum_criterion.py --partition).

Locks (a) the four splits and class-degree vectors of Liu-Sun (eprint
2026/1907, Corollary 5) for DuX, reproduced number by number with the same
alignment (state after a diffusion layer, backward groups (L^{-1}, S^{-1}),
forward groups (S, L)); (b) the new YuX splits; (c) the two alignments.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import zero_sum_criterion as zsc  # noqa: E402


LIU_SUN_COR5 = [
    # (q, active, r_b, r_f, backward class bounds at r_b, forward class bounds at r_f)
    (65537, (0,), 8, 5, [10084, 13775, 18817, 8733], [7680, 4608, 3072, 12288]),
    ("2^16", (0,), 8, 5, [10084, 13775, 18817, 8733], [7680, 4608, 3072, 12288]),
    (65537, (0, 4, 8, 12), 10, 6, [140452, 191861, 262087, 121635], [61440, 36864, 24576, 98304]),
    ("2^16", (0, 4, 8, 12), 10, 6, [140452, 191861, 262087, 121635], [61440, 36864, 24576, 98304]),
    ("2^8", (0, 4), 5, 3, [194, 265, 362, 168], [120, 72, 48, 192]),
    ("2^8", (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14), 6, 4, [1560, 2131, 2911, 1351], [960, 576, 384, 1536]),
]


@pytest.mark.parametrize("q,active,rb,rf,back,fwd", LIU_SUN_COR5)
def test_liu_sun_corollary_5_reproduced(q, active, rb, rf, back, fwd):
    r = zsc.partition(q, active, "dux")
    assert (r["r_b"], r["r_f"]) == (rb, rf)
    assert r["backward_bounds"] == back
    assert r["forward_bounds"] == fwd
    assert max(back) < r["threshold"] and max(fwd) < r["threshold"]
    # one more group on either side is no longer guaranteed
    assert max(r["backward_bounds_next"]) >= r["threshold"]
    assert max(r["forward_bounds_next"]) >= r["threshold"]


def test_liu_sun_active_sets_are_maximal():
    """Their A = {0} (s = 1) and A = C_0 (s = 4) maximise r_b + r_f; C_0 is
    the unique maximiser modulo block rotation."""
    best1 = zsc.partition_search(65537, 1, "dux")
    assert best1[0]["rounds"] == 13 and all(r["rounds"] == 13 for r in best1)
    best4 = zsc.partition_search(65537, 4, "dux")
    assert best4[0]["rounds"] == 16 and best4[0]["active"] == [0, 4, 8, 12]
    assert best4[1]["rounds"] == 15


YUX_SPLITS = [
    # (q, active, free blocks, align, r_b, r_f)
    (65537, (0,), (), "after-L", 8, 5),
    ("2^16", (0,), (), "after-L", 8, 5),
    (65537, (1, 2), (), "after-L", 8, 6),
    (65537, (0, 2, 3, 6), (), "after-L", 9, 6),
    ("2^16", (0, 2, 3, 6), (), "after-L", 9, 6),
    (65537, (), (0,), "after-L", 8, 5),          # a whole block is NOT free after L^{-1}
    (65537, (), (0,), "before-L", 9, 5),         # ... but is free when S^{-1} comes first (O11)
    ("2^8", (1,), (), "after-L", 4, 3),
    ("2^8", (0, 1, 3, 5), (), "after-L", 5, 3),
    ("2^8", (), (0,), "before-L", 5, 3),
]


@pytest.mark.parametrize("q,active,free,align,rb,rf", YUX_SPLITS)
def test_yux_partitions(q, active, free, align, rb, rf):
    r = zsc.partition(q, active, "yux", free, align)
    assert (r["r_b"], r["r_f"]) == (rb, rf)


def test_yux_single_word_bounds():
    r = zsc.partition(65537, (0,), "yux")
    assert r["backward_bounds"] == [16384, 16384, 24576, 32768]
    assert r["forward_bounds"] == [20480, 12800, 7680, 5120]


def test_dux_full_block_before_L():
    """DuX(2^8): one whole block at 2^32 gives 6 + 3 = 9 rounds when the state
    sits before the diffusion layer -- one round short of Liu-Sun's 10 at
    2^96, with 2^64 times fewer texts per coset."""
    r = zsc.partition("2^8", (), "dux", (0,), "before-L")
    assert (r["r_b"], r["r_f"]) == (6, 3)
    r = zsc.partition(65537, (), "dux", (0,), "before-L")
    assert (r["r_b"], r["r_f"]) == (10, 5)


def test_profile_from_matches_profile_without_leading_layer():
    import cipher_degree as cdg
    D0 = [0] * 16
    D0[3] = 1
    for cipher in ("dux", "yux"):
        a = cdg.profile([3], 9, "dec", False, cipher)
        b = cdg.profile_from(D0, 9, "dec", False, cipher, leading_linear=False)
        assert a == b
