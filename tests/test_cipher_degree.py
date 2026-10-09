"""W15-B -- the dual-cipher max-plus registry and the full-block structures.

Three things are locked here:
  (i)   YuX's single-word degree table,
  (ii)  the full-block table (O11), including the CPA direction,
  (iii) that DuX's numbers -- and the 39-cell `--grid` -- did not move.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import cipher_degree as cdg          # noqa: E402
import maxplus                       # noqa: E402
import zero_sum_criterion as zsc     # noqa: E402

T16 = 2 ** 16 - 1


def per_position(Y):
    return [max(Y[4 * b + p] for b in range(4)) for p in range(4)]


# --- (i) YuX single word ---------------------------------------------------
YUX_BLOCK0 = [  # memo Sect. 2, active word = position 0 of block 0, block-0 row
    (1, 1, 1, 2), (2, 3, 4, 5), (12, 14, 19, 26), (49, 49, 75, 98),
    (193, 204, 297, 397), (816, 816, 1228, 1632), (3186, 3227, 4819, 6413),
    (13023, 13060, 19553, 26083), (51763, 51888, 77846, 103651),
    (207924, 208356, 312102, 416280),
]


def test_yux_single_word_degree_table():
    rows = cdg.profile([0], 11, cipher="yux")
    for layer, want in enumerate(YUX_BLOCK0, 1):
        assert tuple(rows[layer - 1][0:4]) == want, f"layer {layer}"
    # the Hamming-weight conversion Ni et al. tabulate
    hw = [tuple((d + 1).bit_length() - 1 for d in r) for r in YUX_BLOCK0]
    assert hw[:9] == [(1, 1, 1, 1), (1, 2, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6),
                      (7, 7, 8, 8), (9, 9, 10, 10), (11, 11, 12, 12),
                      (13, 13, 14, 14), (15, 15, 16, 16)]


def test_yux_single_word_zero_sum_layers():
    """Yu2X-16 / YupX, one active ciphertext word: 8 full layers, then 1100."""
    for q, marg in (("2^16", (13494, 13369)), (65537, (13495, 13370))):
        r = zsc.patterns(q, [0], 11, cipher="yux")
        assert (r["l_full"], r["next"]) == (8, "1100")
        assert tuple(r["margin_per_position"][8][:2]) == marg
    # starting at position 3 the margins are the smaller ones quoted in the memo
    for q, marg in (("2^16", (1327, 1283)), (65537, (1328, 1284))):
        r = zsc.patterns(q, [3], 11, cipher="yux")
        assert (r["l_full"], r["next"]) == (8, "1100")
        assert tuple(r["margin_per_position"][8][:2]) == marg


def test_yux_two_word_and_multiword_structures():
    """Memo Sect. 2 item 4: two words in different blocks reach layer 9."""
    r = zsc.patterns("2^16", [0, 4], 11, cipher="yux")
    assert r["l_full"] == 9 and r["margin_per_position"][8][3] == 10223
    r = zsc.patterns(65537, [0, 4], 11, cipher="yux")
    assert r["l_full"] == 9


# --- (ii) full block -------------------------------------------------------
def test_full_block_matches_memo_table():
    """O11 / memo Sect. 4, cell by cell."""
    r = zsc.patterns("2^16", [], 11, cipher="yux", free_blocks=[0])
    assert r["s"] == 4 and r["threshold"] == 4 * T16
    assert r["patterns"][9] == "1110"
    assert r["margin_per_position"][9] == [131068, 131068, 65532, -4]
    r = zsc.patterns(65537, [], 11, cipher="yux", free_blocks=[0])
    assert r["patterns"][9] == "1110"
    assert r["margin_per_position"][9] == [131072, 131072, 65536, 0]
    # Yu2X-8: full block 2^32 -> layer 6 pattern 1110
    r = zsc.patterns("2^8", [], 8, cipher="yux", free_blocks=[0])
    assert r["patterns"][5] == "1110" and r["margin_per_position"][5] == [508, 508, 252, -4]
    # DuX: full block 2^64 -> layer 11 pattern 0001, margin 57 (F_p) / 53 (F_2^16)
    r = zsc.patterns(65537, [], 12, cipher="dux", free_blocks=[0])
    assert r["patterns"][10] == "0001" and r["margin_per_position"][10][3] == 57
    r = zsc.patterns("2^16", [], 12, cipher="dux", free_blocks=[0])
    assert r["patterns"][10] == "0001" and r["margin_per_position"][10][3] == 53
    # DuX(2^8): full block + words (7, 11), 2^48 -> layer 7 pattern 0001, margin 179
    r = zsc.patterns("2^8", [7, 11], 8, cipher="dux", free_blocks=[0])
    assert r["s"] == 6 and r["threshold"] == 6 * 255
    assert r["patterns"][6] == "0001" and r["margin_per_position"][6][3] == 179
    # DuX full block + word 7 -> layer 11 pattern 1001
    r = zsc.patterns(65537, [7], 12, cipher="dux", free_blocks=[0])
    assert r["patterns"][10] == "1001"


def test_full_block_cpa_direction():
    """Memo Sect. 4, encryption direction: YuX 0111 and DuX 1110 at layer 7."""
    r = zsc.patterns("2^16", [], 8, direction="enc", cipher="yux", free_blocks=[0])
    assert r["patterns"][6] == "0111" and r["margin_per_position"][6][0] == -4
    r = zsc.patterns("2^16", [], 8, direction="enc", cipher="dux", free_blocks=[0])
    assert r["patterns"][6] == "1110" and r["margin_per_position"][6][3] == -4
    # single word, CPA: YuX needs an active word at position 1..3 for 0111
    r = zsc.patterns("2^16", [1], 8, direction="enc", cipher="yux")
    assert (r["l_full"], r["next"]) == (5, "0111")
    assert r["margin_per_position"][5][1] == 4095
    r = zsc.patterns("2^16", [0], 8, direction="enc", cipher="dux")
    assert (r["l_full"], r["next"]) == (5, "1110")
    assert r["margin_per_position"][5][0] == 4095


def test_full_block_needs_a_whole_block():
    with pytest.raises(AssertionError):
        cdg.profile([1], 3, cipher="yux", free_blocks=[0])
    with pytest.raises(AssertionError):
        zsc.threshold(65537, [3], coset=15, free_blocks=[0])


@pytest.mark.parametrize("q,layer", [("2^16", 10), ("2^8", 6)])
def test_full_block_dominates_the_four_word_structure(q, layer):
    """The point of O11: four words in four blocks enter the recurrence at
    (1,1,1,2) per block, a whole block at (1,1,1,1).  Same data (2^{4n}), same
    threshold, but at the first non-full layer the four-word structure balances
    nothing and the full block balances three positions.  (The four-word cell
    sits exactly on the Frobenius boundary D = 4 * 2^n, margin -4, which is the
    62-vs-64 cell of Ni et al.'s Table 7 -- see experiments/Y02.)"""
    a = zsc.patterns(q, [0, 4, 8, 12], layer + 1, cipher="yux")
    b = zsc.patterns(q, [], layer + 1, cipher="yux", free_blocks=[0])
    assert a["threshold"] == b["threshold"] and a["s"] == b["s"] == 4
    assert a["l_full"] == b["l_full"] == layer - 1
    assert a["patterns"][layer - 1] == "0000"
    assert b["patterns"][layer - 1] == "1110"
    assert a["margin_per_position"][layer - 1][:2] == [-4, -4]


# --- (iii) DuX unchanged ---------------------------------------------------
def test_dux_grid_still_39_of_39():
    assert zsc.run_grid()


@pytest.mark.parametrize("pos", [0, 1, 2, 3])
@pytest.mark.parametrize("cipher", ["dux", "yux"])
def test_zero_sum_criterion_agrees_with_maxplus(cipher, pos):
    """Two independently written implementations of the same recurrence."""
    rows = maxplus.run_single_word(pos, 12, cipher=cipher)
    prof = cdg.profile([pos], 12, cipher=cipher)
    for layer, row in enumerate(rows):
        flat = [d for blk in row["sbox_out_per_block"] for d in blk]
        assert flat == prof[layer], f"{cipher} layer {layer + 1}"


@pytest.mark.parametrize("cipher", ["dux", "yux"])
def test_maxplus_full_block_agrees_with_criterion(cipher):
    rows = maxplus.run_full_block([0], 12, cipher=cipher)
    prof = cdg.profile([], 12, cipher=cipher, free_blocks=[0])
    for layer, row in enumerate(rows):
        flat = [d for blk in row["sbox_out_per_block"] for d in blk]
        assert flat == prof[layer]


def test_sbox_degree_profiles():
    """The coordinate degrees the two recurrences imply for a fresh input."""
    one = (1, 1, 1, 1)
    assert cdg.dux_sinv_deg(one) == [2, 3, 4, 2]
    assert cdg.dux_s_deg(one) == [5, 3, 2, 8]
    assert cdg.yux_sinv_deg(one) == [2, 2, 3, 4]
    assert cdg.yux_s_deg(one) == [8, 5, 3, 2]
    for cipher in ("dux", "yux"):
        assert tuple(cdg.CIPHERS[cipher]["sinv_deg"](one)) == \
            cdg.CIPHERS[cipher]["sinv_degrees"]
        assert tuple(cdg.CIPHERS[cipher]["s_deg"](one)) == \
            cdg.CIPHERS[cipher]["s_degrees"]
