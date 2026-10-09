"""Tests for the unified zero-sum criterion (Theorem O7, tools/zero_sum_criterion.py).

The criterion is an upper-bound theorem, so a test can only check two things:
(a) it reproduces every zero-sum cell the repository has actually measured, and
(b) it agrees with the independently written per-position max-plus recurrence in
    tools/maxplus.py (same recurrence, two implementations -- they lock each other).
The third test pins the predictions quoted in the write-up so that an accidental
change of the recurrence or of the threshold is caught immediately.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import maxplus  # noqa: E402
import zero_sum_criterion as zsc  # noqa: E402


def test_grid_reproduced():
    """All 39 measured cells (S1 minimal_data_table + E03 + two toys)."""
    assert zsc.run_grid()


@pytest.mark.parametrize("pos", [0, 1, 2, 3])
def test_matches_maxplus_single_word(pos):
    """s = 1: the joint total-degree recurrence degenerates to the per-position one."""
    layers = 12
    rows = maxplus.run_single_word(pos, layers)
    prof = zsc.profile([pos], layers)
    for layer, row in enumerate(rows):
        flat = [d for blk in row["sbox_out_per_block"] for d in blk]
        assert flat == prof[layer], f"layer {layer + 1} differs"


PAPER_PREDICTIONS = [
    # (q, active words, l_full, pattern of the next layer)
    (65537, (3, 7), 9, "1101"),
    ("2^16", (3, 7), 9, "1101"),
    (65537, (3, 7, 11), 10, "0000"),      # layer 10: all 16 words balanced
    ("2^16", (3, 7, 11), 10, "0000"),
    (65537, (3, 7, 11, 15), 10, "0001"),  # layer 11: block position 3
    ("2^16", (3, 7, 11, 15), 10, "0001"),
    ("2^8", (3, 7, 11, 15), 6, "0000"),
    (193, (3, 7), 5, "0001"),             # S7 toy entry point
]


@pytest.mark.parametrize("q,active,l_full,nxt", PAPER_PREDICTIONS)
def test_predictions_for_paper(q, active, l_full, nxt):
    r = zsc.patterns(q, active, 12)
    assert (r["l_full"], r["next"]) == (l_full, nxt)


MARGINS = [
    # (q, active, layer (1-based), block position, margin T - D) -- measured in E03/E04/E09
    (65537, (3, 7, 11), 10, 2, 45292),
    ("2^16", (3, 7, 11), 10, 2, 45289),
    (65537, (3, 7, 11, 15), 11, 3, 57),
    ("2^16", (3, 7, 11, 15), 11, 3, 53),
    (193, (3, 7), 6, 3, 22),
]


@pytest.mark.parametrize("q,active,layer,pos,margin", MARGINS)
def test_margins(q, active, layer, pos, margin):
    r = zsc.patterns(q, active, 12)
    assert r["margin_per_position"][layer - 1][pos] == margin
