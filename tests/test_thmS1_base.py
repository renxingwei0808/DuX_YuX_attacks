"""S22: the base case of Theorem S1 (O9) holds on the tested keys for every
active position, including forced lambda = 0 keys; see
experiments/E10_cpa/check_thmS1_base.py and its records in results/E10_cpa/."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E10_cpa"))

import check_thmS1_base as CB   # noqa: E402


@pytest.mark.parametrize("instance,layers", [("dux-2^16", 5), ("dux-2^8", 4)])
def test_next_to_leading_vanishes_from_layer_4(instance, layers):
    res = CB.run(instance, layers, keys=3, seed=2026, positions=[0, 1, 2, 3])
    for pos, r in res["positions"].items():
        want = 3 if pos == "3" else 4
        assert r["clean_from_layer"] is not None and r["clean_from_layer"] <= want, (pos, r)
    r1 = res["positions"]["1"]
    assert r1["layer3_block0_u0_u3_same_top_pair"] == 3
    assert r1["layer3_blocks123_y3_degree_at_most_D_minus_3"] == 3


@pytest.mark.parametrize("instance,layers", [("dux-2^16", 5), ("dux-2^8", 4)])
def test_degenerate_keys_are_clean_from_layer_3(instance, layers):
    res = CB.run(instance, layers, keys=2, seed=7, positions=[1], degenerate=True)
    r = res["positions"]["1"]
    assert r["lambda_zero_keys"] == 2
    assert r["clean_from_layer"] <= 3
