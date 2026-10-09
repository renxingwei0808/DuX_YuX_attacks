"""W11 -- the encryption (chosen-plaintext / designer CPA) direction.

Three things are pinned here:
  * `DuX.encrypt_layers` agrees with `DuX.encrypt` (it is only a layer-by-layer
    replay of the same round function);
  * the encryption-direction criterion (`--direction enc`) reproduces the
    measured CPA zero-sums on the toys, and its S max-plus recurrence matches
    the actual coordinate degrees (5,3,2,8);
  * the last-round CPA key recovery returns the true master key.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E10_cpa"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "E08_boolean_degree_extension"))

from dux import DuX  # noqa: E402
import zero_sum_criterion as zsc  # noqa: E402
import run_zero_sum_cp as cp  # noqa: E402


@pytest.mark.parametrize("instance", ["dux-65537", "dux-2^8", "toy-2^4"])
def test_encrypt_layers_matches_encrypt(instance):
    c = DuX(instance, rounds=7)
    K = c.random_key(np.random.default_rng(1))
    rks = c.key_schedule(K)
    P = tuple(int(v) for v in np.random.default_rng(2).integers(0, c.F.q, size=16))
    states = c.encrypt_layers(P, rks, 7, rounds=7, yield_all=True)
    assert len(states) == 7
    assert tuple(c.ARK(states[-1], rks[7])) == tuple(c.encrypt(P, rks, rounds=7))


def test_enc_sbox_degrees():
    """S has coordinate degrees (5,3,2,8) at block positions (0,1,2,3)."""
    d = zsc.s_deg([1, 1, 1, 1])
    assert d == [5, 3, 2, 8]
    # a single active word at position 1 is the cheapest start (docs O4/W11)
    assert zsc.s_deg([0, 1, 0, 0]) == [1, 1, 0, 2]


def test_dec_and_enc_directions_differ():
    r_dec = zsc.patterns(65537, (3,), 12)
    r_enc = zsc.patterns(65537, (1,), 12, direction="enc")
    assert r_dec["l_full"] == 9          # chosen ciphertext
    assert r_enc["l_full"] == 5          # chosen plaintext: 8x growth per layer
    assert r_enc["next"] == "1110"


CP_CELLS = [
    # (instance, active plaintext words, layers, measured l_full, measured next)
    ("toy-193", [1], 5, 3, "0000"),
    ("toy-193", [1, 5], 5, 3, "0010"),
    ("toy-257", [1], 5, 3, "0000"),
    ("toy-257", [1, 5], 5, 3, "0110"),
    ("toy-2^4", [1], 4, 1, "1110"),
]


@pytest.mark.parametrize("instance,active,layers,lfull,nxt", CP_CELLS)
def test_cp_zero_sum_measured(instance, active, layers, lfull, nxt):
    _, res = cp.run(instance, active, layers, 2, 2026)
    pats = [{r["layers"][l]["block_pattern"] for r in res} for l in range(layers)]
    assert all(len(p) == 1 for p in pats), "keys disagree"
    common = [p.pop() for p in pats]
    got = 0
    for m in common:
        if m == "1111":
            got += 1
        else:
            break
    assert (got, common[got]) == (lfull, nxt)


def test_cp_attack_recovers_the_master_key():
    """4-round toy-257 (3-layer CPA zero-sum + last round)."""
    import subprocess
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "..", "experiments", "E10_cpa",
                                      "attack_cp_lastround.py"),
         "--instance", "toy-257", "--rounds", "4", "--active", "1",
         "--coords", "0,1,2,3", "--keys", "1"],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "master key == truth: True" in out.stdout, out.stdout
