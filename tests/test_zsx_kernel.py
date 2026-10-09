"""The generic C zero-sum kernel must reproduce the numpy reference exactly.

`run_zsx.py --verify` recomputes the whole job with dux/ and asserts equality,
so each case below is one cross-check of a different kernel path: prime field
vs F_{2^n}, decryption vs encryption, full field vs affine subspace vs signed
subgroup cosets, one active word vs several.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FASTDIR = os.path.join(ROOT, "experiments", "E09_unified_criterion", "fast")
RUNNER = os.path.join(FASTDIR, "run_zsx.py")
BINARY = os.path.join(FASTDIR, "zsx")

pytestmark = pytest.mark.skipif(
    not os.path.exists(BINARY),
    reason="experiments/E09_unified_criterion/fast/zsx not built")


@pytest.mark.parametrize("args", [
    ["--instance", "toy-193", "--active", "3", "--layers", "6"],
    ["--instance", "toy-193", "--active", "3,7", "--layers", "6"],
    ["--instance", "toy-257", "--active", "3", "--coset", "5", "--layers", "6"],
    ["--instance", "toy-257", "--active", "3,7", "--coset", "4", "--layers", "6"],
    ["--instance", "toy-2^4", "--active", "3,7", "--layers", "6"],
    ["--instance", "dux-2^8", "--active", "3", "--sets", "dim6", "--layers", "6"],
    ["--instance", "toy-193", "--active", "1", "--direction", "enc", "--layers", "5"],
    ["--instance", "toy-2^4", "--active", "1,5", "--direction", "enc", "--layers", "5"],
    ["--instance", "toy-257", "--active", "1", "--direction", "enc", "--layers", "5"],
])
def test_kernel_matches_numpy(args):
    env = dict(os.environ, OMP_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, RUNNER, "--keys", "2", "--verify"] + args,
                          capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "verify: OK" in proc.stdout


def test_mixed_sets_use_the_sum_of_thresholds():
    """A structure whose words use different set types: the O7 threshold is the
    sum of the per-word thresholds, and the prediction must still match."""
    env = dict(os.environ, OMP_NUM_THREADS="2")
    proc = subprocess.run(
        [sys.executable, RUNNER, "--instance", "toy-257", "--active", "3,7",
         "--sets", "full,coset5", "--layers", "6", "--keys", "2", "--verify"],
        capture_output=True, text=True, env=env, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "verify: OK" in proc.stdout
    assert "T=288" in proc.stdout          # (257 - 1) + 2^5
    assert "MATCH" in proc.stdout and "MISMATCH" not in proc.stdout


# ---- O14 (R8 / S16 item 1): the kernel's raw sums on a D = T boundary cell --
YUX_BINARY = os.path.join(FASTDIR, "zsx_yux")


@pytest.mark.skipif(not os.path.exists(YUX_BINARY),
                    reason="experiments/E09_unified_criterion/fast/zsx_yux not built")
@pytest.mark.parametrize("instance,layers,expected", [
    ("yuxtoy-5", 3, [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
    ("yuxtoy-17", 4, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 13]),
])
def test_raw_sums_give_the_o14_constants(instance, layers, expected):
    """`--raw-sums` exposes the 16 word sums; on the full-block D = T cell they
    are the key-independent constants c_b(p) of O14 (Y09 Sect. 2), the same
    for every real master key and every choice of the 12 inactive words."""
    import json
    import tempfile
    env = dict(os.environ, OMP_NUM_THREADS="2")
    with tempfile.TemporaryDirectory() as td:
        proc = subprocess.run(
            [sys.executable, RUNNER, "--instance", instance, "--full-block", "0",
             "--layers", str(layers), "--keys", "3", "--verify", "--raw-sums",
             "--out", td, "--tag", "t"],
            capture_output=True, text=True, env=env, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
        assert "verify: OK" in proc.stdout
        res = json.load(open(os.path.join(td, "t.json")))
    raw = res["raw_sums"]
    assert len(raw) == 3
    for k in raw:
        assert raw[k][f"layer{layers}"] == expected, (k, raw[k][f"layer{layers}"])
        # every earlier layer is a plain zero sum (D < T there)
        for l in range(1, layers):
            assert raw[k][f"layer{l}"] == [0] * 16
