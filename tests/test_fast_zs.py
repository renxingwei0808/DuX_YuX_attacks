"""Cross-check the C zero-sum kernel (experiments/E04_zero_sum_F2n/fast/zs.c)
against the numpy reference implementation.

Skipped when the binary has not been built (`make -C experiments/E04_zero_sum_F2n/fast`).
"""
import importlib.util
import os
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = os.path.join(ROOT, "experiments", "E04_zero_sum_F2n", "fast")
BIN = os.path.join(FAST, "zs")

sys.path.insert(0, ROOT)
from dux import DuX  # noqa: E402

spec = importlib.util.spec_from_file_location("run_fast", os.path.join(FAST, "run_fast.py"))
run_fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_fast)

pytestmark = pytest.mark.skipif(not os.path.exists(BIN), reason="zs binary not built")


@pytest.mark.parametrize("instance,active,dim,layers", [
    ("dux-2^8", [3], 8, 8),          # 2^8
    ("dux-2^8", [3, 7], 6, 7),       # 2^12
    ("dux-2^8", [0, 1], 8, 7),       # 2^16
    ("dux-2^8", [3, 7, 11], 5, 6),   # 2^15
    ("dux-2^16", [3], 12, 8),        # 2^12
    ("dux-2^16", [3, 7], 8, 9),      # 2^16
])
def test_c_matches_numpy(instance, active, dim, layers, tmp_path):
    cipher = DuX(instance)
    job_bytes, jobs = run_fast.build_job(cipher, active, dim, layers, 2, seed=7)
    jobpath = tmp_path / "job.bin"
    jobpath.write_bytes(job_bytes)
    proc = subprocess.run([BIN, str(jobpath)], capture_output=True, text=True,
                          env=dict(os.environ, OMP_NUM_THREADS="4"))
    assert proc.returncode == 0, proc.stderr
    sums = run_fast.parse_output(proc.stdout, 2, layers)
    for k, job in enumerate(jobs):
        ref = run_fast.python_reference(cipher, job, active, dim, layers)
        for l in range(layers):
            assert ref[l] == sums[k][l], f"key {k} layer {l + 1}"


def test_job_header_roundtrip():
    cipher = DuX("dux-2^8")
    job_bytes, _ = run_fast.build_job(cipher, [3], 4, 3, 1, seed=1)
    assert job_bytes[:8] == b"DUXZSJ01"
    n, poly, alpha, rounds, layers, nact, dim, keys = struct.unpack("<8I", job_bytes[8:40])
    assert (n, poly, alpha, rounds, layers, nact, dim, keys) == (8, 0x11B, 179, 12, 3, 1, 4, 1)
