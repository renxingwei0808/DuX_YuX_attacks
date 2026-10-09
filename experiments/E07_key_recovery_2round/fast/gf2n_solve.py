"""Python side of the C blocked Gauss-Jordan over F_{2^n} (`gf2nsolve`).

Same signature and return value as `modp_solve.solve_gf2n`, so the two are
drop-in interchangeable and can be cross-checked against each other
(tests/test_gf2n_solve.py).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(HERE, "gf2nsolve")


def available(binary=None):
    return os.path.exists(binary or DEFAULT_BIN)


def solve(F, rows, rhs, binary=None, block=256, progress=None, threads=None):
    """({column: value}, rank, #free) for the system [rows | rhs] over F_{2^n}.

    The eliminator is OpenMP-parallel, but it is usually called from a driver
    whose *assembly* phase runs many single-threaded worker processes (so the
    driver sets OMP_NUM_THREADS=1).  The subprocess therefore gets its own
    thread count: `threads`, else $DUX_SOLVE_THREADS, else every core."""
    binary = binary or DEFAULT_BIN
    nthreads = threads or int(os.environ.get("DUX_SOLVE_THREADS", 0)) or os.cpu_count()
    A = np.ascontiguousarray(np.asarray(rows, dtype=np.uint16))
    b = np.asarray(rhs, dtype=np.uint16).reshape(-1, 1)
    Aug = np.ascontiguousarray(np.hstack([A, b]))
    del A, b
    m, ncols = Aug.shape[0], Aug.shape[1] - 1
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="duxgf2n_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"DUXGF2N1")
            f.write(struct.pack("<4I", F.n, m, ncols, block))
            f.write(np.asarray(F._log, dtype=np.uint16).tobytes())
            f.write(np.asarray(F._exp, dtype=np.uint16).tobytes())
            f.write(Aug.tobytes())
        env = dict(os.environ, OMP_NUM_THREADS=str(nthreads))
        proc = subprocess.run([binary, path], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"gf2nsolve failed with code {proc.returncode}")
        if progress:
            sys.stderr.write(proc.stderr)
    finally:
        os.unlink(path)
    det, rank, nfree = {}, 0, 0
    for line in proc.stdout.splitlines():
        q = line.split()
        if q[0] == "rank":
            rank, nfree = int(q[1]), int(q[3])
        elif q[0] == "D":
            det[int(q[1])] = int(q[2])
    return det, rank, nfree
