"""S18 (docs/measurement_protocol.md rule 1) -- `--procs 1` must really be one thread.

The protocol reports each attack that fits in 36 h as a run on ONE core whose
thread count, sampled with `ps -o nlwp` while it runs, never left 1.
`multiprocessing.Pool(1)` cannot satisfy that: a pool of one worker still
starts three bookkeeping threads in the parent (`_handle_workers`,
`_handle_tasks`, `_handle_results`), so the sample is 4.  `attack_12round.py`
therefore takes a serial path when `--procs 1` is asked for -- same
initializer, same task function, same shared-memory blocks, no fork.

What is pinned here: the serial path runs the tasks in this very process and
adds no thread, and the driver's two pool sites (characteristic 2 and prime
field) give the SAME system with `--procs 1` as with a pool.
"""
import json
import os
import subprocess
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for _p in (ROOT, E07, os.path.join(E07, "fast"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, _p)

import attack_12round as A12                                     # noqa: E402


def test_make_pool_takes_the_serial_path_for_one_process():
    """One process means no `Pool` object at all; anything else is a `Pool`."""
    seen = []
    with A12._make_pool(1, lambda x: seen.append(x), ("init",)) as pool:
        assert isinstance(pool, A12._SerialPool)
        assert seen == ["init"]
        assert list(pool.imap_unordered(lambda t: t * 2, [1, 2, 3])) == [2, 4, 6]
    from multiprocessing import pool as mp_pool
    with A12._make_pool(2, lambda: None, ()) as pool:
        assert isinstance(pool, mp_pool.Pool)


def test_the_serial_path_starts_no_thread():
    """The whole point of rule 1: `threading.active_count()` is unchanged
    while the tasks run, so `ps -o nlwp` stays at 1."""
    before = threading.active_count()
    inside = []
    with A12._make_pool(1, lambda: None, ()) as pool:
        for _ in pool.imap_unordered(lambda t: inside.append(
                threading.active_count()), range(4)):
            pass
    assert inside == [before] * 4
    assert threading.active_count() == before


def _run(instance, extra, procs, tmp_path, tag):
    out = subprocess.run(
        [sys.executable, os.path.join(E07, "attack_12round.py"),
         "--instance", instance, "--normalise", "--structures", "1",
         "--procs", str(procs), "--seed", "2026",
         "--out", str(tmp_path), "--tag", tag] + extra,
        capture_output=True, text=True, timeout=1800,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                 DUX_SOLVE_THREADS="1"))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.load(open(os.path.join(str(tmp_path), f"s10_{tag}.json")))


KEYS = ("equations", "unknowns", "rank", "free", "pinned", "inner_correct",
        "data_log2")


@pytest.mark.parametrize("instance,extra", [
    # the characteristic-2 pool site (`_block_moments_2n` + `_weight_acc_2n`)
    ("toy-2^4", ["--layers", "3", "--active", "3,7", "--weights", "6",
                 "--combine", "none", "--slice-chunk", "8",
                 "--block-slices", "2"]),
    # the prime-field pool site (`_block_moments` + the parent's dgemm)
    ("toy-257", ["--layers", "4", "--active", "3", "--weights", "40",
                 "--combine", "none", "--slice-chunk", "64",
                 "--block-slices", "8"]),
])
def test_procs1_gives_the_same_system_as_a_pool(instance, extra, tmp_path):
    """Both pool sites: one process and two processes must produce the same
    rows, hence the same rank, the same pinned monomials and the same key
    words.  A protocol rerun of a frozen row is only a protocol rerun if the
    numbers do not move."""
    one = _run(instance, extra, 1, tmp_path, "procs1")
    many = _run(instance, extra, 2, tmp_path, "procs2")
    assert {k: one[k] for k in KEYS} == {k: many[k] for k in KEYS}
    assert one["procs"] == 1 and many["procs"] == 2
