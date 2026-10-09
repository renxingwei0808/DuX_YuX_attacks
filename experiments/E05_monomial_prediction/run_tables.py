"""E05 -- drive the general-monomial-prediction queries in parallel.

Each task is one (n, active words, layer, output word) query, run in its own
process with its own SMT timeout; results are appended to a JSONL file as they
finish, so a long run can be inspected (and resumed) at any time.

Two query modes (see gmp_dux.py):
  top     -- is the single maximal-degree monomial prod_j X_j^{2^n-1} reachable?
             UNSAT proves the zero-sum of that word over the full structure.
  degree  -- binary search for the largest achievable sum_j min(HW(e_j), dim).

Example
-------
  python run_tables.py --spec validate8 --procs 8 --timeout 900 \
      --out results/E05_monomial_prediction/gmp_n8.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import gmp_dux  # noqa: E402

# (n, active, from_layer, to_layer)
SPECS = {
    # validation against the E04/S1 zero-sum experiments on DuX(2^8)
    "validate8": [
        (8, [3], 3, 8),
        (8, [3, 7], 4, 8),
        (8, [3, 7, 11], 4, 8),
        (8, [3, 7, 11, 15], 4, 8),
    ],
    # the same structures on DuX(2^16) -- the extrapolation that matters
    "predict16": [
        (16, [3], 7, 11),
        (16, [3, 7], 8, 11),
        (16, [3, 7, 11], 8, 11),
        (16, [3, 7, 11, 15], 8, 11),
    ],
    "predict16_big": [
        (16, [3, 7, 11, 15, 2], 9, 11),
        (16, [3, 7, 11, 15, 2, 6], 9, 11),
    ],
    # the decisive cells: exactly the layers where the experiment shows the
    # zero-sum appearing or disappearing (long timeouts, run one per process)
    "decide4": [
        (4, [3, 7], 4, 4),
        (4, [3, 7, 11], 4, 4),
        (4, [3, 7, 11, 15], 4, 5),
    ],
    "decide8": [
        (8, [3], 5, 5),               # experiment: 5 full layers
        (8, [3, 7], 6, 6),            # experiment: layer 6 = `1001`
        (8, [3, 7, 11], 6, 6),        # experiment: layer 6 = `1101`
        (8, [3, 7, 11, 15], 6, 6),    # experiment: layer 6 = full zero-sum
    ],
    # positions 0/1 (mod 4), which E04 shows are one step behind
    "pos0": [
        (8, [0], 3, 7),
        (8, [0, 4], 4, 7),
        (16, [0], 7, 10),
    ],
}


def task(args):
    n, active, layer, word, mode, dim, timeout = args
    t0 = time.time()
    if mode == "top":
        r = gmp_dux.top_monomial(n, layer, active, word, timeout=timeout)
    else:
        r = gmp_dux.max_degree(n, layer, active, word, dim, timeout=timeout)
    r.update({"n": n, "active": active, "layer": layer, "word": word,
              "mode": mode, "dim": dim, "timeout": timeout,
              "seconds": round(time.time() - t0, 2)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="validate8", choices=sorted(SPECS))
    ap.add_argument("--mode", choices=["top", "degree"], default="top")
    ap.add_argument("--words", default="0,1,2,3")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    words = [int(v) for v in a.words.split(",")]
    tasks = []
    for n, active, l0, l1 in SPECS[a.spec]:
        for L in range(l0, l1 + 1):
            for w in words:
                tasks.append((n, active, L, w, a.mode, n, a.timeout))
    # cheapest first: fewer layers, fewer active words
    tasks.sort(key=lambda t: (t[2], len(t[1]), t[0]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"{len(tasks)} queries, {a.procs} processes, {a.timeout}s each", flush=True)
    with open(a.out, "w") as fh, Pool(a.procs) as pool:
        for r in pool.imap_unordered(task, tasks):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            head = f"top={r.get('reachable')}" if a.mode == "top" else f"D={r.get('degree')}"
            print(f"n={r['n']} act={r['active']} L={r['layer']} w={r['word']}: "
                  f"{head} [{r['status']}] {r['seconds']}s", flush=True)
    print("saved", a.out)


if __name__ == "__main__":
    main()
