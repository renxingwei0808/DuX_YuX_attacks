"""E04 -- higher-order differential (zero-sum) experiments for DuX(2^n).

Ciphertext structure: a set of `s` active words (--active "i,j,...").  Each
active word ranges over an F_2-affine subspace c + span(b_1..b_m) of
F_{2^n} with m = --dim (default n, i.e. the whole field); the structure is
the product, so the data is 2^{m*s}.  After each of the first `layers`
S^{-1} decryption layers we XOR all states and report which of the 16
words are 0 for every key tried.  A word is guaranteed 0 whenever its
Boolean degree (in the m*s active bits) is < m*s.

The structure is enumerated in chunks (--chunk 2^k) so 2^24-2^28 data fit
in a few GB.  Above ~2^28 use the C kernel in fast/ on a server.

Usage examples:
  python run.py --instance dux-2^8  --active 0 --layers 12 --keys 3
  python run.py --instance dux-2^8  --active 0,1 --layers 12
  python run.py --instance dux-2^8  --active 0,4,8 --layers 10 --chunk 20
  python run.py --instance dux-2^16 --active 3 --layers 12
  python run.py --instance dux-2^16 --active 0,1 --dim 12 --layers 12   # 2^24 data
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dux import DuX  # noqa: E402


def gf2_rank(vs, n):
    rows = list(vs)
    rank = 0
    for bit in reversed(range(n)):
        piv = next((i for i in range(rank, len(rows)) if (rows[i] >> bit) & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> bit) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def random_basis(n, m, rng):
    while True:
        vs = [int(v) for v in rng.integers(1, 1 << n, size=m)]
        if gf2_rank(vs, n) == m:
            return vs


def span_values(basis, c):
    """All 2^m elements c + sum(subset of basis) as a numpy int64 array."""
    m = len(basis)
    vals = np.zeros(1 << m, dtype=np.int64)
    for i, b in enumerate(basis):
        step = 1 << i
        vals[step:2 * step] = vals[:step] ^ b
    return vals ^ c


def run_structure(c, rks, active, dim, layers, rng, chunk_log2):
    n = c.F.n
    consts = [int(v) for v in rng.integers(0, c.F.q, size=16)]
    per_word = []
    for _ in active:
        basis = random_basis(n, dim, rng)
        per_word.append(span_values(basis, int(rng.integers(0, c.F.q))))
    s = len(active)
    total_log2 = dim * s
    chunk_log2 = min(chunk_log2, total_log2)
    # enumerate the product space by splitting the index into per-word parts
    sums = [np.zeros(16, dtype=np.int64) for _ in range(layers)]
    idx_total = 1 << total_log2
    csize = 1 << chunk_log2
    for start in range(0, idx_total, csize):
        idx = np.arange(start, start + csize, dtype=np.int64)
        C = [None] * 16
        for i in range(16):
            C[i] = np.full(csize, consts[i], dtype=np.int64)
        for k, w in enumerate(active):
            sub = (idx >> (dim * k)) & ((1 << dim) - 1)
            C[w] = per_word[k][sub]
        states = c.decrypt_layers(tuple(C), rks, layers, vec=True, yield_all=True)
        for l, st in enumerate(states):
            for i in range(16):
                sums[l][i] ^= np.bitwise_xor.reduce(st[i])
    return [[int(sums[l][i]) == 0 for i in range(16)] for l in range(layers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^8")
    ap.add_argument("--active", default="0", help="comma-separated active word indices")
    ap.add_argument("--dim", type=int, default=None, help="F_2-dimension per active word (default n)")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=20, help="log2 chunk size")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = DuX(a.instance)
    n = c.F.n
    dim = n if a.dim is None else a.dim
    active = [int(v) for v in a.active.split(",")]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    agg = None
    for k in range(a.keys):
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        b = run_structure(c, rks, active, dim, a.layers, rng, a.chunk)
        agg = b if agg is None else [[x and y for x, y in zip(r1, r2)] for r1, r2 in zip(agg, b)]
    counts = [sum(r) for r in agg]
    max_full = max((l + 1 for l in range(a.layers) if counts[l] == 16), default=0)
    last = max((l for l in range(a.layers) if counts[l] > 0), default=-1)
    pattern = "".join("1" if v else "0" for v in agg[last]) if last >= 0 else "-"
    res = {"instance": a.instance, "active": active, "dim": dim, "data_log2": dim * len(active),
           "layers": a.layers, "keys": a.keys, "seed": a.seed,
           "balanced_count_per_layer": counts, "max_layers_all16": max_full,
           "last_nontrivial_layer": last + 1, "pattern_last": pattern,
           "balanced_matrix": agg, "elapsed_s": round(time.time() - t0, 1)}
    print(f"{a.instance} active={active} dim={dim} data=2^{dim * len(active)} keys={a.keys}")
    print(f"  balanced words per layer: {counts}")
    print(f"  all-16 zero-sum up to layer {max_full}; last nontrivial layer {last + 1}: {pattern}")
    print(f"  {res['elapsed_s']} s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"active_{'_'.join(map(str, active))}_dim{dim}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
