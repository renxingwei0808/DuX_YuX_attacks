"""E03 -- zero-sum distinguishers for DuX(p) in the decryption direction.

For each active ciphertext position `pos` we build the set of ciphertexts
    C = (c_0, ..., c_{pos-1}, X, c_{pos+1}, ..., c_15),  X in F_p  (all p values)
with the other 15 words random constants, run `layers` S^{-1} layers of
decryption (round keys rk^r, rk^{r-1}, ...) and test, after every layer,
which of the 16 state words sum to 0 mod p.

Theory (docs/glossary.md O3): the sum over all of F_p of X^d is 0 for
0 <= d < p-1, so a word whose univariate degree in X is < p-1 is balanced.
tools/maxplus.py predicts (upper bound on degree => lower bound on layers):
    pos mod 4 in {2,3}: all 16 words balanced through 9 layers
    pos mod 4 in {0,1}: all 16 words through 8 layers, 12 words at layer 9
Anything beyond that means the true polynomials have cancellations.

Subgroup mode (--subgroup k, Beyne et al. CRYPTO 2020 style): X ranges over
a coset a*H of the order-2^k subgroup H of F_p^* (needs 2^k | p-1).  The sum
over a coset of X^d is 0 unless 2^k | d, and the constant term contributes
|H|*c != 0, so instead of "sum == 0" we test "sum is the same for two
different cosets".  Data 2^k instead of p.

Multi-word mode (--active "i,j,..."): every listed word runs over its own
coset (or over all of F_p if --subgroup is absent, which is only affordable
for one word); the structure is the product, enumerated in chunks of 2^chunk.
For a product of cosets, sum_{X in a1 H x ... x as H} prod X_i^{d_i} is zero
unless 2^k | d_i for EVERY i, so the same two-structure comparison applies.

Usage:
  python run.py --instance dux-65537 --layers 12 --keys 2 --positions all
  python run.py --instance dux-65537 --layers 8 --subgroup 13 --positions 3
  python run.py --instance dux-65537 --layers 9 --subgroup 12 --active 3,7   # 2^24 data
Outputs results/<out>/results.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dux import DuX  # noqa: E402


def subgroup_coset(p, k, rng):
    """Random coset a*H of the order-2^k subgroup H of F_p^*."""
    assert (p - 1) % (1 << k) == 0
    # find generator g of F_p^*
    from dux.field import _prime_factors
    fs = _prime_factors(p - 1)
    g = next(x for x in range(2, p) if all(pow(x, (p - 1) // f, p) != 1 for f in fs))
    h = pow(g, (p - 1) >> k, p)          # generator of H
    H = np.empty(1 << k, dtype=np.int64)
    v = 1
    for i in range(1 << k):
        H[i] = v
        v = (v * h) % p
    a = int(rng.integers(1, p))
    return (H * a) % p


def run_active_set(c, rks, active, layers, rng, subgroup, chunk_log2):
    """Product structure over `active` words; each word runs over a coset of
    the order-2^subgroup subgroup.  Two independent coset tuples are used and
    their layer sums compared (the coset analogue of "sum == 0")."""
    p = c.F.p
    assert subgroup is not None, "multi-word mode needs --subgroup (p^s is too much data)"
    consts = [int(v) for v in rng.integers(0, p, size=16)]
    s = len(active)
    k = subgroup
    total_log2 = k * s
    chunk_log2 = min(chunk_log2, total_log2)
    sums_per_set = []
    for _struct in range(2):
        vals = [subgroup_coset(p, k, rng) for _ in active]
        sums = [np.zeros(16, dtype=np.int64) for _ in range(layers)]
        csize = 1 << chunk_log2
        for start in range(0, 1 << total_log2, csize):
            idx = np.arange(start, start + csize, dtype=np.int64)
            C = [np.full(csize, consts[i], dtype=np.int64) for i in range(16)]
            for j, w in enumerate(active):
                C[w] = vals[j][(idx >> (k * j)) & ((1 << k) - 1)]
            states = c.decrypt_layers(tuple(C), rks, layers, vec=True, yield_all=True)
            for l, st in enumerate(states):
                for i in range(16):
                    sums[l][i] = (sums[l][i] + int(np.sum(st[i] % p) % p)) % p
        sums_per_set.append(sums)
    return [[int(sums_per_set[0][l][i]) == int(sums_per_set[1][l][i]) for i in range(16)]
            for l in range(layers)]


def run_position(c, rks, pos, layers, rng, subgroup=None):
    p = c.F.p
    consts = [int(v) for v in rng.integers(0, p, size=16)]
    if subgroup is None:
        Xs = [np.arange(p, dtype=np.int64)]
    else:
        Xs = [subgroup_coset(p, subgroup, rng), subgroup_coset(p, subgroup, rng)]
    sums_per_set = []
    for X in Xs:
        N = len(X)
        C = tuple(np.full(N, consts[i], dtype=np.int64) if i != pos else X for i in range(16))
        states = c.decrypt_layers(C, rks, layers, vec=True, yield_all=True)
        sums_per_set.append([[c.F.vsum(w) for w in st] for st in states])
    # balanced[l][i] : word i balanced after layer l+1
    balanced = []
    for l in range(layers):
        if subgroup is None:
            balanced.append([sums_per_set[0][l][i] == 0 for i in range(16)])
        else:
            balanced.append([sums_per_set[0][l][i] == sums_per_set[1][l][i] for i in range(16)])
    return balanced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--positions", default="all")
    ap.add_argument("--active", default=None,
                    help="comma-separated active words for the multi-word product mode "
                         "(requires --subgroup); overrides --positions")
    ap.add_argument("--chunk", type=int, default=20, help="log2 chunk size (multi-word mode)")
    ap.add_argument("--subgroup", type=int, default=None, help="use cosets of the order-2^k subgroup")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = DuX(a.instance)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    if a.active is not None:
        active = [int(v) for v in a.active.split(",")]
        agg = None
        for _ in range(a.keys):
            K = c.random_key(rng)
            rks = c.key_schedule(K)
            b = run_active_set(c, rks, active, a.layers, rng, a.subgroup, a.chunk)
            agg = b if agg is None else [[x and y for x, y in zip(r1, r2)] for r1, r2 in zip(agg, b)]
        counts = [sum(r) for r in agg]
        max_full = max((l + 1 for l in range(a.layers) if counts[l] == 16), default=0)
        last = max((l for l in range(a.layers) if counts[l] > 0), default=-1)
        pattern = "".join("1" if v else "0" for v in agg[last]) if last >= 0 else "-"
        res = {"instance": a.instance, "active": active, "subgroup": a.subgroup,
               "data_log2": a.subgroup * len(active), "layers": a.layers, "keys": a.keys,
               "seed": a.seed, "balanced_count_per_layer": counts,
               "max_layers_all16": max_full, "last_nontrivial_layer": last + 1,
               "pattern_last": pattern, "balanced_matrix": agg,
               "elapsed_s": round(time.time() - t0, 1)}
        print(f"{a.instance} active={active} subgroup=2^{a.subgroup} "
              f"data=2^{a.subgroup * len(active)} keys={a.keys}")
        print(f"  balanced words per layer: {counts}")
        print(f"  all-16 up to layer {max_full}; last nontrivial layer {last + 1}: {pattern}")
        print(f"  {res['elapsed_s']} s")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fn = os.path.join(a.out, f"active_{'_'.join(map(str, active))}_sg{a.subgroup}.json")
            json.dump(res, open(fn, "w"), indent=1)
            print("saved", fn)
        return

    positions = list(range(16)) if a.positions == "all" else [int(v) for v in a.positions.split(",")]
    results = {"instance": a.instance, "layers": a.layers, "keys": a.keys, "subgroup": a.subgroup,
               "seed": a.seed, "per_position": {}}
    print(f"{a.instance}: data = {'p=' + str(c.F.p) if a.subgroup is None else '2^' + str(a.subgroup) + ' x2 cosets'}")
    print("pos | layer -> #balanced words (16 = full zero-sum); pattern shown for the last nontrivial layer")
    for pos in positions:
        agg = None
        for k in range(a.keys):
            K = c.random_key(rng)
            rks = c.key_schedule(K)
            b = run_position(c, rks, pos, a.layers, rng, a.subgroup)
            agg = b if agg is None else [[x and y for x, y in zip(r1, r2)] for r1, r2 in zip(agg, b)]
        counts = [sum(r) for r in agg]
        last = max((l for l in range(a.layers) if counts[l] > 0), default=-1)
        pattern = "".join("1" if v else "0" for v in agg[last]) if last >= 0 else "-"
        max_full = max((l + 1 for l in range(a.layers) if counts[l] == 16), default=0)
        results["per_position"][pos] = {"balanced_count_per_layer": counts,
                                        "max_layers_all16": max_full,
                                        "last_nontrivial_layer": last + 1,
                                        "pattern_last": pattern,
                                        "balanced_matrix": agg}
        print(f"{pos:3d} | {counts}  all16<= {max_full}  layer{last + 1}:{pattern}")
    results["elapsed_s"] = round(time.time() - t0, 2)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
        print("saved", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
