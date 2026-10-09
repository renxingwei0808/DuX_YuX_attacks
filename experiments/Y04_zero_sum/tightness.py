"""Y04 / S11-C -- is the max-plus bound TIGHT, word by word and layer by layer?

O12 Sect. 5.3 gives the exact reduced degree of a state word from character
sums: for a single-word full-field structure and 0 <= a <= q-2,

    sum_{x in F_q} x^a Z_i(x)  =  - [X^{q-1-a}] Z_i(x) ,

so ONE sum at a* = q - 1 - D_i decides whether the max-plus bound D_i is
attained: nonzero <=> deg Z_i = D_i exactly (everything above D_i is already
known to vanish, that is what O7 proves).  `tools/degree_spectrum.py` scans a
upwards to find the whole spectrum, which costs O(q * max_a) and is only
affordable on toys; here we only need the single exponent per (layer, word),
which is 16 * layers character sums per key over one 2^16-point structure.

That turns "the 9-layer distinguisher is optimal" from a statement about an
upper bound into an exact statement, for F_{2^16} as well as F_65537 -- the
open item left by report Sect. 12 / Sect. 5.3 of the memo.

    python3 experiments/Y04_zero_sum/tightness.py --instance yupx-65537 --pos 0 \
            --layers 9 --keys 3 --out results/Y04_zero_sum
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux.registry import cipher_family, get_cipher     # noqa: E402
from cipher_degree import profile                      # noqa: E402


def char_sum(F, xs, Z, a):
    """sum_x x^a Z(x) over the whole field."""
    from assemble_fast import pow_field_vec            # noqa: E402
    w = pow_field_vec(F, xs, int(a))
    if F.char == 2:
        return int(np.bitwise_xor.reduce(F.vmul(Z, w)))
    return int(np.sum((Z % F.p) * (w % F.p) % F.p) % F.p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yupx-65537")
    ap.add_argument("--pos", type=int, default=0, help="the active ciphertext word")
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
    c = get_cipher(a.instance, rounds=a.rounds)
    fam = cipher_family(a.instance)
    F = c.F
    q = F.q
    prof = profile([a.pos], a.layers, "dec", F.char == 2, fam)
    xs = np.arange(q, dtype=np.int64)
    rng = np.random.default_rng(a.seed)

    per_key = []
    t0 = time.time()
    for k in range(a.keys):
        rks = c.key_schedule(c.random_key(rng))
        consts = [int(v) for v in rng.integers(0, q, size=16)]
        C = [xs if i == a.pos else np.full(q, consts[i], dtype=np.int64)
             for i in range(16)]
        states = c.decrypt_layers(tuple(C), rks, a.layers, vec=True, yield_all=True)
        rows = []
        for l, st in enumerate(states):
            for i in range(16):
                D = int(prof[l][i])
                if D > q - 2:                # no admissible exponent left
                    rows.append({"layer": l + 1, "word": i, "D": D,
                                 "a_star": None, "sum": None,
                                 "verdict": "out-of-range"})
                    continue
                astar = q - 1 - D
                s = char_sum(F, xs, st[i], astar)
                rows.append({"layer": l + 1, "word": i, "D": D, "a_star": astar,
                             "sum": s, "verdict": "tight" if s else "loose"})
        per_key.append(rows)
        print(f"  key {k}: {sum(1 for r in rows if r['verdict'] == 'tight')}"
              f"/{len(rows)} words tight  ({time.time() - t0:.1f}s)", flush=True)

    # a word/layer counts as tight only if it is tight for EVERY key
    n = len(per_key[0])
    agree, tight_all, loose_all, mixed = [], 0, 0, 0
    for j in range(n):
        vs = {per_key[k][j]["verdict"] for k in range(a.keys)}
        base = dict(per_key[0][j])
        base["verdicts"] = [per_key[k][j]["verdict"] for k in range(a.keys)]
        base["all_keys_agree"] = len(vs) == 1
        agree.append(base)
        if vs == {"tight"}:
            tight_all += 1
        elif vs == {"loose"}:
            loose_all += 1
        elif vs == {"out-of-range"}:
            pass
        else:
            mixed += 1
    inrange = sum(1 for r in agree if r["verdict"] != "out-of-range")
    print(f"{a.instance}: {tight_all}/{inrange} (layer, word) cells tight on all "
          f"{a.keys} keys; {loose_all} loose; {mixed} key-dependent")

    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "rounds": c.r, "active": [a.pos], "layers": a.layers,
           "keys": a.keys, "seed": a.seed,
           "data_log2": round(float(np.log2(q)), 2),
           "cells_in_range": inrange, "tight_all_keys": tight_all,
           "loose_all_keys": loose_all, "key_dependent": mixed,
           "cells": agree, "elapsed_s": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"tightness_{a.instance.replace('^', '')}_pos{a.pos}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
