"""R7 / W23 -- direct-summation check of the known-key zero-sum partitions.

Liu and Sun (eprint 2026/1907, Sect. 6-7) place an intermediate state
immediately after a diffusion layer, let the words of an active set A run over
F_q^s (a coset of the coordinate subspace V_A) and sum the plaintexts and the
ciphertexts of that coset in both directions.  `tools/zero_sum_criterion.py
--partition` predicts the largest split (r_b, r_f); this script does what
their Sect. 7 does on the toy instances: it sums directly at the three splits

    (r_b, r_f)        both sides must be zero-sum on all sixteen words,
    (r_b + 1, r_f)    the criterion guarantees nothing on the plaintext side,
    (r_b, r_f + 1)    the criterion guarantees nothing on the ciphertext side,

for >= 2 random master keys, with the round keys of the instance's own key
schedule (rounds = r_b + r_f (+1)) and the key-dependent diffusion layer of
DuX exercised as it comes.

Alignment (--align):
  after-L   (Liu-Sun)  state = L(SL(x_{r_b-1}))            backward groups (L^{-1}, S^{-1})
  before-L             state = SL(x_{r_b-1})               backward S^{-1} first (a whole
                                                            block is then free, O11), forward L first

Usage:
  python experiments/E09_unified_criterion/partition_check.py --instance toy-2^4 --active 0 --rb 4 --rf 3 --keys 2
  python experiments/E09_unified_criterion/partition_check.py --instance yuxtoy-2^4 --full-block 0 --align before-L --rb 5 --rf 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from dux.registry import get_cipher          # noqa: E402
import zero_sum_criterion as zsc             # noqa: E402


def coset(F, active, free_blocks, rng):
    """One coset of V_A: the active words (and the four words of every free
    block) run over F_q^s in product order, the other words are random
    constants.  Returns a tuple of 16 numpy arrays."""
    words = sorted(set(active) | {4 * b + i for b in free_blocks for i in range(4)})
    s = len(words)
    q = F.q
    N = q ** s
    idx = np.arange(N, dtype=np.int64)
    consts = [int(v) for v in rng.integers(0, q, size=16)]
    X = [np.full(N, consts[i], dtype=np.int64) for i in range(16)]
    for j, w in enumerate(words):
        X[w] = (idx // (q ** (s - 1 - j))) % q
    return tuple(X), words


def both_sides(c, rks, Y, rb, rf, align):
    """Plaintexts and ciphertexts of the coset Y placed at round rb."""
    R = rb + rf
    v = True
    if align == "after-L":
        x = c.SL_inv(c.lin.LM_inv(Y, rks[rb], v), v)           # group 1: L^{-1} then S^{-1}
        for i in range(rb - 1, 0, -1):
            x = c.SL_inv(c.lin.LM_inv(c.ARK_inv(x, rks[i], v), rks[i], v), v)
        P = c.ARK_inv(x, rks[0], v)
        x = c.ARK(Y, rks[rb], v)
    else:
        x = c.SL_inv(Y, v)                                      # S^{-1} first
        for i in range(rb - 1, 0, -1):
            x = c.SL_inv(c.lin.LM_inv(c.ARK_inv(x, rks[i], v), rks[i], v), v)
        P = c.ARK_inv(x, rks[0], v)
        x = c.ARK(c.lin.LM(Y, rks[rb], v), rks[rb], v)          # L first
    for i in range(rb + 1, R):
        x = c.ARK(c.lin.LM(c.SL(x, v), rks[i], v), rks[i], v)
    C = c.ARK(c.SL(x, v), rks[R], v)
    return P, C


def pattern(F, X):
    return "".join("1" if F.vsum(w) == 0 else "0" for w in X)


def check(instance, active, free_blocks, rb, rf, keys, seed, align):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(keys):
        c = get_cipher(instance, rounds=rb + rf)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        Y, words = coset(c.F, active, free_blocks, rng)
        P, C = both_sides(c, rks, Y, rb, rf, align)
        out.append({"key": k, "plaintext_side": pattern(c.F, P),
                    "ciphertext_side": pattern(c.F, C)})
    return out, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-2^4")
    ap.add_argument("--active", default="")
    ap.add_argument("--full-block", default=None)
    ap.add_argument("--rb", type=int, default=None, help="default: the tool's prediction")
    ap.add_argument("--rf", type=int, default=None)
    ap.add_argument("--align", default="after-L", choices=("after-L", "before-L"))
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    active = [int(v) for v in a.active.split(",") if v != ""]
    free = [int(v) for v in a.full_block.split(",")] if a.full_block else []
    c0 = get_cipher(a.instance)
    F = c0.F
    q = f"2^{F.n}" if F.char == 2 else F.q
    cipher = "yux" if a.instance.startswith(("yu", "yux")) else "dux"
    pred = zsc.partition(q, active, cipher, free, a.align)
    rb = pred["r_b"] if a.rb is None else a.rb
    rf = pred["r_f"] if a.rf is None else a.rf
    print(f"{a.instance} active={active} free_blocks={free} align={a.align}: "
          f"predicted r_b + r_f = {pred['r_b']} + {pred['r_f']}, T = {pred['threshold']}")
    t0 = time.time()
    res = {"instance": a.instance, "active": active, "free_blocks": free, "align": a.align,
           "seed": a.seed, "keys": a.keys, "predicted": pred, "splits": {}}
    for (b, f) in [(rb, rf), (rb + 1, rf), (rb, rf + 1)]:
        rows, words = check(a.instance, active, free, b, f, a.keys, a.seed, a.align)
        allP = all(r["plaintext_side"] == "1" * 16 for r in rows)
        allC = all(r["ciphertext_side"] == "1" * 16 for r in rows)
        res["splits"][f"{b}+{f}"] = {"rows": rows, "plaintext_all_balanced": allP,
                                    "ciphertext_all_balanced": allC}
        print(f"  split {b} + {f}: plaintext side {'all 16 balanced' if allP else 'NOT all balanced'}"
              f" {[r['plaintext_side'] for r in rows]}; ciphertext side "
              f"{'all 16 balanced' if allC else 'NOT all balanced'} {[r['ciphertext_side'] for r in rows]}")
    res["elapsed_s"] = round(time.time() - t0, 2)
    res["data_log2"] = pred["log2_data"]
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
