"""E10 / W11 -- chosen-PLAINTEXT (designer CPA model) zero-sum experiments.

The rest of this repository works in the chosen-ciphertext model, where the
S^{-1} coordinate degrees are (2,3,4,2) and L^{-1} is a 5-term rotation sum
that only mixes block positions {p, p+1}.  The designers' own security claim
("r >= 7 resists higher-order differentials") is about the ENCRYPTION
direction, where

  * S has coordinate degrees (5,3,2,8) at block positions (0,1,2,3), and
  * L0 is dense (the circulant M0 has all 16 entries nonzero over F_p, and its
    11-term rotation set over F_{2^n} already covers all four residues mod 4),

so from the second layer on every state word carries the same degree bound and
the growth rate is 8 per layer instead of 2 + sqrt(3).  The same criterion O7
applies (`tools/zero_sum_criterion.py --direction enc`): a word is balanced as
soon as its max-plus total degree D is below the threshold T.

This script measures the real thing: it encrypts full product-set structures
and reports, per layer, which of the 16 state words sum to zero.

    python run_zero_sum_cp.py --instance dux-65537 --active 1 --layers 8 --keys 2
    python run_zero_sum_cp.py --instance dux-2^8   --active 1,5 --layers 6 --keys 3
    python run_zero_sum_cp.py --instance toy-193   --active 1,5 --layers 6 --keys 3
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

from dux import DuX  # noqa: E402
import zero_sum_criterion as zsc  # noqa: E402


def structure(c, active, seed, chunk_log2=22):
    """Full product set on the active plaintext words; yields chunks of the
    16 plaintext arrays so that q^s > 2^22 points still fit in memory."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    s = len(active)
    total = F.q ** s
    csize = min(total, 1 << chunk_log2)
    for start in range(0, total, csize):
        idx = np.arange(start, start + min(csize, total - start), dtype=np.int64)
        P = [np.full(len(idx), consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            P[w] = (idx // (F.q ** t)) % F.q
        yield tuple(P)


def run(instance, active, layers, keys, seed, chunk_log2=22):
    c = DuX(instance, rounds=12)
    F = c.F
    out = []
    for ki in range(keys):
        rng = np.random.default_rng(seed + ki)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        acc = [[0] * 16 for _ in range(layers)]
        for P in structure(c, active, seed + 1000 + ki, chunk_log2):
            states = c.encrypt_layers(P, rks, layers, rounds=12, vec=True,
                                      yield_all=True)
            for l, st in enumerate(states):
                for i in range(16):
                    acc[l][i] = F.add(acc[l][i], F.vsum(st[i]))
        pats = []
        for l in range(layers):
            bits = "".join("1" if acc[l][i] == 0 else "0" for i in range(16))
            blk = "".join("1" if all(acc[l][4 * b + p] == 0 for b in range(4))
                          else "0" for p in range(4))
            pats.append({"layer": l + 1, "words": bits, "block_pattern": blk,
                         "balanced": bits.count("1")})
        out.append({"key_index": ki, "layers": pats})
    return c, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--active", default="1", help="active PLAINTEXT words")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=22)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    active = [int(v) for v in a.active.split(",")]
    t0 = time.time()
    c, res = run(a.instance, active, a.layers, a.keys, a.seed, a.chunk)
    F = c.F
    npts = F.q ** len(active)
    q = a.instance.split("-", 1)[1]
    pred = zsc.patterns(q if q.startswith("2^") else int(q), tuple(active),
                        a.layers, direction="enc")
    print(f"{a.instance} (CPA / encryption direction): active plaintext words "
          f"{active}, {npts} = 2^{np.log2(npts):.1f} chosen plaintexts, "
          f"{a.keys} keys, threshold T = {pred['threshold']}")
    print("layer | predicted (O7) | measured (all keys) | max total degree D | "
          "T - D")
    agree = True
    common = []
    for l in range(a.layers):
        meas = {r["layers"][l]["block_pattern"] for r in res}
        m = meas.pop() if len(meas) == 1 else "/".join(sorted(meas))
        common.append(m)
        p = pred["patterns"][l]
        ok = (len(meas) == 0 and m == p)
        agree &= ok
        print(f"{l + 1:5d} | {p:^14s} | {m:^19s} | "
              f"{str(pred['max_degree_per_position'][l]):>28s} | "
              f"{pred['margin_per_position'][l]}"
              + ("" if ok else "   <-- MISMATCH"))
    lfull = 0
    for m in common:
        if m == "1111":
            lfull += 1
        else:
            break
    print(f"measured l_full = {lfull}, first failing layer {lfull + 1} with "
          f"pattern {common[lfull] if lfull < len(common) else '?'}; "
          f"predicted l_full = {pred['l_full']}, next = {pred['next']}")
    print(f"prediction matches measurement on every layer: {agree}  "
          f"({time.time() - t0:.1f}s)")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"cp_zerosum_{a.instance}_"
                                 f"{'_'.join(map(str, active))}.json")
        json.dump({"instance": a.instance, "direction": "enc",
                   "active": active, "points_per_structure": npts,
                   "data_log2": round(float(np.log2(npts)), 2),
                   "keys": a.keys, "seed": a.seed,
                   "threshold": pred["threshold"],
                   "predicted_patterns": pred["patterns"][:a.layers],
                   "predicted_l_full": pred["l_full"],
                   "predicted_next": pred["next"],
                   "measured_l_full": lfull,
                   "measured": [r["layers"] for r in res],
                   "prediction_matches": bool(agree),
                   "elapsed_s": round(time.time() - t0, 1)},
                  open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
