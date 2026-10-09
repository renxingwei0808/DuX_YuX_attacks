"""Y03 / W17 -- zero-sum (higher-order differential) experiments for YuX.

The YuX twin of experiments/E03 (F_p) and E04 (F_{2^n}), written once for both
fields and both ciphers through `dux.registry.get_cipher`, with the two R5
structure types added:

  --active i,j,...   the classical product structure: each listed ciphertext
                     word runs over all of F_q (or, in characteristic 2, over
                     an F_2-affine subspace of dimension --dim; over F_p, over
                     one coset of the order-2^k subgroup with --coset k).
  --full-block b     O11: the whole 4-word block b runs over F_q^4, so the
                     first S^{-1} layer is free (S^{-1} is a bijection of
                     F_q^4) and the criterion starts at degrees (1,1,1,1).
  --direction enc    the designers' chosen-plaintext model: the structure is
                     in the plaintext and the layers are encryption S layers.
  --weights a,...    O12: also report the WEIGHTED sums sum_x x^a Z(x) (the
                     weight rides on the first active word).  With a single
                     coset and a not a multiple of 2^k this is already a
                     zero-sum, so no signed coset difference is needed
                     (memo Sect. 5.5).

A word counts as balanced only if it is balanced for EVERY key tried
(protocol: >= 2 keys, 3 for a formal result).

Usage
  python experiments/Y03_zero_sum/run.py --instance yuxtoy-257 --active 0 --layers 8 --keys 3
  python experiments/Y03_zero_sum/run.py --instance yuxtoy-2^4 --full-block 0 --layers 6 --keys 3
  python experiments/Y03_zero_sum/run.py --instance yu2x-16 --active 0 --layers 10 --keys 2
  python experiments/Y03_zero_sum/run.py --instance yupx-65537 --active 0 --coset 15 \
      --weights 1,3,5 --layers 10 --keys 2
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

from dux.registry import cipher_family, get_cipher      # noqa: E402
from zero_sum_criterion import patterns                 # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "E04_zero_sum_F2n"))
from run import random_basis, span_values               # noqa: E402


def subgroup_coset(p, k, rng, rep=None):
    """One coset a*H of the order-2^k subgroup H of F_p^*."""
    assert (p - 1) % (1 << k) == 0
    from dux.field import _prime_factors
    fs = _prime_factors(p - 1)
    g = next(x for x in range(2, p) if all(pow(x, (p - 1) // f, p) != 1 for f in fs))
    h = pow(g, (p - 1) >> k, p)
    H = np.empty(1 << k, dtype=np.int64)
    v = 1
    for i in range(1 << k):
        H[i] = v
        v = (v * h) % p
    a = int(rng.integers(1, p)) if rep is None else rep
    return (H * a) % p


def axis_values(c, rng, dim=None, coset=None):
    """The value set one active word runs over, and its log2 size."""
    F = c.F
    if F.char == 2:
        m = F.n if dim is None else dim
        if m == F.n:
            return np.arange(F.q, dtype=np.int64), m
        basis = random_basis(F.n, m, rng)
        return span_values(basis, int(rng.integers(0, F.q))), m
    if coset:
        return subgroup_coset(F.p, coset, rng), coset
    return np.arange(F.q, dtype=np.int64), float(np.log2(F.q))


def build_axes(c, rng, active, free_blocks, dim, coset):
    """(list of (word, values), log2 data)."""
    F = c.F
    axes, bits = [], 0.0
    for b in free_blocks:
        for pos in range(4):
            axes.append((4 * b + pos, np.arange(F.q, dtype=np.int64)))
            bits += float(np.log2(F.q))
    for w in active:
        vals, m = axis_values(c, rng, dim, coset)
        axes.append((w, vals))
        bits += float(m)
    return axes, bits


def structure_chunks(c, axes, consts, chunk_log2=20):
    """Enumerate the product set in chunks; yields (words tuple, x0 values)."""
    sizes = [len(v) for _w, v in axes]
    total = 1
    for s in sizes:
        total *= s
    csize = min(total, 1 << chunk_log2)
    strides = []
    acc = 1
    for s in reversed(sizes):
        strides.append(acc)
        acc *= s
    strides = list(reversed(strides))
    for lo in range(0, total, csize):
        n = min(csize, total - lo)
        idx = np.arange(lo, lo + n, dtype=np.int64)
        cols = {}
        for t, (w, vals) in enumerate(axes):
            cols[w] = vals[(idx // strides[t]) % sizes[t]]
        X = []
        for i in range(16):
            X.append(cols[i] if i in cols else np.full(n, consts[i], dtype=np.int64))
        yield tuple(X), (cols[axes[0][0]] if axes else None)


def run_once(c, rks, axes, layers, consts, direction, weights, chunk_log2):
    """Per-layer sums of every word, plain and weighted."""
    F = c.F
    nw = 1 + len(weights)
    sums = [np.zeros((nw, 16), dtype=np.int64) for _ in range(layers)]
    for X, x0 in structure_chunks(c, axes, consts, chunk_log2):
        if direction == "enc":
            states = c.encrypt_layers(X, rks, layers, vec=True, yield_all=True)
        else:
            states = c.decrypt_layers(X, rks, layers, vec=True, yield_all=True)
        wcols = [np.ones(len(x0), dtype=np.int64)]
        for a in weights:
            col = np.ones(len(x0), dtype=np.int64)
            base = x0.copy()
            e = a
            while e:
                if e & 1:
                    col = F.vmul(col, base) if F.char == 2 else (col * base) % F.p
                base = F.vmul(base, base) if F.char == 2 else (base * base) % F.p
                e >>= 1
            wcols.append(col)
        for l, st in enumerate(states):
            for t, w in enumerate(wcols):
                for i in range(16):
                    z = st[i]
                    if F.char == 2:
                        v = int(np.bitwise_xor.reduce(F.vmul(w, z) if t else z))
                        sums[l][t][i] ^= v
                    else:
                        v = int((w * (z % F.p) % F.p).sum() % F.p) if t else \
                            int((z % F.p).sum() % F.p)
                        sums[l][t][i] = (sums[l][t][i] + v) % F.p
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yuxtoy-257")
    ap.add_argument("--active", default=None, help="comma list of active words")
    ap.add_argument("--full-block", default=None,
                    help="comma list of whole blocks running over F_q^4 (O11)")
    ap.add_argument("--dim", type=int, default=None,
                    help="char 2: dimension of each active word's affine subspace")
    ap.add_argument("--coset", type=int, default=None,
                    help="F_p: order-2^k subgroup coset per active word")
    ap.add_argument("--weights", default="",
                    help="O12: extra weights a (comma list) on the first active word")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--direction", default="dec", choices=("dec", "enc"))
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=20, help="log2 of the chunk size")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    active = [int(v) for v in a.active.split(",")] if a.active else []
    free = [int(v) for v in a.full_block.split(",")] if a.full_block else []
    weights = [int(v) for v in a.weights.split(",") if v.strip()]
    c = get_cipher(a.instance, rounds=a.rounds) if a.rounds else get_cipher(a.instance)
    F = c.F
    fam = cipher_family(a.instance)
    t0 = time.time()

    agg = None
    per_key = []
    bits = None
    for k in range(a.keys):
        rng = np.random.default_rng(a.seed + k)
        rks = c.key_schedule(c.random_key(rng))
        consts = [int(v) for v in rng.integers(0, F.q, size=16)]
        axes, bits = build_axes(c, rng, active, free, a.dim, a.coset)
        sums = run_once(c, rks, axes, a.layers, consts, a.direction, weights,
                        a.chunk)
        bal = [[[bool(sums[l][t][i] % (F.q if F.char == 2 else F.p) == 0)
                 for i in range(16)] for t in range(1 + len(weights))]
               for l in range(a.layers)]
        per_key.append(bal)
        agg = bal if agg is None else \
            [[[x and y for x, y in zip(r1, r2)] for r1, r2 in zip(t1, t2)]
             for t1, t2 in zip(agg, bal)]

    qkey = f"2^{F.n}" if F.char == 2 else F.q
    dims = ([a.dim] * len(active) if (F.char == 2 and a.dim) else None)
    pred = patterns(qkey, active, a.layers, dims, a.coset, a.direction, fam, free)

    # For a SINGLE coset the plain (a = 0) sum is NOT expected to vanish: the
    # constant term contributes |H| c != 0.  O12 Sect. 5.5 says a weight that is
    # not a multiple of 2^k already kills it, so the reference column for a
    # coset structure is the first nonzero weight.
    ref = 0
    if a.coset and weights:
        ref = 1 + next(t for t, w in enumerate(weights) if w % (1 << a.coset))
    ref_label = "weight 0" if ref == 0 else f"weight {weights[ref - 1]}"

    rows = []
    print(f"{a.instance} ({F.name}) {a.direction}: active {active} free blocks {free} "
          f"weights {weights}; data 2^{bits:.2f} per key, {a.keys} keys, seed {a.seed}")
    print(f"layer | measured pattern ({ref_label}) | #balanced | O7 prediction | margins")
    for l in range(a.layers):
        meas = agg[l][ref]
        pat = "".join("1" if all(meas[4 * b + p] for b in range(4)) else "0"
                      for p in range(4))
        rec = {"layer": l + 1,
               "balanced_matrix": "".join("1" if v else "0" for v in meas),
               "balanced_count": int(sum(meas)),
               "pattern": pat,
               "predicted": pred["patterns"][l],
               "margins": pred["margin_per_position"][l],
               "weighted": {str(w): {
                   "balanced_matrix": "".join("1" if v else "0" for v in agg[l][t + 1]),
                   "balanced_count": int(sum(agg[l][t + 1])),
                   "pattern": "".join("1" if all(agg[l][t + 1][4 * b + p]
                                                 for b in range(4)) else "0"
                                      for p in range(4))}
                   for t, w in enumerate(weights)}}
        D = [pred["threshold"] - m for m in rec["margins"]]
        rec["frobenius_boundary"] = [bool(d > 0 and (d & (d - 1)) == 0) for d in D]
        rows.append(rec)
        print(f" {l + 1:4d} | {rec['balanced_matrix']} | {rec['balanced_count']:2d}/16 | "
              f"{rec['predicted']} vs {pat} {'OK ' if pat == rec['predicted'] else 'DIFF'} | "
              f"{rec['margins']}")
    l_full = 0
    for r in rows:
        if r["balanced_count"] == 16:
            l_full += 1
        else:
            break
    first_fail = rows[l_full] if l_full < len(rows) else None
    print(f"  measured full zero-sum layers: {l_full} "
          f"(O7 predicts {pred['l_full']}); first failing layer "
          f"{l_full + 1} pattern {first_fail['pattern'] if first_fail else '-'} "
          f"(predicted {pred['next']})")
    per_weight = {}
    for t in range(1 + len(weights)):
        lab = "0" if t == 0 else str(weights[t - 1])
        lf = 0
        pats = []
        for l in range(a.layers):
            m = agg[l][t]
            pats.append("".join("1" if all(m[4 * b + p] for b in range(4)) else "0"
                                for p in range(4)))
            if sum(m) == 16 and lf == l:
                lf += 1
        per_weight[lab] = {"l_full": lf, "patterns": pats,
                           "agrees_with_O7": pats == pred["patterns"]}
    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "reference_weight": (0 if ref == 0 else weights[ref - 1]),
           "per_weight": per_weight,
           "direction": a.direction, "active": active, "free_blocks": free,
           "dim": a.dim, "coset": a.coset, "weights": weights,
           "seed": a.seed, "keys": a.keys, "data_log2": round(bits, 2),
           "layers": a.layers,
           "balanced_count_per_layer": [r["balanced_count"] for r in rows],
           "balanced_matrix": [r["balanced_matrix"] for r in rows],
           "per_layer": rows,
           "l_full_measured": l_full, "l_full_predicted": pred["l_full"],
           "next_measured": first_fail["pattern"] if first_fail else None,
           "next_predicted": pred["next"],
           "threshold": pred["threshold"],
           "agrees_with_O7": all(r["pattern"] == r["predicted"] for r in rows),
           "disagreements": [{"layer": r["layer"], "measured": r["pattern"],
                              "predicted": r["predicted"], "margins": r["margins"],
                              "frobenius_boundary": r["frobenius_boundary"]}
                             for r in rows if r["pattern"] != r["predicted"]],
           "elapsed_s": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or (f"{a.instance}_{a.direction}"
                        + (f"_fb{''.join(map(str, free))}" if free else "")
                        + (f"_a{'_'.join(map(str, active))}" if active else "")
                        + (f"_d{a.dim}" if a.dim else "")
                        + (f"_k{a.coset}" if a.coset else ""))
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
