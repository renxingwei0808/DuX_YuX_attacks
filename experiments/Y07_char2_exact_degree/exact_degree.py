"""Y07 / S14 -- the EXACT reduced degree of every state word, layer by layer.

Y04's `tightness.py` answers one bit per (layer, word): is the max-plus bound
D attained?  It does that with the single character sum at a* = q - 1 - D,

    sum_{x in F_q} x^a Z_i(x)  =  - [X^{q-1-a}] Z_i(x)          (O12 Sect. 5.3)

for a single-word full-field structure.  When the answer is "loose" -- which
S11 found for Yu2X-16 from layer 4 on, and which we traced to a
key-INDEPENDENT cancellation in characteristic 2 (y_0 + y_1 is constant on a
single-word structure, so the cubic term of S^{-1} collapses) -- the next
question is BY HOW MUCH.  Scanning a upward from q - 1 - D probes the
coefficients of X^D, X^{D-1}, ... in turn, so the FIRST nonzero sum is the
exact degree.  That is what this script does, for every word and every layer,
on several keys.

Cost is (D - D_exact + 1) character sums per (layer, word), each one pass over
the q-point structure; the powers are advanced by one multiplication per step
rather than recomputed.  For Yu2X-16 we expect a ~7 % gap, i.e. a few
thousand sums per cell over 2^16 points -- minutes, not hours.

    python3 experiments/Y07_char2_exact_degree/exact_degree.py \
        --instance yu2x-16 --pos 0 --layers 9 --keys 3 \
        --out results/Y07_char2_exact_degree --tag exact_yu2x-16_pos0
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))

from dux.registry import cipher_family, get_cipher     # noqa: E402
from cipher_degree import profile                      # noqa: E402
from zero_sum_criterion import threshold               # noqa: E402
from assemble_fast import pow_field_vec                # noqa: E402


def _sum(F, Z, w):
    if F.char == 2:
        return int(np.bitwise_xor.reduce(F.vmul(Z, w)))
    return int(np.sum((Z % F.p) * (w % F.p) % F.p) % F.p)


def _step(F, w, xs):
    return F.vmul(w, xs) if F.char == 2 else (w * xs) % F.p


def exact_degree(F, xs, Z, D, max_scan):
    """(exact degree, character sums used) of Z as a polynomial in the active
    word, given the max-plus upper bound D.  `None` if the scan hits its cap
    without finding a nonzero coefficient."""
    q = F.q
    a0 = q - 1 - D
    if a0 < 0:                       # the bound is vacuous (D > q - 1)
        return None, 0
    w = pow_field_vec(F, xs, a0)
    for t in range(max_scan + 1):
        if a0 + t > q - 2:
            return 0, t              # only the constant coefficient can remain
        if _sum(F, Z, w):
            return D - t, t + 1
        w = _step(F, w, xs)
    return None, max_scan + 1


def spectrum(F, xs, Z, amax):
    """The set of a in [0, amax] with a NONZERO character sum, i.e. the nonzero
    coefficients of Z at degrees q-1-a (S14 step 2)."""
    w = np.ones_like(xs)
    nz = set()
    for a in range(amax + 1):
        if _sum(F, Z, w):
            nz.add(a)
        w = _step(F, w, xs)
    return nz


def first_nonzero_weight(F, xw, Z, amax):
    """The smallest a >= 0 with sum_x x_w^a Z(x) != 0 (S14 step 4).

    For an O11 full-block structure the sum is over the whole of F_q^4, so the
    single-variable identity of `exact_degree` does not apply; what the O7
    criterion predicts instead is that the sum vanishes while D + 2a < T -- the
    weight coordinate is the block's position-3 CIPHERTEXT word, which after
    the free first S^{-1} has degree 2 in the new variables, so each power
    costs TWO degree units (Y06 Sect. 5).  Measuring the first surviving a
    tests exactly that factor of two."""
    w = np.ones_like(xw)
    for a in range(amax + 1):
        if _sum(F, Z, w):
            return a
        w = _step(F, w, xw)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yu2x-16")
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--first-layer", type=int, default=1,
                    help="skip the layers below this one (they are cheap but "
                         "uninteresting; 1 = all)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-scan", type=int, default=6000,
                    help="cap on the character sums per (layer, word)")
    ap.add_argument("--words", default=None, help="comma list (default: all 16)")
    ap.add_argument("--spectrum-layers", default=None,
                    help="comma list of layers on which to take the FULL "
                         "spectrum a = 0..q-2 instead of the scan (S14 step 2)")
    ap.add_argument("--free-block", type=int, default=None,
                    help="S14 step 4: instead of a single-word structure, let "
                         "the four ciphertext words of this block run over "
                         "F_q^4 (O11) and report the first weight a on the "
                         "block's position-3 word whose sum survives")
    ap.add_argument("--max-weight", type=int, default=64,
                    help="cap for the --free-block weight scan")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.free_block is not None:
        return full_block_main(a)

    c = get_cipher(a.instance, rounds=a.rounds)
    fam = cipher_family(a.instance)
    F = c.F
    q = F.q
    words = ([int(v) for v in a.words.split(",")] if a.words else list(range(16)))
    spec_layers = ({int(v) for v in a.spectrum_layers.split(",")}
                   if a.spectrum_layers else set())
    prof = profile([a.pos], a.layers, "dec", F.char == 2, fam)
    xs = np.arange(q, dtype=np.int64)
    rng = np.random.default_rng(a.seed)

    print(f"{a.instance} ({F.name}): active ciphertext word {a.pos}, layers "
          f"{a.first_layer}..{a.layers}, {len(words)} words, {a.keys} keys; "
          f"one {q}-point structure per key", flush=True)

    per_key, t0 = [], time.time()
    for k in range(a.keys):
        rks = c.key_schedule(c.random_key(rng))
        consts = [int(v) for v in rng.integers(0, q, size=16)]
        C = [xs if i == a.pos else np.full(q, consts[i], dtype=np.int64)
             for i in range(16)]
        states = c.decrypt_layers(tuple(C), rks, a.layers, vec=True,
                                  yield_all=True)
        rows = []
        for l, st in enumerate(states):
            if l + 1 < a.first_layer:
                continue
            for i in words:
                D = int(prof[l][i])
                rec = {"layer": l + 1, "word": i, "D_maxplus": D}
                if l + 1 in spec_layers:
                    nz = spectrum(F, xs, st[i], q - 2)
                    a0 = min(nz) if nz else None
                    rec["spectrum_nonzero"] = nz          # kept for the merge
                    rec["spectrum_first_nonzero_a"] = a0
                    rec["spectrum_zero_prefix"] = a0 if nz else q - 1
                    rec["spectrum_interior_zeros"] = (
                        (q - 1 - a0) - len(nz) if nz else 0)
                    rec["spectrum_has_interior_hole"] = bool(
                        nz and len(nz) != (q - 1 - a0))
                    rec["D_exact"] = (q - 1 - a0) if nz else 0
                else:
                    de, used = exact_degree(F, xs, st[i], D, a.max_scan)
                    rec["D_exact"] = de
                    rec["sums"] = used
                if rec["D_exact"] is not None and D:
                    rec["ratio"] = round(rec["D_exact"] / D, 4)
                rows.append(rec)
            print(f"  key {k} layer {l + 1}: "
                  + " ".join(str(r["D_exact"]) for r in rows
                             if r["layer"] == l + 1)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
        per_key.append(rows)

    n = len(per_key[0])
    merged, same = [], 0
    for j in range(n):
        vals = [per_key[k][j]["D_exact"] for k in range(a.keys)]
        base = dict(per_key[0][j])
        base["D_exact_per_key"] = vals
        base["key_independent"] = len(set(vals)) == 1
        same += base["key_independent"]
        if "spectrum_nonzero" in base:
            # a coefficient counts as a genuine interior zero only if it is
            # zero on EVERY key: with q = 2^16 a single key produces ~1 zero
            # by chance per cell, and the question S14 step 2 asks is whether
            # the zero set is an initial interval, i.e. whether any structural
            # zero sits above the leading term.
            per = [per_key[k][j]["spectrum_nonzero"] for k in range(a.keys)]
            union = set().union(*per)              # zero on all keys <=> not
            a0 = max(per_key[k][j]["spectrum_first_nonzero_a"]
                     for k in range(a.keys))       # in the union
            holes = sorted(x for x in range(a0 + 1, q - 1) if x not in union)
            base["spectrum_interior_zeros_per_key"] = [
                per_key[k][j]["spectrum_interior_zeros"] for k in range(a.keys)]
            base["common_interior_zeros"] = len(holes)
            base["common_interior_zero_a"] = holes[:64]
            base["common_first_nonzero_a"] = a0
            for k in range(a.keys):
                per_key[k][j].pop("spectrum_nonzero", None)
            base.pop("spectrum_nonzero", None)
        merged.append(base)
    if spec_layers:
        tot = sum(r.get("common_interior_zeros", 0) for r in merged)
        print(f"{a.instance}: {tot} coefficient(s) below the leading term are "
              f"zero on ALL {a.keys} keys "
              f"({'no initial interval' if tot else 'the zero set IS an initial interval'})")
    ok = [r for r in merged if r["D_exact"] is not None and r["D_maxplus"]]
    ratio = (sum(r["D_exact"] for r in ok) / sum(r["D_maxplus"] for r in ok)
             if ok else None)
    print(f"{a.instance}: {same}/{n} (layer, word) cells have the SAME exact "
          f"degree on all {a.keys} keys; sum(D_exact)/sum(D_maxplus) = "
          f"{ratio:.4f}" if ratio else "no cell in range")

    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "rounds": c.r, "active": [a.pos], "layers": a.layers,
           "first_layer": a.first_layer, "words": words, "keys": a.keys,
           "seed": a.seed, "max_scan": a.max_scan,
           "spectrum_layers": sorted(spec_layers),
           "data_log2": round(float(np.log2(q)), 2),
           "key_independent_cells": same, "cells": n,
           "ratio_exact_to_maxplus": (round(ratio, 4) if ratio else None),
           "rows": merged, "seconds": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"exact_{a.instance}_pos{a.pos}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)



def full_block_main(a):
    """S14 step 4: the O11 full-block structure and its weight budget."""
    c = get_cipher(a.instance, rounds=a.rounds)
    fam = cipher_family(a.instance)
    F = c.F
    q = F.q
    b0 = a.free_block
    fkey = f"2^{F.n}" if F.char == 2 else q
    T = threshold(fkey, [], None, None, [b0])
    prof = profile([], a.layers, "dec", F.char == 2, fam, [b0])
    words = ([int(v) for v in a.words.split(",")] if a.words else list(range(16)))
    N = q ** 4
    idx = np.arange(N, dtype=np.int64)
    # the weight word (position 3 of the free block) is listed FIRST so that it
    # is the slowest axis, exactly as attack_12round.py --free-block does
    act = [4 * b0 + 3] + [4 * b0 + i for i in range(3)]
    axes = [(idx // (q ** (3 - j))) % q for j in range(4)]
    xw = axes[0]
    rng = np.random.default_rng(a.seed)
    print(f"{a.instance} ({F.name}): O11 full block {b0} (ciphertext words "
          f"{act}), {N} = q^4 points, T = {T}, layers 1..{a.layers}, "
          f"{a.keys} keys", flush=True)

    per_key, t0 = [], time.time()
    for k in range(a.keys):
        rks = c.key_schedule(c.random_key(rng))
        consts = [int(v) for v in rng.integers(0, q, size=16)]
        C = [np.full(N, consts[i], dtype=np.int64) for i in range(16)]
        for j, wd in enumerate(act):
            C[wd] = np.ascontiguousarray(axes[j])
        states = c.decrypt_layers(tuple(C), rks, a.layers, vec=True,
                                  yield_all=True)
        rows = []
        for l, st in enumerate(states):
            if l + 1 < a.first_layer:
                continue
            for i in words:
                D = int(prof[l][i])
                marg = T - D
                pred = max(0, -(-marg // 2))     # a < marg/2 is predicted zero
                got = first_nonzero_weight(F, xw, st[i], a.max_weight)
                rows.append({"layer": l + 1, "word": i, "D_maxplus": D,
                             "margin": marg, "predicted_first_nonzero_a": pred,
                             "measured_first_nonzero_a": got,
                             "verdict": ("tight" if got == pred else
                                         "loose" if got is None or got > pred
                                         else "VIOLATION")})
            print(f"  key {k} layer {l + 1}: "
                  + " ".join(f"{r['measured_first_nonzero_a']}/"
                             f"{r['predicted_first_nonzero_a']}"
                             for r in rows if r["layer"] == l + 1)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
        per_key.append(rows)

    n = len(per_key[0])
    merged, bad = [], 0
    for j in range(n):
        vals = [per_key[k][j]["measured_first_nonzero_a"] for k in range(a.keys)]
        base = dict(per_key[0][j])
        base["measured_per_key"] = vals
        base["key_independent"] = len(set(vals)) == 1
        merged.append(base)
        bad += base["verdict"] == "VIOLATION"
    tight = sum(1 for r in merged if r["verdict"] == "tight")
    print(f"{a.instance}: {tight}/{n} (layer, word) cells hit the predicted "
          f"boundary exactly; {bad} VIOLATIONS (a weight the criterion calls "
          f"admissible but whose sum is nonzero)")
    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "rounds": c.r, "free_block": b0, "active": act, "threshold": T,
           "layers": a.layers, "first_layer": a.first_layer, "keys": a.keys,
           "seed": a.seed, "max_weight": a.max_weight,
           "data_log2": round(4 * float(np.log2(q)), 2),
           "cells": n, "tight_cells": tight, "violations": bad,
           "rows": merged, "seconds": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"fullblock_{a.instance}_b{b0}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
