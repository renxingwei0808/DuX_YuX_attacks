"""E15 / W29 step 3 -- direct verification of the MIXED zero-sum criterion.

For every layer of the decryption of one mixed structure (a full-field word
times a multiplicative coset, `mixed.mixed_axes`) this sums each of the sixteen
state words, plain and with the admissible weights, and compares the observed
balanced pattern with the max-plus prediction of `tools/zero_sum_criterion.py`
with `--coset k,full`.  Every conclusion is taken on >= 2 random master keys,
as our protocol requires, and the first failing layer is reported with
its 16-bit pattern ("how tight is it").

Usage
  python zero_sum_mixed.py --instance toy-193 --active 3,7 --mixed 6,full \
      --layers 6 --weights 1,2,3 --seeds 2026,7,11 \
      --out ../../results/E15_mixed_coset/zero_sum_toy-193.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
for p_ in (ROOT, os.path.join(ROOT, "tools"), HERE):
    sys.path.insert(0, p_)

import mixed as MX                                              # noqa: E402
import zero_sum_criterion as zsc                                # noqa: E402
from dux.registry import get_cipher                             # noqa: E402


def structure(c, axes, active, consts):
    sizes = [len(v) for v in axes]
    total = 1
    for n in sizes:
        total *= n
    strides, acc = [0] * len(sizes), 1
    for j in range(len(sizes) - 1, -1, -1):
        strides[j] = acc
        acc *= sizes[j]
    idx = np.arange(total, dtype=np.int64)
    cols = {w: axes[j][(idx // strides[j]) % sizes[j]]
            for j, w in enumerate(active)}
    X = [cols[i] if i in cols else np.full(total, consts[i], dtype=np.int64)
         for i in range(16)]
    return tuple(X), cols[active[0]]


def layer_sums(c, rks, X, x0, layers, weights):
    """sums[l][t][i] = sum over the structure of x0^{weights[t]} * state_l[i]."""
    F = c.F
    states = c.decrypt_layers(X, rks, layers, vec=True, yield_all=True)
    cols = [np.ones(len(x0), dtype=np.int64)]
    for a in weights:
        col = np.ones(len(x0), dtype=np.int64)
        base, e = x0.copy(), a
        while e:
            if e & 1:
                col = (col * base) % F.p
            base = (base * base) % F.p
            e >>= 1
        cols.append(col)
    out = []
    for st in states:
        row = [[int((w * (st[i] % F.p) % F.p).sum() % F.p) for i in range(16)]
               for w in cols]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-193")
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--mixed", default="6,full")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--weights", default="1,2,3",
                    help="extra weights on the FIRST active word (the coset one)")
    ap.add_argument("--seeds", default="2026,7,11")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    active = [int(v) for v in a.active.split(",")]
    ks = MX.parse_mixed(a.mixed, len(active))
    weights = [int(v) for v in a.weights.split(",") if v != ""]
    seeds = [int(v) for v in a.seeds.split(",")]
    c = get_cipher(a.instance, rounds=a.layers + 1)
    F = c.F
    pred = zsc.patterns(F.q, active, a.layers, None, ks, "dec", "dux")
    print(f"{a.instance} (F_{F.q}): active {active}, structure {a.mixed}, "
          f"T = {pred['threshold']}, data 2^{pred['log2_data']}")
    print(f"  predicted patterns: {pred['patterns']}")

    per_key = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        axes = MX.mixed_axes(F, ks, seed)
        consts = [int(v) for v in rng.integers(0, F.q, size=16)]
        X, x0 = structure(c, axes, active, consts)
        sums = layer_sums(c, rks, X, x0, a.layers, weights)
        obs, wobs, bits = [], [], []
        for l in range(a.layers):
            bal = ["1" if all(sums[l][0][4 * b + p] == 0 for b in range(4)) else "0"
                   for p in range(4)]
            obs.append("".join(bal))
            bits.append("".join("1" if sums[l][0][i] == 0 else "0" for i in range(16)))
            wobs.append(["".join("1" if all(sums[l][t][4 * b + p] == 0
                                            for b in range(4)) else "0"
                                 for p in range(4))
                         for t in range(1, len(weights) + 1)])
        first_fail = next((l + 1 for l, o in enumerate(obs) if o != "1111"),
                          a.layers + 1)
        per_key.append({"seed": seed, "points": int(len(X[0])),
                        "observed": obs, "observed_16bit": bits,
                        "weighted_observed": wobs,
                        "first_failing_layer": first_fail,
                        "first_failing_pattern16": (bits[first_fail - 1]
                                                    if first_fail <= a.layers else None)})
        print(f"  seed {seed}: observed {obs}; first failing layer {first_fail} "
              f"({per_key[-1]['first_failing_pattern16']})")
        for t, w in enumerate(weights):
            print(f"      weight a = {w}: {wobs[a.layers - 1]} at layer {a.layers}"
                  f"  (all layers: {[x[t] for x in wobs]})")

    # The PLAIN sum of a mixed structure only has the full-field words'
    # threshold (a coset factor at exponent 0 is |H| != 0), so it is checked
    # against `patterns_plain`; the weighted sums with a >= 1 and 2^k not
    # dividing a are the ones the T = sum_i T_i statement covers.
    plain_pred = pred.get("patterns_plain", pred["patterns"])
    marg = pred["margin_per_position"]

    def covers(obs, want):
        return all(obs[p] == "1" for p in range(4) if want[p] == "1")

    sound_plain = all(all(covers(k["observed"][l], plain_pred[l])
                          for l in range(a.layers)) for k in per_key)
    tight_plain = all(k["observed"] == plain_pred for k in per_key)
    sound_w, tight_w = True, True
    for k in per_key:
        for t, w in enumerate(weights):
            for l in range(a.layers):
                want = "".join("1" if w < marg[l][p_] else "0" for p_ in range(4))
                got = k["weighted_observed"][l][t]
                if not covers(got, want):
                    sound_w = False
                if got != want:
                    tight_w = False
    print(f"  PLAIN sums: sound {sound_plain}, tight {tight_plain} "
          f"(threshold_plain = {pred.get('threshold_plain')})")
    print(f"  WEIGHTED sums (a >= 1): sound {sound_w}, tight {tight_w} "
          f"(threshold = {pred['threshold']}; a balanced at position p iff "
          f"a < T - D_p)")
    res = {"instance": a.instance, "p": F.q, "active": active, "mixed": a.mixed,
           "coset_spec": ks, "layers": a.layers, "weights": weights,
           "threshold": pred["threshold"],
           "threshold_plain": pred.get("threshold_plain"),
           "log2_data": pred["log2_data"],
           "predicted": pred["patterns"], "predicted_plain": plain_pred,
           "plain_sound": bool(sound_plain), "plain_tight": bool(tight_plain),
           "weighted_sound": bool(sound_w), "weighted_tight": bool(tight_w),
           "max_degree_per_position": pred["max_degree_per_position"],
           "margin_per_position": pred["margin_per_position"],
           "keys": per_key,
           "sound": bool(sound_plain and sound_w),
           "tight": bool(tight_plain and tight_w)}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
