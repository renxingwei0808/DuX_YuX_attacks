"""S22 -- the base case of Theorem S1 (O9, vanishing next-to-leading coefficient)
checked on many keys.

Theorem S1 of the supplement: for DuX over F_{2^n} in the encryption direction
with one active plaintext word, if the coefficient of X^{D-1} (D the max-plus
degree) vanishes in all 16 inputs of an S-box layer, it vanishes in every word
of all later layers.  The written proof derives the hypothesis at layer 4 for
active position 1 (S2-F).  This script checks it numerically, on random keys
and on keys with lambda = rk0_2 + rk0_0 rk0_3 + alpha = 0, for every active
position, using the exact top-window arithmetic of tools/top_coefficients.py
(window T = 4: the coefficients of X^D, X^{D-1}, X^{D-2}, X^{D-3}, exact).

For every S-box layer it records the INPUTS of the layer (the state words after
the linear layer and the round key) and the OUTPUTS, and counts the words whose
coefficient of X^{D-1} is nonzero.  For position 1 it also checks the two
structural facts of the written base case at layer 3: the inputs u_0 and u_3 of
block 0 share (D, w_0, w_1), and the y_3 outputs of blocks 1, 2, 3 have
w_0 = w_1 = w_2 = 0 (true degree at most D - 3).

Usage
  python experiments/E10_cpa/check_thmS1_base.py --instance dux-2^16 --layers 6 \
      --keys 200 --seed 1 --positions 0,1,2,3 --out results/E10_cpa/thmS1_base_dux-2^16.json
  python experiments/E10_cpa/check_thmS1_base.py --instance dux-2^16 --layers 6 \
      --keys 50 --seed 7 --degenerate --positions 1 --out results/E10_cpa/thmS1_base_dux-2^16_lambda0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import top_coefficients as TC          # noqa: E402
from dux import DuX                     # noqa: E402
from dux.params import WORDS            # noqa: E402


def propagate_io(c, rks, pos, layers, T):
    """Inputs and outputs of every S-box layer, as lists of TopPoly."""
    F = c.F
    words = [TC.TopPoly.const(F, int(rks[0][i]), T) for i in range(WORDS)]
    words[pos] = TC.TopPoly.affine(F, int(rks[0][pos]), T)
    row = TC.fwd_row(c, 0)
    ins, outs_all = [], []
    for layer in range(1, layers + 1):
        ins.append(list(words))
        outs = []
        for b in range(4):
            outs += TC.sbox_enc(F, c.alpha, words[4 * b:4 * b + 4], T)
        outs_all.append(outs)
        nxt = []
        for i in range(WORDS):
            acc = TC.TopPoly.const(F, 0, T)
            for j in range(WORDS):
                if row[j]:
                    acc = TC._add(F, acc, TC._smul(F, outs[(i + j) % WORDS], row[j]))
            nxt.append(TC._add(F, acc, TC.TopPoly.const(F, int(rks[layer][i]), T)))
        words = nxt
    return ins, outs_all


def lam(F, K, alpha):
    """lambda = k_2 + k_0 k_3 + alpha, the leading coefficient of word 3 after
    the first layer for active position 1 (characteristic 2)."""
    return F.add(F.add(int(K[2]), F.mul(int(K[0]), int(K[3]))), alpha)


def run(instance, layers, keys, seed, positions, degenerate=False, T=4):
    c = DuX(instance, rounds=layers + 2)
    F = c.F
    out = {"instance": instance, "q": int(F.q), "char": int(F.char), "layers": layers,
           "keys": keys, "seed": seed, "degenerate": degenerate, "top": T,
           "positions": {}}
    for pos in positions:
        rng = np.random.default_rng(seed + 1000 * pos)
        bad_in = [0] * (layers + 1)
        bad_out = [0] * (layers + 1)
        l3_block0_same = 0
        l3_blocks123_low = 0
        lam0 = 0
        t0 = time.time()
        for _ in range(keys):
            K = [int(v) for v in c.random_key(rng)]
            if degenerate:
                K[2] = F.add(F.mul(K[0], K[3]), c.alpha)      # forces lambda = 0
            if lam(F, K, c.alpha) == 0:
                lam0 += 1
            rks = c.key_schedule(K)
            ins, outs = propagate_io(c, rks, pos, layers, T)
            for l in range(1, layers + 1):
                bad_in[l] += sum(1 for u in ins[l - 1] if u.w[1] != 0)
                bad_out[l] += sum(1 for u in outs[l - 1] if u.w[1] != 0)
            if layers >= 3:
                u = ins[2]
                if (u[0].d, u[0].w[0], u[0].w[1]) == (u[3].d, u[3].w[0], u[3].w[1]):
                    l3_block0_same += 1
                y = outs[2]
                if all(y[4 * b + 3].w[0] == 0 and y[4 * b + 3].w[1] == 0
                       and y[4 * b + 3].w[2] == 0 for b in (1, 2, 3)):
                    l3_blocks123_low += 1
        clean_from = None
        for l in range(1, layers + 1):
            if all(bad_in[m] == 0 and bad_out[m] == 0 for m in range(l, layers + 1)):
                clean_from = l
                break
        out["positions"][str(pos)] = {
            "nonzero_next_to_leading_inputs": bad_in[1:],
            "nonzero_next_to_leading_outputs": bad_out[1:],
            "clean_from_layer": clean_from,
            "lambda_zero_keys": lam0,
            "layer3_block0_u0_u3_same_top_pair": l3_block0_same,
            "layer3_blocks123_y3_degree_at_most_D_minus_3": l3_blocks123_low,
            "elapsed_s": round(time.time() - t0, 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^16")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--keys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--positions", default="0,1,2,3")
    ap.add_argument("--degenerate", action="store_true",
                    help="force lambda = 0 (k_2 = k_0 k_3 + alpha); meaningful for position 1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    positions = [int(p) for p in a.positions.split(",")]
    res = run(a.instance, a.layers, a.keys, a.seed, positions, a.degenerate)
    print(f"{a.instance} (q = {res['q']}), {a.keys} keys"
          f"{' with lambda = 0 forced' if a.degenerate else ''}, layers 1..{a.layers}")
    for pos, r in res["positions"].items():
        cells = ", ".join(f"L{l + 1}:{i}/{o}" for l, (i, o) in enumerate(
            zip(r["nonzero_next_to_leading_inputs"], r["nonzero_next_to_leading_outputs"])))
        print(f"  position {pos}: nonzero X^(D-1) (inputs/outputs) {cells}; "
              f"clean from layer {r['clean_from_layer']}; lambda = 0 on {r['lambda_zero_keys']} keys; "
              f"layer-3 facts {r['layer3_block0_u0_u3_same_top_pair']}/{a.keys} and "
              f"{r['layer3_blocks123_y3_degree_at_most_D_minus_3']}/{a.keys}; {r['elapsed_s']} s")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
