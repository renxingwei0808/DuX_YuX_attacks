"""Y10 (W26 step 2, R8): the Boolean degree d_3 of the THREE-round plaintext-side
extension of Yu2X-16, and what it does to the 13-round Boolean-linearisation
route.

E08 established d_2 = 7 for the two-round extension, exactly and independently
of n, with `keypoly.KeyPoly` (the S-box is a short quadratic chain, so
S(P + k) is an explicit polynomial in the four key words of its block and the
Boolean degree is max_monomials sum_i HW(e_i)).

Three rounds put a FULL S layer between the two the two-round equation uses:

    Y  = SL(P + rk^0)                       inner layer,  k = rk^0 symbolic
    u  = L(Y) + rk^1
    W' = SL(u)                              the extra layer
    v  = L(W') + rk^2
    eq = sum_P [ v3 + v0 v1 + v2 + alpha ]  the cheap coordinate of the last S

so d_3 = deg_2( sum_P v0 v1 ) up to the +-1 of the outer key words, and the two
factors are linear combinations of the SAME sixteen W' words.  The exact
support of a degree-8 S coordinate in sixteen symbolic field variables is far
too large to enumerate, so this script computes

  * a LOWER bound: the same expression with only the CHEAP coordinate of the
    extra S layer kept (it is quadratic, so the support stays small).  The
    top-degree monomials of a product of two cheap coordinates from DIFFERENT
    inner blocks have disjoint variable supports, which is exactly the
    mechanism E08 identified for d_2 = 7, so no cancellation can remove them
    unless it is exact across coordinates -- flagged as an assumption;
  * an UPPER bound from the chain's degree arithmetic (deg of a product <= sum
    of degrees), capped by the number of key bits;

and then the verdict for 13-round Yu2X-16 under Ni et al.'s own cost model.

Usage
  python experiments/Y10_boolean_hybrid/d3.py --out results/Y10_boolean_hybrid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, os.path.join(_ROOT, "experiments", "E08_boolean_degree_extension")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dux.field import make_field                              # noqa: E402
from keypoly import KeyPoly, sbox_enc_chain_yux, YU2X_ROT_FWD  # noqa: E402

WORDS = 16
NV = 16                     # sixteen symbolic inner key WORDS (= 16 n bits)


def _cheap(x, alpha):
    """The cheap coordinate z0 = x3 + x0 x1 + x2 + alpha of the YuX encryption
    S-box (characteristic 2), as a KeyPoly."""
    return (x[3] + x[0] * x[1] + x[2]).add_const(alpha)


def three_layer_lower(F, alpha, nP=3, seed=11, blocks=4, j=0, progress=False):
    """Boolean degree of sum_P v0 v1 with the extra S layer represented by its
    CHEAP coordinate only (a lower bound for d_3).

    `blocks` inner S-box blocks are symbolic; blocks = 4 is the whole inner
    round key of the four blocks that feed one outer block, which is what E08's
    `two_layer_all_blocks` uses."""
    rng = np.random.default_rng(seed)
    rk0 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    rk1 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    acc = KeyPoly(F, None, NV)
    lin = [KeyPoly(F, None, NV) for _ in range(4)]
    for t in range(nP):
        P = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
        # --- layer 1: Y = SL(P + rk^0), rk^0 symbolic on `blocks` blocks -----
        Y = []
        for b in range(4):
            if b < blocks:
                xb = [KeyPoly.var_plus_const(F, 4 * b + i, P[4 * b + i], NV)
                      for i in range(4)]
            else:
                xb = [KeyPoly.const(F, F.add(P[4 * b + i], rk0[4 * b + i]), NV)
                      for i in range(4)]
            Y += list(sbox_enc_chain_yux(xb, alpha))
        # --- L, +rk^1 --------------------------------------------------------
        u = []
        for i in range(WORDS):
            a = KeyPoly(F, None, NV)
            for s in YU2X_ROT_FWD:
                a = a + Y[(i + s) % WORDS]
            u.append(a.add_const(rk1[i]))
        # --- the extra S layer, CHEAP coordinate per block -------------------
        Wp = [KeyPoly(F, None, NV) for _ in range(WORDS)]
        for b in range(4):
            z0 = _cheap(u[4 * b:4 * b + 4], alpha)
            Wp[4 * b + 3] = z0            # position 3 carries the cheap coordinate
        # --- L again ---------------------------------------------------------
        v = []
        for i in range(4):
            a = KeyPoly(F, None, NV)
            for s in YU2X_ROT_FWD:
                a = a + Wp[(4 * j + i + s) % WORDS]
            v.append(a)
        acc = acc + v[0] * v[1]
        for i in range(4):
            lin[i] = lin[i] + v[i]
        if progress:
            print(f"    structure {t + 1}/{nP}: support(sum v0 v1) = "
                  f"{acc.support_size()}, deg = {acc.boolean_degree()}", flush=True)
    return {"deg_sum_v0v1": acc.boolean_degree(),
            "support_sum_v0v1": acc.support_size(),
            "deg_sum_v": [t.boolean_degree() for t in lin],
            "top_monomials": [list(e) for e in acc.top_monomials()[:3]],
            "blocks_symbolic": blocks, "structures": nP}


def moebius_lower(instance="yu2x-16", nbits=20, layers=3, nP=4, seed=11,
                  words=None, progress=False):
    """EXACT Boolean degree of the three-layer equation restricted to `nbits`
    key bits of rk^0 -- a valid LOWER bound on d_3, computed on the REAL
    Yu2X-16 (not a toy), so no exponent wrap-around can distort it.

    The equation is the one the attack uses:
        sum_P [ v3 + v0 v1 + v2 + alpha ]  with  v = L(SL(L(SL(P + rk^0)) +
        rk^1)) + rk^2,
    i.e. the cheap coordinate of the third S layer, summed over an even-sized
    plaintext structure (so the top pure-key term cancels, exactly as in E08).

    `words` picks which rk^0 words carry the symbolic bits; the default spreads
    them over several BLOCKS, because E08 showed the Boolean degree of the
    two-layer equation comes from products across different inner blocks (4
    within one block, 7 across all of them)."""
    from dux.registry import get_cipher                            # noqa: E402
    c = get_cipher(instance)
    F = c.F
    rng = np.random.default_rng(seed)
    if words is None:                       # 5 words spread over 3 blocks
        words = [0, 1, 4, 5, 8]
    per = [nbits // len(words) + (1 if i < nbits % len(words) else 0)
           for i in range(len(words))]
    N = 1 << nbits
    idx = np.arange(N, dtype=np.int64)
    rk0 = np.array([int(v) for v in rng.integers(0, F.q, size=WORDS)], dtype=np.int64)
    rk1 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    rk2 = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
    # the symbolic key: word w gets `per` low bits from the index
    K = [np.full(N, int(rk0[w]), dtype=np.int64) for w in range(WORDS)]
    sh = 0
    for w, b in zip(words, per):
        K[w] = (np.int64(int(rk0[w]) & ~((1 << b) - 1))) | ((idx >> sh) & ((1 << b) - 1))
        sh += b
    acc = np.zeros(N, dtype=np.int64)
    for t in range(nP):
        P = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
        x = tuple(F.vadd(np.full(N, P[i], dtype=np.int64), K[i]) for i in range(WORDS))
        for l in range(layers):
            x = c.SL(x, True)
            if l == layers - 1:
                break
            x = c.lin.L(x, 0, True)
            rk = rk1 if l == 0 else rk2
            x = tuple(F.vadd(x[i], np.int64(rk[i])) for i in range(WORDS))
        # the equation reads the cheap coordinate of the LAST S layer, which is
        # z0 at position 3 of each block -- block 0 here
        acc ^= np.asarray(x[3], dtype=np.int64)
        if progress:
            print(f"    structure {t + 1}/{nP}", flush=True)
    degs = []
    for bit in range(F.n):
        f = ((acc >> bit) & 1).astype(np.uint8)
        g = f.copy()
        for i in range(nbits):                       # Moebius transform
            step = 1 << i
            g = g.reshape(-1, 2 * step)
            g[:, step:] ^= g[:, :step]
            g = g.reshape(-1)
        nz = np.flatnonzero(g)
        hw = np.zeros(nz.size, dtype=np.int64)
        v = nz.copy()
        while v.any():
            hw += (v & 1)
            v >>= 1
        degs.append(int(hw.max()) if nz.size else -1)
    return {"instance": instance, "nbits": nbits, "words": list(words),
            "bits_per_word": per, "structures": nP, "layers": layers,
            "degree_per_output_bit": degs, "degree": max(degs),
            "saturated": max(degs) == nbits}


def upper_bound(n=16):
    """Degree arithmetic of the chain, deg(product) <= sum of degrees.

    One layer: the Yu2X encryption S-box coordinates have Boolean degrees
    (4,3,2,2) in the key bits (E08 Sect. 3.1).  A linear layer does not raise
    the degree.  The second S layer's chain is
        A = x3 + x0 x1 + x2   -> <= 2 d
        B = x2 + A x0 + x1    -> <= 3 d
        C = x1 + B A + x0     -> <= 5 d
        D = x0 + C B + A      -> <= 8 d
    and the third layer's cheap coordinate is a product of two of those."""
    d1 = 4                                   # E08: coordinate 0 of one S layer
    second = [8 * d1, 5 * d1, 3 * d1, 2 * d1]
    dv = max(second)
    return {"one_layer": d1, "second_layer": second, "deg_v_max": dv,
            "d3_upper": min(2 * dv, 16 * n)}


def binom_le(n, d):
    return sum(math.comb(n, i) for i in range(d + 1))


def verdict(n_v=256, n=16, rounds=13, data_log2=64, eq_per_struct=16 * 16):
    """For each candidate d: unknowns, Ni's construction and solving costs.

    Ni et al. (DCC 2026, 94:123) Sect. 4.3, p. 17:
      T_c = 2^m * C(n_v,<=d)/(16 n)   decryptions,
      T_s = C(n_v,<=d)^w bit operations, w = 2.37, converted to full-cipher
            computations by dividing by (rounds * n * 192) bit operations.
    Their own 10-round Yu2X-8 line is the calibration: C(128,<=12) = 2^54.55,
    T_c = 2^95.5 and T_s = 2^115.3 (their Table 1, p. 4)."""
    out = []
    per_round_bits = n * 192
    for d in range(4, 17):
        U = binom_le(n_v, d)
        lU = math.log2(U)
        tc = data_log2 + lU - math.log2(eq_per_struct)
        ts_bits = 2.37 * lU
        ts = ts_bits - math.log2(rounds * per_round_bits)
        out.append({"d": d, "log2_unknowns": round(lU, 2),
                    "log2_Tc": round(tc, 2), "log2_Ts_bitops": round(ts_bits, 2),
                    "log2_Ts": round(ts, 2),
                    "under_2_128": tc < 128 and ts < 128})
    return out


def calibration():
    """Reproduce Ni et al.'s own 10-round Yu2X-8 numbers from their formulas."""
    U = binom_le(128, 12)
    lU = math.log2(U)
    return {"unknowns_C(128,<=12)": round(lU, 2),
            "paper_says": "2^54.6 (p. 17)",
            "log2_Tc": round(48 + lU - math.log2(128), 2),
            "paper_Tc": "2^95.6 (text, p. 17) / 2^95.5 (Table 1, p. 4)",
            "log2_Ts_bitops": round(2.37 * lU, 2),
            "paper_Ts_bitops": "2^129.3 (p. 17)",
            "log2_Ts_10round": round(2.37 * lU - math.log2(10 * 8 * 192), 2),
            "paper_Ts": "2^115.4 (text, p. 17) / 2^115.3 (Table 1, p. 4)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="2^16")
    ap.add_argument("--structures", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--instance", default="yu2x-16")
    ap.add_argument("--moebius-bits", type=int, nargs="*", default=[12, 16, 20])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    F = make_field(a.field)
    alpha = 205 % F.q if F.q < 206 else 205
    print("=== calibration against Ni et al.'s own 10-round Yu2X-8 line ===")
    cal = calibration()
    for k, v in cal.items():
        print(f"  {k:24s} {v}")
    print("\n=== d_3 lower bound A: exact Moebius on a subset of the real key bits ===")
    mob = {}
    for nb in a.moebius_bits:
        mob[nb] = moebius_lower(a.instance, nbits=nb, layers=3,
                                nP=a.structures, seed=a.seed, progress=a.progress)
        print(f"  {nb:2d} symbolic key bits (words {mob[nb]['words']}): "
              f"degree {mob[nb]['degree']} "
              f"{'(SATURATED: the bound is the restriction, not the cipher)' if mob[nb]['saturated'] else ''}")
    print("\n  control: the same routine at 2 layers (E08 measured d_2 = 7)")
    ctrl = {}
    for nb in a.moebius_bits:
        ctrl[nb] = moebius_lower(a.instance, nbits=nb, layers=2,
                                 nP=a.structures, seed=a.seed)
        print(f"  {nb:2d} bits, 2 layers: degree {ctrl[nb]['degree']}")
    print("\n=== d_3 lower bound B: KeyPoly with the cheap coordinate only ===")
    low = three_layer_lower(F, alpha, nP=a.structures, seed=a.seed,
                            blocks=a.blocks, progress=a.progress)
    for k, v in low.items():
        print(f"  {k:22s} {v}")
    low["moebius"] = {str(k): v for k, v in mob.items()}
    low["moebius_control_2_layers"] = {str(k): v for k, v in ctrl.items()}
    low["d3_lower"] = max([low["deg_sum_v0v1"]] + [v["degree"] for v in mob.values()])
    print("\n=== d_3 upper bound (degree arithmetic of the chain) ===")
    up = upper_bound(F.n if F.char == 2 else 16)
    for k, v in up.items():
        print(f"  {k:22s} {v}")
    print("\n=== 13-round Yu2X-16 under Ni et al.'s cost model (n_v = 256) ===")
    rows = verdict()
    print("  d | log2 unknowns | log2 T_c | log2 T_s | < 2^128 ?")
    for r in rows:
        print(f"  {r['d']:2d} | {r['log2_unknowns']:13.2f} | {r['log2_Tc']:8.2f} "
              f"| {r['log2_Ts']:8.2f} | {r['under_2_128']}")
    largest = max((r["d"] for r in rows if r["under_2_128"]), default=None)
    print(f"\n  the route needs d_3 <= {largest}; the measured lower bound is "
          f"{low['d3_lower']}")
    res = {"calibration": cal, "lower_bound": low, "upper_bound": up,
           "verdict_table": rows, "largest_d_under_2_128": largest,
           "closed": low["d3_lower"] > (largest or 0)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "d3.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.out}/d3.json")
    return res


if __name__ == "__main__":
    main()
