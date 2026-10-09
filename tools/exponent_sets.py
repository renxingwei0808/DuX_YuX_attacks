"""Univariate exponent-set propagation for DuX(2^n), both directions.

Ports the machinery of Ni-Wang-Li (DCC 2026, Sect. 3.1) to DuX: every
state word is a polynomial in the single active ciphertext word X; we track
an upper bound E(word) on its exponent set through S^{-1} (Prop. 1:
addition -> union, multiplication -> sumset mod 2^n-1 with the convention
that a nonzero multiple of 2^n-1 stays 2^n-1) and through L^{-1} (union).
The algebraic-degree bound of a word is max_{e in E} HW(e).

Representation: an exponent set is (has_zero, mask) where mask is a boolean
numpy array of length N = 2^n-1 and mask[i] <=> exponent i+1 present.  The
sumset of two masks is a cyclic OR-convolution, done with an FFT (O(N log N)
per product, fine for n = 16).

The output is the analogue of Ni et al. Table 3 for DuX.  It is an upper
bound: the general monomial prediction (experiments/E05) can only be tighter,
and real zero-sum experiments (E04) tighter still.

In the ENCRYPTION direction (--direction enc, W13-B) the same machinery is
run through S and the forward L, and it is then more than a degree bound: it
DECIDES balancedness of a full-field single-word structure.  For q = 2^n,

    sum_{X in F_q} X^e  =  0   for e = 0 (q terms) and for every e > 0 that is
                               not a positive multiple of q - 1,
                        =  1   when (q-1) | e, e > 0,

so a word whose exponent set (an upper bound, i.e. a superset) does not contain
the reduced exponent q - 1 is PROVABLY balanced.  This is what turns the four
"measurement beats the criterion" cells of results/E10_cpa into theorems: the
max-plus criterion O7 uses the UNREDUCED formal degree, so it loses exactly
when the formal degree crosses q and the Frobenius relation x^q = x folds it
back; the exponent set does that reduction automatically (mod q - 1).

Usage:
  python tools/exponent_sets.py --n 8  --pos 3 --layers 8
  python tools/exponent_sets.py --n 16 --pos 3 --layers 11 --json out.json
  python tools/exponent_sets.py --n 16 --pos 1 --layers 8 --direction enc
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dux.params import ROT_FWD_BIN, ROT_INV  # noqa: E402


class ExpSet:
    __slots__ = ("z", "m")

    def __init__(self, n, z=False, m=None):
        self.z = z
        self.m = np.zeros((1 << n) - 1, dtype=bool) if m is None else m

    @classmethod
    def const(cls, n):
        return cls(n, True, None)

    @classmethod
    def affine(cls, n):  # X + c : exponents {0, 1}
        e = cls(n, True, None)
        e.m[0] = True
        return e

    def union(self, other):
        return ExpSet(0, self.z or other.z, self.m | other.m)

    def degree(self):
        idx = np.nonzero(self.m)[0]
        if len(idx) == 0:
            return 0 if self.z else -1
        hw = np.array([bin(int(i) + 1).count("1") for i in idx])
        return int(hw.max())

    def max_exponent(self):
        idx = np.nonzero(self.m)[0]
        return int(idx.max()) + 1 if len(idx) else 0

    def size(self):
        return int(self.m.sum()) + (1 if self.z else 0)


def product(a: ExpSet, b: ExpSet) -> ExpSet:
    N = len(a.m)
    out = np.zeros(N, dtype=bool)
    if a.z:
        out |= b.m
    if b.z:
        out |= a.m
    if a.m.any() and b.m.any():
        fa = np.fft.rfft(a.m.astype(np.float64))
        fb = np.fft.rfft(b.m.astype(np.float64))
        conv = np.fft.irfft(fa * fb, n=N)
        present = conv > 0.5
        # exponents (i+1)+(j+1) = (i+j)+2 -> index (i+j+1) mod N   (see docstring)
        out |= np.roll(present, 1)
    return ExpSet(0, a.z and b.z, out)


def sbox_inv_sets(x, n):
    x0, x1, x2, x3 = x
    c = ExpSet.const(n)
    f = product(x0, x1).union(x3).union(c)       # y3
    g = product(x1, x2).union(x0).union(c)       # y0
    y1 = product(x2, f).union(x1).union(c)
    y2 = product(f, g).union(x2).union(c)
    return [g, y1, y2, f]


def sbox_enc_sets(x, n):
    """Forward S = (R^{-1})^4 (dux/sbox.py, degrees (5, 3, 2, 8)):
        a  = x2 - x0 x3 - alpha      b  = x1 - x3 a - alpha
        c  = x0 - a b  - alpha       y3 = x3 - b c - alpha
    and the output block is (c, b, a, y3) at positions 0..3."""
    x0, x1, x2, x3 = x
    k = ExpSet.const(n)
    a = product(x0, x3).union(x2).union(k)
    b = product(x3, a).union(x1).union(k)
    c = product(a, b).union(x0).union(k)
    y3 = product(b, c).union(x3).union(k)
    return [c, b, a, y3]


def linear_inv_sets(words, rots, n):
    out = []
    for i in range(16):
        acc = ExpSet.const(n)
        for j in rots:
            acc = acc.union(words[(i + j) % 16])
        out.append(acc)
    return out


def balanced_flags(words, n):
    """Provably balanced over the full field F_{2^n}: the reduced exponent
    q - 1 (mask index 2^n - 2) is absent from the exponent-set superset."""
    return [not bool(w.m[(1 << n) - 2]) for w in words]


def run(n, pos, layers, t_seq=None, direction="dec"):
    assert direction in ("dec", "enc")
    sbox = sbox_inv_sets if direction == "dec" else sbox_enc_sets
    rot = ROT_INV if direction == "dec" else ROT_FWD_BIN
    words = [ExpSet.const(n) for _ in range(16)]
    words[pos] = ExpSet.affine(n)
    rows = []
    for k in range(layers):
        outs = []
        for b in range(4):
            outs += sbox(words[4 * b:4 * b + 4], n)
        bal = balanced_flags(outs, n)
        rows.append({"layer": k + 1,
                     "degree": [w.degree() for w in outs],
                     "max_exponent": [w.max_exponent() for w in outs],
                     "set_size": [w.size() for w in outs],
                     "balanced": bal,
                     "pattern": "".join("1" if all(bal[4 * b + p] for b in range(4))
                                        else "0" for p in range(4)),
                     "n_balanced": sum(bal)})
        t = 0 if t_seq is None else t_seq[k]
        words = linear_inv_sets(outs, rot[t], n)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--pos", type=int, default=3)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--direction", choices=["dec", "enc"], default="dec")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = run(a.n, a.pos, a.layers, direction=a.direction)
    what = "S^-1" if a.direction == "dec" else "S"
    print(f"DuX(2^{a.n}), active word {a.pos}, {a.direction}: algebraic-degree upper "
          f"bound per word after each {what} layer")
    print("layer | degree bound words 0..3 (block 0) | max/16 | min/16 | "
          "proved-balanced pattern (count)")
    for r in rows:
        d = r["degree"]
        print(f"{r['layer']:5d} | {str(d[:4]):16s} | {max(d):4d} | {min(d):4d} | "
              f"{r['pattern']} ({r['n_balanced']:2d}/16)")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"n": a.n, "pos": a.pos, "direction": a.direction, "rows": rows},
                  open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
