"""Exact top-of-the-polynomial coefficients of DuX in the encryption direction.

Context (W13-B).  With one active plaintext word X and everything else fixed,
every state word is a univariate polynomial f(X) over F_q.  For a full-field
structure

    sum_{X in F_q} f(X)  =  eps * sum_{k >= 1, k(q-1) <= deg f} c_{k(q-1)},
    eps = 1 (q = 2^n),  eps = -1 (q = p),                                  (*)

because sum_X X^e is 0 unless e is a positive multiple of q - 1.  O7 compares
the max-plus formal degree D against T = q - 1 and concludes "balanced" when
D < T, i.e. when the sum (*) is empty.  The four cells where the E10 / S6
measurements beat O7 are exactly the cells with D = q: there (*) has the single
term c_{q-1} = c_{D-1}, and the criterion cannot see that this coefficient
vanishes.

`tools/exponent_sets.py --direction enc` does not help: in the encryption
direction the exponent sets are FULL INTERVALS [0, D] from the second layer on
(union with a sumset of intervals is an interval), so "q - 1 in E" is just
"D >= q - 1" and the exponent set carries no information beyond D.

This module computes the top `--top` coefficients of every state word EXACTLY
and decides (*) whenever every multiple k(q-1) <= D falls inside that window --
which is precisely the D = q case.  A word is represented by

    (d, w)   with  w[t] = coefficient of X^{d-t},  t = 0 .. T-1,

d the max-plus formal degree bound (so w[0] = 0 simply means the true degree is
smaller; nothing is approximated).  Addition aligns the two windows and
multiplication is the length-T Cauchy product -- both exact, because a
coefficient of X^{D-t} of a product only ever needs coefficients of X^{d-i},
i <= t, of the factors.

Usage
  python tools/top_coefficients.py --instance dux-2^16 --active 1 --layers 7
  python tools/top_coefficients.py --instance dux-2^8 --active 1 --layers 5 --keys 3
  python tools/top_coefficients.py --instance dux-2^8 --active 1 --layers 3 --show-window
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dux import DuX  # noqa: E402
from dux.params import ROT_FWD_BIN, WORDS  # noqa: E402


class TopPoly:
    """Formal degree bound `d` plus the exact coefficients of X^d .. X^{d-T+1}."""

    __slots__ = ("d", "w")

    def __init__(self, d, w):
        self.d, self.w = d, w

    @classmethod
    def const(cls, F, v, T):
        w = [0] * T
        w[0] = int(v) % (F.q if F.char == 2 else F.p)
        return cls(0, w)

    @classmethod
    def affine(cls, F, k, T):
        """X + k."""
        w = [0] * T
        w[0] = 1
        if T > 1:
            w[1] = int(k)
        return cls(1, w)

    def coeff(self, e):
        """Coefficient of X^e, valid for e > self.d - T (else raises)."""
        t = self.d - e
        if t < 0:
            return 0
        if t >= len(self.w):
            raise ValueError(f"exponent {e} is below the tracked window")
        return self.w[t]


def _add(F, u, v):
    T = len(u.w)
    d = max(u.d, v.d)
    w = [0] * T
    for x in (u, v):
        sh = d - x.d
        for t in range(sh, T):
            w[t] = F.add(w[t], x.w[t - sh])
    # exponents below zero cannot exist
    for t in range(T):
        if d - t < 0:
            w[t] = 0
    return TopPoly(d, w)


def _mul(F, u, v):
    T = len(u.w)
    d = u.d + v.d
    w = [0] * T
    for t in range(T):
        acc = 0
        for i in range(t + 1):
            a, b = u.w[i], v.w[t - i]
            if a and b:
                acc = F.add(acc, F.mul(a, b))
        w[t] = acc
    for t in range(T):
        if d - t < 0:
            w[t] = 0
    return TopPoly(d, w)


def _smul(F, u, c):
    c = int(c)
    if c == 0:
        return TopPoly(0, [0] * len(u.w))
    return TopPoly(u.d, [F.mul(x, c) if x else 0 for x in u.w])


def sbox_enc(F, alpha, x, T):
    """Forward S: a = x2 - x0 x3 - alpha, b = x1 - x3 a - alpha,
    c = x0 - a b - alpha, y3 = x3 - b c - alpha; output (c, b, a, y3)."""
    x0, x1, x2, x3 = x
    al = TopPoly.const(F, F.neg(alpha) if hasattr(F, "neg") else alpha, T)
    if F.char != 2:
        al = TopPoly.const(F, (-int(alpha)) % F.p, T)
    a = _add(F, _add(F, x2, _neg(F, _mul(F, x0, x3))), al)
    b = _add(F, _add(F, x1, _neg(F, _mul(F, x3, a))), al)
    c = _add(F, _add(F, x0, _neg(F, _mul(F, a, b))), al)
    y3 = _add(F, _add(F, x3, _neg(F, _mul(F, b, c))), al)
    return [c, b, a, y3]


def _neg(F, u):
    if F.char == 2:
        return u
    return TopPoly(u.d, [(-int(x)) % F.p for x in u.w])


def fwd_row(c, t=0):
    """First row of the forward circulant L_t."""
    if c.F.char == 2:
        return [1 if j in ROT_FWD_BIN[t] else 0 for j in range(WORDS)]
    return list(c.lin._rows[t])


def propagate(c, rks, pos, layers, T):
    """Per-layer list of 16 TopPoly, the state after each S layer."""
    F = c.F
    words = [TopPoly.const(F, int(rks[0][i]), T) for i in range(WORDS)]
    words[pos] = TopPoly.affine(F, int(rks[0][pos]), T)
    row = fwd_row(c, 0)
    out_layers = []
    for layer in range(1, layers + 1):
        outs = []
        for b in range(4):
            outs += sbox_enc(F, c.alpha, words[4 * b:4 * b + 4], T)
        out_layers.append(outs)
        nxt = []
        for i in range(WORDS):
            acc = TopPoly.const(F, 0, T)
            for j in range(WORDS):
                if row[j]:
                    acc = _add(F, acc, _smul(F, outs[(i + j) % WORDS], row[j]))
            nxt.append(_add(F, acc, TopPoly.const(F, int(rks[layer][i]), T)))
        words = nxt
    return out_layers


def verdict(F, u):
    """(status, value) for sum_{X in F_q} u(X) via (*)."""
    q = F.q
    T = len(u.w)
    if u.d < q - 1:
        return "balanced", 0
    mults = [k * (q - 1) for k in range(1, u.d // (q - 1) + 1)]
    if any(u.d - e >= T for e in mults):
        return "undecided", None
    s = 0
    for e in mults:
        s = F.add(s, u.coeff(e))
    if F.char != 2:
        s = (-int(s)) % F.p
    return ("balanced" if s == 0 else "not balanced"), int(s)


def run(instance, pos, layers, T, keys, seed, rounds=None):
    c = DuX(instance, rounds=rounds or (layers + 2))
    F = c.F
    out = []
    for ki in range(keys):
        rng = np.random.default_rng(seed + ki)
        rks = c.key_schedule(c.random_key(rng))
        prof = propagate(c, rks, pos, layers, T)
        rows = []
        for l, outs in enumerate(prof, 1):
            st = [verdict(F, u) for u in outs]
            rows.append({
                "layer": l,
                "degree": [u.d for u in outs],
                "window": [[int(v) for v in u.w] for u in outs],
                "status": [s for s, _ in st],
                "sum": [v for _, v in st],
                "pattern": "".join(
                    "1" if all(st[4 * b + p][0] == "balanced" for b in range(4))
                    else ("?" if any(st[4 * b + p][0] == "undecided" for b in range(4))
                          else "0") for p in range(4)),
            })
        out.append({"seed": seed + ki, "rows": rows})
    return {"instance": instance, "q": F.q, "char": F.char, "active": pos,
            "top": T, "keys": keys, "runs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^16")
    ap.add_argument("--active", type=int, default=1)
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show-window", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = run(a.instance, a.active, a.layers, a.top, a.keys, a.seed)
    print(f"{a.instance} (q = {res['q']}), active plaintext word {a.active}, "
          f"encryption direction, top {a.top} coefficients, {a.keys} keys")
    print("layer | degree (block 0: pos 0..3) | proved-balanced pattern per key "
          "| words with sum != 0")
    for l in range(a.layers):
        deg = res["runs"][0]["rows"][l]["degree"]
        pats = [r["rows"][l]["pattern"] for r in res["runs"]]
        bad = [i for i, s in enumerate(res["runs"][0]["rows"][l]["status"])
               if s != "balanced"]
        print(f"{l+1:5d} | {str(deg[:4]):26s} | {' '.join(pats)} | {bad}")
        if a.show_window:
            for i in range(4):
                r = res["runs"][0]["rows"][l]
                print(f"        w{i}: d = {r['degree'][i]:8d}  "
                      f"top = {r['window'][i]}  {r['status'][i]}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
