"""S6(c) -- which active words buy the most layers, exhaustively.

The zero-sum criterion O7 (tools/zero_sum_criterion.py) is cheap enough to
evaluate on every choice of s active words, so "does more data help?" can be
answered exactly instead of guessed: for each s the script reports the word set
whose pattern at a given layer is best, together with the margins T - D.

This is what settles the CPA question on DuX(2^8): our initial plan for S6(c)
expected `1111` at layer 4 for s = 5, but adding a fifth word necessarily puts
two active words in the same S-box block, which raises the degree bound faster
than it raises the threshold.

    python best_word_sets.py --q 2^8 --direction enc --layer 4 --smax 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from tools.zero_sum_criterion import profile, threshold   # noqa: E402


def scan(q, s, layer, direction):
    char2 = isinstance(q, str) and q.startswith("2^")
    T = threshold(q, list(range(s)))
    best = None
    for act in itertools.combinations(range(16), s):
        Y = profile(list(act), layer, direction, char2)[layer - 1]
        worst = [max(Y[4 * b + p] for b in range(4)) for p in range(4)]
        pat = "".join("1" if w < T else "0" for w in worst)
        margin = [T - w for w in worst]
        key = (pat.count("1"), min(margin))
        if best is None or key > best[0]:
            best = (key, list(act), pat, margin)
    return {"s": s, "threshold": T, "words": best[1], "pattern": best[2],
            "margin": best[3], "balanced_positions": best[2].count("1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="2^8", help="2^8 | 2^16 | 65537 | 193 | 257")
    ap.add_argument("--direction", choices=["dec", "enc"], default="enc")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--smax", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    q = a.q if a.q.startswith("2^") else int(a.q)
    bits = int(a.q[2:]) if a.q.startswith("2^") else None
    rows = []
    print(f"q = {a.q}, direction = {a.direction}, layer {a.layer}: "
          f"best choice of s active words (over all C(16,s))")
    for s in range(1, a.smax + 1):
        r = scan(q, s, a.layer, a.direction)
        data = f"2^{bits * s}" if bits else f"{a.q}^{s}"
        r["data"] = data
        rows.append(r)
        print(f"  s={s} data={data:<7} words={str(r['words']):26s} "
              f"pattern={r['pattern']}  T={r['threshold']:<6} margin={r['margin']}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"q": a.q, "direction": a.direction, "layer": a.layer,
                   "rows": rows}, open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
