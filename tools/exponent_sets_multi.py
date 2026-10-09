"""Multivariate exponent-set degree bounds for DuX(2^n) decryption.

Ni-Wang-Li bound the multi-variable algebraic degree with Proposition 3 (a
component-wise "box" over the s active variables).  For DuX the same idea can
be made sharper for free, because of the following observation.

Let M_l(w) be the set of exponent VECTORS (e_1, ..., e_s) appearing in state
word w after l layers.  Under the Proposition-1 semantics (union for +,
component-wise sumset for x) the projection of M_l(w) onto coordinate j
satisfies exactly the univariate recursion of tools/exponent_sets.py with the
single active word placed at active[j] -- unions project to unions and
sumsets project to sumsets, and the initial data agree.  Hence

    M_l(w)  subset of  E_l(w; active[0]) x ... x E_l(w; active[s-1])

and therefore, for a product of F_2-affine subspaces of dimension m,

    deg_m(w, l) <= sum_j min( d(l, w; active[j]), m ),                     (*)

with d(.) the univariate exponent-set degree bound.  The word is balanced
whenever this is < m*s.  When all active positions are equivalent (all of
them = 3 mod 4, say) the right-hand side is s * min(d, m), so

    balanced  <==  m > d(l, w)                                            (**)

which is INDEPENDENT OF s: the exponent-set machinery predicts exactly the
empirical law (R1) of results/E04_zero_sum_F2n/large/minimal_data_table.md
("the number of zero-sum layers is decided by the per-word dimension m, not
by the number of active words").

The bound is an upper bound only; where the experiments beat it (s = 4 with
m = n buys one extra layer) the gap is real; the E05 records in
results/E05_monomial_prediction/ show it.

Usage
  python tools/exponent_sets_multi.py --n 16 --active 3,7 --layers 11
  python tools/exponent_sets_multi.py --n 8 --active 3,7,11,15 --layers 8 --dim 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools.exponent_sets import run as run_univariate  # noqa: E402


def bounds(n, active, layers, dim=None, direction="dec"):
    """Rows: {layer, degree[16], balanced[16]} for the product structure."""
    m = n if dim is None else dim
    per = {a: run_univariate(n, a, layers, direction=direction)
           for a in sorted(set(active))}
    rows = []
    for l in range(layers):
        deg, bal = [], []
        for w in range(16):
            d = sum(min(per[a][l]["degree"][w], m) for a in active)
            deg.append(d)
            bal.append(d < m * len(active))
        rows.append({"layer": l + 1, "degree": deg, "balanced": bal,
                     "pattern": "".join("1" if b else "0" for b in bal),
                     "count": sum(bal)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--layers", type=int, default=11)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--direction", choices=["dec", "enc"], default="dec")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    active = [int(v) for v in a.active.split(",")]
    m = a.n if a.dim is None else a.dim
    rows = bounds(a.n, active, a.layers, a.dim, a.direction)
    print(f"DuX(2^{a.n}), active {active}, per-word dimension {m}, {a.direction} "
          f"(data 2^{m * len(active)}): separable exponent-set bound")
    print("layer | degree bound (block 0: words 0..3) | max/16s | guaranteed-balanced pattern")
    for r in rows:
        print(f"{r['layer']:5d} | {str(r['degree'][:4]):20s} | "
              f"{max(r['degree']):4d}/{m * len(active):<4d} | {r['pattern']} ({r['count']})")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"n": a.n, "active": active, "dim": m,
                   "direction": a.direction, "rows": rows},
                  open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
