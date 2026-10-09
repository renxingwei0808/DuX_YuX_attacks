"""E05 -- general monomial prediction for DuX(2^n) in the decryption direction.

Method
------
Field-based (general) monomial prediction, Cui-Hu-Wang-Wei, ASIACRYPT 2022
("On the Field-Based Division Property"), in the formulation used by
Ni-Wang-Li (DCC 2026, 94:123, Appendix A, Rules 1-3).  Nothing is copied
from https://github.com/minionsjay/HD-YuX ; only the modelling idea (the
three propagation rules) is reused, and the circuit is DuX's.

Every wire of the decryption circuit carries an exponent u in [0, 2^n-1]
(the power of that wire taken by the monomial trail).  Rules:

  * t-XOR   y = x_0 + ... + x_{t-1} + const :
        the u_i are pairwise bit-disjoint and (u_0 | ... | u_{t-1}) <= v
        (the constant absorbs the remaining bits of v).  Without a constant
        the last inequality is an equality; every XOR in DuX carries either
        alpha or a round key, so we always use the "<=" form.
  * AND     y = x_0 * x_1 :  u_0 = u_1 = v.
  * t-COPY  a wire used t times with exponents v_0..v_{t-1} :
        sum(v_j) = u + s * (2^n - 1) for some 0 <= s <= t-1
        (x^{2^n-1} = 1 for x != 0, hence the wrap).

The algebraic degree of an output word is the largest sum of Hamming
weights of the exponents on the ACTIVE ciphertext words over all trails
reaching that word with exponent 1.  When each active word is restricted to
an F_2-affine subspace of dimension m < n, the monomial X^e contributes only
min(HW(e), m), so the objective becomes sum_j min(HW(e_j), m); the resulting
bound is exactly the "m_min = d + 1" criterion used in E02/E04.

Circuit (dux/cipher.py decrypt_layers, with L0 fixed by observation O2):

    C -> (+ rk^r) -> [ SL^-1 -> (+ rk) -> L0^-1 ] repeated

S^{-1} = R^4.  Unrolling the four R steps gives *literally* the closed form
of dux/sbox.py (see `build_sbox`), so the two modellings (closed form and four R steps)
are the same DAG; `--sbox r4` builds it step by step and
`--check-sbox` asserts the two DAGs are isomorphic.

Usage
-----
  python gmp_dux.py --n 8  --active 3 --layers 7 --words 0,1,2,3
  python gmp_dux.py --n 16 --active 3,7 --layers 10 --words 2 --timeout 3600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import z3

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dux.params import ROT_INV  # noqa: E402

WORDS = 16


# --------------------------------------------------------------- circuit ---
class Circuit:
    """DAG of ops: ("in", ()), ("xc", (a,)), ("xor", (a, b, ...)), ("mul", (a, b))."""

    def __init__(self):
        self.ops: list[tuple[str, tuple]] = []

    def add(self, op, args=()):
        self.ops.append((op, tuple(args)))
        return len(self.ops) - 1

    def __len__(self):
        return len(self.ops)


def build_sbox(c: Circuit, x, mode="closed"):
    """S^{-1} = R^4 on one 4-word block; returns the 4 output node ids."""
    x0, x1, x2, x3 = x
    if mode == "r4":
        st = [x0, x1, x2, x3]
        for _ in range(4):
            p = c.add("mul", (st[0], st[1]))
            z = c.add("xor", (p, st[3]))          # + alpha
            st = [st[1], st[2], z, st[0]]
        return st
    # closed form of dux/sbox.py
    p1 = c.add("mul", (x0, x1))
    f = c.add("xor", (p1, x3))                    # + alpha   -> y3
    p2 = c.add("mul", (x1, x2))
    g = c.add("xor", (p2, x0))                    # + alpha   -> y0
    p3 = c.add("mul", (x2, f))
    y1 = c.add("xor", (p3, x1))                   # + alpha
    p4 = c.add("mul", (f, g))
    y2 = c.add("xor", (p4, x2))                   # + alpha
    return [g, y1, y2, f]


def build_circuit(layers, rots=ROT_INV[0], sbox="closed"):
    """Return (circuit, ciphertext node ids, [state after each S^-1 layer])."""
    c = Circuit()
    inp = [c.add("in") for _ in range(WORDS)]
    x = [c.add("xc", (inp[i],)) for i in range(WORDS)]      # ARK^-1 with rk^r
    states = []
    for l in range(layers):
        y = []
        for b in range(4):
            y += build_sbox(c, x[4 * b:4 * b + 4], sbox)
        states.append(list(y))
        if l < layers - 1:
            # ARK^-1 (constant) folded into the 5-XOR of L0^-1
            x = [c.add("xor", tuple(y[(i + j) % WORDS] for j in rots))
                 for i in range(WORDS)]
    return c, inp, states


def ancestors(c: Circuit, target: int):
    seen, stack = set(), [target]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(c.ops[v][1])
    return seen


# ------------------------------------------------------------------ model ---
def build_model(n, c: Circuit, keep, target, inp, active, dim, target_exp=1):
    """Return (constraints, objective-as-Int) for one (target word) query."""
    M = (1 << n) - 1
    pad = 8
    e = {v: z3.BitVec(f"e{v}", n) for v in keep}
    edge, outs = {}, {v: [] for v in keep}
    cons = []
    for v in sorted(keep):
        _, args = c.ops[v]
        for s, a in enumerate(args):
            w = z3.BitVec(f"w{v}_{s}", n)
            edge[(v, s)] = w
            outs[a].append(w)
    for v in sorted(keep):
        op, args = c.ops[v]
        ws = [edge[(v, s)] for s in range(len(args))]
        if op == "mul":
            cons += [ws[0] == e[v], ws[1] == e[v]]
        elif op in ("xor", "xc"):
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    cons.append(ws[i] & ws[j] == 0)
            orall = ws[0]
            for w in ws[1:]:
                orall = orall | w
            cons.append(orall & ~e[v] == 0)       # submask of e[v]
    for v in sorted(keep):
        o = outs[v]
        if not o:
            continue
        if len(o) == 1:
            cons.append(o[0] == e[v])
        else:
            S = z3.ZeroExt(pad, o[0])
            for w in o[1:]:
                S = S + z3.ZeroExt(pad, w)
            cons.append(z3.Or([S == z3.ZeroExt(pad, e[v]) + k * M
                               for k in range(len(o))]))
    cons.append(e[target] == target_exp)

    terms = []
    for j in active:
        v = inp[j]
        if v not in keep:
            continue
        hw = z3.Sum([z3.If(z3.Extract(i, i, e[v]) == 1, 1, 0) for i in range(n)])
        terms.append(hw if dim >= n else z3.If(hw > dim, dim, hw))
    obj = z3.Sum(terms) if terms else z3.IntVal(0)
    return cons, obj


def max_degree(n, layers, active, word, dim, sbox="closed", timeout=0, verbose=False):
    """Largest achievable sum_j min(HW(e_j), dim) for output `word` after
    `layers` S^{-1} layers.  Binary search on the objective."""
    c, inp, states = build_circuit(layers, sbox=sbox)
    target = states[layers - 1][word]
    keep = ancestors(c, target)
    cons, obj = build_model(n, c, keep, target, inp, active, dim)
    s = z3.Solver()
    if timeout:
        s.set("timeout", int(timeout * 1000))
    s.add(cons)
    hi = min(dim, n) * len(active)
    lo, best = 0, None
    # sat at 0 is guaranteed (the constant monomial), so best is well defined
    while lo <= hi:
        mid = (lo + hi) // 2
        s.push()
        s.add(obj >= mid)
        r = s.check()
        s.pop()
        if verbose:
            print(f"    D>={mid}: {r}", flush=True)
        if r == z3.sat:
            best, lo = mid, mid + 1
        elif r == z3.unsat:
            hi = mid - 1
        else:
            return {"degree": None, "status": "timeout", "at": mid,
                    "nodes": len(keep)}
    return {"degree": best, "status": "ok", "nodes": len(keep)}


def top_monomial(n, layers, active, word, sbox="closed", timeout=0, dim=None):
    """Is the single maximal-degree monomial prod_j X_j^{2^dim-1}-style trail
    reachable in output `word`?  With dim = n this is the monomial
    prod_j X_j^{2^n-1}, whose coefficient is exactly the zero-sum of the word
    over the full product structure; UNSAT therefore *proves* the zero-sum.

    Pinning the active input exponents makes the query far easier than the
    generic "degree >= D" question."""
    M = (1 << n) - 1
    c, inp, states = build_circuit(layers, sbox=sbox)
    target = states[layers - 1][word]
    keep = ancestors(c, target)
    cons, _ = build_model(n, c, keep, target, inp, active, n)
    s = z3.Solver()
    if timeout:
        s.set("timeout", int(timeout * 1000))
    s.add(cons)
    pinned = 0
    for j in active:
        if inp[j] in keep:
            s.add(z3.BitVec(f"e{inp[j]}", n) == M)
            pinned += 1
    if pinned < len(active):
        # an active word that does not even reach the target: top monomial absent
        return {"reachable": False, "status": "structural", "nodes": len(keep)}
    r = s.check()
    status = "sat" if r == z3.sat else ("unsat" if r == z3.unsat else "unknown")
    return {"reachable": (status != "unsat"), "status": status, "nodes": len(keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--active", default="3")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--words", default="0,1,2,3")
    ap.add_argument("--dim", type=int, default=None,
                    help="F_2-dimension per active word (default n)")
    ap.add_argument("--sbox", choices=["closed", "r4"], default="closed")
    ap.add_argument("--from-layer", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=0, help="seconds per SMT query")
    ap.add_argument("--mode", choices=["degree", "top"], default="degree")
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    active = [int(v) for v in a.active.split(",")]
    words = [int(v) for v in a.words.split(",")]
    dim = a.n if a.dim is None else a.dim
    rows = []
    for L in range(a.from_layer, a.layers + 1):
        row = {"layer": L, "degree": {}}
        for w in words:
            t0 = time.time()
            if a.mode == "top":
                r = top_monomial(a.n, L, active, w, a.sbox, a.timeout)
            else:
                r = max_degree(a.n, L, active, w, dim, a.sbox, a.timeout, a.verbose)
            r["seconds"] = round(time.time() - t0, 2)
            row["degree"][str(w)] = r
            head = (f"top={r['reachable']}" if a.mode == "top" else f"D={r['degree']}")
            print(f"n={a.n} active={active} dim={dim} layer={L} word={w}: "
                  f"{head} ({r['status']}, {r['nodes']} nodes, {r['seconds']}s)",
                  flush=True)
        rows.append(row)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump({"n": a.n, "active": active, "dim": dim, "sbox": a.sbox,
                   "mode": a.mode, "rows": rows}, open(a.json, "w"), indent=1)
        print("saved", a.json)


if __name__ == "__main__":
    main()
