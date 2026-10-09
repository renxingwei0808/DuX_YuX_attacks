"""W3 (E02) -- EXACT univariate polynomials of the DuX decryption state.

`tools/maxplus.py` (F_p) and `tools/exponent_sets.py` (F_2^n) give upper
bounds on the algebraic degree / exponent set of each state word as a
polynomial in one active ciphertext word X.  This script computes the
polynomials themselves, so we can see the cancellations and say exactly
where the bounds stop being tight.

Model (identical to experiments/E03 and E04, only symbolic in X):
    C = (c_0, ..., c_{pos-1}, X, c_{pos+1}, ..., c_15)
    state_1 = SL^-1(C - rk^r);  state_{k+1} = SL^-1(L^-1(state_k - rk^{r-k}))
with real round keys from the key schedule and real constants.  Every state
word is an element of F_q[X]/(X^q - X), stored densely as its coefficient
vector (index = exponent, exponents live in {0} u [1, q-1]).

Exactness is affordable because the state is UNIVARIATE: a polynomial has at
most q coefficients no matter how many layers we run.
  * F_p : coefficients are ints mod p; products use Kronecker substitution
          (pack into one big integer, multiply, unpack) -- O~(q) per product.
  * F_2^n : coefficients are field elements; products use the log/antilog
          tables of dux.field.BinaryField, O(q^2) per product (fine for n<=8).

The zero-sum test is exact and needs no experiment:
    sum_{X in F_q} f(X) = -coeff_{q-1}(f)        (both F_p and F_2^n)
so a word is balanced over the full field iff its coefficient at exponent
q-1 vanishes.  This reproduces E03/E04 from the polynomial side.

Usage
  python experiments/E02_degree_bounds/symbolic_layers.py --instance dux-65537 --layers 10
  python experiments/E02_degree_bounds/symbolic_layers.py --instance toy-257  --layers 8
  python experiments/E02_degree_bounds/symbolic_layers.py --instance dux-2^8  --layers 8 --expsets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dux import DuX                                   # noqa: E402
from dux.params import ROT_INV, WORDS                 # noqa: E402
from dux.linear import t_xor                          # noqa: E402
import tools.maxplus as maxplus                       # noqa: E402
import tools.exponent_sets as expsets                 # noqa: E402


# --------------------------------------------------------------------------
# dense univariate polynomials over F_q, reduced modulo X^q - X
# --------------------------------------------------------------------------
class PolyRing:
    """Coefficient vectors (numpy int64, index = exponent, len <= q)."""

    def __init__(self, F):
        self.F = F
        self.q = F.q
        self.char2 = (F.char == 2)

    # -- constructors
    def const(self, v):
        return np.array([int(v) % self.q if not self.char2 else int(v)], dtype=np.int64)

    def linear(self, k):
        """X + k."""
        a = np.zeros(2, dtype=np.int64)
        a[0] = int(k)
        a[1] = 1
        return a

    # -- helpers
    def trim(self, a):
        nz = np.nonzero(a)[0]
        return a[: nz[-1] + 1].copy() if len(nz) else np.zeros(1, dtype=np.int64)

    def reduce(self, a):
        """Fold exponents e >= q down to ((e-1) mod (q-1)) + 1."""
        q = self.q
        if len(a) <= q:
            return self.trim(a)
        out = a[:q].copy()
        tail = a[q:]
        for i, v in enumerate(tail):
            if v:
                e = q + i
                e = ((e - 1) % (q - 1)) + 1
                if self.char2:
                    out[e] ^= v
                else:
                    out[e] = (out[e] + v) % q
        return self.trim(out)

    # -- ring operations
    def add(self, a, b):
        n = max(len(a), len(b))
        r = np.zeros(n, dtype=np.int64)
        if self.char2:
            r[: len(a)] ^= a
            r[: len(b)] ^= b
        else:
            r[: len(a)] += a
            r[: len(b)] += b
            r %= self.q
        return self.trim(r)

    def mul(self, a, b):
        if len(a) == 1 and a[0] == 0:
            return a.copy()
        if len(b) == 1 and b[0] == 0:
            return b.copy()
        raw = self._mul_char2(a, b) if self.char2 else self._mul_fp(a, b)
        return self.reduce(raw)

    def _mul_fp(self, a, b):
        """Kronecker substitution: 8 bytes per coefficient.
        Slot value <= min(len)*p^2 < 2^49 for p <= 65537, so no carry."""
        p = self.q
        la, lb = len(a), len(b)
        assert min(la, lb) * (p - 1) ** 2 < (1 << 64), "slot overflow"
        ia = int.from_bytes(a.astype("<u8").tobytes(), "little")
        ib = int.from_bytes(b.astype("<u8").tobytes(), "little")
        prod = ia * ib
        nslots = la + lb - 1
        buf = prod.to_bytes(8 * (nslots + 1), "little")
        # every slot is < 2^49, so the uint64 view fits in int64 without loss
        out = np.frombuffer(buf, dtype="<u8", count=nslots).astype(np.int64)
        return out % p

    def _mul_char2(self, a, b):
        F = self.F
        if len(a) > len(b):
            a, b = b, a
        out = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
        idx = np.nonzero(a)[0]
        for i in idx:
            out[i: i + len(b)] ^= F.vmul(np.full(len(b), int(a[i]), dtype=np.int64), b)
        return out

    # -- read-out
    def degree(self, a):
        nz = np.nonzero(a)[0]
        return int(nz[-1]) if len(nz) else -1

    def support(self, a):
        return np.nonzero(a)[0]

    def hw_degree(self, a):
        """Algebraic (Boolean) degree = max Hamming weight of an exponent."""
        nz = self.support(a)
        if len(nz) == 0:
            return -1
        return int(max(bin(int(e)).count("1") for e in nz))

    def full_field_sum(self, a):
        """sum_{X in F_q} f(X) = -coeff_{q-1}(f)  (0 if the poly is shorter)."""
        if len(a) < self.q:
            return 0
        v = int(a[self.q - 1])
        return v if self.char2 else (-v) % self.q


# --------------------------------------------------------------------------
# symbolic decryption
# --------------------------------------------------------------------------
def sbox_inv_poly(P, x):
    """S^{-1} = R^4 on four polynomials (closed form of dux/sbox.py)."""
    x0, x1, x2, x3 = x
    al = P.const(P.F.from_int(P.alpha))
    f = P.add(P.add(P.mul(x0, x1), x3), al)          # y3
    g = P.add(P.add(P.mul(x1, x2), x0), al)          # y0
    y1 = P.add(P.add(P.mul(x2, f), x1), al)
    y2 = P.add(P.add(P.mul(f, g), x2), al)
    return [g, y1, y2, f]


def lin_inv_poly(P, words, t):
    rots = ROT_INV[t]
    out = []
    for i in range(WORDS):
        acc = words[(i + rots[0]) % WORDS]
        for j in rots[1:]:
            acc = P.add(acc, words[(i + j) % WORDS])
        out.append(acc)
    return out


def symbolic_layers(c, rks, pos, layers, consts, force_L0=True):
    """Return [state_after_layer_1, ..., state_after_layer_`layers`],
    each a list of 16 coefficient vectors."""
    P = PolyRing(c.F)
    P.alpha = c.alpha
    F = c.F
    r = c.r
    # C - rk^r : the active word becomes X - rk^r_pos, the others constants
    st = []
    for i in range(WORDS):
        if i == pos:
            st.append(P.linear(F.sub(0, rks[r][i]) if not P.char2 else F.sub(0, rks[r][i])))
        else:
            st.append(P.const(F.sub(consts[i], rks[r][i])))
    states = []
    for k in range(layers):
        out = []
        for b in range(4):
            out += sbox_inv_poly(P, st[4 * b:4 * b + 4])
        states.append(out)
        if k + 1 == layers:
            break
        i = r - (k + 1)
        st = [P.add(w, P.const(F.sub(0, rks[i][j]))) for j, w in enumerate(out)]
        t = 0 if force_L0 else t_xor(rks[i])
        st = lin_inv_poly(P, st, t)
    return P, states



def expset_states(n, pos, layers):
    """Same propagation as tools/exponent_sets.run, but keep the ExpSet
    objects so that the bound can be compared set-wise with the exact
    support (and not just through its size / max Hamming weight)."""
    words = [expsets.ExpSet.const(n) for _ in range(WORDS)]
    words[pos] = expsets.ExpSet.affine(n)
    out = []
    for _ in range(layers):
        outs = []
        for b in range(4):
            outs += expsets.sbox_inv_sets(words[4 * b:4 * b + 4], n)
        out.append(outs)
        words = expsets.linear_inv_sets(outs, ROT_INV[0], n)
    return out


def expset_support(e):
    s = set(int(i) + 1 for i in np.nonzero(e.m)[0])
    if e.z:
        s.add(0)
    return s


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--positions", default="0,1,2,3")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--subgroup-scan", action="store_true",
                    help="F_p only: for every k, the largest layer whose exact polynomial is "
                         "constant on cosets of the order-2^k subgroup (exact prediction for E03 --subgroup)")
    ap.add_argument("--expsets", action="store_true",
                    help="also compare exact exponent sets with tools/exponent_sets.py (F_2^n)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    c = DuX(a.instance)
    P0 = PolyRing(c.F)
    rng = np.random.default_rng(a.seed)
    positions = [int(v) for v in a.positions.split(",")]
    out = {"instance": a.instance, "q": c.F.q, "seed": a.seed, "keys": a.keys,
           "layers": a.layers, "per_position": {}}
    print(f"{a.instance}: q = {c.F.q}; exact univariate polynomials, {a.keys} random keys, "
          f"L0 fixed (O2)")

    for pos in positions:
        # upper bounds for comparison
        mp = maxplus.run_single_word(pos, a.layers)
        bound_max_exp = [r["sbox_out_max_per_pos"] for r in mp]
        es = expsets.run(c.F.n, pos, a.layers) if (c.F.char == 2 and a.expsets) else None

        per_key = []
        t0 = time.time()
        for kk in range(a.keys):
            K = c.random_key(rng)
            rks = c.key_schedule(K)
            consts = [int(v) for v in rng.integers(0, c.F.q, size=16)]
            P, states = symbolic_layers(c, rks, pos, a.layers, consts)
            rec = []
            for l, st in enumerate(states):
                degs = [P.degree(w) for w in st]
                sums = [P.full_field_sum(w) for w in st]
                row = {"layer": l + 1,
                       "exact_degree": degs,
                       "balanced": [s == 0 for s in sums],
                       "n_balanced": sum(1 for s in sums if s == 0)}
                if es is not None:
                    row["exact_hw_degree"] = [P.hw_degree(w) for w in st]
                    row["exact_support_size"] = [int(len(P.support(w))) for w in st]
                rec.append(row)
            per_key.append(rec)
        elapsed = time.time() - t0

        # aggregate: exact degrees must agree across keys (generic behaviour)
        agree = all(per_key[0][l]["exact_degree"] == per_key[k][l]["exact_degree"]
                    for k in range(1, a.keys) for l in range(a.layers))
        print(f"\npos {pos}  ({elapsed:.1f}s, degrees agree across keys: {agree})")
        hdr = "layer | exact deg (block0 pos0..3) |  max exact | max bound | tight? | #balanced"
        print(hdr)
        rows = []
        for l in range(a.layers):
            d = per_key[0][l]["exact_degree"]
            mx = max(d)
            bmx = max(bound_max_exp[l])
            nb = min(per_key[k][l]["n_balanced"] for k in range(a.keys))
            rows.append({"layer": l + 1, "exact_degree": d, "max_exact": mx,
                         "max_bound": bmx, "tight": mx == bmx, "n_balanced": nb,
                         "balanced": [all(per_key[k][l]["balanced"][i] for k in range(a.keys))
                                      for i in range(16)]})
            print(f"{l + 1:5d} | {str(d[:4]):26s} | {mx:10d} | {bmx:9d} | "
                  f"{'yes' if mx == bmx else 'NO ':6s} | {nb}")
        entry = {"rows": rows, "bound_max_exp": bound_max_exp,
                 "degrees_agree_across_keys": agree, "elapsed_s": round(elapsed, 2)}
        if es is not None:
            entry["expset_bound_degree"] = [r["degree"] for r in es]
            entry["expset_bound_size"] = [r["set_size"] for r in es]
            entry["exact_hw_degree"] = [per_key[0][l]["exact_hw_degree"] for l in range(a.layers)]
            entry["exact_support_size"] = [per_key[0][l]["exact_support_size"] for l in range(a.layers)]
            # set-wise comparison exact support vs exponent-set bound
            bnd = expset_states(c.F.n, pos, a.layers)
            P = PolyRing(c.F)
            P.alpha = c.alpha
            K = c.random_key(np.random.default_rng(a.seed + 1))
            rks = c.key_schedule(K)
            consts = [int(v) for v in np.random.default_rng(a.seed + 2).integers(0, c.F.q, size=16)]
            _, st_exact = symbolic_layers(c, rks, pos, a.layers, consts)
            cmp_rows = []
            print("  exponent-set bound vs exact support (word = max over the 16):")
            print("  layer | exact HW-deg | bound HW-deg | exact |E| | bound |E| | exact subset of bound?")
            for l in range(a.layers):
                ex = [set(int(v) for v in P.support(w)) for w in st_exact[l]]
                bs = [expset_support(e) for e in bnd[l]]
                subset = all(ex[i] <= bs[i] for i in range(16))
                row = {"layer": l + 1,
                       "exact_hw_degree": [P.hw_degree(w) for w in st_exact[l]],
                       "bound_hw_degree": [e.degree() for e in bnd[l]],
                       "exact_size": [len(e) for e in ex],
                       "bound_size": [len(e) for e in bs],
                       "exact_subset_of_bound": subset,
                       "overcount": [len(bs[i] - ex[i]) for i in range(16)]}
                cmp_rows.append(row)
                print(f"  {l + 1:5d} | {max(row['exact_hw_degree']):12d} | "
                      f"{max(row['bound_hw_degree']):12d} | {max(row['exact_size']):9d} | "
                      f"{max(row['bound_size']):9d} | {subset}")
            entry["expset_comparison"] = cmp_rows
        out["per_position"][pos] = entry

        if a.subgroup_scan and c.F.char != 2:
            p_ = c.F.q
            assert (p_ - 1) & (p_ - 2) == 0, "subgroup scan assumes p-1 is a power of two"
            kmax = (p_ - 1).bit_length() - 1
            P = PolyRing(c.F)
            P.alpha = c.alpha
            rngs = np.random.default_rng(a.seed + 7)
            K = c.random_key(rngs)
            rks = c.key_schedule(K)
            consts = [int(v) for v in rngs.integers(0, c.F.q, size=16)]
            _, st = symbolic_layers(c, rks, pos, a.layers, consts)
            scan = {}
            print("  exact subgroup-coset prediction (k -> largest layer with all 16 words "
                  "constant across cosets of the order-2^k subgroup):")
            for k in range(2, kmax + 1):
                step = 1 << k
                best = 0
                for l in range(a.layers):
                    ok = True
                    for w in st[l]:
                        d = step
                        while d < len(w):
                            if w[d]:
                                ok = False
                                break
                            d += step
                        if not ok:
                            break
                    if ok:
                        best = l + 1
                    else:
                        break
                scan[k] = best
                print(f"    k={k:2d} (data 2^{k}): {best} layers")
            entry["subgroup_scan"] = scan

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)
        print("\nsaved", a.json)


if __name__ == "__main__":
    main()
