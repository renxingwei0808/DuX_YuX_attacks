"""E07 (W7) -- r_KR = 2 key recovery by structured linearisation, prototype.

Setting.  r = l + 2 rounds, an l-layer zero-sum that balances (at least)
block position 2 of every block.  With
    Y   = SL(P + rk^0)                    (16 words, the inner S layer)
    u^j = block j of L_{t1}(Y)            (the linear layer, L0 by O2)
    k'  = block j of rk^1
the coordinate-2 output of the second S layer is a(u^j + k') and the zero-sum
gives ONE equation per outer block j and per structure:

    sum_P u2 - sum_P u0 u3 - k'_3 sum_P u0 - k'_0 sum_P u3 = 0                (*)

(the k'_0 k'_3, k'_2 and alpha terms carry the factor |structure| = 0).

Linearisation.  Expand S(P + k) ONCE as an explicit polynomial in the four key
words AND the four plaintext words (experiments/E08.../keypoly.py).  Then
    T_s   = sum_P Y_s          -> a polynomial in the key words of block(s)
    Q_s,t = sum_P Y_s Y_t      -> a polynomial in the key words of block(s)
                                  and block(t), whose coefficients are joint
                                  plaintext moments of the structure,
so (*) is LINEAR in the monomials
    {1} u {inner monomials} u {inner x inner monomials} u {k'_i x inner}
and every structure contributes one F_q-linear equation per outer block.

`--count-only` prints the exact size of that monomial set (the number S3 needs
to budget the full-scale attack).  Without it the script runs the whole
pipeline -- moments, assembly, rank, back-substitution -- and checks the
recovered key against the real one.  Because the full 4-block system has
~2^15 unknowns (too large for a workstation), the default prototype leaves
ONE inner block unknown (`--unknown-blocks 0`) and takes the other three as
known; that exercises every step and measures the rank behaviour.

Usage
  python attack_2round_toy.py --instance toy-257 --count-only
  python attack_2round_toy.py --instance toy-257 --unknown-blocks 0
  python attack_2round_toy.py --instance toy-257 --unknown-blocks 0 --outer-blocks 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product as iproduct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "E08_boolean_degree_extension"))
from dux import DuX                                        # noqa: E402
from dux.params import ROT_FWD_BIN                          # noqa: E402
from dux.linear import t_xor                                # noqa: E402
from dux.registry import cipher_family, get_cipher          # noqa: E402
from keypoly import KeyPoly, ENC_CHAIN                      # noqa: E402
from keypoly import sbox_enc_chain                          # noqa: E402


# ------------------------------------------------------------------ setup --
def joint_expansion(F, alpha, cipher="dux"):
    """S(P+k): per coordinate, a list of (k-exponents, P-exponents, coeff).

    `cipher` picks the family's ENCRYPTION S-box chain (keypoly.ENC_CHAIN):
    DuX's S = (R^{-1})^4 with degrees (5,3,2,8) or YuX's S = Pf^{-4} with
    (8,5,3,2)."""
    x = [KeyPoly.var_plus_const(F, i, 0, 8) + KeyPoly.var_plus_const(F, 4 + i, 0, 8)
         for i in range(4)]
    out = []
    for t in ENC_CHAIN[cipher](x, alpha):
        out.append([(e[:4], e[4:], co) for e, co in t.d.items()])
    return out


def linear_row(c, t):
    """The first row of the forward linear layer L_t, so that
    L_t(x)_i = sum_l row[l] x_{(i+l) mod 16}.  YuX has a single layer, so `t`
    is ignored there (yux.linear.LinearLayer._rows maps both keys to it)."""
    from yux import YuX
    if isinstance(c, YuX):
        return c.lin._rows[0]
    if c.F.char == 2:
        return tuple(1 if l in ROT_FWD_BIN[t] else 0 for l in range(16))
    return c.lin._rows[t]


def reduce_exp(e, order):
    return ((e - 1) % order + 1) if e else 0


def add_exp(a, b, order):
    return tuple(reduce_exp(x + y, order) for x, y in zip(a, b))


# ---------------------------------------------------------- monomial sets --
def monomial_set(F, alpha, unknown, outer_blocks, joint=None, outer_coords=(0, 3),
                 cipher="dux", extra=()):
    """The exact linearisation monomial set for equation (*).
    A monomial is (outer, inner) where `outer` is None or (j, i) meaning the
    factor k'_i of outer block j, and `inner` is a dict {block: exps}.

    `joint` overrides the inner S-box expansion (E10/W14 uses S^{-1} instead of
    S) and `outer_coords` names the outer key words the equation is linear in
    (E07: the coordinate-2 map a = u2 - u0 u3 pairs u0 with k'_3 and u3 with
    k'_0; the CPA equation uses the coordinate-3 map f = x0 x1 + x3, i.e.
    (0, 1))."""
    order = F.q - 1
    joint = joint if joint is not None else joint_expansion(F, alpha, cipher)
    kU = [ {ke for ke, pe, _ in joint[co]} for co in range(4) ]          # before sum
    kS = [ {ke for ke, pe, _ in joint[co] if any(pe)} for co in range(4) ]  # after sum
    U = set().union(*kU)
    S = set().union(*kS)
    mons = set()
    mons.add((None, ()))                                    # the constant 1
    ub = sorted(unknown)
    # single-block monomials, from T_s of an unknown block
    single = {(b, m) for b in ub for m in S}
    for b, m in single:
        mons.add((None, ((b, m),)))
    # product monomials  Q_{s,t}
    for A in ub:
        for B in ub:
            if A == B:
                for a in U:
                    for bb in U:
                        mons.add((None, ((A, add_exp(a, bb, order)),)))
            elif A < B:
                for a in U:
                    for bb in U:
                        mons.add((None, ((A, a), (B, bb))))
    # products of one unknown block with a KNOWN block: linear in that block
    if len(ub) < 4:
        for b in ub:
            for m in U:
                mons.add((None, ((b, m),)))
    # k'_i x (inner monomial or 1)
    for j in outer_blocks:
        for i in outer_coords:
            mons.add(((j, i), ()))
            for b in ub:
                for m in S:
                    mons.add(((j, i), ((b, m),)))
    # W27 / E13: an optional extension of the column set (the cubic coordinate
    # b adds the Y_s Y_t^2 pair monomials and the k'_0 T^2 columns).  Empty by
    # default, so every existing count is unchanged.
    mons.update(extra)
    return sorted(mons, key=lambda m: ((-1, -1) if m[0] is None else m[0], m[1])), joint, U, S


# --------------------------------------------------------------- moments ---
def active_words(pos, active=None):
    """Normalise the two ways of naming the active ciphertext words:
    `--pos 3` (s = 1, kept as an alias) and `--active 3,7[,11]` (s >= 1)."""
    if active is not None:
        return [int(v) for v in (active.split(",") if isinstance(active, str) else active)]
    return [int(pos)] if isinstance(pos, int) else [int(v) for v in pos]


def _axis_values(F, act, values):
    """The value list of every active word, and the strides of the product set.

    `values is None` is the classical structure: each active word runs over all
    of F_q, axis j has stride q^(s-1-j).  A `values` list (S13-D: one coset of
    the order-2^k subgroup of F_p^*, O12 Sect. 5.5) keeps the same "last axis
    fastest" flattening with the per-axis sizes it gives."""
    s = len(act)
    if values is None:
        vals = [np.arange(F.q, dtype=np.int64)] * s
    else:
        vals = [np.asarray(v, dtype=np.int64) for v in values]
        assert len(vals) == s, "one value list per active word"
    sizes = [len(v) for v in vals]
    strides, acc = [0] * s, 1
    for j in range(s - 1, -1, -1):
        strides[j] = acc
        acc *= sizes[j]
    return vals, sizes, strides, acc


def structure_data(c, rks, rounds, pos, seed, active=None, limit=None,
                   values=None):
    """One chosen-ciphertext structure: the s active ciphertext words each run
    over all of F_q (q^s points, a full product set) and the other 16 - s words
    are random constants.  Returns the decrypted plaintexts as 16 arrays.

    s = 1 reproduces the original single-word structure exactly (same seed ->
    same constants -> same data), so every earlier result stays reproducible.

    `limit` truncates the product set to its first `limit` points in the same
    C order as `np.meshgrid(..., indexing="ij").reshape(-1)`.  It exists only
    to calibrate the cost per point of the assembly for structures whose full
    point set does not fit in memory (S4 step 3); a truncated structure is NOT
    a zero-sum structure, so the equations do not hold on it."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    act = active_words(pos, active)
    s = len(act)
    vals, sizes, strides, N = _axis_values(F, act, values)
    if limit is not None:
        N = min(N, int(limit))
    idx = np.arange(N, dtype=np.int64)
    axes = [vals[j][(idx // strides[j]) % sizes[j]] for j in range(s)]
    C = []
    for i in range(16):
        if i in act:
            C.append(np.ascontiguousarray(axes[act.index(i)]))
        else:
            C.append(np.full(N, consts[i], dtype=np.int64))
    return c.decrypt(tuple(C), rks, rounds=rounds, vec=True)


def structure_stream(c, rks, rounds, pos, seed, active=None, chunk=1 << 20,
                     limit=None, start=0, stop=None, values=None):
    """`structure_data` as a stream of chunks of at most `chunk` points.

    The 12-round structure has q^3 = 2^48 points, which no machine holds at
    once; every quantity the r_KR = 2 assembly needs from a structure is a
    point sum, so the structure is walked in blocks (see
    assemble_fast.Moments).  Chunk c covers the points [c*chunk, (c+1)*chunk)
    of the same product-set order as `structure_data`.

    `start` / `stop` restrict the walk to the point range [start, stop) of that
    same order without changing any constant or any value (S10: worker w takes
    the slices it owns).  They default to the whole structure, so every
    existing caller is unaffected."""
    F = c.F
    rng = np.random.default_rng(seed)
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    act = active_words(pos, active)
    s = len(act)
    vals, sizes, strides, N = _axis_values(F, act, values)
    if limit is not None:
        N = min(N, int(limit))
    if stop is not None:
        N = min(N, int(stop))
    for lo in range(int(start), N, chunk):
        n = min(chunk, N - lo)
        idx = np.arange(lo, lo + n, dtype=np.int64)
        axes = [vals[j][(idx // strides[j]) % sizes[j]] for j in range(s)]
        C = []
        for i in range(16):
            if i in act:
                C.append(np.ascontiguousarray(axes[act.index(i)]))
            else:
                C.append(np.full(n, consts[i], dtype=np.int64))
        yield c.decrypt(tuple(C), rks, rounds=rounds, vec=True)


class PowerCache:
    """P_i^e for one inner block, computed lazily (exponents can reach q-1
    after the modular reduction, so a dense table is out of the question)."""

    def __init__(self, F, P, b):
        self.F = F
        self.base = [P[4 * b + i] for i in range(4)]
        self.cache = [{0: np.ones(len(P[0]), dtype=np.int64), 1: self.base[i]}
                      for i in range(4)]

    def __getitem__(self, i):
        return _Lazy(self.F, self.cache[i], self.base[i])


class _Lazy:
    def __init__(self, F, cache, base):
        self.F, self.cache, self.base = F, cache, base

    def __getitem__(self, e):
        v = self.cache.get(e)
        if v is None:
            v = self.F.vmul(self[e - 1], self.base)
            self.cache[e] = v
        return v


def block_powers(F, P, b, maxe):
    return PowerCache(F, P, b)


def monomial_values(F, pw, exps):
    acc = None
    for i, e in enumerate(exps):
        if e == 0:
            continue
        acc = pw[i][e] if acc is None else F.vmul(acc, pw[i][e])
    return acc


class MomentCache:
    """sum_P prod_i P_i^{e_i} (optionally weighted by a numeric array) for one
    structure and one inner block; every distinct exponent vector is evaluated
    once, which is what makes the assembly affordable."""

    def __init__(self, F, pw, size):
        self.F, self.pw, self.size = F, pw, size
        self.plain, self.weighted = {}, {}

    def plain_moment(self, pe):
        v = self.plain.get(pe)
        if v is None:
            w = monomial_values(self.F, self.pw, pe)
            v = self.F.vsum(w) if w is not None else (
                self.size % self.F.p if self.F.char != 2 else 0)
            self.plain[pe] = v
        return v

    def weighted_moment(self, pe, tag, arr):
        key = (pe, tag)
        v = self.weighted.get(key)
        if v is None:
            w = monomial_values(self.F, self.pw, pe)
            v = self.F.vsum(self.F.vmul(w, arr) if w is not None else arr)
            self.weighted[key] = v
        return v

    def cross_moment(self, pe, other, pe2, cache2):
        """sum_P P_A^{pe} P_B^{pe2}: not memoisable across blocks, computed
        directly (only needed when two inner blocks are both unknown)."""
        F = self.F
        w1 = monomial_values(F, self.pw, pe)
        w2 = monomial_values(F, cache2.pw, pe2)
        w = w1 if w2 is None else (w2 if w1 is None else F.vmul(w1, w2))
        return F.vsum(w) if w is not None else (
            self.size % F.p if F.char != 2 else 0)


def monomial_value(F, m, rk0, rk1, order):
    """The value of a linearisation monomial at the true key.

    A monomial is (outer, inner): `outer` is None or (j, i) meaning the factor
    k'_i = rk1[4j+i] of outer block j, and `inner` is a tuple of (block, exps)
    with exps the exponents of rk0[4b+0..3]."""
    outer, inner = m
    v = 1 if outer is None else rk1[4 * outer[0] + outer[1]]
    for b, exps in inner:
        for i, e in enumerate(exps):
            if e:
                v = F.mul(v, pow_field(F, rk0[4 * b + i], e))
    return v


def pow_field(F, a, e):
    r = 1
    while e:
        if e & 1:
            r = F.mul(r, a)
        a = F.mul(a, a)
        e >>= 1
    return r


# ------------------------------------------------------------------ solve --
def solve_modp_numpy(p, rows, rhs):
    """Gauss-Jordan over F_p with numpy; returns ({col: value}, rank, #free)."""
    A = np.array(rows, dtype=np.int64) % p
    b = np.array(rhs, dtype=np.int64) % p
    m, n = A.shape
    piv, r = [], 0
    for cidx in range(n):
        nz = np.nonzero(A[r:, cidx])[0]
        if len(nz) == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
            b[r], b[pr] = b[pr], b[r]
        inv = pow(int(A[r, cidx]), p - 2, p)
        A[r] = (A[r] * inv) % p
        b[r] = (b[r] * inv) % p
        col = A[:, cidx].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - col[nzr, None] * A[r][None, :]) % p
            b[nzr] = (b[nzr] - col[nzr] * b[r]) % p
        piv.append(cidx)
        r += 1
        if r == m:
            break
    free = [ci for ci in range(n) if ci not in set(piv)]
    fmask = np.zeros(n, dtype=bool)
    fmask[free] = True
    det = {}
    for i, ci in enumerate(piv):
        if not A[i][fmask].any():
            det[ci] = int(b[i])
    return det, r, len(free)


def solve_modp(F, rows, rhs):
    """Gaussian elimination over F_q; returns (solution or None, rank)."""
    m, n = len(rows), len(rows[0])
    A = [list(r) + [b] for r, b in zip(rows, rhs)]
    piv, r = [], 0
    for cidx in range(n):
        p = next((i for i in range(r, m) if A[i][cidx]), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = F.inv(A[r][cidx])
        A[r] = [F.mul(v, inv) for v in A[r]]
        for i in range(m):
            if i != r and A[i][cidx]:
                f = A[i][cidx]
                A[i] = [F.sub(u, F.mul(f, v)) for u, v in zip(A[i], A[r])]
        piv.append(cidx)
        r += 1
        if r == m:
            break
    free = [ci for ci in range(n) if ci not in piv]
    det = {}
    for i, ci in enumerate(piv):
        if all(A[i][fc] == 0 for fc in free):
            det[ci] = A[i][n]
    return det, r, len(free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="toy-257")
    ap.add_argument("--layers", type=int, default=5, help="zero-sum layers l")
    ap.add_argument("--pos", type=int, default=3, help="active ciphertext word (s = 1 alias)")
    ap.add_argument("--active", default=None,
                    help="comma list of active ciphertext words, each over all of F_q "
                         "(q^s points per structure); overrides --pos")
    ap.add_argument("--unknown-blocks", default="0")
    ap.add_argument("--outer-blocks", default="0")
    ap.add_argument("--structures", type=int, default=None)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rounds = a.layers + 2
    c = DuX(a.instance, rounds=rounds)
    F = c.F
    order = F.q - 1
    unknown = [int(v) for v in a.unknown_blocks.split(",")] if a.unknown_blocks else []
    outer = [int(v) for v in a.outer_blocks.split(",")]

    for spec, tag in (([0, 1, 2, 3], "all four inner blocks unknown"),
                      (unknown, f"unknown blocks {unknown}")):
        mons, joint, U, S = monomial_set(F, c.alpha, spec,
                                         [0, 1, 2, 3] if spec == [0, 1, 2, 3] else outer)
        print(f"monomials, {tag}: {len(mons)} = 2^{np.log2(len(mons)):.1f} "
              f"(|U| = {len(U)} per block before the sum, {len(S)} after)")
        if spec == [0, 1, 2, 3]:
            full_count = len(mons)
    if a.count_only:
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            json.dump({"instance": a.instance, "monomials_all_blocks": full_count,
                       "monomials_prototype": len(mons)},
                      open(os.path.join(a.out, "monomial_counts.json"), "w"), indent=1)
        return

    mons, joint, U, S = monomial_set(F, c.alpha, unknown, outer)
    idx = {m: i for i, m in enumerate(mons)}
    n_struct = a.structures or (len(mons) // len(outer) + 8)
    act = active_words(a.pos, a.active)
    npts = F.q ** len(act)
    print(f"prototype: {rounds} rounds, l = {a.layers}, unknown inner blocks {unknown}, "
          f"outer blocks {outer}; active words {act}; "
          f"{n_struct} structures x {npts} chosen ciphertexts "
          f"= 2^{np.log2(n_struct * npts):.1f}")

    rng = np.random.default_rng(a.seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    t1 = t_xor(rks[1])
    row = linear_row(c, t1)
    known = [b for b in range(4) if b not in unknown]
    maxe_k = max(max(pe) for co in range(4) for _, pe, _ in joint[co])

    t0 = time.time()
    rows, rhs = [], []
    for st in range(n_struct):
        P = structure_data(c, rks, rounds, a.pos, a.seed + 5000 + st, a.active)
        # numeric S-box outputs of the KNOWN inner blocks
        Ynum = {}
        for b in known:
            x = tuple(F.vadd(P[4 * b + i], rks[0][4 * b + i]) for i in range(4))
            from dux.sbox import vS
            y = vS(F, x, c.alpha)
            for i in range(4):
                Ynum[4 * b + i] = y[i]
        pw = {b: block_powers(F, P, b, maxe_k) for b in unknown}
        mc = {b: MomentCache(F, pw[b], len(P[0])) for b in unknown}

        def T_poly(s):
            """sum_P Y_s as {inner monomial -> coeff} (or a number if known)."""
            b, co = s // 4, s % 4
            if b in known:
                return {(): F.vsum(Ynum[s])}
            out = {}
            for ke, pe, coeff in joint[co]:
                if not any(pe):
                    continue                    # coefficient |structure| = 0
                v = mc[b].plain_moment(pe)
                if v:
                    key = ((b, ke),)
                    out[key] = F.add(out.get(key, 0), F.mul(coeff, v))
            return out

        def Q_poly(s, t):
            """sum_P Y_s Y_t as {inner monomial -> coeff}."""
            bs, cs = s // 4, s % 4
            bt, ct = t // 4, t % 4
            out = {}
            if bs in known and bt in known:
                return {(): F.vsum(F.vmul(Ynum[s], Ynum[t]))}
            if bs in known or bt in known:
                if bs in known:
                    bs, cs, bt, ct, s, t = bt, ct, bs, cs, t, s
                for ke, pe, coeff in joint[cs]:
                    v = mc[bs].weighted_moment(pe, t, Ynum[t])
                    if v:
                        key = ((bs, ke),)
                        out[key] = F.add(out.get(key, 0), F.mul(coeff, v))
                return out
            for ke1, pe1, c1 in joint[cs]:
                for ke2, pe2, c2 in joint[ct]:
                    if bs == bt:
                        v = mc[bs].plain_moment(add_exp(pe1, pe2, order))
                        key = ((bs, add_exp(ke1, ke2, order)),)
                    else:
                        v = mc[bs].cross_moment(pe1, bt, pe2, mc[bt])
                        key = tuple(sorted([(bs, ke1), (bt, ke2)]))
                    if v:
                        out[key] = F.add(out.get(key, 0), F.mul(F.mul(c1, c2), v))
            return out

        Ts = [T_poly(s) for s in range(16)]
        Qcache = {}

        def Q(s, t):
            key = (s, t)
            v = Qcache.get(key)
            if v is None:
                v = Q_poly(s, t)
                Qcache[key] = v
            return v

        for j in outer:
            m0 = [row[(s - 4 * j - 0) % 16] for s in range(16)]
            m2 = [row[(s - 4 * j - 2) % 16] for s in range(16)]
            m3 = [row[(s - 4 * j - 3) % 16] for s in range(16)]
            acc = {}

            def addto(poly, scale, outer_tag=None):
                for key, v in poly.items():
                    mk = (outer_tag, key)
                    acc[mk] = F.add(acc.get(mk, 0), F.mul(v, scale))

            for s in range(16):
                if m2[s]:
                    addto(Ts[s], m2[s])
                if m0[s]:
                    addto(Ts[s], F.sub(0, m0[s]), (j, 3))
                if m3[s]:
                    addto(Ts[s], F.sub(0, m3[s]), (j, 0))
            for s in range(16):
                if not m0[s]:
                    continue
                for t in range(16):
                    if not m3[t]:
                        continue
                    addto(Q(s, t), F.sub(0, F.mul(m0[s], m3[t])))
            r = [0] * len(mons)
            for mk, v in acc.items():
                if mk not in idx:
                    raise RuntimeError(f"monomial {mk} outside the predicted set")
                r[idx[mk]] = v
            rows.append(r)
            rhs.append(0)
        if (st + 1) % 20 == 0 or st == n_struct - 1:
            print(f"  {st + 1}/{n_struct} structures ({time.time() - t0:.1f}s)")

    # pin the constant monomial and solve
    ci = idx[(None, ())]
    R = [[v for t, v in enumerate(r) if t != ci] for r in rows]
    B = [F.sub(0, r[ci]) for r in rows]
    cols = [m for m in mons if m != (None, ())]
    if F.char != 2:
        det, rank, nfree = solve_modp_numpy(F.p, R, B)
    else:
        det, rank, nfree = solve_modp(F, R, B)
    print(f"  rank {rank} / {len(cols)} unknowns ({nfree} free), "
          f"{len(det)} monomials pinned down")

    ok = {}
    for b in unknown:
        for i in range(4):
            e = tuple(1 if t == i else 0 for t in range(4))
            m = (None, ((b, e),))
            if m in cols and cols.index(m) in det:
                ok[(b, i)] = det[cols.index(m)]
    truth = {(b, i): rks[0][4 * b + i] for b in unknown for i in range(4)}
    print(f"  inner key words recovered: {ok}")
    print(f"  truth                    : {truth}")
    # Once the inner round key is known, equation (*) is LINEAR in (k'_0, k'_3)
    # of each outer block, so two extra structures finish the job.
    outer_ok = {}
    if ok == truth:
        from dux.sbox import vS
        rk0 = list(rks[0])
        for b in unknown:
            for i in range(4):
                rk0[4 * b + i] = ok[(b, i)]
        Ssum = {j: [] for j in outer}
        for st in range(2):
            P = structure_data(c, rks, rounds, a.pos, a.seed + 90000 + st, a.active)
            Y = []
            for b in range(4):
                Y += list(vS(F, tuple(F.vadd(P[4 * b + i], rk0[4 * b + i])
                                      for i in range(4)), c.alpha))
            for j in outer:
                u = []
                for i in range(4):
                    acc = np.zeros(len(P[0]), dtype=np.int64)
                    for l in range(16):
                        if row[l]:
                            acc = F.vadd(acc, F.vmul(Y[(4 * j + i + l) % 16],
                                                     np.full(len(P[0]), row[l],
                                                             dtype=np.int64)))
                    u.append(acc)
                Ssum[j].append((F.vsum(u[0]), F.vsum(u[2]), F.vsum(u[3]),
                                F.vsum(F.vmul(u[0], u[3]))))
        for j in outer:
            (s0a, s2a, s3a, s03a), (s0b, s2b, s3b, s03b) = Ssum[j]
            # k'_3 * s0 + k'_0 * s3 = s2 - s03
            A = [[s0a, s3a], [s0b, s3b]]
            rhs2 = [F.sub(s2a, s03a), F.sub(s2b, s03b)]
            dt = F.sub(F.mul(A[0][0], A[1][1]), F.mul(A[0][1], A[1][0]))
            if dt == 0:
                continue
            iv = F.inv(dt)
            k3p = F.mul(iv, F.sub(F.mul(A[1][1], rhs2[0]), F.mul(A[0][1], rhs2[1])))
            k0p = F.mul(iv, F.sub(F.mul(A[0][0], rhs2[1]), F.mul(A[1][0], rhs2[0])))
            outer_ok[(j, 0)] = k0p
            outer_ok[(j, 3)] = k3p
    print(f"  outer key words recovered: {outer_ok}")
    print("  truth                    : "
          + str({(j, i): rks[1][4 * j + i] for j in outer for i in (0, 3)}))
    success = (ok == truth) and all(outer_ok.get((j, i)) == rks[1][4 * j + i]
                                    for j in outer for i in (0, 3))
    el = time.time() - t0
    print(f"  SUCCESS: {success};  data 2^{np.log2(n_struct * npts):.1f} chosen "
          f"ciphertexts;  {el:.1f}s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({"instance": a.instance, "rounds": rounds, "layers": a.layers,
                   "unknown_blocks": unknown, "outer_blocks": outer,
                   "active_words": act, "points_per_structure": npts,
                   "monomials": len(mons), "monomials_all_blocks": full_count,
                   "structures": n_struct, "rank": rank, "free": nfree,
                   "pinned": len(det), "success": bool(success),
                   "data_log2": round(float(np.log2(n_struct * npts)), 2),
                   "elapsed_s": round(el, 1)},
                  open(os.path.join(a.out, f"toy_{a.instance}_r{rounds}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
