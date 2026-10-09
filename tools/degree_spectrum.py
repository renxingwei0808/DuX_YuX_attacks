"""W17 / O12 Sect. 5.3 -- the EXACT reduced degree of every state word, from
one multiplicative character sum per exponent.

Let Z_i be a state word of a full-field single-word structure, i.e. a
polynomial in the active ciphertext word X reduced modulo X^q - X.  Then

    sum_{x in F_q} x^a Z_i(x)  =  - [X^{q-1-a}] Z_i(x)       for 0 <= a <= q-2,

because sum_x x^m = -1 exactly when m is a positive multiple of q - 1 and 0
otherwise (and sum_x x^0 = q = 0).  So scanning a upwards, the FIRST a* with a
nonzero sum gives the exact reduced degree

    deg Z_i = q - 1 - a*,

and if every a in [0, q-2] gives zero the word is constant.  a = 0 is the
ordinary zero-sum test, so this is the same experiment E03/E04 run for every
weight at once -- the "degree spectrum" of the state.

This replaces the O(q^2) exact univariate polynomial of
experiments/E02_degree_bounds/symbolic_layers.py (which multiplies dense
length-q polynomials layer after layer) by 2^k decryptions plus one weighted
sum per exponent, and it works for BOTH ciphers and both fields through
dux/registry.py.

Modes
  naive   S[a][i] = sum_x x^a Z[x][i] for a = 0 .. max_a, as a Vandermonde
          product in blocks (F_p) or a log-table XOR reduction (F_{2^n}).
          Cost O(q * max_a * 16); q <= 2^8 is instant, q = 2^16 / 65537 needs
          --max-a (then the answer is "the degree is at least q - 1 - max_a").
  ntt     F_65537 only: the multiplicative group is cyclic of order 2^16 with
          generator 3, so the whole spectrum of one word is a single length-2^16
          Fermat NTT of the sequence Z(3^k).  x = 0 contributes only to a = 0.

Usage
  python tools/degree_spectrum.py --instance yuxtoy-257 --pos 0 --layers 8 --keys 2
  python tools/degree_spectrum.py --instance toy-257 --pos 3 --layers 8 --keys 2
  python tools/degree_spectrum.py --instance yupx-65537 --pos 0 --layers 9 \
      --keys 2 --mode ntt
  python tools/degree_spectrum.py --instance yu2x-16 --pos 0 --layers 9 \
      --keys 2 --max-a 20000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from dux.registry import cipher_family, get_cipher  # noqa: E402
from cipher_degree import profile                   # noqa: E402


# ------------------------------------------------------------------ sums ---
def spectrum_naive(F, xs, Z, max_a, block=256):
    """S[a][i] = sum_x x^a Z[x][i] for a = 0..max_a, as an (max_a+1, nwords)."""
    q = len(xs)
    nw = Z.shape[1]
    out = np.zeros((max_a + 1, nw), dtype=np.int64)
    if F.char == 2:
        xp = np.ones(q, dtype=np.int64)
        for a in range(max_a + 1):
            for i in range(nw):
                out[a, i] = np.bitwise_xor.reduce(F.vmul(xp, Z[:, i]))
            xp = F.vmul(xp, xs)
        return out
    p = F.p
    Zf = (Z % p).astype(np.float64)
    chunk = max(1, int(2 ** 53 // (p * p)))
    xp = np.ones(q, dtype=np.int64)
    a = 0
    while a <= max_a:
        k = min(block, max_a + 1 - a)
        V = np.empty((k, q), dtype=np.int64)
        for t in range(k):
            V[t] = xp
            xp = (xp * xs) % p
        Vf = V.astype(np.float64)
        if q <= chunk:
            out[a:a + k] = (Vf @ Zf % p).astype(np.int64)
        else:
            acc = np.zeros((k, nw), dtype=np.int64)
            for lo in range(0, q, chunk):
                hi = min(lo + chunk, q)
                acc = (acc + (Vf[:, lo:hi] @ Zf[lo:hi] % p).astype(np.int64)) % p
            out[a:a + k] = acc
        a += k
    return out


def _ntt_65537(v):
    """Length-2^16 NTT modulo the Fermat prime 65537 (generator 3)."""
    p = 65537
    n = len(v)
    assert n == 1 << 16
    g = pow(3, (p - 1) // n, p)
    a = np.asarray(v, dtype=np.int64) % p
    # iterative Cooley-Tukey, bit-reversal first
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        w = pow(g, n // length, p)
        ws = np.empty(length // 2, dtype=np.int64)
        cur = 1
        for k in range(length // 2):
            ws[k] = cur
            cur = cur * w % p
        for i in range(0, n, length):
            u = a[i:i + length // 2]
            v2 = a[i + length // 2:i + length] * ws % p
            a[i:i + length // 2] = (u + v2) % p
            a[i + length // 2:i + length] = (u - v2) % p
        length <<= 1
    return a


def spectrum_ntt_65537(Z_by_gen, Z0):
    """S[a][i] for a = 0..q-2 from the values on the multiplicative group.

    Z_by_gen[k][i] = Z_i(3^k) and Z0[i] = Z_i(0).  Then
        sum_x x^a Z_i(x) = Z0[i] * [a == 0] + sum_k (3^k)^a Z_i(3^k)
    and the second sum is the NTT of the sequence at index a."""
    p = 65537
    n = Z_by_gen.shape[0]
    out = np.empty((n, Z_by_gen.shape[1]), dtype=np.int64)
    for i in range(Z_by_gen.shape[1]):
        out[:, i] = _ntt_65537(Z_by_gen[:, i].copy())
    out[0] = (out[0] + Z0) % p
    return out


# ------------------------------------------- multi-variable exact degrees --
# W20 / O13.  The univariate scan above answers "what is the exact reduced
# degree of Z_i(X)".  For a structure with s active words -- and for a FULL
# BLOCK (O11), where the four free variables are the first S layer's output
# y = S^{-1}(x + rk^r) -- the quantity the max-plus bound D predicts is the
# exact TOTAL degree of the reduced polynomial Z_i(X_1, ..., X_s).
#
# Reducing modulo X^q - X in every variable, a function F_q^s -> F_q has a
# unique representative with all exponents in [0, q-1], and its coefficients
# come from ONE q x q matrix applied along each axis:
#
#     c_0 = f(0),      c_e = - sum_x x^{q-1-e} f(x)   for 1 <= e <= q-1
#
# (the e >= 1 rows are exactly the character sums the univariate mode uses;
# the e = 0 row is the evaluation at 0, which those sums cannot see).  Taking
# the coefficient array rather than the raw character sums is what makes the
# multi-variable case correct: a monomial with some e_i = 0 is invisible to the
# forward sums, and at layer 2 of YuX those are precisely the monomials that
# survive.
def inverse_evaluation_matrix(F):
    """W[e][x] with c_e = sum_x W[e][x] f(x) for the reduced representative."""
    q = F.q
    W = np.zeros((q, q), dtype=np.int64)
    W[0][0] = 1
    xs = np.arange(q, dtype=np.int64)
    xp = np.ones(q, dtype=np.int64)               # x^a, a = 0, 1, ...
    for a in range(q - 1):                        # e = q - 1 - a
        W[q - 1 - a] = xp if F.char == 2 else F.vneg(xp)
        xp = F.vmul(xp, xs)
    return W


def _axis_transform(F, W, A):
    """out[e, r] = sum_x W[e, x] A[x, r] over the leading axis."""
    q = A.shape[0]
    if F.char != 2:
        p = F.p
        return (W.astype(np.float64) @ (A % p).astype(np.float64) % p
                ).astype(np.int64)
    out = np.zeros_like(A)
    for e in range(q):
        col = W[e]
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        prod = F.vmul(col[nz][:, None], A[nz])
        out[e] = np.bitwise_xor.reduce(prod, axis=0)
    return out


def exact_coefficients(F, Z, s):
    """The reduced coefficient array of Z: (q,)*s x nwords."""
    q = F.q
    nw = Z.shape[1]
    W = inverse_evaluation_matrix(F)
    A = Z.reshape((q,) * s + (nw,))
    for axis in range(s):
        A = np.moveaxis(A, axis, 0)
        shp = A.shape
        A = _axis_transform(F, W, A.reshape(q, -1)).reshape(shp)
        A = np.moveaxis(A, 0, axis)
    return A


def total_degrees(F, Z, s):
    """max{ |e| : c_e != 0 } per word, or None for the zero polynomial."""
    q = F.q
    A = exact_coefficients(F, Z, s)
    nw = A.shape[-1]
    idx = np.arange(q, dtype=np.int64)
    tot = idx.reshape((q,) + (1,) * (s - 1))
    for axis in range(1, s):
        tot = tot + idx.reshape((1,) * axis + (q,) + (1,) * (s - 1 - axis))
    out = []
    for w in range(nw):
        nz = A[..., w] != 0
        out.append(int(tot[nz].max()) if nz.any() else None)
    return out


def multiword_structure(c, rks, active, free_block, consts, rounds=None):
    """The chosen-ciphertext structure and the axis order.

    `active` words each run over F_q (product-set order, axis 0 slowest).  A
    `free_block` runs over F_q^4 in the FIRST S LAYER'S OUTPUT y (O11): the
    ciphertext is C = ARK(SL(Y), rk^r) with block `free_block` of Y free, so
    the state entering layer 2 is a polynomial in the four y variables of
    degree (1,1,1,1) -- exactly what `cipher_degree.profile` assumes."""
    F = c.F
    q = F.q
    r = c.r if rounds is None else rounds
    axes = list(active) if free_block is None else \
        [4 * free_block + t for t in range(4)] + list(active)
    s = len(axes)
    N = q ** s
    idx = np.arange(N, dtype=np.int64)
    vals = [(idx // (q ** (s - 1 - j))) % q for j in range(s)]
    if free_block is None:
        C = [np.full(N, consts[i], dtype=np.int64) for i in range(16)]
        for j, w in enumerate(axes):
            C[w] = vals[j]
        return tuple(C), axes
    Y = [np.full(N, consts[i], dtype=np.int64) for i in range(16)]
    for j, w in enumerate(axes):
        Y[w] = vals[j]
    C = c.ARK(c.SL(tuple(Y), True), rks[r], True)
    return tuple(C), axes


def exact_total_degrees(instance, layers, seed, active=(), free_block=None,
                        rounds=None):
    """[per layer][per word] the exact total degree of the reduced polynomial."""
    c = get_cipher(instance, rounds=rounds) if rounds else get_cipher(instance)
    F = c.F
    rng = np.random.default_rng(seed)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    C, axes = multiword_structure(c, rks, active, free_block, consts, rounds)
    s = len(axes)
    states = decrypt_layers_fixed_L0(c, C, rks, layers, rounds)
    out = []
    for st in states:
        Z = np.stack([np.asarray(st[i], dtype=np.int64) for i in range(16)],
                     axis=1)
        out.append(total_degrees(F, Z, s))
    return c, axes, out


# ------------------------------------------------------------------ run ----
def decrypt_layers_fixed_L0(c, C, rks, layers, rounds=None):
    """`decrypt_layers` with the linear layer forced to L0.

    DuX picks L0 or L1 from two bits of rk^i, and L1 = Rot_{-4} o L0 permutes
    the BLOCKS (O2).  The max-plus profile is computed for L0, so comparing a
    per-word bound against a run of the real cipher would mismatch whenever
    some t(rk^i) = 1.  Analyses fix L0 (O2 says the equation set is the same),
    and so does this tool.  For YuX there is only one layer, so this is
    identical to `decrypt_layers`."""
    r = c.r if rounds is None else rounds
    states = []
    x = c.ARK_inv(tuple(C), rks[r], True)
    x = c.SL_inv(x, True)
    states.append(x)
    for k in range(1, layers):
        i = r - k
        x = c.ARK_inv(x, rks[i], True)
        x = c.lin.L_inv(x, 0, True)
        x = c.SL_inv(x, True)
        states.append(x)
    return states


def exact_degrees(instance, pos, layers, seed, mode="naive", max_a=None,
                  words=None, rounds=None):
    """[per layer][per word] the exact reduced degree, or None if the scan was
    truncated before finding it (then the degree is < q - 1 - max_a is FALSE:
    it means the degree is at most q - 1 - max_a - 1 ... see `truncated`)."""
    c = get_cipher(instance, rounds=rounds) if rounds else get_cipher(instance)
    F = c.F
    q = F.q
    rng = np.random.default_rng(seed)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, q, size=16)]
    wanted = list(range(16)) if words is None else list(words)
    if mode == "ntt":
        assert F.char != 2 and F.q == 65537, "the NTT mode is F_65537 only"
        gpow = np.empty(q - 1, dtype=np.int64)
        cur = 1
        for k in range(q - 1):
            gpow[k] = cur
            cur = cur * 3 % F.p
        xs = np.concatenate([np.array([0], dtype=np.int64), gpow])
    else:
        xs = np.arange(q, dtype=np.int64)
    C = []
    for i in range(16):
        C.append(xs.copy() if i == pos else np.full(len(xs), consts[i], dtype=np.int64))
    states = decrypt_layers_fixed_L0(c, tuple(C), rks, layers, rounds)
    ma = (q - 2) if max_a is None else min(max_a, q - 2)
    out, truncated = [], []
    for st in states:
        Z = np.stack([np.asarray(st[i], dtype=np.int64) for i in wanted], axis=1)
        if mode == "ntt":
            S = spectrum_ntt_65537(Z[1:], Z[0])
        else:
            S = spectrum_naive(F, xs, Z, ma)
        degs, trunc = [], []
        for t in range(len(wanted)):
            nz = np.flatnonzero(S[:, t] % (F.q if F.char == 2 else F.p))
            if nz.size:
                degs.append(int(q - 1 - nz[0]))
                trunc.append(False)
            else:
                degs.append(None)
                trunc.append(S.shape[0] <= q - 1)
        out.append(degs)
        truncated.append(trunc)
    return c, wanted, out, truncated


def multi_main(a):
    """W20: the exact TOTAL degree of a multi-word or full-block structure."""
    fam = cipher_family(a.instance)
    active = [int(v) for v in a.active.split(",") if v != ""] if a.active else []
    free = None if a.full_block is None else int(a.full_block)
    res = {"instance": a.instance, "cipher": fam, "active": active,
           "full_block": free, "layers": a.layers, "seeds": [], "per_key": []}
    t0 = time.time()
    c = None
    seeds = ([int(v) for v in a.seeds.split(",")] if a.seeds
             else [a.seed + k for k in range(a.keys)])
    a.keys = len(seeds)
    for seed in seeds:
        c, axes, tot = exact_total_degrees(a.instance, a.layers, seed, active,
                                           free, a.rounds)
        res["seeds"].append(seed)
        res["per_key"].append(tot)
        res["axes"] = axes
        print(f"--- {a.instance} seed {seed} (axes {axes}) ---")
        for l, row in enumerate(tot, 1):
            print(f"  layer {l:2d}: exact total degrees {row}")
    q = c.F.q
    s = len(res["axes"])
    bound = profile(active, a.layers, "dec", c.F.char == 2, fam,
                    () if free is None else (free,))
    res["maxplus_bound"] = [[int(v) for v in row] for row in bound]
    res["saturation"] = s * (q - 1)
    res["verdicts"] = []
    print("--- exact total degree vs max-plus (saturation at "
          f"s(q-1) = {res['saturation']}) ---")
    for l in range(a.layers):
        b = res["maxplus_bound"][l]
        per = []
        for w in range(16):
            # a word whose bound already reaches s(q-1) cannot be compared:
            # the reduced representative can never show a larger degree.
            if b[w] >= res["saturation"]:
                per.append("saturated")
                continue
            ex = [0 if res["per_key"][k][l][w] is None else
                  res["per_key"][k][l][w] for k in range(a.keys)]
            if len(set(ex)) != 1:
                per.append("KEY-DEPENDENT")
            elif b[w] == 0:
                per.append("constant" if ex[0] == 0 else "LOOSE")
            elif ex[0] == b[w]:
                per.append("tight")
            else:
                per.append("LOOSE")
        res["verdicts"].append(per)
        tally = {v: per.count(v) for v in sorted(set(per))}
        print(f"  layer {l + 1:2d}: {tally}  bound {b[:4]} exact "
              f"{res['per_key'][0][l][:4]}")
    res["all_tight_before_saturation"] = all(
        v not in ("LOOSE", "KEY-DEPENDENT")
        for row in res["verdicts"] for v in row)
    res["key_independent"] = all(v != "KEY-DEPENDENT"
                                 for row in res["verdicts"] for v in row)
    print(f"  => key-independent across {a.keys} keys: {res['key_independent']};"
          f" every unsaturated word meets its bound: "
          f"{res['all_tight_before_saturation']}")
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yuxtoy-257")
    ap.add_argument("--pos", type=int, default=0,
                    help="active ciphertext word (block 0, position `pos`)")
    ap.add_argument("--active", default=None,
                    help="W20: comma list of active ciphertext words; with more "
                         "than one word (or with --full-block) the tool reports "
                         "the exact TOTAL degree instead of the univariate one")
    ap.add_argument("--full-block", default=None,
                    help="W20 / O11: one whole block runs over F_q^4 in the "
                         "first S layer's output y")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seeds", default=None,
                    help="explicit comma list of key seeds (repo convention for "
                         "three keys: 2026,7,11); overrides --keys/--seed")
    ap.add_argument("--mode", default="naive", choices=("naive", "ntt"))
    ap.add_argument("--max-a", type=int, default=None,
                    help="stop the scan at this exponent (big fields)")
    ap.add_argument("--words", default=None, help="comma list, default all 16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nact = len([v for v in (a.active or "").split(",") if v != ""])
    if a.full_block is not None or nact > 1:
        return multi_main(a)
    if nact == 1:
        a.pos = int(a.active)
    words = [int(v) for v in a.words.split(",")] if a.words else None
    fam = cipher_family(a.instance)
    res = {"instance": a.instance, "cipher": fam, "pos": a.pos,
           "layers": a.layers, "mode": a.mode, "max_a": a.max_a,
           "seeds": [], "per_key": []}
    t0 = time.time()
    for k in range(a.keys):
        seed = a.seed + k
        c, wanted, degs, trunc = exact_degrees(a.instance, a.pos, a.layers, seed,
                                               a.mode, a.max_a, words, a.rounds)
        res["seeds"].append(seed)
        res["per_key"].append(degs)
        res["truncated"] = trunc
        res["words"] = wanted
        print(f"--- {a.instance} seed {seed} (active word {a.pos}) ---")
        for l, row in enumerate(degs, 1):
            print(f"  layer {l:2d}: exact degrees {row}")
    bound = profile([a.pos], a.layers, cipher=fam)
    res["maxplus_bound"] = [[int(b[i]) for i in res["words"]] for b in bound]
    res["verdicts"] = []
    q = c.F.q
    print("--- exact vs max-plus, per word ---")
    print("    tight     : exact degree == the bound and below q-1 (the bound is met)")
    print("    saturated : the bound is >= q-1, so the exact degree cannot show it")
    print("    constant  : the bound is 0 and the word really is constant")
    print("    LOOSE     : exact degree < bound while the bound is still < q-1")
    print("    truncated : --max-a stopped the scan before it could see this "
          "degree; all we know is exact < q-1-max_a")
    lo = (q - 1 - a.max_a) if a.max_a is not None else 0
    for l in range(a.layers):
        b = res["maxplus_bound"][l]
        per = []
        for t in range(len(res["words"])):
            ex = [res["per_key"][k][l][t] for k in range(a.keys)]
            if b[t] == 0:
                per.append("constant" if all(e is None for e in ex) else "LOOSE")
            elif b[t] >= q - 1:
                per.append("saturated")
            elif all(e == b[t] for e in ex):
                per.append("tight")
            elif a.max_a is not None and all(e is None for e in ex) and b[t] < lo:
                # the scan never reached a = q-1-b[t], so `None` is "unknown",
                # NOT "the exact degree is smaller than the bound".
                per.append("truncated")
            else:
                per.append("LOOSE")
        res["verdicts"].append(per)
        tally = {v: per.count(v) for v in sorted(set(per))}
        print(f"  layer {l + 1:2d}: {tally}   bound[0:4] {b[:4]} "
              f"exact[0:4] {res['per_key'][0][l][:4]}")
    res["all_tight_before_saturation"] = all(
        v != "LOOSE" for row in res["verdicts"] for v in row)
    res["max_a"] = a.max_a
    ntrunc = sum(row.count("truncated") for row in res["verdicts"])
    print(f"  => every unsaturated word meets its bound: "
          f"{res['all_tight_before_saturation']}"
          + (f"   ({ntrunc} words undecided: --max-a truncated the scan)"
             if ntrunc else ""))
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
