"""W20 / O13 -- the EXACT FORMAL degree of a YuX (or DuX) state word.

`tools/zero_sum_criterion.py` propagates a max-plus UPPER bound on the formal
total degree; `tools/degree_spectrum.py` measures the exact REDUCED degree from
character sums.  Between the two sits the object O13 is about: the exact
FORMAL degree, i.e. the degree of the state word as an unreduced polynomial in
the active ciphertext word(s).  Over F_p the two coincide with the max-plus
bound on every YuX cell we have measured; in characteristic 2 they do not, and
Y07 measures the gap at roughly 7 %.

This tool computes the exact formal degree with no bound, no window and no
sampling: it expands the decryption symbolically, layer by layer, as DENSE
polynomials over F_q, using the closed forms in `dux/sbox.py` / `yux/sbox.py`
and the repository's rotation sets.  Nothing is reduced modulo X^q - X, so the
answer is directly comparable with the max-plus bound, and the whole thing is
exact for as many layers as the degree budget allows (the polynomial length
grows like 4^layer for YuX, 3.73^layer for DuX).

WHY IT IS CHEAP FOR A SINGLE ACTIVE WORD.  With one active word X the first S
layer already collapses to univariate polynomials -- for YuX,

    Y_0 = X + c,     Y_1 = Y_0 + g,     Y_2 = (c_3 + 1) Y_0 + c',
    Y_3 = Y_0 (Y_0 + g) + c''                                            (*)

with g = c_1 + c_2 c_3 + alpha a CONSTANT -- so every later word is a
polynomial in X and dense arithmetic suffices.  (*) is also the mechanism: the
L^{-1} row sums that the max-plus recurrence treats as independent are not.  In
characteristic 2 the two rows that feed positions 1 and 3 of block 1 satisfy

    u_5 + u_7 = (Y_1 + Y_3) + (Y_0 + Y_3) = Y_0 + Y_1 = g ,

so the quadratic term u_2 (u_1 + u_3) of

    y_1 = u_0 + u_1 + u_3 + u_2 (u_1 + u_3)      (YuX S^{-1}, char 2)

degenerates to g * u_2 -- degree 1 instead of 3.  Over F_p the same sum is
Y_0 + Y_1 + 2 Y_3, which is NOT constant, and no cancellation happens.
`--explain` prints, for every block, which L^{-1} row combinations collapse.

Usage
  python tools/char2_exact_degree.py --instance yu2x-8 --pos 0 --layers 4
  python tools/char2_exact_degree.py --instance yuxtoy-257 --pos 0 --layers 4
  python tools/char2_exact_degree.py --instance yu2x-8 --pos 0 --layers 3 \
      --explain --keys 3
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

from cipher_degree import CIPHERS, profile                  # noqa: E402
from dux.registry import cipher_family, get_cipher          # noqa: E402


# --------------------------------------------------------- dense polynomials
class Poly:
    """A dense univariate polynomial over F_q, lowest degree first."""

    __slots__ = ("F", "c")

    def __init__(self, F, c):
        self.F = F
        c = np.asarray(c, dtype=np.int64)
        nz = np.flatnonzero(c)
        self.c = c[:int(nz[-1]) + 1] if nz.size else np.zeros(0, dtype=np.int64)

    @classmethod
    def const(cls, F, v):
        return cls(F, [int(v) % (F.q if F.char == 2 else F.p)])

    @classmethod
    def x(cls, F, shift=0):
        return cls(F, [shift, 1])

    @property
    def degree(self):
        return len(self.c) - 1 if len(self.c) else None

    def __add__(self, other):
        F = self.F
        n = max(len(self.c), len(other.c))
        a = np.zeros(n, dtype=np.int64)
        b = np.zeros(n, dtype=np.int64)
        a[:len(self.c)] = self.c
        b[:len(other.c)] = other.c
        return Poly(F, (a ^ b) if F.char == 2 else (a + b) % F.p)

    __sub__ = __add__ if False else None      # replaced below

    def __mul__(self, other):
        F = self.F
        la, lb = len(self.c), len(other.c)
        if not la or not lb:
            return Poly(F, [])
        if min(la, lb) >= FFT_CUTOFF:
            return Poly(F, _fft_convolve(F, self.c, other.c))
        a, b = (self, other) if la <= lb else (other, self)
        out = np.zeros(la + lb - 1, dtype=np.int64)
        nb = len(b.c)
        if F.char == 2:
            for i, ai in enumerate(a.c):
                if ai:
                    out[i:i + nb] ^= F.vmul(
                        np.full(nb, int(ai), dtype=np.int64), b.c)
        else:
            for i, ai in enumerate(a.c):
                if ai:
                    out[i:i + nb] = (out[i:i + nb] + int(ai) * b.c) % F.p
        return Poly(F, out)


def _scal(self, v):
    F = self.F
    v = int(v) % (F.q if F.char == 2 else F.p)
    if v == 0 or not len(self.c):
        return Poly(F, [])
    return Poly(F, F.vmul(np.full(len(self.c), v, dtype=np.int64), self.c)
                if F.char == 2 else (self.c * v) % F.p)


Poly.scal = _scal


def _sub(self, other):
    F = self.F
    if F.char == 2:
        return self + other
    n = max(len(self.c), len(other.c))
    a = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    a[:len(self.c)] = self.c
    b[:len(other.c)] = other.c
    return Poly(F, (a - b) % F.p)


Poly.__sub__ = _sub


FFT_CUTOFF = 512          # below this the direct convolution is faster


def _bitplanes(F, c):
    """(L, n) 0/1 array: coefficient m of X as a polynomial in t."""
    n = F.n
    sh = np.arange(n, dtype=np.int64)
    return ((c[:, None] >> sh[None, :]) & 1).astype(np.float64)


def _fft_convolve(F, a, b):
    """Exact polynomial product over F_q via a real FFT.

    Characteristic 2: a field element is a polynomial in t of degree < n, so the
    product is a 2-D convolution (length in X, degree in t) of 0/1 arrays; the
    integer entries stay below len * n, far inside float64's exact range, and
    the t-degrees above n - 1 are folded back with the field's own modulus.
    F_p: the coefficients are split into two 8-bit limbs so the convolved
    integers stay below 2^52."""
    la, lb = len(a), len(b)
    nx = 1
    while nx < la + lb - 1:
        nx <<= 1
    if F.char == 2:
        n = F.n
        nt = 1
        while nt < 2 * n - 1:
            nt <<= 1
        A = np.zeros((nx, nt)); B = np.zeros((nx, nt))
        A[:la, :n] = _bitplanes(F, a)
        B[:lb, :n] = _bitplanes(F, b)
        C = np.fft.irfft2(np.fft.rfft2(A) * np.fft.rfft2(B), s=(nx, nt))
        C = np.rint(C).astype(np.int64)[:la + lb - 1, :2 * n - 1] & 1
        # fold t^k, k >= n, back with the field modulus
        red = np.zeros((2 * n - 1, n), dtype=np.int64)
        for k in range(n):
            red[k, k] = 1
        for k in range(n, 2 * n - 1):
            v = 0
            for j in range(n):
                if red[k - 1, j]:
                    v ^= 1 << (j + 1)
            if v >> n & 1:
                v ^= F.poly
            for j in range(n):
                red[k, j] = v >> j & 1
        out = np.zeros((la + lb - 1, n), dtype=np.int64)
        for k in range(2 * n - 1):
            out ^= C[:, k][:, None] * red[k][None, :]
        return (out * (1 << np.arange(n, dtype=np.int64))[None, :]).sum(axis=1)
    p = F.p
    ah, al = a // 256, a % 256
    bh, bl = b // 256, b % 256

    def conv(u, v):
        U = np.zeros(nx); V = np.zeros(nx)
        U[:len(u)] = u; V[:len(v)] = v
        return np.rint(np.fft.irfft(np.fft.rfft(U) * np.fft.rfft(V), n=nx)
                       ).astype(np.int64)[:la + lb - 1]
    hh = conv(ah, bh) % p
    hl = (conv(ah, bl) + conv(al, bh)) % p
    ll = conv(al, bl) % p
    return (hh * 65536 + hl * 256 + ll) % p


# ------------------------------------------------------------- the S-boxes --
def sinv_poly(F, alpha, u, cipher):
    """The decryption S-box in closed form, on four polynomials."""
    a = Poly.const(F, alpha)
    u0, u1, u2, u3 = u
    if cipher == "yux":
        y0 = u0 + u1 * u2 + u3 + a
        y1 = u1 + u2 * u3 + y0 + a
        y2 = u2 + u3 * y0 + y1 + a
        y3 = u3 + y0 * y1 + y2 + a
        return [y0, y1, y2, y3]
    f = u0 * u1 + u3 + a                      # -> position 3
    g = u1 * u2 + u0 + a                      # -> position 0
    y1 = u2 * f + u1 + a
    y2 = f * g + u2 + a
    return [g, y1, y2, f]


def s_poly(F, alpha, u, cipher):
    """The encryption S-box in closed form (the CPA direction's outer map)."""
    a = Poly.const(F, alpha)
    u0, u1, u2, u3 = u
    if cipher == "yux":
        z0 = u3 - u0 * u1 - u2 - a                # position 3
        z1 = u2 - z0 * u0 - u1 - a                # position 2
        z2 = u1 - z1 * z0 - u0 - a                # position 1
        z3 = u0 - z2 * z1 - z0 - a                # position 0
        return [z3, z2, z1, z0]
    aa = u2 - u0 * u3 - a                         # position 2
    b = u1 - u3 * aa - a                          # position 1
    cc = u0 - aa * b - a                          # position 0
    y3 = u3 - b * cc - a                          # position 3
    return [cc, b, aa, y3]


def layer_matrix(c, forward):
    """The 16 x 16 matrix of the cipher's own linear layer, read off by
    applying it to the unit vectors (so F_p coefficients and the
    characteristic-2 XOR sums are both exactly what `dux/` and `yux/` do)."""
    M = [[0] * 16 for _ in range(16)]
    for j in range(16):
        e = [0] * 16
        e[j] = 1
        out = (c.lin.L(tuple(e), 0, False) if forward
               else c.lin.L_inv(tuple(e), 0, False))
        for i in range(16):
            M[i][j] = int(out[i])
    return M


def lin_apply(F, M, Y):
    """The linear layer on 16 polynomials."""
    out = []
    for i in range(16):
        acc = Poly(F, [])
        for j in range(16):
            if M[i][j]:
                acc = acc + Y[j].scal(M[i][j])
        out.append(acc)
    return out


def linv_apply(F, cipher, Y):
    """L^{-1} on 16 polynomials: (L^{-1} Y)_i = sum_{j in ROT_INV} Y_{i+j}."""
    rots = CIPHERS[cipher]["ROT_INV"]
    out = []
    for i in range(16):
        acc = Poly(F, [])
        for j in rots:
            acc = acc + Y[(i + j) % 16]
        out.append(acc)
    return out


def exact_formal_degrees(instance, pos, layers, seed, rounds=None,
                         explain=False, direction="dec"):
    """[per layer][per word] the exact formal degree of the unreduced state.

    `direction` 'dec' follows `decrypt_layers` (chosen ciphertext, the main
    line); 'enc' follows `encrypt_layers` (the designers' CPA model)."""
    c = get_cipher(instance, rounds=rounds) if rounds else get_cipher(instance)
    fam = cipher_family(instance)
    F = c.F
    rng = np.random.default_rng(seed)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    r = c.r if rounds is None else rounds
    sbox = sinv_poly if direction == "dec" else s_poly
    M = layer_matrix(c, forward=(direction == "enc"))

    # the state entering the first S layer
    sub = (lambda v, k: v ^ k) if F.char == 2 else (lambda v, k: (v - k) % F.p)
    add = (lambda v, k: v ^ k) if F.char == 2 else (lambda v, k: (v + k) % F.p)
    mix = sub if direction == "dec" else add
    rk0 = rks[r] if direction == "dec" else rks[0]
    state = []
    for i in range(16):
        if i == pos:
            state.append(Poly(F, [mix(0, rk0[i]), 1]))
        else:
            state.append(Poly.const(F, mix(consts[i], rk0[i])))
    notes = []
    out = []
    for k in range(layers):
        Y = []
        for b in range(4):
            Y += sbox(F, c.alpha, state[4 * b:4 * b + 4], fam)
        out.append([y.degree if y.degree is not None else 0 for y in Y])
        if explain and k == 0:
            U = lin_apply(F, M, Y)
            for b in range(4):
                s13 = U[4 * b + 1] + U[4 * b + 3]
                notes.append({
                    "block": b,
                    "u1_plus_u3_degree": s13.degree if s13.degree is not None
                                         else 0,
                    "input_degrees": [U[4 * b + t].degree
                                      if U[4 * b + t].degree is not None else 0
                                      for t in range(4)],
                    "collapses": (s13.degree or 0) == 0})
        if k + 1 == layers:
            break
        U = lin_apply(F, M, Y)
        i = (r - (k + 1)) if direction == "dec" else (k + 1)
        state = [U[t] - Poly.const(F, rks[i][t]) if direction == "dec"
                 else U[t] + Poly.const(F, rks[i][t]) for t in range(16)]
    return c, out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yu2x-8")
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seeds", default="2026,7,11")
    ap.add_argument("--direction", default="dec", choices=("dec", "enc"),
                    help="dec: chosen ciphertext (main line); enc: the "
                         "designers' chosen-plaintext model")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fam = cipher_family(a.instance)
    seeds = [int(v) for v in a.seeds.split(",")][:a.keys]
    t0 = time.time()
    per_key, notes = [], None
    c = None
    for s in seeds:
        c, degs, nt = exact_formal_degrees(a.instance, a.pos, a.layers, s,
                                           a.rounds, a.explain, a.direction)
        per_key.append(degs)
        notes = notes or nt
        print(f"--- {a.instance} seed {s} (active word {a.pos}) ---")
        for l, row in enumerate(degs, 1):
            print(f"  layer {l:2d}: exact formal degrees {row}")
    bound = profile([a.pos], a.layers, a.direction, c.F.char == 2, fam)
    res = {"instance": a.instance, "cipher": fam, "pos": a.pos,
           "direction": a.direction,
           "layers": a.layers, "seeds": seeds, "q": c.F.q,
           "exact_formal_per_key": per_key,
           "maxplus_bound": [[int(v) for v in row] for row in bound],
           "collapse_notes": notes}
    res["key_independent"] = all(
        len({tuple(per_key[k][l]) for k in range(len(seeds))}) == 1
        for l in range(a.layers))
    res["gap_per_layer"] = [
        [int(bound[l][w]) - int(per_key[0][l][w]) for w in range(16)]
        for l in range(a.layers)]
    print("--- exact formal vs max-plus (gap = bound - exact) ---")
    for l in range(a.layers):
        gap = res["gap_per_layer"][l]
        rel = (100.0 * sum(gap) / max(1, sum(bound[l])))
        print(f"  layer {l + 1:2d}: gap {gap}   ({rel:.1f}% of the bound)")
    print(f"  key-independent across {len(seeds)} keys: "
          f"{res['key_independent']}")
    if a.explain and notes:
        print("--- layer-1 -> layer-2 row collapses (u_1 + u_3 per block) ---")
        for n in notes:
            print(f"  block {n['block']}: inputs {n['input_degrees']}, "
                  f"deg(u_1 + u_3) = {n['u1_plus_u3_degree']}"
                  + ("   <-- CONSTANT: the cubic term of y_1 collapses"
                     if n["collapses"] else ""))
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
