"""O14 (R8): universal top-degree forms of the YuX full-block structure over F_p
and the key-independent constant sums at the exact-boundary cells D = T.

Setting: decryption direction, block 0 fully active (O11 full-block structure),
prime field F_p.  After the Lemma-2 substitution y = S^{-1}(x + k) the four
layer-1 outputs of block 0 are the free variables y0..y3 and the other twelve
words are constants.  From layer 2 on, the top-degree part of every state word
is a product of the top-degree parts of its S^{-1} inputs -- the max-plus
maximiser is unique except for y1, where both maximal terms are constant-free
-- and the layer-2 inputs' top parts are the 0/1 linear forms of L^{-1} in y.
Hence the top form P_i^{(l)}(y) of every word at every layer is a fixed
polynomial over F_p that depends on nothing but p: not on the round keys, not
on alpha, not on the twelve inactive constants.

At a cell where the position-3 formal degree 4^{l-1} equals T = 4(p - 1)
(p - 1 = 4^{l-2}: p = 5 / l = 3, 17 / 4, 257 / 6, 65537 / 10) Theorem 1 kills
every monomial of degree < T, and among the degree-T monomials only
(y0 y1 y2 y3)^{p-1} has a non-zero sum over F_p^4 (its sum is (-1)^4 = 1).  So

    sum_{x} Z_{4b+3}(x) = coeff of (y0 y1 y2 y3)^{p-1} in P_{4b+3}^{(l)}
                        = sum_{y in F_p^4} P_{4b+3}^{(l)}(y) =: c_b(p),

a KEY-INDEPENDENT constant, in general non-zero.  By homogeneity of degree T
(a multiple of p - 1) the sum over F_p^4 minus the origin equals -(sum over the
p^3 + p^2 + p + 1 projective points), which is how the tool evaluates it.

W24 (R8) adds the ENCRYPTION (chosen-plaintext) direction and DuX.  There the
S-box top forms have no ties at all: for YuX, S = Pf^{-4} = (z3, z2, z1, z0)
with z0 = -x0 x1, z1 = -z0 x0, z2 = -z1 z0, z3 = -z2 z1 on top, so position 0
carries degree 8^{l-1}; for DuX, S = (R^{-1})^4 = (c, b, a, y3) with
a = -u0 u3, b = u0 u3^2, c = u0^2 u3^3, y3 = -u0^3 u3^5 on top, so position 3
carries degree 8^{l-1}.  The exact-boundary condition is then 8^{l-1} = 4(p-1),
whose only prime solutions are

    p = 3 / layer 2,   p = 17 / layer 3,   p = 65537 / layer 7.

DuX's two circulants are SINGULAR mod 17 (and mod 5 and 13), so the DuX half of
the encryption check can only be run against the real cipher at p = 3; YuX's
circulant is singular mod 3, so the YuX half can only be run at p = 17.  Both
are covered by experiments/Y09_topform_constants/run.py.  The DuX encryption
direction uses O2's fixed L0 (`--cipher dux --direction enc` reads L(., t=0)).

It does NOT apply to single-word structures (first-layer leading coefficients
contain key-dependent constants) nor in characteristic 2 (D is a power of 4,
T = 4(2^n - 1), never equal).

Usage
  python tools/topform_constants.py --p 5   --layers 3 --check 3   # recursion + direct cipher sums
  python tools/topform_constants.py --p 17  --layers 4 --check 3
  python tools/topform_constants.py --p 257 --layers 6              # projective sum only (seconds)
  python tools/topform_constants.py --p 257 --export-linv linv_257.txt   # matrix for tools/topform_kernels.c
  python tools/topform_constants.py --p 17 --direction enc --cipher yux --affine
  python tools/topform_constants.py --p 3  --direction enc --cipher dux --affine
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dux.field import PrimeField            # noqa: E402
from yux.params import ROT_INV, WORDS       # noqa: E402
from yux.sbox import vS_inv                 # noqa: E402

try:                                        # the repository's own rotation-sum (authoritative)
    from yux.linear import rot_sum as _rot_sum
except Exception:                           # pragma: no cover
    _rot_sum = None


# ----------------------------------------------------------------------------
# L^{-1} as a 0/1 matrix, taken from the repository's implementation
# ----------------------------------------------------------------------------
def linv_matrix(p: int) -> np.ndarray:
    """M with (L^{-1} x)_i = sum_j M[i][j] x_j, read off the repo's `rot_sum`
    on unit vectors (so the rotation direction can never be wrong here)."""
    F = PrimeField(p)
    M = np.zeros((WORDS, WORDS), dtype=np.int64)
    for j in range(WORDS):
        e = [0] * WORDS
        e[j] = 1
        if _rot_sum is not None:
            out = _rot_sum(F, tuple(e), ROT_INV, False)
        else:                                # fallback: x <<< j  ==  out[i] = x[(i + j) mod 16]
            out = tuple(sum(e[(i + r) % WORDS] for r in ROT_INV) % p for i in range(WORDS))
        for i in range(WORDS):
            M[i, j] = int(out[i]) % p
    return M


# ----------------------------------------------------------------------------
# top forms with degree tracking (max-plus with ties summed)
# ----------------------------------------------------------------------------
def _top_sum(terms, p):
    """terms: list of (array_or_scalar, degree).  Keep the terms of maximal
    degree, add them (mod p).  Returns (array, degree)."""
    dmax = max(d for _, d in terms)
    acc = None
    for v, d in terms:
        if d == dmax:
            acc = v if acc is None else (acc + v) % p
    return acc % p, dmax


def sinv_top(u, du, p):
    """Top forms of the four S^{-1} = Pf^4 outputs (YuX, F_p) given the top
    forms u = (u0..u3) and degrees du of the inputs.  alpha is degree 0 and
    never reaches the top."""
    u0, u1, u2, u3 = u
    d0, d1, d2, d3 = du
    y0, e0 = _top_sum([(u0, d0), (u1 * u2 % p, d1 + d2), (u3, d3)], p)
    y1, e1 = _top_sum([(u1, d1), (u2 * u3 % p, d2 + d3), (y0, e0)], p)
    y2, e2 = _top_sum([(u2, d2), (u3 * y0 % p, d3 + e0), (y1, e1)], p)
    y3, e3 = _top_sum([(u3, d3), (y0 * y1 % p, e0 + e1), (y2, e2)], p)
    return (y0, y1, y2, y3), (e0, e1, e2, e3)


def topforms(y, p, layers, M=None):
    """Top forms of all 16 words after `layers` layers, evaluated at the points
    y = (y0, y1, y2, y3) (numpy int64 arrays of equal shape).  Returns
    (list of 16 arrays, list of 16 formal degrees).  Layer 1 = (y0..y3, const*12)."""
    if M is None:
        M = linv_matrix(p)
    zeros = np.zeros_like(y[0])
    state = list(y) + [zeros] * (WORDS - 4)
    deg = [1] * 4 + [0] * (WORDS - 4)
    for _ in range(2, layers + 1):
        # linear layer: keep, per row, the words of maximal degree
        u, du = [], []
        for i in range(WORDS):
            terms = [((M[i, j] * state[j]) % p, deg[j]) for j in range(WORDS) if M[i, j]]
            v, d = _top_sum(terms, p)
            u.append(v)
            du.append(d)
        state, deg = [], []
        for b in range(4):
            ys, es = sinv_top(tuple(u[4 * b:4 * b + 4]), tuple(du[4 * b:4 * b + 4]), p)
            state += list(ys)
            deg += list(es)
    return state, deg


# ----------------------------------------------------------------------------
# W24: the other three (cipher, direction) combinations
#
# `sinv_top` / `topforms` above are the reference implementation for
# (yux, dec) and are deliberately left untouched; `topforms_general` below
# reproduces them exactly for that pair (tests/test_topform_constants.py).
# ----------------------------------------------------------------------------
def sinv_top_dux(u, du, p):
    """DuX decryption S^{-1} = R^4, degrees (2,3,4,2), closed form of
    dux/sbox.py: f = x0 x1 + x3 + a, g = x1 x2 + x0 + a,
    y = (g, x2 f + x1 + a, f g + x2 + a, f)."""
    x0, x1, x2, x3 = u
    d0, d1, d2, d3 = du
    f, ef = _top_sum([(x0 * x1 % p, d0 + d1), (x3, d3)], p)
    g, eg = _top_sum([(x1 * x2 % p, d1 + d2), (x0, d0)], p)
    y1, e1 = _top_sum([(x2 * f % p, d2 + ef), (x1, d1)], p)
    y2, e2 = _top_sum([(f * g % p, ef + eg), (x2, d2)], p)
    return (g, y1, y2, f), (eg, e1, e2, ef)


def s_top_yux(u, du, p):
    """YuX encryption S = Pf^{-4}, output (z3, z2, z1, z0), degrees (8,5,3,2):
    z0 = x3 - x0 x1 - x2 - a, z1 = x2 - z0 x0 - x1 - a,
    z2 = x1 - z1 z0 - x0 - a, z3 = x0 - z2 z1 - z0 - a."""
    x0, x1, x2, x3 = u
    d0, d1, d2, d3 = du
    z0, e0 = _top_sum([(x3, d3), (-x0 * x1 % p, d0 + d1), (-x2 % p, d2)], p)
    z1, e1 = _top_sum([(x2, d2), (-z0 * x0 % p, e0 + d0), (-x1 % p, d1)], p)
    z2, e2 = _top_sum([(x1, d1), (-z1 * z0 % p, e1 + e0), (-x0 % p, d0)], p)
    z3, e3 = _top_sum([(x0, d0), (-z2 * z1 % p, e2 + e1), (-z0 % p, e0)], p)
    return (z3, z2, z1, z0), (e3, e2, e1, e0)


def s_top_dux(u, du, p):
    """DuX encryption S = (R^{-1})^4, output (c, b, a, y3), degrees (5,3,2,8):
    a = x2 - x0 x3 - al, b = x1 - x3 a - al, c = x0 - a b - al,
    y3 = x3 - b c - al."""
    x0, x1, x2, x3 = u
    d0, d1, d2, d3 = du
    a, ea = _top_sum([(x2, d2), (-x0 * x3 % p, d0 + d3)], p)
    b, eb = _top_sum([(x1, d1), (-x3 * a % p, d3 + ea)], p)
    c, ec = _top_sum([(x0, d0), (-a * b % p, ea + eb)], p)
    y3, e3 = _top_sum([(x3, d3), (-b * c % p, eb + ec)], p)
    return (c, b, a, y3), (ec, eb, ea, e3)


SBOX_TOP = {("yux", "dec"): sinv_top, ("yux", "enc"): s_top_yux,
            ("dux", "dec"): sinv_top_dux, ("dux", "enc"): s_top_dux}

# The position that carries the maximal degree of each round function, and the
# per-layer growth rate of that position (used by `exact_hit_layer`).
TOP_POSITION = {("yux", "dec"): 3, ("yux", "enc"): 0,
                ("dux", "dec"): 2, ("dux", "enc"): 3}
GROWTH = {("yux", "dec"): 4, ("yux", "enc"): 8, ("dux", "enc"): 8}


def layer_matrix(p: int, cipher: str = "yux", direction: str = "dec") -> np.ndarray:
    """The 16x16 matrix of the layer's linear map, read off the repository's
    OWN implementation on unit vectors, so neither the rotation direction nor
    the F_p forward circulant can be mis-transcribed here.

    decryption -> L^{-1};  encryption -> L, and for DuX with t = 0, i.e. the
    fixed-L0 cipher of observation O2 (there is no key-dependent layer in YuX,
    whose `L` ignores t)."""
    from dux.registry import get_cipher       # noqa: E402  (lazy: keeps the import graph flat)
    inst = {("yux", 3): None, ("yux", 5): "yuxtoy-5", ("yux", 17): "yuxtoy-17",
            ("yux", 193): "yuxtoy-193", ("yux", 257): "yuxtoy-257",
            ("yux", 65537): "yupx-65537",
            ("dux", 3): "toy-3", ("dux", 193): "toy-193", ("dux", 257): "toy-257",
            ("dux", 65537): "dux-65537"}.get((cipher, p))
    if inst is None:
        raise SystemExit(f"no {cipher} instance over F_{p} (see dux/params.py, yux/params.py)")
    c = get_cipher(inst)
    F = c.F
    M = np.zeros((WORDS, WORDS), dtype=np.int64)
    for j in range(WORDS):
        e = [0] * WORDS
        e[j] = 1
        out = c.lin.L_inv(tuple(e), 0, False) if direction == "dec" else c.lin.L(tuple(e), 0, False)
        for i in range(WORDS):
            M[i, j] = int(out[i]) % F.q
    return M


def topforms_general(y, p, layers, M, top_fn):
    """`topforms` with the S-box top-form rule as a parameter."""
    zeros = np.zeros_like(y[0])
    state = list(y) + [zeros] * (WORDS - 4)
    deg = [1] * 4 + [0] * (WORDS - 4)
    for _ in range(2, layers + 1):
        u, du = [], []
        for i in range(WORDS):
            terms = [((M[i, j] * state[j]) % p, deg[j]) for j in range(WORDS) if M[i, j]]
            v, d = _top_sum(terms, p)
            u.append(v)
            du.append(d)
        state, deg = [], []
        for b in range(4):
            ys, es = top_fn(tuple(u[4 * b:4 * b + 4]), tuple(du[4 * b:4 * b + 4]), p)
            state += list(ys)
            deg += list(es)
    return state, deg


def projective_points(p, chunk=1 << 22):
    """Representatives of P^3(F_p): (1,a,b,c), (0,1,b,c), (0,0,1,c), (0,0,0,1)."""
    rng = np.arange(p, dtype=np.int64)
    # (1, a, b, c): p^3 points, streamed by a
    for a in range(p):
        B, C = np.meshgrid(rng, rng, indexing="ij")
        n = B.size
        yield (np.ones(n, dtype=np.int64), np.full(n, a, dtype=np.int64), B.ravel(), C.ravel())
    B, C = np.meshgrid(rng, rng, indexing="ij")
    n = B.size
    yield (np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), B.ravel(), C.ravel())
    yield (np.zeros(p, dtype=np.int64), np.zeros(p, dtype=np.int64), np.ones(p, dtype=np.int64), rng)
    yield (np.zeros(1, dtype=np.int64),) * 3 + (np.ones(1, dtype=np.int64),)


def _forms(y, p, layers, M, top_fn):
    """Dispatch: the untouched reference `topforms` for (yux, dec), the
    parametrised `topforms_general` otherwise."""
    if top_fn is None or top_fn is sinv_top:
        return topforms(y, p, layers, M)
    return topforms_general(y, p, layers, M, top_fn)


def constants_projective(p, layers, M=None, top_fn=None):
    """c_i(p) = sum_{y in F_p^4} P_i(y) for all 16 words, via the projective
    sum: sum over the non-zero points of a degree-T form with (p-1) | T is
    -(sum over P^3).  Only meaningful for words whose formal degree is a
    multiple of p - 1 (the exact-hit words); the others are reported too."""
    if M is None:
        M = linv_matrix(p)
    acc = np.zeros(WORDS, dtype=np.int64)
    deg = None
    for y in projective_points(p):
        st, deg = _forms(y, p, layers, M, top_fn)
        for i in range(WORDS):
            acc[i] = (acc[i] + int(st[i].sum() % p)) % p
    return [int((-acc[i]) % p) for i in range(WORDS)], deg


def constants_affine(p, layers, M=None, top_fn=None):
    """Same, by brute force over F_p^4 (p <= ~40)."""
    if M is None:
        M = linv_matrix(p)
    grid = np.array(list(itertools.product(range(p), repeat=4)), dtype=np.int64)
    y = tuple(grid[:, i] for i in range(4))
    st, deg = _forms(y, p, layers, M, top_fn)
    return [int(v.sum() % p) for v in st], deg


# ----------------------------------------------------------------------------
# direct sums of the real cipher components (arbitrary round keys)
# ----------------------------------------------------------------------------
def direct_sums(p, layers, seed, alpha=205):
    """sum over x in F_p^4 (block 0, other words = random constants) of all 16
    words after `layers` S^{-1} layers, with independent random round keys.
    The property must hold for arbitrary round keys, so no key schedule is
    used; the repository's `decrypt_layers` on a real key is the same map."""
    F = PrimeField(p)
    alpha %= p
    rng = np.random.default_rng(seed)
    rk = rng.integers(0, p, size=(layers, WORDS), dtype=np.int64)
    cst = rng.integers(0, p, size=WORDS, dtype=np.int64)
    grid = np.array(list(itertools.product(range(p), repeat=4)), dtype=np.int64)
    n = grid.shape[0]
    x = [np.full(n, int(cst[i]), dtype=np.int64) for i in range(WORDS)]
    for i in range(4):
        x[i] = grid[:, i].copy()
    x = tuple((x[i] - rk[0, i]) % p for i in range(WORDS))
    x = _sl_inv(F, x, alpha)
    for l in range(1, layers):
        x = tuple((x[i] - rk[l, i]) % p for i in range(WORDS))
        x = _linv_vec(F, x)
        x = _sl_inv(F, x, alpha)
    return [int(v.sum() % p) for v in x]


def _sl_inv(F, x, alpha):
    out = ()
    for b in range(4):
        out += tuple(vS_inv(F, tuple(x[4 * b:4 * b + 4]), alpha))
    return out


def _linv_vec(F, x):
    if _rot_sum is not None:
        return tuple(_rot_sum(F, x, ROT_INV, True))
    p = F.p
    return tuple(sum(x[(i + r) % WORDS] for r in ROT_INV) % p for i in range(WORDS))


def instance_for(p, cipher):
    """The repository instance name of `cipher` over F_p, or None."""
    return {("yux", 5): "yuxtoy-5", ("yux", 17): "yuxtoy-17",
            ("yux", 193): "yuxtoy-193", ("yux", 257): "yuxtoy-257",
            ("yux", 65537): "yupx-65537",
            ("dux", 3): "toy-3", ("dux", 193): "toy-193", ("dux", 257): "toy-257",
            ("dux", 65537): "dux-65537"}.get((cipher, p))


def full_block_sums(cipher_obj, rks, layers, direction, constants, block=0):
    """Sum over the p^4 points of the full-block structure on `block` of all 16
    state words after `layers` S layers, using the cipher's own layered maps.

    decryption: decrypt_layers on ciphertexts whose `block` ranges over F_p^4;
    encryption: the layered fixed-L0 map of O2 for DuX (lin.L(., t=0)), which
    for YuX is encrypt_layers itself (its L ignores t)."""
    F = cipher_obj.F
    p = F.q
    grid = np.array(list(itertools.product(range(p), repeat=4)), dtype=np.int64)
    n = grid.shape[0]
    x = [np.full(n, int(v) % p, dtype=np.int64) for v in constants]
    for i in range(4):
        x[4 * block + i] = grid[:, i].copy()
    x = tuple(x)
    if direction == "dec":
        out = cipher_obj.decrypt_layers(x, rks, layers, rounds=cipher_obj.r, vec=True)
    else:
        out = cipher_obj.ARK(x, rks[0], True)
        out = cipher_obj.SL(out, True)
        for k in range(1, layers):
            out = cipher_obj.lin.L(out, 0, True)
            out = cipher_obj.ARK(out, rks[k], True)
            out = cipher_obj.SL(out, True)
    return [int(np.asarray(v, dtype=np.int64).sum() % p) for v in out]


def direct_sums_general(p, layers, seed, cipher="yux", direction="dec",
                        constant_sets=1, real_key_schedule=True):
    """Full-block sums of the REAL cipher over F_p, one entry per (master key,
    inactive-constant set).  With `real_key_schedule` the round keys come from
    `key_schedule` on a random master key (W24 step 2's requirement); the
    property itself holds for arbitrary round keys, which `direct_sums` above
    exercises separately.  For DuX in the encryption direction the round keys
    are first replaced by O2's equivalent fixed-L0 keys."""
    from dux.registry import get_cipher       # noqa: E402
    inst = instance_for(p, cipher)
    if inst is None:
        raise SystemExit(f"no {cipher} instance over F_{p}")
    c = get_cipher(inst)
    rng = np.random.default_rng(seed)
    K = c.random_key(rng)
    rks = c.key_schedule(K)
    if cipher == "dux" and direction == "enc":
        rks, _T = c.equivalent_fixedL0_keys(rks)
    rows = []
    for j in range(constant_sets):
        cst = [int(v) for v in rng.integers(0, p, size=WORDS)]
        rows.append({"seed": seed, "constant_set": j, "master_key": list(K),
                     "sums": full_block_sums(c, rks, layers, direction, cst)})
    return rows


# ----------------------------------------------------------------------------
def exact_hit_layer(p, direction="dec", cipher="yux"):
    """The layer l with g^{l-1} = 4(p-1) for the growth rate g of the leading
    position (g = 4 for YuX decryption, g = 8 for either cipher's encryption),
    or None.  DuX decryption grows by 2 + sqrt(3) per layer and never hits."""
    g = GROWTH.get((cipher, direction))
    if g is None:
        return None
    l, d = 1, 1
    while d < 4 * (p - 1):
        d *= g
        l += 1
    return l if d == 4 * (p - 1) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--layers", type=int, default=None, help="default: the exact-hit layer for p")
    ap.add_argument("--check", type=int, default=0, help="number of random key sets for direct sums (p <= 40)")
    ap.add_argument("--affine", action="store_true", help="brute-force sum of the top forms over F_p^4 (p <= 40)")
    ap.add_argument("--export-linv", default=None, help="write the 16x16 linear matrix (for topform_kernels.c)")
    ap.add_argument("--direction", choices=("dec", "enc"), default="dec")
    ap.add_argument("--cipher", choices=("yux", "dux"), default="yux")
    ap.add_argument("--real-keys", type=int, default=0,
                    help="number of REAL master keys (key_schedule) for the direct sums")
    ap.add_argument("--constant-sets", type=int, default=2,
                    help="inactive-constant sets per master key (with --real-keys)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    p = a.p
    key = (a.cipher, a.direction)
    if key not in SBOX_TOP:
        ap.error(f"unknown combination {key}")
    layers = a.layers or exact_hit_layer(p, a.direction, a.cipher)
    if layers is None:
        ap.error(f"{a.cipher}/{a.direction} over F_{p} has no exact-hit layer; give --layers")
    M = linv_matrix(p) if key == ("yux", "dec") else layer_matrix(p, a.cipher, a.direction)
    top_fn = SBOX_TOP[key]
    if a.export_linv:
        with open(a.export_linv, "w") as fh:
            for i in range(WORDS):
                fh.write(" ".join(str(int(v)) for v in M[i]) + "\n")
        print(f"wrote {a.export_linv}")
    out = {"p": p, "layers": layers, "T": 4 * (p - 1),
           "cipher": a.cipher, "direction": a.direction,
           "top_position": TOP_POSITION[key]}
    c, deg = constants_projective(p, layers, M, top_fn)
    T = 4 * (p - 1)
    # the projective value is the sum only where the formal degree is a
    # multiple of p-1 and >= T; below T the sum is 0 by Theorem 1
    c = [0 if deg[i] < T else (c[i] if deg[i] % (p - 1) == 0 else None) for i in range(WORDS)]
    out["formal_degree"] = deg
    out["constants_projective"] = c
    print(f"p={p} layers={layers} T={T} formal degrees {deg}")
    print("predicted sums (16 words; None = D > T, not predicted):", c)
    if a.affine:
        ca, _ = constants_affine(p, layers, M, top_fn)
        ca = [0 if deg[i] < T else (ca[i] if deg[i] % (p - 1) == 0 else None) for i in range(WORDS)]
        out["constants_affine"] = ca
        print("affine-sum constants            :", ca, "  match:", ca == c)
    if a.check:
        if key != ("yux", "dec"):
            ap.error("--check (arbitrary round keys) is the YuX decryption routine; "
                     "use --real-keys for the other combinations")
        out["direct"] = []
        for s in range(1, a.check + 1):
            d = direct_sums(p, layers, seed=s)
            out["direct"].append({"seed": s, "sums": d})
            print(f"direct cipher sums, seed {s}     :", d, "  == projective:", d == c)
    if a.real_keys:
        out["direct_real_keys"] = []
        for s in range(1, a.real_keys + 1):
            for row in direct_sums_general(p, layers, seed=s, cipher=a.cipher,
                                           direction=a.direction,
                                           constant_sets=a.constant_sets):
                row["matches"] = row["sums"] == c
                out["direct_real_keys"].append(row)
                print(f"real key {s} const set {row['constant_set']}: {row['sums']}"
                      f"   == recursion: {row['matches']}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    main()
