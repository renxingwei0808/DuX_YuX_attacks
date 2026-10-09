"""O15 (R8): interpolation (divided-difference) masks -- Theorem 1 / O7 and
the weighted moments O12 for ARBITRARY point sets, not just the full field,
subspaces or subgroup cosets.

For a set U of n distinct points of F_q the divided-difference functional

    f  |->  f[U] = sum_{u in U} w_u f(u),   w_u = prod_{v in U, v != u} (u - v)^{-1}

vanishes on every polynomial of degree <= n - 2 and returns the leading
coefficient of a polynomial of degree n - 1 (it is the top Newton coefficient
of the interpolant).  Hence, for a structure that is a product set
U_1 x ... x U_s with the mask w(x) = prod_i w_{x_i}, every word whose formal
(max-plus) total degree D satisfies

    D  <  T' := sum_i (|U_i| - 1)

has masked sum 0, and the masked weighted moments  sum_x w(x) x_j^a Z(x) = 0
hold for a < T' - D (O12 with the same accounting).  The pure-key terms vanish
because sum_u w_u u^a = 0 for 0 <= a <= n - 2.  The slice decomposition of O12
is unchanged (slice = fixed value of the weight word; the per-slice weight is
w_u u^a).  This works in both characteristics: it is a degree statement about
polynomials over F_q, and for a subspace of dimension m it reduces to the
usual T = 2^m - 1 = |U| - 1.

Gain: the data can be chosen continuously instead of in powers of q.  With
N_w weights needed and the words' maximal formal degree D at the distinguisher
layer, the minimal balanced choice is |U_i| = ceil((D + N_w)/s) + 1.  Examples
(F_p numbers from the paper's tables):
  12-round DuX(65537): 2 words (3,7), D = 70 226, N_w = 9 000  -> n = 39 614, 2^30.55 (was 2^32)
  11-round DuX:        1 word,  D = 40 545 (max over positions), N_w = 2 152 -> n = 42 698, 2^15.38 (was 2^16)
  11-round YupX:       2 words (0,4), D = 120 847, N_w = 2 400 -> n = 61 625, 2^31.82 (was 2^32)
  10-round YupX:       1 word,  D = 26 083, N_w = 2 155 -> n = 28 239, 2^14.79 (was 2^15 coset)
No gain at the 2^64 cells (margins 0 / 57).  Almost certainly known in some
form (Wang-Tang-Wang IPM masks; Beyne-Verbauwhede's Newton-basis integral
cryptanalysis in characteristic p) -- to be cited as a remark, not a theorem.

W25 (R8) adds: the characteristic-2 weights (log/antilog tables of
`dux.field.BinaryField`, blocked so that n up to 2^16 fits in memory), the
formal degrees taken from `tools/cipher_degree.py`'s `profile()` instead of a
private max-plus copy, and multi-word point sets.  Measured cost of the O(n^2)
weight computation in numpy: 1.8 s at n = 10 000, 33 s at n = 42 698, about a
minute at n = 65 536 -- so no C kernel is needed.

Usage
  python tools/interp_mask.py --plan --D 70226 --nw 9000 --words 2
  python tools/interp_mask.py --check --instance yuxtoy-257 --word 3 --n 100 --layers 4 --keys 2
  python tools/interp_mask.py --check --instance toy-257 --word 3 --n 120 --layers 5 --keys 2
  python tools/interp_mask.py --check --instance yuxtoy-2^4 --word 0 --n 12 --layers 3 --keys 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dux.registry import get_cipher      # noqa: E402


# ----------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------
def divided_difference_weights_prime(p: int, U) -> np.ndarray:
    """w_u = 1 / prod_{v != u} (u - v) over F_p, vectorised in chunks."""
    U = np.asarray(U, dtype=np.int64)
    n = U.size
    assert np.unique(U).size == n, "points must be distinct"
    prod = np.ones(n, dtype=np.int64)
    step = max(1, min(n, 2_000_000 // max(n, 1)))
    for s in range(0, n, step):
        blk = U[s:s + step]                       # (k,)
        diff = (blk[:, None] - U[None, :]) % p    # (k, n)
        diff[np.arange(blk.size), np.arange(s, s + blk.size)] = 1
        # product over the row in a loop of chunks to stay in int64
        acc = np.ones(blk.size, dtype=np.int64)
        for t in range(0, n, 64):
            part = diff[:, t:t + 64]
            r = np.ones(blk.size, dtype=np.int64)
            for c in range(part.shape[1]):
                r = (r * part[:, c]) % p
            acc = (acc * r) % p
        prod[s:s + blk.size] = acc
    return np.array([pow(int(v), p - 2, p) for v in prod], dtype=np.int64)


def divided_difference_weights_2n(F, U) -> np.ndarray:
    """w_u = 1 / prod_{v != u} (u + v) over F_{2^n}, through the field's own
    log/antilog tables.  Differences are XORs, so the product is
    exp(-sum of logs); blocked over rows so that n up to 2^16 fits in memory."""
    U = np.asarray(U, dtype=np.int64)
    n = U.size
    assert np.unique(U).size == n, "points must be distinct"
    order = F.order
    logs = np.zeros(n, dtype=np.int64)
    step = max(1, min(n, 4_000_000 // max(n, 1)))
    for s in range(0, n, step):
        blk = U[s:s + step]
        d = blk[:, None] ^ U[None, :]                 # (k, n); 0 on the diagonal
        d[np.arange(blk.size), np.arange(s, s + blk.size)] = 1   # log 1 = 0
        logs[s:s + blk.size] = F._log[d].sum(axis=1) % order
    return F._exp[(order - logs) % order].astype(np.int64)


def divided_difference_weights(F, U) -> np.ndarray:
    """w_u = 1 / prod_{v != u} (u - v), dispatched on the characteristic."""
    if F.char == 2:
        return divided_difference_weights_2n(F, U)
    return divided_difference_weights_prime(F.q, U)


def threshold(sizes) -> int:
    """T' = sum_i (|U_i| - 1)."""
    return sum(n - 1 for n in sizes)


def usable_weights(sizes, D: int) -> int:
    """Number of admissible moment exponents a (a = 0 .. T' - D - 1), i.e. the
    O12 margin for the product set; <= 0 means the cell is not balanced."""
    return threshold(sizes) - D


def min_points(D: int, nw: int, words: int = 1):
    """Balanced product set of `words` factors with usable_weights >= nw."""
    n = math.ceil((D + nw) / words) + 1
    return n, words * math.log2(n)


# ----------------------------------------------------------------------------
# formal degrees: the repository's own tool, not a private copy
# ----------------------------------------------------------------------------
def maxplus_degrees(cipher, family, active, layers, direction="dec"):
    """Formal degrees of the 16 words after `layers` layers, for a structure on
    the words in `active` (all others constant).

    W25: this now calls `tools/cipher_degree.py`'s `profile()` -- the
    repository's reference max-plus -- instead of the private copy the first
    version carried.  For YuX the two agreed word for word (checked on
    yuxtoy-257 at layers 4 and 5); the private DuX branch could not run at all,
    because it imported a `sinv_degree_map` that does not exist in
    `tools/maxplus.py`.  `cipher_degree` is the reference either way."""
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    from cipher_degree import profile            # noqa: E402
    return profile(sorted(active), layers, direction, cipher.F.char == 2, family)


def family_of(instance):
    from dux.registry import cipher_family
    return cipher_family(instance)


# ----------------------------------------------------------------------------
# check on a toy instance: masked sums / masked moments vs the criterion
# ----------------------------------------------------------------------------
def random_point_set(rng, q, n):
    return np.sort(rng.choice(q, size=n, replace=False)).astype(np.int64)


def check(instance, words, n, layers, keys, seed=1, max_a=None, direction="dec"):
    """For each key: build the product point set on `words` (n points per
    word), mask with the divided-difference weights, and find for every state
    word the smallest a with sum_x w(x) x_{words[0]}^a Z_i(x) != 0.

    The criterion predicts (T' - D)^+ zero moments (a = 0 .. T' - D - 1).
    Over F_p the max-plus bound is an equality cell by cell, so observed should
    equal predicted; in characteristic 2 the true degree can be lower (O13), so
    the contract is observed >= predicted."""
    cipher = get_cipher(instance)
    F = cipher.F
    q = F.q
    if isinstance(words, int):
        words = [words]
    words = list(words)
    s = len(words)
    assert n <= q, f"{n} points requested from a field of size {q}"
    family = family_of(instance)
    D = maxplus_degrees(cipher, family, set(words), layers, direction)[layers - 1]
    Tp = s * (n - 1)
    pred = [max(0, Tp - D[i]) for i in range(16)]
    rng = np.random.default_rng(seed)
    results = []
    for k in range(keys):
        K = cipher.random_key(rng)
        rks = cipher.key_schedule(K)
        Us = [random_point_set(rng, q, n) for _ in range(s)]
        ws = [divided_difference_weights(F, U) for U in Us]
        # the product set, axis 0 slowest (the order structure_data uses)
        grid = np.meshgrid(*Us, indexing="ij")
        wgrid = np.meshgrid(*ws, indexing="ij")
        npts = n ** s
        C = [np.full(npts, int(v), dtype=F.dtype) for v in rng.integers(0, q, size=16)]
        for j, wd in enumerate(words):
            C[wd] = grid[j].ravel().astype(F.dtype)
        mask = wgrid[0].ravel().astype(np.int64)
        for j in range(1, s):
            mask = _mul(F, mask, wgrid[j].ravel().astype(np.int64))
        step = cipher.decrypt_layers if direction == "dec" else cipher.encrypt_layers
        Z = step(tuple(C), rks, layers, rounds=cipher.r, vec=True)
        Z = [np.asarray(z, dtype=np.int64) for z in Z]
        xs = grid[0].ravel().astype(np.int64)
        amax = (s * (n - 1)) if max_a is None else max_a
        obs = []
        for i in range(16):
            ua = np.ones(npts, dtype=np.int64) if F.char != 2 \
                else np.ones(npts, dtype=np.int64)
            first = None
            for a in range(0, amax + 1):
                t = _mul(F, mask, ua)
                acc = _mul(F, t, Z[i])
                val = int(np.bitwise_xor.reduce(acc)) if F.char == 2 else int(acc.sum() % q)
                if val != 0:
                    first = a
                    break
                ua = _mul(F, ua, xs)
            obs.append(first)
        ok = all(o is None or o >= t for o, t in zip(obs, pred))
        results.append({"key": k, "n": n, "observed_first_nonzero_a": obs,
                        "predicted_zero_moments": pred, "criterion_respected": ok,
                        "tight": obs == [None if t == 0 and o is None else o
                                         for o, t in zip(obs, pred)]})
        print(f"{instance} words {words} n={n} layers={layers} key {k}: T'={Tp}")
        print(f"   D                               : {D}")
        print(f"   predicted #zero moments (T'-D)+ : {pred}")
        print(f"   observed first non-zero a       : {obs}   criterion respected: {ok}")
    return {"instance": instance, "words": words, "n": n, "layers": layers,
            "direction": direction, "threshold": Tp, "D": D,
            "predicted_zero_moments": pred, "results": results,
            "all_respected": all(r["criterion_respected"] for r in results),
            "exact": all(r["observed_first_nonzero_a"] == pred for r in results)}


def _mul(F, a, b):
    """Vectorised field product for either characteristic."""
    if F.char == 2:
        return F.vmul(a.astype(F.dtype), b.astype(F.dtype)).astype(np.int64)
    return (a * b) % F.q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--D", type=int, help="max formal degree of the rows used")
    ap.add_argument("--nw", type=int, help="weights needed")
    ap.add_argument("--words", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--instance", default="yuxtoy-257")
    ap.add_argument("--word", default="3",
                    help="active word, or a comma-separated list for a product set")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--direction", choices=("dec", "enc"), default="dec")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = None
    if a.plan:
        n, lg = min_points(a.D, a.nw, a.words)
        out = {"D": a.D, "nw": a.nw, "words": a.words, "n_per_word": n, "log2_data": round(lg, 3),
               "usable_weights": usable_weights([n] * a.words, a.D)}
        print(json.dumps(out))
    if a.check:
        words = [int(v) for v in str(a.word).split(",")]
        out = check(a.instance, words, a.n, a.layers, a.keys, a.seed,
                    direction=a.direction)
    if a.json and out is not None:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
