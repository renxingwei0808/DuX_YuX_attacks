"""E06 -- r_KR = 1 key recovery on (l+1)-round DuX from an l-layer zero-sum.

Setting (chosen ciphertext).  Let the cipher have r = l + 1 rounds.  The
attacker picks a structure of ciphertexts with one active word (all q
values, position `pos`), obtains the plaintexts P, and knows (E03/E04)
that after l S^{-1} layers the state Z' sums to 0 on all 16 words.

Because the remaining decryption steps are ARK(rk^1), L^{-1}, S^{-1},
ARK(rk^0), the state W = SL(P + rk^0) satisfies
    sum_P W = L^{-1}( sum Z' - |P| rk^1 ) = 0        (|P| = q = 0 in F_q)
and W depends only on P and the whitening key rk^0 = K.  So for every
S-box block j we get 4 equations in the 4 key words (k0,k1,k2,k3) of that
block.  With S = (c, b, a, x3 - b c - alpha) (see dux/sbox.py):

  coordinate 2 (a = x2 - x0 x3 - alpha):
      sum a(P+k) = -k0*sum P3 - k3*sum P0 + (sum P2 - sum P0 P3)   -> affine in (k0,k3)
      (the k0 k3 and k2 terms are multiplied by |P| = 0)
  coordinate 1 (b = x1 - x3 a - alpha): given k0,k3, affine in k2
  coordinate 3 (y3 = x3 - b c - alpha): given k0,k2,k3, affine in k1
      (the k1^2 term has coefficient sum a = 0 by the coordinate-2 equation)
  coordinate 0 (c): consistency check only (k1 drops out)

We never expand the polynomials by hand: an affine function is fully
determined by evaluating the (vectorised) sum at 2 (or 3) points, so the
whole attack is a handful of batched S-box evaluations plus 2x2 linear
algebra over F_q.  Two structures are enough (two independent equations for
(k0,k3)); a third is used to double-check.

Complexity: data 2 * q chosen ciphertexts, time ~ 3 structures of q
S-layer evaluations.  Recovering rk^0 = K gives the master key directly.

The structure is parameterised (`--active` word list, `--dim` F_2-dimension
per word), so as soon as a longer distinguisher is available -- e.g. a 10-layer
all-16-word zero-sum on DuX(2^16) from the large E04 runs -- the same script mounts the
11-round attack with `--rounds 11 --active <words> --dim <m>`.

`--keys N` repeats the whole attack on N random master keys and reports the
success rate, the number of structures actually needed and the failure reasons
(singular 2x2 system, or a vanishing coefficient in one of the affine steps);
on failure it automatically retries with one more structure.

Usage:
  python attack_1round.py --instance dux-65537 --rounds 10 --pos 3
  python attack_1round.py --instance dux-2^16  --rounds 10 --pos 3
  python attack_1round.py --instance dux-2^8   --rounds 6  --pos 3
  python attack_1round.py --instance dux-2^8   --rounds 6  --pos 3 --keys 50
  python attack_1round.py --instance dux-2^16  --rounds 11 --active 3,7 --dim 16   # once S1 delivers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dux import DuX  # noqa: E402
from dux.registry import cipher_family, get_cipher  # noqa: E402
from dux.sbox import vS  # noqa: E402
from yux.sbox import vS as vS_yux  # noqa: E402
from yux import YuX  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "E04_zero_sum_F2n"))
from run import random_basis, span_values  # noqa: E402


def structure_plaintexts(c, rks, rounds, active, rng, dim=None):
    """Chosen-ciphertext structure: every word in `active` runs over an
    F_2-affine subspace of dimension `dim` (F_{2^n}; default the whole field)
    or, over a prime field, one word runs over all of F_p.

    Returns (plaintexts as 16 arrays, the values of the FIRST active
    ciphertext word at every point).  The latter is what O12's weights ride
    on: the equations become sum_x x^a S_c(P(x) + k) = 0 (see `--weight`)."""
    F = c.F
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    if F.char != 2:
        assert len(active) == 1, "prime fields: one active word per structure"
        xs = np.arange(F.q, dtype=np.int64)
        C = tuple(np.full(F.q, consts[i], dtype=np.int64) if i != active[0]
                  else xs for i in range(16))
        return c.decrypt(C, rks, rounds=rounds, vec=True), xs
    per = F.n if dim is None else dim
    vals = [span_values(random_basis(F.n, per, rng), int(rng.integers(0, F.q)))
            for _ in active]
    N = 1 << (per * len(active))
    idx = np.arange(N, dtype=np.int64)
    C = [np.full(N, consts[i], dtype=np.int64) for i in range(16)]
    for t, w in enumerate(active):
        C[w] = vals[t][(idx >> (per * t)) & ((1 << per) - 1)]
    return (c.decrypt(tuple(C), rks, rounds=rounds, vec=True), C[active[0]])


def weight_col(F, xs, weight):
    """x^weight over the structure's points (None for the trivial weight)."""
    if not weight:
        return None
    r = np.ones_like(xs)
    b = np.asarray(xs, dtype=np.int64).copy()
    e = weight
    while e:
        if e & 1:
            r = F.vmul(r, b) if F.char == 2 else (r * b) % F.p
        b = F.vmul(b, b) if F.char == 2 else (b * b) % F.p
        e >>= 1
    return r


def coord_sum(c, Pblk, k, coord, w=None):
    """sum (optionally weighted by w) over the structure of coordinate `coord`
    of S(P_blk + k)."""
    F = c.F
    x = tuple(F.vadd(Pblk[i], k[i]) for i in range(4))
    y = (vS_yux if isinstance(c, YuX) else vS)(F, x, c.alpha)
    return F.vsum(y[coord] if w is None else F.vmul(y[coord], w))


def affine_coeffs_1(fun, F):
    """fun(t) affine in t: return (a0, a1) with fun(t) = a0 + a1 t."""
    f0, f1 = fun(0), fun(1)
    return f0, F.sub(f1, f0)


def _independent_rows(F, cand, nvars):
    """Pick `nvars` rows of `cand` = [(coeffs, rhs)] that are independent."""
    chosen, basis = [], []
    for row, rhs in cand:
        v = list(row)
        for b, piv in basis:
            if v[piv]:
                f = F.mul(v[piv], F.inv(b[piv]))
                v = [F.sub(x, F.mul(f, y)) for x, y in zip(v, b)]
        piv = next((i for i, x in enumerate(v) if x), None)
        if piv is None:
            continue
        basis.append((v, piv))
        chosen.append((row, rhs))
        if len(chosen) == nvars:
            return chosen
    return None


def _solve_small(F, rows, rhs):
    """Gaussian elimination on an n x n system over F_q."""
    n = len(rows)
    A = [list(r) + [b] for r, b in zip(rows, rhs)]
    for c in range(n):
        pr = next((i for i in range(c, n) if A[i][c]), None)
        if pr is None:
            return None
        A[c], A[pr] = A[pr], A[c]
        inv = F.inv(A[c][c])
        A[c] = [F.mul(v, inv) for v in A[c]]
        for i in range(n):
            if i != c and A[i][c]:
                f = A[i][c]
                A[i] = [F.sub(u, F.mul(f, v)) for u, v in zip(A[i], A[c])]
    return [A[i][n] for i in range(n)]


def recover_block_yux(c, pairs, j, verbose=False):
    """O5 sequential substitution for YuX (S = Pf^{-4}, output (z3,z2,z1,z0)).

    `pairs` is a list of (block plaintexts, weight column); every entry is one
    usable equation per coordinate.  The chain is the mirror image of DuX's and
    the ORDER is different:

      position 3, z0 = x3 - x0 x1 - x2 - alpha
          sum z0 = sum P3 - sum P0P1 - k1 sum P0 - k0 sum P1 - sum P2
          -- AFFINE in (k0, k1); k2 and k3 only ever appear with the factor
          |structure| = 0, so two independent equations give (k0, k1).
      position 2, z1 = x2 - z0 x0 - x1 - alpha
          k2 and k3 enter only through (k3 - k2) sum P0, so one equation gives
          the DIFFERENCE d = k3 - k2 and neither word on its own.
      position 0, z3 = x0 - z2 z1 - z0 - alpha
          substituting k3 = k2 + d makes z0 independent of k2, so z1 = A + k2,
          z2 = B - k2 z0 and z3 is quadratic in k2 with leading coefficient
          -sum z0 = 0.  It is therefore AFFINE in k2 -- one equation.
      position 1, z2 = x1 - z1 z0 - x0 - alpha
          sum z2 loses its k2 term for the same reason, so it determines
          nothing new and serves as the consistency check.

    THE DEGENERACY (the YuX analogue of O8's k1 mechanism).  Position 3's two
    coefficients are sum P1 and sum P0, i.e. the layer-(l+1) state at block
    POSITIONS 0 AND 1 -- and for every YuX single-word structure the
    layer-(l+1) pattern is exactly `1100`, so both vanish.  Substituting that
    back makes k2 invisible in all four coordinates as well (each later
    coefficient reduces to sum z0 = 0 or sum z1 z0 = 0), so the plain O5
    attack recovers NOTHING.  The fix is O12: run it on the WEIGHTED sums
    sum_x x^a (...) with a above the layer-(l+1) margin and below the layer-l
    one (`--auto-weight`).  Individual words can also be balanced by accident
    even when their position is not, which is why the equations are collected
    over several (structure, weight) pairs and independent ones are picked."""
    F = c.F
    cand = []
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        f00 = coord_sum(c, Pb, (0, 0, 0, 0), 3, w)
        f10 = coord_sum(c, Pb, (1, 0, 0, 0), 3, w)
        f01 = coord_sum(c, Pb, (0, 1, 0, 0), 3, w)
        cand.append((( F.sub(f10, f00), F.sub(f01, f00)), F.neg(f00)))
    sel = _independent_rows(F, cand, 2)
    if sel is None:
        return None, False, "coordinate 3 does not determine (k0,k1)"
    sol = _solve_small(F, [r for r, _ in sel], [b for _, b in sel])
    if sol is None:
        return None, False, "singular 2x2 system for (k0,k1)"
    k0, k1 = sol

    # --- d = k3 - k2 from coordinate 2 (the two coefficients must cancel)
    d = None
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        g00 = coord_sum(c, Pb, (k0, k1, 0, 0), 2, w)
        c2 = F.sub(coord_sum(c, Pb, (k0, k1, 1, 0), 2, w), g00)
        c3 = F.sub(coord_sum(c, Pb, (k0, k1, 0, 1), 2, w), g00)
        if F.add(c2, c3) != 0:
            return None, False, "coordinate 2 is not a function of k3 - k2 alone"
        if c3:
            d = F.mul(F.neg(g00), F.inv(c3))
            break
    if d is None:
        return None, False, "vanishing (k3-k2) coefficient (coordinate 2)"

    # --- k2 from coordinate 0, with k3 = k2 + d (affine because sum z0 = 0)
    k2 = None
    two = F.from_int(2)
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        h = lambda t: coord_sum(c, Pb, (k0, k1, t, F.add(t, d)), 0, w)  # noqa: E731
        h0, h1 = affine_coeffs_1(h, F)
        if h1 == 0:
            continue
        if h(two) != F.add(h0, F.mul(h1, two)):
            return None, False, "coordinate 0 is not affine in k2"
        k2 = F.mul(F.neg(h0), F.inv(h1))
        break
    if k2 is None:
        return None, False, "vanishing k2 coefficient (coordinate 0)"
    k3 = F.add(k2, d)

    k = (k0, k1, k2, k3)
    ok = all(coord_sum(c, tuple(P[4 * j + i] for i in range(4)), k, co, w) == 0
             for P, w in pairs for co in range(4))
    if verbose:
        print(f"  block {j}: k = {k}  consistent on {len(pairs)} equations: {ok}")
    return k, ok, None if ok else "consistency check failed"


def recover_block_dux(c, pairs, j, verbose=False):
    F = c.F
    cand = []
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        f = lambda k0, k3: coord_sum(c, Pb, (k0, 0, 0, k3), 2, w)   # noqa: E731
        f00, f10, f01 = f(0, 0), f(1, 0), f(0, 1)
        cand.append(((F.sub(f10, f00), F.sub(f01, f00)), F.neg(f00)))
    sel = _independent_rows(F, cand, 2)
    if sel is None:
        return None, False, "singular 2x2 system for (k0,k3)"
    sol = _solve_small(F, [r for r, _ in sel], [b for _, b in sel])
    if sol is None:
        return None, False, "singular 2x2 system for (k0,k3)"
    k0, k3 = sol

    # --- k2 from coordinate 1 (affine in k2 given k0,k3)
    k2 = None
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        g0, g1 = affine_coeffs_1(
            lambda t: coord_sum(c, Pb, (k0, 0, t, k3), 1, w), F)
        if g1:
            k2 = F.mul(F.neg(g0), F.inv(g1))
            break
    if k2 is None:
        return None, False, "vanishing k2 coefficient (coordinate 1)"

    # --- k1 from coordinate 3 (affine in k1 given k0,k2,k3)
    k1 = None
    for P, w in pairs:
        Pb = tuple(P[4 * j + i] for i in range(4))
        h0, h1 = affine_coeffs_1(
            lambda t: coord_sum(c, Pb, (k0, t, k2, k3), 3, w), F)
        if h1:
            k1 = F.mul(F.neg(h0), F.inv(h1))
            break
    if k1 is None:
        return None, False, "vanishing k1 coefficient (coordinate 3)"

    k = (k0, k1, k2, k3)
    ok = all(coord_sum(c, tuple(P[4 * j + i] for i in range(4)), k, co, w) == 0
             for P, w in pairs for co in range(4))
    if verbose:
        print(f"  block {j}: k = {k}  consistent on {len(pairs)} equations: {ok}")
    return k, ok, None if ok else "consistency check failed"


def recover_block(c, pairs, j, verbose=False):
    """Dispatch on the cipher family: the two S-boxes need different orders."""
    fn = recover_block_yux if isinstance(c, YuX) else recover_block_dux
    return fn(c, pairs, j, verbose=verbose)


def auto_weights(c, active, rounds, dim=None, extra=4):
    """Weights a that make the r_KR = 1 equations both VALID and USEFUL.

    The equation sum_x x^a S_c(P + k) = 0 is valid while the layer-l state
    (l = rounds - 1) stays balanced under the weight, i.e. a < hi with
    hi = min over words of (T - D) at layer l.  It is USEFUL only where the
    plaintext moments it multiplies do NOT vanish, i.e. where the
    layer-(l + 1) state is unbalanced.  Since the max-plus bound is met
    exactly by YuX (W17), word i's moment first becomes nonzero at
    a = T - D_i(layer l+1), so the candidate set is exactly those thresholds,
    per WORD (a position can be unbalanced overall while one block's word is
    balanced -- that is what breaks a naive per-position choice).

    For DuX with an active word at position 3 the layer-(l+1) pattern is
    `0000`, every threshold is <= 0 and the returned list starts at a = 0, so
    nothing changes.  For YuX the pattern is `1100` and a weight is mandatory:
    without one the attack recovers NOTHING (see recover_block_yux)."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      "..", "..", "tools"))
    from zero_sum_criterion import patterns as _patterns
    from cipher_degree import profile as _profile
    from dux.registry import cipher_family
    F = c.F
    fam = cipher_family(c.instance)
    q = f"2^{F.n}" if F.char == 2 else F.q
    dims = [dim] * len(active) if (F.char == 2 and dim) else None
    r = _patterns(q, list(active), rounds, dims, None, "dec", fam, [])
    T = r["threshold"]
    prof = _profile(list(active), rounds, "dec", F.char == 2, fam, [])
    hi = min(T - d for d in prof[rounds - 2])           # layer l still balanced
    thresholds = sorted({max(0, T - d) for d in prof[rounds - 1]})
    lo = min(thresholds)
    cand = sorted({a for a in thresholds if a < hi}
                  | {lo + t for t in range(extra) if lo + t < hi})
    assert cand and lo < hi, (
        f"no usable weight: layer {rounds} needs a >= {lo}, "
        f"layer {rounds - 1} allows a < {hi}")
    return cand, lo, hi


def build_data(c, rks, rounds, active, dim, rng, structures):
    """`structures` chosen-ciphertext structures as (plaintexts, active values)."""
    return [structure_plaintexts(c, rks, rounds, active, rng, dim)
            for _ in range(structures)]


def _moment_spectrum(F, xs, cols, a_lo, a_hi, block=512):
    """Yield (a, [sum_x x^a col for col in cols]) for a = a_lo .. a_hi - 1.

    Over F_{2^n} the products go through the log tables in BATCHES of weights:
    log(x^a) = a log(x) mod (q-1), so one (block x N) index array plus a gather
    and an XOR-reduce gives `block` weights at once.  That matters because the
    characteristic-2 instances need the scan to run over the whole validity
    window (their max-plus bound is not tight, W17), which is tens of thousands
    of weights at q = 2^16.

    x = 0 is excluded from the log-table sums (0^a = 0 for a >= 1) and added
    back for a = 0, where 0^0 = 1."""
    xs = np.asarray(xs, dtype=np.int64)
    zero_terms = None
    if a_lo == 0:
        z = np.flatnonzero(xs % (F.q if F.char == 2 else F.p) == 0)
        zero_terms = [int(F.vsum(np.asarray(c, dtype=np.int64)[z])) if z.size else 0
                      for c in cols]

    def _fix(a, v):
        if a or zero_terms is None:
            return v
        return [F.add(x, y) for x, y in zip(v, zero_terms)]

    if F.char != 2:
        p = F.p
        nz = xs % p != 0
        xn = xs[nz] % p
        cn = [np.asarray(c, dtype=np.int64)[nz] % p for c in cols]
        xp = np.array([pow(int(v), a_lo, p) for v in xn], dtype=np.int64)
        for a in range(a_lo, a_hi):
            yield a, _fix(a, [int((xp * c % p).sum() % p) for c in cn])
            xp = (xp * xn) % p
        return
    order = F.order
    nz = xs != 0
    lx = np.asarray(F._log, dtype=np.int64)[xs[nz]]
    exp = np.asarray(F._exp, dtype=np.int64)
    lcols = []
    for c in cols:
        c = np.asarray(c, dtype=np.int64)[nz]
        lc = np.asarray(F._log, dtype=np.int64)[c].copy()
        lcols.append((lc, c != 0))
    a = a_lo
    while a < a_hi:
        n = min(block, a_hi - a)
        av = np.arange(a, a + n, dtype=np.int64)
        alx = (av[:, None] * lx[None, :]) % order
        out = []
        for lc, mask in lcols:
            v = exp[(alx + lc[None, :]) % order]
            v = np.where(mask[None, :], v, 0)
            out.append(np.bitwise_xor.reduce(v, axis=1))
        for t in range(n):
            yield a + t, _fix(a + t, [int(o[t]) for o in out])
        a += n


def measured_weights(c, data, j, lo, hi, cap=None, extra=4):
    """Weights that actually make the block-j equations solvable, measured.

    `auto_weights` derives its candidates from the max-plus bound, which YuX
    meets exactly over F_p but NOT over F_{2^n} (characteristic 2 collapses
    some degrees, W17), so there the predicted thresholds are only lower
    bounds.  This scan finds the real ones: the two coefficients of the first
    equation are exactly (-sum_x x^a P_{4j+1}, -sum_x x^a P_{4j+0}) for YuX and
    (-sum_x x^a P_{4j+3}, -sum_x x^a P_{4j+0}) for DuX, i.e. two weighted
    moments per weight instead of a full S-box pass.  Weights are collected
    until those coefficient vectors span the plane -- which is what determines
    (k0,k1) resp. (k0,k3) -- plus a few spare ones for the substitution steps."""
    from yux import YuX as _YuX
    F = c.F
    words = (1, 0) if isinstance(c, _YuX) else (3, 0)
    hi = hi if cap is None else min(hi, lo + cap)
    out, basis, spare = [], [], 0
    gens = [(_moment_spectrum(F, xs, [P[4 * j + w] for w in words], lo, hi), )
            for P, xs in data]
    for step in zip(*[g[0] for g in gens]):
        a = step[0][0]
        useful = False
        for _a, v in step:
            if not any(v):
                continue
            v = list(v)
            for b, piv in basis:
                if v[piv]:
                    f = F.mul(v[piv], F.inv(b[piv]))
                    v = [F.sub(x, F.mul(f, y)) for x, y in zip(v, b)]
            piv = next((i for i, x in enumerate(v) if x), None)
            if piv is not None and len(basis) < 2:
                basis.append((v, piv))
                useful = True
            elif len(basis) >= 2:
                useful = True
        if useful:
            out.append(a)
            if len(basis) >= 2:
                spare += 1
                if spare > extra:
                    break
    return out


def weight_batches(weights, lo, hi, batch=8, cap=512):
    """The weight lists to try, in order: the explicit candidates first, then a
    scan upwards from `lo`.

    `auto_weights` derives its candidates from the MAX-PLUS bound, which YuX
    meets exactly over F_p but not over F_{2^n} (characteristic 2 collapses
    some degrees, W17).  There the true threshold is only >= the predicted one,
    so the scan is what makes the binary instances work."""
    yield list(weights)
    if hi is None:
        return
    a = lo
    tried = 0
    while a < hi and tried < cap:
        nxt = [v for v in range(a, min(a + batch, hi)) if v not in weights]
        a += batch
        tried += batch
        if nxt:
            yield nxt


def attack_once(c, rks, rounds, active, dim, rng, structures, max_extra=3,
                verbose=False, weights=(0,), lo=None, hi=None, report=None):
    """Run the attack; add structures (and, for YuX, weights) until it works.

    Returns (recovered key or None, structures used, failure reasons).  The
    optional `report` dict collects the weights that actually produced the
    equations (O12); with the default weights = (0,) it stays {0}, so the DuX
    behaviour and this signature are unchanged."""
    F = c.F
    weights = list(weights)
    data = build_data(c, rks, rounds, active, dim, rng, structures)
    reasons, used = [], structures
    for _extra in range(max_extra + 1):
        rec, bad, wused = [], None, set()
        for j in range(4):
            batches = [list(weights)]
            if hi is not None:
                batches.append(measured_weights(c, data, j, lo, hi))
            pairs, k, ok, why = [], None, False, "no weights tried"
            for batch in batches:
                if not batch:
                    continue
                pairs += [(P, weight_col(F, xs, a)) for (P, xs) in data
                          for a in batch]
                k, ok, why = recover_block(c, pairs, j, verbose=verbose)
                if k is not None and ok:
                    wused |= set(batch)
                    break
            if k is None or not ok:
                bad = why
                break
            rec += list(k)
        if bad is None:
            if report is not None:
                report["weights_used"] = sorted(wused)
            return tuple(rec), used, reasons
        reasons.append(bad)
        data += build_data(c, rks, rounds, active, dim, rng, 1)
        used += 1
    if report is not None:
        report["weights_used"] = []
    return None, used, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537",
                    help="dux-* / toy-* or yu2x-* / yupx-* / yuxtoy-* "
                         "(the cipher is chosen by dux.registry)")
    ap.add_argument("--rounds", type=int, default=10, help="reduced rounds r = l + 1")
    ap.add_argument("--pos", type=int, default=3, help="active ciphertext word")
    ap.add_argument("--active", default=None,
                    help="comma-separated active words (overrides --pos)")
    ap.add_argument("--dim", type=int, default=None,
                    help="F_2-dimension per active word (F_{2^n}); default n")
    ap.add_argument("--structures", type=int, default=3)
    ap.add_argument("--weight", type=int, default=0,
                    help="O12: use the weighted sums sum_x x^a (...) instead of "
                         "the plain ones.  Needed for YuX, whose layer-(l+1) "
                         "pattern `1100` kills exactly the two moments the "
                         "cheapest equation uses")
    ap.add_argument("--nweights", type=int, default=4,
                    help="with --auto-weight: how many consecutive weights to "
                         "add below the per-word thresholds; without it, how "
                         "many consecutive weights starting at --weight")
    ap.add_argument("--auto-weight", action="store_true",
                    help="pick the smallest weight that breaks the layer-(l+1) "
                         "balance while staying inside the layer-l margin")
    ap.add_argument("--keys", type=int, default=1,
                    help=">1: success-rate mode over that many random master keys")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ks-literal", action="store_true",
                    help="YuX only: read Algorithm 1 line 2 as four SLIDING "
                         "windows (key_i..key_{i+3}) instead of the "
                         "non-overlapping blocks key_{4i}..key_{4i+3}.  The "
                         "attack recovers rk^0 = the master key itself, which "
                         "is the same under both readings (yux.keyschedule "
                         "guarantees rks[0] == tuple(K)), so this is a control: "
                         "the result must not change.  Silently ignored for "
                         "DuX, which has no such ambiguity.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    c = get_cipher(a.instance, ks_literal=a.ks_literal) if a.ks_literal \
        else get_cipher(a.instance)
    active = [a.pos] if a.active is None else [int(v) for v in a.active.split(",")]
    weights, wlo, whi = [a.weight], None, None
    if a.auto_weight:
        weights, wlo, whi = auto_weights(c, active, a.rounds, a.dim, a.nweights)
        print(f"  auto weights: a in {weights} "
              f"(the layer-{a.rounds} balance of some word breaks at each of "
              f"them; the layer-{a.rounds - 1} zero-sum holds for a < {whi}; "
              f"a scan up from {wlo} is the fallback)")
    elif a.nweights > 1:
        weights = [a.weight + t for t in range(a.nweights)]
    dim = a.dim
    per = (c.F.n if dim is None else dim) if c.F.char == 2 else int(np.log2(c.F.q))
    data_log2_struct = per * len(active)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    if a.keys > 1:
        ok_count, used_hist, reasons = 0, [], {}
        weights_used = set()
        for _ in range(a.keys):
            K = c.random_key(rng)
            rks = c.key_schedule(K)
            rep = {}
            rec, used, why = attack_once(
                c, rks, a.rounds, active, dim, rng, a.structures,
                weights=weights, lo=wlo, hi=whi, report=rep)
            weights_used |= set(rep.get("weights_used", []))
            ok_count += (rec == tuple(K))
            used_hist.append(used)
            for w in why:
                reasons[w] = reasons.get(w, 0) + 1
        el = time.time() - t0
        print(f"{a.instance}, {a.rounds} rounds, active {active}, "
              f"{a.structures} structures x 2^{data_log2_struct} chosen ciphertexts")
        print(f"  success {ok_count}/{a.keys} = {100.0 * ok_count / a.keys:.1f}%   "
              f"structures used: min {min(used_hist)}, max {max(used_hist)}, "
              f"mean {sum(used_hist) / len(used_hist):.2f}")
        print(f"  retries triggered by: {reasons or 'none'}   "
              f"({el:.1f}s, {el / a.keys:.2f}s per key)")
        res = {"instance": a.instance, "rounds": a.rounds, "active": active,
               "dim": dim, "structures": a.structures, "keys": a.keys,
               "weights": weights, "weights_used": sorted(weights_used),
               "weight_range": [wlo, whi],
               "success": ok_count, "success_rate": ok_count / a.keys,
               "structures_used": used_hist, "retry_reasons": reasons,
               "data_log2_per_structure": data_log2_struct,
               "seed": a.seed, "ks_literal": a.ks_literal, "elapsed_s": round(el, 1)}
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fn = os.path.join(a.out, f"success_rate_{a.instance}_r{a.rounds}.json")
            json.dump(res, open(fn, "w"), indent=1)
            print("saved", fn)
        return

    K = c.random_key(rng)
    rks = c.key_schedule(K)
    data = build_data(c, rks, a.rounds, active, dim, rng, a.structures)
    t_data = time.time() - t0
    print(f"{a.instance}, {a.rounds} rounds, {a.structures} structures x "
          f"2^{data_log2_struct} chosen ciphertexts (active {active}); "
          f"decryption oracle time {t_data:.1f}s")
    rec, oks, weights_used = [], [], set()
    for j in range(4):
        batches = [list(weights)]
        if whi is not None:
            batches.append(measured_weights(c, data, j, wlo, whi))
        pairs, k, ok, why = [], None, False, "no weights tried"
        for batch in batches:
            if not batch:
                continue
            pairs += [(P, weight_col(c.F, xs, w)) for (P, xs) in data
                      for w in batch]
            k, ok, why = recover_block(c, pairs, j, verbose=True)
            if k is not None and ok:
                weights_used |= set(batch)
                break
        if k is None:
            print(f"  block {j}: FAILED ({why})")
            rec += [None] * 4
            oks.append(False)
            continue
        rec += list(k)
        oks.append(ok)
    rec = tuple(rec)
    success = rec == tuple(K)
    print(f"recovered rk^0 == master key K: {success}   "
          f"(elapsed {time.time() - t0:.1f}s)")
    if not success:
        print("  true K:", K)
        print("  rec  K:", rec)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"{a.instance}_r{a.rounds}_"
                                 f"{'_'.join(map(str, active))}.json")
        json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                   "dim": dim, "structures": a.structures, "weights": weights,
                   "weights_used": sorted(weights_used),
                   "data_log2": data_log2_struct + float(np.log2(a.structures)),
                   "success": success, "block_consistency": oks,
                   "elapsed_s": round(time.time() - t0, 1)}, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
