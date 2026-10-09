"""E07 / W16 -- weighted moments (O12) + cheap combined rows (O10) as a driver.

One structure with s active ciphertext words gives, instead of one batch of
equations, ONE BATCH PER ADMISSIBLE WEIGHT a in N^s with |a| < T - D:

    sum_x x^a * (the r_KR = 2 equation)  =  0 .

The admissible range is set by the LARGEST formal degree among the layer-l
words the equation actually uses:

  * the plain E07 equation for outer block j is the coordinate-2 output of the
    outer S-box, i.e. [L^{-1}(Z)]_{4j+2}, and L^{-1} maps position 2 to the
    positions {2, 3} of Z (O3), so  a < T - max(D_2, D_3);
  * the O10 combined row for the pattern `0001` only uses Z's position-3 rows,
    so  a < T - D_3  -- a much longer weight range (toy-257 layer 5: 159
    instead of 47);
  * in the CPA direction the outer map is S^{-1} and the equation uses L, which
    is dense in both characteristics, so a < T - max_p D_p.

`weight_margin` reads those degrees straight out of
`tools/zero_sum_criterion.py`'s `margin_per_position`.

The moments themselves come from `assemble_fast.weighted_moment_stream`, i.e.
the slice decomposition of memo Sect. 5.4: per-slice plain moments (exactly
what `Moments.add` computes) times a Vandermonde matrix, no NTT.
"""
from __future__ import annotations

import math
import os
import sys
from typing import NamedTuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "E08_boolean_degree_extension"))

from assemble_fast import (MomentLayout, _pow_matrix, combine_rows,  # noqa: E402
                           rows_from_moments, weight_vector_count,
                           weight_vectors, weighted_moment_stream)
from assemble_fast_2n import rows_from_moments_2n                    # noqa: E402
from attack_2round_toy import active_words, structure_data           # noqa: E402
import zero_sum_criterion as zsc                                     # noqa: E402
from cheap_rows import block_coeffs                                  # noqa: E402


def field_key(F):
    """The `--q` string tools/zero_sum_criterion.py expects."""
    return f"2^{F.n}" if F.char == 2 else F.q


def used_positions(direction="dec", cipher="dux", combine=None):
    """The layer-l positions of Z the equation rows actually read.

    The weight range is set by the largest formal degree among THESE positions,
    not by the whole state (that is the point of the O10 combined rows)."""
    if combine:
        # the O10 combined row only touches the balanced positions of the pattern
        return [p for p in range(4) if combine[p] == "1"]
    if direction == "enc":
        return list(range(4))                 # L is dense in both characteristics
    # W'_i = [L^{-1}(Z - rk^2)]_i uses Z at the positions the rotation
    # offsets of L^{-1} reach: {0,1} for DuX (O3), all four for YuX.
    from cipher_degree import CIPHERS
    offs = {o % 4 for o in CIPHERS[cipher]["ROT_INV"]}
    cheap = 2 if cipher == "dux" else 3       # cheap coordinate of the outer S
    return sorted({(cheap + o) % 4 for o in offs})


def weight_margin(F, active, layers, direction="dec", cipher="dux",
                  combine=None, free_blocks=(), subspace_dims=None, coset=None,
                  pointset=None):
    """T - max_{positions the equation uses} D, i.e. the number of admissible
    one-dimensional weights (a = 0 .. margin - 1) BEFORE the R6 corrections.

    `subspace_dims` / `coset` are passed straight to the criterion so that the
    subspace and coset thresholds are the ones of O7's table; `usable_weights`
    (below) is what a caller should actually use."""
    r = zsc.patterns(field_key(F), list(active), layers, subspace_dims, coset,
                     direction, cipher, list(free_blocks), pointset)
    marg = r["margin_per_position"][layers - 1]
    used = used_positions(direction, cipher, combine)
    return min(marg[p] for p in used), r


def point_sets(F, nwords, n, seed):
    """`nwords` random point sets of n distinct elements of F_q (O15).

    Sorted, so the structure's axis order is deterministic; the seed is the
    only source of randomness, as the protocol requires."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(F.q, size=n, replace=False)).astype(np.int64)
            for _ in range(nwords)]


def structure_powers(c, rks, rounds, pos, seed, active, pre, points=None):
    """Decrypt one structure and return (PW, xvals, npoints).

    `PW[b]` is the (nP x N) matrix of plaintext monomial values of inner block
    b and `xvals[j]` the value of the j-th active CIPHERTEXT word at every
    point (axis 0 slowest, matching `structure_data`'s product-set order).

    With `points` (O15) the active words run over the given point sets instead
    of the whole field; `structure_data`'s `values` argument already supports
    that (it was added for the S13-D coset run)."""
    F = c.F
    act = active_words(pos, active)
    s = len(act)
    P = structure_data(c, rks, rounds, pos, seed, active,
                       values=None if points is None else list(points))
    N = len(P[0])
    idx = np.arange(N, dtype=np.int64)
    if points is None:
        xvals = [(idx // (F.q ** (s - 1 - j))) % F.q for j in range(s)]
    else:
        sizes = [len(u) for u in points]
        xvals = []
        for j in range(s):
            stride = 1
            for n_ in sizes[j + 1:]:
                stride *= n_
            xvals.append(np.asarray(points[j], dtype=np.int64)[(idx // stride) % sizes[j]])
    PW = {b: _pow_matrix(F, P, b, pre.pe_arr) for b in pre.unknown}
    return PW, xvals, N


def true_key_vector(c, pre, rks):
    """The linearisation monomials at the key of the EQUIVALENT fixed-L0 cipher
    (O2): the assembly always uses L0, so the outer key words are
    rk'^1 = Rot_{4 t(rk^1)}(rk^1).  The inner key rk^0 is unchanged, which is
    why the attack still returns the master key."""
    import numpy as _np
    from attack_2round_toy import monomial_value
    eq, _T = c.equivalent_fixedL0_keys(rks)
    return _np.array([monomial_value(c.F, m, eq[0], eq[1], c.F.q - 1)
                      for m in pre.mons], dtype=_np.int64)


def weighted_rows(pre, c, rks, rounds, pos, seed, active, weights, lrow,
                  ycomb=None, progress=None, points=None, mask=None):
    """All equation rows of ONE structure, one batch per weight.

    Without `ycomb` there are len(pre.outer) rows per weight (the plain E07
    equations); with it there is one row per kernel vector of O10.

    `points` / `mask` (O15) replace the full-domain product set by a product of
    arbitrary point sets and their divided-difference weights."""
    F = c.F
    asm = rows_from_moments_2n if F.char == 2 else rows_from_moments
    PW, xvals, N = structure_powers(c, rks, rounds, pos, seed, active, pre, points)
    dims = len(weights[0])
    assert dims <= len(xvals), "more weight axes than active words"
    layout = MomentLayout(pre)
    out, order = [], []
    for a, mom in weighted_moment_stream(pre, PW, xvals[:dims], weights, layout,
                                         progress=progress, points=points,
                                         mask=mask):
        rows = asm(pre, mom, lrow)
        if ycomb is None:
            out.extend(rows)
            order.extend([(a, j) for j in pre.outer])
        else:
            for y in ycomb:
                out.append(combine_rows(rows, y, F))
                order.append((a, tuple(y)))
    return np.asarray(out, dtype=np.int64), order, N


def combine_vectors(cipher, F, direction, pattern, cheap=None):
    """The per-outer-block coefficient vectors of the O10 combined rows.

    `cheap` overrides the family's cheap set.  T1 (R9) uses {1, 2} for DuX in
    characteristic 2 (the cubic coordinate b is Frobenius-bilinear there), which
    turns `1101` from dim K = 1 into dim K = 4; with two cheap positions
    `block_coeffs` returns a dict keyed by (outer block, cheap position), which
    is what `experiments/E13_b_coordinate/bcoord.combine_rows_b` consumes."""
    return block_coeffs(cipher, str(field_key(F)), direction, pattern, cheap)


class Plan(NamedTuple):
    """What `plan` decides for one (structure, layer, row set).

    `margin` is the raw degree budget T - max_{used} D; `usable_weights` is the
    number of weight vectors a caller may actually use, i.e. `margin` AFTER the
    four R6 corrections (S12 Sect. 7.5) AND the W25 axis cap.  Read
    `usable_weights`, not `margin`."""
    weights: list          # the admissible weight vectors, already filtered/capped
    margin: int            # T - max_{used positions} D, in degree units
    crit: dict             # the full tools/zero_sum_criterion.py result
    usable_weights: int    # number of admissible weight vectors at this `dims`
    weight_rule: str       # which rule produced it (one line, for the log)
    usable_norm: int       # the effective bound: |a| < usable_norm
    weight_word: int       # full block: the ciphertext word weights must use
    a0_only_rows: list     # CPA: positions with D = q (O9) -- those take a = 0
    coset_exclude: list    # coset: drop a with coset_exclude[i] | a_i
    axis_cap: int = 1 << 62   # W25: prod of the weight axes' point counts
    degree_weights: int = 0   # weights the DEGREE margin alone admits
    weight_words: tuple = ()  # T3: full block, the ciphertext words used
    cheap_set: tuple = ()     # T1: the cheap coordinate set of the combined row


# T1 (R9): the extra usable classes characteristic 2 gains from the cheap set
# {1, 2} -- position 1 (the cubic b) plus position 2 (the quadratic a).  A
# combined row of one of these patterns reads positions 1 and 3, hence the name
# "x1x1", and has dim K = 4 (`results/E13_b_coordinate/kernel_cheap12.json`).
X1X1_CLASSES = ("0101", "0111", "1101", "1111")


def plan(F, active, layers, direction="dec", cipher="dux", combine=None,
         dims=1, nweights=None, free_blocks=(), subspace_dims=None, coset=None,
         pointset=None, cheap_set=None):
    """Pick the weight set: `dims`-dimensional, |a| < usable_norm, capped at
    `nweights` (in the |a| order `weight_vectors` produces).

    The four corrections `margin` alone does NOT contain (S12 Sect. 7.5, R6
    rule 3), each of which used to be the caller's job to remember:

    * **full-domain word** -- nothing to correct, `usable_norm = margin`.
    * **full-block structure** (O11, `free_blocks` non-empty) -- the weights can
      only be put on a CIPHERTEXT word, and after the substitution
      y = S^{-1}(x + k) the cheapest such coordinate (DuX position 2, YuX
      position 3) has degree 2 in y, so every power eats TWO degree units:
      `usable_norm = margin // 2`, and `weight_word` says which word.
    * **subspace word** (characteristic 2, `subspace_dims`) -- the threshold is
      already sum_i (2^{m_i} - 1), so `usable_norm = margin` = 2^m - 1 - D for a
      single active word; the rule string records that it is a subspace bound.
    * **point set** (O15, `pointset` = the sizes n_i) -- the divided-difference
      mask makes the threshold T' = sum_i (n_i - 1) in either characteristic,
      so `usable_norm = margin` exactly as for full-domain words; the caller is
      responsible for passing the same point sets to the assembly.
    * **coset** (F_p, order-2^k subgroups) -- Theorem O7's coset factor is
      nonzero exactly when 2^{k_i} divides a_i + e_i, and the counting argument
      forces one such index to be 0; a single coset therefore needs EVERY axis
      to satisfy 2^{k_i} does not divide a_i (O12 Sect. 5.5).  Those vectors are
      filtered out and `coset_exclude` returns the moduli.
    * **CPA rows on an O9 cell** -- a word whose formal degree is exactly q is
      balanced because its next-to-leading coefficient c_{q-1} vanishes (O9),
      which says nothing about c_{q-1-a} for a >= 1.  If such a position is
      among the rows used, the equation admits `a = 0` only; `a0_only_rows`
      lists the positions."""
    free_blocks = tuple(sorted(set(free_blocks)))
    if cheap_set is not None:
        cheap_set = tuple(sorted(cheap_set))
        assert cheap_set == (2,) or (F.char == 2 and cheap_set == (1, 2)
                                     and cipher == "dux" and direction == "dec"), \
            "the only non-default cheap set is T1's {1,2}: DuX, decryption, " \
            "characteristic 2"
        if cheap_set == (1, 2):
            assert combine in X1X1_CLASSES, (
                f"the cheap set {{1,2}} needs a pattern with positions 1 and 3 "
                f"balanced, i.e. one of {list(X1X1_CLASSES)}, not {combine}")
    margin, crit = weight_margin(F, active, layers, direction, cipher, combine,
                                 free_blocks, subspace_dims, coset, pointset)
    used = used_positions(direction, cipher, combine)
    D = crit["max_degree_per_position"][layers - 1]
    a0_only = [p for p in range(4) if D[p] == F.q]
    weight_word, coset_exclude, block_costs = -1, [], None
    weight_words = ()

    if a0_only and any(p in used for p in a0_only):
        # O9 cell among the rows we read: the plain sum only.
        norm = 1
        rule = (f"O9 cell: positions {sorted(set(a0_only) & set(used))} have "
                f"D = q = {F.q}; balanced by the vanishing c_(q-1), so a = 0 only")
    else:
        assert margin > 0, (
            f"layer {layers} leaves no margin for the positions the equation "
            f"uses ({used}): {crit['margin_per_position'][layers - 1]}")
        if free_blocks and dims > 1:
            # T3 (R9): spread the weight over the `dims` cheapest words of the
            # block, each power costing that word's degree in y.
            costs_pos = block_weight_costs(cipher, direction)[:dims]
            block_costs = [d for _p, d in costs_pos]
            weight_words = tuple(4 * free_blocks[0] + p for p, _d in costs_pos)
            weight_word = weight_words[0]
            norm = margin
            rule = (f"full block {list(free_blocks)} (O11), T3 multi-index: "
                    f"weights on ciphertext words {list(weight_words)} "
                    f"(positions {[p for p, _ in costs_pos]}, degrees "
                    f"{block_costs} after the substitution) => "
                    f"sum_i a_i deg_i < {margin}")
        elif free_blocks:
            from cipher_degree import CIPHERS
            cheap = CIPHERS[cipher]["cheap_dec" if direction == "dec"
                                    else "cheap_enc"][0]
            weight_word = 4 * free_blocks[0] + cheap
            norm = margin // 2
            rule = (f"full block {list(free_blocks)} (O11): weights on ciphertext "
                    f"word {weight_word} (position {cheap}, degree 2 after the "
                    f"substitution) => 2|a| < {margin}, so |a| < {norm}")
        elif pointset:
            norm = margin
            rule = (f"O15 point sets of sizes {list(pointset)}: T' = "
                    f"{sum(n - 1 for n in pointset)} = sum_i (n_i - 1), "
                    f"|a| < T' - D = {margin}")
        elif coset:
            ks = coset if isinstance(coset, list) else [coset] * len(active)
            # T2 (R9): an entry None is a FULL-FIELD word.  Its sum over F_p
            # kills the constant term all by itself (sum_x 1 = p = 0), so it
            # carries no exclusion; a coset axis still needs a_i >= 1 and
            # 2^{k_i} not dividing a_i, which is what a modulus 0 / m encodes.
            coset_exclude = [0 if k is None else 2 ** k for k in ks]
            norm = margin
            kinds = ["full" if k is None else f"2^{k}" for k in ks]
            rule = (f"structure words {kinds}: |a| < {margin}"
                    + ("; every coset axis needs a_i >= 1 and 2^k not dividing "
                       "a_i (O12 Sect. 5.5), the full-field axes are "
                       "unconstrained (T2)" if any(k is None for k in ks)
                       else " and no axis with 2^k | a_i (O12 Sect. 5.5)"))
        elif subspace_dims:
            norm = margin
            rule = (f"subspace dims {list(subspace_dims)}: |a| < T - D = "
                    f"sum_i (2^m_i - 1) - D = {margin}")
        else:
            norm = margin
            rule = f"full-domain words: |a| < T - D = {margin}"
        assert norm >= 1, (f"the correction leaves no usable weight: "
                           f"margin {margin}, rule '{rule}'")

    if block_costs:
        usable = cost_weight_count(block_costs, norm)
        ws = cost_weight_vectors(block_costs, norm, limit=nweights)
    elif any(coset_exclude):
        # the filter is not a prefix of the order, so the vectors must be
        # enumerated; the COUNT, though, has a closed form (`coset_weight_count`)
        # which the mixed structures of T2 need -- their two-dimensional weight
        # sets run into the millions.
        mods = (coset_exclude * dims)[:dims]
        usable = coset_weight_count(dims, norm, mods)
        ws = filtered_weight_vectors(dims, norm, mods, limit=nweights)
    else:
        # the closed form, so that a two-dimensional five-digit margin (billions
        # of vectors) can be PLANNED without being BUILT (W25 / O15).
        usable = weight_vector_count(dims, norm)
        ws = weight_vectors(dims, norm, limit=nweights)

    # ---- W25: the RANK cap the degree margin does not see -------------------
    # The weighted rows of one structure are the images of the functionals
    # x -> w(x) x^a on the weight axes.  An axis of m points carries only an
    # m-dimensional space of functions, and 1, x, ..., x^{m-1} already span it
    # (Vandermonde), so AT MOST m WEIGHTS PER AXIS ARE INDEPENDENT, whatever
    # the degree margin says.  Measured on toy-2^4 (three words, layer 3, four
    # rows per weight): 16 / 24 / 30 full-domain weights all give rank 64 =
    # 4 x min(N_w, q), and a 14-point set gives 4 x 14 = 56.  No published row
    # is affected -- every one of them has N_w below its axis count -- but a
    # PLAN can exceed it, and the O15 point sets make that easy to do by
    # accident, so the cap belongs here.
    axis = _axis_sizes(F, dims, free_blocks, subspace_dims, coset, pointset)
    cap = 1
    for m in axis:
        cap *= m
    degree_weights = usable
    if cap < usable:
        usable = cap
        rule += (f"; capped at {cap} by the weight axes {axis} (at most |axis| "
                 f"independent weights per axis, W25)")
        if nweights is None or nweights > cap:
            ws = ws[:cap]
    if cheap_set == (1, 2):
        rule += ("; T1 cheap set {1,2} (characteristic 2 only): the combined "
                 "row reads positions 1 and 3, dim K = 4")
    return Plan(ws, margin, crit, usable, rule, norm, weight_word,
                a0_only, coset_exclude, cap, degree_weights, weight_words,
                tuple(cheap_set) if cheap_set else ())


# ---- T3 (R9): the per-word cost of a weight on a FULL-BLOCK structure -------
# With the block b of the ciphertext running over F_q^4 the substitution
# y = S^{-1}(x + rk) makes y the free variable (O11), so a weight prod_i x_i^{a_i}
# becomes prod_i (S_i(y) - rk_i)^{a_i}, of degree sum_i a_i deg(S_i) in y.  The
# paper's rule ("weights on the cheapest ciphertext word, two degree units per
# power") is the dims = 1 case of exactly this, with deg = 2.
#
#   DuX  S   = (c, b, a, y3) at positions 0..3, degrees (5, 3, 2, 8)
#   YuX  S   = (z3, z2, z1, z0)                 degrees (8, 5, 3, 2)
#   DuX  S^{-1}                                 degrees (2, 3, 4, 2)   (CPA)
#   YuX  S^{-1}                                 degrees (2, 2, 3, 4)   (CPA)
BLOCK_WEIGHT_DEGREES = {
    ("dux", "dec"): (5, 3, 2, 8),
    ("yux", "dec"): (8, 5, 3, 2),
    ("dux", "enc"): (2, 3, 4, 2),
    ("yux", "enc"): (2, 2, 3, 4),
}


def block_weight_costs(cipher, direction="dec"):
    """[(position, cost), ...] for the four words of a free block, cheapest
    first -- DuX/dec gives [(2,2), (1,3), (0,5), (3,8)]."""
    degs = BLOCK_WEIGHT_DEGREES[(cipher, direction)]
    return sorted(((p, d) for p, d in enumerate(degs)), key=lambda t: (t[1], t[0]))


def cost_weight_count(costs, norm):
    """#{a in N^len(costs) : sum_i a_i costs[i] < norm}, by one truncated
    polynomial product per axis (the multi-index generalisation of
    `assemble_fast.weight_vector_count`, which is this with every cost 1)."""
    acc = np.zeros(norm, dtype=object)
    acc[0] = 1
    for c in costs:
        col = np.zeros(norm, dtype=object)
        col[::c] = 1                                  # 1 at 0, c, 2c, ...
        acc = np.convolve(acc, col)[:norm]
    return int(acc.sum())


def cost_weight_vectors(costs, norm, limit=None):
    """The admissible a themselves, ordered by sum_i a_i costs[i] then
    lexicographically (so the first `limit` of them are the cheapest)."""
    dims = len(costs)
    out = []
    for total in range(norm):
        def rec(k, left, acc):
            if limit is not None and len(out) >= limit:
                return
            if k == dims - 1:
                if left % costs[k] == 0:
                    out.append(tuple(acc + [left // costs[k]]))
                return
            for v in range(0, left + 1, costs[k]):
                rec(k + 1, left - v, acc + [v // costs[k]])
        rec(0, total, [])
        if limit is not None and len(out) >= limit:
            break
    return out[:limit] if limit is not None else out


def coset_weight_count(dims, norm, mods):
    """#{a in N^dims : |a| < norm, a_j >= 1 and mods[j] does not divide a_j
    on every axis with mods[j] != 0}.

    One truncated polynomial product per axis: axis j contributes the series
    sum_t [t admissible] x^t, and the answer is the sum of the coefficients of
    the product up to degree norm - 1."""
    acc = np.zeros(norm, dtype=object)
    acc[0] = 1
    for j in range(dims):
        m = mods[j] if j < len(mods) else 0
        col = np.ones(norm, dtype=object)
        if m:
            col[0] = 0
            col[::m] = 0                       # every multiple of m, 0 included
        acc = np.convolve(acc, col)[:norm]
    return int(acc.sum())


def filtered_weight_vectors(dims, norm, mods, limit=None):
    """The admissible weight vectors themselves, in the |a| order, at most
    `limit` of them (the cheapest ones, which is what every caller wants)."""
    out = []
    for total in range(norm):
        def rec(k, left, acc):
            if limit is not None and len(out) >= limit:
                return
            if k == dims - 1:
                m = mods[k] if k < len(mods) else 0
                if not m or (left and left % m):
                    out.append(tuple(acc + [left]))
                return
            m = mods[k] if k < len(mods) else 0
            for v in range(left + 1):
                if m and (not v or v % m == 0):
                    continue
                rec(k + 1, left - v, acc + [v])
        rec(0, total, [])
        if limit is not None and len(out) >= limit:
            break
    return out[:limit] if limit is not None else out


def _axis_sizes(F, dims, free_blocks, subspace_dims, coset, pointset):
    """Number of points on each of the first `dims` weight axes."""
    if pointset:
        sizes = list(pointset)
    elif coset:
        ks = coset if isinstance(coset, list) else [coset] * dims
        sizes = [F.q if k is None else 2 ** k for k in ks]
    elif subspace_dims:
        sizes = [2 ** m for m in subspace_dims]
    else:
        sizes = [F.q] * max(dims, 1)          # full domain, and the O11 free block
    return (sizes + [F.q] * dims)[:dims]
