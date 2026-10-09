"""Unified zero-sum criterion for DuX and YuX (R5: both ciphers, both directions).

Formal total degree (max-plus) bound, all active words tracked jointly:

    one S^{-1} layer (closed form S^{-1} = R^4, see dux/sbox.py):
        d0 = max(e1+e2, e0)      d3 = max(e0+e1, e3)
        d1 = max(e2+d3, e1)      d2 = max(d3+d0, e2)
    L^{-1} = Rot1 + Rot4 + Rot8 + Rot9 + Rot13  (convention (X<<<j)_i = X_{i+j}):
        e'_i = max(d_{i+1}, d_{i+4}, d_{i+8}, d_{i+9}, d_{i+13})
    (L1^{-1} = Rot4 o L0^{-1} only permutes blocks; the block-symmetric bound
     dominates every block from layer 3 on, so patterns are rotation-independent.)

Theorem (O7).  Let the s active ciphertext words range over U_1 x ... x U_s and
let D be the max-plus bound on the total degree of a state word.  The word is
balanced (zero-sum) as soon as  D < T  with
    T = sum_i (2^{m_i} - 1)          F_{2^n}, U_i affine subspace of dim m_i (m_i = n: full field)
    T = s * (p - 1)                  F_p, every U_i = F_p
    T = sum_i 2^{k_i}                F_p, U_i = coset of the order-2^{k_i} subgroup, *signed*
                                     2^s-coset inclusion-exclusion (s = 1: difference of two cosets)
    T = sum_i T_i                    F_p, MIXED (T2, R9): pass `coset` as a list with
                                     None for a full-field word (T_i = p - 1) and k for
                                     a coset word (T_i = 2^k); the proof is per word.
                                     CAVEAT: an UNSIGNED coset factor at exponent 0 is
                                     |H| = 2^k != 0, so a coset word only reaches its
                                     T_i = 2^k once the WEIGHT on it satisfies a_i >= 1
                                     and 2^{k_i} not dividing a_i (O12 Sect. 5.5).  The
                                     PLAIN sum of a mixed structure therefore only has
                                     `threshold_plain` = sum over the full-field words
                                     of (p - 1); `patterns` below is the weighted
                                     statement and `patterns_plain` the unweighted one.
                                     (E15 measured both on toy-193: layer 5, position 2,
                                     D = 209 -- balanced for every a < 47 = 256 - 209 and
                                     NOT balanced at a = 0 or a >= 47, on three keys.)
Proof: a monomial prod x_i^{a_i} survives the sum only if every a_i >= 2^{m_i}-1
(resp. a_i a positive multiple of p-1 / of 2^{k_i}); hence total degree >= T.
For a_i = 0 the factor is |U_i| = 0 in F_q (full field / affine subspace) or is
killed by the signed coset difference.

FULL-BLOCK structures (O11, R5).  `--full-block b` lets the whole 4-word block
b of the ciphertext run over F_q^4.  S^{-1} is a bijection of F_q^4, so the
substitution y = S^{-1}(x + k) makes those four words free variables: the first
S layer is FREE and the recurrence starts at (1,1,1,1) on that block, while the
threshold counts s = 4 * (#free blocks) + (#other active words).  Only a WHOLE
block qualifies (a lower-dimensional affine subspace is not mapped to one).

`--cipher yux` switches the S-box recurrence and the rotation sets to YuX
(tools/cipher_degree.py); the default `dux` reproduces every earlier number,
including the 39-cell `--grid`.

KNOWN-KEY ZERO-SUM PARTITIONS (R7 / W23, after Liu-Sun eprint 2026/1907 Sect. 6).
`--partition` places the intermediate state immediately AFTER a diffusion
layer and propagates in both directions: every backward group is
(L^{-1}, S^{-1}) -- a LEADING inverse diffusion layer, unlike the chosen-
ciphertext convention above, whose first layer is S^{-1} -- and every forward
group is (S, L).  The cosets of the coordinate subspace of the active set have
q^s texts, so the threshold is the full-field one, T = s (q - 1) in both
characteristics, and all sixteen words must clear it on both sides.  The tool
prints the largest split (r_b, r_f) and the class degree bounds at r_b, r_f
and one group further.  `--partition-align before-L` places the state between
the S layer and the diffusion layer instead (backward: S^{-1} first -- there a
full block IS free by O11 -- forward: L first).  `--partition-search s`
enumerates every active set of s words (modulo block rotation) and reports the
ones maximising r_b + r_f.

Usage:
    python tools/zero_sum_criterion.py --q 65537 --active 3,7,11 --layers 12
    python tools/zero_sum_criterion.py --q 2^16 --active 3,7 --dims 16,16
    python tools/zero_sum_criterion.py --q 65537 --active 3 --coset 15
    python tools/zero_sum_criterion.py --q 65537 --active 3,7 --coset 14,full --layers 10
    python tools/zero_sum_criterion.py --cipher yux --q 2^16 --active 0 --layers 11
    python tools/zero_sum_criterion.py --cipher yux --q 2^16 --full-block 0 --layers 11
    python tools/zero_sum_criterion.py --grid           # reproduce the S1 / E03 tables
    python tools/zero_sum_criterion.py --q 65537 --active 0 --partition        # 8 + 5 = 13
    python tools/zero_sum_criterion.py --q 2^8 --active 0,4 --partition        # 5 + 3 = 8
    python tools/zero_sum_criterion.py --cipher yux --q 65537 --partition-search 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cipher_degree import CIPHERS, profile as _profile_generic, profile_from  # noqa: E402

ROT_INV = (1, 4, 8, 9, 13)
# Encryption direction: L0 = sum of the rotations below (paper Sect. 3.2.2).
# Over F_p the circulant M0 has all 16 entries nonzero, so the layer is fully
# dense; over F_{2^n} it is the 11-element XOR sum, whose offsets already cover
# all four residues mod 4, so from the second layer on every word carries the
# same bound in both cases.
ROT_FWD_BIN = (0, 1, 3, 4, 7, 8, 9, 10, 12, 14, 15)
ROT_FWD_FP = tuple(range(16))


def sinv_deg(e, cipher="dux"):
    """Decryption S-box degree recurrence (tools/cipher_degree.py)."""
    return CIPHERS[cipher]["sinv_deg"](e)


def s_deg(e, cipher="dux"):
    """Encryption S-box degree recurrence.  DuX: S = (R^{-1})^4, output
    (c, b, a, y3) at positions 0..3, degrees (5,3,2,8).  YuX: S = Pf^{-4},
    output (z3, z2, z1, z0), degrees (8,5,3,2)."""
    return CIPHERS[cipher]["s_deg"](e)


def profile(active, layers, direction="dec", char2=False, cipher="dux",
            free_blocks=()):
    """Per-layer, per-word max-plus total-degree bounds after the S layer.

    `free_blocks` are whole 4-word blocks running over F_q^4 (O11): they skip
    the first S layer and enter the recurrence at degrees (1,1,1,1)."""
    return _profile_generic(active, layers, direction, char2, cipher, free_blocks)


def threshold(q, active, dims=None, coset=None, free_blocks=(), pointset=None):
    """T of Theorem O7.  Free blocks contribute four full-field words each.

    O15 (`pointset`, a list of point-set sizes n_i, one per active word): the
    divided-difference mask of a product set U_1 x ... x U_s kills every
    monomial of degree below T' = sum_i (|U_i| - 1), which is the threshold in
    BOTH characteristics and degenerates to the subspace / full-domain values
    when the point set is a subspace / the whole field."""
    if pointset is not None:
        assert not free_blocks, "a full block runs over F_q^4, not over a point set"
        assert not coset and not dims, "the point set already fixes the threshold"
        sizes = list(pointset)
        assert len(sizes) == len(active), "one point-set size per active word"
        return sum(n - 1 for n in sizes)
    nfree = 4 * len(set(free_blocks))
    s = len(active) + nfree
    if isinstance(q, str) and q.startswith("2^"):
        n = int(q[2:])
        dims = (list(dims) + [n] * nfree) if dims else [n] * s
        assert len(dims) == s, "one subspace dimension per active word"
        return sum(2 ** m - 1 for m in dims)
    p = int(q)
    if coset:
        ks = coset if isinstance(coset, list) else [coset] * len(active)
        assert not free_blocks, "a full block is a full field, not a coset"
        assert len(ks) == len(active), "one coset order per active word"
        # T2 (R9): a MIXED structure names the word types one by one -- an
        # entry None means "this word runs over all of F_p" and contributes
        # p - 1, an entry k means "a coset of the order-2^k subgroup" and
        # contributes 2^k.  Theorem 1's proof is per active word, so the
        # thresholds simply add.
        return sum((p - 1) if k is None else 2 ** k for k in ks)
    return s * (p - 1)


# ------------------------------------------------ the mixed-structure grid --
# T2 (R9).  The tightness grid `--grid` reproduces the published S1 / E03
# tables and is not touched; this is the same kind of table for the MIXED
# structures of T2 -- the "full-field word x multiplicative coset" column.  Two thresholds per cell, because an unsigned coset
# factor at exponent 0 is |H| != 0: `T` is the WEIGHTED statement (a_i >= 1 and
# 2^{k_i} not dividing a_i on every coset axis), `T_plain` the unweighted one.
# The measured counterpart is `experiments/E15_mixed_coset/zero_sum_mixed.py`,
# which sums the states directly on >= 2 random master keys.
MIXED_GRID = [
    # (p, active words, layer, the coset orders to try on the FIRST word)
    (65537, [3, 7], 10, [12, 13, 14, 16]),
    (257, [3, 7], 6, [5, 6, 7, 8]),
    (193, [3, 7], 5, [4, 5, 6]),
]


def run_mixed_grid(cipher="dux"):
    print(f"mixed structures (T2), {cipher}: one coset word x full-field words")
    print(f"{'p':>7} {'active':>9} {'layer':>5} {'coset':>6} {'T':>7} "
          f"{'T_plain':>7} {'data':>8}  weighted  plain   margins (weighted)")
    ok = True
    for p, active, layer, ks in MIXED_GRID:
        for k in ks:
            spec = [k] + [None] * (len(active) - 1)
            r = patterns(p, active, layer, None, spec, "dec", cipher)
            w = r["patterns"][layer - 1]
            pl = r.get("patterns_plain", [None] * layer)[layer - 1]
            print(f"{p:>7} {str(active):>9} {layer:>5} {'2^' + str(k):>6} "
                  f"{r['threshold']:>7} {str(r.get('threshold_plain')):>7} "
                  f"{'2^' + str(r['log2_data']):>8}  {w:>8}  {str(pl):>5}   "
                  f"{r['margin_per_position'][layer - 1]}")
            if pl is None:
                ok = False
    return ok


def parse_coset(spec, nactive):
    """`--coset` -> None, an int, or a per-word list with None for full words."""
    if spec is None or spec == "":
        return None
    parts = [t.strip().lower() for t in str(spec).split(",")]
    ks = [None if t in ("full", "none", "") else int(t) for t in parts]
    if len(ks) == 1 and nactive > 1:
        ks = ks * nactive
    if all(k is not None for k in ks) and len(set(ks)) == 1 and len(ks) == nactive:
        return ks[0] if nactive == 1 else ks
    return ks


def threshold_plain(q, active, coset):
    """The threshold of the UNWEIGHTED sum of a mixed / coset structure.

    An unsigned coset word contributes nothing at all to it (its factor at
    exponent 0 is |H| != 0), so only the full-field words count."""
    ks = coset if isinstance(coset, list) else [coset] * len(active)
    p = int(q)
    return sum((p - 1) for k in ks if k is None)


def patterns(q, active, layers=12, dims=None, coset=None, direction="dec",
             cipher="dux", free_blocks=(), pointset=None):
    free_blocks = tuple(sorted(set(free_blocks)))
    T = threshold(q, active, dims, coset, free_blocks, pointset)
    char2 = isinstance(q, str) and q.startswith("2^")
    prof = profile(active, layers, direction, char2, cipher, free_blocks)
    pats, worst = [], []
    for Y in prof:
        pats.append("".join("1" if all(Y[4 * b + p] < T for b in range(4)) else "0"
                            for p in range(4)))
        worst.append([max(Y[4 * b + p] for b in range(4)) for p in range(4)])
    lfull = 0
    for pat in pats:
        if pat == "1111":
            lfull += 1
        else:
            break
    nxt = pats[lfull] if lfull < len(pats) else "?"
    margin = [[T - w for w in row] for row in worst]
    qn = 2 ** int(str(q)[2:]) if str(q).startswith("2^") else int(q)
    s_words = len(active) + 4 * len(free_blocks)
    if pointset:
        log2_data = round(sum(math.log2(n) for n in pointset), 2)
    elif coset:
        ks = coset if isinstance(coset, list) else [coset] * len(active)
        log2_data = round(sum(math.log2(qn) if k is None else float(k) for k in ks), 2)
    else:
        log2_data = round(s_words * math.log2(qn), 2)
    extra = {}
    if coset and not str(q).startswith("2^"):
        ks = coset if isinstance(coset, list) else [coset] * len(active)
        if any(k is None for k in ks):
            Tp = threshold_plain(q, active, coset)
            extra["threshold_plain"] = Tp
            extra["patterns_plain"] = [
                "".join("1" if all(Y[4 * b + p_] < Tp for b in range(4)) else "0"
                        for p_ in range(4)) for Y in prof]
            extra["margin_plain_per_position"] = [[Tp - w for w in row]
                                                  for row in worst]
    return {**extra, "q": str(q), "cipher": cipher, "active": list(active),
            "free_blocks": list(free_blocks), "s": s_words,
            "log2_data": log2_data,
            "dims": dims, "coset": coset, "pointset": list(pointset) if pointset else None,
            "direction": direction, "threshold": T, "l_full": lfull, "next": nxt, "patterns": pats,
            "max_degree_per_position": worst, "margin_per_position": margin}



# ------------------------------------------------ known-key partitions ----
def _per_position(Y):
    return [max(Y[4 * b + p] for b in range(4)) for p in range(4)]


def _per_class(Y):
    """Liu-Sun's class-wise vector: the largest bound in each residue class
    of word indices modulo four (= our per-position vector)."""
    return _per_position(Y)


def _layers_below(prof, T):
    """Number of leading layers whose sixteen words all stay below T."""
    n = 0
    for Y in prof:
        if max(Y) < T:
            n += 1
        else:
            break
    return n


def partition(q, active, cipher="dux", free_blocks=(), align="after-L",
              max_groups=48):
    """Known-key zero-sum partition split for one active set (Liu-Sun Sect. 6).

    The intermediate state is a coset of the coordinate subspace V_A (the words
    of `active`, plus the four words of every block in `free_blocks`, run over
    F_q^s; the other words are constant).  Returns the largest r_b, r_f such
    that all sixteen words are balanced after r_b backward groups and after
    r_f forward groups, the class degree bounds there, and the bounds one
    group further (the first group at which the criterion guarantees nothing).

    align = "after-L"  (Liu-Sun): backward groups (L^{-1}, S^{-1}), forward (S, L).
    align = "before-L":            backward groups (S^{-1}, L^{-1}) -- the
                                   chosen-ciphertext convention, in which a
                                   full block passes the first S^{-1} for free
                                   (O11) -- and forward groups (L, S).
    """
    free_blocks = tuple(sorted(set(free_blocks)))
    active = tuple(sorted(set(active)))
    for a in active:
        assert a // 4 not in free_blocks, \
            f"word {a} lies in free block {a // 4}; list it as a free block only"
    char2 = isinstance(q, str) and q.startswith("2^")
    qn = 2 ** int(q[2:]) if char2 else int(q)
    s = len(active) + 4 * len(free_blocks)
    T = s * (qn - 1)
    D0 = [0] * 16
    for a in active:
        D0[a] = 1
    for b in free_blocks:
        for i in range(4):
            D0[4 * b + i] = 1
    if align == "after-L":
        back = profile_from(D0, max_groups, "dec", char2, cipher, leading_linear=True)
        fwd = profile_from(D0, max_groups, "enc", char2, cipher, leading_linear=False)
    elif align == "before-L":
        back = _profile_generic(active, max_groups, "dec", char2, cipher, free_blocks)
        fwd = profile_from(D0, max_groups, "enc", char2, cipher, leading_linear=True)
    else:
        raise ValueError(align)
    rb = _layers_below(back, T)
    rf = _layers_below(fwd, T)
    assert rb < max_groups and rf < max_groups, "raise max_groups"

    def at(prof, k):
        return _per_class(prof[k - 1]) if k >= 1 else [0, 0, 0, 0]
    return {"q": str(q), "cipher": cipher, "align": align,
            "active": list(active), "free_blocks": list(free_blocks), "s": s,
            "log2_data": round(s * math.log2(qn), 2), "threshold": T,
            "r_b": rb, "r_f": rf, "rounds": rb + rf,
            "backward_bounds": at(back, rb), "forward_bounds": at(fwd, rf),
            "backward_bounds_next": at(back, rb + 1),
            "forward_bounds_next": at(fwd, rf + 1),
            "backward_max_per_layer": [max(Y) for Y in back[:rb + 1]],
            "forward_max_per_layer": [max(Y) for Y in fwd[:rf + 1]]}


def _canonical(words):
    """Canonical representative of a word set modulo the four block rotations."""
    return min(tuple(sorted((w + 4 * k) % 16 for w in words)) for k in range(4))


def partition_search(q, s, cipher="dux", align="after-L", with_full_block=False):
    """All active sets of s words (modulo block rotation), sorted by r_b + r_f.
    With `with_full_block` the search also tries one whole block plus s - 4
    words (s >= 4)."""
    import itertools
    seen, rows = set(), []
    for act in itertools.combinations(range(16), s):
        c = _canonical(act)
        if c in seen:
            continue
        seen.add(c)
        rows.append(partition(q, c, cipher, (), align))
    if with_full_block and s >= 4:
        for act in itertools.combinations(range(4, 16), s - 4):
            r = partition(q, act, cipher, (0,), align)
            rows.append(r)
    rows.sort(key=lambda r: (-r["rounds"], -r["r_b"], r["active"]))
    return rows


GRID = [  # (label, q, active, dims/coset, observed l_full, observed next) -- measured in E03 and E04 (S1 run matrix)
    ("toy-2^4 s1", "2^4", (3,), [4], 2, "1101"), ("toy-2^4 s2", "2^4", (3, 7), [4, 4], 3, "0001"),
    ("toy-2^4 s3", "2^4", (3, 7, 11), [4] * 3, 3, "1101"), ("toy-2^4 s4", "2^4", (3, 7, 11, 15), [4] * 4, 4, "0000"),
    ("2^8 s1", "2^8", (3,), [8], 5, "0000"), ("2^8 s2", "2^8", (3, 7), [8, 8], 5, "1001"),
    ("2^8 s3", "2^8", (3, 7, 11), [8] * 3, 5, "1101"),
    ("2^8 s4 m4", "2^8", (3, 7, 11, 15), [4] * 4, 4, "0000"), ("2^8 s4 m5", "2^8", (3, 7, 11, 15), [5] * 4, 4, "1001"),
    ("2^8 s4 m6", "2^8", (3, 7, 11, 15), [6] * 4, 5, "0000"), ("2^8 s4 m7", "2^8", (3, 7, 11, 15), [7] * 4, 5, "1001"),
    ("2^8 s4 m8", "2^8", (3, 7, 11, 15), [8] * 4, 6, "0000"), ("2^8 s4 (2,6,10,14)", "2^8", (2, 6, 10, 14), [8] * 4, 6, "0000"),
    ("2^8 s8 m4 pairs", "2^8", (2, 3, 6, 7, 10, 11, 14, 15), [4] * 8, 4, "0000"),
    ("2^16 s1", "2^16", (3,), [16], 9, "0000"), ("2^16 s2 m12", "2^16", (3, 7), [12] * 2, 7, "1101"),
    ("2^16 s2 m13", "2^16", (3, 7), [13] * 2, 8, "0000"), ("2^16 s2 m14", "2^16", (3, 7), [14] * 2, 8, "1101"),
    ("2^16 s2 m15", "2^16", (3, 7), [15] * 2, 9, "0000"), ("2^16 s2 m16", "2^16", (3, 7), [16] * 2, 9, "1101"),
    ("2^16 s2 (2,3) same block", "2^16", (2, 3), [16] * 2, 9, "0000"),
    ("2^16 s3 m10", "2^16", (3, 7, 11), [10] * 3, 7, "0000"), ("2^16 s4 m8", "2^16", (3, 7, 11, 15), [8] * 4, 6, "0000"),
    # E03 (F_65537): full field, start position 3 vs 0; subgroup cosets k=10..15 (two-coset difference)
    ("65537 pos3", 65537, (3,), None, 9, "0000"), ("65537 pos0", 65537, (0,), None, 8, "1101"),
    ("65537 k10 pos3", 65537, (3,), "k10", 6, "0000"), ("65537 k11 pos3", 65537, (3,), "k11", 6, "1001"),
    ("65537 k12 pos3", 65537, (3,), "k12", 7, "0000"), ("65537 k13 pos3", 65537, (3,), "k13", 7, "1101"),
    ("65537 k14 pos3", 65537, (3,), "k14", 8, "0000"), ("65537 k15 pos3", 65537, (3,), "k15", 8, "1101"),
    ("65537 k10 pos0", 65537, (0,), "k10", 5, "1101"), ("65537 k11 pos0", 65537, (0,), "k11", 6, "0000"),
    ("65537 k12 pos0", 65537, (0,), "k12", 6, "1101"), ("65537 k13 pos0", 65537, (0,), "k13", 7, "0000"),
    ("65537 k14 pos0", 65537, (0,), "k14", 7, "1101"), ("65537 k15 pos0", 65537, (0,), "k15", 8, "0001"),
    ("toy-257 pos3", 257, (3,), None, 5, "0000"), ("toy-193 pos3", 193, (3,), None, 4, "1101"),
]


def run_grid():
    ok = 0
    for label, q, active, extra, ol, on in GRID:
        dims = extra if isinstance(extra, list) else None
        coset = int(extra[1:]) if isinstance(extra, str) else None
        r = patterns(q, active, 12, dims, coset)
        good = (r["l_full"] == ol and r["next"] == on)
        ok += good
        print(f"{'OK' if good else 'XX'}  {label:28s} T={r['threshold']:7d}  predicted {r['l_full']}+{r['next']}  observed {ol}+{on}")
    print(f"{ok}/{len(GRID)} observed cells reproduced")
    return ok == len(GRID)


PAPER_INSTANCES = [("65537", 65537), ("2^16", "2^16"), ("2^8", "2^8")]
TOY_INSTANCES = [("toy-193", 193), ("toy-257", 257)]


def active_set(start, s):
    """Canonical s-word structure with block position `start`: one word per
    block first (that is what keeps the profile lowest), then the next block
    position, and so on."""
    order = []
    for d in range(4):
        pos = (start + d) % 4
        order += [pos + 4 * b for b in range(4)]
    return tuple(sorted(order[:s]))


def paper_predictions(layers=12, direction="dec", starts=(2, 3), smax=4):
    """The prediction table used in the write-up: the three paper instances
    (and the two toys) x s active words x start block position."""
    out = []
    for label, q in PAPER_INSTANCES + TOY_INSTANCES:
        for start in starts:
            for s in range(1, smax + 1):
                active = active_set(start, s)
                r = patterns(q, active, layers, None, None, direction)
                r["label"] = label
                r["start_position"] = start
                r["s"] = s
                # data cost: log2 of the number of chosen ciphertexts
                qn = 2 ** int(str(q)[2:]) if str(q).startswith("2^") else int(q)
                r["log2_data"] = round(s * math.log2(qn), 2)
                out.append(r)
    return out


def min_active_for_layer(q, layer, max_s=12, dims=None, direction="dec", want=None):
    """Smallest number of active words that makes *some* block position balanced
    at `layer` (1-based).  Exhaustive over all subsets of the 16 words: with only
    4 blocks, the 5th active word has to share a block with an earlier one, which
    raises that block's profile -- so the answer is not simply `layer`-driven."""
    import itertools
    for s in range(1, max_s + 1):
        best = None
        for act in itertools.combinations(range(16), s):
            r = patterns(q, act, layer, dims, None, direction)
            pat = r["patterns"][layer - 1]
            hit = (pat == want) if want else (pat != "0000")
            if hit:
                cand = {"s": s, "active": list(act), "pattern": pat,
                        "threshold": r["threshold"],
                        "margin": r["margin_per_position"][layer - 1]}
                if best is None:
                    best = cand
                if not want:
                    return cand
        if best:
            return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="dux", choices=tuple(CIPHERS),
                    help="which S-box recurrence and rotation sets to use")
    ap.add_argument("--full-block", default=None,
                    help="comma list of whole blocks running over F_q^4 (O11)")
    ap.add_argument("--q", default="65537", help="65537 | 257 | 193 | 2^8 | 2^16 | 2^4")
    ap.add_argument("--active", default="3", help="comma list of active ciphertext words")
    ap.add_argument("--dims", default=None, help="per-word subspace dims (char 2)")
    ap.add_argument("--coset", default=None,
                    help="F_p: order-2^k subgroup cosets (signed).  One k for "
                         "every active word, or a comma list with one entry per "
                         "active word -- an entry 'full' (or 'none') makes that "
                         "word run over all of F_p, which is the MIXED structure "
                         "of T2 (R9), e.g. --active 3,7 --coset 14,full")
    ap.add_argument("--pointset", default=None,
                    help="O15 (R8): the active words run over point sets of these "
                         "sizes (one per active word, or one size for all), with "
                         "the divided-difference mask of tools/interp_mask.py; "
                         "the threshold becomes T' = sum_i (n_i - 1) in EITHER "
                         "characteristic")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--direction", default="dec", choices=("dec", "enc"),
                    help="dec: chosen ciphertext (S^{-1}, sparse L^{-1}); "
                         "enc: chosen plaintext / designer's CPA model "
                         "(S, dense L)")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--mixed-grid", action="store_true",
                    help="T2 (R9): the tightness grid for MIXED structures "
                         "(one coset word times full-field words), with both "
                         "thresholds per cell")
    ap.add_argument("--want", default=None,
                    help="with --min-words-for-layer: require this exact 4-bit pattern")
    ap.add_argument("--min-words-for-layer", type=int, default=None,
                    help="search the smallest active-word count giving any balanced position at this layer")
    ap.add_argument("--partition", action="store_true",
                    help="known-key zero-sum partition split (r_b, r_f) for --active/--full-block")
    ap.add_argument("--partition-align", default="after-L", choices=("after-L", "before-L"),
                    help="where the intermediate state sits: after the diffusion layer "
                         "(Liu-Sun, backward groups start with L^{-1}) or before it")
    ap.add_argument("--partition-search", type=int, default=None,
                    help="enumerate every active set of this many words and rank by r_b + r_f")
    ap.add_argument("--with-full-block", action="store_true",
                    help="with --partition-search: also try one whole block plus s - 4 words")
    ap.add_argument("--top", type=int, default=12, help="with --partition-search: rows to print")
    ap.add_argument("--json", default=None)
    ap.add_argument("--predictions-json", default=None,
                    help="write the full paper prediction table (3 instances + 2 toys)")
    ap.add_argument("--starts", default=None,
                    help="with --predictions-json: comma list of start block positions")
    ap.add_argument("--smax", type=int, default=4,
                    help="with --predictions-json: largest number of active words")
    a = ap.parse_args()
    if a.grid:
        raise SystemExit(0 if run_grid() else 1)
    if a.mixed_grid:
        raise SystemExit(0 if run_mixed_grid(a.cipher) else 1)
    if a.min_words_for_layer:
        q = a.q if a.q.startswith("2^") else int(a.q)
        r = min_active_for_layer(q, a.min_words_for_layer, direction=a.direction,
                                 want=a.want)
        print(f"q={a.q} layer {a.min_words_for_layer}: {r}")
        raise SystemExit(0)
    if a.predictions_json:
        starts = tuple(int(v) for v in a.starts.split(",")) if a.starts else (2, 3)
        rows = paper_predictions(a.layers, a.direction, starts, a.smax)
        json.dump(rows, open(a.predictions_json, "w"), indent=1)
        print(f"{len(rows)} predictions -> {a.predictions_json}")
        for r in rows:
            print(f"{r['label']:8s} start {r['start_position']} s={r['s']} "
                  f"active={r['active']} T={r['threshold']:8d} data=2^{r['log2_data']:<5} "
                  f"l_full={r['l_full']:2d} next={r['next']}")
        raise SystemExit(0)
    free = [int(v) for v in a.full_block.split(",")] if a.full_block else []
    active = [int(v) for v in a.active.split(",") if v != ""] if a.active else []
    if free and a.active == ap.get_default("active"):
        active = []                     # --full-block alone means "no extra words"
    dims = [int(v) for v in a.dims.split(",")] if a.dims else None
    q = a.q if a.q.startswith("2^") else int(a.q)
    if a.partition_search is not None:
        rows = partition_search(q, a.partition_search, a.cipher, a.partition_align,
                                a.with_full_block)
        print(f"{a.cipher} q={a.q} s={a.partition_search} align={a.partition_align}: "
              f"{len(rows)} active sets modulo block rotation, best first")
        for r in rows[:a.top]:
            fb = f" +block{r['free_blocks']}" if r["free_blocks"] else ""
            print(f"  {r['rounds']:2d} = {r['r_b']} + {r['r_f']}  active={r['active']}{fb}  "
                  f"T={r['threshold']}  back={r['backward_bounds']}  fwd={r['forward_bounds']}")
        if a.json:
            json.dump(rows, open(a.json, "w"), indent=1)
        raise SystemExit(0)
    if a.partition:
        r = partition(q, active, a.cipher, free, a.partition_align)
        print(f"{a.cipher} q={a.q} active={r['active']} free_blocks={r['free_blocks']} "
              f"s={r['s']} (2^{r['log2_data']} texts per coset), T = {r['threshold']}, "
              f"align={r['align']}")
        print(f"  backward: r_b = {r['r_b']} groups, class bounds {r['backward_bounds']}; "
              f"group {r['r_b'] + 1} would give {r['backward_bounds_next']}")
        print(f"  forward:  r_f = {r['r_f']} groups, class bounds {r['forward_bounds']}; "
              f"group {r['r_f'] + 1} would give {r['forward_bounds_next']}")
        print(f"  zero-sum partition over r_b + r_f = {r['rounds']} rounds")
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)
        raise SystemExit(0)
    pointset = None
    if a.pointset:
        sizes = [int(v) for v in str(a.pointset).split(",")]
        pointset = sizes if len(sizes) > 1 else sizes * len(active)
    r = patterns(q, active, a.layers, dims, parse_coset(a.coset, len(active)),
                 a.direction, a.cipher, free, pointset)
    if free:
        print(f"free blocks {free} (first S layer is free, O11); "
              f"s = {r['s']} active field variables, data 2^{r['log2_data']}")
    for l, (pat, w) in enumerate(zip(r["patterns"], r["max_degree_per_position"]), 1):
        line = (f"layer {l:2d}: {pat}  max total degree per position {w}  "
                f"(T = {r['threshold']})")
        if "patterns_plain" in r:
            line += f"   plain: {r['patterns_plain'][l - 1]} (T = {r['threshold_plain']})"
        print(line)
    print(f"l_full = {r['l_full']}, next = {r['next']}")
    if "patterns_plain" in r:
        print("  (mixed structure: the T = sum_i T_i patterns are the WEIGHTED "
              "statement, a_i >= 1 and 2^k_i not dividing a_i on every coset "
              "axis; the plain column is the unweighted one)")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
