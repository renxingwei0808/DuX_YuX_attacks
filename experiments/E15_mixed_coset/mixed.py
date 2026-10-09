"""E15 / W29 -- MIXED structures over F_p: one full-field word times one
multiplicative coset (R9 memo Sect. 1, T2).

THE CRITERION.  Theorem 1's proof is per active word: a monomial prod_i x_i^{e_i}
survives the structure sum only if EVERY factor sum_{x_i in U_i} x_i^{e_i + a_i}
is nonzero, and each factor's condition is a statement about that word alone.
So a structure whose words are of DIFFERENT kinds simply adds the thresholds,

    U = F_p                      ->  the factor is nonzero iff (p-1) | (e+a)
                                     AND e + a >= p-1   (e + a = 0 gives p = 0),
                                     so it costs  T_i = p - 1 ;
    U = g H, |H| = 2^k           ->  the factor is g^{e+a} |H| [ 2^k | (e+a) ],
                                     nonzero iff 2^k | (e + a); e + a = 0 is NOT
                                     killed (|H| = 2^k != 0 mod p), so the WEIGHT
                                     must satisfy a >= 1, and then e + a >= 2^k,
                                     costing  T_i = 2^k .

    ==>  T  =  sum_i T_i  and, for every weight a with a_i >= 1 on each coset
         axis,  sum_{x in U} x^a Z(x) = 0  as soon as  |a| < T - D .

That is the same statement Theorem 4 makes for full-field structures, with the
same proof, so Lemma 1 (the pure-key terms drop out) and the whole r_KR = 2
machinery carry over verbatim: the full-field word kills the constant term by
sum_{x in F_p} 1 = p = 0, and on a coset axis the requirement a_i >= 1 does it.
(2^{k_i} must also not divide a_i -- otherwise e_i = 0 is admissible and the
monomial survives at total degree p - 1 only.  Both conditions are what
`weighted.plan(coset=[k, None])` filters on.)

WHY IT PAYS.  12-round DuX(65537) needs only position 3 of layer 10 balanced
(D_3 = 70 226) because the O10 combined row of `0001` reads position 3 alone.
Dropping one of the two full words to a coset of order 2^k costs 65 536 - 2^k
of threshold and 16 - k bits of data:

    k       T        margin      data
    2^12    69 632   -594        (unusable)
    2^13    73 728    3 502      2^29   (needs two-axis weights)
    2^14    81 920   11 694      2^30   (one-axis weights suffice)
    2^16   131 072   60 846      2^32   (the current run)

The axis cap of W25 is what decides between the last two: the weights live on
the coset axis, which has 2^k points, so at most 2^k of them are independent,
and the attack needs N_w = 8 608.

Usage
    python -m mixed --p 193 --k 6            # the coset a*H itself
    python run_toy.py --instance toy-193 ...  # the end-to-end toy
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
for p_ in (ROOT, os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "experiments", "E07_key_recovery_2round")):
    sys.path.insert(0, p_)


def subgroup_coset(p, k, rng=None, rep=None):
    """One coset a*H of the order-2^k subgroup H of F_p^*, sorted.

    Same construction as `experiments/E03_zero_sum_Fp/run.py` and
    `experiments/Y03_zero_sum/run.py` (a primitive root, its (p-1)/2^k-th
    power, and a random representative); sorted so that the structure's axis
    order is deterministic given the seed."""
    assert (p - 1) % (1 << k) == 0, f"2^{k} does not divide {p} - 1"
    from dux.field import _prime_factors
    fs = _prime_factors(p - 1)
    g = next(x for x in range(2, p)
             if all(pow(x, (p - 1) // f, p) != 1 for f in fs))
    h = pow(g, (p - 1) >> k, p)
    H = np.empty(1 << k, dtype=np.int64)
    v = 1
    for i in range(1 << k):
        H[i] = v
        v = (v * h) % p
    a = rep if rep is not None else (int(rng.integers(1, p)) if rng else 1)
    return np.sort((H * a) % p)


def parse_mixed(spec, nactive):
    """'14,full' -> [14, None]; 'full,13' -> [None, 13]; '14' -> [14] * s."""
    parts = [t.strip().lower() for t in str(spec).split(",")]
    ks = [None if t in ("full", "none", "") else int(t) for t in parts]
    if len(ks) == 1 and nactive > 1:
        ks = ks * nactive
    assert len(ks) == nactive, "one entry per active word"
    return ks


def mixed_axes(F, ks, seed):
    """The value set of every active word, in the order `structure_data` wants.

    A None entry gives the whole of F_p; an entry k gives one coset of the
    order-2^k subgroup.  The COSET word must be axis 0 (the slowest axis), since
    the slice decomposition of O12 weights along that axis and T2 puts the
    weights on the coset word."""
    rng = np.random.default_rng(seed)
    out = []
    for j, k in enumerate(ks):
        if k is None:
            out.append(np.arange(F.q, dtype=np.int64))
        else:
            assert j == 0 or all(kk is not None for kk in ks[:j]), \
                "the coset words must come first (the weight axis is the slowest)"
            out.append(subgroup_coset(F.p, k, rng))
    return out


def log2_data(F, ks):
    return float(sum(np.log2(F.q) if k is None else k for k in ks))


def weight_moduli(F, ks):
    """The modulus of each weight axis: on a coset of order 2^k the map
    x -> x^a only depends on `a mod 2^k` up to the constant factor g^{2^k},
    and on the whole field on `a mod (p-1)` (x = 0 gives 0 either way for
    a >= 1).  Two weights whose exponents agree modulo these give
    PROPORTIONAL rows, hence no new rank."""
    return [(1 << k) if k is not None else (F.q - 1) for k in ks]


def independent_weights(vectors, moduli):
    """How many of `vectors` are pairwise non-proportional, i.e. how many rows
    a single structure can really contribute.

    `weighted.plan`'s `usable_weights` counts the ADMISSIBLE exponents and caps
    them by W25's axis bound; that is an upper bound, and it is loose exactly
    when a weight margin exceeds a coset order, because then the admissible
    exponents wrap around the residue.  On toy-193 (coset 2^6, margin 159) the
    first 8300 vectors have only 6174 distinct residues -- and 6174 is exactly
    the rank the attack measures.  At the three S20 targets the margin is below
    the coset order (11694 < 2^14, 3502 < 2^13, 13951 < 2^15), so no two
    admissible exponents collide and the bound is attained."""
    return len({tuple(int(a[i]) % int(m) for i, m in enumerate(moduli))
                for a in vectors})


def residues_collide(margin, ks):
    """True when some coset axis is shorter than the weight margin, i.e. when
    `usable_weights` overstates the rows one structure yields."""
    return any(k is not None and margin > (1 << k) for k in ks)
