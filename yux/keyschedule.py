"""YuX key schedule (paper Algorithm 1) and round constants (Algorithm 2).

Algorithm 1 (verbatim):
    1: rk^0 = Key
    2: X_i = (key_i, key_{i+1}, key_{i+2}, key_{i+3}), i in {0,...,3}
    3: for i = 1..r:
    4:     rk^i = ()
    5:     for j = 0..3:
    6:         X4 = X0 + S((X1 + X2 + X3) <<< 3) + (rc^{i-1}_{4j},...,rc^{i-1}_{4j+3})
    7:         rk^i = rk^i || X4
    8:         X0 = X1, X1 = X2, X2 = X3, X3 = X4

Four NLFSR branches of 4 words each; "<<< 3" is a rotation of the four words of
a block with the state convention (X <<< 3)_i = X_{(i+3) mod 4} (footnote 2).

AMBIGUITY (docs/yux_specification.md, ambiguity A1).  Line 2 taken literally
gives four SLIDING windows, so the NLFSR is seeded by key_0..key_6 only and
rk^1..rk^r carry just 7 words of entropy.  Sect. VI-D of the same paper writes
RK_{0,i} = [k_{4i}, k_{4i+1}, k_{4i+2}, k_{4i+3}], the NON-OVERLAPPING reading,
under which the seed state IS rk^0.  The reference model implements the
non-overlapping reading (`ks_literal=False`, default) and offers the literal
one; both are invertible as state recursions.  The attacks in this repository
recover rk^0 = Key directly, so neither reading affects their results.

Round constants (Algorithm 2): for i = 0..r-1 and j = 0..3, id = 16 i + 4 j and
(c0,c1,c2,c3) = S(id+1, id+2, id+3, id+4) with S the ENCRYPTION S-box and the
integers mapped into F_q by `F.from_int` (the paper does not state the map;
dux/keyschedule.py makes the same choice -- ambiguity A2).
"""
from .sbox import S


def rot3(blk):
    """(X <<< 3)_i = X_{(i+3) mod 4} on one 4-word block."""
    return tuple(blk[(i + 3) % 4] for i in range(4))


def round_constants(F, alpha, rounds):
    """[rc^0, ..., rc^{rounds-1}], each 16 words (Algorithm 2)."""
    rcs = []
    for i in range(rounds):
        rc = []
        for j in range(4):
            t = 16 * i + 4 * j
            rc.extend(S(F, tuple(F.from_int(t + k) for k in (1, 2, 3, 4)), alpha))
        rcs.append(tuple(rc))
    return rcs


def _blk_add(F, *blocks):
    out = list(blocks[0])
    for b in blocks[1:]:
        out = [F.add(u, v) for u, v in zip(out, b)]
    return tuple(out)


def seed_state(K, ks_literal=False):
    """(X0, X1, X2, X3) of Algorithm 1 line 2.

    ks_literal=False (default, Sect. VI-D reading): X_i = key_{4i..4i+3}.
    ks_literal=True  (Algorithm 1 read verbatim): X_i = key_{i..i+3}."""
    if ks_literal:
        return tuple(tuple(K[i + t] for t in range(4)) for i in range(4))
    return tuple(tuple(K[4 * i + t] for t in range(4)) for i in range(4))


def ks_round(F, alpha, state, rc):
    """One outer iteration (Algorithm 1 lines 4-9): state -> (rk^i, new state).

    The four blocks produced in this round ARE the new state, so `rk^i` and the
    state coincide; the return value keeps them separate only for clarity."""
    X = list(state)
    out = []
    for j in range(4):
        X4 = _blk_add(F, X[0], S(F, rot3(_blk_add(F, X[1], X[2], X[3])), alpha),
                      tuple(rc[4 * j:4 * j + 4]))
        out.append(X4)
        X = [X[1], X[2], X[3], X4]
    return tuple(w for blk in out for w in blk), tuple(X)


def ks_round_inverse(F, alpha, rk_i, rc):
    """Inverse of `ks_round`: the state that produced rk^i.

    With Y_j the four blocks of rk^i the recursion unrolls backwards without
    ever inverting S:
        X3 = Y3 - S((Y0+Y1+Y2) <<< 3) - rc_3
        X2 = Y2 - S((X3+Y0+Y1) <<< 3) - rc_2
        X1 = Y1 - S((X2+X3+Y0) <<< 3) - rc_1
        X0 = Y0 - S((X1+X2+X3) <<< 3) - rc_0
    """
    Y = [tuple(rk_i[4 * j:4 * j + 4]) for j in range(4)]
    X = [None] * 4
    for j in (3, 2, 1, 0):
        prev = [X[t] for t in range(j + 1, 4)] + [Y[t] for t in range(0, j)]
        s = S(F, rot3(_blk_add(F, *prev)), alpha)
        X[j] = tuple(F.sub(F.sub(a, b), c)
                     for a, b, c in zip(Y[j], s, rc[4 * j:4 * j + 4]))
    return tuple(X)


def key_expand(F, alpha, K, rounds, ks_literal=False):
    """Return [rk^0, ..., rk^rounds] (rounds+1 round keys of 16 words)."""
    assert len(K) == 16
    rcs = round_constants(F, alpha, rounds)
    state = seed_state(tuple(K), ks_literal)
    rks = [tuple(K)]
    for i in range(1, rounds + 1):
        rk, state = ks_round(F, alpha, state, rcs[i - 1])
        rks.append(rk)
    return rks


def ks_inverse(F, alpha, rk_i, i, rounds, ks_literal=False):
    """rk^{i-1} from rk^i (i >= 1), i.e. the key schedule run backwards.

    For i >= 2 the state before round i is rk^{i-1} in block form, so this is
    exact.  For i = 1 the state before round 1 is the SEED state, which equals
    rk^0 only in the non-overlapping reading; with `ks_literal=True` the return
    value is the seed state (key_0..key_6 in four sliding windows) laid out as
    16 words, not rk^0."""
    assert 1 <= i <= rounds
    rcs = round_constants(F, alpha, rounds)
    state = ks_round_inverse(F, alpha, tuple(rk_i), rcs[i - 1])
    return tuple(w for blk in state for w in blk)
