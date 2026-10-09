"""DuX key schedule (paper Sect. 4.3, Fig. 3, Algorithms 1-2).

Rf(X, rc) with X = (x0,x1,x2,x3), x_i in F_q^4, rc in F_q^4:
    y0 = S(x0) + x2 + rc
    y1 = S(x1) + x0 + x3 + rc
    y2 = S(x1) + x3 + 2*x0 + rc      (2*x0 = 0 in characteristic 2!  -> O5)
    y3 = x1
Each round key = 4 iterations of Rf on the previous round key with the
four 4-word sub-constants rc^{i-1}_{0..3}, rc^{i-1}_{4..7}, ...

Round constants: for round i and j = 0..3, t = 16 i + 4 j and
    (c0,c1,c2,c3) = S(t+1, t+2, t+3, t+4)
with S the ENCRYPTION S-box and t+k read as field elements (from_int).
"""
from .sbox import S


def round_constants(F, alpha, rounds):
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


def Rf(F, alpha, X, rc):
    x0, x1, x2, x3 = (tuple(X[4 * k:4 * k + 4]) for k in range(4))
    s0 = S(F, x0, alpha)
    s1 = S(F, x1, alpha)
    two_x0 = tuple(F.add(v, v) for v in x0)
    y0 = _blk_add(F, s0, x2, rc)
    y1 = _blk_add(F, s1, x0, x3, rc)
    y2 = _blk_add(F, s1, x3, two_x0, rc)
    y3 = x1
    return y0 + y1 + y2 + y3


def key_expand(F, alpha, K, rounds):
    """Return [rk^0, ..., rk^rounds] (rounds+1 round keys of 16 words)."""
    assert len(K) == 16
    rcs = round_constants(F, alpha, rounds)
    rks = [tuple(K)]
    for i in range(1, rounds + 1):
        st = rks[i - 1]
        rc = rcs[i - 1]
        for j in range(4):
            st = Rf(F, alpha, st, rc[4 * j:4 * j + 4])
        rks.append(st)
    return rks


def Rf_inv(F, alpha, Y, rc):
    """Inverse of Rf (the key schedule is invertible; recovering any round
    key recovers the master key)."""
    y0, y1, y2, y3 = (tuple(Y[4 * k:4 * k + 4]) for k in range(4))
    x1 = y3
    # y1 - y2 = x0 - 2 x0 = -x0  =>  x0 = y2 - y1
    x0 = tuple(F.sub(a, b) for a, b in zip(y2, y1))
    s1 = S(F, x1, alpha)
    x3 = tuple(F.sub(F.sub(F.sub(a, b), c), d) for a, b, c, d in zip(y1, s1, x0, rc))
    s0 = S(F, x0, alpha)
    x2 = tuple(F.sub(F.sub(a, b), c) for a, b, c in zip(y0, s0, rc))
    return x0 + x1 + x2 + x3
