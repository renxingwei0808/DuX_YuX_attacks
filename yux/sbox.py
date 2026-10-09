"""YuX S-box (paper Sect. III, Construction 1 + Fig. 1; Sect. IV-A.2).

    Pf(x0,x1,x2,x3)      = (x1, x2, x3, x0 + x1 x2 + x3 + alpha)
    Pf^{-1}(x0,x1,x2,x3) = (x3 - x0 x1 - x2 - alpha, x0, x1, x2)
    S^{-1} = Pf^4        (used in DECRYPTION; degrees (2,2,3,4))
    S      = Pf^{-4}     (used in ENCRYPTION; degrees (8,5,3,2))

These definitions are the ground truth.  The explicit quadratic system printed
in Sect. VI-D ("y3 = x0 x1 + x2 + alpha - x3", ...) equals Pf^{-4} only in
characteristic 2; over F_p every one of its four lines differs from the chain
below by signs (see docs/yux_specification.md, erratum E1, and
tests/test_yux.py::test_paper_VI_D_system_is_a_sign_typo_over_Fp).

Closed form of S^{-1} = Pf^4 (obtained by iterating Pf; Pf^k moves the input
words left and appends one new word, so the four new words are exactly):
    y0 = x0 + x1 x2 + x3 + alpha        (deg 2)
    y1 = x1 + x2 x3 + y0 + alpha        (deg 2)
    y2 = x2 + x3 y0 + y1 + alpha        (deg 3)
    y3 = x3 + y0 y1 + y2 + alpha        (deg 4)

Closed form of S = Pf^{-4} (same argument on the other side):
    z0 = x3 - x0 x1 - x2 - alpha        (deg 2)
    z1 = x2 - z0 x0 - x1 - alpha        (deg 3)
    z2 = x1 - z1 z0 - x0 - alpha        (deg 5)
    z3 = x0 - z2 z1 - z0 - alpha        (deg 8)
    S(x) = (z3, z2, z1, z0)             -- degrees (8,5,3,2) at positions 0..3

Compare dux/sbox.py: DuX's S^{-1} = R^4 has degrees (2,3,4,2) and its S has
(5,3,2,8).  Over F_{2^n} the two S-box families are word-affine equivalent
(DuX paper, Appendix B); over F_p they are not.
"""


def Pf(F, x, alpha):
    x0, x1, x2, x3 = x
    return (x1, x2, x3, F.add(F.add(F.add(x0, F.mul(x1, x2)), x3), alpha))


def Pf_inv(F, y, alpha):
    y0, y1, y2, y3 = y
    return (F.sub(F.sub(F.sub(y3, F.mul(y0, y1)), y2), alpha), y0, y1, y2)


def S_inv(F, x, alpha):
    """Decryption S-box S^{-1} = Pf^4 (iterated definition)."""
    for _ in range(4):
        x = Pf(F, x, alpha)
    return x


def S(F, x, alpha):
    """Encryption S-box S = Pf^{-4} (iterated definition)."""
    for _ in range(4):
        x = Pf_inv(F, x, alpha)
    return x


def S_inv_closed(F, x, alpha):
    """Closed form of S^{-1} = Pf^4; must equal S_inv (tested)."""
    x0, x1, x2, x3 = x
    y0 = F.add(F.add(F.add(x0, F.mul(x1, x2)), x3), alpha)
    y1 = F.add(F.add(F.add(x1, F.mul(x2, x3)), y0), alpha)
    y2 = F.add(F.add(F.add(x2, F.mul(x3, y0)), y1), alpha)
    y3 = F.add(F.add(F.add(x3, F.mul(y0, y1)), y2), alpha)
    return (y0, y1, y2, y3)


def S_closed(F, x, alpha):
    """Closed form of S = Pf^{-4}; must equal S (tested)."""
    x0, x1, x2, x3 = x
    z0 = F.sub(F.sub(F.sub(x3, F.mul(x0, x1)), x2), alpha)
    z1 = F.sub(F.sub(F.sub(x2, F.mul(z0, x0)), x1), alpha)
    z2 = F.sub(F.sub(F.sub(x1, F.mul(z1, z0)), x0), alpha)
    z3 = F.sub(F.sub(F.sub(x0, F.mul(z2, z1)), z0), alpha)
    return (z3, z2, z1, z0)


def paper_VI_D_system(F, x, alpha):
    """The four quadratic equations printed in Sect. VI-D, taken literally:
        y3 = x0 x1 + x2 + alpha - x3
        y2 = y3 x0 + x2 + alpha - x1
        y1 = y2 y3 + x0 + alpha - x1
        y0 = y1 y2 + y3 + alpha - x0
    Returned as (y0, y1, y2, y3).  In characteristic 2 this is exactly S(x)
    (erratum E1); over F_p it is not, which is why the reference model uses the
    Construction-1 chain instead.  Kept here only so the test suite can state
    the discrepancy precisely."""
    x0, x1, x2, x3 = x
    y3 = F.sub(F.add(F.add(F.mul(x0, x1), x2), alpha), x3)
    y2 = F.sub(F.add(F.add(F.mul(y3, x0), x2), alpha), x1)
    y1 = F.sub(F.add(F.add(F.mul(y2, y3), x0), alpha), x1)
    y0 = F.sub(F.add(F.add(F.mul(y1, y2), y3), alpha), x0)
    return (y0, y1, y2, y3)


# --- vectorised versions (numpy arrays per word) -----------------------------

def vPf(F, x, alpha):
    x0, x1, x2, x3 = x
    return (x1, x2, x3, F.vadd(F.vadd(F.vadd(x0, F.vmul(x1, x2)), x3), alpha))


def vPf_inv(F, y, alpha):
    y0, y1, y2, y3 = y
    return (F.vsub(F.vsub(F.vsub(y3, F.vmul(y0, y1)), y2), alpha), y0, y1, y2)


def vS_inv(F, x, alpha):
    for _ in range(4):
        x = vPf(F, x, alpha)
    return x


def vS(F, x, alpha):
    for _ in range(4):
        x = vPf_inv(F, x, alpha)
    return x
