"""DuX S-box (paper Sect. 3.1.1).

    R(x0,x1,x2,x3)      = (x1, x2, x0*x1 + x3 + alpha, x0)
    R^{-1}(x0,x1,x2,x3) = (x3, x0, x1, x2 - x0*x3 - alpha)
    S^{-1} = R^4      (used in DECRYPTION; cheap, degrees (2,3,4,2))
    S      = (R^{-1})^4 (used in ENCRYPTION; degrees (5,3,2,8))

These definitions are the ground truth.  The explicit polynomial forms
printed in the paper (Sect. 3.1.1, 5.3, 5.5.1) disagree with each other
in signs / alpha terms, so we never use them directly; instead
tests/test_sbox.py checks the closed forms derived here against R^4.

Closed form of S^{-1} (derived by iterating R, see docs/glossary.md):
    f  = x0*x1 + x3 + alpha
    g  = x1*x2 + x0 + alpha
    y0 = g
    y1 = x2*f + x1 + alpha
    y2 = f*g + x2 + alpha
    y3 = f
"""


def R(F, x, alpha):
    x0, x1, x2, x3 = x
    return (x1, x2, F.add(F.add(F.mul(x0, x1), x3), alpha), x0)


def R_inv(F, y, alpha):
    y0, y1, y2, y3 = y
    return (y3, y0, y1, F.sub(F.sub(y2, F.mul(y0, y3)), alpha))


def S_inv(F, x, alpha):
    for _ in range(4):
        x = R(F, x, alpha)
    return x


def S(F, x, alpha):
    for _ in range(4):
        x = R_inv(F, x, alpha)
    return x


def S_inv_closed(F, x, alpha):
    """Closed form of S^{-1}; must equal S_inv (tested)."""
    x0, x1, x2, x3 = x
    f = F.add(F.add(F.mul(x0, x1), x3), alpha)
    g = F.add(F.add(F.mul(x1, x2), x0), alpha)
    y1 = F.add(F.add(F.mul(x2, f), x1), alpha)
    y2 = F.add(F.add(F.mul(f, g), x2), alpha)
    return (g, y1, y2, f)


def S_closed(F, x, alpha):
    """Closed form of S = (R^{-1})^4; must equal S (tested).

    a  = x2 - x0*x3 - alpha           (deg 2)   -> y2
    b  = x1 - x3*a  - alpha           (deg 3)   -> y1
    c  = x0 - a*b   - alpha           (deg 5)   -> y0
    y3 = x3 - b*c   - alpha           (deg 8)
    """
    x0, x1, x2, x3 = x
    a = F.sub(F.sub(x2, F.mul(x0, x3)), alpha)
    b = F.sub(F.sub(x1, F.mul(x3, a)), alpha)
    c = F.sub(F.sub(x0, F.mul(a, b)), alpha)
    y3 = F.sub(F.sub(x3, F.mul(b, c)), alpha)
    return (c, b, a, y3)


# --- vectorised versions (numpy arrays per word) -----------------------------

def vR(F, x, alpha):
    x0, x1, x2, x3 = x
    return (x1, x2, F.vadd(F.vadd(F.vmul(x0, x1), x3), alpha), x0)


def vR_inv(F, y, alpha):
    y0, y1, y2, y3 = y
    return (y3, y0, y1, F.vsub(F.vsub(y2, F.vmul(y0, y3)), alpha))


def vS_inv(F, x, alpha):
    for _ in range(4):
        x = vR(F, x, alpha)
    return x


def vS(F, x, alpha):
    for _ in range(4):
        x = vR_inv(F, x, alpha)
    return x
