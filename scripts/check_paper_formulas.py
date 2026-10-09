"""Symbolically compare the polynomial formulas PRINTED in the DuX paper
with the ones FORCED by its own definitions (Sect. 3.1.1: R, S^-1 = R^4,
S = (R^-1)^4).

Definitions are the ground truth; every mismatch printed here is a typo in the
paper and is listed in docs/dux_specification.md (errata); the output of
this script is results/W1_spec_audit/paper_formula_check.txt.

Usage: python scripts/check_paper_formulas.py
"""
import sympy as sp

x0, x1, x2, x3, al = sp.symbols("x0 x1 x2 x3 alpha")


def R(v):
    a, b, c, d = v
    return (b, c, sp.expand(a * b + d + al), a)


def R_inv(v):
    a, b, c, d = v
    return (d, a, b, sp.expand(c - a * d - al))


def iterate(f, v, k):
    for _ in range(k):
        v = f(v)
    return tuple(sp.expand(t) for t in v)


def report(label, printed, true):
    d = sp.simplify(sp.expand(printed - true))
    tag = "OK " if d == 0 else "TYPO"
    print(f"  [{tag}] {label:38s} printed - definition = {d}")
    return d == 0


def main():
    X = (x0, x1, x2, x3)
    R2 = iterate(R, X, 2)
    S_inv = iterate(R, X, 4)          # decryption S-box
    Rm2 = iterate(R_inv, X, 2)
    S_ = iterate(R_inv, X, 4)         # encryption S-box

    print("Sect. 3.1.1  R^2 (printed):")
    for i, pr in enumerate([x2, x0 * x1 + x3 + al, x1 * x2 + x0 + al, x1]):
        report(f"R^2 component {i}", pr, R2[i])

    print("\nSect. 3.1.1  R^-2 (printed):")
    a_ = x2 - x0 * x3 - al
    for i, pr in enumerate([a_, x3, x0, x1 - x3 * a_]):
        report(f"R^-2 component {i}", pr, Rm2[i])

    print("\nSect. 3.1.1  S^-1 = R^4 closed form (printed):")
    f = x0 * x1 + x3 + al
    g = x1 * x2 + x0 + al
    for i, pr in enumerate([g, x2 * f + x1 + al, f * g + x2 + al, f]):
        report(f"S^-1 y{i}", pr, S_inv[i])

    print("\nSect. 5.3  S = (R^-1)^4 expansion (printed):")
    y2p = -x0 * x3 + x2 - al
    y1p = x0 * x3 ** 2 - x2 * x3 + x1 + al * x3
    y0p = x0 - (x2 - x0 * x3 - al) * (x1 - x3 * (x2 - x0 * x3 - al)) - al
    y3p = x3 - y0p * y1p
    for i, pr in enumerate([y0p, y1p, y2p, y3p]):
        report(f"S y{i}", pr, S_[i])
    print("  (degrees of the definitional S: "
          f"{tuple(sp.Poly(t, x0, x1, x2, x3).total_degree() for t in S_)}"
          "  <- the paper's claim (5,3,2,8) is correct)")

    print("\n  Root cause of the Sect. 5.3 mismatches: the chain a -> b -> c -> y3 is\n"
          "  a = x2-x0x3-alpha, b = x1-x3a-alpha, c = x0-ab-alpha, y3 = x3-bc-alpha;\n"
          "  the paper keeps the -alpha only in a and drops it in b, c and y3.\n"
          "  Feeding the TRUE y0,y1 into the paper's y3 = x3 - y0y1 still misses -alpha:")
    cc, bb = S_[0], S_[1]
    report("S y3 (true y0,y1 substituted)", sp.expand(x3 - cc * bb), S_[3])

    print("\nSect. 5.5.1  quadratic system for y = S(x) (printed):")
    Y2 = sp.expand(x0 * x3 + al - x2)
    Y1 = sp.expand(Y2 * x3 - x1)
    Y0 = sp.expand(Y1 * Y2 + al - x0)
    Y3 = sp.expand(Y0 * Y1 - x3)
    for i, pr in enumerate([Y0, Y1, Y2, Y3]):
        report(f"5.5.1 y{i} vs +S", pr, S_[i])
        report(f"5.5.1 y{i} vs -S", pr, sp.expand(-S_[i]))


if __name__ == "__main__":
    main()
