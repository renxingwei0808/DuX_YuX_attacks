"""E11 / W15-B -- the cheap-coordinate elimination lemma (O10) as a tool.

SETTING (r_KR = 2, chosen-ciphertext main line).  With
    Y   = SL(P + rk^0)            the inner S layer (encryption S-box),
    W'  = the outer S layer's output block-wise,
    Z   = L(W') + rk^2            the state whose layer-l zero-sum we exploit,
a balanced block POSITION p of Z contributes the four linear identities
    sum_P [L . sum_P W']_i = 0      for i in {p, p+4, p+8, p+12}.
The outer S-box has exactly one CHEAP coordinate -- quadratic and bilinear, so
its linearisation monomial set stays small:
    DuX   encryption S:  a  = x2 - x0 x3 - alpha        at position 2
    YuX   encryption S:  z0 = x3 - x0 x1 - x2 - alpha   at position 3
and, in the chosen-plaintext (CPA) direction where the outer map is S^{-1},
    DuX   S^{-1}: f = x0 x1 + x3 + alpha (position 3) and
                  g = x1 x2 + x0 + alpha (position 0)   -> cheap columns {0,3}
    YuX   S^{-1}: y0 = x0 + x1 x2 + x3 + alpha and y1 - y0 = x1 + x2 x3
                                                        -> cheap columns {0,1}
The other coordinates have degree 3/5/8 and blow the linearisation up.

LEMMA (O10).  Let R = {i : i mod 4 in balanced_positions} be the usable rows of
the linear layer and let the "expensive" columns be the words whose position is
not cheap.  Every vector of the LEFT KERNEL
        K = { y : y . L[R, expensive columns] = 0 }
gives one equation containing ONLY cheap coordinates,
        sum_b (yL)_{4b+c} . sum_P W'_{4b+c} = 0        (c the cheap position),
whose linearisation unknowns are the ones E07 already uses -- so each structure
(and, with O12, each weight) yields dim K equations instead of one per block.

The matrix L is the FORWARD (encryption) linear layer in the decryption
direction and L^{-1} in the CPA direction; with the repository's convention
(L x)_i = sum_l row[l] x_{(i+l) mod 16} that means L[i][j] = row[(j-i) mod 16].
Everything is computed with exact elimination over F_2 (characteristic 2: the
matrices are 0/1, so the rank is the same over every F_{2^n}) or over F_p.

O2 caveat: with DuX's L1 the kernel vectors get rotated and change sign, but
the EQUATION SET is the same; analyses fix L0 (`--t 0`, the default) and
`--t 1` is available as the control.

Usage:
  python tools/cheap_rows.py --cipher dux --field 65537 --direction dec --balanced 0001
  python tools/cheap_rows.py --cipher yux --field 2^16  --direction dec --balanced 1110
  python tools/cheap_rows.py --cipher dux --field 65537 --direction enc --balanced 0001
  python tools/cheap_rows.py --table                 # the whole O10 table
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from cipher_degree import CIPHERS  # noqa: E402

WORDS = 16


# --------------------------------------------------------------- matrices --
def layer_row(cipher, field, direction, t=0):
    """First row of the 16 x 16 circulant used by the lemma.

    direction 'dec': the ENCRYPTION linear layer L (the outer layer of the
    r_KR = 2 equation is the encryption S-box, preceded by L).
    direction 'enc': its inverse L^{-1} (the CPA equation's outer map is
    S^{-1}, preceded by L^{-1})."""
    from dux.field import make_field
    F = make_field(field)
    if direction == "enc":
        rots = (CIPHERS[cipher]["ROT_INV"] if t == 0
                else CIPHERS[cipher]["ROT_INV_L1"])
        row = [0] * WORDS
        for j in rots:
            row[j] += 1
        return tuple(v % (2 if F.char == 2 else F.p) for v in row), F
    if cipher == "dux":
        from dux.linear import inv_matrix_rows
        from dux.params import ROT_FWD_BIN
        if F.char == 2:
            return tuple(1 if j in ROT_FWD_BIN[t] else 0 for j in range(WORDS)), F
        return inv_matrix_rows(F)[t], F
    from yux.linear import forward_row
    return forward_row(F), F


def layer_matrix(row, p=None):
    """L[i][j] = row[(j - i) mod 16]."""
    return [[row[(j - i) % WORDS] for j in range(WORDS)] for i in range(WORDS)]


# --------------------------------------------------------------- kernels ---
def _left_kernel(M, mod):
    """Basis of {y : y M = 0} over F_mod (mod = 2 or an odd prime), exact."""
    m = len(M)
    n = len(M[0]) if m else 0
    # augment M^T-style: work on rows of [M | I] and eliminate columns of M
    A = [list(M[i]) + [1 if k == i else 0 for k in range(m)] for i in range(m)]
    A = [[v % mod for v in r] for r in A]
    r = 0
    for c in range(n):
        piv = next((i for i in range(r, m) if A[i][c] % mod), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][c], mod - 2, mod)
        A[r] = [(v * inv) % mod for v in A[r]]
        for i in range(m):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(u - f * v) % mod for u, v in zip(A[i], A[r])]
        r += 1
        if r == m:
            break
    ker = [row[n:] for row in A[r:]]        # rows that became zero on M
    return ker


def _normalise(y, mod):
    """Scale a kernel vector so its first nonzero entry is 1 (or -1 -> 1)."""
    c = next((v for v in y if v % mod), None)
    if c is None:
        return list(y)
    inv = pow(c % mod, mod - 2, mod)
    return [(v * inv) % mod for v in y]


def _signed(v, mod):
    """Print an F_p element as a small signed integer when it is +-1, +-2, ..."""
    v %= mod
    return v - mod if v > mod // 2 else v


def positions(balanced):
    """Accept '0001' (position 0 leftmost) or an iterable of positions."""
    if isinstance(balanced, str):
        assert len(balanced) == 4 and set(balanced) <= {"0", "1"}
        return [p for p in range(4) if balanced[p] == "1"]
    return sorted(int(v) for v in balanced)


def cheap_rows(cipher, field, direction, balanced_positions,
               cheap_positions=None, t=0):
    """Return [(y, yL)] -- one entry per basis vector of the left kernel.

    `y` is indexed by the usable rows R (in increasing order) and `yL` is the
    full 16-vector of coefficients of the combined equation, which is zero on
    every expensive column by construction."""
    bal = positions(balanced_positions)
    cheap = tuple(cheap_positions) if cheap_positions is not None else \
        CIPHERS[cipher]["cheap_dec" if direction == "dec" else "cheap_enc"]
    row, F = layer_row(cipher, field, direction, t)
    mod = 2 if F.char == 2 else F.p
    L = layer_matrix(row)
    R = [i for i in range(WORDS) if i % 4 in bal]
    exp_cols = [j for j in range(WORDS) if j % 4 not in cheap]
    sub = [[L[i][j] % mod for j in exp_cols] for i in R]
    ker = _left_kernel(sub, mod)
    out = []
    for y in ker:
        y = _normalise(y, mod)
        yL = [sum(y[t_] * L[R[t_]][j] for t_ in range(len(R))) % mod
              for j in range(WORDS)]
        assert all(yL[j] == 0 for j in exp_cols), "kernel vector is not cheap-only"
        out.append((y, yL))
    return out


def describe(cipher, field, direction, balanced, cheap=None, t=0):
    rows = cheap_rows(cipher, field, direction, balanced, cheap, t)
    row, F = layer_row(cipher, field, direction, t)
    mod = 2 if F.char == 2 else F.p
    bal = positions(balanced)
    cheapp = tuple(cheap) if cheap is not None else \
        CIPHERS[cipher]["cheap_dec" if direction == "dec" else "cheap_enc"]
    R = [i for i in range(WORDS) if i % 4 in bal]
    return {"cipher": cipher, "field": field, "direction": direction,
            "balanced": balanced if isinstance(balanced, str) else bal,
            "balanced_positions": bal, "cheap_positions": list(cheapp),
            "rows": R, "modulus": mod, "dim_K": len(rows),
            "kernel": [{"y": [_signed(v, mod) for v in y],
                        "yL": [_signed(v, mod) for v in yL],
                        "cheap_coeffs": {str(j): _signed(yL[j], mod)
                                         for j in range(WORDS) if j % 4 in cheapp}}
                       for y, yL in rows]}


def block_coeffs(cipher, field, direction, balanced, cheap=None, t=0):
    """The per-outer-block coefficients of each combined equation.

    `cheap_rows` returns the 16-vector yL, which is supported on the cheap
    columns {4b + c : b = 0..3, c cheap}.  With one cheap position (the
    decryption direction) that is one coefficient per outer block b, which is
    exactly the vector `assemble_fast.combine_rows` needs to fold the four
    per-outer-block rows of `rows_from_moments` into one equation.  With two
    cheap positions (the CPA direction) the equation touches two coordinates
    per block and the caller must combine per (block, coordinate) instead;
    those are returned as a dict."""
    from cipher_degree import CIPHERS
    cheapp = tuple(cheap) if cheap is not None else \
        CIPHERS[cipher]["cheap_dec" if direction == "dec" else "cheap_enc"]
    out = []
    for y, yL in cheap_rows(cipher, field, direction, balanced, cheap, t):
        if len(cheapp) == 1:
            c = cheapp[0]
            out.append([yL[4 * b + c] for b in range(4)])
        else:
            out.append({(b, c): yL[4 * b + c] for b in range(4) for c in cheapp})
    return out


PATTERNS = ["0001", "0010", "0011", "0100", "0101", "0110", "0111",
            "1000", "1001", "1010", "1011", "1100", "1101", "1110", "1111"]
TABLE_FIELDS = ["2^16", "65537", "193", "257"]
# W27 / E13: the six fields the T1 kernel table is asked for.
ALL_FIELDS = ["2^4", "2^8", "2^16", "193", "257", "65537"]


def table(fields=None, patterns=None, ciphers=None, directions=None, cheap=None):
    """The O10 table.  `ciphers` / `directions` / `cheap` default to the whole
    grid with each family's own cheap set, which is what every earlier call
    produced; W27 passes a single (cipher, direction) and an explicit cheap
    set to get the extended {1, 2} rows."""
    fields = fields or TABLE_FIELDS
    patterns = patterns or PATTERNS
    out = []
    for cipher in (ciphers or ("dux", "yux")):
        for direction in (directions or ("dec", "enc")):
            for field in fields:
                for pat in patterns:
                    out.append(describe(cipher, field, direction, pat, cheap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="dux", choices=tuple(CIPHERS))
    ap.add_argument("--field", default="65537",
                    help="65537 | 257 | 193 | 2^16 | 2^8 | 2^4")
    ap.add_argument("--direction", default="dec", choices=("dec", "enc"),
                    help="dec: chosen ciphertext (outer map = encryption S, "
                         "uses L); enc: CPA (outer map = S^{-1}, uses L^{-1})")
    ap.add_argument("--balanced", default="0001",
                    help="4-bit pattern of balanced block positions, "
                         "position 0 leftmost")
    ap.add_argument("--cheap", default=None,
                    help="override the cheap columns, e.g. '0,3'")
    ap.add_argument("--t", type=int, default=0, choices=(0, 1),
                    help="DuX only: L0 (default) or L1 -- O2 control")
    ap.add_argument("--table", action="store_true", help="the whole O10 table; "
                    "with --cheap it is restricted to --cipher / --direction and "
                    "uses the given cheap set")
    ap.add_argument("--all-fields", action="store_true",
                    help="--table over 2^4, 2^8, 2^16, 193, 257, 65537")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.table:
        cheap_t = [int(v) for v in a.cheap.split(",")] if a.cheap else None
        rows = table(fields=(ALL_FIELDS if a.all_fields else None),
                     ciphers=(a.cipher,) if a.cheap else None,
                     directions=(a.direction,) if a.cheap else None,
                     cheap=cheap_t)
        for r in rows:
            print(f"{r['cipher']:4s} {r['direction']:4s} {r['field']:6s} "
                  f"{r['balanced']}  cheap {r['cheap_positions']}  "
                  f"dim K = {r['dim_K']}")
        if a.json:
            json.dump(rows, open(a.json, "w"), indent=1)
        return
    cheap = [int(v) for v in a.cheap.split(",")] if a.cheap else None
    r = describe(a.cipher, a.field, a.direction, a.balanced, cheap, a.t)
    print(f"{r['cipher']} / {r['direction']} / F_{r['field']} / balanced "
          f"{r['balanced']} (positions {r['balanced_positions']})")
    print(f"  rows used          : {r['rows']}")
    print(f"  cheap columns      : positions {r['cheap_positions']}")
    print(f"  dim K              : {r['dim_K']}")
    for k, e in enumerate(r["kernel"]):
        print(f"   y[{k}] = {e['y']}   ->  cheap coefficients {e['cheap_coeffs']}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
