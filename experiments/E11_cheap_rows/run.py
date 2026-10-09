"""E11 -- generate the full O10 cheap-row table (all patterns x 2 ciphers x
2 directions x 6 fields) as markdown + JSON.

    python experiments/E11_cheap_rows/run.py --out results/E11_cheap_rows
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from cheap_rows import PATTERNS, describe  # noqa: E402

FIELDS = ["2^4", "2^8", "2^16", "193", "257", "65537"]


def build():
    recs = []
    for cipher in ("dux", "yux"):
        for direction in ("dec", "enc"):
            for field in FIELDS:
                for pat in PATTERNS:
                    recs.append(describe(cipher, field, direction, pat))
    return recs


def markdown(recs):
    L = ["# E11 . Cheap-coordinate elimination (O10): dim K for every pattern\n",
         "`dim K` is the dimension of the left kernel "
         "`{ y : y L[balanced rows, expensive columns] = 0 }`; each basis "
         "vector is one linearised equation that contains ONLY the cheap "
         "(bilinear) coordinate of the outer S-box, so a structure yields "
         "`dim K` equations instead of one per outer block.\n",
         "* `dec` (chosen ciphertext, r_KR on the plaintext side): the outer "
         "map is the ENCRYPTION S-box and the matrix is the encryption linear "
         "layer L.  Cheap column: DuX position 2 (`a = x2 - x0 x3 - alpha`), "
         "YuX position 3 (`z0 = x3 - x0 x1 - x2 - alpha`).\n",
         "* `enc` (chosen plaintext / CPA): the outer map is `S^{-1}` and the "
         "matrix is `L^{-1}`.  Cheap columns: DuX {0,3} (`g` and `f`), YuX "
         "{0,1} (`y0` and `y1 - y0`).\n",
         "Patterns are block positions 0..3 with position 0 leftmost; "
         "`1110` means positions 0, 1, 2 balanced.\n"]
    for cipher in ("dux", "yux"):
        for direction in ("dec", "enc"):
            L.append(f"## {cipher.upper()}, direction `{direction}`\n")
            L.append("| field | " + " | ".join(PATTERNS) + " |")
            L.append("|---" * (len(PATTERNS) + 1) + "|")
            for field in FIELDS:
                vals = [str(next(r["dim_K"] for r in recs
                                 if r["cipher"] == cipher and r["direction"] == direction
                                 and r["field"] == field and r["balanced"] == p))
                        for p in PATTERNS]
                L.append(f"| {field} | " + " | ".join(vals) + " |")
            L.append("")
    L.append("## The equations the R5 attacks use\n")
    for cipher, direction, pat in (("dux", "dec", "0001"), ("yux", "dec", "1110"),
                                   ("dux", "enc", "0001"), ("yux", "enc", "0111"),
                                   ("dux", "enc", "1110"), ("yux", "enc", "1110")):
        for field in ("65537", "2^16"):
            r = next(x for x in recs if x["cipher"] == cipher
                     and x["direction"] == direction and x["field"] == field
                     and x["balanced"] == pat)
            L.append(f"### {cipher} / {direction} / `{pat}` / F_{field} "
                     f"-- dim K = {r['dim_K']}, rows {r['rows']}\n")
            for k, e in enumerate(r["kernel"]):
                nz = {j: v for j, v in e["cheap_coeffs"].items() if v}
                L.append(f"* `y[{k}] = {e['y']}`  ->  "
                         + " + ".join(f"{v:+d}*W'_{j}" for j, v in nz.items())
                         + " = 0")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = build()
    for cipher in ("dux", "yux"):
        for direction in ("dec", "enc"):
            for field in ("2^16", "65537"):
                dims = [next(r["dim_K"] for r in recs if r["cipher"] == cipher
                             and r["direction"] == direction and r["field"] == field
                             and r["balanced"] == p) for p in PATTERNS]
                print(f"{cipher} {direction} {field:6s} " +
                      " ".join(f"{p}:{d}" for p, d in zip(PATTERNS, dims)))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(recs, open(os.path.join(a.out, "cheap_rows_table.json"), "w"), indent=1)
        open(os.path.join(a.out, "cheap_rows_table.md"), "w").write(markdown(recs))
        print("wrote cheap_rows_table.{json,md}")


if __name__ == "__main__":
    main()
