"""Y02 / W15-B -- Ni et al.'s algebraic-degree tables, reproduced cell by cell.

SOURCE (cross-check literature, NOT a specification): Jianqiang Ni, Gaoli Wang,
Yingxin Li, "Tracking Algebraic Degree with Exponent Sets: Higher-Order
Differential Attacks on FHE-Friendly Cipher Yu2X", DCC 2026, 94:123.
  Table 3 (p. 13)  univariate,   Yu2X-8 and Yu2X-16, input index 0/1/2 and 3
  Table 4 (p. 13)  two variables (0,1) / (0,2)
  Table 6 (pp. 20-21) Yu2X-8 multivariate, 15 word sets
  Table 7 (p. 22)  Yu2X-16 multivariate, 5 word sets
The numbers are transcribed verbatim into ni_tables.json ("-" = not listed).

CONVENTIONS (docs/yux_specification.md):
  * Ni's "round r" is our "layer r": the state after r decryption S^{-1}
    layers.  Their round 0 is the input, where every active word has degree 1.
  * Their "output index i" is block 0, position i, and their input indices are
    absolute ciphertext word indices (a single index j means the word j).
  * They report the BOOLEAN degree; we compute the max-plus FORMAL degree D
    (the bound on the SUM of the exponents) and convert it into a Boolean
    degree.  For one variable that conversion is Ni et al.'s own Lemma 1,
    deg_2 <= floor(log2(D+1)), the one our single-word YuX table
    uses.  For s variables the sharpest conversion a total-degree bound admits
    is the knapsack
        hw(D, s, n) = max { sum_i HW(a_i) : sum_i a_i <= D, a_i <= 2^n - 1 },
    because a variable carrying Hamming weight w needs exponent >= 2^w - 1 and
    the cost 2^w - 1 is convex, so the optimum spreads the weight as evenly as
    possible over the s variables.  Note hw(D, s, n) = s*n exactly when
    D >= T = s*(2^n - 1): the Boolean criterion "deg < s n" and Theorem O7's
    "D < T" are the SAME condition, so the two machines always agree about
    which layers are balanced, whatever slack the intermediate cells show.
  * D is taken on BLOCK 0 exactly (no max over blocks), because that is what
    their per-block table reports.

Three outcomes per cell:
  agree            floor(log2(D+1)) == Ni
  maxplus_looser   floor(log2(D+1)) >  Ni   (their SMT/exponent-set bound is
                                             sharper -- the interesting case)
  maxplus_tighter  floor(log2(D+1)) <  Ni
plus a FROBENIUS flag when D is exactly a power of two, where the leading
monomial can vanish for field-theoretic reasons and a strictly smaller Boolean
degree is expected (memo Sect. 2 item 1, the 62-vs-64 cell), and a SAME_BLOCK
flag for cells where every active word sits in one 4-word block.  The knapsack
conversion is exact only when all s variables reach the output word
symmetrically; for active words spread over several blocks it has slack (the
total degree cannot see that early layers only mix the same-block variable),
which is where Ni's per-variable exponent sets are genuinely sharper.

Usage:
    python experiments/Y02_degree_bounds/ni_tables.py \
        --out results/Y02_degree_bounds
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))

from cipher_degree import profile  # noqa: E402

D = "-"          # "not listed" in the paper

# --------------------------------------------------------------- Table 3 ---
# univariate; key = (cipher, input word, output position) -> rounds 0..10
TABLE3 = {
    "page": 13,
    "caption": "The algebraic degree upper bounds of Yu2X in univariate setting",
    "rounds": list(range(11)),
    "rows": [
        # Yu2X-8, input index 0/1/2 (printed once for the three inputs)
        ("Yu2X-8", [0], "0/1/2", 0, [1, 1, 1, 3, 5, 7, 8, D, D, D, D]),
        ("Yu2X-8", [0], "0/1/2", 1, [1, 1, 2, 3, 5, 7, 8, D, D, D, D]),
        ("Yu2X-8", [0], "0/1/2", 2, [1, 1, 2, 4, 6, 8, D, D, D, D, D]),
        ("Yu2X-8", [0], "0/1/2", 3, [1, 1, 2, 4, 6, 8, D, D, D, D, D]),
        # Yu2X-8, input index 3
        ("Yu2X-8", [3], "3", 0, [1, 1, 2, 4, 6, 8, D, D, D, D, D]),
        ("Yu2X-8", [3], "3", 1, [1, 1, 2, 4, 6, 8, D, D, D, D, D]),
        ("Yu2X-8", [3], "3", 2, [1, 1, 2, 4, 6, 8, D, D, D, D, D]),
        ("Yu2X-8", [3], "3", 3, [1, 1, 3, 5, 7, 8, D, D, D, D, D]),
        # Yu2X-16, input index 0/1/2
        ("Yu2X-16", [0], "0/1/2", 0, [1, 1, 1, 3, 5, 7, 9, 11, 13, 15, 16]),
        ("Yu2X-16", [0], "0/1/2", 1, [1, 1, 2, 3, 5, 7, 9, 11, 13, 15, 16]),
        ("Yu2X-16", [0], "0/1/2", 2, [1, 1, 2, 4, 6, 8, 10, 12, 14, 16, D]),
        ("Yu2X-16", [0], "0/1/2", 3, [1, 1, 2, 4, 6, 8, 10, 12, 14, 16, D]),
        # Yu2X-16, input index 3
        ("Yu2X-16", [3], "3", 0, [1, 1, 2, 4, 6, 8, 10, 12, 14, 16, D]),
        ("Yu2X-16", [3], "3", 1, [1, 1, 2, 4, 6, 8, 10, 12, 14, 16, D]),
        ("Yu2X-16", [3], "3", 2, [1, 1, 2, 4, 6, 8, 10, 12, 14, 16, D]),
        ("Yu2X-16", [3], "3", 3, [1, 1, 3, 5, 7, 9, 11, 13, 15, 16, D]),
    ],
    # the paper prints one row for the three inputs 0, 1, 2; we check all three
    "input_aliases": {"0/1/2": [0, 1, 2], "3": [3]},
}

# --------------------------------------------------------------- Table 4 ---
TABLE4 = {
    "page": 13,
    "caption": "The algebraic degree upper bounds of Yu2X in two variables setting",
    "rounds": list(range(11)),
    "rows": [
        ("Yu2X-8", [0, 1], "(0,1)/(0,2)", 0, [1, 1, 2, 5, 9, 13, 16, D, D, D, D]),
        ("Yu2X-8", [0, 1], "(0,1)/(0,2)", 1, [1, 1, 2, 5, 9, 13, 16, D, D, D, D]),
        ("Yu2X-8", [0, 1], "(0,1)/(0,2)", 2, [1, 1, 3, 6, 10, 14, 16, D, D, D, D]),
        ("Yu2X-8", [0, 1], "(0,1)/(0,2)", 3, [1, 2, 3, 7, 11, 15, 16, D, D, D, D]),
        ("Yu2X-16", [0, 1], "(0,1)/(0,2)", 0, [1, 1, 2, 5, 9, 13, 17, 21, 25, 29, 32]),
        ("Yu2X-16", [0, 1], "(0,1)/(0,2)", 1, [1, 1, 2, 5, 9, 13, 17, 21, 25, 29, 32]),
        ("Yu2X-16", [0, 1], "(0,1)/(0,2)", 2, [1, 1, 3, 6, 10, 14, 18, 22, 26, 30, 32]),
        ("Yu2X-16", [0, 1], "(0,1)/(0,2)", 3, [1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 32]),
    ],
    "input_aliases": {"(0,1)/(0,2)": [[0, 1], [0, 2]]},
}

# --------------------------------------------------------------- Table 6 ---
# Yu2X-8 multivariate, rounds 0..7.  Rows printed as "0/1" are one row for the
# two output positions 0 and 1; they are expanded here.
_T6 = [
    ([0, 4, 8], {"0/1": [1, 1, 2, 6, 12, 18, 24, D],
                 "2": [1, 1, 3, 8, 14, 20, 24, D],
                 "3": [1, 1, 3, 9, 15, 21, 24, D]}),
    ([0, 4, 8, 12], {"0/1": [1, 1, 2, 6, 14, 22, 30, 32],
                     "2": [1, 1, 3, 9, 17, 25, 32, D],
                     "3": [1, 1, 3, 11, 18, 26, 32, D]}),
    ([0, 1, 4, 8, 12], {"0": [1, 1, 2, 8, 16, 26, 36, 40],
                        "1": [1, 1, 3, 8, 17, 27, 37, 40],
                        "2": [1, 1, 4, 11, 20, 30, 40, D],
                        "3": [1, 2, 4, 12, 21, 31, 40, D]}),
    ([0, 1, 3, 4, 8, 12], {"0": [1, 1, 3, 8, 19, 31, 43, 48],
                           "1": [1, 1, 3, 8, 19, 31, 43, 48],
                           "2": [1, 2, 4, 11, 22, 34, 46, 48],
                           "3": [1, 2, 5, 12, 23, 35, 47, 48]}),
    ([0, 1, 3, 4, 6, 8, 12], {"0": [1, 1, 3, 10, 21, 35, 49, 56],
                              "1": [1, 1, 4, 10, 21, 35, 49, 56],
                              "2": [1, 2, 5, 13, 25, 39, 53, 56],
                              "3": [1, 2, 5, 13, 26, 40, 54, 56]}),
    ([0, 1, 4, 5, 8, 9, 12, 13], {"0": [1, 1, 4, 12, 21, 37, 53, 64],
                                  "1": [1, 1, 4, 12, 21, 37, 53, 64],
                                  "2": [1, 1, 6, 16, 32, 48, 64, D],
                                  "3": [1, 2, 6, 17, 33, 49, 64, D]}),
    ([0, 1, 3, 4, 5, 8, 9, 12, 13], {"0": [1, 1, 4, 12, 22, 40, 58, 72],
                                     "1": [1, 1, 4, 12, 22, 40, 58, 72],
                                     "2": [1, 2, 6, 18, 36, 54, 72, D],
                                     "3": [1, 2, 6, 18, 36, 54, 72, D]}),
    ([0, 1, 3, 4, 5, 8, 9, 12, 13, 15], {"0": [1, 1, 4, 12, 23, 43, 63, 80],
                                         "1": [1, 1, 4, 12, 23, 43, 63, 80],
                                         "2": [1, 2, 6, 18, 38, 58, 78, 80],
                                         "3": [1, 2, 6, 18, 38, 58, 78, 80]}),
    ([0, 1, 3, 4, 5, 7, 8, 9, 12, 13, 15], {"0": [1, 1, 4, 12, 24, 46, 68, 88],
                                            "1": [1, 1, 4, 12, 24, 46, 68, 88],
                                            "2": [1, 2, 6, 18, 30, 52, 74, 88],
                                            "3": [1, 2, 6, 18, 30, 52, 74, 88]}),
    ([0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15], {"0": [1, 1, 4, 12, 26, 50, 74, 96],
                                                "1": [1, 1, 4, 12, 26, 50, 74, 96],
                                                "2": [1, 2, 6, 19, 43, 67, 91, 96],
                                                "3": [1, 2, 6, 19, 43, 67, 91, 96]}),
    ([0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
     {"0": [1, 2, 5, 14, 28, 54, 80, 104],
      "1": [1, 2, 5, 14, 28, 54, 80, 104],
      "2": [1, 3, 7, 21, 47, 73, 99, 104],
      "3": [1, 3, 8, 22, 48, 74, 100, 104]}),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15],
     {"0": [1, 2, 5, 16, 30, 58, 86, 112],
      "1": [1, 2, 5, 16, 30, 58, 86, 112],
      "2": [1, 3, 8, 24, 52, 80, 108, 112],
      "3": [1, 3, 8, 24, 52, 80, 108, 112]}),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
     {"0": [1, 2, 6, 17, 36, 66, 96, 120],
      "1": [1, 2, 6, 17, 36, 66, 96, 120],
      "2": [1, 3, 9, 26, 56, 86, 116, 120],
      "3": [1, 3, 9, 26, 56, 86, 116, 120]}),
    # printed as "(0,...,12,14,13,15)": all sixteen words, the order is a typo
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
     {"0": [1, 2, 6, 18, 38, 70, 102, 128],
      "1": [1, 2, 6, 18, 38, 70, 102, 128],
      "2": [1, 3, 9, 28, 60, 92, 124, 128],
      "3": [1, 3, 9, 28, 60, 92, 124, 128]}),
]

# --------------------------------------------------------------- Table 7 ---
_T7 = [
    ([0, 4, 8], {"0/1": [1, 1, 2, 6, 12, 18, 24, 30, 36, 42, 48, D],
                 "2": [1, 1, 3, 8, 14, 20, 26, 32, 38, 44, 48, D],
                 "3": [1, 1, 3, 9, 15, 21, 27, 33, 39, 45, 48, D]}),
    ([0, 4, 8, 12], {"0/1": [1, 1, 2, 6, 14, 22, 30, 38, 46, 54, 62, 64],
                     "2": [1, 1, 3, 9, 17, 25, 33, 41, 49, 57, 64, D],
                     "3": [1, 1, 3, 11, 18, 26, 34, 42, 50, 58, 64, D]}),
    ([0, 1, 4, 8, 12], {"0": [1, 1, 2, 8, 16, 26, 36, 46, 56, 66, 76, 80],
                        "1": [1, 1, 3, 8, 17, 27, 37, 47, 57, 67, 77, 80],
                        "2": [1, 1, 4, 11, 20, 30, 40, 50, 60, 70, 80, D],
                        "3": [1, 2, 4, 12, 21, 31, 41, 51, 61, 71, 80, D]}),
    ([0, 1, 3, 4, 8, 12], {"0": [1, 1, 3, 8, 19, 31, 43, 55, 67, 79, 91, 96],
                           "1": [1, 1, 3, 8, 19, 31, 43, 55, 67, 79, 91, 96],
                           "2": [1, 2, 4, 11, 22, 34, 46, 58, 70, 82, 94, 96],
                           "3": [1, 2, 5, 12, 23, 35, 47, 59, 71, 83, 95, 96]}),
    ([0, 1, 3, 4, 6, 8, 12], {"0": [1, 1, 3, 10, 21, 35, 49, 63, 77, 91, 105, 112],
                              "1": [1, 1, 4, 10, 21, 35, 49, 63, 77, 91, 105, 112],
                              "2": [1, 2, 5, 13, 25, 39, 53, 67, 81, 95, 109, 112],
                              "3": [1, 2, 5, 13, 26, 40, 54, 68, 82, 96, 110, 112]}),
]


def _expand(sets, cipher, page, caption, nrounds):
    rows = []
    for words, per_out in sets:
        for key, vals in per_out.items():
            assert len(vals) == nrounds, (words, key, len(vals))
            for out in (int(v) for v in key.split("/")):
                rows.append((cipher, list(words), key, out, list(vals)))
    return {"page": page, "caption": caption,
            "rounds": list(range(nrounds)), "rows": rows, "input_aliases": {}}


TABLE6 = _expand(_T6, "Yu2X-8", "20-21",
                 "The algebraic degree upper bounds of Yu2X-8 in multivariate setting", 8)
TABLE7 = _expand(_T7, "Yu2X-16", 22,
                 "The algebraic degree upper bounds of Yu2X-16 in multivariate setting", 12)

TABLES = {"table3": TABLE3, "table4": TABLE4, "table6": TABLE6, "table7": TABLE7}
NBITS = {"Yu2X-8": 8, "Yu2X-16": 16}


def hw_bound(Dval, s=1, n=16):
    """max { sum_i HW(a_i) : sum a_i <= Dval, a_i <= 2^n - 1 }.

    A variable of Hamming weight w costs at least 2^w - 1, and 2^w - 1 is
    convex in w, so the cheapest way to buy a total weight W with s variables
    is to spread it as evenly as possible.  For s = 1 this is exactly Ni et
    al.'s Lemma 1, floor(log2(D+1))."""
    if s == 1:
        return min((Dval + 1).bit_length() - 1, n)
    best = 0
    for W in range(s * n + 1):
        a, b = divmod(W, s)
        if b * (2 ** (a + 1) - 1) + (s - b) * (2 ** a - 1) <= Dval:
            best = W
        else:
            break
    return best


def cells(table_name):
    """One record per (row, round) cell with the max-plus comparison."""
    tab = TABLES[table_name]
    out = []
    for cipher, words, in_label, outpos, vals in tab["rows"]:
        n = NBITS[cipher]
        prof = profile(words, len(tab["rounds"]), cipher="yux")
        for r, v in enumerate(vals):
            if r == 0:
                continue                       # round 0 is the input
            Dv = prof[r - 1][outpos]           # block 0, position `outpos`
            hw = hw_bound(Dv, len(words), n)
            frob = Dv > 0 and (Dv & (Dv - 1)) == 0
            rec = {"table": table_name, "cipher": cipher, "input_words": words,
                   "input_label": in_label, "output_index": outpos, "round": r,
                   "ni": None if v == D else v, "maxplus_D": Dv,
                   "maxplus_hw": hw, "frobenius_boundary": frob,
                   "same_block": len({w // 4 for w in words}) == 1,
                   "full_degree": n * len(words),
                   "o7_balanced": hw < n * len(words),
                   "ni_balanced": (None if v == D else v < n * len(words)),
                   "saturated": hw >= n * len(words)}
            if v == D:
                rec["verdict"] = "not_listed"
            elif hw == v:
                rec["verdict"] = "agree"
            elif hw > v:
                rec["verdict"] = "maxplus_looser"
            else:
                rec["verdict"] = "maxplus_tighter"
            out.append(rec)
    return out


def alias_consistency(table_name):
    """Ni prints one row for several input indices ("0/1/2", "(0,1)/(0,2)").
    Check that the max-plus bound really is the same for all of them."""
    tab = TABLES[table_name]
    issues = []
    for cipher, words, in_label, outpos, vals in tab["rows"]:
        alt = tab["input_aliases"].get(in_label)
        if not alt:
            continue
        base = profile(words, len(tab["rounds"]), cipher="yux")
        for a in alt:
            a = a if isinstance(a, list) else [a]
            if a == words:
                continue
            other = profile(a, len(tab["rounds"]), cipher="yux")
            for r in range(len(tab["rounds"]) - 1):
                if base[r][outpos] != other[r][outpos]:
                    issues.append({"table": table_name, "cipher": cipher,
                                   "printed_for": in_label, "words": words,
                                   "alias": a, "output_index": outpos,
                                   "round": r + 1,
                                   "D_printed_row": base[r][outpos],
                                   "D_alias": other[r][outpos]})
    return issues


def summarise(recs):
    tally = {}
    for r in recs:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return tally


def markdown(all_recs, aliases):
    L = []
    L.append("# Y02 . Ni et al.'s degree tables vs the max-plus criterion (O7)\n")
    L.append("Source (cross-check literature, not a specification): Ni, Wang, Li, "
             "*Tracking Algebraic Degree with Exponent Sets*, DCC 2026, 94:123 -- "
             "Table 3 and Table 4 (p. 13), Table 6 (pp. 20-21), Table 7 (p. 22). "
             "Every printed number is transcribed verbatim into "
             "`results/Y02_degree_bounds/ni_tables.json`.\n")
    L.append("**Conventions.** Their round *r* is our layer *r* (their round 0 is "
             "the input); their output index *i* is block 0, position *i*; our "
             "formal degree `D` is the max-plus bound on the SUM of the "
             "exponents, taken on block 0 exactly (no max over blocks). The "
             "Boolean degree it implies is the knapsack\n")
    L.append("```\nhw(D, s, n) = max { sum_i HW(a_i) : sum_i a_i <= D, "
             "a_i <= 2^n - 1 }\n```\n")
    L.append("which for `s = 1` is Ni et al.'s own Lemma 1, "
             "`floor(log2(D+1))`. Because a Hamming weight `w` costs at least "
             "`2^w - 1` and that cost is convex, `hw(D, s, n) = s n` exactly "
             "when `D >= T = s (2^n - 1)`: **the Boolean criterion "
             "`deg < s n` and Theorem O7's `D < T` are literally the same "
             "condition**, so whatever slack the intermediate cells show, the "
             "two machines can only disagree about a balanced layer when one of "
             "the two bounds is not tight.\n")
    L.append("The conversion is exact only when the `s` variables reach the "
             "output word symmetrically -- in particular when they all sit in "
             "one 4-word block. For active words spread over several blocks the "
             "total degree cannot see that the early layers mix only the "
             "same-block variable, so `hw` overshoots; that is where Ni's "
             "per-variable exponent sets are genuinely sharper.\n")

    L.append("## 1. Cell counts\n")
    L.append("| table | cells compared | agree | max-plus looser (Ni sharper) | max-plus tighter (O7 sharper) | not listed |")
    L.append("|---|---|---|---|---|---|")
    tot = {}
    for name in ("table3", "table4", "table6", "table7"):
        recs = [r for r in all_recs if r["table"] == name]
        t = summarise(recs)
        cmpd = t.get("agree", 0) + t.get("maxplus_looser", 0) + t.get("maxplus_tighter", 0)
        L.append(f"| {name} | {cmpd} | {t.get('agree', 0)} | "
                 f"{t.get('maxplus_looser', 0)} | {t.get('maxplus_tighter', 0)} | "
                 f"{t.get('not_listed', 0)} |")
        for k, v in t.items():
            tot[k] = tot.get(k, 0) + v
    cmpd = tot.get("agree", 0) + tot.get("maxplus_looser", 0) + tot.get("maxplus_tighter", 0)
    L.append(f"| **all** | **{cmpd}** | **{tot.get('agree', 0)}** | "
             f"**{tot.get('maxplus_looser', 0)}** | **{tot.get('maxplus_tighter', 0)}** | "
             f"**{tot.get('not_listed', 0)}** |")
    L.append("")
    L.append("Split by whether the active words share one block (where the "
             "conversion is exact):\n")
    L.append("| active words | cells | agree | max-plus looser | max-plus tighter |")
    L.append("|---|---|---|---|---|")
    for flag, label in ((True, "same block"), (False, "several blocks")):
        recs = [r for r in all_recs if r["same_block"] is flag and r["ni"] is not None]
        t = summarise(recs)
        L.append(f"| {label} | {len(recs)} | {t.get('agree', 0)} | "
                 f"{t.get('maxplus_looser', 0)} | {t.get('maxplus_tighter', 0)} |")
    L.append("")

    L.append("## 2. Where the two machines disagree about a BALANCED word\n")
    L.append("These are the only cells that change a distinguisher. "
             "`O7 balanced` means `hw < s n` (equivalently `D < T`); "
             "`Ni balanced` means their printed degree is `< s n`.\n")
    L.append("| table | cipher | active words | out | layer | Ni | s n | D | hw | O7 | Ni | D = 2^k |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    ndiff = 0
    for r in all_recs:
        if r["ni"] is None or r["o7_balanced"] == r["ni_balanced"]:
            continue
        ndiff += 1
        L.append(f"| {r['table']} | {r['cipher']} | {r['input_words']} | "
                 f"{r['output_index']} | {r['round']} | {r['ni']} | "
                 f"{r['full_degree']} | {r['maxplus_D']} | {r['maxplus_hw']} | "
                 f"{'balanced' if r['o7_balanced'] else '-'} | "
                 f"{'balanced' if r['ni_balanced'] else '-'} | "
                 f"{'yes' if r['frobenius_boundary'] else ''} |")
    L.append("")
    L.append(f"{ndiff} cells out of {cmpd} compared.\n")

    L.append("## 3. Saturation cells with D exactly a power of two (Frobenius boundary)\n")
    L.append("A formal degree that is exactly `2^k` has a Frobenius power as its "
             "leading monomial, whose coefficient is often forced to vanish; O7 "
             "cannot use that and reports `not balanced` with margin 0 or -4.\n")
    L.append("| table | cipher | active words | out | layer | Ni | s n | D | O7 | Ni |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in all_recs:
        if not (r["frobenius_boundary"] and r["ni"] is not None and r["saturated"]):
            continue
        L.append(f"| {r['table']} | {r['cipher']} | {r['input_words']} | "
                 f"{r['output_index']} | {r['round']} | {r['ni']} | "
                 f"{r['full_degree']} | {r['maxplus_D']} | "
                 f"{'balanced' if r['o7_balanced'] else '-'} | "
                 f"{'balanced' if r['ni_balanced'] else '-'} |")
    L.append("")

    L.append("## 4. Same-block cells that disagree\n")
    L.append("| table | cipher | active words | out | layer | Ni | D | hw | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in all_recs:
        if r["same_block"] and r["verdict"] in ("maxplus_looser", "maxplus_tighter"):
            L.append(f"| {r['table']} | {r['cipher']} | {r['input_words']} | "
                     f"{r['output_index']} | {r['round']} | {r['ni']} | "
                     f"{r['maxplus_D']} | {r['maxplus_hw']} | {r['verdict']} |")
    L.append("")

    L.append(f"## 5. Rows printed for several input indices: {len(aliases)} inconsistencies\n")
    if not aliases:
        L.append("Every row Ni prints for a set of input indices (\"0/1/2\", "
                 "\"(0,1)/(0,2)\") really does carry the same max-plus bound for "
                 "each member of the set.\n")
    else:
        L.append("| table | printed for | words | alias | out | layer | D printed | D alias |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in aliases:
            L.append(f"| {a['table']} | {a['printed_for']} | {a['words']} | "
                     f"{a['alias']} | {a['output_index']} | {a['round']} | "
                     f"{a['D_printed_row']} | {a['D_alias']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    all_recs, aliases = [], []
    for name in TABLES:
        all_recs += cells(name)
        aliases += alias_consistency(name)
    tally = summarise(all_recs)
    print("verdict tally:", tally)
    print("alias inconsistencies:", len(aliases))
    for r in all_recs:
        if r["verdict"] in ("maxplus_looser", "maxplus_tighter"):
            print(f"  {r['table']} {r['cipher']} in={r['input_words']} "
                  f"out={r['output_index']} round={r['round']}: Ni {r['ni']} vs "
                  f"D={r['maxplus_D']} -> {r['maxplus_hw']} ({r['verdict']}"
                  f"{', Frobenius' if r['frobenius_boundary'] else ''})")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({"source": "Ni, Wang, Li, DCC 2026, 94:123",
                   "tables": {k: {"page": v["page"], "caption": v["caption"],
                                  "rounds": v["rounds"], "rows": v["rows"]}
                              for k, v in TABLES.items()},
                   "cells": all_recs, "alias_issues": aliases, "tally": tally},
                  open(os.path.join(a.out, "ni_tables.json"), "w"), indent=1)
        open(os.path.join(a.out, "ni_vs_maxplus.md"), "w").write(
            markdown(all_recs, aliases))
        print("wrote", os.path.join(a.out, "ni_tables.json"),
              "and ni_vs_maxplus.md")


if __name__ == "__main__":
    main()
