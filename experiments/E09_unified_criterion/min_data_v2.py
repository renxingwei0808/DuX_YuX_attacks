"""E09 / W19-D -- the minimal-data table v2, inside the O7 framework.

`results/E09_unified_criterion/dux2_8_layer7_search.txt` answered one instance
of one question: what is the cheapest structure that makes SOME block position
of DuX(2^8) balanced at layer 7?  (Answer: 11 active words, 2^88, pattern
`0001`.)  R6 needs the same answer for every target the two attack routes can
consume, which means generalising the search along four axes:

  * both ciphers (`--cipher dux|yux`), both directions (`dec` = chosen
    ciphertext, `enc` = the designers' CPA model);
  * FULL BLOCKS (O11): zero or one whole 4-word block running over F_q^4, which
    passes the first S layer for free and contributes four full-field words to
    the threshold;
  * MIXED subspace dimensions in characteristic 2 -- the data cost of an active
    word is m bits and its threshold contribution 2^m - 1, so allocations like
    (16, 16, 15) are strictly cheaper than (16, 16, 16) whenever they still
    clear the degree;
  * PATTERN CLASSES rather than "any balanced position", because what the key
    recovery can use is decided by O10:

        1111   every position balanced -- the four plain E07 rows;
        xxx1   position 3 balanced     -- DuX dec, dim K = 1  (and DuX CPA,
                                          dim K = 4);
        1110   positions 0,1,2         -- YuX dec, dim K = 3  (and DuX CPA,
                                          dim K = 4);
        0111   positions 1,2,3         -- YuX CPA, dim K = 4.

    R6 rule 2: in characteristic 2 the `xxx1` class is ONLY usable through the
    dim K = 1 combined row, which collapses (W19-B / S10-B).  Those rows are
    printed with a `char-2 unusable` flag.

COST MODEL (exactly Theorem O7's thresholds, `tools/zero_sum_criterion.py`):

  characteristic 2   an active word with an m-dimensional affine subspace costs
                     m bits and contributes 2^m - 1; a free block costs 4n bits
                     and contributes 4 (2^n - 1).
  F_p                an active word is the whole field: log2 p bits, p - 1; a
                     free block 4 log2 p bits, 4 (p - 1).  (Order-2^k cosets
                     are cheaper per bit only before the signed 2^s
                     inclusion-exclusion, which multiplies the point count by
                     2^s; see the summary.)

Since the max-plus profile depends only on WHICH words are active (not on their
subspace dimensions) the search factorises: enumerate active sets, then solve a
tiny convex allocation problem for the cheapest budget that clears the degree.
2^m - 1 is convex, so for a fixed bit budget the threshold is largest at the
extreme allocation (as many words as possible at the full dimension), which is
what `min_bits_for` uses.

Block rotation by one block permutes the profile, so active sets are
canonicalised modulo the four rotations.

Usage
  python experiments/E09_unified_criterion/min_data_v2.py \
      --out results/E09_unified_criterion/min_data_table_v2.md \
      --json results/E09_unified_criterion/min_data_table_v2.json
  python experiments/E09_unified_criterion/min_data_v2.py --instance yu2x-16 \
      --direction dec --layers 12 --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cipher_degree import profile as degree_profile               # noqa: E402

WORDS = 16

# (label, cipher, q, char2, n_or_p)
INSTANCES = [
    ("DuX(65537)", "dux", "65537"),
    ("DuX(2^16)", "dux", "2^16"),
    ("DuX(2^8)", "dux", "2^8"),
    ("YupX-65537", "yux", "65537"),
    ("Yu2X-16", "yux", "2^16"),
    ("Yu2X-8", "yux", "2^8"),
]

# pattern class -> the positions that must be balanced
CLASSES = {
    "1111": (0, 1, 2, 3),
    "xxx1": (3,),
    "1110": (0, 1, 2),
    "0111": (1, 2, 3),
}

# which classes each (cipher, direction) can actually consume, and the dim K
USABLE = {
    ("dux", "dec"): [("1111", 4), ("xxx1", 1)],
    ("yux", "dec"): [("1111", 4), ("1110", 3)],
    ("dux", "enc"): [("1111", 8), ("xxx1", 4), ("1110", 4)],
    ("yux", "enc"): [("1111", 8), ("0111", 4), ("1110", 4)],
}


def field_of(q):
    if q.startswith("2^"):
        n = int(q[2:])
        return True, n, 2 ** n
    p = int(q)
    return False, int(math.log2(p - 1)) + 1, p


# --------------------------------------------------------------- the cost --
def word_cost(char2, n, q):
    """(max bits, threshold contribution at that many bits) for one word."""
    if char2:
        return n, 2 ** n - 1
    return math.log2(q), q - 1


def max_threshold(char2, n, q, s, bits):
    """The largest threshold s active words can reach with `bits` bits of data.

    2^m - 1 is convex in m, so for a fixed sum of dimensions the extreme
    allocation (as many words as possible at the maximum) wins."""
    if not char2:
        return None                        # F_p words are all-or-nothing
    if bits < s or bits > s * n:
        return None
    full, rest = divmod(bits - s, n - 1) if n > 1 else (bits - s, 0)
    dims = [n] * full + ([1 + rest] if full < s else []) + [1] * (s - full - 1)
    dims = (dims + [1] * s)[:s]
    assert sum(dims) == bits, (dims, bits)
    return sum(2 ** m - 1 for m in dims), dims


def min_bits_for(char2, n, q, s, nfree, need):
    """Cheapest (log2 data, per-word dims) with s subspace words and nfree free
    blocks whose threshold strictly exceeds `need`."""
    free_T = nfree * 4 * (2 ** n - 1 if char2 else q - 1)
    free_bits = nfree * 4 * (n if char2 else math.log2(q))
    if not char2:
        T = free_T + s * (q - 1)
        if T <= need:
            return None
        return free_bits + s * math.log2(q), None
    for bits in range(s, s * n + 1):
        got = max_threshold(char2, n, q, s, bits) if s else (0, [])
        if got is None:
            continue
        T, dims = got
        if free_T + T > need:
            return free_bits + bits, dims
    return None


# ------------------------------------------------------------- the search --
def canonical(mask):
    """Smallest block rotation of a 16-bit active mask."""
    best = mask
    m = mask
    for _ in range(3):
        m = ((m << 4) | (m >> 12)) & 0xFFFF
        best = min(best, m)
    return best


def profiles_for(cipher, direction, char2, free_block, layers):
    """{canonical mask: [per-layer 16 degrees]} for every active set."""
    out = {}
    forbidden = 0 if free_block is None else (0xF << (4 * free_block))
    for mask in range(1 << WORDS):
        if mask & forbidden:
            continue
        if not mask and free_block is None:
            continue
        c = canonical(mask) if free_block is None else mask
        if c in out:
            continue
        active = [w for w in range(WORDS) if mask >> w & 1]
        free = () if free_block is None else (free_block,)
        out[c] = degree_profile(active, layers, direction, char2, cipher, free)
    return out


def search(instance, cipher, q, direction, layers, verbose=False):
    char2, n, qn = field_of(q)
    rows = []
    prof_cache = {}
    for free_block in (None, 0):
        key = (free_block,)
        if key not in prof_cache:
            prof_cache[key] = profiles_for(cipher, direction, char2, free_block,
                                           layers)
    for layer in range(1, layers + 1):
        for cls, dimK in USABLE[(cipher, direction)]:
            need_pos = CLASSES[cls]
            best = None
            for free_block in (None, 0):
                nfree = 0 if free_block is None else 1
                for mask, prof in prof_cache[(free_block,)].items():
                    D = prof[layer - 1]
                    need = max(max(D[4 * b + p] for b in range(4))
                               for p in need_pos)
                    s = bin(mask).count("1")
                    got = min_bits_for(char2, n, qn, s, nfree, need)
                    if got is None:
                        continue
                    bits, dims = got
                    if best is None or bits < best["log2_data"]:
                        best = {"log2_data": round(bits, 2),
                                "active": [w for w in range(WORDS)
                                           if mask >> w & 1],
                                "free_blocks": [] if free_block is None
                                               else [free_block],
                                "dims": dims, "s_words": s + 4 * nfree,
                                "degree_needed": need}
            frob = None
            if best is None:
                # is any position at the Frobenius boundary D = q (O9 cell)?
                for free_block in (None, 0):
                    for mask, prof in prof_cache[(free_block,)].items():
                        D = prof[layer - 1]
                        if any(D[i] == qn for i in range(WORDS)):
                            frob = True
                            break
                    if frob:
                        break
            rows.append({"instance": instance, "cipher": cipher, "q": q,
                         "direction": direction, "layer": layer, "class": cls,
                         "dim_K": dimK, "reachable": best is not None,
                         "frobenius_candidate": bool(frob),
                         "char2_dimK1_unusable": char2 and dimK == 1,
                         **(best or {})})
            if verbose:
                b = f"2^{best['log2_data']}" if best else "unreachable"
                print(f"  layer {layer:2d} {cls}: {b}"
                      + (f"  active {best['active']} free {best['free_blocks']}"
                         f" dims {best['dims']}" if best else ""))
    return rows


# ---------------------------------------------------------------- report ---
def markdown(rows):
    out = ["# E09 · minimal-data table v2 inside the O7 framework",
           "",
           "For every (instance, direction, layer l, pattern class) this gives the smallest log2 data",
           "**inside the O7 framework** that makes the pattern hold, with a witness structure. Generated by:",
           "",
           "```bash",
           "python experiments/E09_unified_criterion/min_data_v2.py \\",
           "    --out results/E09_unified_criterion/min_data_table_v2.md \\",
           "    --json results/E09_unified_criterion/min_data_table_v2.json",
           "```",
           "",
           "**Reading the table**: in the `Structure` column, `B0` = block 0 ranges over all of F_q^4 (O11, the first layer is free); the other entries are active word indices;",
           "`Dims` are the affine subspace dimensions of the active words in characteristic 2 (over F_p only the full field).",
           "`dim K` is the dimension of the left kernel of O10 on the pattern; **rows with dim K = 1 are unusable in characteristic 2**,",
           "marked (unusable). `†` = the layer has a Frobenius boundary word with D = q (O9 may rescue it; the criterion alone does not).",
           ""]
    # --- conclusions: the deepest reachable layer per (instance, direction,
    #     class) and its cheapest structure ---------------------------------
    deep = {}
    for r in rows:
        if not r["reachable"]:
            continue
        k = (r["instance"], r["direction"], r["class"])
        if k not in deep or r["layer"] > deep[k]["layer"]:
            deep[k] = r
    out += ["## Summary: deepest reachable layer and cheapest structure per (instance, direction, class)", "",
            "| Instance | Direction | Class | dim K | Deepest layer | Min. data | Structure | Dims |",
            "|---|---|---|---|---|---|---|---|"]
    for (inst, direction, cls), r in sorted(deep.items()):
        st = ", ".join([f"B{b}" for b in r["free_blocks"]]
                       + [str(w) for w in r["active"]]) or "—"
        dims = "—" if r["dims"] is None else ",".join(str(d) for d in r["dims"])
        flag = " (unusable)" if r["char2_dimK1_unusable"] else ""
        out.append(f"| {inst} | {'CCA' if direction == 'dec' else 'CPA'} | "
                   f"`{cls}`{flag} | {r['dim_K']} | **{r['layer']}** | "
                   f"2^{r['log2_data']:g} | {st} | {dims} |")
    out.append("")

    by = {}
    for r in rows:
        by.setdefault((r["instance"], r["direction"]), []).append(r)
    for (inst, direction), rs in by.items():
        model = "chosen ciphertext (CCA)" if direction == "dec" else "chosen plaintext (CPA)"
        out += [f"## {inst} · {model}", "",
                "| Layer | Class | dim K | Min. data | Structure | Dims | D needed |",
                "|---|---|---|---|---|---|---|"]
        for r in rs:
            if not r["reachable"]:
                mark = "†" if r["frobenius_candidate"] else ""
                out.append(f"| {r['layer']} | `{r['class']}` | {r['dim_K']} | "
                           f"**unreachable**{mark} | — | — | — |")
                continue
            st = ", ".join([f"B{b}" for b in r["free_blocks"]]
                           + [str(w) for w in r["active"]])
            dims = "—" if r["dims"] is None else ",".join(str(d) for d in r["dims"])
            flag = " (unusable)" if r["char2_dimK1_unusable"] else ""
            out.append(f"| {r['layer']} | `{r['class']}`{flag} | {r['dim_K']} | "
                       f"2^{r['log2_data']:g} | {st} | {dims} | "
                       f"{r['degree_needed']} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None,
                    help="restrict to one instance label")
    ap.add_argument("--direction", default=None, choices=("dec", "enc"))
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = []
    for label, cipher, q in INSTANCES:
        if a.instance and a.instance != label:
            continue
        for direction in ("dec", "enc"):
            if a.direction and a.direction != direction:
                continue
            print(f"{label} / {direction} ...", flush=True)
            rows += search(label, cipher, q, direction, a.layers, a.verbose)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(rows, open(a.json, "w"), indent=1)
        print("saved", a.json)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(markdown(rows) + "\n")
        print("saved", a.out)


if __name__ == "__main__":
    main()
