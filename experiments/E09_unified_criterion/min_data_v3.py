"""E09 -- the minimal-data table v3.

v2 (`min_data_v2.py`, R6) is kept BYTE FOR BYTE; v3 extends it along the axis
R8 needs:

  * **all fifteen balanced-position classes**, not only the four O10 can
    consume.  Our earlier hand-made joint search reports `1100` cells --
    Yu2X-8 layer 6 at 2^40, Yu2X-16 layer 10 at 2^80 -- and those have to be
    recomputed with the repository's own criterion and then classified.
  * **dim K printed for every class**, from `tools/cheap_rows.py`, so that a
    cell that is reachable as a DISTINGUISHER but useless for key recovery is
    visible as such.  (It turns out `1100` has dim K = 0 in the YuX decryption
    direction, and the plain E07 rows need positions {2,3} for DuX and all four
    for YuX, so those two `1100` cells are distinguisher-only.)
  * the free-block + extra-words search v2 already performs (`free_block` in
    {None, 0} x every subset of the remaining words) is reused unchanged, which
    covers "one free block + at most k extra words" for every k.

Everything else -- the cost model, the convex allocation, the canonicalisation
modulo block rotation -- is imported from v2.

Usage
  python experiments/E09_unified_criterion/min_data_v3.py \
      --out results/E09_unified_criterion/min_data_table_v3.md \
      --json results/E09_unified_criterion/min_data_table_v3.json
  # S17: the encryption (CPA) direction; the report on top compares with the
  # encryption rows of Table 12 (tab:opt) and with v2's encryption cells
  python experiments/E09_unified_criterion/min_data_v3.py --direction enc \
      --out results/E09_unified_criterion/min_data_table_v3_enc.md \
      --json results/E09_unified_criterion/min_data_table_v3_enc.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, os.path.join(_ROOT, "tools"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from min_data_v2 import (INSTANCES, WORDS, field_of, min_bits_for,   # noqa: E402
                         profiles_for)
from cheap_rows import describe as cheap_describe                    # noqa: E402

ALL_CLASSES = ["".join("1" if i in pos else "0" for i in range(4))
               for r in range(1, 5)
               for pos in itertools.combinations(range(4), r)]

# the positions the PLAIN (uncombined) E07 equation reads, per
# (cipher, direction) -- the same rule as
# experiments/E07_key_recovery_2round/weighted.py::used_positions
PLAIN_ROWS = {("dux", "dec"): {2, 3}, ("yux", "dec"): {0, 1, 2, 3},
              ("dux", "enc"): {0, 1, 2, 3}, ("yux", "enc"): {0, 1, 2, 3}}


def dimK_table(cipher, field, direction, cheap=None):
    """dim K per class.  `cheap` overrides the family's cheap set; T1 (R9) uses
    {1, 2} for DuX in characteristic 2 (`experiments/E13_b_coordinate/`), which
    is what turns the classes with positions 1 and 3 balanced -- 0101, 0111,
    1101, 1111, the "x1x1" family -- into usable key-recovery cells."""
    return {cls: cheap_describe(cipher, field, direction,
                                [i for i in range(4) if cls[i] == "1"],
                                cheap)["dim_K"]
            for cls in ALL_CLASSES}


def usability(cipher, direction, cls, dimK, char2, cheap=None):
    """What the key recovery can do with a cell of this class."""
    bal = {i for i in range(4) if cls[i] == "1"}
    plain = PLAIN_ROWS[(cipher, direction)] <= bal
    if dimK == 0 and not plain:
        return "distinguisher only"
    if cheap and tuple(sorted(cheap)) == (1, 2) and char2 and dimK >= 2:
        return f"dim K = {dimK} with the T1 cheap set {{1,2}}"
    if dimK == 1 and char2:
        return "char-2 unusable (dim K = 1 combined row collapses in characteristic 2)"
    if dimK == 0:
        return f"plain rows only ({len(PLAIN_ROWS[(cipher, direction)])} positions)"
    return f"dim K = {dimK}" + (" + plain rows" if plain else "")


def search_all(instance, cipher, q, direction, layers, verbose=False,
               cheap=None):
    char2, n, qn = field_of(q)
    dk = dimK_table(cipher, q, direction, cheap)
    prof = {fb: profiles_for(cipher, direction, char2, fb, layers)
            for fb in (None, 0)}
    rows = []
    for layer in range(1, layers + 1):
        for cls in ALL_CLASSES:
            need_pos = [i for i in range(4) if cls[i] == "1"]
            best = None
            for free_block in (None, 0):
                nfree = 0 if free_block is None else 1
                for mask, pr in prof[free_block].items():
                    D = pr[layer - 1]
                    need = max(max(D[4 * b + p] for b in range(4))
                               for p in need_pos)
                    s = bin(mask).count("1")
                    got = min_bits_for(char2, n, qn, s, nfree, need)
                    if got is None:
                        continue
                    bits, dims = got
                    if best is None or bits < best["log2_data"]:
                        best = {"log2_data": round(bits, 2),
                                "active": [w for w in range(WORDS) if mask >> w & 1],
                                "free_blocks": [] if free_block is None else [free_block],
                                "dims": dims, "s_words": s + 4 * nfree,
                                "extra_words": s, "degree_needed": need}
            rows.append({"instance": instance, "cipher": cipher, "q": q,
                         "direction": direction, "layer": layer, "class": cls,
                         "dim_K": dk[cls],
                         "cheap_set": list(cheap) if cheap else None,
                         "usable": usability(cipher, direction, cls, dk[cls],
                                             char2, cheap),
                         "reachable": best is not None, **(best or {})})
            if verbose and best:
                print(f"  layer {layer:2d} {cls}: 2^{best['log2_data']} "
                      f"(free {best['free_blocks']}, +{best['extra_words']} words, "
                      f"dim K = {dk[cls]})")
    return rows


MEMO_CELLS = [
    # (instance, direction, layer, class, memo's log2 data or None)
    ("Yu2X-8", "dec", 6, "1100", 40.0),
    ("Yu2X-8", "dec", 6, "1111", 56.0),
    ("Yu2X-8", "dec", 6, "1110", 32.0),
    ("Yu2X-8", "dec", 7, None, None),          # unreachable with <= 6 extra words
    ("Yu2X-16", "dec", 10, "1100", 80.0),
    ("Yu2X-16", "dec", 10, "1110", 64.0),
    ("Yu2X-16", "dec", 10, "1111", 112.0),
    ("Yu2X-16", "dec", 11, None, None),
    ("YupX-65537", "dec", 11, None, None),     # >= 9 words = 2^144
    ("DuX(65537)", "dec", 12, None, None),     # ~ 15 words
]


def compare_memo(rows):
    """Cell-by-cell comparison with our earlier hand-made table (MEMO_CELLS)."""
    idx = {(r["instance"], r["direction"], r["layer"], r["class"]): r for r in rows}
    out = []
    for inst, direction, layer, cls, memo in MEMO_CELLS:
        if cls is None:
            reach = [r for r in rows
                     if (r["instance"], r["direction"], r["layer"]) == (inst, direction, layer)
                     and r["reachable"]]
            cheapest = min((r["log2_data"] for r in reach), default=None)
            out.append({"cell": f"{inst} / layer {layer} / any class",
                        "memo": "unreachable (<= 6 extra words)",
                        "v3": ("unreachable" if cheapest is None
                               else f"2^{cheapest} with {min(r['s_words'] for r in reach if r['log2_data'] == cheapest)} words"),
                        "agrees": cheapest is None or cheapest >= 128})
            continue
        r = idx.get((inst, direction, layer, cls))
        got = r["log2_data"] if r and r["reachable"] else None
        out.append({"cell": f"{inst} / layer {layer} / {cls}",
                    "memo": f"2^{memo}", "v3": f"2^{got}" if got is not None else "unreachable",
                    "dim_K": r["dim_K"] if r else None,
                    "usable": r["usable"] if r else None,
                    "agrees": got is not None and abs(got - memo) < 0.01})
    return out


# ------------------------------------------------- encryption direction ---
# Table 12 (appF_optimality.tex, tab:opt) of eprint v1, encryption rows, plus
# its two footnote claims.  (instance, layer, pattern, structure, log2 data)
TABLE12_ENC = [
    ("DuX(65537)", 7, "1110", "full block", 64.0),
    ("DuX(2^16)", 7, "1110", "full block", 64.0),
    ("YupX-65537", 7, "0111", "full block", 64.0),
    ("Yu2X-16", 7, "0111", "full block", 64.0),
    ("DuX(2^8)", 4, None, None, None),          # "cap at layer 4"
    ("Yu2X-8", 4, None, None, None),            # "cap at layer 4"
]
# the pattern classes Table 12 counts as "usable" in the encryption direction
# (appF: 1111, xxx1 = 0001 for DuX, 1110 for both, 0111 for YuX)
TABLE12_USABLE = {"dux": {"1111", "0001", "1110"}, "yux": {"1111", "0111", "1110"}}
# v2 class names -> v3 class names (v2's `xxx1` = position 3 balanced only)
V2_CLASS = {"xxx1": "0001"}


def _structure(r):
    """One-line description of a witness structure of a v3 row."""
    fb = f"free block {r['free_blocks']}" if r.get("free_blocks") else "no free block"
    words = r.get("active") or []
    dims = r.get("dims")
    return (f"{fb}, words {words}" + (f", dims {dims}" if dims else "")
            + f" ({r['s_words']} words, degree {r['degree_needed']})")


def compare_table12(rows):
    """Cell-by-cell comparison with the encryption rows of Table 12 (tab:opt).

    For every instance: the deepest layer with a USABLE class (Table 12's
    notion: 1111, 0001/xxx1 and 1110 for DuX, 1111, 0111 and 1110 for YuX),
    the cheapest structure reaching it, the deepest layer with ANY class the
    key recovery can consume (dim K >= 2, or the plain rows), the deepest
    layer reachable by ANY class at all, and what -- if anything -- reaches
    the layer after the deepest usable one."""
    UNUSABLE = ("distinguisher only", "char-2 unusable")
    out = []
    for inst, t_layer, t_cls, t_struct, t_data in TABLE12_ENC:
        rs = [r for r in rows if r["instance"] == inst and r["direction"] == "enc"]
        if not rs:
            continue
        cipher = rs[0]["cipher"]
        usable = [r for r in rs if r["reachable"] and r["class"] in TABLE12_USABLE[cipher]]
        deepest_u = max(r["layer"] for r in usable)
        at_deep = [r for r in usable if r["layer"] == deepest_u]
        cheapest = min(at_deep, key=lambda r: (r["log2_data"], r["class"]))
        cheapest_all = [r for r in at_deep if r["log2_data"] == cheapest["log2_data"]]
        deepest_kr = max(r["layer"] for r in rs if r["reachable"]
                         and not r["usable"].startswith(UNUSABLE))
        deepest_any = max(r["layer"] for r in rs if r["reachable"])
        next_layer = deepest_u + 1
        next_reach = [r for r in rs if r["layer"] == next_layer and r["reachable"]]
        rec = {"instance": inst,
               "table12": {"layer": t_layer, "pattern": t_cls, "structure": t_struct,
                           "log2_data": t_data},
               "v3_deepest_usable_layer": deepest_u,
               "v3_cheapest_at_deepest": {"classes": sorted(r["class"] for r in cheapest_all),
                                          "log2_data": cheapest["log2_data"],
                                          "structure": _structure(cheapest)},
               "v3_deepest_key_recovery_layer": deepest_kr,
               "v3_deepest_any_class_layer": deepest_any,
               "v3_next_layer": next_layer,
               "v3_next_layer_unreachable": not next_reach,
               "v3_next_layer_cells": [{"class": r["class"], "log2_data": r["log2_data"],
                                        "dim_K": r["dim_K"], "usable": r["usable"],
                                        "structure": _structure(r)} for r in next_reach]}
        # Table 12 is about usable patterns: the deepest usable layer must
        # match, no class the key recovery can consume may go deeper, and for
        # the two explicit rows the pattern and the data must match too.
        rec["agrees"] = (deepest_u == t_layer and deepest_kr == t_layer)
        if t_cls is not None:
            rec["agrees"] = (rec["agrees"] and t_cls in rec["v3_cheapest_at_deepest"]["classes"]
                             and abs(cheapest["log2_data"] - t_data) < 0.01)
        out.append(rec)
    return out


def compare_v2(rows, v2_rows):
    """Cell-by-cell comparison with the encryption rows of min_data_table_v2.

    v2 only searched the classes its key recoveries consume; every such cell is
    matched with the v3 cell of the same (instance, layer, class) and labelled
    same / cheaper / more expensive; a cheaper v3 cell reports its witness."""
    idx = {(r["instance"], r["layer"], r["class"]): r for r in rows if r["direction"] == "enc"}
    out = []
    for v in v2_rows:
        if v["direction"] != "enc":
            continue
        cls = V2_CLASS.get(v["class"], v["class"])
        r = idx[(v["instance"], v["layer"], cls)]
        a = v["log2_data"] if v["reachable"] else None
        b = r["log2_data"] if r["reachable"] else None
        if a == b:
            verdict = "same"
        elif b is None or (a is not None and b > a):
            verdict = "more expensive"
        else:
            verdict = "cheaper"
        out.append({"instance": v["instance"], "layer": v["layer"],
                    "v2_class": v["class"], "class": cls,
                    "v2": a, "v3": b, "verdict": verdict,
                    "v2_dim_K": v["dim_K"], "dim_K": r["dim_K"],
                    "witness": _structure(r) if verdict == "cheaper" else None})
    return out


def _instance_tables(rows):
    """The per-instance / per-layer / per-class tables (same layout as v3 dec)."""
    L = []
    by_inst = {}
    for r in rows:
        by_inst.setdefault((r["instance"], r["direction"]), []).append(r)
    for (inst, direction), rs in by_inst.items():
        L.append(f"### {inst} · {'decryption (CCA)' if direction == 'dec' else 'encryption (CPA)'}")
        L.append("")
        L.append("| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            if not r["reachable"]:
                continue
            L.append(f"| {r['layer']} | `{r['class']}` | {r['dim_K']} | {r['usable']} "
                     f"| 2^{r['log2_data']} | {r['free_blocks'] or '—'} "
                     f"| {r['extra_words']} | {r['dims']} | {r['degree_needed']} |")
        L.append("")
    return L


def markdown_enc(rows, t12, v2cmp):
    """The encryption-direction report: Table 12 comparison on top, then the
    comparison with v2's encryption rows, then the full per-class tables."""
    L = ["# E09 · minimal-data table v3 inside the O7 framework, **encryption (CPA) direction**",
         "",
         "The decryption-direction v3 (`min_data_table_v3.md`) is kept **unchanged**; this file is the output of the same script,",
         "`min_data_v3.py --direction enc`: all 15 balanced-position classes, the dim K of each class",
         "(`tools/cheap_rows.py --direction enc`) and its usability. The search space is that of v2:",
         "0 or 1 free block (O11) x every subset of the remaining words x every layer, active sets canonicalised modulo block rotation;",
         "characteristic 2 allows mixed subspace dimensions.",
         "",
         "## 1. Cell-by-cell comparison with the encryption rows of Table 12 (eprint v1, Appendix F, `tab:opt`)",
         "",
         "Table 12 has two encryption rows, the two DuX instances at layer 7 with a full block `1110` @2^64 and the two YuX instances at layer 7 with a full block",
         "`0111` @2^64, and two footnote sentences: \"DuX(2^8) and Yu2X-8 stop at layer 4 in the encryption direction\" and",
         "\"every other next layer is unreachable for any structure in the search space\". \"Usable classes\" are those of Appendix F: `1111`, `xxx1` (= `0001`) for DuX,",
         "`1110` for both families and `0111` for YuX.",
         "",
         "| Instance | Table 12 | v3: deepest usable layer | Cheapest class / data reaching it | Witness structure | Deepest layer of any key-recovery class | Deepest layer of any class | Next layer | Agrees |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in t12:
        t = c["table12"]
        t_txt = (f"layer {t['layer']} `{t['pattern']}` {t['structure']} 2^{t['log2_data']:g}"
                 if t["pattern"] else f"stops at layer {t['layer']}")
        ch = c["v3_cheapest_at_deepest"]
        if c["v3_next_layer_unreachable"]:
            nxt = f"layer {c['v3_next_layer']} is **unreachable** for any structure"
        else:
            nxt = (f"layer {c['v3_next_layer']} has only "
                   + ", ".join(f"`{x['class']}` @2^{x['log2_data']} (dim K = {x['dim_K']}, {x['usable']})"
                               for x in c["v3_next_layer_cells"]))
        L.append(f"| {c['instance']} | {t_txt} | {c['v3_deepest_usable_layer']} "
                 f"| {' / '.join('`' + k + '`' for k in ch['classes'])} @2^{ch['log2_data']} "
                 f"| {ch['structure']} | {c['v3_deepest_key_recovery_layer']} "
                 f"| {c['v3_deepest_any_class_layer']} | {nxt} "
                 f"| {'✅' if c['agrees'] else '**no**'} |")
    L += ["",
          "\"Usable\" follows Table 12 (the classes listed in Appendix F); \"any key-recovery class\" means the classes among the 15 with dim K >= 2 or with plain rows;",
          "\"any class\" includes the distinguisher-only classes and the unusable dim K = 1 classes in characteristic 2. The two footnote sentences of Table 12 are read over the usable classes:",
          "if the next layer is reachable only by unusable classes, \"stops at layer l\" still holds, and the reachable cells are listed in the \"next layer\" column.",
          "",
          "**The YuX layer-7 full-block cell**: the criterion gives `0111` (the margin at position 0 is exactly 0: D = 8^6 = 262 144 = T = 4 x 65 536).",
          "By the rank-3 argument of `experiments/Y09_topform_constants/run.py --part rank` (the top forms of YuX in the encryption direction lie in",
          "F_p[y0, y1, sigma], with fibres of size p), **the sum at that boundary position is always 0**, so the paper writes this cell as `1111`",
          "(O14-CPA); this table keeps the criterion's own `0111` and is not edited by hand. The corresponding eight rows of DuX have rank 4, so there is no such shortcut;",
          "the constant c_b(65537) at position 3 of layer 7 of DuX(65537) was not computed, and `1110` stands.",
          "",
          "## 2. Cell-by-cell comparison with the encryption rows of `min_data_table_v2.md`",
          ""]
    same = [c for c in v2cmp if c["verdict"] == "same"]
    cheaper = [c for c in v2cmp if c["verdict"] == "cheaper"]
    dearer = [c for c in v2cmp if c["verdict"] == "more expensive"]
    L.append(f"v2 has {len(v2cmp)} encryption cells"
             + (" (6 instances x 12 layers x 3 classes" if len(v2cmp) == 216 else " (")
             + "; v2's `xxx1` corresponds to v3's `0001`):")
    L.append(f"**{len(same)} identical, {len(cheaper)} cheaper, {len(dearer)} more expensive**.")
    L.append("")
    if cheaper:
        L += ["### Cheaper cells (witness structures)", "",
              "| Instance | Layer | Class | v2 | v3 | Witness structure |", "|---|---|---|---|---|---|"]
        for c in cheaper:
            L.append(f"| {c['instance']} | {c['layer']} | `{c['class']}` | "
                     f"{'2^' + str(c['v2']) if c['v2'] is not None else 'unreachable'} | "
                     f"2^{c['v3']} | {c['witness']} |")
        L.append("")
    if dearer:
        L += ["### More expensive cells", "",
              "| Instance | Layer | Class | v2 | v3 |", "|---|---|---|---|---|"]
        for c in dearer:
            L.append(f"| {c['instance']} | {c['layer']} | `{c['class']}` | 2^{c['v2']} | "
                     f"{'2^' + str(c['v3']) if c['v3'] is not None else 'unreachable'} |")
        L.append("")
    L += ["### Identical cells", "",
          "| Instance | Class (v2 / v3) | Min. data at layers 1-12 (log2; `—` = unreachable) | dim K (v2 / v3) |",
          "|---|---|---|---|"]
    keyed = {}
    for c in same:
        keyed.setdefault((c["instance"], c["v2_class"], c["class"], c["v2_dim_K"], c["dim_K"]), {})[c["layer"]] = c["v3"]
    for (inst, vc, cls, dk2, dk3), cells in keyed.items():
        vals = ", ".join("—" if cells.get(l) is None else f"{cells[l]:g}" for l in range(1, 13))
        L.append(f"| {inst} | `{vc}` / `{cls}` | {vals} | {dk2} / {dk3} |")
    L += ["", "## 3. Minimal data per instance, layer and class", "",
          "Only reachable cells are listed; a layer marked `—` is unreachable inside the O7 framework.", ""]
    L += _instance_tables(rows)
    return "\n".join(L)


def markdown(rows, cmp_rows):
    L = ["# E09 · minimal-data table v3 inside the O7 framework",
         "",
         "v2 (`min_data_table_v2.md`) is kept **unchanged**; v3 extends the search to **all 15 balanced-position classes**",
         "and prints for each class the dim K of `tools/cheap_rows.py` and its usability for key recovery.",
         "The search space is that of v2: 0 or 1 free block (O11) x every subset of the remaining words x every layer,",
         "active sets canonicalised modulo block rotation; characteristic 2 allows mixed subspace dimensions (convexity => the extreme allocation is optimal).",
         "",
         "## 1. Cell-by-cell comparison with the earlier hand-made table",
         "",
         "| Cell | Hand-made | v3 | dim K | Usability | Agrees |",
         "|---|---|---|---|---|---|"]
    for c in cmp_rows:
        L.append(f"| {c['cell']} | {c['memo']} | {c['v3']} | {c.get('dim_K', '—')} "
                 f"| {c.get('usable', '—')} | {'✅' if c['agrees'] else '**no**'} |")
    L += ["", "## 2. Minimal data per instance, layer and class", "",
          "Only reachable cells are listed; a layer marked `—` is unreachable inside the O7 framework.", ""]
    by_inst = {}
    for r in rows:
        by_inst.setdefault((r["instance"], r["direction"]), []).append(r)
    for (inst, direction), rs in by_inst.items():
        L.append(f"### {inst} · {'decryption (CCA)' if direction == 'dec' else 'encryption (CPA)'}")
        L.append("")
        L.append("| Layer | Class | dim K | Usability | Min. data | Free block | Extra words | Dims | Degree needed |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            if not r["reachable"]:
                continue
            L.append(f"| {r['layer']} | `{r['class']}` | {r['dim_K']} | {r['usable']} "
                     f"| 2^{r['log2_data']} | {r['free_blocks'] or '—'} "
                     f"| {r['extra_words']} | {r['dims']} | {r['degree_needed']} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--direction", default="dec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--v2-json",
                    default=os.path.join(_ROOT, "results", "E09_unified_criterion",
                                         "min_data_table_v2.json"),
                    help="v2 rows to compare against (encryption direction only)")
    a = ap.parse_args(argv)
    rows = []
    for label, cipher, q in INSTANCES:
        if a.verbose:
            print(f"{label} / {a.direction}")
        rows += search_all(label, cipher, q, a.direction, a.layers, a.verbose)
    if a.direction == "enc":
        # S17: the memo's cells are all decryption-direction, so the
        # encryption report compares with Table 12 and with v2 instead.
        t12 = compare_table12(rows)
        with open(a.v2_json) as fh:
            v2cmp = compare_v2(rows, json.load(fh))
        print("\n=== comparison with Table 12 (tab:opt), encryption rows ===")
        for c in t12:
            ch = c["v3_cheapest_at_deepest"]
            print(f"  {c['instance']:12s} deepest usable layer {c['v3_deepest_usable_layer']} "
                  f"({'/'.join(ch['classes'])} @2^{ch['log2_data']}), key-recovery class "
                  f"{c['v3_deepest_key_recovery_layer']}, any class "
                  f"{c['v3_deepest_any_class_layer']}, layer {c['v3_next_layer']} "
                  + ("unreachable" if c["v3_next_layer_unreachable"] else
                     "only " + ", ".join(f"{x['class']}@2^{x['log2_data']} ({x['usable']})"
                                         for x in c["v3_next_layer_cells"]))
                  + f"  {'ok' if c['agrees'] else 'MISMATCH'}")
        n = {k: sum(1 for c in v2cmp if c["verdict"] == k)
             for k in ("same", "cheaper", "more expensive")}
        print(f"=== comparison with v2 encryption rows: {n} ===")
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as fh:
                fh.write(markdown_enc(rows, t12, v2cmp) + "\n")
            print(f"wrote {a.out}")
        if a.json:
            with open(a.json, "w") as fh:
                json.dump({"rows": rows, "table12_comparison": t12,
                           "v2_comparison": v2cmp}, fh, indent=1)
            print(f"wrote {a.json}")
        return rows
    cmp_rows = compare_memo(rows)
    print("\n=== comparison with the earlier hand-made table ===")
    for c in cmp_rows:
        print(f"  {c['cell']:44s} memo {c['memo']:<12s} v3 {c['v3']:<28s} "
              f"{'ok' if c['agrees'] else 'MISMATCH'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(markdown(rows, cmp_rows) + "\n")
        print(f"wrote {a.out}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"rows": rows, "memo_comparison": cmp_rows}, fh, indent=1)
        print(f"wrote {a.json}")
    return rows


if __name__ == "__main__":
    main()
