"""Y07 / W20 -- the exact-degree grid behind O13.

For every (toy instance, structure) cell this runs `tools/degree_spectrum.py`'s
exact-coefficient machinery on THREE random master keys and reports, per layer
and per state word,

  * the exact total degree of the reduced polynomial,
  * the max-plus bound (`tools/cipher_degree.profile`),
  * whether the three keys agree (KEY-INDEPENDENT is the whole point of O13),
  * and, for single-word structures, whether the common zero set of the
    character sums has any INTERIOR hole -- i.e. whether there is an O9-style
    exploitable zero coefficient below the leading one.

Cells (R6 / W20 step 1):
  yuxtoy-2^4   single word position 0 and 3, two words (0,4), FULL BLOCK 0
  yu2x-8       single word position 0 and 3, two words (0,4)   [layers 1-4]
  yuxtoy-257   the prime-field control: expected exact = max-plus everywhere

The Yu2X-8 full block is 2^32 points and is handed to S14, not run here.

Usage
  python experiments/Y07_char2_exact_degree/run_grid.py \
      --out results/Y07_char2_exact_degree
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import degree_spectrum as DS                                  # noqa: E402
from cipher_degree import profile                             # noqa: E402
from dux.registry import cipher_family, get_cipher            # noqa: E402

SEEDS = (2026, 7, 11)          # three keys for a formal row (docs/measurement_protocol.md)

CELLS = [
    # (instance, label, active words, free block, layers)
    ("yuxtoy-2^4", "1 word, position 0", (0,), None, 5),
    ("yuxtoy-2^4", "1 word, position 3", (3,), None, 5),
    ("yuxtoy-2^4", "2 words (0,4)", (0, 4), None, 5),
    ("yuxtoy-2^4", "full block 0 (2^16)", (), 0, 6),
    ("yu2x-8", "1 word, position 0", (0,), None, 4),
    ("yu2x-8", "1 word, position 3", (3,), None, 4),
    ("yu2x-8", "2 words (0,4)", (0, 4), None, 4),
    ("yuxtoy-257", "1 word, position 0 (F_p control)", (0,), None, 5),
    ("yuxtoy-257", "1 word, position 3 (F_p control)", (3,), None, 5),
    ("yuxtoy-257", "2 words (0,4) (F_p control)", (0, 4), None, 4),
]


def coefficient_arrays(instance, layers, seed, active, free_block):
    """Per layer, the full reduced coefficient array (q,)*s x 16."""
    c = get_cipher(instance)
    F = c.F
    rng = np.random.default_rng(seed)
    rks = c.key_schedule(c.random_key(rng))
    consts = [int(v) for v in rng.integers(0, F.q, size=16)]
    C, axes = DS.multiword_structure(c, rks, active, free_block, consts)
    states = DS.decrypt_layers_fixed_L0(c, C, rks, layers)
    out = []
    for st in states:
        Z = np.stack([np.asarray(st[i], dtype=np.int64) for i in range(16)],
                     axis=1)
        out.append(DS.exact_coefficients(F, Z, len(axes)))
    return c, axes, out


def total_degree_from_coeffs(A, q, s):
    idx = np.arange(q, dtype=np.int64)
    tot = idx.reshape((q,) + (1,) * (s - 1))
    for axis in range(1, s):
        tot = tot + idx.reshape((1,) * axis + (q,) + (1,) * (s - 1 - axis))
    out = []
    for w in range(A.shape[-1]):
        nz = A[..., w] != 0
        out.append(int(tot[nz].max()) if nz.any() else 0)
    return out


def run_cell(instance, label, active, free_block, layers):
    fam = cipher_family(instance)
    t0 = time.time()
    per_key_deg, per_key_coeff = [], []
    axes = None
    c = None
    for seed in SEEDS:
        c, axes, coeffs = coefficient_arrays(instance, layers, seed, active,
                                             free_block)
        s = len(axes)
        per_key_deg.append([total_degree_from_coeffs(A, c.F.q, s)
                            for A in coeffs])
        per_key_coeff.append(coeffs)
    q, s = c.F.q, len(axes)
    sat = s * (q - 1)
    bound = profile(list(active), layers, "dec", c.F.char == 2, fam,
                    () if free_block is None else (free_block,))
    verdicts, holes = [], []
    for l in range(layers):
        row, hrow = [], []
        for w in range(16):
            b = int(bound[l][w])
            ex = [per_key_deg[k][l][w] for k in range(len(SEEDS))]
            if b >= sat:
                row.append("saturated")
            elif len(set(ex)) != 1:
                row.append("KEY-DEPENDENT")
            elif b == 0:
                row.append("constant" if ex[0] == 0 else "LOOSE")
            elif ex[0] == b:
                row.append("tight")
            else:
                row.append("LOOSE")
            # interior holes of the common zero set (single-word cells only)
            if s == 1 and row[-1] not in ("saturated",):
                d = ex[0]
                nz = None
                for e in range(d + 1):
                    allzero = all(int(per_key_coeff[k][l][e, w]) == 0
                                  for k in range(len(SEEDS)))
                    if allzero:
                        nz = (nz or 0) + 1
                hrow.append(nz or 0)
            else:
                hrow.append(None)
        verdicts.append(row)
        holes.append(hrow)
    return {"instance": instance, "cipher": fam, "label": label,
            "active": list(active), "free_block": free_block, "s": s,
            "q": q, "layers": layers, "seeds": list(SEEDS),
            "saturation": sat,
            "exact_per_key": per_key_deg,
            "maxplus_bound": [[int(v) for v in row] for row in bound],
            "verdicts": verdicts,
            "common_interior_zero_coefficients": holes,
            "key_independent": all(v != "KEY-DEPENDENT"
                                   for r in verdicts for v in r),
            "loose_words_per_layer": [r.count("LOOSE") for r in verdicts],
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/Y07_char2_exact_degree")
    ap.add_argument("--only", default=None, help="substring filter on the label")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for inst, label, active, free, layers in CELLS:
        if a.only and a.only not in f"{inst} {label}":
            continue
        print(f"=== {inst} · {label} ===", flush=True)
        r = run_cell(inst, label, active, free, layers)
        rows.append(r)
        for l in range(r["layers"]):
            tally = {v: r["verdicts"][l].count(v)
                     for v in sorted(set(r["verdicts"][l]))}
            print(f"  layer {l + 1}: {tally}  bound {r['maxplus_bound'][l][:4]} "
                  f"exact {r['exact_per_key'][0][l][:4]}", flush=True)
        print(f"  key-independent: {r['key_independent']}   "
              f"loose/layer {r['loose_words_per_layer']}   "
              f"{r['elapsed_s']} s", flush=True)
    path = os.path.join(a.out, "spectrum_grid.json")
    json.dump(rows, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
