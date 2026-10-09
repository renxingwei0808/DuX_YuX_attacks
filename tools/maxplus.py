"""Per-position max-exponent (F_q-degree) upper bounds for DuX / YuX decryption.

Model (docs/glossary.md, O3).  Each state word is a polynomial in one
active ciphertext word X.  Track e_p = max exponent at block position p.
One S^{-1} layer (degrees from the closed form in dux/sbox.py):
    d0 = max(e1+e2, e0)        # y0 = x1 x2 + x0 + a
    d3 = max(e0+e1, e3)        # y3 = x0 x1 + x3 + a
    d1 = max(e2+d3, e1)        # y1 = x2 y3 + x1 + a
    d2 = max(d3+d0, e2)        # y2 = y3 y0 + x2 + a
L^{-1} (rotations {1,4,8,9,13} / {1,5,8,12,13}) maps position p to the two
positions {p, p+1} (mod 4) of every block, so the next-layer input is
    e'_p = max over blocks of max(d_p, d_{p+1}).
For F_p the zero-sum over all of F_p holds while max exponent < p-1.
For F_{2^n} the Boolean degree is bounded by floor(log2(e+1)) (Ni et al.
Lemma 1) and by n; this is only a coarse model there (use
tools/exponent_sets.py instead).

Two granularities:
  * `run_all_active`: all 16 words start with degree 1 (worst case after
    full diffusion).
  * `run_single_word`: only one word starts active (position `pos` of block
    0); other words are constants (degree 0).  Blocks are tracked
    separately for the first layers until the bound merges.

`--cipher yux` swaps in YuX's S^{-1} = Pf^4 recurrence and its 7-term rotation
set {0,3,4,8,9,12,14} (tools/cipher_degree.py); the default `dux` reproduces
every earlier number.  `--full-block b` starts the recurrence with a whole
block at degree (1,1,1,1) because S^{-1} is a bijection of F_q^4 (O11).

Usage:  python tools/maxplus.py --q 65537 --pos 3 --layers 12
        python tools/maxplus.py --cipher yux --q 65537 --pos 0 --layers 11
        python tools/maxplus.py --cipher yux --q 65537 --full-block 0 --layers 11
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cipher_degree import CIPHERS  # noqa: E402


def sbox_inv_degrees(e, cipher="dux"):
    return CIPHERS[cipher]["sinv_deg"](e)


ROT_INV = {0: (1, 4, 8, 9, 13), 1: (1, 5, 8, 12, 13)}


def rot_inv(cipher):
    return {0: CIPHERS[cipher]["ROT_INV"], 1: CIPHERS[cipher]["ROT_INV_L1"]}


def linear_bound(blocks, rots):
    """blocks: list of 4 lists (per-block per-position degrees).
    Returns the new per-block degrees after word i <- sum_j X_{i+j}."""
    flat = [d for b in blocks for d in b]
    new = [max(flat[(i + j) % 16] for j in rots) for i in range(16)]
    return [new[4 * b:4 * b + 4] for b in range(4)]


def run(start_blocks, layers, t_seq=None, cipher="dux", free_blocks=()):
    """start_blocks: 4 lists of 4 ints (degree of each word entering layer 1).
    t_seq: per-layer choice of L0/L1 (default all L0; O2 says it does not
    matter for the bound, but we allow it for checking).
    free_blocks: whole blocks running over F_q^4, which skip the first S layer
    (O11) and enter the recurrence at (1,1,1,1)."""
    rows = []
    rots = rot_inv(cipher)
    blocks = [list(b) for b in start_blocks]
    free_blocks = tuple(sorted(set(free_blocks)))
    for k in range(layers):
        outs = [([1, 1, 1, 1] if (k == 0 and b in free_blocks)
                 else sbox_inv_degrees(blocks[b], cipher)) for b in range(4)]
        rows.append({"layer": k + 1,
                     "sbox_out_per_block": outs,
                     "sbox_out_max_per_pos": [max(o[p] for o in outs) for p in range(4)]})
        t = 0 if t_seq is None else t_seq[k]
        blocks = linear_bound(outs, rots[t])
    return rows


def run_all_active(layers, cipher="dux"):
    return run([[1, 1, 1, 1]] * 4, layers, cipher=cipher)


def run_single_word(pos, layers, cipher="dux"):
    start = [[0, 0, 0, 0] for _ in range(4)]
    start[0][pos] = 1
    return run(start, layers, cipher=cipher)


def run_full_block(blocks, layers, cipher="dux", extra=()):
    start = [[0, 0, 0, 0] for _ in range(4)]
    for w in extra:
        start[w // 4][w % 4] = 1
    return run(start, layers, cipher=cipher, free_blocks=blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cipher", default="dux", choices=tuple(CIPHERS))
    ap.add_argument("--q", type=int, default=65537)
    ap.add_argument("--pos", type=int, default=3, help="active position (0-3) or -1 for all-active")
    ap.add_argument("--full-block", default=None,
                    help="comma list of whole blocks over F_q^4 (O11); "
                         "with --extra-words for additional single words")
    ap.add_argument("--extra-words", default=None)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()
    if a.full_block:
        blocks = [int(v) for v in a.full_block.split(",")]
        extra = [int(v) for v in a.extra_words.split(",")] if a.extra_words else []
        rows = run_full_block(blocks, a.layers, a.cipher, extra)
    elif a.pos < 0:
        rows = run_all_active(a.layers, a.cipher)
    else:
        rows = run_single_word(a.pos, a.layers, a.cipher)
    limit = a.q - 1
    print(f"q={a.q}  zero-sum over all of F_q needs max exponent < q-1 = {limit}")
    print("layer | max exponent per position (0,1,2,3) | positions below q-1")
    for r in rows:
        m = r["sbox_out_max_per_pos"]
        ok = [p for p in range(4) if m[p] < limit]
        print(f"{r['layer']:5d} | {m} | {ok}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
