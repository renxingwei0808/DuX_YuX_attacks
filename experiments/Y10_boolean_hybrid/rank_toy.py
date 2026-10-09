"""Y10 (W26 step 3, R8): does a WEIGHTED row carry rank inside a Boolean
linearised system?

This is the gate for the hybrid route (our full-block zero sum combined with
Ni et al.'s Boolean linearisation).  That route feeds our full-block zero sum to Ni et al.'s Boolean
linearisation and counts (1 + N_w) x (#balanced words) x n equations per
structure instead of Ni's (#balanced words) x n.  The count is only worth
anything if the weighted rows are LINEARLY INDEPENDENT of the plain ones --
Proposition 5 of the paper (and the characteristic-2 collapse of W19-B) are
two cases where extra rows turned out to carry no new rank at all.

Setting (yuxtoy-2^4, F_{2^4}):
  * chosen-ciphertext structure on two active words, 256 points, layer 3, where
    the criterion gives the pattern `1110` with margins (15, 14, 7, -1) -- the
    same SHAPE as the Yu2X-8 layer-6 full-block cell (508, 508, 252, -4) that
    the hybrid route would use, at 1/256 of the cost.  The full-block analogue
    on this toy is layer 4 (margins 28, 28, 12, -4) at 2^16 points; with a
    2^16-point structure the 2^16 candidate keys of the Moebius step make the
    double loop 2^32 per structure, which does not fit in this environment, so
    the smaller cell is used and the deviation is recorded.
  * two encryption rounds from the plaintext side (r_KR = 2), with ONE block of
    rk^0 symbolic: 16 Boolean key bits, and the rest of the key known.
  * for every balanced word i and every weight a, the row is the exact ANF (a
    Moebius transform over the 2^16 candidate blocks) of
        sum_x  x_0^a * Z_i(x, k)      as a Boolean function of the 16 bits,
    split into n = 4 rows over F_2 (one per output bit).

Measured: the rank of the accumulated rows as a function of the number of
structures, with weights and without, and the saturation point.

Usage
  python experiments/Y10_boolean_hybrid/rank_toy.py --structures 12 --weights 8 \
      --keys 2 --out results/Y10_boolean_hybrid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, os.path.join(_ROOT, "tools"),
           os.path.join(_ROOT, "experiments", "E07_key_recovery_2round")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dux.registry import get_cipher                 # noqa: E402
import zero_sum_criterion as zsc                    # noqa: E402

WORDS = 16
NBITS = 16                      # one block of rk^0 over F_{2^4}


def cell(instance, active, layers, cipher):
    """The criterion's verdict for this cell: pattern and per-position margins."""
    F = get_cipher(instance).F
    q = f"2^{F.n}" if F.char == 2 else F.q
    r = zsc.patterns(q, list(active), layers, cipher=cipher)
    return {"pattern": r["patterns"][layers - 1],
            "margins": r["margin_per_position"][layers - 1],
            "threshold": r["threshold"],
            "D": r["max_degree_per_position"][layers - 1]}


def moebius_rows(c, rks, act, layers, seed, weights_per_pos, balanced, progress=False):
    """One structure -> the exact ANF rows.

    Returns a dict {(word, weight): (nbits-bit ANF as a uint64 bitmap array of
    length 2^NBITS)} ... concretely an array `A` of shape
    (#rows, 2^NBITS) over F_2, one row per (balanced word, weight, output bit).
    """
    F = c.F
    q = F.q
    rng = np.random.default_rng(seed)
    s = len(act)
    npts = q ** s
    # the chosen-ciphertext structure, decrypted with the TRUE key
    consts = [int(v) for v in rng.integers(0, q, size=WORDS)]
    idx = np.arange(npts, dtype=np.int64)
    C = []
    for i in range(WORDS):
        if i in act:
            j = act.index(i)
            C.append((idx // (q ** (s - 1 - j))) % q)
        else:
            C.append(np.full(npts, consts[i], dtype=np.int64))
    P = c.decrypt(tuple(C), rks, rounds=c.r, vec=True)
    P = [np.asarray(v, dtype=np.int64) for v in P]
    x0 = np.asarray(C[act[0]], dtype=np.int64)

    # all 2^16 candidate values of block 0 of rk^0
    K = 1 << NBITS
    kidx = np.arange(K, dtype=np.int64)
    kw = [(kidx >> (4 * t)) & 0xF for t in range(4)]

    nw = max(weights_per_pos.values())
    acc = np.zeros((WORDS, nw, K), dtype=np.int64)
    for t in range(npts):
        st = []
        for i in range(WORDS):
            if i < 4:
                st.append(F.vadd(np.full(K, int(P[i][t]), dtype=np.int64), kw[i]))
            else:
                st.append(np.full(K, F.add(int(P[i][t]), int(rks[0][i])), dtype=np.int64))
        st = c.SL(tuple(st), True)
        st = c.lin.L(st, 0, True)
        st = tuple(F.vadd(st[i], np.int64(rks[1][i])) for i in range(WORDS))
        st = c.SL(st, True)
        Z = np.stack([np.asarray(v, dtype=np.int64) for v in st])       # (16, K)
        p_ = 1
        for a in range(nw):
            if a:
                p_ = F.mul(p_, int(x0[t]))
            if p_ == 0:
                continue
            acc[:, a, :] ^= F.vmul(Z, np.int64(p_))
        if progress and (t + 1) % 64 == 0:
            print(f"      point {t + 1}/{npts}", flush=True)

    rows, tags = [], []
    for i in balanced:
        nwi = weights_per_pos[i % 4]
        for a in range(nwi):
            for b in range(F.n):
                f = ((acc[i, a] >> b) & 1).astype(np.uint8)
                g = f.copy()
                for j in range(NBITS):
                    step = 1 << j
                    g = g.reshape(-1, 2 * step)
                    g[:, step:] ^= g[:, :step]
                    g = g.reshape(-1)
                rows.append(g)
                tags.append((i, a, b))
    return np.asarray(rows, dtype=np.uint8), tags


def pack(rows):
    """Bit-pack an F_2 matrix into uint64 words for the rank routine."""
    n, m = rows.shape
    w = (m + 63) // 64
    out = np.zeros((n, w), dtype=np.uint64)
    padded = np.zeros((n, w * 64), dtype=np.uint8)
    padded[:, :m] = rows
    bits = padded.reshape(n, w, 64)
    for k in range(64):
        out |= (bits[:, :, k].astype(np.uint64) << np.uint64(k))
    return out


def gf2_rank(M):
    """Rank over F_2 of a bit-packed matrix (destroys M)."""
    n, w = M.shape
    r = 0
    for col in range(w * 64):
        wi, bi = col >> 6, np.uint64(col & 63)
        mask = np.uint64(1) << bi
        piv = None
        nz = np.flatnonzero(M[r:, wi] & mask)
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        hit = np.flatnonzero(M[:, wi] & mask)
        hit = hit[hit != r]
        if hit.size:
            M[hit] ^= M[r]
        r += 1
        if r == n:
            break
    return r


def degree_columns(nbits, d):
    """Indices of the ANF columns of Hamming weight <= d."""
    v = np.arange(1 << nbits, dtype=np.int64)
    hw = np.zeros_like(v)
    u = v.copy()
    while u.any():
        hw += u & 1
        u >>= 1
    return np.flatnonzero(hw <= d), hw


def run(instance="yuxtoy-2^4", active=(0, 4), layers=3, structures=12,
        nweights=8, keys=2, seed=2026, dmax=None, progress=False, checkpoint=2):
    cipher = "yux" if instance.startswith(("yu2x", "yupx", "yuxtoy")) else "dux"
    info = cell(instance, active, layers, cipher)
    balanced_pos = [p for p in range(4) if info["pattern"][p] == "1"]
    balanced = [4 * b + p for b in range(4) for p in balanced_pos]
    wpp = {p: max(1, min(nweights, info["margins"][p])) for p in range(4)}
    print(f"{instance} layer {layers}, active {list(active)}: pattern "
          f"{info['pattern']}, margins {info['margins']}")
    print(f"  balanced words {balanced}; weights per position {wpp}")
    out = {"instance": instance, "active": list(active), "layers": layers,
           "cell": info, "balanced_words": balanced, "weights_per_position": wpp,
           "structures": structures, "keys": []}
    _, hw = degree_columns(NBITS, NBITS)
    for kk in range(keys):
        c = get_cipher(instance, rounds=layers + 2)
        rng = np.random.default_rng(seed + kk)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        allrows, tags = [], []
        curve = []
        t0 = time.time()
        for st in range(structures):
            R, T = moebius_rows(c, rks, list(active), layers,
                                seed + 5000 + 97 * kk + st, wpp, balanced,
                                progress=progress)
            allrows.append(R)
            tags += T
            if (st + 1) % checkpoint and st + 1 != structures:
                print(f"  key {kk} structures {st + 1:2d}: assembled "
                      f"({time.time() - t0:.0f}s)", flush=True)
                continue
            A = np.concatenate(allrows, axis=0)
            tg = np.array([t[1] for t in tags])
            deg = int(hw[np.flatnonzero(A.any(axis=0))].max()) if A.any() else -1
            r_all = gf2_rank(pack(A))
            r_plain = gf2_rank(pack(A[tg == 0]))
            curve.append({"structures": st + 1, "rows_all": int(A.shape[0]),
                          "rank_all": r_all,
                          "rows_plain": int((tg == 0).sum()), "rank_plain": r_plain,
                          "anf_degree_seen": deg})
            print(f"  key {kk} structures {st + 1:2d}: rows {A.shape[0]:5d} "
                  f"rank {r_all:5d} | a=0 rows {(tg == 0).sum():4d} rank {r_plain:4d} "
                  f"| ANF degree {deg}  ({time.time() - t0:.0f}s)", flush=True)
        out["keys"].append({"key": kk, "seed": seed + kk, "curve": curve})
    # verdict
    last = [k["curve"][-1] for k in out["keys"]]
    out["weights_add_rank"] = all(c["rank_all"] > c["rank_plain"] for c in last)
    out["rank_per_row_all"] = [round(c["rank_all"] / c["rows_all"], 4) for c in last]
    out["rank_per_row_plain"] = [round(c["rank_plain"] / c["rows_plain"], 4) for c in last]
    out["identical_across_keys"] = len({tuple((c["rank_all"], c["rank_plain"])) for c in last}) == 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default="yuxtoy-2^4")
    ap.add_argument("--active", default="0,4")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--structures", type=int, default=12)
    ap.add_argument("--weights", type=int, default=8)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--checkpoint", type=int, default=2,
                    help="compute the rank every N structures (the F_2 "
                         "elimination is the expensive part)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = run(a.instance, tuple(int(v) for v in a.active.split(",")), a.layers,
              a.structures, a.weights, a.keys, a.seed, progress=a.progress,
              checkpoint=a.checkpoint)
    print(f"\n  weights add rank: {res['weights_add_rank']}")
    print(f"  rank per row, all rows   : {res['rank_per_row_all']}")
    print(f"  rank per row, a = 0 only : {res['rank_per_row_plain']}")
    print(f"  identical across keys    : {res['identical_across_keys']}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "rank_toy.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"wrote {a.out}/rank_toy.json")
    return res


if __name__ == "__main__":
    main()
