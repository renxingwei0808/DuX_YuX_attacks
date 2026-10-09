"""Y07 / S14 step 3 -- exact degrees of a TWO-WORD structure at 2^32.

W20 computed the exact reduced degrees of the paper instances symbolically, but
only for SINGLE-WORD structures (`tools/char2_exact_degree.py`); the multi-word
and full-block cells of its per-cell verdicts use a gap EXTRAPOLATED from the
toys (about 1.26 % for two words, 0 % for a full block,
against 6.1 % for one word).  This script measures the two-word case directly.

The identity.  For the product structure {(x0, x1)} = F_q x F_q and a weight on
the first axis,

    sum_{x0, x1} x0^a Z(x0, x1)
        = sum_{d0, d1} c_{d0, d1} (sum_{x0} x0^{a+d0}) (sum_{x1} x1^{d1}) ,

and sum_{x in F_q} x^m is nonzero exactly when (q-1) | m and m > 0.  Reduced
exponents live in [0, q-1], so exactly one term survives: the sum equals
+- c_{q-1-a, q-1}.  Scanning a upward from 0 therefore reads off the
coefficients of x0^{q-1-a} x1^{q-1}, i.e. of TOTAL degree T - a with
T = 2(q-1), and the first nonzero one gives

    D_exact  >=  T - a*        (a lower bound on the total degree, which is
                                what the criterion needs: it must not exceed
                                T - a for the weights the attack uses).

Everything the scan needs from the structure is a per-slice sum, so one
decryption pass over the q^2 points produces BOTH axes at once:

    rows[v0][i] = sum_{x1} Z_i(v0, x1)      cols[v1][i] = sum_{x0} Z_i(x0, v1)

and the weighted sums are two (q x N_a) Vandermonde products afterwards.  At
2^32 that pass is ~13 core-hours per key in numpy, which is why it is chunked
across a process pool rather than done in one array.

    python3 experiments/Y07_char2_exact_degree/exact_degree_2word.py \
        --instance yu2x-16 --active 0,4 --layers 10 --rounds 12 --keys 3 \
        --max-weight 3000 --procs 76 --out results/Y07_char2_exact_degree
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))

from dux.registry import cipher_family, get_cipher     # noqa: E402
from cipher_degree import profile                      # noqa: E402
from zero_sum_criterion import threshold               # noqa: E402

_W = {}


def _init(instance, rounds, key_seed, struct_seed, active, layers, words):
    try:
        from threadpoolctl import threadpool_limits
        _W["_tp"] = threadpool_limits(limits=1)
    except Exception:
        pass
    c = get_cipher(instance, rounds=rounds)
    rng = np.random.default_rng(key_seed)
    rks = c.key_schedule(c.random_key(rng))
    # the same constants `structure_data` would draw for this seed
    consts = [int(v) for v in np.random.default_rng(struct_seed)
              .integers(0, c.F.q, size=16)]
    _W.update(c=c, rks=rks, rounds=rounds, active=active, seed=struct_seed,
              layers=layers, words=words, consts=consts)


def _block(arg):
    """Row sums for the slices [v0, v1) and this block's share of the column
    sums, from one decryption pass over its (v1 - v0) * q points."""
    v0, v1 = arg
    c, rks = _W["c"], _W["rks"]
    F, q = c.F, c.F.q
    layers, words = _W["layers"], _W["words"]
    nw = len(words)
    rows = np.zeros((v1 - v0, nw), dtype=np.int64)
    cols = np.zeros((q, nw), dtype=np.int64)
    # the CIPHERTEXT block, in `structure_data`'s product-set order (axis 0
    # slowest).  `structure_stream` would hand back the fully decrypted
    # plaintext; what the degree scan needs is the state after `layers` inverse
    # layers, so the structure is rebuilt here and fed to `decrypt_layers`.
    act = _W["active"]
    consts = _W["consts"]
    idx = np.arange(v0 * q, v1 * q, dtype=np.int64)
    axes = [(idx // q) % q, idx % q]
    C = [np.ascontiguousarray(axes[act.index(i)]) if i in act
         else np.full(idx.size, consts[i], dtype=np.int64) for i in range(16)]
    st = c.decrypt_layers(tuple(C), rks, layers, vec=True, yield_all=True)[-1]
    for j, i in enumerate(words):
        Z = np.asarray(st[i], dtype=np.int64).reshape(v1 - v0, q)
        if F.char == 2:
            rows[:, j] = np.bitwise_xor.reduce(Z, axis=1)
            cols[:, j] = np.bitwise_xor.reduce(Z, axis=0)
        else:
            rows[:, j] = Z.sum(axis=1) % F.p
            cols[:, j] = Z.sum(axis=0) % F.p
    return v0, v1, rows, cols


def first_nonzero(F, vals, table, amax, progress=0):
    """For every column of `table` (one per state word), the smallest a >= 0
    with sum_v v^a table[v] != 0, scanning the powers incrementally."""
    n = table.shape[1]
    out = [None] * n
    w = np.ones(len(vals), dtype=np.int64)
    for a in range(amax + 1):
        if F.char == 2:
            s = np.bitwise_xor.reduce(F.vmul(table, w[:, None]), axis=0)
        else:
            s = (table * w[:, None] % F.p).sum(axis=0) % F.p
        for j in range(n):
            if out[j] is None and int(s[j]):
                out[j] = a
        if all(v is not None for v in out):
            break
        w = F.vmul(w, vals) if F.char == 2 else (w * vals) % F.p
        if progress and a % progress == 0:
            print(f"    a = {a}: {sum(v is not None for v in out)}/{n} words "
                  f"done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="yu2x-16")
    ap.add_argument("--active", default="0,4")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--words", default=None, help="comma list (default: all 16)")
    ap.add_argument("--max-weight", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--block-slices", type=int, default=64)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="save the running row/column sums here every "
                         "--checkpoint-every tasks and resume from them.  The "
                         "machine these runs live on kills long jobs every few "
                         "hours and one pass over 2^32 points is hours, so "
                         "without this the scan never finishes.")
    ap.add_argument("--checkpoint-every", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    act = [int(v) for v in a.active.split(",")]
    assert len(act) == 2, "this is the two-word scan"
    words = ([int(v) for v in a.words.split(",")] if a.words else list(range(16)))
    c = get_cipher(a.instance, rounds=a.rounds)
    fam = cipher_family(a.instance)
    F, q = c.F, c.F.q
    fkey = f"2^{F.n}" if F.char == 2 else q
    T = threshold(fkey, act, None, None, [])
    prof = profile(act, a.layers, "dec", F.char == 2, fam, [])
    vals = np.arange(q, dtype=np.int64)
    print(f"{a.instance} ({F.name}): 2-word structure {act}, {q*q} = 2^"
          f"{2*np.log2(q):.0f} points, layer {a.layers}, T = {T}, "
          f"{a.keys} keys, scan a <= {a.max_weight}", flush=True)

    from multiprocessing import Pool
    per_key, t0 = [], time.time()
    for k in range(a.keys):
        key_seed = a.seed + k        # same convention as the other Y07 scripts
        rowsum = np.zeros((q, len(words)), dtype=np.int64)
        colsum = np.zeros((q, len(words)), dtype=np.int64)
        tasks = [(v, min(v + a.block_slices, q))
                 for v in range(0, q, a.block_slices)]
        ckpt = (os.path.join(a.checkpoint_dir, f"{a.tag or 'tw'}_key{k}.npz")
                if a.checkpoint_dir else None)
        pending = list(range(len(tasks)))
        if ckpt and os.path.exists(ckpt):
            try:
                z = np.load(ckpt)
                rowsum, colsum = z["rowsum"], z["colsum"]
                mask = z["mask"]
                pending = [i for i in range(len(tasks)) if not mask[i]]
                print(f"  key {k}: resumed from {ckpt}, "
                      f"{len(tasks) - len(pending)}/{len(tasks)} tasks already "
                      f"done", flush=True)
            except Exception as e:
                print(f"  key {k}: checkpoint {ckpt} unusable ({e})", flush=True)
                pending = list(range(len(tasks)))
        mask = np.zeros(len(tasks), dtype=bool)
        mask[[i for i in range(len(tasks)) if i not in set(pending)]] = True
        idx_of = {tasks[i]: i for i in range(len(tasks))}
        with Pool(a.procs, initializer=_init,
                  initargs=(a.instance, c.r, key_seed, a.seed + 5000, act,
                            a.layers, words)) as pool:
            done = int(mask.sum()) * a.block_slices
            since = 0
            for v0, v1, rows, cols in pool.imap_unordered(
                    _block, [tasks[i] for i in pending], chunksize=1):
                rowsum[v0:v1] = rows
                if F.char == 2:
                    colsum ^= cols
                else:
                    colsum = (colsum + cols) % F.p
                mask[idx_of[(v0, v1)]] = True
                done += v1 - v0
                since += 1
                if ckpt and since >= a.checkpoint_every:
                    os.makedirs(a.checkpoint_dir, exist_ok=True)
                    np.savez(ckpt + ".tmp.npz", rowsum=rowsum, colsum=colsum,
                             mask=mask)
                    os.replace(ckpt + ".tmp.npz", ckpt)   # atomic
                    since = 0
                if a.progress and done % (a.progress * a.block_slices) == 0:
                    print(f"  key {k}: {done}/{q} slices "
                          f"({time.time() - t0:.0f}s)", flush=True)
        t_dec = time.time() - t0
        ax0 = first_nonzero(F, vals, rowsum, a.max_weight)
        ax1 = first_nonzero(F, vals, colsum, a.max_weight)
        rec = []
        for j, i in enumerate(words):
            D = int(prof[a.layers - 1][i])
            rec.append({"word": i, "D_maxplus": D, "T": T,
                        "a_star_axis0": ax0[j], "a_star_axis1": ax1[j],
                        "lower_bound_axis0": (T - ax0[j]) if ax0[j] is not None
                        else None,
                        "lower_bound_axis1": (T - ax1[j]) if ax1[j] is not None
                        else None})
        per_key.append(rec)
        print(f"  key {k}: axis0 a* = {ax0}", flush=True)
        print(f"  key {k}: axis1 a* = {ax1}   (decryption {t_dec:.0f}s, "
              f"total {time.time() - t0:.0f}s)", flush=True)

    merged = []
    same = 0
    for j in range(len(words)):
        v0 = [per_key[k][j]["a_star_axis0"] for k in range(a.keys)]
        v1 = [per_key[k][j]["a_star_axis1"] for k in range(a.keys)]
        base = dict(per_key[0][j])
        base["a_star_axis0_per_key"] = v0
        base["a_star_axis1_per_key"] = v1
        base["key_independent"] = len(set(v0)) == 1 and len(set(v1)) == 1
        same += base["key_independent"]
        merged.append(base)
    print(f"{a.instance}: {same}/{len(words)} words key-independent on both axes")

    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "rounds": c.r, "active": act, "layers": a.layers, "threshold": T,
           "keys": a.keys, "seed": a.seed, "max_weight": a.max_weight,
           "words": words, "data_log2": round(2 * float(np.log2(q)), 2),
           "key_independent_words": same, "rows": merged,
           "seconds": round(time.time() - t0, 1)}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"twoword_{a.instance}_L{a.layers}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
