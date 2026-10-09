"""E06b / S1 -- r_KR = 1 key recovery from a PARTIAL zero-sum, C-backed.

Identical mathematics to experiments/E06_key_recovery_1round/attack_1round_partial.py
(read that file first); the only change is that the per-structure plaintext
moments are produced by fast/mom.c, which makes 2^32-ciphertext structures --
and therefore the 11-round DuX(2^16) attack enabled by the layer-10 partial
zero-sum of S1 -- actually runnable.

  # 7-round DuX(2^8) reproduction of W6 (3 words, 2^24 per structure)
  python attack_1round_partial_fast.py --instance dux-2^8 --rounds 7 \
      --active 3,7,11 --dim 8 --coords 0 --structures 18

  # 11-round DuX(2^16): layer-10 `1101` zero-sum from 2 words over the whole
  # field (2^32 per structure)
  python attack_1round_partial_fast.py --instance dux-2^16 --rounds 11 \
      --active 3,7 --dim 16 --coords 0 --structures 18
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))

from dux import DuX                                        # noqa: E402
from attack_1round_partial import (equation_template, determined_values,  # noqa: E402
                                   brute_force_free_words, sequential_solve)
from experiments_e04_helpers import random_basis, span_values  # noqa: E402


def build_job(cipher, rounds, active, dim, structures, needed, seed):
    F = cipher.F
    rng = np.random.default_rng(seed)
    K = cipher.random_key(rng)
    rks = cipher.key_schedule(K)
    assert len(rks) == rounds + 1
    consts, vals = [], []
    for _ in range(structures):
        consts.append([int(v) for v in rng.integers(0, F.q, size=16)])
        vals.append([span_values(random_basis(F.n, dim, rng),
                                 int(rng.integers(0, F.q))) for _ in active])
    maxe = max(max(pe) for pe in needed)
    out = bytearray(b"DUXMOMJ1")
    out += struct.pack("<9I", F.n, F.poly, cipher.alpha, rounds,
                       len(active), dim, structures, len(needed), maxe)
    out += np.array(active, dtype=np.uint32).tobytes()
    out += np.asarray(F._log, dtype=np.uint16).tobytes()
    out += np.asarray(F._exp, dtype=np.uint16).tobytes()
    out += np.array([w for rk in rks for w in rk], dtype=np.uint16).tobytes()
    for cs in consts:
        out += np.array(cs, dtype=np.uint16).tobytes()
    for vs in vals:
        for v in vs:
            out += np.asarray(v, dtype=np.uint16).tobytes()
    out += np.array(needed, dtype=np.uint8).tobytes()
    return bytes(out), K, rks, consts, vals


def python_moments(cipher, rks, rounds, active, dim, consts, vals, needed,
                   chunk_log2=20):
    """Reference moments for one structure (small data only)."""
    F = cipher.F
    total_log2 = dim * len(active)
    csize = 1 << min(total_log2, chunk_log2)
    maxe = max(max(pe) for pe in needed)
    moments = [{pe: 0 for pe in needed} for _ in range(4)]
    for start in range(0, 1 << total_log2, csize):
        idx = np.arange(start, start + csize, dtype=np.int64)
        C = [np.full(csize, consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            C[w] = vals[t][(idx >> (dim * t)) & ((1 << dim) - 1)]
        P = cipher.decrypt(tuple(C), rks, rounds=rounds, vec=True)
        for j in range(4):
            pw = []
            for i in range(4):
                col = [np.ones(csize, dtype=np.int64)]
                for _ in range(maxe):
                    col.append(F.vmul(col[-1], P[4 * j + i]))
                pw.append(col)
            for pe in needed:
                acc = None
                for i in range(4):
                    if pe[i]:
                        acc = pw[i][pe[i]] if acc is None else F.vmul(acc, pw[i][pe[i]])
                v = F.vsum(acc) if acc is not None else 0
                moments[j][pe] = F.add(moments[j][pe], v)
    return moments


def parse_moments(txt, structures, needed):
    out = [[None] * 4 for _ in range(structures)]
    for line in txt.splitlines():
        if not line.startswith("M "):
            continue
        p = line.split()
        st, b = int(p[1]), int(p[2])
        out[st][b] = {pe: int(v) for pe, v in zip(needed, p[3:])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^16")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--coords", default="0")
    ap.add_argument("--sequential", action="store_true",
                    help="W balanced on all four coordinates (`1111`): substitute "
                         "coordinate 2 -> (k0,k3), 1 -> k2, 3 -> k1 (Lemma O8) "
                         "instead of linearising; needs --coords 1,2,3 and 2-3 "
                         "structures")
    ap.add_argument("--structures", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bin", default=os.path.join(HERE, "mom"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--job", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cipher = DuX(a.instance, rounds=a.rounds)
    F = cipher.F
    assert F.char == 2
    dim = F.n if a.dim is None else a.dim
    active = [int(v) for v in a.active.split(",")]
    coords = [int(v) for v in a.coords.split(",")]

    tmpl = {co: equation_template(F, cipher.alpha, co) for co in coords}
    mons = sorted({m for co in coords for m in tmpl[co][0]})
    needed = sorted({pe for co in coords for lst in tmpl[co][1].values()
                     for pe, _ in lst})
    n_struct = a.structures or (3 if a.sequential else len(mons) // len(coords) + 4)
    print(f"{a.instance}: {a.rounds} rounds, coords {coords}, {len(mons)} key "
          f"monomials, {len(needed)} plaintext moments, {n_struct} structures "
          f"x 2^{dim * len(active)} chosen ciphertexts "
          f"= 2^{dim * len(active) + np.log2(n_struct):.1f}")

    job, K, rks, consts, vals = build_job(cipher, a.rounds, active, dim,
                                          n_struct, needed, a.seed)
    jobpath = a.job or os.path.join("/tmp", f"duxmom_{os.getpid()}.bin")
    open(jobpath, "wb").write(job)
    env = dict(os.environ)
    if a.threads:
        env["OMP_NUM_THREADS"] = str(a.threads)
    t0 = time.time()
    proc = subprocess.run([a.bin, jobpath], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit("mom failed")
    moms = parse_moments(proc.stdout, n_struct, needed)
    t_data = time.time() - t0
    if a.job is None:
        os.unlink(jobpath)
    print(f"  moments done in {t_data:.1f}s")

    if a.verify:
        for st in range(min(2, n_struct)):
            ref = python_moments(cipher, rks, a.rounds, active, dim,
                                 consts[st], vals[st], needed)
            for b in range(4):
                assert ref[b] == moms[st][b], f"mismatch structure {st} block {b}"
        print("  verify: OK (C moments match the numpy reference)")

    rows = {j: [] for j in range(4)}
    seq = {j: {} for j in range(4)}
    for st in range(n_struct):
        for j in range(4):
            for co in coords:
                _, terms = tmpl[co]
                row = []
                for m in mons:
                    v = 0
                    for pe, coeff in terms.get(m, []):
                        v = F.add(v, F.mul(coeff, moms[st][j][pe]))
                    row.append(v)
                rows[j].append(row)
                seq[j].setdefault(co, []).append(
                    {m: row[t] for t, m in enumerate(mons) if row[t]})

    if a.sequential:
        rec, seq_info = [], []
        for j in range(4):
            kj, info = sequential_solve(F, seq[j])
            seq_info.append({"block": j, "steps": info})
            rec += kj
            print(f"  block {j} (sequential): k = {kj} "
                  f"(true {list(K[4 * j:4 * j + 4])})")
        success = tuple(rec) == tuple(K)
        total_log2 = dim * len(active) + float(np.log2(n_struct))
        print(f"recovered rk^0 == master key: {success}; data 2^{total_log2:.1f}; "
              f"{time.time() - t0:.1f}s")
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            fn = os.path.join(a.out, f"fast_sequential_{a.instance.replace('^', '')}"
                                     f"_r{a.rounds}_{'_'.join(map(str, active))}_d{dim}.json")
            json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                       "dim": dim, "coords": coords, "mode": "sequential",
                       "structures": n_struct, "steps": seq_info,
                       "success": bool(success),
                       "data_log2_per_structure": dim * len(active),
                       "data_log2_total": round(total_log2, 2),
                       "moments_s": round(t_data, 1),
                       "elapsed_s": round(time.time() - t0, 1)},
                      open(fn, "w"), indent=1)
            print("saved", fn)
        return

    const = (0, 0, 0, 0)
    cols_all = [m for m in mons if m != const]
    rec, ranks, bruted = [], [], []
    for j in range(4):
        R, B = [], []
        for row in rows[j]:
            b = F.sub(0, row[mons.index(const)]) if const in mons else 0
            R.append([v for t, v in enumerate(row) if mons[t] != const])
            B.append(b)
        det, rank = determined_values(F, R, B, cols_all)
        ranks.append(rank)
        kj = [det.get(tuple(1 if t == i else 0 for t in range(4))) for i in range(4)]
        free = [i for i in range(4) if kj[i] is None]
        if free:
            known = {i: kj[i] for i in range(4) if kj[i] is not None}
            cand = brute_force_free_words(F, rows[j], mons, known, free)
            bruted.append({"block": j, "free_words": free,
                           "candidates": None if cand is None else len(cand)})
            if cand and len(cand) == 1:
                for slot, i in enumerate(free):
                    kj[i] = cand[0][slot]
        rec += kj
        print(f"  block {j}: rank {rank}/{len(cols_all)}; k = {kj} "
              f"(true {list(K[4 * j:4 * j + 4])})")
    success = tuple(rec) == tuple(K)
    total_log2 = dim * len(active) + float(np.log2(n_struct))
    print(f"recovered rk^0 == master key: {success}; data 2^{total_log2:.1f}; "
          f"{time.time() - t0:.1f}s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"fast_partial_{a.instance.replace('^', '')}"
                                 f"_r{a.rounds}_{'_'.join(map(str, active))}"
                                 f"_d{dim}_c{a.coords.replace(',', '')}.json")
        json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                   "dim": dim, "coords": coords, "monomials": len(mons),
                   "structures": n_struct, "ranks": ranks,
                   "data_log2_per_structure": dim * len(active),
                   "data_log2_total": round(total_log2, 2),
                   "brute_forced": bruted, "success": bool(success),
                   "moments_s": round(t_data, 1),
                   "elapsed_s": round(time.time() - t0, 1)},
                  open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
