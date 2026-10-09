"""E10 / S8 -- chosen-plaintext last-round key recovery, C-backed.

Identical mathematics to experiments/E10_cpa/attack_cp_lastround.py (read that
file first); the only change is that the per-structure CIPHERTEXT moments are
produced by fast/mom_cp.c, which makes 2^32-plaintext structures -- and
therefore the 8-round DuX(2^16) attack enabled by the layer-7 `0010` zero-sum
of S6(c) -- actually runnable.

  # S8: 8-round DuX(2^16), 2 active plaintext words over the whole field
  python attack_cp_lastround_fast.py --instance dux-2^16 --rounds 8 \
      --active 1,5 --dim 16 --coords 2 --structures 20 --keys 2

  # DuX(2^8) at the same data scale, r_KR = 1 rehearsal (layer-4 zero-sum
  # `1111` from four active words, S6(c))
  python attack_cp_lastround_fast.py --instance dux-2^8 --rounds 5 \
      --active 1,5,9,13 --dim 8 --coords 0,1,2,3 --structures 6

Because the moments are the whole cost, the driver solves every nested prefix
N = 1..structures of the structure list and reports N_min (the smallest N that
recovers the master key) together with the rank curve -- the same measurement
S4/S5 make for the chosen-ciphertext attack, for free.
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "E06_key_recovery_1round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))

from dux import DuX                                              # noqa: E402
from dux.keyschedule import round_constants                      # noqa: E402
from attack_cp_lastround import (system_spec, structure_rows,    # noqa: E402
                                 solve_system, master_from_last_round_key)
from experiments_e04_helpers import random_basis, span_values    # noqa: E402


def build_job(cipher, rounds, active, dim, structures, needed, seed):
    """Job file for mom_cp.c plus the plaintext structures it encodes."""
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
    out = bytearray(b"DUXCPMJ1")
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


def python_moments_cp(cipher, rks, rounds, active, dim, consts, vals, needed,
                      chunk_log2=20):
    """Reference ciphertext moments for one structure (small data only)."""
    F = cipher.F
    total_log2 = dim * len(active)
    csize = 1 << min(total_log2, chunk_log2)
    maxe = max(max(pe) for pe in needed)
    moments = [{pe: 0 for pe in needed} for _ in range(4)]
    for start in range(0, 1 << total_log2, csize):
        idx = np.arange(start, start + csize, dtype=np.int64)
        P = [np.full(csize, consts[i], dtype=np.int64) for i in range(16)]
        for t, w in enumerate(active):
            P[w] = vals[t][(idx >> (dim * t)) & ((1 << dim) - 1)]
        C = cipher.encrypt(tuple(P), rks, rounds=rounds, vec=True)
        for b in range(4):
            pw = []
            for i in range(4):
                col = [np.ones(csize, dtype=np.int64)]
                for _ in range(maxe):
                    col.append(F.vmul(col[-1], C[4 * b + i]))
                pw.append(col)
            for pe in needed:
                acc = None
                for i in range(4):
                    if pe[i]:
                        acc = pw[i][pe[i]] if acc is None else F.vmul(acc, pw[i][pe[i]])
                v = F.vsum(acc) if acc is not None else 0
                moments[b][pe] = F.add(moments[b][pe], v)
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
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--active", default="1,5", help="active PLAINTEXT words")
    ap.add_argument("--dim", type=int, default=None,
                    help="F_2-dimension per active word (default: n = whole field)")
    ap.add_argument("--coords", default="2",
                    help="balanced block positions of Y (the CPA distinguisher)")
    ap.add_argument("--structures", type=int, default=None)
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bin", default=os.path.join(HERE, "mom_cp"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the C moments against the numpy reference "
                         "of attack_cp_lastround.py (only feasible for small --dim)")
    ap.add_argument("--job", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cipher = DuX(a.instance, rounds=a.rounds)
    F = cipher.F
    assert F.char == 2, "mom_cp.c is the characteristic-2 kernel"
    dim = F.n if a.dim is None else a.dim
    active = [int(v) for v in a.active.split(",")]
    pos = [int(v) for v in a.coords.split(",")]
    spec = system_spec(F, cipher.alpha, pos)
    needed, cols, n_eq = spec["needed"], spec["cols"], spec["n_eq"]
    n_struct = a.structures or (len(cols) // n_eq + 3)
    ptlog2 = dim * len(active)
    print(f"{a.instance} CPA: {a.rounds} rounds, active plaintext words {active} "
          f"(2^{ptlog2} per structure), balanced positions {pos}")
    print(f"  {len(cols)} unknowns ({len(spec['kmons'])} kappa monomials per "
          f"block), {len(needed)} ciphertext moments, {n_eq} equations/structure, "
          f"{n_struct} structures = 2^{ptlog2 + np.log2(n_struct):.2f} chosen "
          f"plaintexts")

    rcs = round_constants(F, cipher.alpha, a.rounds)
    results = []
    for ki in range(a.keys):
        t0 = time.time()
        job, K, rks, consts, vals = build_job(cipher, a.rounds, active, dim,
                                              n_struct, needed, a.seed + ki)
        jobpath = a.job or os.path.join("/tmp", f"duxcpmom_{os.getpid()}_{ki}.bin")
        open(jobpath, "wb").write(job)
        env = dict(os.environ)
        if a.threads:
            env["OMP_NUM_THREADS"] = str(a.threads)
        td = time.time()
        proc = subprocess.run([a.bin, jobpath], capture_output=True, text=True,
                              env=env)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            raise SystemExit("mom_cp failed")
        moms = parse_moments(proc.stdout, n_struct, needed)
        t_data = time.time() - td
        per_struct = [float(s.split("(")[1].split()[0])
                      for s in proc.stderr.splitlines() if "done (" in s]
        if a.job is None:
            os.unlink(jobpath)
        print(f"  key {ki}: moments done in {t_data:.1f}s "
              f"({t_data / n_struct:.1f}s/structure)")

        if a.verify:
            for st in range(min(2, n_struct)):
                ref = python_moments_cp(cipher, rks, a.rounds, active, dim,
                                        consts[st], vals[st], needed)
                for b in range(4):
                    assert ref[b] == moms[st][b], \
                        f"mismatch structure {st} block {b}"
            print("    verify: OK (C moments match the numpy reference)")

        # nested prefixes: the moments are the cost, the solve is free
        rows, rhs = [], []
        blkrows = {b: [] for b in range(4)}
        Pv = tuple(int(v) for v in np.random.default_rng(a.seed + 77 + ki)
                   .integers(0, F.q, size=16))
        Cv = tuple(cipher.encrypt(Pv, rks, rounds=a.rounds))
        curve, nmin, best = [], None, None
        for st in range(n_struct):
            structure_rows(spec, F, pos, moms[st], rows, rhs, blkrows)
            kappa, rank, n_unknown, free_words, blk_cand = solve_system(
                spec, F, rows, blkrows)
            rk_last = [None if v is None else F.sub(0, v) for v in kappa]
            words = sum(1 for i in range(16)
                        if rk_last[i] is not None and rk_last[i] == rks[a.rounds][i])
            master = (master_from_last_round_key(cipher, a.rounds, rcs, rk_last)
                      if all(v is not None for v in rk_last) else None)
            ok = (master is not None
                  and tuple(cipher.encrypt(Pv, cipher.key_schedule(master),
                                           rounds=a.rounds)) == Cv)
            curve.append({"N": st + 1, "rank": rank, "words": words,
                          "free_words": free_words, "known_pair_ok": bool(ok),
                          "master_key_ok": bool(master == tuple(K))})
            print(f"    N={st + 1:3d}  rank {rank:4d}/{n_unknown}  words "
                  f"{words}/16  master {master == tuple(K)}  known-pair {ok}")
            if ok and nmin is None:
                nmin = st + 1
            if ok:
                best = {"master_key_ok": bool(master == tuple(K)),
                        "known_pair_ok": True}
        el = time.time() - t0
        rank_max = max(c["rank"] for c in curve)
        dmin = ("n/a" if nmin is None
                else f"2^{ptlog2 + np.log2(nmin):.2f}")
        print(f"  key {ki}: N_min = {nmin}, rank_max = {rank_max}, data = {dmin} "
              f"(used {n_struct} = 2^{ptlog2 + np.log2(n_struct):.2f})  ({el:.1f}s)")
        results.append({"key_index": ki, "n_min": nmin, "rank_max": rank_max,
                        "unknowns": n_unknown,
                        "data_log2_at_nmin": (None if nmin is None else
                                              round(ptlog2 + float(np.log2(nmin)), 2)),
                        "moments_seconds": round(t_data, 1),
                        "seconds_per_structure": (round(float(np.mean(per_struct)), 1)
                                                  if per_struct else None),
                        "elapsed_s": round(el, 1), "curve": curve,
                        "final": best})

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, f"cp_attack_{a.instance}_r{a.rounds}_"
                                 f"{'_'.join(map(str, active))}_c"
                                 f"{'_'.join(map(str, pos))}{a.tag}.json")
        json.dump({"instance": a.instance, "rounds": a.rounds, "active": active,
                   "dim": dim, "balanced_positions": pos,
                   "unknowns": len(cols),
                   "kappa_monomials_per_block": len(spec["kmons"]),
                   "moments": len(needed), "equations_per_structure": n_eq,
                   "structures": n_struct,
                   "points_per_structure_log2": ptlog2,
                   "data_log2": round(ptlog2 + float(np.log2(n_struct)), 2),
                   "keys": a.keys, "seed": a.seed,
                   "threads": a.threads, "results": results},
                  open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
