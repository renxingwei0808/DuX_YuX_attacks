"""E04-fast -- driver for the C zero-sum kernel (run matrix S1).

Builds the job file (round keys from dux.keyschedule, ciphertext structure,
field tables from dux.field), runs ./zs, and writes a JSON in the same
schema as experiments/E04_zero_sum_F2n/run.py so the two are directly
comparable.  `--verify` recomputes the same job with the numpy reference
implementation and asserts the two agree word by word (only usable for
small data).

Examples
--------
  # smoke test against the Python reference (2^16 data)
  python run_fast.py --instance dux-2^8 --active 3,7 --layers 8 --keys 2 --verify

  # the decisive DuX(2^16) run: 2 words, whole field, 2^32 data
  python run_fast.py --instance dux-2^16 --active 3,7 --layers 12 --keys 3 \
      --out results/E04_zero_sum_F2n/large
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
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from dux import DuX  # noqa: E402


def gf2_rank(vs, n):
    rows = list(vs)
    rank = 0
    for bit in reversed(range(n)):
        piv = next((i for i in range(rank, len(rows)) if (rows[i] >> bit) & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> bit) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def random_basis(n, m, rng):
    while True:
        vs = [int(v) for v in rng.integers(1, 1 << n, size=m)]
        if gf2_rank(vs, n) == m:
            return vs


def span_values(basis, c):
    m = len(basis)
    vals = np.zeros(1 << m, dtype=np.int64)
    for i, b in enumerate(basis):
        step = 1 << i
        vals[step:2 * step] = vals[:step] ^ b
    return vals ^ c


def build_job(cipher, active, dim, layers, keys, seed):
    """Return (job_bytes, per-key dicts of {rks, consts, vals})."""
    F = cipher.F
    n = F.n
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(keys):
        K = cipher.random_key(rng)
        rks = cipher.key_schedule(K)
        consts = [int(v) for v in rng.integers(0, F.q, size=16)]
        vals = []
        for _w in active:
            basis = random_basis(n, dim, rng)
            vals.append(span_values(basis, int(rng.integers(0, F.q))))
        jobs.append({"K": K, "rks": rks, "consts": consts, "vals": vals})

    out = bytearray()
    out += b"DUXZSJ01"
    out += struct.pack("<8I", n, F.poly, cipher.alpha, cipher.r,
                       layers, len(active), dim, keys)
    out += np.array(active, dtype=np.uint32).tobytes()
    out += np.asarray(F._log, dtype=np.uint16).tobytes()
    out += np.asarray(F._exp, dtype=np.uint16).tobytes()
    for j in jobs:
        out += np.array([w for rk in j["rks"] for w in rk], dtype=np.uint16).tobytes()
    for j in jobs:
        out += np.array(j["consts"], dtype=np.uint16).tobytes()
    for j in jobs:
        for v in j["vals"]:
            out += np.asarray(v, dtype=np.uint16).tobytes()
    return bytes(out), jobs


def python_reference(cipher, job, active, dim, layers, chunk_log2=20):
    """XOR sums per layer, computed with the numpy reference (small data only)."""
    s = len(active)
    total_log2 = dim * s
    chunk_log2 = min(chunk_log2, total_log2)
    sums = [np.zeros(16, dtype=np.int64) for _ in range(layers)]
    csize = 1 << chunk_log2
    for start in range(0, 1 << total_log2, csize):
        idx = np.arange(start, start + csize, dtype=np.int64)
        C = [np.full(csize, job["consts"][i], dtype=np.int64) for i in range(16)]
        for k, w in enumerate(active):
            sub = (idx >> (dim * k)) & ((1 << dim) - 1)
            C[w] = job["vals"][k][sub]
        states = cipher.decrypt_layers(tuple(C), job["rks"], layers,
                                       vec=True, yield_all=True)
        for l, st in enumerate(states):
            for i in range(16):
                sums[l][i] ^= np.bitwise_xor.reduce(st[i])
    return [[int(v) for v in sums[l]] for l in range(layers)]


def parse_output(txt, keys, layers):
    sums = [[None] * layers for _ in range(keys)]
    for line in txt.splitlines():
        if not line.startswith("S "):
            continue
        parts = line.split()
        k, l = int(parts[1]), int(parts[2]) - 1
        sums[k][l] = [int(v) for v in parts[3:]]
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-2^8")
    ap.add_argument("--active", default="3,7")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=None,
                    help="number of cipher rounds (default: the instance's)")
    ap.add_argument("--keys", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bin", default=os.path.join(HERE, "zs"))
    ap.add_argument("--job", default=None, help="where to write the job file")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="also run the numpy reference and compare")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cipher = DuX(a.instance, rounds=a.rounds)
    assert cipher.F.char == 2, "this kernel is for DuX(2^n) only"
    dim = cipher.F.n if a.dim is None else a.dim
    active = [int(v) for v in a.active.split(",")]
    job_bytes, jobs = build_job(cipher, active, dim, a.layers, a.keys, a.seed)

    jobpath = a.job or os.path.join("/tmp", f"duxzs_{os.getpid()}.bin")
    with open(jobpath, "wb") as f:
        f.write(job_bytes)

    env = dict(os.environ)
    if a.threads:
        env["OMP_NUM_THREADS"] = str(a.threads)
    t0 = time.time()
    proc = subprocess.run([a.bin, jobpath], capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"zs failed with code {proc.returncode}")
    sums = parse_output(proc.stdout, a.keys, a.layers)
    if a.job is None:
        os.unlink(jobpath)

    if a.verify:
        for k, job in enumerate(jobs):
            ref = python_reference(cipher, job, active, dim, a.layers)
            for l in range(a.layers):
                assert ref[l] == sums[k][l], (
                    f"MISMATCH key {k} layer {l + 1}:\n  C   {sums[k][l]}\n  py  {ref[l]}")
        print(f"verify: OK ({a.keys} keys x {a.layers} layers match the numpy reference)")

    # a word is balanced only if it is zero for EVERY key
    agg = [[all(sums[k][l][i] == 0 for k in range(a.keys)) for i in range(16)]
           for l in range(a.layers)]
    counts = [sum(r) for r in agg]
    max_full = max((l + 1 for l in range(a.layers) if counts[l] == 16), default=0)
    last = max((l for l in range(a.layers) if counts[l] > 0), default=-1)
    pattern = "".join("1" if v else "0" for v in agg[last]) if last >= 0 else "-"
    first_fail = max_full + 1 if max_full < a.layers else None
    fail_pat = ("".join("1" if v else "0" for v in agg[max_full])
                if max_full < a.layers else "-")

    res = {"instance": a.instance, "active": active, "dim": dim,
           "rounds": cipher.r,
           "data_log2": dim * len(active), "layers": a.layers, "keys": a.keys,
           "seed": a.seed, "kernel": "c-omp",
           "threads": a.threads or int(env.get("OMP_NUM_THREADS", 0)) or "all",
           "balanced_count_per_layer": counts, "max_layers_all16": max_full,
           "first_failing_layer": first_fail, "first_failing_pattern": fail_pat,
           "last_nontrivial_layer": last + 1, "pattern_last": pattern,
           "balanced_matrix": agg,
           "patterns_per_layer": ["".join("1" if v else "0" for v in agg[l])
                                  for l in range(a.layers)],
           "elapsed_s": round(elapsed, 1)}
    print(f"{a.instance} active={active} dim={dim} data=2^{dim * len(active)} keys={a.keys}")
    print(f"  balanced words per layer: {counts}")
    print(f"  all-16 zero-sum up to layer {max_full}; "
          f"first failing layer {first_fail} pattern {fail_pat}")
    print(f"  last nontrivial layer {last + 1}: {pattern}")
    print(f"  {res['elapsed_s']} s")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or f"{a.instance.replace('^', '')}_a{'_'.join(map(str, active))}_d{dim}"
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
