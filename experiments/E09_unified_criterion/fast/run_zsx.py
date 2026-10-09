"""S6 -- driver for the generic zero-sum kernel `zsx`.

Builds the job file from the Python reference (`dux/`), runs ./zsx, and reports
the measured balanced-word pattern per layer next to the prediction of theorem
O7 (`tools/zero_sum_criterion.py`), with the margin T - D per block position.

Structures.  Each active word carries a value set, given by --sets (one entry
per active word) or by the shorthands --dim / --coset:

    full      the whole field                      (T contribution q - 1)
    dimM      F_2-affine subspace of dimension M   (T contribution 2^M - 1)
    cosetK    coset_0 (+1) u coset_1 (-1) of the order-2^K subgroup of F_p^*
              (T contribution 2^K; over s words this is the signed 2^s-coset
              inclusion-exclusion, not the "difference of two product sets"
              that experiments/E03_zero_sum_Fp/run.py measures)

Examples
--------
  # S6(a): the F_p multi-word full-field run that E03 declared useless
  python run_zsx.py --instance dux-65537 --active 3,7 --layers 12 --keys 3

  # S6(c): CPA direction, two active plaintext words
  python run_zsx.py --instance dux-65537 --active 1,5 --direction enc --layers 8 --keys 3

  # cross-check against the numpy reference
  python run_zsx.py --instance toy-193 --active 3,7 --layers 6 --keys 2 --verify

  # S11: YuX (the registry picks the cipher; --bin defaults to ./zsx_yux)
  python run_zsx.py --instance yu2x-8 --full-block 0 --layers 8 --keys 3
  python run_zsx.py --instance yupx-65537 --active 0,4 --rounds 14 --layers 11 --keys 3

O11 full-block structures (--full-block b) make the four words of block b run
over all of F_q^4; S^{-1} is a bijection there, so the first layer is free and
theorem O7 starts from s = 4 with first-layer degrees (1,1,1,1).  The kernel
needs no change for this -- it is just four active words in one block -- but
the O7 prediction does, which is what `free_blocks=` passes on.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
from dux.field import _prime_factors                   # noqa: E402
from dux.registry import cipher_family, get_cipher      # noqa: E402
from tools.zero_sum_criterion import patterns as o7_patterns   # noqa: E402

WORDS = 16


# ---------------------------------------------------------------- value sets
def gf2_rank(vs, n):
    rows, rank = list(vs), 0
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
    vals = np.zeros(1 << len(basis), dtype=np.int64)
    for i, b in enumerate(basis):
        step = 1 << i
        vals[step:2 * step] = vals[:step] ^ b
    return vals ^ c


def subgroup(p, k):
    """The order-2^k subgroup H of F_p^*, as an array."""
    assert (p - 1) % (1 << k) == 0, f"2^{k} does not divide {p} - 1"
    fs = _prime_factors(p - 1)
    g = next(x for x in range(2, p) if all(pow(x, (p - 1) // f, p) != 1 for f in fs))
    h = pow(g, (p - 1) >> k, p)
    H = np.empty(1 << k, dtype=np.int64)
    v = 1
    for i in range(1 << k):
        H[i] = v
        v = (v * h) % p
    return H


def build_set(F, spec, rng):
    """Return (values, signs) for one active word."""
    if spec == "full":
        return np.arange(F.q, dtype=np.int64), np.ones(F.q, dtype=np.int8)
    if spec.startswith("dim"):
        m = int(spec[3:])
        assert F.char == 2, "dimM sets are F_2-affine subspaces"
        v = span_values(random_basis(F.n, m, rng), int(rng.integers(0, F.q)))
        return v, np.ones(len(v), dtype=np.int8)
    if spec.startswith("coset"):
        k = int(spec[5:])
        assert F.char != 2, "cosetK sets are multiplicative cosets in F_p"
        H = subgroup(F.p, k)
        # The two cosets must be DISTINCT: a0 H = a1 H exactly when
        # (a0/a1)^{2^k} = 1, and then the signed difference annihilates every
        # function, which would look like a zero-sum at every layer.  With
        # k = 15 and p = 65537 there are only two cosets, so a uniformly random
        # pair coincides half the time.
        a0 = int(rng.integers(1, F.p))
        while True:
            a1 = int(rng.integers(1, F.p))
            if pow(a1 * pow(a0, F.p - 2, F.p) % F.p, 1 << k, F.p) != 1:
                break
        v = np.concatenate([(H * a0) % F.p, (H * a1) % F.p])
        sg = np.concatenate([np.ones(len(H), dtype=np.int8),
                             -np.ones(len(H), dtype=np.int8)])
        return v, sg
    raise SystemExit(f"unknown set spec {spec!r}")


def threshold_of(F, specs):
    T = 0
    for sp in specs:
        if sp == "full":
            T += (F.q - 1) if F.char == 2 else (F.p - 1)
        elif sp.startswith("dim"):
            T += (1 << int(sp[3:])) - 1
        else:
            T += 1 << int(sp[5:])
    return T


# --------------------------------------------------------------------- job
def build_job(cipher, active, specs, layers, keys, seed, direction):
    F = cipher.F
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(keys):
        rks = cipher.key_schedule(cipher.random_key(rng))
        consts = [int(v) for v in rng.integers(0, F.q, size=WORDS)]
        sets = [build_set(F, sp, rng) for sp in specs]
        jobs.append({"rks": rks, "consts": consts, "sets": sets})
    sizes = [len(jobs[0]["sets"][j][0]) for j in range(len(active))]

    out = bytearray()
    out += b"DUXZSX01"
    out += struct.pack("<9I", 1 if F.char == 2 else 0,
                       F.n if F.char == 2 else 0,
                       F.q if F.char == 2 else F.p,
                       cipher.alpha, cipher.r, layers, len(active), keys,
                       1 if direction == "enc" else 0)
    out += np.array(active, dtype=np.uint32).tobytes()
    out += np.array(sizes, dtype=np.uint32).tobytes()
    if F.char == 2:
        out += np.asarray(F._log, dtype=np.uint16).tobytes()
        out += np.asarray(F._exp, dtype=np.uint16).tobytes()
    fwd = np.zeros((2, WORDS), dtype=np.uint32)
    if F.char != 2:
        for t in (0, 1):
            fwd[t] = np.array(cipher.lin._rows[t], dtype=np.uint32)
    out += fwd.tobytes()
    for j in jobs:
        out += np.array([w for rk in j["rks"] for w in rk], dtype=np.uint32).tobytes()
    for j in jobs:
        out += np.array(j["consts"], dtype=np.uint32).tobytes()
    for j in jobs:
        for v, _s in j["sets"]:
            out += np.asarray(v, dtype=np.uint32).tobytes()
    for j in jobs:
        for _v, s in j["sets"]:
            out += np.asarray(s, dtype=np.int8).tobytes()
    return bytes(out), jobs, sizes


def python_reference(cipher, job, active, sizes, layers, direction, chunk=1 << 18):
    """Weighted per-layer sums with the numpy reference (small structures)."""
    F = cipher.F
    total = int(np.prod(sizes))
    strides = [int(np.prod(sizes[j + 1:])) for j in range(len(sizes))]
    sums = [np.zeros(WORDS, dtype=np.int64) for _ in range(layers)]
    for start in range(0, total, chunk):
        n = min(chunk, total - start)
        idx = np.arange(start, start + n, dtype=np.int64)
        X = [np.full(n, job["consts"][i], dtype=np.int64) for i in range(WORDS)]
        w = np.ones(n, dtype=np.int64)
        for j, word in enumerate(active):
            sub = (idx // strides[j]) % sizes[j]
            v, sg = job["sets"][j]
            X[word] = v[sub]
            w = w * sg[sub]
        f = cipher.decrypt_layers if direction == "dec" else cipher.encrypt_layers
        states = f(tuple(X), job["rks"], layers, vec=True, yield_all=True)
        for l, st in enumerate(states):
            for i in range(WORDS):
                if F.char == 2:
                    sums[l][i] ^= int(np.bitwise_xor.reduce(st[i]))
                else:
                    sums[l][i] = int((sums[l][i] + int(np.sum(w * st[i]) % F.p)) % F.p)
    return [[int(v) for v in sums[l]] for l in range(layers)]


def parse_output(txt, keys, layers):
    sums = [[None] * layers for _ in range(keys)]
    for line in txt.splitlines():
        if line.startswith("S "):
            q = line.split()
            sums[int(q[1])][int(q[2]) - 1] = [int(v) for v in q[3:]]
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="dux-65537")
    ap.add_argument("--active", default=None,
                    help="comma list of active words (default 3,7 unless "
                         "--full-block is given)")
    ap.add_argument("--full-block", default=None,
                    help="O11: comma list of blocks whose four words all run "
                         "over F_q (added to --active; the O7 prediction is "
                         "recomputed with the free-block substitution)")
    ap.add_argument("--sets", default=None,
                    help="one spec per active word: full | dimM | cosetK")
    ap.add_argument("--dim", type=int, default=None, help="shorthand: dimM for every word")
    ap.add_argument("--coset", type=int, default=None, help="shorthand: cosetK for every word")
    ap.add_argument("--direction", choices=["dec", "enc"], default="dec")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bin", default=None,
                    help="kernel binary (default: ./zsx for DuX, ./zsx_yux for YuX)")
    ap.add_argument("--job", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--raw-sums", action="store_true",
                    help="also print and record the 16 raw word sums of every "
                         "key and layer (O14 / S16: a D = T boundary cell is "
                         "not 'balanced or not' but a specific constant)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    fam = cipher_family(a.instance)
    cipher = get_cipher(a.instance, rounds=a.rounds)
    F = cipher.F
    free_blocks = ([int(v) for v in a.full_block.split(",")]
                   if a.full_block else [])
    # `active` is what the KERNEL walks (a free block is simply its four
    # words); `plain_active` is what the O7 tool takes, which wants the free
    # blocks listed separately (cipher_degree.profile asserts on the overlap).
    act_spec = a.active if a.active is not None else ("" if free_blocks else "3,7")
    plain_active = sorted({int(v) for v in act_spec.split(",")} if act_spec else set())
    plain_active = [w for w in plain_active if w // 4 not in free_blocks]
    active = sorted(set(plain_active) | {4 * b + i for b in free_blocks
                                         for i in range(4)})
    binpath = a.bin or os.path.join(HERE, "zsx_yux" if fam == "yux" else "zsx")
    if a.sets:
        specs = a.sets.split(",")
    elif a.dim is not None:
        specs = [f"dim{a.dim}"] * len(active)
    elif a.coset is not None:
        specs = [f"coset{a.coset}"] * len(active)
    else:
        specs = ["full"] * len(active)
    assert len(specs) == len(active)

    job_bytes, jobs, sizes = build_job(cipher, active, specs, a.layers, a.keys,
                                       a.seed, a.direction)
    npoints = int(np.prod([int(s) for s in sizes]))
    jobpath = a.job or os.path.join("/tmp", f"duxzsx_{os.getpid()}.bin")
    open(jobpath, "wb").write(job_bytes)

    env = dict(os.environ)
    if a.threads:
        env["OMP_NUM_THREADS"] = str(a.threads)
    t0 = time.time()
    proc = subprocess.run([binpath, jobpath], capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"zsx failed with code {proc.returncode}")
    sums = parse_output(proc.stdout, a.keys, a.layers)
    if a.job is None:
        os.unlink(jobpath)

    if a.verify:
        for k, job in enumerate(jobs):
            ref = python_reference(cipher, job, active, sizes, a.layers, a.direction)
            for l in range(a.layers):
                assert ref[l] == sums[k][l], (
                    f"MISMATCH key {k} layer {l + 1}:\n  C  {sums[k][l]}\n  py {ref[l]}")
        print(f"verify: OK ({a.keys} keys x {a.layers} layers match the numpy reference)")

    agg = [[all(sums[k][l][i] == 0 for k in range(a.keys)) for i in range(WORDS)]
           for l in range(a.layers)]
    per_layer16 = ["".join("1" if v else "0" for v in row) for row in agg]
    # block pattern: position pp balanced iff balanced in all four blocks
    per_layer4 = ["".join("1" if all(row[4 * b + pp] for b in range(4)) else "0"
                          for pp in range(4)) for row in agg]
    l_full = 0
    for pat in per_layer4:
        if pat == "1111":
            l_full += 1
        else:
            break
    nxt = per_layer4[l_full] if l_full < a.layers else "?"

    # ---- prediction (theorem O7) -----------------------------------------
    # The max-plus degree profile does not depend on the value sets, only the
    # threshold T does, so the patterns are recomputed here from the profile
    # with `threshold_of`.  That also covers structures whose words use
    # DIFFERENT set types, which the tool's own T formulas cannot express.
    if F.char == 2:
        q = f"2^{F.n}"
        dims = [F.n if sp == "full" else int(sp[3:])
                for w, sp in zip(active, specs) if w // 4 not in free_blocks]
        pred = o7_patterns(q, plain_active, a.layers, dims=dims,
                           direction=a.direction, cipher=fam,
                           free_blocks=free_blocks)
    elif all(sp.startswith("coset") for sp in specs):
        pred = o7_patterns(F.p, plain_active, a.layers,
                           coset=[int(sp[5:]) for sp in specs],
                           direction=a.direction, cipher=fam,
                           free_blocks=free_blocks)
    else:
        pred = o7_patterns(F.p, plain_active, a.layers, direction=a.direction,
                           cipher=fam, free_blocks=free_blocks)
    T = threshold_of(F, specs)
    worst = pred["max_degree_per_position"]
    pred["threshold"] = T
    pred["margin_per_position"] = [[T - w for w in row] for row in worst]
    pred["patterns"] = ["".join("1" if w < T else "0" for w in row) for row in worst]
    pred["l_full"] = 0
    for pat in pred["patterns"]:
        if pat == "1111":
            pred["l_full"] += 1
        else:
            break
    pred["next"] = (pred["patterns"][pred["l_full"]]
                    if pred["l_full"] < a.layers else "?")

    match = pred["l_full"] == l_full and pred["next"] == nxt
    res = {"instance": a.instance, "cipher": fam, "field": F.name,
           "active": active, "full_block": free_blocks,
           "sets": specs, "direction": a.direction, "rounds": cipher.r,
           "layers": a.layers, "keys": a.keys, "seed": a.seed,
           "points_per_structure": npoints,
           "data_log2": round(math.log2(npoints), 2),
           "threshold_T": threshold_of(F, specs),
           "measured": {"l_full": l_full, "next": nxt,
                        "patterns4": per_layer4, "patterns16": per_layer16,
                        "balanced_count_per_layer": [sum(r) for r in agg]},
           "predicted_O7": {"l_full": pred["l_full"], "next": pred["next"],
                            "patterns4": pred["patterns"],
                            "threshold": pred["threshold"],
                            "max_degree_per_position": pred["max_degree_per_position"],
                            "margin_per_position": pred["margin_per_position"]},
           "prediction_matches": bool(match),
           "kernel": os.path.basename(binpath), "threads": a.threads or env.get("OMP_NUM_THREADS", "all"),
           "elapsed_s": round(elapsed, 1)}
    print(f"{a.instance} {a.direction} active={active} sets={specs} "
          f"data=2^{res['data_log2']} keys={a.keys} T={res['threshold_T']}")
    print(f"  measured : l_full = {l_full}, next = {nxt}")
    print(f"  predicted: l_full = {pred['l_full']}, next = {pred['next']}  "
          f"({'MATCH' if match else 'MISMATCH'})")
    for l in range(a.layers):
        m = pred["margin_per_position"][l]
        print(f"    layer {l + 1:2d}  measured {per_layer4[l]}  "
              f"predicted {pred['patterns'][l]}  margin T-D {m}")
    print(f"  {res['elapsed_s']} s")
    if a.raw_sums:
        # O14 (S16 item 1): on a D = T cell the sum is a key-independent
        # constant c_b(p), so the raw numbers are the result, not the pattern.
        res["raw_sums"] = {f"key{k}": {f"layer{l + 1}": sums[k][l]
                                       for l in range(a.layers)}
                           for k in range(a.keys)}
        for k in range(a.keys):
            for l in range(a.layers):
                print(f"  raw key {k} layer {l + 1:2d}: "
                      + " ".join(str(v) for v in sums[k][l]))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or (f"{a.instance.replace('^', '')}_{a.direction}_"
                        + (f"fb{'_'.join(map(str, free_blocks))}_" if free_blocks else "")
                        + f"a{'_'.join(map(str, active))}_{'-'.join(specs)}")
        fn = os.path.join(a.out, f"{tag}.json")
        json.dump(res, open(fn, "w"), indent=1)
        print("saved", fn)


if __name__ == "__main__":
    main()
