"""S6(b)/(c) -- the O7 tightness grid, driven by the generic `zsx` kernel.

Every cell is one (instance, direction, active words, value sets) combination.
For each cell the O7 prediction is computed first (that also fixes how many
layers have to be evaluated: l_full + 2 is enough to see the first failing
layer and the one after it), the kernel is run on `--keys` master keys, and the
measured pattern is compared with the prediction.  Nothing here ever changes
the prediction: a mismatch is reported, not repaired.

    python s6_grid.py --list                    # the plan with data sizes
    python s6_grid.py --which b --threads 96 --out results/E09_unified_criterion/grid
    python s6_grid.py --table results/E09_unified_criterion/grid > tightness_grid.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
from dux import DuX                                             # noqa: E402
from tools.zero_sum_criterion import patterns as o7_patterns    # noqa: E402

# (group, instance, direction, active, sets)
CELLS = [
    # ---- (b) decryption, F_p toys ----------------------------------------
    ("b", "toy-193", "dec", "3", "full"),
    ("b", "toy-193", "dec", "3,7", "full,full"),
    ("b", "toy-193", "dec", "3,7,11", "full,full,full"),
    ("b", "toy-193", "dec", "3,7,11,15", "full,full,full,full"),
    ("b", "toy-193", "dec", "2,3", "full,full"),
    ("b", "toy-193", "dec", "2,3,6,7", "full,full,full,full"),
    ("b", "toy-257", "dec", "3", "full"),
    ("b", "toy-257", "dec", "3,7", "full,full"),
    ("b", "toy-257", "dec", "3,7,11", "full,full,full"),
    ("b", "toy-257", "dec", "3,7,11,15", "full,full,full,full"),
    ("b", "toy-257", "dec", "2,3", "full,full"),
    # ---- (b) decryption, F_2^n ------------------------------------------
    ("b", "dux-2^8", "dec", "3,7", "full,full"),
    ("b", "dux-2^8", "dec", "3,7,11", "full,full,full"),
    ("b", "dux-2^8", "dec", "2,3", "full,full"),
    ("b", "dux-2^8", "dec", "2,3,6,7", "full,full,full,full"),
    ("b", "dux-2^8", "dec", "3,7,11,15", "full,full,full,full"),
    ("b", "dux-2^16", "dec", "3", "full"),
    ("b", "dux-2^16", "dec", "3,7", "full,full"),
    ("b", "dux-2^16", "dec", "3,7,11", "dim8,dim8,dim8"),
    ("b", "dux-2^16", "dec", "3,7,11", "dim10,dim10,dim10"),
    ("b", "dux-2^16", "dec", "3,7,11", "dim11,dim11,dim11"),
    # ---- (c) encryption (designer CPA model) -----------------------------
    ("c", "toy-193", "enc", "1", "full"),
    ("c", "toy-193", "enc", "1,5", "full,full"),
    ("c", "toy-257", "enc", "1,5", "full,full"),
    ("c", "dux-2^8", "enc", "1", "full"),
    ("c", "dux-2^8", "enc", "1,5", "full,full"),
    ("c", "dux-2^8", "enc", "1,5,9", "full,full,full"),
    ("c", "dux-2^8", "enc", "1,5,9,13", "full,full,full,full"),
    ("c", "dux-65537", "enc", "1", "full"),
    ("c", "dux-2^16", "enc", "1", "full"),
    ("c", "dux-65537", "enc", "1,5", "full,full"),
    ("c", "dux-2^16", "enc", "1,5", "full,full"),
]


def set_size(F, spec):
    if spec == "full":
        return F.q
    if spec.startswith("dim"):
        return 1 << int(spec[3:])
    return 1 << (int(spec[5:]) + 1)          # two signed cosets


def cell_info(instance, direction, active, sets, maxlayers=12):
    c = DuX(instance)
    F = c.F
    act = [int(v) for v in active.split(",")]
    specs = sets.split(",")
    npts = 1
    for sp in specs:
        npts *= set_size(F, sp)
    if F.char == 2:
        dims = [F.n if sp == "full" else int(sp[3:]) for sp in specs]
        pred = o7_patterns(f"2^{F.n}", act, maxlayers, dims=dims, direction=direction)
    elif all(sp.startswith("coset") for sp in specs):
        pred = o7_patterns(F.p, act, maxlayers,
                           coset=[int(sp[5:]) for sp in specs], direction=direction)
    else:
        pred = o7_patterns(F.p, act, maxlayers, direction=direction)
    layers = min(maxlayers, pred["l_full"] + 2)
    return {"npts": npts, "data_log2": round(math.log2(npts), 2),
            "layers": layers, "pred": pred, "active": act, "specs": specs}


def tag_of(instance, direction, active, sets):
    return (f"{instance.replace('^', '')}_{direction}_a{active.replace(',', '_')}"
            f"_{sets.replace(',', '-')}")


def run_cell(cell, args):
    group, instance, direction, active, sets = cell
    info = cell_info(instance, direction, active, sets, args.maxlayers)
    tag = tag_of(instance, direction, active, sets)
    cmd = [sys.executable, os.path.join(HERE, "run_zsx.py"),
           "--instance", instance, "--active", active, "--sets", sets,
           "--direction", direction, "--layers", str(info["layers"]),
           "--keys", str(args.keys), "--seed", str(args.seed),
           "--threads", str(args.threads), "--out", args.out, "--tag", tag]
    print(f"[{group}] {tag}  data=2^{info['data_log2']}  layers={info['layers']}",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"  !! FAILED ({time.time() - t0:.0f}s)", flush=True)
    return proc.returncode == 0


def table(outdir):
    rows = []
    for fn in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        r = json.load(open(fn))
        pl = r["predicted_O7"]
        m = pl["margin_per_position"]
        pf = pl["l_full"]
        # tightness: how much room is left in the LAST all-balanced layer, and
        # by how little the FIRST failing layer misses
        last_ok = min(m[pf - 1]) if pf >= 1 else None
        # tightness at the transition layer: the narrowest correct "balanced"
        # call (smallest positive margin) and the narrowest correct
        # "unbalanced" call (largest non-positive margin)
        tp = tn = None
        if pf < len(m):
            pos = [v for v in m[pf] if v > 0]
            neg = [v for v in m[pf] if v <= 0]
            tp = min(pos) if pos else None
            tn = max(neg) if neg else None
        rows.append((r["instance"], r["direction"], ",".join(map(str, r["active"])),
                     "-".join(r["sets"]), r["data_log2"], r["keys"],
                     pl["l_full"], pl["next"],
                     r["measured"]["l_full"], r["measured"]["next"],
                     "yes" if r["prediction_matches"] else "**NO**",
                     last_ok, tp if tp is not None else "-",
                     tn if tn is not None else "-", r["elapsed_s"]))
    hdr = ("| 实例 | 方向 | 激活字 | 结构 | 数据 log2 | 密钥 | 预测 l_full | 预测 next | "
           "实测 l_full | 实测 next | 一致 | 第 l_full 层最薄余量 | "
           "第 l_full+1 层最薄正余量 | 第 l_full+1 层最薄负余量 | 秒 |")
    out = [hdr, "|" + "---|" * 15]
    for r in rows:
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    ok = sum(1 for r in rows if r[10] == "yes")
    out.append("")
    out.append(f"**{ok}/{len(rows)} 格吻合**")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", help="b | c | all | a comma list of instances")
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--maxlayers", type=int, default=12)
    ap.add_argument("--max-data-log2", type=float, default=33.0)
    ap.add_argument("--out", default="results/E09_unified_criterion/grid")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--table", default=None)
    a = ap.parse_args()

    if a.table:
        print(table(a.table))
        return

    cells = [c for c in CELLS if a.which == "all" or c[0] == a.which
             or c[1] in a.which.split(",")]
    plan = []
    for c in cells:
        info = cell_info(c[1], c[2], c[3], c[4], a.maxlayers)
        if info["data_log2"] > a.max_data_log2:
            print(f"SKIP (data 2^{info['data_log2']} > 2^{a.max_data_log2}): {c}")
            continue
        plan.append((c, info))
    if a.list:
        tot = 0.0
        for c, info in plan:
            # rough cost model: points x layers x keys, normalised to the
            # measured 12.7 s for 3.4e7 points x 12 layers x 3 keys at 32 threads
            cost = info["npts"] * info["layers"] * a.keys / (3.36e7 * 12 * 3) * 12.7 * 32 / a.threads
            tot += cost
            print(f"{c[0]} {c[1]:11s} {c[2]} a={c[3]:12s} {c[4]:24s} "
                  f"data=2^{info['data_log2']:<5} layers={info['layers']:2d} "
                  f"pred l_full={info['pred']['l_full']} next={info['pred']['next']} "
                  f"~{cost:.0f}s")
        print(f"total ~{tot / 60:.0f} min at {a.threads} threads")
        return
    os.makedirs(a.out, exist_ok=True)
    for c, _info in plan:
        run_cell(c, a)


if __name__ == "__main__":
    main()
