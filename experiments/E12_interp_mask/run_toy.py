"""E12 (W25, R8): O15 divided-difference masks -- the data plan for the paper
instances and the toy end-to-end runs.

Two modes:

  --plan       For every ledger row that uses O12 weights on a full-domain
               product set, recompute D with the repository's own criterion,
               then report the smallest point set that still admits the N_w
               weights the run actually used, the new data in log2 and the
               scaling of the assembly cost.  Nothing is run.

  --toy        Print the toy end-to-end commands (baseline and masked).

Two accountings are reported for every row, because they differ whenever a row
uses more than one active word:

  dims = 1   the accounting the R8 memo uses AND the one the streaming pipeline
             implements (`attack_12round.py` builds `weights = [int(w[0])]`,
             i.e. the weights live on the FIRST active word only), so N_w
             weights need T' - D >= N_w and n = ceil((D + N_w)/s) + 1;
  dims = s   multi-index weights a in N^s, which `attack_2round_fast.py`
             supports: the first N_w vectors in |a| order only reach |a| = m
             with C(m + s, s) >= N_w, so a much smaller margin -- and hence a
             smaller point set -- suffices.  Reported as an upper bound on what
             is achievable, NOT as a number anyone has run.

Usage
  python experiments/E12_interp_mask/run_toy.py --plan --out results/E12_interp_mask
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (_ROOT, os.path.join(_ROOT, "tools"),
           os.path.join(_ROOT, "experiments", "E07_key_recovery_2round")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dux.field import make_field                      # noqa: E402
import weighted as WT                                 # noqa: E402
from assemble_fast import weight_vector_count         # noqa: E402


# (label, ledger section, q, active words, layers, cipher, combine, N_w,
#  structures, current log2 data per key)
ROWS = [
    ("DuX(65537) 11/12", "A", "65537", [3], 9, "dux", None, 2152, 1, 16.00),
    ("DuX(65537) 12/12", "A", "65537", [3, 7], 10, "dux", "0001", 9000, 1, 32.00),
    ("DuX(2^16) 11/12", "A", "2^16", [3], 9, "dux", None, 1189, 1, 16.00),
    ("DuX(2^8) 7/12", "A", "2^8", [3], 5, "dux", None, 46, 26, 12.70),
    ("DuX(2^8) 8/12", "A", "2^8", [3, 7, 11, 15], 6, "dux", None, 240, 5, 34.32),
    ("YupX-65537 10/14 (ii)", "A'", "65537", [0], 8, "yux", None, 2155, 1, 16.00),
    ("YupX-65537 11/14", "A'", "65537", [0, 4], 9, "yux", None, 2400, 1, 32.00),
    ("Yu2X-16 10/12 (ii)", "A'", "2^16", [0], 8, "yux", None, 1238, 1, 16.00),
    ("Yu2X-16 11/12", "A'", "2^16", [0, 4], 9, "yux", None, 1300, 1, 32.00),
    ("Yu2X-8 7/12", "A'", "2^8", [0, 4], 5, "yux", None, 34, 40, 21.32),
]

NO_GAIN = [
    ("DuX(65537) 13 轮（理论）",
     "全块 2^64（O11）：自由块必须跑遍 F_q^4，点集不适用；何况第 11 层的余量只有 57"),
    ("DuX(2^16) 12/12（理论）",
     "已经是维数 (16,16,15) 的子空间结构 = 2^47；点集与子空间是同一类节省，"
     "而那里的维数正是由余量定死的"),
    ("YupX-65537 12/14 与 Yu2X-16 12/12（理论）",
     "全块 2^64，权重位置的余量是 0 / -4"),
    ("Yu2X-8 8/12", "全块 2^32（O11）"),
    ("所有 r_KR = 1 的行（9 轮 YupX、10 轮的路线 (i)、5/6 轮各行）",
     "它们用的是普通零和而不是权重；同一换元仍然成立，但省下的不到 1 位"),
]


def min_n_for(D, nw, s, dims):
    """Smallest per-word point count n whose margin admits `nw` weights."""
    norm = 1
    while weight_vector_count(dims, norm) < nw:
        norm += 1
    n = math.ceil((D + norm) / s) + 1
    while s * (n - 1) - D < norm:
        n += 1
    return n, norm


def plan_table():
    out = []
    for (label, sec, q, act, layers, cip, comb, nw, nstruct, cur) in ROWS:
        F = make_field(q)
        pl = WT.plan(F, act, layers, "dec", cip, comb, 1, 1)
        used = WT.used_positions("dec", cip, comb)
        D = max(pl.crit["max_degree_per_position"][layers - 1][p] for p in used)
        s = len(act)
        n1, norm1 = min_n_for(D, nw, s, 1)
        ns, norms = min_n_for(D, nw, s, s)
        row = {
            "row": label, "ledger": sec, "q": q, "active": act, "layers": layers,
            "cipher": cip, "combine": comb, "used_positions": used, "D": D,
            "T_full": pl.crit["threshold"], "margin_full": pl.margin,
            "N_w": nw, "structures": nstruct, "s": s,
            "current_log2_per_key": cur,
            "dims1": {"n": n1, "norm": norm1,
                      "log2_per_key": round(math.log2(nstruct) + s * math.log2(n1), 3),
                      "assembly_factor": round((n1 / F.q) ** s, 4)},
            "dims_s": {"n": ns, "norm": norms, "dims": s,
                       "log2_per_key": round(math.log2(nstruct) + s * math.log2(ns), 3),
                       "assembly_factor": round((ns / F.q) ** s, 4)},
        }
        row["gain_bits_dims1"] = round(cur - row["dims1"]["log2_per_key"], 3)
        row["gain_bits_dims_s"] = round(cur - row["dims_s"]["log2_per_key"], 3)
        out.append(row)
    return out


def markdown(rows):
    L = ["| 行 | 总账 | 结构 | 层 | D | T（全域） | N_w | 现数据 | n/字（dims 1） | 新数据 | 省 | 装配系数 | n（dims = s） | 数据 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = f"{r['structures']} × " if r["structures"] > 1 else ""
        act = ",".join(str(v) for v in r["active"])
        L.append(
            f"| {r['row']} | {r['ledger']} | {st}{r['s']} 字 ({act})"
            f"{'，combine ' + r['combine'] if r['combine'] else ''} | {r['layers']} "
            f"| {r['D']} | {r['T_full']} | {r['N_w']} "
            f"| 2^{r['current_log2_per_key']:.2f} "
            f"| **{r['dims1']['n']}** | **2^{r['dims1']['log2_per_key']:.2f}** "
            f"| {r['gain_bits_dims1']:.2f} 位 "
            f"| ×{r['dims1']['assembly_factor']:.3f} "
            f"| {r['dims_s']['n']} | 2^{r['dims_s']['log2_per_key']:.2f} |")
    return "\n".join(L)


TOY_COMMANDS = """# toy-257, 7 rounds (layer-5 distinguisher + r_KR = 2), 2 keys, n = 250 < 257
python experiments/E07_key_recovery_2round/attack_2round_fast.py --instance toy-257 \\
    --layers 5 --pos 3 --unknown-blocks 0,1,2,3 --outer-blocks 0,1,2,3 \\
    --structures 55 --weights 40 --weight-dims 1 --normalise --seed {seed} \\
    [--point-set 250]

# yuxtoy-257, 7 rounds, 2 keys
python experiments/E07_key_recovery_2round/attack_2round_fast.py --instance yuxtoy-257 \\
    --layers 5 --pos 0 --unknown-blocks 0,1,2,3 --outer-blocks 0,1,2,3 \\
    --structures {N} --weights {Nw} --weight-dims 1 --normalise --seed {seed} \\
    [--point-set {n}]

# characteristic 2: toy-2^4 / yuxtoy-2^4
python experiments/E07_key_recovery_2round/attack_2round_fast.py --instance toy-2^4 \\
    --layers 3 --active 3,7,11 --unknown-blocks 0,1,2,3 --structures {N} \\
    --combine 0001 --weight-dims 1 --normalise --seed {seed} [--point-set {n}]
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    if a.plan or not a.toy:
        rows = plan_table()
        res["plan"] = rows
        res["no_gain"] = [{"row": r, "reason": w} for r, w in NO_GAIN]
        print(markdown(rows))
        print()
        for r, w in NO_GAIN:
            print(f"  no gain: {r} -- {w}")
    if a.toy:
        print(TOY_COMMANDS)
    if a.out and res:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "data_plan.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        with open(os.path.join(a.out, "data_plan.md"), "w") as fh:
            fh.write("# E12 · O15 数据账（W25 步骤 5；只算不跑）\n\n")
            fh.write("由 `python experiments/E12_interp_mask/run_toy.py --plan` 生成；"
                     "D 与 T 由 `tools/zero_sum_criterion.py` 现算，N_w 取总账里实跑用到的权重数。\n\n")
            fh.write(markdown(res["plan"]) + "\n\n")
            fh.write("## 不适用的行\n\n")
            for r, w in NO_GAIN:
                fh.write(f"* **{r}** —— {w}\n")
        print(f"\nwrote {a.out}/data_plan.json and data_plan.md")
    return res


if __name__ == "__main__":
    main()
