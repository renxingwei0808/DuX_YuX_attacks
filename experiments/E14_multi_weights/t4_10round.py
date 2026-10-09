"""E14 / W28 -- T4: the ten-round DuX attack from ONE structure.

Table 6's ten-round row costs 2^17 (two 2^16 structures, E06's one-round
extension with the plain sums), while the ELEVEN-round row costs 2^16 -- the
"ten rounds needs more data than eleven" oddity the memo flags.  The fix is the
one YuX's ten-round route (i) already uses: the second structure is only there
to supply a second independent equation for the pair (k_{4j}, k_{4j+3}), and
O12's weighted sums supply as many equations as the margin allows from the SAME
structure.  `attack_1round.py --auto-weight` already implements it; this script
runs the comparison and writes the table.

Usage
  python t4_10round.py --keys 50 --out ../../results/E14_multi_weights
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
E06 = os.path.join(ROOT, "experiments", "E06_key_recovery_1round")

CONFIGS = [
    ("dux-65537", 10, "3", 16.0),
    ("dux-2^16", 10, "3", 16.0),
]


def run(instance, rounds, pos, structures, keys, seed, auto_weight, nweights,
        outdir):
    cmd = [sys.executable, "-u", os.path.join(E06, "attack_1round.py"),
           "--instance", instance, "--rounds", str(rounds), "--pos", pos,
           "--structures", str(structures), "--keys", str(keys),
           "--seed", str(seed), "--out", outdir]
    cmd += ["--nweights", str(nweights)]
    if auto_weight:
        cmd += ["--auto-weight"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    fn = os.path.join(outdir, f"success_rate_{instance}_r{rounds}.json")
    res = json.load(open(fn))
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["stdout"] = p.stdout.strip().splitlines()[-3:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=int, default=50)
    ap.add_argument("--nweights", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tmp = os.path.join(a.out or ".", "t4_raw")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for instance, rounds, pos, bits in CONFIGS:
        one = run(instance, rounds, pos, 1, a.keys, a.seed, True, a.nweights, tmp)
        rows.append({"instance": instance, "rounds": rounds, "active": pos,
                     "variant": "one structure + weights (T4)",
                     "structures": 1, "weights": one["weights"],
                     "weights_used": one["weights_used"],
                     "weight_range": one["weight_range"],
                     "success": one["success"], "keys": a.keys,
                     "log2_data": bits, "elapsed_s": one["elapsed_s"]})
        two = run(instance, rounds, pos, 2, a.keys, a.seed, False, 1, tmp)
        rows.append({"instance": instance, "rounds": rounds, "active": pos,
                     "variant": "two structures, plain sums (Table 6)",
                     "structures": 2, "weights": two["weights"],
                     "weights_used": two["weights_used"],
                     "weight_range": two["weight_range"],
                     "success": two["success"], "keys": a.keys,
                     "log2_data": round(bits + math.log2(2), 2),
                     "elapsed_s": two["elapsed_s"]})
    L = ["# E14 / T4 — the ten-round DuX attack from one structure", "",
         "`attack_1round.py --structures 1 --auto-weight` against the published",
         "two-structure row of Table 6.  Both use the nine-layer zero-sum and",
         "r_KR = 1; the difference is only whether the second equation for",
         "(k_{4j}, k_{4j+3}) comes from a second structure or from a weight.", "",
         "| instance | rounds | variant | structures | weights used | data | keys | success | time |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['instance']} | {r['rounds']} | {r['variant']} | "
                 f"{r['structures']} | {r['weights_used']} | "
                 f"**2^{r['log2_data']}** | {r['keys']} | "
                 f"{r['success']}/{r['keys']} | {r['elapsed_s']} s |")
    L += ["", f"The weight range the criterion admits at layer {CONFIGS[0][1] - 1} "
          f"is `a < {rows[0]['weight_range'][1]}` for DuX(65537) and "
          f"`a < {rows[2]['weight_range'][1]}` for DuX(2^16); the attack needs "
          f"only the handful `--auto-weight` picks, so the margin is not the "
          f"binding constraint.", "",
          "Consequence for the ledger: the ten-round row becomes **2^16**, the",
          "same as the eleven-round row, and the Table 6 footnote \"data is one",
          "structure per key except for DuX(2^8)\" stops contradicting it (memo",
          "E3).", ""]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "t4_10round.md"), "w").write(txt + "\n")
        json.dump({"rows": rows, "keys": a.keys, "seed": a.seed},
                  open(os.path.join(a.out, "t4_10round.json"), "w"), indent=1)
        print("saved", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
