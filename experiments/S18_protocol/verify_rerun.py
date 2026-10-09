"""S18 -- check every protocol run against the record it has to reproduce.

The protocol's hard rule (docs/measurement_protocol.md, rule 1) is
that a rerun changes only HOW a row was measured, never WHAT it measured: the
rank, the pinned monomials, the 16/16 and the row/column counts must come out
exactly as they stand in the ledger.  This script states that as a list of
(rerun, frozen record, invariants) triples and checks them, so "identical
numbers" is a claim the reader can re-run rather than take on trust.

The third key (seed 11) of a headline row is compared against the SAME cell's
seed-2026 run: a different master key must give the same rank, the same pinned
count and the same 16/16 -- that is exactly the evidence a third key adds.

    python verify_rerun.py [--out results/S18_protocol/verify.json]

Exit status is non-zero if any invariant moved.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")

S18 = "results/S18_protocol"

# Invariants per driver shape.  `equations` is the row count of the streamed
# CCA driver; the CPA drivers record `equations_per_structure` + `structures`;
# `attack_1round.py` records `keys` / `success`; `partial_lcomb.py` records a
# `results` list, one entry per key.
CCA = ("equations", "unknowns", "rank", "free", "pinned", "inner_correct",
       "data_log2")
CCA_KEY = ("unknowns", "rank", "pinned", "inner_correct")      # across keys
CPA = ("equations_per_structure", "structures", "unknowns", "rank",
       "kappa_correct", "data_log2")
CP1 = ("equations_per_structure", "structures", "unknowns", "data_log2",
       "kappa_monomials_per_block", "moments")
RATE = ("keys", "success", "structures", "data_log2_per_structure")
# `attack_1round.py` writes `data_log2_per_structure`; the T4 table that froze
# this row wrote the same number as `log2_data`.  A pair (new, old) compares
# two differently named fields.
RATE_T4 = ("keys", "success", "structures", "weights_used",
           ("data_log2_per_structure", "log2_data"))
LCOMB = ("weights", "unknowns", "equations_per_structure", "structures",
         "data_log2", "keys")

PAIRS = [
    # ---- item 2: same key (2026), same command, one core ------------------
    (f"{S18}/s10_S18_1t_dux65537_r11_coset15_seed2026.json",
     "results/R9_server/s10_S20_dux65537_r11_coset15_0001_seed2026.json", CCA),
    (f"{S18}/s10_S18_1t_dux2p16_r11_pset41735_seed2026.json",
     "results/R8_server/s10_R8_dux2p16_r11_pset41735_seed2026.json", CCA),
    (f"{S18}/s10_S18_1t_dux2p8_r7_seed2026.json",
     "results/E07_key_recovery_2round/s9/s10_S9_step4_dux2p8_r7_seed2026.json",
     CCA),
    (f"{S18}/s10_S18_1t_yupx_r10_pset28239_seed2026.json",
     "results/R8_server/s10_R8_yupx_r10_pset28239_seed2026.json", CCA),
    (f"{S18}/s10_S18_1t_yupx_r10_coset15_seed2026.json",
     "results/R6_third_keys/s10_R6D3_yupx_r10_coset15_seed2026.json", CCA),
    (f"{S18}/S18_1t_yupx_r10_lcomb_seed2026.json",
     "results/Y05_key_recovery/Y05-A4_yupx_r10.json", LCOMB),
    (f"{S18}/S18_1t_yu2x16_r10_lcomb_seed2026.json",
     "results/Y05_key_recovery/Y05-A4_yu2x16_r10.json", LCOMB),
    (f"{S18}/kr2cpa_S18_1t_cpa8_dux2p16_seed2026.json",
     "results/E10_cpa/s9/kr2cpa_S9_cpa8_dux2p16_seed2026.json", CPA),
    (f"{S18}/kr2cpa_S18_1t_cpa8_dux65537_seed2026.json",
     "results/E10_cpa/s9/kr2cpa_S9_cpa8_dux65537_seed2026.json", CPA),
    (f"{S18}/S18_1t_cpa8_dux2p16_nw_seed2026/*.json",
     "results/E10_cpa/w14/kr2cpa_dux-2^16_r8_s2026.json",
     ("unknowns", "rank")),
    (f"{S18}/S18_1t_cpa7_dux65537_seed2026/*.json",
     "results/E10_cpa/cp_attack_dux-65537_r7_1.json", CP1),
    (f"{S18}/S18_1t_cpa7_dux2p16_seed2026/*.json",
     "results/E10_cpa/cp_attack_dux-2^16_r7_1.json", CP1),
    # the T4 rows: `t4_raw/` holds the TWO-structure control (the Table 6
    # row); the one-structure T4 row lives in the table `t4_10round.json`.
    (f"{S18}/S18_1t_t4_dux65537_r10_seed2026/*.json",
     "results/E14_multi_weights/t4_10round.json"
     "#dux-65537|one structure + weights (T4)", RATE_T4),
    (f"{S18}/S18_1t_t4_dux2p16_r10_seed2026/*.json",
     "results/E14_multi_weights/t4_10round.json"
     "#dux-2^16|one structure + weights (T4)", RATE_T4),

    # ---- item 1: the third key (11) against the frozen first key (2026) ---
    (f"{S18}/s10_S18_mp_dux2p16_r12_2w_cheap12_seed11.json",
     "results/R9_server/s10_S19_dux2p16_r12_2w_cheap12_seed2026.json", CCA_KEY),
    (f"{S18}/s10_S18_1t_yu2x8_r7_2w_2axis_seed11.json",
     "results/R9_server/s10_S21_yu2x8_r7_2w_2axis_seed2026.json", CCA_KEY),
    (f"{S18}/s10_S18_1t_dux65537_r12_k13_2axis_seed11.json",
     "results/R9_server/s10_S20_dux65537_r12_mixed13full_2axis_seed2026.json",
     CCA_KEY),
    (f"{S18}/s10_S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11.json",
     "results/R9_server/s10_S19_dux2p8_r8_3w_cheap12_2axis_seed2026.json",
     CCA_KEY),
    (f"{S18}/s10_S18_mp_yu2x8_r8_fb0_2axis_seed11.json",
     "results/R9_server/s10_S21_yu2x8_r8_fb0_2axis_seed2026.json", CCA_KEY),
    (f"{S18}/s10_S18_mp_dux65537_r12_k14_seed11.json",
     "results/R9_server/s10_S20_dux65537_r12_mixed14full_seed2026.json",
     CCA_KEY),
]


def load(pattern):
    """`path` or `path/glob`, optionally `...#<instance>|<variant>` to pick one
    row out of a table that froze several variants in a `rows` list."""
    pattern, _, sel = pattern.partition("#")
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    hits = sorted(glob.glob(p))
    if not hits:
        return None, None
    d = json.load(open(hits[0]))
    name = os.path.relpath(hits[0], ROOT)
    if sel:
        inst, _, variant = sel.partition("|")
        d = next((r for r in d.get("rows", [])
                  if r.get("instance") == inst and r.get("variant") == variant),
                 None)
        name += "#" + sel
    return d, name


def key_results(d):
    """`partial_lcomb.py` / the CPA drivers list one entry per key; every key
    must give the same (determined, correct, rank)."""
    out = []
    for r in d.get("results") or []:
        out.append(tuple(r.get(k) for k in ("determined", "correct", "rank")
                         if k in r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S18, "verify.json"))
    a = ap.parse_args()
    report, bad, missing = [], 0, 0
    for new_pat, old_pat, keys in PAIRS:
        new, new_p = load(new_pat)
        old, old_p = load(old_pat)
        if new is None:
            report.append({"rerun": new_pat, "status": "not run yet"})
            missing += 1
            continue
        if old is None:
            report.append({"rerun": new_p, "status": "no frozen record",
                           "frozen": old_pat})
            bad += 1
            continue
        diff = {}
        for k in keys:
            kn, ko = k if isinstance(k, tuple) else (k, k)
            if old.get(ko) != new.get(kn):
                diff[kn] = [old.get(ko), new.get(kn)]
        kr_new, kr_old = key_results(new), key_results(old)
        if kr_new and kr_old and kr_new[0] != kr_old[0]:
            diff["results[0]"] = [kr_old[0], kr_new[0]]
        prot = new.get("protocol") or {}
        rec = {"rerun": new_p, "frozen": old_p,
               "checked": [k if isinstance(k, str) else list(k) for k in keys],
               "differences": diff,
               "same": not diff,
               "wall_over_cpu": prot.get("wall_over_cpu"),
               "max_nlwp": (prot.get("threads") or {}).get("max_nlwp"),
               "single_threaded": prot.get("single_threaded"),
               "not_preempted": prot.get("not_preempted")}
        report.append(rec)
        if diff:
            bad += 1
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump({"pairs": report}, open(os.path.join(ROOT, a.out), "w"), indent=1)
    for r in report:
        if r.get("status"):
            print(f"  ..  {r['rerun']}: {r['status']}")
        elif r["same"]:
            print(f"  OK  {os.path.basename(r['rerun'])[:52]:<52} "
                  f"= {os.path.basename(r['frozen'])[:44]}")
        else:
            print(f"  XX  {os.path.basename(r['rerun'])}: {r['differences']}")
    print(f"{len(report) - bad - missing} identical, {bad} different, "
          f"{missing} not run yet")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
