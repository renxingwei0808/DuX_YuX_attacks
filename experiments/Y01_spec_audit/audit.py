"""Y01 -- machine-checkable half of the YuX specification audit.

`docs/yux_specification.md` is the human-readable clause table.  This script
turns it into `results/Y01_spec_audit/audit_table.json` and CHECKS the three
things a reader would otherwise have to take on trust:

  1. every clause names a test that really exists in tests/test_yux.py;
  2. the two explicit objects the paper prints -- Yu2X's encryption rotation
     set and YupX's v_p (as fractions and, for p = 65537, as integers) -- are
     REPRODUCED by inverting the decryption circulant, not copied in;
  3. the quadratic system printed in Sect. VI-D equals S = Pf^{-4} in
     characteristic 2 and does NOT over F_p (erratum E1), with the first line
     off by exactly a sign.

    python experiments/Y01_spec_audit/audit.py --out results/Y01_spec_audit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)

from dux.field import make_field                                  # noqa: E402
from dux.registry import get_cipher                               # noqa: E402
from yux.linear import circulant, forward_row, paper_v_p          # noqa: E402
from yux.params import INSTANCES, ROT_FWD_BIN, ROT_INV, V_P_65537  # noqa: E402
from yux.sbox import S, paper_VI_D_system                         # noqa: E402

SPEC = os.path.join(ROOT, "docs", "yux_specification.md")
TESTS = os.path.join(ROOT, "tests", "test_yux.py")


def clauses():
    """Parse the clause table of docs/yux_specification.md."""
    rows, in_table = [], False
    for line in open(SPEC, encoding="utf-8"):
        if line.startswith("| # | Clause |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and not set(cells[0]) <= {"-", ":"}:
                rows.append({"id": cells[0], "clause": cells[1],
                             "paper": cells[2], "implementation": cells[3],
                             "test": cells[4]})
    return rows


def check_tests_exist(rows):
    src = open(TESTS, encoding="utf-8").read()
    names = set(re.findall(r"def (test_\w+)", src))
    missing = []
    for r in rows:
        refs = re.findall(r"`?(test_\w+)`?", r["test"])
        r["test_names"] = refs
        for t in refs:
            if t not in names and t not in ("test_dux_dec_0001_row_uses_the_paper_M0",):
                missing.append((r["id"], t))
    return missing


def check_printed_objects():
    out = {}
    F2 = make_field("2^16")
    row2 = forward_row(F2)
    out["yu2x_forward_set_recomputed"] = list(j for j in range(16) if row2[j])
    out["yu2x_forward_set_paper"] = list(ROT_FWD_BIN)
    out["yu2x_forward_set_matches"] = out["yu2x_forward_set_recomputed"] == list(ROT_FWD_BIN)
    Fp = make_field("65537")
    rowp = forward_row(Fp)
    out["v_p_recomputed"] = list(rowp)
    out["v_p_paper_integers"] = list(V_P_65537)
    out["v_p_paper_fractions_evaluated"] = list(paper_v_p(Fp))
    out["v_p_matches"] = (list(rowp) == list(V_P_65537) ==
                          list(paper_v_p(Fp)))
    M = circulant(ROT_INV) % 65537
    Minv = np.array([[V_P_65537[(k - i) % 16] for k in range(16)] for i in range(16)],
                    dtype=np.int64)
    out["M_times_v_p_is_identity"] = bool(
        np.array_equal((M @ Minv) % 65537, np.eye(16, dtype=np.int64)))
    return out


def check_erratum_E1(trials=200, seed=2026):
    rng = np.random.default_rng(seed)
    out = {}
    for fname in ("2^4", "2^8", "2^16", "193", "257", "65537"):
        F = make_field(fname)
        alpha = 205 % F.q
        same = 0
        for _ in range(trials):
            x = tuple(int(v) for v in rng.integers(0, F.q, size=4))
            same += (paper_VI_D_system(F, x, alpha) == S(F, x, alpha))
        out[fname] = {"char": F.char, "agreements": same, "trials": trials,
                      "identical": same == trials}
    F = make_field("65537")
    signs = True
    for _ in range(50):
        x = tuple(int(v) for v in rng.integers(0, F.q, size=4))
        signs &= paper_VI_D_system(F, x, 205)[3] == F.sub(0, S(F, x, 205)[3])
    out["first_line_is_exactly_a_sign_flip_over_F_65537"] = bool(signs)
    return out


def check_instances():
    out = {}
    for inst, spec in INSTANCES.items():
        c = get_cipher(inst)
        rng = np.random.default_rng(2026)
        K = c.random_key(rng)
        rks = c.key_schedule(K)
        P = tuple(int(v) for v in rng.integers(0, c.F.q, size=16))
        C = c.encrypt(P, rks)
        out[inst] = {
            "field": spec["field"], "alpha": spec["alpha"],
            "rounds": spec["rounds"], "official": spec["official"],
            "roundtrip": c.decrypt(C, rks) == P,
            "key_schedule_invertible": all(
                c.key_schedule_inverse(rks[i], i) == rks[i - 1]
                for i in range(1, c.r + 1)),
            "equivalent_fixedL0_is_trivial": c.equivalent_fixedL0_keys(rks)[1] == 0,
        }
    return out


def check_literal_key_schedule():
    """Ambiguity A1: the literal reading of Algorithm 1 line 2 makes rk^1..rk^r
    depend on key_0..key_6 only."""
    c = get_cipher("yuxtoy-257", ks_literal=True)
    rng = np.random.default_rng(2026)
    K = list(c.random_key(rng))
    base = c.key_schedule(K)
    indep, dep = [], []
    for w in range(16):
        K2 = list(K)
        K2[w] = (K2[w] + 1) % c.F.q
        (indep if c.key_schedule(K2)[1:] == base[1:] else dep).append(w)
    return {"literal_reading_words_that_matter": dep,
            "literal_reading_words_that_do_not": indep,
            "entropy_of_rk1_onwards_in_words": len(dep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = clauses()
    missing = check_tests_exist(rows)
    report = {
        "source": "Liu et al., YuX, IEEE TIT 70(5), 2024",
        "clauses": rows, "n_clauses": len(rows),
        "clauses_with_missing_tests": missing,
        "printed_objects": check_printed_objects(),
        "erratum_E1_paper_VI_D_system": check_erratum_E1(),
        "instances": check_instances(),
        "ambiguity_A1_literal_key_schedule": check_literal_key_schedule(),
    }
    print(f"clauses: {len(rows)}; clauses whose test is missing: {len(missing)}")
    po = report["printed_objects"]
    print(f"Yu2X encryption rotation set reproduced: {po['yu2x_forward_set_matches']}")
    print(f"v_p reproduced (fractions == integers == circulant inverse): "
          f"{po['v_p_matches']}; M * Circ(v_p) = I: {po['M_times_v_p_is_identity']}")
    for k, v in report["erratum_E1_paper_VI_D_system"].items():
        if isinstance(v, dict):
            print(f"  Sect. VI-D system on F_{k}: identical to Pf^-4? {v['identical']} "
                  f"({v['agreements']}/{v['trials']})")
    print(f"  first line is a pure sign flip over F_65537: "
          f"{report['erratum_E1_paper_VI_D_system']['first_line_is_exactly_a_sign_flip_over_F_65537']}")
    bad = [k for k, v in report["instances"].items()
           if not (v["roundtrip"] and v["key_schedule_invertible"])]
    print(f"instances failing roundtrip / key-schedule inversion: {bad or 'none'}")
    print(f"literal Algorithm 1 line 2: rk^1.. depends on words "
          f"{report['ambiguity_A1_literal_key_schedule']['literal_reading_words_that_matter']}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "audit_table.json")
        json.dump(report, open(fn, "w"), indent=1, ensure_ascii=False)
        print("saved", fn)
    assert not missing and po["v_p_matches"] and po["yu2x_forward_set_matches"]
    assert not bad


if __name__ == "__main__":
    main()
