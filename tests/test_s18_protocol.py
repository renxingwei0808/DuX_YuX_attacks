"""S18 -- the protocol's two new tools.

`merge_protocol.py` turns what `s18_run.sh` measured (GNU `time -v`, a thread
sample every 10 s, the core list) into the `protocol` block of a run's JSON,
which is where docs/measurement_protocol.md rule 1's two acceptance conditions are read from.

`opcounts.py` states every reported row as a count of field multiplications
under ONE formula, so that an executed row and a theoretical one are
comparable.  The sharpest test of that formula is that it reproduces the five
exponents the paper already quotes for its theoretical rows -- they were
computed as `points * W`, which is this formula's dominant term.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "S18_protocol"))

import merge_protocol as MP                                      # noqa: E402
import opcounts as OC                                            # noqa: E402
import verify_rerun as VR                                        # noqa: E402


TIMEV = """\tCommand being timed: "python3 attack_12round.py --tag T"
\tUser time (seconds): 2914.02
\tSystem time (seconds): 0.80
\tPercent of CPU this job got: 99%
\tElapsed (wall clock) time (h:mm:ss or m:ss): 48:35.09
\tMaximum resident set size (kbytes): 6510000
\tExit status: 0
"""

THREADS = """1789960000 procs 2 max_nlwp 1 sum_nlwp 2
1789960010 procs 2 max_nlwp 1 sum_nlwp 2
1789960020 procs 0 max_nlwp 0 sum_nlwp 0
1789960030 procs 3 max_nlwp 1 sum_nlwp 3
"""

ENV = """tag T
cores 31
solve_threads 1
start 2026-09-21T00:00:00-07:00
loadavg_start 40.00 41.00 42.00
end 2026-09-21T00:48:35-07:00
loadavg_end 40.10 41.10 42.10
exit 0
"""


def test_timev_elapsed_is_not_confused_by_the_label_colons():
    """`Elapsed (wall clock) time (h:mm:ss or m:ss): 48:35.09` has colons in
    its own label; the parser must take the value, not the label."""
    tv = MP.parse_timev(TIMEV)
    assert tv["wall_s"] == 2915.09
    assert tv["user_s"] == 2914.02 and tv["sys_s"] == 0.80
    assert tv["maxrss_kb"] == 6510000
    assert MP.parse_elapsed("2:03:04") == 7384.0


def test_protocol_block_states_rule_1s_two_acceptance_conditions():
    prot = MP.build(TIMEV, THREADS, ENV)
    assert prot["cores"] == "31" and prot["solve_threads"] == 1
    assert prot["cpu_s"] == 2914.82
    assert prot["wall_over_cpu"] == 1.0          # 2915.09 / 2914.82
    assert prot["threads"]["max_nlwp"] == 1
    assert prot["threads"]["samples"] == 3       # the procs == 0 line is skipped
    assert prot["single_threaded"] is True
    assert prot["not_preempted"] is True
    assert prot["maxrss_gb"] == 6.21


def test_a_preempted_or_multi_threaded_run_is_rejected():
    slow = TIMEV.replace("48:35.09", "2:00:00")          # wall 2.47x the CPU
    assert MP.build(slow, THREADS, ENV)["not_preempted"] is False
    pooled = THREADS.replace("max_nlwp 1", "max_nlwp 4")
    assert MP.build(TIMEV, pooled, ENV)["single_threaded"] is False


def test_merge_writes_the_block_into_the_runs_json(tmp_path):
    log = tmp_path / "logs"
    log.mkdir()
    (log / "T.timev.txt").write_text(TIMEV)
    (log / "T.threads.log").write_text(THREADS)
    (log / "T.env.txt").write_text(ENV)
    (tmp_path / "s10_T.json").write_text(json.dumps({"rank": 4756, "procs": 1}))
    sys.argv = ["merge_protocol.py", "--tag", "T", "--out", str(tmp_path)]
    MP.main()
    got = json.load(open(tmp_path / "s10_T.json"))
    assert got["rank"] == 4756                      # the record is not touched
    assert got["protocol"]["procs"] == 1
    assert got["protocol"]["threads"]["max_nlwp"] == 1


# --------------------------------------------------------------------------

def test_op_count_formula_is_the_sum_of_its_four_terms():
    row = dict(data_log2=10.0, structures=1, n_w=3, slices=4)
    got = OC.op_counts(row, width=5, rows_=7, cols=11, rank=2)
    assert got["oracle"] == 1024
    assert got["per_point_mults"] == 1024 * 5
    assert got["weighted_mults"] == 3 * 5 * 4
    assert got["row_mults"] == 77
    assert got["solve_mults"] == 77 * 2
    assert got["total_mults"] == 1024 * 5 + 60 + 77 + 154


def test_widths_are_built_from_the_real_layouts():
    """The three layouts the routes actually use.  A change in the
    linearisation must move these numbers, which is why the table builds them
    instead of quoting them."""
    assert OC.width_kr2("dux-65537", 12)[0] == 56550
    assert OC.width_kr2("dux-2^16", 11)[0] == 25200
    assert OC.width_kr2_b("dux-2^16", 12)[0] == 65400
    assert OC.width_cpa("dux-2^16", 8)[0] == 3686


PUBLISHED = {                      # ledger A / A' / C, eprint v1; the tolerance
    "dux2p16_r12_theory": (62.0, 0.5),      # is the precision the paper quotes
    "yu2x16_r12_theory": (79.0, 0.5),       # the exponent with: "~2^79" is one
    "yupx_r12_theory": (80.3, 0.05),        # integer, "2^42.46" two decimals
    "dux65537_r13_theory": (88.0, 0.5),
    "cpa8_dux2p16_rkr1": (42.46, 0.005),
}


@pytest.mark.parametrize("row_id,published,tol", sorted(
    (k, v[0], v[1]) for k, v in PUBLISHED.items()))
def test_the_published_theoretical_exponents_come_back(row_id, published, tol):
    """Every exponent the paper already states for a theoretical row is
    reproduced by the unified formula to within the rounding it was quoted
    with.  If one of these moves, docs/measurement_protocol.md section 4 has to say so."""
    rec = next(r for r in OC.build_cached() if r["id"] == row_id)
    assert abs(rec["total_log2"] - published) <= tol, rec


def test_us_per_point_is_only_reported_where_the_points_dominate(tmp_path):
    """On a point-set row a slice IS a point, so `N_w * W * slices` dwarfs
    `oracle * W`: dividing the assembly time by the point count would measure
    the weighted sum and call it a per-point cost.  The gate must drop it."""
    rec_pt = dict(id="x", label="point-bound", instance="dux-2^8",
                  rounds=8, structures=1, oracle=2. ** 24,
                  per_point_mults=1.1e12, assemble_mults=1.16e12,
                  solve_mults=1e10, json_path=str(tmp_path / "r.json"))
    rec_wt = dict(rec_pt, id="y", label="weight-bound",
                  per_point_mults=1.0e9, assemble_mults=1.25e12)
    (tmp_path / "r.json").write_text(json.dumps(
        {"assemble_s": 1000.0, "solve_s": 10.0, "per_structure": [{}],
         "protocol": {"single_threaded": True, "not_preempted": True}}))
    cal = OC.calibrate([rec_pt, rec_wt], str(tmp_path))
    by = {c["id"]: c for c in cal["runs"]}
    assert by["x"]["point_bound"] is True and by["x"]["us_per_point"] is not None
    assert by["y"]["point_bound"] is False and by["y"]["us_per_point"] is None
    assert by["x"]["ns_per_mult_solve"] == round(10.0 * 1e9 / 1e10, 3)


def test_the_cpa_drivers_json_shape_is_understood(tmp_path):
    """`kr2_cpa.py` records `equations_per_structure` and `structures`, not
    `equations`; the table must not silently fall back to its own guess."""
    p = tmp_path / "kr2cpa_X.json"
    p.write_text(json.dumps({"equations_per_structure": 1600, "structures": 2,
                             "unknowns": 3248, "rank": 1086}))
    rows_, cols, rank, _ = OC.from_json(str(p))
    assert (rows_, cols, rank) == (3200, 3248, 1086)


# --------------------------------------------------------------------------

def test_a_frozen_table_row_can_be_selected_by_instance_and_variant(tmp_path,
                                                                    monkeypatch):
    """Some rows were frozen inside a table of several variants (the T4 row
    sits next to its two-structure control).  The verifier has to pick the
    right one, or it compares a run against the wrong record -- which is
    exactly how the `--nweights` and `--coords` omissions were caught."""
    monkeypatch.setattr(VR, "ROOT", str(tmp_path))
    (tmp_path / "t.json").write_text(json.dumps({"rows": [
        {"instance": "dux-65537", "variant": "one structure + weights (T4)",
         "structures": 1, "log2_data": 16.0},
        {"instance": "dux-65537", "variant": "two structures, plain sums",
         "structures": 2, "log2_data": 17.0}]}))
    got, name = VR.load("t.json#dux-65537|one structure + weights (T4)")
    assert got["structures"] == 1 and got["log2_data"] == 16.0
    assert name.endswith("(T4)")
    miss, _ = VR.load("t.json#dux-65537|no such variant")
    assert miss is None


def test_differently_named_fields_are_compared_as_a_pair(tmp_path, monkeypatch):
    """`attack_1round.py` writes `data_log2_per_structure` where the frozen
    table wrote `log2_data`; a (new, old) pair must compare them, and must
    still report a real difference."""
    monkeypatch.setattr(VR, "ROOT", str(tmp_path))
    (tmp_path / "new.json").write_text(json.dumps(
        {"keys": 50, "success": 50, "data_log2_per_structure": 16}))
    (tmp_path / "old.json").write_text(json.dumps(
        {"keys": 50, "success": 50, "log2_data": 16}))
    monkeypatch.setattr(VR, "PAIRS", [
        ("new.json", "old.json",
         ("keys", "success", ("data_log2_per_structure", "log2_data")))])
    out = tmp_path / "v.json"
    monkeypatch.setattr(sys, "argv", ["verify_rerun.py", "--out", str(out)])
    assert VR.main() == 0
    assert json.load(open(out))["pairs"][0]["same"] is True

    (tmp_path / "old.json").write_text(json.dumps(
        {"keys": 50, "success": 50, "log2_data": 17}))
    assert VR.main() == 1
    assert json.load(open(out))["pairs"][0]["differences"] == {
        "data_log2_per_structure": [17, 16]}


def test_the_field_comes_from_the_registry_not_the_instance_name():
    """`yu2x-8` is F_{2^8} and `yupx-65537` is F_p, but neither name says so;
    the calibration constants differ by an order of magnitude between the two
    fields, so a name-based guess would mix them."""
    assert OC.field_of("yu2x-8", 7) == "char2"
    assert OC.field_of("yu2x-16", 11) == "char2"
    assert OC.field_of("dux-2^16", 12) == "char2"
    assert OC.field_of("yupx-65537", 10) == "prime"
    assert OC.field_of("dux-65537", 12) == "prime"


def test_a_replaced_sampler_splits_the_thread_samples():
    """The first sampler matched `gf2nsolve` by name and so counted another
    job's solve as this run's.  When it is replaced in flight the log carries
    a marker; the headline `max_nlwp` must then come from the corrected
    segment, with the earlier one reported separately rather than dropped."""
    log = ("1 procs 3 max_nlwp 1 sum_nlwp 3\n"
           "2 procs 5 max_nlwp 19 sum_nlwp 23\n"
           "# 2026-09-21T08:00:00-07:00 sampler replaced: own process tree only\n"
           "3 procs 3 max_nlwp 1 sum_nlwp 3\n"
           "4 procs 3 max_nlwp 1 sum_nlwp 3\n")
    th = MP.parse_threads(log)
    assert th["max_nlwp"] == 1 and th["samples"] == 2
    assert th["sampler_replaced"] is True
    assert th["pre_swap_samples"] == 2 and th["pre_swap_max_nlwp"] == 19
    # a log with no marker is unchanged
    plain = MP.parse_threads("1 procs 3 max_nlwp 1 sum_nlwp 3\n")
    assert plain["max_nlwp"] == 1 and "sampler_replaced" not in plain


# --------------------------------------------------------------------------

def test_the_run_table_reads_every_driver_shape():
    """The run table is generated, not transcribed, so it has to understand
    all four JSON shapes: the streamed CCA driver (`equations`), the CPA one
    (`equations_per_structure` x `structures`), the r_KR = 1 success-rate one
    (`success` / `keys`) and `partial_lcomb` (rank inside `results`)."""
    import run_table as RT
    assert RT.system({"equations": 15120, "rank": 12056, "pinned": 5812,
                      "inner_correct": 16}) == "15120 / 12056 / 5812 / 16/16"
    assert RT.system({"equations_per_structure": 8, "structures": 172,
                      "rank": 1336, "kappa_correct": 16}).startswith("1376 / 1336")
    assert RT.system({"success": 50, "keys": 50}) == "50/50 密钥"
    assert RT.system({"equations": 480,
                      "results": [{"rank": 136}]}).startswith("480 / 136")
    assert RT.hms(26840) == "7:27:20" and RT.hms(2915) == "48:35"
