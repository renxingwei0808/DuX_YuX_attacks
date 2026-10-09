"""S18 -- build the run table from the runs' own JSONs.

The run tables of the paper and its supplement need this
table: for every protocol run, what was measured and under what conditions.
Writing it by hand invites transcription errors of exactly the kind
`verify_rerun.py` exists to catch, so it is generated instead.

Columns follow docs/measurement_protocol.md: the attack's own numbers (rows / rank / pinned /
16-16), then how it was measured (mode, cores, processes, solve threads),
then what was measured (wall, CPU, peak RSS), then the two acceptance numbers
(`max_nlwp`, `wall/(user+sys)`), and finally the machine's concurrent load --
which docs/measurement_protocol.md section 5.1 requires, because the all-core turbo bin moves the
wall clock by up to 1.5x without moving the ratio at all.

    python run_table.py [--out results/S18_protocol/run_table.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
S18 = os.path.join(ROOT, "results", "S18_protocol")

# tag -> the ledger row it reports, in the order the table should list them
LABEL = [
    ("S18_mp_dux2p16_r12_2w_cheap12_seed11", "DuX(2^16) 12/12 @2^32", "11"),
    ("S18_1t_dux65537_r12_k13_2axis_seed11", "DuX(65537) 12/12 @2^29", "11"),
    ("S18_mp_dux65537_r12_k14_seed11", "DuX(65537) 12/12 @2^30 (variant)", "11"),
    ("S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11", "DuX(2^8) 8/12 @2^24", "11"),
    ("S18_mp_yu2x8_r8_fb0_2axis_seed11", "Yu2X-8 8/12 @2^32", "11"),
    ("S18_1t_yu2x8_r7_2w_2axis_seed11", "Yu2X-8 7/12 @2^17.58", "11"),
    ("S18_1t_dux65537_r11_coset15_seed2026", "DuX(65537) 11/12 @2^15", "2026"),
    ("S18_1t_dux2p16_r11_pset41735_seed2026", "DuX(2^16) 11/12 @2^15.35", "2026"),
    ("S18_1t_t4_dux65537_r10_seed2026", "DuX(65537) 10/12 @2^16 (T4)", "2026"),
    ("S18_1t_t4_dux2p16_r10_seed2026", "DuX(2^16) 10/12 @2^16 (T4)", "2026"),
    ("S18_1t_dux2p8_r7_seed2026", "DuX(2^8) 7/12 @2^12.70", "2026"),
    ("S18_1t_yupx_r10_lcomb_seed2026", "YupX 10/14 (i) @2^16", "2026"),
    ("S18_1t_yupx_r10_pset28239_seed2026", "YupX 10/14 (ii') @2^14.79", "2026"),
    ("S18_1t_yupx_r10_coset15_seed2026", "YupX 10/14 (iii) @2^15", "2026"),
    ("S18_1t_yu2x16_r10_lcomb_seed2026", "Yu2X-16 10/12 (i) @2^16", "2026"),
    ("S18_1t_cpa7_dux65537_seed2026", "CPA DuX(65537) 7/12 @2^18.81", "2026"),
    ("S18_1t_cpa7_dux2p16_seed2026", "CPA DuX(2^16) 7/12 @2^18.58", "2026"),
    ("S18_1t_cpa8_dux65537_seed2026", "CPA DuX(65537) 8/12 @2^16", "2026"),
    ("S18_1t_cpa8_dux2p16_seed2026", "CPA DuX(2^16) 8/12 @2^16", "2026"),
    ("S18_1t_cpa8_dux2p16_nw_seed2026", "CPA DuX(2^16) 8/12 @2^23.43", "2026"),
]


def hms(s):
    if s is None:
        return "—"
    s = int(round(s))
    h, m = divmod(s, 3600)
    m, sec = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, sec) if h else "%d:%02d" % (m, sec)


def load(tag):
    for p in (f"s10_{tag}.json", f"kr2cpa_{tag}.json", f"{tag}.json"):
        fn = os.path.join(S18, p)
        if os.path.exists(fn):
            return json.load(open(fn)), os.path.relpath(fn, ROOT)
    hits = sorted(glob.glob(os.path.join(S18, tag, "*.json")))
    if hits:
        return json.load(open(hits[0])), os.path.relpath(hits[0], ROOT)
    return None, None


def system(d):
    rows = d.get("equations")
    if rows is None and d.get("equations_per_structure"):
        rows = d["equations_per_structure"] * d.get("structures", 1)
    rank = d.get("rank")
    if rank is None and d.get("results"):
        rank = d["results"][0].get("rank")
    ok = d.get("inner_correct") or d.get("kappa_correct")
    if ok is None and d.get("success") is not None:
        return f"{d['success']}/{d.get('keys')} 密钥"
    bits = [str(rows or "—"), str(rank or "—"), str(d.get("pinned") or "—"),
            f"{ok}/16" if ok is not None else "—"]
    return " / ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(S18, "run_table.md"))
    a = ap.parse_args()
    head = ("| 报告的行 | 密钥 | 行 / 秩 / 被定住 / 密钥字 | 模式 | 核 | 进程 | "
            "消元线程 | 墙钟 | CPU 核时 | 峰值 | `max_nlwp` | `wall/cpu` | "
            "同时段负载 | 记录 |")
    out = [head, "|" + "---|" * 14]
    missing = []
    for tag, label, seed in LABEL:
        d, path = load(tag)
        if d is None:
            missing.append(tag)
            continue
        p = d.get("protocol") or {}
        th = p.get("threads") or {}
        procs = p.get("procs") or d.get("procs") or 1
        mode = "规范 3（多进程不间断）" if procs and procs > 1 else "规范 1（单线程）"
        cpu = p.get("cpu_s")
        la = (p.get("loadavg_start") or "").split(" ")[0]
        lb = (p.get("loadavg_end") or "").split(" ")[0]
        out.append("| %s | %s | %s | %s | `%s` | %s | %s | **%s** | %s | %s | "
                   "**%s** | **%s** | %s → %s | `%s` |"
                   % (label, seed, system(d), mode, p.get("cores", "—"), procs,
                      p.get("solve_threads", "—"), hms(p.get("wall_s")),
                      ("%.3f" % (cpu / 3600.0)) if cpu else "—",
                      f"{p.get('maxrss_gb')} GB" if p.get("maxrss_gb") else "—",
                      th.get("max_nlwp", "—"), p.get("wall_over_cpu", "—"),
                      la or "—", lb or "—", path))
    text = "\n".join(out) + "\n"
    if missing:
        text += "\n未完成：" + "、".join(missing) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(
        "# S18 运行表（由 `experiments/S18_protocol/run_table.py` 从各运行的 JSON 生成）\n\n"
        "列的口径见 `docs/measurement_protocol.md`：`max_nlwp` 与 `wall/cpu` 是规范 1 的\n"
        "两个验收量（单线程行要 1 与 ≤ 1.1）；**同时段负载**是 §5.1 要求的——\n"
        "全核睿频会在不改变 `wall/cpu` 的情况下把墙钟改变最多 1.5 倍。\n\n" + text)
    print(text)


if __name__ == "__main__":
    main()
