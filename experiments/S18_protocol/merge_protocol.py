"""S18 -- fold the protocol's measurements into a run's JSON.

`scripts/run_pinned.sh` leaves three files next to every run:

    logs/<tag>.timev.txt    GNU `/usr/bin/time -v` output
    logs/<tag>.threads.log  one line every 10 s:
                            "<unix ts> procs <n> max_nlwp <m> sum_nlwp <s>"
    logs/<tag>.env.txt      cores, solve threads, load average, start/end

This script parses them and writes a `"protocol"` block into
`results/S18_protocol/s10_<tag>.json`, so that docs/measurement_protocol.md rule 1's two
acceptance conditions can be checked straight from the JSON:

    protocol.threads.max_nlwp == 1          (the run really was single-threaded)
    protocol.wall_over_cpu    <= 1.1        (it was not preempted)

`wall_over_cpu` is wall / (user + sys).  For a multi-process run (rule 3) the
ratio is meaningless -- `procs` is then > 1 and the field is reported but not
used as an acceptance test.

    python merge_protocol.py --tag S18_1t_dux65537_r11_seed2026 \
        --out results/S18_protocol
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def parse_timev(text):
    """GNU `time -v` output -> {wall_s, user_s, sys_s, maxrss_kb, cpu_pct}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"User time \(seconds\):\s*([\d.]+)", line)
        if m:
            out["user_s"] = float(m.group(1))
        m = re.match(r"System time \(seconds\):\s*([\d.]+)", line)
        if m:
            out["sys_s"] = float(m.group(1))
        m = re.match(r"Maximum resident set size \(kbytes\):\s*(\d+)", line)
        if m:
            out["maxrss_kb"] = int(m.group(1))
        m = re.match(r"Percent of CPU this job got:\s*(\d+)%", line)
        if m:
            out["cpu_pct"] = int(m.group(1))
        m = re.match(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)$",
                     line)
        if m:
            out["wall_s"] = parse_elapsed(m.group(1))
    return out


def parse_elapsed(s):
    """'h:mm:ss' or 'm:ss.ss' -> seconds."""
    parts = s.strip().split(":")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return round(sec, 2)


def parse_threads(text):
    """The sampler's lines -> {samples, max_nlwp, max_procs, first_ts, last_ts}.

    `max_nlwp` is the largest thread count any single process of the run ever
    showed; it is 1 exactly when every process stayed single-threaded -- that
    is the number rule 1 asks for.  `max_procs` is a loose upper bound: the
    sampler matches on the run's `--tag`, and its own `grep` carries the tag
    on its command line, so up to two helper processes can be counted with
    the run's own.  The authoritative worker count is `protocol.procs`, taken
    from the driver's own record."""
    n = mx = mp = 0
    first = last = None
    pre_n = pre_mx = 0
    swapped = False
    for line in text.splitlines():
        line = line.strip()
        # The first version of the sampler matched `gf2nsolve` BY NAME, which
        # on a shared machine also catches another job's solve and reports its
        # thread count as this run's.  A run whose sampler was replaced in
        # flight carries this marker; the headline numbers are then taken from
        # the corrected segment, and the earlier one is reported separately.
        if line.startswith("#") and "sampler replaced" in line:
            pre_n, pre_mx = n, mx
            n = mx = mp = 0
            swapped = True
            continue
        m = re.match(r"(\d+) procs (\d+) max_nlwp (\d+) sum_nlwp (\d+)", line)
        if not m:
            continue
        ts, procs, nlwp = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if procs == 0:                       # between exec and the first fork
            continue
        n += 1
        mx = max(mx, nlwp)
        mp = max(mp, procs)
        first = ts if first is None else min(first, ts)
        last = ts if last is None else max(last, ts)
    out = {"samples": n, "max_nlwp": mx, "max_procs": mp,
           "first_ts": first, "last_ts": last}
    if swapped:
        out["sampler_replaced"] = True
        out["pre_swap_samples"] = pre_n
        out["pre_swap_max_nlwp"] = pre_mx
    return out


def parse_env(text):
    out = {}
    for line in text.splitlines():
        k, _, v = line.strip().partition(" ")
        if k in ("cores", "solve_threads", "start", "end", "exit",
                 "loadavg_start", "loadavg_end"):
            out[k] = v
    return out


def build(timev_text, threads_text, env_text):
    tv = parse_timev(timev_text)
    th = parse_threads(threads_text)
    env = parse_env(env_text)
    cpu = tv.get("user_s", 0.0) + tv.get("sys_s", 0.0)
    prot = {"cores": env.get("cores"),
            "solve_threads": int(env.get("solve_threads", 0) or 0),
            "wall_s": tv.get("wall_s"), "user_s": tv.get("user_s"),
            "sys_s": tv.get("sys_s"), "cpu_s": round(cpu, 2),
            "core_hours": round(cpu / 3600.0, 3) if cpu else None,
            "maxrss_kb": tv.get("maxrss_kb"),
            "maxrss_gb": (round(tv["maxrss_kb"] / 1048576.0, 2)
                          if "maxrss_kb" in tv else None),
            "cpu_pct": tv.get("cpu_pct"),
            "wall_over_cpu": (round(tv["wall_s"] / cpu, 3)
                              if cpu and tv.get("wall_s") else None),
            "threads": th,
            "loadavg_start": env.get("loadavg_start"),
            "loadavg_end": env.get("loadavg_end"),
            "start": env.get("start"), "end": env.get("end"),
            "exit": env.get("exit"),
            "machine": "2 x Intel Xeon Gold 6230R (26 cores / 52 threads each, "
                       "2.10 GHz base, 4.00 GHz max turbo), 251 GB RAM"}
    prot["single_threaded"] = bool(th["samples"] and th["max_nlwp"] == 1)
    prot["not_preempted"] = bool(prot["wall_over_cpu"] is not None
                                 and prot["wall_over_cpu"] <= 1.1)
    return prot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/S18_protocol")
    a = ap.parse_args()
    log = os.path.join(a.out, "logs")
    read = lambda fn: (open(fn).read() if os.path.exists(fn) else "")   # noqa: E731
    prot = build(read(os.path.join(log, a.tag + ".timev.txt")),
                 read(os.path.join(log, a.tag + ".threads.log")),
                 read(os.path.join(log, a.tag + ".env.txt")))
    # the streamed CCA driver writes s10_<tag>.json, the CPA one
    # kr2cpa_<tag>.json; the drivers with no --tag of their own were given
    # `--out <out>/<tag>` instead and named the file themselves.
    cand = [os.path.join(a.out, p + a.tag + ".json")
            for p in ("s10_", "kr2cpa_", "success_rate_", "")]
    cand += sorted(glob.glob(os.path.join(a.out, a.tag, "*.json")),
                   key=os.path.getmtime, reverse=True)
    fn = next((c for c in cand if os.path.exists(c)),
              os.path.join(a.out, f"s10_{a.tag}.json"))
    if not os.path.exists(fn):
        print(f"merge_protocol: {fn} does not exist (run failed?); "
              f"protocol block written to {fn}.protocol.json")
        json.dump(prot, open(fn + ".protocol.json", "w"), indent=1)
        return
    res = json.load(open(fn))
    prot["procs"] = res.get("procs")          # the driver's own worker count
    res["protocol"] = prot
    json.dump(res, open(fn, "w"), indent=1)
    print(f"merge_protocol: {a.tag}: wall {prot['wall_s']} s, "
          f"cpu {prot['cpu_s']} s, wall/cpu {prot['wall_over_cpu']}, "
          f"max_nlwp {prot['threads']['max_nlwp']} over "
          f"{prot['threads']['samples']} samples, "
          f"peak {prot['maxrss_gb']} GB")


if __name__ == "__main__":
    main()
