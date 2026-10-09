#!/bin/bash
# Run ONE attack under the measurement protocol (docs/measurement_protocol.md,
# rules 1 and 3): pinned to a fixed set of cores, with every thread pool pinned
# to one thread, and record everything the protocol asks for:
#
#   * `taskset -c <cores>`          -- the run never leaves the reserved cores
#   * OPENBLAS/OMP/MKL = 1          -- no BLAS or OpenMP pool inside the workers
#   * DUX_SOLVE_THREADS = <n>       -- the only place where >1 thread is allowed
#                                      (rule 1: n = 1 for single-thread runs)
#   * /usr/bin/time -v              -- wall, user, sys, maximum resident set
#   * a thread sample every 10 s    -- `nlwp` of every process of THIS run's
#                                      own process tree, so the JSON can show
#                                      that the run really was single-threaded
#
# The sample matches on the run's own --tag (the pool workers are forks, so
# their /proc/<pid>/cmdline is the parent's and the tag catches all of them)
# and then adds every process whose PARENT is one of those -- which is how the
# `gf2nsolve` child, whose own argv carries no tag, gets counted.  Matching
# `gf2nsolve` by name instead would be wrong on a shared machine: it would
# catch the elimination of any other job.
#
#   bash scripts/run_pinned.sh <cores> <solve_threads> <tag> <command...>
#
# The command must write its JSON into <OUT> (as s10_<tag>.json, kr2cpa_<tag>.json
# or <tag>.json) or into the directory <OUT>/<tag>/; OUT defaults to
# results/S18_protocol.  Afterwards
# experiments/S18_protocol/merge_protocol.py folds the time, memory and thread
# samples into that JSON under the key `protocol`.  Example (DuX(2^16), 11
# rounds, single thread on core 30):
#
#   bash scripts/run_pinned.sh 30 1 S18_1t_dux2p16_r11_pset41735_seed2026 \
#       python3 experiments/E07_key_recovery_2round/attack_12round.py \
#       --instance dux-2^16 --layers 9 --active 3 --point-set 41735 \
#       --weights 1189 --combine none --normalise --structures 1 \
#       --slice-chunk 8192 --procs 1 --seed 2026 \
#       --out results/S18_protocol --tag S18_1t_dux2p16_r11_pset41735_seed2026
set -u
CORES=$1; STH=$2; TAG=$3; shift 3
cd "$(dirname "$0")/.." || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 OPENBLAS_MAIN_FREE=1
export DUX_SOLVE_THREADS="$STH"
OUT=${OUT:-results/S18_protocol}; LOG=$OUT/logs; mkdir -p "$LOG" "$OUT"
TV="$LOG/$TAG.timev.txt"; TH="$LOG/$TAG.threads.log"; RL="$LOG/$TAG.log"
: > "$TH"
{ echo "tag $TAG"; echo "cores $CORES"; echo "solve_threads $STH";
  echo "start $(date -Is)"; echo "loadavg_start $(cut -d' ' -f1-3 /proc/loadavg)";
  echo "cmd $*"; } > "$LOG/$TAG.env.txt"

taskset -c "$CORES" /usr/bin/time -v -o "$TV" "$@" > "$RL" 2>&1 &
PID=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    ps -eo pid=,ppid=,nlwp=,args= 2>/dev/null \
      | awk -v t="$TAG" -v ts="$(date +%s)" '
          { par[$1] = $2; nlwp[$1] = $3 }
          index($0, t) && !/run_pinned\.sh|awk -v t=/ { ours[$1] = 1 }
          END { for (p in nlwp) if (ours[p] || ours[par[p]]) {
                    n++; s += nlwp[p]; if (nlwp[p] > m) m = nlwp[p] }
                printf "%d procs %d max_nlwp %d sum_nlwp %d\n", ts, n+0, m+0, s+0 }' \
      >> "$TH"
    sleep 10
  done
) &
SAMP=$!
wait "$PID"; RC=$?
kill "$SAMP" 2>/dev/null
{ echo "end $(date -Is)"; echo "loadavg_end $(cut -d' ' -f1-3 /proc/loadavg)";
  echo "exit $RC"; } >> "$LOG/$TAG.env.txt"
python3 experiments/S18_protocol/merge_protocol.py --tag "$TAG" --out "$OUT" \
    >> "$LOG/$TAG.env.txt" 2>&1
exit "$RC"
