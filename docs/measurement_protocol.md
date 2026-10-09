# Measurement protocol

This is the protocol behind every time, memory and multiplication count in the
paper (Section VII of the paper and section 5 of the supplement).
It fixes how a run is reported; it never changes what an attack computes.  A
rerun under the protocol must reproduce the rank, the number of determined
monomials and the 16/16 key words of the original record exactly;
`experiments/S18_protocol/verify_rerun.py` checks this for every rerun.

## 0. Machine

| Item | Value |
|---|---|
| CPU | 2 x Intel Xeon Gold 6230R (26 cores / 52 threads each): 52 physical / 104 logical cores |
| Clock | 2.10 GHz base, 4.00 GHz maximum turbo |
| Cache | L2 52 MiB, L3 71.5 MiB (35.75 MiB per socket) |
| NUMA | 2 nodes: node 0 = logical 0-25 / 52-77, node 1 = 26-51 / 78-103; logical i and i+52 are the two hyperthreads of one physical core |
| Memory | 251 GB |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12, NumPy 2.2.6 (scipy-openblas64) |
| C kernels | gcc 11.4.0, `-O3 -march=native -funroll-loops -fopenmp` (`experiments/*/fast/Makefile`) |

The machine was shared.  Every reported run used physical cores reserved with
`taskset`, with the sibling hyperthreads left idle.

## 1. Rule 1: single-thread runs

An executed attack that one thread completes within 36 hours is reported from
a run on one reserved physical core:

```bash
taskset -c <core>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
DUX_SOLVE_THREADS=1                      # the C elimination kernel uses one thread
python3 <driver> ... --procs 1
/usr/bin/time -v                         # wall, user, sys, maximum resident set
```

`scripts/run_pinned.sh` does all of this and records the run.  `--procs 1`
really is one thread: `attack_12round.py` then takes a serial path instead of
a one-worker `multiprocessing.Pool` (which would add three helper threads);
`tests/test_procs1.py` checks that the serial and the parallel path give the
same rows, rank, determined monomials and key words.  The other drivers are
single-process anyway, and their only thread pool is the BLAS one, pinned by
the environment variables.

The command of a rerun is the command of the original record, plus
`--procs 1` and the core pinning.

## 2. Rule 2: extrapolation

An attack made of N independent structures or N identical passes that takes
more than 36 hours on one thread may be reported as N x t_1 + t_e, with t_1
the single-thread time of one structure or pass and t_e the single-thread
elimination time, keeping a full multi-process run as the evidence of 16/16.
No row of the paper uses this rule.

## 3. Rule 3: uninterrupted multi-process runs

An attack that takes more than 36 hours on one thread and cannot be split by
structure stays multi-process.  We report the number of processes, the wall
clock, the CPU time (user + sys), the peak memory and the number of
elimination threads.  The run must be uninterrupted: not restarted by a
watchdog and not resumed from a checkpoint (the checkpoint and row directories
are cleared before the start).  Workers are single-threaded; only the
elimination may use several threads, recorded as `protocol.solve_threads`.
The job is pinned to a reserved range of physical cores and runs alone in it.

## 4. Rule 4: multiplication counts

Every executed and every theoretical row gets a machine-independent count of
field multiplications, from one formula:

```
oracle    = N_struct x n_points                     chosen texts (= oracle queries)
assemble  = passes x oracle x W                     moment accumulation per point
          + N_w x W x n_slices                      weighted sums (Vandermonde / dgemm)
          + n_rows x n_cols                         assembly of the combined rows
solve     = n_rows x n_cols x rank                  elimination
total     = assemble + solve
```

`passes` is the number of times the assembly walks the structure: 1 for
one-dimensional weights, one pass per exponent of every extra weight axis
otherwise, and one pass for m axes joined by `--vandermonde-axes m`.  The
chosen texts are not multiplied by the passes: the oracle is queried once.
`W` is the width of the moment layout, built by
`experiments/S18_protocol/opcounts.py` with the same classes the attack uses.
`n_rows`, `n_cols` and `rank` come from the run's own JSON for executed rows.

Output: `results/S18_protocol/opcounts.{md,json}`; test:
`tests/test_s18_protocol.py`, which also pins the published exponents of the
theoretical rows.

**4.2 Calibration.** From single-thread runs on structures of at least 2^15
points: 43.2 us per point for the prime-field decryption direction
(DuX(65537) 12/12 at 2^29, W = 56 550, 3 passes) and 87.0 us per point in
characteristic 2 (DuX(2^8) 8/12 at 2^24, W = 65 400, 20 passes).  The
elimination costs 0.23-0.35 ns per multiplication over F_p and 0.40-0.48 ns
in characteristic 2.

## 5. Checks of every run

| JSON field | Meaning | Acceptance |
|---|---|---|
| `protocol.threads.max_nlwp` | largest thread count of any process of the run, sampled every 10 s | = 1 for single-thread rows |
| `protocol.wall_over_cpu` | wall / (user + sys) | <= 1.1 |

The sampler counts only the run's own process tree (its tagged processes and
their children, which catches the `gf2nsolve` elimination child).  A run with
wall / (user + sys) > 1.1 was preempted and is rerun, never corrected.  Each
run also records `loadavg_start` and `loadavg_end`.

**5.1 Turbo frequency.** A reserved core does not run at the same speed every
time: the all-core turbo depends on how many cores are busy, which lowers wall
and CPU time together and is therefore invisible to the ratio above.  The
reported single-thread rows were all measured while we used 3-8 cores and the
machine load was 80-105; comparisons across machines use the multiplication
counts of Rule 4, not wall clock.

## 6. Memory

`Maximum resident set size` of `/usr/bin/time -v`, in GB
(`protocol.maxrss_gb`), the peak of the whole process tree.  A single-thread
run can show a higher peak than a multi-process run of the same row, because
the shared slice tables are then counted in one process (example: DuX(2^16)
11/12 at 2^15.35, 3.8 GB multi-process, 6.21 GB single-thread).  The paper
reports the single-thread figure.

## 7. Keys

* Every cell is run on at least 2 random master keys; the headline rows on 3
  (seeds 2026, 7 and 11).
* Success means `inner_correct = 16/16`: the 16 words of rk^0, i.e. the
  master key; chosen-plaintext rows also check `master_key_ok` and
  `known_pair_ok`.
* Across keys, the rank, the determined monomials and the structure or weight
  curves must agree exactly.
* The success-rate rows (9 and 10 rounds) are runs over 50 keys.

## 8. Which run a cell reports

1. A single-thread run under Rule 1, if there is one.
2. Otherwise an uninterrupted multi-process run under Rule 3.
3. Otherwise no time is reported, only the multiplication count.

All other runs (other keys, other process counts) are kept as evidence in
`results/` and listed in `results/README.md`.
