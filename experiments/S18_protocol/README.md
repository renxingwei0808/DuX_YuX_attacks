# S18: measurement protocol, run table and multiplication counts

Scripts behind the times, memory figures and multiplication counts of the
paper (`docs/measurement_protocol.md`).

| File | Role |
|---|---|
| `../../scripts/run_pinned.sh` | runs one attack pinned to reserved cores with all thread pools at one thread, under `/usr/bin/time -v`, sampling the thread count every 10 s |
| `merge_protocol.py` | folds the time, memory and thread samples into the run's JSON (`protocol` block: `wall_s`, `cpu_s`, `maxrss_gb`, `wall_over_cpu`, `threads.max_nlwp`, ...) |
| `verify_rerun.py` | checks that every protocol rerun reproduces the rank, determined monomials and 16/16 of the record it replaces |
| `run_table.py` | builds the run table from the JSONs (`results/S18_protocol/run_table.md`) |
| `opcounts.py` | the machine-independent multiplication counts of every executed and theoretical row (`results/S18_protocol/opcounts.{md,json}`) |

```bash
python experiments/S18_protocol/opcounts.py --out results/S18_protocol
python experiments/S18_protocol/run_table.py --out results/S18_protocol/run_table.md
python experiments/S18_protocol/verify_rerun.py --out results/S18_protocol/verify.json
```

These three run in seconds on the committed records.  Tests:
`tests/test_s18_protocol.py`, `tests/test_procs1.py`.
