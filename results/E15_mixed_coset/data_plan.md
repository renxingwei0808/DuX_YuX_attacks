# E15 / T2 — the S20 data plan for mixed structures over F_p

Every number below is recomputed by `tools/zero_sum_criterion.py` and
`experiments/E07_key_recovery_2round/weighted.py::plan`; the cost model
scales the S16 run `results/R8_server/s10_R8optA_dux65537_r12_pset39614_seed2026.json` phase by
phase (slice moments ~ points, weight dgemm ~ N_w x slices, row
assembly ~ N_w, elimination unchanged).

## 1. The rows

| row | structure | T | D (layer) | margin | usable weights | residues collide? | N_w needed | data | slices | est. core-hours |
|---|---|---|---|---|---|---|---|---|---|---|
| DuX(65537) 12/12, k = 14 | 2^14 x full | 81920 | [81090, 110771, 151316, 70226] | 11694 | 11693 (dims 1) | no | 8608 | **2^30.0** | 16384 | 15.1 |
| DuX(65537) 12/12, k = 13 | 2^13 x full | 73728 | [81090, 110771, 151316, 70226] | 3502 | 6130251 (dims 2) | no | 8608 | **2^29.0** | 8192 | 10.7 |
| DuX(65537) 11/12, k = 15 | 2^15 | 32768 | [21728, 29681, 40545, 18817] | 13951 | 13950 (dims 1) | no | 8608 | **2^15.0** | 32768 | 9.1 |
| DuX(65537) 12/12, full x full (current) | full x full | 131072 | [81090, 110771, 151316, 70226] | 60846 | 60846 (dims 1) | no | 8608 | **2^32.0** | 65537 | 41.5 |

The weight rules, verbatim from `plan`:

* **DuX(65537) 12/12, k = 14** — structure words ['2^14', 'full']: |a| < 11694; every coset axis needs a_i >= 1 and 2^k not dividing a_i (O12 Sect. 5.5), the full-field axes are unconstrained (T2)
* **DuX(65537) 12/12, k = 13** — structure words ['2^13', 'full']: |a| < 3502; every coset axis needs a_i >= 1 and 2^k not dividing a_i (O12 Sect. 5.5), the full-field axes are unconstrained (T2)
* **DuX(65537) 11/12, k = 15** — structure words ['2^15']: |a| < 13951 and no axis with 2^k | a_i (O12 Sect. 5.5)
* **DuX(65537) 12/12, full x full (current)** — full-domain words: |a| < T - D = 60846

## 2. The commands

```bash
# DuX(65537) 12/12, k = 14: 8608 weights, 2^30.0 chosen ciphertexts per key
python experiments/E07_key_recovery_2round/attack_12round.py \
    --instance dux-65537 --layers 10 --active 3,7 \
    --mixed 14,full --weights 8608 --combine 0001 \
    --normalise --structures 1 --procs 32 --seed 2026
```

```bash
# DuX(65537) 12/12, k = 13: 8608 weights, 2^29.0 chosen ciphertexts per key
python experiments/E07_key_recovery_2round/attack_12round.py \
    --instance dux-65537 --layers 10 --active 3,7 \
    --mixed 13,full --weights 8608 --weight-dims 2 --combine 0001 \
    --normalise --structures 1 --procs 32 --seed 2026
```

```bash
# DuX(65537) 11/12, k = 15: 8608 weights, 2^15.0 chosen ciphertexts per key
python experiments/E07_key_recovery_2round/attack_12round.py \
    --instance dux-65537 --layers 9 --active 3 \
    --coset 15 --weights 8608 --combine 0001 \
    --normalise --structures 1 --procs 32 --seed 2026
```

> `attack_12round.py --mixed` is the streamed driver; the in-memory
> `attack_2round_fast.py --mixed 14,full` is the same rows for a toy
> size.  Both default to the old behaviour without the switch.

## 3. The 'coset x point set' variant (T = 2^k + (n - 1))

| k | coset points | point-set size n | T | D_3 | margin | data |
|---|---|---|---|---|---|---|
| 14 | 16384 | 62452 | 78835 | 70226 | 8609 | 2^29.93 |
| 15 | 32768 | 46068 | 78835 | 70226 | 8609 | 2^30.49 |
| 16 | 65536 | 13300 | 78835 | 70226 | 8609 | 2^29.7 |

Computed only, not run: it needs the divided-difference mask of the
point-set axis AND the coset algebra on the other, which the criterion
supports (the thresholds add, Theorem 1 being per word) but no driver
combines yet.

