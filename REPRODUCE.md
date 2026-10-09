# Reproducing the results

This guide lists, for every executed attack of the paper, the exact command of
the reported run, the record it produced, and what a rerun must reproduce.
`results/README.md` maps the remaining tables and statements to their records.

All times below were measured on our server (2 x Xeon Gold 6230R, 52 physical
cores, 251 GB; see `docs/measurement_protocol.md`).  A rerun must reproduce
the rank, the number of determined monomials (`pinned`) and the 16/16
recovered key words exactly; time and memory depend on the machine.

## 1. Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, sympy, pytest
pip install threadpoolctl                # optional: pins BLAS threads inside the attack workers
pip install z3-solver                    # optional: only for the monomial prediction of E05

# C kernels (gcc with OpenMP)
make -C experiments/E04_zero_sum_F2n/fast
make -C experiments/E06_key_recovery_1round/fast
make -C experiments/E07_key_recovery_2round/fast
make -C experiments/E09_unified_criterion/fast
make -C experiments/E10_cpa/fast
```

We used Python 3.10.12, NumPy 2.2.6 and gcc 11.4.0 on Ubuntu 22.04.5.

## 2. Quick checks (minutes on a laptop)

```bash
python -m pytest -q tests                                  # cipher models, regression vectors, tools, small attacks
python experiments/E01_structure/check_structure.py        # O2, O3, O5'
python scripts/check_paper_formulas.py                     # errata of the DuX paper (sympy)
python experiments/Y01_spec_audit/audit.py --out /tmp/y01  # YuX specification audit

# the criterion, e.g. the 12-round DuX(65537) structure: layer 10 of two words
python tools/zero_sum_criterion.py --q 65537 --active 3,7 --layers 11

# the algebra of the 12-round attack on the toy field (seconds)
python experiments/E07_key_recovery_2round/attack_12round.py --instance toy-257 \
    --layers 6 --active 3,7 --weights 150 --combine 0001 --normalise --structures 1 \
    --procs 1 --seed 11 --out /tmp/smoke --tag smoke_toy257

# tables recomputed from the committed records (seconds)
python experiments/S18_protocol/opcounts.py --out /tmp/opcounts      # multiplication counts
python experiments/S18_protocol/run_table.py --out /tmp/run_table.md  # the run table
python experiments/S18_protocol/verify_rerun.py --out /tmp/verify.json
```

## 3. How the reported runs were made

Every reported run went through `scripts/run_pinned.sh`, which pins the run to
reserved cores, fixes every thread pool at one thread, measures it with
`/usr/bin/time -v`, samples the thread count every 10 s, and merges the
measurements into the JSON record (`protocol` block):

```bash
bash scripts/run_pinned.sh <cores> <solve_threads> <tag> <command ...>
```

Single-thread runs used one core and `--procs 1`; multi-process runs used 19
reserved physical cores (`32-43,45-51` on our machine) and `--procs 19`.
The original command line of each run is the `cmd` line of
`results/S18_protocol/logs/<tag>.env.txt`.  In the commands below, `$DATA` is a
scratch directory for checkpoints and row files (a few GB).  Seeds 2026, 7 and
11 draw the random master keys; the key is used only to play the oracle and,
after solving, to check the result.

## 4. Chosen-ciphertext attacks on DuX (Tables I, III and IV of the paper; section 4 of the supplement)

### DuX(65537), 12/12 rounds, 2^29 chosen ciphertexts, single thread

Mixed structure: a coset of order 2^13 on word 3 times the full field on
word 7, two-dimensional weights, combined row `0001`.

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-65537 \
    --layers 10 --active 3,7 --mixed 13,full --weight-dims 2 --weight-a1 0,1,2 \
    --combine 0001 --normalise --structures 1 --slice-chunk 4096 --block-slices 1 \
    --checkpoint-every 4096 --procs 1 \
    --checkpoint-dir $DATA/S18_1t_dux65537_r12_k13_2axis_seed11 \
    --rows-dir $DATA/S18_1t_dux65537_r12_k13_2axis_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_1t_dux65537_r12_k13_2axis_seed11
```

Expected: 10 500 rows, rank 8188, 2592 determined monomials, 16/16.  Ours:
70 753 s (19.7 h) on one core, 13.84 GB.
Record: `results/S18_protocol/s10_S18_1t_dux65537_r12_k13_2axis_seed11.json`;
other keys: `results/R9_server/s10_S20_dux65537_r12_mixed13full_2axis_seed{2026,7}.json`.

### DuX(65537), 12/12 rounds, 2^30, variant (supplement), 19 processes

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-65537 \
    --layers 10 --active 3,7 --mixed 14,full --weights 9000 --combine 0001 --normalise \
    --structures 1 --slice-chunk 4096 --block-slices 1 --checkpoint-every 4096 --procs 19 \
    --checkpoint-dir $DATA/S18_mp_dux65537_r12_k14_seed11 \
    --rows-dir $DATA/S18_mp_dux65537_r12_k14_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_mp_dux65537_r12_k14_seed11
```

Expected: rank 8188, 2592, 16/16.  Ours: 4069 s, 11.5 CPU hours, 19.63 GB.
Record: `results/S18_protocol/s10_S18_mp_dux65537_r12_k14_seed11.json`;
other keys: `results/R9_server/s10_S20_dux65537_r12_mixed14full_seed{2026,7}.json`.

### DuX(2^16), 12/12 rounds, 2^32, 19 processes

Two full words (3, 7), the cheap set {1, 2} (cubic coordinate b), combined
rows `1101`, 3780 weights.

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-2^16 \
    --layers 10 --active 3,7 --cheap-set 1,2 --combine 1101 --weights 3780 \
    --weight-grid 3600 --normalise --structures 1 --slice-chunk 4096 --block-slices 1 \
    --point-chunk 65536 --checkpoint-every 8192 --procs 19 \
    --checkpoint-dir $DATA/S18_mp_dux2p16_r12_2w_cheap12_seed11 \
    --rows-dir $DATA/S18_mp_dux2p16_r12_2w_cheap12_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_mp_dux2p16_r12_2w_cheap12_seed11
```

Expected: 15 120 rows, rank 12 056, 5812 determined monomials, 16/16 (the
threshold is crossed at 14 400 rows, recorded by `--weight-grid 3600`).
Ours: 26 840 s, 133.6 CPU hours, 26.13 GB.
Record: `results/S18_protocol/s10_S18_mp_dux2p16_r12_2w_cheap12_seed11.json`;
other keys: `results/R9_server/s10_S19_dux2p16_r12_2w_cheap12_seed{2026,7}.json`.

### DuX(65537), 11/12 rounds, 2^15, single thread

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-65537 \
    --layers 9 --active 3 --coset 15 --weights 9000 --weight-grid 8188,8608 \
    --combine 0001 --normalise --structures 1 --slice-chunk 4096 --block-slices 512 \
    --procs 1 --seed 2026 --out results/S18_protocol \
    --tag S18_1t_dux65537_r11_coset15_seed2026
```

Expected: rank 8188, 2592, 16/16.  Ours: 5597 s, 20.76 GB.
Record: `results/S18_protocol/s10_S18_1t_dux65537_r11_coset15_seed2026.json`;
three keys: `results/R9_server/s10_S20_dux65537_r11_coset15_0001_seed{2026,7,11}.json`.

### DuX(2^16), 11/12 rounds, 2^15.35, single thread

Point set of 41 735 points on word 3 (O15).

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-2^16 \
    --layers 9 --active 3 --point-set 41735 --weights 1189 --combine none --normalise \
    --structures 1 --slice-chunk 8192 --procs 1 --seed 2026 --out results/S18_protocol \
    --tag S18_1t_dux2p16_r11_pset41735_seed2026
```

Expected: 4756 rows, rank 4756, 2188, 16/16.  Ours: 2039 s, 6.21 GB.
Record: `results/S18_protocol/s10_S18_1t_dux2p16_r11_pset41735_seed2026.json`;
second key: `results/R8_server/s10_R8_dux2p16_r11_pset41735_seed7.json`.

### DuX(65537) and DuX(2^16), 10/12 rounds, 2^16, 50 keys, single thread

```bash
python3 experiments/E06_key_recovery_1round/attack_1round.py --instance dux-65537 \
    --rounds 10 --auto-weight --nweights 8 --keys 50 --structures 1 --seed 2026 \
    --out results/S18_protocol/S18_1t_t4_dux65537_r10_seed2026
python3 experiments/E06_key_recovery_1round/attack_1round.py --instance dux-2^16 \
    --rounds 10 --auto-weight --nweights 8 --keys 50 --structures 1 --seed 2026 \
    --out results/S18_protocol/S18_1t_t4_dux2p16_r10_seed2026
```

Expected: 50/50 keys.  Ours: 215 s and 464 s for all 50 keys.
Records: `results/S18_protocol/S18_1t_t4_dux{65537,2p16}_r10_seed2026/`.

### DuX(2^8), 8/12 rounds, 2^24, single thread

Three full words (3, 7, 11), cheap set {1, 2}, two-dimensional weights.

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-2^8 \
    --layers 6 --active 3,7,11 --cheap-set 1,2 --combine 1101 --normalise --structures 1 \
    --weight-dims 2 --weights-a1-n 20 --slice-chunk 256 --block-slices 1 --slice-parts 4 \
    --point-chunk 16384 --procs 1 \
    --checkpoint-dir $DATA/S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11 \
    --rows-dir $DATA/S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11
```

Expected: 14 760 rows, rank 12 056, 5812, 16/16.  Ours: 32 834 s, 16.33 GB.
Record: `results/S18_protocol/s10_S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11.json`;
other keys: `results/R9_server/s10_S19_dux2p8_r8_3w_cheap12_2axis_seed{2026,7}.json`.

### DuX(2^8), 7/12 rounds, 2^12.70, single thread

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance dux-2^8 \
    --layers 5 --active 3 --combine none --normalise --structures 26 \
    --weight-grid 10,20,30,40,46 --procs 1 --seed 2026 --out results/S18_protocol \
    --tag S18_1t_dux2p8_r7_seed2026
```

Expected: 4784 rows, rank 4756, 2188, 16/16.  Ours: 824 s, 2.06 GB.
Record: `results/S18_protocol/s10_S18_1t_dux2p8_r7_seed2026.json`;
second key: `results/E07_key_recovery_2round/s9/s10_S9_step4_dux2p8_r7_seed7.json`.

## 5. Chosen-ciphertext attacks on YuX (Tables I and IV of the paper; section 4 of the supplement)

### YupX-65537, 11/14 rounds, 2^32, 24 processes

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yupx-65537 \
    --layers 9 --active 0,4 --weights 2400 --combine none --normalise --structures 1 \
    --procs 24 --slice-chunk 4096 --block-slices 1 --checkpoint-every 16384 \
    --checkpoint-dir $DATA/R6D1_yupx_r11_seed11 --progress 1 \
    --seed 11 --out results/R6_third_keys --tag R6D1_yupx_r11_seed11
```

Expected: 9600 rows, rank 8608, 2088, 16/16.  Ours: 16 427 s, 92.9 CPU hours,
19.4 GB.  Record: `results/R6_third_keys/s10_R6D1_yupx_r11_seed11.json`;
other keys: `results/Y06_kr2/y06/s10_Y06-2_yupx_r11_seed{2026,7}.json`.

### Yu2X-16, 11/14 rounds, 2^32, 36 processes

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yu2x-16 \
    --layers 9 --active 0,4 --weights 1300 --combine none --normalise --structures 1 \
    --procs 36 --slice-chunk 1024 --block-slices 1 --checkpoint-every 8192 \
    --checkpoint-dir $DATA/R7B_seed11 --progress 1 --seed 11 --weight-grid 1238,1300 \
    --out results/R7_keys --tag R7B_yu2x16_r11_seed11
```

Expected: 5200 rows, rank 4940, 1606, 16/16 (also at the minimal 1238
weights).  Ours: 14 636 s, 144.4 CPU hours, 5.48 GB.
Record: `results/R7_keys/s10_R7B_yu2x16_r11_seed11.json`;
other keys: `results/Y06_kr2/y08/s10_Y08-B_yu2x16_r11_seed{2026,7}.json`.

### Yu2X-8, 8/12 rounds, 2^32, 19 processes

Full block 0 (O11), three combined rows `1110`, two-dimensional weights with a
joint Vandermonde over both axes.

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yu2x-8 \
    --layers 6 --full-block 0 --weight-dims 2 --vandermonde-axes 2 --max-vectors 1400 \
    --combine 1110 --normalise --structures 1 --slice-chunk 4096 --block-slices 1 \
    --point-chunk 65536 --checkpoint-every 8192 --procs 19 \
    --checkpoint-dir $DATA/S18_mp_yu2x8_r8_fb0_2axis_seed11 \
    --rows-dir $DATA/S18_mp_yu2x8_r8_fb0_2axis_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_mp_yu2x8_r8_fb0_2axis_seed11
```

Expected: 4200 rows, rank 3822, 1296, 16/16.  Ours: 13 108 s, 67.2 CPU
hours, 4.56 GB.  Record: `results/S18_protocol/s10_S18_mp_yu2x8_r8_fb0_2axis_seed11.json`;
other keys: `results/R9_server/s10_S21_yu2x8_r8_fb0_2axis_seed{2026,7}.json`.

### Yu2X-8, 7/12 rounds, 2^17.58, single thread

```bash
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yu2x-8 \
    --layers 5 --active 0,4 --weight-dims 2 --vandermonde-axes 2 --combine none \
    --normalise --structures 4 --structure-grid 2,3 --slice-chunk 4096 --block-slices 256 \
    --point-chunk 65536 --procs 1 \
    --checkpoint-dir $DATA/S18_1t_yu2x8_r7_2w_2axis_seed11 \
    --rows-dir $DATA/S18_1t_yu2x8_r7_2w_2axis_seed11 --progress 1 \
    --seed 11 --out results/S18_protocol --tag S18_1t_yu2x8_r7_2w_2axis_seed11
```

Expected: the structure curve reaches rank 4940, 1606 determined monomials
and 16/16 at 3 structures (2^17.58) and does not move at 4.  Ours: 9947 s,
6.68 GB.  Record: `results/S18_protocol/s10_S18_1t_yu2x8_r7_2w_2axis_seed11.json`;
other keys: `results/R9_server/s10_S21_yu2x8_r7_2w_2axis_seed{2026,7}.json`.

### YupX-65537, 10/14 rounds, single thread (three routes)

```bash
# (i) 2^16, partial zero sum and forward combined equations, three keys in one run
python3 experiments/Y05_key_recovery/partial_lcomb.py --instance yupx-65537 --rounds 10 \
    --active 0 --weights 60 --keys 3 --seed 2026 --out results/S18_protocol \
    --tag S18_1t_yupx_r10_lcomb_seed2026
# (ii) 2^14.79, point set of 28 239 points
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yupx-65537 \
    --layers 8 --active 0 --point-set 28239 --weights 2155 --combine none --normalise \
    --structures 1 --slice-chunk 4096 --block-slices 512 --procs 1 --seed 2026 \
    --out results/S18_protocol --tag S18_1t_yupx_r10_pset28239_seed2026
# (iii) 2^15, one coset of order 2^15
python3 experiments/E07_key_recovery_2round/attack_12round.py --instance yupx-65537 \
    --layers 8 --active 0 --coset 15 --weights 2400 --weight-grid 2155,2400 \
    --combine none --normalise --structures 1 --slice-chunk 4096 --block-slices 512 \
    --procs 1 --seed 2026 --out results/S18_protocol --tag S18_1t_yupx_r10_coset15_seed2026
```

Expected: (i) rank 136, 3 keys; (ii) rank 8608, 2088, 16/16; (iii) rank 8608,
2088, 16/16.  Ours: 115 s, 1706 s and 4988 s.  Records:
`results/S18_protocol/S18_1t_yupx_r10_lcomb_seed2026.json`,
`results/S18_protocol/s10_S18_1t_yupx_r10_{pset28239,coset15}_seed2026.json`;
other keys: `results/R8_server/s10_R8_yupx_r10_pset28239_seed{7,11}.json`,
`results/R6_third_keys/s10_R6D3_yupx_r10_coset15_seed7.json`.

### Yu2X-16, 10/14 rounds, 2^16, single thread

```bash
python3 experiments/Y05_key_recovery/partial_lcomb.py --instance yu2x-16 --rounds 10 \
    --active 0 --weights 60 --keys 3 --seed 2026 --out results/S18_protocol \
    --tag S18_1t_yu2x16_r10_lcomb_seed2026
```

Expected: rank 104, 3 keys.  Ours: 55 s.
Record: `results/S18_protocol/S18_1t_yu2x16_r10_lcomb_seed2026.json`.

## 6. Chosen-plaintext attacks on DuX (Appendix A and supplement)

```bash
# 7 rounds, last-round key recovery
python3 experiments/E10_cpa/attack_cp_lastround.py --instance dux-65537 --rounds 7 --active 1 \
    --keys 2 --seed 2026 --out results/S18_protocol/S18_1t_cpa7_dux65537_seed2026
python3 experiments/E10_cpa/attack_cp_lastround.py --instance dux-2^16 --rounds 7 --active 1 \
    --coords 0,1,2,3 --keys 2 --seed 2026 --out results/S18_protocol/S18_1t_cpa7_dux2p16_seed2026
# 8 rounds from one structure of 2^16 plaintexts with 400 weights
python3 experiments/E10_cpa/kr2_cpa.py --instance dux-65537 --layers 6 --active 1 --combine 1110 \
    --weights 400 --structures 1 --seed 2026 --out results/S18_protocol --tag S18_1t_cpa8_dux65537_seed2026
python3 experiments/E10_cpa/kr2_cpa.py --instance dux-2^16 --layers 6 --active 1 --combine 1110 \
    --weights 400 --structures 1 --seed 2026 --out results/S18_protocol --tag S18_1t_cpa8_dux2p16_seed2026
# 8 rounds without weights: 172 structures with checkpoints; the key is determined
# from 167 structures (2^23.38), the value the paper reports
python3 experiments/E10_cpa/kr2_cpa.py --instance dux-2^16 --layers 6 --active 1 --structures 172 \
    --checkpoints 166,167,168,170 --seed 2026 --out results/S18_protocol/S18_1t_cpa8_dux2p16_nw_seed2026
```

Expected ranks: 44 and 40 (7 rounds), 1234 and 1086 (8 rounds with weights),
1336 (8 rounds without weights), with the master key and a known pair
checked.  Ours: 42 s, 2.8 s, 125 s, 1190 s and 774 s.  Records: the
corresponding files in `results/S18_protocol/`; second keys in
`results/E10_cpa/` and `results/E10_cpa/s9/`.

## 7. Distinguishers, criterion checks and theoretical rows

See `results/README.md`, sections B, B', D and E, for the commands and
records behind Table II of the paper, the tightness grid of the
criterion, the template bounds, the characteristic-2 collapse, the exact
degrees, the minimal-data search and the multiplication counts of the
theoretical rows.
