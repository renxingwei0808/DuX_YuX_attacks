# Results index

Every number of the paper and its supplement can be traced to a record in this
directory.  This page maps the tables and statements to the records; the
commands of the executed attacks are in `../REPRODUCE.md`.

## Reading a record

The key recovery drivers write one JSON per run.  The fields that matter:

| Field | Meaning |
|---|---|
| `instance`, `rounds`, `layers`, `active_words`, `seed` | the attack and the random master key (seeds 2026, 7, 11) |
| `data_log2` | chosen ciphertexts (plaintexts) per key, log2 |
| `structures`, `points_per_structure`, `weights` | the data and the weights actually used (`weights.n`, `weights.range`) |
| `equation_rows`, `unknowns`, `rank` | size and rank of the assembled linear system |
| `pinned` | number of linearisation monomials the system determines |
| `inner_correct` / `inner_total` | recovered words of rk^0, i.e. of the master key (16/16 = success) |
| `weight_curve`, `structure_curve` | rank and recovered words as a function of the number of weights or structures |
| `assemble_s`, `solve_s` | assembly and elimination time |
| `protocol` | the measurement (`docs/measurement_protocol.md`): `wall_s`, `cpu_s`, `maxrss_gb`, `wall_over_cpu`, `threads.max_nlwp`, `procs`, `cores`, load average |

The success-rate runs (`success_rate_*.json`) record `keys`, `success`,
`success_rate` and the structures used per key.  The distinguisher runs record
`per_layer[i].pattern` (measured), the prediction of the criterion and the
margins T - D.  The file `S18_protocol/logs/<tag>.env.txt` holds the exact
command line (`cmd`), the cores and the load average of each reported run,
and `<tag>.timev.txt` the output of `/usr/bin/time -v`.

## A. Key recovery on DuX, chosen ciphertexts (Tables I, III and IV of the paper; section 4 of the supplement)

| Attack | Data | Reported run | Further keys | Variants (same rank and key words) |
|---|---|---|---|---|
| DuX(65537) 10/12 | 2^16 | `S18_protocol/S18_1t_t4_dux65537_r10_seed2026/` (50/50 keys) | `E14_multi_weights/t4_raw/`, `t4_10round.{md,json}` | 2^17, two structures: `E06_key_recovery_1round/w6/success_rate_dux-65537_r10.json` |
| DuX(65537) 11/12 | 2^15 | `S18_protocol/s10_S18_1t_dux65537_r11_coset15_seed2026.json` | `R9_server/s10_S20_dux65537_r11_coset15_0001_seed{2026,7,11}.json` | 2^15.38: `R8_server/s10_R8_dux65537_r11_pset42698_seed*.json`; 2^16: `E07_key_recovery_2round/s9/s10_S9_step2_*.json`, `R6_third_keys/s10_R6D1_dux65537_r11_seed11.json` |
| DuX(65537) 12/12 | 2^29 | `S18_protocol/s10_S18_1t_dux65537_r12_k13_2axis_seed11.json` | `R9_server/s10_S20_dux65537_r12_mixed13full_2axis_seed{2026,7}.json` | 2^30: `S18_protocol/s10_S18_mp_dux65537_r12_k14_seed11.json`, `R9_server/s10_S20_dux65537_r12_mixed14full_seed*.json`; 2^30.55: `R8_server/s10_R8optA_dux65537_r12_pset39614_seed*.json`; 2^32: `E07_key_recovery_2round/s10/s10_S10_dux65537_r12_seed*.json`, `R6_third_keys/s10_R6D1_dux65537_r12_seed11.json` |
| DuX(2^16) 10/12 | 2^16 | `S18_protocol/S18_1t_t4_dux2p16_r10_seed2026/` (50/50 keys) | `E14_multi_weights/t4_raw/` | 2^17: `E06_key_recovery_1round/w6/success_rate_dux-2^16_r10.json` |
| DuX(2^16) 11/12 | 2^15.35 | `S18_protocol/s10_S18_1t_dux2p16_r11_pset41735_seed2026.json` | `R8_server/s10_R8_dux2p16_r11_pset41735_seed{2026,7}.json` | 2^15.38: `R8_server/s10_R8_dux2p16_r11_pset42698_seed*.json`; 2^16: `E07_key_recovery_2round/s9/s10_S9_step3_*.json`, `R6_third_keys/s10_R6D1_dux2p16_r11_seed11.json` |
| DuX(2^16) 12/12 | 2^32 | `S18_protocol/s10_S18_mp_dux2p16_r12_2w_cheap12_seed11.json` | `R9_server/s10_S19_dux2p16_r12_2w_cheap12_seed{2026,7}.json` | row-subset control: `R9_server/subsets_S19_*.json` |
| DuX(2^8) 7/12 | 2^12.70 | `S18_protocol/s10_S18_1t_dux2p8_r7_seed2026.json` | `E07_key_recovery_2round/s9/s10_S9_step4_dux2p8_r7_seed{2026,7}.json` | -- |
| DuX(2^8) 8/12 | 2^24 | `S18_protocol/s10_S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11.json` | `R9_server/s10_S19_dux2p8_r8_3w_cheap12_2axis_seed{2026,7}.json` | 2^28.09 (17 structures): `R9_server/s10_S19_dux2p8_r8_3w_cheap12_seed*.json`; 2^29 (one subspace, four weight axes): `R9_server/s10_S21_dux2p8_r8_sub8885_4axis_seed*.json`; 2^34.32: `R6_third_keys/s10_R6D4_dux2p8_r8_seed2026.json`, `R7_keys/s10_R7D4_dux2p8_r8_seed7.json` |
| DuX(65537) 13, theoretical | 2^65.59 | -- | -- | criterion: `tools/zero_sum_criterion.py --q 65537 --full-block 0 --layers 11`; weights: `tests/test_usable_weights.py`, `tests/test_multi_index_weights.py` |
| DuX(2^8) 9 rounds | unreachable | -- | -- | `E09_unified_criterion/min_data_table_v2.md` (DuX(2^8), layer 7), `E09_unified_criterion/dux2_8_layer7_search.txt` |

## A'. Key recovery on YuX, chosen ciphertexts (Tables I and IV of the paper; section 4 of the supplement)

| Attack | Data | Reported run | Further keys | Variants |
|---|---|---|---|---|
| YupX-65537 10/14 (i) | 2^16 | `S18_protocol/S18_1t_yupx_r10_lcomb_seed2026.json` (3 keys) | `Y05_key_recovery/Y05-A4_yupx_r10.json` | -- |
| YupX-65537 10/14 (ii) | 2^14.79 | `S18_protocol/s10_S18_1t_yupx_r10_pset28239_seed2026.json` | `R8_server/s10_R8_yupx_r10_pset28239_seed{2026,7,11}.json` | 2^16: `Y06_kr2/y06/s10_Y06-1_yupx_r10_seed*.json`, `R6_third_keys/s10_R6D1_yupx_r10_seed11.json` |
| YupX-65537 10/14 (iii) | 2^15 | `S18_protocol/s10_S18_1t_yupx_r10_coset15_seed2026.json` | `R6_third_keys/s10_R6D3_yupx_r10_coset15_seed{2026,7}.json` | -- |
| YupX-65537 11/14 | 2^32 | `R6_third_keys/s10_R6D1_yupx_r11_seed11.json` (24 processes) | `Y06_kr2/y06/s10_Y06-2_yupx_r11_seed{2026,7}.json` | -- |
| YupX-65537 12/14, theoretical | 2^64 | -- | -- | template bound of the `1110` rows over F_65537: `Y06_kr2/collapse_yupx-65537_comb1110{,_seed7}.json` (ceiling 8468, 16/16 key words); `Y06_kr2/rank_phi_yupx-65537_*.json` |
| Yu2X-16 10/14 | 2^16 | `S18_protocol/S18_1t_yu2x16_r10_lcomb_seed2026.json` (3 keys) | `Y05_key_recovery/Y05-A4_yu2x16_r10.json` | two extension rounds: `Y06_kr2/y06/s10_Y06-1_yu2x16_r10_seed*.json`, `R6_third_keys/s10_R6D1_yu2x16_r10_seed11.json` |
| Yu2X-16 11/14 | 2^32 | `R7_keys/s10_R7B_yu2x16_r11_seed11.json` (36 processes) | `Y06_kr2/y08/s10_Y08-B_yu2x16_r11_seed{2026,7}.json` | -- |
| Yu2X-16 12/14, theoretical | 2^64 | -- | -- | `Y06_kr2/rank_phi_yu2x-16_*.json` |
| Yu2X-8 7/12 | 2^17.58 | `S18_protocol/s10_S18_1t_yu2x8_r7_2w_2axis_seed11.json` | `R9_server/s10_S21_yu2x8_r7_2w_2axis_seed{2026,7}.json` | 2^21.32 (40 structures): `Y06_kr2/y06/s10_Y06-4_yu2x8_r7_seed*.json` |
| Yu2X-8 8/12 | 2^32 | `S18_protocol/s10_S18_mp_yu2x8_r8_fb0_2axis_seed11.json` | `R9_server/s10_S21_yu2x8_r8_fb0_2axis_seed{2026,7}.json` | 2^35.46 (11 structures): `Y06_kr2/y08/s10_Y08-C_yu2x8_r8_seed*.json`; template bound `Y06_kr2/rank_phi_yu2x-8_comb1110.json` |

Further YuX key recoveries that the paper does not tabulate:
`Y05_key_recovery/success_rate_{yupx-65537_r9,yu2x-16_r9,yu2x-8_r5}.json`
(50/50 keys each, one structure of 2^16 resp. 2^8 per key) and the toy
pipelines of `Y05_key_recovery/`.

## B, B'. Zero-sum distinguishers (Table II of the paper; section 2 of the supplement)

| Instance and structure | Records |
|---|---|
| DuX(65537), one word, 9 layers | `E03_zero_sum_Fp/full_field_k5/results.json`, `E03_zero_sum_Fp/prelim_full_field/results.json`; exact tightness: `E02_degree_bounds/symbolic_p65537*.json` |
| DuX(65537), cosets of order 2^k | `E03_zero_sum_Fp/subgroup_k*/results.json` |
| DuX(2^16), one word | `E04_zero_sum_F2n/prelim_2-16/active_*_dim16.json` |
| DuX two words (3,7), 2^32, 9 layers + `1101` | `E03_zero_sum_Fp/multiword_full/p65537_dec_a3_7_full.json`, `E04_zero_sum_F2n/large/b2_p37_d16.json` |
| DuX(65537), signed coset differences | `E03_zero_sum_Fp/multiword_full/p65537_dec_a3_7_coset15.json` |
| DuX(65537), coset 2^13 x full field (weighted, `0001`) | the 12-round runs of section A; criterion: `tools/zero_sum_criterion.py --coset 13,full`; toy check of the mixed criterion: `E15_mixed_coset/zero_sum_toy-193_c6.json` |
| DuX(2^8), one to four words | `E04_zero_sum_F2n/prelim_2-8/`, `E04_zero_sum_F2n/large/a*_d8.json` |
| DuX, four words or full block, 2^64 | criterion only (Theorem "unified zero-sum criterion"): `tools/zero_sum_criterion.py --full-block 0 --layers 11` |
| YuX, one word, 8 layers + `1100` | `Y03_zero_sum/Y03-5_*.json`, `Y04_zero_sum/Y04-1_*.json`, `Y04_zero_sum/Y04-2_*.json` |
| YuX, two words, 9 layers | `Y04_zero_sum/Y04-3_*.json` |
| Yu2X-8, full block 2^32, layer 6 `1110` | `Y04_zero_sum/Y04-4_yu2x8_fullblock.json` |
| Yu2X-8, one, two and four words | `Y04_zero_sum/Y04-5*.json`, `Y03_zero_sum/Y03-4_*.json` |
| YupX-65537, single cosets with weights | `Y03_zero_sum/Y03-6_yupx_coset*.json` |
| YuX full blocks at 2^64 (margins of the table) | `Y02_degree_bounds/o7_yux_tables.{md,json}` |

## C, C'. Chosen plaintexts (Appendix A of the paper; section 7 of the supplement)

| Result | Records |
|---|---|
| Distinguishers, encryption direction | `E10_cpa/cp_zerosum_*.json`, `E10_cpa/predictions_enc.json`, `E09_unified_criterion/grid/*_enc_*.json`, `Y03_zero_sum/Y03-7_*.json`, `Y04_zero_sum/Y04-7_*.json` |
| The six-round integral examples of the DuX paper | `E10_cpa/cp_zerosum_dux-2^16_*.json`, `E10_cpa/cp_zerosum_dux-65537_*.json` |
| Theorem on the next-to-leading coefficient (O9) | `E10_cpa/w13/top_coeffs_*.json`, `E10_cpa/thmS1_base_*.json`, `E10_cpa/logs/S22_thmS1_base.log` |
| DuX 7/12, 2^18.8 and 2^18.6 | reported: `S18_protocol/S18_1t_cpa7_dux65537_seed2026/`, `S18_protocol/S18_1t_cpa7_dux2p16_seed2026/`; also `E10_cpa/cp_attack_dux-*_r7_1.json` |
| DuX 8/12, 2^16 with 400 weights | reported: `S18_protocol/kr2cpa_S18_1t_cpa8_dux{65537,2p16}_seed2026.json`; second key: `E10_cpa/s9/kr2cpa_S9_cpa8_*_seed{2026,7}.json` |
| DuX(2^16) 8/12 without weights, 2^23.43 | reported: `S18_protocol/S18_1t_cpa8_dux2p16_nw_seed2026/`; further keys in `E10_cpa/w14/` |
| DuX(2^16) 8/12, one extension round, 2^35.46 | `E10_cpa/s8/cp_attack_dux-2^16_r8_1_5_c2*.json` |
| Deepest layers in the encryption direction | `E09_unified_criterion/min_data_table_v2.md`, `E09_unified_criterion/min_data_table_v3_enc.md` |

## D. Criterion, techniques and controls (Sections III, IV, VII; supplement)

| Statement | Records and scripts |
|---|---|
| The criterion reproduces 39 grid cells, 21 toy cells and 17 YuX cells | `E09_unified_criterion/grid/`, `E03_zero_sum_Fp/toy*_full/`, `E04_zero_sum_F2n/large/toy4_*.json`, `Y03_zero_sum/`, `Y04_zero_sum/`; `tests/test_zero_sum_criterion.py`, `tests/test_cipher_degree.py` |
| The four-word Yu2X-8 cell at a Frobenius boundary is balanced at two more positions | `Y04_zero_sum/Y04-5a_yu2x8_4words.json` |
| Tightness of the max-plus bound | `E02_degree_bounds/spectrum_*.json`, `Y04_zero_sum/tightness_*.json` |
| Exact degrees in characteristic 2 (O13) | `Y07_char2_exact_degree/` |
| Class degrees and the comparison with Ni et al. | `Y02_degree_bounds/ni_tables.json`, `ni_vs_maxplus.md`, `o7_yux_tables.md`; `E05_monomial_prediction/` |
| Key dependence and the rank formula, one extension round (Proposition "key dependence") | `E06_key_recovery_1round/determinacy.json`, `Y05_key_recovery/determinacy_yux.json` |
| Template bounds (Proposition "template bound") | `E07_key_recovery_2round/s9/rank_phi_*.json`, `E07_key_recovery_2round/s10/rank_phi_dux-2^16_comb0001.json`, `Y05_key_recovery/rank_phi_*.json`, `Y06_kr2/rank_phi_*.json`, `E13_b_coordinate/rank_*_1101_*.json`, `R9_server/rank_dux-2^16_1101_*_S19.json` |
| Kernel dimensions of the cheap rows (supplement table) | `E11_cheap_rows/cheap_rows_table.{md,json}`, `E13_b_coordinate/kernel_cheap12*.json`, `E13_b_coordinate/kernel_cheap23*.json` |
| Characteristic-2 collapse and the prime-field case | `E11_cheap_rows/w19/collapse_*.json`, `Y06_kr2/collapse_yupx-65537_comb1110*.json` |
| Cubic coordinate (O16): toy end to end | `E13_b_coordinate/toy_toy-2^4_1101_cheap12.json`, `E13_b_coordinate/weight_subset_*.json` |
| Mixed structures (O17): toy | `E15_mixed_coset/fast_toy-193_*.json`, `E15_mixed_coset/zero_sum_toy-193_c6.json`, `E15_mixed_coset/mixed_grid.txt`, `E15_mixed_coset/data_plan.md` |
| Point sets (O15): toys and data plan | `E12_interp_mask/` |
| Multi-index weights (O18): rank curves and data plan | `E14_multi_weights/rank_curves_toy-2^4.json`, `E14_multi_weights/data_plan.md` |
| Eight-round toy analog over F_257 of the 12-round attack (rank 8188, 2592 monomials) | `E07_key_recovery_2round/w16/fast_T2norm_toy257_r8_1d.json` |
| A weight beyond the margin gives 16 "determined" but wrong key words | `Y05_key_recovery/Y05-A4_yuxtoy2p4_r4_overshoot.json` |
| Boundary cells: key-independent constants (O14) | `Y09_topform_constants/y09_all.json`, `R8_server/R8_O14_yuxtoy257_fb0_l6_3keys.json`, `R8_server/direct_p257_l6_server.txt` |
| Minimal data inside the framework (section 8 of the supplement) | `E09_unified_criterion/min_data_table_v2.{md,json}`, `min_data_table_v3.{md,json}`, `min_data_table_v3_enc.{md,json}`, `E13_b_coordinate/min_data_x1x1.md` |
| Known-key zero-sum partitions (eprint version only, not in the TC paper) | `E09_unified_criterion/partitions/` |
| Weighted moments on the reduced configuration of Liu and Sun (C1) | `E14_multi_weights/C1_ls_reduced.{md,json}` |
| Boolean degree of the two-round extension: 7 | `E08_boolean_degree_extension/boolean_degrees.json` |
| Three-round Boolean degree of Yu2X-16 and the hybrid route | `Y10_boolean_hybrid/d3.json`, `Y10_boolean_hybrid/rank_toy.json` |
| Specification checks and errata | `W1_spec_audit/paper_formula_check.txt`, `Y01_spec_audit/audit_table.json`, `../docs/dux_specification.md`, `../docs/yux_specification.md` |

## E. Measurement and multiplication counts (Section VII; supplement)

| What | Records |
|---|---|
| Run table of every reported run | `S18_protocol/run_table.md` (from `experiments/S18_protocol/run_table.py`) |
| Reruns reproduce the earlier records | `S18_protocol/verify.json` (from `experiments/S18_protocol/verify_rerun.py`) |
| Multiplication counts of all rows, calibration | `S18_protocol/opcounts.{md,json}` (from `experiments/S18_protocol/opcounts.py`) |
| Command, cores, load and `time -v` of each run | `S18_protocol/logs/*.env.txt`, `S18_protocol/logs/*.timev.txt` |
| Per-point assembly cost | `E07_key_recovery_2round/s4/throughput_3word.json` |

## Directory map

| Directory | Contents |
|---|---|
| `E01_structure` ... `E15_mixed_coset` | records of the DuX experiment series (scripts in `experiments/` under the same names) |
| `Y01_spec_audit` ... `Y10_boolean_hybrid` | records of the YuX experiment series |
| `Y06_kr2` | two-round key recovery on YuX (`y06/`, `y08/`) and its template bounds |
| `S18_protocol` | the reported run of every executed attack, under the measurement protocol, with logs, run table and multiplication counts |
| `R9_server` | full runs of techniques T1-T3 on the real instances (three keys per row with `S18_protocol`) |
| `R8_server` | point-set runs (O15) and the p = 257 boundary-cell check (O14) |
| `R6_third_keys`, `R7_keys` | further keys of the headline rows (seed 11 and others) |
| `W1_spec_audit` | output of `scripts/check_paper_formulas.py` |

Subdirectory names such as `s9/`, `s10/`, `w6/`, `w16/` are the batch labels
of `docs/glossary.md`.
