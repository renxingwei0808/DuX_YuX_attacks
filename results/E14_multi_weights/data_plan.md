# E14 / T3 — the S21 data plan (multi-index weights)

`rank_max` is the MEASURED ceiling of the ledger row; a structure
yields `dim K` (or 4) rows per weight, so it needs
`N_w = ceil(rank_max / rows_per_weight)` weights and
`ceil(rank_max / rows_per_structure)` structures.  The only new
quantity is the weight count; whether one structure's multi-index
rows actually reach `rank_max` is what S21 has to measure.

| row | points/structure | rank_max | dims = 1: weights / structures / data | dims = s: weights / structures / **data** | published |
|---|---|---|---|---|---|
| DuX(2^8) 8/12, four full words | 2^32.0 | 4756 | 240 / 5 / 2^34.32 | 141722460 / 1 / **2^32.0** | 5 structures, 2^34.32 |
| DuX(2^8) 8/12, dims (8,8,8,5) | 2^29.0 | 4756 | 16 / 75 / 2^35.23 | 3876 / 1 / **2^29.0** | 75 structures, 2^35.2 |
| Yu2X-8 8/12, full block | 2^32.0 | 3822 | 126 / 11 / 2^35.46 | 798873 / 1 / **2^32.0** | 11 structures, 2^35.46 |
| Yu2X-8 7/12, two words | 2^16.0 | 4940 | 34 / 37 / 2^21.21 | 595 / 3 / **2^17.58** | 40 structures, 2^21.32 |
| DuX(65537) 13 (dagger), full block | 2^64.0 | 8608 | 28 / 308 / 2^72.27 | 3158 / 3 / **2^65.59** | 308 structures, 2^72.3 |

Notes.

* The dims = 1 column is the repository's own accounting, not the
  published one: the published rows were run BEFORE the axis cap of
  W25 and before `plan` existed, and some of them used fewer weights
  than the margin allows (the ledger records what was run).
* The memo's targets were: DuX(2^8) 8/12 one structure 2^32 or
  (8,8,8,5) 2^29; Yu2X-8 8/12 one structure 2^32; Yu2X-8 7/12 three
  structures 2^17.6; DuX(65537) 13(dagger) three structures 2^65.6.
* The 13(dagger) row is THEORY only (prime field, dim K = 1 combined
  row); it moves the ledger's J section, not A.

* **DuX(2^8) 8/12, four full words** — dims = 4: full-domain words: |a| < T - D = 240
* **DuX(2^8) 8/12, dims (8,8,8,5)** — dims = 4: subspace dims [8, 8, 8, 5]: |a| < T - D = sum_i (2^m_i - 1) - D = 16
* **Yu2X-8 8/12, full block** — dims = 4: full block [0] (O11), T3 multi-index: weights on ciphertext words [3, 2, 1, 0] (positions [3, 2, 1, 0], degrees [2, 3, 5, 8] after the substitution) => sum_i a_i deg_i < 252
* **Yu2X-8 7/12, two words** — dims = 2: full-domain words: |a| < T - D = 34
* **DuX(65537) 13 (dagger), full block** — dims = 4: full block [0] (O11), T3 multi-index: weights on ciphertext words [2, 1, 0, 3] (positions [2, 1, 0, 3], degrees [2, 3, 5, 8] after the substitution) => sum_i a_i deg_i < 57
