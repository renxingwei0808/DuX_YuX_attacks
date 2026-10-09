# E14 / C1 — weighted moments on the reduced configuration of [LS26 v2, Section 7]

**One sentence (paper-ready).**  *One subspace of 2^24 chosen ciphertexts with the
weights a = 0, ..., 40 that the margin at W1 admits attains the ranks 8, 4 and 7 of
the three stages of [LS26 v2, Theorem 3] on three random master keys, against the
10, 4 and 4 subspaces used in [LS26 v2, Section 7].*

## The configuration

* DuX(2^8), six-layer distinguisher, seven rounds attacked, s = 3, active words [0, 2, 3].
* Layer 6 class degree bounds **[836, 1142, 1560, 724]**, T = 765,
  margins [-71, -377, -795, 41], pattern `0001`.
  [LS26 v2] Table C3 gives **[836, 1142, 1560, 724]** at W1, guaranteed C3,
  margin 765 - 724 = 41 -- `tools/zero_sum_criterion.py` reproduces that
  vector exactly (100 of the 560 three-word active sets give it).
* Mask spaces from `tools/cheap_rows.py` (their V_C is our O10 left kernel for r_KR = 1):
  {'V_{2}': 1, 'V_{1,2}': 2, 'V_{0,1,2}': 3, 'V_{0,1,2,3}': 4}; [LS26 v2] Sect. 5.1 states 1, 2, 3, 4.

## The measurement

| key seed | stage | masks | unknowns (effective columns) | rank at a = 0 only | weights to reach full rank | rank with all 41 weights | correct |
|---|---|---|---|---|---|---|---|
| 2026 | (i)  a_{4k}, a_{4k+3} | 1 | 8 (8) | 1 | **8** | 8 | True |
| 2026 | (ii) a_{4k+2} | 1 | 4 (4) | 1 | **4** | 4 | True |
| 2026 | (iii) a_{4k+1}, a^2_{4k+1} | 4 | 8 (7) | 2 | **4** | 7 | True |
| 7 | (i)  a_{4k}, a_{4k+3} | 1 | 8 (8) | 1 | **8** | 8 | True |
| 7 | (ii) a_{4k+2} | 1 | 4 (4) | 1 | **4** | 4 | True |
| 7 | (iii) a_{4k+1}, a^2_{4k+1} | 4 | 8 (7) | 2 | **4** | 7 | True |
| 11 | (i)  a_{4k}, a_{4k+3} | 1 | 8 (8) | 1 | **8** | 8 | True |
| 11 | (ii) a_{4k+2} | 1 | 4 (4) | 1 | **4** | 4 | True |
| 11 | (iii) a_{4k+1}, a^2_{4k+1} | 4 | 8 (7) | 2 | **4** | 7 | True |

All three keys: **16/16 key words from ONE subspace of 2^24.0 chosen ciphertexts** (`all_keys_16_of_16 = True`).

## What it says

* **Without weights** a subspace contributes exactly 1, 1 and 2 rows of rank to the
  three stages -- which is [LS26 v2] Proposition 4 / Table C4's `rho_j = 1, 1, 2`,
  reproduced here as a measurement.  That is why they need ceil(8/1), ceil(4/1) and
  ceil(7/2) = 8, 4, 4 subspaces (they used 10, 4, 4).
* **With weights** every admissible weight contributes the SAME rho_j rows of rank:
  stage (i) reaches 8 at 8 weights, stage (ii) 4 at 4, stage (iii) 7 at 4.  The margin
  at W1 is 41, so one subspace has five times what the attack needs.
* The data therefore drops from 10 * 2^24 = 2^27.32 to **2^24**, and the time with it:
  one decryption pass of the subspace (about 20 s here) instead of ten.

## Reproducing

```bash
python experiments/E14_multi_weights/c1_ls_reduced.py \
    --instance dux-2^8 --rounds 7 --active 0,2,3 \
    --keys 2026,7,11 --weights 41 \
    --out results/E14_multi_weights/C1_ls_reduced.json
```

