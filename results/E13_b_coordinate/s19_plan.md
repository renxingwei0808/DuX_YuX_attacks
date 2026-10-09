# E13 / T1 — the S19 data and cost account

All numbers recomputed here; `rank_max_b = 14400` is the ROW COUNT at which the extended template determines the sixteen key words (step 3).  It is not a rank: the system is already rank-deficient there (12056 of 16000 rows on toy-2^4), and it is rows, not rank, that the data account has to buy.

## 1. The cells

| row | structure | T | D (layer) | margins | rows read | weight margin | weight axes | usable weights | dim K |
|---|---|---|---|---|---|---|---|---|---|
| DuX(2^16) 12/12 | 2 words [3, 7], 2^32.0 | 131070 | [81090, 110771, 151316, 70226] | [49980, 20299, -20246, 60844] | positions [0, 1, 3] | 20299 | 1 | 20299 | 4 |
| DuX(2^8) 8/12, three whole words | 3 words [3, 7, 11], 2^24.0 | 765 | [418, 571, 780, 362] | [347, 194, -15, 403] | positions [0, 1, 3] | 194 | 1 | 194 | 4 |
| DuX(2^8) 8/12, T1 + T3 (three weight axes) | 3 words [3, 7, 11], 2^24.0 | 765 | [418, 571, 780, 362] | [347, 194, -15, 403] | positions [0, 1, 3] | 194 | 3 | 1235780 | 4 |
| DuX(2^8) 8/12, subspace (8,8,7) | 3 words [3, 7, 11], dims [8, 8, 7], 2^23.0 | 637 | [418, 571, 780, 362] | [219, 66, -143, 275] | positions [0, 1, 3] | 66 | 1 | 66 | 4 |
| DuX(2^8) 8/12, subspace (8,8,6) | 3 words [3, 7, 11], dims [8, 8, 6], 2^22.0 | 573 | [418, 571, 780, 362] | [155, 2, -207, 211] | positions [0, 1, 3] | 2 | 1 | 2 | 4 |

## 2. The system

| row | M (paper) | M_b (extended) | N_w | rows/structure | structures | total rows | **total data** | row stack | elimination matrix |
|---|---|---|---|---|---|---|---|---|---|
| DuX(2^16) 12/12 | 18025 | **47749** | 3600 | 14400 | 1 | 14400 | **2^32.0** | 1.28 GB | **1.28 GB** |
| DuX(2^8) 8/12, three whole words | 18025 | **47749** | 194 | 776 | 19 | 14744 | **2^28.25** | 1.31 GB | **1.28 GB** |
| DuX(2^8) 8/12, T1 + T3 (three weight axes) | 18025 | **47749** | 3600 | 14400 | 1 | 14400 | **2^24.0** | 1.28 GB | **1.28 GB** |
| DuX(2^8) 8/12, subspace (8,8,7) | 18025 | **47749** | 66 | 264 | 55 | 14520 | **2^28.78** | 1.29 GB | **1.28 GB** |
| DuX(2^8) 8/12, subspace (8,8,6) | 18025 | **47749** | 2 | 8 | 1800 | 14400 | **2^32.81** | 1.28 GB | **1.28 GB** |

## 3. The cost

The pair-moment count goes from 10 (unordered, `Mom` symmetric) to
10 + 16 = 26, because `Mom2[A][B] = sum_P P_A^e (P_B^f)^2` has no
symmetry.  Scaling Y08 Sect. 4.1's measured characteristic-2
throughput by the gather count:

| row | gathers/structure | weight accumulation | core-hours (quiet machine) | core-hours (loaded) |
|---|---|---|---|---|
| DuX(2^16) 12/12 | 2^47.99 | 2^44.11 | **71.7** | 187.2 |
| DuX(2^8) 8/12, three whole words | 2^44.24 | 2^36.14 | **5.3** | 13.9 |
| DuX(2^8) 8/12, T1 + T3 (three weight axes) | 2^39.99 | 2^36.11 | **0.3** | 0.7 |
| DuX(2^8) 8/12, subspace (8,8,7) | 2^44.77 | 2^36.12 | **7.7** | 20.1 |
| DuX(2^8) 8/12, subspace (8,8,6) | 2^48.8 | 2^36.11 | **126.1** | 329.0 |

> Y08 Sect. 4.1 measured 43.8 core-hours per 2^32 structure on a
> quiet machine and 114.3 on a loaded one for `10 * 63^2 * 2^32`
> gathers; the two columns are that spread carried over.  The R9
> memo's estimate was 250-500 core-hours, from scaling the F_p
> 12-round run (45 core-hours) by 2.6 for the characteristic-2
> kernel and by 2 for the extra moment family.

## 4. The commands

```bash
# DuX(2^16) 12/12: 3600 weights x dim K 4 = 14400 rows from each of 1 structure(s) of 2^32.0 chosen ciphertexts (2^32.0 in total)
python experiments/E13_b_coordinate/run_toy.py \
    --instance dux-2^16 --layers 10 --active 3,7 \
    --combine 1101 --structures 1 --seeds 2026,7,11
```

```bash
# DuX(2^8) 8/12, three whole words: 194 weights x dim K 4 = 776 rows from each of 19 structure(s) of 2^24.0 chosen ciphertexts (2^28.25 in total)
python experiments/E13_b_coordinate/run_toy.py \
    --instance dux-2^8 --layers 6 --active 3,7,11 \
    --combine 1101 --structures 19 --seeds 2026,7,11
```

```bash
# DuX(2^8) 8/12, T1 + T3 (three weight axes): 3600 weights x dim K 4 = 14400 rows from each of 1 structure(s) of 2^24.0 chosen ciphertexts (2^24.0 in total)
python experiments/E13_b_coordinate/run_toy.py \
    --instance dux-2^8 --layers 6 --active 3,7,11 \
    --combine 1101 --structures 1 --weight-dims 3 --seeds 2026,7,11
```

```bash
# DuX(2^8) 8/12, subspace (8,8,7): 66 weights x dim K 4 = 264 rows from each of 55 structure(s) of 2^23.0 chosen ciphertexts (2^28.78 in total)
python experiments/E13_b_coordinate/run_toy.py \
    --instance dux-2^8 --layers 6 --active 3,7,11 \
    --combine 1101 --structures 55 --seeds 2026,7,11
```

```bash
# DuX(2^8) 8/12, subspace (8,8,6): 2 weights x dim K 4 = 8 rows from each of 1800 structure(s) of 2^22.0 chosen ciphertexts (2^32.81 in total)
python experiments/E13_b_coordinate/run_toy.py \
    --instance dux-2^8 --layers 6 --active 3,7,11 \
    --combine 1101 --structures 1800 --seeds 2026,7,11
```

> `run_toy.py` is the in-memory driver; at 2^32 the structure does
> not fit and S19 needs the streamed variant (the same slice
> decomposition `bcoord.weighted_moments_b` implements, moved into
> `attack_12round.py`'s worker layout).  That port is the one piece
> of T1 this round leaves to the server.

