# S18 -- operation counts (docs/measurement_protocol.md section 4)

`oracle = N_struct * n_points`; `assemble = oracle * W + N_w * W * slices + rows * cols`; `solve = rows * cols * rank`; the total is their sum, in field multiplications.  `W` is the moment-layout width, built by `experiments/S18_protocol/opcounts.py` from the same `Precomp` / `MomentLayout` the attack uses.

| row | status | data (log2) | W | rows x cols x rank | assemble (log2) | solve (log2) | **total (log2)** |
|---|---|---|---|---|---|---|---|
| DuX(65537) 12/12 @2^29 | executed | 29.0 | 56550 | 10500 x 38040 x 8188 | 46.45 | 41.57 | **46.5** |
| DuX(65537) 12/12 @2^30 (variant) | executed | 30.0 | 56550 | 9000 x 38040 x 8188 | 45.97 | 41.35 | **46.03** |
| DuX(65537) 12/12 @2^30.55 (variant) | executed | 30.55 | 56550 | 9000 x 38040 x 8188 | 46.63 | 41.35 | **46.67** |
| DuX(65537) 12/12 @2^32 (eprint v1) | executed | 32.0 | 56550 | 9000 x 38040 x 8188 | 47.97 | 41.35 | **47.99** |
| DuX(2^16) 12/12 @2^32 | executed | 32.0 | 65400 | 15120 x 47738 x 12056 | 48.08 | 42.98 | **48.12** |
| DuX(2^16) 12/12 @2^47 (eprint v1 theory) | theoretical | 47.0 | 25200 | 50084 x 18014 x 4756 | 61.87 | 41.96 | **61.87** |
| DuX(65537) 11/12 @2^15 | executed | 15.0 | 56550 | 9000 x 38040 x 8188 | 43.92 | 41.35 | **44.15** |
| DuX(65537) 11/12 @2^15.38 (variant) | executed | 15.38 | 56550 | 8608 x 38040 x 8608 | 42.24 | 41.36 | **42.87** |
| DuX(2^16) 11/12 @2^15.35 | executed | 15.35 | 25200 | 4756 x 18014 x 4756 | 40.19 | 38.57 | **40.59** |
| DuX(2^8) 8/12 @2^24 | executed | 24.0 | 65400 | 14760 x 47738 x 12056 | 44.32 | 42.95 | **44.79** |
| DuX(2^8) 8/12 @2^28.09 (variant) | executed | 28.09 | 65400 | 16296 x 47738 x 12056 | 44.09 | 43.09 | **44.68** |
| DuX(2^8) 8/12 @2^34.32 (eprint v1) | executed | 34.32 | 25200 | 4800 x 18014 x 4756 | 48.94 | 38.58 | **48.94** |
| DuX(2^8) 7/12 @2^12.70 | executed | 12.7 | 25200 | 4784 x 18014 x 4756 | 32.89 | 38.58 | **38.6** |
| Yu2X-8 8/12 @2^32 | executed | 32.0 | 39942 | 4200 x 27884 x 3822 | 47.32 | 38.7 | **47.32** |
| Yu2X-8 8/12 @2^35.46 (eprint v1) | executed | 35.46 | 39942 | 4158 x 27884 x 3822 | 50.75 | 38.69 | **50.75** |
| Yu2X-8 7/12 @2^17.58 | executed | 17.58 | 39942 | 9520 x 27884 x 4940 | 42.09 | 40.25 | **42.45** |
| Yu2X-16 11/12 @2^32 | executed | 32.0 | 39942 | 5200 x 27884 x 4940 | 47.31 | 39.38 | **47.32** |
| Yu2X-16 12/12 @2^64 (theory) | theoretical | 64.0 | 39942 | 4950 x 20250 x 4940 | 79.29 | 38.85 | **79.29** |
| YupX-65537 11/14 @2^32 | executed | 32.0 | 81360 | 9600 x 53846 x 8608 | 48.36 | 42.02 | **48.38** |
| YupX-65537 10/14 @2^14.79 | executed | 14.79 | 81360 | 8620 x 53846 x 8608 | 42.17 | 41.86 | **43.02** |
| YupX-65537 10/14 @2^15 (coset variant) | executed | 15.0 | 81360 | 9600 x 53846 x 8608 | 42.54 | 42.02 | **43.3** |
| YupX-65537 12/14 @2^64 (theory) | theoretical | 64.0 | 81360 | 8619 x 32560 x 8608 | 80.31 | 41.14 | **80.31** |
| DuX(65537) 13 @2^72.3 (theory) | theoretical | 72.3 | 56550 | 8624 x 56550 x 8188 | 88.09 | 41.86 | **88.09** |
| DuX(65537) 13 @2^65.59 (T3 candidate) | theoretical | 65.59 | 56550 | 9474 x 56550 x 8188 | 81.38 | 42.0 | **81.38** |
| CPA DuX(2^16) 8/12 @2^16 | executed | 16.0 | 3686 | 1600 x 3248 x 1086 | 36.5 | 32.39 | **36.58** |
| CPA DuX(65537) 8/12 @2^16 | executed | 16.0 | 3686 | 1600 x 3296 x 1234 | 36.5 | 32.6 | **36.59** |
| CPA DuX(2^16) 8/12 @2^23.38 (no weights) | executed | 23.38 | 3686 | 1376 x 3248 x 1336 | 36.23 | 32.48 | **36.33** |
| DuX(65537) 10/12 @2^16 (T4) | executed | 16.0 | 12 | 8 x 2 x 2 | 22.75 | 5.0 | **22.75** |
| DuX(2^16) 10/12 @2^16 (T4) | executed | 16.0 | 12 | 8 x 2 x 2 | 22.75 | 5.0 | **22.75** |
| DuX 10/12 @2^17 (two structures, variant) | executed | 17.0 | 12 | 8 x 2 x 2 | 21.58 | 5.0 | **21.58** |
| YupX-65537 9/9 (FHE parameters) @2^16 | executed | 16.0 | 12 | 8 x 2 x 2 | 22.75 | 5.0 | **22.75** |
| YupX-65537 10/14 (i) @2^16 (L-combination) | executed | 16.0 | 320 | 480 x 320 x 136 | 30.25 | 24.32 | **30.28** |
| Yu2X-16 10/12 (i) @2^16 (L-combination) | executed | 16.0 | 200 | 480 x 200 x 104 | 29.57 | 23.25 | **29.59** |
| DuX(65537) 11/12 @2^16 (eprint v1 variant) | executed | 16.0 | 56550 | 9600 x 38040 x 8608 | 42.86 | 41.52 | **43.34** |
| DuX(2^16) 11/12 @2^16 (eprint v1 variant) | executed | 16.0 | 25200 | 6000 x 18014 x 4756 | 40.84 | 38.9 | **41.17** |
| DuX(2^8) 8/12 @2^29 (T3 variant) | executed | 29.0 | 25200 | 5780 x 18014 x 4756 | 47.54 | 38.85 | **47.55** |
| CPA DuX(65537) 7/12 @2^18.81 | executed | 18.81 | 18 | 84 x 57 x 44 | 23.98 | 17.68 | **24.0** |
| CPA DuX(2^16) 7/12 @2^18.58 | executed | 18.58 | 18 | 96 x 53 x 40 | 23.75 | 17.63 | **23.77** |
| CPA DuX(2^16) 8/12 @2^35.46 (r_KR = 1) | theoretical | 35.46 | 128 | 176 x 49 x 40 | 42.46 | 18.4 | **42.46** |

## Calibration (docs/measurement_protocol.md section 4.2)

Single-thread runs only, each on a reserved physical core with `wall/(user+sys) <= 1.1`.

| run | field | dir | point-bound | passes | assemble s | solve s | us/point | ns/mult (assemble) | ns/mult (solve) |
|---|---|---|---|---|---|---|---|---|---|
| DuX(65537) 12/12 @2^29 | prime | dec | True | 3 | 69612.0 | 1127.0 | 43.221 | 0.726 | 0.345 |
| DuX(2^16) 11/12 @2^15.35 | char2 | dec | False | 1 | 1866.1 | 170.4 | None | 1.491 | 0.418 |
| DuX(2^8) 8/12 @2^24 | char2 | dec | True | 20 | 29181.6 | 3640.5 | 86.968 | 1.326 | 0.429 |
| DuX(2^8) 7/12 @2^12.70 | char2 | dec | False | 26 | 218.1 | 162.3 | None | 27.367 | 0.396 |
| Yu2X-8 7/12 @2^17.58 | char2 | dec | False | 4 | 7867.4 | 630.9 | None | 1.681 | 0.481 |
| YupX-65537 10/14 @2^14.79 | prime | dec | False | 1 | 769.0 | 922.9 | None | 0.155 | 0.231 |
| YupX-65537 10/14 @2^15 (coset variant) | prime | dec | False | 1 | 876.8 | 1495.2 | None | 0.137 | 0.336 |
| CPA DuX(2^16) 8/12 @2^16 | char2 | enc | False | 1 | 1185.9 | None | None | 12.242 | None |
| CPA DuX(65537) 8/12 @2^16 | prime | enc | False | 1 | 110.1 | None | None | 1.137 | None |
| CPA DuX(2^16) 8/12 @2^23.38 (no weights) | char2 | enc | False | 167 | 755.2 | None | None | 9.372 | None |
