# Y04: zero sums on YuX (large structures) and tightness

`grid.sh [THREADS] [KEYS]` runs the full-scale YuX grid with the C kernel of
E09 (`zsx.c` compiled with `-DCIPHER_YUX`, picked by `run_zsx.py` from the
instance name): the 2^32 structures (two words, four words, the full block of
Yu2X-8, the chosen-plaintext two-word control) and a three-key rerun of the
2^16 cells, so that every row comes from the same binary.  `tightness.py`
checks whether the max-plus bound D is attained, with the character sum at
`a* = q - 1 - D`.

```bash
make -C experiments/E09_unified_criterion/fast
sh experiments/Y04_zero_sum/grid.sh 64 3
python3 experiments/Y04_zero_sum/tightness.py --instance yupx-65537 --pos 0 \
        --layers 9 --keys 3 --out results/Y04_zero_sum
```

Records: `results/Y04_zero_sum/`.  Together with Y03 these are the YuX rows of
Table "zero-sum distinguishers" of the paper.
