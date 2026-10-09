# Y03: zero sums on YuX (small structures)

`run.py` measures the zero-sum pattern of all 16 words layer by layer on the
three YuX instances and three toys, and compares every cell with the
criterion.  `grid.sh` runs the 24 cells up to 2^16 data (under a CPU minute);
the larger cells are in Y04.

```bash
sh experiments/Y03_zero_sum/grid.sh
python3 experiments/Y03_zero_sum/run.py --instance yu2x-16 --active 0 --layers 10 --keys 2
python3 experiments/Y03_zero_sum/run.py --instance yuxtoy-2^4 --full-block 0 --layers 6 --keys 3
python3 experiments/Y03_zero_sum/run.py --instance yupx-65537 --active 0 --coset 15 \
        --weights 1,3,5 --layers 10 --keys 2
```

Options: `--active` (each word runs over F_q), `--full-block` (O11),
`--dim` (F_2-subspaces), `--coset` (multiplicative cosets, F_p),
`--weights` (also report the weighted sums of O12), `--direction dec|enc`,
`--layers`, `--keys`, `--seed`, `--chunk`.

JSON fields: `per_layer[i].pattern` (4-bit pattern), `per_layer[i].predicted`
(the criterion), `per_layer[i].margins` (T - D per position),
`l_full_measured` / `l_full_predicted`, `agrees_with_O7`, `data_log2`.

Two remarks: a single coset needs a nonzero weight (at a = 0 the sum over a
coset is 2^k != 0); in characteristic 2 the criterion is a lower bound, and
cells flagged `frobenius_boundary` can be better than predicted.
