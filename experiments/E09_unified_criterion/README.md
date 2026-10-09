# E09: the unified zero-sum criterion (O7) and its tightness

The criterion itself is `tools/zero_sum_criterion.py` (Theorem "unified
zero-sum criterion"): the formal degree D of a word against the threshold T of
the structure, for subspaces, full fields and cosets, in both characteristics
and both directions.  This directory holds the generic measurement kernel and
the exhaustive minimal-data search.

## `fast/zsx.c`: one kernel for all structures

Each active word carries a table of `(value, sign)` pairs; a structure is the
product set, a point is weighted by the product of the signs.

| Set | Meaning | Contribution to T |
|---|---|---|
| `full` | the whole of F_q | q - 1 |
| `dimM` | an F_2-affine subspace of dimension M (characteristic 2) | 2^M - 1 |
| `cosetK` | coset_0 (+1) and coset_1 (-1) of the subgroup of order 2^K (F_p) | 2^K |

```bash
make -C experiments/E09_unified_criterion/fast
python experiments/E09_unified_criterion/fast/run_zsx.py --instance dux-65537 --active 3,7 \
    --layers 12 --keys 3 --threads 64 --out results/E03_zero_sum_Fp/multiword_full
python experiments/E09_unified_criterion/fast/run_zsx.py --instance dux-2^16 --active 1,5 \
    --direction enc --layers 8 --keys 3
python experiments/E09_unified_criterion/fast/run_zsx.py --instance toy-193 --active 3,7 \
    --layers 6 --keys 2 --verify          # compare with the numpy reference
```

Every run prints the O7 prediction next to the measurement (fields
`predicted_O7`, `measured`, `prediction_matches` in the JSON).

## `fast/s6_grid.py`: the tightness grid

```bash
python experiments/E09_unified_criterion/fast/s6_grid.py --which all --keys 3 --threads 64 \
    --out results/E09_unified_criterion/grid
python experiments/E09_unified_criterion/fast/s6_grid.py --table results/E09_unified_criterion/grid
```

## Minimal data inside the framework (supplement, "Optimality of the structures")

`min_data_v2.py` and `min_data_v3.py` search exhaustively over both ciphers,
both directions, zero or one free block (O11), every subset of the remaining
words and mixed subspace dimensions, and report the cheapest structure for
every (instance, layer, pattern class); v3 covers all 15 pattern classes and
prints dim K and the usability for key recovery.

```bash
python experiments/E09_unified_criterion/min_data_v3.py \
    --out results/E09_unified_criterion/min_data_table_v3.md \
    --json results/E09_unified_criterion/min_data_table_v3.json
python experiments/E09_unified_criterion/min_data_v3.py --direction enc \
    --out results/E09_unified_criterion/min_data_table_v3_enc.md \
    --json results/E09_unified_criterion/min_data_table_v3_enc.json
```

`partition_check.py` checks the known-key zero-sum partitions
(`tools/zero_sum_criterion.py --partition`) by direct summation on the toys.
