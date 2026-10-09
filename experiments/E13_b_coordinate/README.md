# E13: the cubic coordinate b as a second cheap coordinate (technique T1)

Lemma "cubic coordinate" of the paper.  The two-round equation normally uses
only the quadratic coordinate `a = x2 - x0 x3 - alpha` of DuX's outer S-box.
In characteristic 2 the cubic coordinate

```
b = x1 - x3 a - alpha = x1 + x2 x3 + x0 x3^2 + alpha x3 + alpha
```

is also cheap, because `x3^2` is Frobenius-linear: `x0 x3^2` is still a
product of two sums.  Linearising `sum_P b` needs one new moment family
`Mom2[A][B][e][f] = sum_P P_A^e (P_B^f)^2` and no triple moments.  With the
cheap set {1, 2} the left kernel of the pattern `1101` has dimension 4 instead
of 1, and it does not collapse under the characteristic-2 collapse theorem.
This is what makes the 12-round attack on DuX(2^16) and the 8-round attack on
DuX(2^8) from one structure possible.

| File | Role |
|---|---|
| `bcoord.py` | the extended linearisation (`PrecompB`, `MomentsB`, `rows_from_moments_b`, `combine_rows_b`, `weighted_moments_b`, pseudo-structure generators, `column_count`) |
| `kernel_table.py` | the {1,2} kernel table over the six fields and the tau-valuations |
| `rank_b.py` | the template bound of the extended system and whether it determines the 16 inner key words |
| `run_toy.py` | end-to-end runs on the toy F_{2^4} (>= 2 keys, 16/16 and re-encryption check) |
| `min_data_x1x1.py` | the (instance, layer) cells that the new usable class opens |
| `rows_subsets.py` | control: odd weights against all weights at equal row counts |
| `s19_plan.py` | data, memory and core-hour plan of the full runs |

```bash
python tools/cheap_rows.py --cipher dux --direction dec --cheap 1,2 --table \
    --all-fields --json results/E13_b_coordinate/kernel_cheap12.json
python experiments/E13_b_coordinate/kernel_table.py --out results/E13_b_coordinate/kernel_cheap12_tau.json
python experiments/E13_b_coordinate/rank_b.py --instance toy-2^4 --rounds 5 \
    --combine 1101 --structures 4000 --checkpoints 1000,2000,3000,4000
python experiments/E13_b_coordinate/run_toy.py --instance toy-2^4 --layers 3 \
    --seeds 2026,7 --out results/E13_b_coordinate
```

The full runs use `attack_12round.py --cheap-set 1,2` (see `REPRODUCE.md`).
The change is additive: without `--cheap-set` every column count is the
published one (M = 18 025 over F_{2^n}, 38 051 over F_p).
