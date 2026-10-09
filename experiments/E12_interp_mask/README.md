# E12: point sets with the divided-difference mask (O15)

Proposition "point sets" of the paper: if every active word runs over an
arbitrary set U_i of n_i points and the sum is weighted by the
divided-difference mask `w(x) = prod_i w_{x_i}`, `w_u = prod_{v != u} (u - v)^-1`,
the criterion holds with the threshold `T' = sum (n_i - 1)`, and so do the
weighted moments for `|a| < T' - D`, in both characteristics.  The data of an
attack then becomes a continuous quantity instead of a power of q.

| Where | What |
|---|---|
| `tools/interp_mask.py` | the mask (both characteristics) and the `--check` of the criterion on toys |
| `tools/zero_sum_criterion.py` | `threshold(..., pointset=)`, `patterns(..., pointset=)` |
| `experiments/E07_key_recovery_2round/weighted.py` | `plan(..., pointset=)`, `point_sets()` |
| `experiments/E07_key_recovery_2round/assemble_fast.py` | `weighted_moment_stream(..., points=, mask=)` |
| `attack_2round_fast.py`, `attack_12round.py` | `--point-set N` (the point set is derived from `--seed + 4000` and recorded as `point_set` in the JSON) |

```bash
python tools/interp_mask.py --check --instance yuxtoy-257 --word 3 --n 100 --layers 4 --keys 2
python experiments/E12_interp_mask/run_toy.py --plan --out results/E12_interp_mask
python experiments/E07_key_recovery_2round/attack_2round_fast.py --instance toy-257 \
    --layers 5 --pos 3 --unknown-blocks 0,1,2,3 --outer-blocks 0,1,2,3 \
    --structures 55 --weights 40 --weight-dims 1 --normalise --seed 2026 \
    --point-set 250 --out results/E12_interp_mask --tag E12_pset250_toy257_r7_seed2026
```

Records: `results/E12_interp_mask/`; the point-set attacks on the real
instances are in `results/R8_server/`.
