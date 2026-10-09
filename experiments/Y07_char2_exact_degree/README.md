# Y07: exact degrees in characteristic 2 (O13)

In characteristic 2 the exact degree of YuX's decryption state lies below the
max-plus bound, independently of the key (supplement, "The Characteristic-2
Mechanism").  These scripts measure by how much.

| File | Role |
|---|---|
| `exact_degree.py` | exact reduced degree of every word and layer for a single-word structure, by scanning the character sums `sum_x x^a Z(x)` upwards from `a = q - 1 - D` |
| `exact_degree_2word.py` | the same for a two-word structure at 2^32 |
| `run_grid.py` | the grid over instances, positions and three keys |

The symbolic counterpart is `tools/char2_exact_degree.py` (exact formal
degrees from the reduced coefficient arrays); `tests/test_char2_exact_degree.py`
checks that the two implementations agree.

```bash
python experiments/Y07_char2_exact_degree/run_grid.py --out results/Y07_char2_exact_degree
```

Records: `results/Y07_char2_exact_degree/`.
