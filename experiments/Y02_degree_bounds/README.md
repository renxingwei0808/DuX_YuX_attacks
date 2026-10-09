# Y02: YuX degree bounds and the comparison with Ni et al.

1. `ni_tables.py` transcribes the degree tables 3, 4, 6 and 7 of Ni, Wang and
   Li (DCC 2026, 94:123) and compares them cell by cell with the max-plus
   formal degrees of the criterion.
2. `o7_tables.py` produces the criterion's prediction for every YuX structure
   we use: layers, pattern of the first non-full layer and per-position
   margins T - D.

Conventions: their "round r" is our "layer r"; their "output index i" is
block 0, position i.  Our formal degree D bounds the sum of the exponents; the
Boolean degree follows from `hw(D, s, n) = max { sum HW(a_i) : sum a_i <= D,
a_i <= 2^n - 1 }`, and `hw(D, s, n) = s n` exactly when `D >= T = s(2^n - 1)`,
so the Boolean criterion `deg < s n` and the criterion `D < T` coincide.

```bash
python experiments/Y02_degree_bounds/ni_tables.py --out results/Y02_degree_bounds
python experiments/Y02_degree_bounds/o7_tables.py --out results/Y02_degree_bounds
python tools/zero_sum_criterion.py --cipher yux --q 2^16 --active 0 --layers 11
python tools/zero_sum_criterion.py --cipher yux --q 2^16 --full-block 0 --layers 11
```

Records: `results/Y02_degree_bounds/` (`ni_vs_maxplus.md`, `o7_yux_tables.md`).
Tests: `tests/test_cipher_degree.py`.
