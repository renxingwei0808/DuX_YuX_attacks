# E15: mixed structures over F_p (technique T2)

Proposition "mixed structures" of the paper.  The proof of the criterion is
per active word, so a structure whose words are of different kinds adds the
thresholds:

| Word | `sum_{x in U} x^{e+a}` is nonzero iff | Threshold `T_i` |
|---|---|---|
| `U = F_p` | `(p-1)` divides `e+a` and `e+a >= p-1` | `p - 1` |
| `U = gH`, H of order 2^k | `2^k` divides `e+a`, including `e+a = 0` | `2^k` |

so `T = sum_i T_i`, and the weighted sum vanishes for `|a| < T - D` provided
every coset axis carries `a_i >= 1` with `2^{k_i}` not dividing `a_i`.
`tools/zero_sum_criterion.py` reports both the weighted (`threshold`,
`patterns`) and the unweighted statement (`threshold_plain`,
`patterns_plain`); `zero_sum_mixed.py` measures both.  On toy-193, layer 5,
position 2 (D = 209, T = 256) the weighted sums vanish for every a < 47 =
T - D and not for a = 0 or a >= 47, on three keys.

For 12-round DuX(65537), replacing one full-field word by a coset of order
2^k costs 65 536 - 2^k of threshold and saves 16 - k bits of data, which gives
the 2^29 and 2^30 attacks.

| File | Role |
|---|---|
| `mixed.py` | coset construction, parsing of `--mixed`, the value sets per word |
| `zero_sum_mixed.py` | direct per-layer sums, plain and weighted, against the criterion |
| `data_plan.py` | the data plan of the mixed structures (`results/E15_mixed_coset/data_plan.md`) |

Entry points: `attack_12round.py --mixed 13,full`,
`attack_2round_fast.py --mixed 6,full`, `tools/zero_sum_criterion.py --coset 14,full`.
