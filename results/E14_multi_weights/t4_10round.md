# E14 / T4 — the ten-round DuX attack from one structure

`attack_1round.py --structures 1 --auto-weight` against the published
two-structure row of Table 6.  Both use the nine-layer zero-sum and
r_KR = 1; the difference is only whether the second equation for
(k_{4j}, k_{4j+3}) comes from a second structure or from a weight.

| instance | rounds | variant | structures | weights used | data | keys | success | time |
|---|---|---|---|---|---|---|---|---|
| dux-65537 | 10 | one structure + weights (T4) | 1 | [0, 1, 2, 3, 4, 5, 6, 7] | **2^16.0** | 50 | 50/50 | 93.3 s |
| dux-65537 | 10 | two structures, plain sums (Table 6) | 2 | [0] | **2^17.0** | 50 | 50/50 | 50.6 s |
| dux-2^16 | 10 | one structure + weights (T4) | 1 | [0, 1, 2, 3, 4, 5, 6, 7] | **2^16.0** | 50 | 50/50 | 511.3 s |
| dux-2^16 | 10 | two structures, plain sums (Table 6) | 2 | [0] | **2^17.0** | 50 | 50/50 | 26.9 s |

The weight range the criterion admits at layer 9 is `a < 24991` for DuX(65537) and `a < 24990` for DuX(2^16); the attack needs only the handful `--auto-weight` picks, so the margin is not the binding constraint.

Consequence for the ledger: the ten-round row becomes **2^16**, the
same as the eleven-round row, and the Table 6 footnote "data is one
structure per key except for DuX(2^8)" stops contradicting it (memo
E3).

