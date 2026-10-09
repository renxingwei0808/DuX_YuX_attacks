# Y05: key recovery on YuX

The key recovery routes of DuX carry over to YuX through `dux/registry.py`:

| Route | Requires | Script |
|---|---|---|
| A1: r_KR = 1, full zero sum | all 16 words balanced at layer l | `experiments/E06_key_recovery_1round/attack_1round.py --instance yu*` (sequential substitution, O5) |
| A4: r_KR = 1, partial zero sum and forward combined equations | some positions balanced at layer l | `partial_lcomb.py` (this directory) |
| B: r_KR = 2, structured linearisation | all 16 words balanced at layer l | `experiments/E07_key_recovery_2round/attack_2round_fast.py` / `attack_12round.py --instance yu*` |

**Why YuX needs `partial_lcomb.py`.** The rotation offsets of YuX's `L^-1`
cover all four residues mod 4, so every word of `W = L^-1(Z_l - rk^1)` depends
on all four positions of Z and a partial pattern leaves no balanced word of W
(`tools/cheap_rows.py` gives dim K = 0 for `1100`).  The forward form works
instead (supplement, "Forward Combined Equations"):

```
Z_l = L(W) + rk^1,  W = SL(P + rk^0)
for every balanced word i of layer l:  sum_P Z_{l,i} = [L(sum_P W)]_i = 0
```

With the weights of O12 one structure usually suffices.

```bash
# 9 rounds of YupX-65537 (the recommended FHE parameters), one structure per key,
# 50 random keys; the weights used are recorded in
# results/Y05_key_recovery/success_rate_yupx-65537_r9.json
python3 experiments/E06_key_recovery_1round/attack_1round.py --instance yupx-65537 \
        --rounds 9 --pos 0 --structures 1 --auto-weight --keys 50 --seed 2026 \
        --out results/Y05_key_recovery
# 10 rounds, partial zero sum and forward combined equations
python3 experiments/Y05_key_recovery/partial_lcomb.py --instance yupx-65537 --rounds 10 \
        --active 0 --weights 60 --keys 3 --seed 2026 --out results/Y05_key_recovery --tag Y05-A4_yupx_r10
# key dependence table (Proposition "key dependence", YuX version)
python3 experiments/E06_key_recovery_1round/determinacy.py --cipher yux --out results/Y05_key_recovery
```

Pitfall: `--weights` beyond the margin makes the equations false at the true
key and the solver silently returns a wrong answer;
`results/Y05_key_recovery/Y05-A4_yuxtoy2p4_r4_overshoot.json` is kept as the
counterexample (margin 1, 8 weights: 16 "determined" key words, at most one of
them correct).

Records: `results/Y05_key_recovery/`; the two-round YuX attacks are in
`results/Y06_kr2/`.
