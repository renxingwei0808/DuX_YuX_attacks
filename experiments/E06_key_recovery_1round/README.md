# E06: key recovery with one extension round (r_KR = 1)

* `attack_1round.py`: full 16-word zero sum after l layers, then the
  sequential linear solution of the four coordinate equations of the last
  S-box layer (O5).  Covers DuX and YuX through `dux/registry.py`.
  `--auto-weight --nweights N --structures 1` is the ten-round attack from
  one structure (technique T4).
* `attack_1round_partial.py` (and `fast/`): partial zero sums, using only the
  S-box coordinates that a partial pattern leaves balanced.
* `determinacy.py`: which key words an r_KR = 1 equation determines and the
  rank formula of the linearised system (Proposition "key dependence", O8),
  for DuX and `--cipher yux`.
* `lcombined_check_toy257.py`: stand-alone check of the forward combined
  equations on the toy DuX(257).

```bash
# ten rounds, 50 random keys, one structure of 2^16 ciphertexts per key
python experiments/E06_key_recovery_1round/attack_1round.py --instance dux-65537 \
    --rounds 10 --auto-weight --nweights 8 --keys 50 --structures 1 --seed 2026 \
    --out results/S18_protocol/S18_1t_t4_dux65537_r10_seed2026
python experiments/E06_key_recovery_1round/determinacy.py --out results/E06_key_recovery_1round
```

Records: `results/E06_key_recovery_1round/`, `results/E14_multi_weights/t4_raw/`.
