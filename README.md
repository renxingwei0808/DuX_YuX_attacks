# DuX_YuX_attacks

Source code, run records and reproduction instructions for our paper

> **Practical Key Recovery Attacks on Full DuX and Reduced-Round YuX**
> Xingwei Ren, Bo Xu, Zhenyu Xiong, Yongqiang Li, Xichao Hu, Lin Jiao and Mingsheng Wang.
> Submitted to *IEEE Transactions on Computers*.

DuX (Wu et al., DCC 2026) and YuX (Liu et al., IEEE TIT 2024) are FHE-friendly
block ciphers over large finite fields.  We exploit the slow degree growth of
their decryption functions: a sufficient criterion for zero sums, a structured
linearisation of two extension rounds, and three techniques (full block
structures, cheap coordinate elimination and weighted moments) give practical
key recovery attacks in the non-adaptive chosen-ciphertext model.  This
repository contains everything we used to obtain the numbers of the paper and
its supplementary material: the reference implementations of both ciphers,
the analysis tools, the attack code, the record of every run, and the scripts
that turn the records into the tables.

## Main results

All attacks below were executed end to end on the real instances with their
real key schedules, from random master keys, and recovered all 16 words of the
master key (chosen-ciphertext model, data per key).

| Instance | Rounds | Data | Field multiplications | Time of the reported run | Keys |
|---|---|---|---|---|---|
| DuX(65537) | **12/12 (full)** | 2^29 | 2^46.50 | 19.7 h on one core | 3 |
| DuX(2^16) | **12/12 (full)** | 2^32 | 2^48.12 | 7.5 h on 19 cores | 3 |
| DuX(65537) | 11/12 | 2^15 | 2^44.15 | 1.6 h on one core | 3 |
| DuX(2^16) | 11/12 | 2^15.35 | 2^40.59 | 34 min on one core | 2 |
| DuX(65537), DuX(2^16) | 10/12 | 2^16 | 2^22.75 | seconds per key | 50 |
| DuX(2^8) | 8/12 | 2^24 | 2^44.79 | 9.1 h on one core | 3 |
| DuX(2^8) | 7/12 | 2^12.70 | 2^38.60 | 14 min on one core | 2 |
| YupX-65537 | 11/14 | 2^32 | 2^48.38 | 4.6 h on 24 cores | 3 |
| Yu2X-16 | 11/14 | 2^32 | 2^47.32 | 4.1 h on 36 cores | 3 |
| Yu2X-8 | 8/12 | 2^32 | 2^47.32 | 3.6 h on 19 cores | 3 |
| Yu2X-8 | 7/12 | 2^17.58 | 2^42.45 | 2.8 h on one core | 3 |

The complete list, including the ten-round YuX attacks, the chosen-plaintext
attacks and the theoretical routes, is in `REPRODUCE.md` and
`results/README.md`.  All runs used one server with two Xeon Gold 6230R
processors (52 physical cores, 251 GB), measured as described in
`docs/measurement_protocol.md`.

## Repository layout

| Path | Contents |
|---|---|
| `dux/` | reference implementation of DuX (scalar and NumPy-vectorised; the three instances of the design paper and toy instances over small fields) |
| `yux/` | reference implementation of YuX with the same interface (Yu2X-8, Yu2X-16, YupX-65537 and toys) |
| `tools/` | degree bounds (max-plus recursion, exponent sets), the zero-sum criterion, cheap-row kernels, point-set masks, exact degrees, top-form constants |
| `experiments/` | the experiments, one directory per series: `E01`-`E15` for DuX and the shared attack code, `Y01`-`Y10` for YuX, `S18_protocol` for measurement and multiplication counts; each has a README |
| `experiments/E07_key_recovery_2round/attack_12round.py` | the streamed driver of every two-round key recovery in the paper, including the full-round attacks |
| `results/` | the JSON record of every run, the logs of the reported runs, and generated tables; `results/README.md` maps the paper to them |
| `tests/` | unit and regression tests: cipher models, test vectors, tools, and small end-to-end attacks |
| `tests/vectors/` | five (K, P, C) test vectors per instance with all round keys and intermediate states |
| `scripts/` | test-vector generation, the symbolic check of the formulas printed in the DuX paper, and `run_pinned.sh`, the wrapper of the measurement protocol |
| `docs/` | how we read the two specifications, the measurement protocol, and a glossary of the labels used in code and records |

## Getting started

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

The C kernels need gcc with OpenMP:

```bash
for d in E04_zero_sum_F2n E06_key_recovery_1round E07_key_recovery_2round E09_unified_criterion E10_cpa; do
    make -C experiments/$d/fast
done
```

A few things to try (seconds to minutes):

```bash
# the zero-sum criterion: two active words of DuX(65537), layers 1..11
python tools/zero_sum_criterion.py --q 65537 --active 3,7 --layers 11

# measured zero sums of one active word of DuX(65537), two keys
python experiments/E03_zero_sum_Fp/run.py --instance dux-65537 --layers 12 --keys 2 \
    --positions 2,3 --out /tmp/e03

# ten-round DuX(65537) from one structure of 2^16 chosen ciphertexts, 5 random keys
python experiments/E06_key_recovery_1round/attack_1round.py --instance dux-65537 \
    --rounds 10 --auto-weight --nweights 8 --keys 5 --structures 1 --seed 2026 --out /tmp/t4

# the algebra of the 12-round attack on a toy field
python experiments/E07_key_recovery_2round/attack_12round.py --instance toy-257 \
    --layers 6 --active 3,7 --weights 150 --combine 0001 --normalise --structures 1 \
    --procs 1 --seed 11 --out /tmp/smoke --tag smoke_toy257
```

`REPRODUCE.md` gives the exact command of every reported run, the record it
produced, and the rank, determined monomials and key words a rerun must
reproduce.

## Notes on the implementation

* The cipher code follows the definitional parts of the two design papers.
  Our reading of the DuX specification was confirmed with its designers;
  the public implementation of DuX differs from its paper and was not used
  (`docs/dux_specification.md`).  For YuX, `docs/yux_specification.md` maps
  every clause of the paper to code and tests and lists the ambiguities, none
  of which affects the recovered key rk^0.
* The test vectors in `tests/vectors/` were generated by us, since no public
  vectors agree with the papers; they freeze the semantics of the reference
  models.
* Every attack starts from a fresh random master key drawn from `--seed`,
  reads only the returned plaintexts and the public parameters, and uses the
  key only to answer the oracle queries and, after solving, to check the
  result.
* The implementation is Python with NumPy; the inner loops of the moment
  assembly and of the characteristic-2 elimination are in C.  The same code
  computes the criterion, the kernels, the template bounds, the weight ranges
  and the multiplication counts.
* Directory names, run tags and comments carry our internal labels (for
  example `O12` for the weighted-moments theorem, `T1` for the cubic
  coordinate, `S18` for the measurement protocol); `docs/glossary.md` maps
  them to the paper.
