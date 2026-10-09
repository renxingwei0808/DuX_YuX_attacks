#!/usr/bin/env bash
# Re-run every preliminary experiment in this repo (a few minutes on one core).
set -e
cd "$(dirname "$0")/.."
pytest -q tests
python experiments/E01_structure/check_structure.py | tee results/E01_structure/report_prelim.txt
for pos in 0 3; do python tools/maxplus.py --q 65537 --pos $pos --layers 11 --json results/E02_degree_bounds/maxplus_p65537_pos$pos.json; done
python tools/exponent_sets.py --n 8 --pos 3 --layers 7 --json results/E02_degree_bounds/expset_n8_pos3.json
python experiments/E03_zero_sum_Fp/run.py --instance dux-65537 --layers 12 --keys 2 --positions all --out results/E03_zero_sum_Fp/prelim_full_field
for pos in 0 1 2 3; do python experiments/E04_zero_sum_F2n/run.py --instance dux-2^8 --active $pos --layers 10 --keys 3 --out results/E04_zero_sum_F2n/prelim_2-8; done
for pos in 0 1 2 3; do python experiments/E04_zero_sum_F2n/run.py --instance dux-2^16 --active $pos --layers 12 --keys 2 --out results/E04_zero_sum_F2n/prelim_2-16; done
python experiments/E06_key_recovery_1round/attack_1round.py --instance dux-65537 --rounds 10 --pos 3 --out results/E06_key_recovery_1round/prelim
python experiments/E06_key_recovery_1round/attack_1round.py --instance dux-2^16 --rounds 10 --pos 3 --out results/E06_key_recovery_1round/prelim
python experiments/E06_key_recovery_1round/attack_1round.py --instance dux-2^8 --rounds 6 --pos 3 --out results/E06_key_recovery_1round/prelim
