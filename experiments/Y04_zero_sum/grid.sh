#!/bin/sh
# Y04 (S11-B) -- YuX zero-sum grid at full scale.
#
# The <=2^16 cells were already measured on a workstation (Y03); this grid is
# the server half: the 2^32 structures (two-word, four-word, full-block, and
# the CPA two-word control) plus a 3-key re-run of the 2^16 cells with the C
# kernel, so that every row of the table comes from the same binary.
#
# The kernel is experiments/E09_unified_criterion/fast/zsx_yux (S11-A: zsx.c
# compiled with -DCIPHER_YUX).  `run_zsx.py` picks it from the instance name.
#
#   sh experiments/Y04_zero_sum/grid.sh [THREADS] [KEYS]
set -e
T=${1:-64}
K=${2:-3}
R="$(cd "$(dirname "$0")/../.." && pwd)"
Z="$R/experiments/E09_unified_criterion/fast/run_zsx.py"
O="$R/results/Y04_zero_sum"
mkdir -p "$O"
run() { echo "=== $* ==="; python3 "$Z" --threads "$T" --keys "$K" --seed 2026 --out "$O" "$@"; echo; }

# --- Y04-1/2: single word, 2^16 (predicted 8 layers + layer 9 `1100`) --------
run --instance yupx-65537 --active 0  --layers 10 --tag Y04-1_yupx_pos0
run --instance yupx-65537 --active 3  --layers 10 --tag Y04-1_yupx_pos3
run --instance yu2x-16    --active 0  --layers 10 --tag Y04-2_yu2x16_pos0
run --instance yu2x-16    --active 3  --layers 10 --tag Y04-2_yu2x16_pos3

# --- Y04-5c: Yu2X-8 single word 2^8 and two words 2^16 -----------------------
run --instance yu2x-8 --active 0   --layers 8 --tag Y04-5c_yu2x8_1word_pos0
run --instance yu2x-8 --active 3   --layers 8 --tag Y04-5c_yu2x8_1word_pos3
run --instance yu2x-8 --active 0,4 --layers 8 --tag Y04-5b_yu2x8_2words

# --- Y04-4: Yu2X-8 FULL BLOCK 2^32 (O11; predicted layer 6 `1110`) ----------
run --instance yu2x-8 --full-block 0 --layers 8 --tag Y04-4_yu2x8_fullblock

# --- Y04-5a: Yu2X-8 four words (0,4,8,12) 2^32 (Ni table 6 comparison) ------
run --instance yu2x-8 --active 0,4,8,12 --layers 8 --tag Y04-5a_yu2x8_4words

# --- Y04-3: two words 2^32 (predicted 9 layers + layer 10 `0000`) -----------
run --instance yu2x-16    --active 0,4 --layers 11 --tag Y04-3_yu2x16_a0_4
run --instance yu2x-16    --active 0,1 --layers 11 --tag Y04-3_yu2x16_a0_1
run --instance yupx-65537 --active 0,4 --layers 11 --tag Y04-3_yupx_a0_4
run --instance yupx-65537 --active 0,1 --layers 11 --tag Y04-3_yupx_a0_1

# --- Y04-7: CPA (encryption) direction --------------------------------------
run --instance yu2x-16    --active 1   --direction enc --layers 8 --tag Y04-7_yu2x16_cpa_1word
run --instance yupx-65537 --active 1   --direction enc --layers 8 --tag Y04-7_yupx_cpa_1word
run --instance yu2x-16    --active 1,5 --direction enc --layers 8 --tag Y04-7_yu2x16_cpa_2words
run --instance yupx-65537 --active 1,5 --direction enc --layers 8 --tag Y04-7_yupx_cpa_2words
