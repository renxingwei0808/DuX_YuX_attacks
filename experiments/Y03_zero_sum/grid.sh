#!/bin/sh
# Y03 (W17) -- the zero-sum grid the task table asks for.  Every cell writes a
# JSON into results/Y03_zero_sum/.  Cells marked SERVER are beyond the
# workstation budget (>= 2^24 chosen ciphertexts) and are run by Y04.
set -e
R=results/Y03_zero_sum
RUN="python3 experiments/Y03_zero_sum/run.py --out $R"

# --- Y03-1  toy F_p instances, single word, 3 keys ------------------------
$RUN --instance yuxtoy-193 --active 0 --layers 7 --keys 3 --tag Y03-1_193_pos0
$RUN --instance yuxtoy-193 --active 3 --layers 7 --keys 3 --tag Y03-1_193_pos3
$RUN --instance yuxtoy-257 --active 0 --layers 7 --keys 3 --tag Y03-1_257_pos0
$RUN --instance yuxtoy-257 --active 3 --layers 7 --keys 3 --tag Y03-1_257_pos3

# --- Y03-2  toy-257 two words (0,4) 2^16 ----------------------------------
$RUN --instance yuxtoy-257 --active 0,4 --layers 7 --keys 2 --tag Y03-2_257_a0_4
# full block 2^32 for yuxtoy-257: SERVER

# --- Y03-3  Yu2X-4: full block, four words, single word, 3 keys -----------
$RUN --instance yuxtoy-2^4 --full-block 0 --layers 6 --keys 3 --tag Y03-3_2^4_fullblock
$RUN --instance yuxtoy-2^4 --active 0,4,8,12 --layers 6 --keys 3 --tag Y03-3_2^4_4words
$RUN --instance yuxtoy-2^4 --active 0 --layers 6 --keys 3 --tag Y03-3_2^4_1word
$RUN --instance yuxtoy-2^4 --active 3 --layers 6 --keys 3 --tag Y03-3_2^4_1word_pos3

# --- Y03-4  Yu2X-8: single word 2^8, two words 2^16 -----------------------
$RUN --instance yu2x-8 --active 0 --layers 7 --keys 3 --tag Y03-4_yu2x8_1word
$RUN --instance yu2x-8 --active 3 --layers 7 --keys 3 --tag Y03-4_yu2x8_1word_pos3
$RUN --instance yu2x-8 --active 0,4 --layers 7 --keys 2 --chunk 22 --tag Y03-4_yu2x8_2words
# 4 words (0,4,8,12) 2^32 and the full block 2^32: SERVER

# --- Y03-5  Yu2X-16 / YupX single word 2^16, 2 keys -----------------------
$RUN --instance yu2x-16 --active 0 --layers 10 --keys 2 --tag Y03-5_yu2x16_pos0
$RUN --instance yu2x-16 --active 3 --layers 10 --keys 2 --tag Y03-5_yu2x16_pos3
$RUN --instance yupx-65537 --active 0 --layers 10 --keys 2 --tag Y03-5_yupx_pos0
$RUN --instance yupx-65537 --active 3 --layers 10 --keys 2 --tag Y03-5_yupx_pos3

# --- Y03-6  YupX single subgroup coset + weights (O12 Sect. 5.5) ----------
for k in 10 12 14 15; do
  $RUN --instance yupx-65537 --active 0 --coset $k --weights 1,3,5 \
       --layers 10 --keys 2 --tag Y03-6_yupx_coset$k
done

# --- Y03-7  CPA direction, single plaintext word 2^16, 2 keys -------------
$RUN --instance yu2x-16 --active 1 --direction enc --layers 7 --keys 2 --tag Y03-7_yu2x16_cpa
$RUN --instance yupx-65537 --active 1 --direction enc --layers 7 --keys 2 --tag Y03-7_yupx_cpa
$RUN --instance yuxtoy-257 --active 1 --direction enc --layers 6 --keys 3 --tag Y03-7_257_cpa
$RUN --instance yuxtoy-2^4 --full-block 0 --direction enc --layers 6 --keys 3 --tag Y03-7_2^4_cpa_fullblock
