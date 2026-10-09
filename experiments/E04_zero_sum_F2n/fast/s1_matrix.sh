#!/usr/bin/env bash
# S1 run matrix -- large zero-sum experiments for DuX(2^n).
# Ordered cheapest first so that the decisive cheap runs land early.
#
#   bash s1_matrix.sh <a|b|c>          a = DuX(2^8), b = DuX(2^16), c = >=2^39
#
# Results: results/E04_zero_sum_F2n/large/*.json, log on stdout.
set -u
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
R=experiments/E04_zero_sum_F2n/fast/run_fast.py
OUT=results/E04_zero_sum_F2n/large
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-104}
mkdir -p "$OUT"

run () {  # run <tag> <instance> <active> <dim> <layers>
  echo "=== $(date -Is) $1  $2 active=$3 dim=$4 ==="
  $PY $R --instance "$2" --active "$3" --dim "$4" --layers "$5" \
        --keys 3 --seed 2026 --out "$OUT" --tag "$1" 2>&1
  echo
}

case "${1:-a}" in
a)   # ---------------- DuX(2^8): full (s, dim) law, up to 2^32 --------------
  run a1_p3_d8        dux-2^8 3               8 9
  run a2_p37_d8       dux-2^8 3,7             8 9
  run a3_p3711_d8     dux-2^8 3,7,11          8 9
  run a4_d4           dux-2^8 3,7,11,15       4 9
  run a4_d5           dux-2^8 3,7,11,15       5 9
  run a4_d6           dux-2^8 3,7,11,15       6 9
  run a4_d7           dux-2^8 3,7,11,15       7 9
  run a4_d8           dux-2^8 3,7,11,15       8 9     # 2^32  *** T1 ***
  run a4b_d8          dux-2^8 2,6,10,14       8 9     # 2^32
  run a8_d4           dux-2^8 2,3,6,7,10,11,14,15 4 9  # 2^32, same-block test
  ;;
b)   # ---------------- DuX(2^16): up to 2^36 ---------------------------------
  run b2_p37_d15      dux-2^16 3,7            15 12   # 2^30
  run b3_p3711_d10    dux-2^16 3,7,11         10 12   # 2^30
  run b2_p37_d16      dux-2^16 3,7            16 12   # 2^32  *** decisive ***
  run b4_d8           dux-2^16 3,7,11,15       8 12   # 2^32
  run b2sb_p23_d16    dux-2^16 2,3            16 12   # 2^32, same block
  run b3_p3711_d11    dux-2^16 3,7,11         11 12   # 2^33
  run b3_p3711_d12    dux-2^16 3,7,11         12 12   # 2^36
  run b4_d9           dux-2^16 3,7,11,15       9 12   # 2^36
  ;;
c)   # ---------------- DuX(2^16): the expensive tail -------------------------
  run b3_p3711_d13    dux-2^16 3,7,11         13 11   # 2^39
  run b4_d10          dux-2^16 3,7,11,15      10 11   # 2^40
  ;;
esac
echo "DONE $(date -Is)"
