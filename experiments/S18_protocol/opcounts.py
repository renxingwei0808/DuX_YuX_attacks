"""S18 item 4 -- the machine-independent operation count of every reported row.

The protocol (docs/measurement_protocol.md section 4) states each attack twice: once as a wall-clock
time on a named machine, and once as a count of field operations that does not
depend on the machine at all.  This script computes the second one, with the
SAME formula for an executed row and for a theoretical one, so that the two
kinds of row in Table 1 are comparable.

    oracle    = N_struct * n_points                      chosen ciphertexts
    assemble  = passes * oracle * W                      per-point moments
              + N_w * W * n_slices                       the weighted sums
              + n_rows * n_cols                          the combined rows
    solve     = n_rows * n_cols * rank                   the elimination
    total     = assemble + solve                         field multiplications

`W` is the width of the moment layout -- the number of point sums the assembly
keeps per structure -- and is not transcribed from anywhere: it is built here
from the same `Precomp` / `MomentLayout` (or `PrecompB` / `LayoutBStream` for
the T1 cheap set, or `kr2_cpa.make_precomp` for the encryption direction) that
the attack itself uses, so a change in the linearisation changes this table.
`n_rows`, `n_cols` and `rank` come from the run's own JSON wherever the row was
executed; a theoretical row states them in its spec.

The dominant term is `oracle * W`, which is exactly how the published
theoretical rows were quoted (2^47 * 2^14.95 = 2^62 for DuX(2^16) 12/12, and so
on); the other three terms are what this script adds, and section 4 of
docs/measurement_protocol.md
records which quoted exponents move as a result.

    python opcounts.py --out results/S18_protocol
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
E07 = os.path.join(ROOT, "experiments", "E07_key_recovery_2round")
for _p in (ROOT, E07, os.path.join(E07, "fast"),
           os.path.join(ROOT, "experiments", "E10_cpa"),
           os.path.join(ROOT, "experiments", "E13_b_coordinate"),
           os.path.join(ROOT, "tools")):
    sys.path.insert(0, _p)

from dux.registry import cipher_family, get_cipher                # noqa: E402


# --------------------------------------------------------------------------
# the moment-layout width, built the way each route builds it

@functools.lru_cache(maxsize=None)
def width_kr2(instance, rounds):
    """r_KR = 2, decryption direction, the ordinary linearisation:
    `MomentLayout(Precomp(...))` -- 4 blocks of nP monomials plus the 10
    unordered pairs of nP x nP joint moments."""
    from assemble_fast import MomentLayout, Precomp
    c = get_cipher(instance, rounds=rounds)
    pre = Precomp(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3],
                  cipher=cipher_family(instance))
    lay = MomentLayout(pre)
    return lay.width, lay.nP, len(pre.mons)


@functools.lru_cache(maxsize=None)
def width_kr2_b(instance, rounds):
    """T1 (cheap set {1, 2}, characteristic 2, DuX): the extended system of
    E13 -- `LayoutBStream(PrecompB(...))`."""
    import bcoord as BC
    c = get_cipher(instance, rounds=rounds)
    pre = BC.PrecompB(c.F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3])
    lay = BC.LayoutBStream(pre)
    return lay.width, lay.nP, len(pre.mons)


@functools.lru_cache(maxsize=None)
def width_cpa(instance, rounds, outer="0,1,2,3", coords="0,3"):
    """r_KR = 2, encryption direction (`kr2_cpa.py`)."""
    from assemble_fast import MomentLayout
    import kr2_cpa
    c = get_cipher(instance, rounds=rounds)
    pre = kr2_cpa.make_precomp(c, [int(x) for x in outer.split(",")],
                               [int(x) for x in coords.split(",")])
    lay = MomentLayout(pre)
    return lay.width, lay.nP, len(pre.mons)


WIDTH_FN = {"kr2": width_kr2, "kr2b": width_kr2_b, "cpa": width_cpa}


@functools.lru_cache(maxsize=None)
def field_of(instance, rounds):
    """"char2" or "prime", from the registry -- the instance name does not
    say it (`yu2x-8` is F_{2^8}, `yupx-65537` is F_p)."""
    return "char2" if get_cipher(instance, rounds=rounds).F.char == 2 else "prime"


# --------------------------------------------------------------------------
# the rows.  `width` is either a key of WIDTH_FN (built here) or an integer
# with a note saying where it comes from; `json` points at the run's record.

def R(**kw):
    kw.setdefault("structures", 1)
    kw.setdefault("passes", 1)
    kw.setdefault("status", "executed")
    kw.setdefault("note", "")
    return kw


ROWS = [
    # ---- DuX, chosen ciphertext (ledger A) --------------------------------
    R(id="dux65537_r12_2p29", label="DuX(65537) 12/12 @2^29", instance="dux-65537",
      rounds=12, route="2^13 coset x full field, two-axis weights (T2+T3)",
      data_log2=29.0, width="kr2", n_w=10500, slices=8192, passes=3,
      json="results/S18_protocol/s10_S18_1t_dux65537_r12_k13_2axis_seed11.json,"
           "results/R9_server/s10_S20_dux65537_r12_mixed13full_2axis_seed2026.json"),
    R(id="dux65537_r12_2p30", label="DuX(65537) 12/12 @2^30 (variant)",
      instance="dux-65537", rounds=12, route="2^14 coset x full field (T2)",
      data_log2=30.0, width="kr2", n_w=9000, slices=16384,
      json="results/R9_server/s10_S20_dux65537_r12_mixed14full_seed2026.json"),
    R(id="dux65537_r12_pset", label="DuX(65537) 12/12 @2^30.55 (variant)",
      instance="dux-65537", rounds=12, route="two 39 614-point sets (O15)",
      data_log2=30.55, width="kr2", n_w=9000, slices=39614,
      json="results/R8_server/s10_R8optA_dux65537_r12_pset39614_seed2026.json"),
    R(id="dux65537_r12_full", label="DuX(65537) 12/12 @2^32 (eprint v1)",
      instance="dux-65537", rounds=12, route="two words, full domain",
      data_log2=32.0, width="kr2", n_w=9000, slices=65537,
      json="results/E07_key_recovery_2round/s10/s10_S10_dux65537_r12_seed2026.json"),
    R(id="dux2p16_r12_2p32", label="DuX(2^16) 12/12 @2^32", instance="dux-2^16",
      rounds=12, route="two words, full domain, cheap set {1,2} (T1)",
      data_log2=32.0, width="kr2b", n_w=3780, slices=65536,
      json="results/S18_protocol/s10_S18_mp_dux2p16_r12_2w_cheap12_seed11.json,"
           "results/R9_server/s10_S19_dux2p16_r12_2w_cheap12_seed2026.json"),
    R(id="dux2p16_r12_theory", label="DuX(2^16) 12/12 @2^47 (eprint v1 theory)",
      instance="dux-2^16", rounds=12, status="theoretical",
      route="three words, affine subspace (16,16,15), 16-word zero sum",
      data_log2=47.0, width="kr2", n_w=12521, slices=2 ** 31, dim_k=4,
      rows=50084, cols=18014, rank=4756,
      note="published as ~2^62; rank/cols from the eleven-round row's system"),
    R(id="dux65537_r11_coset", label="DuX(65537) 11/12 @2^15", instance="dux-65537",
      rounds=11, route="one 2^15 coset, `0001` combined row (T2)",
      data_log2=15.0, width="kr2", n_w=9000, slices=32768,
      json="results/R9_server/s10_S20_dux65537_r11_coset15_0001_seed2026.json"),
    R(id="dux65537_r11_pset", label="DuX(65537) 11/12 @2^15.38 (variant)",
      instance="dux-65537", rounds=11, route="42 698-point set (O15)",
      data_log2=15.38, width="kr2", n_w=2152, slices=42698,
      json="results/R8_server/s10_R8_dux65537_r11_pset42698_seed2026.json"),
    R(id="dux2p16_r11_pset", label="DuX(2^16) 11/12 @2^15.35", instance="dux-2^16",
      rounds=11, route="41 735-point set (O15)",
      data_log2=15.35, width="kr2", n_w=1189, slices=41735,
      json="results/S18_protocol/s10_S18_1t_dux2p16_r11_pset41735_seed2026.json"),
    R(id="dux2p8_r8_2p24", label="DuX(2^8) 8/12 @2^24", instance="dux-2^8",
      rounds=8, route="three words, cheap set {1,2}, two-axis weights (T1+T3)",
      data_log2=24.0, width="kr2b", n_w=3690, slices=256, passes=20,
      json="results/S18_protocol/s10_S18_1t_dux2p8_r8_3w_cheap12_2axis_seed11.json,"
           "results/R9_server/s10_S19_dux2p8_r8_3w_cheap12_2axis_seed2026.json"),
    R(id="dux2p8_r8_2p28", label="DuX(2^8) 8/12 @2^28.09 (variant)",
      instance="dux-2^8", rounds=8, structures=17,
      route="17 three-word 2^24 structures, cheap set {1,2} (T1)",
      data_log2=28.09, width="kr2b", n_w=194, slices=256,
      json="results/R9_server/s10_S19_dux2p8_r8_3w_cheap12_seed2026.json"),
    R(id="dux2p8_r8_2p34", label="DuX(2^8) 8/12 @2^34.32 (eprint v1)",
      instance="dux-2^8", rounds=8, structures=5,
      route="five four-word 2^32 structures, plain rows",
      data_log2=34.32, width="kr2", n_w=240, slices=256,
      json="results/R6_third_keys/s10_R6D4_dux2p8_r8_seed2026.json"),
    R(id="dux2p8_r7", label="DuX(2^8) 7/12 @2^12.70", instance="dux-2^8",
      rounds=7, structures=26, route="26 one-word 2^8 structures, plain rows",
      data_log2=12.70, width="kr2", n_w=46, slices=256,
      json="results/S18_protocol/s10_S18_1t_dux2p8_r7_seed2026.json,"
           "results/E07_key_recovery_2round/s9/s10_S9_step4_dux2p8_r7_seed2026.json"),

    # ---- YuX, chosen ciphertext (ledger A') -------------------------------
    R(id="yu2x8_r8", label="Yu2X-8 8/12 @2^32", instance="yu2x-8", rounds=8,
      route="one full block, two-axis weights (T3)",
      data_log2=32.0, width="kr2", n_w=1400, slices=65536,
      json="results/S18_protocol/s10_S18_mp_yu2x8_r8_fb0_2axis_seed11.json,"
           "results/R9_server/s10_S21_yu2x8_r8_fb0_2axis_seed2026.json"),
    R(id="yu2x8_r8_11", label="Yu2X-8 8/12 @2^35.46 (eprint v1)",
      instance="yu2x-8", rounds=8, structures=11,
      route="11 full blocks, 126 one-axis weights",
      data_log2=35.46, width="kr2", n_w=126, slices=65536,
      json="results/Y06_kr2/y08/s10_Y08-C_yu2x8_r8_seed2026.json"),
    R(id="yu2x8_r7", label="Yu2X-8 7/12 @2^17.58", instance="yu2x-8", rounds=7,
      structures=3, route="three two-word 2^16 structures, two-axis weights (T3)",
      data_log2=17.58, width="kr2", n_w=595, slices=65536,
      json="results/S18_protocol/s10_S18_1t_yu2x8_r7_2w_2axis_seed11.json,"
           "results/R9_server/s10_S21_yu2x8_r7_2w_2axis_seed2026.json"),
    R(id="yu2x16_r11", label="Yu2X-16 11/12 @2^32", instance="yu2x-16", rounds=11,
      route="two words, full domain", data_log2=32.0, width="kr2",
      n_w=1300, slices=65536,
      json="results/Y06_kr2/y08/s10_Y08-B_yu2x16_r11_seed2026.json"),
    # S22 note: n_w = 1650 = ceil(4 950 / 3) is the batch count from the
    # four-row template bound, not the weight count the paper cites -- that is
    # the 32 766 usable weights (margin 65 532, 2a < 65 532).  The total is
    # dominated by passes * N * W = 2^64 * 39 942 = 2^79.29; even at 32 766
    # the N_w * W * slices term is only ~2^62.3, so both decimals stay.
    R(id="yu2x16_r12_theory", label="Yu2X-16 12/12 @2^64 (theory)",
      instance="yu2x-16", rounds=12, status="theoretical",
      route="one full block 2^64, `1110` combined rows",
      data_log2=64.0, width="kr2", n_w=1650, slices=2 ** 32, dim_k=3,
      rows=4950, cols=20250, rank=4940,
      note="published as ~2^79"),
    R(id="yupx_r11", label="YupX-65537 11/14 @2^32", instance="yupx-65537",
      rounds=11, route="two words, full domain", data_log2=32.0, width="kr2",
      n_w=2400, slices=65537,
      json="results/Y06_kr2/y06/s10_Y06-2_yupx_r11_seed2026.json"),
    R(id="yupx_r10_pset", label="YupX-65537 10/14 @2^14.79", instance="yupx-65537",
      rounds=10, route="28 239-point set (O15)", data_log2=14.79, width="kr2",
      n_w=2155, slices=28239,
      json="results/S18_protocol/s10_S18_1t_yupx_r10_pset28239_seed2026.json"),
    R(id="yupx_r10_coset", label="YupX-65537 10/14 @2^15 (coset variant)",
      instance="yupx-65537", rounds=10, route="one 2^15 coset",
      data_log2=15.0, width="kr2", n_w=2400, slices=32768,
      json="results/S18_protocol/s10_S18_1t_yupx_r10_coset15_seed2026.json,"
           "results/R6_third_keys/s10_R6D3_yupx_r10_coset15_seed2026.json"),
    # S22 note: n_w = 2873 = ceil(8 618 / 3) is the batch count from the
    # four-row template bound, not the weight count the paper cites -- that is
    # the 32 768 usable weights.  The total is dominated by passes * N * W =
    # 2^64 * 81 360 = 2^80.31; even at 32 768 the N_w * W * slices term is
    # only ~2^63.3, so both decimals stay.
    R(id="yupx_r12_theory", label="YupX-65537 12/14 @2^64 (theory)",
      instance="yupx-65537", rounds=12, status="theoretical",
      route="one full block 2^64, `1110` combined rows (dim K = 3)",
      data_log2=64.0, width="kr2", n_w=2873, slices=2 ** 32, dim_k=3,
      rows=8619, cols=32560, rank=8608,
      note="published as ~2^80.3 (44 us/point)"),
    R(id="dux65537_r13_theory", label="DuX(65537) 13 @2^72.3 (theory)",
      instance="dux-65537", rounds=13, status="theoretical", structures=308,
      route="full block 2^64, `0001` combined row, 28 weights per structure",
      data_log2=72.3, width="kr2", n_w=28, slices=2 ** 32, dim_k=1,
      rows=8624, cols=56550, rank=8188,
      note="published as ~2^88"),
    R(id="dux65537_r13_t3", label="DuX(65537) 13 @2^65.59 (T3 candidate)",
      instance="dux-65537", rounds=13, status="theoretical", structures=3,
      route="full block 2^64, multi-index weights, 3 158 per structure",
      data_log2=65.59, width="kr2", n_w=3158, slices=2 ** 32, dim_k=1,
      rows=9474, cols=56550, rank=8188,
      note="J-section candidate, never quoted before"),

    # ---- chosen plaintext (ledger C) --------------------------------------
    R(id="cpa8_dux2p16", label="CPA DuX(2^16) 8/12 @2^16", instance="dux-2^16",
      rounds=8, route="one word 2^16, `1110` combined rows, 400 weights",
      data_log2=16.0, width="cpa", n_w=400, slices=65536,
      json="results/S18_protocol/kr2cpa_S18_1t_cpa8_dux2p16_seed2026.json"),
    R(id="cpa8_dux65537", label="CPA DuX(65537) 8/12 @2^16", instance="dux-65537",
      rounds=8, route="one word 2^16, `1110` combined rows, 400 weights",
      data_log2=16.0, width="cpa", n_w=400, slices=65536,
      json="results/S18_protocol/kr2cpa_S18_1t_cpa8_dux65537_seed2026.json,"
           "results/E10_cpa/s9/kr2cpa_S9_cpa8_dux65537_seed2026.json"),
    R(id="cpa8_dux2p16_nw", label="CPA DuX(2^16) 8/12 @2^23.38 (no weights)",
      instance="dux-2^16", rounds=8, structures=167,
      route="167 one-word 2^16 structures, plain rows",
      data_log2=23.38, width="cpa", n_w=1, slices=65536,
      json="results/S18_protocol/S18_1t_cpa8_dux2p16_nw_seed2026"),
    # ---- r_KR = 1 rows.  These have no big linearisation at all: each
    # block is recovered from two independent rows of a tiny system, so the
    # per-point work is the 4 blocks' balanced S^-1 coordinate sums.  W = 12
    # (4 blocks x 3 coordinates) is that moment set, NOT a `MomentLayout`;
    # the row/column/rank triple is the 2 x 2 solve per block.
    R(id="dux65537_r10_t4", label="DuX(65537) 10/12 @2^16 (T4)",
      instance="dux-65537", rounds=10, route="one structure + 8 weights (T4)",
      data_log2=16.0, width=12, n_w=8, slices=65536, rows=8, cols=2, rank=2,
      note="r_KR = 1; 50 keys; W = 12 is the four blocks' coordinate sums",
      json="results/S18_protocol/S18_1t_t4_dux65537_r10_seed2026"),
    R(id="dux2p16_r10_t4", label="DuX(2^16) 10/12 @2^16 (T4)",
      instance="dux-2^16", rounds=10, route="one structure + 8 weights (T4)",
      data_log2=16.0, width=12, n_w=8, slices=65536, rows=8, cols=2, rank=2,
      note="r_KR = 1; 50 keys",
      json="results/S18_protocol/S18_1t_t4_dux2p16_r10_seed2026"),
    R(id="dux_r10_2structs", label="DuX 10/12 @2^17 (two structures, variant)",
      instance="dux-65537", rounds=10, structures=2,
      route="two structures, plain sums (eprint v1 Table 6)",
      data_log2=17.0, width=12, n_w=1, slices=65536, rows=8, cols=2, rank=2,
      note="r_KR = 1; 50 keys"),
    R(id="yupx_r9", label="YupX-65537 9/9 (FHE parameters) @2^16",
      instance="yupx-65537", rounds=9, route="layer 8 zero sum + r_KR = 1",
      data_log2=16.0, width=12, n_w=8, slices=65536, rows=8, cols=2, rank=2,
      note="r_KR = 1; 50/50 keys"),
    R(id="yupx_r10_lcomb", label="YupX-65537 10/14 (i) @2^16 (L-combination)",
      instance="yupx-65537", rounds=10,
      route="layer 9 `1100` partial sum + forward L-combination + 60 weights",
      data_log2=16.0, width=320, n_w=60, slices=65536,
      note="W = 4 blocks x 80 monomials, from the run's own record",
      json="results/S18_protocol/S18_1t_yupx_r10_lcomb_seed2026.json"),
    R(id="yu2x16_r10_lcomb", label="Yu2X-16 10/12 (i) @2^16 (L-combination)",
      instance="yu2x-16", rounds=10,
      route="layer 9 `1100` partial sum + forward L-combination + 60 weights",
      data_log2=16.0, width=200, n_w=60, slices=65536,
      note="W = 4 blocks x 50 monomials",
      json="results/S18_protocol/S18_1t_yu2x16_r10_lcomb_seed2026.json"),

    # ---- full-domain variants kept in the ledger --------------------------
    R(id="dux65537_r11_full", label="DuX(65537) 11/12 @2^16 (eprint v1 variant)",
      instance="dux-65537", rounds=11, route="one word, full domain",
      data_log2=16.0, width="kr2", n_w=2152, slices=65537,
      json="results/E07_key_recovery_2round/s9/s10_S9_step2_dux65537_r11_seed2026.json"),
    R(id="dux2p16_r11_full", label="DuX(2^16) 11/12 @2^16 (eprint v1 variant)",
      instance="dux-2^16", rounds=11, route="one word, full domain",
      data_log2=16.0, width="kr2", n_w=1189, slices=65536,
      json="results/E07_key_recovery_2round/s9/s10_S9_step3_dux2p16_r11_seed2026.json"),
    R(id="dux2p8_r8_2p29", label="DuX(2^8) 8/12 @2^29 (T3 variant)",
      instance="dux-2^8", rounds=8,
      route="one (8,8,8,5) subspace structure, four-axis weights (T3)",
      data_log2=29.0, width="kr2", n_w=1445, slices=65536, passes=15,
      json="results/R9_server/s10_S21_dux2p8_r8_sub8885_4axis_seed2026.json"),
    # the seven-round CPA rows are r_KR = 1 as well: W = 18 is the moment set
    # `attack_cp_lastround.py` records for one structure.
    R(id="cpa7_dux65537", label="CPA DuX(65537) 7/12 @2^18.81", instance="dux-65537",
      rounds=7, structures=7, route="seven one-word 2^16 structures, r_KR = 1",
      data_log2=18.81, width=18, n_w=1, slices=65536,
      json="results/S18_protocol/S18_1t_cpa7_dux65537_seed2026"),
    R(id="cpa7_dux2p16", label="CPA DuX(2^16) 7/12 @2^18.58", instance="dux-2^16",
      rounds=7, structures=6, route="six one-word 2^16 structures, r_KR = 1",
      data_log2=18.58, width=18, n_w=1, slices=65536,
      json="results/S18_protocol/S18_1t_cpa7_dux2p16_seed2026"),
    R(id="cpa8_dux2p16_rkr1", label="CPA DuX(2^16) 8/12 @2^35.46 (r_KR = 1)",
      instance="dux-2^16", rounds=8, status="theoretical", structures=11,
      route="two words 2^32, layer 7 `0010`, balanced position 2 only",
      data_log2=35.46, width=128, n_w=1, slices=65536,
      rows=176, cols=49, rank=40,
      note="published as 2^42.46; W = 128 is the r_KR = 1 moment set "
           "(`attack_cp_lastround.py`), not a `MomentLayout`"),
]


def op_counts(row, width, rows_, cols, rank):
    n_points = 2.0 ** row["data_log2"] / max(1, row["structures"])
    oracle = row["structures"] * n_points
    # `passes` is how many times the assembly walks the structure: 1 for a
    # one-axis weight route, one per extra weight-axis exponent otherwise
    # (`--weight-dims 2 --weight-a1 0,1,2` is three passes; the joint
    # Vandermonde of `--vandermonde-axes m` collapses m axes into ONE pass).
    # The chosen-ciphertext count does NOT multiply -- the same points are
    # reused -- but the per-point moment work does.
    per_point = row.get("passes", 1) * oracle * width
    weighted = row["n_w"] * width * row["slices"] * row["structures"]
    assembled = rows_ * cols
    solve = rows_ * cols * rank
    total = per_point + weighted + assembled + solve
    lg = lambda x: (round(math.log2(x), 2) if x > 0 else None)      # noqa: E731
    return {"oracle": oracle, "oracle_log2": lg(oracle),
            "per_point_mults": per_point, "weighted_mults": weighted,
            "row_mults": assembled, "assemble_mults": per_point + weighted + assembled,
            "assemble_log2": lg(per_point + weighted + assembled),
            "solve_mults": solve, "solve_log2": lg(solve),
            "total_mults": total, "total_log2": lg(total)}


def rel(path):
    return os.path.relpath(path, ROOT) if path else path


def resolve(path):
    """The row's evidence file.  `path` may list several candidates separated
    by commas -- the S18 protocol rerun first, the frozen record it has to
    agree with second -- and the first one that exists wins.  Both give the
    same rows / columns / rank (that is what makes a protocol rerun a rerun),
    so the count is stable while the reruns are still going.  A `--out <dir>`
    run names its own JSON, so a directory resolves to the JSON inside it."""
    if not path:
        return None
    for one in path.split(","):
        p = one if os.path.isabs(one) else os.path.join(ROOT, one)
        if os.path.isdir(p):
            cand = [f for f in sorted(os.listdir(p)) if f.endswith(".json")]
            p = os.path.join(p, cand[0]) if cand else None
        if p and os.path.exists(p):
            return p
    return None


def from_json(path):
    """rows / cols / rank of an executed row, from its own record."""
    p = resolve(path)
    if not p or not os.path.exists(p):
        return None
    d = json.load(open(p))
    rows_ = d.get("equations") or d.get("rows")
    if rows_ is None and d.get("equations_per_structure"):   # the CPA driver
        rows_ = d["equations_per_structure"] * d.get("structures", 1)
    cols = d.get("unknowns")
    rank = d.get("rank")
    if rank is None and d.get("results"):     # one entry per key (r_KR = 1,
        rank = d["results"][0].get("rank")    # partial_lcomb, the CPA drivers)
    if isinstance(rank, list):
        rank = rank[0]
    if rows_ is None or cols is None or rank is None:
        return None
    return int(rows_), int(cols), int(rank), d


@functools.lru_cache(maxsize=None)
def build_cached():
    return tuple(build())


def build(rows=ROWS):
    out = []
    for row in rows:
        w = row["width"]
        if isinstance(w, str):
            width, nP, cols_pre = WIDTH_FN[w](row["instance"], row["rounds"])
            wsrc = w
        else:
            width, nP, cols_pre = int(w), None, None
            wsrc = "given"
        got = from_json(row.get("json"))
        if got:
            rows_, cols, rank, _d = got
            src = "run JSON"
        else:
            rows_, cols, rank = (row.get("rows"), row.get("cols"), row.get("rank"))
            src = "spec" if rows_ else "missing"
            if rows_ is None:
                rows_ = row["n_w"] * row.get("dim_k", 4) * row["structures"]
                cols, rank = cols_pre or 0, 0
                src = "derived"
        rec = {k: row[k] for k in ("id", "label", "instance", "rounds", "route",
                                   "data_log2", "structures", "n_w", "slices",
                                   "status", "note")}
        rec.update({"width": width, "width_source": wsrc, "nP": nP,
                    "rows": rows_, "cols": cols, "rank": rank,
                    "system_source": src, "json_path": rel(resolve(row.get("json")))})
        rec.update(op_counts(row, width, rows_, cols, rank))
        out.append(rec)
    return out


def calibrate(recs, out_dir):
    """docs/measurement_protocol.md section 4.2: turn the single-thread runs into the two constants
    that let an operation count be read as a time.

    Only runs that passed rule 1 (`protocol.single_threaded` and
    `not_preempted`) are used -- a preempted run would understate the machine.
    The constants are reported PER FIELD: the weighted accumulation is a
    mod-p dgemm on the prime-field instances and a log-table gather in
    characteristic 2, and the two are not within a factor of each other."""
    cal = []
    for r in recs:
        path = os.path.join(ROOT, r["json_path"]) if r.get("json_path") else None
        if not path or not os.path.exists(path):
            continue
        d = json.load(open(path))
        prot = d.get("protocol") or {}
        if not (prot.get("single_threaded") and prot.get("not_preempted")):
            continue
        # `per_structure` already has one entry per structure (and per extra
        # weight-axis pass), so it IS the pass count; do not multiply by the
        # structure count again.
        passes = len(d.get("per_structure") or []) or r["structures"]
        asm, slv = d.get("assemble_s"), d.get("solve_s")
        if asm is None:              # the r_KR = 1 drivers time the whole run
            continue                 # only; nothing to calibrate a term with
        solve_mults = r["solve_mults"]
        asm_mults = r["assemble_mults"]
        # "microseconds per point" is only meaningful where the per-point term
        # is what the assembly actually spends its time on.  On a point-set or
        # a joint-Vandermonde row a slice is a single point, so `N_w * W *
        # slices` dwarfs `oracle * W` and the ratio would measure the weighted
        # sum, not the points.  Gate it at half the assembly.
        point_bound = r["per_point_mults"] >= 0.5 * r["assemble_mults"]
        rec = {"id": r["id"], "label": r["label"],
               "field": field_of(r["instance"], r["rounds"]),
               "direction": "enc" if r["id"].startswith("cpa") else "dec",
               "point_bound": bool(point_bound), "passes": passes,
               "assemble_s": asm, "solve_s": slv,
               "us_per_point": (round(asm * 1e6 / (r["oracle"] * passes), 3)
                                if asm and point_bound else None),
               "ns_per_mult_assemble": (round(asm * 1e9 / asm_mults, 3)
                                        if asm and asm_mults else None),
               "ns_per_mult_solve": (round(slv * 1e9 / solve_mults, 3)
                                     if slv and solve_mults else None)}
        cal.append(rec)
    by_field = {}
    for c in cal:
        by_field.setdefault(f"{c['field']}_{c['direction']}", []).append(c)
    summary = {}
    for f, cs in by_field.items():
        for key in ("us_per_point", "ns_per_mult_assemble", "ns_per_mult_solve"):
            vals = [c[key] for c in cs if c[key]]
            if vals:
                summary[f"{f}_{key}"] = {"n": len(vals),
                                         "min": round(min(vals), 3),
                                         "max": round(max(vals), 3),
                                         "median": round(sorted(vals)[len(vals) // 2], 3)}
    return {"runs": cal, "per_field": summary}


def markdown(recs):
    head = ("| row | status | data (log2) | W | rows x cols x rank | "
            "assemble (log2) | solve (log2) | **total (log2)** |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in recs:
        lines.append(
            f"| {r['label']} | {r['status']} | {r['data_log2']} | {r['width']} | "
            f"{r['rows']} x {r['cols']} x {r['rank']} | {r['assemble_log2']} | "
            f"{r['solve_log2']} | **{r['total_log2']}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "S18_protocol"))
    a = ap.parse_args()
    recs = build()
    cal = calibrate(recs, a.out)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"formula": "oracle*W + N_w*W*slices + rows*cols + rows*cols*rank",
               "calibration": cal, "rows": recs},
              open(os.path.join(a.out, "opcounts.json"), "w"), indent=1)
    with open(os.path.join(a.out, "opcounts.md"), "w") as fh:
        fh.write("# S18 -- operation counts (docs/measurement_protocol.md section 4)\n\n"
                 "`oracle = N_struct * n_points`; "
                 "`assemble = oracle * W + N_w * W * slices + rows * cols`; "
                 "`solve = rows * cols * rank`; the total is their sum, in "
                 "field multiplications.  `W` is the moment-layout width, "
                 "built by `experiments/S18_protocol/opcounts.py` from the "
                 "same `Precomp` / `MomentLayout` the attack uses.\n\n")
        fh.write(markdown(recs) + "\n")
        if cal["runs"]:
            fh.write("\n## Calibration (docs/measurement_protocol.md section 4.2)\n\n"
                     "Single-thread runs only, each on a reserved physical "
                     "core with `wall/(user+sys) <= 1.1`.\n\n"
                     "| run | field | dir | point-bound | passes | assemble s | solve s | "
                     "us/point | ns/mult (assemble) | ns/mult (solve) |\n"
                     "|---|---|---|---|---|---|---|---|---|---|\n")
            for c in cal["runs"]:
                fh.write(f"| {c['label']} | {c['field']} | {c['direction']} | "
                         f"{c['point_bound']} | {c['passes']} | "
                         f"{c['assemble_s']} | {c['solve_s']} | "
                         f"{c['us_per_point']} | {c['ns_per_mult_assemble']} | "
                         f"{c['ns_per_mult_solve']} |\n")
    for r in recs:
        print(f"{r['label']:<46} W {r['width']:>7}  total 2^{r['total_log2']}"
              f"  ({r['system_source']})")


if __name__ == "__main__":
    main()
