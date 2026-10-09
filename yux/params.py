"""YuX parameter sets (Liu et al., IEEE TIT 70(5), 2024, Table II) plus toys.

Official instances (block = 16 words of F_q, 4 blocks of 4, alpha = 205):
  Yu2X-8      : F_{2^8},  x^8 + x^4 + x^3 + x + 1     (0x11B),  12 rounds
  Yu2X-16     : F_{2^16}, x^16 + x^12 + x^3 + x + 1   (0x1100B), 12 rounds
                (recommended for FHE) / 14 rounds (128-bit claim)
  YupX-65537  : F_65537,                                9 rounds
                (recommended for FHE, 118-bit claim) / 14 rounds (128-bit)

Table II lists two round numbers for Yu2X-16 and for YupX-65537; the starred
one is "the recommended number of rounds in fully homomorphic application
scenarios".  We keep BOTH: `rounds` is the default the reference model uses and
the other one is recorded next to it (see docs/yux_specification.md, clause
S6.*).  `rounds` = 12 for Yu2X-16 (the FHE instance, which is also the number
Ni et al. attack) and 14 for YupX-65537 (the 128-bit instance), with
`rounds_128` / `rounds_fhe` naming the other one.

Toy instances (NOT in the paper; same round function, smaller field) exist only
to validate attack pipelines cheaply, exactly as dux/params.py's toys do:
  yuxtoy-2^4 : F_{2^4}, x^4 + x + 1 (0x13), alpha = 5
  yuxtoy-193 : F_193,  alpha = 205 mod 193 = 12
  yuxtoy-257 : F_257,  alpha = 205
  yuxtoy-5   : F_5,    alpha = 205 mod 5  = 0   (O14: exact-hit layer 3)
  yuxtoy-17  : F_17,   alpha = 205 mod 17 = 1   (O14: exact-hit layer 4)
(For F_p toys the circulant M = sum of rotations must be invertible mod p;
5, 17, 193, 257 and 65537 are -- unlike DuX's two circulants, which are
singular mod 5, 13 and 17; see `assert_invertible` in yux/linear.py.
yuxtoy-5 / yuxtoy-17 exist only for O14, where the smallest primes with
p - 1 = 4^{l-2} are the ones whose full-block sum can be taken by brute
force; alpha = 0 for p = 5 is a degenerate-looking but legitimate value --
Pf is a permutation for every alpha, and O14 is alpha-independent anyway.)

SPECIFICATION SOURCE: the YuX paper only (Liu et al., IEEE TIT 70(5), 2024) --
Sect. III (Construction 1, Fig. 1), Sect. IV-A/B (Cipher Description,
Algorithms 1-3, Table II, footnote 2).  Ni et al. (DCC 2026, 94:123) is a
CROSS-CHECK, not a specification source; the authors' public code is not used.

Interpretive choices where the paper is ambiguous (all recorded with their
consequences in docs/yux_specification.md, ambiguity table):
  * Algorithm 1 line 2 reads X_i = (key_i, key_{i+1}, key_{i+2}, key_{i+3}),
    i in {0..3} -- four SLIDING windows using only key_0..key_6.  Sect. VI-D
    writes RK_{0,i} = [k_{4i}, ..., k_{4i+3}], i.e. NON-OVERLAPPING blocks.
    We implement the non-overlapping reading (`ks_literal=False`, the default)
    and provide the literal one as an option.
  * Round constants (Algorithm 2) feed the integers id+1..id+4 into S; the
    integer -> field element map is not stated.  We use `F.from_int`
    (identity on [0, q), the same choice dux/keyschedule.py makes).
  * alpha = 205 is listed once per instance in Table II and is used for the
    S-box, the key schedule and the round constants alike.
  * "<<< 3" inside the key schedule is a rotation of the FOUR WORDS of a block,
    with the same convention as the state rotation (footnote 2).
"""
from dux.params import FIELDS  # noqa: F401  (re-exported: yux reuses dux's fields)

WORDS = 16          # state words
BLOCKS = 4          # S-boxes per layer
BLOCK_WORDS = 4     # words per S-box

# 'official' means: parameter set defined in the paper (True) vs toy (False).
INSTANCES = {
    "yu2x-8":      {"field": "2^8",   "alpha": 205, "rounds": 12,
                    "official": True,  "security": "128 bit"},
    "yu2x-16":     {"field": "2^16",  "alpha": 205, "rounds": 12,
                    "rounds_128": 14, "official": True,
                    "security": "128 bit", "rounds_note": "12* (FHE) / 14"},
    "yupx-65537":  {"field": "65537", "alpha": 205, "rounds": 14,
                    "rounds_fhe": 9, "official": True,
                    "security": "118 bit* (9 rounds) / 128 bit (14 rounds)",
                    "rounds_note": "9* (FHE) / 14"},
    "yuxtoy-2^4":  {"field": "2^4",   "alpha": 5,           "rounds": 12, "official": False},
    "yuxtoy-5":    {"field": "5",     "alpha": 205 % 5,     "rounds": 12, "official": False},
    "yuxtoy-17":   {"field": "17",    "alpha": 205 % 17,    "rounds": 12, "official": False},
    "yuxtoy-193":  {"field": "193",   "alpha": 205 % 193,   "rounds": 12, "official": False},
    "yuxtoy-257":  {"field": "257",   "alpha": 205 % 257,   "rounds": 12, "official": False},
}

# Linear layer as word rotations, convention (X <<< j)_i = X_{(i+j) mod 16}
# (paper footnote 2: (x_0,...,x_15) <<< 1 = (x_1,...,x_15,x_0)).
# DECRYPTION direction (Sect. IV-A.3), identical over F_{2^n} and F_p:
#     LP_inv(x) = sum_{j in ROT_INV} (x <<< j)
ROT_INV = (0, 3, 4, 8, 9, 12, 14)
# ENCRYPTION direction over F_{2^n} (Sect. IV-B.1): the XOR of these rotations.
# It is the circulant inverse of ROT_INV over F_2; yux/linear.py computes the
# inverse generically and tests/test_yux.py locks it to this set.
ROT_FWD_BIN = (1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15)
# ENCRYPTION direction over F_p (Sect. IV-B.2): the circulant with first row
# v_p; LP(x)_i = sum_j v_p[j] x_{(i+j) mod 16}.  Printed as fractions and, for
# p = 65537, as explicit integers.  Both are locked in tests/test_yux.py.
V_P_FRACTIONS = ((4, 7), (5, 21), (5, 21), (5, 21), (4, 7), (5, 21), (5, 21),
                 (5, 21), (-3, 7), (-16, 21), (-16, 21), (-16, 21), (-3, 7),
                 (5, 21), (5, 21), (5, 21))
V_P_65537 = (9363, 53054, 53054, 53054, 9363, 53054, 53054, 53054,
             9362, 53053, 53053, 53053, 9362, 53054, 53054, 53054)
