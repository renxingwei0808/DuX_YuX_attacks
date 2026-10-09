"""DuX parameter sets (Wu et al., DCC 2026, Table 2) plus toy instances.

Official instances (block = 16 words of F_q, 12 rounds, alpha = 179):
  DuX(2^8)   : F_{2^8},  x^8 + x^4 + x^3 + x + 1     (0x11B)
  DuX(2^16)  : F_{2^16}, x^16 + x^12 + x^3 + x + 1   (0x1100B)
  DuX(65537) : F_65537

Toy instances (NOT in the paper; same round function, smaller field) are
used only to validate attack pipelines cheaply before scaling up:
  toy-2^4 : F_{2^4}, x^4 + x + 1 (0x13)
  toy-193 : F_193  (193 - 1 = 2^6 * 3: large 2-part, small field)
  toy-257 : F_257  (257 = 2^8 + 1, Fermat prime like 65537)
  toy-3   : F_3    (O14 only: the CPA exact-boundary condition 8^{l-1} =
                    4(p - 1) has the three solutions p = 3 / layer 2,
                    p = 17 / layer 3 and p = 65537 / layer 7, and 17 is the
                    one prime of the three where DuX's circulant is singular,
                    so F_3 is the only field in which the DuX half of O14 can
                    be checked against the real cipher by brute force)
(p = 5, 13, 17 are NOT usable for DuX: both DuX circulants L0^{-1} and
L1^{-1} are singular mod each of them -- verified in tests/test_dux.py.  The
FIELDS entries "5" and "17" below exist only because YuX's circulant IS
invertible mod 5 and mod 17, which yux/params.py uses for the O14 toys
yuxtoy-5 / yuxtoy-17; no DuX instance references them.)

SPECIFICATION SOURCE: the paper only (Wu et al., DCC 2026, 94:140).  The
authors' GitHub code is known to differ from the paper and is NOT used as a
reference anywhere in this repository (docs/dux_specification.md only records
the differences).

Interpretive choices where the paper is ambiguous (all follow the paper's
definitional statements; see docs/dux_specification.md):
  * alpha = 179 for all three instances (Table 2 lists it once).
  * t_xor is computed from round-key word 15 (Algorithm 3, line 5).
  * Round constants use the ENCRYPTION S-box S (Algorithm 1 writes S).
  * S-box closed forms are derived from S^{-1} = R^4 (Sect. 3.1.1), since the
    explicit formulas printed in Sect. 3.1.1 / 5.3 / 5.5.1 contradict each other.
  * Word rotation (X <<< j)_i = X_{(i+j) mod 16}; fixed by matching the paper's
    explicit M0 for p = 65537 (tests/test_dux.py::test_paper_M0_row_p65537).
"""

FIELDS = {
    "2^8":   {"char": 2, "n": 8,  "poly": 0x11B,   "q": 1 << 8},
    "2^16":  {"char": 2, "n": 16, "poly": 0x1100B, "q": 1 << 16},
    "65537": {"char": 65537, "q": 65537},
    # toys
    "2^4":   {"char": 2, "n": 4,  "poly": 0x13,    "q": 1 << 4},
    "193":   {"char": 193, "q": 193},
    "257":   {"char": 257, "q": 257},
    # O14 toys (see the module docstring): "5" and "17" are used by YuX only,
    # "3" by DuX only -- no official instance references any of them.
    "3":     {"char": 3,   "q": 3},
    "5":     {"char": 5,   "q": 5},
    "17":    {"char": 17,  "q": 17},
}

# 'official' means: parameter set defined in the paper (True) vs toy (False).
INSTANCES = {
    "dux-2^8":   {"field": "2^8",   "alpha": 179, "rounds": 12, "official": True},
    "dux-2^16":  {"field": "2^16",  "alpha": 179, "rounds": 12, "official": True},
    "dux-65537": {"field": "65537", "alpha": 179, "rounds": 12, "official": True},
    "toy-2^4":   {"field": "2^4",   "alpha": 5,   "rounds": 12, "official": False},
    "toy-193":   {"field": "193",   "alpha": 179, "rounds": 12, "official": False},
    "toy-257":   {"field": "257",   "alpha": 179, "rounds": 12, "official": False},
    "toy-3":     {"field": "3",     "alpha": 179 % 3, "rounds": 12, "official": False},
}

WORDS = 16          # state words
BLOCKS = 4          # S-boxes per layer
BLOCK_WORDS = 4     # words per S-box

# Linear layer as word rotations: (X <<< j)_i = X_{(i+j) mod 16}.
# Decryption direction (paper Sect. 3.2.3), valid over F_{2^n} and F_p:
ROT_INV = {0: (1, 4, 8, 9, 13), 1: (1, 5, 8, 12, 13)}
# Encryption direction over F_{2^n} (paper Sect. 3.2.1).  Over F_p the
# forward matrix is the circulant inverse of ROT_INV (paper Sect. 3.2.2);
# we compute it generically and check it against the paper's M0 in tests.
ROT_FWD_BIN = {0: (0, 1, 3, 4, 7, 8, 9, 10, 12, 14, 15),
               1: (0, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15)}

# Paper's explicit M0 first row for p = 65537 (Sect. 3.2.2)
M0_ROW_65537 = (23249, 3677, 25325, 6346, 19394, 26808, 17615, 52608,
                3974, 53794, 52311, 41042, 7829, 30663, 60021, 60318)

# Word whose two lowest bits select L0/L1 (Algorithm 3, line 5 uses rk^i_15)
TXOR_WORD = 15
