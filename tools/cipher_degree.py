"""R5 / W15-B -- the per-cipher data the max-plus degree machinery needs.

`tools/zero_sum_criterion.py` and `tools/maxplus.py` used to hard-code DuX's
S-box degree recurrence and rotation sets.  From R5 on the same criterion is
applied to YuX as well, so both live here and the two tools take a `--cipher`
switch.  DuX's entries are byte-for-byte the ones the tools used before, so
every existing prediction (and the 39-cell `--grid`) is unchanged.

Formal total degree ("max-plus") of one S-box layer.  Each coordinate of the
closed form is a sum of products of earlier coordinates, so its degree bound is
the max over its terms of the sum of the factors' bounds:

  DuX (dux/sbox.py):  S^{-1} = R^4                    S = (R^{-1})^4
      f  = x0 x1 + x3 + a       -> position 3         a  = x2 - x0 x3 - a  (pos 2)
      g  = x1 x2 + x0 + a       -> position 0         b  = x1 - x3 a  - a  (pos 1)
      y1 = x2 f  + x1 + a       -> position 1         c  = x0 - a b   - a  (pos 0)
      y2 = f  g  + x2 + a       -> position 2         y3 = x3 - b c   - a  (pos 3)
      degrees (2,3,4,2)                               degrees (5,3,2,8)

  YuX (yux/sbox.py):  S^{-1} = Pf^4                   S = Pf^{-4}
      y0 = x0 + x1 x2 + x3 + a  (pos 0)               z0 = x3 - x0x1 - x2 - a (pos 3)
      y1 = x1 + x2 x3 + y0 + a  (pos 1)               z1 = x2 - z0x0 - x1 - a (pos 2)
      y2 = x2 + x3 y0 + y1 + a  (pos 2)               z2 = x1 - z1z0 - x0 - a (pos 1)
      y3 = x3 + y0 y1 + y2 + a  (pos 3)               z3 = x0 - z2z1 - z0 - a (pos 0)
      degrees (2,2,3,4)                               degrees (8,5,3,2)

Rotation sets, convention (X <<< j)_i = X_{(i+j) mod 16}:
  DuX  decryption {1,4,8,9,13} (L0; L1 = {1,5,8,12,13} only permutes blocks),
       encryption {0,1,3,4,7,8,9,10,12,14,15} over F_{2^n}, dense over F_p.
  YuX  decryption {0,3,4,8,9,12,14} (both fields),
       encryption {1,2,3,5,6,7,8,12,13,14,15} over F_{2^n}, dense over F_p.
DuX's decryption offsets cover only the residues {0,1} mod 4 (O3: position p
receives positions p and p+1), YuX's cover all four -- which is why YuX grows
at 4 per layer and DuX at 2 + sqrt(3) ~ 3.73.

FULL-BLOCK structures (O11).  When a whole 4-word block of the ciphertext runs
over F_q^4, S^{-1} is a bijection of F_q^4 and the substitution y = S^{-1}(x+k)
turns that block into four free variables: the first S layer is FREE and the
recurrence starts at degrees (1,1,1,1) for that block, with threshold computed
from s = 4 * (#free blocks) + (#other active words).  The same holds in the
encryption direction with S in place of S^{-1}.  Only a WHOLE block works: the
S-box maps a lower-dimensional affine subspace to a non-affine variety.
"""
from __future__ import annotations


# ------------------------------------------------------------------ DuX ----
def dux_sinv_deg(e):
    """DuX S^{-1} = R^4 -- degree bounds of (y0, y1, y2, y3)."""
    e0, e1, e2, e3 = e
    d0 = max(e1 + e2, e0)
    d3 = max(e0 + e1, e3)
    d1 = max(e2 + d3, e1)
    d2 = max(d3 + d0, e2)
    return [d0, d1, d2, d3]


def dux_s_deg(e):
    """DuX S = (R^{-1})^4 -- output (c, b, a, y3) at positions 0..3."""
    e0, e1, e2, e3 = e
    da = max(e2, e0 + e3)
    db = max(e1, e3 + da)
    dc = max(e0, db + da)
    dy3 = max(e3, dc + db)
    return [dc, db, da, dy3]


# ------------------------------------------------------------------ YuX ----
def yux_sinv_deg(e):
    """YuX S^{-1} = Pf^4 -- degree bounds of (y0, y1, y2, y3)."""
    e0, e1, e2, e3 = e
    d0 = max(e0, e1 + e2, e3)
    d1 = max(e1, e2 + e3, d0)
    d2 = max(e2, e3 + d0, d1)
    d3 = max(e3, d0 + d1, d2)
    return [d0, d1, d2, d3]


def yux_s_deg(e):
    """YuX S = Pf^{-4} -- output (z3, z2, z1, z0) at positions 0..3."""
    e0, e1, e2, e3 = e
    z0 = max(e3, e0 + e1, e2)
    z1 = max(e2, z0 + e0, e1)
    z2 = max(e1, z1 + z0, e0)
    z3 = max(e0, z2 + z1, z0)
    return [z3, z2, z1, z0]


CIPHERS = {
    "dux": {
        "sinv_deg": dux_sinv_deg,
        "s_deg": dux_s_deg,
        "ROT_INV": (1, 4, 8, 9, 13),
        "ROT_INV_L1": (1, 5, 8, 12, 13),
        "ROT_FWD_BIN": (0, 1, 3, 4, 7, 8, 9, 10, 12, 14, 15),
        "ROT_FWD_FP": tuple(range(16)),
        "sinv_degrees": (2, 3, 4, 2),
        "s_degrees": (5, 3, 2, 8),
        "cheap_dec": (2,),          # cheap coordinate of the ENCRYPTION S-box
        "cheap_enc": (0, 3),        # cheap coordinates of S^{-1} (g, f)
    },
    "yux": {
        "sinv_deg": yux_sinv_deg,
        "s_deg": yux_s_deg,
        "ROT_INV": (0, 3, 4, 8, 9, 12, 14),
        "ROT_INV_L1": (0, 3, 4, 8, 9, 12, 14),
        "ROT_FWD_BIN": (1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15),
        "ROT_FWD_FP": tuple(range(16)),
        "sinv_degrees": (2, 2, 3, 4),
        "s_degrees": (8, 5, 3, 2),
        "cheap_dec": (3,),          # z0 of the ENCRYPTION S-box
        "cheap_enc": (0, 1),        # y0 and y1 - y0 of S^{-1}
    },
}


def rotations(cipher, direction, char2):
    c = CIPHERS[cipher]
    if direction == "dec":
        return c["ROT_INV"]
    return c["ROT_FWD_BIN"] if char2 else c["ROT_FWD_FP"]


def sbox_deg(cipher, direction):
    c = CIPHERS[cipher]
    return c["sinv_deg"] if direction == "dec" else c["s_deg"]


def profile(active, layers, direction="dec", char2=False, cipher="dux",
            free_blocks=()):
    """Per-layer, per-word max-plus total-degree bounds AFTER the S layer.

    `active` are the active ciphertext (resp. plaintext) words that go through
    the first S layer normally; `free_blocks` are whole blocks that run over
    F_q^4 and therefore pass the first S layer for free (O11), entering the
    recurrence with degrees (1,1,1,1)."""
    free_blocks = tuple(sorted(set(free_blocks)))
    sbox = sbox_deg(cipher, direction)
    rots = rotations(cipher, direction, char2)
    D = [0] * 16
    for a in active:
        assert a // 4 not in free_blocks, \
            f"word {a} lies in free block {a // 4}; list it as a free block only"
        D[a] = 1
    out = []
    for layer in range(layers):
        Y = []
        for b in range(4):
            if layer == 0 and b in free_blocks:
                Y += [1, 1, 1, 1]           # the substitution y = S^{-1}(x + k)
            else:
                Y += sbox(D[4 * b:4 * b + 4])
        out.append(Y)
        D = [max(Y[(i + j) % 16] for j in rots) for i in range(16)]
    return out


def structure_size(active, free_blocks=()):
    """Number of free field variables s of the structure."""
    return len(active) + 4 * len(set(free_blocks))

# ------------------------------------------------ known-key partitions ----
def profile_from(D0, layers, direction="dec", char2=False, cipher="dux",
                 leading_linear=False):
    """Max-plus profile starting from an ARBITRARY degree vector `D0` on the
    sixteen words (R7 / W23).  Each layer is one S-box layer followed, between
    layers, by the linear layer of `direction`; with `leading_linear=True` the
    linear layer is applied BEFORE the first S-box layer as well.

    This is what the known-key zero-sum partitions of Liu and Sun need: the
    intermediate state sits immediately after a diffusion layer, so every
    backward group is (L^{-1}, S^{-1}) -- a leading L^{-1} -- while every
    forward group is (S, L).  `profile` is the special case D0 = indicator of
    the active words, leading_linear=False."""
    sbox = sbox_deg(cipher, direction)
    rots = rotations(cipher, direction, char2)
    D = list(D0)
    if leading_linear:
        D = [max(D[(i + j) % 16] for j in rots) for i in range(16)]
    out = []
    for _ in range(layers):
        Y = []
        for b in range(4):
            Y += sbox(D[4 * b:4 * b + 4])
        out.append(Y)
        D = [max(Y[(i + j) % 16] for j in rots) for i in range(16)]
    return out
