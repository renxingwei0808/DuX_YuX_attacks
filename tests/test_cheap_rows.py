"""E11 / W15-B -- the cheap-coordinate elimination lemma (O10), cell by cell.

`tools/cheap_rows.py` computes dim K = dim {y : y L[balanced rows, expensive
columns] = 0} exactly (over F_2 in characteristic 2, over F_p otherwise).  The
tables below are the ones of the cheap-coordinate lemma (O10, see
docs/glossary.md) in the paper and its supplement, plus the two cells where this
run DISAGREES with them (both in the DuX/CPA row and both harmless -- see
results/E11_cheap_rows/cheap_rows_table.md).
"""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

from cheap_rows import cheap_rows, describe, layer_row, layer_matrix  # noqa: E402

BIN_FIELDS = ["2^4", "2^8", "2^16"]
FP_FIELDS = ["193", "257", "65537"]


def dim(cipher, field, direction, pattern):
    return describe(cipher, field, direction, pattern)["dim_K"]


# --- DuX, decryption direction (cheap column = position 2) -----------------
@pytest.mark.parametrize("field", BIN_FIELDS + FP_FIELDS)
@pytest.mark.parametrize("pattern,want", [("0001", 1), ("1001", 1), ("1101", 1),
                                          ("1110", 0), ("1100", 0), ("0111", 4)])
def test_dux_dec_dim_K(field, pattern, want):
    assert dim("dux", field, "dec", pattern) == want


def test_dux_dec_0001_kernel_vector():
    """O10: the kernel vector is the alternating sum of the four position-3
    rows; over F_p its cheap coefficients are proportional to (1,-1,1,-1) on
    the words {2,6,10,14}, over F_2 they are all ones."""
    for field in FP_FIELDS:
        r = describe("dux", field, "dec", "0001")
        assert r["rows"] == [3, 7, 11, 15]
        assert r["kernel"][0]["y"] == [1, -1, 1, -1]
        assert [r["kernel"][0]["cheap_coeffs"][str(j)] for j in (2, 6, 10, 14)] \
            == [1, -1, 1, -1]
    for field in BIN_FIELDS:
        r = describe("dux", field, "dec", "0001")
        assert r["kernel"][0]["y"] == [1, 1, 1, 1]
        assert [r["kernel"][0]["cheap_coeffs"][str(j)] for j in (2, 6, 10, 14)] \
            == [1, 1, 1, 1]


def test_dux_dec_0001_row_uses_the_paper_M0():
    """Cross-check against the M0 first row printed in the DuX paper."""
    from dux.params import M0_ROW_65537
    row, F = layer_row("dux", "65537", "dec")
    assert row == M0_ROW_65537
    L = layer_matrix(row)
    y = [1, -1, 1, -1]
    R = [3, 7, 11, 15]
    for j in range(16):
        v = sum(y[t] * L[R[t]][j] for t in range(4)) % F.p
        assert (v == 0) == (j % 4 != 2), f"word {j} should be {'zero' if j % 4 != 2 else 'nonzero'}"


def test_dux_L1_control_keeps_the_equation_set():
    """O2: with L1 the kernel vector is the same and its cheap coefficients
    flip sign, so the equation set is unchanged (analyses fix L0)."""
    a = describe("dux", "65537", "dec", "0001", t=0)["kernel"][0]
    b = describe("dux", "65537", "dec", "0001", t=1)["kernel"][0]
    assert a["y"] == b["y"]
    assert [b["cheap_coeffs"][k] for k in b["cheap_coeffs"]] == \
           [-v for v in (a["cheap_coeffs"][k] for k in a["cheap_coeffs"])]


# --- YuX, decryption direction (cheap column = position 3) -----------------
@pytest.mark.parametrize("field", BIN_FIELDS + FP_FIELDS)
@pytest.mark.parametrize("pattern,want", [("1110", 3), ("1100", 0), ("1101", 0),
                                          ("1011", 0), ("0111", 0)])
def test_yux_dec_dim_K(field, pattern, want):
    assert dim("yux", field, "dec", pattern) == want


def test_yux_dec_1110_equations_are_word_pairs():
    """O10: the three equations pair position-3 words of two blocks each,
    with +-1 coefficients over F_p and 1 over F_2."""
    for field in ("65537", "2^16"):
        r = describe("yux", field, "dec", "1110")
        assert r["dim_K"] == 3
        for e in r["kernel"]:
            nz = {int(j): v for j, v in e["cheap_coeffs"].items() if v}
            assert len(nz) == 2 and set(nz) <= {3, 7, 11, 15}
            assert all(abs(v) == 1 for v in nz.values())
        pairs = {frozenset(int(j) for j, v in e["cheap_coeffs"].items() if v)
                 for e in r["kernel"]}
        assert len(pairs) == 3 and all(7 in p for p in pairs)


# --- CPA direction ---------------------------------------------------------
@pytest.mark.parametrize("field", BIN_FIELDS + FP_FIELDS)
@pytest.mark.parametrize("pattern,want", [("0001", 4), ("1110", 4), ("1111", 8)])
def test_dux_cpa_dim_K_field_independent_cells(field, pattern, want):
    assert dim("dux", field, "enc", pattern) == want


def test_dux_cpa_0001_rows_are_individually_cheap():
    """O10: L^{-1}'s position-3 rows only touch words of positions {3,0}, both
    cheap for S^{-1} (f and g), so each of rows 3,7,11,15 is usable alone."""
    for field in ("65537", "2^16"):
        r = describe("dux", field, "enc", "0001")
        assert r["rows"] == [3, 7, 11, 15] and r["dim_K"] == 4
        assert [e["y"] for e in r["kernel"]] == \
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


@pytest.mark.parametrize("field", BIN_FIELDS + FP_FIELDS)
@pytest.mark.parametrize("pattern,want", [("0111", 4), ("1110", 4), ("1111", 8),
                                          ("1100", 4), ("1101", 4), ("0011", 0)])
def test_yux_cpa_dim_K(field, pattern, want):
    assert dim("yux", field, "enc", pattern) == want


@pytest.mark.parametrize("pattern,bin_dim,fp_dim", [("0110", 2, 1), ("0111", 6, 5)])
def test_dux_cpa_two_cells_are_field_dependent(pattern, bin_dim, fp_dim):
    """DISAGREEMENT with O10 / memo Sect. 3.2, which state the whole table is
    field independent.  It is, except for these two DuX/CPA cells: L^{-1} is a
    0/1 matrix, so its F_2 rank can be strictly smaller than its F_p rank and
    the left kernel correspondingly larger.  Neither pattern is used by any
    attack in this repository (the CPA line uses `0001` and `1110`, which do
    agree), so this is a bookkeeping correction, not a result change."""
    for field in BIN_FIELDS:
        assert dim("dux", field, "enc", pattern) == bin_dim
    for field in FP_FIELDS:
        assert dim("dux", field, "enc", pattern) == fp_dim


def test_every_other_cell_is_field_independent():
    from cheap_rows import PATTERNS
    exceptions = {("dux", "enc", "0110"), ("dux", "enc", "0111")}
    for cipher in ("dux", "yux"):
        for direction in ("dec", "enc"):
            for pat in PATTERNS:
                d2 = {dim(cipher, f, direction, pat) for f in BIN_FIELDS}
                dp = {dim(cipher, f, direction, pat) for f in FP_FIELDS}
                assert len(d2) == 1 and len(dp) == 1, "dim K depends on n or p"
                if (cipher, direction, pat) not in exceptions:
                    assert d2 == dp, (cipher, direction, pat)


# --- the lemma itself ------------------------------------------------------
@pytest.mark.parametrize("cipher", ["dux", "yux"])
@pytest.mark.parametrize("direction", ["dec", "enc"])
@pytest.mark.parametrize("field", ["2^16", "65537"])
def test_kernel_vectors_kill_every_expensive_column(cipher, direction, field):
    from cheap_rows import PATTERNS
    from cipher_degree import CIPHERS
    cheap = CIPHERS[cipher]["cheap_dec" if direction == "dec" else "cheap_enc"]
    row, F = layer_row(cipher, field, direction)
    mod = 2 if F.char == 2 else F.p
    L = layer_matrix(row)
    for pat in PATTERNS:
        R = [i for i in range(16) if i % 4 in [p for p in range(4) if pat[p] == "1"]]
        for y, yL in cheap_rows(cipher, field, direction, pat):
            assert len(y) == len(R)
            for j in range(16):
                v = sum(y[t] * L[R[t]][j] for t in range(len(R))) % mod
                assert v == yL[j] % mod
                if j % 4 not in cheap:
                    assert v == 0
            assert any(v % mod for v in yL), "the zero vector is not an equation"


# ---------------------------------------------------------------------------
# W19-B: why the dim K = 1 combined row collapses in characteristic 2.
#
# `experiments/E11_cheap_rows/char2_collapse.py` proves and measures it; these
# tests pin the two structural facts the lemma rests on:
#
#   (1) Phi_j = pi^j . Phi_0 . rho^j  -- outer block j's template is block 0's
#       with the inner blocks and the outer key slot rotated by j;
#   (2) over F_2 the group algebra of the rotation is LOCAL, so a kernel vector
#       y is tau^v times a unit (tau = 1 + pi) and the combined row restricted
#       to the rho-invariant moments has rank(tau^v).  DuX's `0001` kernel
#       vector (1,1,1,1) has v = 3, the largest possible: one coefficient per
#       orbit, i.e. ORBIT SUMS only, so the four key words of an orbit are
#       indistinguishable.  YuX's `1110` vectors have v in {1,2} and keep 77 %.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def collapse_mod():
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E11_cheap_rows"))
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
    sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round",
                                    "fast"))
    import char2_collapse
    return char2_collapse


@pytest.mark.parametrize("y,v", [((1, 1, 1, 1), 3),        # DuX `0001`
                                 ((1, 1, 0, 0), 1),        # YuX `1110`
                                 ((0, 1, 0, 1), 2),
                                 ((0, 1, 1, 0), 1),
                                 ((1, 0, 0, 0), 0)])
def test_tau_valuation_of_the_kernel_vectors(collapse_mod, y, v):
    assert collapse_mod.tau_valuation(list(y)) == v


def test_tau_rank_from_the_orbit_histogram(collapse_mod):
    """rank(tau^e) = sum over orbits of max(0, L - e)."""
    sizes = {1: 1, 2: 50, 4: 4481}          # toy-2^4's 18 025 columns
    assert collapse_mod.tau_rank(sizes, 1) == 50 + 3 * 4481
    assert collapse_mod.tau_rank(sizes, 2) == 2 * 4481
    assert collapse_mod.tau_rank(sizes, 3) == 4481      # one per orbit: the sums


@pytest.mark.parametrize("instance,rounds,cipher,combine",
                         [("toy-2^4", 6, "dux", "0001"),
                          ("toy-257", 7, "dux", "0001")])
def test_outer_block_templates_are_rotations_of_each_other(
        collapse_mod, instance, rounds, cipher, combine):
    """(1): Phi_j m = pi^j (Phi_0 (rho^j m)) for random moment vectors."""
    import numpy as np
    from dux.registry import get_cipher
    from assemble_fast import Precomp
    from attack_2round_toy import linear_row
    import rank_phi as RP

    c = get_cipher(instance, rounds=rounds)
    F = c.F
    pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher=cipher)
    lrow = linear_row(c, 0)
    perms = {k: collapse_mod.column_rotation(pre, k) for k in (1, 2, 3)}
    rng = np.random.default_rng(2026)
    mom = RP.realizable_moments(pre, rng, 32)
    rows = collapse_mod._rows(pre, mom, lrow, F.char == 2)
    mod = F.q if F.char == 2 else F.p
    for k in (1, 2, 3):
        rot = collapse_mod._rows(pre, collapse_mod.rotate_moments(mom, k),
                                 lrow, F.char == 2)[0]
        got = collapse_mod.apply_perm(perms[k], rot) % mod
        assert np.array_equal(got, rows[k] % mod)


def test_char2_all_ones_row_is_an_orbit_sum_vector(collapse_mod):
    """(2): on the rho-invariant part of the moment space the DuX `0001`
    combined row is pi-INVARIANT (constant on every orbit), while over F_p the
    same construction is pi-ANTI-invariant -- the sign that saves F_p."""
    import numpy as np
    from dux.registry import get_cipher
    from assemble_fast import Precomp, combine_rows
    from attack_2round_toy import linear_row
    from cheap_rows import block_coeffs
    import rank_phi as RP

    for instance, rounds, want_sign in (("toy-2^4", 6, +1), ("toy-257", 7, -1)):
        c = get_cipher(instance, rounds=rounds)
        F = c.F
        pre = Precomp(F, c.alpha, [0, 1, 2, 3], [0, 1, 2, 3], cipher="dux")
        lrow = linear_row(c, 0)
        fkey = f"2^{F.n}" if F.char == 2 else str(F.q)
        y = block_coeffs("dux", fkey, "dec", "0001")[0]
        mod = F.q if F.char == 2 else F.p
        rng = np.random.default_rng(11)
        mom = RP.realizable_moments(pre, rng, 32)
        # sigma^3 m = sum_j rho^j m -- the rho-invariant part
        pm, Mom = {}, {}
        for x in mom.pm:
            acc = mom.pm[x].copy()
            for j in (1, 2, 3):
                r = collapse_mod.rotate_moments(mom, j).pm[x]
                acc = (acc ^ r) if F.char == 2 else (acc + r) % mod
            pm[x] = acc
        for x in mom.Mom:
            acc = mom.Mom[x].copy()
            for j in (1, 2, 3):
                r = collapse_mod.rotate_moments(mom, j).Mom[x]
                acc = (acc ^ r) if F.char == 2 else (acc + r) % mod
            Mom[x] = acc
        sym = RP.PseudoMoments(pm, Mom)
        rows = collapse_mod._rows(pre, sym, lrow, F.char == 2)
        v = np.asarray(combine_rows(rows, y, F), dtype=np.int64) % mod
        perm = collapse_mod.column_rotation(pre, 1)
        rot = collapse_mod.apply_perm(perm, v) % mod
        want = v if want_sign > 0 else (-v) % mod
        assert np.array_equal(rot, want), instance
        if want_sign > 0:
            # pi-invariant == constant on orbits == the 16 degree-1 key columns
            # enter only through four orbit sums.
            d1 = collapse_mod.degree_one_columns(pre)
            for cpos in range(4):
                vals = {int(v[d1[str(4 * b + cpos)]]) for b in range(4)}
                assert len(vals) == 1
