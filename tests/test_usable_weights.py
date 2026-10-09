"""W19-C -- `weighted.plan` returns the weights a caller may ACTUALLY use.

Until R6, `plan` returned only the raw degree budget `margin = T - max_used D`
and every caller had to remember the four corrections by hand (S12 Sect. 7.5,
Y06 Sect. 7 item 5).  Two of them had already been written down as footnotes in
the summaries after the fact:

  * a FULL-BLOCK structure (O11) can only be weighted on a ciphertext word, and
    after the substitution y = S^{-1}(x + k) that word has degree 2 in y, so
    every power eats two degree units -- `usable_weights = margin // 2`;
  * a single COSET needs a weight that is not a multiple of 2^k (O12 Sect. 5.5),
    so a = 0 drops out;
  * a SUBSPACE word's budget is (2^m - 1) - D, which is what the criterion's
    threshold already gives -- the rule string records it;
  * a CPA row sitting on an O9 cell (formal degree exactly q) is balanced only
    because c_{q-1} vanishes, which says nothing about c_{q-1-a}: a = 0 only.

The numbers below are the ones the R5 write-ups computed by hand, so this file
also pins them: 126 (Yu2X-8 8 rounds, Y06 Sect. 4), 28 (DuX(65537) 13 rounds,
ledger G.1), 32 768 / 32 766 (the two 12-round YuX accountings, Y06 Sect. 5).
"""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "E07_key_recovery_2round"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E08_boolean_degree_extension"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dux.field import make_field                                 # noqa: E402
import weighted as WT                                            # noqa: E402


# ---------------------------------------------------------- full blocks ----
def test_full_block_halves_the_margin_yu2x8_layer6():
    """Y06 Sect. 4: Yu2X-8, full block 0 (2^32 points), layer 6 `1110`, three
    O10 rows.  `plan` used to return 252 and the operator had to halve it."""
    pl = WT.plan(make_field("2^8"), [], 6, "dec", "yux", "1110",
                 free_blocks=(0,))
    assert pl.margin == 252
    assert pl.usable_weights == 126
    assert pl.usable_norm == 126
    assert pl.weights[-1] == (125,)
    assert "full block" in pl.weight_rule


@pytest.mark.parametrize("q,cipher,combine,layers,margin,usable", [
    ("2^8", "yux", "1110", 6, 252, 126),        # Y06 Sect. 4  (8 rounds)
    ("2^16", "yux", "1110", 10, 65532, 32766),  # Y06 Sect. 5  (Yu2X-16, 12 rounds)
    ("65537", "yux", "1110", 10, 65536, 32768),  # Y06 Sect. 5  (YupX, 12 rounds)
    ("65537", "dux", "0001", 11, 57, 28),       # ledger G.1   (DuX 13 rounds)
    ("2^16", "dux", "0001", 11, 53, 26),        # the same route over F_{2^16}
])
def test_full_block_usable_weights_match_the_hand_accountings(
        q, cipher, combine, layers, margin, usable):
    pl = WT.plan(make_field(q), [], layers, "dec", cipher, combine,
                 free_blocks=(0,))
    assert (pl.margin, pl.usable_weights) == (margin, usable)


def test_full_block_weight_word_is_the_cheap_ciphertext_coordinate():
    """The weights have to live on the block's own cheap coordinate: DuX
    position 2 (a), YuX position 3 (z0) -- the only ones of degree 2 in y."""
    for b in range(4):
        pl = WT.plan(make_field("2^8"), [], 6, "dec", "yux", "1110",
                     free_blocks=(b,))
        assert pl.weight_word == 4 * b + 3
        pl = WT.plan(make_field("65537"), [], 11, "dec", "dux", "0001",
                     free_blocks=(b,))
        assert pl.weight_word == 4 * b + 2


# ------------------------------------------------------- the O9 cells ------
def test_cpa_o9_cell_admits_only_the_plain_sum():
    """DuX(2^16), CPA direction, one active plaintext word at position 1,
    layer 6: the formal degrees are (40960, 24576, 16384, 65536) against
    T = 65535, so position 3 sits exactly at D = q.  O9 says the word is still
    balanced -- but that is a statement about c_{q-1} only."""
    pl = WT.plan(make_field("2^16"), [1], 6, "enc", "dux")
    assert pl.a0_only_rows == [3]
    assert pl.usable_weights == 1 and pl.weights == [(0,)]
    assert "O9" in pl.weight_rule
    assert pl.crit["max_degree_per_position"][5][3] == 65536


def test_cpa_o9_cell_does_not_bite_when_the_combined_row_avoids_it():
    """The S9 run used `--combine 1110`, whose rows only read positions 0..2;
    position 3 is still an O9 cell but is not among the rows, so the full
    24 575-weight budget is available (S9 used 400 of them)."""
    pl = WT.plan(make_field("2^16"), [1], 6, "enc", "dux", "1110")
    assert pl.a0_only_rows == [3]
    assert pl.usable_weights == 24575 and pl.usable_norm == 24575
    assert "O9" not in pl.weight_rule


def test_fp_has_no_o9_cell_at_the_same_layer():
    """Over F_65537 the same layer has D = 65536 = q - 1 = T, so the criterion
    reports margin 0 for position 3 and there is no O9 exemption to claim
    (which is exactly why DuX(65537) stops one layer earlier in CPA)."""
    pl = WT.plan(make_field("65537"), [1], 6, "enc", "dux", "1110")
    assert pl.a0_only_rows == []
    assert pl.crit["patterns"][5] == "1110"


# ----------------------------------------------------------- cosets --------
def test_single_coset_drops_the_multiples_of_2k():
    """toy-257, order-2^6 subgroup cosets: T = 64, so the only multiple of 64
    inside the budget is a = 0 -- and a = 0 is exactly the weight that
    `tests/test_weighted_rows.py` measures as NOT a zero sum."""
    pl = WT.plan(make_field("257"), [3], 4, "dec", "dux", coset=6)
    assert pl.margin == 8
    assert pl.coset_exclude == [64]
    assert pl.usable_weights == 7
    assert pl.weights == [(a,) for a in range(1, 8)]
    assert all(a[0] % 64 for a in pl.weights)
    assert "2^k | a_i" in pl.weight_rule


def test_coset_budget_matches_the_measured_yuxtoy257_layers():
    """`test_single_coset_with_a_nonzero_weight_is_already_a_zero_sum` measures
    layers 2 and 3 balanced for a in {1,2,3,5,7} and layer 4 failing.  The plan
    has to agree: a budget at layers 2/3, none at layer 4."""
    for layers, usable in ((2, 56), (3, 37)):
        pl = WT.plan(make_field("257"), [0], layers, "dec", "yux", coset=6)
        assert pl.usable_weights == usable
        assert max(a[0] for a in pl.weights) >= 7
    with pytest.raises(AssertionError):
        WT.plan(make_field("257"), [0], 4, "dec", "yux", coset=6)


# ---------------------------------------------- full domain and subspace ---
def test_full_domain_and_subspace_rules_are_the_old_margin():
    """Regression: the two cases the R5 drivers actually ran must be unchanged,
    and the subspace budget is the criterion's (2^m - 1) - D."""
    pl = WT.plan(make_field("257"), [3], 5)
    assert (pl.margin, pl.usable_weights) == (47, 47)
    assert "full-domain" in pl.weight_rule
    pl = WT.plan(make_field("257"), [3], 5, combine="0001")
    assert (pl.margin, pl.usable_weights) == (159, 159)
    for m, usable in ((14, 5519), (15, 21903), (16, 54671)):
        pl = WT.plan(make_field("2^16"), [3], 8, "dec", "dux",
                     subspace_dims=[m])
        assert pl.usable_weights == usable
        D = pl.crit["max_degree_per_position"][7]
        assert pl.margin == (2 ** m - 1) - max(D[2], D[3])
        assert "subspace" in pl.weight_rule


def test_nweights_caps_but_usable_weights_reports_the_budget():
    """`--weights N` must be checked against `usable_weights`, not against
    `len(weights)` (which is already capped)."""
    pl = WT.plan(make_field("2^8"), [], 6, "dec", "yux", "1110", nweights=40,
                 free_blocks=(0,))
    assert len(pl.weights) == 40
    assert pl.usable_weights == 126           # the budget, not the cap


# ----------------------------------------- the published commands still run --
PUBLISHED = [
    # every --weights value that appears in a committed run command
    ("S10 12-round DuX(65537)", "65537", [3, 7], 10, "dec", "dux", "0001", 9000),
    ("S9 11-round DuX(65537)", "65537", [3], 9, "dec", "dux", None, 2152),
    ("S9 11-round DuX(2^16)", "2^16", [3], 9, "dec", "dux", None, 1189),
    ("S9 7-round DuX(2^8)", "2^8", [3], 5, "dec", "dux", None, 46),
    ("Y06-2 11-round YupX", "65537", [0, 4], 9, "dec", "yux", None, 2400),
    ("Y06-1 10-round Yu2X-16", "2^16", [0], 8, "dec", "yux", None, 1238),
    ("Y06-1 10-round YupX", "65537", [0], 8, "dec", "yux", None, 2155),
    ("Y06-4 7-round Yu2X-8", "2^8", [0, 4], 5, "dec", "yux", None, 34),
    ("S9 8-round CPA DuX(2^16)", "2^16", [1], 6, "enc", "dux", "1110", 400),
    ("S9 8-round CPA DuX(65537)", "65537", [1], 6, "enc", "dux", "1110", 400),
]


@pytest.mark.parametrize("label,q,active,layers,direction,cipher,combine,weights",
                         PUBLISHED)
def test_every_published_weight_count_is_admissible(
        label, q, active, layers, direction, cipher, combine, weights):
    """The `--weights N` of every committed run command must pass the assert the
    drivers now make.  Two of them (46 and 34) sit exactly at the limit, which
    is also a check on the margin itself."""
    pl = WT.plan(make_field(q), active, layers, direction, cipher, combine,
                 nweights=weights)
    assert weights <= pl.usable_weights, f"{label}: {pl.weight_rule}"
    assert len(pl.weights) == weights


# --------------------------------------------------------------------------
# W25 / O15: point sets (divided-difference masks) as a fifth weight rule.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label,q,active,layers,cipher,combine,nw,n", [
    # the four rows of the R8 memo Sect. 2, recomputed by the criterion itself
    ("12-round DuX(65537), 2 words", "65537", [3, 7], 10, "dux", "0001", 9000, 39614),
    ("11-round DuX(65537), 1 word", "65537", [3], 9, "dux", None, 2152, 42698),
    ("10-round YupX, 1 word", "65537", [0], 8, "yux", None, 2155, 28239),
    ("11-round YupX, 2 words", "65537", [0, 4], 9, "yux", None, 2400, 61625),
])
def test_pointset_sizes_of_the_memo(label, q, active, layers, cipher, combine, nw, n):
    """T' = sum_i (n_i - 1) must leave exactly the weights the run used.

    The memo's n is the one-dimensional accounting (weights on the first active
    word), which is what `attack_12round.py` implements -- so `usable_norm`,
    not the number of weight VECTORS, is the quantity that must reach N_w."""
    F = make_field(q)
    s = len(active)
    pl = WT.plan(F, active, layers, "dec", cipher, combine, 1, 1,
                 pointset=[n] * s)
    assert pl.usable_norm >= nw, f"{label}: {pl.weight_rule}"
    # and one point fewer is not enough: the memo's n is minimal
    smaller = WT.plan(F, active, layers, "dec", cipher, combine, 1, 1,
                      pointset=[n - 1] * s)
    assert smaller.usable_norm < nw, f"{label}: n = {n} is not minimal"
    assert "O15 point sets" in pl.weight_rule


def test_pointset_threshold_matches_interp_mask():
    """The criterion's T' and tools/interp_mask.py must agree."""
    import interp_mask as im
    for sizes in ([100], [60, 60], [39614, 39614]):
        assert im.threshold(sizes) == sum(n - 1 for n in sizes)
    assert im.min_points(70226, 9000, 2)[0] == 39614
    assert im.min_points(26083, 2155, 1)[0] == 28239


def test_pointset_rejects_free_blocks_and_cosets():
    F = make_field("65537")
    with pytest.raises(AssertionError):
        WT.plan(F, [], 11, "dec", "dux", "0001", 1, 1, free_blocks=(0,),
                pointset=[1000])
    with pytest.raises(AssertionError):
        WT.plan(F, [3], 9, "dec", "dux", None, 1, 1, coset=15, pointset=[1000])


def test_weight_vector_count_is_the_closed_form():
    """`plan` counts the admissible weight vectors without building them."""
    from assemble_fast import weight_vector_count, weight_vectors
    for dims in (1, 2, 3):
        for norm in (1, 2, 5, 11):
            assert weight_vector_count(dims, norm) == len(weight_vectors(dims, norm))
    # the limit keeps the |a| order
    ws = weight_vectors(2, 50, limit=9)
    assert ws == weight_vectors(2, 50)[:9]
    # billions of vectors: counted, never built
    assert weight_vector_count(2, 60846) == 1851148281


def test_axis_cap_bounds_usable_weights():
    """W25: at most |axis| independent weights per weight axis.

    Measured on toy-2^4 (3 words, layer 3, 4 rows per weight): 16 / 24 / 30
    full-domain weights all give rank 64 = 4 x min(N_w, q), and a 14-point set
    gives 4 x 14 = 56.  `plan` must report that cap, not the degree margin."""
    F = make_field("2^4")
    pl = WT.plan(F, [3, 7, 11], 3, "dec", "dux", None, 1, None)
    assert pl.degree_weights == 30 and pl.axis_cap == 16
    assert pl.usable_weights == 16 and len(pl.weights) == 16
    assert "capped at 16" in pl.weight_rule
    p2 = WT.plan(F, [3, 7, 11], 3, "dec", "dux", None, 2, None)
    assert p2.axis_cap == 256 and p2.usable_weights == 256
    ps = WT.plan(F, [3, 7, 11], 3, "dec", "dux", None, 1, None, pointset=[14] * 3)
    assert ps.degree_weights == 24 and ps.axis_cap == 14 and ps.usable_weights == 14


def test_axis_cap_changes_no_published_row():
    """Every committed `--weights N` stays below its axis count, so the cap is
    a new guard and not a retraction."""
    for label, q, active, layers, direction, cipher, combine, weights in PUBLISHED:
        F = make_field(q)
        pl = WT.plan(F, active, layers, direction, cipher, combine, nweights=weights)
        assert weights <= pl.axis_cap, f"{label}: axis cap {pl.axis_cap}"
        assert weights <= pl.usable_weights, f"{label}: {pl.weight_rule}"
