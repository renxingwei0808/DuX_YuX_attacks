"""S17: the encryption-direction report of min_data_v3 (Table 12 and v2
comparisons).  The search itself is v2's code and is pinned by the committed
tables; here the comparison logic is checked on synthetic rows, and the
committed encryption report is checked against the claims the ledger makes."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E09_unified_criterion"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import min_data_v3 as V3  # noqa: E402

ENC_JSON = os.path.join(ROOT, "results", "E09_unified_criterion",
                        "min_data_table_v3_enc.json")


def _row(inst, cipher, layer, cls, log2=None, **kw):
    r = {"instance": inst, "cipher": cipher, "direction": "enc", "layer": layer,
         "class": cls, "dim_K": kw.get("dim_K", 0), "reachable": log2 is not None,
         "usable": kw.get("usable", "distinguisher only")}
    if log2 is not None:
        r.update({"log2_data": log2, "active": kw.get("active", []),
                  "free_blocks": kw.get("free_blocks", [0]), "dims": kw.get("dims"),
                  "s_words": kw.get("s_words", 4), "degree_needed": kw.get("deg", 1)})
    return r


def _synthetic():
    """A YuX-like instance: layers 1-7 reachable, layer 7 only by 0111 (full
    block, 2^64) and by 1100 (a distinguisher-only class), layer 8 by nothing."""
    rows = []
    for layer in range(1, 9):
        for cls in V3.ALL_CLASSES:
            usable = "dim K = 4" if cls in ("0111", "1110", "1111") else "distinguisher only"
            if layer <= 6:
                rows.append(_row("YupX-65537", "yux", layer, cls, 16.0, active=[1],
                                 free_blocks=[], s_words=1, usable=usable))
            elif layer == 7 and cls == "0111":
                rows.append(_row("YupX-65537", "yux", layer, cls, 64.0, dim_K=4, usable=usable))
            elif layer == 7 and cls == "1100":
                rows.append(_row("YupX-65537", "yux", layer, cls, 48.0, active=[1, 2, 3],
                                 free_blocks=[], s_words=3))
            elif layer == 8 and cls == "0001":
                # a distinguisher-only class one layer deeper (as the two
                # small instances show at layer 5): does not move the cap
                rows.append(_row("YupX-65537", "yux", layer, cls, 100.0))
            else:
                rows.append(_row("YupX-65537", "yux", layer, cls))
    return rows


def test_table12_comparison_reads_the_deepest_usable_layer():
    rows = _synthetic()
    out = [c for c in V3.compare_table12(rows) if c["instance"] == "YupX-65537"]
    assert len(out) == 1
    c = out[0]
    assert c["v3_deepest_usable_layer"] == 7
    assert c["v3_cheapest_at_deepest"]["classes"] == ["0111"]
    assert c["v3_cheapest_at_deepest"]["log2_data"] == 64.0
    assert "free block [0]" in c["v3_cheapest_at_deepest"]["structure"]
    # 1100 at layer 7 is reachable but not a Table-12 class, and the
    # distinguisher-only 0001 at layer 8 does not move the cap: they count
    # for "any class", not for "usable" / "key recovery"
    assert c["v3_deepest_key_recovery_layer"] == 7
    assert c["v3_deepest_any_class_layer"] == 8
    assert c["v3_next_layer"] == 8 and not c["v3_next_layer_unreachable"]
    assert [(x["class"], x["log2_data"]) for x in c["v3_next_layer_cells"]] == [("0001", 100.0)]
    assert c["agrees"]


def test_table12_comparison_flags_a_deeper_usable_layer():
    rows = _synthetic()
    rows.append(_row("YupX-65537", "yux", 8, "1111", 200.0, dim_K=8, usable="dim K = 8 + plain rows"))
    c = [c for c in V3.compare_table12(rows) if c["instance"] == "YupX-65537"][0]
    assert c["v3_deepest_usable_layer"] == 8 and c["v3_deepest_key_recovery_layer"] == 8
    assert not c["agrees"]              # Table 12 says layer 8 is unreachable
    rows = _synthetic()
    rows.append(_row("YupX-65537", "yux", 8, "1101", 200.0, dim_K=4, usable="dim K = 4"))
    c = [c for c in V3.compare_table12(rows) if c["instance"] == "YupX-65537"][0]
    assert c["v3_deepest_usable_layer"] == 7 and c["v3_deepest_key_recovery_layer"] == 8
    assert not c["agrees"]              # a non-Table-12 class the key recovery could use


def test_v2_comparison_maps_xxx1_and_labels_the_three_verdicts():
    rows = [_row("DuX(65537)", "dux", 7, "0001", 64.0, active=[3]),
            _row("DuX(65537)", "dux", 7, "1110", 64.0),
            _row("DuX(65537)", "dux", 7, "1111", 208.0),
            _row("DuX(65537)", "dux", 8, "1110")]
    v2 = [{"instance": "DuX(65537)", "direction": "enc", "layer": 7, "class": "xxx1",
           "reachable": True, "log2_data": 208.0, "dim_K": 4},
          {"instance": "DuX(65537)", "direction": "enc", "layer": 7, "class": "1110",
           "reachable": True, "log2_data": 64.0, "dim_K": 4},
          {"instance": "DuX(65537)", "direction": "enc", "layer": 7, "class": "1111",
           "reachable": True, "log2_data": 64.0, "dim_K": 8},
          {"instance": "DuX(65537)", "direction": "enc", "layer": 8, "class": "1110",
           "reachable": False, "dim_K": 4},
          {"instance": "DuX(65537)", "direction": "dec", "layer": 7, "class": "1111",
           "reachable": True, "log2_data": 1.0, "dim_K": 4}]
    out = V3.compare_v2(rows, v2)
    assert [c["class"] for c in out] == ["0001", "1110", "1111", "1110"]   # dec row skipped
    assert [c["verdict"] for c in out] == ["cheaper", "same", "more expensive", "same"]
    assert out[0]["witness"] is not None and "words [3]" in out[0]["witness"]
    assert out[1]["witness"] is None
    assert out[3]["v2"] is None and out[3]["v3"] is None


@pytest.mark.skipif(not os.path.exists(ENC_JSON), reason="encryption report not generated")
def test_committed_encryption_report_matches_table12_and_v2():
    d = json.load(open(ENC_JSON))
    assert all(c["agrees"] for c in d["table12_comparison"]), \
        [c["instance"] for c in d["table12_comparison"] if not c["agrees"]]
    deep = {c["instance"]: (c["v3_deepest_usable_layer"], c["v3_deepest_key_recovery_layer"])
            for c in d["table12_comparison"]}
    assert deep == {"DuX(65537)": (7, 7), "DuX(2^16)": (7, 7), "YupX-65537": (7, 7),
                    "Yu2X-16": (7, 7), "DuX(2^8)": (4, 4), "Yu2X-8": (4, 4)}
    # layer 8 is unreachable for any structure on the four large instances;
    # on the two small ones layer 5 has exactly one cell, unusable, at 2^100
    nxt = {c["instance"]: c["v3_next_layer_unreachable"] for c in d["table12_comparison"]}
    assert nxt == {"DuX(65537)": True, "DuX(2^16)": True, "YupX-65537": True,
                   "Yu2X-16": True, "DuX(2^8)": False, "Yu2X-8": False}
    small = {c["instance"]: [(x["class"], x["log2_data"]) for x in c["v3_next_layer_cells"]]
             for c in d["table12_comparison"] if c["instance"] in ("DuX(2^8)", "Yu2X-8")}
    assert small == {"DuX(2^8)": [("0010", 100)], "Yu2X-8": [("0001", 100)]}
    verdicts = {c["verdict"] for c in d["v2_comparison"]}
    assert verdicts == {"same"}, verdicts
    assert len(d["v2_comparison"]) == 6 * 12 * 3
    # the YuX layer-7 full-block cell: the criterion itself says 0111 (the
    # paper's 1111 is O14-CPA, not the criterion)
    yux7 = {r["class"]: r for r in d["rows"]
            if r["instance"] == "YupX-65537" and r["layer"] == 7 and r["reachable"]}
    assert yux7["0111"]["log2_data"] == 64.0 and yux7["0111"]["free_blocks"] == [0]
    assert "1111" not in yux7 or yux7["1111"]["log2_data"] > 64.0
