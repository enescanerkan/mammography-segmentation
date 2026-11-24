"""Label primitives and splits (synthetic masks only)."""
import numpy as np
import pandas as pd
import pytest

from breast_seg.datasets.labels import (
    check_label_text,
    masks_to_lines,
    parse_polygons,
    polygons_to_masks,
    rasterise,
)
from breast_seg.datasets.splits import (
    FoldAssignment,
    LeakageError,
    assert_no_leaks,
    find_leaks,
    folds_from_pairs,
)


def square(x0, y0, x1, y1):
    return f"{x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}"


def test_polygons_round_trip_through_masks():
    text = f"1 {square(0.1, 0.1, 0.9, 0.9)}\n2 {square(0.4, 0.4, 0.6, 0.6)}\n"
    polys = parse_polygons(text)
    assert set(polys) == {1, 2}
    masks = polygons_to_masks(polys, 100, 100)
    assert masks[0].sum() == 0 and masks[1].sum() > 5000 and 300 < masks[2].sum() < 500
    lines = masks_to_lines(masks)
    again = polygons_to_masks(parse_polygons("\n".join(lines)), 100, 100)
    for c in (1, 2):
        inter = (again[c] & masks[c]).sum()
        assert inter / max(masks[c].sum(), 1) > 0.95


def test_masks_to_lines_drops_tiny_contours():
    m = np.zeros((50, 50), np.uint8)
    m[10:13, 10:13] = 1  # area 4 < 20
    assert masks_to_lines({1: m}) == []


def test_rasterise_paint_order_breast_then_pectoral_then_nipple():
    # breast covers everything, pectoral the top-left, nipple a small square inside the pectoral
    text = "\n".join([f"1 {square(0, 0, 1, 1)}", f"0 {square(0, 0, 0.5, 0.5)}", f"2 {square(0.2, 0.2, 0.3, 0.3)}"])
    label, overwritten = rasterise(parse_polygons(text), 40, 40)
    assert label[35, 35] == 2          # breast tissue
    assert label[5, 5] == 1            # pectoral painted over breast
    assert label[10, 10] == 3          # nipple on top
    assert overwritten > 0


def test_check_label_text_reports_errors():
    counts, errors = check_label_text(f"1 {square(0.1, 0.1, 0.5, 0.5)}\n7 0 0 1 1 1 0\n0 0.1 0.2\n2 0 0 2 2 0 1\n")
    assert counts == {1: 1}
    assert len(errors) == 3


def test_find_leaks():
    split_of = {"a": "train", "b": "test", "c": "train"}
    groups = {"a": ["study:1"], "b": ["study:1", "series:9"], "c": ["series:9x"]}
    assert set(find_leaks(split_of, groups)) == {"study:1"}
    with pytest.raises(LeakageError):
        assert_no_leaks(split_of, groups)


def fold_table():
    rows = []
    for g in range(10):
        for v in ("m", "cc_m"):
            rows.append({"stem": f"{v}{g}", "fold": g % 5, "group": f"g{g}", "labelled": 0 if (g == 3 and v == "m") else 1})
    return pd.DataFrame(rows)


def test_fold_assignment_protocol():
    a = FoldAssignment.from_frame(fold_table(), "stem", "fold", "group", "labelled", n_folds=5)
    s = a.split(2)
    assert set(s.test) == {"m2", "cc_m2", "m7", "cc_m7"}
    assert set(s.val) == {"m3", "cc_m3", "m8", "cc_m8"} - {"m3"}   # unlabelled m3 never validates
    assert "m3" not in s.train and not set(s.train) & set(s.test)
    s3 = a.split(3)
    assert "m3" in s3.test and "m3" in s3.unlabelled


def test_fold_assignment_rejects_split_group():
    df = fold_table()
    df.loc[0, "fold"] = 4
    with pytest.raises(LeakageError):
        FoldAssignment.from_frame(df, "stem", "fold", "group", n_folds=5)


def test_extend_by_group_and_pairs_conversion():
    pairs = pd.DataFrame({"StudyInstanceUID": ["g1", "g2"], "SOPInstanceUID_mlo": ["a", "b"],
                          "SOPInstanceUID_cc": ["c", "d"], "fold": [0, 1]})
    long = folds_from_pairs(pairs, "SOPInstanceUID_mlo", "SOPInstanceUID_cc", group_column="StudyInstanceUID")
    assert list(long.stem) == ["a", "cc_c", "b", "cc_d"]
    a = FoldAssignment.from_frame(long, "stem", "fold", "group", n_folds=2)
    left = a.extend_by_group(["a", "x", "y"], {"x": "g2", "y": "g9"})
    assert a.fold_of["x"] == 1 and left == ["y"]
