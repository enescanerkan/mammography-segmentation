"""VinDr-Mammo dataset builder on a synthetic cohort (phantom DICOMs, fake ids); also hosts the cohort helper."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("pydicom")

from breast_seg.cli import main  # noqa: E402
from breast_seg.config import PathsConfig  # noqa: E402
from breast_seg.datasets import (  # noqa: E402
    DatasetAuditError,
    VinDrDatasetBuilder,
    find_dicoms,
    make_label_map,
    read_labels,
)
from breast_seg.datasets.builder import read_pixels  # noqa: E402
from breast_seg.datasets.labels import rasterise, read_polygons  # noqa: E402
from breast_seg.imaging import preprocess, read_dicom  # noqa: E402
from breast_seg.imaging.phantom import phantom_dicom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, SPACING = 320, 400, 0.1


def _mirror(points, side):
    pts = np.asarray(points, float)
    if side == "R":
        pts[:, 0] = (WIDTH - 1) - pts[:, 0]
    return pts


def _vertices(points):
    return str({"vertices": [[round(float(x), 2), round(float(y), 2)] for x, y in points]})


def _box(cx, cy, half=5.0):
    return str({"x": cx - half, "y": cy - half, "width": 2 * half, "height": 2 * half})


def _geometry(view, side):
    """Polygons (DICOM pixels) that follow the phantom: half-ellipse breast, pectoral wedge, nipple at the apex."""
    semi_x, semi_y, cy = WIDTH * 0.62, HEIGHT * 0.42, HEIGHT * 0.5
    t = np.linspace(-np.pi / 2, np.pi / 2, 40)
    breast = np.stack([np.minimum(semi_x * np.cos(t), WIDTH - 1), cy + semi_y * np.sin(t)], 1)
    nipple_xy = (min(semi_x, WIDTH - 8.0), cy)
    nipple = [(nipple_xy[0] - 5, cy - 5), (nipple_xy[0] + 5, cy - 5), (nipple_xy[0] + 5, cy + 5), (nipple_xy[0] - 5, cy + 5)]
    shapes = {"Breast": _mirror(breast, side), "Nipple": _mirror(nipple, side)}
    if view == "MLO":
        shapes["Pectoral muscle"] = _mirror([(0, cy - semi_y), (WIDTH * 0.3, cy - semi_y), (0, HEIGHT * 0.55)], side)
    return shapes, _mirror([nipple_xy], side)[0]


def make_cohort(tmp: Path, n_studies: int = 8, n_folds: int = 5, layout_mix: bool = True):
    """Synthetic VinDr-Mammo cohort: per study two breasts with an MLO and a CC phantom DICOM; the released label
    tables (polygons, Split, Fold, canvas.csv) and radiologist landmark tables.

    Writes ``<tmp>/dicom`` (half ``images/<study>/<sop>.dicom``, half ``<study>/<series>/<sop>.dcm``),
    ``<tmp>/labels/vindr-mammo`` and ``<tmp>/data/landmarks``; returns ``(paths, labels_dir, dicom_root)`` with a
    :class:`PathsConfig` rooted at ``tmp`` whose ``extra`` entries match ``configs/paths.example.yaml``.
    """
    labels_dir, dicom_root, landmarks = tmp / "labels/vindr-mammo", tmp / "dicom", tmp / "data/landmarks"
    for d in (labels_dir, dicom_root, landmarks):
        d.mkdir(parents=True, exist_ok=True)
    seg = {"MLO": [], "CC": []}
    marks = {"MLO": [], "CC": []}
    canvas, k = [], 0
    for j in range(n_studies):
        study = f"st{j}"
        split = "Test" if j == 0 else "Validation" if j == 1 else "Train"
        for side in ("L", "R"):
            for view in ("MLO", "CC"):
                sop = f"{'m' if view == 'MLO' else 'c'}{j}{side}"
                data = phantom_dicom(view, side, width=WIDTH, height=HEIGHT, spacing=SPACING, seed=k,
                                     attributes={"SOPInstanceUID": sop})
                k += 1
                if layout_mix and k % 2:
                    path = dicom_root / "images" / study / f"{sop}.dicom"
                else:
                    path = dicom_root / study / f"se{j}" / f"{sop}.dcm"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                pixels, meta = read_dicom(data, path.name)
                _, tr = preprocess(pixels, meta, enhance=False)
                canvas.append({"SOPInstanceUID": sop, "crop_x1": tr.crop_x1, "crop_y1": tr.crop_y1,
                               "crop_x2": tr.crop_x2, "crop_y2": tr.crop_y2, "scale": round(tr.scale, 10),
                               "pad_left": tr.pad_left, "pad_top": tr.pad_top, "pad_right": tr.pad_right,
                               "pad_bottom": tr.pad_bottom})
                shapes, nip = _geometry(view, side)
                common = {"StudyInstanceUID": study, "SOPInstanceUID": sop, "SeriesDescription": f"{side}-{view}"}
                for name, pts in shapes.items():
                    seg[view].append({**common, "annotationMode": "polygon", "labelName": name, "data": _vertices(pts),
                                      "height": HEIGHT, "width": WIDTH, "ImagerPixelSpacing": f"{SPACING}\\{SPACING}",
                                      "Split": split, "Fold": j % n_folds})
                rows = [{**common, "annotationMode": "bounding_box", "labelName": "Nipple", "data": _box(*nip)}]
                if view == "MLO":
                    top, bottom = _mirror([(WIDTH * 0.3, HEIGHT * 0.08), (0, HEIGHT * 0.55)], side)
                    rows.append({**common, "annotationMode": "line", "labelName": "Pectoralis",
                                 "data": _vertices([top, bottom])})
                for r in rows:
                    r.update({"height": HEIGHT, "width": WIDTH, "Split": split,
                              "qualitativeLabel": ("Good" if j % 3 else "Bad") if view == "CC" else ""})
                marks[view] += rows
    for view in ("MLO", "CC"):
        pd.DataFrame(seg[view]).to_csv(labels_dir / f"{view.lower()}_segmentation_labels.csv", index=False)
        pd.DataFrame(marks[view]).to_csv(landmarks / f"{view.lower()}_labels.csv", index=False)
    pd.DataFrame(canvas).to_csv(labels_dir / "canvas.csv", index=False)
    paths_file = tmp / "paths.yaml"
    paths_file.write_text(yaml.safe_dump({"root": ".", "extra": {"mlo_landmarks": "data/landmarks/mlo_labels.csv",
                                                                 "cc_landmarks": "data/landmarks/cc_labels.csv"}}))
    return PathsConfig.load(paths_file), labels_dir, dicom_root


def build_cohort(tmp: Path, n_studies: int = 8, n_folds: int = 5) -> PathsConfig:
    """:func:`make_cohort` built into ``<tmp>/data/vindr`` (the layout the committed configs read)."""
    paths, labels_dir, dicom_root = make_cohort(tmp, n_studies, n_folds)
    VinDrDatasetBuilder(dicom_root, labels_dir, tmp / "data/vindr").build()
    return paths


def test_builds_every_layout_from_synthetic_dicoms(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=4)
    out = tmp_path / "data/vindr"
    report = VinDrDatasetBuilder(dicom_root, labels_dir, out).build()
    assert report.ok and not report.missing
    assert report.counts == {"train": {"MLO": 4, "CC": 4, "pairs": 4}, "val": {"MLO": 2, "CC": 2, "pairs": 2},
                             "test": {"MLO": 2, "CC": 2, "pairs": 2}}
    audit = json.loads((out / "audit.json").read_text())
    assert audit["problems"] == [] and audit["no_nipple"] == []
    # YOLO layout
    data = yaml.safe_load((out / "yolo/data.yaml").read_text())
    assert data["names"] == ["pectoral", "breast-tissue", "nipple"] and Path(data["path"]) == out / "yolo"
    assert sorted(p.stem for p in (out / "yolo/images/test").glob("*.png")) == ["cc_c0L", "cc_c0R", "m0L", "m0R"]
    for txt in (out / "yolo/labels").rglob("*.txt"):
        classes = {ln.split()[0] for ln in txt.read_text().splitlines()}
        assert classes == ({"1", "2"} if txt.stem.startswith("cc_") else {"0", "1", "2"})
    # semantic layout and nnU-Net raw dataset (fold 0 = the held-out train / val split)
    index = pd.read_csv(out / "semantic/index.csv")
    assert len(index) == 16 and set(index.split) == {"train", "val", "test"}
    assert (index[index.view == "CC"].has_pectoral == 0).all() and index["has_breast-tissue"].all()
    raw = out / "nnunet/raw/Dataset601_BreastSeg"
    assert len(list((raw / "imagesTr").glob("*_0000.png"))) == 12 and len(list((raw / "imagesTs").glob("*.png"))) == 4
    splits = json.loads((out / "nnunet/preprocessed/Dataset601_BreastSeg/splits_final.json").read_text())
    assert len(splits) == 1 and len(splits[0]["train"]) == 8 and len(splits[0]["val"]) == 4
    # one file, linked into every layout
    canvas = out / "canvases/m2L.png"
    assert canvas.stat().st_nlink == 4 and (out / "label_maps/m2L.png").stat().st_nlink == 3
    # tables for cross-validation and evaluation
    folds = pd.read_csv(out / "folds.csv")
    assert set(folds.columns) == {"stem", "view", "study", "side", "split", "fold"} and len(folds) == 16
    assert folds.groupby("study").fold.nunique().max() == 1
    meta = pd.read_csv(out / "metadata_cc.csv")
    assert set(meta.laterality) == {"L", "R"} and (meta.pixel_spacing == SPACING).all()
    assert (meta.original_width == WIDTH).all() and (meta.original_height == HEIGHT).all()


def test_canvas_is_the_baseline640_input_of_the_app(tmp_path):
    """With the crop / padding of canvas.csv the builder writes exactly what the review app feeds the models."""
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=1)
    out = tmp_path / "out"
    VinDrDatasetBuilder(dicom_root, labels_dir, out).build()
    for path in sorted(dicom_root.rglob("*.dc*")):
        pixels, meta = read_dicom(path.read_bytes(), path.name)
        expected, _ = preprocess(pixels, meta, enhance=False)
        stem = path.stem if meta["view"] == "MLO" else f"cc_{path.stem}"
        built = cv2.imread(str(out / "canvases" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        assert built.shape == (640, 640) and np.array_equal(built, expected)
        assert read_pixels(path)[0].shape == (HEIGHT, WIDTH)


def test_label_map_paint_order_padding_and_yolo_polygons(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=1)
    images = read_labels(labels_dir)
    mlo, cc = images["m0R"], images["c0L"]
    assert [v for v, _ in mlo.polygons] == [2, 1, 3] and mlo.stem == "m0R" and cc.stem == "cc_c0L"
    label = make_label_map(mlo.polygons, mlo.canvas)
    assert set(np.unique(label)) == {0, 1, 2, 3}
    pad = int(mlo.canvas["pad_left"])                 # right breast: chest wall right, padding on the left
    assert pad > 0 and not label[:, :pad].any()
    out = tmp_path / "out"
    VinDrDatasetBuilder(dicom_root, labels_dir, out).build()
    for stem in ("m0R", "cc_c0L"):
        lab = cv2.imread(str(out / "label_maps" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        yolo, _ = rasterise(read_polygons(out / "yolo/labels/test" / f"{stem}.txt"), 640, 640)
        for value in (1, 2, 3):
            a, b = lab == value, yolo == value
            if a.any() or b.any():
                assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.97


def test_dicoms_are_found_by_name_in_any_layout_or_by_header(tmp_path):
    _, _, dicom_root = make_cohort(tmp_path, n_studies=1)
    moved = dicom_root / "elsewhere" / "IM0001.dcm"
    moved.parent.mkdir()
    (dicom_root / "images/st0/m0L.dicom").rename(moved)            # PhysioNet layout, then moved
    (dicom_root / "images/notes.txt").write_text("not a DICOM")
    found = find_dicoms(dicom_root, ["m0L", "m0R", "c0L", "c0R", "absent"])
    assert set(found) == {"m0L", "m0R", "c0L", "c0R"} and found["m0L"] == moved
    assert found["m0R"] == dicom_root / "images/st0/m0R.dicom" and found["c0L"] == dicom_root / "st0/se0/c0L.dcm"


def test_missing_dicoms_fail_unless_allowed(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=3)
    for p in (dicom_root / "images" / "st2").glob("*"):
        p.unlink()
    with pytest.raises(DatasetAuditError, match="no DICOM file"):
        VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "a").build()
    assert not (tmp_path / "a/yolo").exists()
    report = VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "b", allow_missing=True).build()
    assert report.ok and len(report.missing) == 2 and (tmp_path / "b/yolo/data.yaml").is_file()


@pytest.mark.parametrize("change, message", [
    ("cc_pectoral", "pectoral muscle on a CC view"),
    ("study_in_two_splits", "is in more than one split"),
    ("chest_side_padding", "padding on the chest-wall side"),
])
def test_audit_failures_stop_before_the_layouts(tmp_path, change, message):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=3)
    cc = pd.read_csv(labels_dir / "cc_segmentation_labels.csv")
    if change == "cc_pectoral":
        row = cc[(cc.SOPInstanceUID == "c2L") & (cc.labelName == "Breast")].copy()
        row["labelName"] = "Pectoral muscle"
        cc = pd.concat([cc, row])
    elif change == "study_in_two_splits":
        cc.loc[cc.SOPInstanceUID == "c2L", "Split"] = "Test"
    else:
        canvas = pd.read_csv(labels_dir / "canvas.csv")
        i = canvas.index[canvas.SOPInstanceUID == "c2L"]
        canvas.loc[i, ["pad_left", "pad_right"]] = canvas.loc[i, ["pad_right", "pad_left"]].to_numpy()
        canvas.to_csv(labels_dir / "canvas.csv", index=False)
    cc.to_csv(labels_dir / "cc_segmentation_labels.csv", index=False)
    with pytest.raises(DatasetAuditError) as err:
        VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "out").build()
    assert any(message in p for p in err.value.problems)
    assert not (tmp_path / "out/semantic").exists() and (tmp_path / "out/audit.json").is_file()


def test_parallel_build_and_command_line(tmp_path, capsys):
    paths, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=2)
    assert main(["build-dataset", "--dicom-root", str(dicom_root), "--labels", str(labels_dir),
                 "--out", str(tmp_path / "cli"), "--workers", "2", "--paths", str(paths.source)]) == 0
    assert "test: 2 pairs (2 MLO, 2 CC)" in capsys.readouterr().out
    VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "serial").build()
    for p in sorted((tmp_path / "serial/label_maps").glob("*.png")):
        for sub in ("canvases", "label_maps"):
            a = cv2.imread(str(tmp_path / "serial" / sub / p.name), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(tmp_path / "cli" / sub / p.name), cv2.IMREAD_UNCHANGED)
            assert np.array_equal(a, b)
    assert main(["build-dataset", "--dicom-root", str(tmp_path / "nothing"), "--labels", str(labels_dir),
                 "--out", str(tmp_path / "empty"), "--paths", str(paths.source)]) == 1
    assert "no DICOM file" in capsys.readouterr().err


def test_command_line_defaults_follow_the_paths_file(tmp_path, monkeypatch, capsys):
    """Without --out / --labels the command writes where the training configs read (``${data_root}/vindr``, the
    nnU-Net dataset in ``nnunet_raw`` / ``nnunet_preprocessed``) and reads the labels under the paths root, whatever
    the working directory. A rebuild replaces Dataset601 only."""
    from breast_seg.training import build_trainer, load_train_config

    _, _, dicom_root = make_cohort(tmp_path, n_studies=2)          # labels in <tmp>/labels/vindr-mammo
    paths_file = tmp_path / "configs/paths.yaml"
    paths_file.parent.mkdir()
    paths_file.write_text(yaml.safe_dump({"root": "..", "data_root": "big/data", "nnunet_raw": "fast/raw",
                                          "nnunet_preprocessed": "fast/pre"}))
    other_raw, other_pre = tmp_path / "fast/raw/Dataset777_Other", tmp_path / "fast/pre/Dataset777_Other"
    for d in (other_raw, other_pre):
        d.mkdir(parents=True)
        (d / "keep.json").write_text("{}")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    for _ in range(2):
        assert main(["build-dataset", "--dicom-root", str(dicom_root), "--paths", str(paths_file)]) == 0
        (tmp_path / "fast/raw/Dataset601_BreastSeg/stale.txt").write_text("from the first build")
    assert main(["build-dataset", "--dicom-root", str(dicom_root), "--paths", str(paths_file)]) == 0
    out = tmp_path / "big/data/vindr"
    assert f"dataset written to {out}" in capsys.readouterr().out
    assert (out / "semantic/index.csv").is_file() and (out / "yolo/data.yaml").is_file() and not (out / "nnunet").exists()
    raw, pre = tmp_path / "fast/raw/Dataset601_BreastSeg", tmp_path / "fast/pre/Dataset601_BreastSeg"
    assert (raw / "dataset.json").is_file() and not (raw / "stale.txt").exists()
    assert len(json.loads((pre / "splits_final.json").read_text())) == 1
    assert (other_raw / "keep.json").is_file() and (other_pre / "keep.json").is_file()
    assert not list((tmp_path / "elsewhere").iterdir())
    paths = PathsConfig.load(paths_file)
    for config in ("segformer_b3.yaml", "unet.yaml"):
        assert paths.resolve(load_train_config(f"{ROOT}/configs/train/{config}").data.dataset_dir) == out / "semantic"
    assert paths.resolve(load_train_config(f"{ROOT}/configs/train/yolo26m_seg.yaml").data.dataset_dir) == out / "yolo"
    nnunet = build_trainer(f"{ROOT}/configs/train/nnunet_resenc_m.yaml", paths)
    assert nnunet.raw_dir / nnunet.dataset_folder == raw and nnunet.preprocessed_dir / nnunet.dataset_folder == pre


def test_nnunet_folders_of_the_builder(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=2)
    (tmp_path / "out/nnunet/raw/Dataset999_Mine").mkdir(parents=True)
    VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "out", nnunet_raw_dir=tmp_path / "r",
                        nnunet_preprocessed_dir=tmp_path / "p").build()
    assert (tmp_path / "r/Dataset601_BreastSeg/dataset.json").is_file()
    assert (tmp_path / "p/Dataset601_BreastSeg/splits_final.json").is_file()
    assert (tmp_path / "out/nnunet/raw/Dataset999_Mine").is_dir()        # the output folder's nnunet/ is not wiped


def test_dicom_matrix_other_than_the_label_table_is_an_audit_entry(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=3)
    cc = pd.read_csv(labels_dir / "cc_segmentation_labels.csv")
    cc.loc[cc.SOPInstanceUID == "c2L", "height"] = HEIGHT // 2          # e.g. another image under that name
    cc.to_csv(labels_dir / "cc_segmentation_labels.csv", index=False)
    with pytest.raises(DatasetAuditError) as err:
        VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "out", workers=2).build()
    assert [p for p in err.value.problems if "DICOM matrix" in p] == [p for p in err.value.problems if "cc_c2L" in p]
    assert len(err.value.problems) == 1
    assert not (tmp_path / "out/canvases/cc_c2L.png").exists() and (tmp_path / "out/canvases/m2L.png").is_file()
    assert len(list((tmp_path / "out/canvases").glob("*.png"))) == 11


def test_a_failing_image_or_file_write_does_not_stop_the_build(tmp_path, monkeypatch):
    from breast_seg.datasets import builder

    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=3)
    real_canvas, real_write = builder.make_canvas, cv2.imwrite
    calls = {"n": 0}

    def make_canvas(pixels, photometric, canvas):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("broken pixel data")
        return real_canvas(pixels, photometric, canvas)

    monkeypatch.setattr(builder, "make_canvas", make_canvas)
    monkeypatch.setattr(cv2, "imwrite", lambda path, img, *a: False if path.endswith("label_maps/m2R.png")
                        else real_write(path, img, *a))
    with pytest.raises(DatasetAuditError) as err:
        VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "out").build()
    problems = err.value.problems
    assert len(problems) == 2
    assert any("canvas or label map failed (ValueError: broken pixel data)" in p for p in problems)
    assert any(p.startswith("m2R:") and "could not write" in p for p in problems)
    audit = json.loads((tmp_path / "out/audit.json").read_text())
    assert audit["problems"] == problems and not (tmp_path / "out/semantic").exists()
    assert len(list((tmp_path / "out/canvases").glob("*.png"))) == 11      # every other image was processed


def test_identical_canvases_in_two_splits_fail_and_are_listed(tmp_path):
    _, labels_dir, dicom_root = make_cohort(tmp_path, n_studies=3)
    canvas = pd.read_csv(labels_dir / "canvas.csv").set_index("SOPInstanceUID")

    def same_image(src, dst):
        """``dst`` gets the pixels and the canvas of ``src`` (one image under two SOPInstanceUIDs)."""
        next(dicom_root.rglob(f"{dst}.*")).write_bytes(next(dicom_root.rglob(f"{src}.*")).read_bytes())
        canvas.loc[dst] = canvas.loc[src]
        canvas.reset_index().to_csv(labels_dir / "canvas.csv", index=False)

    same_image("m2L", "c2L")                      # same breast, split and fold: listed, not a problem
    report = VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "same").build()
    assert report.ok
    assert json.loads((tmp_path / "same/audit.json").read_text())["identical_canvases"] == [["cc_c2L", "m2L"]]
    same_image("m0L", "m1L")                      # Test / fold 0 and Validation / fold 1: a leak
    with pytest.raises(DatasetAuditError) as err:
        VinDrDatasetBuilder(dicom_root, labels_dir, tmp_path / "leak").build()
    assert err.value.problems == ["identical canvases ['m0L', 'm1L'] are in more than one split or fold "
                                  "(splits ['test', 'val'], folds [0, 1])"]


def test_find_dicoms_skips_folder_links_and_reads_only_unnamed_headers(tmp_path, monkeypatch):
    import pydicom

    _, _, dicom_root = make_cohort(tmp_path, n_studies=1)
    (dicom_root / "loop").symlink_to(dicom_root, target_is_directory=True)        # a cycle when links are followed
    original = dicom_root / "images/st0/m0R.dicom"
    (dicom_root / "zcopies").mkdir()
    (dicom_root / "zcopies/m0R.dcm").write_bytes(original.read_bytes())          # named after a wanted image
    moved = dicom_root / "zz/IM0001.dcm"
    moved.parent.mkdir()
    (dicom_root / "images/st0/m0L.dicom").rename(moved)
    (dicom_root / "zz/IM0002.dcm").write_bytes(b"not needed once every image is found")
    read = []
    real = pydicom.dcmread
    monkeypatch.setattr(pydicom, "dcmread", lambda path, *a, **k: read.append(Path(path)) or real(path, *a, **k))
    found = find_dicoms(dicom_root, ["m0L", "m0R", "c0L", "c0R"])
    assert found == {"m0L": moved, "m0R": original, "c0L": dicom_root / "st0/se0/c0L.dcm",
                     "c0R": next(p for p in dicom_root.rglob("c0R.*") if "loop" not in p.parts)}
    assert read == [moved]
