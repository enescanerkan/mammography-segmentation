"""Training data of the four segmentation models, from the VinDr-Mammo DICOM files and the released labels.

    breast-seg build-dataset --dicom-root /data/vindr-mammo [--labels labels/vindr-mammo] [--out <data_root>/vindr]

    from breast_seg.datasets import VinDrDatasetBuilder
    report = VinDrDatasetBuilder("/data/vindr-mammo").build()

Inputs (``labels/vindr-mammo``, see ``labels/README.md``): ``mlo_segmentation_labels.csv`` and
``cc_segmentation_labels.csv`` (one polygon per row in DICOM pixels, with the held-out ``Split`` and the
cross-validation ``Fold`` of every image) and ``canvas.csv`` (crop, scale and padding of each image's 640 x 640
canvas). The DICOM file of an image is found anywhere under the DICOM root by its SOPInstanceUID: as the file name
(``images/<study>/<image>.dicom`` of the PhysioNet download, ``<study>/<series>/<sop>.dcm``), else from the header
of the files not already identified by name. Symbolic links to folders are not followed.

The canvases are identical to the model inputs of the study; the label maps are the released (cleaned and
smoothed) annotations, reproduced from their polygons up to rasterisation (Dice about 0.999 for breast tissue and
pectoral muscle, about 0.997 for the small nipple region):

* **canvas** (``baseline640``): VOI LUT, MONOCHROME1 inverted, the crop of ``canvas.csv``, min-max to 8 bit,
  bilinear resize to the size the padding of ``canvas.csv`` leaves, zero padding to 640 x 640;
* **label map**: the polygons mapped onto the canvas (``canvas = (dicom - crop) * scale + pad``) and filled in the
  order Breast = 2, Pectoral muscle = 1, Nipple = 3; the padding carries no label.

Written under the output folder (stem = SOPInstanceUID for MLO, ``cc_<SOPInstanceUID>`` for CC)::

    canvases/<stem>.png                              640 x 640 grey canvases
    label_maps/<stem>.png                            0 background, 1 pectoral muscle, 2 breast tissue, 3 nipple
    yolo/{images,labels}/<split>/<stem>, data.yaml   YOLO26m-seg (polygons of the label map)
    semantic/{images,masks}/<split>/<stem>.png       SegFormer-B3, U-Net (+ index.csv)
    folds.csv                                        stem, view, study, side, split, fold (10-fold cross-validation)
    metadata_mlo.csv, metadata_cc.csv                per-image geometry of the evaluation (canvas <-> DICOM, mm)
    audit.json                                       counts and every problem the audit found

and, in the nnU-Net folders (``nnunet_raw`` / ``nnunet_preprocessed`` of the paths config, which the nnU-Net
trainer reads; default ``<output>/nnunet/raw`` and ``<output>/nnunet/preprocessed``)::

    <nnunet_raw>/Dataset601_BreastSeg/                            nnU-Net v2 raw dataset (test images in imagesTs)
    <nnunet_preprocessed>/Dataset601_BreastSeg/splits_final.json  fold 0 = the held-out train / validation split

A rebuild replaces only these two Dataset601 outputs; other nnU-Net datasets in the same folders are kept.

``<split>`` is ``train`` / ``val`` / ``test`` (``Split`` = Train / Validation / Test). The layouts hard-link the
canvases and label maps (they are copied where the file system has no hard links). The layouts are written only
when the audit passes: every DICOM readable and of the labelled matrix size, every file written, canvas and
label-map shapes and values, no pectoral muscle on a CC view, breast tissue on every image, padding on the side away
from the chest wall, no study in two splits or folds, and no identical canvas in two splits or folds.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from breast_seg.config import LABEL_VALUES, YOLO_CLASSES
from breast_seg.datasets.formats import NnUNetExporter, link_or_copy, semantic_paths, write_yolo_data_yaml
from breast_seg.datasets.labels import masks_to_lines, write_lines
from breast_seg.datasets.splits import SPLITS, find_leaks
from breast_seg.imaging.preprocess import TARGET_SIZE, quick_normalize

PathLike = Union[str, Path]

LABEL_FILES = {"MLO": "mlo_segmentation_labels.csv", "CC": "cc_segmentation_labels.csv"}
CANVAS_FILE = "canvas.csv"
#: ``Split`` value of the label tables -> split folder name.
SPLIT_NAMES = {"Train": "train", "Validation": "val", "Test": "test"}
#: Polygon class -> label-map value, in painting order.
PAINT = (("Breast", 2), ("Pectoral muscle", 1), ("Nipple", 3))
CC_PREFIX = "cc_"
DICOM_SUFFIXES = (".dcm", ".dicom")
#: YOLO polygons of a label map: external contours of at least this many pixels, simplified with
#: ``approxPolyDP(YOLO_EPSILON * perimeter)``.
YOLO_MIN_AREA = 20.0
YOLO_EPSILON = 0.002
NNUNET_DATASET = (601, "BreastSeg")
NNUNET_DESCRIPTION = "breast / pectoral / nipple segmentation, 640x640 canvases, study-level split; test held out"
CANVAS_COLUMNS = ("crop_x1", "crop_y1", "crop_x2", "crop_y2", "scale", "pad_left", "pad_top", "pad_right", "pad_bottom")


class DatasetAuditError(RuntimeError):
    """The built dataset failed its audit (``problems`` lists every finding)."""

    def __init__(self, problems: List[str]):
        super().__init__(f"dataset audit failed ({len(problems)} problem(s)): " + "; ".join(problems[:5]))
        self.problems = problems


@dataclass
class LabelledImage:
    """One image of the label tables: identifiers, split, fold, DICOM geometry, canvas and polygons."""

    sop: str
    study: str
    view: str
    side: str
    split: str
    fold: int
    height: int
    width: int
    spacing_mm: float
    canvas: Dict[str, float]
    polygons: List[Tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return f"{CC_PREFIX}{self.sop}" if self.view == "CC" else self.sop


@dataclass
class BuildReport:
    """Outcome of :meth:`VinDrDatasetBuilder.build`."""

    output_dir: Path
    counts: Dict[str, Dict[str, int]]
    missing: List[str]
    problems: List[str]
    audit: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return not self.problems


# ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────


def _vertices(raw: str) -> np.ndarray:
    data = ast.literal_eval(raw)
    return np.asarray(data["vertices"], dtype=np.float64).reshape(-1, 2)


def _one(values: pd.Series, what: str, sop: str) -> Any:
    distinct = values.dropna().unique()
    if len(distinct) != 1:
        raise ValueError(f"{sop}: {len(distinct)} values of {what} in the label table")
    return distinct[0]


def read_labels(labels_dir: PathLike) -> Dict[str, LabelledImage]:
    """``{SOPInstanceUID: LabelledImage}`` from the two polygon tables and ``canvas.csv`` of ``labels_dir``."""
    labels_dir = Path(labels_dir)
    canvas = pd.read_csv(labels_dir / CANVAS_FILE, dtype={"SOPInstanceUID": str}).set_index("SOPInstanceUID")
    paint = dict(PAINT)
    out: Dict[str, LabelledImage] = {}
    for view, name in LABEL_FILES.items():
        table = pd.read_csv(labels_dir / name, dtype={"StudyInstanceUID": str, "SOPInstanceUID": str})
        for sop, rows in table.groupby("SOPInstanceUID", sort=True):
            if sop in out:
                raise ValueError(f"{sop}: listed in both label tables")
            if sop not in canvas.index:
                raise ValueError(f"{sop}: no row in {CANVAS_FILE}")
            unknown = set(rows.labelName) - set(paint)
            if unknown:
                raise ValueError(f"{sop}: unknown labelName {sorted(unknown)}")
            series = str(_one(rows.SeriesDescription, "SeriesDescription", sop))
            split = str(_one(rows.Split, "Split", sop))
            if split not in SPLIT_NAMES:
                raise ValueError(f"{sop}: Split {split!r} is not one of {sorted(SPLIT_NAMES)}")
            spacing = str(_one(rows.ImagerPixelSpacing, "ImagerPixelSpacing", sop))
            polygons = [(paint[n], _vertices(d)) for n, d in zip(rows.labelName, rows.data)]
            polygons.sort(key=lambda p: [v for _, v in PAINT].index(p[0]))
            out[sop] = LabelledImage(
                sop=sop, study=str(_one(rows.StudyInstanceUID, "StudyInstanceUID", sop)), view=view,
                side=series[:1].upper(), split=SPLIT_NAMES[split], fold=int(_one(rows.Fold, "Fold", sop)),
                height=int(_one(rows.height, "height", sop)), width=int(_one(rows.width, "width", sop)),
                spacing_mm=float(spacing.replace("\\", "/").split("/")[0]),
                canvas={c: float(canvas.at[sop, c]) for c in CANVAS_COLUMNS}, polygons=polygons)
    return out


def find_dicoms(root: PathLike, wanted: Iterable[str]) -> Dict[str, Path]:
    """``{SOPInstanceUID: DICOM path}`` for every wanted image found under ``root``.

    A ``.dcm`` / ``.dicom`` file named after the SOPInstanceUID is taken as it is (the first one in walk order);
    for images still missing after that, the headers of the DICOM files not named after a wanted image are read
    (``SOPInstanceUID`` only) until every image is found. Symbolic links to folders are not followed, so a link
    cycle cannot make the walk endless and no file is visited twice through a link.
    """
    wanted = set(wanted)
    found: Dict[str, Path] = {}
    others: List[Path] = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            stem, suffix = os.path.splitext(name)
            if suffix.lower() not in DICOM_SUFFIXES:
                continue
            path = Path(folder) / name
            if stem in wanted:
                found.setdefault(stem, path)
            else:
                others.append(path)
    if len(found) < len(wanted) and others:
        import pydicom

        for path in others:
            try:
                uid = str(pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["SOPInstanceUID"]).SOPInstanceUID)
            except Exception:  # noqa: BLE001 - an unreadable file is simply not one of ours
                continue
            if uid in wanted and uid not in found:
                found[uid] = path
                if len(found) == len(wanted):
                    break
    return found


# ── one image ────────────────────────────────────────────────────────────────────────────────────────────────


def make_canvas(pixels: np.ndarray, photometric: str, canvas: Mapping[str, float]) -> np.ndarray:
    """The ``baseline640`` canvas of DICOM ``pixels`` (after the VOI LUT) with the crop / padding of ``canvas``."""
    if str(photometric).strip().upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    x1, y1, x2, y2 = (int(canvas[c]) for c in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
    left, top, right, bottom = (int(canvas[c]) for c in ("pad_left", "pad_top", "pad_right", "pad_bottom"))
    crop = quick_normalize(pixels[y1:y2, x1:x2])
    resized = cv2.resize(crop, (TARGET_SIZE - left - right, TARGET_SIZE - top - bottom), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def to_canvas(points: np.ndarray, canvas: Mapping[str, float]) -> np.ndarray:
    """DICOM pixel coordinates ``(N, 2)`` -> canvas coordinates."""
    out = np.empty_like(np.asarray(points, dtype=np.float64))
    out[:, 0] = (points[:, 0] - canvas["crop_x1"]) * canvas["scale"] + canvas["pad_left"]
    out[:, 1] = (points[:, 1] - canvas["crop_y1"]) * canvas["scale"] + canvas["pad_top"]
    return out


def make_label_map(polygons: Sequence[Tuple[int, np.ndarray]], canvas: Mapping[str, float]) -> np.ndarray:
    """Label map of the canvas: the polygons (``(value, DICOM vertices)``) filled in the given order."""
    label = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
    for value, vertices in polygons:
        cv2.fillPoly(label, [np.round(to_canvas(vertices, canvas)).astype(np.int32)], int(value))
    left, top, right, bottom = (int(canvas[c]) for c in ("pad_left", "pad_top", "pad_right", "pad_bottom"))
    label[:top] = 0
    label[TARGET_SIZE - bottom:] = 0
    label[:, :left] = 0
    label[:, TARGET_SIZE - right:] = 0
    return label


def yolo_lines(label: np.ndarray) -> List[str]:
    """YOLO-seg lines of a label map: pectoral muscle (class 0), breast tissue with the nipple area (1), nipple (2)."""
    masks = {0: label == 1, 1: (label == 2) | (label == 3), 2: label == 3}
    return masks_to_lines({c: m.astype(np.uint8) for c, m in masks.items()}, YOLO_MIN_AREA, YOLO_EPSILON)


def read_pixels(path: PathLike) -> Tuple[np.ndarray, str]:
    """DICOM pixels after the VOI LUT (float32, as the review app reads them) and the photometric interpretation."""
    import pydicom

    from breast_seg.imaging.dicom import apply_voi_lut

    dataset = pydicom.dcmread(str(path))
    pixels = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
    return pixels, str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def _process(task: Tuple[LabelledImage, str, str]) -> Dict[str, Any]:
    """Read one DICOM, write its canvas and label map, return what the audit checks.

    Never raises: a failure of one image is returned as its ``error`` and the build goes on with the others."""
    image, dicom, out = task
    result: Dict[str, Any] = {"sop": image.sop}
    try:
        pixels, photometric = read_pixels(dicom)
    except Exception as exc:  # noqa: BLE001 - reported per image, the build goes on
        result["error"] = f"{image.stem}: DICOM not readable ({exc})"
        return result
    result["dicom_shape"] = [int(pixels.shape[0]), int(pixels.shape[1])] if pixels.ndim == 2 else list(pixels.shape)
    if result["dicom_shape"] != [image.height, image.width]:
        result["error"] = (f"{image.stem}: DICOM matrix {result['dicom_shape']} is not the labelled "
                           f"{image.height} x {image.width} ({dicom})")
        return result
    try:
        canvas = make_canvas(pixels, photometric, image.canvas)
        label = make_label_map(image.polygons, image.canvas)
        _write_png(Path(out) / "canvases" / f"{image.stem}.png", canvas)
        _write_png(Path(out) / "label_maps" / f"{image.stem}.png", label)
        result.update(canvas_shape=list(canvas.shape), canvas_dtype=str(canvas.dtype),
                      canvas_sha1=hashlib.sha1(np.ascontiguousarray(canvas).tobytes()).hexdigest(),
                      pixels={int(v): int((label == v).sum()) for v in LABEL_VALUES}, lines=yolo_lines(label))
    except Exception as exc:  # noqa: BLE001 - reported per image, the build goes on
        result = {"sop": image.sop, "dicom_shape": result["dicom_shape"],
                  "error": f"{image.stem}: canvas or label map failed ({type(exc).__name__}: {exc})"}
    return result


# ── builder ──────────────────────────────────────────────────────────────────────────────────────────────────


class VinDrDatasetBuilder:
    """Builds every training layout of the VinDr-Mammo cohort (see the module docstring)."""

    def __init__(self, dicom_root: PathLike, labels_dir: PathLike = "labels/vindr-mammo", output_dir: PathLike = "data/vindr",
                 workers: int = 1, allow_missing: bool = False, sops: Optional[Iterable[str]] = None,
                 nnunet_raw_dir: Optional[PathLike] = None, nnunet_preprocessed_dir: Optional[PathLike] = None):
        self.dicom_root = Path(dicom_root)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        #: nnU-Net raw / preprocessed folders (the ``nnunet_raw`` / ``nnunet_preprocessed`` of the paths config,
        #: where the nnU-Net trainer reads them); default ``<output>/nnunet/raw`` and ``<output>/nnunet/preprocessed``.
        self.nnunet_raw_dir = Path(nnunet_raw_dir) if nnunet_raw_dir is not None else self.output_dir / "nnunet" / "raw"
        self.nnunet_preprocessed_dir = (Path(nnunet_preprocessed_dir) if nnunet_preprocessed_dir is not None
                                        else self.output_dir / "nnunet" / "preprocessed")
        self.workers = max(1, int(workers))
        self.allow_missing = allow_missing
        self.sops = set(sops) if sops is not None else None

    # ── audit ───────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def check_image(image: LabelledImage, result: Mapping[str, Any]) -> List[str]:
        """The problems of one processed image."""
        where = image.stem
        problems = []
        if result.get("dicom_shape") != [image.height, image.width]:
            problems.append(f"{where}: DICOM matrix {result.get('dicom_shape')} is not the labelled "
                            f"{image.height} x {image.width}")
        if result.get("canvas_shape") != [TARGET_SIZE, TARGET_SIZE] or result.get("canvas_dtype") != "uint8":
            problems.append(f"{where}: canvas {result.get('canvas_shape')} {result.get('canvas_dtype')}")
        c = image.canvas
        for axis, (lo, hi, pad_a, pad_b) in (("x", ("crop_x1", "crop_x2", "pad_left", "pad_right")),
                                             ("y", ("crop_y1", "crop_y2", "pad_top", "pad_bottom"))):
            size = TARGET_SIZE - c[pad_a] - c[pad_b]
            if abs((c[hi] - c[lo]) * c["scale"] - size) > 1.0 + 1e-6:  # the resize truncates: up to one pixel
                problems.append(f"{where}: crop, scale and padding of {CANVAS_FILE} disagree on {axis}")
        chest_pad = c["pad_left"] if image.side == "L" else c["pad_right"]
        if image.side not in ("L", "R") or chest_pad > 0:
            problems.append(f"{where}: padding on the chest-wall side ({image.side})")
        px = result.get("pixels", {})
        if not px.get(2):
            problems.append(f"{where}: no breast tissue")
        if image.view == "CC" and px.get(1):
            problems.append(f"{where}: pectoral muscle on a CC view")
        if image.view == "MLO" and not px.get(1):
            problems.append(f"{where}: no pectoral muscle on an MLO view")
        return problems

    @staticmethod
    def check_groups(images: Sequence[LabelledImage]) -> List[str]:
        """No study in two splits or two folds; both views of every breast in the same split and fold."""
        problems = []
        for what, value in (("split", lambda i: i.split), ("fold", lambda i: str(i.fold))):
            leaks = find_leaks({i.stem: value(i) for i in images}, {i.stem: [i.study] for i in images})
            problems += [f"study {g} is in more than one {what}: {sorted(s)}" for g, s in leaks.items()]
        return problems

    @staticmethod
    def duplicate_canvases(images: Sequence[LabelledImage], results: Mapping[str, Mapping[str, Any]]) -> List[List[str]]:
        """Stems of the images whose canvases are pixel-identical, one sorted list per group of two or more."""
        by_digest: Dict[str, List[str]] = {}
        for image in images:
            digest = results[image.sop].get("canvas_sha1")
            if digest:
                by_digest.setdefault(digest, []).append(image.stem)
        return sorted(sorted(stems) for stems in by_digest.values() if len(stems) > 1)

    @staticmethod
    def check_duplicates(images: Sequence[LabelledImage], duplicates: Sequence[Sequence[str]]) -> List[str]:
        """No identical canvas in two splits or two folds (the same image under two SOPInstanceUIDs would leak)."""
        of = {i.stem: i for i in images}
        problems = []
        for stems in duplicates:
            splits = sorted({of[s].split for s in stems})
            folds = sorted({of[s].fold for s in stems})
            if len(splits) > 1 or len(folds) > 1:
                problems.append(f"identical canvases {list(stems)} are in more than one split or fold "
                                f"(splits {splits}, folds {folds})")
        return problems

    # ── layouts ─────────────────────────────────────────────────────────────────────────────────────────────

    def write_layouts(self, images: Sequence[LabelledImage], results: Mapping[str, Mapping[str, Any]]) -> None:
        out = self.output_dir
        for sub in ("yolo", "semantic"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
        rows = []
        for image in images:
            stem, split = image.stem, image.split
            canvas, label = out / "canvases" / f"{stem}.png", out / "label_maps" / f"{stem}.png"
            link_or_copy(canvas, out / "yolo" / "images" / split / f"{stem}.png", "hardlink")
            (out / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
            write_lines(out / "yolo" / "labels" / split / f"{stem}.txt", results[image.sop]["lines"])
            link_or_copy(canvas, out / "semantic" / "images" / split / f"{stem}.png", "hardlink")
            link_or_copy(label, out / "semantic" / "masks" / split / f"{stem}.png", "hardlink")
            px = results[image.sop]["pixels"]
            rows.append({"stem": stem, "split": split, "view": image.view, "group": image.study,
                         "height": TARGET_SIZE, "width": TARGET_SIZE,
                         **{f"has_{n}": int(px.get(v, 0) > 0) for v, n in LABEL_VALUES.items() if v},
                         **{f"px_{n}": int(px.get(v, 0)) for v, n in LABEL_VALUES.items() if v}})
        for split in SPLITS:
            (out / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        write_yolo_data_yaml(out / "yolo" / "data.yaml", out / "yolo", tuple(YOLO_CLASSES.values()))
        index = pd.DataFrame(rows)
        index.to_csv(out / "semantic" / "index.csv", index=False)
        # replaces <raw>/Dataset601_BreastSeg and <preprocessed>/Dataset601_BreastSeg/splits_final.json only
        NnUNetExporter(self.nnunet_raw_dir, self.nnunet_preprocessed_dir, *NNUNET_DATASET, NNUNET_DESCRIPTION,
                       link_mode="hardlink").export(semantic_paths(out / "semantic", index))

    def write_tables(self, images: Sequence[LabelledImage]) -> None:
        out = self.output_dir
        pd.DataFrame([{"stem": i.stem, "view": i.view, "study": i.study, "side": i.side, "split": i.split, "fold": i.fold}
                      for i in images]).to_csv(out / "folds.csv", index=False)
        for view in ("MLO", "CC"):
            pd.DataFrame([{"sop_uid": i.sop, **{c: i.canvas[c] for c in CANVAS_COLUMNS}, "pixel_spacing": i.spacing_mm,
                           "laterality": i.side, "original_width": i.width, "original_height": i.height}
                          for i in images if i.view == view]).to_csv(out / f"metadata_{view.lower()}.csv", index=False)

    # ── build ───────────────────────────────────────────────────────────────────────────────────────────────

    def build(self, strict: bool = True) -> BuildReport:
        """Read, write and audit every image; write the layouts when the audit passes.

        With ``strict`` an audit failure raises :class:`DatasetAuditError` (the canvases, label maps and the audit
        file are written either way)."""
        labels = read_labels(self.labels_dir)
        if self.sops is not None:
            labels = {s: i for s, i in labels.items() if s in self.sops}
        dicoms = find_dicoms(self.dicom_root, labels)
        missing = sorted(s for s in labels if s not in dicoms)
        problems: List[str] = []
        if missing and not self.allow_missing:
            problems += [f"{labels[s].stem}: no DICOM file under the DICOM root" for s in missing]
        images = [labels[s] for s in sorted(labels) if s in dicoms]
        out = self.output_dir
        for sub in ("canvases", "label_maps"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
            (out / sub).mkdir(parents=True)
        tasks = [(image, str(dicoms[image.sop]), str(out)) for image in images]
        if self.workers > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                processed = list(pool.map(_process, tasks, chunksize=8))
        else:
            processed = [_process(t) for t in tasks]
        results: Dict[str, Dict[str, Any]] = {}
        for image, result in zip(images, processed):
            if "error" in result:
                problems.append(result["error"])
                continue
            results[image.sop] = result
            problems += self.check_image(image, result)
        images = [i for i in images if i.sop in results]
        problems += self.check_groups(images)
        duplicates = self.duplicate_canvases(images, results)
        problems += self.check_duplicates(images, duplicates)
        counts = self.counts(images)
        audit = out / "audit.json"
        audit.write_text(json.dumps({"counts": counts, "missing_dicom": [labels[s].stem for s in missing],
                                     "no_nipple": sorted(i.stem for i in images if not results[i.sop]["pixels"].get(3)),
                                     "identical_canvases": duplicates, "problems": problems}, indent=2))
        report = BuildReport(out, counts, [labels[s].stem for s in missing], problems, audit)
        if problems:
            if strict:
                raise DatasetAuditError(problems)
            return report
        self.write_tables(images)
        self.write_layouts(images, results)
        return report

    @staticmethod
    def counts(images: Sequence[LabelledImage]) -> Dict[str, Dict[str, int]]:
        """Images per view and complete MLO-CC pairs per split."""
        out: Dict[str, Dict[str, int]] = {}
        for split in SPLITS:
            sub = [i for i in images if i.split == split]
            views = Counter(i.view for i in sub)
            breasts: Dict[Tuple[str, str], Set[str]] = {}
            for i in sub:
                breasts.setdefault((i.study, i.side), set()).add(i.view)
            out[split] = {"MLO": views.get("MLO", 0), "CC": views.get("CC", 0),
                          "pairs": sum(v == {"MLO", "CC"} for v in breasts.values())}
        return out


__all__ = ["BuildReport", "DatasetAuditError", "LabelledImage", "VinDrDatasetBuilder", "find_dicoms", "make_canvas",
           "make_label_map", "read_labels", "read_pixels", "to_canvas", "yolo_lines"]
