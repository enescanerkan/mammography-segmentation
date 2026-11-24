"""The dataset layouts the model families read.

* YOLO: ``images/<split>/<stem>.png``, ``labels/<split>/<stem>.txt`` and ``data.yaml`` (:func:`write_yolo_data_yaml`).
* Semantic (SegFormer, U-Net): ``images/<split>/<stem>.png`` (grey), ``masks/<split>/<stem>.png``
  (label map ``0 bg / 1 pectoral / 2 breast-tissue / 3 nipple``) and ``index.csv`` (:func:`semantic_paths`).
* nnU-Net v2 raw: ``Dataset<ID>_<name>/{imagesTr,labelsTr,imagesTs,labelsTs}`` with ``<stem>_0000.png`` inputs,
  ``dataset.json`` and ``splits_final.json`` (:class:`NnUNetExporter`).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
import yaml

from breast_seg.config import LABEL_VALUES, YOLO_CLASSES

PathLike = Union[str, Path]


def write_yolo_data_yaml(path: PathLike, root: PathLike, names: Sequence[str] = tuple(YOLO_CLASSES.values()),
                         splits: Optional[Mapping[str, Optional[str]]] = None) -> Path:
    """Write an Ultralytics ``data.yaml`` (``path`` = absolute ``root``; the file stays local, it is never committed)."""
    splits = dict(splits or {"train": "images/train", "val": "images/val", "test": "images/test"})
    data: Dict[str, object] = {"path": str(Path(root).resolve())}
    data.update({k: v for k, v in splits.items() if v})
    data["nc"] = len(names)
    data["names"] = list(names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def link_or_copy(src: PathLike, dst: PathLike, mode: str = "symlink") -> None:
    """Place ``src`` at ``dst`` by ``symlink`` (absolute target), ``hardlink`` (a copy where the file system has no
    hard links) or ``copy``."""
    src, dst = Path(src).resolve(), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    elif mode == "copy":
        shutil.copyfile(src, dst)
    else:
        raise ValueError(f"unknown link mode {mode!r}")


def view_of(stem: str, cc_prefix: str = "cc_") -> str:
    """Dataset naming rule: CC files carry the ``cc_`` prefix, everything else is MLO."""
    return "CC" if stem.startswith(cc_prefix) else "MLO"


def semantic_paths(root: PathLike, index: pd.DataFrame) -> pd.DataFrame:
    """Add absolute ``image`` / ``mask`` columns to a semantic ``index.csv`` frame (keeps existing ones)."""
    root = Path(root)
    df = index.copy()
    if "image" not in df:
        df["image"] = [str(root / "images" / sp / f"{s}.png") for s, sp in zip(df.stem, df.split)]
    if "mask" not in df:
        df["mask"] = [str(root / "masks" / sp / f"{s}.png") for s, sp in zip(df.stem, df.split)]
    return df


@dataclass
class NnUNetExporter:
    """Semantic dataset -> nnU-Net v2 raw dataset.

    Train and validation images go to ``imagesTr``; which of them validate is decided by ``splits_final.json``
    (written to ``<preprocessed>/<dataset>/``). Fold 0 is the dataset's own train/val split. Test images go to
    ``imagesTs`` and are never preprocessed.
    """

    raw_dir: Path
    preprocessed_dir: Path
    dataset_id: int
    dataset_name: str
    description: str = ""
    link_mode: str = "copy"

    @property
    def dataset_folder(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.dataset_folder

    def dataset_json(self, n_training: int) -> Dict[str, object]:
        return {"channel_names": {"0": "greyscale"},
                "labels": {name: value for value, name in LABEL_VALUES.items()},
                "numTraining": n_training, "file_ending": ".png",
                "overwrite_image_reader_writer": "NaturalImage2DIO",
                "description": self.description}

    def export(self, index: pd.DataFrame, splits: Optional[List[Dict[str, List[str]]]] = None,
               extra_test_dirs: Optional[Mapping[str, Sequence[str]]] = None) -> Path:
        """Write the raw dataset from an index with ``stem, split, image, mask`` columns.

        ``splits`` replaces the default fold list (used by cross-validation); ``extra_test_dirs`` maps a folder
        name (e.g. ``imagesTest_fold3``) to the stems linked into it (unlabelled images allowed, ``mask`` may be
        empty for them).
        """
        ds = self.dataset_dir
        if ds.exists():
            shutil.rmtree(ds)
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (ds / sub).mkdir(parents=True)
        labelled = _labelled(index)
        by_split = {sp: sorted(labelled.stem[labelled.split == sp]) for sp in ("train", "val", "test")}
        for r in labelled.itertuples(index=False):
            tr = "Ts" if r.split == "test" else "Tr"
            link_or_copy(r.image, ds / f"images{tr}" / f"{r.stem}_0000.png", self.link_mode)
            link_or_copy(r.mask, ds / f"labels{tr}" / f"{r.stem}.png", self.link_mode)
        image_of = dict(zip(index.stem, index.image))
        for folder, stems in (extra_test_dirs or {}).items():
            for s in stems:
                link_or_copy(image_of[s], ds / folder / f"{s}_0000.png", "symlink")
        n_tr = len(by_split["train"]) + len(by_split["val"])
        (ds / "dataset.json").write_text(json.dumps(self.dataset_json(n_tr), indent=2))
        self.write_splits(index, splits)
        return ds

    @property
    def splits_file(self) -> Path:
        return self.preprocessed_dir / self.dataset_folder / "splits_final.json"

    def write_splits(self, index: pd.DataFrame, splits: Optional[List[Dict[str, List[str]]]] = None) -> Path:
        """Write ``<preprocessed>/<dataset>/splits_final.json``: ``splits``, else fold 0 = the index's own
        train / val split. Every named image must be a training image of the raw dataset (``imagesTr``)."""
        labelled = _labelled(index)
        if splits is None:
            splits = [{"train": sorted(labelled.stem[labelled.split == "train"]),
                       "val": sorted(labelled.stem[labelled.split == "val"])}]
        tr_dir = self.dataset_dir / "imagesTr"
        tr_stems = ({p.name[:-len("_0000.png")] for p in tr_dir.glob("*_0000.png")} if tr_dir.is_dir()
                    else set(labelled.stem[labelled.split != "test"]))
        for k, sp in enumerate(splits):
            stray = (set(sp["train"]) | set(sp["val"])) - tr_stems
            if stray:
                raise ValueError(f"split {k} names {len(stray)} image(s) that are not in imagesTr "
                                 f"(e.g. {sorted(stray)[0]})")
        path = self.splits_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(splits, indent=2))
        return path


def _labelled(index: pd.DataFrame) -> pd.DataFrame:
    """The rows of a dataset index that have a mask."""
    return index[index["mask"].fillna("").astype(str) != ""] if "mask" in index else index
