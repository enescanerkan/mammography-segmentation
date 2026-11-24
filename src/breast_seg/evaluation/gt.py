"""Reference annotations from the landmark label tables.

The tables (one for MLO, one for CC) have one row per annotation with at least ``StudyInstanceUID``,
``SOPInstanceUID``, ``labelName``, ``data``, ``qualitativeLabel`` and ``SeriesDescription`` columns:

* ``labelName == "Nipple"``: ``data`` is a box ``{"x", "y", "width", "height"}``; the landmark is its centre.
* ``labelName == "Pectoralis"``: ``data`` holds ``{"vertices": [[x, y], ...]}``; the top / bottom pectoral
  points are the vertices with the smallest / largest ``y``.

Coordinates are in the original image grid. ``qualitativeLabel`` (good / bad) is the reference positioning label of
the MLO/CC pair; ``SeriesDescription`` starts with the side (``L-CC``, ``R-MLO``...).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

MLO_ROLES = ("nipple", "pec_top", "pec_bot")


def parse_data(raw: Any) -> Optional[dict]:
    """The ``data`` cell as a dict, or ``None`` when empty / unparsable."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    try:
        out = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return None
    return out if isinstance(out, dict) else None


def box_centre(d: Mapping[str, Any]) -> np.ndarray:
    return np.array([float(d["x"]) + float(d["width"]) / 2, float(d["y"]) + float(d["height"]) / 2])


def mlo_landmarks(labels: pd.DataFrame, complete_only: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """``{sop: {"nipple", "pec_top", "pec_bot"}}``; with ``complete_only`` images missing one of them are dropped."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sop, rows in labels.groupby("SOPInstanceUID"):
        d: Dict[str, np.ndarray] = {}
        for name, raw in zip(rows["labelName"], rows["data"]):
            v = parse_data(raw)
            if v is None:
                continue
            if name == "Nipple":
                d["nipple"] = box_centre(v)
            elif name == "Pectoralis":
                pts = sorted((tuple(map(float, q)) for q in v.get("vertices", [])), key=lambda q: q[1])
                if len(pts) >= 2:
                    d["pec_top"], d["pec_bot"] = np.array(pts[0]), np.array(pts[-1])
        if d and (not complete_only or set(MLO_ROLES) <= d.keys()):
            out[str(sop)] = d
    return out


def cc_nipples(labels: pd.DataFrame) -> Dict[str, np.ndarray]:
    """``{sop: nipple}`` from the CC table (the first parsable ``Nipple`` row of each image)."""
    out: Dict[str, np.ndarray] = {}
    rows = labels[labels["labelName"] == "Nipple"] if "labelName" in labels else labels
    for sop, raw in zip(rows["SOPInstanceUID"], rows["data"]):
        v = parse_data(raw)
        if v is not None and str(sop) not in out:
            out[str(sop)] = box_centre(v)
    return out


def quality_labels(labels: pd.DataFrame) -> Dict[str, str]:
    """``{sop: "good" | "bad"}`` (lower-cased ``qualitativeLabel``)."""
    out: Dict[str, str] = {}
    for sop, q in zip(labels["SOPInstanceUID"], labels["qualitativeLabel"]):
        if pd.notna(q) and str(sop) not in out:
            out[str(sop)] = str(q).strip().lower()
    return out


@dataclass
class GroundTruth:
    """Reference landmarks and labels of one cohort (original-grid coordinates)."""

    mlo: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    cc: Dict[str, np.ndarray] = field(default_factory=dict)
    quality: Dict[str, str] = field(default_factory=dict)
    mlo_table: Optional[pd.DataFrame] = None
    cc_table: Optional[pd.DataFrame] = None

    @classmethod
    def from_tables(cls, mlo_labels: pd.DataFrame, cc_labels: pd.DataFrame) -> GroundTruth:
        return cls(mlo_landmarks(mlo_labels), cc_nipples(cc_labels), quality_labels(cc_labels), mlo_labels, cc_labels)

    @classmethod
    def from_csv(cls, mlo_csv: Union[str, Path], cc_csv: Union[str, Path]) -> GroundTruth:
        return cls.from_tables(pd.read_csv(mlo_csv, dtype={"SOPInstanceUID": str}),
                               pd.read_csv(cc_csv, dtype={"SOPInstanceUID": str}))


#: Landmark-model role names -> the names used by the measurements.
ROLE_ALIASES = {"pectoral_top": "pec_top", "pectoral_bottom": "pec_bot"}


def landmarks_by_stem(table: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """``stem, role, x, y`` long table -> ``{stem: {role: point}}`` (``pectoral_top`` / ``pectoral_bottom`` of
    the pose models become ``pec_top`` / ``pec_bot``)."""
    return {str(s): {ROLE_ALIASES.get(r.role, r.role): np.array([float(r.x), float(r.y)]) for r in g.itertuples()}
            for s, g in table.groupby("stem")}
