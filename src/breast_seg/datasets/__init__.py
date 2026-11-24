"""Training data: the VinDr-Mammo builder, the YOLO / semantic / nnU-Net layouts, label primitives and splits.

    from breast_seg.datasets import VinDrDatasetBuilder
    VinDrDatasetBuilder("/data/vindr-mammo", "labels/vindr-mammo", "data/vindr").build()
"""
from breast_seg.datasets.builder import (
    BuildReport,
    DatasetAuditError,
    LabelledImage,
    VinDrDatasetBuilder,
    find_dicoms,
    make_canvas,
    make_label_map,
    read_labels,
)
from breast_seg.datasets.formats import NnUNetExporter, semantic_paths, write_yolo_data_yaml
from breast_seg.datasets.labels import masks_to_lines, parse_polygons, polygons_to_masks, rasterise, read_polygons
from breast_seg.datasets.splits import FoldAssignment, FoldSplit, LeakageError, find_leaks, folds_from_pairs

__all__ = [
    "BuildReport",
    "DatasetAuditError",
    "FoldAssignment",
    "FoldSplit",
    "LabelledImage",
    "LeakageError",
    "NnUNetExporter",
    "VinDrDatasetBuilder",
    "find_dicoms",
    "find_leaks",
    "folds_from_pairs",
    "make_canvas",
    "make_label_map",
    "masks_to_lines",
    "parse_polygons",
    "polygons_to_masks",
    "rasterise",
    "read_labels",
    "read_polygons",
    "semantic_paths",
    "write_yolo_data_yaml",
]
