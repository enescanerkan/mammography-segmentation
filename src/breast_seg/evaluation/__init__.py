"""Evaluation: mask metrics, landmark distances, the nipple and pectoral line rules, the ΔD decision, the MLO-only
criterion, paired comparison, out-of-fold aggregation and the end-to-end scorer of a prediction directory. Pure
numpy / pandas / OpenCV (no framework imports)."""
from breast_seg.evaluation.evaluator import EvaluationConfig, EvaluationResult, SegmentationEvaluator, load_evaluation_config
from breast_seg.evaluation.gt import GroundTruth
from breast_seg.evaluation.landmarks import (
    MLO_ONLY_MARGIN,
    NIPPLE_PROBABILITY_THRESHOLD,
    CanvasTransform,
    ImageGeometry,
    centroid,
    chest_wall_mm,
    distance_mm,
    mlo_only_good,
    nipple_point,
    pnl_mm,
    point_line_distance,
)
from breast_seg.evaluation.masks import PooledDice, dice_iou
from breast_seg.evaluation.measure import MeasureOptions, measure_cc, measure_mlo
from breast_seg.evaluation.oof import OOFReport, merge_oof_predictions
from breast_seg.evaluation.pairs import pairs_from_fold_table, pairs_from_labels, score_pairs
from breast_seg.evaluation.predictions import (
    clear_nipple_fallback,
    instances_to_label_map,
    read_nipple_fallback,
    write_nipple_fallback,
)
from breast_seg.evaluation.rule import FLAT_RULE, THRESHOLD_MM, TOLERANCE_RULE, PNLRule, decision_summary
from breast_seg.evaluation.stats import McNemarResult, group_bootstrap, mcnemar_exact, paired_mcnemar

__all__ = [
    "FLAT_RULE",
    "MLO_ONLY_MARGIN",
    "NIPPLE_PROBABILITY_THRESHOLD",
    "THRESHOLD_MM",
    "TOLERANCE_RULE",
    "CanvasTransform",
    "EvaluationConfig",
    "EvaluationResult",
    "GroundTruth",
    "ImageGeometry",
    "McNemarResult",
    "MeasureOptions",
    "OOFReport",
    "PNLRule",
    "PooledDice",
    "SegmentationEvaluator",
    "centroid",
    "chest_wall_mm",
    "clear_nipple_fallback",
    "decision_summary",
    "dice_iou",
    "distance_mm",
    "group_bootstrap",
    "instances_to_label_map",
    "load_evaluation_config",
    "mcnemar_exact",
    "measure_cc",
    "measure_mlo",
    "merge_oof_predictions",
    "mlo_only_good",
    "nipple_point",
    "paired_mcnemar",
    "pairs_from_fold_table",
    "pairs_from_labels",
    "pnl_mm",
    "point_line_distance",
    "read_nipple_fallback",
    "score_pairs",
    "write_nipple_fallback",
]
