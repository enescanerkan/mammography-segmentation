# Evaluation

Two layers are measured and should always be read together:

1. **Mask quality**: per-class Dice and IoU of label maps, by view.
2. **Downstream measurements**: millimetre errors against the reference landmarks and the PNL positioning
   decision. A model with an excellent pectoral Dice can still misplace the lower end of the muscle by a few
   pixels and shift the posterior nipple line by much more; the downstream layer is the arbiter.

Everything lives in `breast_seg.evaluation` (numpy / pandas only).

## Prediction format

One `uint8` label map per image, `<stem>.png`, same size as the canvas: `0 background, 1 pectoral,
2 breast-tissue, 3 nipple`. Semantic models write it directly. Instance models (YOLO) are flattened by
`instances_to_label_map`: instances above the confidence threshold (0.25) are unioned per class and painted breast
tissue, pectoral, nipple. A class with no instance above the threshold gets its single best instance back when
that one is above a floor (0.01), because semantic models have no confidence gate and every mammogram has a
nipple; the rescue is not applied to the pectoral class of a CC view. CC images are recognised by the `cc_` file
prefix. Points of a landmark model are given to the evaluator as a long table (`stem, role, x, y` in canvas
pixels; the pose roles `pectoral_top` / `pectoral_bottom` are read as `pec_top` / `pec_bot`); no command of the
package writes that table, it is passed as `landmarks_csv` (see below).

A view whose label map has no nipple pixel also gets its **nipple fallback** from the same forward pass, in
`nipple/<stem>` next to the label maps (`write_nipple_fallback` / `read_nipple_fallback`): the nipple probability
map of a semantic model (`.npz`, float16; nnU-Net through `--save_probabilities`) or YOLO's most confident nipple
instance at confidence 0.001 (`.png`, 0 / 1). `breast-seg predict` and the cross-validation write them; the
out-of-fold merge copies them with the label maps.

## Mask metrics

* `dice_iou(pred, ref)`: per image and class, `Dice = 2|P∩G| / (|P| + |G|)`, `IoU = |P∩G| / |P∪G|`. A class
  absent from both maps is `NaN` (no evidence) and is skipped in means. Means are reported by view; the pectoral
  class is not reported on CC views.
* `PooledDice`: intersections and sums pooled over a whole split before dividing; the validation metric of the
  PyTorch training loops (`mean_fg` = mean over the foreground classes).

## Landmarks and distances

All distances are measured in the original image grid and converted with the pixel spacing (mm per pixel).
`CanvasTransform` maps canvas points back: `original = (canvas - pad) / scale + crop`.

* **Nipple** (`nipple_point`, no minimum size): the centroid of the largest 8-connected nipple component of the
  label map (`nipple_source = argmax`); without nipple pixels, the nipple fallback of the same forward pass: the
  probability map binarised at 0.02, centroid of its largest component (`low_confidence`), or its highest pixel
  when no pixel reaches 0.02 (`max_probability`); YOLO's low-confidence instance mask, centroid of its largest
  component (`low_confidence`). Without either the nipple is *not found* (`none`).
* **Pectoral line** = the line fitted to the pectoral / breast-tissue border (`breast_seg.geometry.fit_pectoral_line`);
  for landmark models the two predicted pectoral points.
* **PNL (MLO)** = perpendicular distance from the nipple to the *infinite* line through the two pectoral points.
* **D_CC (CC)** = horizontal distance from the nipple to the chest-wall image edge: `x` for a left breast,
  `width - x` for a right breast. The breast mask does not enter this measurement, so `e_cc` is effectively the
  horizontal nipple error.
* **MLO-only criterion** (`mlo_only_good`): an MLO view is `good` when the foot of the perpendicular from the
  nipple to the infinite pectoral line lies inside the original image, widened by 0.25 % of its longest side
  (`MLO_ONLY_MARGIN`), and `poor` otherwise; computed for the model (`mlo_only`) and for the reference
  landmarks (`gt_mlo_only`).
* Per-image errors (`measure_mlo`, `measure_cc`): `e_nipple`, `e_pec_top`, `e_pec_bot`, `e_pnl = |PNL - PNL_ref|`,
  `e_cc = |D_CC - D_CC_ref|`. The reference value is always recorded, even when the model produced nothing, so all
  models are scored on the same reference subset. `e_pec_top` compares two different definitions of the upper
  pectoral point (where the annotator stops versus the fitted line's end) and is descriptive, not an error.
* Optional contour fallback (`MeasureOptions(fallback=True)`, off in the paper's evaluation): without a nipple,
  measure at the breast-tissue contour point that reaches farthest from the pectoral line (MLO) or the chest wall
  (CC); such rows carry `fallback = True`. It replaces only a missing nipple, never a found one.

Reference landmarks come from the landmark tables (`GroundTruth.from_csv`): a `Nipple` box (centre) and
`Pectoralis` vertices (smallest / largest `y` = top / bottom) per image, a `qualitativeLabel` (good / bad) per
pair:

| Set | Pairs | Reference |
|---|---|---|
| VinDr-Mammo | 1,463 (test 146) | radiologist landmarks (`mlo_landmarks` / `cc_landmarks` of the paths file, published with [mammography-positioning-yolo](https://github.com/enescanerkan/mammography-positioning-yolo/tree/main/data/labels)); reference masks: the polygons of `labels/vindr-mammo` |
| EMBED | 203 | radiologist landmarks + segmentation polygons (`labels/external/embed_*`) |
| CMMD | 97 | radiologist landmarks only (`labels/external/cmmd_*`) |

On EMBED, D_CC is measured towards the side of the stored pixel array that holds the chest wall (`ChestWallSide`,
see [labels/README.md](../labels/README.md#external-test-sets)).

## The PNL rule

For the MLO/CC pair of one breast, `delta = |PNL_MLO - D_CC|`; the pair is **good** when `delta <= 10 mm`.

Two scorings against the reference label (`PNLRule`, `FLAT_RULE`, `TOLERANCE_RULE`):

* **Flat 10 mm**: the rule itself.
* **+-1 mm tolerance**: the threshold depends on the reference class; a reference-good pair is correct when
  `delta <= 11`, a reference-bad pair when `delta > 9`, so a measurement within `(9, 11]` agrees with either
  label. Because it looks at the reference, this is an evaluation tolerance, not a decision rule; report it next to
  the flat rule, never instead of it.

Counting convention (both scorings): a pair without a measurement is never correct and never leaves a
denominator. Unanswered reference-good pairs are false negatives; unanswered reference-bad pairs stay outside the
confusion matrix but inside the specificity denominator:

```
Recall      = TP / (TP + FN)
Specificity = TN / (TN + FP + U_bad)
Accuracy    = (TP + TN) / n
```

`n_total` adds pairs that could not be matched at all to the denominator, as failures. If the reference label is
itself derived from the expert landmarks with the same rule, decision accuracy is a thresholded view of the mm
errors rather than an independent endpoint; report the mm errors alongside it.

## Paired comparison and uncertainty

* `mcnemar_exact(correct_a, correct_b)`: exact McNemar test (two-sided binomial on the discordant pairs) for two
  models scored on the same pairs; `paired_mcnemar` joins two per-pair tables on `(group, side)` and checks that
  their reference labels agree.
* `group_bootstrap(df, "group", statistic)`: percentile confidence interval resampling whole studies (both breasts
  of a study together).

## Out-of-fold aggregation

In cross-validation each image is predicted by the single fold model that never trained on it.
`merge_oof_predictions(fold_dirs, out_dir, expected)` copies every fold's own test images into one directory,
ignores predictions a fold made for images it does not own and fails when an image is missing; the merged
directory is then scored once like any other prediction directory.

## Running an evaluation

```python
from breast_seg.config import PathsConfig
from breast_seg.evaluation import SegmentationEvaluator, load_evaluation_config

cfg = load_evaluation_config("configs/cv/_cv10_base.yaml", ["evaluation.pairs_split=Test"])
result = SegmentationEvaluator.from_config(cfg, PathsConfig.load()).evaluate("outputs/predictions/<model>")
result.save("outputs/results", "<model>")      # <model>_views.csv, <model>_pairs.csv, <model>_summary.json
```

The `evaluation:` section names the label tables, the canvas metadata, the semantic dataset providing reference
masks and how pairs are formed (`labels`: same study and side in the label tables, optionally restricted to one
`Split`; `fold_table`: a pair-level table). Landmark-model output is scored with
`SegmentationEvaluator.from_config(cfg, paths, landmarks_csv=...)`. The summary holds Dice by view, the
distribution of each error (mean, median, 90th percentile, count), the MLO-only agreement with the reference
(`mlo_only`: images, answered, agreement, reference-good and predicted-good counts), both decision scorings and
the error of `delta` itself against the reference `delta`.

```bash
breast-seg evaluate --config configs/cv/cv10_segformer_b3.yaml --preds outputs/predictions/segformer_b3_test \
    --set pairs_split=Test --out outputs/results --name segformer_b3_test
```
