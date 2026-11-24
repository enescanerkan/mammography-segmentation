# Labels

Annotations of the study, as CSV tables in the original DICOM pixel coordinates. Images are not included: they are
obtained from the source databases under their own terms, and every row is keyed by the DICOM `StudyInstanceUID` /
`SOPInstanceUID`.

| Folder | Cohort | Content |
|---|---|---|
| [`vindr-mammo/`](#vindr-mammo-segmentation) | VinDr-Mammo, 1,463 MLO–CC pairs (827 studies) | breast, pectoral muscle and nipple polygons of every image |
| [`external/`](#external-test-sets) | EMBED (203 pairs) and CMMD (97 pairs) | radiologist landmarks and the reference positioning label of each pair; on EMBED also the radiologist's breast, pectoral muscle and nipple polygons |

The radiologist landmarks of the VinDr-Mammo pairs (nipple, pectoral line and the positioning label of each pair),
which are the reference of every positioning measurement, are published with
[mammography-positioning-yolo](https://github.com/enescanerkan/mammography-positioning-yolo/tree/main/data/labels)
in the same CSV schema.

## VinDr-Mammo segmentation

`mlo_segmentation_labels.csv` and `cc_segmentation_labels.csv` hold one row per polygon; `canvas.csv` holds the 640 ×
640 model canvas of every image.

| Column | Meaning |
|---|---|
| `StudyInstanceUID`, `SOPInstanceUID` | VinDr-Mammo identifiers (`study_id`, `image_id` in the VinDr-Mammo tables) |
| `SeriesDescription` | `L-MLO`, `R-MLO`, `L-CC`, `R-CC`; an MLO and a CC with the same study and side form a pair |
| `annotationMode` | `polygon` |
| `labelName` | `Breast`, `Pectoral muscle` (MLO only) or `Nipple` |
| `data` | `{'vertices': [[x, y], ...]}`, closed polygon in original DICOM pixels |
| `height`, `width` | DICOM matrix size |
| `ImagerPixelSpacing` | pixel spacing in mm (row \ column) |
| `Split` | `Train` / `Validation` / `Test`: the held-out split (1,186 / 131 / 146 pairs) |
| `Fold` | 0–9: fold of the 10-fold cross-validation; both breasts of a study are in the same fold |

**Classes.** `Breast` is the outline of the breast region: the breast tissue and, on MLO, the pectoral muscle.
`Pectoral muscle` and `Nipple` are drawn on top of it. To make the label map of the study (0 background, 1 pectoral
muscle, 2 breast tissue, 3 nipple), fill `Breast` with 2, then `Pectoral muscle` with 1, then `Nipple` with 3.
Every image has one polygon of each class, except two images without a nipple polygon (one MLO, one CC).

**How they were made.** All masks were drawn by one annotator. 50 MLO and 50 CC images were annotated in
[CVAT](https://github.com/cvat-ai/cvat) with SAM 2; MedSAM2 was fine-tuned on them with box prompts, and every image
of the cohort was then annotated with the fine-tuned model in the annotation toolkit of this repository
(`tools/annotation`): a box or point prompt returns a mask, which was corrected as a polygon where needed. The
released polygons are the cleaned and smoothed annotations: each class keeps one connected region without holes,
boundaries are smoothed with a Gaussian filter (σ = 3 px on the canvas, nipple 2 px), unlabelled tissue between the
pectoral muscle and the breast tissue goes to the nearer class, on CC views the breast tissue reaches the chest-wall
edge of the image, and nothing lies in the zero padding of the canvas.

**The 640 canvas** (`canvas.csv`: `crop_x1`, `crop_y1`, `crop_x2`, `crop_y2`, `scale`, `pad_left`, `pad_top`,
`pad_right`, `pad_bottom`) is the input of the segmentation models: the breast crop, resized with its aspect ratio
kept to a longer side of 640 pixels, and zero-padded to 640 × 640.

```
canvas_x = (dicom_x − crop_x1) × scale + pad_left        dicom_x = (canvas_x − pad_left) / scale + crop_x1
canvas_y = (dicom_y − crop_y1) × scale + pad_top         dicom_y = (canvas_y − pad_top)  / scale + crop_y1
```

`breast-seg build-dataset` reads the VinDr-Mammo DICOM files and these tables and writes the canvases and the
label maps in the layouts of the four segmentation models (see [docs/TRAINING.md](../docs/TRAINING.md)).

```python
import ast
import cv2
import numpy as np
import pandas as pd

rows = pd.read_csv("labels/vindr-mammo/mlo_segmentation_labels.csv")
image = rows[rows.SOPInstanceUID == "2ddfad7286c2b016931ceccd1e2c7bbc"]
label = np.zeros((int(image.height.iloc[0]), int(image.width.iloc[0])), np.uint8)
for name, value in (("Breast", 2), ("Pectoral muscle", 1), ("Nipple", 3)):
    for data in image.loc[image.labelName == name, "data"]:
        points = np.round(np.array(ast.literal_eval(data)["vertices"])).astype(np.int32)
        cv2.fillPoly(label, [points], value)
# label[y, x] is the class of DICOM pixel (x, y)
```

## External test sets

`external/` holds the reference annotations of the two external cohorts (`Split` is always `Test`): landmark tables in
the schema of the VinDr-Mammo landmark tables, and for EMBED segmentation tables in the schema of the VinDr-Mammo
segmentation tables above.

| Files | Cohort |
|---|---|
| `cmmd_mlo_labels.csv`, `cmmd_cc_labels.csv` | Chinese Mammography Database (CMMD, TCIA): 97 pairs |
| `embed_mlo_labels.csv`, `embed_cc_labels.csv` | EMBED (Emory Breast Imaging Dataset, open subset): 203 pairs |
| `embed_mlo_segmentation_labels.csv`, `embed_cc_segmentation_labels.csv`, `embed_canvas.csv` | EMBED: breast, pectoral muscle and nipple polygons of the same 406 images, and their 640 canvas |

- `Nipple` rows (`annotationMode = bounding_box`): the box centre is the nipple point.
- `Pectoralis` rows (`annotationMode = line`): `vertices` = [upper end, lower end] of the pectoral muscle line.
- `qualitativeLabel` on the CC rows is the reference label of the pair: `Good` when |PNL<sub>MLO</sub> −
  D<sub>CC</sub>| ≤ 10 mm from the reference landmarks.
- One breast radiologist placed every landmark. On EMBED the radiologist also outlined the breast tissue, the
  pectoral muscle and the nipple; CMMD has landmarks only.
- The EMBED polygons are cleaned and smoothed like the VinDr-Mammo ones (one region per class, Gaussian smoothing,
  same classes and paint order; on CC the pectoral muscle is part of `Breast`). They are in original DICOM pixels
  and may extend beyond the 640 canvas crop, because they were drawn on the whole image. The two extra columns
  `ChestWallSide` and `FieldOfViewHorizontalFlip` are the same as in the EMBED landmark tables.
- EMBED stores part of its images mirrored: `ChestWallSide` (`L` / `R`) gives the side of the stored pixel array on
  which the chest wall lies, and `FieldOfViewHorizontalFlip` is DICOM (0018,7034) as stored. D<sub>CC</sub> is
  measured towards `ChestWallSide`. In VinDr-Mammo and CMMD the chest wall is always on the laterality side.
- `ImagerPixelSpacing` is DICOM (0018,1164); for the Hologic units of EMBED it is 0.070 mm (the
  magnification-corrected `PixelSpacing`, 0.065238 mm, is not used).

## License

The tables are released under the license of the repository (MIT). The images belong to VinDr-Mammo (PhysioNet),
CMMD (TCIA) and EMBED, each under its own terms.
