# Model catalogue

The models of the study: what each one is, what it is fed, what it predicts, how it was trained, the config that
reproduces the recipe and the weight file the tools expect. The released weights are in the
[v1.0.0 release](https://github.com/enescanerkan/mammography-segmentation/releases/tag/v1.0.0)
(`python packaging/weights.py weights`, see [APP.md](APP.md#weights-folder)).

This page describes recipes only; it contains no results. Layer diagrams of every architecture are in the
[README](../README.md#models) and in [`docs/figures/architectures/`](figures/architectures).

## Conventions

* **Canvas.** All models except MedSAM2 work on a 640 x 640 canvas made from the DICOM: VOI LUT, MONOCHROME1
  inversion, breast bounding box, crop, min-max normalisation of the crop to 8 bit, aspect-preserving resize to a
  longest side of 640 and black padding on the side away from the chest wall. Per-image metadata (crop offset,
  scale, padding, pixel spacing, laterality) maps canvas points back to the original image; for the VinDr-Mammo
  cohort it is `labels/vindr-mammo/canvas.csv`. Two input recipes exist: `baseline640` (the canvas as is) and
  `wavelet640` (the canvas enhanced by non-local-means denoising, a Daubechies-4 level-2 detail boost and CLAHE,
  applied before padding).
* **Segmentation classes.** YOLO class ids `0 pectoral`, `1 breast-tissue`, `2 nipple`. Semantic label maps use
  `0 background`, `1 pectoral`, `2 breast-tissue`, `3 nipple`; instance masks are flattened in the order breast
  tissue, pectoral over it, nipple on top (see `breast_seg.config.PAINT_ORDER`).
* **Landmarks.** MLO: `nipple`, `pectoral_top`, `pectoral_bottom`; CC: `nipple`.
* **Anatomy-safe augmentation.** Horizontal flips are allowed (left and right breasts both occur), vertical flips
  never (the pectoral muscle is always at the top of an MLO view).
* **Configs.** `configs/train/<name>.yaml`, loaded by `breast_seg.training.load_train_config`; files starting with
  `_` are shared fragments inherited through `base:`. `configs/cv/cv10_<model>.yaml` runs the 10-fold
  cross-validation of the same recipe.
* **Data.** Every segmentation model is trained on the VinDr-Mammo cohort as `breast-seg build-dataset` writes it
  (see [TRAINING.md](TRAINING.md)): held-out split Train / Validation / Test = 1,186 / 131 / 146 MLO–CC pairs, MLO and
  CC in one model (CC images carry the `cc_` file prefix).

## Summary

| Model | Family | Training config | CV config | Weight file (release v1.0.0) |
|---|---|---|---|---|
| SegFormer-B3 | `segformer` | `segformer_b3.yaml` | `cv10_segformer_b3.yaml` | `segmentation/segformer_b3/` (Hugging Face folder) |
| nnU-Net ResEnc-M | `nnunet` | `nnunet_resenc_m.yaml` | `cv10_nnunet.yaml` | not released (nnU-Net model export `nnunet_resenc_m_fold0.zip`) |
| YOLO26m-seg | `yolo_seg` | `yolo26m_seg.yaml` | `cv10_yolo26m.yaml` | `segmentation/yolo26m_seg.pt` |
| U-Net | `unet` | `unet.yaml` | `cv10_unet.yaml` | `segmentation/unet.pt` |
| YOLO26l-pose, MLO / CC | (landmark models of the app) | – | – | `positioning/yolo26l_pose_mlo.pt`, `positioning/yolo26l_pose_cc.pt` |
| MedSAM2 fine-tune | (annotation toolkit) | – | – | `medsam2/medsam_model_best.pth` (not distributed) |

`breast-seg export` copies the selected checkpoint of a run to `<weights_dir>/segmentation/<weight file>`, the
path `configs/models.yaml` and the release use (`export.dir` of the training config). Inside a run
directory the checkpoints are `weights/best.pt` (YOLO), `best.pt` / `last.pt` (U-Net), `best/` / `last/`
(SegFormer) and `<nnunet_results>/Dataset601_BreastSeg/<trainer>__<plans>__2d/fold_0/checkpoint_final.pth`
(nnU-Net).

## SegFormer-B3

* **Architecture.** SegFormer with the MiT-B3 hierarchical transformer encoder (`nvidia/mit-b3`, ImageNet
  pre-trained) and the all-MLP decoder, four output classes; logits at 1/4 resolution are upsampled bilinearly to
  the canvas.
* **Input.** `baseline640` grey canvas repeated on three channels, ImageNet mean / std normalisation.
* **Training recipe.** Cross-entropy + soft Dice (foreground classes), AdamW `lr` 6e-5, weight decay 0.01,
  polynomial decay (power 1) per iteration, batch 16, 300 epochs, early stopping after 50 epochs without
  improvement of the validation mean foreground Dice (cross-validation: patience 300, the full schedule), bf16
  autocast, seed 42. Augmentation: horizontal flip, affine (rotation 8, scale 0.8-1.2, translation 5 %),
  brightness / contrast, gamma, CLAHE, Gaussian noise, blur / median blur, sharpen. Prediction averages the softmax
  with the horizontally flipped pass.
* **Dataset.** The semantic layout (`<data_root>/vindr/semantic`).

## nnU-Net ResEnc-M

* **Architecture.** nnU-Net v2 residual-encoder U-Net, ResEnc-M preset (`nnUNetPlannerResEncM`), 2D
  configuration, deep supervision.
* **Deviations from nnU-Net's defaults** (`nnUNetTrainerMammo`, shipped in `breast_seg/training/nnunet_variants.py`
  and installed into nnU-Net by `prepare()`): the patch is the full 640 x 640 canvas with batch size 12 (plans
  `nnUNetResEncUNetMPlans_fullM`, same network as the preset; the planner's smaller patch would crop wide
  breasts), and mirroring is horizontal only, in training and at inference.
* **Training recipe.** 300 epochs of nnU-Net's fixed 250 iterations, no early stopping, prediction with
  `checkpoint_final.pth`. Everything else is nnU-Net's own pipeline.
* **Dataset.** `Dataset601_BreastSeg`: train + validation in `imagesTr`, test in `imagesTs`; `splits_final.json`
  fold 0 = the held-out train / validation split. The cross-validation uses one dataset, `Dataset603_BreastSegCV10`,
  with the plans of 601 transferred unchanged and one `splits_final.json` entry per fold.

## YOLO26m-seg

* **Architecture.** Ultralytics YOLO26m-seg (NMS-free detection head with mask prototypes and per-instance mask
  coefficients), initialised from the COCO weights `yolo26m-seg.pt`.
* **Input.** `baseline640` canvases.
* **Outputs.** Instances of pectoral, breast tissue and nipple. For evaluation they are flattened into a label map
  (confidence 0.25; if a class has no instance above it, its best instance is taken back when above 0.01, except
  the pectoral class on CC views).
* **Training recipe** (`_yolo_seg_base.yaml`). AdamW, `lr0` 0.001, batch 16, 640 input, AMP, seed 42, 300 epochs,
  patience 50 (cross-validation: patience 300, the full schedule); rotation 8 degrees, translate 0.05, scale 0.2,
  horizontal flip 0.5, no vertical flip, brightness jitter only (HSV 0 / 0 / 0.3), mosaic 0.5 closed for the last
  15 epochs, no mixup / copy-paste; only best / last checkpoints. Pixel-level Albumentations replace Ultralytics'
  default list (CLAHE, brightness / contrast, gamma, Gaussian / ISO / multiplicative / salt-and-pepper noise, blur,
  median blur, sharpen).
* **Dataset.** The YOLO layout (`<data_root>/vindr/yolo`): polygons of the label maps (external contours of at least
  20 px, `approxPolyDP` with 0.2 % of the perimeter); the breast-tissue polygon includes the nipple area.

## U-Net

* **Architecture.** Plain U-Net: four levels of double 3x3 convolution + BatchNorm + ReLU (widths 64-512,
  bottleneck 1024), max-pool down, transposed convolution up, concatenated skips, 1x1 head; random
  initialisation. The reference baseline without attention or residual blocks.
* **Input.** Single-channel canvas scaled to [0, 1] (no ImageNet statistics for a network trained from scratch).
* **Training recipe.** As SegFormer (loss, optimizer, schedule, augmentation, early stopping, flip TTA) except
  `lr` 1e-3 and batch 8.
* **Dataset.** The semantic layout (`<data_root>/vindr/semantic`).

## Landmark models (YOLO26l-pose, MLO / CC)

The landmark route of the review app. Ultralytics YOLO26l-pose, one class (`Breast`) whose box carries the
keypoints (MLO: nipple, pectoral_top, pectoral_bottom; CC: nipple), on `wavelet640` canvases (feeding raw canvases
is out of distribution). The posterior nipple line is the perpendicular from the nipple to the line through the
two pectoral points. The weights are released for the app; their training is not part of this repository.

## MedSAM2 fine-tune (prompted segmentation)

SAM2 (Hiera-tiny image encoder) fine-tuned with box prompts on 50 MLO and 50 CC images through the MedSAM2 branch
of the MedSAM code base. The annotation toolkit runs it in its SAM Label mode to turn a box or point prompt into a
mask (input: the image at 1024 x 1024, output: one 256 x 256 mask per prompt); the masks of `labels/vindr-mammo`
were made with it. As a prompted model it is not compared with the automatic models.
