# Building the dataset, training and cross-validation

The four segmentation models of the study (SegFormer-B3, nnU-Net ResEnc-M, YOLO26m-seg, U-Net) are trained on the
VinDr-Mammo cohort of `labels/vindr-mammo`. `breast-seg build-dataset` turns the VinDr-Mammo DICOM files and the
released labels into the training data, `configs/train/` holds one recipe per model and `configs/cv/` the 10-fold
cross-validation of each. Every recipe is in [MODELS.md](MODELS.md); metric definitions are in
[EVALUATION.md](EVALUATION.md).

## Setup

```bash
pip install -e ".[models,train,dicom]"   # YOLO26m-seg, U-Net, DICOM codecs
pip install -e ".[segformer]"            # SegFormer-B3 (transformers, timm)
pip install -e ".[nnunet]"               # nnU-Net v2
cp configs/paths.example.yaml configs/paths.yaml   # then edit (git-ignored)
```

`configs/paths.yaml` is the only place where local paths are written. Committed configs refer to its entries as
`${name}`: the fixed locations (`data_root`, `runs_dir`, `weights_dir`, `predictions_dir`, `results_dir`,
`nnunet_raw`, `nnunet_preprocessed`, `nnunet_results`) and the named inputs under `extra:` (the radiologist landmark
tables of the evaluation, `mlo_landmarks` / `cc_landmarks`). Relative values are resolved against `root`, which is
relative to the paths file. Another file can be selected with `BREAST_SEG_PATHS`; without any file every location
defaults to a folder under the working directory.

## Building the dataset

The images come from [VinDr-Mammo on PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/).

```bash
breast-seg build-dataset --dicom-root <VinDr-Mammo images> [--workers 8] [--paths configs/paths.yaml]
                         [--labels labels/vindr-mammo] [--out <data_root>/vindr]
```

The DICOM file of every labelled image is found anywhere under `--dicom-root` by its SOPInstanceUID (the file name
`images/<study>/<image>.dicom` of the PhysioNet download, `<study>/<series>/<sop>.dcm`, or the header of the files
not named after a labelled image; symbolic links to folders are not followed). The canvases are identical to the
model inputs of the study; the label maps are the released (cleaned and smoothed) annotations, reproduced from
their polygons up to rasterisation (Dice about 0.999 for breast tissue and pectoral muscle, about 0.997 for the
small nipple region):

* **Canvas** (`baseline640`): VOI LUT, MONOCHROME1 inverted, the crop of `canvas.csv`, min-max to 8 bit, bilinear
  resize to the size the padding of `canvas.csv` leaves, zero padding to 640 x 640. The crop is taken from
  `canvas.csv`, never recomputed.
* **Label map**: the polygons mapped onto the canvas (`canvas = (dicom - crop) * scale + pad`) and filled in the
  order Breast = 2, Pectoral muscle = 1, Nipple = 3; the padding carries no label.

The defaults come from the paths file, so the training configs find what the command writes: `--out` is
`<data_root>/vindr` (the `${data_root}/vindr/...` of `configs/train/`), `--labels` is `labels/vindr-mammo` under its
`root` (the repository), and the nnU-Net dataset goes to its `nnunet_raw` / `nnunet_preprocessed`, where the nnU-Net
trainer reads it. Written (stem = SOPInstanceUID for MLO, `cc_<SOPInstanceUID>` for CC; `<split>` = `train` / `val` /
`test` from the `Split` column):

```
<data_root>/vindr/
  canvases/<stem>.png                       640 x 640 grey canvases
  label_maps/<stem>.png                     0 background, 1 pectoral muscle, 2 breast tissue, 3 nipple
  yolo/{images,labels}/<split>/, data.yaml  YOLO26m-seg: polygons of the label maps
  semantic/{images,masks}/<split>/, index.csv   SegFormer-B3, U-Net (and the source of the nnU-Net datasets)
  folds.csv                                 stem, view, study, side, split, fold (the 10-fold cross-validation)
  metadata_mlo.csv, metadata_cc.csv         per-image geometry of the evaluation (canvas <-> DICOM, mm, laterality)
  audit.json                                counts per split, identical canvases and every problem found
<nnunet_raw>/Dataset601_BreastSeg/          nnU-Net v2 raw dataset (train + validation in imagesTr, test in imagesTs)
<nnunet_preprocessed>/Dataset601_BreastSeg/splits_final.json   fold 0 = the held-out train / validation split
```

The layouts hard-link the canvases and label maps (they are copied where the file system has no hard links). A
rebuild replaces only `Dataset601_BreastSeg` in the nnU-Net folders; other nnU-Net datasets there are kept.
**Audit**: every DICOM readable and of the matrix size of the label table, every file written, canvas and
label-map shape and values, padding on the side away from the chest wall, breast tissue on every image, pectoral
muscle on every MLO and on no CC view, no study in two splits or two folds, and no pixel-identical canvas in two
splits or two folds. An image that fails is reported and the others are still processed. The layouts and tables
are written only when the audit passes; otherwise `DatasetAuditError` (exit code 1) lists every problem and
`audit.json` keeps them. An image whose DICOM file is missing is a problem too, unless `--allow-missing` builds
from the images that were found. A full build gives 1,186 / 131 / 146 pairs in train / val / test.

```python
from breast_seg.config import PathsConfig
from breast_seg.datasets import VinDrDatasetBuilder

paths = PathsConfig.load()
report = VinDrDatasetBuilder("<VinDr-Mammo images>", "labels/vindr-mammo", paths.resolve("${data_root}/vindr"),
                             workers=8, nnunet_raw_dir=paths.dir("nnunet_raw"),
                             nnunet_preprocessed_dir=paths.dir("nnunet_preprocessed")).build()
```

Without `nnunet_raw_dir` / `nnunet_preprocessed_dir` the class writes the nnU-Net dataset to `<output>/nnunet/raw`
and `<output>/nnunet/preprocessed`.

## Config files

* **Inheritance.** `base: other.yaml` (or a list) merges the base first; nested mappings merge key by key, lists
  are replaced. Files starting with `_` are fragments meant only as bases.
* **Overrides.** Dotted `key=value` strings, the value parsed as YAML: `optim.epochs=300`, `device=1`,
  `augment=[]`.
* **Validation.** Configs become typed dataclasses; unknown keys, wrong types and missing required values raise a
  `ConfigError` naming the dotted key.

```python
from breast_seg.training import load_train_config
cfg = load_train_config("configs/train/segformer_b3.yaml", ["device=1"])
```

## Training

```bash
breast-seg train configs/train/segformer_b3.yaml --set device=0
breast-seg predict configs/train/segformer_b3.yaml --split test     # label maps + nipple fallbacks
breast-seg export configs/train/segformer_b3.yaml                   # -> <weights_dir>/segmentation/segformer_b3
```

```python
from breast_seg.training import build_trainer

trainer = build_trainer("configs/train/yolo26m_seg.yaml", overrides=["device=0"])
trainer.prepare()          # writes <runs_dir>/<name>/data.yaml and train_config.json; trains nothing
result = trainer.fit()     # result.weights = the selected checkpoint
trainer.evaluate("val")    # family metrics on a split (see EVALUATION.md)
trainer.predict(trainer.split_images("test"), out_dir)   # common label-map format
trainer.export()           # copy the checkpoint to <export.dir>/<export.filename> (<weights_dir>/segmentation/...)
```

The run directory is `<runs_dir>/<name>`. `predict` writes one label map per image and, for a view whose label map
has no nipple pixel, its nipple fallback (`nipple/<stem>`), the input of the nipple rule of the evaluation. Family
notes:

* **YOLO26m-seg (`yolo_seg`).** Base weights are looked up under `weights_dir` and otherwise downloaded by
  Ultralytics. Only non-null `hyp` values are passed to `YOLO.train`, everything else keeps the Ultralytics
  default. The Albumentations patch (`albumentations.enabled`) swaps Ultralytics' default pixel-level list for the
  configured one while training runs; it works for single-device training in the current process (a multi-GPU DDP
  launch starts new interpreters that do not see it).
* **SegFormer-B3, U-Net.** One PyTorch loop: AdamW, polynomial decay per iteration, autocast
  (`optim.amp_dtype`), validation every epoch, early stopping on the monitored metric (mean foreground Dice),
  `best` and `last` checkpoints, `history.csv`. Albumentations is needed when `augment` is not empty.
* **nnU-Net ResEnc-M.** `prepare()` uses the raw dataset of `build-dataset` (or converts the semantic layout when
  it is missing), writes `splits_final.json` again from the semantic index (fold 0 = the held-out split, also when
  the raw dataset is reused), runs `nnUNetv2_plan_and_preprocess`, `nnUNetv2_plan_experiment -pl nnUNetPlannerResEncM`, derives
  the full-canvas plans (`patch_size`, `batch_size`), preprocesses with them and installs `nnUNetTrainerMammo` into
  the nnU-Net package. `fit()` runs `nnUNetv2_train` (continuing from `checkpoint_latest.pth` when a run was
  interrupted). The three nnU-Net locations come from the paths file and are exported to every command.

Training a recipe exactly as the config states reproduces the training *procedure*; bit-identical weights also
depend on library versions, hardware and non-deterministic kernels.

## Cross-validation

```bash
breast-seg cv configs/cv/cv10_segformer_b3.yaml                  # every fold, merge, evaluate
breast-seg cv configs/cv/cv10_segformer_b3.yaml --fold 3         # one fold (train + predict)
breast-seg cv configs/cv/cv10_segformer_b3.yaml --no-train --no-predict --out outputs/results --name cv10_segformer_b3
```

* **Folds are read, never redrawn**, from `folds.csv` (the `Fold` column of the released labels): both breasts of
  a study share a fold, so every model is compared case by case on the same folds.
* **Protocol** of fold `i`: test = fold `i`, validation = fold `(i + 1) % 10`, train = the other eight folds.
* **Per family.** YOLO folds are trees of links (`images/` and `labels/`), the PyTorch families get a fold index
  with absolute paths, nnU-Net gets one shared dataset (`Dataset603_BreastSegCV10`) with one `splits_final.json`
  entry per fold and `imagesTest_fold<i>` folders; it transfers the plans of `Dataset601_BreastSeg`, so plan the
  held-out nnU-Net first (`breast-seg train configs/train/nnunet_resenc_m.yaml --prepare-only`).
* **Outputs.** Fold runs are `<runs_dir>/<cv name>_fold<i>`; each fold model predicts only its test fold into
  `<predictions_dir>/<cv name>_fold<i>`; `merge()` copies them (with their nipple fallbacks) into
  `<predictions_dir>/<cv name>` and fails if an image has no prediction from its own fold; `evaluate()` scores the
  merged directory and adds a per-fold decision table. Finished folds are skipped (`skip_finished`), so a run can be
  restarted.
* **Configs.** `cv10_segformer_b3`, `cv10_nnunet`, `cv10_yolo26m` and `cv10_unet` reuse the training recipes;
  SegFormer-B3, U-Net and YOLO26m-seg run the full 300-epoch schedule in every fold (patience 300, the checkpoint
  still selected on the fold's validation fold).
