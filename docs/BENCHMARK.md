# Runtime benchmark

`breast-seg benchmark runtime` times one MLO–CC pair **end to end**: reading the two DICOM files (pydicom, pixel
decoding, VOI LUT), the pipeline's released preprocessing, the network(s), and the measurement rule (pectoral line
fit, nipple, PNL and D<sub>CC</sub>, `ΔD ≤ 10 mm`) up to the adequacy decision. The configurations of the paper's
runtime table:

| Configuration | Pipeline |
|---|---|
| `craunet` | CRAUNet, two view-specific landmark models (MLO + CC), 512 × 512, its own released preprocessing (one setting) |
| `segformer_b3_tta` / `_no_tta` | SegFormer-B3 with / without horizontal-flip test-time augmentation |
| `nnunet_tta` / `_no_tta` | nnU-Net ResEnc-M with / without its mirroring |
| `yolo26m` | YOLO26m-seg (one setting, no TTA) |
| `unet_tta` / `_no_tta` | U-Net with / without horizontal flip |

Precision is that of the evaluation code the table timed: on the GPU bfloat16 autocast for SegFormer-B3 and U-Net
(their training configs' `amp_dtype`), nnU-Net's own float16 autocast and float32 for YOLO; on the CPU float32 for
every pipeline, as stated in the paper. (The trainers' own `predict` autocasts on the CPU too; `--cpu-autocast`
times that instead.) nnU-Net mirrors as `nnUNetPredictor` does, over every combination of the checkpoint's mirroring
axes (horizontal only for the released model). D<sub>CC</sub> is measured to the chest-wall edge of the CC pixel
data, so images stored mirrored (`FieldOfViewHorizontalFlip=YES`) are measured from the correct side.

**Protocol.** All pipelines run in one process. Every pair runs every configuration, and the order of the
configurations rotates by one from pair to pair (seeded base order), so a background load hits all of them alike.
Five warm-up pairs are not measured; then all pairs are passed twice, one pair at a time, the two views in one
batched forward wherever one model serves both (`_unbatched` configurations run image by image). The process is
pinned to the *N* least-loaded physical cores and runs *N* threads (with `--cores` the thread count is the number of
cores given; if fewer free cores are found, the thread count is lowered and recorded). A GPU session checks its card for other compute
processes at the start, every 20 pairs and at the end, and stops with exit code 4 (`MEASUREMENT INVALID`) if it
finds one (`--allowed-pids` lists persistent processes that may hold memory on the card). A GPU index is the one
`nvidia-smi` shows (`CUDA_DEVICE_ORDER=PCI_BUS_ID`).

```bash
pip install -e ".[models,dicom,segformer,nnunet]"
# GPU 0 (keep the card otherwise idle)
breast-seg benchmark runtime configs/benchmark/runtime_paper.yaml --device 0 \
    --pairs configs/benchmark/vindr_heldout_pairs.csv --dicom-root <VinDr-Mammo images> \
    --craunet-code <CRAUNet repository> --craunet-weights <CRAUNet cv_results> --out runtime --name runtime_gpu
# CPU: 8 threads on the 8 least-loaded physical cores
breast-seg benchmark runtime configs/benchmark/runtime_paper.yaml --device cpu --threads 8 \
    --pairs configs/benchmark/vindr_heldout_pairs.csv --dicom-root <VinDr-Mammo images> --out runtime --name runtime_cpu
# report (and the paper-table layout of the last session per device)
breast-seg benchmark summary runtime/runtime_gpu.json runtime/runtime_cpu.json
```

Weights follow the training configs (the selected checkpoint of `<runs_dir>/<name>`, nnU-Net under
`<nnunet_results>`; `configs/paths.yaml`, see [TRAINING.md](TRAINING.md)); `--weights NAME=PATH` overrides one
pipeline, e.g. the released models: `--weights segformer_b3=weights/segmentation/segformer_b3 --weights
yolo26m=weights/segmentation/yolo26m_seg.pt --weights unet=weights/segmentation/unet.pt` (nnU-Net: the trained model
folder). CRAUNet is not part of this package: give the root of its released repository and its cross-validation
folder (`mlo/fold0/mlo_model.pth`, `cc/fold0/cc_model.pth`); without `--craunet-code` the `craunet` configuration is
skipped. It needs scikit-image, SciPy and torchvision (its own dependencies).

**Outputs.** `<name>.csv` has one row per pair and configuration (read / preprocessing / network / measurement
seconds, PNL, D<sub>CC</sub>, ΔD, decision, nipple source, load average). The stage split is informational: when a
view's label map has no nipple, the fallback is counted in the network stage for YOLO26m-seg (its second pass at
confidence 0.001) and in the measurement stage for SegFormer-B3, U-Net and nnU-Net (copying, respectively exporting,
the nipple probability of the same pass); a probability map that is zero everywhere gives no measurement.
`<name>.json` gives, per configuration, the seconds per pair as median [IQR] over all measurements, pairs per minute
(`60 × measurements / summed time`), every pass on its own, the spread between the passes (flagged above 10 %),
model size and GFLOPs, and the disturbed passes. `--stored <pipeline>=<csv>` (columns `study`, `side`, `delta_mm`)
checks every decision against stored per-pair predictions (checked at the start; a failed check is recorded, the
timing is kept); a `reference_delta_mm` column in the pairs file adds the decision accuracy.

**Disturbed passes.** A pass is *disturbed* when more than 5 % of its measurements take longer than 1.5 × the median
of the same configuration in the same session: something else on the machine slowed it down (pinning keeps the
benchmark on its cores, not other processes off them). The paper's CPU column comes from the undisturbed passes of
two CPU sessions:

```bash
breast-seg benchmark summary runtime/runtime_cpu.json runtime/runtime_cpu_2.json --compose runtime_cpu_undisturbed
```

which keeps the undisturbed passes in time order and recomputes every timing number from them (the sessions must
agree in device, threads, warm-up and pairs; the output checks come from the first session whose pass 1 is kept).

**Pairs.** `configs/benchmark/vindr_heldout_pairs.csv` lists the 146 held-out pairs the table was measured on by
their VinDr-Mammo identifiers: `study` is the `study_id`, `mlo_sop` / `cc_sop` the `image_id` of the MLO and CC
image (in the files they are the StudyInstanceUID and SOPInstanceUID), and `side` the laterality. The files are found
by name (`<image_id>.dicom` or `.dcm`) anywhere under `--dicom-root`; a pairs file with `mlo_dicom`
/ `cc_dicom` paths works as well. Absolute times depend on the machine; the protocol makes the configurations
comparable within a session.
