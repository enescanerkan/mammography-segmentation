"""Semantic segmentation networks (SegFormer-B3, plain U-Net) behind the Ultralytics-style interface of the engine.

The engine asks a segmentation model for instances: ``predict(image, conf=..., ...) -> [result]`` with
``result.masks.data`` (one 640 x 640 mask per instance), ``result.boxes.cls`` and ``result.boxes.conf``, and the
model's ``names``. A semantic network has no instances, so :class:`SemanticModel` returns one per class: the largest
connected component of the class in the arg-max map, with the mean class probability over it as the confidence.

A pass at a confidence below the model's own also takes the pixels whose class probability reaches that
confidence. The nipple itself is located by :meth:`SemanticModel.nipple_by_rule`, the paper's rule, which the engine
applies to a semantic model whenever its normal pass returns no nipple, with no condition on the pose model:

1. the largest 8-connected nipple component of the arg-max map (of any size);
2. else the largest component of the nipple probability >= :data:`NIPPLE_PROBABILITY` (0.02);
3. else the pixel of the highest nipple probability.

As in the evaluation, the probabilities are the softmax averaged over the image and its horizontally flipped pass,
in bfloat16 autocast on a GPU and float32 on the CPU; both passes of one image share one forward pass (cached).

PyTorch (and, for SegFormer, ``transformers``: the ``segformer`` extra) is imported only when a model is loaded.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..imaging import TARGET_SIZE
from ..registry import ModelSpec

#: nipple probability threshold of the paper's rule (fixed before the experiments; not a registry setting)
NIPPLE_PROBABILITY = 0.02
#: radius of the disc drawn around the highest-probability pixel (step 3), and around the centroid of a component too
#: small to outline (steps 1-2), so the contour is drawable; the disc is symmetric, so its centroid is that point
NIPPLE_DISC_PX = 2
#: a nipple component of fewer pixels than this is replaced by the disc at its centroid (a one- or two-pixel component
#: has no outline of three vertices, and the stored polygon is the only shape the measurement sees)
MIN_NIPPLE_PX = 13

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SemanticMasks:
    data: np.ndarray  # (N, 640, 640) float32 in {0, 1}


@dataclass
class SemanticBoxes:
    cls: np.ndarray  # (N,) class index into ``names``
    conf: np.ndarray  # (N,)
    xyxy: np.ndarray  # (N, 4)


@dataclass
class SemanticResult:
    masks: Optional[SemanticMasks]
    boxes: SemanticBoxes


def largest_component(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a boolean mask (empty in, empty out)."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def torch_device(device: str) -> str:
    """Ultralytics device strings (``"cpu"``, ``"0"``, ``"cuda:1"``, a list such as ``"0,1"``) -> a torch device
    string. A semantic network runs on one device: of a list, the first GPU, the one Ultralytics runs the pose models
    on.

    Ultralytics narrows ``CUDA_VISIBLE_DEVICES`` to the devices it is given before CUDA starts, so after the first pose
    call of a process started with ``device: "1"`` (or ``"cuda:1"``, ``"1,2"``) physical GPU 1 is the process's
    ``cuda:0``. A GPU index named in ``CUDA_VISIBLE_DEVICES`` is therefore taken by its place in that list whenever
    torch sees exactly those devices (or fewer than the index needs); otherwise (no narrowing, or one made after CUDA
    started, which torch ignores) it is the index itself."""
    device = str(device).strip().lower()
    if "," in device:
        device = device.split(",")[0].strip()
    if device.startswith("cuda:") and device[5:].isdigit():
        device = device[5:]
    if device.isdigit():
        torch = sys.modules.get("torch")
        count = torch.cuda.device_count() if torch is not None else 0
        visible = [entry.strip() for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
        if device in visible and (count == len(visible) or int(device) >= count):
            return f"cuda:{visible.index(device)}"
        return f"cuda:{device}"
    return device or "cpu"


def disc_at(point: Tuple[float, float], shape: Tuple[int, int] = (TARGET_SIZE, TARGET_SIZE),
            radius: int = NIPPLE_DISC_PX) -> np.ndarray:
    """A filled disc (uint8) of ``radius`` around ``point`` = (x, y)."""
    disc = np.zeros(shape, np.uint8)
    cv2.circle(disc, (int(round(point[0])), int(round(point[1]))), int(radius), 1, -1)
    return disc


class SemanticModel:
    """A semantic network on the 640 grey canvas, answering ``predict`` the way an Ultralytics model does.

    Args:
        net: the PyTorch module; ``forward(x)`` returns logits ``(1, 1 + len(classes), H', W')`` at any resolution.
        to_input: ``(640, 640) uint8 -> (C, 640, 640) float32`` (the input transform the network was trained with).
        classes: class names in label order (label ``i + 1`` is ``classes[i]``; label 0 is the background).
        conf: the confidence of a normal pass; ``predict`` with a lower ``conf`` adds that probability level.
        flip: average the softmax with that of the horizontally flipped image (the evaluated setting).
    """

    def __init__(self, net: Any, to_input: Callable[[np.ndarray], np.ndarray], classes: Sequence[str],
                 conf: float = 0.25, flip: bool = True) -> None:
        self.net = net.eval()
        self.to_input = to_input
        self.names: Dict[int, str] = dict(enumerate(classes))
        self.conf = float(conf)
        self.flip = bool(flip)
        self._device: Optional[str] = None
        # (device, digest, image bytes) -> probabilities of the last image asked
        self._cache: Optional[Tuple[Tuple[str, bytes], bytes, np.ndarray]] = None

    def _on(self, device: str) -> str:
        dev = torch_device(device)
        if dev != self._device:
            self.net = self.net.to(dev)
            self._device = dev
        return dev

    def probabilities(self, grey: np.ndarray, device: str = "cpu") -> np.ndarray:
        """``(640, 640) uint8`` -> class probabilities ``(1 + len(classes), 640, 640)`` float32."""
        import torch
        import torch.nn.functional as F

        dev = self._on(device)
        x = torch.from_numpy(self.to_input(grey)[None].copy()).to(dev)
        cuda = dev.startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda):
            batch = torch.cat([x, x.flip(-1)]) if self.flip else x
            logits = self.net(batch)
            # In the evaluated order (SegFormerTrainer.forward, benchmark SegFormerPipeline): upsampled inside the
            # autocast region (bf16 on a GPU), and only then cast to float32.
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            p = logits.float().softmax(1)
            p = (p[0] + p[1].flip(-1)) / 2 if self.flip else p[0]
        return p.float().cpu().numpy()

    def _probabilities_of(self, image: np.ndarray, device: str) -> np.ndarray:
        """Probabilities of one image (grey or BGR), reused while the same image is asked again on the same device
        (the engine's normal pass, its low-confidence pass and the nipple rule share one forward pass)."""
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if grey.shape != (TARGET_SIZE, TARGET_SIZE):
            raise ValueError(f"expected a {TARGET_SIZE} x {TARGET_SIZE} canvas, got {grey.shape}")
        data = np.ascontiguousarray(grey).tobytes()
        key = (torch_device(device), hashlib.blake2b(data, digest_size=16).digest())
        # The digest finds the entry; the stored bytes confirm it (a collision can never hand over another image).
        if self._cache is None or self._cache[0] != key or self._cache[1] != data:
            self._cache = (key, data, self.probabilities(grey, device))
        return self._cache[2]

    def nipple_by_rule(self, image: np.ndarray, device: str = "cpu") -> Tuple[Optional[np.ndarray], str, float]:
        """The paper's nipple rule (module docstring, steps 1-3) -> ``(mask, source, confidence)``.

        ``mask`` is a 640 x 640 uint8 mask (one component), ``source`` ``"argmax"``, ``"probability"`` or
        ``"highest"``, ``confidence`` the mean nipple probability over the mask; ``(None, "none", 0.0)`` for a model
        without a nipple class."""
        index = next((i for i, name in self.names.items() if name == "nipple"), None)
        if index is None:
            return None, "none", 0.0
        p = self._probabilities_of(image, device)
        q = p[index + 1]
        mask = largest_component(p.argmax(0) == index + 1)
        source = "argmax"
        if not mask.any():
            mask, source = largest_component(q >= NIPPLE_PROBABILITY), "probability"
        if not mask.any():
            y, x = np.unravel_index(int(np.argmax(q)), q.shape)
            mask, source = disc_at((x, y), q.shape) > 0, "highest"
        elif mask.sum() < MIN_NIPPLE_PX:
            # Too small to outline: the disc at its centroid (the rule's nipple is the centroid either way).
            ys, xs = np.nonzero(mask)
            confidence = float(q[mask].mean())
            return disc_at((xs.mean(), ys.mean()), q.shape), source, confidence
        return mask.astype(np.uint8), source, float(q[mask].mean())

    def predict(self, image: np.ndarray, imgsz: int = TARGET_SIZE, device: str = "cpu", conf: Optional[float] = None,
                iou: Optional[float] = None, retina_masks: bool = True, verbose: bool = False,
                **_: Any) -> List[SemanticResult]:
        """One image (grey, or BGR as the engine passes it) -> ``[result]`` with one instance per class found."""
        p = self._probabilities_of(image, device)
        labels = p.argmax(0)
        low = conf is not None and float(conf) < self.conf
        masks, classes, confs, boxes = [], [], [], []
        for index in self.names:
            region = labels == index + 1
            if low:
                region = region | (p[index + 1] >= float(conf))
            region = largest_component(region)
            if not region.any():
                continue
            ys, xs = np.nonzero(region)
            masks.append(region.astype(np.float32))
            classes.append(index)
            confs.append(float(p[index + 1][region].mean()))
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        if not masks:
            return [SemanticResult(None, SemanticBoxes(np.zeros(0), np.zeros(0), np.zeros((0, 4))))]
        return [SemanticResult(SemanticMasks(np.stack(masks)),
                               SemanticBoxes(np.asarray(classes, float), np.asarray(confs, float),
                                             np.asarray(boxes, float)))]


def _imagenet3(img: np.ndarray) -> np.ndarray:
    x = np.repeat(img[..., None].astype(np.float32) / 255.0, 3, axis=2)
    return ((x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)).transpose(2, 0, 1)


def _grey01(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0)[None]


class _SegFormerLogits:
    """``SegformerForSemanticSegmentation`` -> a callable returning logits (as :class:`SemanticModel` expects)."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def eval(self) -> _SegFormerLogits:
        self.model.eval()
        return self

    def to(self, device: str) -> _SegFormerLogits:
        self.model.to(device)
        return self

    def __call__(self, x: Any) -> Any:
        return self.model(pixel_values=x).logits


# The decode head's four per-stage projections: ``linear_c.N`` up to transformers 4 (the last line for Python 3.8),
# ``linear_projections.N`` from transformers 5, which saved the study's checkpoints.
_HEAD_NAMES = ("decode_head.linear_projections.", "decode_head.linear_c.")


def _segformer_key(key: str, expected: Any) -> str:
    if key in expected:
        return key
    for old, new in (_HEAD_NAMES, _HEAD_NAMES[::-1]):
        if key.startswith(old) and new + key[len(old):] in expected:
            return new + key[len(old):]
    return key


def _segformer_weights(folder: Any) -> Dict[str, Any]:
    if (folder / "model.safetensors").is_file():
        from safetensors.torch import load_file

        return load_file(str(folder / "model.safetensors"))
    if (folder / "pytorch_model.bin").is_file():
        import torch

        return torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"{folder.name}: neither model.safetensors nor pytorch_model.bin")


def load_segformer(folder: Any) -> Any:
    """A ``save_pretrained`` SegFormer folder -> ``SegformerForSemanticSegmentation`` in eval mode, every weight loaded.

    Not ``from_pretrained``: a folder saved by transformers 5 (the study's) names the decode head's projections
    ``linear_projections.N``, and transformers 4 (``linear_c.N``) would leave those four layers randomly initialised
    with no more than a log warning — a network that runs and segments nonsense. The weights are read here, the head's
    names mapped either way, and a weight left unloaded (or one the network has no place for) is an error."""
    from pathlib import Path

    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    folder = Path(folder)
    model = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(str(folder)))
    expected = model.state_dict()
    state = {_segformer_key(key, expected): value for key, value in _segformer_weights(folder).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"{folder.name}: the SegFormer weights do not fit the network ({len(missing)} missing, e.g. "
                         f"{missing[:2]}; {len(unexpected)} unexpected, e.g. {unexpected[:2]})")
    return model.eval()


def load_semantic(spec: ModelSpec) -> SemanticModel:
    """Build the network of a ``segformer`` or ``unet`` registry entry (CPU; moved to the device on first use)."""
    classes = spec.classes or ("pectoral", "breast-tissue", "nipple")
    if spec.framework == "segformer":
        net = _SegFormerLogits(load_segformer(spec.weights_path))
        to_input = _imagenet3
    elif spec.framework == "unet":
        from .unet import UNet, load_checkpoint, upgrade_state_dict

        ck = load_checkpoint(spec.weights_path)
        # A U-Net checkpoint is a dict (weights under "model" plus the architecture: base_channels / num_classes, in
        # the released unet.pt named taban / sinif); a bare state dict carries only the weights, and the
        # architecture falls back to the defaults.
        meta = ck if isinstance(ck, dict) and "model" in ck else {}
        state = ck.get("model", ck) if isinstance(ck, dict) else ck
        if not isinstance(state, dict):
            raise ValueError(f"{spec.id}: {spec.weights_path.name} holds no state dict (got {type(state).__name__})")
        net = UNet(meta.get("in_channels", 1), meta.get("num_classes", meta.get("sinif", 1 + len(classes))),
                   meta.get("base_channels", meta.get("taban", 64)))
        net.load_state_dict(upgrade_state_dict(state))
        to_input = _grey01
    else:
        raise ValueError(f"{spec.id}: framework {spec.framework!r} is not a semantic network")
    return SemanticModel(net, to_input, classes, conf=spec.conf)
