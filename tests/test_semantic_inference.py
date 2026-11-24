"""Semantic segmentation networks (SegFormer, U-Net) served through the engine's Ultralytics-style interface."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breast_seg.imaging import TARGET_SIZE, read_dicom
from breast_seg.imaging.phantom import phantom_dicom
from breast_seg.inference import LandmarkEngine, default_loader
from breast_seg.inference.testing import fake_loader, placeholder_weights
from breast_seg.registry import Registry, RegistryError

torch = pytest.importorskip("torch")

from breast_seg.geometry import compute_nipple_centroid  # noqa: E402
from breast_seg.inference.engine import drawable_polygon  # noqa: E402
from breast_seg.inference.semantic import (  # noqa: E402
    MIN_NIPPLE_PX,
    SemanticModel,
    largest_component,
    load_semantic,
    torch_device,
)
from breast_seg.positioning.measures import masks_from_polygons  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"
CLASSES = ("pectoral", "breast-tissue", "nipple")


class FixedNet(torch.nn.Module):
    """Returns the same logits for every input (quarter resolution, as SegFormer does), mirrored with the input so
    the flip average reproduces them."""

    def __init__(self, logits: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("logits", torch.from_numpy(logits[None]))

    def forward(self, x):
        out = torch.nn.functional.interpolate(self.logits, size=(TARGET_SIZE // 4, TARGET_SIZE // 4), mode="nearest")
        out = out.expand(x.shape[0], -1, -1, -1).clone()
        if x.shape[0] == 2:  # second item is the flipped image: flip the answer too
            out[1] = out[1].flip(-1)
        return out


def _logits(nipple_peak: float = 10.0) -> np.ndarray:
    """Background everywhere, breast in the middle, pectoral in the top-right corner, a nipple blob at (480, 320).
    A nipple peak below the breast logit (9) leaves the nipple out of the arg-max map but above 0.02 in probability."""
    z = np.zeros((4, TARGET_SIZE, TARGET_SIZE), np.float32)
    z[0] = 6.0
    z[2, 100:560, 120:520] = 9.0
    z[1, 0:200, 440:640] = 11.0
    z[3, 300:340, 460:500] = nipple_peak
    z[1, 600:610, 0:10] = 9.0  # a small second pectoral blob: dropped by the largest-component rule
    return z


def _model(**kw) -> SemanticModel:
    return SemanticModel(FixedNet(_logits(**kw)), lambda g: (g.astype(np.float32) / 255.0)[None], CLASSES)


def test_registry_framework():
    raw = {"models": [{"id": "s", "task": "segmentation", "recipe": "baseline640", "weights": "s",
                       "framework": "segformer"}]}
    spec = Registry.from_dict(raw).models[0]
    assert spec.framework == "segformer" and spec.to_dict()["framework"] == "segformer"
    assert Registry.from_dict({"models": [{"id": "y", "task": "segmentation", "recipe": "baseline640",
                                           "weights": "y.pt"}]}).models[0].framework == "ultralytics"
    with pytest.raises(RegistryError):
        Registry.from_dict({"models": [dict(raw["models"][0], framework="keras")]})
    with pytest.raises(RegistryError):
        Registry.from_dict({"models": [dict(raw["models"][0], task="landmark")]})


def test_segformer_folder_is_available(tmp_path):
    raw = {"models": [{"id": "s", "task": "segmentation", "recipe": "baseline640", "weights": "seg/segformer_b3",
                       "framework": "segformer"}]}
    spec = Registry.from_dict(raw, weights_root=tmp_path).models[0]
    assert not spec.available
    (tmp_path / "seg/segformer_b3").mkdir(parents=True)
    assert not spec.available
    (tmp_path / "seg/segformer_b3/config.json").write_text("{}")
    assert spec.available


def test_default_registry_serves_segformer(tmp_path):
    registry = Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    seg = registry.select("segmentation")
    assert seg.id == "seg_segformer_b3" and seg.framework == "segformer"
    assert seg.nipple_rescue.enabled and seg.nipple_rescue.conf == pytest.approx(0.02)


def test_device_strings(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert torch_device("cpu") == "cpu" and torch_device("0") == "cuda:0" and torch_device("cuda:1") == "cuda:1"
    # an Ultralytics device list runs a semantic network on its first GPU; case and spaces do not matter
    assert torch_device("0,1") == "cuda:0" and torch_device(" 1 , 0 ") == "cuda:1"
    assert torch_device("CUDA:1") == "cuda:1" and torch_device("CPU") == "cpu" and torch_device("") == "cpu"
    assert torch_device("mps") == "mps"


@pytest.mark.parametrize("device, visible, seen, expected", [
    ("1", "1", 1, "cuda:0"),          # Ultralytics narrowed the process to GPU 1 before CUDA started: it is cuda:0
    ("cuda:1", "1", 1, "cuda:0"),     # ... however the device is written
    ("1,2", "1,2", 2, "cuda:0"),      # a list: the pose models run on its first GPU, physical 1 = cuda:0
    ("1", "1", 2, "cuda:1"),          # narrowed after CUDA started (torch ignores it): still both GPUs
    ("1", "", 2, "cuda:1"),           # nothing narrowed
    ("1", "0,1", 2, "cuda:1"),
])
def test_a_gpu_index_follows_what_ultralytics_left_visible(monkeypatch, device, visible, seen, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: seen)
    assert torch_device(device) == expected


def test_largest_component():
    m = np.zeros((10, 10), bool)
    m[0:2, 0:2] = True
    m[5:9, 5:9] = True
    out = largest_component(m)
    assert out.sum() == 16 and not out[0, 0]
    assert not largest_component(np.zeros((4, 4), bool)).any()


def test_one_instance_per_class_from_the_argmax_map():
    model = _model()
    (result,) = model.predict(np.zeros((TARGET_SIZE, TARGET_SIZE, 3), np.uint8), conf=0.25)
    names = [model.names[int(c)] for c in result.boxes.cls]
    assert names == list(CLASSES)
    pectoral = result.masks.data[names.index("pectoral")]
    assert pectoral[10:190, 450:630].all() and not pectoral[600:610, 0:10].any()  # the small blob is dropped
    assert all(0.0 < c <= 1.0 for c in result.boxes.conf)


def test_low_confidence_pass_recovers_a_weak_nipple():
    model = _model(nipple_peak=7.0)  # below the breast logit: absent from the arg-max map
    grey = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
    (normal,) = model.predict(grey, conf=0.25)
    assert "nipple" not in [model.names[int(c)] for c in normal.boxes.cls]
    (low,) = model.predict(grey, conf=0.02)
    names = [model.names[int(c)] for c in low.boxes.cls]
    nipple = low.masks.data[names.index("nipple")]
    ys, xs = np.nonzero(nipple)
    assert 300 <= ys.mean() < 340 and 460 <= xs.mean() < 500


@pytest.mark.parametrize("peak, source", [(10.0, "argmax"), (7.0, "probability"), (4.5, "highest")])
def test_nipple_rule_steps(peak, source):
    """The paper's rule: arg-max component, else probability >= 0.02, else the highest-probability pixel."""
    model = _model(nipple_peak=peak)
    mask, found, confidence = model.nipple_by_rule(np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8))
    assert found == source and mask.dtype == np.uint8 and mask.any() and 0.0 < confidence <= 1.0
    ys, xs = np.nonzero(mask)
    assert 296 <= ys.mean() < 344 and 456 <= xs.mean() < 504  # on the nipple blob in every case
    if source == "highest":
        assert mask.sum() <= 13 and confidence < 0.02  # a 2 px disc around one pixel


def test_nipple_rule_without_a_nipple_class():
    model = SemanticModel(FixedNet(_logits()), lambda g: (g.astype(np.float32) / 255.0)[None],
                          ("pectoral", "breast-tissue", "other"))
    assert model.nipple_by_rule(np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)) == (None, "none", 0.0)


def test_one_forward_pass_per_image():
    model = _model()
    calls = []
    forward = model.probabilities
    model.probabilities = lambda grey, device="cpu": calls.append(1) or forward(grey, device)
    image = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
    model.predict(image, conf=0.25)
    model.predict(np.stack([image] * 3, -1), conf=0.02)  # the engine passes BGR
    model.nipple_by_rule(image)
    assert len(calls) == 1
    model.predict(image + 1, conf=0.25)
    assert len(calls) == 2


@pytest.mark.parametrize("pose_finds_nothing", [True, False])
def test_engine_takes_the_paper_nipple_without_the_pose_model(tmp_path, pose_finds_nothing):
    """A nipple missing from the arg-max map comes from the probability map, whatever the pose model found."""
    registry = Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    fake = fake_loader(find_nothing=pose_finds_nothing)

    def loader(spec):
        return _model(nipple_peak=7.0) if spec.task == "segmentation" else fake(spec)

    engine = LandmarkEngine(registry, device="cpu", preview_size=640, loader=loader).load()
    result = engine.analyse(*read_dicom(phantom_dicom("CC", "L"), "L-CC.dcm"))
    nipple = result["polygons"]["nipple"]
    # the rule's step, not a pose-vouched rescue (the page must not say the pose model confirmed it)
    assert nipple["rule"] == "probability" and not nipple["rescued"] and not nipple["estimated"] and nipple["points"]
    ys, xs = np.nonzero(result["masks"]["nipple"])
    assert 296 <= ys.mean() < 344 and 456 <= xs.mean() < 504


def test_engine_runs_a_semantic_segmentation_model(tmp_path):
    registry = Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    fake = fake_loader()

    def loader(spec):
        return _model() if spec.task == "segmentation" else fake(spec)

    engine = LandmarkEngine(registry, device="cpu", preview_size=640, loader=loader).load()
    assert engine.ready
    result = engine.analyse(*read_dicom(phantom_dicom("MLO", "L"), "L-MLO.dcm"))
    assert set(result["masks"]) == set(CLASSES)
    assert result["polygons"]["nipple"]["points"]


def test_unet_lives_in_the_inference_package():
    """The desktop build leaves breast_seg.training out; the U-Net loader must not need it."""
    import breast_seg.inference.unet as inference_unet
    from breast_seg.training import nets

    assert nets.UNet is inference_unet.UNet and nets.load_checkpoint is inference_unet.load_checkpoint
    source = (REPO / "src/breast_seg/inference/semantic.py").read_text()
    assert "training" not in source.split("def load_semantic", 1)[1]


def test_load_semantic_unet(tmp_path):
    from breast_seg.inference.unet import UNet

    torch.save({"model": UNet(1, 4, 4).state_dict(), "base_channels": 4, "num_classes": 4, "in_channels": 1},
               tmp_path / "unet.pt")
    raw = {"models": [{"id": "u", "task": "segmentation", "recipe": "baseline640", "weights": "unet.pt",
                       "framework": "unet"}]}
    spec = Registry.from_dict(raw, weights_root=tmp_path).models[0]
    model = default_loader(spec)
    assert isinstance(model, SemanticModel)
    (result,) = model.predict(np.full((TARGET_SIZE, TARGET_SIZE, 3), 80, np.uint8), conf=0.25)
    assert result.boxes.cls.shape == result.boxes.conf.shape


def _tiny_segformer(transformers):
    config = transformers.SegformerConfig(num_labels=4, num_encoder_blocks=4, depths=[1, 1, 1, 1],
                                          hidden_sizes=[8, 16, 32, 64], num_attention_heads=[1, 1, 2, 2],
                                          decoder_hidden_size=16)
    return transformers.SegformerForSemanticSegmentation(config).eval()


def test_a_segformer_folder_saved_by_transformers_5_loads_every_weight(tmp_path):
    """transformers 5 (which saved the study's checkpoints) names the decode head's projections linear_projections.N;
    transformers 4 calls them linear_c.N and from_pretrained left them randomly initialised (the study's SegFormer-B3
    then segmented every pixel as background). load_segformer maps the names, and refuses a folder it cannot fill."""
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file, save_file

    from breast_seg.inference.semantic import load_segformer

    original = _tiny_segformer(transformers)
    original.save_pretrained(tmp_path / "segformer")
    weights = tmp_path / "segformer" / "model.safetensors"
    state = load_file(str(weights))
    remapped = {key.replace("decode_head.linear_c.", "decode_head.linear_projections."): value
               for key, value in state.items()}
    assert remapped != state
    save_file(remapped, str(weights), metadata={"format": "pt"})
    loaded = load_segformer(tmp_path / "segformer")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(loaded(pixel_values=x).logits, original(pixel_values=x).logits)
    assert not loaded.training

    save_file({key: value for key, value in remapped.items() if "classifier" not in key}, str(weights),
              metadata={"format": "pt"})
    with pytest.raises(ValueError, match="do not fit the network"):
        load_segformer(tmp_path / "segformer")


def test_load_semantic_segformer(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.SegformerConfig(num_labels=4, num_encoder_blocks=4, depths=[1, 1, 1, 1],
                                          hidden_sizes=[8, 16, 32, 64], num_attention_heads=[1, 1, 2, 2],
                                          decoder_hidden_size=16)
    transformers.SegformerForSemanticSegmentation(config).save_pretrained(tmp_path / "segformer_b3")
    raw = {"models": [{"id": "s", "task": "segmentation", "recipe": "baseline640", "weights": "segformer_b3",
                       "framework": "segformer"}]}
    spec = Registry.from_dict(raw, weights_root=tmp_path).models[0]
    assert spec.available
    model = load_semantic(spec)
    p = model.probabilities(np.full((TARGET_SIZE, TARGET_SIZE), 80, np.uint8))
    assert p.shape == (4, TARGET_SIZE, TARGET_SIZE) and np.allclose(p.sum(0), 1.0, atol=1e-4)


def _probabilities(nipple_pixels):
    """Breast tissue everywhere except a few pixels where the nipple wins the arg-max."""
    p = np.zeros((4, TARGET_SIZE, TARGET_SIZE), np.float32)
    p[0], p[2] = 0.1, 0.9
    for y, x in nipple_pixels:
        p[:, y, x] = (0.0, 0.0, 0.05, 0.95)
    return p


def _fixed(p):
    model = _model()
    model.probabilities = lambda grey, device="cpu": p
    return model


@pytest.mark.parametrize("pixels", [[(320, 480)], [(320, 480), (321, 480)]])
def test_a_nipple_too_small_to_outline_is_the_disc_at_its_centroid(pixels):
    mask, source, confidence = _fixed(_probabilities(pixels)).nipple_by_rule(np.zeros((TARGET_SIZE,) * 2, np.uint8))
    assert source == "argmax" and confidence == pytest.approx(0.95)
    ys, xs = np.nonzero(mask)
    assert mask.sum() >= MIN_NIPPLE_PX and len(drawable_polygon(mask)) >= 3
    assert ys.mean() == pytest.approx(np.mean([y for y, _ in pixels]), abs=0.6) and xs.mean() == pytest.approx(480)


def test_drawable_polygon_never_has_fewer_than_three_vertices():
    for pixels in ([(10, 10)], [(10, 10), (10, 11)]):
        binary = np.zeros((64, 64), np.uint8)
        for y, x in pixels:
            binary[y, x] = 1
        polygon = drawable_polygon(binary)
        assert len(polygon) >= 3
        centre = np.mean(polygon, axis=0)
        assert np.allclose(centre, np.mean([(x, y) for y, x in pixels], axis=0), atol=1.0)
    assert drawable_polygon(np.zeros((8, 8), np.uint8)) == []


@pytest.mark.parametrize("view", ["CC", "MLO"])
def test_a_one_pixel_nipple_survives_as_a_contour_the_measurement_uses(tmp_path, view):
    """A one-pixel arg-max nipple is stored as a contour that rasterises back to it: before, a one-vertex polygon was
    stored, dropped when the labels were read, and the nipple lost."""
    registry = Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    fake, model = fake_loader(), _fixed(_probabilities([(320, 480)]))
    engine = LandmarkEngine(registry, device="cpu", preview_size=640,
                            loader=lambda spec: model if spec.task == "segmentation" else fake(spec)).load()
    result = engine.analyse(*read_dicom(phantom_dicom(view, "L"), f"L-{view}.dcm"))
    nipple = result["polygons"]["nipple"]
    assert len(nipple["points"]) >= 3 and nipple.get("rule", "argmax") == "argmax" and not nipple.get("estimated")
    centre = compute_nipple_centroid(masks_from_polygons(result["polygons"])["nipple"])
    assert abs(centre[0] - 480) <= 1 and abs(centre[1] - 320) <= 1


def test_an_image_changed_in_place_is_run_again():
    """The cache holds the bytes it was computed from, not the caller's buffer."""
    model = _model()
    calls = []
    forward = model.probabilities
    model.probabilities = lambda grey, device="cpu": calls.append(device) or forward(grey, device)
    image = np.zeros((TARGET_SIZE, TARGET_SIZE), np.uint8)
    model.predict(image, conf=0.25)
    image[0, 0] = 7  # the same buffer, refilled
    model.predict(image, conf=0.25)
    model.predict(image.copy(), conf=0.02)
    assert len(calls) == 2


def test_load_semantic_unet_from_a_bare_state_dict(tmp_path):
    from breast_seg.inference.unet import UNet

    torch.save(UNet(1, 4, 64).state_dict(), tmp_path / "bare.pt")  # weights only: the default architecture
    raw = {"models": [{"id": "u", "task": "segmentation", "recipe": "baseline640", "weights": "bare.pt",
                       "framework": "unet"}]}
    model = load_semantic(Registry.from_dict(raw, weights_root=tmp_path).models[0])
    assert isinstance(model, SemanticModel)


def test_a_unet_file_without_a_state_dict_is_refused(tmp_path):
    torch.save([1, 2, 3], tmp_path / "list.pt")
    raw = {"models": [{"id": "u", "task": "segmentation", "recipe": "baseline640", "weights": "list.pt",
                       "framework": "unet"}]}
    with pytest.raises(ValueError, match="holds no state dict"):
        load_semantic(Registry.from_dict(raw, weights_root=tmp_path).models[0])
