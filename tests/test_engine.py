"""LandmarkEngine with injected fake models (no weights, CPU), plus one opt-in run on real weights.

The real-weights test reads the weights root from ``BREAST_SEG_WEIGHTS`` and is skipped when the files are absent.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from breast_seg.imaging import TARGET_SIZE, Transform, preprocess_pair, read_dicom
from breast_seg.imaging.phantom import phantom_dicom
from breast_seg.inference import BrightenSearch, EngineError, LandmarkEngine, landmark_record, largest_polygon
from breast_seg.inference.brighten import distance_to, mask_gap_mm
from breast_seg.inference.testing import fake_loader, placeholder_weights
from breast_seg.registry import WEIGHTS_ENV, Registry

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"


@pytest.fixture()
def registry(tmp_path):
    return Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))


@pytest.fixture()
def instance_registry(tmp_path):
    """The registry with the instance model (seg_yolo26m) serving the segmentation role: the pose-anchored rescue and the
    brightened re-search are the instance path's (a semantic network takes the paper's nipple rule instead)."""
    root = placeholder_weights(MODELS_YAML, tmp_path)
    (root / "segmentation" / "segformer_b3").unlink()
    registry = Registry.load(MODELS_YAML, weights_root=root)
    assert registry.select("segmentation").id == "seg_yolo26m"
    return registry


def _engine(registry, **options):
    return LandmarkEngine(registry, device="cpu", preview_size=960, loader=fake_loader(**options)).load()


def _image(view="MLO", laterality="L", **kwargs):
    return read_dicom(phantom_dicom(view, laterality, **kwargs), f"{laterality}-{view}.dcm")


def test_loads_the_role_models(registry):
    engine = _engine(registry)
    status = engine.status()
    assert engine.ready and status["errors"] == {}
    assert status["loaded"] == ["pose_cc", "pose_mlo", "seg_segformer_b3"]
    assert status["roles"]["segmentation"] == "seg_segformer_b3" and not status["roles"].get("detection")
    assert status["device"] == "cpu"
    assert engine.loaded_weights()["seg_segformer_b3"] == "segformer_b3"


def test_analyse_mlo(registry):
    engine = _engine(registry)
    pixels, meta = _image("MLO", "L")
    out = engine.analyse(pixels, meta)
    assert set(out) == {"transform", "mm_per_px", "display", "landmarks", "masks", "polygons", "lesions"}
    assert out["display"].shape == (960, 960)
    assert out["mm_per_px"] == pytest.approx(Transform(**out["transform"]).mm_per_640px)
    landmarks = out["landmarks"]
    assert landmarks["detected"] and not landmarks["mirrored"] and landmarks["confidence"] == 0.9
    assert set(landmarks["points"]) == {"nipple", "pectoral_top", "pectoral_bottom"}
    assert set(out["polygons"]) == {"breast-tissue", "pectoral", "nipple"}
    for name, mask in out["masks"].items():
        assert mask.shape == (TARGET_SIZE, TARGET_SIZE) and mask.dtype == np.uint8
        polygon = out["polygons"][name]["points"]
        assert len(polygon) >= 3 and all(0 <= c <= TARGET_SIZE for point in polygon for c in point)
    assert out["lesions"] == []  # no lesion detector is served
    # every model call ran on the configured device, at the trained input size
    models = engine._models
    assert models["pose_mlo"].calls and models["seg_segformer_b3"].calls
    assert not models["pose_cc"].calls
    for model in models.values():
        assert all(call["device"] == "cpu" and call["imgsz"] == TARGET_SIZE for call in model.calls)


def test_cc_has_only_the_nipple_keypoint(registry):
    out = _engine(registry).analyse(*_image("CC", "R"))
    assert set(out["landmarks"]["points"]) == {"nipple"}


def test_nipple_rescue(instance_registry):
    out = _engine(instance_registry, nipple_conf=0.1).analyse(*_image("MLO", "R"))
    nipple = out["polygons"]["nipple"]
    assert nipple["rescued"] and nipple["confidence"] == 0.1 and not nipple["brightened"]
    assert out["polygons"]["breast-tissue"]["rescued"] is False


@pytest.mark.parametrize("conf, step, estimated", [(0.1, "probability", False), (0.01, "highest", True)])
def test_a_semantic_network_takes_the_papers_nipple_rule(registry, conf, step, estimated):
    """SegFormer (the default): a nipple missing from the arg-max map comes from the paper's rule, not from the
    pose-anchored rescue, and carries the rule's step. It is never "rescued": the pose model vouched for nothing."""
    engine = _engine(registry, nipple_conf=conf)
    out = engine.analyse(*_image("MLO", "R"))
    nipple = out["polygons"]["nipple"]
    assert nipple["rule"] == step and nipple["estimated"] is estimated
    assert nipple["rescued"] is False and nipple["brightened"] is False and len(nipple["points"]) >= 3
    model = engine._models["seg_segformer_b3"]
    assert [call["conf"] for call in model.calls] == [0.25] and len(model.rule_calls) == 1  # no rescue pass


@pytest.mark.parametrize("conf, step, estimated", [(0.1, "probability", False), (0.01, "highest", True)])
def test_analyse_landmarks_carries_the_nipple_rule_step(registry, conf, step, estimated):
    """The cross-view record of a semantic network keeps the rule's step: a nipple placed at the single most probable
    pixel reaches the matcher marked as an estimate, not as a detected nipple."""
    engine = _engine(registry, nipple_conf=conf)
    pixels, meta = _image("MLO", "R")
    record = engine.analyse_landmarks(pixels, meta)
    out = engine.analyse(pixels, meta)
    nipple = record["polygons"]["nipple"]
    assert nipple["rule"] == step and nipple["estimated"] is estimated and nipple["rescued"] is False
    expected = _crossview_summary(out["landmarks"], out["polygons"], out["transform"])
    expected["polygons"]["nipple"].update(rule=step, estimated=estimated)
    assert {key: record[key] for key in ("points", "conf", "polygons")} == expected
    assert all("rule" not in shape for name, shape in record["polygons"].items() if name != "nipple")


class _DarkNipple:
    """Segmentation stand-in for a dark image: the nipple is found only on the brightened canvas, i.e. on any image
    other than the first one it is shown (the normal pass and the rescue see the same canvas).

    ``shift`` moves the nipple it finds there that many pixels along x (away from the breast, to test the adjacency
    check); ``short`` makes the normal pass draw the breast that many pixels short of its anterior edge, as on a
    dark image whose periphery only the brightened pass sees (chest wall on the left).
    """

    def __init__(self, inner, shift=0, short=0):
        self.inner, self.names, self.calls, self.original = inner, inner.names, inner.calls, None
        self.shift, self.short = shift, short

    def predict(self, image, **kwargs):
        key = hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()
        self.original = self.original or key
        result = self.inner.predict(image, **kwargs)[0]
        if result.masks is None:
            return [result]
        named = {self.names[int(c)]: i for i, c in enumerate(result.boxes.cls)}
        if key == self.original:
            if self.short:
                breast = result.masks.data[named["breast-tissue"]]
                breast[:, int(np.nonzero(breast.any(axis=0))[0].max()) - self.short:] = 0
            keep = [i for i in range(len(result.boxes)) if i != named.get("nipple")]
            result.masks.data = result.masks.data[keep]
            result.boxes = type(result.boxes)(result.boxes.conf[keep], result.boxes.cls[keep], result.boxes.xyxy[keep])
        elif self.shift and "nipple" in named:
            result.masks.data[named["nipple"]] = np.roll(result.masks.data[named["nipple"]], self.shift, axis=1)
        return [result]


ON = BrightenSearch()  # as the review app builds its engine (window 50 mm, no adjacency check)


def _dark_engine(registry, brighten=ON, **dark):
    inner = fake_loader()

    def load(spec):
        model = inner(spec)
        return _DarkNipple(model, **dark) if spec.task == "segmentation" else model
    return LandmarkEngine(registry, device="cpu", preview_size=960, loader=load, brighten=brighten).load()


@pytest.mark.parametrize("view,laterality", [("MLO", "L"), ("CC", "R")])
def test_brightened_search_finds_the_nipple_both_passes_missed(instance_registry, view, laterality):
    pixels, meta = _image(view, laterality)
    engine = _dark_engine(instance_registry)
    out = engine.analyse(pixels, meta)
    nipple = out["polygons"]["nipple"]
    assert nipple["brightened"] and not nipple["rescued"] and nipple["confidence"] == 0.8
    # normal pass, rescue next to the pose nipple, then the brightened canvas at the rescue confidence
    assert [call["conf"] for call in engine._models["seg_yolo26m"].calls] == [0.25, 0.02, 0.02]
    # the contour sits at the anterior apex the pose model also points at
    pose = np.asarray(out["landmarks"]["points"]["nipple"])
    centre = np.argwhere(out["masks"]["nipple"] > 0).mean(axis=0)[::-1]
    assert np.linalg.norm(centre - pose) < 25
    record = landmark_record(out["landmarks"], out["polygons"], Transform(**out["transform"]))
    assert record["polygons"]["nipple"]["brightened"] is True


def _gap(out):
    mm = Transform(**out["transform"]).mm_per_640px
    return mask_gap_mm(out["masks"]["nipple"], distance_to(out["masks"]["breast-tissue"]), mm)


def test_the_adjacency_check_is_off_unless_a_distance_is_given(instance_registry):
    """As configured, the brightened search takes the most confident nipple in the window wherever it lies: the
    check never turned down a wrong candidate in the evaluation, so it is not on by default."""
    assert BrightenSearch().max_gap_mm is None
    away = _dark_engine(instance_registry, shift=70).analyse(*_image("MLO", "L"))
    assert away["polygons"]["nipple"]["brightened"] and _gap(away) > 3.0


def test_with_a_distance_a_brightened_nipple_must_touch_the_breast(instance_registry):
    """With ``max_gap_mm`` set, the brightened search does not take a nipple that floats in the background, more than
    that far from the breast-tissue mask."""
    checked = BrightenSearch(max_gap_mm=3.0)
    pixels, meta = _image("MLO", "L")                 # chest wall on the left: +x is away from the breast
    touching = _dark_engine(instance_registry, checked).analyse(pixels, meta)
    assert touching["polygons"]["nipple"]["brightened"] and _gap(touching) == 0.0
    near = _dark_engine(instance_registry, checked, shift=20).analyse(pixels, meta)           # a little off the skin line
    assert near["polygons"]["nipple"]["brightened"] and 0.0 < _gap(near) <= 3.0
    away = _dark_engine(instance_registry, checked, shift=70).analyse(pixels, meta)           # out in the background
    assert "nipple" not in away["polygons"] and set(away["polygons"]) == {"breast-tissue", "pectoral"}


def test_the_breast_the_brightened_pass_sees_counts_for_the_adjacency(instance_registry):
    """On a dark image the normal pass stops short of the skin line; the nipple then lies far from that mask and
    right on the breast the brightened pass itself draws. It is taken."""
    out = _dark_engine(instance_registry, BrightenSearch(max_gap_mm=3.0), short=30).analyse(*_image("MLO", "L"))
    assert out["polygons"]["nipple"]["brightened"]
    assert _gap(out) > 3.0                             # against the normal pass alone it would have been dropped


@pytest.mark.parametrize("search", [ON, BrightenSearch(max_gap_mm=3.0)])
def test_the_brightened_pass_adds_a_nipple_and_changes_nothing_else(instance_registry, search):
    """The brightened pass sees more of the breast than the normal pass did (here 30 px more). What is stored stays
    the normal pass's breast and pectoral: the same masks and contours as with the brightened search switched off."""
    pixels, meta = _image("MLO", "L")
    without = _dark_engine(instance_registry, None, short=30).analyse(pixels, meta)
    found = _dark_engine(instance_registry, search, short=30).analyse(pixels, meta)
    assert "nipple" not in without["polygons"] and found["polygons"]["nipple"]["brightened"]
    for name in ("breast-tissue", "pectoral"):
        assert (found["masks"][name] == without["masks"][name]).all()
        assert found["polygons"][name] == without["polygons"][name]


def test_a_failing_brightened_search_never_fails_the_image(instance_registry, monkeypatch, caplog):
    pixels, meta = _image("MLO", "L")
    # a header that says the pixels are 0 mm wide cannot size the window: no search, no error
    zero = dict(meta, pixel_spacing=0.0)
    out = _dark_engine(instance_registry).analyse(pixels, zero)
    assert "nipple" not in out["polygons"] and set(out["polygons"]) == {"breast-tissue", "pectoral"}

    # and whatever else goes wrong inside the optional step, the image is analysed as before it existed
    import breast_seg.inference.engine as engine_module

    def broken(*args, **kwargs):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(engine_module, "brightened_canvas", broken)
    with caplog.at_level("WARNING", logger="breast_seg.inference.engine"):
        out = _dark_engine(instance_registry).analyse(pixels, meta)
    assert "nipple" not in out["polygons"] and out["landmarks"]["detected"]
    assert any("brightened nipple search failed" in message for message in caplog.messages)


def test_the_brightened_search_is_the_callers_choice(instance_registry):
    """Built from the shared registry alone, the engine never brightens: every other user of the registry (a
    cross-view matcher reading ``analyse_landmarks``) gets the nipples the model passes find, and nothing else."""
    assert LandmarkEngine(instance_registry).brighten is None
    pixels, meta = _image("MLO", "L")
    plain = _dark_engine(instance_registry, None)
    assert "nipple" not in plain.analyse(pixels, meta)["polygons"]
    assert [call["conf"] for call in plain._models["seg_yolo26m"].calls] == [0.25, 0.02]     # no third pass
    record = _dark_engine(instance_registry, None).analyse_landmarks(pixels, meta)
    assert "nipple" not in record["polygons"] and not any("brightened" in shape for shape in record["polygons"].values())
    assert _dark_engine(instance_registry).analyse_landmarks(pixels, meta)["polygons"]["nipple"]["brightened"] is True
    for bad in ({"window_mm": 0}, {"max_gap_mm": -1}):
        with pytest.raises(ValueError):
            BrightenSearch(**bad)


def test_nipple_rescue_disabled(tmp_path):
    raw = Registry.load(MODELS_YAML)
    models = [{"id": m.id, "task": m.task, "view": m.view, "recipe": m.recipe, "weights": m.weights,
               "enabled": m.enabled, "classes": list(m.classes), "keypoints": list(m.keypoints),
               "nipple_rescue": {"enabled": False}} for m in raw.models]
    registry = Registry.from_dict({"models": models}, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    out = _engine(registry, nipple_conf=0.1).analyse(*_image("MLO", "R"))
    assert "nipple" not in out["polygons"]


def test_mirror_fallback_maps_points_back(registry):
    pixels, meta = _image("MLO", "L")
    plain = _engine(registry).analyse(pixels, meta)["landmarks"]
    mirrored = _engine(registry, only_mirrored=True).analyse(pixels, meta)["landmarks"]
    assert mirrored["mirrored"] and not plain["mirrored"]
    for name in ("nipple", "pectoral_bottom"):
        assert np.allclose(mirrored["points"][name], plain["points"][name], atol=2.0)


def test_nothing_found(registry):
    out = _engine(registry, find_nothing=True).analyse(*_image("CC", "L"))
    assert out["landmarks"] == {"detected": False, "mirrored": False, "confidence": 0.0, "points": {}}


def test_missing_weights_are_reported_not_raised(tmp_path):
    registry = Registry.load(MODELS_YAML, weights_root=tmp_path)
    engine = LandmarkEngine(registry, loader=fake_loader()).load()
    assert not engine.ready
    expected = "weights file not found: segmentation/segformer_b3 (under the weights root)"
    assert engine.errors["seg_segformer_b3"] == expected
    assert str(tmp_path) not in str(engine.status())
    with pytest.raises(EngineError):
        engine.analyse(*_image())
    with pytest.raises(EngineError):
        engine.analyse_landmarks(*_image())


def test_a_later_entry_stands_in_for_missing_weights(tmp_path):
    root = placeholder_weights(MODELS_YAML, tmp_path)
    (root / "segmentation" / "segformer_b3").unlink()  # SegFormer-B3 folder absent
    engine = LandmarkEngine(Registry.load(MODELS_YAML, weights_root=root), loader=fake_loader()).load()
    status = engine.status()
    assert engine.ready and status["errors"] == {}
    assert status["roles"]["segmentation"] == "seg_yolo26m"
    assert status["skipped"] == {"seg_segformer_b3": "weights file not found: segmentation/segformer_b3 (under the "
                                                     "weights root); seg_yolo26m serves the segmentation role"}
    assert set(engine.analyse(*_image())["polygons"]) == {"breast-tissue", "pectoral", "nipple"}


def test_a_later_entry_stands_in_for_a_load_failure(registry):
    fake = fake_loader()

    def loader(spec):
        if spec.id == "seg_segformer_b3":
            raise ImportError("No module named 'transformers'")
        return fake(spec)

    engine = LandmarkEngine(registry, loader=loader).load()
    assert engine.ready and engine.errors == {}
    assert engine.roles["segmentation"].id == "seg_yolo26m"
    assert "No module named 'transformers'" in engine.skipped["seg_segformer_b3"]


def test_loader_failure_is_recorded(registry):
    def broken(spec):
        raise OSError("corrupt checkpoint")

    engine = LandmarkEngine(registry, loader=broken).load()
    assert not engine.ready and engine.errors["pose_mlo"] == "failed to load: corrupt checkpoint"


def test_missing_role(tmp_path):
    registry = Registry.from_dict({"models": []}, weights_root=tmp_path)
    engine = LandmarkEngine(registry, loader=fake_loader()).load()
    assert not engine.ready and "no enabled segmentation model" in engine.errors["segmentation"]


def _crossview_summary(landmarks, polygons, transform):
    """The landmark record exactly as the cross-view project's adapter built it from the web app's engine."""
    t = Transform(**transform)
    pts = {k: t.to_original(np.array([p]))[0].round(1).tolist() for k, p in (landmarks.get("points") or {}).items()}
    shapes = {}
    for k, e in polygons.items():
        if e["points"]:
            shapes[k] = dict(p=t.to_original(np.array(e["points"], dtype=float)).round(1).tolist(), conf=e["confidence"],
                             rescued=e.get("rescued", False))
    return dict(points=pts, conf=landmarks.get("confidence"), polygons=shapes)


@pytest.mark.parametrize("view,laterality,flipped", [("MLO", "L", False), ("CC", "R", False), ("CC", "L", True)])
def test_analyse_landmarks_reproduces_the_crossview_record(instance_registry, view, laterality, flipped):
    engine = _engine(instance_registry, nipple_conf=0.1)
    pixels, meta = _image(view, laterality, flipped=flipped)
    record = engine.analyse_landmarks(pixels, meta)
    out = engine.analyse(pixels, meta)
    expected = _crossview_summary(out["landmarks"], out["polygons"], out["transform"])
    expected.update(chest_side=meta["chest_side"], lat=meta["laterality"], view=meta["view"],
                    W=int(pixels.shape[1]), H=int(pixels.shape[0]))
    assert record == expected
    assert set(record) == {"points", "conf", "polygons", "chest_side", "lat", "view", "W", "H"}
    assert record["polygons"]["nipple"]["rescued"] is True
    # original pixel coordinates: inside the image, not the 640 frame
    xs = [p[0] for p in record["polygons"]["breast-tissue"]["p"]]
    assert max(xs) <= record["W"] and max(xs) > TARGET_SIZE * 0.5


def test_landmark_record_skips_empty_polygons():
    transform = Transform(100, 100, 0, 0, 100, 100, 2.0, 0, 0, 0, 0, 0.1)
    record = landmark_record({"points": {"nipple": [20.0, 40.0]}, "confidence": 0.5},
                             {"nipple": {"points": [], "confidence": 0.3}}, transform)
    assert record == {"points": {"nipple": [10.0, 20.0]}, "conf": 0.5, "polygons": {}}


def test_largest_polygon():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1
    mask[30:45, 30:45] = 1
    polygon = largest_polygon(mask)
    assert min(p[0] for p in polygon) >= 30
    assert largest_polygon(np.zeros((5, 5), np.uint8)) == []


def test_importing_the_engine_does_not_import_ultralytics():
    code = ("import sys, breast_seg.inference, breast_seg.apps.positioning; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "False False"


# ── real weights (opt-in) ──────────────────────────────────────────────────
def _real_registry():
    root = os.environ.get(WEIGHTS_ENV, "").strip()
    if not root:
        pytest.skip(f"set {WEIGHTS_ENV} to the weights root to run against real weights")
    registry = Registry.load(MODELS_YAML, weights_root=root)
    needed = [registry.select("landmark", "MLO"), registry.select("landmark", "CC"), registry.select("segmentation")]
    if not all(spec is not None and spec.available for spec in needed):
        pytest.skip("weights not found under the weights root")
    return registry


@pytest.mark.weights
def test_real_weights_on_cpu():
    pytest.importorskip("ultralytics")
    engine = LandmarkEngine(_real_registry(), device="cpu", preview_size=640).load()
    assert engine.ready, engine.errors
    pixels, meta = _image("MLO", "L")
    out = engine.analyse(pixels, meta)
    assert out["display"].shape == (640, 640)
    assert set(out) >= {"landmarks", "polygons", "lesions"}
    record = engine.analyse_landmarks(pixels, meta)
    assert record["W"] == pixels.shape[1] and record["view"] == "MLO"
    baseline, _, _ = preprocess_pair(pixels, meta)
    assert baseline.shape == (TARGET_SIZE, TARGET_SIZE)


@pytest.mark.weights
@pytest.mark.parametrize("seg_id", ["seg_segformer_b3", "seg_unet", "seg_yolo26m"])
def test_every_released_segmentation_model_runs_on_cpu(seg_id):
    """Each released segmentation checkpoint loads into the engine and segments a phantom, on its own."""
    import yaml

    pytest.importorskip("ultralytics")
    if seg_id == "seg_segformer_b3":
        pytest.importorskip("transformers")
    root = _real_registry().weights_root
    raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    for entry in raw["models"]:
        if entry["task"] == "segmentation":
            entry["enabled"] = entry["id"] == seg_id
    engine = LandmarkEngine(Registry.from_dict(raw, MODELS_YAML.parent, weights_root=root), device="cpu",
                            preview_size=640).load()
    assert engine.ready, engine.errors
    assert engine.roles["segmentation"].id == seg_id and not engine.skipped
    pixels, meta = _image("MLO", "L")
    out = engine.analyse(pixels, meta)
    assert out["masks"] and all(m.shape == (TARGET_SIZE, TARGET_SIZE) for m in out["masks"].values())


def _pose_entry(spec, **changes):
    entry = {"id": spec.id, "task": spec.task, "view": spec.view, "recipe": spec.recipe, "weights": spec.weights,
             "keypoints": list(spec.keypoints)}
    entry.update(changes)
    return entry


def test_a_model_that_fails_for_a_role_left_without_one_stays_an_error(tmp_path, caplog):
    """One pose entry serving both views fails to load: the MLO role has a stand-in, the CC role has none. The
    failure is an error (the CC role is unserved), not a "skipped" note, and the entry is not tried twice."""
    base = {spec.id: spec for spec in Registry.load(MODELS_YAML).models}
    models = [
        _pose_entry(base["pose_mlo"], id="pose_both", view="both", weights="both.pt"),
        _pose_entry(base["pose_mlo"], id="pose_mlo_only", weights="mlo.pt"),
        {"id": "seg", "task": "segmentation", "recipe": "baseline640", "weights": "seg.pt"},
    ]
    for name in ("both.pt", "mlo.pt", "seg.pt"):
        (tmp_path / name).touch()
    registry = Registry.from_dict({"models": models}, weights_root=tmp_path)
    fake, tried = fake_loader(), []

    def loader(spec):
        tried.append(spec.id)
        if spec.id == "pose_both":
            raise OSError("corrupt checkpoint")
        return fake(spec)

    with caplog.at_level("WARNING", logger="breast_seg.inference.engine"):
        engine = LandmarkEngine(registry, loader=loader).load()
    assert tried.count("pose_both") == 1
    reported = [record for record in caplog.records if record.getMessage().startswith("pose_both: failed to load")]
    assert reported and reported[0].exc_info and reported[0].exc_info[0] is OSError   # with its traceback
    assert not engine.ready and engine.roles["pose_mlo"].id == "pose_mlo_only"
    assert engine.errors == {"pose_both": "failed to load: corrupt checkpoint"} and engine.skipped == {}


def test_a_second_load_rebuilds_what_was_skipped(registry):
    fake, broken = fake_loader(), {"seg_segformer_b3"}

    def loader(spec):
        if spec.id in broken:
            raise ImportError("No module named 'transformers'")
        return fake(spec)

    engine = LandmarkEngine(registry, loader=loader).load()
    assert engine.roles["segmentation"].id == "seg_yolo26m" and set(engine.skipped) == {"seg_segformer_b3"}
    broken.clear()  # the package installed, the engine loaded again
    engine.load()
    assert engine.roles["segmentation"].id == "seg_segformer_b3" and engine.skipped == {} and engine.errors == {}
