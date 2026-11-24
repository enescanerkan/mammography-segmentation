"""YOLO trainers without training: data.yaml, train kwargs, the Albumentations patch, fold links and prediction
flattening with a fake model."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from breast_seg.config import PathsConfig
from breast_seg.training import SplitSpec, build_trainer
from breast_seg.training.augment import build_transforms, ultralytics_albumentations, validate_specs

ROOT = Path(__file__).resolve().parents[1]


def yolo_layout(root: Path, splits=("train", "val", "test"), n=2):
    for sp in splits:
        (root / "images" / sp).mkdir(parents=True)
        (root / "labels" / sp).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / "images" / sp / f"img{sp}{i}.png"), np.full((32, 32), 100, np.uint8))
            (root / "labels" / sp / f"img{sp}{i}.txt").write_text("1 0 0 1 0 1 1 0 1\n")


def trainer_for(config, tmp_path, *overrides):
    paths = PathsConfig(root=str(tmp_path))
    return build_trainer(ROOT / "configs/train" / config, paths, overrides=list(overrides))


def test_prepare_writes_valid_seg_data_yaml(tmp_path):
    yolo_layout(tmp_path / "data/vindr/yolo")
    t = trainer_for("yolo26m_seg.yaml", tmp_path)
    path = t.prepare()
    data = yaml.safe_load(path.read_text())
    assert path == tmp_path / "runs/yolo26m_seg/data.yaml"
    assert Path(data["path"]) == tmp_path / "data/vindr/yolo"
    assert (data["train"], data["val"], data["test"]) == ("images/train", "images/val", "images/test")
    assert data["names"] == ["pectoral", "breast-tissue", "nipple"] and data["nc"] == 3
    assert (t.run_dir / "train_config.json").is_file()
    kw = t.train_kwargs()
    expected = {"epochs": 300, "patience": 50, "batch": 16, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.001, "amp": True,
                "workers": 8, "degrees": 8.0, "translate": 0.05, "scale": 0.2, "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.0,
                "hsv_s": 0.0, "hsv_v": 0.3, "mosaic": 0.5, "close_mosaic": 15, "mixup": 0.0, "copy_paste": 0.0,
                "save_period": -1, "plots": True, "seed": 42, "name": "yolo26m_seg"}
    assert {k: kw[k] for k in expected} == expected
    assert kw["data"] == str(path) and Path(kw["project"]) == tmp_path / "runs"
    assert "lrf" not in kw                                   # unset -> Ultralytics default


def test_prepare_needs_the_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        trainer_for("yolo26m_seg.yaml", tmp_path).prepare()


def test_model_source_prefers_local_weights(tmp_path):
    t = trainer_for("yolo26m_seg.yaml", tmp_path)
    assert t.model_source() == "yolo26m-seg.pt"
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/yolo26m-seg.pt").write_bytes(b"x")
    assert t.model_source() == str(tmp_path / "weights/yolo26m-seg.pt")


def test_albumentations_specs_and_patch(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, **kw):
            calls.append(kw)

    module = SimpleNamespace(CLAHE=Fake, OneOf=Fake, Blur=Fake)
    out = build_transforms([{"name": "CLAHE", "clip_limit": [1, 4], "p": 0.2},
                            {"name": "OneOf", "transforms": [{"name": "Blur", "blur_limit": 3}], "p": 0.1}], module)
    assert len(out) == 2 and calls[0] == {"clip_limit": (1, 4), "p": 0.2} and isinstance(calls[2]["transforms"][0], Fake)
    with pytest.raises(ValueError):
        build_transforms([{"name": "Nope"}], module)
    with pytest.raises(ValueError):
        validate_specs([{"p": 1}])

    aug = pytest.importorskip("ultralytics.data.augment")
    original = aug.Albumentations.__init__
    seen = {}
    monkeypatch.setattr("breast_seg.training.augment.build_transforms", lambda specs: seen.setdefault("specs", specs) and [])
    with ultralytics_albumentations([{"name": "CLAHE", "p": 0.25}]):
        assert aug.Albumentations.__init__ is not original
        aug.Albumentations(p=1.0)
    assert aug.Albumentations.__init__ is original
    assert seen["specs"] == [{"name": "CLAHE", "p": 0.25}]


def test_use_split_links_fold_tree(tmp_path):
    src = tmp_path / "src"
    yolo_layout(src, splits=("train",), n=3)
    imgs = {p.stem: p for p in (src / "images/train").glob("*.png")}
    lbls = {p.stem: src / "labels/train" / f"{p.stem}.txt" for p in imgs.values()}
    split_of = {"imgtrain0": "train", "imgtrain1": "val", "imgtrain2": "test"}
    t = trainer_for("yolo26m_seg.yaml", tmp_path, "name=cv_fold0")
    lbls.pop("imgtrain2")                                   # unlabelled test image is allowed
    ft = t.use_split(SplitSpec(split_of, imgs, lbls), tmp_path / "work/fold0")
    assert (tmp_path / "work/fold0/images/test/imgtrain2.png").is_symlink()
    assert (tmp_path / "work/fold0/labels/val/imgtrain1.txt").is_file()
    assert not (tmp_path / "work/fold0/labels/test/imgtrain2.txt").exists()
    data = yaml.safe_load(ft.prepare().read_text())
    assert Path(data["path"]) == tmp_path / "work/fold0" and ft.config.name == "cv_fold0"
    with pytest.raises(ValueError, match="unlabelled"):
        t.use_split(SplitSpec({"imgtrain2": "train"}, imgs, {}), tmp_path / "work/x")


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def fake_result(shape, masks, cls, conf):
    return SimpleNamespace(orig_shape=shape, masks=SimpleNamespace(data=_Arr(masks)),
                           boxes=SimpleNamespace(cls=_Arr(cls), conf=_Arr(conf)))


def test_seg_predict_flattens_instances(tmp_path, monkeypatch):
    images = []
    for name in ("m1", "cc_c1"):
        p = tmp_path / f"{name}.png"
        cv2.imwrite(str(p), np.zeros((16, 16), np.uint8))
        images.append(p)
    m = np.zeros((2, 16, 16), np.float32)
    m[0, :, :8] = 1
    m[1, 0:4, 0:4] = 1

    nipple = np.zeros((1, 16, 16), np.float32)
    nipple[0, 6:9, 12:15] = 1

    class FakeModel:
        calls = []

        def predict(self, paths, **kw):
            FakeModel.calls.append(kw["conf"])
            if kw["conf"] == 0.001:                         # the nipple fallback pass, one image at a time
                assert isinstance(paths, str)
                return [fake_result((16, 16), nipple, [2], [0.004])]
            assert kw["conf"] == 0.01                       # inference at the rescue floor
            return [fake_result((16, 16), m, [1, 0], [0.9, 0.1]) for _ in paths]

    t = trainer_for("yolo26m_seg.yaml", tmp_path)
    monkeypatch.setattr(type(t), "_model", lambda self, weights: FakeModel())
    out = t.predict(images, tmp_path / "pred")
    mlo = cv2.imread(str(out[0]), cv2.IMREAD_GRAYSCALE)
    cc = cv2.imread(str(out[1]), cv2.IMREAD_GRAYSCALE)
    assert mlo[0, 0] == 1 and mlo[10, 2] == 2           # low-confidence pectoral rescued on MLO
    assert cc[0, 0] == 2                                # ... but not on CC
    assert (tmp_path / "pred/flatten_stats.csv").is_file()
    from breast_seg.evaluation.predictions import read_nipple_fallback

    assert FakeModel.calls == [0.01, 0.001, 0.001]          # neither view has a nipple at the floor
    fallback = read_nipple_fallback(tmp_path / "pred", "m1")
    assert fallback.dtype == np.uint8 and fallback[7, 13] == 1 and fallback.sum() == 9


def test_seg_predict_deletes_stale_nipple_fallbacks(tmp_path, monkeypatch):
    """A fallback an earlier run left in the prediction folder never outlives its label map: deleted when the view
    now has a nipple, replaced (and no stale ``.npz`` shadowing the new ``.png``) when it still has none."""
    from breast_seg.evaluation.predictions import read_nipple_fallback, write_nipple_fallback

    images = []
    for name in ("m1", "m2"):
        p = tmp_path / f"{name}.png"
        cv2.imwrite(str(p), np.zeros((16, 16), np.uint8))
        images.append(p)
    pred = tmp_path / "pred"
    for stem in ("m1", "m2"):
        write_nipple_fallback(pred, stem, np.full((16, 16), 0.5, np.float32))      # .npz of an earlier run
        cv2.imwrite(str(pred / "nipple" / f"{stem}.png"), np.ones((16, 16), np.uint8))
    breast = np.zeros((2, 16, 16), np.float32)
    breast[0, :, :8] = 1
    breast[1, 6:9, 12:15] = 1
    nipple = np.zeros((1, 16, 16), np.float32)
    nipple[0, 0:2, 0:2] = 1

    class FakeModel:
        def predict(self, paths, **kw):
            if kw["conf"] == 0.001:                         # fallback pass: only m2 has no nipple
                assert paths.endswith("m2.png")
                return [fake_result((16, 16), nipple, [2], [0.004])]
            return [fake_result((16, 16), breast, [1, 2], [0.9, 0.9 if Path(x).stem == "m1" else 0.0001])
                    for x in paths]

    t = trainer_for("yolo26m_seg.yaml", tmp_path)
    monkeypatch.setattr(type(t), "_model", lambda self, weights: FakeModel())
    t.predict(images, pred)
    assert (cv2.imread(str(pred / "m1.png"), cv2.IMREAD_GRAYSCALE) == 3).any()
    assert read_nipple_fallback(pred, "m1") is None and not list((pred / "nipple").glob("m1.*"))
    assert sorted(p.name for p in (pred / "nipple").glob("m2.*")) == ["m2.png"]
    fallback = read_nipple_fallback(pred, "m2")
    assert fallback.dtype == np.uint8 and fallback.sum() == 4 and fallback[0, 0] == 1


def test_export_copies_weights_to_the_registry_layout(tmp_path):
    """``<weights_dir>/segmentation/<file>``: the path configs/models.yaml and the release use."""
    t = trainer_for("yolo26m_seg.yaml", tmp_path)
    with pytest.raises(FileNotFoundError):
        t.export()
    t.best_weights().parent.mkdir(parents=True)
    t.best_weights().write_bytes(b"w")
    assert t.export() == tmp_path / "weights/segmentation/yolo26m_seg.pt"
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    registered = {m["weights"] for m in models["models"] if m.get("task") == "segmentation"}
    for config in ("yolo26m_seg.yaml", "unet.yaml", "segformer_b3.yaml"):
        c = trainer_for(config, tmp_path).config.export
        assert c.dir == "${weights_dir}/segmentation" and f"segmentation/{c.filename}" in registered
