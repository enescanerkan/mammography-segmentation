"""nnU-Net trainer with a recording command runner (nnU-Net itself is never invoked)."""
import json
from pathlib import Path

import pytest

from breast_seg.training import SplitSpec, load_train_config
from breast_seg.training.common import CommandRunner
from breast_seg.training.nnunet import NnUNetTrainer, derive_plans, install_trainer_variants, keep_nipple_probabilities
from tests.test_datasets_builder import build_cohort

ROOT = Path(__file__).resolve().parents[1]


def base_plans(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plans_name": "nnUNetResEncUNetMPlans",
                                "configurations": {"2d": {"patch_size": [576, 448], "batch_size": 13,
                                                          "data_identifier": "nnUNetPlans_2d",
                                                          "architecture": {"arch_kwargs": {"n_stages": 7}}}}}))


@pytest.fixture()
def cohort(tmp_path):
    return build_cohort(tmp_path, n_studies=6)


def test_derive_plans_keeps_topology(tmp_path):
    base_plans(tmp_path / "a.json")
    out = derive_plans(tmp_path / "a.json", tmp_path / "b.json", "P_fullM", [640, 640], 12)
    conf = out["configurations"]["2d"]
    assert out["plans_name"] == "P_fullM" and conf["patch_size"] == [640, 640] and conf["batch_size"] == 12
    assert conf["architecture"]["arch_kwargs"]["n_stages"] == 7 and conf["data_identifier"] == "nnUNetPlans_2d"


def test_prepare_fit_predict_commands(cohort, tmp_path):
    cfg = load_train_config(ROOT / "configs/train/nnunet_resenc_m.yaml")
    runner = CommandRunner(dry_run=True)
    t = NnUNetTrainer(cfg, cohort, runner)
    base_plans(cohort.dir("nnunet_preprocessed") / "Dataset601_BreastSeg/nnUNetResEncUNetMPlans.json")
    ds = t.prepare()                                    # the raw dataset of build-dataset is reused
    assert (ds / "dataset.json").is_file() and len(list((ds / "imagesTs").glob("*_0000.png"))) == 4
    assert t.convert() == ds and len(list((ds / "imagesTr").glob("*_0000.png"))) == 20
    labels = json.loads((ds / "dataset.json").read_text())["labels"]
    assert labels == {"background": 0, "pectoral": 1, "breast-tissue": 2, "nipple": 3}
    splits = json.loads((cohort.dir("nnunet_preprocessed") / "Dataset601_BreastSeg/splits_final.json").read_text())
    assert len(splits) == 1
    pre = cohort.dir("nnunet_preprocessed") / "Dataset601_BreastSeg"
    derived = json.loads((pre / "nnUNetResEncUNetMPlans_fullM.json").read_text())
    assert derived["configurations"]["2d"]["patch_size"] == [640, 640]
    tools = [c[0] for c in runner.history]
    assert tools == ["nnUNetv2_plan_and_preprocess", "nnUNetv2_plan_experiment", "nnUNetv2_preprocess"]
    assert runner.history[1][-2:] == ["-pl", "nnUNetPlannerResEncM"]
    assert "nnUNetResEncUNetMPlans_fullM" in runner.history[2]
    t.fit()
    assert runner.history[-1][:4] == ["nnUNetv2_train", "601", "2d", "0"] and "nnUNetTrainerMammo" in runner.history[-1]
    imgs = sorted((ds / "imagesTs").glob("*.png"))
    out = t.predict(imgs, tmp_path / "pred")
    cmd = runner.history[-1]
    assert cmd[0] == "nnUNetv2_predict" and cmd[cmd.index("-chk") + 1] == "checkpoint_final.pth"
    assert out[0].parent == tmp_path / "pred" and "--disable_progress_bar" in cmd and "--save_probabilities" in cmd
    t.export()
    assert runner.history[-1][0] == "nnUNetv2_export_model_to_zip"
    assert t.env()["nnUNet_raw"] == str(cohort.dir("nnunet_raw"))


def test_prepare_folds_builds_one_cv_dataset(cohort):
    cfg = load_train_config(ROOT / "configs/train/nnunet_resenc_m.yaml",
                            {"dataset_id": 603, "dataset_name": "BreastSegCV10", "transfer_plans_from": 601})
    runner = CommandRunner(dry_run=True)
    t = NnUNetTrainer(cfg, cohort, runner)
    sem = cohort.resolve("${data_root}/vindr/semantic")
    stems = sorted(p.stem for p in (sem / "images/train").glob("*.png"))[:6]
    images = {s: sem / "images/train" / f"{s}.png" for s in stems}
    masks = {s: sem / "masks/train" / f"{s}.png" for s in stems[:5]}       # last one unlabelled
    specs = []
    for f in range(2):
        split_of = {s: ("test" if i % 2 == f else "train" if i < 4 else "val") for i, s in enumerate(stems)}
        split_of = {s: sp for s, sp in split_of.items() if sp == "test" or s in masks}
        specs.append(SplitSpec(split_of, images, masks, masks, fold=f))
    ds = t.prepare_folds(specs)
    splits = json.loads((cohort.dir("nnunet_preprocessed") / "Dataset603_BreastSegCV10/splits_final.json").read_text())
    assert len(splits) == 2 and len(list((ds / "imagesTr").glob("*.png"))) == 5
    assert len(list((ds / "imagesTest_fold0").glob("*_0000.png"))) == len(specs[0].stems("test"))
    assert [c[0] for c in runner.history] == ["nnUNetv2_extract_fingerprint", "nnUNetv2_move_plans_between_datasets",
                                             "nnUNetv2_preprocess"]
    assert t.use_split(specs[1], Path("unused")).config.fold == 1


def test_install_trainer_variants(tmp_path):
    target = install_trainer_variants(tmp_path / "variants")
    text = target.read_text()
    assert "class nnUNetTrainerMammo(" in text and "mirror_axes = (1,)" in text


def test_prepare_rewrites_the_held_out_splits_when_reusing_the_raw_dataset(cohort):
    """``reuse_raw`` skips the conversion but never the splits: a stale ``splits_final.json`` (here ten folds of an
    earlier cross-validation run) is replaced by fold 0 = the held-out train / validation split of the index."""
    import pandas as pd

    cfg = load_train_config(ROOT / "configs/train/nnunet_resenc_m.yaml")
    assert cfg.data.reuse_raw
    t = NnUNetTrainer(cfg, cohort, CommandRunner(dry_run=True))
    base_plans(cohort.dir("nnunet_preprocessed") / "Dataset601_BreastSeg/nnUNetResEncUNetMPlans.json")
    splits_file = cohort.dir("nnunet_preprocessed") / "Dataset601_BreastSeg/splits_final.json"
    raw_json = cohort.dir("nnunet_raw") / "Dataset601_BreastSeg/dataset.json"
    stale = [{"train": ["x"], "val": ["y"]}] * 10
    splits_file.write_text(json.dumps(stale))
    before = raw_json.stat().st_mtime_ns
    t.prepare()
    assert raw_json.stat().st_mtime_ns == before                 # the raw dataset was reused, not converted
    splits = json.loads(splits_file.read_text())
    index = pd.read_csv(cohort.resolve("${data_root}/vindr/semantic/index.csv"), dtype={"stem": str})
    assert splits == [{"train": sorted(index.stem[index.split == "train"]), "val": sorted(index.stem[index.split == "val"])}]
    # a semantic index naming images the reused raw dataset does not have is refused
    index.loc[index.split == "test", "split"] = "val"
    index.drop(columns=["image", "mask"], errors="ignore").to_csv(
        cohort.resolve("${data_root}/vindr/semantic/index.csv"), index=False)
    with pytest.raises(ValueError, match="not in imagesTr"):
        t.prepare()


def test_exported_probabilities_become_nipple_fallbacks(tmp_path):
    import cv2
    import numpy as np

    from breast_seg.evaluation.predictions import read_nipple_fallback

    with_nipple, without = np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8)
    with_nipple[2, 2] = 3
    prob = np.zeros((4, 1, 8, 8), np.float32)
    prob[3, 0, 5, 6] = 0.3
    for stem, label in (("a", with_nipple), ("b", without)):
        cv2.imwrite(str(tmp_path / f"{stem}.png"), label)
        np.savez(tmp_path / f"{stem}.npz", probabilities=prob)
        (tmp_path / f"{stem}.pkl").write_bytes(b"x")
    keep_nipple_probabilities(tmp_path)
    assert not list(tmp_path.glob("*.npz")) and not list(tmp_path.glob("*.pkl"))
    assert read_nipple_fallback(tmp_path, "a") is None
    p = read_nipple_fallback(tmp_path, "b")
    assert p.shape == (8, 8) and p.dtype == np.float16 and float(p[5, 6]) == pytest.approx(0.3, abs=1e-3)
