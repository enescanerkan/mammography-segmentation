"""Cross-validation runner end to end on the synthetic cohort with a fake trainer (no framework, no training)."""
from pathlib import Path

import cv2
import pytest

from breast_seg.training import CrossValidationRunner, load_cv_config
from breast_seg.training.yolo import YoloSegTrainer
from tests.test_datasets_builder import build_cohort

ROOT = Path(__file__).resolve().parents[1]


class FakeFoldTrainer(YoloSegTrainer):
    """Fit writes a marker; predict writes the reference label map of each test image."""

    fitted = []

    def fit(self):
        self.prepare()
        FakeFoldTrainer.fitted.append(self.config.name)
        return None

    def predict(self, images, out_dir, weights=None):
        out = []
        for p in images:
            p = Path(p).resolve()                                   # <out>/yolo/images/<split>/<stem>.png
            label = cv2.imread(str(p.parents[3] / "label_maps" / p.name), cv2.IMREAD_UNCHANGED)
            target = Path(out_dir) / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), label)
            out.append(target)
        return out


def factory(cfg, paths):
    return FakeFoldTrainer(cfg, paths)


@pytest.fixture()
def cohort(tmp_path):
    return build_cohort(tmp_path, n_studies=10, n_folds=5)


@pytest.mark.parametrize("name", ["cv10_yolo26m", "cv10_segformer_b3", "cv10_unet", "cv10_nnunet"])
def test_every_committed_cv_config_reads_the_built_folds(cohort, name):
    cfg = load_cv_config(ROOT / f"configs/cv/{name}.yaml", {"n_folds": 5})
    runner = CrossValidationRunner(cfg, cohort, ROOT / "configs/cv")
    a = runner.assignment()
    assert len(a.fold_of) == 40 and all(a.is_labelled(s) for s in a.fold_of)
    assert a.fold_of["m4R"] == a.fold_of["cc_c4L"] == 4                 # both breasts of a study in one fold
    s = runner.split(2)
    assert set(s.test) == {"m2L", "m2R", "cc_c2L", "cc_c2R", "m7L", "m7R", "cc_c7L", "cc_c7R"}
    assert not set(s.test) & (set(s.train) | set(s.val))


def test_cv_runner_on_committed_config(cohort):
    FakeFoldTrainer.fitted = []
    cfg = load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"n_folds": 5})
    assert (cfg.overrides["hyp.epochs"], cfg.overrides["hyp.patience"]) == (300, 300)
    runner = CrossValidationRunner(cfg, cohort, ROOT / "configs/cv", trainer_factory=factory)
    report = runner.run()
    assert sorted(FakeFoldTrainer.fitted) == [f"cv10_yolo26m_fold{i}" for i in range(5)]
    assert report.oof.complete and report.oof.n_images == 40
    fold_tree = cohort.dir("runs_dir") / "cv/cv10_yolo26m/fold2"
    assert (fold_tree / "images/test/m7R.png").is_symlink() and (fold_tree / "labels/test/m7R.txt").is_symlink()
    assert (cohort.dir("runs_dir") / "cv10_yolo26m_fold2/data.yaml").is_file()
    ev = report.evaluation
    assert len(ev.pairs) == 20 and ev.views.fold.notna().all()
    assert len(ev.summary["per_fold"]) == 5 and "decision" in ev.summary
    assert ev.summary["dice"]["MLO"]["pectoral"] == pytest.approx(1.0)
    assert ev.summary["mlo_only"]["n"] == 20 and ev.views[ev.views.view == "MLO"].nipple_source.eq("argmax").all()
    res = cohort.dir("results_dir")
    assert (res / "cv10_yolo26m_pairs.csv").is_file() and (res / "cv10_yolo26m_summary.json").is_file()
    # a second run skips finished folds
    FakeFoldTrainer.fitted = []
    runner.run()
    assert FakeFoldTrainer.fitted == []


def test_cv_runner_rejects_unknown_fold_and_missing_sections(cohort):
    from breast_seg.config import ConfigError

    with pytest.raises(ConfigError):
        load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"run_folds": [12]})
    with pytest.raises(ConfigError):
        load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"folds": None})
