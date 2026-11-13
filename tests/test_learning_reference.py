"""Saved OOF references must be complete, finite and paired by identity."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import depthlab.calibration_study as study
import depthlab.learning_curve as learning


@pytest.fixture
def setup(tmp_path, monkeypatch):
    ids = np.arange(10)
    scenes = np.repeat([f"room{i}" for i in range(5)], 2)
    assignment = np.repeat(np.arange(5), 2)
    raw = np.ones((480, 640), dtype=np.float32)
    class SyntheticTraining:
        def __enter__(self):
            self.ids, self.scenes, self.assignment = ids, scenes, assignment
            self.ratios = np.ones(10)
            self.features = np.zeros((10, 8))
            self.samples = self.targets = np.ones((10, 8), dtype=np.float32)
            self.predictions = {int(index): raw for index in ids}
            self.handle = None
            return self
        def __exit__(self, *exc): pass
        def fingerprint(self, module): return {"scope": "synthetic_only"}
    scalar = {"kind": "scalar", "scale": 1.}
    models = [{"median": scalar, "spline_rmse": scalar} for _ in range(5)]
    base = tmp_path / "base"
    base.mkdir()
    (base / "selection.json").write_text("synthetic reference")
    for fold in range(5):
        (base / f"fold{fold}").mkdir()
        (base / f"fold{fold}" / "models.json").write_text(json.dumps(models[fold]))
    rows = [{"dataset_index": int(index), "scene_id": str(scenes[index]),
             "fold": int(assignment[index]), "method": method, "abs_rel": 0., "rmse": 0., "d1": 1.}
            for index in ids for method in ("median", "spline_rmse")]
    monkeypatch.setattr(learning, "TrainingData", lambda *args: SyntheticTraining())
    monkeypatch.setattr(learning, "load_base_study", lambda *args: models)
    monkeypatch.setattr(learning, "fit_map", lambda *args: (scalar, {}))
    monkeypatch.setattr(study, "nyu_depth", lambda *args: raw)
    return SimpleNamespace(base=base, rows=rows, output=tmp_path / "output")


def run_reference(setup):
    with (setup.base / "oof_per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(setup.rows[0]))
        writer.writeheader()
        writer.writerows(setup.rows)
    learning.run(SimpleNamespace(data_root="unused", predictions="unused", output=setup.output,
                                 base_study=setup.base, legacy_base_frozen=None,
                                 expected_legacy_frozen_sha256=None, smoke_only=False))


@pytest.mark.parametrize("changed", ["nan", "duplicate", "scene", "fold", "missing"])
def test_learning_rejects_invalid_saved_reference_before_completion(setup, changed):
    if changed == "nan": setup.rows[0]["abs_rel"] = float("nan")
    elif changed == "duplicate": setup.rows[0] = dict(setup.rows[2])
    elif changed == "scene": setup.rows[0]["scene_id"] = "other_room"
    elif changed == "fold": setup.rows[0]["fold"] = 4
    else: setup.rows.pop()
    with pytest.raises(ValueError, match="[Rr]eference"):
        run_reference(setup)
    assert not (setup.output / "completion.json").exists()
    assert not (setup.output / "learning_curve.csv").exists()


def test_learning_accepts_matching_saved_reference(setup):
    run_reference(setup)
    receipt = json.loads((setup.output / "completion.json").read_text())
    assert receipt["complete"] is True and receipt["full_oof_max_difference"] == 0.
    assert len(receipt["reference_oof_sha256"]) == 64
