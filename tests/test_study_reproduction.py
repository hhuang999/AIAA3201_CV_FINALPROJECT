import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from depthlab.calibration_study import (
    CANDIDATES, apply_calibration, choose_primary, fit_fold_models, fresh_output,
    huber_log_scale, select_from_oof,
)
from depthlab.learning_curve import FRACTIONS, nested_scene_subsets, summarize_learning
from depthlab.calibration_analysis import describe, saved_metric_arrays, signed_bias_rows, summarize_saved


def test_discovery_contains_original_18_candidates():
    assert len(CANDIDATES) == 18 and len({s["name"] for s in CANDIDATES}) == 18
    assert {s["kind"] for s in CANDIDATES} == {"scalar", "affine", "power", "bounded", "spline", "ridge"}
    assert sum(s.get("loss") == "balanced" for s in CANDIDATES) == 5


def test_all_discovery_candidates_fit_on_tiny_cpu_inputs():
    torch.set_num_threads(2)
    samples = np.geomspace(.3, 8, 384).reshape(3, 128).astype(np.float32)
    targets = (.91 * samples + .1).astype(np.float32)
    data = SimpleNamespace(samples=samples, targets=targets,
                           ratios=np.median(targets / samples, axis=1).astype(np.float64),
                           features=np.random.default_rng(9).normal(size=(3, 8)))
    models, details = fit_fold_models(data, np.arange(3))
    assert len(models) == 21 and len(details) == 18
    depth = np.geomspace(.01, 100, 200)
    for name, model in models.items():
        output = apply_calibration(depth, model, features=data.features[0])
        assert np.isfinite(output).all() and np.all((output >= .001) & (output <= 10.))
        assert np.all(np.diff(output) >= -1e-12), name
        if name in details and details[name]["optimization"]:
            trace = details[name]["optimization"]
            assert trace["objective_last"] <= trace["objective_first"] + 1e-8, name


def test_huber_resists_extreme_image_ratio_and_handles_constant_inputs():
    assert huber_log_scale([.9, .9, .9, 100]) == .9
    np.testing.assert_allclose(huber_log_scale([.9] * 4), .9)
    with pytest.raises(ValueError): huber_log_scale([0., 1.])


def test_original_discovery_selection_confident_and_provisional_branches():
    rows = [{"method": "a", "abs_rel": .09, "rmse": .3, "d1": .92,
             "joint_positive_means": True, "joint_adjusted_positive": False, "minimum_normalized_gain": .1},
            {"method": "b", "abs_rel": .1, "rmse": .31, "d1": .91,
             "joint_positive_means": True, "joint_adjusted_positive": True, "minimum_normalized_gain": .05}]
    assert choose_primary(rows) == ("b", "simultaneous_confidence_gate")
    rows[1]["joint_adjusted_positive"] = False
    assert choose_primary(rows) == ("a", "provisional_mean_only")
    for row in rows: row["joint_positive_means"] = False
    assert choose_primary(rows) == (None, "no_joint_positive_candidate")


def test_training_oof_scene_bootstrap_selects_without_test_access():
    groups = np.repeat(np.arange(4), 2)
    base = np.tile([.11, .4, .89], (8, 1))
    values = {name: base.copy() for name in ("raw", "median", "huber")}
    for spec in CANDIDATES:
        values[spec["name"]] = base + [-.005, -.01, .005]
    values["scalar_absrel"] = base + [-.01, -.02, .01]
    table, intervals, selected = select_from_oof(values, groups, repeats=64)
    assert selected["primary"] == "scalar_absrel"
    assert selected["test_read"] is False and selected["deployment_changed"] is False
    assert len(table) == 21 and len(intervals) == 18
    np.testing.assert_allclose(intervals["scalar_absrel"]["mean_gain"]["abs_rel"], .01)
    broken = copy.deepcopy(values)
    broken["raw"][0, 0] = np.nan
    with pytest.raises(ValueError): select_from_oof(broken, groups, repeats=8)


def test_nested_scene_subsets_have_original_order_and_no_heldout_leakage():
    scenes = np.repeat([f"room_{i:02d}" for i in range(15)], 2)
    assignment = np.repeat(np.arange(15) % 5, 2)
    subsets, heldout = nested_scene_subsets(assignment, scenes, 2)
    fitting = np.flatnonzero(assignment != 2)
    shuffled = np.random.default_rng(20261111 + 2).permutation(np.unique(scenes[fitting]))
    previous = set()
    for fraction in FRACTIONS:
        wanted = fitting[np.isin(scenes[fitting], shuffled[:int(np.ceil(fraction * len(shuffled)))])]
        np.testing.assert_array_equal(subsets[fraction], wanted)
        assert previous <= set(subsets[fraction])
        assert not set(scenes[subsets[fraction]]) & set(scenes[heldout])
        previous = set(subsets[fraction])
    np.testing.assert_array_equal(subsets[1.], fitting)


def learning_rows():
    rows, fits = [], []
    for fraction in FRACTIONS:
        fits.append({"fraction": fraction, "fit_images": 10, "fit_scenes": 5})
        for index in range(4):
            for method, metrics in (("median", [.1, .4, .9]), ("spline_rmse", [.09, .38, .91])):
                rows.append({"fraction": fraction, "method": method, "dataset_index": index,
                             "scene_id": f"room{index // 2}", "abs_rel": metrics[0], "rmse": metrics[1], "d1": metrics[2]})
    return rows, fits


def test_learning_curve_summary_uses_paired_scene_intervals():
    rows, fits = learning_rows()
    table, intervals, values = summarize_learning(rows, fits, np.arange(4), ["room0", "room0", "room1", "room1"], repeats=32)
    assert len(table) == 8 and len(values) == 8
    np.testing.assert_allclose(intervals["1.0"]["gain"], [.01, .02, .01])
    with pytest.raises(ValueError): summarize_learning(rows[:-1], fits, np.arange(4), ["room0", "room0", "room1", "room1"], repeats=8)


def saved_rows(new_names=False):
    rows = []
    names = ("raw", "global_scale", "monotone_curve", "residual_scale") if new_names else ("raw", "median", "spline_rmse", "original_residual")
    for pos, name in enumerate(names):
        for index in range(5):
            row = {"method": name, "dataset_index": index, "scene_id": f"room{index // 2}",
                   "abs_rel": .12 - pos * .01, "rmse": .45 - pos * .01, "d1": .86 + pos * .01}
            if new_names:
                row["index"] = row.pop("dataset_index")
                row["scene"] = row.pop("scene_id")
                row["delta1"] = row.pop("d1")
            rows.append(row)
    return rows


@pytest.mark.parametrize("new_names", [False, True])
def test_saved_distribution_accepts_old_and_rebuilt_columns(new_names):
    rows = saved_rows(new_names)
    distributions, comparisons, intervals, metadata = summarize_saved(rows, expected_images=5, repeats=32)
    assert len(distributions) == 12 and len(comparisons) == 15 and len(intervals) == 5
    assert metadata["model_selection"] is False and metadata["deployment_changed"] is False
    primary = "residual_scale" if new_names else "original_residual"
    np.testing.assert_allclose(intervals[primary + "_vs_spline_rmse"]["gain"], [.01, .01, .01])
    with pytest.raises(ValueError): saved_metric_arrays(rows + rows[:1], expected_images=5)


def test_signed_bias_retains_direction_and_clipped_scale_definition():
    truth = np.array([.5, 1.5, 3., 4.])
    rows = signed_bias_rows(2 * truth, truth, .5)
    raw = next(row for row in rows if row["method"] == "raw" and row["depth_bin"] == "all")
    corrected = next(row for row in rows if row["method"] == "median" and row["depth_bin"] == "all")
    assert raw["signed_relative_error"] == 1. and raw["overprediction_fraction"] == 1.
    assert corrected["signed_relative_error"] == 0. and corrected["median_gt_over_prediction"] == 1.
    assert describe([1., 2., 3.])["median"] == 2.


def test_studies_refuse_to_overwrite_evidence(tmp_path):
    with pytest.raises(FileExistsError): fresh_output(tmp_path)


def test_legacy_learning_reference_is_bound_to_frozen_hash_chain(tmp_path):
    import json
    from depthlab.cache import sha256
    from depthlab.learning_curve import load_legacy_base_study
    prediction_root = tmp_path / "cache"
    prediction_root.mkdir()
    np.save(prediction_root / "depth.npy", np.ones((5, 2, 2), dtype=np.float32))
    base = tmp_path / "base"
    selection_root = base / "selection"
    selection_root.mkdir(parents=True)
    ids = np.arange(5)
    assignment = np.arange(5)
    data = SimpleNamespace(ids=ids, assignment=assignment, prediction_root=prediction_root,
                           fingerprint=lambda module: {"dataset_sha256": "a" * 64, "split_sha256": "b" * 64})
    evidence = []
    for fold in range(5):
        folder = base / "cv" / f"fold{fold}"
        folder.mkdir(parents=True)
        (folder / "models.json").write_text(json.dumps({"median": {"kind": "scalar", "scale": .95}}))
        (folder / "per_image.csv").write_text("synthetic fixture only")
        (folder / "completion.json").write_text(json.dumps({"complete": True, "fold": fold,
                "fit_ids": ids[assignment != fold].tolist(), "validation_ids": ids[assignment == fold].tolist()}))
        evidence.append({"fold": fold, "completion_sha256": sha256(folder / "completion.json"),
                         "metrics_sha256": sha256(folder / "per_image.csv"), "models_sha256": sha256(folder / "models.json")})
    selection = selection_root / "selection.json"
    selection.write_text(json.dumps({"test_read": False, "fold_evidence": evidence}))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"primary": "spline_rmse", "test_read": False, "train_ids": ids.tolist(),
                                "dataset_sha256": "a" * 64, "split_sha256": "b" * 64,
                                "train_cache_sha256": sha256(prediction_root / "depth.npy"),
                                "selection_sha256": sha256(selection)}))
    models = load_legacy_base_study(base, frozen, data, sha256(frozen))
    assert len(models) == 5 and models[0]["median"]["scale"] == .95
    (base / "cv" / "fold0" / "models.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        load_legacy_base_study(base, frozen, data, sha256(frozen))


def test_saved_distribution_rejects_missing_metric_instead_of_aliasing_it():
    rows = saved_rows()
    del rows[0]["abs_rel"]
    with pytest.raises(ValueError, match="metric column"):
        saved_metric_arrays(rows, expected_images=5)


def test_discovery_and_learning_smoke_entrypoints_on_synthetic_training_only(monkeypatch, tmp_path):
    import json
    import depthlab.calibration_study as study
    import depthlab.learning_curve as learning
    from depthlab.frozen import depth_features
    raw = np.broadcast_to(np.linspace(.5, 6., 640, dtype=np.float32), (480, 640)).copy()
    truth = (1.03 * raw).astype(np.float32)

    class SyntheticTraining:
        def __enter__(self):
            self.ids = np.arange(10)
            self.scenes = np.repeat([f"room{i}" for i in range(5)], 2)
            self.assignment = np.repeat(np.arange(5), 2)
            self.predictions = {int(index): raw for index in self.ids}
            self.handle = None
            self.features = np.tile(depth_features(raw), (10, 1))
            self.ratios = np.full(10, 1.03)
            self.samples = np.tile(np.linspace(.5, 6., 64, dtype=np.float32), (10, 1))
            self.targets = (1.03 * self.samples).astype(np.float32)
            return self

        def __exit__(self, *exc): pass
        def fingerprint(self, module): return {"scope": "synthetic_training_only", "test_read": False}

    monkeypatch.setattr(study, "TrainingData", lambda *args: SyntheticTraining())
    monkeypatch.setattr(learning, "TrainingData", lambda *args: SyntheticTraining())
    monkeypatch.setattr(study, "nyu_depth", lambda *args: truth)
    discovery_output = tmp_path / "discovery"
    study.run(SimpleNamespace(data_root="unused", predictions="unused", output=discovery_output, smoke_only=True))
    assert json.loads((discovery_output / "completion.json").read_text())["selection_performed"] is False
    assert len(json.loads((discovery_output / "fold0" / "models.json").read_text())) == 21
    assert not (discovery_output / "selection.json").exists()

    base = tmp_path / "base"
    (base / "fold0").mkdir(parents=True)
    (base / "fold0" / "models.json").write_text("synthetic reference fixture")
    (base / "selection.json").write_text("synthetic reference fixture")
    monkeypatch.setattr(learning, "load_base_study", lambda *args: [{}] * 5)
    learning_output = tmp_path / "learning"
    learning.run(SimpleNamespace(data_root="unused", predictions="unused", output=learning_output,
                                base_study=base, legacy_base_frozen=None, expected_legacy_frozen_sha256=None, smoke_only=True))
    completion = json.loads((learning_output / "completion.json").read_text())
    assert completion["smoke"] is True and completion["model_selection"] is False
    assert (learning_output / "fold0" / "fits.json").exists()
