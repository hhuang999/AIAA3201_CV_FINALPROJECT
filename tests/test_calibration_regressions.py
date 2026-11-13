import copy
import hashlib
import json

import numpy as np
import pytest

from depthlab.calibration import (
    SEED, SAMPLE_SEED, candidate_specs, fit_curve, fit_residual,
    log_features, prepare_training, residual_log_scale, scene_folds,
)
from depthlab.frozen import apply_frozen, frozen_predictions, load_frozen, validate_frozen


def frozen_fixture():
    knots = np.log([.3, .7, 1.5, 3., 6., 12.]).tolist()
    spline = {"kind": "spline", "x_knots": knots, "y_knots": knots}
    frozen = {"schema": "B_residual_v1", "complete": True,
              "primary": "residual_median_ratio_l1_a0.5",
              "candidate_spec": {"name": "residual_median_ratio_l1_a0.5", "kind": "ridge",
                                 "target": "median_ratio", "lam": 1., "shrinkage": .5, "max_log_scale": .15},
              "base_model": spline,
              "residual_model": {"kind": "ridge_residual", "feature_mean": [0.] * 8,
                                 "feature_std": [1.] * 8, "coefficients": [0.] * 8,
                                 "bias": -.4, "ridge_lambda": 1., "shrinkage": .5, "max_log_scale": .15},
              "controls": {"raw": {"kind": "scalar", "scale": 1.},
                           "median": {"kind": "scalar", "scale": .95}, "spline_rmse": copy.deepcopy(spline)},
              "train_ids": list(range(795)), "training_scenes": 249,
              "source_sha256": {"original.py": "0" * 64}, "test_read": False,
              "historical_test_exposure": True, "final_decision": "Prefrozen selected candidate; no retuning"}
    for key in ("dataset_sha256", "split_sha256", "train_cache_sha256", "selection_sha256",
                "selection_protocol_sha256", "prepared_manifest_sha256", "base_frozen_sha256"):
        frozen[key] = "0" * 64
    return frozen


def test_constant_residual_features_remain_finite():
    features = np.ones((4, 8), dtype=np.float64)
    target = np.array([-.03, -.01, .01, .03])
    model = fit_residual(features, target, 1., .5)
    assert np.isfinite(model["coefficient"]).all()
    np.testing.assert_allclose(residual_log_scale(features, model), 0.)


def test_residual_with_one_constant_column_remains_finite():
    features = np.random.default_rng(13).normal(size=(10, 8))
    features[:, 3] = 1.
    model = fit_residual(features, np.arange(10) / 100, 1., .5)
    assert model["coefficient"][3] == 0.
    assert np.isfinite(residual_log_scale(features, model)).all()


def test_original_candidate_grid_is_preserved():
    specs = candidate_specs()
    assert len(specs) == 14 and len({s["name"] for s in specs}) == 14
    assert {s["target"] for s in specs} == {"median_ratio", "rmse_ratio"}
    assert sum(s["kind"] == "constant" for s in specs) == 2
    assert next(s for s in specs if s["name"] == "residual_median_ratio_l1_a0.5")["lam"] == 1.


def test_original_scene_fold_assignment():
    names = np.repeat([f"room_{i:02d}" for i in range(17)], 3)
    groups = np.unique(names)
    np.random.default_rng(20261031).shuffle(groups)
    expected = [np.flatnonzero(np.isin(names, chunk)) for chunk in np.array_split(groups, 5)]
    assert SEED == 20261031 and SAMPLE_SEED == 20261105
    for actual, wanted in zip(scene_folds(names), expected):
        np.testing.assert_array_equal(actual, wanted)


def test_sampling_is_per_identity_and_independent_of_iteration_order(monkeypatch):
    import depthlab.calibration as calibration
    pixels = np.arange(480 * 640, dtype=np.float32).reshape(480, 640)
    raw = 1 + pixels / 100000
    truth = 1.1 * raw
    monkeypatch.setattr(calibration, "nyu_depth", lambda handle, index: truth)
    a = prepare_training(None, {2: raw, 9: raw}, [2, 9])
    b = prepare_training(None, {2: raw, 9: raw}, [9, 2])
    np.testing.assert_array_equal(a[2][0], b[2][1])
    np.testing.assert_array_equal(a[3][0], b[3][1])
    crop = raw[45:471, 41:601].ravel()
    chosen = np.random.default_rng(20261105 + 2).choice(np.arange(len(crop)), 4096, replace=False)
    np.testing.assert_array_equal(a[2][0], crop[chosen])
    assert not np.array_equal(a[2][0], a[2][1])


def test_frozen_features_preserve_tiny_positive_depth():
    features = log_features(np.full((480, 640), 1e-9))
    np.testing.assert_allclose(features[:6], np.log(1e-9), rtol=0, atol=1e-13)
    np.testing.assert_allclose(features[6:], 0., rtol=0, atol=1e-13)


def test_frozen_deployment_clips_base_before_residual():
    frozen = validate_frozen(frozen_fixture())
    depth = np.full((480, 640), 20.)
    output = apply_frozen(depth, frozen)
    np.testing.assert_allclose(output, 10 * np.exp(-.15), rtol=0, atol=1e-14)
    predictions = frozen_predictions(depth, frozen)
    np.testing.assert_array_equal(predictions["residual_scale"], output)
    np.testing.assert_array_equal(predictions["monotone_curve"], 10.)
    np.testing.assert_array_equal(predictions["raw"], 10.)


def test_frozen_load_preserves_metadata_and_checks_digest(tmp_path):
    frozen = frozen_fixture()
    path = tmp_path / "frozen.json"
    payload = json.dumps(frozen).encode()
    path.write_bytes(payload)
    assert load_frozen(path, hashlib.sha256(payload).hexdigest()) == frozen
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen(path, "f" * 64)


@pytest.mark.parametrize("change", ["zero_std", "different_base", "duplicate_ids", "wrong_selection", "missing_digest"])
def test_frozen_rejects_inconsistent_artifacts(change):
    frozen = frozen_fixture()
    if change == "zero_std": frozen["residual_model"]["feature_std"][0] = 0.
    if change == "different_base": frozen["controls"]["spline_rmse"]["y_knots"][0] += .1
    if change == "duplicate_ids": frozen["train_ids"][0] = frozen["train_ids"][1]
    if change == "wrong_selection": frozen["candidate_spec"]["lam"] = 10.
    if change == "missing_digest": del frozen["selection_sha256"]
    with pytest.raises(ValueError): validate_frozen(frozen)


def test_tiny_cpu_curve_fit_has_finite_monotone_mapping():
    from depthlab.calibration import apply_curve
    p = np.geomspace(.3, 8, 512).reshape(4, 128).astype(np.float32)
    gt = .93 * p + .07
    curve = fit_curve(p, gt, .96)
    output = apply_curve(np.geomspace(.01, 100, 1000), curve)
    assert np.isfinite(output).all() and np.all(np.diff(output) >= 0)
    assert curve["objective_history"][-1] <= curve["objective_history"][0] + 1e-9
