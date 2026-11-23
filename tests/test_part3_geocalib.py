"""Unit and artifact integrity tests for standalone Part 3 GeoCalib-3D."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depthlab.geocalib_diagnose import (
    fit_inv_affine_oracle,
    fit_log_affine_oracle,
    fit_spatial_oracle,
    optimal_absrel_scalar,
    optimal_rmse_scalar,
    roc_auc_mann_whitney,
)
from depthlab.geocalib_evaluate import compute_change_counts, paired_scene_bootstrap
from depthlab.geocalib_io import file_sha256, scene_folds_seeded, write_csv_lf, write_json_lf
from depthlab.geocalib_model import (
    DEPTH_KNOTS,
    LOCKED_EVAL_ORDER,
    PRIMARY_GEO_METHOD,
    PRIMARY_MULTIPASS_METHOD,
    PRIMARY_SINGLE_PASS_METHOD,
    apply_geocalib_model,
    apply_spline,
    build_candidate_specs,
    count_model_parameters,
    validate_frozen_geocalib,
)


ROOT = Path(__file__).resolve().parents[1]


def test_scene_folds_seeded_deterministic_and_disjoint():
    scenes = np.repeat([f"kitchen_{i:02d}" for i in range(25)], 4)
    f1 = scene_folds_seeded(scenes, count=5, seed=20261031)
    f2 = scene_folds_seeded(scenes, count=5, seed=20261031)
    assert len(f1) == 5
    for a, b in zip(f1, f2):
        np.testing.assert_array_equal(a, b)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not (set(scenes[f1[i]]) & set(scenes[f1[j]]))
    np.testing.assert_array_equal(np.sort(np.concatenate(f1)), np.arange(len(scenes)))


def test_lf_writers_emit_zero_crlf(tmp_path):
    jpath = tmp_path / "sample.json"
    cpath = tmp_path / "sample.csv"
    write_json_lf(jpath, {"ok": True, "vals": [1, 2]})
    write_csv_lf(cpath, pd.DataFrame([{"a": 1, "b": "x"}]))
    assert b"\r" not in jpath.read_bytes()
    assert b"\r" not in cpath.read_bytes()
    assert len(file_sha256(jpath)) == 64


def test_diagnosis_oracles_and_roc_auc():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.5, 5.0, size=512)
    gt = 1.25 * p
    assert abs(optimal_absrel_scalar(p, gt) - 1.25) < 1e-6
    assert abs(optimal_rmse_scalar(p, gt) - 1.25) < 1e-6

    gt_aff = np.exp(0.15 + 1.08 * np.log(p))
    np.testing.assert_allclose(fit_log_affine_oracle(p, gt_aff), np.clip(gt_aff, 0.001, 10.0), rtol=1e-5)

    inv_gt = 1.0 / np.maximum(0.9 / p + 0.05, 1e-4)
    np.testing.assert_allclose(fit_inv_affine_oracle(p, inv_gt), np.clip(inv_gt, 0.001, 10.0), rtol=1e-5)

    u = rng.uniform(-1.0, 1.0, size=512)
    v = rng.uniform(-1.0, 1.0, size=512)
    r2 = u * u + v * v
    gt_sp = p * np.exp(0.04 * v - 0.03 * r2)
    corr, _ = fit_spatial_oracle(p, gt_sp, u, v, r2, "spatial_linear")
    np.testing.assert_allclose(corr, np.clip(gt_sp, 0.001, 10.0), rtol=1e-5)

    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    assert roc_auc_mann_whitney(scores, labels) == 1.0
    assert roc_auc_mann_whitney(scores[::-1], labels) == 0.0


def test_candidate_specs_unique_and_non_legacy():
    specs = build_candidate_specs()
    assert len(specs) == 26
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert PRIMARY_SINGLE_PASS_METHOD in names
    assert PRIMARY_MULTIPASS_METHOD in names


def test_apply_geocalib_model_scalar_and_spline():
    depth = np.full((480, 640), 2.0, dtype=np.float64)
    k = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out_s = apply_geocalib_model(depth, k, {"kind": "scalar", "scale": 0.9})
    np.testing.assert_allclose(out_s, 1.8)

    spline_m = {"kind": "monotone_spline", "x_knots": DEPTH_KNOTS.tolist(), "y_knots": (DEPTH_KNOTS + np.log(0.9)).tolist()}
    out_sp = apply_geocalib_model(depth, k, spline_m)
    np.testing.assert_allclose(out_sp, apply_spline(depth, spline_m), rtol=1e-6)


def test_paired_scene_bootstrap_and_change_counts():
    rows = []
    for s in range(10):
        for k in range(3):
            idx = s * 3 + k
            sc = f"scene_{s}"
            rows.append({"index": idx, "scene": sc, "method": "ref", "abs_rel": 0.12, "rmse": 0.40, "delta1": 0.88})
            rows.append({"index": idx, "scene": sc, "method": "cand", "abs_rel": 0.08, "rmse": 0.34, "delta1": 0.94})
    df = pd.DataFrame(rows)
    pb = paired_scene_bootstrap(df, [("ref", "cand")], replicates=200)
    assert len(pb) == 3
    assert bool((pb["ci95_low"] > 0.0).all())
    cc = compute_change_counts(df, [("ref", "cand")])
    assert int(cc.loc[0, "all_better"]) == 30
    assert int(cc.loc[0, "all_worse"]) == 0


def test_validate_frozen_geocalib_rejects_test_leakage():
    payload = {
        "schema": "GeoCalib_3D",
        "complete": True,
        "test_read": True,
        "train_ids": list(range(795)),
        "training_scenes": 249,
        "dataset_sha256": "0" * 64,
        "split_sha256": "1" * 64,
        "primary_single_pass_method": PRIMARY_SINGLE_PASS_METHOD,
        "primary_multipass_method": PRIMARY_MULTIPASS_METHOD,
        "locked_eval_order": ["raw"],
        "models": {"raw": {"kind": "scalar", "scale": 1.0}},
    }
    with pytest.raises(ValueError, match="test_read"):
        validate_frozen_geocalib(payload)


def test_delivered_geocalib_artifacts_and_parity_contracts():
    p3_dir = ROOT / "results" / "part3_geocalib"
    frozen_path = p3_dir / "frozen_geocalib_model.json"
    if not frozen_path.is_file():
        pytest.skip("Delivered results not present")
    payload = validate_frozen_geocalib(json.loads(frozen_path.read_text(encoding="utf-8")))
    assert payload["test_read"] is False
    assert payload["training_images"] == 795
    assert payload["training_scenes"] == 249
    assert tuple(payload["locked_eval_order"]) == LOCKED_EVAL_ORDER

    sp_raw, sp_dof = count_model_parameters(payload["models"][PRIMARY_SINGLE_PASS_METHOD])
    assert sp_raw == payload["primary_single_pass_raw_params"] == 5018
    assert abs(sp_dof - payload["primary_single_pass_effective_dof"]) < 1e-6

    test_df = pd.read_csv(p3_dir / "test_summary.csv").set_index("method")
    assert list(test_df.index) == list(LOCKED_EVAL_ORDER)
    assert abs(float(test_df.loc[PRIMARY_GEO_METHOD, "abs_rel"]) - 0.086535) < 1e-5
    assert abs(float(test_df.loc[PRIMARY_SINGLE_PASS_METHOD, "abs_rel"]) - 0.083024) < 1e-5
    assert abs(float(test_df.loc[PRIMARY_SINGLE_PASS_METHOD, "rmse"]) - 0.340792) < 1e-5
    assert abs(float(test_df.loc[PRIMARY_SINGLE_PASS_METHOD, "delta1"]) - 0.938413) < 1e-5
    assert abs(float(test_df.loc[PRIMARY_MULTIPASS_METHOD, "abs_rel"]) - 0.083189) < 1e-5


def test_plot_camera_diagnostic_regenerates_png(tmp_path):
    from depthlab.geocalib_figures import plot_camera_diagnostic

    p2_dir = ROOT / "results" / "part2_depth_video"
    per_csv = p2_dir / "camera_diagnostic_per_image.csv"
    rep_json = p2_dir / "camera_diagnostic.json"
    if not (per_csv.is_file() and rep_json.is_file()):
        pytest.skip("Delivered Part 2 camera diagnostic files not present")
    out_png = tmp_path / "camera_diagnostic.png"
    plot_camera_diagnostic(out_png, per_csv, rep_json)
    assert out_png.is_file()
    assert out_png.stat().st_size > 10_000
