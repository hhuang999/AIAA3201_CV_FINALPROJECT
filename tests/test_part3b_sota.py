"""Unit tests for Part 3B SOTA I/O, split isolation, and diagnostic helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depthlab.calibration import scene_folds
from depthlab.sota_diagnose import (
    CROP_R2,
    CROP_U_TILDE,
    CROP_V_TILDE,
    extract_intrinsics_features,
    extract_rgb_features,
    extract_spatial_3d_features,
    fit_inv_affine_oracle,
    fit_log_affine_oracle,
    fit_spatial_oracle,
    optimal_absrel_scalar,
    optimal_rmse_scalar,
    weighted_median,
)
from depthlab.sota_io import load_nyu_rgb_cache, scene_folds_seeded, write_csv_lf, write_json_lf, write_text_lf


def test_lf_writers_produce_no_crlf(tmp_path):
    txt_path = tmp_path / "sample.txt"
    json_path = tmp_path / "sample.json"
    csv_path = tmp_path / "sample.csv"
    write_text_lf(txt_path, "line1\r\nline2\n")
    write_json_lf(json_path, {"a": 1, "b": [2.0, 3.0]})
    write_csv_lf(csv_path, pd.DataFrame([{"x": 1, "y": 2.5}, {"x": 3, "y": 4.5}]))
    for path in (txt_path, json_path, csv_path):
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1, "b": [2.0, 3.0]}


def test_scene_folds_seeded_matches_historical_and_disjoint():
    names = np.array([f"scene_{i // 3}" for i in range(60)])
    f_ref = scene_folds(names, count=5)
    f_new = scene_folds_seeded(names, count=5, seed=20261031)
    for a, b in zip(f_ref, f_new):
        assert np.array_equal(a, b)
    # Scene disjointness across all folds for alternative seed
    f_alt = scene_folds_seeded(names, count=5, seed=42)
    seen = []
    for part in f_alt:
        seen.extend(part.tolist())
    assert sorted(seen) == list(range(60))
    for i in range(5):
        for j in range(i + 1, 5):
            assert not set(names[f_alt[i]]) & set(names[f_alt[j]])


def test_weighted_median_and_scalar_oracles_exact():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.5, 5.0, size=200)
    g = p * rng.lognormal(mean=0.08, sigma=0.15, size=200)

    s_ar = optimal_absrel_scalar(p, g)
    grid = np.linspace(s_ar * 0.85, s_ar * 1.15, 401)
    grid_vals = [np.mean(np.abs(s * p - g) / g) for s in grid]
    ar_opt = np.mean(np.abs(s_ar * p - g) / g)
    assert ar_opt <= min(grid_vals) + 1e-10

    s_rmse = optimal_rmse_scalar(p, g)
    rmse_opt = np.sqrt(np.mean((s_rmse * p - g) ** 2))
    rmse_grid = [np.sqrt(np.mean((s * p - g) ** 2)) for s in grid]
    assert rmse_opt <= min(rmse_grid) + 1e-10

    # Weighted median sanity on simple analytic case
    wm = weighted_median(np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 10.0]))
    assert wm == 4.0


def test_log_inv_and_spatial_oracles_reduce_synthetic_errors():
    u = CROP_U_TILDE.ravel()
    v = CROP_V_TILDE.ravel()
    r2 = CROP_R2.ravel()
    rng = np.random.default_rng(11)
    pred = rng.uniform(0.6, 6.0, size=u.size)

    # Exact log-affine recovery
    gt_log = np.clip(np.exp(0.15 + 1.12 * np.log(pred)), 0.001, 10.0)
    rec_log = fit_log_affine_oracle(pred, gt_log)
    assert np.max(np.abs(rec_log - gt_log)) < 1e-10

    # Exact inverse-affine recovery
    gt_inv = np.clip(1.0 / (0.85 / pred + 0.04), 0.001, 10.0)
    rec_inv = fit_inv_affine_oracle(pred, gt_inv)
    assert np.max(np.abs(rec_inv - gt_inv)) < 1e-10

    # Exact spatial linear tilt recovery
    gt_sp = np.clip(pred * np.exp(0.05 - 0.08 * u + 0.12 * v - 0.10 * r2), 0.001, 10.0)
    rec_sp, coef = fit_spatial_oracle(pred, gt_sp, u, v, r2, "spatial_linear")
    assert np.allclose(coef, [0.05, -0.08, 0.12, -0.10], atol=1e-6)
    assert np.max(np.abs(rec_sp - gt_sp)) < 1e-6


def test_feature_extractors_finite_and_independent_of_true_k():
    K_a = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_b = np.array([[620.0, 0.0, 310.0], [0.0, 615.0, 250.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    fa = extract_intrinsics_features(K_a)
    fb = extract_intrinsics_features(K_b)
    assert all(np.isfinite(list(fa.values())))
    assert fa["log_fx_w"] < fb["log_fx_w"]
    assert fa["fov_x"] > fb["fov_x"]

    rng = np.random.default_rng(19)
    depth = rng.uniform(0.8, 5.0, size=(480, 640))
    rgb = rng.integers(20, 235, size=(480, 640, 3), dtype=np.uint8)
    s_feats = extract_spatial_3d_features(depth, K_a)
    r_feats = extract_rgb_features(rgb, depth)
    assert len(s_feats) == 27
    assert len(r_feats) == 16
    assert all(np.isfinite(list(s_feats.values())))
    assert all(np.isfinite(list(r_feats.values())))


def test_verified_cache_train_guard_if_data_present():
    repo = Path(__file__).resolve().parents[1]
    pred_dir = repo / "outputs" / "nyu_rgb"
    data_dir = repo / "data"
    if not (pred_dir / "depth.npy").is_file() or not (data_dir / "splits.mat").is_file():
        pytest.skip("Local NYU cache/data not present")
    cache = load_nyu_rgb_cache(pred_dir, data_dir, allowed_split="train", compute_hashes=False, verify_scenes=False)
    assert len(cache.train_ids) == 795
    assert len(cache.test_ids) == 654
    _ = cache[int(cache.train_ids[0])]
    with pytest.raises(PermissionError):
        _ = cache[int(cache.test_ids[0])]
    with pytest.raises(PermissionError):
        _ = cache.intrinsics(int(cache.test_ids[0]))
