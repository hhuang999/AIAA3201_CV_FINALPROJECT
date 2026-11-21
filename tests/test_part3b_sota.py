"""Unit tests for Part 3B SOTA I/O, split isolation, and diagnostic helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from depthlab.calibration import scene_folds
from depthlab.metrics import CROP
from depthlab.sota_calibration import (
    GeoCalibSpec,
    _CROP_D2_CENTER,
    _CROP_R2,
    _CROP_V_TILDE,
    _JointGeoCalibModule,
    _MEAN_R2_CROP,
    _centered_log_cos_hat_crop,
    _cond2d_feature_names,
    _feature_matrix,
    _scale_feature_names,
    _slope_feature_names,
    _standardize_apply,
    _standardize_fit,
    apply_sota_model,
    count_parameters,
    extract_sota_features,
    fit_joint_sota_model,
    load_frozen_sota,
    validate_frozen_sota,
)
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


def _make_synthetic_batch(n_imgs=12, samples_per_img=128):
    rng = np.random.default_rng(123)
    depths = []
    intrinsics = []
    samples_p = np.empty((n_imgs, samples_per_img), dtype=np.float64)
    samples_gt = np.empty((n_imgs, samples_per_img), dtype=np.float64)
    samples_row = np.empty((n_imgs, samples_per_img), dtype=np.int32)
    samples_col = np.empty((n_imgs, samples_per_img), dtype=np.int32)
    feats = []
    ratios = np.empty(n_imgs, dtype=np.float64)

    for i in range(n_imgs):
        f = rng.uniform(350.0, 620.0)
        K = np.array([[f, 0.0, 321.0], [0.0, f * 1.01, 244.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        d = rng.uniform(0.6, 5.5, size=(480, 640))
        true_scale = 1.05 * (550.0 / f) ** 0.3
        gt = np.clip(d * true_scale, 0.001, 10.0)
        rr = rng.integers(0, 426, size=samples_per_img)
        cc = rng.integers(0, 560, size=samples_per_img)
        samples_row[i] = rr
        samples_col[i] = cc
        samples_p[i] = d[CROP][rr, cc]
        samples_gt[i] = gt[CROP][rr, cc]
        depths.append(d)
        intrinsics.append(K)
        feats.append(extract_sota_features(d, K, None, need_rgb=False))
        ratios[i] = true_scale

    return depths, np.stack(intrinsics), samples_p, samples_gt, samples_row, samples_col, feats, ratios


def test_sota_model_torch_numpy_parity():
    import torch

    depths, intr, sp, sgt, srow, scol, feats, ratios = _make_synthetic_batch()
    spec = GeoCalibSpec(
        name="parity_test",
        use_focal_curve=True,
        use_geo_cxcy=True,
        use_spatial3d=True,
        use_depth_slope=True,
        use_ray_unwarp=True,
        use_cond_spatial2d=True,
        loss_mode="ar_rmse_d1_1512",
        reg_l2=0.01,
    )
    model = fit_joint_sota_model(spec, sp, sgt, srow, scol, feats, intr, ratios, max_iter=30)
    assert count_parameters(model) == 37
    # Focal spline must be active (not frozen near zero) on focal-correlated synthetic data
    assert min(model["focal_spline"]["slopes"]) < -0.15

    # Verify torch fast path vs pure NumPy path across all pixels
    out_np = apply_sota_model(depths[0], intr[0], model, use_torch_fast=False)
    out_torch = apply_sota_model(depths[0], intr[0], model, use_torch_fast=True)
    assert out_np.shape == (480, 640)
    assert np.max(np.abs(out_np - out_torch)) < 1e-11

    # Verify against _JointGeoCalibModule forward pass on sampled crop pixels
    mod = _JointGeoCalibModule(
        spec,
        model["init_scale"],
        model["ray_unwarp"]["f_ref_tilde"],
        len(_scale_feature_names(spec)),
        len(_slope_feature_names(spec)),
        len(_cond2d_feature_names(spec)),
    )
    # Reconstruct exact module state by running forward check with exported dict
    # Check directly that focal_spline anchor at index 3 == 0.0 and slopes are strictly valid
    assert abs(model["focal_spline"]["y_knots"][3]) < 1e-12
    assert all(s <= 0.0 for s in model["focal_spline"]["slopes"])
    assert all(0.25 - 1e-9 <= s <= 2.0 + 1e-9 for s in model["depth_spline"]["slopes"])

    # Verify module parity via explicit parameter loading (module-level with torch.no_grad)
    with torch.no_grad():
        ds = model["depth_spline"]
        offset = ds["y_knots"][0] - ds["x_knots"][0]
        mod.raw_offset.copy_(torch.tensor(np.arctanh(offset / np.log(2.0)), dtype=torch.float64))
        sl_frac = (np.asarray(ds["slopes"]) - 0.25) / 1.75
        mod.raw_depth_slopes.copy_(torch.tensor(np.log(sl_frac / (1.0 - sl_frac)), dtype=torch.float64))
        fs = model["focal_spline"]
        f_frac = np.clip(-np.asarray(fs["slopes"]) / 2.5, 1e-12, 1.0 - 1e-12)
        mod.raw_focal_slopes.copy_(torch.tensor(np.log(f_frac / (1.0 - f_frac)), dtype=torch.float64))
        mod.w_scale.copy_(torch.tensor(model["scale_residual"]["weights"], dtype=torch.float64))
        mod.w_slope.copy_(torch.tensor(model["depth_slope_mod"]["weights"], dtype=torch.float64))
        ru = model["ray_unwarp"]
        mod.raw_gamma_ray.copy_(torch.tensor(np.arctanh(ru["gamma_ray"] / 1.5), dtype=torch.float64))
        fr_frac = (ru["f_ref_tilde"] - 0.8) / 0.6
        mod.raw_f_ref.copy_(torch.tensor(np.log(fr_frac / (1.0 - fr_frac)), dtype=torch.float64))
        c2 = model["cond_spatial2d"]
        mod.a_v0.copy_(torch.tensor(c2["a_v0"], dtype=torch.float64))
        mod.a_r0.copy_(torch.tensor(c2["a_r0"], dtype=torch.float64))
        mod.w_v.copy_(torch.tensor(c2["w_v"], dtype=torch.float64))
        mod.w_r.copy_(torch.tensor(c2["w_r"], dtype=torch.float64))

        z_sc = _standardize_apply(
            _feature_matrix([feats[0]], _scale_feature_names(spec)),
            np.asarray(model["scale_residual"]["mean"]),
            np.asarray(model["scale_residual"]["std"]),
        )
        z_sl = _standardize_apply(
            _feature_matrix([feats[0]], _slope_feature_names(spec)),
            np.asarray(model["depth_slope_mod"]["mean"]),
            np.asarray(model["depth_slope_mod"]["std"]),
        )
        z_c2 = _standardize_apply(
            _feature_matrix([feats[0]], _cond2d_feature_names(spec)),
            np.asarray(c2["mean"]),
            np.asarray(c2["std"]),
        )
        lch = _centered_log_cos_hat_crop(intr[0])[srow[0], scol[0]][None, :]
        f_ref_v = ru["f_ref_tilde"]
        log_cos_ref_full = -0.5 * np.log(1.0 + _CROP_D2_CENTER / ((480.0 * f_ref_v) ** 2))
        pred_t = mod(
            torch.tensor(np.log(sp[0:1]), dtype=torch.float64),
            torch.tensor([feats[0]["f_tilde"]], dtype=torch.float64),
            torch.tensor(z_sc, dtype=torch.float64),
            torch.tensor(z_sl, dtype=torch.float64),
            torch.tensor(z_c2, dtype=torch.float64),
            torch.tensor(_CROP_V_TILDE[srow[0], scol[0]][None, :], dtype=torch.float64),
            torch.tensor((_CROP_R2[srow[0], scol[0]] - _MEAN_R2_CROP)[None, :], dtype=torch.float64),
            torch.tensor(_CROP_D2_CENTER[srow[0], scol[0]][None, :], dtype=torch.float64),
            torch.tensor(lch, dtype=torch.float64),
        ).numpy()[0]
        sample_only_lcr = log_cos_ref_full[srow[0], scol[0]]
        mean_diff = float(sample_only_lcr.mean() - log_cos_ref_full.mean())
        pred_t_exact = np.clip(pred_t * np.exp(ru["gamma_ray"] * mean_diff), 0.001, 10.0)
    assert np.max(np.abs(pred_t_exact - out_np[CROP][srow[0], scol[0]])) < 1e-11


def test_sota_model_strict_depth_monotonicity():
    depths, intr, sp, sgt, srow, scol, feats, ratios = _make_synthetic_batch()
    spec = GeoCalibSpec(
        name="mono_test",
        use_focal_curve=True,
        use_geo_cxcy=True,
        use_spatial3d=True,
        use_depth_slope=True,
        use_ray_unwarp=True,
        use_cond_spatial2d=True,
    )
    model = fit_joint_sota_model(spec, sp, sgt, srow, scol, feats, intr, ratios, max_iter=20)
    # Push depth_slope weights to extreme magnitude to force tanh -> -1 (delta_b -> -0.20)
    model["depth_slope_mod"]["weights"] = [100.0] * len(model["depth_slope_mod"]["weights"])

    z_grid = np.linspace(0.05, 15.0, 200)
    d_low = np.full((480, 640), 1.0)
    d_high = np.full((480, 640), 1.0)
    fixed_feat = extract_sota_features(depths[0], intr[0], need_rgb=False)
    prev = np.zeros((480, 640))
    for zv in z_grid:
        d_cur = np.full((480, 640), zv)
        cur = apply_sota_model(d_cur, intr[0], model, cached_features=fixed_feat)
        assert np.all(cur >= prev - 1e-12)
        prev = cur
    # Also check strict ordering between two distinct levels
    d_low[200, 300] = 1.2
    d_high[200, 300] = 2.5
    out_low = apply_sota_model(d_low, intr[0], model, cached_features=fixed_feat)
    out_high = apply_sota_model(d_high, intr[0], model, cached_features=fixed_feat)
    assert out_high[200, 300] > out_low[200, 300]


def test_sota_model_bounded_extrapolation():
    depths, intr, sp, sgt, srow, scol, feats, ratios = _make_synthetic_batch()
    spec = GeoCalibSpec(
        name="bounds_test",
        use_focal_curve=True,
        use_geo_cxcy=True,
        use_spatial3d=True,
        use_depth_slope=True,
        use_ray_unwarp=True,
        use_cond_spatial2d=True,
    )
    model = fit_joint_sota_model(spec, sp, sgt, srow, scol, feats, intr, ratios, max_iter=20)
    d_ext = np.full((480, 640), 1e-4)
    d_ext[0:240, :] = 100.0
    for f_val in (50.0, 5000.0):
        K_ext = np.array([[f_val, 0.0, 320.0], [0.0, f_val, 240.0], [0.0, 0.0, 1.0]])
        out = apply_sota_model(d_ext, K_ext, model)
        assert np.isfinite(out).all()
        assert np.all((out >= 0.001) & (out <= 10.0))


def test_sota_inference_latency_sanity():
    from time import perf_counter

    depths, intr, sp, sgt, srow, scol, feats, ratios = _make_synthetic_batch(n_imgs=6)
    spec = GeoCalibSpec(name="lat_test", use_focal_curve=True, use_geo_cxcy=True, use_depth_slope=True)
    model = fit_joint_sota_model(spec, sp, sgt, srow, scol, feats, intr, ratios, max_iter=15)
    _ = apply_sota_model(depths[0], intr[0], model)
    t0 = perf_counter()
    for _ in range(10):
        _ = apply_sota_model(depths[0], intr[0], model)
    ms = (perf_counter() - t0) * 1000.0 / 10.0
    assert np.isfinite(ms) and ms < 200.0


def test_sota_outputs_have_lf_line_endings():
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "results" / "part3b_sota"
    if not out_dir.exists():
        pytest.skip("Gate 2 outputs not yet generated")
    for p in list(out_dir.glob("*.csv")) + list(out_dir.glob("*.json")):
        assert b"\r\n" not in p.read_bytes(), f"CRLF detected in {p}"


def test_frozen_sota_load_and_tamper_rejection(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    frozen_path = repo / "results" / "part3b_sota" / "frozen_sota_model.json"
    cv_path = repo / "results" / "part3b_sota" / "cv_selection_report.json"
    if not frozen_path.is_file() or not cv_path.is_file():
        pytest.skip("Gate 3 frozen_sota_model.json not yet generated")

    frozen = load_frozen_sota(frozen_path)
    assert frozen["test_read"] is False
    assert frozen["primary_params"] == 25
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    inspected_yk = np.asarray(cv["winner_full_train_inspection"]["depth_spline"]["y_knots"], dtype=np.float64)
    locked_yk = np.asarray(frozen["primary_model"]["depth_spline"]["y_knots"], dtype=np.float64)
    assert np.max(np.abs(locked_yk - inspected_yk)) < 1e-12

    # Wrong expected SHA-256 must fail
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_sota(frozen_path, expected_sha256="0" * 64)

    # Tampered test_read=True must fail
    tampered_tr = json.loads(json.dumps(frozen))
    tampered_tr["test_read"] = True
    with pytest.raises(ValueError, match="test_read=False"):
        validate_frozen_sota(tampered_tr)

    # Tampered NaN weight must fail (write as raw string since allow_nan=False rejects NaN JSON normally)
    tampered_nan = json.loads(json.dumps(frozen))
    tampered_nan["primary_model"]["scale_residual"]["weights"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_frozen_sota(tampered_nan)

    # Tampered negative depth slope must fail
    tampered_slope = json.loads(json.dumps(frozen))
    yk = list(tampered_slope["primary_model"]["depth_spline"]["y_knots"])
    yk[2] = yk[1] - 0.1
    tampered_slope["primary_model"]["depth_spline"]["y_knots"] = yk
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_frozen_sota(tampered_slope)

    # Round-trip valid file via tmp_path rejects wrong hash
    copy_p = tmp_path / "frozen.json"
    copy_p.write_bytes(frozen_path.read_bytes())
    assert load_frozen_sota(copy_p)["primary_params"] == 25
