"""Training-only residual and oracle error diagnosis for Part 3B SOTA calibration."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .calibration import (
    apply_curve,
    apply_residual,
    fit_curve,
    fit_residual,
    fit_scale,
    prepare_training,
    residual_targets,
)
from .data import nyu_depth, nyu_image, open_nyu, scene_names
from .frozen import load_frozen
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, clip_depth, evaluate_image, metrics, valid_depth
from .sota_io import load_nyu_rgb_cache, scene_folds_seeded, write_csv_lf, write_json_lf


# CROP = (slice(45, 471), slice(41, 601)) -> shape (426, 560) inside 480x640 image
_FULL_U, _FULL_V = np.meshgrid(
    np.arange(640, dtype=np.float64),
    np.arange(480, dtype=np.float64),
    indexing="xy",
)
CROP_U_PIX = _FULL_U[CROP]
CROP_V_PIX = _FULL_V[CROP]
CROP_U_TILDE = (CROP_U_PIX - 319.5) / 320.0
CROP_V_TILDE = (CROP_V_PIX - 239.5) / 240.0
CROP_R2 = CROP_U_TILDE ** 2 + CROP_V_TILDE ** 2


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute exact weighted median minimizing sum_i w_i * |s - v_i|."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0 or v.shape != w.shape:
        raise ValueError("Expected nonempty aligned value/weight vectors")
    if not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("Expected finite nonnegative weights with positive sum")
    order = np.argsort(v, kind="mergesort")
    v_s = v[order]
    w_s = w[order]
    cum_w = np.cumsum(w_s)
    half = 0.5 * cum_w[-1]
    idx = int(np.searchsorted(cum_w, half, side="left"))
    return float(v_s[min(idx, len(v_s) - 1)])


def optimal_absrel_scalar(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute exact scalar s > 0 minimizing sum |s * p - g| / g."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    g = np.asarray(gt, dtype=np.float64).ravel()
    ratios = g / p
    weights = p / g
    return float(np.clip(weighted_median(ratios, weights), 1e-4, 100.0))


def optimal_rmse_scalar(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute exact scalar s > 0 minimizing sum (s * p - g)^2."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    g = np.asarray(gt, dtype=np.float64).ravel()
    denom = float(np.dot(p, p))
    if denom <= 0:
        return 1.0
    return float(np.clip(np.dot(p, g) / denom, 1e-4, 100.0))


def fit_log_affine_oracle(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image 2-param log-affine exp(a + b * log(pred))."""
    lp = np.log(pred)
    lg = np.log(gt)
    design = np.column_stack([np.ones_like(lp), lp])
    coef, _, _, _ = np.linalg.lstsq(design, lg, rcond=None)
    return clip_depth(np.exp(np.clip(design @ coef, np.log(0.001), np.log(10.0))))


def fit_inv_affine_oracle(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image 2-param inverse-depth affine 1 / max(a / pred + b, 1e-4)."""
    inv_p = 1.0 / pred
    inv_g = 1.0 / gt
    design = np.column_stack([inv_p, np.ones_like(inv_p)])
    coef, _, _, _ = np.linalg.lstsq(design, inv_g, rcond=None)
    inv_hat = np.maximum(design @ coef, 1e-4)
    return clip_depth(1.0 / inv_hat)


def fit_spatial_oracle(
    pred: np.ndarray,
    gt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    r2: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-image spatial/depth multiplicative field in log-residual space."""
    log_res = np.log(gt / pred)
    if mode == "spatial_linear":
        design = np.column_stack([np.ones_like(u), u, v, r2])
    elif mode == "spatial_quadric":
        design = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    elif mode == "spatial_plus_depth":
        design = np.column_stack([np.ones_like(u), np.log(pred), u, v, r2])
    else:
        raise ValueError(f"Unknown spatial oracle mode: {mode}")
    coef, _, _, _ = np.linalg.lstsq(design, log_res, rcond=None)
    corrected = clip_depth(pred * np.exp(np.clip(design @ coef, -2.0, 2.0)))
    return corrected, coef


def extract_intrinsics_features(K: np.ndarray) -> dict[str, float]:
    """Extract strictly RGB-only self-predicted camera geometry features from K_hat."""
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Expected finite 3x3 predicted intrinsics matrix")
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Predicted focal lengths must be positive")
    cos_theta = 1.0 / np.sqrt(
        1.0 + ((CROP_U_PIX - cx) / fx) ** 2 + ((CROP_V_PIX - cy) / fy) ** 2
    )
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "log_fx_w": float(np.log(fx / 640.0)),
        "log_fy_h": float(np.log(fy / 480.0)),
        "log_f_geom": float(0.5 * np.log((fx * fy) / (640.0 * 480.0))),
        "log_aspect": float(np.log(fx / fy)),
        "cx_off": float((cx - 320.0) / 640.0),
        "cy_off": float((cy - 240.0) / 480.0),
        "fov_x": float(2.0 * np.arctan(320.0 / fx)),
        "fov_y": float(2.0 * np.arctan(240.0 / fy)),
        "mean_ray_cos": float(cos_theta.mean()),
        "std_ray_cos": float(cos_theta.std()),
    }


def extract_spatial_3d_features(raw_depth: np.ndarray, K: np.ndarray) -> dict[str, float]:
    """Extract B3 8 features, spatial depth gradients, and pseudo-3D geometry from (Z_hat, K_hat)."""
    p = np.asarray(raw_depth, dtype=np.float64)
    if p.shape != (480, 640) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Expected finite positive 480x640 depth map")
    crop_p = p[CROP]
    lp = np.log(crop_p)
    q10, q25, q50, q75, q90 = np.quantile(lp, [0.1, 0.25, 0.5, 0.75, 0.9])
    lp_mean = float(lp.mean())
    lp_std = float(lp.std())
    grad_v = float(np.mean(np.abs(np.diff(lp, axis=0))))
    grad_u = float(np.mean(np.abs(np.diff(lp, axis=1))))
    grad = 0.5 * (grad_v + grad_u)

    # Spatial OLS slopes of log(Z_hat) wrt (u_tilde, v_tilde, r^2)
    u_c = CROP_U_TILDE - CROP_U_TILDE.mean()
    v_c = CROP_V_TILDE - CROP_V_TILDE.mean()
    r2_c = CROP_R2 - CROP_R2.mean()
    lp_c = lp - lp_mean
    dz_du = float(np.mean(u_c * lp_c) / np.mean(u_c * u_c))
    dz_dv = float(np.mean(v_c * lp_c) / np.mean(v_c * v_c))
    dz_dr2 = float(np.mean(r2_c * lp_c) / np.mean(r2_c * r2_c))

    h_c = lp.shape[0]
    top_slice = lp[: h_c // 3]
    bot_slice = lp[(2 * h_c) // 3 :]
    center_mask = CROP_R2 < 0.35
    border_mask = CROP_R2 > 0.75

    skew_logz = float(np.mean(((lp - lp_mean) / max(lp_std, 1e-8)) ** 3))

    # Pseudo-3D back-projection using ONLY self-predicted K_hat
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x3d = crop_p * ((CROP_U_PIX - cx) / fx)
    y3d = crop_p * ((CROP_V_PIX - cy) / fy)
    span_x = float(max(np.quantile(x3d, 0.95) - np.quantile(x3d, 0.05), 1e-6))
    span_y = float(max(np.quantile(y3d, 0.95) - np.quantile(y3d, 0.05), 1e-6))
    span_z = float(max(np.quantile(crop_p, 0.95) - np.quantile(crop_p, 0.05), 1e-6))
    inv_z = 1.0 / crop_p

    return {
        "b3_q10": float(q10),
        "b3_q25": float(q25),
        "b3_q50": float(q50),
        "b3_q75": float(q75),
        "b3_q90": float(q90),
        "b3_mean": lp_mean,
        "b3_std": lp_std,
        "b3_grad": float(grad),
        "dz_du": dz_du,
        "dz_dv": dz_dv,
        "dz_dr2": dz_dr2,
        "top_minus_bot_logz": float(top_slice.mean() - bot_slice.mean()),
        "center_minus_border_logz": float(lp[center_mask].mean() - lp[border_mask].mean()),
        "iqr_logz": float(q75 - q25),
        "skew_logz": skew_logz,
        "frac_near_1p2m": float(np.mean(crop_p < 1.2)),
        "frac_far_4m": float(np.mean(crop_p > 4.0)),
        "grad_u": grad_u,
        "grad_v": grad_v,
        "grad_aniso_uv": float((grad_v - grad_u) / (grad_v + grad_u + 1e-8)),
        "log_span_x": float(np.log(span_x)),
        "log_span_y": float(np.log(span_y)),
        "log_span_z": float(np.log(span_z)),
        "span_y_over_z": float(span_y / span_z),
        "span_x_over_z": float(span_x / span_z),
        "mean_inv_z": float(inv_z.mean()),
        "std_inv_z": float(inv_z.std()),
    }


def extract_rgb_features(rgb_uint8: np.ndarray, raw_depth: np.ndarray) -> dict[str, float]:
    """Extract strictly legal RGB appearance, illumination, and texture cues on CROP."""
    rgb = np.asarray(rgb_uint8[CROP], dtype=np.float64) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Unexpected cropped RGB shape: {rgb.shape}")
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_p10, lum_p50, lum_p90 = np.quantile(lum, [0.1, 0.5, 0.9])
    h_c = lum.shape[0]
    lum_top_minus_bot = float(lum[: h_c // 3].mean() - lum[(2 * h_c) // 3 :].mean())

    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-6)

    dv = np.abs(np.diff(lum, axis=0))
    du = np.abs(np.diff(lum, axis=1))
    rgb_grad_v = float(dv.mean())
    rgb_grad_u = float(du.mean())
    # Interior gradient magnitude aligned on (H-1, W-1)
    grad_mag = np.sqrt(dv[:, :-1] ** 2 + du[:-1, :] ** 2)

    lp = np.log(np.asarray(raw_depth, dtype=np.float64)[CROP])
    lp_dv = np.abs(np.diff(lp, axis=0))[:, :-1]
    lp_du = np.abs(np.diff(lp, axis=1))[:-1, :]
    lp_grad_mag = np.sqrt(lp_dv ** 2 + lp_du ** 2)
    # Correlation between RGB edge magnitude and predicted log-depth edge magnitude
    a_c = grad_mag.ravel() - grad_mag.mean()
    b_c = lp_grad_mag.ravel() - lp_grad_mag.mean()
    denom = float(np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c)))
    rgb_depth_grad_corr = float(np.dot(a_c, b_c) / denom) if denom > 1e-12 else 0.0

    return {
        "lum_p10": float(lum_p10),
        "lum_p50": float(lum_p50),
        "lum_p90": float(lum_p90),
        "lum_mean": float(lum.mean()),
        "lum_std": float(lum.std()),
        "lum_top_minus_bot": lum_top_minus_bot,
        "frac_overexp": float(np.mean((lum > 0.92) | (max_c >= 250.0 / 255.0))),
        "frac_underexp": float(np.mean(lum < 0.08)),
        "sat_mean": float(sat.mean()),
        "sat_std": float(sat.std()),
        "r_minus_b_mean": float((r - b).mean()),
        "rgb_grad_mean": float(grad_mag.mean()),
        "rgb_grad_std": float(grad_mag.std()),
        "rgb_grad_aniso_uv": float((rgb_grad_v - rgb_grad_u) / (rgb_grad_v + rgb_grad_u + 1e-8)),
        "frac_low_texture": float(np.mean(grad_mag < 0.02)),
        "rgb_depth_grad_corr": rgb_depth_grad_corr,
    }


def scene_cv_ridge_r2(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    lam: float = 1.0,
) -> float:
    """Compute 5-fold scene-grouped out-of-fold R^2 with standardized linear ridge."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    y_arr = np.asarray(y, dtype=np.float64)
    oof_pred = np.empty_like(y_arr)
    for val in folds:
        tr = np.setdiff1d(np.arange(len(y_arr)), val)
        mu = x_arr[tr].mean(axis=0)
        sd = np.maximum(x_arr[tr].std(axis=0), 1e-8)
        z_tr = (x_arr[tr] - mu) / sd
        z_val = (x_arr[val] - mu) / sd
        y0 = y_arr[tr].mean()
        coef = np.linalg.solve(
            z_tr.T @ z_tr / len(tr) + lam * np.eye(z_tr.shape[1]),
            z_tr.T @ (y_arr[tr] - y0) / len(tr),
        )
        oof_pred[val] = y0 + z_val @ coef
    ss_res = float(np.sum((y_arr - oof_pred) ** 2))
    # Use fold-mean baseline for fair out-of-fold R^2
    oof_null = np.empty_like(y_arr)
    for val in folds:
        tr = np.setdiff1d(np.arange(len(y_arr)), val)
        oof_null[val] = y_arr[tr].mean()
    ss_tot = float(np.sum((y_arr - oof_null) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _spatial_basis(u: np.ndarray, v: np.ndarray, r2: np.ndarray, log_z: np.ndarray | None = None) -> np.ndarray:
    cols = [np.ones_like(u), u, v, u * u, u * v, v * v, r2]
    if log_z is not None:
        lz = log_z - log_z.mean()
        cols.extend([u * lz, v * lz, r2 * lz])
    return np.column_stack(cols)


def run_diagnosis(
    data_root: Path,
    prediction_dir: Path,
    frozen_model_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute Gate 0 baseline verification and Gate 1 training-set residual diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_nyu_rgb_cache(prediction_dir, data_root, allowed_split="train", compute_hashes=False)
    frozen = load_frozen(frozen_model_path)
    train_ids = cache.train_ids
    if train_ids.tolist() != frozen["train_ids"]:
        raise ValueError("Official train IDs mismatch frozen_model.json")

    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        scenes = all_scenes[train_ids]
        folds = scene_folds_seeded(scenes, count=5, seed=20261031)
        features_b3, ratios, samples, targets = prepare_training(handle, cache, train_ids)

        # 1. Exact 5-fold OOF baseline reproduction on the 795 training images
        n = len(train_ids)
        oof_names = ["raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"]
        oof_scores = np.full((n, len(oof_names), 3), np.nan)
        oof_b3_preds = np.empty((n, 480, 640), dtype=np.float32)
        fold_id = np.empty(n, dtype=int)
        fold_b3_curves = []
        fold_b3_residuals = []

        for f_idx, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            scale = fit_scale(ratios[tr])
            curve = fit_curve(samples[tr], targets[tr], scale)
            res_y = residual_targets(handle, cache, train_ids[tr], curve)[:, 0]  # median_ratio
            res_model = fit_residual(features_b3[tr], res_y, regularization=1.0, shrinkage=0.5)
            fold_b3_curves.append(curve)
            fold_b3_residuals.append(res_model)

            for row in val:
                idx = int(train_ids[row])
                pred = cache[idx].astype(np.float64)
                gt = nyu_depth(handle, idx)
                b1 = clip_depth(scale * pred)
                b2 = apply_curve(pred, curve)
                b3 = apply_residual(b2, features_b3[row], res_model)
                oof_b3_preds[row] = b3.astype(np.float32)
                for col, d in enumerate((clip_depth(pred), b1, b2, b3)):
                    oof_scores[row, col] = evaluate_image(d, gt)
                fold_id[row] = f_idx

        # Save OOF baseline table
        oof_rows = []
        for col, name in enumerate(oof_names):
            row_dict = {
                "method": name,
                "images": n,
                **dict(zip(METRIC_NAMES, oof_scores[:, col].mean(axis=0).tolist())),
            }
            for f_idx in range(5):
                f_mask = fold_id == f_idx
                f_vals = oof_scores[f_mask, col].mean(axis=0)
                row_dict[f"fold{f_idx}_abs_rel"] = float(f_vals[0])
                row_dict[f"fold{f_idx}_rmse"] = float(f_vals[1])
                row_dict[f"fold{f_idx}_delta1"] = float(f_vals[2])
            oof_rows.append(row_dict)
        oof_df = pd.DataFrame(oof_rows)
        write_csv_lf(output_dir / "diag_oof_baseline.csv", oof_df)

        # 2. Oracle error ceilings, residual targets, features, and depth-bin breakdown
        oracle_names = [
            "raw",
            "oof_b3_residual_v1",
            "raw_oracle_med_scalar",
            "raw_oracle_absrel_scalar",
            "raw_oracle_rmse_scalar",
            "b3_oracle_med_scalar",
            "b3_oracle_absrel_scalar",
            "b3_oracle_rmse_scalar",
            "b3_oracle_log_affine",
            "b3_oracle_inv_affine",
            "b3_oracle_spatial_linear",
            "b3_oracle_spatial_quadric",
            "b3_oracle_spatial_plus_depth",
        ]
        oracle_scores = np.full((n, len(oracle_names), 3), np.nan)

        e_raw_med = np.empty(n, dtype=np.float64)
        e_med = np.empty(n, dtype=np.float64)
        e_absrel = np.empty(n, dtype=np.float64)
        e_rmse = np.empty(n, dtype=np.float64)
        e_slope_u = np.empty(n, dtype=np.float64)
        e_slope_v = np.empty(n, dtype=np.float64)
        e_radial = np.empty(n, dtype=np.float64)

        feat_records: list[dict[str, float]] = []
        bin_records: list[dict[str, Any]] = []

        # Accumulate 2D spatial residual statistics over CROP (426x560)
        res_sum = np.zeros(CROP_U_TILDE.shape, dtype=np.float64)
        res_cnt = np.zeros(CROP_U_TILDE.shape, dtype=np.int64)
        res_sum_by_depth = {b: np.zeros(CROP_U_TILDE.shape, dtype=np.float64) for b in range(5)}
        res_cnt_by_depth = {b: np.zeros(CROP_U_TILDE.shape, dtype=np.int64) for b in range(5)}

        # Also collect sampled pixels for global OOF spatial field check
        sample_rng = np.random.default_rng(20261121)
        sampled_log_res = []
        sampled_u = []
        sampled_v = []
        sampled_r2 = []
        sampled_lz = []
        sampled_row = []

        intr_all = cache.intrinsics_for(train_ids)
        # Check cross-metric scalar multipliers on OOF B3
        b3_all_ratios_ar = []
        b3_all_p = []
        b3_all_g = []

        for row, idx in enumerate(train_ids):
            idx_int = int(idx)
            raw_full = cache[idx_int].astype(np.float64)
            b3_full = oof_b3_preds[row].astype(np.float64)
            gt_full = nyu_depth(handle, idx_int).astype(np.float64)
            rgb_full = nyu_image(handle, idx_int)
            K = intr_all[row]

            gt_c = gt_full[CROP]
            valid = valid_depth(gt_c)
            raw_c = raw_full[CROP][valid]
            b3_c = b3_full[CROP][valid]
            g_c = gt_c[valid]
            u_c = CROP_U_TILDE[valid]
            v_c = CROP_V_TILDE[valid]
            r2_c = CROP_R2[valid]

            # Oracles
            raw_med_s = float(np.median(g_c / raw_c))
            raw_ar_s = optimal_absrel_scalar(raw_c, g_c)
            raw_rmse_s = optimal_rmse_scalar(raw_c, g_c)

            b3_med_s = float(np.median(g_c / b3_c))
            b3_ar_s = optimal_absrel_scalar(b3_c, g_c)
            b3_rmse_s = optimal_rmse_scalar(b3_c, g_c)
            b3_log_aff = fit_log_affine_oracle(b3_c, g_c)
            b3_inv_aff = fit_inv_affine_oracle(b3_c, g_c)
            b3_sp_lin, coef_lin = fit_spatial_oracle(b3_c, g_c, u_c, v_c, r2_c, "spatial_linear")
            b3_sp_quad, _ = fit_spatial_oracle(b3_c, g_c, u_c, v_c, r2_c, "spatial_quadric")
            b3_sp_plus, _ = fit_spatial_oracle(b3_c, g_c, u_c, v_c, r2_c, "spatial_plus_depth")

            e_raw_med[row] = float(np.log(raw_med_s))
            e_med[row] = float(np.log(b3_med_s))
            e_absrel[row] = float(np.log(b3_ar_s))
            e_rmse[row] = float(np.log(b3_rmse_s))
            e_slope_u[row] = float(coef_lin[1])
            e_slope_v[row] = float(coef_lin[2])
            e_radial[row] = float(coef_lin[3])

            eval_arrays = [
                clip_depth(raw_full[CROP][valid]),
                b3_c,
                clip_depth(raw_med_s * raw_c),
                clip_depth(raw_ar_s * raw_c),
                clip_depth(raw_rmse_s * raw_c),
                clip_depth(b3_med_s * b3_c),
                clip_depth(b3_ar_s * b3_c),
                clip_depth(b3_rmse_s * b3_c),
                b3_log_aff,
                b3_inv_aff,
                b3_sp_lin,
                b3_sp_quad,
                b3_sp_plus,
            ]
            for col, arr in enumerate(eval_arrays):
                oracle_scores[row, col] = metrics(arr, g_c, np.ones(len(g_c), dtype=bool))

            # Depth bin profile of OOF B3 and Raw
            for b_idx, (lo, hi) in enumerate(zip(DEPTH_BINS, DEPTH_BINS[1:])):
                b_mask = valid & (gt_c >= lo) & (gt_c < hi)
                if b_mask.any():
                    m_raw = metrics(clip_depth(raw_full[CROP]), gt_c, b_mask)
                    m_b3 = metrics(b3_full[CROP], gt_c, b_mask)
                    signed_log = float(np.mean(np.log(b3_full[CROP][b_mask] / gt_c[b_mask])))
                    bin_records.append(
                        {
                            "row": row,
                            "index": idx_int,
                            "lower_m": lo,
                            "upper_m": hi,
                            "pixels": int(b_mask.sum()),
                            "raw_abs_rel": float(m_raw[0]),
                            "raw_rmse": float(m_raw[1]),
                            "raw_delta1": float(m_raw[2]),
                            "b3_abs_rel": float(m_b3[0]),
                            "b3_rmse": float(m_b3[1]),
                            "b3_delta1": float(m_b3[2]),
                            "b3_signed_log_err": signed_log,
                        }
                    )

            # Spatial map accumulation
            log_err_map = np.zeros_like(gt_c)
            log_err_map[valid] = np.log(g_c / b3_c)
            res_sum += log_err_map
            res_cnt += valid.astype(np.int64)
            for b_idx, (lo, hi) in enumerate(zip(DEPTH_BINS, DEPTH_BINS[1:])):
                b_mask = valid & (gt_c >= lo) & (gt_c < hi)
                res_sum_by_depth[b_idx] += np.where(b_mask, log_err_map, 0.0)
                res_cnt_by_depth[b_idx] += b_mask.astype(np.int64)

            # Sample 1024 valid pixels per image for global OOF spatial field test
            pick = sample_rng.choice(len(g_c), size=min(1024, len(g_c)), replace=False)
            sampled_log_res.append(np.log(g_c[pick] / b3_c[pick]))
            sampled_u.append(u_c[pick])
            sampled_v.append(v_c[pick])
            sampled_r2.append(r2_c[pick])
            sampled_lz.append(np.log(b3_c[pick]))
            sampled_row.append(np.full(len(pick), row, dtype=np.int32))
            b3_all_p.append(b3_c[pick])
            b3_all_g.append(g_c[pick])

            # Extract strictly RGB-only legal features
            feats = {}
            feats.update(extract_intrinsics_features(K))
            feats.update(extract_spatial_3d_features(raw_full, K))
            feats.update(extract_rgb_features(rgb_full, raw_full))
            feat_records.append(feats)

        # 3. Evaluate 5-fold OOF global spatial field on top of OOF B3
        s_log_res = np.concatenate(sampled_log_res)
        s_u = np.concatenate(sampled_u)
        s_v = np.concatenate(sampled_v)
        s_r2 = np.concatenate(sampled_r2)
        s_lz = np.concatenate(sampled_lz)
        s_row = np.concatenate(sampled_row)

        oof_spatial_only_scores = np.full((n, 3), np.nan)
        oof_spatial_depth_scores = np.full((n, 3), np.nan)
        spatial_coefs_folds = []

        for val in folds:
            tr_mask = ~np.isin(s_row, val)
            x_sp_tr = _spatial_basis(s_u[tr_mask], s_v[tr_mask], s_r2[tr_mask], None)
            coef_sp = np.linalg.solve(
                x_sp_tr.T @ x_sp_tr + 10.0 * np.eye(x_sp_tr.shape[1]),
                x_sp_tr.T @ s_log_res[tr_mask],
            )
            spatial_coefs_folds.append(coef_sp.tolist())

            x_spd_tr = _spatial_basis(s_u[tr_mask], s_v[tr_mask], s_r2[tr_mask], s_lz[tr_mask])
            coef_spd = np.linalg.solve(
                x_spd_tr.T @ x_spd_tr + 10.0 * np.eye(x_spd_tr.shape[1]),
                x_spd_tr.T @ s_log_res[tr_mask],
            )

            # Evaluate full-resolution on validation images
            for row in val:
                idx_int = int(train_ids[row])
                gt_c = nyu_depth(handle, idx_int)[CROP].astype(np.float64)
                b3_c_full = oof_b3_preds[row][CROP].astype(np.float64)
                valid = valid_depth(gt_c)

                sp_map = (
                    coef_sp[0]
                    + coef_sp[1] * CROP_U_TILDE
                    + coef_sp[2] * CROP_V_TILDE
                    + coef_sp[3] * (CROP_U_TILDE ** 2)
                    + coef_sp[4] * (CROP_U_TILDE * CROP_V_TILDE)
                    + coef_sp[5] * (CROP_V_TILDE ** 2)
                    + coef_sp[6] * CROP_R2
                )
                pred_sp = clip_depth(b3_c_full * np.exp(sp_map))
                oof_spatial_only_scores[row] = metrics(pred_sp, gt_c, valid)

                lz_c = np.log(b3_c_full)
                lz_c_centered = lz_c - s_lz[tr_mask].mean()
                spd_map = (
                    coef_spd[0]
                    + coef_spd[1] * CROP_U_TILDE
                    + coef_spd[2] * CROP_V_TILDE
                    + coef_spd[3] * (CROP_U_TILDE ** 2)
                    + coef_spd[4] * (CROP_U_TILDE * CROP_V_TILDE)
                    + coef_spd[5] * (CROP_V_TILDE ** 2)
                    + coef_spd[6] * CROP_R2
                    + coef_spd[7] * CROP_U_TILDE * lz_c_centered
                    + coef_spd[8] * CROP_V_TILDE * lz_c_centered
                    + coef_spd[9] * CROP_R2 * lz_c_centered
                )
                pred_spd = clip_depth(b3_c_full * np.exp(spd_map))
                oof_spatial_depth_scores[row] = metrics(pred_spd, gt_c, valid)

    # Assemble Oracle table
    oracle_rows = []
    for col, name in enumerate(oracle_names):
        m = oracle_scores[:, col].mean(axis=0)
        oracle_rows.append({"method": name, "abs_rel": float(m[0]), "rmse": float(m[1]), "delta1": float(m[2])})
    oracle_rows.append(
        {
            "method": "oof_global_spatial_quadric",
            "abs_rel": float(oof_spatial_only_scores.mean(axis=0)[0]),
            "rmse": float(oof_spatial_only_scores.mean(axis=0)[1]),
            "delta1": float(oof_spatial_only_scores.mean(axis=0)[2]),
        }
    )
    oracle_rows.append(
        {
            "method": "oof_global_spatial_x_depth",
            "abs_rel": float(oof_spatial_depth_scores.mean(axis=0)[0]),
            "rmse": float(oof_spatial_depth_scores.mean(axis=0)[1]),
            "delta1": float(oof_spatial_depth_scores.mean(axis=0)[2]),
        }
    )
    oracle_df = pd.DataFrame(oracle_rows)
    write_csv_lf(output_dir / "diag_oracles.csv", oracle_df)

    # Check global residual cross-metric bias on OOF B3
    p_flat = np.concatenate(b3_all_p)
    g_flat = np.concatenate(b3_all_g)
    global_b3_med = float(np.median(g_flat / p_flat))
    global_b3_ar = optimal_absrel_scalar(p_flat, g_flat)
    global_b3_rmse = optimal_rmse_scalar(p_flat, g_flat)

    # 4. Correlations & 5-fold scene-CV R^2 for every feature
    feat_df = pd.DataFrame(feat_records)
    corr_rows = []
    group_map = {}
    for k in extract_intrinsics_features(intr_all[0]).keys():
        group_map[k] = "A_intrinsics"
    for k in extract_spatial_3d_features(cache[int(train_ids[0])].astype(np.float64), intr_all[0]).keys():
        group_map[k] = "B_b3_orig" if k.startswith("b3_") else "B_spatial_3d"
    for k in feat_df.columns:
        group_map.setdefault(k, "C_rgb")

    for col in feat_df.columns:
        vals = feat_df[col].to_numpy(dtype=np.float64)
        pr_med, pp_med = pearsonr(vals, e_med)
        sr_med, sp_med = spearmanr(vals, e_med)
        pr_ar, pp_ar = pearsonr(vals, e_absrel)
        sr_ar, sp_ar = spearmanr(vals, e_absrel)
        pr_raw, _ = pearsonr(vals, e_raw_med)
        pr_sv, _ = pearsonr(vals, e_slope_v)
        pr_rad, _ = pearsonr(vals, e_radial)
        r2_med = scene_cv_ridge_r2(vals, e_med, folds, lam=1.0)
        r2_ar = scene_cv_ridge_r2(vals, e_absrel, folds, lam=1.0)
        corr_rows.append(
            {
                "feature": col,
                "group": group_map[col],
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "pearson_e_med": float(pr_med),
                "p_e_med": float(pp_med),
                "spearman_e_med": float(sr_med),
                "p_spearman_e_med": float(sp_med),
                "cv_r2_e_med": float(r2_med),
                "pearson_e_absrel": float(pr_ar),
                "spearman_e_absrel": float(sr_ar),
                "cv_r2_e_absrel": float(r2_ar),
                "pearson_e_raw_med": float(pr_raw),
                "pearson_e_slope_v": float(pr_sv),
                "pearson_e_radial": float(pr_rad),
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values("cv_r2_e_med", ascending=False)
    write_csv_lf(output_dir / "diag_correlations.csv", corr_df)

    # Multivariate scene-CV R^2 subsets
    b3_cols = [c for c in feat_df.columns if c.startswith("b3_")]
    geo_cols = ["log_fx_w", "log_fy_h", "log_aspect", "cx_off", "cy_off", "mean_ray_cos", "std_ray_cos"]
    sp3d_cols = [
        "dz_du", "dz_dv", "dz_dr2", "top_minus_bot_logz", "center_minus_border_logz",
        "iqr_logz", "skew_logz", "frac_near_1p2m", "frac_far_4m", "grad_aniso_uv",
        "log_span_x", "log_span_y", "log_span_z", "span_y_over_z", "span_x_over_z",
        "mean_inv_z", "std_inv_z",
    ]
    rgb_cols = [c for c in feat_df.columns if group_map[c] == "C_rgb"]
    decol_cols = [
        "log_f_geom", "log_aspect", "cx_off", "cy_off", "mean_ray_cos",
        "b3_q50", "b3_std", "b3_grad", "dz_dv", "dz_dr2", "span_y_over_z",
        "mean_inv_z", "lum_mean", "rgb_grad_mean", "frac_low_texture",
    ]
    subset_r2 = {}
    for s_name, cols in [
        ("B3_8feat", b3_cols),
        ("Geo_K", geo_cols),
        ("Spatial3D", sp3d_cols),
        ("RGB", rgb_cols),
        ("Geo_K+B3_8feat", geo_cols + b3_cols),
        ("Geo_K+Spatial3D", geo_cols + sp3d_cols),
        ("Decollinearized_All", decol_cols),
    ]:
        X_sub = feat_df[cols].to_numpy(dtype=np.float64)
        subset_r2[s_name] = {
            "dims": len(cols),
            "cv_r2_e_med_lam1": scene_cv_ridge_r2(X_sub, e_med, folds, lam=1.0),
            "cv_r2_e_med_lam10": scene_cv_ridge_r2(X_sub, e_med, folds, lam=10.0),
            "cv_r2_e_absrel_lam1": scene_cv_ridge_r2(X_sub, e_absrel, folds, lam=1.0),
            "cv_r2_e_absrel_lam10": scene_cv_ridge_r2(X_sub, e_absrel, folds, lam=10.0),
        }

    # Condition number of original B3_8feat vs Geo_K
    b3_std_mat = (feat_df[b3_cols] - feat_df[b3_cols].mean()) / feat_df[b3_cols].std()
    cond_b3 = float(np.linalg.cond(b3_std_mat.to_numpy(dtype=np.float64)))
    geo_std_mat = (feat_df[geo_cols] - feat_df[geo_cols].mean()) / feat_df[geo_cols].std()
    cond_geo = float(np.linalg.cond(geo_std_mat.to_numpy(dtype=np.float64)))

    # 5. Spatial 4x4 grid + radial ring summary + depth bin summary
    mean_res_map = res_sum / np.maximum(res_cnt, 1)
    spatial_rows = []
    v_edges = np.linspace(0, mean_res_map.shape[0], 5, dtype=int)
    u_edges = np.linspace(0, mean_res_map.shape[1], 5, dtype=int)
    for vi in range(4):
        for ui in range(4):
            patch = mean_res_map[v_edges[vi] : v_edges[vi + 1], u_edges[ui] : u_edges[ui + 1]]
            spatial_rows.append(
                {
                    "region_type": "grid_4x4",
                    "region": f"v{vi}_u{ui}",
                    "mean_log_gt_over_b3": float(patch.mean()),
                    "mean_abs_log_err": float(np.abs(patch).mean()),
                }
            )
    for r_lo, r_hi, lbl in [(0.0, 0.25, "center_r2<0.25"), (0.25, 0.65, "mid_0.25<=r2<0.65"), (0.65, 2.5, "periphery_r2>=0.65")]:
        m = (CROP_R2 >= r_lo) & (CROP_R2 < r_hi)
        spatial_rows.append(
            {
                "region_type": "radial_ring",
                "region": lbl,
                "mean_log_gt_over_b3": float(mean_res_map[m].mean()),
                "mean_abs_log_err": float(np.abs(mean_res_map[m]).mean()),
            }
        )
    for strip_lbl, sl in [("top_third", slice(0, 142)), ("mid_third", slice(142, 284)), ("bot_third", slice(284, 426))]:
        spatial_rows.append(
            {
                "region_type": "vertical_strip",
                "region": strip_lbl,
                "mean_log_gt_over_b3": float(mean_res_map[sl, :].mean()),
                "mean_abs_log_err": float(np.abs(mean_res_map[sl, :]).mean()),
            }
        )
    spatial_df = pd.DataFrame(spatial_rows)
    write_csv_lf(output_dir / "diag_spatial_bins.csv", spatial_df)

    bin_df = pd.DataFrame(bin_records)
    bin_summary = (
        bin_df.groupby(["lower_m", "upper_m"], as_index=False)[
            ["raw_abs_rel", "raw_rmse", "raw_delta1", "b3_abs_rel", "b3_rmse", "b3_delta1", "b3_signed_log_err"]
        ]
        .mean()
    )
    bin_summary["images"] = bin_df.groupby(["lower_m", "upper_m"])["index"].count().values
    bin_summary["pixels"] = bin_df.groupby(["lower_m", "upper_m"])["pixels"].sum().values
    write_csv_lf(output_dir / "diag_depth_bins.csv", bin_summary)

    # Intrinsics distribution stats (no K_true used!)
    intr_stats = {}
    for col in ["fx", "fy", "cx", "cy", "fov_x", "fov_y"]:
        vals = feat_df[col].to_numpy(dtype=np.float64)
        qs = np.quantile(vals, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        intr_stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(qs[0]),
            "p05": float(qs[1]),
            "p25": float(qs[2]),
            "median": float(qs[3]),
            "p75": float(qs[4]),
            "p95": float(qs[5]),
            "max": float(qs[6]),
        }

    # Global spatial field mean fold coefs
    mean_sp_coef = np.mean(spatial_coefs_folds, axis=0).tolist()

    summary = {
        "split": "official795_training_only",
        "images": int(n),
        "scenes": int(len(np.unique(scenes))),
        "test_read": False,
        "oof_baseline": oof_rows,
        "oracles": oracle_rows,
        "cross_metric_bias_on_oof_b3": {
            "global_median_multiplier": global_b3_med,
            "global_absrel_multiplier": global_b3_ar,
            "global_rmse_multiplier": global_b3_rmse,
        },
        "residual_target_stats": {
            "e_med_mean": float(e_med.mean()),
            "e_med_std": float(e_med.std()),
            "e_absrel_mean": float(e_absrel.mean()),
            "e_absrel_std": float(e_absrel.std()),
            "e_slope_v_mean": float(e_slope_v.mean()),
            "e_slope_v_std": float(e_slope_v.std()),
            "e_radial_mean": float(e_radial.mean()),
            "e_radial_std": float(e_radial.std()),
        },
        "intrinsics_hat_stats": intr_stats,
        "feature_condition_numbers": {"B3_8feat": cond_b3, "Geo_K": cond_geo},
        "multivariate_scene_cv_r2": subset_r2,
        "global_spatial_field_mean_coef_1_u_v_u2_uv_v2_r2": mean_sp_coef,
        "depth_bin_summary": bin_summary.to_dict(orient="records"),
        "spatial_summary": spatial_rows,
    }
    write_json_lf(output_dir / "diag_results.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_diagnosis(args.data_root, args.predictions, args.frozen_model, args.output)
    print("Diagnosis complete on", summary["images"], "training images /", summary["scenes"], "scenes.", flush=True)


if __name__ == "__main__":
    main()
