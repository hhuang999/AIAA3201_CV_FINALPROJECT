"""Training-only residual, oracle ceiling, signal predictability, and left-tail bimodality diagnosis."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import nyu_depth, open_nyu, scene_names
from .geocalib_cache import CONF_SUMMARY_NAMES, load_geocalib_cache
from .geocalib_io import scene_folds_seeded, write_csv_lf, write_json_lf, write_text_lf
from .metrics import CROP, DEPTH_BINS, clip_depth, evaluate_image, metrics, valid_depth


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
CROP_SUB = (slice(45, 471, 4), slice(41, 601, 4))


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


def roc_auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact rank-based ROC-AUC (with tie averaging); labels are 0/1 booleans."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("Expected finite aligned scores and boolean labels")
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    u = float(np.sum(ranks[y]) - n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute standardized mean difference (pos - neg) / pooled_sd."""
    a = np.asarray(pos, dtype=np.float64)
    b = np.asarray(neg, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


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
    oof_null = np.empty_like(y_arr)
    for val in folds:
        tr = np.setdiff1d(np.arange(len(y_arr)), val)
        mu = x_arr[tr].mean(axis=0)
        sd = np.maximum(x_arr[tr].std(axis=0), 1e-8)
        z_tr = (x_arr[tr] - mu) / sd
        z_val = (x_arr[val] - mu) / sd
        y0 = y_arr[tr].mean()
        oof_null[val] = y0
        coef = np.linalg.solve(
            z_tr.T @ z_tr / len(tr) + lam * np.eye(z_tr.shape[1]),
            z_tr.T @ (y_arr[tr] - y0) / len(tr),
        )
        oof_pred[val] = y0 + z_val @ coef
    ss_res = float(np.sum((y_arr - oof_pred) ** 2))
    ss_tot = float(np.sum((y_arr - oof_null) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def scene_cv_ridge_r2_pca(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    lam: float = 10.0,
    pca_k: int | None = None,
) -> tuple[float, np.ndarray]:
    """Fold-strict out-of-fold R^2 with optional fold-internal SVD PCA + standardized ridge."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    y_arr = np.asarray(y, dtype=np.float64)
    oof_pred = np.empty_like(y_arr)
    oof_null = np.empty_like(y_arr)

    for val in folds:
        tr = np.setdiff1d(np.arange(len(y_arr)), val)
        mu = x_arr[tr].mean(axis=0)
        sd = np.maximum(x_arr[tr].std(axis=0), 1e-8)
        z_tr = (x_arr[tr] - mu) / sd
        z_val = (x_arr[val] - mu) / sd

        if pca_k is not None and pca_k < z_tr.shape[1]:
            _, _, vt = np.linalg.svd(z_tr, full_matrices=False)
            proj = vt[:pca_k].T
            f_tr = z_tr @ proj
            f_val = z_val @ proj
            f_mu = f_tr.mean(axis=0)
            f_sd = np.maximum(f_tr.std(axis=0), 1e-8)
            f_tr = (f_tr - f_mu) / f_sd
            f_val = (f_val - f_mu) / f_sd
        else:
            f_tr, f_val = z_tr, z_val

        y0 = float(y_arr[tr].mean())
        oof_null[val] = y0
        n_tr = len(tr)
        d = f_tr.shape[1]
        if n_tr < d:
            gram = f_tr @ f_tr.T / n_tr + lam * np.eye(n_tr)
            alpha = np.linalg.solve(gram, y_arr[tr] - y0)
            oof_pred[val] = y0 + (f_val @ f_tr.T / n_tr) @ alpha
        else:
            coef = np.linalg.solve(
                f_tr.T @ f_tr / n_tr + lam * np.eye(d),
                f_tr.T @ (y_arr[tr] - y0) / n_tr,
            )
            oof_pred[val] = y0 + f_val @ coef

    ss_res = float(np.sum((y_arr - oof_pred) ** 2))
    ss_tot = float(np.sum((y_arr - oof_null) ** 2))
    return float(1.0 - ss_res / max(ss_tot, 1e-12)), oof_pred


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
    """Extract log-depth summary, spatial depth gradients, and pseudo-3D geometry from (Z_hat, K_hat)."""
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

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x3d = crop_p * ((CROP_U_PIX - cx) / fx)
    y3d = crop_p * ((CROP_V_PIX - cy) / fy)
    span_x = float(max(np.quantile(x3d, 0.95) - np.quantile(x3d, 0.05), 1e-6))
    span_y = float(max(np.quantile(y3d, 0.95) - np.quantile(y3d, 0.05), 1e-6))
    span_z = float(max(np.quantile(crop_p, 0.95) - np.quantile(crop_p, 0.05), 1e-6))
    inv_z = 1.0 / crop_p

    return {
        "log_q10": float(q10),
        "log_q25": float(q25),
        "log_q50": float(q50),
        "log_q75": float(q75),
        "log_q90": float(q90),
        "log_mean": lp_mean,
        "log_std": lp_std,
        "log_grad": float(grad),
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
    grad_mag = np.sqrt(dv[:, :-1] ** 2 + du[:-1, :] ** 2)

    lp = np.log(np.asarray(raw_depth, dtype=np.float64)[CROP])
    lp_dv = np.abs(np.diff(lp, axis=0))[:, :-1]
    lp_du = np.abs(np.diff(lp, axis=1))[:-1, :]
    lp_grad_mag = np.sqrt(lp_dv ** 2 + lp_du ** 2)
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


def compute_flip_features(
    raw_depth: np.ndarray,
    k_orig: np.ndarray,
    flip_depth: np.ndarray,
    k_flip: np.ndarray,
    conf_orig: np.ndarray | None = None,
    conf_flip: np.ndarray | None = None,
    cls_orig: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute label-free 2-pass horizontal-flip self-consistency features."""
    f_o = 0.5 * (float(k_orig[0, 0]) + float(k_orig[1, 1])) / 480.0
    f_f = 0.5 * (float(k_flip[0, 0]) + float(k_flip[1, 1])) / 480.0
    p1 = np.asarray(raw_depth[CROP_SUB], dtype=np.float64)
    p2 = np.asarray(flip_depth[CROP_SUB], dtype=np.float64)
    dlog = np.log(p1) - np.log(p2)
    adlog = np.abs(dlog)
    rel_diff = np.abs(p1 - p2) / np.maximum(0.5 * (p1 + p2), 1e-6)

    cx_err = abs((float(k_orig[0, 2]) - float(k_flip[0, 2]))) / 640.0
    feats = {
        "f_orig_tilde": f_o,
        "f_flip_tilde": f_f,
        "f_tilde_mean": 0.5 * (f_o + f_f),
        "f_tilde_max": max(f_o, f_f),
        "f_tilde_min": min(f_o, f_f),
        "f_abs_diff": abs(f_o - f_f),
        "f_rel_diff": abs(f_o - f_f) / max(0.5 * (f_o + f_f), 1e-6),
        "f_signed_diff": f_f - f_o,
        "cx_sym_err": cx_err,
        "flip_log_abs_mean": float(adlog.mean()),
        "flip_log_abs_med": float(np.median(adlog)),
        "flip_log_abs_p90": float(np.quantile(adlog, 0.90)),
        "flip_signed_log_mean": float(dlog.mean()),
        "flip_rel_diff_mean": float(rel_diff.mean()),
    }
    if conf_orig is not None and conf_flip is not None:
        c1 = np.asarray(conf_orig[CROP_SUB], dtype=np.float64)
        c2 = np.asarray(conf_flip[CROP_SUB], dtype=np.float64)
        feats["conf_flip_abs_mean"] = float(np.mean(np.abs(c1 - c2)))
    else:
        feats["conf_flip_abs_mean"] = 0.0
    if cls_orig is not None and cls_flip is not None:
        a = np.asarray(cls_orig, dtype=np.float64)
        b = np.asarray(cls_flip, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        feats["cls_flip_cos"] = float(np.dot(a, b) / denom) if denom > 1e-12 else 1.0
    else:
        feats["cls_flip_cos"] = 1.0
    return feats


def extract_geo_interaction_features(
    raw_depth: np.ndarray,
    k_mat: np.ndarray,
    base_feat: dict[str, float],
    sp_feat: dict[str, float],
) -> dict[str, float]:
    """Compute geometric and K_hat x Spatial3D interaction features."""
    f_t = float(base_feat["f_tilde"])
    f_hinge = max(0.0, 1.14 - f_t)
    log_sy = float(base_feat["log_span_y"])
    std_lz = float(base_feat["std_logz"])
    dz_dv = float(base_feat["dz_dv"])
    mean_iz = float(sp_feat["mean_inv_z"])
    std_iz = float(base_feat["std_inv_z"])
    q25 = float(base_feat["q25_logz"])
    return {
        "f_tilde": f_t,
        "f_tilde_sq": f_t * f_t,
        "f_tilde_hinge": f_hinge,
        "f_tilde_hinge_sq": f_hinge * f_hinge,
        "log_f_tilde": float(np.log(max(f_t, 1e-4))),
        "inv_f_tilde": float(1.0 / max(f_t, 1e-4)),
        "log_aspect": float(np.log(float(k_mat[0, 0]) / max(float(k_mat[1, 1]), 1e-6))),
        "f_x_log_spany": f_t * log_sy,
        "f_x_std_logz": f_t * std_lz,
        "f_x_dz_dv": f_t * dz_dv,
        "f_x_mean_invz": f_t * mean_iz,
        "hinge_x_log_spany": f_hinge * log_sy,
        "hinge_x_q25": f_hinge * q25,
        "hinge_x_std_invz": f_hinge * std_iz,
        "std_logz_x_dz_dv": std_lz * dz_dv,
        "q25_x_std_invz": q25 * std_iz,
        "near_far_ratio": float(sp_feat["frac_near_1p2m"]) / (float(sp_feat["frac_far_4m"]) + 0.05),
    }


def _scene_type(name: str) -> str:
    return re.sub(r"_[0-9]+.*$", "", str(name))


def run_diagnosis(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute training-only post-geometry oracle, signal R^2, and left-tail bimodality diagnosis."""
    from .geocalib_model import (
        CV_SAMPLES_PER_IMAGE,
        GEO_ONLY_PRIMARY_SPEC,
        SEED,
        apply_geocalib_model,
        apply_spline,
        fit_geo_only_model,
        fit_monotone_spline,
        fit_scale,
        prepare_training_bundle,
    )

    torch.set_num_threads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_geocalib_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids

    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        scenes = all_scenes[train_ids]
        folds = scene_folds_seeded(scenes, count=5, seed=SEED)
        data = prepare_training_bundle(handle, cache.base, train_ids)
        n = len(train_ids)

        oof_geo = np.empty((n, 480, 640), dtype=np.float32)
        oof_spline = np.empty((n, 480, 640), dtype=np.float32)
        oof_geo_flip_pass = np.empty((n, 480, 640), dtype=np.float32)
        raw_scores = np.empty((n, 3), dtype=np.float64)
        spline_scores = np.empty((n, 3), dtype=np.float64)
        geo_scores = np.empty((n, 3), dtype=np.float64)
        raw_flip_log_scores = np.empty((n, 3), dtype=np.float64)
        raw_flip_conf_scores = np.empty((n, 3), dtype=np.float64)
        geo_applied_flip_avg_scores = np.empty((n, 3), dtype=np.float64)
        geo_flip_log_scores = np.empty((n, 3), dtype=np.float64)
        geo_flip_conf_scores = np.empty((n, 3), dtype=np.float64)
        fold_id = np.empty(n, dtype=int)

        intr_flip_all = cache.intrinsics_flip_for(train_ids)

        for f_idx, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            scale = fit_scale(data["ratios"][tr])
            spline = fit_monotone_spline(data["samples_spline"][tr], data["targets_spline"][tr], scale)

            geo_model = fit_geo_only_model(
                spec=GEO_ONLY_PRIMARY_SPEC,
                samples_p=data["samples_p"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_gt=data["samples_gt"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_row=data["samples_row"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_col=data["samples_col"][tr, :CV_SAMPLES_PER_IMAGE],
                features=[data["base_feats"][i] for i in tr],
                intrinsics=data["intrinsics"][tr],
                ratios=data["ratios"][tr],
                max_iter=65,
            )

            for row in val:
                idx = int(train_ids[row])
                fold_id[row] = f_idx
                p1 = cache[idx].astype(np.float64)
                p2 = cache.depth_flip(idx).astype(np.float64)
                c1 = np.maximum(cache.confidence(idx).astype(np.float64), 1e-4)
                c2 = np.maximum(cache.confidence_flip(idx).astype(np.float64), 1e-4)
                k1 = data["intrinsics"][row]
                k2 = intr_flip_all[row]
                gt = nyu_depth(handle, idx)

                sp_out = apply_spline(p1, spline)
                g_out = apply_geocalib_model(p1, k1, geo_model, cached_features=data["base_feats"][row])

                p_flip_log = clip_depth(np.exp(0.5 * (np.log(p1) + np.log(p2))))
                w1 = c1 / (c1 + c2)
                p_flip_conf = clip_depth(np.exp(w1 * np.log(p1) + (1.0 - w1) * np.log(p2)))
                k_mean = 0.5 * (k1 + k2)
                g_on_avg = apply_geocalib_model(p_flip_log, k_mean, geo_model)
                g_pass2 = apply_geocalib_model(p2, k2, geo_model)
                g_flog = clip_depth(np.exp(0.5 * (np.log(g_out) + np.log(g_pass2))))
                g_fconf = clip_depth(np.exp(w1 * np.log(g_out) + (1.0 - w1) * np.log(g_pass2)))

                oof_spline[row] = sp_out.astype(np.float32)
                oof_geo[row] = g_out.astype(np.float32)
                oof_geo_flip_pass[row] = g_flog.astype(np.float32)

                raw_scores[row] = evaluate_image(clip_depth(p1), gt)
                spline_scores[row] = evaluate_image(sp_out, gt)
                geo_scores[row] = evaluate_image(g_out, gt)
                raw_flip_log_scores[row] = evaluate_image(p_flip_log, gt)
                raw_flip_conf_scores[row] = evaluate_image(p_flip_conf, gt)
                geo_applied_flip_avg_scores[row] = evaluate_image(g_on_avg, gt)
                geo_flip_log_scores[row] = evaluate_image(g_flog, gt)
                geo_flip_conf_scores[row] = evaluate_image(g_fconf, gt)

        geo_m = geo_scores.mean(axis=0)
        oracle_names = [
            "raw",
            "monotone_spline",
            "geo_focal_3d",
            "geo_oracle_med_scalar",
            "geo_oracle_absrel_scalar",
            "geo_oracle_rmse_scalar",
            "geo_oracle_log_affine",
            "geo_oracle_inv_affine",
            "geo_oracle_spatial_linear",
            "geo_oracle_spatial_quadric",
            "geo_oracle_spatial_plus_depth",
            "raw_flip_log_avg",
            "raw_flip_conf_avg",
            "geo_applied_to_flip_avg",
            "geo_flip_log_avg",
            "geo_flip_conf_avg",
            "geo_flip_oracle_absrel_scalar",
        ]
        oracle_arr = np.empty((n, len(oracle_names), 3), dtype=np.float64)
        e_geo_med = np.empty(n, dtype=np.float64)
        e_geo_absrel = np.empty(n, dtype=np.float64)
        e_geo_rmse = np.empty(n, dtype=np.float64)
        e_geo_slope_v = np.empty(n, dtype=np.float64)
        e_geo_radial = np.empty(n, dtype=np.float64)
        raw_bias = np.empty(n, dtype=np.float64)
        geo_bias = np.empty(n, dtype=np.float64)

        sp_rows: list[dict[str, float]] = []
        inter_rows: list[dict[str, float]] = []
        flip_rows: list[dict[str, float]] = []
        bin_records: list[dict[str, Any]] = []

        cls_orig_all = cache.cls_tokens_for(train_ids)
        cls_flip_all = cache.cls_tokens_flip_for(train_ids)
        pat_mean_all = cache.patch_mean_for(train_ids)
        pat_std_all = cache.patch_std_for(train_ids)
        df_stats_all = cache.depth_feat_stats_for(train_ids)
        conf_sum_all = cache.confidence_summary_for(train_ids)

        for row, index in enumerate(train_ids):
            idx = int(index)
            gt_c = nyu_depth(handle, idx)[CROP].astype(np.float64)
            valid = valid_depth(gt_c)
            g_c = gt_c[valid]
            raw_full = cache[idx].astype(np.float64)
            raw_c = raw_full[CROP][valid]
            g_pred_c = oof_geo[row][CROP][valid].astype(np.float64)
            gf_pred_c = oof_geo_flip_pass[row][CROP][valid].astype(np.float64)

            u_c = CROP_U_TILDE[valid]
            v_c = CROP_V_TILDE[valid]
            r2_c = CROP_R2[valid]

            med_s = float(np.median(g_c / g_pred_c))
            ar_s = optimal_absrel_scalar(g_pred_c, g_c)
            rm_s = optimal_rmse_scalar(g_pred_c, g_c)
            log_aff = fit_log_affine_oracle(g_pred_c, g_c)
            inv_aff = fit_inv_affine_oracle(g_pred_c, g_c)
            sp_lin, coef_lin = fit_spatial_oracle(g_pred_c, g_c, u_c, v_c, r2_c, "spatial_linear")
            sp_quad, _ = fit_spatial_oracle(g_pred_c, g_c, u_c, v_c, r2_c, "spatial_quadric")
            sp_plus, _ = fit_spatial_oracle(g_pred_c, g_c, u_c, v_c, r2_c, "spatial_plus_depth")

            f_ar_s = optimal_absrel_scalar(gf_pred_c, g_c)

            e_geo_med[row] = float(np.log(med_s))
            e_geo_absrel[row] = float(np.log(ar_s))
            e_geo_rmse[row] = float(np.log(rm_s))
            e_geo_slope_v[row] = float(coef_lin[2])
            e_geo_radial[row] = float(coef_lin[3])
            raw_bias[row] = float(np.median(np.log(raw_c / g_c)))
            geo_bias[row] = float(np.median(np.log(g_pred_c / g_c)))

            oracle_arr[row, 0] = raw_scores[row]
            oracle_arr[row, 1] = spline_scores[row]
            oracle_arr[row, 2] = geo_scores[row]
            for c_idx, pred_1d in (
                (3, clip_depth(med_s * g_pred_c)),
                (4, clip_depth(ar_s * g_pred_c)),
                (5, clip_depth(rm_s * g_pred_c)),
                (6, log_aff),
                (7, inv_aff),
                (8, sp_lin),
                (9, sp_quad),
                (10, sp_plus),
            ):
                oracle_arr[row, c_idx] = metrics(pred_1d, g_c, np.ones(len(g_c), dtype=bool))
            oracle_arr[row, 11] = raw_flip_log_scores[row]
            oracle_arr[row, 12] = raw_flip_conf_scores[row]
            oracle_arr[row, 13] = geo_applied_flip_avg_scores[row]
            oracle_arr[row, 14] = geo_flip_log_scores[row]
            oracle_arr[row, 15] = geo_flip_conf_scores[row]
            oracle_arr[row, 16] = metrics(clip_depth(f_ar_s * gf_pred_c), g_c, np.ones(len(g_c), dtype=bool))

            sp_f = extract_spatial_3d_features(raw_full, data["intrinsics"][row])
            sp_rows.append(sp_f)
            inter_rows.append(
                extract_geo_interaction_features(raw_full, data["intrinsics"][row], data["base_feats"][row], sp_f)
            )
            flip_rows.append(
                compute_flip_features(
                    raw_full,
                    data["intrinsics"][row],
                    cache.depth_flip(idx),
                    intr_flip_all[row],
                    conf_orig=cache.confidence(idx),
                    conf_flip=cache.confidence_flip(idx),
                    cls_orig=cls_orig_all[row],
                    cls_flip=cls_flip_all[row],
                )
            )

            for lo, hi in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                b_mask = valid & (gt_c >= lo) & (gt_c < hi)
                if b_mask.any():
                    m_sp = metrics(oof_spline[row][CROP], gt_c, b_mask)
                    m_g = metrics(oof_geo[row][CROP], gt_c, b_mask)
                    m_gf = metrics(oof_geo_flip_pass[row][CROP], gt_c, b_mask)
                    bin_records.append(
                        {
                            "row": row,
                            "lower_m": lo,
                            "upper_m": hi,
                            "spline_abs_rel": float(m_sp[0]),
                            "geo_abs_rel": float(m_g[0]),
                            "geo_flip_abs_rel": float(m_gf[0]),
                            "spline_delta1": float(m_sp[2]),
                            "geo_delta1": float(m_g[2]),
                            "geo_flip_delta1": float(m_gf[2]),
                        }
                    )

        oracle_df = pd.DataFrame(
            [
                {
                    "method": name,
                    "abs_rel": float(oracle_arr[:, c, 0].mean()),
                    "rmse": float(oracle_arr[:, c, 1].mean()),
                    "delta1": float(oracle_arr[:, c, 2].mean()),
                    "gain_vs_geo_abs_rel": float(oracle_arr[:, 2, 0].mean() - oracle_arr[:, c, 0].mean()),
                    "gain_vs_geo_rmse": float(oracle_arr[:, 2, 1].mean() - oracle_arr[:, c, 1].mean()),
                    "gain_vs_geo_delta1": float(oracle_arr[:, c, 2].mean() - oracle_arr[:, 2, 2].mean()),
                }
                for c, name in enumerate(oracle_names)
            ]
        )
        write_csv_lf(output_dir / "diag_oracles.csv", oracle_df)

        base_feat_df = pd.DataFrame(data["base_feats"])
        sp_df = pd.DataFrame(sp_rows)
        inter_df = pd.DataFrame(inter_rows)
        flip_df = pd.DataFrame(flip_rows)
        conf_df = pd.DataFrame(conf_sum_all, columns=list(CONF_SUMMARY_NAMES))

        geo_used_cols = [
            "f_tilde", "cx_off", "cy_off", "q25_logz", "std_logz", "grad_logz",
            "dz_dv", "grad_aniso_uv", "log_span_y", "span_x_over_z", "std_inv_z",
        ]
        X_geo_ctrl = base_feat_df[geo_used_cols].to_numpy(dtype=np.float64)
        X_inter = inter_df.to_numpy(dtype=np.float64)
        X_conf = conf_df.to_numpy(dtype=np.float64)
        X_flip = flip_df.to_numpy(dtype=np.float64)
        X_pat = np.hstack([pat_mean_all, pat_std_all])
        X_single_all = np.hstack([X_geo_ctrl, X_inter, X_conf, df_stats_all])
        X_plus_flip = np.hstack([X_single_all, X_flip])

        signal_specs: list[tuple[str, str, np.ndarray, list[tuple[int | None, float]]]] = [
            ("geo_sp3d_control", "single_pass", X_geo_ctrl, [(None, 1.0), (None, 10.0)]),
            ("geo_sp3d_interactions", "single_pass", X_inter, [(None, 1.0), (None, 10.0), (None, 100.0)]),
            ("confidence_stats", "single_pass", X_conf, [(None, 1.0), (None, 10.0), (None, 100.0)]),
            ("depth_feat_stats", "single_pass", df_stats_all, [(8, 10.0), (16, 10.0), (32, 10.0), (None, 100.0), (None, 1000.0)]),
            ("cls_token", "single_pass", cls_orig_all, [(8, 10.0), (16, 10.0), (32, 10.0), (None, 100.0), (None, 1000.0)]),
            ("patch_mean_std", "single_pass", X_pat, [(8, 10.0), (16, 10.0), (32, 10.0), (None, 100.0), (None, 1000.0)]),
            ("all_single_pass_compact", "single_pass", X_single_all, [(16, 10.0), (32, 10.0), (None, 100.0), (None, 1000.0)]),
            ("flip_consistency", "multi_pass", X_flip, [(None, 1.0), (None, 10.0), (None, 100.0)]),
            ("all_single_plus_flip", "multi_pass", X_plus_flip, [(16, 10.0), (32, 10.0), (None, 100.0), (None, 1000.0)]),
        ]

        r2_rows: list[dict[str, Any]] = []
        oof_cls_med = None
        oof_df_med = None
        best_cls_r2 = -999.0
        best_df_r2 = -999.0
        for block_name, track, x_mat, cfgs in signal_specs:
            for pca_k, lam in cfgs:
                r2_m, pred_m = scene_cv_ridge_r2_pca(x_mat, e_geo_med, folds, lam=lam, pca_k=pca_k)
                r2_a, _ = scene_cv_ridge_r2_pca(x_mat, e_geo_absrel, folds, lam=lam, pca_k=pca_k)
                r2_sv, _ = scene_cv_ridge_r2_pca(x_mat, e_geo_slope_v, folds, lam=lam, pca_k=pca_k)
                r2_rad, _ = scene_cv_ridge_r2_pca(x_mat, e_geo_radial, folds, lam=lam, pca_k=pca_k)
                r2_rows.append(
                    {
                        "feature_block": block_name,
                        "track": track,
                        "raw_dims": int(x_mat.shape[1]),
                        "pca_k": -1 if pca_k is None else int(pca_k),
                        "ridge_lam": float(lam),
                        "cv_r2_e_geo_med": r2_m,
                        "cv_r2_e_geo_absrel": r2_a,
                        "cv_r2_e_geo_slope_v": r2_sv,
                        "cv_r2_e_geo_radial": r2_rad,
                    }
                )
                if block_name == "cls_token" and r2_m > best_cls_r2:
                    best_cls_r2 = r2_m
                    oof_cls_med = pred_m
                if block_name == "depth_feat_stats" and r2_m > best_df_r2:
                    best_df_r2 = r2_m
                    oof_df_med = pred_m

        r2_df = pd.DataFrame(r2_rows).sort_values("cv_r2_e_geo_med", ascending=False)
        write_csv_lf(output_dir / "diag_signal_r2.csv", r2_df)

        f_tilde_arr = base_feat_df["f_tilde"].to_numpy(dtype=np.float64)
        regimes = np.full(n, "mild", dtype=object)
        regimes[raw_bias < -0.12] = "true_collapse"
        regimes[raw_bias >= -0.08] = "false_alarm"

        lt_df = pd.DataFrame(
            {
                "index": train_ids.astype(int),
                "scene": scenes,
                "scene_type": [_scene_type(s) for s in scenes],
                "fold": fold_id,
                "f_tilde": f_tilde_arr,
                "raw_med_bias": raw_bias,
                "geo_med_bias": geo_bias,
                "raw_abs_rel": raw_scores[:, 0],
                "geo_abs_rel": geo_scores[:, 0],
                "gain_geo_vs_raw_ar": raw_scores[:, 0] - geo_scores[:, 0],
                "geo_flip_abs_rel": geo_flip_log_scores[:, 0],
                "regime": regimes,
                "oof_cls_pred_e_geo_med": oof_cls_med if oof_cls_med is not None else np.zeros(n),
                "oof_df_pred_e_geo_med": oof_df_med if oof_df_med is not None else np.zeros(n),
            }
        )
        disc_pool = pd.concat(
            [
                inter_df.drop(columns=["f_tilde"]),
                sp_df[["dz_du", "dz_dr2", "top_minus_bot_logz", "center_minus_border_logz", "frac_near_1p2m", "frac_far_4m", "mean_inv_z", "iqr_logz"]],
                base_feat_df[["q25_logz", "std_logz", "grad_logz", "dz_dv", "log_span_y", "span_x_over_z", "std_inv_z"]],
                conf_df,
                flip_df,
            ],
            axis=1,
        )
        lt_full_df = pd.concat([lt_df, disc_pool], axis=1)
        lt_sub = lt_full_df[lt_full_df["f_tilde"] < 0.75].sort_values("f_tilde")
        write_csv_lf(output_dir / "diag_left_tail.csv", lt_sub)

        lt_bin = lt_sub[lt_sub["regime"].isin(["true_collapse", "false_alarm"])]
        y_collapse = (lt_bin["regime"] == "true_collapse").to_numpy()
        disc_rows: list[dict[str, Any]] = []
        cand_cols = [c for c in list(disc_pool.columns) + ["f_tilde", "oof_cls_pred_e_geo_med", "oof_df_pred_e_geo_med"] if c != "f_orig_tilde"]
        if int(y_collapse.sum()) >= 2 and int((~y_collapse).sum()) >= 2:
            for col in cand_cols:
                vals = lt_bin[col].to_numpy(dtype=np.float64)
                auc = roc_auc_mann_whitney(vals, y_collapse)
                cd = cohens_d(vals[y_collapse], vals[~y_collapse])
                track_lbl = "multi_pass_flip" if col in flip_df.columns else (
                    "single_pass_conf" if col in conf_df.columns else (
                        "single_pass_sem" if col.startswith("oof_") else "single_pass_geo"
                    )
                )
                disc_rows.append(
                    {
                        "feature": col,
                        "track": track_lbl,
                        "n_true_collapse": int(y_collapse.sum()),
                        "n_false_alarm": int((~y_collapse).sum()),
                        "mean_true_collapse": float(vals[y_collapse].mean()),
                        "mean_false_alarm": float(vals[~y_collapse].mean()),
                        "roc_auc": auc,
                        "abs_auc": max(auc, 1.0 - auc),
                        "cohens_d": cd,
                    }
                )
        disc_df = pd.DataFrame(disc_rows).sort_values("abs_auc", ascending=False)
        write_csv_lf(output_dir / "diag_left_tail_discriminators.csv", disc_df)

        scene_rows: list[dict[str, Any]] = []
        for stype, grp in lt_df.groupby("scene_type"):
            scene_rows.append(
                {
                    "group_type": "scene_type",
                    "name": str(stype),
                    "count": int(len(grp)),
                    "raw_abs_rel": float(grp["raw_abs_rel"].mean()),
                    "geo_abs_rel": float(grp["geo_abs_rel"].mean()),
                    "geo_flip_abs_rel": float(grp["geo_flip_abs_rel"].mean()),
                    "gain_geo_vs_raw": float(grp["raw_abs_rel"].mean() - grp["geo_abs_rel"].mean()),
                    "gain_flip_vs_geo": float(grp["geo_abs_rel"].mean() - grp["geo_flip_abs_rel"].mean()),
                }
            )
        bin_df = pd.DataFrame(bin_records)
        for (lo, hi), grp in bin_df.groupby(["lower_m", "upper_m"]):
            scene_rows.append(
                {
                    "group_type": "depth_bin",
                    "name": f"[{lo:g},{hi:g})m",
                    "count": int(len(grp)),
                    "raw_abs_rel": np.nan,
                    "geo_abs_rel": float(grp["geo_abs_rel"].mean()),
                    "geo_flip_abs_rel": float(grp["geo_flip_abs_rel"].mean()),
                    "gain_geo_vs_raw": float(grp["spline_abs_rel"].mean() - grp["geo_abs_rel"].mean()),
                    "gain_flip_vs_geo": float(grp["geo_abs_rel"].mean() - grp["geo_flip_abs_rel"].mean()),
                }
            )
        write_csv_lf(output_dir / "diag_scene_breakdown.csv", pd.DataFrame(scene_rows))

    best_r2_per_block = (
        r2_df.sort_values("cv_r2_e_geo_med", ascending=False)
        .groupby("feature_block", as_index=False)
        .first()
        .sort_values("cv_r2_e_geo_med", ascending=False)
    )
    summary_payload = {
        "split": "official795_training_only",
        "test_read": False,
        "images": int(n),
        "scenes": int(len(np.unique(scenes))),
        "reconstructed_geo_oof": {"abs_rel": float(geo_m[0]), "rmse": float(geo_m[1]), "delta1": float(geo_m[2])},
        "oracles": oracle_df.to_dict(orient="records"),
        "best_signal_r2_per_block": best_r2_per_block.to_dict(orient="records"),
        "left_tail_count_ftilde_lt_075": int(len(lt_sub)),
        "left_tail_count_ftilde_lt_055": int((lt_sub["f_tilde"] < 0.55).sum()),
        "left_tail_true_collapse_count": int((lt_sub["regime"] == "true_collapse").sum()),
        "left_tail_false_alarm_count": int((lt_sub["regime"] == "false_alarm").sum()),
        "top_discriminators_lt075": disc_df.head(15).to_dict(orient="records"),
    }
    write_json_lf(output_dir / "diag_results.json", summary_payload)
    md_lines = [
        "# Training-Only Post-Geometry Residual, Bimodality & Signal Predictability Diagnosis",
        "",
        "> Strictly training-only (`splits['train']`, 795 images / 249 scenes). `test_read: false`.",
        "",
        f"- Reconstructed `+Focal & 3D Geo` Canonical OOF: `AbsRel={geo_m[0]:.6f}`, `RMSE={geo_m[1]:.6f}`, `delta1={geo_m[2]:.6f}`",
        "",
        "## 1. Post-Geometry Remaining Oracles & Train Flip-TTA Baselines",
        oracle_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 2. Best 5-Fold Scene-CV R^2 Predicting Post-Geometry Residual Error",
        best_r2_per_block[["feature_block", "track", "raw_dims", "pca_k", "ridge_lam", "cv_r2_e_geo_med", "cv_r2_e_geo_absrel", "cv_r2_e_geo_slope_v", "cv_r2_e_geo_radial"]].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    write_text_lf(output_dir / "DIAGNOSIS.md", "\n".join(md_lines) + "\n")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--orig-preds", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--flip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/part3_geocalib"))
    args = parser.parse_args()
    run_diagnosis(args.data_root, args.orig_preds, args.aux_dir, args.flip_dir, args.output)


if __name__ == "__main__":
    main()
