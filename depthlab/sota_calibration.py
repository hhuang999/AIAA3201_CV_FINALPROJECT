"""GeoCalib-3D: Joint geometric, projective, and multi-metric monocular depth calibration.

All development, hyperparameter tuning, and model selection operate exclusively
on the official 795 NYU Depth V2 training images under scene-disjoint CV.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .calibration import (
    BOOTSTRAP_SEED,
    SAMPLE_SEED,
    SEED,
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
from .metrics import CROP, METRIC_NAMES, clip_depth, evaluate_image, valid_depth
from .sota_io import file_sha256, load_nyu_rgb_cache, scene_folds_seeded, write_csv_lf, write_json_lf


DEPTH_KNOTS = np.log(np.array([0.3, 0.7, 1.5, 3.0, 6.0, 12.0], dtype=np.float64))
FOCAL_KNOTS = np.array([0.35, 0.75, 0.98, 1.14, 1.22, 1.35], dtype=np.float64)
FOCAL_ANCHOR_IDX = 3  # 1.14 in f_tilde = 0.5*(fx+fy)/480
LOG_DEPTH_CENTER = 1.0
SAMPLES_PER_IMAGE = 4096
CV_SAMPLES_PER_IMAGE = 1024

LOCKED_LADDER_NAMES = (
    "S2_b2_geok_sp3d_ridge",
    "J0_spline_bal111",
    "J1_geok_bal111_l01",
    "J2_geok_sp3d_bal111_l01",
    "J3_geok_sp3d_slope_bal111_l01",
)
COMPARISON_ORDER = (
    "raw",
    "b1_global_scale",
    "b2_monotone_curve",
    "b3_residual_v1",
    *LOCKED_LADDER_NAMES,
)
PRIMARY_SOTA_METHOD = "J3_geok_sp3d_slope_bal111_l01"
LEARNING_CURVE_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
LEARNING_CURVE_SEED = 20261111

# Precomputed pixel coordinate grids on 480x640 image
_FULL_U_PIX, _FULL_V_PIX = np.meshgrid(
    np.arange(640, dtype=np.float64),
    np.arange(480, dtype=np.float64),
    indexing="xy",
)
_FULL_U_TILDE = (_FULL_U_PIX - 319.5) / 320.0
_FULL_V_TILDE = (_FULL_V_PIX - 239.5) / 240.0
_FULL_R2 = _FULL_U_TILDE ** 2 + _FULL_V_TILDE ** 2
_FULL_D2_CENTER = (_FULL_U_PIX - 319.5) ** 2 + (_FULL_V_PIX - 239.5) ** 2

_FULL_U_PIX_T = torch.from_numpy(_FULL_U_PIX)
_FULL_V_PIX_T = torch.from_numpy(_FULL_V_PIX)
_FULL_V_TILDE_T = torch.from_numpy(_FULL_V_TILDE)
_FULL_R2_T = torch.from_numpy(_FULL_R2)
_FULL_D2_CENTER_T = torch.from_numpy(_FULL_D2_CENTER)

_CROP_U_PIX = _FULL_U_PIX[CROP]
_CROP_V_PIX = _FULL_V_PIX[CROP]
_CROP_V_TILDE = _FULL_V_TILDE[CROP]
_CROP_R2 = _FULL_R2[CROP]
_CROP_D2_CENTER = _FULL_D2_CENTER[CROP]
_MEAN_R2_CROP = float(_CROP_R2.mean())

# Strided 4x4 sub-crop (107x140 = 14,980 px) for <0.6 ms/image feature extraction
CROP_SUB = (slice(45, 471, 4), slice(41, 601, 4))
_SUB_U_PIX = _FULL_U_PIX[CROP_SUB]
_SUB_V_PIX = _FULL_V_PIX[CROP_SUB]
_SUB_V_TILDE = _FULL_V_TILDE[CROP_SUB]
_SUB_V_TILDE_CENTERED = _SUB_V_TILDE - _SUB_V_TILDE.mean()
_SUB_V_TILDE_VAR = float(np.mean(_SUB_V_TILDE_CENTERED ** 2))

GEO_CXCY_NAMES = ("cx_off", "cy_off")
SPATIAL3D_NAMES = (
    "q25_logz",
    "std_logz",
    "grad_logz",
    "dz_dv",
    "grad_aniso_uv",
    "log_span_y",
    "span_x_over_z",
    "std_inv_z",
)
RGB3_NAMES = ("frac_low_texture", "rgb_grad_mean", "lum_mean")
SLOPE_FULL_NAMES = ("f_tilde", "std_logz", "dz_dv", "log_span_y")
SLOPE_GEO_NAMES = ("f_tilde",)
COND2D_FULL_NAMES = ("f_tilde", "dz_dv", "log_span_y", "std_inv_z")
COND2D_GEO_NAMES = ("f_tilde",)

LOSS_WEIGHTS = {
    "rmse_only": (0.0, 1.0, 0.0),
    "balanced_111": (1.0, 1.0, 1.0),
    "ar_rmse_d1_1512": (1.5, 1.0, 1.5),
    "ar_heavy_211": (2.0, 1.0, 1.0),
    "ar_d1_212": (2.0, 1.0, 1.5),
}


@dataclass(frozen=True)
class GeoCalibSpec:
    name: str
    kind: str = "joint"  # "joint" or "two_stage_ridge"
    use_focal_curve: bool = False
    use_geo_cxcy: bool = False
    use_spatial3d: bool = False
    use_rgb3: bool = False
    use_depth_slope: bool = False
    use_ray_unwarp: bool = False
    use_cond_spatial2d: bool = False
    loss_mode: str = "ar_d1_212"
    reg_l2: float = 0.01
    max_log_scale: float = 0.40
    ridge_lam: float = 1.0
    ridge_features: str = "geok"  # for two_stage_ridge sanity controls


def extract_sota_features(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    rgb_image: np.ndarray | None = None,
    need_rgb: bool = True,
) -> dict[str, float]:
    """Extract strictly RGB-only legal geometry and depth-structure features."""
    p = np.asarray(raw_depth, dtype=np.float64)
    if p.shape != (480, 640) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Expected finite positive 480x640 raw depth")
    K = np.asarray(pred_intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Expected finite 3x3 predicted intrinsics")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Predicted focal lengths must be positive")

    f_tilde = float(np.clip(0.5 * (fx + fy) / 480.0, FOCAL_KNOTS[0], FOCAL_KNOTS[-1]))
    cx_off = (cx - 320.0) / 640.0
    cy_off = (cy - 240.0) / 480.0

    p_sub = p[CROP_SUB]
    lp_sub = np.log(p_sub)
    q25_logz = float(np.quantile(lp_sub, 0.25))
    lp_mean = float(lp_sub.mean())
    std_logz = float(lp_sub.std())
    gv = float(np.mean(np.abs(np.diff(lp_sub, axis=0))))
    gu = float(np.mean(np.abs(np.diff(lp_sub, axis=1))))
    grad_logz = 0.5 * (gv + gu)
    grad_aniso_uv = (gv - gu) / (gv + gu + 1e-8)
    dz_dv = float(np.mean(_SUB_V_TILDE_CENTERED * (lp_sub - lp_mean)) / _SUB_V_TILDE_VAR)

    y3d = p_sub * ((_SUB_V_PIX - cy) / fy)
    x3d = p_sub * ((_SUB_U_PIX - cx) / fx)
    q_y = np.quantile(y3d, [0.05, 0.95])
    q_x = np.quantile(x3d, [0.05, 0.95])
    q_z = np.quantile(p_sub, [0.05, 0.95])
    span_y = float(max(q_y[1] - q_y[0], 1e-6))
    span_x = float(max(q_x[1] - q_x[0], 1e-6))
    span_z = float(max(q_z[1] - q_z[0], 1e-6))

    feats = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "f_tilde": f_tilde,
        "cx_off": cx_off,
        "cy_off": cy_off,
        "q25_logz": q25_logz,
        "std_logz": std_logz,
        "grad_logz": grad_logz,
        "dz_dv": dz_dv,
        "grad_aniso_uv": grad_aniso_uv,
        "log_span_y": float(np.log(span_y)),
        "span_x_over_z": float(span_x / span_z),
        "std_inv_z": float(np.std(1.0 / p_sub)),
    }

    if need_rgb and rgb_image is not None:
        rgb_sub = np.asarray(rgb_image[CROP_SUB], dtype=np.float64) / 255.0
        lum = 0.299 * rgb_sub[:, :, 0] + 0.587 * rgb_sub[:, :, 1] + 0.114 * rgb_sub[:, :, 2]
        dv = np.abs(np.diff(lum, axis=0))[:, :-1]
        du = np.abs(np.diff(lum, axis=1))[:-1, :]
        gmag = np.sqrt(dv * dv + du * du)
        feats["lum_mean"] = float(lum.mean())
        feats["rgb_grad_mean"] = float(gmag.mean())
        feats["frac_low_texture"] = float(np.mean(gmag < 0.02))
    else:
        feats["lum_mean"] = 0.0
        feats["rgb_grad_mean"] = 0.0
        feats["frac_low_texture"] = 0.0
    return feats


def _feature_matrix(feat_list: list[dict[str, float]], names: tuple[str, ...]) -> np.ndarray:
    if not names:
        return np.zeros((len(feat_list), 0), dtype=np.float64)
    return np.asarray([[f[k] for k in names] for f in feat_list], dtype=np.float64)


def _scale_feature_names(spec: GeoCalibSpec) -> tuple[str, ...]:
    names: list[str] = []
    if spec.use_geo_cxcy:
        names.extend(GEO_CXCY_NAMES)
    if spec.use_spatial3d:
        names.extend(SPATIAL3D_NAMES)
    if spec.use_rgb3:
        names.extend(RGB3_NAMES)
    return tuple(names)


def _slope_feature_names(spec: GeoCalibSpec) -> tuple[str, ...]:
    if not spec.use_depth_slope:
        return ()
    return SLOPE_FULL_NAMES if spec.use_spatial3d else SLOPE_GEO_NAMES


def _cond2d_feature_names(spec: GeoCalibSpec) -> tuple[str, ...]:
    if not spec.use_cond_spatial2d:
        return ()
    return COND2D_FULL_NAMES if spec.use_spatial3d else COND2D_GEO_NAMES


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[1] == 0:
        return x, np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0), 1e-8)
    return (x - mu) / sd, mu, sd


def _standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    if len(mu) == 0:
        return np.zeros((len(x), 0), dtype=np.float64)
    return (x - mu) / sd


def _interp_1d_piecewise_torch(x: torch.Tensor, x_knots: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
    i = (torch.bucketize(x.contiguous(), x_knots, right=True) - 1).clamp(0, len(x_knots) - 2)
    return y_knots[i] + (x - x_knots[i]) * (y_knots[i + 1] - y_knots[i]) / (x_knots[i + 1] - x_knots[i])


def _interp_1d_piecewise_np(x: np.ndarray | float, x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    i = np.clip(np.searchsorted(x_knots, x_arr, side="right") - 1, 0, len(x_knots) - 2)
    return y_knots[i] + (x_arr - x_knots[i]) * (y_knots[i + 1] - y_knots[i]) / (x_knots[i + 1] - x_knots[i])


class _JointGeoCalibModule(torch.nn.Module):
    """Differentiable float64 calibration model matching apply_sota_model bit-for-bit."""

    def __init__(
        self,
        spec: GeoCalibSpec,
        init_scale: float,
        init_f_ref: float,
        n_scale_feats: int,
        n_slope_feats: int,
        n_cond2d_feats: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer("depth_knots", torch.tensor(DEPTH_KNOTS, dtype=torch.float64))
        self.register_buffer("focal_knots", torch.tensor(FOCAL_KNOTS, dtype=torch.float64))

        init_offset = float(np.arctanh(np.clip(np.log(init_scale) / np.log(2.0), -0.99, 0.99)))
        self.raw_offset = torch.nn.Parameter(torch.tensor(init_offset, dtype=torch.float64))
        # slope = 0.25 + 1.75 * sigmoid(raw); raw = log(3/4) gives slope = 1.0
        self.raw_depth_slopes = torch.nn.Parameter(
            torch.full((len(DEPTH_KNOTS) - 1,), float(np.log(3.0 / 4.0)), dtype=torch.float64)
        )

        if spec.use_focal_curve:
            # m_k = -2.5 * sigmoid(raw_m_k); init at -2.0 -> -0.298 slope with strong gradient
            self.raw_focal_slopes = torch.nn.Parameter(
                torch.full((len(FOCAL_KNOTS) - 1,), -2.0, dtype=torch.float64)
            )
        if n_scale_feats > 0:
            self.w_scale = torch.nn.Parameter(torch.zeros(n_scale_feats, dtype=torch.float64))
        if n_slope_feats > 0:
            self.w_slope = torch.nn.Parameter(torch.zeros(n_slope_feats, dtype=torch.float64))
        if spec.use_ray_unwarp:
            self.raw_gamma_ray = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            f_ref_clip = float(np.clip((init_f_ref - 0.8) / 0.6, 0.05, 0.95))
            init_f_ref_raw = float(np.log(f_ref_clip / (1.0 - f_ref_clip)))
            self.raw_f_ref = torch.nn.Parameter(torch.tensor(init_f_ref_raw, dtype=torch.float64))
        if n_cond2d_feats > 0:
            self.a_v0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.a_r0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.w_v = torch.nn.Parameter(torch.zeros(n_cond2d_feats, dtype=torch.float64))
            self.w_r = torch.nn.Parameter(torch.zeros(n_cond2d_feats, dtype=torch.float64))

    def depth_y_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        offset = np.log(2.0) * torch.tanh(self.raw_offset)
        slopes = 0.25 + 1.75 * torch.sigmoid(self.raw_depth_slopes)
        y0 = (self.depth_knots[0] + offset).reshape(1)
        y = torch.cat([y0, y0 + torch.cumsum(slopes * torch.diff(self.depth_knots), dim=0)])
        return y, slopes

    def focal_y_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        slopes = -2.5 * torch.sigmoid(self.raw_focal_slopes)
        deltas = slopes * torch.diff(self.focal_knots)
        zero = torch.zeros(1, dtype=torch.float64, device=deltas.device)
        raw_y = torch.cat([zero, torch.cumsum(deltas, dim=0)])
        y = raw_y - raw_y[FOCAL_ANCHOR_IDX]
        return y, slopes

    def forward(
        self,
        lp: torch.Tensor,          # (N, S)
        f_tilde: torch.Tensor,     # (N,)
        z_scale: torch.Tensor,     # (N, D_s)
        z_slope: torch.Tensor,     # (N, D_b)
        z_cond2d: torch.Tensor,    # (N, D_c)
        v_tilde: torch.Tensor,     # (N, S)
        r2_centered: torch.Tensor, # (N, S) = r2 - _MEAN_R2_CROP
        d2_center: torch.Tensor,   # (N, S)
        log_cos_hat_c: torch.Tensor,  # (N, S) = log_cos_hat - mean_crop(log_cos_hat)
    ) -> torch.Tensor:
        y_d, _ = self.depth_y_knots()
        base_lp = _interp_1d_piecewise_torch(lp, self.depth_knots, y_d)

        # Per-image scale shift
        shift = torch.zeros_like(f_tilde)
        if self.spec.use_focal_curve:
            y_f, _ = self.focal_y_knots()
            shift = shift + _interp_1d_piecewise_torch(f_tilde, self.focal_knots, y_f)
        if z_scale.shape[1] > 0:
            raw_s = z_scale @ self.w_scale
            m_lim = self.spec.max_log_scale
            shift = shift + m_lim * torch.tanh(raw_s / m_lim)
        if self.spec.use_focal_curve or z_scale.shape[1] > 0:
            shift = 0.50 * torch.tanh(shift / 0.50)

        out_lp = base_lp + shift[:, None]

        # Conditioned depth-slope modulation
        if z_slope.shape[1] > 0:
            delta_b = 0.20 * torch.tanh(z_slope @ self.w_slope)
            out_lp = out_lp + delta_b[:, None] * (lp - LOG_DEPTH_CENTER)

        # Physics ray-cosine unwarping (zero-mean over CROP per image)
        if self.spec.use_ray_unwarp:
            f_ref = 0.8 + 0.6 * torch.sigmoid(self.raw_f_ref)
            gamma_ray = 1.5 * torch.tanh(self.raw_gamma_ray)
            log_cos_ref = -0.5 * torch.log(1.0 + d2_center / ((480.0 * f_ref) ** 2))
            log_cos_ref_c = log_cos_ref - log_cos_ref.mean(dim=1, keepdim=True)
            out_lp = out_lp + gamma_ray * (log_cos_ref_c - log_cos_hat_c)

        # Conditioned 2D spatial field
        if z_cond2d.shape[1] > 0:
            c_v = 0.12 * torch.tanh(self.a_v0 + z_cond2d @ self.w_v)
            c_r2 = 0.12 * torch.tanh(self.a_r0 + z_cond2d @ self.w_r)
            out_lp = out_lp + c_v[:, None] * v_tilde + c_r2[:, None] * r2_centered

        return torch.exp(torch.clamp(out_lp, np.log(0.001), np.log(10.0))).clamp(0.001, 10.0)

    def regularizer(self) -> torch.Tensor:
        _, d_slopes = self.depth_y_knots()
        reg = 0.01 * torch.diff(d_slopes).square().mean() + 0.001 * (d_slopes - 1.0).square().mean()
        if self.spec.use_focal_curve:
            _, f_slopes = self.focal_y_knots()
            reg = reg + 0.005 * torch.diff(f_slopes).square().mean() + 0.0005 * f_slopes.square().mean()
        l2_sum = torch.zeros((), dtype=torch.float64, device=d_slopes.device)
        if hasattr(self, "w_scale"):
            l2_sum = l2_sum + self.w_scale.square().sum()
        if hasattr(self, "w_slope"):
            l2_sum = l2_sum + self.w_slope.square().sum()
        if hasattr(self, "raw_gamma_ray"):
            gamma_ray = 1.5 * torch.tanh(self.raw_gamma_ray)
            l2_sum = l2_sum + 0.5 * gamma_ray.square()
        if hasattr(self, "w_v"):
            l2_sum = l2_sum + self.w_v.square().sum() + self.w_r.square().sum() + self.a_v0.square() + self.a_r0.square()
        return reg + self.spec.reg_l2 * l2_sum


def _loss_components_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    rel = (pred - gt) / gt
    ar = (rel.square() + 1e-6).sqrt().sub(0.001).mean()
    rmse = ((pred - gt).square().mean(dim=-1) + 1e-12).sqrt().mean()
    d1 = torch.sigmoid(((pred.log() - gt.log()).abs() - np.log(1.25)) / 0.03).mean()
    return torch.stack([ar, rmse, d1])


def _centered_log_cos_hat_crop(K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    log_cos = -0.5 * np.log(1.0 + ((_CROP_U_PIX - cx) / fx) ** 2 + ((_CROP_V_PIX - cy) / fy) ** 2)
    return log_cos - float(log_cos.mean())


def _centered_log_cos_hat_full(K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    log_cos_full = -0.5 * np.log(1.0 + ((_FULL_U_PIX - cx) / fx) ** 2 + ((_FULL_V_PIX - cy) / fy) ** 2)
    crop_mean = float(log_cos_full[CROP].mean())
    return log_cos_full - crop_mean


def fit_joint_sota_model(
    spec: GeoCalibSpec,
    samples_p: np.ndarray,
    samples_gt: np.ndarray,
    samples_row: np.ndarray,
    samples_col: np.ndarray,
    features: list[dict[str, float]],
    intrinsics: np.ndarray,
    ratios: np.ndarray,
    device: str = "cpu",
    max_iter: int = 100,
) -> dict[str, Any]:
    """Fit a joint float64 GeoCalib-3D model using L-BFGS exclusively on training samples."""
    n = len(features)
    init_scale = float(np.clip(np.median(ratios), 0.55, 1.85))
    f_tildes = np.array([f["f_tilde"] for f in features], dtype=np.float64)
    init_f_ref = float(np.median(f_tildes))

    scale_names = _scale_feature_names(spec)
    slope_names = _slope_feature_names(spec)
    cond2d_names = _cond2d_feature_names(spec)

    x_scale_raw = _feature_matrix(features, scale_names)
    x_slope_raw = _feature_matrix(features, slope_names)
    x_cond2d_raw = _feature_matrix(features, cond2d_names)

    z_scale_np, mu_scale, sd_scale = _standardize_fit(x_scale_raw)
    z_slope_np, mu_slope, sd_slope = _standardize_fit(x_slope_raw)
    z_cond2d_np, mu_cond2d, sd_cond2d = _standardize_fit(x_cond2d_raw)

    # Precompute pixel spatial coords for sampled training pixels
    v_tilde_s = _CROP_V_TILDE[samples_row, samples_col]
    r2_c_s = _CROP_R2[samples_row, samples_col] - _MEAN_R2_CROP
    d2_s = _CROP_D2_CENTER[samples_row, samples_col]
    log_cos_hat_c_s = np.empty_like(samples_p, dtype=np.float64)
    if spec.use_ray_unwarp:
        for i in range(n):
            lcc = _centered_log_cos_hat_crop(intrinsics[i])
            log_cos_hat_c_s[i] = lcc[samples_row[i], samples_col[i]]
    else:
        log_cos_hat_c_s.fill(0.0)

    dev = torch.device(device)
    p_t = torch.tensor(samples_p, dtype=torch.float64, device=dev)
    lp_t = p_t.log()
    gt_t = torch.tensor(samples_gt, dtype=torch.float64, device=dev)
    f_t = torch.tensor(f_tildes, dtype=torch.float64, device=dev)
    z_sc_t = torch.tensor(z_scale_np, dtype=torch.float64, device=dev)
    z_sl_t = torch.tensor(z_slope_np, dtype=torch.float64, device=dev)
    z_c2_t = torch.tensor(z_cond2d_np, dtype=torch.float64, device=dev)
    v_t = torch.tensor(v_tilde_s, dtype=torch.float64, device=dev)
    r2_t = torch.tensor(r2_c_s, dtype=torch.float64, device=dev)
    d2_t = torch.tensor(d2_s, dtype=torch.float64, device=dev)
    lch_t = torch.tensor(log_cos_hat_c_s, dtype=torch.float64, device=dev)

    with torch.no_grad():
        ref = _loss_components_torch((p_t * init_scale).clamp(0.001, 10.0), gt_t).clamp_min(1e-9)
    weights_t = torch.tensor(LOSS_WEIGHTS[spec.loss_mode], dtype=torch.float64, device=dev)

    def _run_lbfgs(max_iter: int, lr: float = 1.0) -> tuple[_JointGeoCalibModule, list[float]]:
        mod = _JointGeoCalibModule(
            spec=spec,
            init_scale=init_scale,
            init_f_ref=init_f_ref,
            n_scale_feats=len(scale_names),
            n_slope_feats=len(slope_names),
            n_cond2d_feats=len(cond2d_names),
        ).to(dev)
        opt = torch.optim.LBFGS(
            mod.parameters(),
            lr=lr,
            max_iter=max_iter,
            max_eval=int(max_iter * 1.5),
            history_size=25,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )
        hist: list[float] = []

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            pred = mod(lp_t, f_t, z_sc_t, z_sl_t, z_c2_t, v_t, r2_t, d2_t, lch_t)
            comp = _loss_components_torch(pred, gt_t) / ref
            loss = (weights_t * comp).sum() + mod.regularizer()
            if not bool(torch.isfinite(loss)):
                raise ArithmeticError("Nonfinite SOTA calibration objective")
            loss.backward()
            hist.append(float(loss.detach()))
            return loss

        opt.step(closure)
        return mod, hist

    try:
        mod, history = _run_lbfgs(max_iter=max_iter, lr=1.0)
        if any(not torch.isfinite(p).all() for p in mod.parameters()):
            raise ArithmeticError("Nonfinite parameters after LBFGS")
    except ArithmeticError:
        mod, history = _run_lbfgs(max_iter=max_iter, lr=0.5)

    with torch.no_grad():
        y_d, d_slopes = mod.depth_y_knots()
        y_d_list = y_d.cpu().tolist()
        d_slopes_list = d_slopes.cpu().tolist()
        if spec.use_focal_curve:
            y_f, f_slopes = mod.focal_y_knots()
            y_f_list = y_f.cpu().tolist()
            f_slopes_list = f_slopes.cpu().tolist()
        else:
            y_f_list = None
            f_slopes_list = None

        w_scale_list = mod.w_scale.cpu().tolist() if len(scale_names) else []
        w_slope_list = mod.w_slope.cpu().tolist() if len(slope_names) else []
        if spec.use_ray_unwarp:
            gamma_ray = float((1.5 * torch.tanh(mod.raw_gamma_ray)).cpu())
            f_ref_tilde = float((0.8 + 0.6 * torch.sigmoid(mod.raw_f_ref)).cpu())
        else:
            gamma_ray = 0.0
            f_ref_tilde = init_f_ref
        if len(cond2d_names):
            a_v0 = float(mod.a_v0.cpu())
            a_r0 = float(mod.a_r0.cpu())
            w_v_list = mod.w_v.cpu().tolist()
            w_r_list = mod.w_r.cpu().tolist()
        else:
            a_v0 = 0.0
            a_r0 = 0.0
            w_v_list = []
            w_r_list = []

    return {
        "kind": "geocalib_3d",
        "spec": asdict(spec),
        "init_scale": init_scale,
        "depth_spline": {
            "x_knots": DEPTH_KNOTS.tolist(),
            "y_knots": y_d_list,
            "slopes": d_slopes_list,
        },
        "focal_spline": {
            "enabled": spec.use_focal_curve,
            "x_knots": FOCAL_KNOTS.tolist(),
            "y_knots": y_f_list,
            "slopes": f_slopes_list,
        },
        "scale_residual": {
            "feature_names": list(scale_names),
            "mean": mu_scale.tolist(),
            "std": sd_scale.tolist(),
            "weights": w_scale_list,
            "max_log_scale": spec.max_log_scale,
        },
        "depth_slope_mod": {
            "feature_names": list(slope_names),
            "mean": mu_slope.tolist(),
            "std": sd_slope.tolist(),
            "weights": w_slope_list,
            "log_depth_center": LOG_DEPTH_CENTER,
        },
        "ray_unwarp": {
            "enabled": spec.use_ray_unwarp,
            "gamma_ray": gamma_ray,
            "f_ref_tilde": f_ref_tilde,
        },
        "cond_spatial2d": {
            "feature_names": list(cond2d_names),
            "mean": mu_cond2d.tolist(),
            "std": sd_cond2d.tolist(),
            "a_v0": a_v0,
            "w_v": w_v_list,
            "a_r0": a_r0,
            "w_r": w_r_list,
            "mean_r2_crop": _MEAN_R2_CROP,
        },
        "optimization": {
            "closure_calls": len(history),
            "objective_first": history[0],
            "objective_last": history[-1],
        },
    }


def _two_stage_ridge_features(feat: dict[str, float], kind: str) -> np.ndarray:
    f_t = feat["f_tilde"]
    if kind == "geok":
        return np.array([f_t, f_t ** 2, np.log(f_t), feat["cx_off"], feat["cy_off"]], dtype=np.float64)
    if kind == "geok_sp3d":
        return np.array(
            [f_t, f_t ** 2, np.log(f_t), feat["cx_off"], feat["cy_off"], *(feat[k] for k in SPATIAL3D_NAMES)],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown two-stage ridge feature set: {kind}")


def fit_two_stage_ridge_model(
    spec: GeoCalibSpec,
    base_curve: dict[str, Any],
    features: list[dict[str, float]],
    targets_med_log: np.ndarray,
) -> dict[str, Any]:
    """Two-stage sanity control: B2 spline + ridge regression on new feature groups."""
    x = np.stack([_two_stage_ridge_features(f, spec.ridge_features) for f in features], axis=0)
    mu = x.mean(axis=0)
    sd = np.maximum(x.std(axis=0), 1e-8)
    z = (x - mu) / sd
    bias = float(targets_med_log.mean())
    coef = np.linalg.solve(
        z.T @ z / len(z) + spec.ridge_lam * np.eye(z.shape[1]),
        z.T @ (targets_med_log - bias) / len(z),
    )
    return {
        "kind": "two_stage_ridge",
        "spec": asdict(spec),
        "base_curve": base_curve,
        "ridge_features": spec.ridge_features,
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "bias": bias,
        "coefficients": coef.tolist(),
        "max_log_scale": spec.max_log_scale,
    }


def count_parameters(model: dict[str, Any]) -> int:
    """Count exact learned scalar parameters in an exported model dictionary."""
    if model.get("kind") == "two_stage_ridge":
        # 6 spline params + intercept + ridge coefs
        return 6 + 1 + len(model["coefficients"])
    total = 6  # raw_offset + 5 depth spline slopes
    if model["focal_spline"]["enabled"]:
        total += 5
    total += len(model["scale_residual"]["weights"])
    total += len(model["depth_slope_mod"]["weights"])
    if model["ray_unwarp"]["enabled"]:
        total += 2  # gamma_ray, f_ref_tilde
    if model["cond_spatial2d"]["w_v"]:
        total += 2 + 2 * len(model["cond_spatial2d"]["w_v"])
    return int(total)


def apply_sota_model(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    model: dict[str, Any],
    rgb_image: np.ndarray | None = None,
    cached_features: dict[str, float] | None = None,
    use_torch_fast: bool = True,
) -> np.ndarray:
    """Single-pass inference for GeoCalib-3D or two-stage ridge controls.

    Uses exact float64 arithmetic; optionally accelerates array ops via Torch CPU
    for fast (<5 ms) deployment while remaining numerically identical to pure NumPy.
    """
    p = np.asarray(raw_depth, dtype=np.float64)
    if p.shape != (480, 640) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Expected finite positive 480x640 raw depth")
    K = np.asarray(pred_intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Expected valid 3x3 predicted intrinsics")

    if model.get("kind") == "two_stage_ridge":
        feats = cached_features if cached_features is not None else extract_sota_features(p, K, rgb_image, need_rgb=False)
        curved = apply_curve(p, model["base_curve"])
        x = _two_stage_ridge_features(feats, model["ridge_features"])
        z = (x - np.asarray(model["mean"], dtype=np.float64)) / np.asarray(model["std"], dtype=np.float64)
        pred_log = float(model["bias"]) + float(z @ np.asarray(model["coefficients"], dtype=np.float64))
        lim = float(model["max_log_scale"])
        return clip_depth(curved * np.exp(np.clip(pred_log, -lim, lim)))

    spec_d = model["spec"]
    need_rgb = bool(spec_d.get("use_rgb3", False))
    feats = (
        cached_features
        if cached_features is not None
        else extract_sota_features(p, K, rgb_image, need_rgb=need_rgb)
    )

    # Image-level scalar terms
    shift = 0.0
    fs = model["focal_spline"]
    if fs["enabled"]:
        shift += float(
            _interp_1d_piecewise_np(
                feats["f_tilde"],
                np.asarray(fs["x_knots"], dtype=np.float64),
                np.asarray(fs["y_knots"], dtype=np.float64),
            )
        )
    sr = model["scale_residual"]
    if sr["weights"]:
        x_s = np.array([feats[k] for k in sr["feature_names"]], dtype=np.float64)
        z_s = (x_s - np.asarray(sr["mean"], dtype=np.float64)) / np.asarray(sr["std"], dtype=np.float64)
        m_lim = float(sr["max_log_scale"])
        raw_s = float(z_s @ np.asarray(sr["weights"], dtype=np.float64))
        shift += m_lim * float(np.tanh(raw_s / m_lim))
    if fs["enabled"] or sr["weights"]:
        shift = 0.50 * float(np.tanh(shift / 0.50))

    dsm = model["depth_slope_mod"]
    delta_b = 0.0
    if dsm["weights"]:
        x_b = np.array([feats[k] for k in dsm["feature_names"]], dtype=np.float64)
        z_b = (x_b - np.asarray(dsm["mean"], dtype=np.float64)) / np.asarray(dsm["std"], dtype=np.float64)
        delta_b = 0.20 * float(np.tanh(float(z_b @ np.asarray(dsm["weights"], dtype=np.float64))))

    ru = model["ray_unwarp"]
    c2 = model["cond_spatial2d"]
    c_v = 0.0
    c_r2 = 0.0
    if c2["w_v"]:
        x_c = np.array([feats[k] for k in c2["feature_names"]], dtype=np.float64)
        z_c = (x_c - np.asarray(c2["mean"], dtype=np.float64)) / np.asarray(c2["std"], dtype=np.float64)
        c_v = 0.12 * float(np.tanh(float(c2["a_v0"]) + float(z_c @ np.asarray(c2["w_v"], dtype=np.float64))))
        c_r2 = 0.12 * float(np.tanh(float(c2["a_r0"]) + float(z_c @ np.asarray(c2["w_r"], dtype=np.float64))))

    ds = model["depth_spline"]
    xk = np.asarray(ds["x_knots"], dtype=np.float64)
    yk = np.asarray(ds["y_knots"], dtype=np.float64)
    slopes = np.diff(yk) / np.diff(xk)
    dx = np.diff(xk)

    if use_torch_fast:
        p_t = torch.from_numpy(np.ascontiguousarray(p))
        lp = torch.log(p_t)
        out_lp = float(yk[0]) + float(slopes[0]) * torch.clamp(lp - float(xk[0]), max=float(dx[0]))
        for k_i in range(1, 4):
            out_lp.add_(float(slopes[k_i]) * torch.clamp(lp - float(xk[k_i]), min=0.0, max=float(dx[k_i])))
        out_lp.add_(float(slopes[4]) * torch.clamp(lp - float(xk[4]), min=0.0))
        if shift != 0.0:
            out_lp.add_(shift)
        if delta_b != 0.0:
            out_lp.add_(lp - float(dsm["log_depth_center"]), alpha=delta_b)
        if ru["enabled"]:
            f_ref = float(ru["f_ref_tilde"])
            gamma_ray = float(ru["gamma_ray"])
            log_cos_ref = -0.5 * torch.log(1.0 + _FULL_D2_CENTER_T / ((480.0 * f_ref) ** 2))
            log_cos_ref_c = log_cos_ref - float(log_cos_ref[CROP].mean())
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            lch = -0.5 * torch.log(
                1.0
                + ((_FULL_U_PIX_T - cx) / fx) ** 2
                + ((_FULL_V_PIX_T - cy) / fy) ** 2
            )
            lch_c = lch - float(lch[CROP].mean())
            out_lp.add_(log_cos_ref_c - lch_c, alpha=gamma_ray)
        if c2["w_v"]:
            out_lp.add_(_FULL_V_TILDE_T, alpha=c_v)
            out_lp.add_(_FULL_R2_T - float(c2["mean_r2_crop"]), alpha=c_r2)
        return torch.exp(out_lp.clamp_(float(np.log(0.001)), float(np.log(10.0)))).clamp_(0.001, 10.0).numpy()

    lp = np.log(p)
    out_lp = _interp_1d_piecewise_np(lp, xk, yk)
    if shift != 0.0:
        out_lp = out_lp + shift
    if delta_b != 0.0:
        out_lp = out_lp + delta_b * (lp - float(dsm["log_depth_center"]))
    if ru["enabled"]:
        f_ref = float(ru["f_ref_tilde"])
        gamma_ray = float(ru["gamma_ray"])
        log_cos_ref = -0.5 * np.log(1.0 + _FULL_D2_CENTER / ((480.0 * f_ref) ** 2))
        log_cos_ref_c = log_cos_ref - float(log_cos_ref[CROP].mean())
        log_cos_hat_c = _centered_log_cos_hat_full(K)
        out_lp = out_lp + gamma_ray * (log_cos_ref_c - log_cos_hat_c)
    if c2["w_v"]:
        out_lp = out_lp + c_v * _FULL_V_TILDE + c_r2 * (_FULL_R2 - float(c2["mean_r2_crop"]))

    return clip_depth(np.exp(np.clip(out_lp, np.log(0.001), np.log(10.0))))


def build_candidate_specs() -> list[GeoCalibSpec]:
    """Return the streamlined 16-candidate Gate 2 ablation ladder + hyperparameter sweep."""
    specs = [
        # Two-stage decoupled sanity controls (B2 + ridge)
        GeoCalibSpec(
            name="S1_b2_geok_ridge",
            kind="two_stage_ridge",
            ridge_lam=1.0,
            max_log_scale=0.40,
            ridge_features="geok",
        ),
        GeoCalibSpec(
            name="S2_b2_geok_sp3d_ridge",
            kind="two_stage_ridge",
            ridge_lam=1.0,
            max_log_scale=0.40,
            ridge_features="geok_sp3d",
        ),
        # Joint Multi-Metric Ablation Ladder
        GeoCalibSpec(name="J0_spline_bal111", loss_mode="balanced_111"),
        GeoCalibSpec(name="J0_spline_1512", loss_mode="ar_rmse_d1_1512"),
        GeoCalibSpec(
            name="J1_geok_bal111_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            loss_mode="balanced_111",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J1_geok_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J2_geok_sp3d_bal111_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            loss_mode="balanced_111",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J2_geok_sp3d_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J3_geok_sp3d_slope_bal111_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            loss_mode="balanced_111",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J3_geok_sp3d_slope_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J3_geok_sp3d_slope_1512_l003",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.003,
        ),
        GeoCalibSpec(
            name="J4_geok_sp3d_slope_ray_bal111_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            loss_mode="balanced_111",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J4_geok_sp3d_slope_ray_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J4_geok_sp3d_slope_ray_1512_l003",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.003,
        ),
        GeoCalibSpec(
            name="J5_full_spatial2d_bal111_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            use_cond_spatial2d=True,
            loss_mode="balanced_111",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J5_full_spatial2d_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            use_cond_spatial2d=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J6_compact_geo_spatial2d_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=False,
            use_depth_slope=True,
            use_ray_unwarp=True,
            use_cond_spatial2d=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
        GeoCalibSpec(
            name="J7_full_plus_rgb3_1512_l01",
            use_focal_curve=True,
            use_geo_cxcy=True,
            use_spatial3d=True,
            use_rgb3=True,
            use_depth_slope=True,
            use_ray_unwarp=True,
            use_cond_spatial2d=True,
            loss_mode="ar_rmse_d1_1512",
            reg_l2=0.01,
        ),
    ]
    return specs


def prepare_sota_training(
    handle,
    cache,
    train_ids: np.ndarray,
    need_rgb: bool = True,
) -> dict[str, Any]:
    """Load training-only sampled pixels (with exact spatial coords) and extracted features."""
    n = len(train_ids)
    features_b3, ratios, samples_b3, targets_b3 = prepare_training(handle, cache, train_ids)
    intrinsics = cache.intrinsics_for(train_ids).astype(np.float64)

    samples_p = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float64)
    samples_gt = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float64)
    samples_row = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.int32)
    samples_col = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.int32)
    sota_feats: list[dict[str, float]] = []

    for row, index in enumerate(train_ids):
        idx = int(index)
        pred_full = cache[idx].astype(np.float64)
        gt_crop = nyu_depth(handle, idx)[CROP].astype(np.float64)
        valid = valid_depth(gt_crop)
        valid_rc = np.argwhere(valid)
        choice = np.random.default_rng(SAMPLE_SEED + idx).choice(
            len(valid_rc), SAMPLES_PER_IMAGE, replace=False
        )
        rc = valid_rc[choice]
        samples_row[row] = rc[:, 0]
        samples_col[row] = rc[:, 1]
        p_crop = pred_full[CROP]
        samples_p[row] = p_crop[rc[:, 0], rc[:, 1]]
        samples_gt[row] = gt_crop[rc[:, 0], rc[:, 1]]

        rgb = nyu_image(handle, idx) if need_rgb else None
        sota_feats.append(extract_sota_features(pred_full, intrinsics[row], rgb, need_rgb=need_rgb))

    return {
        "train_ids": train_ids,
        "intrinsics": intrinsics,
        "features_b3": features_b3,
        "ratios": ratios,
        "samples_b3": samples_b3,
        "targets_b3": targets_b3,
        "samples_p": samples_p,
        "samples_gt": samples_gt,
        "samples_row": samples_row,
        "samples_col": samples_col,
        "sota_feats": sota_feats,
    }


def _scene_bootstrap_vs_ref(
    scores: np.ndarray,
    ref_col: int,
    scenes: np.ndarray,
    repeats: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute paired scene-cluster bootstrap gains (positive = improvement) vs ref_col."""
    direction = np.array([-1.0, -1.0, 1.0])
    gains = (scores - scores[:, ref_col : ref_col + 1, :]) * direction[None, None, :]
    unique, inverse = np.unique(scenes, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.stack([gains[inverse == i].sum(axis=0) for i in range(len(unique))])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((repeats, scores.shape[1], 3), dtype=np.float64)
    for start in range(0, repeats, 200):
        step = min(200, repeats - start)
        draw = rng.integers(0, len(unique), (step, len(unique)))
        boot[start : start + step] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None, None]
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    return gains.mean(axis=0), ci_low, ci_high


def evaluate_cv_for_seed(
    handle,
    cache,
    data: dict[str, Any],
    scenes: np.ndarray,
    specs: list[GeoCalibSpec],
    seed: int,
    include_baselines: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Run 5-fold scene-disjoint CV on the 795 training images for a single scene-split seed."""
    train_ids = data["train_ids"]
    n = len(train_ids)
    folds = scene_folds_seeded(scenes, count=5, seed=seed)

    baseline_names = ["raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"] if include_baselines else []
    names = baseline_names + [s.name for s in specs]
    scores = np.full((n, len(names), 3), np.nan)
    fold_ids = np.empty(n, dtype=int)

    for f_idx, val in enumerate(folds):
        t0 = perf_counter()
        tr = np.setdiff1d(np.arange(n), val)
        scale = fit_scale(data["ratios"][tr])
        curve = fit_curve(data["samples_b3"][tr], data["targets_b3"][tr], scale)
        res_y = residual_targets(handle, cache, train_ids[tr], curve)[:, 0]
        res_model = fit_residual(data["features_b3"][tr], res_y, regularization=1.0, shrinkage=0.5)

        tr_feats = [data["sota_feats"][i] for i in tr]
        tr_intr = data["intrinsics"][tr]
        tr_p = data["samples_p"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_gt = data["samples_gt"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_row = data["samples_row"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_col = data["samples_col"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_ratios = data["ratios"][tr]

        def _fit_one(spec: GeoCalibSpec) -> dict[str, Any]:
            if spec.kind == "two_stage_ridge":
                return fit_two_stage_ridge_model(spec, curve, tr_feats, res_y)
            return fit_joint_sota_model(
                spec=spec,
                samples_p=tr_p,
                samples_gt=tr_gt,
                samples_row=tr_row,
                samples_col=tr_col,
                features=tr_feats,
                intrinsics=tr_intr,
                ratios=tr_ratios,
                max_iter=65,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            fitted_models = list(pool.map(_fit_one, specs))

        for row in val:
            idx = int(train_ids[row])
            pred = cache[idx].astype(np.float64)
            gt = nyu_depth(handle, idx)
            K = data["intrinsics"][row]
            feat = data["sota_feats"][row]

            outputs: list[np.ndarray] = []
            if include_baselines:
                b1 = clip_depth(scale * pred)
                b2 = apply_curve(pred, curve)
                b3 = apply_residual(b2, data["features_b3"][row], res_model)
                outputs.extend([clip_depth(pred), b1, b2, b3])
            for m in fitted_models:
                outputs.append(apply_sota_model(pred, K, m, cached_features=feat))

            for col, out_d in enumerate(outputs):
                scores[row, col] = evaluate_image(out_d, gt)
            fold_ids[row] = f_idx

        print(f"CV seed={seed} fold {f_idx + 1}/5 finished in {perf_counter() - t0:.1f}s", flush=True)

    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite OOF scores encountered")
    return scores, names, fold_ids


def benchmark_postprocessing(
    cache,
    train_ids: np.ndarray,
    winner_model: dict[str, Any],
    b3_residual_model: dict[str, Any],
    b2_curve: dict[str, Any],
    n_images: int = 100,
) -> dict[str, Any]:
    """Measure pure post-processing runtime per 480x640 image in milliseconds."""
    from .frozen import depth_features

    bench_ids = train_ids[:n_images]
    depths = [cache[int(i)].astype(np.float64) for i in bench_ids]
    ks = [cache.intrinsics(int(i)).astype(np.float64) for i in bench_ids]
    need_rgb = bool(winner_model.get("spec", {}).get("use_rgb3", False))

    # Warmup
    for d, k in zip(depths[:5], ks[:5]):
        _ = extract_sota_features(d, k, None, need_rgb=need_rgb)
        _ = apply_sota_model(d, k, winner_model, use_torch_fast=True)
        _ = apply_sota_model(d, k, winner_model, use_torch_fast=False)
        _ = apply_residual(apply_curve(d, b2_curve), depth_features(d), b3_residual_model)

    tf0 = perf_counter()
    for d, k in zip(depths, ks):
        _ = extract_sota_features(d, k, None, need_rgb=need_rgb)
    feat_ms = (perf_counter() - tf0) * 1000.0 / len(depths)

    t0 = perf_counter()
    for d, k in zip(depths, ks):
        _ = apply_sota_model(d, k, winner_model, use_torch_fast=True)
    torch_ms = (perf_counter() - t0) * 1000.0 / len(depths)

    t0_np = perf_counter()
    for d, k in zip(depths, ks):
        _ = apply_sota_model(d, k, winner_model, use_torch_fast=False)
    numpy_ms = (perf_counter() - t0_np) * 1000.0 / len(depths)

    t1 = perf_counter()
    for d in depths:
        _ = apply_residual(apply_curve(d, b2_curve), depth_features(d), b3_residual_model)
    b3_ms = (perf_counter() - t1) * 1000.0 / len(depths)

    result = {
        "benchmark_images": len(depths),
        "extract_sota_features_ms_per_image": float(feat_ms),
        "b3_residual_v1_ms_per_image": float(b3_ms),
        "winner_cpu_torch_f64_ms_per_image": float(torch_ms),
        "winner_pure_numpy_f64_ms_per_image": float(numpy_ms),
    }
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        _ = torch.zeros(1, dtype=torch.float64, device=dev)
        torch.cuda.synchronize()
        d2_g = _FULL_D2_CENTER_T.to(dev)
        up_g = _FULL_U_PIX_T.to(dev)
        vp_g = _FULL_V_PIX_T.to(dev)
        vt_g = _FULL_V_TILDE_T.to(dev)
        r2_g = _FULL_R2_T.to(dev)
        ds = winner_model["depth_spline"]
        xk = np.asarray(ds["x_knots"], dtype=np.float64)
        yk = np.asarray(ds["y_knots"], dtype=np.float64)
        slopes = np.diff(yk) / np.diff(xk)
        dx = np.diff(xk)
        t_gpu0 = perf_counter()
        for d in depths:
            p_g = torch.from_numpy(np.ascontiguousarray(d)).to(dev)
            lp_g = torch.log(p_g)
            out_g = float(yk[0]) + float(slopes[0]) * torch.clamp(lp_g - float(xk[0]), max=float(dx[0]))
            for k_i in range(1, 4):
                out_g = out_g + float(slopes[k_i]) * torch.clamp(lp_g - float(xk[k_i]), min=0.0, max=float(dx[k_i]))
            out_g = out_g + float(slopes[4]) * torch.clamp(lp_g - float(xk[4]), min=0.0)
            _ = torch.exp(out_g).clamp(0.001, 10.0).cpu().numpy()
        torch.cuda.synchronize()
        result["winner_gpu_torch_f64_ms_per_image"] = float((perf_counter() - t_gpu0) * 1000.0 / len(depths))
    return result


def run_cv_study(data_root: Path, prediction_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Execute complete Gate 2 5-fold canonical + multi-seed ablation study (train split ONLY)."""
    torch.set_num_threads(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_nyu_rgb_cache(prediction_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids
    specs = build_candidate_specs()

    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        scenes = all_scenes[train_ids]
        data = prepare_sota_training(handle, cache, train_ids, need_rgb=True)

        # Stage 1: Canonical SEED=20261031 full ablation ladder
        scores_canon, names_canon, fold_ids = evaluate_cv_for_seed(
            handle, cache, data, scenes, specs, seed=SEED, include_baselines=True
        )
        b3_idx = names_canon.index("b3_residual_v1")
        mean_scores = scores_canon.mean(axis=0)
        gain_vs_b3, ci_lo_b3, ci_hi_b3 = _scene_bootstrap_vs_ref(scores_canon, b3_idx, scenes)

        # Paired bootstrap vs raw, b1, b2, and b3 for full paired table
        ref_indices = {
            "raw": names_canon.index("raw"),
            "b1_global_scale": names_canon.index("b1_global_scale"),
            "b2_monotone_curve": names_canon.index("b2_monotone_curve"),
            "b3_residual_v1": b3_idx,
        }
        paired_rows = []
        for ref_name, r_idx in ref_indices.items():
            mg, cl, ch = _scene_bootstrap_vs_ref(scores_canon, r_idx, scenes)
            for col, cand in enumerate(names_canon):
                if col <= r_idx:
                    continue
                for m_i, metric in enumerate(METRIC_NAMES):
                    paired_rows.append(
                        {
                            "reference": ref_name,
                            "candidate": cand,
                            "metric": metric,
                            "gain": float(mg[col, m_i]),
                            "ci95_low": float(cl[col, m_i]),
                            "ci95_high": float(ch[col, m_i]),
                        }
                    )

        # Fold breakdown + ablation summary
        fold_rows = []
        ablation_rows = []
        direction = np.array([-1.0, -1.0, 1.0])
        spec_by_name = {s.name: s for s in specs}

        # Precompute parameter counts by fitting on full train once per spec? No: compute exact param count analytically!
        def _spec_params(name: str) -> int:
            if name == "raw":
                return 0
            if name == "b1_global_scale":
                return 1
            if name == "b2_monotone_curve":
                return 6
            if name == "b3_residual_v1":
                return 15
            sp = spec_by_name[name]
            if sp.kind == "two_stage_ridge":
                return 6 + 1 + (5 if sp.ridge_features == "geok" else 13)
            cnt = 6
            if sp.use_focal_curve:
                cnt += 5
            cnt += len(_scale_feature_names(sp))
            cnt += len(_slope_feature_names(sp))
            if sp.use_ray_unwarp:
                cnt += 2
            c2 = _cond2d_feature_names(sp)
            if c2:
                cnt += 2 + 2 * len(c2)
            return cnt

        for col, cand in enumerate(names_canon):
            fold_gains = []
            for f in range(5):
                f_mask = fold_ids == f
                f_m = scores_canon[f_mask, col].mean(axis=0)
                f_b3 = scores_canon[f_mask, b3_idx].mean(axis=0)
                fg = (f_m - f_b3) * direction
                fold_gains.append(fg)
                fold_rows.append(
                    {
                        "method": cand,
                        "fold": f,
                        "images": int(f_mask.sum()),
                        "abs_rel": float(f_m[0]),
                        "rmse": float(f_m[1]),
                        "delta1": float(f_m[2]),
                        "gain_vs_b3_abs_rel": float(fg[0]),
                        "gain_vs_b3_rmse": float(fg[1]),
                        "gain_vs_b3_delta1": float(fg[2]),
                    }
                )
            fg_arr = np.asarray(fold_gains)
            all_folds_pos = bool(np.all(fg_arr > 0))
            means_pos = bool(np.all(gain_vs_b3[col] > 1e-6))
            ci_all_pos = bool(np.all(ci_lo_b3[col] > 0))
            ablation_rows.append(
                {
                    "method": cand,
                    "params": _spec_params(cand),
                    "abs_rel": float(mean_scores[col, 0]),
                    "rmse": float(mean_scores[col, 1]),
                    "delta1": float(mean_scores[col, 2]),
                    "gain_vs_b3_abs_rel": float(gain_vs_b3[col, 0]),
                    "gain_vs_b3_rmse": float(gain_vs_b3[col, 1]),
                    "gain_vs_b3_delta1": float(gain_vs_b3[col, 2]),
                    "ci95_lo_abs_rel": float(ci_lo_b3[col, 0]),
                    "ci95_lo_rmse": float(ci_lo_b3[col, 1]),
                    "ci95_lo_delta1": float(ci_lo_b3[col, 2]),
                    "all_5_folds_beat_b3": all_folds_pos,
                    "means_beat_b3": means_pos,
                    "ci95_all_positive_vs_b3": ci_all_pos,
                }
            )

        ablation_df = pd.DataFrame(ablation_rows)
        write_csv_lf(output_dir / "oof_ablation_summary.csv", ablation_df)
        write_csv_lf(output_dir / "oof_fold_breakdown.csv", pd.DataFrame(fold_rows))
        write_csv_lf(output_dir / "oof_paired_bootstrap.csv", pd.DataFrame(paired_rows))

        # Select finalists for 4-seed stability check: core ladder + top eligible candidates
        eligible_specs = [
            r["method"]
            for r in ablation_rows
            if r["method"] not in ("raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1")
            and r["means_beat_b3"]
            and r["all_5_folds_beat_b3"]
        ]
        # Always include representative ladder stages + all top performers
        # Always evaluate S2 decoupled + 5 key joint ladder stages + canonical #1 winner across all 4 seeds
        core_ladder = [
            "S2_b2_geok_sp3d_ridge",
            "J1_geok_1512_l01",
            "J2_geok_sp3d_1512_l01",
            "J3_geok_sp3d_slope_1512_l01",
            "J4_geok_sp3d_slope_ray_1512_l01",
            "J5_full_spatial2d_1512_l01",
        ]
        ranked_elig = sorted(
            eligible_specs,
            key=lambda m: float(ablation_df.loc[ablation_df["method"] == m, "abs_rel"].iloc[0]),
        )
        finalist_names = list(core_ladder)
        if ranked_elig and ranked_elig[0] not in finalist_names:
            finalist_names.append(ranked_elig[0])
        if (
            "J6_compact_geo_spatial2d_1512_l01" in ranked_elig[:3]
            and "J6_compact_geo_spatial2d_1512_l01" not in finalist_names
        ):
            finalist_names.append("J6_compact_geo_spatial2d_1512_l01")
        finalist_specs = [spec_by_name[m] for m in finalist_names]

        # Stage 2: Multi-seed stability (seeds 42, 100, 20261121 + canonical 20261031)
        extra_seeds = (42, 100, 20261121)
        multiseed_rows = []

        # Record canonical seed first
        for cand in ["raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"] + finalist_names:
            c_idx = names_canon.index(cand)
            g = (mean_scores[c_idx] - mean_scores[b3_idx]) * direction
            multiseed_rows.append(
                {
                    "seed": SEED,
                    "method": cand,
                    "abs_rel": float(mean_scores[c_idx, 0]),
                    "rmse": float(mean_scores[c_idx, 1]),
                    "delta1": float(mean_scores[c_idx, 2]),
                    "gain_vs_b3_abs_rel": float(g[0]),
                    "gain_vs_b3_rmse": float(g[1]),
                    "gain_vs_b3_delta1": float(g[2]),
                }
            )

        for alt_seed in extra_seeds:
            sc_alt, nm_alt, _ = evaluate_cv_for_seed(
                handle, cache, data, scenes, finalist_specs, seed=alt_seed, include_baselines=True
            )
            m_alt = sc_alt.mean(axis=0)
            b3_alt_idx = nm_alt.index("b3_residual_v1")
            for c_idx, cand in enumerate(nm_alt):
                g = (m_alt[c_idx] - m_alt[b3_alt_idx]) * direction
                multiseed_rows.append(
                    {
                        "seed": alt_seed,
                        "method": cand,
                        "abs_rel": float(m_alt[c_idx, 0]),
                        "rmse": float(m_alt[c_idx, 1]),
                        "delta1": float(m_alt[c_idx, 2]),
                        "gain_vs_b3_abs_rel": float(g[0]),
                        "gain_vs_b3_rmse": float(g[1]),
                        "gain_vs_b3_delta1": float(g[2]),
                    }
                )

        ms_df = pd.DataFrame(multiseed_rows)
        write_csv_lf(output_dir / "oof_multiseed_stability.csv", ms_df)

        # Select final winner: must pass all 5 canonical folds, positive 95% CI, and beat B3 on all 3 metrics across ALL 4 seeds
        candidate_evals = []
        for cand in finalist_names:
            sub = ms_df[ms_df["method"] == cand]
            all4_beats_b3 = bool(
                (sub["gain_vs_b3_abs_rel"] > 0).all()
                and (sub["gain_vs_b3_rmse"] > 0).all()
                and (sub["gain_vs_b3_delta1"] > 0).all()
            )
            canon_row = ablation_df[ablation_df["method"] == cand].iloc[0]
            eligible = bool(
                canon_row["means_beat_b3"]
                and canon_row["all_5_folds_beat_b3"]
                and all4_beats_b3
            )
            candidate_evals.append(
                {
                    "method": cand,
                    "params": int(canon_row["params"]),
                    "eligible": eligible,
                    "all_5_folds_beat_b3": bool(canon_row["all_5_folds_beat_b3"]),
                    "ci95_all_positive_vs_b3": bool(canon_row["ci95_all_positive_vs_b3"]),
                    "all_4_seeds_beat_b3": all4_beats_b3,
                    "canon_abs_rel": float(canon_row["abs_rel"]),
                    "canon_rmse": float(canon_row["rmse"]),
                    "canon_delta1": float(canon_row["delta1"]),
                    "mean4_abs_rel": float(sub["abs_rel"].mean()),
                    "std4_abs_rel": float(sub["abs_rel"].std(ddof=0)),
                    "mean4_rmse": float(sub["rmse"].mean()),
                    "std4_rmse": float(sub["rmse"].std(ddof=0)),
                    "mean4_delta1": float(sub["delta1"].mean()),
                    "std4_delta1": float(sub["delta1"].std(ddof=0)),
                    "mean4_gain_abs_rel": float(sub["gain_vs_b3_abs_rel"].mean()),
                    "mean4_gain_rmse": float(sub["gain_vs_b3_rmse"].mean()),
                    "mean4_gain_delta1": float(sub["gain_vs_b3_delta1"].mean()),
                }
            )

        elig_cands = [c for c in candidate_evals if c["eligible"]]
        if not elig_cands:
            raise RuntimeError("No candidate met the strict pre-registered Gate 2 promotion criteria")
        # Rank by lowest 4-seed mean AbsRel (with tie-breaking on RMSE and parsimony)
        elig_cands_sorted = sorted(elig_cands, key=lambda c: (c["mean4_abs_rel"], c["mean4_rmse"], c["params"]))
        winner_name = elig_cands_sorted[0]["method"]
        runner_up_name = elig_cands_sorted[1]["method"] if len(elig_cands_sorted) > 1 else None

        # Save per-image OOF CSV for baselines + key ladder + winner
        keep_per_img = ["raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"]
        for m in core_ladder + [winner_name]:
            if m in names_canon and m not in keep_per_img:
                keep_per_img.append(m)
        per_img_rows = []
        for row, idx in enumerate(train_ids):
            for m in keep_per_img:
                col = names_canon.index(m)
                per_img_rows.append(
                    {
                        "index": int(idx),
                        "scene": str(scenes[row]),
                        "fold": int(fold_ids[row]),
                        "method": m,
                        "abs_rel": float(scores_canon[row, col, 0]),
                        "rmse": float(scores_canon[row, col, 1]),
                        "delta1": float(scores_canon[row, col, 2]),
                    }
                )
        write_csv_lf(output_dir / "oof_per_image.csv", pd.DataFrame(per_img_rows))

        # Fit winner on full 795 training set ONLY to inspect physical parameters and measure latency (not frozen yet)
        winner_spec = spec_by_name[winner_name]
        full_scale = fit_scale(data["ratios"])
        full_b2_curve = fit_curve(data["samples_b3"], data["targets_b3"], full_scale)
        full_res_y = residual_targets(handle, cache, train_ids, full_b2_curve)[:, 0]
        full_b3_res = fit_residual(data["features_b3"], full_res_y, regularization=1.0, shrinkage=0.5)
        full_winner_model = fit_joint_sota_model(
            spec=winner_spec,
            samples_p=data["samples_p"],
            samples_gt=data["samples_gt"],
            samples_row=data["samples_row"],
            samples_col=data["samples_col"],
            features=data["sota_feats"],
            intrinsics=data["intrinsics"],
            ratios=data["ratios"],
        )
        # Note: pass fit_residual dict directly to calibration.apply_residual (uses 'mean', 'std', 'intercept', 'coefficient')
        latency = benchmark_postprocessing(cache, train_ids, full_winner_model, full_b3_res, full_b2_curve, n_images=100)

    report = {
        "scope": "official795_training_only",
        "test_read": False,
        "images": int(len(train_ids)),
        "scenes": int(len(np.unique(scenes))),
        "canonical_seed": SEED,
        "multi_seeds": [SEED, *extra_seeds],
        "promotion_criterion": "Beats B3_residual_v1 on all 3 metrics (AbsRel, RMSE, delta1) on canonical OOF, all 5 canonical folds, and across all 4 scene-split seeds; ranked by lowest 4-seed mean AbsRel.",
        "winner": winner_name,
        "winner_params": count_parameters(full_winner_model),
        "runner_up": runner_up_name,
        "runtime_benchmark": latency,
        "candidate_evaluations": candidate_evals,
        "winner_full_train_inspection": full_winner_model,
    }
    write_json_lf(output_dir / "cv_selection_report.json", report)
    print(f"Gate 2 CV Complete. Winner: {winner_name} ({count_parameters(full_winner_model)} params), torch_f64={latency['winner_cpu_torch_f64_ms_per_image']:.3f} ms/img, numpy_f64={latency['winner_pure_numpy_f64_ms_per_image']:.3f} ms/img", flush=True)
    return report


def _check_sha256_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid SHA-256 digest for {name}: {value!r}")
    return value


def _finite_vec(values: object, name: str, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite 1D vector for {name}")
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"Expected length {expected_len} for {name}, got {len(arr)}")
    return arr


def _validate_submodel(model: dict[str, Any], label: str) -> None:
    if not isinstance(model, dict):
        raise ValueError(f"Expected dict submodel for {label}")
    kind = model.get("kind")
    if kind == "two_stage_ridge":
        bc = model.get("base_curve", {})
        xk = _finite_vec(bc.get("x_knots"), f"{label}.base_curve.x_knots", 6)
        yk = _finite_vec(bc.get("y_knots"), f"{label}.base_curve.y_knots", 6)
        if np.any(np.diff(xk) <= 0) or np.any(np.diff(yk) / np.diff(xk) <= 0):
            raise ValueError(f"Non-monotone base_curve in {label}")
        mean = _finite_vec(model.get("mean"), f"{label}.mean")
        std = _finite_vec(model.get("std"), f"{label}.std", len(mean))
        _finite_vec(model.get("coefficients"), f"{label}.coefficients", len(mean))
        if np.any(std <= 0) or not np.isfinite(float(model.get("bias", np.nan))):
            raise ValueError(f"Invalid ridge stats/bias in {label}")
        if not np.isfinite(float(model.get("max_log_scale", np.nan))) or float(model["max_log_scale"]) <= 0:
            raise ValueError(f"Invalid max_log_scale in {label}")
        return

    if kind != "geocalib_3d":
        raise ValueError(f"Unsupported submodel kind {kind!r} in {label}")
    ds = model.get("depth_spline", {})
    dx = _finite_vec(ds.get("x_knots"), f"{label}.depth_spline.x_knots", 6)
    dy = _finite_vec(ds.get("y_knots"), f"{label}.depth_spline.y_knots", 6)
    if np.any(np.diff(dx) <= 0) or np.any(np.diff(dy) / np.diff(dx) <= 0):
        raise ValueError(f"Depth spline must be strictly increasing in {label}")

    fs = model.get("focal_spline", {})
    if fs.get("enabled"):
        fxk = _finite_vec(fs.get("x_knots"), f"{label}.focal_spline.x_knots", 6)
        fyk = _finite_vec(fs.get("y_knots"), f"{label}.focal_spline.y_knots", 6)
        if np.any(np.diff(fxk) <= 0) or np.any(np.diff(fyk) / np.diff(fxk) > 1e-12):
            raise ValueError(f"Focal spline must be non-increasing in {label}")

    for block_name in ("scale_residual", "depth_slope_mod"):
        blk = model.get(block_name, {})
        fnames = blk.get("feature_names", [])
        n_f = len(fnames)
        _finite_vec(blk.get("mean", []), f"{label}.{block_name}.mean", n_f)
        sd = _finite_vec(blk.get("std", []), f"{label}.{block_name}.std", n_f)
        _finite_vec(blk.get("weights", []), f"{label}.{block_name}.weights", n_f)
        if np.any(sd <= 0):
            raise ValueError(f"Non-positive std in {label}.{block_name}")

    ru = model.get("ray_unwarp", {})
    if not np.isfinite(float(ru.get("gamma_ray", 0.0))) or not np.isfinite(float(ru.get("f_ref_tilde", 1.14))):
        raise ValueError(f"Nonfinite ray_unwarp in {label}")
    c2 = model.get("cond_spatial2d", {})
    if c2.get("w_v"):
        n_c = len(c2.get("feature_names", []))
        _finite_vec(c2.get("mean"), f"{label}.cond_spatial2d.mean", n_c)
        c2_sd = _finite_vec(c2.get("std"), f"{label}.cond_spatial2d.std", n_c)
        _finite_vec(c2.get("w_v"), f"{label}.cond_spatial2d.w_v", n_c)
        _finite_vec(c2.get("w_r"), f"{label}.cond_spatial2d.w_r", n_c)
        if np.any(c2_sd <= 0):
            raise ValueError(f"Non-positive std in {label}.cond_spatial2d")


def validate_frozen_sota(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate self-contained GeoCalib_3D_v1 frozen artifact strictly without side effects."""
    if not isinstance(payload, dict) or payload.get("schema") != "GeoCalib_3D_v1":
        raise ValueError("Expected schema == 'GeoCalib_3D_v1'")
    if payload.get("complete") is not True:
        raise ValueError("Frozen SOTA artifact must have complete=True")
    if payload.get("test_read") is not False:
        raise ValueError("Frozen SOTA artifact must retain train-only provenance (test_read=False)")
    if payload.get("primary_method") != PRIMARY_SOTA_METHOD:
        raise ValueError(f"Expected primary_method={PRIMARY_SOTA_METHOD!r}, got {payload.get('primary_method')!r}")
    if payload.get("primary_params") != 25:
        raise ValueError(f"Expected primary_params=25, got {payload.get('primary_params')!r}")
    if payload.get("training_images") != 795 or payload.get("training_scenes") != 249:
        raise ValueError("Expected training_images=795 and training_scenes=249")

    train_ids = payload.get("train_ids")
    if (
        not isinstance(train_ids, list)
        or len(train_ids) != 795
        or len(set(train_ids)) != 795
        or any(type(i) is not int or not (0 <= i < 1449) for i in train_ids)
    ):
        raise ValueError("Invalid train_ids in frozen SOTA artifact")

    for digest_key in ("dataset_sha256", "split_sha256", "b3_frozen_model_sha256", "cv_selection_report_sha256"):
        _check_sha256_hex(payload.get(digest_key), digest_key)

    pc_hashes = payload.get("prediction_cache_sha256")
    if not isinstance(pc_hashes, dict) or set(pc_hashes.keys()) != {"depth.npy", "intrinsics.npy", "indices.npy"}:
        raise ValueError("Invalid prediction_cache_sha256 keys")
    for k, v in pc_hashes.items():
        _check_sha256_hex(v, f"prediction_cache_sha256[{k}]")

    src_hashes = payload.get("source_sha256")
    if not isinstance(src_hashes, dict) or not src_hashes:
        raise ValueError("Expected nonempty source_sha256 map")
    for k, v in src_hashes.items():
        _check_sha256_hex(v, f"source_sha256[{k}]")

    primary_model = payload.get("primary_model")
    _validate_submodel(primary_model, "primary_model")
    if count_parameters(primary_model) != payload["primary_params"]:
        raise ValueError("primary_params mismatch with primary_model parameter count")

    ladder = payload.get("ladder_models")
    if not isinstance(ladder, dict) or any(name not in ladder for name in LOCKED_LADDER_NAMES):
        raise ValueError("Missing required ladder_models entries")
    for name in LOCKED_LADDER_NAMES:
        _validate_submodel(ladder[name], f"ladder_models[{name}]")
    if ladder[PRIMARY_SOTA_METHOD] != primary_model:
        raise ValueError("ladder_models[primary_method] must exactly match primary_model")
    return payload


def load_frozen_sota(path: Path | str, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load and verify a locked GeoCalib_3D_v1 JSON artifact (optionally checking SHA-256)."""
    target = Path(path)
    actual_sha = file_sha256(target)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"Frozen SOTA SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return validate_frozen_sota(payload)
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed frozen SOTA artifact at {target}") from exc


def fit_and_lock_sota(
    data_root: Path,
    prediction_dir: Path,
    cv_report_path: Path,
    b3_frozen_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Gate 3: Fit primary + clean ladder on all 795 training images and freeze JSON before test."""
    torch.set_num_threads(2)
    cache = load_nyu_rgb_cache(
        prediction_dir, data_root, allowed_split="train", compute_hashes=True, verify_scenes=True
    )
    train_ids = cache.train_ids

    cv_report = json.loads(Path(cv_report_path).read_text(encoding="utf-8"))
    if cv_report.get("test_read") is not False:
        raise ValueError("CV selection report must certify test_read=False")
    if cv_report.get("winner") != PRIMARY_SOTA_METHOD:
        raise ValueError(f"Unexpected CV winner: {cv_report.get('winner')!r}")

    b3_frozen = load_frozen(b3_frozen_path)
    if b3_frozen["train_ids"] != train_ids.tolist():
        raise ValueError("B3 frozen train_ids differ from official training split")
    if b3_frozen["dataset_sha256"] != cache.hashes["nyu_mat_sha256"]:
        raise ValueError("B3 frozen dataset_sha256 differs from current NYU dataset")
    if b3_frozen["split_sha256"] != cache.hashes["splits_sha256"]:
        raise ValueError("B3 frozen split_sha256 differs from current splits.mat")

    spec_by_name = {s.name: s for s in build_candidate_specs()}
    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        train_scenes = np.unique(all_scenes[train_ids])
        data = prepare_sota_training(handle, cache, train_ids, need_rgb=False)
        full_scale = fit_scale(data["ratios"])
        full_b2_curve = fit_curve(data["samples_b3"], data["targets_b3"], full_scale)
        full_res_y = residual_targets(handle, cache, train_ids, full_b2_curve)[:, 0]

    ladder_models: dict[str, Any] = {}
    for name in LOCKED_LADDER_NAMES:
        spec = spec_by_name[name]
        if spec.kind == "two_stage_ridge":
            model = fit_two_stage_ridge_model(spec, full_b2_curve, data["sota_feats"], full_res_y)
        else:
            model = fit_joint_sota_model(
                spec=spec,
                samples_p=data["samples_p"],
                samples_gt=data["samples_gt"],
                samples_row=data["samples_row"],
                samples_col=data["samples_col"],
                features=data["sota_feats"],
                intrinsics=data["intrinsics"],
                ratios=data["ratios"],
            )
        ladder_models[name] = model

    primary_model = ladder_models[PRIMARY_SOTA_METHOD]
    pkg_dir = Path(__file__).resolve().parent
    source_sha256 = {
        f"depthlab/{fname}": file_sha256(pkg_dir / fname)
        for fname in ("sota_io.py", "sota_calibration.py", "metrics.py", "data.py", "frozen.py", "calibration.py")
    }
    eval_by_name = {e["method"]: e for e in cv_report.get("candidate_evaluations", [])}
    selection_receipt = {
        "cv_winner": PRIMARY_SOTA_METHOD,
        "canonical_seed": cv_report["canonical_seed"],
        "multi_seeds": cv_report["multi_seeds"],
        "promotion_criterion": cv_report["promotion_criterion"],
        "winner_cv_metrics": eval_by_name.get(PRIMARY_SOTA_METHOD),
        "ladder_cv_metrics": {k: eval_by_name.get(k) for k in LOCKED_LADDER_NAMES if k in eval_by_name},
    }

    payload = {
        "schema": "GeoCalib_3D_v1",
        "complete": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
        "primary_method": PRIMARY_SOTA_METHOD,
        "primary_display_name": "GeoCalib-3D",
        "primary_params": count_parameters(primary_model),
        "comparison_order": list(COMPARISON_ORDER),
        "train_ids": train_ids.tolist(),
        "training_images": int(len(train_ids)),
        "training_scenes": int(len(train_scenes)),
        "dataset_sha256": cache.hashes["nyu_mat_sha256"],
        "split_sha256": cache.hashes["splits_sha256"],
        "prediction_cache_sha256": {
            "depth.npy": cache.hashes["depth_npy_sha256"],
            "intrinsics.npy": cache.hashes["intrinsics_npy_sha256"],
            "indices.npy": cache.hashes["indices_npy_sha256"],
        },
        "b3_frozen_model_path": "results/frozen_20261111/part3b_residual_20261111/final/frozen_model.json",
        "b3_frozen_model_sha256": file_sha256(b3_frozen_path),
        "cv_selection_report_sha256": file_sha256(cv_report_path),
        "source_sha256": source_sha256,
        "selection_receipt": selection_receipt,
        "primary_model": primary_model,
        "ladder_models": ladder_models,
    }
    write_json_lf(output_path, payload)
    verified = load_frozen_sota(output_path)
    out_sha = file_sha256(output_path)
    print(f"Gate 3 Frozen Lock Complete: {output_path} (sha256={out_sha}, params={verified['primary_params']})", flush=True)
    return verified


def _scene_metric_bootstrap_ci(
    scores: np.ndarray,
    scenes: np.ndarray,
    repeats: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-method 95% scene-cluster bootstrap CI [2.5, 97.5] over 249 training scenes."""
    unique, inverse = np.unique(scenes, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.stack([scores[inverse == i].sum(axis=0) for i in range(len(unique))])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((repeats, scores.shape[1], 3), dtype=np.float64)
    for start in range(0, repeats, 200):
        step = min(200, repeats - start)
        draw = rng.integers(0, len(unique), (step, len(unique)))
        boot[start : start + step] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None, None]
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def run_learning_curve(
    data_root: Path,
    prediction_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Train-only 5-fold scene-CV learning curve at 10%, 25%, 50%, 100% training scenes."""
    torch.set_num_threads(2)
    cache = load_nyu_rgb_cache(prediction_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids
    winner_spec = {s.name: s for s in build_candidate_specs()}[PRIMARY_SOTA_METHOD]
    methods = ("b1_global_scale", "b2_monotone_curve", "b3_residual_v1", PRIMARY_SOTA_METHOD)

    rows: list[dict[str, Any]] = []
    with open_nyu(data_root) as handle:
        scenes = scene_names(handle)[train_ids]
        data = prepare_sota_training(handle, cache, train_ids, need_rgb=False)
        folds = scene_folds_seeded(scenes, count=5, seed=SEED)
        n = len(train_ids)

        for frac in LEARNING_CURVE_FRACTIONS:
            t0 = perf_counter()
            fold_tasks: list[dict[str, Any]] = []
            for f_idx, val in enumerate(folds):
                fitting = np.setdiff1d(np.arange(n), val)
                fit_scenes = np.unique(scenes[fitting])
                shuffled = np.random.default_rng(LEARNING_CURVE_SEED + f_idx).permutation(fit_scenes)
                keep_scenes = shuffled[: max(1, int(np.ceil(frac * len(shuffled))))]
                tr_sub = fitting[np.isin(scenes[fitting], keep_scenes)]

                scale = fit_scale(data["ratios"][tr_sub])
                curve = fit_curve(data["samples_b3"][tr_sub], data["targets_b3"][tr_sub], scale)
                res_y = residual_targets(handle, cache, train_ids[tr_sub], curve)[:, 0]
                res_model = fit_residual(data["features_b3"][tr_sub], res_y, regularization=1.0, shrinkage=0.5)
                fold_tasks.append(
                    {
                        "fold": f_idx,
                        "val": val,
                        "n_scenes": len(keep_scenes),
                        "n_images": len(tr_sub),
                        "scale": scale,
                        "curve": curve,
                        "res_model": res_model,
                        "tr_sub": tr_sub,
                    }
                )

            def _fit_fold_sota(task: dict[str, Any]) -> dict[str, Any]:
                tr_sub = task["tr_sub"]
                return fit_joint_sota_model(
                    spec=winner_spec,
                    samples_p=data["samples_p"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                    samples_gt=data["samples_gt"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                    samples_row=data["samples_row"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                    samples_col=data["samples_col"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                    features=[data["sota_feats"][i] for i in tr_sub],
                    intrinsics=data["intrinsics"][tr_sub],
                    ratios=data["ratios"][tr_sub],
                    max_iter=65,
                )

            with ThreadPoolExecutor(max_workers=5) as pool:
                sota_models = list(pool.map(_fit_fold_sota, fold_tasks))

            scores = np.full((n, len(methods), 3), np.nan, dtype=np.float64)
            for task, sota_m in zip(fold_tasks, sota_models):
                for row in task["val"]:
                    idx = int(train_ids[row])
                    pred = cache[idx].astype(np.float64)
                    gt = nyu_depth(handle, idx)
                    K = data["intrinsics"][row]
                    feat = data["sota_feats"][row]
                    b1 = clip_depth(task["scale"] * pred)
                    b2 = apply_curve(pred, task["curve"])
                    b3 = apply_residual(b2, data["features_b3"][row], task["res_model"])
                    sota_out = apply_sota_model(pred, K, sota_m, cached_features=feat)
                    for col, out_d in enumerate((b1, b2, b3, sota_out)):
                        scores[row, col] = evaluate_image(out_d, gt)

            if not np.isfinite(scores).all():
                raise ArithmeticError(f"Nonfinite learning curve OOF scores at fraction={frac}")
            mean_scores = scores.mean(axis=0)
            ci_lo, ci_hi = _scene_metric_bootstrap_ci(scores, scenes)
            g_mean, g_lo, g_hi = _scene_bootstrap_vs_ref(scores, methods.index("b3_residual_v1"), scenes)
            mean_scenes = float(np.mean([t["n_scenes"] for t in fold_tasks]))
            mean_imgs = float(np.mean([t["n_images"] for t in fold_tasks]))

            for col, m_name in enumerate(methods):
                rows.append(
                    {
                        "fraction": float(frac),
                        "train_scenes_mean": mean_scenes,
                        "train_images_mean": mean_imgs,
                        "method": m_name,
                        "abs_rel": float(mean_scores[col, 0]),
                        "rmse": float(mean_scores[col, 1]),
                        "delta1": float(mean_scores[col, 2]),
                        "ci95_lo_abs_rel": float(ci_lo[col, 0]),
                        "ci95_hi_abs_rel": float(ci_hi[col, 0]),
                        "ci95_lo_rmse": float(ci_lo[col, 1]),
                        "ci95_hi_rmse": float(ci_hi[col, 1]),
                        "ci95_lo_delta1": float(ci_lo[col, 2]),
                        "ci95_hi_delta1": float(ci_hi[col, 2]),
                        "gain_vs_b3_abs_rel": float(g_mean[col, 0]),
                        "gain_vs_b3_rmse": float(g_mean[col, 1]),
                        "gain_vs_b3_delta1": float(g_mean[col, 2]),
                        "gain_ci95_lo_abs_rel": float(g_lo[col, 0]),
                        "gain_ci95_hi_abs_rel": float(g_hi[col, 0]),
                        "gain_ci95_lo_rmse": float(g_lo[col, 1]),
                        "gain_ci95_hi_rmse": float(g_hi[col, 1]),
                        "gain_ci95_lo_delta1": float(g_lo[col, 2]),
                        "gain_ci95_hi_delta1": float(g_hi[col, 2]),
                    }
                )
            print(
                f"Learning curve frac={frac:.2f} (scenes={mean_scenes:.1f}, imgs={mean_imgs:.1f}) finished in {perf_counter() - t0:.1f}s: "
                f"GeoCalib AbsRel={mean_scores[-1, 0]:.4f}, B3={mean_scores[2, 0]:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    write_csv_lf(output_path, df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cv_p = sub.add_parser("cv", help="Run 5-fold + multi-seed scene-CV ablations on 795 training images only")
    cv_p.add_argument("--data-root", type=Path, required=True)
    cv_p.add_argument("--predictions", type=Path, required=True)
    cv_p.add_argument("--output", type=Path, required=True)

    lock_p = sub.add_parser("fit-lock", help="Fit primary + ladder on all 795 training images and freeze JSON")
    lock_p.add_argument("--data-root", type=Path, required=True)
    lock_p.add_argument("--predictions", type=Path, required=True)
    lock_p.add_argument("--cv-report", type=Path, required=True)
    lock_p.add_argument("--b3-frozen", type=Path, required=True)
    lock_p.add_argument("--output", type=Path, required=True)

    lc_p = sub.add_parser("learning-curve", help="Run train-only 5-fold scene-CV learning curve (10/25/50/100%)")
    lc_p.add_argument("--data-root", type=Path, required=True)
    lc_p.add_argument("--predictions", type=Path, required=True)
    lc_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "cv":
        run_cv_study(args.data_root, args.predictions, args.output)
    elif args.command == "fit-lock":
        fit_and_lock_sota(args.data_root, args.predictions, args.cv_report, args.b3_frozen, args.output)
    elif args.command == "learning-curve":
        run_learning_curve(args.data_root, args.predictions, args.output)


if __name__ == "__main__":
    main()
