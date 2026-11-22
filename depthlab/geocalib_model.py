"""GeoCalib-3D: Joint geometric, inverse-depth, and internal semantic monocular depth calibration.

All feature scaling, SVD PCA bases, spline knots, regularization, and model parameters
are fit strictly on the 795 NYU Depth V2 training images (249 scenes) with zero
ground-truth camera intrinsics K_true.
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
import torch.nn.functional as F

from .data import nyu_depth, nyu_image, open_nyu, scene_names
from .geocalib_cache import CONF_SUMMARY_NAMES, VerifiedGeoCalibCache, load_geocalib_cache
from .geocalib_diagnose import CROP_R2, CROP_SUB, CROP_U_TILDE, CROP_V_TILDE, compute_flip_features
from .geocalib_io import file_sha256, scene_folds_seeded, write_csv_lf, write_json_lf
from .metrics import CROP, METRIC_NAMES, clip_depth, evaluate_image, valid_depth


SEED = 20261031
SAMPLE_SEED = 20261105
BOOTSTRAP_SEED = 20261111
MULTI_SEEDS = (20261031, 42, 100, 20261121)

DEPTH_KNOTS = np.log(np.array([0.3, 0.7, 1.5, 3.0, 6.0, 12.0], dtype=np.float64))
INV_DEPTH_KNOTS = np.array([1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 1.5, 1.0 / 0.7, 1.0 / 0.3], dtype=np.float64)
FOCAL_KNOTS = np.array([0.35, 0.75, 0.98, 1.14, 1.22, 1.35], dtype=np.float64)
FOCAL_ANCHOR_IDX = 3
FOCAL_KNOTS_Q7 = np.array([0.35, 0.58, 0.78, 0.98, 1.14, 1.22, 1.35], dtype=np.float64)
FOCAL_Q7_ANCHOR_IDX = 4
LOG_DEPTH_CENTER = 1.0
SAMPLES_PER_IMAGE = 4096
CV_SAMPLES_PER_IMAGE = 1024
LEARNING_CURVE_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
LEARNING_CURVE_SEED = 20261111

PRIMARY_GEO_METHOD = "geo_focal_3d"
PRIMARY_SINGLE_PASS_METHOD = "geocalib_3d"
PRIMARY_MULTIPASS_METHOD = "geocalib_3d_flip"
LOCKED_EVAL_ORDER = (
    "raw",
    "global_scale",
    "monotone_spline",
    "geo_focal_3d",
    "geo_qknots_robust",
    "geo_inv_residual",
    "geosem_scale",
    "geocalib_3d",
    "flip_geo_only",
    "flip_geo_gated",
    "geocalib_3d_flip",
)

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

GEO_CLEAN_NAMES = ("cy_off", *SPATIAL3D_NAMES)
GEO_EXT_NAMES = (*GEO_CLEAN_NAMES, "frac_far_4m", "left_hinge_x_log_spany", "left_hinge_x_frac_far")
GEO_CONF_NAMES = (*GEO_CXCY_NAMES, *SPATIAL3D_NAMES, *CONF_SUMMARY_NAMES)
FLIP_CONSISTENCY_NAMES = (
    "f_abs_diff",
    "f_tilde_min",
    "depth_flip_log_std",
    "conf_flip_abs_mean",
    "cls_flip_cos_dist",
)
GATE_SINGLE_NAMES = ("f_tilde", "log_span_y", "frac_far_4m", "left_hinge")
GATE_FLIP_EXTRA_NAMES = ("f_abs_diff", "f_tilde_min", "depth_flip_log_std")

SEM_DIMS = {
    "none": 0,
    "cls": 384,
    "df": 512,
    "cls_pm_df": 1280,
    "all_sp": 1664,
    "all_mp": 1664,
}

LOSS_WEIGHTS = {
    "rmse_only": (0.0, 1.0, 0.0),
    "balanced_111": (1.0, 1.0, 1.0),
    "ar_rmse_d1_1512": (1.5, 1.0, 1.5),
}


@dataclass(frozen=True)
class GeoOnlySpec:
    """Declarative configuration for pure-geometry calibrator ablations."""

    name: str
    kind: str = "joint"
    use_focal_curve: bool = False
    use_geo_cxcy: bool = False
    use_spatial3d: bool = False
    use_rgb3: bool = False
    use_depth_slope: bool = False
    use_ray_unwarp: bool = False
    use_cond_spatial2d: bool = False
    loss_mode: str = "balanced_111"
    reg_l2: float = 0.01
    max_log_scale: float = 0.40


GEO_ONLY_PRIMARY_SPEC = GeoOnlySpec(
    name=PRIMARY_GEO_METHOD,
    use_focal_curve=True,
    use_geo_cxcy=True,
    use_spatial3d=True,
    use_depth_slope=True,
    use_ray_unwarp=False,
    use_cond_spatial2d=False,
    loss_mode="balanced_111",
    reg_l2=0.01,
)


@dataclass(frozen=True)
class GeoCalibSpec:
    """Declarative specification for unified geometric, inverse-depth, and semantic calibrators."""

    name: str
    direction: str  # "deepen" | "new_param" | "new_info" | "combine" | "polish" | "multipass"
    track: str = "single_pass"  # "single_pass" | "multi_pass_flip_tta"
    depth_param: str = "log_spline"  # "log_spline" | "inv_spline" | "rational"
    use_quantile_knots: bool = False
    use_inv_residual: bool = False
    free_left_focal: bool = False
    use_focal_gate: bool = False
    geo_mode: str = "clean"  # "full_cxcy" | "clean" | "extended" | "with_conf"
    z_clip: float | None = 4.0
    sem_mode: str = "none"  # "none" | "cls" | "df" | "cls_pm_df" | "all_sp" | "all_mp"
    sem_pca_k: int | None = None
    sem_reg: float = 0.05
    sem_scale_cap: float = 0.40
    use_sem_spatial2d: bool = False
    sem_sp_reg: float = 0.08
    use_sem_slope: bool = False
    loss_mode: str = "balanced_111"  # "balanced_111" | "robust_huber_111"
    flip_blend_mode: str = "none"  # "none" | "learnable" | "conf" | "sym_geo_zero_retrain"
    use_flip_consistency: bool = False
    blend_members: tuple[str, str] | None = None


def fit_scale(ratios: np.ndarray) -> float:
    """Compute global median training depth ratio."""
    arr = np.asarray(ratios, dtype=np.float64)
    if not arr.size or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError("Expected finite positive training ratios")
    return float(np.median(arr))


def apply_spline(depth: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Apply piecewise linear log-depth monotone spline calibration and clip."""
    p = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Raw depth must be finite and positive")
    x = np.asarray(model["x_knots"], dtype=np.float64)
    y = np.asarray(model["y_knots"], dtype=np.float64)
    lp = np.log(p)
    i = np.clip(np.searchsorted(x, lp, side="right") - 1, 0, len(x) - 2)
    out = np.exp(y[i] + (lp - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
    return clip_depth(out)


def fit_monotone_spline(samples: np.ndarray, targets: np.ndarray, scale: float) -> dict[str, Any]:
    """Fit 6-knot monotone log-depth spline initialized at median scale."""
    p = torch.tensor(np.asarray(samples), dtype=torch.float64)
    target = torch.tensor(np.asarray(targets), dtype=torch.float64)
    if (
        p.ndim != 2
        or target.shape != p.shape
        or not bool(torch.isfinite(p).all())
        or not bool(torch.isfinite(target).all())
        or bool((p <= 0).any())
        or bool((target <= 0).any())
        or not 0.5 < scale < 2.0
    ):
        raise ValueError("Expected positive paired image samples and 0.5 < initial scale < 2")
    knots = torch.tensor(DEPTH_KNOTS, dtype=torch.float64)
    raw = torch.tensor(
        [np.arctanh(np.log(scale) / np.log(2.0)), *([np.log(3.0 / 4.0)] * 5)],
        dtype=torch.float64,
        requires_grad=True,
    )

    def _loss_comp(pred: torch.Tensor) -> torch.Tensor:
        rel = (pred - target) / target
        ar = (rel.square() + 1e-6).sqrt().sub(0.001).mean()
        rmse = ((pred - target).square().mean(-1) + 1e-12).sqrt().mean()
        delta = torch.nn.functional.softplus(((pred.log() - target.log()).abs() - np.log(1.25)) / 0.02).mean() * 0.02
        return torch.stack([ar, rmse, delta])

    with torch.no_grad():
        reference = _loss_comp((p * scale).clamp(0.001, 10.0)).clamp_min(1e-9)
    optimizer = torch.optim.LBFGS(
        [raw],
        max_iter=80,
        max_eval=120,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
    )

    def decoded() -> tuple[torch.Tensor, torch.Tensor]:
        offset = np.log(2.0) * raw[0].tanh()
        slopes = 0.25 + 1.75 * raw[1:].sigmoid()
        y = torch.cat(
            [
                (knots[0] + offset).reshape(1),
                knots[0] + offset + torch.cumsum(slopes * torch.diff(knots), dim=0),
            ]
        )
        return offset, y

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _, y = decoded()
        lp = p.log()
        i = (torch.bucketize(lp.contiguous(), knots, right=True) - 1).clamp(0, len(knots) - 2)
        mapped = (y[i] + (lp - knots[i]) * (y[i + 1] - y[i]) / (knots[i + 1] - knots[i])).exp().clamp(0.001, 10.0)
        slopes = torch.diff(y) / torch.diff(knots)
        terms = _loss_comp(mapped) / reference
        loss = terms[1] + (0.01 * torch.diff(slopes).square().mean() + 0.001 * (slopes - 1).square().mean())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        offset, y = decoded()
        return {
            "kind": "monotone_spline",
            "offset": float(offset),
            "slopes": (torch.diff(y) / torch.diff(knots)).tolist(),
            "x_knots": DEPTH_KNOTS.tolist(),
            "y_knots": y.tolist(),
        }


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        denom = float(max(np.linalg.norm(arr), 1e-8))
        return arr / denom
    norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    return arr / norms


def assemble_semantic_matrix(
    sem_mode: str,
    cls_orig: np.ndarray | None = None,
    patch_mean: np.ndarray | None = None,
    patch_std: np.ndarray | None = None,
    depth_feat_stats: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble L2-normalized semantic block matrix cleanly without label leakage."""
    if sem_mode == "none":
        n_rows = len(cls_orig) if cls_orig is not None else 0
        return np.zeros((n_rows, 0), dtype=np.float64)
    if sem_mode == "cls":
        assert cls_orig is not None
        return _l2_normalize_rows(cls_orig)
    if sem_mode == "df":
        assert depth_feat_stats is not None
        return _l2_normalize_rows(depth_feat_stats)
    if sem_mode == "cls_pm_df":
        assert cls_orig is not None and patch_mean is not None and depth_feat_stats is not None
        return np.hstack([_l2_normalize_rows(cls_orig), _l2_normalize_rows(patch_mean), _l2_normalize_rows(depth_feat_stats)])
    if sem_mode == "all_sp":
        assert cls_orig is not None and patch_mean is not None and patch_std is not None and depth_feat_stats is not None
        return np.hstack([
            _l2_normalize_rows(cls_orig),
            _l2_normalize_rows(patch_mean),
            _l2_normalize_rows(patch_std),
            _l2_normalize_rows(depth_feat_stats),
        ])
    if sem_mode == "all_mp":
        assert (
            cls_orig is not None
            and cls_flip is not None
            and patch_mean is not None
            and patch_std is not None
            and depth_feat_stats is not None
        )
        cls_sym = _l2_normalize_rows(0.5 * (_l2_normalize_rows(cls_orig) + _l2_normalize_rows(cls_flip)))
        return np.hstack([
            cls_sym,
            _l2_normalize_rows(patch_mean),
            _l2_normalize_rows(patch_std),
            _l2_normalize_rows(depth_feat_stats),
        ])
    raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")


def assemble_semantic_vector(
    sem_mode: str,
    cls_orig: np.ndarray | None = None,
    patch_mean: np.ndarray | None = None,
    patch_std: np.ndarray | None = None,
    depth_feat_stats: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
) -> np.ndarray:
    """Single-image L2-normalized semantic vector matching assemble_semantic_matrix bit-for-bit."""
    if sem_mode == "none":
        return np.zeros(0, dtype=np.float64)
    if sem_mode == "cls":
        assert cls_orig is not None
        return _l2_normalize_rows(cls_orig)
    if sem_mode == "df":
        assert depth_feat_stats is not None
        return _l2_normalize_rows(depth_feat_stats)
    if sem_mode == "cls_pm_df":
        assert cls_orig is not None and patch_mean is not None and depth_feat_stats is not None
        return np.concatenate([_l2_normalize_rows(cls_orig), _l2_normalize_rows(patch_mean), _l2_normalize_rows(depth_feat_stats)])
    if sem_mode == "all_sp":
        assert cls_orig is not None and patch_mean is not None and patch_std is not None and depth_feat_stats is not None
        return np.concatenate([
            _l2_normalize_rows(cls_orig),
            _l2_normalize_rows(patch_mean),
            _l2_normalize_rows(patch_std),
            _l2_normalize_rows(depth_feat_stats),
        ])
    if sem_mode == "all_mp":
        assert (
            cls_orig is not None
            and cls_flip is not None
            and patch_mean is not None
            and patch_std is not None
            and depth_feat_stats is not None
        )
        cls_sym = _l2_normalize_rows(0.5 * (_l2_normalize_rows(cls_orig) + _l2_normalize_rows(cls_flip)))
        return np.concatenate([
            cls_sym,
            _l2_normalize_rows(patch_mean),
            _l2_normalize_rows(patch_std),
            _l2_normalize_rows(depth_feat_stats),
        ])
    raise ValueError(f"Unsupported sem_mode: {sem_mode!r}")


def extract_base_geo_features(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    rgb_image: np.ndarray | None = None,
    need_rgb: bool = False,
) -> dict[str, float]:
    """Extract single-pass RGB-only geometry and pseudo-3D depth structure features."""
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


def extract_geocalib_features(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    cached_base_feat: dict[str, float] | None = None,
    conf_summary: np.ndarray | None = None,
    flip_depth: np.ndarray | None = None,
    flip_intrinsics: np.ndarray | None = None,
    conf_orig: np.ndarray | None = None,
    conf_flip: np.ndarray | None = None,
    cls_orig: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract RGB-only single-pass + optional 2-pass flip-consistency geometry features."""
    feats = (
        dict(cached_base_feat)
        if cached_base_feat is not None
        else extract_base_geo_features(raw_depth, pred_intrinsics, None, need_rgb=False)
    )
    p = np.asarray(raw_depth, dtype=np.float64)
    p_sub = p[CROP_SUB]
    frac_far = float(np.mean(p_sub > 4.0))
    f_t = float(feats["f_tilde"])
    left_h = max(0.75 - f_t, 0.0)
    feats["frac_far_4m"] = frac_far
    feats["left_hinge"] = left_h
    feats["left_hinge_x_log_spany"] = left_h * float(feats["log_span_y"])
    feats["left_hinge_x_frac_far"] = left_h * frac_far

    if conf_summary is not None:
        if len(conf_summary) != len(CONF_SUMMARY_NAMES):
            raise ValueError(f"Expected {len(CONF_SUMMARY_NAMES)} conf summary entries")
        for c_name, c_val in zip(CONF_SUMMARY_NAMES, conf_summary):
            feats[c_name] = float(c_val)

    if flip_depth is not None and flip_intrinsics is not None:
        fl = compute_flip_features(
            raw_depth=p,
            k_orig=pred_intrinsics,
            flip_depth=flip_depth,
            k_flip=flip_intrinsics,
            conf_orig=conf_orig,
            conf_flip=conf_flip,
            cls_orig=cls_orig,
            cls_flip=cls_flip,
        )
        p2_sub = np.asarray(flip_depth[CROP_SUB], dtype=np.float64)
        feats["f2_tilde"] = float(np.clip(fl["f_flip_tilde"], FOCAL_KNOTS[0], FOCAL_KNOTS[-1]))
        feats["f_tilde_mean"] = float(np.clip(0.5 * (f_t + feats["f2_tilde"]), FOCAL_KNOTS[0], FOCAL_KNOTS[-1]))
        feats["f_abs_diff"] = float(fl["f_abs_diff"])
        feats["f_tilde_min"] = float(fl["f_tilde_min"])
        feats["depth_flip_log_std"] = float(np.std(np.log(p_sub) - np.log(p2_sub)))
        feats["conf_flip_abs_mean"] = float(fl["conf_flip_abs_mean"])
        feats["cls_flip_cos_dist"] = float(1.0 - fl["cls_flip_cos"])
    return feats


def _feature_matrix(feat_list: list[dict[str, float]], names: tuple[str, ...]) -> np.ndarray:
    if not names:
        return np.zeros((len(feat_list), 0), dtype=np.float64)
    return np.asarray([[f[k] for k in names] for f in feat_list], dtype=np.float64)


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


def _std_z(feats: dict[str, float], blk: dict[str, Any]) -> np.ndarray:
    names = blk["feature_names"]
    if not names:
        return np.zeros(0, dtype=np.float64)
    x = np.array([feats[k] for k in names], dtype=np.float64)
    z = (x - np.asarray(blk["mean"], dtype=np.float64)) / np.asarray(blk["std"], dtype=np.float64)
    z_clip = blk.get("z_clip")
    if z_clip is not None:
        zc = float(z_clip)
        z = zc * np.tanh(z / zc)
    return z


def _clip_z(z: np.ndarray, clip_val: float | None) -> np.ndarray:
    if clip_val is None or z.shape[1] == 0:
        return z
    return clip_val * np.tanh(z / clip_val)


def _interp_1d_piecewise_torch(x: torch.Tensor, x_knots: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
    i = (torch.bucketize(x.contiguous(), x_knots, right=True) - 1).clamp(0, len(x_knots) - 2)
    return y_knots[i] + (x - x_knots[i]) * (y_knots[i + 1] - y_knots[i]) / (x_knots[i + 1] - x_knots[i])


def _interp_1d_piecewise_np(x: np.ndarray | float, x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    i = np.clip(np.searchsorted(x_knots, x_arr, side="right") - 1, 0, len(x_knots) - 2)
    return y_knots[i] + (x_arr - x_knots[i]) * (y_knots[i + 1] - y_knots[i]) / (x_knots[i + 1] - x_knots[i])


def _loss_components_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    rel = (pred - gt) / gt
    ar = (rel.square() + 1e-6).sqrt().sub(0.001).mean()
    rmse = ((pred - gt).square().mean(dim=-1) + 1e-12).sqrt().mean()
    d1 = torch.sigmoid(((pred.log() - gt.log()).abs() - np.log(1.25)) / 0.03).mean()
    return torch.stack([ar, rmse, d1])


def _loss_components_geocalib_torch(pred: torch.Tensor, gt: torch.Tensor, loss_mode: str) -> torch.Tensor:
    if loss_mode == "balanced_111":
        return _loss_components_torch(pred, gt)
    if loss_mode == "robust_huber_111":
        rel = (pred - gt) / gt
        ar_img = (rel.square() + 1e-6).sqrt().sub(0.001).mean(dim=-1)
        rmse_img = ((pred - gt).square().mean(dim=-1) + 1e-12).sqrt()
        d1 = torch.sigmoid(((pred.log() - gt.log()).abs() - np.log(1.25)) / 0.03).mean()
        ar = (0.25 * torch.log1p(ar_img / 0.25)).mean()
        rmse = (0.80 * torch.log1p(rmse_img / 0.80)).mean()
        return torch.stack([ar, rmse, d1])
    raise ValueError(f"Unsupported loss_mode: {loss_mode!r}")


def _centered_log_cos_hat_crop(K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    log_cos = -0.5 * np.log(1.0 + ((_CROP_U_PIX - cx) / fx) ** 2 + ((_CROP_V_PIX - cy) / fy) ** 2)
    return log_cos - float(log_cos.mean())


def _centered_log_cos_hat_full(K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    log_cos_full = -0.5 * np.log(1.0 + ((_FULL_U_PIX - cx) / fx) ** 2 + ((_FULL_V_PIX - cy) / fy) ** 2)
    return log_cos_full - float(log_cos_full[CROP].mean())


def _scale_feature_names(spec: GeoOnlySpec) -> tuple[str, ...]:
    names: list[str] = []
    if spec.use_geo_cxcy:
        names.extend(GEO_CXCY_NAMES)
    if spec.use_spatial3d:
        names.extend(SPATIAL3D_NAMES)
    if spec.use_rgb3:
        names.extend(RGB3_NAMES)
    return tuple(names)


def _slope_feature_names(spec: GeoOnlySpec) -> tuple[str, ...]:
    if not spec.use_depth_slope:
        return ()
    return SLOPE_FULL_NAMES if spec.use_spatial3d else SLOPE_GEO_NAMES


def _cond2d_feature_names(spec: GeoOnlySpec) -> tuple[str, ...]:
    if not spec.use_cond_spatial2d:
        return ()
    return COND2D_FULL_NAMES if spec.use_spatial3d else COND2D_GEO_NAMES


class _JointGeoOnlyModule(torch.nn.Module):
    """Differentiable float64 geometry-only calibrator module."""

    def __init__(
        self,
        spec: GeoOnlySpec,
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
        self.raw_depth_slopes = torch.nn.Parameter(
            torch.full((len(DEPTH_KNOTS) - 1,), float(np.log(3.0 / 4.0)), dtype=torch.float64)
        )
        if spec.use_focal_curve:
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
        return raw_y - raw_y[FOCAL_ANCHOR_IDX], slopes

    def forward(
        self,
        lp: torch.Tensor,
        f_tilde: torch.Tensor,
        z_scale: torch.Tensor,
        z_slope: torch.Tensor,
        z_cond2d: torch.Tensor,
        v_tilde: torch.Tensor,
        r2_centered: torch.Tensor,
        d2_center: torch.Tensor,
        log_cos_hat_c: torch.Tensor,
    ) -> torch.Tensor:
        y_d, _ = self.depth_y_knots()
        base_lp = _interp_1d_piecewise_torch(lp, self.depth_knots, y_d)

        shift = torch.zeros_like(f_tilde)
        if self.spec.use_focal_curve:
            y_f, _ = self.focal_y_knots()
            shift = shift + _interp_1d_piecewise_torch(f_tilde, self.focal_knots, y_f)
        if z_scale.shape[1] > 0:
            m_lim = self.spec.max_log_scale
            shift = shift + m_lim * torch.tanh((z_scale @ self.w_scale) / m_lim)
        if self.spec.use_focal_curve or z_scale.shape[1] > 0:
            shift = 0.50 * torch.tanh(shift / 0.50)

        out_lp = base_lp + shift[:, None]
        if z_slope.shape[1] > 0:
            delta_b = 0.20 * torch.tanh(z_slope @ self.w_slope)
            out_lp = out_lp + delta_b[:, None] * (lp - LOG_DEPTH_CENTER)
        if self.spec.use_ray_unwarp:
            f_ref = 0.8 + 0.6 * torch.sigmoid(self.raw_f_ref)
            gamma_ray = 1.5 * torch.tanh(self.raw_gamma_ray)
            log_cos_ref = -0.5 * torch.log(1.0 + d2_center / ((480.0 * f_ref) ** 2))
            log_cos_ref_c = log_cos_ref - log_cos_ref.mean(dim=1, keepdim=True)
            out_lp = out_lp + gamma_ray * (log_cos_ref_c - log_cos_hat_c)
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


def fit_geo_only_model(
    spec: GeoOnlySpec,
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
    """Fit a joint float64 pure-geometry calibrator using L-BFGS exclusively on training samples."""
    n = len(features)
    init_scale = float(np.clip(np.median(ratios), 0.55, 1.85))
    f_tildes = np.array([f["f_tilde"] for f in features], dtype=np.float64)
    init_f_ref = float(np.median(f_tildes))

    scale_names = _scale_feature_names(spec)
    slope_names = _slope_feature_names(spec)
    cond2d_names = _cond2d_feature_names(spec)

    z_scale_np, mu_scale, sd_scale = _standardize_fit(_feature_matrix(features, scale_names))
    z_slope_np, mu_slope, sd_slope = _standardize_fit(_feature_matrix(features, slope_names))
    z_cond2d_np, mu_cond2d, sd_cond2d = _standardize_fit(_feature_matrix(features, cond2d_names))

    v_tilde_s = _CROP_V_TILDE[samples_row, samples_col]
    r2_c_s = _CROP_R2[samples_row, samples_col] - _MEAN_R2_CROP
    d2_s = _CROP_D2_CENTER[samples_row, samples_col]
    log_cos_hat_c_s = np.empty_like(samples_p, dtype=np.float64)
    if spec.use_ray_unwarp:
        for i in range(n):
            log_cos_hat_c_s[i] = _centered_log_cos_hat_crop(intrinsics[i])[samples_row[i], samples_col[i]]
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

    def _run_lbfgs(lr: float) -> tuple[_JointGeoOnlyModule, list[float]]:
        mod = _JointGeoOnlyModule(
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
                raise ArithmeticError("Nonfinite calibration objective")
            loss.backward()
            hist.append(float(loss.detach()))
            return loss

        opt.step(closure)
        return mod, hist

    try:
        mod, history = _run_lbfgs(lr=1.0)
        if any(not torch.isfinite(p).all() for p in mod.parameters()):
            raise ArithmeticError("Nonfinite parameters after LBFGS")
    except ArithmeticError:
        mod, history = _run_lbfgs(lr=0.5)

    with torch.no_grad():
        y_d, d_slopes = mod.depth_y_knots()
        y_d_list, d_slopes_list = y_d.cpu().tolist(), d_slopes.cpu().tolist()
        if spec.use_focal_curve:
            y_f, f_slopes = mod.focal_y_knots()
            y_f_list, f_slopes_list = y_f.cpu().tolist(), f_slopes.cpu().tolist()
        else:
            y_f_list, f_slopes_list = None, None
        w_scale_list = mod.w_scale.cpu().tolist() if len(scale_names) else []
        w_slope_list = mod.w_slope.cpu().tolist() if len(slope_names) else []
        if spec.use_ray_unwarp:
            gamma_ray = float((1.5 * torch.tanh(mod.raw_gamma_ray)).cpu())
            f_ref_tilde = float((0.8 + 0.6 * torch.sigmoid(mod.raw_f_ref)).cpu())
        else:
            gamma_ray, f_ref_tilde = 0.0, init_f_ref
        if len(cond2d_names):
            a_v0, a_r0 = float(mod.a_v0.cpu()), float(mod.a_r0.cpu())
            w_v_list, w_r_list = mod.w_v.cpu().tolist(), mod.w_r.cpu().tolist()
        else:
            a_v0, a_r0, w_v_list, w_r_list = 0.0, 0.0, [], []

    return {
        "kind": "geocalib_geo",
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


def _geo_feature_names(spec: GeoCalibSpec) -> tuple[str, ...]:
    if spec.geo_mode == "full_cxcy":
        base: tuple[str, ...] = (*GEO_CXCY_NAMES, *SPATIAL3D_NAMES)
    elif spec.geo_mode == "clean":
        base = GEO_CLEAN_NAMES
    elif spec.geo_mode == "extended":
        base = GEO_EXT_NAMES
    elif spec.geo_mode == "with_conf":
        base = GEO_CONF_NAMES
    else:
        raise ValueError(f"Unsupported geo_mode: {spec.geo_mode!r}")
    if spec.use_flip_consistency:
        base = (*base, *FLIP_CONSISTENCY_NAMES)
    return base


def _gate_feature_names(spec: GeoCalibSpec) -> tuple[str, ...]:
    if not spec.use_focal_gate:
        return ()
    if spec.use_flip_consistency:
        return (*GATE_SINGLE_NAMES, *GATE_FLIP_EXTRA_NAMES)
    return GATE_SINGLE_NAMES


def _clip_z(z: np.ndarray, clip_val: float | None) -> np.ndarray:
    if clip_val is None or z.shape[1] == 0:
        return z
    return clip_val * np.tanh(z / clip_val)


def _compute_quantile_knots(samples_p: np.ndarray, depth_param: str) -> tuple[np.ndarray, int]:
    """Compute strictly increasing depth spline knots inside current training fold."""
    qs = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
    if depth_param == "inv_spline":
        vals = np.quantile(1.0 / np.clip(samples_p, 0.05, 12.0), qs)
        raw = np.concatenate([[1.0 / 12.0], vals, [1.0 / 0.3]])
        raw = np.maximum.accumulate(raw)
        knots = raw.copy()
        for j in range(1, len(knots)):
            knots[j] = max(knots[j], knots[j - 1] + 0.04)
        return knots, len(knots)
    vals = np.quantile(np.log(np.clip(samples_p, 0.05, 12.0)), qs)
    raw = np.concatenate([[np.log(0.3)], vals, [np.log(12.0)]])
    knots = raw.copy()
    for j in range(1, len(knots)):
        knots[j] = max(knots[j], knots[j - 1] + 0.12)
    return knots, len(knots)


def _softplus_inv(val: float) -> float:
    return float(np.log(np.expm1(max(val - 1e-4, 1e-4))))


class _JointGeoCalibModule(torch.nn.Module):
    """Differentiable float64 joint calibrator supporting inverse-depth, focal, semantic, and flip fusion."""

    def __init__(
        self,
        spec: GeoCalibSpec,
        depth_knots: np.ndarray,
        focal_knots: np.ndarray,
        focal_anchor_idx: int,
        init_scale: float,
        n_geo: int,
        n_slope: int,
        n_gate: int,
        d_sem: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.focal_anchor_idx = int(focal_anchor_idx)
        self.init_scale = float(init_scale)
        self.register_buffer("depth_knots", torch.tensor(depth_knots, dtype=torch.float64))
        self.register_buffer("focal_knots", torch.tensor(focal_knots, dtype=torch.float64))

        n_seg = len(depth_knots) - 1
        if spec.depth_param == "log_spline":
            init_off = float(np.arctanh(np.clip(np.log(init_scale) / np.log(2.0), -0.99, 0.99)))
            self.raw_offset = torch.nn.Parameter(torch.tensor(init_off, dtype=torch.float64))
            self.raw_depth_slopes = torch.nn.Parameter(
                torch.full((n_seg,), float(np.log(3.0 / 4.0)), dtype=torch.float64)
            )
        elif spec.depth_param == "inv_spline":
            s0 = float(np.clip(1.0 / init_scale, 0.30, 1.80))
            self.raw_offset = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.raw_depth_slopes = torch.nn.Parameter(
                torch.full((n_seg,), float(np.log((s0 - 0.25) / (2.0 - s0))), dtype=torch.float64)
            )
        elif spec.depth_param == "rational":
            self.raw_a = torch.nn.Parameter(torch.tensor(_softplus_inv(init_scale), dtype=torch.float64))
            self.raw_b = torch.nn.Parameter(torch.tensor(_softplus_inv(0.05), dtype=torch.float64))
            self.raw_c = torch.nn.Parameter(torch.tensor(_softplus_inv(0.05 / init_scale), dtype=torch.float64))
        else:
            raise ValueError(f"Unsupported depth_param: {spec.depth_param!r}")

        n_fseg = len(focal_knots) - 1
        self.raw_focal_slopes = torch.nn.Parameter(torch.full((n_fseg,), -2.0, dtype=torch.float64))
        if spec.free_left_focal:
            self.raw_free_left_slope = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

        if n_gate > 0:
            self.b_gate = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.w_gate = torch.nn.Parameter(torch.zeros(n_gate, dtype=torch.float64))

        self.w_scale = torch.nn.Parameter(torch.zeros(n_geo, dtype=torch.float64))
        self.w_slope = torch.nn.Parameter(torch.zeros(n_slope, dtype=torch.float64))

        if spec.use_inv_residual:
            self.w_inv = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        if d_sem > 0:
            self.w_sem_s = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
            if spec.use_sem_spatial2d:
                self.a_v0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
                self.a_r0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
                self.w_sem_v = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
                self.w_sem_r = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
            if spec.use_sem_slope:
                self.w_sem_b = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))

        if spec.flip_blend_mode == "learnable":
            self.raw_w_flip = torch.nn.Parameter(torch.tensor(float(np.log(0.55 / 0.45)), dtype=torch.float64))

    def depth_y_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        slopes = 0.25 + 1.75 * torch.sigmoid(self.raw_depth_slopes)
        if self.spec.depth_param == "log_spline":
            offset = np.log(2.0) * torch.tanh(self.raw_offset)
            y0 = (self.depth_knots[0] + offset).reshape(1)
        else:
            offset = 0.08 * torch.tanh(self.raw_offset)
            y0 = (self.depth_knots[0] / self.init_scale + offset).reshape(1)
        y = torch.cat([y0, y0 + torch.cumsum(slopes * torch.diff(self.depth_knots), dim=0)])
        return y, slopes

    def rational_abc(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return F.softplus(self.raw_a) + 1e-4, F.softplus(self.raw_b) + 1e-4, F.softplus(self.raw_c) + 1e-4

    def focal_y_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        slopes = -2.5 * torch.sigmoid(self.raw_focal_slopes)
        if self.spec.free_left_focal:
            left_s = (1.5 * torch.tanh(self.raw_free_left_slope)).reshape(1)
            slopes = torch.cat([left_s, slopes[1:]])
        deltas = slopes * torch.diff(self.focal_knots)
        zero = torch.zeros(1, dtype=torch.float64, device=deltas.device)
        raw_y = torch.cat([zero, torch.cumsum(deltas, dim=0)])
        return raw_y - raw_y[self.focal_anchor_idx], slopes

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor | None,
        w1_conf: torch.Tensor | None,
        f_vec: torch.Tensor,
        z_geo: torch.Tensor,
        z_slope: torch.Tensor,
        z_gate: torch.Tensor,
        x_sem: torch.Tensor,
        v_tilde: torch.Tensor,
        r2_centered: torch.Tensor,
    ) -> torch.Tensor:
        if self.spec.flip_blend_mode == "learnable":
            assert p2 is not None
            w_f = torch.sigmoid(self.raw_w_flip)
            p_in = torch.exp(w_f * torch.log(p1) + (1.0 - w_f) * torch.log(p2))
        elif self.spec.flip_blend_mode == "conf":
            assert p2 is not None and w1_conf is not None
            p_in = torch.exp(w1_conf * torch.log(p1) + (1.0 - w1_conf) * torch.log(p2))
        else:
            p_in = p1

        lp = torch.log(p_in)
        invp = torch.reciprocal(p_in)

        if self.spec.depth_param == "log_spline":
            yd, _ = self.depth_y_knots()
            out_lp = _interp_1d_piecewise_torch(lp, self.depth_knots, yd)
        elif self.spec.depth_param == "inv_spline":
            yd, _ = self.depth_y_knots()
            d_spl = _interp_1d_piecewise_torch(invp, self.depth_knots, yd).clamp_min(0.04)
            out_lp = -torch.log(d_spl)
        else:
            a, b, c = self.rational_abc()
            out_lp = lp + torch.log(a + b * p_in) - torch.log(1.0 + c * p_in)

        yf, _ = self.focal_y_knots()
        hf = _interp_1d_piecewise_torch(f_vec, self.focal_knots, yf)
        if self.spec.use_focal_gate:
            g_rel = torch.sigmoid(self.b_gate + z_gate @ self.w_gate)
            hf = hf * g_rel

        shift = hf + 0.40 * torch.tanh((z_geo @ self.w_scale) / 0.40)
        if x_sem.shape[1] > 0:
            cap = self.spec.sem_scale_cap
            shift = shift + cap * torch.tanh((x_sem @ self.w_sem_s) / cap)
        shift = 0.50 * torch.tanh(shift / 0.50)
        out_lp = out_lp + shift[:, None]

        slope_lin = z_slope @ self.w_slope
        if x_sem.shape[1] > 0 and self.spec.use_sem_slope:
            slope_lin = slope_lin + x_sem @ self.w_sem_b
        out_lp = out_lp + (0.20 * torch.tanh(slope_lin))[:, None] * (lp - LOG_DEPTH_CENTER)

        if self.spec.use_inv_residual:
            out_lp = out_lp + (0.20 * torch.tanh(self.w_inv)) * (invp - 0.5)

        if x_sem.shape[1] > 0 and self.spec.use_sem_spatial2d:
            cv = 0.14 * torch.tanh(self.a_v0 + x_sem @ self.w_sem_v)
            cr = 0.14 * torch.tanh(self.a_r0 + x_sem @ self.w_sem_r)
            out_lp = out_lp + cv[:, None] * v_tilde + cr[:, None] * r2_centered

        return torch.exp(torch.clamp(out_lp, np.log(0.001), np.log(10.0))).clamp(0.001, 10.0)

    def regularizer(self) -> torch.Tensor:
        if self.spec.depth_param in ("log_spline", "inv_spline"):
            _, d_slopes = self.depth_y_knots()
            target_s = 1.0 if self.spec.depth_param == "log_spline" else (1.0 / self.init_scale)
            reg = 0.01 * torch.diff(d_slopes).square().mean() + 0.001 * (d_slopes - target_s).square().mean()
        else:
            _, b, c = self.rational_abc()
            reg = 0.002 * (b.square() + c.square())

        _, f_slopes = self.focal_y_knots()
        reg = reg + 0.005 * torch.diff(f_slopes).square().mean() + 0.0005 * f_slopes.square().mean()
        reg = reg + 0.01 * (self.w_scale.square().sum() + self.w_slope.square().sum())
        if self.spec.use_inv_residual:
            reg = reg + 0.01 * self.w_inv.square()
        if self.spec.use_focal_gate:
            reg = reg + 0.02 * (self.w_gate.square().sum() + (self.b_gate - 2.0).square())
        if hasattr(self, "w_sem_s"):
            reg = reg + self.spec.sem_reg * self.w_sem_s.square().sum()
        if hasattr(self, "w_sem_v"):
            reg = reg + self.spec.sem_sp_reg * (
                self.w_sem_v.square().sum() + self.w_sem_r.square().sum() + self.a_v0.square() + self.a_r0.square()
            )
        if hasattr(self, "w_sem_b"):
            reg = reg + self.spec.sem_reg * self.w_sem_b.square().sum()
        if self.spec.flip_blend_mode == "learnable":
            reg = reg + 0.002 * (self.raw_w_flip - float(np.log(0.55 / 0.45))).square()
        return reg


def _estimate_ridge_dof(x_sem_tr: np.ndarray, h_scale_curvature: float, reg_coeff: float) -> float:
    """Compute Gauss-Newton effective degrees of freedom tr(H) from centered training semantics."""
    if x_sem_tr.shape[1] == 0 or reg_coeff <= 0.0:
        return 0.0
    n_tr = max(len(x_sem_tr), 1)
    svals = np.linalg.svd(x_sem_tr, full_matrices=False, compute_uv=False)
    eig = max(float(h_scale_curvature), 0.0) * (svals ** 2) / n_tr
    return float(np.sum(eig / (eig + 2.0 * float(reg_coeff) + 1e-12)))


def fit_geocalib_model(
    spec: GeoCalibSpec,
    samples_p1: np.ndarray,
    samples_gt: np.ndarray,
    samples_row: np.ndarray,
    samples_col: np.ndarray,
    features: list[dict[str, float]],
    ratios: np.ndarray,
    x_sem_raw: np.ndarray,
    samples_p2: np.ndarray | None = None,
    samples_w1: np.ndarray | None = None,
    max_iter: int = 75,
) -> dict[str, Any]:
    """Fit a joint float64 GeoCalib-3D model exclusively on training samples."""
    init_scale = float(np.clip(np.median(ratios), 0.55, 1.85))
    if spec.use_quantile_knots:
        d_knots, _ = _compute_quantile_knots(samples_p1, spec.depth_param)
        f_knots, f_anchor_idx = FOCAL_KNOTS_Q7.copy(), FOCAL_Q7_ANCHOR_IDX
    else:
        d_knots = INV_DEPTH_KNOTS.copy() if spec.depth_param == "inv_spline" else DEPTH_KNOTS.copy()
        f_knots, f_anchor_idx = FOCAL_KNOTS.copy(), FOCAL_ANCHOR_IDX

    geo_names = _geo_feature_names(spec)
    slope_names = SLOPE_FULL_NAMES
    gate_names = _gate_feature_names(spec)

    z_geo_raw, mu_geo, sd_geo = _standardize_fit(_feature_matrix(features, geo_names))
    z_geo = _clip_z(z_geo_raw, spec.z_clip)
    z_slope, mu_slope, sd_slope = _standardize_fit(_feature_matrix(features, slope_names))
    z_gate_raw, mu_gate, sd_gate = _standardize_fit(_feature_matrix(features, gate_names))
    z_gate = _clip_z(z_gate_raw, spec.z_clip) if spec.use_focal_gate else z_gate_raw

    d_sem_orig = x_sem_raw.shape[1]
    if d_sem_orig > 0:
        mu_sem = x_sem_raw.mean(axis=0)
        x_sem_c = x_sem_raw - mu_sem
        if spec.sem_pca_k is not None and spec.sem_pca_k < min(x_sem_c.shape[0], x_sem_c.shape[1]):
            _, _, vt = np.linalg.svd(x_sem_c, full_matrices=False)
            proj_k = vt[: spec.sem_pca_k].T
            x_sem_fit = x_sem_c @ proj_k
        else:
            proj_k = None
            x_sem_fit = x_sem_c
    else:
        mu_sem = np.zeros(0, dtype=np.float64)
        proj_k = None
        x_sem_fit = x_sem_raw

    f_key = "f_tilde_mean" if spec.flip_blend_mode != "none" else "f_tilde"
    f_vec = np.array([f[f_key] for f in features], dtype=np.float64)
    v_s = _CROP_V_TILDE[samples_row, samples_col]
    r2_s = _CROP_R2[samples_row, samples_col] - _MEAN_R2_CROP

    p1_t = torch.tensor(samples_p1, dtype=torch.float64)
    p2_t = torch.tensor(samples_p2, dtype=torch.float64) if samples_p2 is not None else None
    w1_t = torch.tensor(samples_w1, dtype=torch.float64) if samples_w1 is not None else None
    gt_t = torch.tensor(samples_gt, dtype=torch.float64)
    f_t = torch.tensor(f_vec, dtype=torch.float64)
    zg_t = torch.tensor(z_geo, dtype=torch.float64)
    zsl_t = torch.tensor(z_slope, dtype=torch.float64)
    zgt_t = torch.tensor(z_gate, dtype=torch.float64)
    xsem_t = torch.tensor(x_sem_fit, dtype=torch.float64)
    vt_t = torch.tensor(v_s, dtype=torch.float64)
    r2_t = torch.tensor(r2_s, dtype=torch.float64)

    with torch.no_grad():
        ref = _loss_components_geocalib_torch((p1_t * init_scale).clamp(0.001, 10.0), gt_t, spec.loss_mode).clamp_min(1e-9)

    def _run(lr: float) -> tuple[_JointGeoCalibModule, list[float]]:
        mod = _JointGeoCalibModule(
            spec=spec,
            depth_knots=d_knots,
            focal_knots=f_knots,
            focal_anchor_idx=f_anchor_idx,
            init_scale=init_scale,
            n_geo=len(geo_names),
            n_slope=len(slope_names),
            n_gate=len(gate_names),
            d_sem=x_sem_fit.shape[1],
        )
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
            pred = mod(p1_t, p2_t, w1_t, f_t, zg_t, zsl_t, zgt_t, xsem_t, vt_t, r2_t)
            comp = _loss_components_geocalib_torch(pred, gt_t, spec.loss_mode) / ref
            loss = comp.sum() + mod.regularizer()
            if not bool(torch.isfinite(loss)):
                raise ArithmeticError(f"Nonfinite objective for {spec.name}")
            loss.backward()
            hist.append(float(loss.detach()))
            return loss

        opt.step(closure)
        return mod, hist

    try:
        mod, history = _run(lr=1.0)
        if any(not torch.isfinite(p).all() for p in mod.parameters()):
            raise ArithmeticError(f"Nonfinite parameters after LBFGS for {spec.name}")
    except ArithmeticError:
        mod, history = _run(lr=0.5)

    with torch.no_grad():
        sem_dof = 0.0
        sem_sp_dof = 0.0
        sem_slope_dof = 0.0
        if d_sem_orig > 0:
            pred0 = mod(p1_t, p2_t, w1_t, f_t, zg_t, zsl_t, zgt_t, xsem_t, vt_t, r2_t)
            eps = 1e-3
            l0 = float((_loss_components_geocalib_torch(pred0, gt_t, spec.loss_mode) / ref).sum())
            lp_up = float((_loss_components_geocalib_torch((pred0 * float(np.exp(eps))).clamp(0.001, 10.0), gt_t, spec.loss_mode) / ref).sum())
            lp_dn = float((_loss_components_geocalib_torch((pred0 * float(np.exp(-eps))).clamp(0.001, 10.0), gt_t, spec.loss_mode) / ref).sum())
            h_scale = max((lp_up - 2.0 * l0 + lp_dn) / (eps * eps), 0.0)
            sem_dof = _estimate_ridge_dof(x_sem_fit, h_scale, spec.sem_reg)
            if spec.use_sem_spatial2d:
                h_v = float(np.mean(_CROP_V_TILDE ** 2)) * 0.14 * 0.14 * h_scale
                h_r = float(np.mean((_CROP_R2 - _MEAN_R2_CROP) ** 2)) * 0.14 * 0.14 * h_scale
                sem_sp_dof = _estimate_ridge_dof(x_sem_fit, h_v, spec.sem_sp_reg) + _estimate_ridge_dof(
                    x_sem_fit, h_r, spec.sem_sp_reg
                )
            if spec.use_sem_slope:
                sem_slope_dof = _estimate_ridge_dof(x_sem_fit, 0.04 * h_scale, spec.sem_reg)

        if spec.depth_param in ("log_spline", "inv_spline"):
            yd, d_slopes = mod.depth_y_knots()
            depth_block: dict[str, Any] = {
                "param": spec.depth_param,
                "x_knots": d_knots.tolist(),
                "y_knots": yd.cpu().tolist(),
                "slopes": d_slopes.cpu().tolist(),
            }
        else:
            a, b, c = mod.rational_abc()
            depth_block = {"param": "rational", "a": float(a), "b": float(b), "c": float(c)}

        yf, f_slopes = mod.focal_y_knots()
        w_scale_l = mod.w_scale.cpu().tolist()
        w_slope_l = mod.w_slope.cpu().tolist()
        w_inv_val = float(0.20 * torch.tanh(mod.w_inv)) if spec.use_inv_residual else 0.0
        w_gate_l = mod.w_gate.cpu().tolist() if spec.use_focal_gate else []
        b_gate_v = float(mod.b_gate) if spec.use_focal_gate else 2.0
        w_flip_v = (
            float(torch.sigmoid(mod.raw_w_flip))
            if spec.flip_blend_mode == "learnable"
            else (0.55 if spec.flip_blend_mode != "none" else 1.0)
        )

        def _collapse_w(param: torch.Tensor) -> list[float]:
            vec = param.cpu().numpy()
            if proj_k is not None:
                vec = proj_k @ vec
            return vec.tolist()

        w_sem_s_l = _collapse_w(mod.w_sem_s) if d_sem_orig > 0 else []
        if d_sem_orig > 0 and spec.use_sem_spatial2d:
            sem_sp_blk: dict[str, Any] = {
                "enabled": True,
                "a_v0": float(mod.a_v0),
                "a_r0": float(mod.a_r0),
                "w_v": _collapse_w(mod.w_sem_v),
                "w_r": _collapse_w(mod.w_sem_r),
                "mean_r2_crop": _MEAN_R2_CROP,
            }
        else:
            sem_sp_blk = {"enabled": False, "a_v0": 0.0, "a_r0": 0.0, "w_v": [], "w_r": [], "mean_r2_crop": _MEAN_R2_CROP}

        w_sem_b_l = _collapse_w(mod.w_sem_b) if (d_sem_orig > 0 and spec.use_sem_slope) else []

    raw_param_count = count_raw_parameters_from_spec(spec)
    low_dim_count = (
        (len(d_knots) if spec.depth_param in ("log_spline", "inv_spline") else 3)
        + len(f_slopes)
        + len(w_scale_l)
        + len(w_slope_l)
        + (1 if spec.use_inv_residual else 0)
        + (len(w_gate_l) + 1 if spec.use_focal_gate else 0)
        + (1 if spec.flip_blend_mode == "learnable" else 0)
        + (2 if (d_sem_orig > 0 and spec.use_sem_spatial2d) else 0)
    )
    effective_dof = float(low_dim_count + sem_dof + sem_sp_dof + sem_slope_dof)

    return {
        "kind": "geocalib_3d",
        "spec": asdict(spec),
        "init_scale": init_scale,
        "depth_branch": depth_block,
        "focal_spline": {
            "x_knots": f_knots.tolist(),
            "y_knots": yf.cpu().tolist(),
            "slopes": f_slopes.cpu().tolist(),
            "focal_key": f_key,
            "free_left_slope": spec.free_left_focal,
        },
        "focal_gate": {
            "enabled": spec.use_focal_gate,
            "feature_names": list(gate_names),
            "mean": mu_gate.tolist(),
            "std": sd_gate.tolist(),
            "z_clip": spec.z_clip,
            "bias": b_gate_v,
            "weights": w_gate_l,
        },
        "scale_residual": {
            "feature_names": list(geo_names),
            "mean": mu_geo.tolist(),
            "std": sd_geo.tolist(),
            "z_clip": spec.z_clip,
            "weights": w_scale_l,
        },
        "depth_slope_mod": {
            "feature_names": list(slope_names),
            "mean": mu_slope.tolist(),
            "std": sd_slope.tolist(),
            "weights": w_slope_l,
            "log_depth_center": LOG_DEPTH_CENTER,
        },
        "inv_residual": {"enabled": spec.use_inv_residual, "weight": w_inv_val},
        "semantic_branch": {
            "sem_mode": spec.sem_mode,
            "pca_k": spec.sem_pca_k,
            "raw_dims": d_sem_orig,
            "mean": mu_sem.tolist(),
            "scale_cap": spec.sem_scale_cap,
            "w_scale": w_sem_s_l,
            "spatial2d": sem_sp_blk,
            "w_slope": w_sem_b_l,
        },
        "flip_fusion": {"mode": spec.flip_blend_mode, "w_flip": w_flip_v},
        "complexity": {
            "raw_params": raw_param_count,
            "low_dim_params": int(low_dim_count),
            "effective_dof": effective_dof,
            "sem_scale_dof": float(sem_dof),
            "sem_spatial2d_dof": float(sem_sp_dof),
            "sem_slope_dof": float(sem_slope_dof),
        },
        "optimization": {
            "closure_calls": len(history),
            "objective_first": history[0],
            "objective_last": history[-1],
        },
    }


def count_raw_parameters_from_spec(spec: GeoCalibSpec) -> int:
    """Count exact tunable scalar parameters in a GeoCalibSpec before regularization shrinkage."""
    if spec.flip_blend_mode == "sym_geo_zero_retrain":
        return 25
    if spec.blend_members is not None:
        return 1
    n_depth = 3 if spec.depth_param == "rational" else (8 if spec.use_quantile_knots else 6)
    n_focal = 6 if spec.use_quantile_knots else 5
    n_geo = len(_geo_feature_names(spec))
    n_slope = len(SLOPE_FULL_NAMES)
    n_inv = 1 if spec.use_inv_residual else 0
    n_gate = (len(_gate_feature_names(spec)) + 1) if spec.use_focal_gate else 0
    n_flip = 1 if spec.flip_blend_mode == "learnable" else 0
    d_sem_raw = SEM_DIMS[spec.sem_mode]
    d_fit = spec.sem_pca_k if (spec.sem_pca_k is not None and spec.sem_pca_k < d_sem_raw) else d_sem_raw
    sem_heads = (1 if d_fit > 0 else 0) + (2 if (d_fit > 0 and spec.use_sem_spatial2d) else 0) + (1 if (d_fit > 0 and spec.use_sem_slope) else 0)
    sem_bias = 2 if (d_fit > 0 and spec.use_sem_spatial2d) else 0
    return int(n_depth + n_focal + n_geo + n_slope + n_inv + n_gate + n_flip + sem_bias + sem_heads * d_fit)


def count_model_parameters(model: dict[str, Any]) -> tuple[int, float]:
    """Return (raw_params, effective_dof) for any exported model dictionary."""
    kind = model.get("kind")
    if kind == "scalar":
        n = 0 if float(model.get("scale", 1.0)) == 1.0 else 1
        return n, float(n)
    if kind == "monotone_spline":
        n = len(model["y_knots"])
        return n, float(n)
    if kind == "geocalib_geo":
        total = 6 + (5 if model["focal_spline"]["enabled"] else 0)
        total += len(model["scale_residual"]["weights"]) + len(model["depth_slope_mod"]["weights"])
        if model["ray_unwarp"]["enabled"]:
            total += 2
        if model["cond_spatial2d"]["w_v"]:
            total += 2 + 2 * len(model["cond_spatial2d"]["w_v"])
        return int(total), float(total)
    if kind == "geocalib_3d":
        c = model["complexity"]
        return int(c["raw_params"]), float(c["effective_dof"])
    if kind == "geocalib_flip_geo":
        return count_model_parameters(model["geo_model"])
    if kind == "geocalib_ensemble":
        m0_r, m0_d = count_model_parameters(model["members"][0])
        m1_r, m1_d = count_model_parameters(model["members"][1])
        return int(m0_r + m1_r + 1), float(m0_d + m1_d + 1.0)
    raise ValueError(f"Unknown model kind: {kind!r}")


def _apply_geo_only_model(
    p: np.ndarray,
    k: np.ndarray,
    model: dict[str, Any],
    rgb_image: np.ndarray | None = None,
    cached_features: dict[str, float] | None = None,
    use_torch_fast: bool = True,
) -> np.ndarray:
    spec_d = model["spec"]
    need_rgb = bool(spec_d.get("use_rgb3", False))
    feats = (
        cached_features
        if cached_features is not None
        else extract_base_geo_features(p, k, rgb_image, need_rgb=need_rgb)
    )
    shift = 0.0
    fs = model["focal_spline"]
    if fs["enabled"]:
        shift += float(_interp_1d_piecewise_np(feats["f_tilde"], np.asarray(fs["x_knots"], dtype=np.float64), np.asarray(fs["y_knots"], dtype=np.float64)))
    sr = model["scale_residual"]
    if sr["weights"]:
        x_s = np.array([feats[key] for key in sr["feature_names"]], dtype=np.float64)
        z_s = (x_s - np.asarray(sr["mean"], dtype=np.float64)) / np.asarray(sr["std"], dtype=np.float64)
        m_lim = float(sr["max_log_scale"])
        shift += m_lim * float(np.tanh(float(z_s @ np.asarray(sr["weights"], dtype=np.float64)) / m_lim))
    if fs["enabled"] or sr["weights"]:
        shift = 0.50 * float(np.tanh(shift / 0.50))

    dsm = model["depth_slope_mod"]
    delta_b = 0.0
    if dsm["weights"]:
        x_b = np.array([feats[key] for key in dsm["feature_names"]], dtype=np.float64)
        z_b = (x_b - np.asarray(dsm["mean"], dtype=np.float64)) / np.asarray(dsm["std"], dtype=np.float64)
        delta_b = 0.20 * float(np.tanh(float(z_b @ np.asarray(dsm["weights"], dtype=np.float64))))

    ru = model["ray_unwarp"]
    c2 = model["cond_spatial2d"]
    c_v = 0.0
    c_r2 = 0.0
    if c2["w_v"]:
        x_c = np.array([feats[key] for key in c2["feature_names"]], dtype=np.float64)
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
            fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            lch = -0.5 * torch.log(1.0 + ((_FULL_U_PIX_T - cx) / fx) ** 2 + ((_FULL_V_PIX_T - cy) / fy) ** 2)
            out_lp.add_(log_cos_ref_c - (lch - float(lch[CROP].mean())), alpha=gamma_ray)
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
        out_lp = out_lp + gamma_ray * ((log_cos_ref - float(log_cos_ref[CROP].mean())) - _centered_log_cos_hat_full(k))
    if c2["w_v"]:
        out_lp = out_lp + c_v * _FULL_V_TILDE + c_r2 * (_FULL_R2 - float(c2["mean_r2_crop"]))
    return clip_depth(np.exp(np.clip(out_lp, np.log(0.001), np.log(10.0))))


def apply_geocalib_model(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    model: dict[str, Any],
    flip_depth: np.ndarray | None = None,
    flip_intrinsics: np.ndarray | None = None,
    conf_orig: np.ndarray | None = None,
    conf_flip: np.ndarray | None = None,
    conf_summary: np.ndarray | None = None,
    cls_orig: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
    cached_features: dict[str, float] | None = None,
    cached_sem: np.ndarray | None = None,
    use_torch_fast: bool = True,
) -> np.ndarray:
    """Apply any GeoCalib-3D baseline or calibrated model with float64 Torch/NumPy parity."""
    p1 = np.asarray(raw_depth, dtype=np.float64)
    if p1.shape != (480, 640) or not np.isfinite(p1).all() or np.any(p1 <= 0):
        raise ValueError("Expected finite positive 480x640 raw depth")
    k1 = np.asarray(pred_intrinsics, dtype=np.float64)
    if k1.shape != (3, 3) or not np.isfinite(k1).all() or k1[0, 0] <= 0 or k1[1, 1] <= 0:
        raise ValueError("Expected valid 3x3 predicted intrinsics")

    kind = model.get("kind")
    if kind == "scalar":
        return clip_depth(p1 * float(model["scale"]))
    if kind == "monotone_spline":
        return apply_spline(p1, model)
    if kind == "geocalib_geo":
        return _apply_geo_only_model(p1, k1, model, cached_features=cached_features, use_torch_fast=use_torch_fast)
    if kind == "geocalib_flip_geo":
        assert flip_depth is not None and flip_intrinsics is not None
        p2 = np.asarray(flip_depth, dtype=np.float64)
        k2_unmir = np.asarray(flip_intrinsics, dtype=np.float64).copy()
        k2_unmir[0, 2] = 639.0 - k2_unmir[0, 2]
        k_avg = 0.5 * (k1 + k2_unmir)
        w_f = float(model.get("w_flip", 0.55))
        p_log = clip_depth(np.exp(w_f * np.log(p1) + (1.0 - w_f) * np.log(p2)))
        return _apply_geo_only_model(p_log, k_avg, model["geo_model"], use_torch_fast=use_torch_fast)
    if kind == "geocalib_ensemble":
        w0, w1 = float(model["weights"][0]), float(model["weights"][1])
        sem0 = cached_sem if isinstance(cached_sem, np.ndarray) else None
        out0 = apply_geocalib_model(
            p1, k1, model["members"][0], flip_depth=flip_depth, flip_intrinsics=flip_intrinsics,
            conf_orig=conf_orig, conf_flip=conf_flip, conf_summary=conf_summary,
            cls_orig=cls_orig, cls_flip=cls_flip, cached_features=cached_features,
            cached_sem=sem0, use_torch_fast=use_torch_fast,
        )
        out1 = apply_geocalib_model(
            p1, k1, model["members"][1], flip_depth=flip_depth, flip_intrinsics=flip_intrinsics,
            conf_orig=conf_orig, conf_flip=conf_flip, conf_summary=conf_summary,
            cls_orig=cls_orig, cls_flip=cls_flip, cached_features=cached_features,
            cached_sem=sem0, use_torch_fast=use_torch_fast,
        )
        return clip_depth(np.exp(w0 * np.log(out0) + w1 * np.log(out1)))
    if kind != "geocalib_3d":
        raise ValueError(f"Unsupported GeoCalib-3D model kind: {kind!r}")

    feats = (
        cached_features
        if cached_features is not None
        else extract_geocalib_features(
            p1, k1, conf_summary=conf_summary, flip_depth=flip_depth, flip_intrinsics=flip_intrinsics,
            conf_orig=conf_orig, conf_flip=conf_flip, cls_orig=cls_orig, cls_flip=cls_flip,
        )
    )

    ff = model["flip_fusion"]
    fmode = ff["mode"]
    if fmode == "learnable":
        assert flip_depth is not None
        p2 = np.asarray(flip_depth, dtype=np.float64)
        w_f = float(ff["w_flip"])
        p_in = np.exp(w_f * np.log(p1) + (1.0 - w_f) * np.log(p2))
    elif fmode == "conf":
        assert flip_depth is not None and conf_orig is not None and conf_flip is not None
        p2 = np.asarray(flip_depth, dtype=np.float64)
        c1 = np.maximum(np.asarray(conf_orig, dtype=np.float64), 1e-4)
        c2 = np.maximum(np.asarray(conf_flip, dtype=np.float64), 1e-4)
        wc1 = c1 / (c1 + c2)
        p_in = np.exp(wc1 * np.log(p1) + (1.0 - wc1) * np.log(p2))
    else:
        p_in = p1

    fs = model["focal_spline"]
    hf = float(_interp_1d_piecewise_np(feats[fs["focal_key"]], np.asarray(fs["x_knots"], dtype=np.float64), np.asarray(fs["y_knots"], dtype=np.float64)))
    fg = model["focal_gate"]
    if fg["enabled"]:
        zg = _std_z(feats, fg)
        logit_g = float(fg["bias"]) + float(zg @ np.asarray(fg["weights"], dtype=np.float64))
        hf *= float(1.0 / (1.0 + np.exp(-logit_g)))

    sr = model["scale_residual"]
    z_sc = _std_z(feats, sr)
    shift = hf + 0.40 * float(np.tanh(float(z_sc @ np.asarray(sr["weights"], dtype=np.float64)) / 0.40))

    sb = model["semantic_branch"]
    x_sem_c = None
    if sb["raw_dims"] > 0:
        assert cached_sem is not None
        x_sem_c = np.asarray(cached_sem, dtype=np.float64) - np.asarray(sb["mean"], dtype=np.float64)
        cap = float(sb["scale_cap"])
        shift += cap * float(np.tanh(float(x_sem_c @ np.asarray(sb["w_scale"], dtype=np.float64)) / cap))
    shift = 0.50 * float(np.tanh(shift / 0.50))

    dsm = model["depth_slope_mod"]
    z_sl = _std_z(feats, dsm)
    slope_lin = float(z_sl @ np.asarray(dsm["weights"], dtype=np.float64))
    if x_sem_c is not None and sb["w_slope"]:
        slope_lin += float(x_sem_c @ np.asarray(sb["w_slope"], dtype=np.float64))
    delta_b = 0.20 * float(np.tanh(slope_lin))

    ir = model["inv_residual"]
    w_inv = float(ir["weight"]) if ir["enabled"] else 0.0

    sp2d = sb["spatial2d"]
    c_v = 0.0
    c_r2 = 0.0
    if x_sem_c is not None and sp2d["enabled"]:
        c_v = 0.14 * float(np.tanh(float(sp2d["a_v0"]) + float(x_sem_c @ np.asarray(sp2d["w_v"], dtype=np.float64))))
        c_r2 = 0.14 * float(np.tanh(float(sp2d["a_r0"]) + float(x_sem_c @ np.asarray(sp2d["w_r"], dtype=np.float64))))

    db = model["depth_branch"]
    dparam = db["param"]

    if use_torch_fast:
        pt = torch.from_numpy(np.ascontiguousarray(p_in))
        lp = torch.log(pt)
        if dparam == "log_spline":
            xk = np.asarray(db["x_knots"], dtype=np.float64)
            yk = np.asarray(db["y_knots"], dtype=np.float64)
            slopes = np.diff(yk) / np.diff(xk)
            dx = np.diff(xk)
            out_lp = float(yk[0]) + float(slopes[0]) * torch.clamp(lp - float(xk[0]), max=float(dx[0]))
            for ki in range(1, len(slopes) - 1):
                out_lp.add_(float(slopes[ki]) * torch.clamp(lp - float(xk[ki]), min=0.0, max=float(dx[ki])))
            out_lp.add_(float(slopes[-1]) * torch.clamp(lp - float(xk[-2]), min=0.0))
        elif dparam == "inv_spline":
            xk = np.asarray(db["x_knots"], dtype=np.float64)
            yk = np.asarray(db["y_knots"], dtype=np.float64)
            slopes = np.diff(yk) / np.diff(xk)
            dx = np.diff(xk)
            invpt = torch.reciprocal(pt)
            dspl = float(yk[0]) + float(slopes[0]) * torch.clamp(invpt - float(xk[0]), max=float(dx[0]))
            for ki in range(1, len(slopes) - 1):
                dspl.add_(float(slopes[ki]) * torch.clamp(invpt - float(xk[ki]), min=0.0, max=float(dx[ki])))
            dspl.add_(float(slopes[-1]) * torch.clamp(invpt - float(xk[-2]), min=0.0))
            out_lp = -torch.log(dspl.clamp_min_(0.04))
        else:
            out_lp = lp + torch.log(float(db["a"]) + float(db["b"]) * pt) - torch.log(1.0 + float(db["c"]) * pt)

        out_lp.add_(shift)
        if delta_b != 0.0:
            out_lp.add_(lp - float(dsm["log_depth_center"]), alpha=delta_b)
        if w_inv != 0.0:
            out_lp.add_(torch.reciprocal(pt) - 0.5, alpha=w_inv)
        if c_v != 0.0 or c_r2 != 0.0:
            out_lp.add_(_FULL_V_TILDE_T, alpha=c_v)
            out_lp.add_(_FULL_R2_T - float(sp2d["mean_r2_crop"]), alpha=c_r2)
        return torch.exp(out_lp.clamp_(float(np.log(0.001)), float(np.log(10.0)))).clamp_(0.001, 10.0).numpy()

    lp = np.log(p_in)
    if dparam == "log_spline":
        out_lp = _interp_1d_piecewise_np(lp, np.asarray(db["x_knots"], dtype=np.float64), np.asarray(db["y_knots"], dtype=np.float64))
    elif dparam == "inv_spline":
        dspl = _interp_1d_piecewise_np(1.0 / p_in, np.asarray(db["x_knots"], dtype=np.float64), np.asarray(db["y_knots"], dtype=np.float64))
        out_lp = -np.log(np.maximum(dspl, 0.04))
    else:
        out_lp = lp + np.log(float(db["a"]) + float(db["b"]) * p_in) - np.log(1.0 + float(db["c"]) * p_in)

    out_lp = out_lp + shift
    if delta_b != 0.0:
        out_lp = out_lp + delta_b * (lp - float(dsm["log_depth_center"]))
    if w_inv != 0.0:
        out_lp = out_lp + w_inv * (1.0 / p_in - 0.5)
    if c_v != 0.0 or c_r2 != 0.0:
        out_lp = out_lp + c_v * _FULL_V_TILDE + c_r2 * (_FULL_R2 - float(sp2d["mean_r2_crop"]))
    return clip_depth(np.exp(np.clip(out_lp, np.log(0.001), np.log(10.0))))


def build_candidate_specs() -> list[GeoCalibSpec]:
    """Return all 26 structured calibration candidates spanning single-pass and 2-pass Flip-TTA tracks."""
    return [
        GeoCalibSpec(name="geo_free_left_focal", direction="deepen", free_left_focal=True, geo_mode="full_cxcy", z_clip=None),
        GeoCalibSpec(name="geo_left_rel_gate", direction="deepen", use_focal_gate=True, geo_mode="extended", z_clip=4.0),
        GeoCalibSpec(name="geo_qknots_robust", direction="deepen", use_quantile_knots=True, geo_mode="clean", z_clip=4.0, loss_mode="robust_huber_111"),

        GeoCalibSpec(name="geo_inv_spline", direction="new_param", depth_param="inv_spline", geo_mode="full_cxcy", z_clip=None),
        GeoCalibSpec(name="geo_inv_residual", direction="new_param", use_inv_residual=True, geo_mode="full_cxcy", z_clip=None),
        GeoCalibSpec(name="geo_rational_depth", direction="new_param", depth_param="rational", geo_mode="full_cxcy", z_clip=None),

        GeoCalibSpec(name="geo_conf_ctrl", direction="new_info", geo_mode="with_conf", z_clip=4.0, sem_mode="none"),
        GeoCalibSpec(name="geosem_cls", direction="new_info", geo_mode="clean", sem_mode="cls", sem_reg=0.06),
        GeoCalibSpec(name="geosem_df", direction="new_info", geo_mode="clean", sem_mode="df", sem_reg=0.06),
        GeoCalibSpec(name="geosem_all_r008", direction="new_info", geo_mode="clean", sem_mode="all_sp", sem_reg=0.08),
        GeoCalibSpec(name="geosem_scale", direction="new_info", geo_mode="clean", sem_mode="all_sp", sem_reg=0.05),
        GeoCalibSpec(name="geosem_pca128", direction="new_info", geo_mode="clean", sem_mode="all_sp", sem_pca_k=128, sem_reg=0.05),

        GeoCalibSpec(name="geosem_sp2d_r005", direction="combine", geo_mode="clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibSpec(name="geosem_sp2d_r008", direction="polish", geo_mode="clean", sem_mode="all_sp", sem_reg=0.08, use_sem_spatial2d=True, sem_sp_reg=0.10),
        GeoCalibSpec(name="geosem_sp2d_pca128", direction="polish", geo_mode="clean", sem_mode="all_sp", sem_pca_k=128, sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibSpec(name="geosem_sp2d_inv_gate", direction="combine", geo_mode="extended", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_inv_residual=True, use_focal_gate=True),
        GeoCalibSpec(name="geosem_sp2d_qknots", direction="polish", geo_mode="clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_quantile_knots=True, loss_mode="robust_huber_111"),
        GeoCalibSpec(name="geocalib_3d", direction="combine", depth_param="inv_spline", geo_mode="clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibSpec(name="geocalib_3d_blend", direction="combine", blend_members=("geosem_sp2d_r005", "geocalib_3d")),

        GeoCalibSpec(name="flip_geo_only", direction="multipass", track="multi_pass_flip_tta", flip_blend_mode="sym_geo_zero_retrain"),
        GeoCalibSpec(name="flip_geo_gated", direction="multipass", track="multi_pass_flip_tta", geo_mode="extended", use_focal_gate=True, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibSpec(name="flip_geosem_sp2d", direction="multipass", track="multi_pass_flip_tta", geo_mode="clean", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibSpec(name="flip_geosem_conf_gate", direction="multipass", track="multi_pass_flip_tta", geo_mode="extended", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_focal_gate=True, flip_blend_mode="conf", use_flip_consistency=True),
        GeoCalibSpec(name="flip_geosem_pca128", direction="multipass", track="multi_pass_flip_tta", geo_mode="clean", sem_mode="all_mp", sem_pca_k=128, sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibSpec(name="geocalib_3d_flip", direction="multipass", track="multi_pass_flip_tta", depth_param="inv_spline", geo_mode="clean", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibSpec(name="geocalib_3d_flip_blend", direction="multipass", track="multi_pass_flip_tta", blend_members=("flip_geosem_sp2d", "geocalib_3d_flip")),
    ]


def prepare_training_bundle(handle, base_cache, train_ids: np.ndarray) -> dict[str, Any]:
    """Load training-only sampled pixels and extracted base geometric features."""
    n = len(train_ids)
    intrinsics = base_cache.intrinsics_for(train_ids).astype(np.float64)
    ratios = np.empty(n, dtype=np.float64)
    samples_spline = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float32)
    targets_spline = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float32)
    samples_p = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float64)
    samples_gt = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.float64)
    samples_row = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.int32)
    samples_col = np.empty((n, SAMPLES_PER_IMAGE), dtype=np.int32)
    base_feats: list[dict[str, float]] = []

    for row, index in enumerate(train_ids):
        idx = int(index)
        pred_full = base_cache[idx].astype(np.float64)
        gt_crop = nyu_depth(handle, idx)[CROP].astype(np.float64)
        p_crop = pred_full[CROP]
        valid = valid_depth(gt_crop)
        ratios[row] = np.median(gt_crop[valid] / p_crop[valid])

        choice_lin = np.random.default_rng(SAMPLE_SEED + idx).choice(valid.sum(), SAMPLES_PER_IMAGE, replace=False)
        samples_spline[row] = p_crop[valid][choice_lin].astype(np.float32)
        targets_spline[row] = gt_crop[valid][choice_lin].astype(np.float32)

        valid_rc = np.argwhere(valid)
        choice_rc = np.random.default_rng(SAMPLE_SEED + idx).choice(len(valid_rc), SAMPLES_PER_IMAGE, replace=False)
        rc = valid_rc[choice_rc]
        samples_row[row] = rc[:, 0]
        samples_col[row] = rc[:, 1]
        samples_p[row] = p_crop[rc[:, 0], rc[:, 1]]
        samples_gt[row] = gt_crop[rc[:, 0], rc[:, 1]]
        base_feats.append(extract_base_geo_features(pred_full, intrinsics[row], None, need_rgb=False))

    return {
        "train_ids": train_ids,
        "intrinsics": intrinsics,
        "ratios": ratios,
        "samples_spline": samples_spline,
        "targets_spline": targets_spline,
        "samples_p": samples_p,
        "samples_gt": samples_gt,
        "samples_row": samples_row,
        "samples_col": samples_col,
        "base_feats": base_feats,
    }


def prepare_full_bundle(handle, cache: VerifiedGeoCalibCache, split_ids: np.ndarray, is_train: bool = True) -> dict[str, Any]:
    """Load split arrays cleanly inside allowed_split; GT/targets built only when is_train=True."""
    n = len(split_ids)
    cls_orig = cache.cls_tokens_for(split_ids)
    cls_flip = cache.cls_tokens_flip_for(split_ids)
    pat_mean = cache.patch_mean_for(split_ids)
    pat_std = cache.patch_std_for(split_ids)
    df_stats = cache.depth_feat_stats_for(split_ids)
    conf_sum = cache.confidence_summary_for(split_ids)
    k1_all = cache.intrinsics_for(split_ids).astype(np.float64)
    k2_all = cache.intrinsics_flip_for(split_ids).astype(np.float64)

    p1_full = [cache[int(i)].astype(np.float64) for i in split_ids]
    p2_full = [cache.depth_flip(int(i)).astype(np.float64) for i in split_ids]
    c1_full = [np.maximum(cache.confidence(int(i)).astype(np.float64), 1e-4) for i in split_ids]
    c2_full = [np.maximum(cache.confidence_flip(int(i)).astype(np.float64), 1e-4) for i in split_ids]

    geo_feats: list[dict[str, float]] = []
    for row in range(n):
        geo_feats.append(
            extract_geocalib_features(
                raw_depth=p1_full[row],
                pred_intrinsics=k1_all[row],
                conf_summary=conf_sum[row],
                flip_depth=p2_full[row],
                flip_intrinsics=k2_all[row],
                conf_orig=c1_full[row],
                conf_flip=c2_full[row],
                cls_orig=cls_orig[row],
                cls_flip=cls_flip[row],
            )
        )

    sem_mats = {
        mode: assemble_semantic_matrix(mode, cls_orig, pat_mean, pat_std, df_stats, cls_flip)
        for mode in SEM_DIMS
    }

    bundle: dict[str, Any] = {
        "ids": split_ids,
        "intrinsics_orig": k1_all,
        "intrinsics_flip": k2_all,
        "p1_full": p1_full,
        "p2_full": p2_full,
        "c1_full": c1_full,
        "c2_full": c2_full,
        "geo_feats": geo_feats,
        "sem_mats": sem_mats,
    }
    if is_train:
        base_data = prepare_training_bundle(handle, cache.base, split_ids)
        samples_p2 = np.empty_like(base_data["samples_p"])
        samples_w1 = np.empty_like(base_data["samples_p"])
        for row in range(n):
            p2c = p2_full[row][CROP]
            wc = c1_full[row][CROP] / (c1_full[row][CROP] + c2_full[row][CROP])
            rr = base_data["samples_row"][row]
            cc = base_data["samples_col"][row]
            samples_p2[row] = p2c[rr, cc]
            samples_w1[row] = wc[rr, cc]
        bundle["base_data"] = base_data
        bundle["samples_p2"] = samples_p2
        bundle["samples_w1"] = samples_w1
    return bundle


def _validate_single_model_params(name: str, model: dict[str, Any]) -> None:
    kind = model.get("kind")
    if kind == "scalar":
        val = float(model["scale"])
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"Model {name!r} has invalid positive scalar")
        return
    if kind == "monotone_spline":
        yk = np.asarray(model["y_knots"], dtype=np.float64)
        if not np.isfinite(yk).all() or np.any(np.diff(yk) <= 0):
            raise ValueError(f"Model {name!r} monotone spline y_knots must be finite and strictly increasing")
        return
    if kind == "geocalib_geo":
        ds = model["depth_spline"]
        yk = np.asarray(ds["y_knots"], dtype=np.float64)
        if not np.isfinite(yk).all() or np.any(np.diff(yk) <= 0):
            raise ValueError(f"Model {name!r} depth_spline y_knots must be finite and strictly increasing")
        if not np.isfinite(np.asarray(model["scale_residual"]["weights"], dtype=np.float64)).all():
            raise ValueError(f"Model {name!r} has nonfinite scale_residual weights")
        return
    if kind == "geocalib_flip_geo":
        _validate_single_model_params(f"{name}.geo_model", model["geo_model"])
        return
    if kind == "geocalib_ensemble":
        for i, m in enumerate(model["members"]):
            _validate_single_model_params(f"{name}.members[{i}]", m)
        return
    if kind != "geocalib_3d":
        raise ValueError(f"Unknown model kind {kind!r} for {name!r}")
    db = model["depth_branch"]
    if db["param"] in ("log_spline", "inv_spline"):
        yk = np.asarray(db["y_knots"], dtype=np.float64)
        if not np.isfinite(yk).all() or np.any(np.diff(yk) <= 0):
            raise ValueError(f"Model {name!r} depth_branch y_knots must be finite and strictly increasing")
    if not np.isfinite(np.asarray(model["scale_residual"]["weights"], dtype=np.float64)).all():
        raise ValueError(f"Model {name!r} has nonfinite scale_residual weights")


def validate_frozen_geocalib(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate self-contained GeoCalib_3D frozen artifact strictly."""
    if not isinstance(payload, dict) or payload.get("schema") != "GeoCalib_3D":
        raise ValueError("Expected schema == 'GeoCalib_3D'")
    if payload.get("complete") is not True:
        raise ValueError("Frozen GeoCalib_3D artifact must have complete=True")
    if payload.get("test_read") is not False:
        raise ValueError("Frozen GeoCalib_3D artifact must certify test_read=False")
    train_ids = payload.get("train_ids")
    if (
        not isinstance(train_ids, list)
        or len(train_ids) != 795
        or len(set(train_ids)) != 795
        or any(type(i) is not int or not 0 <= i < 1449 for i in train_ids)
        or payload.get("training_scenes") != 249
    ):
        raise ValueError("Invalid official 795-image / 249-scene train_ids in frozen model artifact")
    for key in ("dataset_sha256", "split_sha256"):
        val = payload.get(key)
        if not isinstance(val, str) or re.fullmatch(r"[0-9a-f]{64}", val) is None:
            raise ValueError(f"Missing or malformed 64-hex digest: {key}")
    models = payload.get("models", {})
    win_sp = payload.get("primary_single_pass_method")
    win_mp = payload.get("primary_multipass_method")
    if win_sp not in models or win_mp not in models:
        raise ValueError("Primary single-pass or multi-pass model missing from frozen artifact")
    for m_name in payload.get("locked_eval_order", []):
        if m_name not in models:
            raise ValueError(f"Locked eval method {m_name!r} missing from frozen models")
    for m_name, m_obj in models.items():
        _validate_single_model_params(m_name, m_obj)
    return payload


def load_frozen_geocalib(path: Path | str, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load and SHA-256-verify self-contained GeoCalib_3D frozen artifact."""
    target = Path(path)
    actual_sha = file_sha256(target)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"Frozen model SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return validate_frozen_geocalib(payload)


def _scene_bootstrap_vs_ref(
    scores: np.ndarray,
    ref_col: int,
    scenes: np.ndarray,
    repeats: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return gains.mean(axis=0), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def _scene_metric_bootstrap_ci(
    scores: np.ndarray,
    scenes: np.ndarray,
    repeats: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
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


def _fit_blend_weight_strict(
    m0: dict[str, Any],
    m1: dict[str, Any],
    bundle: dict[str, Any],
    tr_idx: np.ndarray,
    s_per_img: int = 512,
) -> float:
    base = bundle["base_data"]
    sm0 = m0["semantic_branch"]["sem_mode"]
    sm1 = m1["semantic_branch"]["sem_mode"]
    x0 = bundle["sem_mats"][sm0]
    x1 = bundle["sem_mats"][sm1]

    rr = base["samples_row"][tr_idx, :s_per_img]
    cc = base["samples_col"][tr_idx, :s_per_img]
    gt = base["samples_gt"][tr_idx, :s_per_img]

    vals0 = np.empty((len(tr_idx), s_per_img), dtype=np.float64)
    vals1 = np.empty((len(tr_idx), s_per_img), dtype=np.float64)
    for k, row in enumerate(tr_idx):
        d0 = apply_geocalib_model(
            bundle["p1_full"][row], bundle["intrinsics_orig"][row], m0,
            flip_depth=bundle["p2_full"][row], flip_intrinsics=bundle["intrinsics_flip"][row],
            conf_orig=bundle["c1_full"][row], conf_flip=bundle["c2_full"][row],
            cached_features=bundle["geo_feats"][row], cached_sem=x0[row] if x0.shape[1] else None,
        )[CROP]
        d1 = apply_geocalib_model(
            bundle["p1_full"][row], bundle["intrinsics_orig"][row], m1,
            flip_depth=bundle["p2_full"][row], flip_intrinsics=bundle["intrinsics_flip"][row],
            conf_orig=bundle["c1_full"][row], conf_flip=bundle["c2_full"][row],
            cached_features=bundle["geo_feats"][row], cached_sem=x1[row] if x1.shape[1] else None,
        )[CROP]
        vals0[k] = d0[rr[k], cc[k]]
        vals1[k] = d1[rr[k], cc[k]]

    lp0, lp1 = np.log(vals0), np.log(vals1)
    best_a, best_ar = 0.5, float("inf")
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        pred_s = clip_depth(np.exp(alpha * lp0 + (1.0 - alpha) * lp1))
        ar = float(np.mean(np.abs(pred_s - gt) / gt))
        if ar < best_ar:
            best_ar = ar
            best_a = alpha
    return best_a


def evaluate_cv_for_seed(
    handle,
    bundle: dict[str, Any],
    scenes: np.ndarray,
    specs: list[GeoCalibSpec],
    seed: int,
    max_iter: int = 75,
    complexity_out: dict[str, tuple[int, float]] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Run 5-fold scene-disjoint CV on 795 training images only for all specs."""
    base = bundle["base_data"]
    train_ids = bundle["ids"]
    n = len(train_ids)
    folds = scene_folds_seeded(scenes, count=5, seed=seed)
    gts = [nyu_depth(handle, int(i)) for i in train_ids]

    names = ["raw", "global_scale", "monotone_spline", PRIMARY_GEO_METHOD, *[s.name for s in specs]]
    scores = np.full((n, len(names), 3), np.nan, dtype=np.float64)
    fold_ids = np.empty(n, dtype=int)

    single_specs = [s for s in specs if s.blend_members is None and s.flip_blend_mode != "sym_geo_zero_retrain"]
    blend_specs = [s for s in specs if s.blend_members is not None]
    sym_geo_specs = [s for s in specs if s.flip_blend_mode == "sym_geo_zero_retrain"]
    fold_dofs: dict[str, list[tuple[int, float]]] = {s.name: [] for s in specs}

    for f_idx, val in enumerate(folds):
        t0 = perf_counter()
        tr = np.setdiff1d(np.arange(n), val)
        scale = fit_scale(base["ratios"][tr])
        spline = fit_monotone_spline(base["samples_spline"][tr], base["targets_spline"][tr], scale)

        tr_p1 = base["samples_p"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_p2 = bundle["samples_p2"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_w1 = bundle["samples_w1"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_gt = base["samples_gt"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_row = base["samples_row"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_col = base["samples_col"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_feats = [bundle["geo_feats"][i] for i in tr]
        tr_ratios = base["ratios"][tr]

        geo_model = fit_geo_only_model(
            spec=GEO_ONLY_PRIMARY_SPEC,
            samples_p=tr_p1,
            samples_gt=tr_gt,
            samples_row=tr_row,
            samples_col=tr_col,
            features=[base["base_feats"][i] for i in tr],
            intrinsics=base["intrinsics"][tr],
            ratios=tr_ratios,
            max_iter=65,
        )

        def _fit_one(sp: GeoCalibSpec) -> tuple[str, dict[str, Any]]:
            use_p2 = tr_p2 if sp.flip_blend_mode != "none" else None
            use_w1 = tr_w1 if sp.flip_blend_mode == "conf" else None
            m = fit_geocalib_model(
                spec=sp,
                samples_p1=tr_p1,
                samples_gt=tr_gt,
                samples_row=tr_row,
                samples_col=tr_col,
                features=tr_feats,
                ratios=tr_ratios,
                x_sem_raw=bundle["sem_mats"][sp.sem_mode][tr],
                samples_p2=use_p2,
                samples_w1=use_w1,
                max_iter=max_iter,
            )
            return sp.name, m

        with ThreadPoolExecutor(max_workers=4) as pool:
            fitted_map = dict(pool.map(_fit_one, single_specs))

        for sp in sym_geo_specs:
            fitted_map[sp.name] = {"kind": "geocalib_flip_geo", "geo_model": geo_model, "w_flip": 0.55}

        for sp in blend_specs:
            n0, n1 = sp.blend_members  # type: ignore[misc]
            m0, m1 = fitted_map[n0], fitted_map[n1]
            alpha = _fit_blend_weight_strict(m0, m1, bundle, tr)
            fitted_map[sp.name] = {
                "kind": "geocalib_ensemble",
                "members": [m0, m1],
                "weights": [alpha, 1.0 - alpha],
                "blend_space": "log",
            }

        for sp in specs:
            fold_dofs[sp.name].append(count_model_parameters(fitted_map[sp.name]))

        for row in val:
            fold_ids[row] = f_idx
            p1 = bundle["p1_full"][row]
            p2 = bundle["p2_full"][row]
            c1 = bundle["c1_full"][row]
            c2 = bundle["c2_full"][row]
            k1 = bundle["intrinsics_orig"][row]
            k2 = bundle["intrinsics_flip"][row]
            gt = gts[row]

            scores[row, 0] = evaluate_image(clip_depth(p1), gt)
            scores[row, 1] = evaluate_image(clip_depth(p1 * scale), gt)
            scores[row, 2] = evaluate_image(apply_spline(p1, spline), gt)
            scores[row, 3] = evaluate_image(
                apply_geocalib_model(p1, k1, geo_model, cached_features=base["base_feats"][row]), gt
            )

            for s_idx, sp in enumerate(specs):
                m_obj = fitted_map[sp.name]
                if sp.blend_members is not None:
                    sm = fitted_map[sp.blend_members[0]]["semantic_branch"]["sem_mode"]
                elif sp.sem_mode != "none":
                    sm = sp.sem_mode
                else:
                    sm = "none"
                x_row = bundle["sem_mats"][sm][row] if bundle["sem_mats"][sm].shape[1] else None
                out = apply_geocalib_model(
                    p1, k1, m_obj,
                    flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2,
                    cached_features=bundle["geo_feats"][row], cached_sem=x_row,
                )
                scores[row, 4 + s_idx] = evaluate_image(out, gt)

        print(f"CV seed={seed} fold {f_idx + 1}/5 finished in {perf_counter() - t0:.1f}s", flush=True)

    if complexity_out is not None:
        complexity_out["raw"] = (0, 0.0)
        complexity_out["global_scale"] = (1, 1.0)
        complexity_out["monotone_spline"] = (6, 6.0)
        complexity_out[PRIMARY_GEO_METHOD] = (25, 25.0)
        for nm, pairs in fold_dofs.items():
            complexity_out[nm] = (pairs[0][0], float(np.mean([p[1] for p in pairs])))

    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite OOF scores in evaluate_cv_for_seed")
    return scores, names, fold_ids


def freeze_full_training_models(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Fit all locked baseline, geometric, semantic, and flip-TTA models on all 795 training images."""
    torch.set_num_threads(2)
    cache = load_geocalib_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=True, verify_scenes=True)
    train_ids = cache.train_ids
    spec_by_name = {s.name: s for s in build_candidate_specs()}

    with open_nyu(data_root) as handle:
        train_scenes = np.unique(scene_names(handle)[train_ids])
        bundle = prepare_full_bundle(handle, cache, train_ids, is_train=True)
    base = bundle["base_data"]

    scale = fit_scale(base["ratios"])
    spline = fit_monotone_spline(base["samples_spline"], base["targets_spline"], scale)
    geo_model = fit_geo_only_model(
        spec=GEO_ONLY_PRIMARY_SPEC,
        samples_p=base["samples_p"],
        samples_gt=base["samples_gt"],
        samples_row=base["samples_row"],
        samples_col=base["samples_col"],
        features=base["base_feats"],
        intrinsics=base["intrinsics"],
        ratios=base["ratios"],
        max_iter=120,
    )

    locked_models: dict[str, Any] = {
        "raw": {"kind": "scalar", "scale": 1.0},
        "global_scale": {"kind": "scalar", "scale": scale},
        "monotone_spline": spline,
        PRIMARY_GEO_METHOD: geo_model,
        "flip_geo_only": {"kind": "geocalib_flip_geo", "geo_model": geo_model, "w_flip": 0.55},
    }

    for nm in ("geo_qknots_robust", "geo_inv_residual", "geosem_scale", "geocalib_3d", "flip_geo_gated", "geocalib_3d_flip"):
        sp = spec_by_name[nm]
        p2u = bundle["samples_p2"] if sp.flip_blend_mode != "none" else None
        w1u = bundle["samples_w1"] if sp.flip_blend_mode == "conf" else None
        locked_models[nm] = fit_geocalib_model(
            spec=sp,
            samples_p1=base["samples_p"],
            samples_gt=base["samples_gt"],
            samples_row=base["samples_row"],
            samples_col=base["samples_col"],
            features=bundle["geo_feats"],
            ratios=base["ratios"],
            x_sem_raw=bundle["sem_mats"][sp.sem_mode],
            samples_p2=p2u,
            samples_w1=w1u,
            max_iter=110,
        )

    sp_raw, sp_dof = count_model_parameters(locked_models[PRIMARY_SINGLE_PASS_METHOD])
    mp_raw, mp_dof = count_model_parameters(locked_models[PRIMARY_MULTIPASS_METHOD])
    pkg_dir = Path(__file__).resolve().parent
    src_hashes = {
        f"depthlab/{fn}": file_sha256(pkg_dir / fn)
        for fn in ("geocalib_io.py", "geocalib_cache.py", "geocalib_diagnose.py", "geocalib_model.py", "metrics.py")
        if (pkg_dir / fn).is_file()
    }

    payload = {
        "schema": "GeoCalib_3D",
        "complete": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
        "primary_single_pass_method": PRIMARY_SINGLE_PASS_METHOD,
        "primary_single_pass_display": "GeoCalib-3D (Single-Pass)",
        "primary_single_pass_raw_params": sp_raw,
        "primary_single_pass_effective_dof": sp_dof,
        "primary_multipass_method": PRIMARY_MULTIPASS_METHOD,
        "primary_multipass_display": "GeoCalib-3D + Flip-TTA",
        "primary_multipass_raw_params": mp_raw,
        "primary_multipass_effective_dof": mp_dof,
        "locked_eval_order": list(LOCKED_EVAL_ORDER),
        "train_ids": train_ids.tolist(),
        "training_images": int(len(train_ids)),
        "training_scenes": int(len(train_scenes)),
        "dataset_sha256": cache.hashes["nyu_mat_sha256"],
        "split_sha256": cache.hashes["splits_sha256"],
        "cache_sha256": cache.hashes,
        "source_sha256": src_hashes,
        "models": {k: locked_models[k] for k in LOCKED_EVAL_ORDER},
    }
    write_json_lf(output_path, payload)
    return validate_frozen_geocalib(payload)


def run_cv_study(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run training-only 5-fold scene-CV across all 26 candidates + 4-seed stability."""
    torch.set_num_threads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_geocalib_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids
    specs = build_candidate_specs()
    spec_by_name = {s.name: s for s in specs}

    with open_nyu(data_root) as handle:
        scenes = scene_names(handle)[train_ids]
        bundle = prepare_full_bundle(handle, cache, train_ids, is_train=True)
        complexity_map: dict[str, tuple[int, float]] = {}
        scores_canon, names_canon, fold_ids = evaluate_cv_for_seed(
            handle, bundle, scenes, specs, seed=SEED, max_iter=75, complexity_out=complexity_map
        )

    geo_idx = names_canon.index(PRIMARY_GEO_METHOD)
    mean_scores = scores_canon.mean(axis=0)
    _, lo_g, hi_g = _scene_bootstrap_vs_ref(scores_canon, geo_idx, scenes)
    direction_vec = np.array([-1.0, -1.0, 1.0])
    rows = []
    for col, cand in enumerate(names_canon):
        sp_obj = spec_by_name.get(cand)
        track = sp_obj.track if sp_obj is not None else "single_pass"
        raw_p, eff_dof = complexity_map[cand]
        gains_g = (mean_scores[col] - mean_scores[geo_idx]) * direction_vec
        all5 = bool(
            np.all(
                [
                    ((scores_canon[fold_ids == f, col].mean(axis=0) - scores_canon[fold_ids == f, geo_idx].mean(axis=0)) * direction_vec) > 0
                    for f in range(5)
                ]
            )
        )
        rows.append({
            "method": cand,
            "track": track,
            "backbone_passes": 2 if track == "multi_pass_flip_tta" else 1,
            "direction": sp_obj.direction if sp_obj is not None else "baseline",
            "raw_params": int(raw_p),
            "effective_dof": float(eff_dof),
            "abs_rel": float(mean_scores[col, 0]),
            "rmse": float(mean_scores[col, 1]),
            "delta1": float(mean_scores[col, 2]),
            "gain_vs_geo_abs_rel": float(gains_g[0]),
            "gain_vs_geo_rmse": float(gains_g[1]),
            "gain_vs_geo_delta1": float(gains_g[2]),
            "ci95_lo_geo_abs_rel": float(lo_g[col, 0]),
            "ci95_hi_geo_abs_rel": float(hi_g[col, 0]),
            "all_5_folds_beat_geo": all5,
        })
    write_csv_lf(output_dir / "oof_ablation_summary.csv", pd.DataFrame(rows))
    report = {
        "scope": "official795_training_only",
        "test_read": False,
        "images": int(len(train_ids)),
        "scenes": int(len(np.unique(scenes))),
        "canonical_seed": SEED,
        "multi_seeds": list(MULTI_SEEDS),
        "primary_single_pass_method": PRIMARY_SINGLE_PASS_METHOD,
        "primary_multipass_method": PRIMARY_MULTIPASS_METHOD,
    }
    write_json_lf(output_dir / "cv_selection_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cv_p = sub.add_parser("cv", help="Run training-only 5-fold scene CV study")
    cv_p.add_argument("--data-root", type=Path, required=True)
    cv_p.add_argument("--orig-preds", type=Path, required=True)
    cv_p.add_argument("--aux-dir", type=Path, required=True)
    cv_p.add_argument("--flip-dir", type=Path, required=True)
    cv_p.add_argument("--output", type=Path, required=True)

    fr_p = sub.add_parser("freeze", help="Full-795 training-only model fit and JSON freeze")
    fr_p.add_argument("--data-root", type=Path, required=True)
    fr_p.add_argument("--orig-preds", type=Path, required=True)
    fr_p.add_argument("--aux-dir", type=Path, required=True)
    fr_p.add_argument("--flip-dir", type=Path, required=True)
    fr_p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "cv":
        run_cv_study(args.data_root, args.orig_preds, args.aux_dir, args.flip_dir, args.output)
    elif args.mode == "freeze":
        freeze_full_training_models(args.data_root, args.orig_preds, args.aux_dir, args.flip_dir, args.output)


if __name__ == "__main__":
    main()
