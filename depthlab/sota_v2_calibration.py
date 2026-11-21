"""Round 2 GeoCalib-3D v2: Dual-track geometric, inverse-depth & semantic calibration.

Tracks are strictly separated:
  - Track A (`single_pass`): Uses only a single RGB forward pass of frozen UniDepthV2 ViT-S
    (depth, predicted K, and internal encoder/decoder representations), zero extra passes.
  - Track B (`multi_pass_flip_tta`): Runs 2 passes (original + horizontal-flip TTA) and
    exploits flip-consistency + confidence/log-depth fusion with explicit pass disclosure.

Zero ground-truth intrinsics K_true are ever used. All candidate screening, multi-seed
stability checks, and full-795 parameter freezing strictly forbid reading splits["test"].
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .calibration import (
    BOOTSTRAP_SEED,
    SEED,
    apply_curve,
    apply_residual,
    fit_curve,
    fit_residual,
    fit_scale,
    residual_targets,
)
from .data import nyu_depth, open_nyu, scene_names
from .frozen import load_frozen
from .metrics import CROP, METRIC_NAMES, clip_depth, evaluate_image
from .sota_calibration import (
    CROP_SUB,
    CV_SAMPLES_PER_IMAGE,
    DEPTH_KNOTS,
    FOCAL_ANCHOR_IDX,
    FOCAL_KNOTS,
    GEO_CXCY_NAMES,
    LEARNING_CURVE_FRACTIONS,
    LEARNING_CURVE_SEED,
    LOG_DEPTH_CENTER,
    PRIMARY_SOTA_METHOD,
    SAMPLES_PER_IMAGE,
    SLOPE_FULL_NAMES,
    SPATIAL3D_NAMES,
    _CROP_R2,
    _CROP_V_TILDE,
    _FULL_R2,
    _FULL_R2_T,
    _FULL_V_TILDE,
    _FULL_V_TILDE_T,
    _MEAN_R2_CROP,
    _feature_matrix,
    _interp_1d_piecewise_np,
    _interp_1d_piecewise_torch,
    _loss_components_torch,
    _scene_bootstrap_vs_ref,
    _scene_metric_bootstrap_ci,
    _standardize_apply,
    _standardize_fit,
    apply_sota_model,
    build_candidate_specs,
    extract_sota_features,
    fit_joint_sota_model,
    prepare_sota_training,
)
from .sota_io import file_sha256, scene_folds_seeded, write_csv_lf, write_json_lf
from .sota_v2_cache import CONF_SUMMARY_NAMES, VerifiedV2Cache, load_v2_cache
from .sota_v2_diagnose import compute_flip_features


INV_DEPTH_KNOTS = np.array([1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 1.5, 1.0 / 0.7, 1.0 / 0.3], dtype=np.float64)
FOCAL_KNOTS_Q7 = np.array([0.35, 0.58, 0.78, 0.98, 1.14, 1.22, 1.35], dtype=np.float64)
FOCAL_Q7_ANCHOR_IDX = 4
MULTI_SEEDS = (20261031, 42, 100, 20261121)

GEO_V2_CLEAN_NAMES = ("cy_off", *SPATIAL3D_NAMES)
GEO_V2_EXT_NAMES = (*GEO_V2_CLEAN_NAMES, "frac_far_4m", "left_hinge_x_log_spany", "left_hinge_x_frac_far")
GEO_V2_CONF_NAMES = (*GEO_CXCY_NAMES, *SPATIAL3D_NAMES, *CONF_SUMMARY_NAMES)
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


@dataclass(frozen=True)
class GeoCalibV2Spec:
    """Declarative specification for any Gate 2 Round 2 model candidate."""

    name: str
    direction: str  # "baseline" | "deepen" | "new_param" | "new_info" | "combine" | "polish" | "multipass"
    track: str = "single_pass"  # "single_pass" | "multi_pass_flip_tta"
    depth_param: str = "log_spline"  # "log_spline" | "inv_spline" | "rational"
    use_quantile_knots: bool = False
    use_inv_residual: bool = False
    free_left_focal: bool = False
    use_focal_gate: bool = False
    geo_mode: str = "v2_clean"  # "v1" | "v2_clean" | "v2_ext" | "v2_conf"
    z_clip: float | None = 4.0
    sem_mode: str = "none"  # "none" | "cls" | "df" | "cls_pm_df" | "all_sp" | "all_mp"
    sem_pca_k: int | None = None
    sem_reg: float = 0.05
    sem_scale_cap: float = 0.40
    use_sem_spatial2d: bool = False
    sem_sp_reg: float = 0.08
    use_sem_slope: bool = False
    loss_mode: str = "balanced_111"  # "balanced_111" | "robust_huber_111"
    flip_blend_mode: str = "none"  # "none" | "learnable" | "conf" | "sym_v1_zero_retrain"
    use_flip_consistency: bool = False
    blend_members: tuple[str, str] | None = None


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


def extract_v2_geo_features(
    raw_depth: np.ndarray,
    pred_intrinsics: np.ndarray,
    cached_v1_feat: dict[str, float] | None = None,
    conf_summary: np.ndarray | None = None,
    flip_depth: np.ndarray | None = None,
    flip_intrinsics: np.ndarray | None = None,
    conf_orig: np.ndarray | None = None,
    conf_flip: np.ndarray | None = None,
    cls_orig: np.ndarray | None = None,
    cls_flip: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract RGB-only single-pass + optional 2-pass flip-consistency geometry features."""
    feats = dict(cached_v1_feat) if cached_v1_feat is not None else extract_sota_features(raw_depth, pred_intrinsics, None, need_rgb=False)
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


def _geo_feature_names(spec: GeoCalibV2Spec) -> tuple[str, ...]:
    if spec.geo_mode == "v1":
        base: tuple[str, ...] = (*GEO_CXCY_NAMES, *SPATIAL3D_NAMES)
    elif spec.geo_mode == "v2_clean":
        base = GEO_V2_CLEAN_NAMES
    elif spec.geo_mode == "v2_ext":
        base = GEO_V2_EXT_NAMES
    elif spec.geo_mode == "v2_conf":
        base = GEO_V2_CONF_NAMES
    else:
        raise ValueError(f"Unsupported geo_mode: {spec.geo_mode!r}")
    if spec.use_flip_consistency:
        base = (*base, *FLIP_CONSISTENCY_NAMES)
    return base


def _gate_feature_names(spec: GeoCalibV2Spec) -> tuple[str, ...]:
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
        # Ensure minimum positive spacing >= 0.04 in inverse-meter units
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


def _loss_components_v2_torch(pred: torch.Tensor, gt: torch.Tensor, loss_mode: str) -> torch.Tensor:
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


class _JointGeoCalibV2Module(torch.nn.Module):
    """Differentiable float64 Round 2 joint calibrator supporting both tracks and all ablations."""

    def __init__(
        self,
        spec: GeoCalibV2Spec,
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

        # 1. Depth curve parameters
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

        # 2. Focal spline + optional free left-tail slope
        n_fseg = len(focal_knots) - 1
        self.raw_focal_slopes = torch.nn.Parameter(torch.full((n_fseg,), -2.0, dtype=torch.float64))
        if spec.free_left_focal:
            self.raw_free_left_slope = torch.nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

        # 3. Left-tail focal reliability gate
        if n_gate > 0:
            self.b_gate = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
            self.w_gate = torch.nn.Parameter(torch.zeros(n_gate, dtype=torch.float64))

        # 4. Geometric scale + depth-slope modulation
        self.w_scale = torch.nn.Parameter(torch.zeros(n_geo, dtype=torch.float64))
        self.w_slope = torch.nn.Parameter(torch.zeros(n_slope, dtype=torch.float64))

        # 5. Hybrid inverse-depth residual
        if spec.use_inv_residual:
            self.w_inv = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # 6. Regularized internal semantic branch (fit inside tr fold only)
        if d_sem > 0:
            self.w_sem_s = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
            if spec.use_sem_spatial2d:
                self.a_v0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
                self.a_r0 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
                self.w_sem_v = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
                self.w_sem_r = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))
            if spec.use_sem_slope:
                self.w_sem_b = torch.nn.Parameter(torch.zeros(d_sem, dtype=torch.float64))

        # 7. Learnable flip fusion weight
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


def _estimate_ridge_dof(
    x_sem_tr: np.ndarray,
    h_scale_curvature: float,
    reg_coeff: float,
) -> float:
    """Compute Gauss-Newton effective degrees of freedom tr(H) from centered training semantics."""
    if x_sem_tr.shape[1] == 0 or reg_coeff <= 0.0:
        return 0.0
    n_tr = max(len(x_sem_tr), 1)
    svals = np.linalg.svd(x_sem_tr, full_matrices=False, compute_uv=False)
    eig = max(float(h_scale_curvature), 0.0) * (svals ** 2) / n_tr
    dof = float(np.sum(eig / (eig + 2.0 * float(reg_coeff) + 1e-12)))
    return dof


def fit_joint_v2_model(
    spec: GeoCalibV2Spec,
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
    """Fit a joint float64 GeoCalib-3D v2 model exclusively on training samples."""
    n = len(features)
    init_scale = float(np.clip(np.median(ratios), 0.55, 1.85))

    # Knot grids computed strictly inside training fold
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

    # Fold-internal semantic centering & optional SVD PCA projection
    d_sem_orig = x_sem_raw.shape[1]
    if d_sem_orig > 0:
        mu_sem = x_sem_raw.mean(axis=0)
        x_sem_c = x_sem_raw - mu_sem
        if spec.sem_pca_k is not None and spec.sem_pca_k < min(x_sem_c.shape[0], x_sem_c.shape[1]):
            _, _, vt = np.linalg.svd(x_sem_c, full_matrices=False)
            proj_k = vt[: spec.sem_pca_k].T  # (D_orig, K)
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
        ref = _loss_components_v2_torch((p1_t * init_scale).clamp(0.001, 10.0), gt_t, spec.loss_mode).clamp_min(1e-9)

    def _run(lr: float) -> tuple[_JointGeoCalibV2Module, list[float]]:
        mod = _JointGeoCalibV2Module(
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
            comp = _loss_components_v2_torch(pred, gt_t, spec.loss_mode) / ref
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
        # Estimate scalar Gauss-Newton curvature h_bar for exact effective DoF reporting
        sem_dof = 0.0
        sem_sp_dof = 0.0
        sem_slope_dof = 0.0
        if d_sem_orig > 0:
            pred0 = mod(p1_t, p2_t, w1_t, f_t, zg_t, zsl_t, zgt_t, xsem_t, vt_t, r2_t)
            eps = 1e-3
            l0 = float((_loss_components_v2_torch(pred0, gt_t, spec.loss_mode) / ref).sum())
            lp_up = float((_loss_components_v2_torch((pred0 * float(np.exp(eps))).clamp(0.001, 10.0), gt_t, spec.loss_mode) / ref).sum())
            lp_dn = float((_loss_components_v2_torch((pred0 * float(np.exp(-eps))).clamp(0.001, 10.0), gt_t, spec.loss_mode) / ref).sum())
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
        w_flip_v = float(torch.sigmoid(mod.raw_w_flip)) if spec.flip_blend_mode == "learnable" else (0.55 if spec.flip_blend_mode != "none" else 1.0)

        # Collapse PCA weights W_eff = V_k @ w_k directly in original D_sem space
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

    raw_param_count = count_v2_raw_parameters_from_spec(spec)
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
        "kind": "geocalib_3d_v2",
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
        "flip_fusion": {
            "mode": spec.flip_blend_mode,
            "w_flip": w_flip_v,
        },
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


def count_v2_raw_parameters_from_spec(spec: GeoCalibV2Spec) -> int:
    """Count exact tunable scalar parameters in a GeoCalibV2Spec before regularization shrinkage."""
    if spec.flip_blend_mode == "sym_v1_zero_retrain":
        return 25
    if spec.blend_members is not None:
        return 1  # convex blend weight on top of 2 member models
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


def count_v2_parameters(model: dict[str, Any]) -> tuple[int, float]:
    """Return (raw_params, effective_dof) for any exported v1 or v2 model dict."""
    kind = model.get("kind")
    if kind == "geocalib_3d_v2":
        c = model["complexity"]
        return int(c["raw_params"]), float(c["effective_dof"])
    if kind == "geocalib_v2_ensemble":
        m0_r, m0_d = count_v2_parameters(model["members"][0])
        m1_r, m1_d = count_v2_parameters(model["members"][1])
        return int(m0_r + m1_r + 1), float(m0_d + m1_d + 1.0)
    if kind == "geocalib_v2_sym_v1_wrapper":
        return 25, 25.0
    raise ValueError(f"Unknown model kind: {kind!r}")


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


def apply_sota_v2_model(
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
    """Single-pass or 2-pass Flip-TTA inference matching PyTorch training float64 numerics."""
    p1 = np.asarray(raw_depth, dtype=np.float64)
    if p1.shape != (480, 640) or not np.isfinite(p1).all() or np.any(p1 <= 0):
        raise ValueError("Expected finite positive 480x640 raw depth")
    k1 = np.asarray(pred_intrinsics, dtype=np.float64)
    if k1.shape != (3, 3) or not np.isfinite(k1).all() or k1[0, 0] <= 0 or k1[1, 1] <= 0:
        raise ValueError("Expected valid 3x3 predicted intrinsics")

    kind = model.get("kind")
    if kind == "geocalib_v2_sym_v1_wrapper":
        assert flip_depth is not None and flip_intrinsics is not None
        p2 = np.asarray(flip_depth, dtype=np.float64)
        k2_unmir = np.asarray(flip_intrinsics, dtype=np.float64).copy()
        k2_unmir[0, 2] = 639.0 - k2_unmir[0, 2]
        k_avg = 0.5 * (k1 + k2_unmir)
        w_f = float(model.get("w_flip", 0.55))
        p_log = clip_depth(np.exp(w_f * np.log(p1) + (1.0 - w_f) * np.log(p2)))
        return apply_sota_model(p_log, k_avg, model["v1_model"], use_torch_fast=use_torch_fast)

    if kind == "geocalib_v2_ensemble":
        w0, w1 = float(model["weights"][0]), float(model["weights"][1])
        sem0 = cached_sem if isinstance(cached_sem, np.ndarray) else None
        out0 = apply_sota_v2_model(
            p1, k1, model["members"][0], flip_depth=flip_depth, flip_intrinsics=flip_intrinsics,
            conf_orig=conf_orig, conf_flip=conf_flip, conf_summary=conf_summary,
            cls_orig=cls_orig, cls_flip=cls_flip, cached_features=cached_features,
            cached_sem=sem0, use_torch_fast=use_torch_fast,
        )
        out1 = apply_sota_v2_model(
            p1, k1, model["members"][1], flip_depth=flip_depth, flip_intrinsics=flip_intrinsics,
            conf_orig=conf_orig, conf_flip=conf_flip, conf_summary=conf_summary,
            cls_orig=cls_orig, cls_flip=cls_flip, cached_features=cached_features,
            cached_sem=sem0, use_torch_fast=use_torch_fast,
        )
        return clip_depth(np.exp(w0 * np.log(out0) + w1 * np.log(out1)))

    if kind != "geocalib_3d_v2":
        raise ValueError(f"Unsupported v2 model kind: {kind!r}")

    feats = (
        cached_features
        if cached_features is not None
        else extract_v2_geo_features(
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

    # Exact pure NumPy path
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


def build_v2_candidate_specs() -> list[GeoCalibV2Spec]:
    """Return 26 structured Round 2 candidates spanning all 4 directions and both tracks."""
    return [
        # Direction 1: Deepen Pure Geometry (Track A: Single-Pass)
        GeoCalibV2Spec(name="D1_free_left_focal", direction="deepen", free_left_focal=True, geo_mode="v1", z_clip=None),
        GeoCalibV2Spec(name="D2_left_rel_gate", direction="deepen", use_focal_gate=True, geo_mode="v2_ext", z_clip=4.0),
        GeoCalibV2Spec(name="D3_qknots_robust", direction="deepen", use_quantile_knots=True, geo_mode="v2_clean", z_clip=4.0, loss_mode="robust_huber_111"),

        # Direction 2A: New Depth Parametrizations (Track A: Single-Pass)
        GeoCalibV2Spec(name="N1_inv_depth_spline", direction="new_param", depth_param="inv_spline", geo_mode="v1", z_clip=None),
        GeoCalibV2Spec(name="N2_hybrid_log_inv_res", direction="new_param", use_inv_residual=True, geo_mode="v1", z_clip=None),
        GeoCalibV2Spec(name="N3_rational_depth", direction="new_param", depth_param="rational", geo_mode="v1", z_clip=None),

        # Direction 2B: Single-Pass New Information Sources (Confidence control + Semantic hierarchy)
        GeoCalibV2Spec(name="S0_conf_summary_ctrl", direction="new_info", geo_mode="v2_conf", z_clip=4.0, sem_mode="none"),
        GeoCalibV2Spec(name="S1_sem_cls_r006", direction="new_info", geo_mode="v2_clean", sem_mode="cls", sem_reg=0.06),
        GeoCalibV2Spec(name="S2_sem_df_r006", direction="new_info", geo_mode="v2_clean", sem_mode="df", sem_reg=0.06),
        GeoCalibV2Spec(name="S3_sem_all_r008", direction="new_info", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.08),
        GeoCalibV2Spec(name="S4_sem_all_r005", direction="new_info", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.05),
        GeoCalibV2Spec(name="S5_sem_all_pca128_r005", direction="new_info", geo_mode="v2_clean", sem_mode="all_sp", sem_pca_k=128, sem_reg=0.05),

        # Direction 3 & 4: Combine / Hybrid & Polish (Track A: Single-Pass)
        GeoCalibV2Spec(name="H1_geosem_sp2d_r005_sp008", direction="combine", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibV2Spec(name="H2_geosem_sp2d_r008_sp010", direction="polish", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.08, use_sem_spatial2d=True, sem_sp_reg=0.10),
        GeoCalibV2Spec(name="H3_geosem_sp2d_pca128", direction="polish", geo_mode="v2_clean", sem_mode="all_sp", sem_pca_k=128, sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibV2Spec(name="H4_geosem_sp2d_inv_fgate", direction="combine", geo_mode="v2_ext", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_inv_residual=True, use_focal_gate=True),
        GeoCalibV2Spec(name="H5_geosem_sp2d_qknots_opt", direction="polish", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_quantile_knots=True, loss_mode="robust_huber_111"),
        GeoCalibV2Spec(name="H6_geosem_inv_spline", direction="combine", depth_param="inv_spline", geo_mode="v2_clean", sem_mode="all_sp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08),
        GeoCalibV2Spec(name="H7_sp_hybrid_blend", direction="combine", blend_members=("H1_geosem_sp2d_r005_sp008", "H6_geosem_inv_spline")),

        # Track B: Multi-Pass / 2-Pass Horizontal-Flip TTA Extension Track
        GeoCalibV2Spec(name="M1_v1_flip_sym_zero_retrain", direction="multipass", track="multi_pass_flip_tta", flip_blend_mode="sym_v1_zero_retrain"),
        GeoCalibV2Spec(name="M2_flip_geo_gated", direction="multipass", track="multi_pass_flip_tta", geo_mode="v2_ext", use_focal_gate=True, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibV2Spec(name="M3_flip_geosem_sp2d_r005", direction="multipass", track="multi_pass_flip_tta", geo_mode="v2_clean", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibV2Spec(name="M4_flip_geosem_sp2d_conf_fgate", direction="multipass", track="multi_pass_flip_tta", geo_mode="v2_ext", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, use_focal_gate=True, flip_blend_mode="conf", use_flip_consistency=True),
        GeoCalibV2Spec(name="M5_flip_geosem_sp2d_pca128", direction="multipass", track="multi_pass_flip_tta", geo_mode="v2_clean", sem_mode="all_mp", sem_pca_k=128, sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibV2Spec(name="M6_flip_inv_geosem", direction="multipass", track="multi_pass_flip_tta", depth_param="inv_spline", geo_mode="v2_clean", sem_mode="all_mp", sem_reg=0.05, use_sem_spatial2d=True, sem_sp_reg=0.08, flip_blend_mode="learnable", use_flip_consistency=True),
        GeoCalibV2Spec(name="M7_flip_hybrid_blend", direction="multipass", track="multi_pass_flip_tta", blend_members=("M3_flip_geosem_sp2d_r005", "M6_flip_inv_geosem")),
    ]


def prepare_v2_bundle(handle, cache: VerifiedV2Cache, split_ids: np.ndarray, is_train: bool = True) -> dict[str, Any]:
    """Load split arrays safely inside allowed_split; GT/targets built only when is_train=True."""
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

    v2_feats: list[dict[str, float]] = []
    for row in range(n):
        v2_feats.append(
            extract_v2_geo_features(
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
        "v2_feats": v2_feats,
        "sem_mats": sem_mats,
    }
    if is_train:
        base_data = prepare_sota_training(handle, cache.base, split_ids, need_rgb=False)
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


def _fit_blend_weight_strict(
    m0: dict[str, Any],
    m1: dict[str, Any],
    bundle: dict[str, Any],
    tr_idx: np.ndarray,
    s_per_img: int = 512,
) -> float:
    """Choose convex log-blend alpha in {0.0, 0.25, 0.5, 0.75, 1.0} strictly on training rows."""
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
        d0 = apply_sota_v2_model(
            bundle["p1_full"][row], bundle["intrinsics_orig"][row], m0,
            flip_depth=bundle["p2_full"][row], flip_intrinsics=bundle["intrinsics_flip"][row],
            conf_orig=bundle["c1_full"][row], conf_flip=bundle["c2_full"][row],
            cached_features=bundle["v2_feats"][row], cached_sem=x0[row] if x0.shape[1] else None,
        )[CROP]
        d1 = apply_sota_v2_model(
            bundle["p1_full"][row], bundle["intrinsics_orig"][row], m1,
            flip_depth=bundle["p2_full"][row], flip_intrinsics=bundle["intrinsics_flip"][row],
            conf_orig=bundle["c1_full"][row], conf_flip=bundle["c2_full"][row],
            cached_features=bundle["v2_feats"][row], cached_sem=x1[row] if x1.shape[1] else None,
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


def evaluate_v2_cv_for_seed(
    handle,
    cache: VerifiedV2Cache,
    bundle: dict[str, Any],
    scenes: np.ndarray,
    specs: list[GeoCalibV2Spec],
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

    v1_spec = {s.name: s for s in build_candidate_specs()}[PRIMARY_SOTA_METHOD]
    names = ["raw", "b3_residual_v1", "v1_geocalib3d", *[s.name for s in specs]]
    scores = np.full((n, len(names), 3), np.nan, dtype=np.float64)
    fold_ids = np.empty(n, dtype=int)

    single_specs = [s for s in specs if s.blend_members is None and s.flip_blend_mode != "sym_v1_zero_retrain"]
    blend_specs = [s for s in specs if s.blend_members is not None]
    sym_v1_specs = [s for s in specs if s.flip_blend_mode == "sym_v1_zero_retrain"]
    fold_dofs: dict[str, list[tuple[int, float]]] = {s.name: [] for s in specs}

    for f_idx, val in enumerate(folds):
        t0 = perf_counter()
        tr = np.setdiff1d(np.arange(n), val)
        scale = fit_scale(base["ratios"][tr])
        curve = fit_curve(base["samples_b3"][tr], base["targets_b3"][tr], scale)
        res_y = residual_targets(handle, cache.base, train_ids[tr], curve)[:, 0]
        res_model = fit_residual(base["features_b3"][tr], res_y, regularization=1.0, shrinkage=0.5)

        tr_p1 = base["samples_p"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_p2 = bundle["samples_p2"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_w1 = bundle["samples_w1"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_gt = base["samples_gt"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_row = base["samples_row"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_col = base["samples_col"][tr, :CV_SAMPLES_PER_IMAGE]
        tr_feats = [bundle["v2_feats"][i] for i in tr]
        tr_ratios = base["ratios"][tr]

        v1_model = fit_joint_sota_model(
            spec=v1_spec,
            samples_p=tr_p1,
            samples_gt=tr_gt,
            samples_row=tr_row,
            samples_col=tr_col,
            features=[base["sota_feats"][i] for i in tr],
            intrinsics=base["intrinsics"][tr],
            ratios=tr_ratios,
            max_iter=65,
        )

        def _fit_one(sp: GeoCalibV2Spec) -> tuple[str, dict[str, Any]]:
            use_p2 = tr_p2 if sp.flip_blend_mode != "none" else None
            use_w1 = tr_w1 if sp.flip_blend_mode == "conf" else None
            m = fit_joint_v2_model(
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

        for sp in sym_v1_specs:
            fitted_map[sp.name] = {"kind": "geocalib_v2_sym_v1_wrapper", "v1_model": v1_model, "w_flip": 0.55}

        for sp in blend_specs:
            n0, n1 = sp.blend_members  # type: ignore[misc]
            m0, m1 = fitted_map[n0], fitted_map[n1]
            alpha = _fit_blend_weight_strict(m0, m1, bundle, tr)
            fitted_map[sp.name] = {
                "kind": "geocalib_v2_ensemble",
                "members": [m0, m1],
                "weights": [alpha, 1.0 - alpha],
                "blend_space": "log",
            }

        for sp in specs:
            fold_dofs[sp.name].append(count_v2_parameters(fitted_map[sp.name]))

        for row in val:
            fold_ids[row] = f_idx
            p1 = bundle["p1_full"][row]
            p2 = bundle["p2_full"][row]
            c1 = bundle["c1_full"][row]
            c2 = bundle["c2_full"][row]
            k1 = bundle["intrinsics_orig"][row]
            k2 = bundle["intrinsics_flip"][row]
            gt = gts[row]

            b2 = apply_curve(p1, curve)
            b3 = apply_residual(b2, base["features_b3"][row], res_model)
            v1_out = apply_sota_model(p1, k1, v1_model, cached_features=base["sota_feats"][row])

            scores[row, 0] = evaluate_image(clip_depth(p1), gt)
            scores[row, 1] = evaluate_image(b3, gt)
            scores[row, 2] = evaluate_image(v1_out, gt)

            for s_idx, sp in enumerate(specs):
                m_obj = fitted_map[sp.name]
                if sp.blend_members is not None:
                    sm = fitted_map[sp.blend_members[0]]["semantic_branch"]["sem_mode"]
                elif sp.sem_mode != "none":
                    sm = sp.sem_mode
                else:
                    sm = "none"
                x_row = bundle["sem_mats"][sm][row] if bundle["sem_mats"][sm].shape[1] else None
                out = apply_sota_v2_model(
                    p1, k1, m_obj,
                    flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2,
                    cached_features=bundle["v2_feats"][row], cached_sem=x_row,
                )
                scores[row, 3 + s_idx] = evaluate_image(out, gt)

        print(f"V2 CV seed={seed} fold {f_idx + 1}/5 finished in {perf_counter() - t0:.1f}s", flush=True)

    if complexity_out is not None:
        complexity_out["raw"] = (0, 0.0)
        complexity_out["b3_residual_v1"] = (15, 15.0)
        complexity_out["v1_geocalib3d"] = (25, 25.0)
        for nm, pairs in fold_dofs.items():
            complexity_out[nm] = (pairs[0][0], float(np.mean([p[1] for p in pairs])))

    if not np.isfinite(scores).all():
        raise ArithmeticError("Nonfinite OOF scores in evaluate_v2_cv_for_seed")
    return scores, names, fold_ids


def _composite_norm_gain(cand_m: np.ndarray, v1_m: np.ndarray) -> float:
    return float(
        (v1_m[0] - cand_m[0]) / max(v1_m[0], 1e-12)
        + (v1_m[1] - cand_m[1]) / max(v1_m[1], 1e-12)
        + (cand_m[2] - v1_m[2]) / max(1.0 - v1_m[2], 1e-12)
    )


def run_v2_cv_study(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute complete Round 2 Gate 2 dual-track ablation + 4-seed CV + learning curve (795 Train ONLY)."""
    torch.set_num_threads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_v2_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids
    specs = build_v2_candidate_specs()
    spec_by_name = {s.name: s for s in specs}

    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        scenes = all_scenes[train_ids]
        bundle = prepare_v2_bundle(handle, cache, train_ids, is_train=True)

        # Stage 1: Canonical 5-fold CV (SEED=20261031) across all 26 candidates + baselines
        complexity_map: dict[str, tuple[int, float]] = {}
        scores_canon, names_canon, fold_ids = evaluate_v2_cv_for_seed(
            handle, cache, bundle, scenes, specs, seed=SEED, max_iter=75, complexity_out=complexity_map
        )
        b3_idx = names_canon.index("b3_residual_v1")
        v1_idx = names_canon.index("v1_geocalib3d")
        mean_scores = scores_canon.mean(axis=0)

        # Verify exact Round 1 OOF v1 reproduction
        if abs(float(mean_scores[v1_idx, 0]) - 0.09250162171078427) > 1e-6:
            raise RuntimeError(f"Canonical OOF v1 reproduction drifted: {mean_scores[v1_idx].tolist()}")

        g_b3, lo_b3, hi_b3 = _scene_bootstrap_vs_ref(scores_canon, b3_idx, scenes)
        g_v1, lo_v1, hi_v1 = _scene_bootstrap_vs_ref(scores_canon, v1_idx, scenes)

        # Paired bootstrap rows vs BOTH b3_residual_v1 and v1_geocalib3d
        paired_rows = []
        for ref_name, ref_idx in (("b3_residual_v1", b3_idx), ("v1_geocalib3d", v1_idx)):
            mg, cl, ch = _scene_bootstrap_vs_ref(scores_canon, ref_idx, scenes)
            for col, cand in enumerate(names_canon):
                if cand == ref_name:
                    continue
                for mi, metric in enumerate(METRIC_NAMES):
                    paired_rows.append({
                        "reference": ref_name,
                        "candidate": cand,
                        "metric": metric,
                        "gain": float(mg[col, mi]),
                        "ci95_low": float(cl[col, mi]),
                        "ci95_high": float(ch[col, mi]),
                    })

        direction_vec = np.array([-1.0, -1.0, 1.0])
        fold_rows = []
        ablation_rows = []
        for col, cand in enumerate(names_canon):
            fg_v1_list = []
            for f in range(5):
                fm = fold_ids == f
                m_f = scores_canon[fm, col].mean(axis=0)
                v1_f = scores_canon[fm, v1_idx].mean(axis=0)
                b3_f = scores_canon[fm, b3_idx].mean(axis=0)
                fg_v1 = (m_f - v1_f) * direction_vec
                fg_b3 = (m_f - b3_f) * direction_vec
                fg_v1_list.append(fg_v1)
                fold_rows.append({
                    "method": cand,
                    "fold": f,
                    "images": int(fm.sum()),
                    "abs_rel": float(m_f[0]),
                    "rmse": float(m_f[1]),
                    "delta1": float(m_f[2]),
                    "gain_vs_v1_abs_rel": float(fg_v1[0]),
                    "gain_vs_v1_rmse": float(fg_v1[1]),
                    "gain_vs_v1_delta1": float(fg_v1[2]),
                    "gain_vs_b3_abs_rel": float(fg_b3[0]),
                })

            sp_obj = spec_by_name.get(cand)
            track = sp_obj.track if sp_obj is not None else "single_pass"
            passes = 2 if track == "multi_pass_flip_tta" else 1
            dir_lbl = sp_obj.direction if sp_obj is not None else "baseline"
            raw_p, eff_dof = complexity_map[cand]
            all5_v1 = bool(np.all(np.asarray(fg_v1_list) > 0))
            means_v1 = bool(np.all(g_v1[col] > 1e-6))
            ci_v1 = bool(np.all(lo_v1[col] > 0))

            ablation_rows.append({
                "method": cand,
                "track": track,
                "backbone_passes": passes,
                "direction": dir_lbl,
                "raw_params": int(raw_p),
                "effective_dof": float(eff_dof),
                "abs_rel": float(mean_scores[col, 0]),
                "rmse": float(mean_scores[col, 1]),
                "delta1": float(mean_scores[col, 2]),
                "gain_vs_b3_abs_rel": float(g_b3[col, 0]),
                "gain_vs_b3_rmse": float(g_b3[col, 1]),
                "gain_vs_b3_delta1": float(g_b3[col, 2]),
                "gain_vs_v1_abs_rel": float(g_v1[col, 0]),
                "gain_vs_v1_rmse": float(g_v1[col, 1]),
                "gain_vs_v1_delta1": float(g_v1[col, 2]),
                "ci95_lo_v1_abs_rel": float(lo_v1[col, 0]),
                "ci95_hi_v1_abs_rel": float(hi_v1[col, 0]),
                "ci95_lo_v1_rmse": float(lo_v1[col, 1]),
                "ci95_hi_v1_rmse": float(hi_v1[col, 1]),
                "ci95_lo_v1_delta1": float(lo_v1[col, 2]),
                "ci95_hi_v1_delta1": float(hi_v1[col, 2]),
                "all_5_folds_beat_v1": all5_v1,
                "means_beat_v1": means_v1,
                "ci95_all_positive_vs_v1": ci_v1,
            })

        ablation_df = pd.DataFrame(ablation_rows)
        write_csv_lf(output_dir / "oof_ablation_summary.csv", ablation_df)
        write_csv_lf(output_dir / "oof_fold_breakdown.csv", pd.DataFrame(fold_rows))

        # Select representative top contenders for 4-seed stability test
        sp_cands = ablation_df[
            (ablation_df["track"] == "single_pass") & (ablation_df["direction"].isin(["deepen", "new_param", "new_info", "combine", "polish"]))
        ].sort_values("abs_rel")
        mp_cands = ablation_df[ablation_df["track"] == "multi_pass_flip_tta"].sort_values("abs_rel")

        sp_elig = sp_cands[sp_cands["means_beat_v1"] & sp_cands["all_5_folds_beat_v1"]]["method"].tolist()[:4]
        mp_elig = mp_cands[mp_cands["means_beat_v1"] & mp_cands["all_5_folds_beat_v1"]]["method"].tolist()[:4]
        if len(sp_elig) < 4:
            for nm in sp_cands["method"].tolist():
                if nm not in sp_elig and len(sp_elig) < 4:
                    sp_elig.append(nm)
        if len(mp_elig) < 3:
            for nm in mp_cands["method"].tolist():
                if nm not in mp_elig and len(mp_elig) < 3:
                    mp_elig.append(nm)

        # Ensure any blend dependency is included in finalists
        finalist_names: list[str] = []
        for nm in sp_elig + mp_elig + ["M1_v1_flip_sym_zero_retrain", "M2_flip_geo_gated"]:
            sp_o = spec_by_name[nm]
            if sp_o.blend_members is not None:
                for dep in sp_o.blend_members:
                    if dep not in finalist_names:
                        finalist_names.append(dep)
            if nm not in finalist_names:
                finalist_names.append(nm)
        finalist_specs = [spec_by_name[nm] for nm in finalist_names]

        # Stage 2: 4-seed scene-CV stability
        ms_rows = []
        for cand in ("raw", "b3_residual_v1", "v1_geocalib3d", *finalist_names):
            ci = names_canon.index(cand)
            gv = (mean_scores[ci] - mean_scores[v1_idx]) * direction_vec
            gb = (mean_scores[ci] - mean_scores[b3_idx]) * direction_vec
            ms_rows.append({
                "seed": SEED, "method": cand,
                "track": spec_by_name[cand].track if cand in spec_by_name else "single_pass",
                "abs_rel": float(mean_scores[ci, 0]), "rmse": float(mean_scores[ci, 1]), "delta1": float(mean_scores[ci, 2]),
                "gain_vs_v1_abs_rel": float(gv[0]), "gain_vs_v1_rmse": float(gv[1]), "gain_vs_v1_delta1": float(gv[2]),
                "gain_vs_b3_abs_rel": float(gb[0]),
            })

        for alt_seed in MULTI_SEEDS[1:]:
            sc_alt, nm_alt, _ = evaluate_v2_cv_for_seed(
                handle, cache, bundle, scenes, finalist_specs, seed=alt_seed, max_iter=75
            )
            m_alt = sc_alt.mean(axis=0)
            v1_alt_i = nm_alt.index("v1_geocalib3d")
            b3_alt_i = nm_alt.index("b3_residual_v1")
            for ci, cand in enumerate(nm_alt):
                gv = (m_alt[ci] - m_alt[v1_alt_i]) * direction_vec
                gb = (m_alt[ci] - m_alt[b3_alt_i]) * direction_vec
                ms_rows.append({
                    "seed": alt_seed, "method": cand,
                    "track": spec_by_name[cand].track if cand in spec_by_name else "single_pass",
                    "abs_rel": float(m_alt[ci, 0]), "rmse": float(m_alt[ci, 1]), "delta1": float(m_alt[ci, 2]),
                    "gain_vs_v1_abs_rel": float(gv[0]), "gain_vs_v1_rmse": float(gv[1]), "gain_vs_v1_delta1": float(gv[2]),
                    "gain_vs_b3_abs_rel": float(gb[0]),
                })

        ms_df = pd.DataFrame(ms_rows)
        write_csv_lf(output_dir / "oof_multiseed_stability.csv", ms_df)

        # Evaluate strict promotion criteria per track
        v1_ms = ms_df[ms_df["method"] == "v1_geocalib3d"].set_index("seed")
        promo_evals = []
        for cand in finalist_names:
            sub = ms_df[ms_df["method"] == cand].set_index("seed")
            all4_v1 = bool(
                (sub["gain_vs_v1_abs_rel"] > 0).all()
                and (sub["gain_vs_v1_rmse"] > 0).all()
                and (sub["gain_vs_v1_delta1"] > 0).all()
            )
            crow = ablation_df[ablation_df["method"] == cand].iloc[0]
            elig = bool(
                crow["means_beat_v1"]
                and crow["all_5_folds_beat_v1"]
                and crow["ci95_all_positive_vs_v1"]
                and all4_v1
            )
            comp_gains = [
                _composite_norm_gain(
                    sub.loc[s, ["abs_rel", "rmse", "delta1"]].to_numpy(dtype=np.float64),
                    v1_ms.loc[s, ["abs_rel", "rmse", "delta1"]].to_numpy(dtype=np.float64),
                )
                for s in MULTI_SEEDS
            ]
            promo_evals.append({
                "method": cand,
                "track": str(crow["track"]),
                "raw_params": int(crow["raw_params"]),
                "effective_dof": float(crow["effective_dof"]),
                "eligible": elig,
                "all_5_folds_beat_v1": bool(crow["all_5_folds_beat_v1"]),
                "ci95_all_positive_vs_v1": bool(crow["ci95_all_positive_vs_v1"]),
                "all_4_seeds_beat_v1": all4_v1,
                "mean4_abs_rel": float(sub["abs_rel"].mean()),
                "std4_abs_rel": float(sub["abs_rel"].std(ddof=0)),
                "mean4_rmse": float(sub["rmse"].mean()),
                "std4_rmse": float(sub["rmse"].std(ddof=0)),
                "mean4_delta1": float(sub["delta1"].mean()),
                "std4_delta1": float(sub["delta1"].std(ddof=0)),
                "mean4_composite_norm_gain_vs_v1": float(np.mean(comp_gains)),
            })

        sp_quals = sorted(
            [c for c in promo_evals if c["track"] == "single_pass" and c["eligible"]],
            key=lambda c: (-c["mean4_composite_norm_gain_vs_v1"], c["effective_dof"]),
        )
        mp_quals = sorted(
            [c for c in promo_evals if c["track"] == "multi_pass_flip_tta" and c["eligible"]],
            key=lambda c: (-c["mean4_composite_norm_gain_vs_v1"], c["effective_dof"]),
        )
        if not sp_quals or not mp_quals:
            raise RuntimeError(f"Promotion check failed: sp_quals={len(sp_quals)}, mp_quals={len(mp_quals)}")

        win_sp = sp_quals[0]["method"]
        win_mp = mp_quals[0]["method"]

        # Append paired bootstrap for Track B winner vs Track A winner
        win_sp_col = names_canon.index(win_sp)
        mg_mp_vs_sp, lo_mp_vs_sp, hi_mp_vs_sp = _scene_bootstrap_vs_ref(scores_canon, win_sp_col, scenes)
        for cand in mp_cands["method"].tolist():
            cc = names_canon.index(cand)
            for mi, metric in enumerate(METRIC_NAMES):
                paired_rows.append({
                    "reference": win_sp,
                    "candidate": cand,
                    "metric": metric,
                    "gain": float(mg_mp_vs_sp[cc, mi]),
                    "ci95_low": float(lo_mp_vs_sp[cc, mi]),
                    "ci95_high": float(hi_mp_vs_sp[cc, mi]),
                })
        write_csv_lf(output_dir / "oof_paired_bootstrap.csv", pd.DataFrame(paired_rows))

        # Pick best single ablation representatives cleanly on canonical OOF
        best_D = str(ablation_df[ablation_df["direction"] == "deepen"].sort_values("abs_rel").iloc[0]["method"])
        best_N = str(ablation_df[ablation_df["direction"] == "new_param"].sort_values("abs_rel").iloc[0]["method"])
        best_S = str(ablation_df[ablation_df["method"].isin(["S1_sem_cls_r006", "S2_sem_df_r006", "S3_sem_all_r008", "S4_sem_all_r005", "S5_sem_all_pca128_r005"])].sort_values("abs_rel").iloc[0]["method"])

        locked_eval_methods = []
        for nm in ("raw", "b3_residual_v1", "v1_geocalib3d", best_D, best_N, best_S, win_sp, "M1_v1_flip_sym_zero_retrain", "M2_flip_geo_gated", win_mp):
            if nm not in locked_eval_methods:
                locked_eval_methods.append(nm)

        # Export per-image OOF CSV for locked methods
        per_img_rows = []
        for row, idx in enumerate(train_ids):
            for nm in locked_eval_methods:
                col = names_canon.index(nm)
                per_img_rows.append({
                    "index": int(idx),
                    "scene": str(scenes[row]),
                    "fold": int(fold_ids[row]),
                    "method": nm,
                    "abs_rel": float(scores_canon[row, col, 0]),
                    "rmse": float(scores_canon[row, col, 1]),
                    "delta1": float(scores_canon[row, col, 2]),
                })
        write_csv_lf(output_dir / "oof_per_image.csv", pd.DataFrame(per_img_rows))

        # Stage 3: Train-only Learning Curve (10%, 25%, 50%, 100%)
        lc_df = _run_v2_learning_curve(handle, cache, bundle, scenes, spec_by_name[win_sp], spec_by_name[win_mp], output_dir / "oof_learning_curve.csv")

    report = {
        "scope": "official795_training_only",
        "test_read": False,
        "images": int(len(train_ids)),
        "scenes": int(len(np.unique(scenes))),
        "canonical_seed": SEED,
        "multi_seeds": list(MULTI_SEEDS),
        "promotion_criterion": "Beats v1_geocalib3d on all 3 metrics (AbsRel, RMSE, delta1) on canonical OOF, 5/5 folds, 4/4 seeds, and 10k-replicate bootstrap 95% CI > 0 vs v1; ranked by 4-seed composite normalized gain.",
        "primary_single_pass_method": win_sp,
        "primary_multipass_method": win_mp,
        "ablation_representatives": {
            "best_D_geo": best_D,
            "best_N_param": best_N,
            "best_S_sem_scale": best_S,
            "M1_v1_flip_sym": "M1_v1_flip_sym_zero_retrain",
            "M2_flip_geo_gated": "M2_flip_geo_gated",
        },
        "locked_ladder_order": locked_eval_methods,
        "track_a_qualifiers": sp_quals,
        "track_b_qualifiers": mp_quals,
        "candidate_evaluations": promo_evals,
        "learning_curve_10_to_100_rows": int(len(lc_df)),
    }
    write_json_lf(output_dir / "cv_v2_selection_report.json", report)
    print(f"Round 2 Gate 2 Complete: Track A Winner={win_sp}, Track B Winner={win_mp}", flush=True)
    return report


def _run_v2_learning_curve(
    handle,
    cache: VerifiedV2Cache,
    bundle: dict[str, Any],
    scenes: np.ndarray,
    sp_winner: GeoCalibV2Spec,
    mp_winner: GeoCalibV2Spec,
    output_path: Path,
) -> pd.DataFrame:
    """Train-only 5-fold learning curve (10/25/50/100% scenes) comparing B3, v1, v2_sp, v2_mp."""
    base = bundle["base_data"]
    train_ids = bundle["ids"]
    n = len(train_ids)
    folds = scene_folds_seeded(scenes, count=5, seed=SEED)
    gts = [nyu_depth(handle, int(i)) for i in train_ids]
    v1_spec = {s.name: s for s in build_candidate_specs()}[PRIMARY_SOTA_METHOD]
    spec_by_name = {s.name: s for s in build_v2_candidate_specs()}
    methods = ("b3_residual_v1", "v1_geocalib3d", sp_winner.name, mp_winner.name)

    def _fit_v2_on_sub(sp: GeoCalibV2Spec, sub: np.ndarray) -> dict[str, Any]:
        if sp.blend_members is not None:
            m0 = _fit_v2_on_sub(spec_by_name[sp.blend_members[0]], sub)
            m1 = _fit_v2_on_sub(spec_by_name[sp.blend_members[1]], sub)
            alpha = _fit_blend_weight_strict(m0, m1, bundle, sub)
            return {"kind": "geocalib_v2_ensemble", "members": [m0, m1], "weights": [alpha, 1.0 - alpha], "blend_space": "log"}
        p2u = bundle["samples_p2"][sub, :CV_SAMPLES_PER_IMAGE] if sp.flip_blend_mode != "none" else None
        w1u = bundle["samples_w1"][sub, :CV_SAMPLES_PER_IMAGE] if sp.flip_blend_mode == "conf" else None
        return fit_joint_v2_model(
            spec=sp,
            samples_p1=base["samples_p"][sub, :CV_SAMPLES_PER_IMAGE],
            samples_gt=base["samples_gt"][sub, :CV_SAMPLES_PER_IMAGE],
            samples_row=base["samples_row"][sub, :CV_SAMPLES_PER_IMAGE],
            samples_col=base["samples_col"][sub, :CV_SAMPLES_PER_IMAGE],
            features=[bundle["v2_feats"][i] for i in sub],
            ratios=base["ratios"][sub],
            x_sem_raw=bundle["sem_mats"][sp.sem_mode][sub],
            samples_p2=p2u,
            samples_w1=w1u,
            max_iter=75,
        )

    rows = []
    for frac in LEARNING_CURVE_FRACTIONS:
        t0 = perf_counter()
        scores = np.full((n, len(methods), 3), np.nan, dtype=np.float64)
        n_scenes_list, n_imgs_list = [], []
        for f_idx, val in enumerate(folds):
            fitting = np.setdiff1d(np.arange(n), val)
            f_sc = np.unique(scenes[fitting])
            shuffled = np.random.default_rng(LEARNING_CURVE_SEED + f_idx).permutation(f_sc)
            keep_sc = shuffled[: max(1, int(np.ceil(frac * len(shuffled))))]
            tr_sub = fitting[np.isin(scenes[fitting], keep_sc)]
            n_scenes_list.append(len(keep_sc))
            n_imgs_list.append(len(tr_sub))

            sc = fit_scale(base["ratios"][tr_sub])
            cv = fit_curve(base["samples_b3"][tr_sub], base["targets_b3"][tr_sub], sc)
            ry = residual_targets(handle, cache.base, train_ids[tr_sub], cv)[:, 0]
            rm = fit_residual(base["features_b3"][tr_sub], ry, regularization=1.0, shrinkage=0.5)
            v1_m = fit_joint_sota_model(
                spec=v1_spec,
                samples_p=base["samples_p"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                samples_gt=base["samples_gt"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                samples_row=base["samples_row"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                samples_col=base["samples_col"][tr_sub, :CV_SAMPLES_PER_IMAGE],
                features=[base["sota_feats"][i] for i in tr_sub],
                intrinsics=base["intrinsics"][tr_sub],
                ratios=base["ratios"][tr_sub],
                max_iter=65,
            )
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_sp = pool.submit(_fit_v2_on_sub, sp_winner, tr_sub)
                f_mp = pool.submit(_fit_v2_on_sub, mp_winner, tr_sub)
                m_sp, m_mp = f_sp.result(), f_mp.result()

            sm_sp = m_sp["members"][0]["semantic_branch"]["sem_mode"] if m_sp["kind"] == "geocalib_v2_ensemble" else sp_winner.sem_mode
            sm_mp = m_mp["members"][0]["semantic_branch"]["sem_mode"] if m_mp["kind"] == "geocalib_v2_ensemble" else mp_winner.sem_mode

            for row in val:
                p1 = bundle["p1_full"][row]; p2 = bundle["p2_full"][row]
                c1 = bundle["c1_full"][row]; c2 = bundle["c2_full"][row]
                k1 = bundle["intrinsics_orig"][row]; k2 = bundle["intrinsics_flip"][row]
                gt = gts[row]
                b3 = apply_residual(apply_curve(p1, cv), base["features_b3"][row], rm)
                v1_o = apply_sota_model(p1, k1, v1_m, cached_features=base["sota_feats"][row])
                xs = bundle["sem_mats"][sm_sp][row] if bundle["sem_mats"][sm_sp].shape[1] else None
                xm = bundle["sem_mats"][sm_mp][row] if bundle["sem_mats"][sm_mp].shape[1] else None
                sp_o = apply_sota_v2_model(p1, k1, m_sp, cached_features=bundle["v2_feats"][row], cached_sem=xs)
                mp_o = apply_sota_v2_model(p1, k1, m_mp, flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2, cached_features=bundle["v2_feats"][row], cached_sem=xm)
                for col, out in enumerate((b3, v1_o, sp_o, mp_o)):
                    scores[row, col] = evaluate_image(out, gt)

        mean_m = scores.mean(axis=0)
        ci_lo, ci_hi = _scene_metric_bootstrap_ci(scores, scenes)
        g_v1, g_v1_lo, g_v1_hi = _scene_bootstrap_vs_ref(scores, 1, scenes)
        for col, m_name in enumerate(methods):
            rows.append({
                "fraction": float(frac),
                "train_scenes_mean": float(np.mean(n_scenes_list)),
                "train_images_mean": float(np.mean(n_imgs_list)),
                "method": m_name,
                "abs_rel": float(mean_m[col, 0]),
                "rmse": float(mean_m[col, 1]),
                "delta1": float(mean_m[col, 2]),
                "ci95_lo_abs_rel": float(ci_lo[col, 0]),
                "ci95_hi_abs_rel": float(ci_hi[col, 0]),
                "ci95_lo_rmse": float(ci_lo[col, 1]),
                "ci95_hi_rmse": float(ci_hi[col, 1]),
                "ci95_lo_delta1": float(ci_lo[col, 2]),
                "ci95_hi_delta1": float(ci_hi[col, 2]),
                "gain_vs_v1_abs_rel": float(g_v1[col, 0]),
                "gain_ci95_lo_v1_abs_rel": float(g_v1_lo[col, 0]),
                "gain_ci95_hi_v1_abs_rel": float(g_v1_hi[col, 0]),
            })
        print(f"V2 Learning curve frac={frac:.2f} ({perf_counter() - t0:.1f}s): v1={mean_m[1,0]:.4f}, sp={mean_m[2,0]:.4f}, mp={mean_m[3,0]:.4f}", flush=True)

    df = pd.DataFrame(rows)
    write_csv_lf(output_path, df)
    return df


def validate_frozen_sota_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate self-contained GeoCalib_3D_v2 pre-test frozen artifact strictly."""
    if not isinstance(payload, dict) or payload.get("schema") != "GeoCalib_3D_v2":
        raise ValueError("Expected schema == 'GeoCalib_3D_v2'")
    if payload.get("complete") is not True:
        raise ValueError("Frozen v2 artifact must have complete=True")
    if payload.get("test_read") is not False:
        raise ValueError("Frozen v2 artifact must certify test_read=False")
    train_ids = payload.get("train_ids")
    if not isinstance(train_ids, list) or len(train_ids) != 795 or len(set(train_ids)) != 795:
        raise ValueError("Invalid train_ids in frozen_sota_v2_model.json")
    for key in ("dataset_sha256", "split_sha256", "b3_frozen_model_sha256", "v1_frozen_sota_sha256", "cv_v2_selection_report_sha256"):
        val = payload.get(key)
        if not isinstance(val, str) or len(val) != 64:
            raise ValueError(f"Missing or malformed {key}")
    models = payload.get("models", {})
    win_sp = payload.get("primary_single_pass_method")
    win_mp = payload.get("primary_multipass_method")
    if win_sp not in models or win_mp not in models:
        raise ValueError("Primary single-pass or multi-pass model missing from frozen v2 artifact")
    return payload


def load_frozen_sota_v2(path: Path | str, expected_sha256: str | None = None) -> dict[str, Any]:
    target = Path(path)
    actual_sha = file_sha256(target)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"Frozen v2 SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return validate_frozen_sota_v2(payload)


def freeze_sota_v2_models(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    cv_report_path: Path,
    b3_frozen_path: Path,
    v1_frozen_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Gate 3: Fit promoted Track A & B winners + key ablation ladder on all 795 training images and lock JSON."""
    torch.set_num_threads(2)
    cache = load_v2_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=True, verify_scenes=True)
    train_ids = cache.train_ids
    cv_report = json.loads(Path(cv_report_path).read_text(encoding="utf-8"))
    if cv_report.get("test_read") is not False:
        raise ValueError("CV selection report must certify test_read=False")

    b3_frozen = load_frozen(b3_frozen_path)
    if b3_frozen["train_ids"] != train_ids.tolist():
        raise ValueError("Train split mismatch vs b3_frozen")
    v1_frozen = json.loads(Path(v1_frozen_path).read_text(encoding="utf-8"))

    win_sp = cv_report["primary_single_pass_method"]
    win_mp = cv_report["primary_multipass_method"]
    reps = cv_report["ablation_representatives"]
    freeze_names = []
    for nm in (reps["best_D_geo"], reps["best_N_param"], reps["best_S_sem_scale"], win_sp, reps["M2_flip_geo_gated"], win_mp):
        if nm not in freeze_names:
            freeze_names.append(nm)

    spec_by_name = {s.name: s for s in build_v2_candidate_specs()}
    with open_nyu(data_root) as handle:
        train_scenes = np.unique(scene_names(handle)[train_ids])
        bundle = prepare_v2_bundle(handle, cache, train_ids, is_train=True)
    base = bundle["base_data"]

    # Ensure blend dependencies are fitted first
    to_fit: list[str] = []
    for nm in freeze_names:
        sp = spec_by_name[nm]
        if sp.blend_members is not None:
            for dep in sp.blend_members:
                if dep not in to_fit:
                    to_fit.append(dep)
        if nm not in to_fit:
            to_fit.append(nm)

    fitted: dict[str, Any] = {
        "M1_v1_flip_sym_zero_retrain": {
            "kind": "geocalib_v2_sym_v1_wrapper",
            "v1_model": v1_frozen["primary_model"],
            "w_flip": 0.55,
        }
    }
    for nm in to_fit:
        sp = spec_by_name[nm]
        if sp.blend_members is not None:
            m0, m1 = fitted[sp.blend_members[0]], fitted[sp.blend_members[1]]
            alpha = _fit_blend_weight_strict(m0, m1, bundle, np.arange(len(train_ids)), s_per_img=1024)
            fitted[nm] = {"kind": "geocalib_v2_ensemble", "members": [m0, m1], "weights": [alpha, 1.0 - alpha], "blend_space": "log"}
        else:
            p2u = bundle["samples_p2"] if sp.flip_blend_mode != "none" else None
            w1u = bundle["samples_w1"] if sp.flip_blend_mode == "conf" else None
            fitted[nm] = fit_joint_v2_model(
                spec=sp,
                samples_p1=base["samples_p"],
                samples_gt=base["samples_gt"],
                samples_row=base["samples_row"],
                samples_col=base["samples_col"],
                features=bundle["v2_feats"],
                ratios=base["ratios"],
                x_sem_raw=bundle["sem_mats"][sp.sem_mode],
                samples_p2=p2u,
                samples_w1=w1u,
                max_iter=110,
            )

    locked_models = {nm: fitted[nm] for nm in (*freeze_names, "M1_v1_flip_sym_zero_retrain")}
    sp_raw, sp_dof = count_v2_parameters(locked_models[win_sp])
    mp_raw, mp_dof = count_v2_parameters(locked_models[win_mp])

    pkg_dir = Path(__file__).resolve().parent
    src_hashes = {
        f"depthlab/{fn}": file_sha256(pkg_dir / fn)
        for fn in ("sota_io.py", "sota_calibration.py", "sota_v2_cache.py", "sota_v2_diagnose.py", "sota_v2_calibration.py", "metrics.py")
    }

    payload = {
        "schema": "GeoCalib_3D_v2",
        "complete": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_read": False,
        "primary_single_pass_method": win_sp,
        "primary_single_pass_display": "GeoCalib-3D v2",
        "primary_single_pass_raw_params": sp_raw,
        "primary_single_pass_effective_dof": sp_dof,
        "primary_multipass_method": win_mp,
        "primary_multipass_display": "GeoCalib-3D v2 + Flip-TTA",
        "primary_multipass_raw_params": mp_raw,
        "primary_multipass_effective_dof": mp_dof,
        "ablation_representatives": reps,
        "locked_eval_order": [
            "raw", "b3_residual_v1", "v1_geocalib3d",
            reps["best_D_geo"], reps["best_N_param"], reps["best_S_sem_scale"],
            win_sp, "M1_v1_flip_sym_zero_retrain", reps["M2_flip_geo_gated"], win_mp,
        ],
        "train_ids": train_ids.tolist(),
        "training_images": int(len(train_ids)),
        "training_scenes": int(len(train_scenes)),
        "dataset_sha256": cache.hashes["nyu_mat_sha256"],
        "split_sha256": cache.hashes["splits_sha256"],
        "cache_sha256": cache.hashes,
        "b3_frozen_model_sha256": file_sha256(b3_frozen_path),
        "v1_frozen_sota_sha256": file_sha256(v1_frozen_path),
        "cv_v2_selection_report_sha256": file_sha256(cv_report_path),
        "source_sha256": src_hashes,
        "models": locked_models,
    }
    write_json_lf(output_path, payload)
    out_sha = file_sha256(output_path)
    print(f"Gate 3 Freeze Complete: {output_path} (sha256={out_sha}, TrackA={win_sp}[{sp_raw} raw / {sp_dof:.1f} dof], TrackB={win_mp}[{mp_raw} raw / {mp_dof:.1f} dof])", flush=True)
    return validate_frozen_sota_v2(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    cv_p = sub.add_parser("cv", help="Run Gate 2 dual-track CV + 4-seed + learning curve on 795 train ONLY")
    cv_p.add_argument("--data-root", type=Path, required=True)
    cv_p.add_argument("--orig-preds", type=Path, required=True)
    cv_p.add_argument("--aux-dir", type=Path, required=True)
    cv_p.add_argument("--flip-dir", type=Path, required=True)
    cv_p.add_argument("--output", type=Path, required=True)

    fr_p = sub.add_parser("freeze", help="Gate 3 full-795 fitting and pre-test JSON lock")
    fr_p.add_argument("--data-root", type=Path, required=True)
    fr_p.add_argument("--orig-preds", type=Path, required=True)
    fr_p.add_argument("--aux-dir", type=Path, required=True)
    fr_p.add_argument("--flip-dir", type=Path, required=True)
    fr_p.add_argument("--cv-report", type=Path, required=True)
    fr_p.add_argument("--b3-frozen", type=Path, required=True)
    fr_p.add_argument("--v1-frozen", type=Path, required=True)
    fr_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "cv":
        run_v2_cv_study(args.data_root, args.orig_preds, args.aux_dir, args.flip_dir, args.output)
    elif args.mode == "freeze":
        freeze_sota_v2_models(
            args.data_root, args.orig_preds, args.aux_dir, args.flip_dir,
            args.cv_report, args.b3_frozen, args.v1_frozen, args.output,
        )


if __name__ == "__main__":
    main()
