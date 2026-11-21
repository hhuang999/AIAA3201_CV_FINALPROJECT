"""Round 2 Gate 1: Deep mechanism & residual bimodality diagnosis on 795 training images ONLY."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from .calibration import (
    SEED,
    apply_curve,
    apply_residual,
    fit_curve,
    fit_residual,
    fit_scale,
    residual_targets,
)
from .data import nyu_depth, open_nyu, scene_names
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, clip_depth, evaluate_image, metrics, valid_depth
from .sota_calibration import (
    CROP_SUB,
    CV_SAMPLES_PER_IMAGE,
    GeoCalibSpec,
    apply_sota_model,
    build_candidate_specs,
    extract_sota_features,
    fit_joint_sota_model,
    prepare_sota_training,
)
from .sota_diagnose import (
    CROP_R2,
    CROP_U_TILDE,
    CROP_V_TILDE,
    extract_spatial_3d_features,
    fit_inv_affine_oracle,
    fit_log_affine_oracle,
    fit_spatial_oracle,
    optimal_absrel_scalar,
    optimal_rmse_scalar,
)
from .sota_io import scene_folds_seeded, write_csv_lf, write_json_lf, write_text_lf
from .sota_v2_cache import CONF_SUMMARY_NAMES, load_v2_cache


CANON_V1_ABSREL = 0.09250162171078427
CANON_V1_RMSE = 0.37923017857116015
CANON_V1_DELTA1 = 0.9266869297546273


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
            # Dual kernel ridge for fast high-dimensional exact solve
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
    sota_feat: dict[str, float],
    sp_feat: dict[str, float],
) -> dict[str, float]:
    """Compute richer geometric + K_hat x Spatial3D interaction features cleanly."""
    f_t = float(sota_feat["f_tilde"])
    f_hinge = max(0.0, 1.14 - f_t)
    log_sy = float(sota_feat["log_span_y"])
    std_lz = float(sota_feat["std_logz"])
    dz_dv = float(sota_feat["dz_dv"])
    mean_iz = float(sp_feat["mean_inv_z"])
    std_iz = float(sota_feat["std_inv_z"])
    q25 = float(sota_feat["q25_logz"])
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


def run_v2_diagnosis(
    data_root: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute Round 2 deep post-v1 mechanism diagnosis strictly on 795 training images."""
    torch.set_num_threads(2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_v2_cache(orig_pred_dir, aux_dir, flip_dir, data_root, allowed_split="train", compute_hashes=False)
    train_ids = cache.train_ids
    v1_spec: GeoCalibSpec = {s.name: s for s in build_candidate_specs()}["J3_geok_sp3d_slope_bal111_l01"]

    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        scenes = all_scenes[train_ids]
        folds = scene_folds_seeded(scenes, count=5, seed=SEED)
        data = prepare_sota_training(handle, cache.base, train_ids, need_rgb=False)
        n = len(train_ids)

        # 1. Reconstruct exact canonical 5-fold OOF GeoCalib-3D v1 and B3 baseline maps
        oof_v1 = np.empty((n, 480, 640), dtype=np.float32)
        oof_b3 = np.empty((n, 480, 640), dtype=np.float32)
        oof_v1_on_flip_avg = np.empty((n, 480, 640), dtype=np.float32)
        oof_v1_flip_pass = np.empty((n, 480, 640), dtype=np.float32)
        raw_scores = np.empty((n, 3), dtype=np.float64)
        b3_scores = np.empty((n, 3), dtype=np.float64)
        v1_scores = np.empty((n, 3), dtype=np.float64)
        raw_flip_log_scores = np.empty((n, 3), dtype=np.float64)
        raw_flip_conf_scores = np.empty((n, 3), dtype=np.float64)
        v1_applied_flip_avg_scores = np.empty((n, 3), dtype=np.float64)
        v1_flip_log_scores = np.empty((n, 3), dtype=np.float64)
        v1_flip_conf_scores = np.empty((n, 3), dtype=np.float64)
        fold_id = np.empty(n, dtype=int)

        intr_flip_all = cache.intrinsics_flip_for(train_ids)

        for f_idx, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            scale = fit_scale(data["ratios"][tr])
            curve = fit_curve(data["samples_b3"][tr], data["targets_b3"][tr], scale)
            res_y = residual_targets(handle, cache.base, train_ids[tr], curve)[:, 0]
            res_model = fit_residual(data["features_b3"][tr], res_y, regularization=1.0, shrinkage=0.5)

            v1_model = fit_joint_sota_model(
                spec=v1_spec,
                samples_p=data["samples_p"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_gt=data["samples_gt"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_row=data["samples_row"][tr, :CV_SAMPLES_PER_IMAGE],
                samples_col=data["samples_col"][tr, :CV_SAMPLES_PER_IMAGE],
                features=[data["sota_feats"][i] for i in tr],
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

                b2 = apply_curve(p1, curve)
                b3 = apply_residual(b2, data["features_b3"][row], res_model)
                v1 = apply_sota_model(p1, k1, v1_model, cached_features=data["sota_feats"][row])

                # Flip-TTA variations
                p_flip_log = clip_depth(np.exp(0.5 * (np.log(p1) + np.log(p2))))
                w1 = c1 / (c1 + c2)
                p_flip_conf = clip_depth(np.exp(w1 * np.log(p1) + (1.0 - w1) * np.log(p2)))
                k_mean = 0.5 * (k1 + k2)
                v1_on_avg = apply_sota_model(p_flip_log, k_mean, v1_model)
                v1_pass2 = apply_sota_model(p2, k2, v1_model)
                v1_flog = clip_depth(np.exp(0.5 * (np.log(v1) + np.log(v1_pass2))))
                v1_fconf = clip_depth(np.exp(w1 * np.log(v1) + (1.0 - w1) * np.log(v1_pass2)))

                oof_b3[row] = b3.astype(np.float32)
                oof_v1[row] = v1.astype(np.float32)
                oof_v1_on_flip_avg[row] = v1_on_avg.astype(np.float32)
                oof_v1_flip_pass[row] = v1_flog.astype(np.float32)

                raw_scores[row] = evaluate_image(clip_depth(p1), gt)
                b3_scores[row] = evaluate_image(b3, gt)
                v1_scores[row] = evaluate_image(v1, gt)
                raw_flip_log_scores[row] = evaluate_image(p_flip_log, gt)
                raw_flip_conf_scores[row] = evaluate_image(p_flip_conf, gt)
                v1_applied_flip_avg_scores[row] = evaluate_image(v1_on_avg, gt)
                v1_flip_log_scores[row] = evaluate_image(v1_flog, gt)
                v1_flip_conf_scores[row] = evaluate_image(v1_fconf, gt)

        v1_m = v1_scores.mean(axis=0)
        if (
            abs(float(v1_m[0]) - CANON_V1_ABSREL) > 1e-6
            or abs(float(v1_m[1]) - CANON_V1_RMSE) > 1e-6
            or abs(float(v1_m[2]) - CANON_V1_DELTA1) > 1e-6
        ):
            raise RuntimeError(f"Reconstructed OOF GeoCalib-3D v1 mismatch: {v1_m.tolist()}")
        print(f"Verified exact v1 OOF reconstruction: {v1_m.round(6).tolist()}", flush=True)

        # 2. Oracles & residuals on oof_geocalib_v1 + oof_v1_flip
        oracle_names = [
            "raw",
            "oof_b3_residual_v1",
            "oof_geocalib_v1",
            "v1_oracle_med_scalar",
            "v1_oracle_absrel_scalar",
            "v1_oracle_rmse_scalar",
            "v1_oracle_log_affine",
            "v1_oracle_inv_affine",
            "v1_oracle_spatial_linear",
            "v1_oracle_spatial_quadric",
            "v1_oracle_spatial_plus_depth",
            "raw_flip_log_avg",
            "raw_flip_conf_avg",
            "v1_applied_to_flip_avg",
            "v1_flip_log_avg",
            "v1_flip_conf_avg",
            "v1_flip_oracle_absrel_scalar",
        ]
        oracle_arr = np.empty((n, len(oracle_names), 3), dtype=np.float64)
        e_v1_med = np.empty(n, dtype=np.float64)
        e_v1_absrel = np.empty(n, dtype=np.float64)
        e_v1_rmse = np.empty(n, dtype=np.float64)
        e_v1_slope_v = np.empty(n, dtype=np.float64)
        e_v1_radial = np.empty(n, dtype=np.float64)
        raw_bias = np.empty(n, dtype=np.float64)
        v1_bias = np.empty(n, dtype=np.float64)

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
            v1_c = oof_v1[row][CROP][valid].astype(np.float64)
            v1f_c = oof_v1_flip_pass[row][CROP][valid].astype(np.float64)

            u_c = CROP_U_TILDE[valid]
            v_c = CROP_V_TILDE[valid]
            r2_c = CROP_R2[valid]

            med_s = float(np.median(g_c / v1_c))
            ar_s = optimal_absrel_scalar(v1_c, g_c)
            rm_s = optimal_rmse_scalar(v1_c, g_c)
            log_aff = fit_log_affine_oracle(v1_c, g_c)
            inv_aff = fit_inv_affine_oracle(v1_c, g_c)
            sp_lin, coef_lin = fit_spatial_oracle(v1_c, g_c, u_c, v_c, r2_c, "spatial_linear")
            sp_quad, _ = fit_spatial_oracle(v1_c, g_c, u_c, v_c, r2_c, "spatial_quadric")
            sp_plus, _ = fit_spatial_oracle(v1_c, g_c, u_c, v_c, r2_c, "spatial_plus_depth")

            f_ar_s = optimal_absrel_scalar(v1f_c, g_c)

            e_v1_med[row] = float(np.log(med_s))
            e_v1_absrel[row] = float(np.log(ar_s))
            e_v1_rmse[row] = float(np.log(rm_s))
            e_v1_slope_v[row] = float(coef_lin[2])
            e_v1_radial[row] = float(coef_lin[3])
            raw_bias[row] = float(np.median(np.log(raw_c / g_c)))
            v1_bias[row] = float(np.median(np.log(v1_c / g_c)))

            oracle_arr[row, 0] = raw_scores[row]
            oracle_arr[row, 1] = b3_scores[row]
            oracle_arr[row, 2] = v1_scores[row]
            for c_idx, pred_1d in (
                (3, clip_depth(med_s * v1_c)),
                (4, clip_depth(ar_s * v1_c)),
                (5, clip_depth(rm_s * v1_c)),
                (6, log_aff),
                (7, inv_aff),
                (8, sp_lin),
                (9, sp_quad),
                (10, sp_plus),
            ):
                oracle_arr[row, c_idx] = metrics(pred_1d, g_c, np.ones(len(g_c), dtype=bool))
            oracle_arr[row, 11] = raw_flip_log_scores[row]
            oracle_arr[row, 12] = raw_flip_conf_scores[row]
            oracle_arr[row, 13] = v1_applied_flip_avg_scores[row]
            oracle_arr[row, 14] = v1_flip_log_scores[row]
            oracle_arr[row, 15] = v1_flip_conf_scores[row]
            oracle_arr[row, 16] = metrics(clip_depth(f_ar_s * v1f_c), g_c, np.ones(len(g_c), dtype=bool))

            sp_f = extract_spatial_3d_features(raw_full, data["intrinsics"][row])
            sp_rows.append(sp_f)
            inter_rows.append(
                extract_geo_interaction_features(raw_full, data["intrinsics"][row], data["sota_feats"][row], sp_f)
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

            # Depth bins
            for lo, hi in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                b_mask = valid & (gt_c >= lo) & (gt_c < hi)
                if b_mask.any():
                    m_b3 = metrics(oof_b3[row][CROP], gt_c, b_mask)
                    m_v1 = metrics(oof_v1[row][CROP], gt_c, b_mask)
                    m_vf = metrics(oof_v1_flip_pass[row][CROP], gt_c, b_mask)
                    bin_records.append(
                        {
                            "row": row,
                            "lower_m": lo,
                            "upper_m": hi,
                            "b3_abs_rel": float(m_b3[0]),
                            "v1_abs_rel": float(m_v1[0]),
                            "v1_flip_abs_rel": float(m_vf[0]),
                            "b3_delta1": float(m_b3[2]),
                            "v1_delta1": float(m_v1[2]),
                            "v1_flip_delta1": float(m_vf[2]),
                        }
                    )

        oracle_df = pd.DataFrame(
            [
                {
                    "method": name,
                    "abs_rel": float(oracle_arr[:, c, 0].mean()),
                    "rmse": float(oracle_arr[:, c, 1].mean()),
                    "delta1": float(oracle_arr[:, c, 2].mean()),
                    "gain_vs_v1_abs_rel": float(oracle_arr[:, 2, 0].mean() - oracle_arr[:, c, 0].mean()),
                    "gain_vs_v1_rmse": float(oracle_arr[:, 2, 1].mean() - oracle_arr[:, c, 1].mean()),
                    "gain_vs_v1_delta1": float(oracle_arr[:, c, 2].mean() - oracle_arr[:, 2, 2].mean()),
                }
                for c, name in enumerate(oracle_names)
            ]
        )
        write_csv_lf(output_dir / "diag_v2_oracles.csv", oracle_df)

        # 3. 5-Fold Scene-CV Predictability Audit (R^2) of Remaining v1 Residuals
        sota_feat_df = pd.DataFrame(data["sota_feats"])
        sp_df = pd.DataFrame(sp_rows)
        inter_df = pd.DataFrame(inter_rows)
        flip_df = pd.DataFrame(flip_rows)
        conf_df = pd.DataFrame(conf_sum_all, columns=list(CONF_SUMMARY_NAMES))

        v1_used_cols = [
            "f_tilde", "cx_off", "cy_off", "q25_logz", "std_logz", "grad_logz",
            "dz_dv", "grad_aniso_uv", "log_span_y", "span_x_over_z", "std_inv_z",
        ]
        X_v1_ctrl = sota_feat_df[v1_used_cols].to_numpy(dtype=np.float64)
        X_inter = inter_df.to_numpy(dtype=np.float64)
        X_conf = conf_df.to_numpy(dtype=np.float64)
        X_flip = flip_df.to_numpy(dtype=np.float64)
        X_pat = np.hstack([pat_mean_all, pat_std_all])
        X_single_all = np.hstack([X_v1_ctrl, X_inter, X_conf, df_stats_all])
        X_plus_flip = np.hstack([X_single_all, X_flip])

        signal_specs: list[tuple[str, str, np.ndarray, list[tuple[int | None, float]]]] = [
            ("v1_geo_sp3d_control", "single_pass", X_v1_ctrl, [(None, 1.0), (None, 10.0)]),
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
                r2_m, pred_m = scene_cv_ridge_r2_pca(x_mat, e_v1_med, folds, lam=lam, pca_k=pca_k)
                r2_a, _ = scene_cv_ridge_r2_pca(x_mat, e_v1_absrel, folds, lam=lam, pca_k=pca_k)
                r2_sv, _ = scene_cv_ridge_r2_pca(x_mat, e_v1_slope_v, folds, lam=lam, pca_k=pca_k)
                r2_rad, _ = scene_cv_ridge_r2_pca(x_mat, e_v1_radial, folds, lam=lam, pca_k=pca_k)
                r2_rows.append(
                    {
                        "feature_block": block_name,
                        "track": track,
                        "raw_dims": int(x_mat.shape[1]),
                        "pca_k": -1 if pca_k is None else int(pca_k),
                        "ridge_lam": float(lam),
                        "cv_r2_e_v1_med": r2_m,
                        "cv_r2_e_v1_absrel": r2_a,
                        "cv_r2_e_v1_slope_v": r2_sv,
                        "cv_r2_e_v1_radial": r2_rad,
                    }
                )
                if block_name == "cls_token" and r2_m > best_cls_r2:
                    best_cls_r2 = r2_m
                    oof_cls_med = pred_m
                if block_name == "depth_feat_stats" and r2_m > best_df_r2:
                    best_df_r2 = r2_m
                    oof_df_med = pred_m

        r2_df = pd.DataFrame(r2_rows).sort_values("cv_r2_e_v1_med", ascending=False)
        write_csv_lf(output_dir / "diag_v2_signal_r2.csv", r2_df)

        # 4. Left-Tail Focal Bimodality Analysis (f_tilde < 0.75)
        f_tilde_arr = sota_feat_df["f_tilde"].to_numpy(dtype=np.float64)
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
                "v1_med_bias": v1_bias,
                "raw_abs_rel": raw_scores[:, 0],
                "b3_abs_rel": b3_scores[:, 0],
                "v1_abs_rel": v1_scores[:, 0],
                "gain_v1_vs_raw_ar": raw_scores[:, 0] - v1_scores[:, 0],
                "gain_v1_vs_b3_ar": b3_scores[:, 0] - v1_scores[:, 0],
                "v1_flip_abs_rel": v1_flip_log_scores[:, 0],
                "regime": regimes,
                "oof_cls_pred_e_v1_med": oof_cls_med if oof_cls_med is not None else np.zeros(n),
                "oof_df_pred_e_v1_med": oof_df_med if oof_df_med is not None else np.zeros(n),
            }
        )
        # Merge key candidate discriminator features
        disc_pool = pd.concat(
            [
                inter_df.drop(columns=["f_tilde"]),
                sp_df[["dz_du", "dz_dr2", "top_minus_bot_logz", "center_minus_border_logz", "frac_near_1p2m", "frac_far_4m", "mean_inv_z", "iqr_logz"]],
                sota_feat_df[["q25_logz", "std_logz", "grad_logz", "dz_dv", "log_span_y", "span_x_over_z", "std_inv_z"]],
                conf_df,
                flip_df,
            ],
            axis=1,
        )
        lt_full_df = pd.concat([lt_df, disc_pool], axis=1)
        lt_sub = lt_full_df[lt_full_df["f_tilde"] < 0.75].sort_values("f_tilde")
        write_csv_lf(output_dir / "diag_v2_left_tail.csv", lt_sub)

        lt_bin = lt_sub[lt_sub["regime"].isin(["true_collapse", "false_alarm"])]
        y_collapse = (lt_bin["regime"] == "true_collapse").to_numpy()
        disc_rows: list[dict[str, Any]] = []
        cand_cols = [c for c in list(disc_pool.columns) + ["f_tilde", "oof_cls_pred_e_v1_med", "oof_df_pred_e_v1_med"] if c != "f_orig_tilde"]
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
        write_csv_lf(output_dir / "diag_v2_left_tail_discriminators.csv", disc_df)

        # 5. Scene-Type & Worst Failure Taxonomy
        scene_rows: list[dict[str, Any]] = []
        for stype, grp in lt_df.groupby("scene_type"):
            scene_rows.append(
                {
                    "group_type": "scene_type",
                    "name": str(stype),
                    "count": int(len(grp)),
                    "raw_abs_rel": float(grp["raw_abs_rel"].mean()),
                    "b3_abs_rel": float(grp["b3_abs_rel"].mean()),
                    "v1_abs_rel": float(grp["v1_abs_rel"].mean()),
                    "v1_flip_abs_rel": float(grp["v1_flip_abs_rel"].mean()),
                    "gain_v1_vs_b3": float(grp["b3_abs_rel"].mean() - grp["v1_abs_rel"].mean()),
                    "gain_flip_vs_v1": float(grp["v1_abs_rel"].mean() - grp["v1_flip_abs_rel"].mean()),
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
                    "b3_abs_rel": float(grp["b3_abs_rel"].mean()),
                    "v1_abs_rel": float(grp["v1_abs_rel"].mean()),
                    "v1_flip_abs_rel": float(grp["v1_flip_abs_rel"].mean()),
                    "gain_v1_vs_b3": float(grp["b3_abs_rel"].mean() - grp["v1_abs_rel"].mean()),
                    "gain_flip_vs_v1": float(grp["v1_abs_rel"].mean() - grp["v1_flip_abs_rel"].mean()),
                }
            )
        breakdown_df = pd.DataFrame(scene_rows)
        write_csv_lf(output_dir / "diag_v2_scene_breakdown.csv", breakdown_df)

        worst20 = lt_df.sort_values("v1_abs_rel", ascending=False).head(20)
        write_csv_lf(output_dir / "diag_v2_worst20_v1.csv", worst20)

    # Best R^2 per feature block
    best_r2_per_block = (
        r2_df.sort_values("cv_r2_e_v1_med", ascending=False)
        .groupby("feature_block", as_index=False)
        .first()
        .sort_values("cv_r2_e_v1_med", ascending=False)
    )

    summary_payload = {
        "split": "official795_training_only",
        "test_read": False,
        "images": int(n),
        "scenes": int(len(np.unique(scenes))),
        "reconstructed_v1_oof": {"abs_rel": float(v1_m[0]), "rmse": float(v1_m[1]), "delta1": float(v1_m[2])},
        "oracles": oracle_df.to_dict(orient="records"),
        "best_signal_r2_per_block": best_r2_per_block.to_dict(orient="records"),
        "left_tail_count_ftilde_lt_075": int(len(lt_sub)),
        "left_tail_count_ftilde_lt_055": int((lt_sub["f_tilde"] < 0.55).sum()),
        "left_tail_true_collapse_count": int((lt_sub["regime"] == "true_collapse").sum()),
        "left_tail_false_alarm_count": int((lt_sub["regime"] == "false_alarm").sum()),
        "top_discriminators_lt075": disc_df.head(15).to_dict(orient="records"),
    }
    write_json_lf(output_dir / "diag_v2_results.json", summary_payload)

    # Write clean human-readable Markdown summary
    md_lines = [
        "# Round 2 Part 1: Post-v1 Residual, Bimodality & Signal Predictability Diagnosis",
        "",
        "> Strictly training-only (`splits['train']`, 795 images / 249 scenes). `test_read: false`.",
        "",
        f"- Reconstructed `GeoCalib-3D v1` Canonical OOF: `AbsRel={v1_m[0]:.6f}`, `RMSE={v1_m[1]:.6f}`, `delta1={v1_m[2]:.6f}`",
        "",
        "## 1. Post-v1 Remaining Oracles & Train Flip-TTA Baselines",
        oracle_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 2. Best 5-Fold Scene-CV R^2 Predicting Residual v1 Error (`e_v1_med`, `e_v1_absrel`)",
        best_r2_per_block[["feature_block", "track", "raw_dims", "pca_k", "ridge_lam", "cv_r2_e_v1_med", "cv_r2_e_v1_absrel", "cv_r2_e_v1_slope_v", "cv_r2_e_v1_radial"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## 3. Left-Tail (`f_tilde < 0.75`, n={len(lt_sub)}) Bimodality Top Discriminators",
        f"- Counts: `true_collapse={(lt_sub['regime']=='true_collapse').sum()}`, `false_alarm={(lt_sub['regime']=='false_alarm').sum()}`, `mild={(lt_sub['regime']=='mild').sum()}`",
        "",
        disc_df.head(15)[["feature", "track", "mean_true_collapse", "mean_false_alarm", "roc_auc", "abs_auc", "cohens_d"]].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    write_text_lf(output_dir / "DIAGNOSIS_V2.md", "\n".join(md_lines) + "\n")
    print(f"Round 2 Gate 1 Diagnosis Complete -> {output_dir}", flush=True)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--orig-preds", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--flip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_v2_diagnosis(args.data_root, args.orig_preds, args.aux_dir, args.flip_dir, args.output)


if __name__ == "__main__":
    main()
