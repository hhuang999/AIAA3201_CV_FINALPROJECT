"""Held-out 654-image NYU test evaluation, 10k scene bootstrap CIs, and latency benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import nyu_depth, open_nyu, scene_names
from .geocalib_cache import load_geocalib_cache
from .geocalib_io import file_sha256, write_csv_lf, write_json_lf
from .geocalib_model import (
    _FULL_R2_T,
    _FULL_V_TILDE_T,
    BOOTSTRAP_SEED,
    PRIMARY_GEO_METHOD,
    PRIMARY_MULTIPASS_METHOD,
    PRIMARY_SINGLE_PASS_METHOD,
    _std_z,
    apply_geocalib_model,
    assemble_semantic_matrix,
    assemble_semantic_vector,
    count_model_parameters,
    extract_base_geo_features,
    extract_geocalib_features,
    load_frozen_geocalib,
)
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, evaluate_image, metrics, valid_depth


LEFT_TAIL_AUDIT_IDS = (839, 832, 326, 1150, 1151, 1097)


def _git_head(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def paired_scene_bootstrap(
    per_image_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    seed: int = BOOTSTRAP_SEED,
    replicates: int = 10000,
) -> pd.DataFrame:
    """Compute paired scene-cluster bootstrap CIs and one-sided p-values over image rows."""
    required_cols = {"index", "scene", "method", *METRIC_NAMES}
    missing = required_cols - set(per_image_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for paired bootstrap: {sorted(missing)}")
    if per_image_df.duplicated(["method", "index"]).any():
        raise ValueError("Duplicate (method, index) rows in per_image_df")
    if not np.isfinite(per_image_df[list(METRIC_NAMES)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite metrics in per_image_df")

    tables = {
        m: per_image_df[per_image_df["method"] == m].set_index("index").sort_index()
        for m in per_image_df["method"].unique()
    }
    direction = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for ref, cand in pairs:
        before, after = tables[ref], tables[cand]
        if not before.index.equals(after.index):
            raise ValueError(f"Mismatched image IDs between {ref!r} and {cand!r}")
        gain = (
            before[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
            - after[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        ) * direction[None, :]

        _, inverse = np.unique(before["scene"].to_numpy(), return_inverse=True)
        n_scenes = int(inverse.max() + 1)
        counts = np.bincount(inverse)
        sums = np.stack([np.bincount(inverse, weights=gain[:, c]) for c in range(3)], axis=1)

        boot = np.empty((replicates, 3), dtype=np.float64)
        for start in range(0, replicates, 250):
            step = min(250, replicates - start)
            draws = rng.integers(0, n_scenes, size=(step, n_scenes))
            boot[start : start + step] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]

        for col, metric in enumerate(METRIC_NAMES):
            low, high = np.percentile(boot[:, col], [2.5, 97.5])
            p_one_sided = float(np.mean(boot[:, col] <= 0.0))
            rows.append(
                {
                    "reference": ref,
                    "candidate": cand,
                    "metric": metric,
                    "gain": float(gain[:, col].mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "p_value_one_sided": p_one_sided,
                    "scenes": n_scenes,
                    "replicates": int(replicates),
                }
            )
    return pd.DataFrame(rows)


def compute_change_counts(
    per_image_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Count images improved, worsened, mixed/tied, or unchanged across paired methods."""
    tables = {
        m: per_image_df[per_image_df["method"] == m].set_index("index").sort_index()
        for m in per_image_df["method"].unique()
    }
    direction = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for ref, cand in pairs:
        before, after = tables[ref], tables[cand]
        if not before.index.equals(after.index):
            raise ValueError(f"Mismatched IDs for {ref!r} vs {cand!r}")
        gain = (
            before[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
            - after[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        ) * direction[None, :]
        all_better = int(np.sum((gain > 0).all(axis=1)))
        all_worse = int(np.sum((gain < 0).all(axis=1)))
        unchanged = int(np.sum((gain == 0).all(axis=1)))
        mixed = int(len(gain) - all_better - all_worse - unchanged)
        rows.append(
            {
                "reference": ref,
                "candidate": cand,
                "all_better": all_better,
                "all_worse": all_worse,
                "mixed_or_tied": mixed,
                "unchanged": unchanged,
                "abs_rel_improved": int(np.sum(gain[:, 0] > 0)),
                "rmse_improved": int(np.sum(gain[:, 1] > 0)),
                "delta1_improved": int(np.sum(gain[:, 2] > 0)),
                "total_images": int(len(gain)),
            }
        )
    return pd.DataFrame(rows)


def select_representative_cases(
    per_image_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    reference: str = "geo_focal_3d",
    candidate: str = PRIMARY_SINGLE_PASS_METHOD,
) -> pd.DataFrame:
    """Select deterministic largest-gain, percentile, and largest-regression cases."""
    tables = {
        m: per_image_df[per_image_df["method"] == m].set_index("index").sort_index()
        for m in per_image_df["method"].unique()
    }
    raw_tbl = tables["raw"]
    ref_tbl = tables[reference]
    cand_tbl = tables[candidate]
    diag_tbl = diag_df.set_index("index").reindex(ref_tbl.index)

    direction = np.array([1.0, 1.0, -1.0], dtype=np.float64)
    ref_vals = ref_tbl[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    cand_vals = cand_tbl[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    gain = (ref_vals - cand_vals) * direction[None, :]
    ref_means = np.maximum(ref_vals.mean(axis=0), 1e-12)
    comp_score = (gain / ref_means[None, :]).mean(axis=1)

    order = np.argsort(comp_score)
    n = len(order)
    selections = [
        ("largest_gain_1", int(order[-1])),
        ("largest_gain_2", int(order[-2])),
        ("largest_gain_3", int(order[-3])),
        ("p75_gain", int(order[int(round(0.75 * (n - 1)))])),
        ("median_gain", int(order[int(round(0.50 * (n - 1)))])),
        ("largest_regression_1", int(order[0])),
        ("largest_regression_2", int(order[1])),
        ("largest_regression_3", int(order[2])),
    ]

    rows: list[dict[str, Any]] = []
    for label, row_pos in selections:
        idx = int(ref_tbl.index[row_pos])
        rows.append(
            {
                "case_label": label,
                "index": idx,
                "scene": str(ref_tbl["scene"].iloc[row_pos]),
                "composite_score": float(comp_score[row_pos]),
                "f_tilde": float(diag_tbl["f_tilde"].iloc[row_pos]),
                "raw_abs_rel": float(raw_tbl["abs_rel"].iloc[row_pos]),
                "raw_rmse": float(raw_tbl["rmse"].iloc[row_pos]),
                "raw_delta1": float(raw_tbl["delta1"].iloc[row_pos]),
                "ref_abs_rel": float(ref_vals[row_pos, 0]),
                "ref_rmse": float(ref_vals[row_pos, 1]),
                "ref_delta1": float(ref_vals[row_pos, 2]),
                "cand_abs_rel": float(cand_vals[row_pos, 0]),
                "cand_rmse": float(cand_vals[row_pos, 1]),
                "cand_delta1": float(cand_vals[row_pos, 2]),
                "gain_abs_rel": float(gain[row_pos, 0]),
                "gain_rmse": float(gain[row_pos, 1]),
                "gain_delta1": float(gain[row_pos, 2]),
            }
        )
    return pd.DataFrame(rows)


def build_generalization_gap(oof_summary_df: pd.DataFrame, test_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare 5-fold OOF metrics and gains vs monotone_spline and geo_focal_3d against test."""
    oof_t = oof_summary_df.set_index("method")
    test_t = test_summary_df.set_index("method")
    oof_sp = oof_t.loc["monotone_spline", list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    test_sp = test_t.loc["monotone_spline", list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    oof_geo = oof_t.loc[PRIMARY_GEO_METHOD, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    test_geo = test_t.loc[PRIMARY_GEO_METHOD, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    direction = np.array([-1.0, -1.0, 1.0], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for method in test_summary_df["method"].tolist():
        if method not in oof_t.index:
            continue
        o_v = oof_t.loc[method, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        t_v = test_t.loc[method, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        o_g_sp = (o_v - oof_sp) * direction
        t_g_sp = (t_v - test_sp) * direction
        o_g_geo = (o_v - oof_geo) * direction
        t_g_geo = (t_v - test_geo) * direction
        rows.append(
            {
                "method": method,
                "oof_abs_rel": float(o_v[0]),
                "test_abs_rel": float(t_v[0]),
                "oof_gain_vs_spline_abs_rel": float(o_g_sp[0]),
                "test_gain_vs_spline_abs_rel": float(t_g_sp[0]),
                "test_minus_oof_gain_spline_abs_rel": float(t_g_sp[0] - o_g_sp[0]),
                "oof_rmse": float(o_v[1]),
                "test_rmse": float(t_v[1]),
                "oof_gain_vs_spline_rmse": float(o_g_sp[1]),
                "test_gain_vs_spline_rmse": float(t_g_sp[1]),
                "test_minus_oof_gain_spline_rmse": float(t_g_sp[1] - o_g_sp[1]),
                "oof_delta1": float(o_v[2]),
                "test_delta1": float(t_v[2]),
                "oof_gain_vs_spline_delta1": float(o_g_sp[2]),
                "test_gain_vs_spline_delta1": float(t_g_sp[2]),
                "test_minus_oof_gain_spline_delta1": float(t_g_sp[2] - o_g_sp[2]),
                "oof_gain_vs_geo_abs_rel": float(o_g_geo[0]),
                "test_gain_vs_geo_abs_rel": float(t_g_geo[0]),
                "test_minus_oof_gain_geo_abs_rel": float(t_g_geo[0] - o_g_geo[0]),
                "oof_gain_vs_geo_rmse": float(o_g_geo[1]),
                "test_gain_vs_geo_rmse": float(t_g_geo[1]),
                "test_minus_oof_gain_geo_rmse": float(t_g_geo[1] - o_g_geo[1]),
                "oof_gain_vs_geo_delta1": float(o_g_geo[2]),
                "test_gain_vs_geo_delta1": float(t_g_geo[2]),
                "test_minus_oof_gain_geo_delta1": float(t_g_geo[2] - o_g_geo[2]),
            }
        )
    return pd.DataFrame(rows)


def decompose_shift_and_spatial(
    model: dict[str, Any],
    feats: dict[str, float],
    sem_vec: np.ndarray | None,
) -> dict[str, float]:
    """Extract exact scalar shift and 2D spatial bias terms for diagnostic interpretability."""
    if model.get("kind") != "geocalib_3d":
        return {"focal": 0.0, "geo": 0.0, "sem": 0.0, "total": 0.0, "c_v": 0.0, "c_r2": 0.0}
    fs = model["focal_spline"]
    hf = float(np.interp(feats[fs["focal_key"]], fs["x_knots"], fs["y_knots"]))
    fg = model["focal_gate"]
    if fg["enabled"]:
        zg = _std_z(feats, fg)
        hf *= float(1.0 / (1.0 + np.exp(-(float(fg["bias"]) + float(zg @ np.asarray(fg["weights"], dtype=np.float64))))))
    sr = model["scale_residual"]
    z_sc = _std_z(feats, sr)
    geo_s = 0.40 * float(np.tanh(float(z_sc @ np.asarray(sr["weights"], dtype=np.float64)) / 0.40))
    sb = model["semantic_branch"]
    sem_s, c_v, c_r2 = 0.0, 0.0, 0.0
    if sb["raw_dims"] > 0 and sem_vec is not None:
        xc = np.asarray(sem_vec, dtype=np.float64) - np.asarray(sb["mean"], dtype=np.float64)
        cap = float(sb["scale_cap"])
        sem_s = cap * float(np.tanh(float(xc @ np.asarray(sb["w_scale"], dtype=np.float64)) / cap))
        sp2d = sb["spatial2d"]
        if sp2d["enabled"]:
            c_v = 0.14 * float(np.tanh(float(sp2d["a_v0"]) + float(xc @ np.asarray(sp2d["w_v"], dtype=np.float64))))
            c_r2 = 0.14 * float(np.tanh(float(sp2d["a_r0"]) + float(xc @ np.asarray(sp2d["w_r"], dtype=np.float64))))
    total = 0.50 * float(np.tanh((hf + geo_s + sem_s) / 0.50))
    return {"focal": hf, "geo": geo_s, "sem": sem_s, "total": total, "c_v": c_v, "c_r2": c_r2}


def _apply_geocalib_gpu_exact(
    p1_np: np.ndarray,
    k1_np: np.ndarray,
    model: dict[str, Any],
    dev: torch.device,
    flip_depth_np: np.ndarray | None = None,
    flip_intrinsics_np: np.ndarray | None = None,
    conf_orig_np: np.ndarray | None = None,
    conf_flip_np: np.ndarray | None = None,
    conf_sum_np: np.ndarray | None = None,
    cls_orig_np: np.ndarray | None = None,
    cls_flip_np: np.ndarray | None = None,
    pat_m_np: np.ndarray | None = None,
    pat_s_np: np.ndarray | None = None,
    df_np: np.ndarray | None = None,
) -> torch.Tensor:
    """Exact CUDA float64 forward matching pure NumPy within <= 1e-13 m."""
    feats = extract_geocalib_features(
        p1_np, k1_np, conf_summary=conf_sum_np, flip_depth=flip_depth_np,
        flip_intrinsics=flip_intrinsics_np, conf_orig=conf_orig_np, conf_flip=conf_flip_np,
        cls_orig=cls_orig_np, cls_flip=cls_flip_np,
    )
    sb = model["semantic_branch"]
    sem_vec = (
        assemble_semantic_vector(sb["sem_mode"], cls_orig_np, pat_m_np, pat_s_np, df_np, cls_flip_np)
        if sb["raw_dims"] > 0
        else None
    )
    dec = decompose_shift_and_spatial(model, feats, sem_vec)

    dsm = model["depth_slope_mod"]
    z_sl = _std_z(feats, dsm)
    slope_lin = float(z_sl @ np.asarray(dsm["weights"], dtype=np.float64))
    if sem_vec is not None and sb["w_slope"]:
        xc = sem_vec - np.asarray(sb["mean"], dtype=np.float64)
        slope_lin += float(xc @ np.asarray(sb["w_slope"], dtype=np.float64))
    delta_b = 0.20 * float(np.tanh(slope_lin))

    p1_g = torch.from_numpy(np.ascontiguousarray(p1_np)).to(dev)
    ff = model["flip_fusion"]
    if ff["mode"] == "learnable":
        assert flip_depth_np is not None
        p2_g = torch.from_numpy(np.ascontiguousarray(flip_depth_np)).to(dev)
        wf = float(ff["w_flip"])
        pin_g = torch.exp(wf * torch.log(p1_g) + (1.0 - wf) * torch.log(p2_g))
    else:
        pin_g = p1_g

    lp_g = torch.log(pin_g)
    db = model["depth_branch"]
    xk = np.asarray(db["x_knots"], dtype=np.float64)
    yk = np.asarray(db["y_knots"], dtype=np.float64)
    slopes = np.diff(yk) / np.diff(xk)
    dx = np.diff(xk)
    if db["param"] == "inv_spline":
        invp_g = torch.reciprocal(pin_g)
        dspl = float(yk[0]) + float(slopes[0]) * torch.clamp(invp_g - float(xk[0]), max=float(dx[0]))
        for ki in range(1, len(slopes) - 1):
            dspl.add_(float(slopes[ki]) * torch.clamp(invp_g - float(xk[ki]), min=0.0, max=float(dx[ki])))
        dspl.add_(float(slopes[-1]) * torch.clamp(invp_g - float(xk[-2]), min=0.0))
        out_g = -torch.log(dspl.clamp_min_(0.04))
    else:
        out_g = float(yk[0]) + float(slopes[0]) * torch.clamp(lp_g - float(xk[0]), max=float(dx[0]))
        for ki in range(1, len(slopes) - 1):
            out_g.add_(float(slopes[ki]) * torch.clamp(lp_g - float(xk[ki]), min=0.0, max=float(dx[ki])))
        out_g.add_(float(slopes[-1]) * torch.clamp(lp_g - float(xk[-2]), min=0.0))

    out_g.add_(dec["total"])
    if delta_b != 0.0:
        out_g.add_(lp_g - float(dsm["log_depth_center"]), alpha=delta_b)
    if dec["c_v"] != 0.0 or dec["c_r2"] != 0.0:
        out_g.add_(_FULL_V_TILDE_T.to(dev), alpha=dec["c_v"])
        out_g.add_(_FULL_R2_T.to(dev) - float(sb["spatial2d"]["mean_r2_crop"]), alpha=dec["c_r2"])
    return torch.exp(out_g.clamp_(float(np.log(0.001)), float(np.log(10.0)))).clamp_(0.001, 10.0)


def benchmark_uncontended_runtime(
    cache,
    test_ids: np.ndarray,
    models: dict[str, Any],
    aux_runtime_path: Path,
    flip_runtime_path: Path,
    n_images: int = 200,
) -> dict[str, Any]:
    """Measure uncontended post-processing runtime and Torch/NumPy/GPU parity across test images."""
    bench_ids = test_ids[: min(n_images, len(test_ids))]
    p1_l = [cache[int(i)].astype(np.float64) for i in bench_ids]
    p2_l = [cache.depth_flip(int(i)).astype(np.float64) for i in bench_ids]
    c1_l = [np.maximum(cache.confidence(int(i)).astype(np.float64), 1e-4) for i in bench_ids]
    c2_l = [np.maximum(cache.confidence_flip(int(i)).astype(np.float64), 1e-4) for i in bench_ids]
    k1_l = [cache.intrinsics(int(i)).astype(np.float64) for i in bench_ids]
    k2_l = [cache.intrinsics_flip(int(i)).astype(np.float64) for i in bench_ids]
    cs_l = cache.confidence_summary_for(bench_ids)
    cls1_l = cache.cls_tokens_for(bench_ids)
    cls2_l = cache.cls_tokens_flip_for(bench_ids)
    pm_l = cache.patch_mean_for(bench_ids)
    ps_l = cache.patch_std_for(bench_ids)
    df_l = cache.depth_feat_stats_for(bench_ids)

    m_sp_line = models["monotone_spline"]
    m_geo = models[PRIMARY_GEO_METHOD]
    m_full = models[PRIMARY_SINGLE_PASS_METHOD]
    m_flip = models[PRIMARY_MULTIPASS_METHOD]
    sm_sp = m_full["semantic_branch"]["sem_mode"]
    sm_mp = m_flip["semantic_branch"]["sem_mode"]

    max_diff_sp_cpu, max_diff_mp_cpu = 0.0, 0.0
    for j in range(min(20, len(bench_ids))):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        s_t = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
        s_n = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
        max_diff_sp_cpu = max(max_diff_sp_cpu, float(np.max(np.abs(s_t - s_n))))

        f_t = apply_geocalib_model(
            p1_l[j], k1_l[j], m_flip, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )
        f_n = apply_geocalib_model(
            p1_l[j], k1_l[j], m_flip, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=False,
        )
        max_diff_mp_cpu = max(max_diff_mp_cpu, float(np.max(np.abs(f_t - f_n))))

    for j in range(10):
        _ = apply_geocalib_model(p1_l[j], k1_l[j], m_sp_line, use_torch_fast=True)
        _ = apply_geocalib_model(p1_l[j], k1_l[j], m_geo, use_torch_fast=True)
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        _ = apply_geocalib_model(
            p1_l[j], k1_l[j], m_flip, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )

    t0 = perf_counter()
    for p, k in zip(p1_l, k1_l):
        _ = apply_geocalib_model(p, k, m_sp_line, use_torch_fast=True)
    spline_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for p, k in zip(p1_l, k1_l):
        _ = apply_geocalib_model(p, k, m_geo, use_torch_fast=True)
    geo_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
    full_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
    full_np_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        _ = apply_geocalib_model(
            p1_l[j], k1_l[j], m_flip, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )
    flip_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    aux_info = json.loads(aux_runtime_path.read_text(encoding="utf-8")) if aux_runtime_path.is_file() else {}
    flip_info = json.loads(flip_runtime_path.read_text(encoding="utf-8")) if flip_runtime_path.is_file() else {}
    bb_1pass_ms = round(float(aux_info.get("mean_seconds_per_image", 0.04304)) * 1000.0, 2)
    bb_flip_delta_ms = round(float(flip_info.get("mean_seconds_per_image", 0.04413)) * 1000.0, 2)
    bb_2pass_ms = round(bb_1pass_ms + bb_flip_delta_ms, 2)

    res: dict[str, Any] = {
        "benchmark_images": len(bench_ids),
        "uncontended_threads": torch.get_num_threads(),
        "backbone_device": aux_info.get("gpu_name", "GPU"),
        "backbone_single_pass_ms_per_image": bb_1pass_ms,
        "backbone_two_pass_flip_ms_per_image": bb_2pass_ms,
        "monotone_spline_cpu_post_ms": float(spline_cpu_ms),
        "geo_focal_3d_cpu_post_ms": float(geo_cpu_ms),
        "geocalib_3d_cpu_post_ms": float(full_cpu_ms),
        "geocalib_3d_numpy_post_ms": float(full_np_ms),
        "geocalib_3d_flip_cpu_post_ms": float(flip_cpu_ms),
        "end_to_end_spline_ms": float(bb_1pass_ms + spline_cpu_ms),
        "end_to_end_geo_focal_3d_ms": float(bb_1pass_ms + geo_cpu_ms),
        "end_to_end_geocalib_3d_cpu_post_ms": float(bb_1pass_ms + full_cpu_ms),
        "end_to_end_geocalib_3d_flip_cpu_post_ms": float(bb_2pass_ms + flip_cpu_ms),
        "max_abs_diff_sp_cpu_vs_numpy": float(max_diff_sp_cpu),
        "max_abs_diff_mp_cpu_vs_numpy": float(max_diff_mp_cpu),
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        max_diff_sp_gpu, max_diff_mp_gpu = 0.0, 0.0
        for j in range(10):
            xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
            g_sp = _apply_geocalib_gpu_exact(
                p1_l[j], k1_l[j], m_full, dev, conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            ).cpu().numpy()
            n_sp = apply_geocalib_model(p1_l[j], k1_l[j], m_full, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
            max_diff_sp_gpu = max(max_diff_sp_gpu, float(np.max(np.abs(g_sp - n_sp))))

            xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
            g_mp = _apply_geocalib_gpu_exact(
                p1_l[j], k1_l[j], m_flip, dev, flip_depth_np=p2_l[j], flip_intrinsics_np=k2_l[j],
                conf_orig_np=c1_l[j], conf_flip_np=c2_l[j], conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], cls_flip_np=cls2_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            ).cpu().numpy()
            n_mp = apply_geocalib_model(
                p1_l[j], k1_l[j], m_flip, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
                conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
                cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=False,
            )
            max_diff_mp_gpu = max(max_diff_mp_gpu, float(np.max(np.abs(g_mp - n_mp))))

        torch.cuda.synchronize()
        t_g0 = perf_counter()
        for j in range(len(bench_ids)):
            _ = _apply_geocalib_gpu_exact(
                p1_l[j], k1_l[j], m_full, dev, conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            )
        torch.cuda.synchronize()
        sp_gpu_ms = (perf_counter() - t_g0) * 1000.0 / len(bench_ids)

        t_g1 = perf_counter()
        for j in range(len(bench_ids)):
            _ = _apply_geocalib_gpu_exact(
                p1_l[j], k1_l[j], m_flip, dev, flip_depth_np=p2_l[j], flip_intrinsics_np=k2_l[j],
                conf_orig_np=c1_l[j], conf_flip_np=c2_l[j], conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], cls_flip_np=cls2_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            )
        torch.cuda.synchronize()
        mp_gpu_ms = (perf_counter() - t_g1) * 1000.0 / len(bench_ids)

        res["geocalib_3d_gpu_post_ms"] = float(sp_gpu_ms)
        res["geocalib_3d_flip_gpu_post_ms"] = float(mp_gpu_ms)
        res["end_to_end_geocalib_3d_gpu_ms"] = float(bb_1pass_ms + sp_gpu_ms)
        res["end_to_end_geocalib_3d_flip_gpu_ms"] = float(bb_2pass_ms + mp_gpu_ms)
        res["max_abs_diff_sp_gpu_vs_numpy"] = float(max_diff_sp_gpu)
        res["max_abs_diff_mp_gpu_vs_numpy"] = float(max_diff_mp_gpu)
        res["gpu_name"] = str(torch.cuda.get_device_name(0))
    return res


def evaluate_heldout_test(
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    frozen_model_path: Path,
    oof_summary_path: Path,
    output_dir: Path,
    expected_model_sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute held-out 654-image NYU test evaluation from the verified frozen artifact."""
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    summary_file = output_dir / "test_summary.csv"
    if summary_file.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {summary_file}; pass --force for an explicit replay"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_frozen_geocalib(frozen_model_path, expected_sha256=expected_model_sha256)
    frozen_sha = file_sha256(frozen_model_path)

    cache = load_geocalib_cache(orig_preds, aux_dir, flip_dir, data_root, allowed_split="test", compute_hashes=True, verify_scenes=True)
    for k, expected_d in frozen["cache_sha256"].items():
        if cache.hashes.get(k) != expected_d:
            raise ValueError(f"Cache SHA-256 mismatch for {k}")

    test_ids = cache.test_ids
    win_sp = frozen["primary_single_pass_method"]
    win_mp = frozen["primary_multipass_method"]
    methods = tuple(frozen["locked_eval_order"])
    models = frozen["models"]

    k1_all = cache.intrinsics_for(test_ids).astype(np.float64)
    k2_all = cache.intrinsics_flip_for(test_ids).astype(np.float64)
    cs_all = cache.confidence_summary_for(test_ids)
    cls1_all = cache.cls_tokens_for(test_ids)
    cls2_all = cache.cls_tokens_flip_for(test_ids)
    pm_all = cache.patch_mean_for(test_ids)
    ps_all = cache.patch_std_for(test_ids)
    df_all = cache.depth_feat_stats_for(test_ids)

    sem_mats = {
        sm: assemble_semantic_matrix(sm, cls1_all, pm_all, ps_all, df_all, cls2_all)
        for sm in ("none", "all_sp", "all_mp")
    }

    all_f_tilde = 0.5 * (k1_all[:, 0, 0] + k1_all[:, 1, 1]) / 480.0
    f_q25 = float(np.quantile(all_f_tilde, 0.25))
    f_q75 = float(np.quantile(all_f_tilde, 0.75))

    row_bins = np.array_split(np.arange(426), 12)
    col_bins = np.array_split(np.arange(560), 16)
    row_starts = np.array([int(b[0]) for b in row_bins], dtype=np.int64)
    col_starts = np.array([int(b[0]) for b in col_bins], dtype=np.int64)
    row_edges = [45 + int(b[0]) for b in row_bins] + [471]
    col_edges = [41 + int(b[0]) for b in col_bins] + [601]

    spatial_methods = ("raw", "monotone_spline", PRIMARY_GEO_METHOD, win_sp, win_mp)
    spatial_subsets = ("all", "low_f_tilde", "high_f_tilde", "left_tail_lt055")
    sp_log_sum = {sub: {m: np.zeros((12, 16), dtype=np.float64) for m in spatial_methods} for sub in spatial_subsets}
    sp_cnt_sum = {sub: np.zeros((12, 16), dtype=np.float64) for sub in spatial_subsets}

    def _reduce_12x16(arr2d: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(arr2d, row_starts, axis=0), col_starts, axis=1)

    records: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    audit_set = set(LEFT_TAIL_AUDIT_IDS)

    with open_nyu(data_root) as handle:
        scenes = scene_names(handle)
        for pos, idx in enumerate(test_ids):
            idx_int = int(idx)
            sc = str(scenes[idx_int])
            p1 = cache[idx_int].astype(np.float64)
            p2 = cache.depth_flip(idx_int).astype(np.float64)
            c1 = np.maximum(cache.confidence(idx_int).astype(np.float64), 1e-4)
            c2 = np.maximum(cache.confidence_flip(idx_int).astype(np.float64), 1e-4)
            k1 = k1_all[pos]
            k2 = k2_all[pos]
            gt_full = nyu_depth(handle, idx_int)
            gt_crop = gt_full[CROP]
            valid = valid_depth(gt_crop)

            base_feats = extract_base_geo_features(p1, k1, None, need_rgb=False)
            full_feats = extract_geocalib_features(
                p1, k1, cached_base_feat=base_feats, conf_summary=cs_all[pos],
                flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2,
                cls_orig=cls1_all[pos], cls_flip=cls2_all[pos],
            )

            preds: dict[str, np.ndarray] = {}
            for nm in methods:
                m_obj = models[nm]
                if m_obj.get("kind") == "geocalib_3d":
                    sm = m_obj["semantic_branch"]["sem_mode"]
                    x_row = sem_mats[sm][pos] if (sm in sem_mats and sem_mats[sm].shape[1]) else None
                    preds[nm] = apply_geocalib_model(
                        p1, k1, m_obj, flip_depth=p2, flip_intrinsics=k2,
                        conf_orig=c1, conf_flip=c2, conf_summary=cs_all[pos],
                        cls_orig=cls1_all[pos], cls_flip=cls2_all[pos],
                        cached_features=full_feats, cached_sem=x_row, use_torch_fast=True,
                    )
                else:
                    preds[nm] = apply_geocalib_model(
                        p1, k1, m_obj, flip_depth=p2, flip_intrinsics=k2,
                        cached_features=base_feats, use_torch_fast=True,
                    )

            img_scores: dict[str, tuple[float, float, float]] = {}
            for m_name in methods:
                d_pred = preds[m_name]
                ar, rm, d1 = evaluate_image(d_pred, gt_full)
                img_scores[m_name] = (float(ar), float(rm), float(d1))
                records.append({"index": idx_int, "scene": sc, "method": m_name, "abs_rel": float(ar), "rmse": float(rm), "delta1": float(d1)})
                d_crop = d_pred[CROP]
                for lower, upper in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                    mask = valid & (gt_crop >= lower) & (gt_crop < upper)
                    if mask.any():
                        b_ar, b_rmse, b_d1 = metrics(d_crop, gt_crop, mask)
                        binned.append({
                            "index": idx_int, "scene": sc, "method": m_name,
                            "lower_m": float(lower), "upper_m": float(upper),
                            "pixels": int(mask.sum()), "abs_rel": float(b_ar), "rmse": float(b_rmse), "delta1": float(b_d1),
                        })

            dec_sp = decompose_shift_and_spatial(models[win_sp], full_feats, sem_mats[models[win_sp]["semantic_branch"]["sem_mode"]][pos])
            dec_mp = decompose_shift_and_spatial(models[win_mp], full_feats, sem_mats[models[win_mp]["semantic_branch"]["sem_mode"]][pos])

            diag_rows.append({
                "index": idx_int,
                "scene": sc,
                "f_tilde": float(full_feats["f_tilde"]),
                "f_flip_tilde": float(full_feats["f2_tilde"]),
                "f_abs_diff": float(full_feats["f_abs_diff"]),
                "fx": float(k1[0, 0]),
                "fy": float(k1[1, 1]),
                "log_span_y": float(full_feats["log_span_y"]),
                "frac_far_4m": float(full_feats["frac_far_4m"]),
                "median_log_err_raw": float(np.median(np.log(preds["raw"][CROP][valid] / gt_crop[valid]))),
                "median_log_err_spline": float(np.median(np.log(preds["monotone_spline"][CROP][valid] / gt_crop[valid]))),
                "median_log_err_geo": float(np.median(np.log(preds[PRIMARY_GEO_METHOD][CROP][valid] / gt_crop[valid]))),
                "median_log_err_sp": float(np.median(np.log(preds[win_sp][CROP][valid] / gt_crop[valid]))),
                "median_log_err_mp": float(np.median(np.log(preds[win_mp][CROP][valid] / gt_crop[valid]))),
                "abs_rel_geo": img_scores[PRIMARY_GEO_METHOD][0],
                "abs_rel_sp": img_scores[win_sp][0],
                "abs_rel_mp": img_scores[win_mp][0],
                "shift_sp_total": dec_sp["total"],
                "shift_sp_focal": dec_sp["focal"],
                "shift_sp_geo": dec_sp["geo"],
                "shift_sp_sem": dec_sp["sem"],
                "sp2d_v_sp": dec_sp["c_v"],
                "sp2d_r2_sp": dec_sp["c_r2"],
                "shift_mp_total": dec_mp["total"],
            })

            if idx_int in audit_set:
                for m_lbl in ("raw", "monotone_spline", PRIMARY_GEO_METHOD, win_sp, win_mp):
                    ar, rm, d1 = img_scores[m_lbl]
                    audit_rows.append({
                        "index": idx_int,
                        "scene": sc,
                        "regime_group": "false_focal_alarm" if idx_int in (839, 832, 326) else "wide_fov_collapse",
                        "method": m_lbl,
                        "f_tilde": float(full_feats["f_tilde"]),
                        "f_flip_tilde": float(full_feats["f2_tilde"]),
                        "log_span_y": float(full_feats["log_span_y"]),
                        "frac_far_4m": float(full_feats["frac_far_4m"]),
                        "abs_rel": ar,
                        "rmse": rm,
                        "delta1": d1,
                        "delta_vs_geo_abs_rel": img_scores[PRIMARY_GEO_METHOD][0] - ar,
                        "delta_vs_geo_delta1": d1 - img_scores[PRIMARY_GEO_METHOD][2],
                        "shift_sp_focal": dec_sp["focal"],
                        "shift_sp_geo": dec_sp["geo"],
                        "shift_sp_sem": dec_sp["sem"],
                        "shift_sp_total": dec_sp["total"],
                    })

            active_subs = ["all"]
            if full_feats["f_tilde"] <= f_q25:
                active_subs.append("low_f_tilde")
            elif full_feats["f_tilde"] >= f_q75:
                active_subs.append("high_f_tilde")
            if full_feats["f_tilde"] < 0.55:
                active_subs.append("left_tail_lt055")
            cnt_grid = _reduce_12x16(valid.astype(np.float64))
            for sub in active_subs:
                sp_cnt_sum[sub] += cnt_grid
            gt_safe = np.where(valid, gt_crop, 1.0)
            for m_name in spatial_methods:
                pc = preds[m_name][CROP]
                lg_grid = _reduce_12x16(np.where(valid, np.log(pc / gt_safe), 0.0))
                for sub in active_subs:
                    sp_log_sum[sub][m_name] += lg_grid

            if (pos + 1) % 150 == 0 or pos + 1 == len(test_ids):
                print(f"Held-out test eval {pos + 1}/{len(test_ids)}", flush=True)

    per_image_df = pd.DataFrame(records)
    diag_df = pd.DataFrame(diag_rows)
    bins_raw_df = pd.DataFrame(binned)
    audit_df = pd.DataFrame(audit_rows)

    oof_df = pd.read_csv(oof_summary_path)
    oof_t = oof_df.set_index("method")
    meta_map: dict[str, tuple[str, int, int, float]] = {}
    for nm in methods:
        rp, dof = count_model_parameters(models[nm])
        if nm in oof_t.index:
            tr = str(oof_t.loc[nm, "track"])
            bp = int(oof_t.loc[nm, "backbone_passes"])
        else:
            bp = 2 if nm.startswith("flip_") or nm.endswith("_flip") else 1
            tr = "multi_pass_flip_tta" if bp == 2 else "single_pass"
        meta_map[nm] = (tr, bp, rp, dof)

    sum_df = per_image_df.groupby("method", sort=False)[list(METRIC_NAMES)].mean().reindex(methods).reset_index()
    raw_arr = sum_df.loc[sum_df["method"] == "raw", list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]
    sp_arr = sum_df.loc[sum_df["method"] == "monotone_spline", list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]
    geo_arr = sum_df.loc[sum_df["method"] == PRIMARY_GEO_METHOD, list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]

    sum_df["track"] = [meta_map[m][0] for m in sum_df["method"]]
    sum_df["backbone_passes"] = [meta_map[m][1] for m in sum_df["method"]]
    sum_df["raw_params"] = [meta_map[m][2] for m in sum_df["method"]]
    sum_df["effective_dof"] = [meta_map[m][3] for m in sum_df["method"]]
    sum_df["gain_vs_raw_abs_rel"] = raw_arr[0] - sum_df["abs_rel"]
    sum_df["gain_vs_raw_rmse"] = raw_arr[1] - sum_df["rmse"]
    sum_df["gain_vs_raw_delta1"] = sum_df["delta1"] - raw_arr[2]
    sum_df["gain_vs_spline_abs_rel"] = sp_arr[0] - sum_df["abs_rel"]
    sum_df["gain_vs_spline_rmse"] = sp_arr[1] - sum_df["rmse"]
    sum_df["gain_vs_spline_delta1"] = sum_df["delta1"] - sp_arr[2]
    sum_df["gain_vs_geo_abs_rel"] = geo_arr[0] - sum_df["abs_rel"]
    sum_df["gain_vs_geo_rmse"] = geo_arr[1] - sum_df["rmse"]
    sum_df["gain_vs_geo_delta1"] = sum_df["delta1"] - geo_arr[2]
    sum_df = sum_df[
        [
            "method", "track", "backbone_passes", "raw_params", "effective_dof",
            "abs_rel", "rmse", "delta1",
            "gain_vs_raw_abs_rel", "gain_vs_raw_rmse", "gain_vs_raw_delta1",
            "gain_vs_spline_abs_rel", "gain_vs_spline_rmse", "gain_vs_spline_delta1",
            "gain_vs_geo_abs_rel", "gain_vs_geo_rmse", "gain_vs_geo_delta1",
        ]
    ]

    pairs: list[tuple[str, str]] = []
    for cand_m in methods:
        if cand_m != "raw":
            pairs.append(("raw", cand_m))
    for cand_m in methods:
        if cand_m not in ("raw", "global_scale", "monotone_spline"):
            pairs.append(("monotone_spline", cand_m))
    for cand_m in methods:
        if cand_m not in ("raw", "global_scale", "monotone_spline", PRIMARY_GEO_METHOD):
            pairs.append((PRIMARY_GEO_METHOD, cand_m))
    pairs.append((win_sp, win_mp))

    boot_df = paired_scene_bootstrap(per_image_df, pairs)
    change_df = compute_change_counts(per_image_df, pairs)

    c_spline = select_representative_cases(per_image_df, diag_df, reference="monotone_spline", candidate=win_sp)
    c_spline.insert(0, "comparison", f"{win_sp}_vs_monotone_spline")
    c_geo = select_representative_cases(per_image_df, diag_df, reference=PRIMARY_GEO_METHOD, candidate=win_sp)
    c_geo.insert(0, "comparison", f"{win_sp}_vs_{PRIMARY_GEO_METHOD}")
    c_mp = select_representative_cases(per_image_df, diag_df, reference=PRIMARY_GEO_METHOD, candidate=win_mp)
    c_mp.insert(0, "comparison", f"{win_mp}_vs_{PRIMARY_GEO_METHOD}")
    cases_df = pd.concat([c_spline, c_geo, c_mp], ignore_index=True)

    bins_df = (
        bins_raw_df.groupby(["method", "lower_m", "upper_m"], sort=False)
        .agg(abs_rel=("abs_rel", "mean"), rmse=("rmse", "mean"), delta1=("delta1", "mean"), images=("index", "count"), pixels=("pixels", "sum"))
        .reset_index()
    )
    m_rank = {m: i for i, m in enumerate(methods)}
    bins_df["_r"] = bins_df["method"].map(m_rank)
    bins_df = bins_df.sort_values(["_r", "lower_m"]).drop(columns=["_r"]).reset_index(drop=True)

    gap_df = build_generalization_gap(oof_df, sum_df)

    spatial_payload: dict[str, Any] = {
        "row_edges": row_edges,
        "col_edges": col_edges,
        "f_tilde_q25": f_q25,
        "f_tilde_q75": f_q75,
        "maps": {},
    }
    for sub in spatial_subsets:
        cnt = np.maximum(sp_cnt_sum[sub], 1.0)
        spatial_payload["maps"][sub] = {}
        for m_name in spatial_methods:
            log_map = sp_log_sum[sub][m_name] / cnt
            spatial_payload["maps"][sub][m_name] = {
                "mean_log_residual_12x16": log_map.tolist(),
                "rms_spatial_log_bias": float(np.sqrt(np.mean(log_map ** 2))),
                "max_abs_spatial_log_bias": float(np.max(np.abs(log_map))),
            }

    runtime = benchmark_uncontended_runtime(
        cache, test_ids, models, aux_dir / "runtime_aux.json", flip_dir / "runtime_flip.json", n_images=200
    )

    write_csv_lf(output_dir / "test_summary.csv", sum_df)
    write_csv_lf(output_dir / "test_per_image.csv", per_image_df)
    write_csv_lf(output_dir / "test_paired_bootstrap.csv", boot_df)
    write_csv_lf(output_dir / "test_depth_bins.csv", bins_df)
    write_csv_lf(output_dir / "test_change_counts.csv", change_df)
    write_csv_lf(output_dir / "test_left_tail_case_audit.csv", audit_df)
    write_csv_lf(output_dir / "test_representative_cases.csv", cases_df)
    write_csv_lf(output_dir / "generalization_gap.csv", gap_df)
    write_csv_lf(output_dir / "test_mechanism_diagnostics.csv", diag_df)
    write_json_lf(output_dir / "test_spatial_residuals.json", spatial_payload)
    write_json_lf(output_dir / "runtime_overhead.json", runtime)

    repo_root = Path(__file__).resolve().parents[1]
    sp_row = sum_df[sum_df["method"] == win_sp].iloc[0]
    mp_row = sum_df[sum_df["method"] == win_mp].iloc[0]
    receipt = {
        "schema": "GeoCalib_3D_test_receipt",
        "complete": True,
        "one_shot_test_evaluation": True,
        "git_head_at_eval": _git_head(repo_root),
        "frozen_model_path": str(frozen_model_path),
        "frozen_model_sha256": frozen_sha,
        "test_images": int(len(test_ids)),
        "test_scenes": int(per_image_df["scene"].nunique()),
        "track_a_winner": win_sp,
        "track_a_test_metrics": {k: float(sp_row[k]) for k in ("abs_rel", "rmse", "delta1", "gain_vs_raw_abs_rel", "gain_vs_spline_abs_rel", "gain_vs_geo_abs_rel", "gain_vs_geo_rmse", "gain_vs_geo_delta1")},
        "track_b_winner": win_mp,
        "track_b_test_metrics": {k: float(mp_row[k]) for k in ("abs_rel", "rmse", "delta1", "gain_vs_raw_abs_rel", "gain_vs_spline_abs_rel", "gain_vs_geo_abs_rel", "gain_vs_geo_rmse", "gain_vs_geo_delta1")},
        "runtime_summary": runtime,
    }
    write_json_lf(output_dir / "test_evaluation_receipt.json", receipt)
    print(
        f"Test Eval Complete:\n"
        f"  Track A ({win_sp}): AbsRel={sp_row['abs_rel']:.6f}, RMSE={sp_row['rmse']:.6f}, delta1={sp_row['delta1']:.6f}\n"
        f"  Track B ({win_mp}): AbsRel={mp_row['abs_rel']:.6f}, RMSE={mp_row['rmse']:.6f}, delta1={mp_row['delta1']:.6f}",
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--orig-preds", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--flip-dir", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--oof-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate_heldout_test(
        data_root=args.data_root,
        orig_preds=args.orig_preds,
        aux_dir=args.aux_dir,
        flip_dir=args.flip_dir,
        frozen_model_path=args.frozen_model,
        oof_summary_path=args.oof_summary,
        output_dir=args.output,
        expected_model_sha256=args.expected_model_sha256,
        force=args.force,
    )


if __name__ == "__main__":
    main()
