"""Gate 4 one-shot evaluation and statistical audit for GeoCalib-3D on held-out test split."""
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

from .calibration import BOOTSTRAP_SEED
from .data import nyu_depth, open_nyu, scene_names
from .frozen import apply_frozen, frozen_predictions, load_frozen
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, evaluate_image, metrics, valid_depth
from .sota_calibration import (
    _FULL_D2_CENTER_T,
    _FULL_R2_T,
    _FULL_U_PIX_T,
    _FULL_V_PIX_T,
    _FULL_V_TILDE_T,
    LOCKED_LADDER_NAMES,
    PRIMARY_SOTA_METHOD,
    apply_sota_model,
    count_parameters,
    extract_sota_features,
    load_frozen_sota,
)
from .sota_io import file_sha256, load_nyu_rgb_cache, write_csv_lf, write_json_lf


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
    reference: str = "b3_residual_v1",
    candidate: str = PRIMARY_SOTA_METHOD,
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

    order = np.argsort(comp_score)  # ascending: worst regressions first, largest gains last
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
    """Compare 5-fold OOF metrics and gains vs B3 against held-out test metrics and gains."""
    oof_t = oof_summary_df.set_index("method")
    test_t = test_summary_df.set_index("method")
    oof_b3 = oof_t.loc["b3_residual_v1", list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    test_b3 = test_t.loc["b3_residual_v1", list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    direction = np.array([-1.0, -1.0, 1.0], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for method in test_summary_df["method"].tolist():
        if method not in oof_t.index:
            continue
        o_v = oof_t.loc[method, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        t_v = test_t.loc[method, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        o_g = (o_v - oof_b3) * direction
        t_g = (t_v - test_b3) * direction
        rows.append(
            {
                "method": method,
                "oof_abs_rel": float(o_v[0]),
                "test_abs_rel": float(t_v[0]),
                "oof_gain_vs_b3_abs_rel": float(o_g[0]),
                "test_gain_vs_b3_abs_rel": float(t_g[0]),
                "test_minus_oof_gain_abs_rel": float(t_g[0] - o_g[0]),
                "oof_rmse": float(o_v[1]),
                "test_rmse": float(t_v[1]),
                "oof_gain_vs_b3_rmse": float(o_g[1]),
                "test_gain_vs_b3_rmse": float(t_g[1]),
                "test_minus_oof_gain_rmse": float(t_g[1] - o_g[1]),
                "oof_delta1": float(o_v[2]),
                "test_delta1": float(t_v[2]),
                "oof_gain_vs_b3_delta1": float(o_g[2]),
                "test_gain_vs_b3_delta1": float(t_g[2]),
                "test_minus_oof_gain_delta1": float(t_g[2] - o_g[2]),
            }
        )
    return pd.DataFrame(rows)


def _apply_geocalib_gpu_exact(
    p_np: np.ndarray,
    k_np: np.ndarray,
    model: dict[str, Any],
    dev: torch.device,
) -> torch.Tensor:
    """Apply GeoCalib-3D on CUDA float64 tensors cleanly matching pure NumPy semantics."""
    feats = extract_sota_features(p_np, k_np, rgb_image=None, need_rgb=False)
    fs = model["focal_spline"]
    shift = 0.0
    if fs["enabled"]:
        shift += float(np.interp(feats["f_tilde"], fs["x_knots"], fs["y_knots"]))
    sr = model["scale_residual"]
    if sr["weights"]:
        x_s = np.array([feats[k] for k in sr["feature_names"]], dtype=np.float64)
        z_s = (x_s - np.asarray(sr["mean"], dtype=np.float64)) / np.asarray(sr["std"], dtype=np.float64)
        m_lim = float(sr["max_log_scale"])
        shift += m_lim * float(np.tanh(float(z_s @ np.asarray(sr["weights"], dtype=np.float64)) / m_lim))
    if fs["enabled"] or sr["weights"]:
        shift = 0.50 * float(np.tanh(shift / 0.50))

    dsm = model["depth_slope_mod"]
    delta_b = 0.0
    if dsm["weights"]:
        x_b = np.array([feats[k] for k in dsm["feature_names"]], dtype=np.float64)
        z_b = (x_b - np.asarray(dsm["mean"], dtype=np.float64)) / np.asarray(dsm["std"], dtype=np.float64)
        delta_b = 0.20 * float(np.tanh(float(z_b @ np.asarray(dsm["weights"], dtype=np.float64))))

    ds = model["depth_spline"]
    xk = np.asarray(ds["x_knots"], dtype=np.float64)
    yk = np.asarray(ds["y_knots"], dtype=np.float64)
    slopes = np.diff(yk) / np.diff(xk)
    dx = np.diff(xk)

    p_g = torch.from_numpy(np.ascontiguousarray(p_np)).to(dev)
    lp_g = torch.log(p_g)
    out_g = float(yk[0]) + float(slopes[0]) * torch.clamp(lp_g - float(xk[0]), max=float(dx[0]))
    for k_i in range(1, 4):
        out_g.add_(float(slopes[k_i]) * torch.clamp(lp_g - float(xk[k_i]), min=0.0, max=float(dx[k_i])))
    out_g.add_(float(slopes[4]) * torch.clamp(lp_g - float(xk[4]), min=0.0))
    if shift != 0.0:
        out_g.add_(shift)
    if delta_b != 0.0:
        out_g.add_(lp_g - float(dsm["log_depth_center"]), alpha=delta_b)
    return torch.exp(out_g.clamp_(float(np.log(0.001)), float(np.log(10.0)))).clamp_(0.001, 10.0)


def benchmark_test_runtime(
    cache,
    test_ids: np.ndarray,
    primary_model: dict[str, Any],
    b3_frozen: dict[str, Any],
    n_images: int = 200,
) -> dict[str, Any]:
    """Measure pure post-processing runtime and verify torch-vs-numpy parity on held-out test."""
    bench_ids = test_ids[: min(n_images, len(test_ids))]
    depths = [cache[int(i)].astype(np.float64) for i in bench_ids]
    ks = [cache.intrinsics(int(i)).astype(np.float64) for i in bench_ids]

    # Verify numerical parity across first 20 test images
    max_diff_cpu = 0.0
    for d, k in zip(depths[:20], ks[:20]):
        out_np = apply_sota_model(d, k, primary_model, use_torch_fast=False)
        out_th = apply_sota_model(d, k, primary_model, use_torch_fast=True)
        max_diff_cpu = max(max_diff_cpu, float(np.max(np.abs(out_np - out_th))))

    # Warmup
    for d, k in zip(depths[:5], ks[:5]):
        _ = extract_sota_features(d, k, None, need_rgb=False)
        _ = apply_sota_model(d, k, primary_model, use_torch_fast=True)
        _ = apply_sota_model(d, k, primary_model, use_torch_fast=False)
        _ = apply_frozen(d, b3_frozen)

    tf0 = perf_counter()
    for d, k in zip(depths, ks):
        _ = extract_sota_features(d, k, None, need_rgb=False)
    feat_ms = (perf_counter() - tf0) * 1000.0 / len(depths)

    t_cpu0 = perf_counter()
    for d, k in zip(depths, ks):
        _ = apply_sota_model(d, k, primary_model, use_torch_fast=True)
    cpu_torch_ms = (perf_counter() - t_cpu0) * 1000.0 / len(depths)

    t_np0 = perf_counter()
    for d, k in zip(depths, ks):
        _ = apply_sota_model(d, k, primary_model, use_torch_fast=False)
    numpy_ms = (perf_counter() - t_np0) * 1000.0 / len(depths)

    t_b3 = perf_counter()
    for d in depths:
        _ = apply_frozen(d, b3_frozen)
    b3_ms = (perf_counter() - t_b3) * 1000.0 / len(depths)

    result: dict[str, Any] = {
        "benchmark_images": len(depths),
        "extract_sota_features_ms_per_image": float(feat_ms),
        "geocalib3d_cpu_torch_f64_ms_per_image": float(cpu_torch_ms),
        "geocalib3d_pure_numpy_f64_ms_per_image": float(numpy_ms),
        "b3_residual_v1_ms_per_image": float(b3_ms),
        "max_abs_diff_cpu_torch_vs_numpy": float(max_diff_cpu),
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        # Reference GPU parity check & warmup
        max_diff_gpu = 0.0
        for d, k in zip(depths[:10], ks[:10]):
            out_g = _apply_geocalib_gpu_exact(d, k, primary_model, dev).cpu().numpy()
            out_n = apply_sota_model(d, k, primary_model, use_torch_fast=False)
            max_diff_gpu = max(max_diff_gpu, float(np.max(np.abs(out_g - out_n))))
        torch.cuda.synchronize()
        t_gpu0 = perf_counter()
        for d, k in zip(depths, ks):
            _ = _apply_geocalib_gpu_exact(d, k, primary_model, dev)
        torch.cuda.synchronize()
        gpu_ms = (perf_counter() - t_gpu0) * 1000.0 / len(depths)
        result["geocalib3d_gpu_torch_f64_ms_per_image"] = float(gpu_ms)
        result["max_abs_diff_gpu_torch_vs_numpy"] = float(max_diff_gpu)
        result["gpu_name"] = str(torch.cuda.get_device_name(0))
    return result


def _compare_b3_test_reference(per_image_df: pd.DataFrame, reference_path: Path) -> dict[str, Any]:
    ref_raw = pd.read_csv(reference_path)
    aliases = {
        "raw": "raw",
        "median": "b1_global_scale",
        "spline_rmse": "b2_monotone_curve",
        "residual_median_ratio_l1_a0.5": "b3_residual_v1",
    }
    expected = (
        ref_raw[ref_raw["method"].isin(aliases.keys())]
        .rename(columns={"dataset_index": "index", "d1": "delta1"})
        .copy()
    )
    expected["method"] = expected["method"].replace(aliases)
    keys = ["method", "index"]
    expected = expected.set_index(keys).sort_index()
    observed = per_image_df[per_image_df["method"].isin(aliases.values())].set_index(keys).sort_index()
    if not expected.index.equals(observed.index):
        raise ValueError("Reference method/image IDs do not match Gate 4 baseline evaluation")
    err = np.abs(
        expected[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        - observed[list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(err).all():
        raise ValueError("Nonfinite error in frozen B3 reference check")
    per_image_max_err = float(np.max(err))
    exp_means = expected.groupby(level=0)[list(METRIC_NAMES)].mean()
    obs_means = observed.groupby(level=0)[list(METRIC_NAMES)].mean()
    delta_means = obs_means - exp_means
    max_mean_err = float(np.max(np.abs(delta_means.to_numpy(dtype=np.float64))))
    if max_mean_err > 5e-4:
        raise ValueError(f"Frozen B3 reference aggregate mismatch: max_mean_abs_error={max_mean_err}")
    return {
        "reference_path": str(reference_path),
        "reference_sha256": file_sha256(reference_path),
        "verified_rows": int(len(observed)),
        "cache_note": (
            "Local outputs/nyu_rgb/depth.npy (e84ee7d5...) vs historical replay cache "
            "(2d2e63d2...); all paired comparisons use identical local depth.npy"
        ),
        "max_mean_abs_error": max_mean_err,
        "per_image_max_abs_error": per_image_max_err,
        "mean_tolerance": 5e-4,
        "mean_deltas_by_method": {
            str(m): {col: float(delta_means.loc[m, col]) for col in METRIC_NAMES}
            for m in delta_means.index
        },
    }


def _git_head(repo_root: Path) -> str | None:
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
        return None


def evaluate_heldout_test(
    data_root: Path,
    prediction_dir: Path,
    frozen_sota_path: Path,
    b3_frozen_path: Path,
    cv_dir: Path,
    b3_test_reference_path: Path,
    output_dir: Path,
    expected_sota_sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run strictly one-shot evaluation of frozen GeoCalib-3D + baselines on 654 test images."""
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    summary_file = output_dir / "test_summary.csv"
    if summary_file.exists() and not force:
        raise FileExistsError(
            f"Refusing to re-run one-shot test evaluation because {summary_file} already exists (pass --force only if intentional)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    sota = load_frozen_sota(frozen_sota_path, expected_sha256=expected_sota_sha256)
    sota_sha = file_sha256(frozen_sota_path)
    b3 = load_frozen(b3_frozen_path, expected_sha256=sota["b3_frozen_model_sha256"])

    cache = load_nyu_rgb_cache(
        prediction_dir, data_root, allowed_split="test", compute_hashes=True, verify_scenes=True
    )
    if sota["dataset_sha256"] != cache.hashes["nyu_mat_sha256"]:
        raise ValueError("dataset_sha256 mismatch between frozen model and NYU data")
    if sota["split_sha256"] != cache.hashes["splits_sha256"]:
        raise ValueError("split_sha256 mismatch between frozen model and splits.mat")
    for fname, cache_key in (
        ("depth.npy", "depth_npy_sha256"),
        ("intrinsics.npy", "intrinsics_npy_sha256"),
        ("indices.npy", "indices_npy_sha256"),
    ):
        if sota["prediction_cache_sha256"][fname] != cache.hashes[cache_key]:
            raise ValueError(f"Prediction cache hash mismatch for {fname}")

    test_ids = cache.test_ids
    methods = tuple(sota["comparison_order"])
    ladder_models = sota["ladder_models"]

    # Precompute f_tilde quartiles on held-out test predicted intrinsics for spatial bin breakdown
    test_ks = cache.intrinsics_for(test_ids).astype(np.float64)
    all_f_tilde = 0.5 * (test_ks[:, 0, 0] + test_ks[:, 1, 1]) / 480.0
    f_tilde_q25 = float(np.quantile(all_f_tilde, 0.25))
    f_tilde_q75 = float(np.quantile(all_f_tilde, 0.75))

    # Precompute 12x16 crop spatial grid reduceat boundaries
    row_bins = np.array_split(np.arange(426), 12)
    col_bins = np.array_split(np.arange(560), 16)
    row_starts = np.array([int(b[0]) for b in row_bins], dtype=np.int64)
    col_starts = np.array([int(b[0]) for b in col_bins], dtype=np.int64)
    row_edges = [45 + int(b[0]) for b in row_bins] + [471]
    col_edges = [41 + int(b[0]) for b in col_bins] + [601]

    spatial_methods = ("raw", "b3_residual_v1", PRIMARY_SOTA_METHOD)
    spatial_subsets = ("all", "low_f_tilde", "high_f_tilde")
    sp_log_sum = {sub: {m: np.zeros((12, 16), dtype=np.float64) for m in spatial_methods} for sub in spatial_subsets}
    sp_ar_sum = {sub: {m: np.zeros((12, 16), dtype=np.float64) for m in spatial_methods} for sub in spatial_subsets}
    sp_cnt_sum = {sub: np.zeros((12, 16), dtype=np.float64) for sub in spatial_subsets}

    def _reduce_12x16(arr2d: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(arr2d, row_starts, axis=0), col_starts, axis=1)

    records: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    with open_nyu(data_root) as handle:
        scenes = scene_names(handle)
        for pos, idx in enumerate(test_ids):
            idx_int = int(idx)
            sc = str(scenes[idx_int])
            raw_p = cache[idx_int].astype(np.float64)
            K = test_ks[pos]
            gt_full = nyu_depth(handle, idx_int)
            gt_crop = gt_full[CROP]
            valid = valid_depth(gt_crop)

            f_outs = frozen_predictions(raw_p, b3)
            feats = extract_sota_features(raw_p, K, rgb_image=None, need_rgb=False)
            preds: dict[str, np.ndarray] = {
                "raw": f_outs["raw"],
                "b1_global_scale": f_outs["global_scale"],
                "b2_monotone_curve": f_outs["monotone_curve"],
                "b3_residual_v1": f_outs["residual_scale"],
            }
            for lname in LOCKED_LADDER_NAMES:
                preds[lname] = apply_sota_model(
                    raw_p, K, ladder_models[lname], cached_features=feats, use_torch_fast=True
                )

            for m_name in methods:
                d_pred = preds[m_name]
                ar, rmse, d1 = evaluate_image(d_pred, gt_full)
                records.append(
                    {
                        "index": idx_int,
                        "scene": sc,
                        "method": m_name,
                        "abs_rel": float(ar),
                        "rmse": float(rmse),
                        "delta1": float(d1),
                    }
                )
                d_crop = d_pred[CROP]
                for lower, upper in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                    mask = valid & (gt_crop >= lower) & (gt_crop < upper)
                    if mask.any():
                        b_ar, b_rmse, b_d1 = metrics(d_crop, gt_crop, mask)
                        binned.append(
                            {
                                "index": idx_int,
                                "scene": sc,
                                "method": m_name,
                                "lower_m": float(lower),
                                "upper_m": float(upper),
                                "pixels": int(mask.sum()),
                                "abs_rel": float(b_ar),
                                "rmse": float(b_rmse),
                                "delta1": float(b_d1),
                            }
                        )

            diag_rows.append(
                {
                    "index": idx_int,
                    "scene": sc,
                    "f_tilde": float(feats["f_tilde"]),
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx_off": float(feats["cx_off"]),
                    "cy_off": float(feats["cy_off"]),
                    "median_log_err_raw": float(np.median(np.log(preds["raw"][CROP][valid] / gt_crop[valid]))),
                    "median_log_err_b3": float(
                        np.median(np.log(preds["b3_residual_v1"][CROP][valid] / gt_crop[valid]))
                    ),
                    "median_log_err_geocalib3d": float(
                        np.median(np.log(preds[PRIMARY_SOTA_METHOD][CROP][valid] / gt_crop[valid]))
                    ),
                }
            )

            # Accumulate spatial residual maps
            active_subs = ["all"]
            if feats["f_tilde"] <= f_tilde_q25:
                active_subs.append("low_f_tilde")
            elif feats["f_tilde"] >= f_tilde_q75:
                active_subs.append("high_f_tilde")
            valid_f = valid.astype(np.float64)
            cnt_grid = _reduce_12x16(valid_f)
            for sub in active_subs:
                sp_cnt_sum[sub] += cnt_grid
            gt_safe = np.where(valid, gt_crop, 1.0)
            for m_name in spatial_methods:
                pc = preds[m_name][CROP]
                log_r = np.where(valid, np.log(pc / gt_safe), 0.0)
                ar_r = np.where(valid, np.abs(pc - gt_safe) / gt_safe, 0.0)
                lg_grid = _reduce_12x16(log_r)
                ar_grid = _reduce_12x16(ar_r)
                for sub in active_subs:
                    sp_log_sum[sub][m_name] += lg_grid
                    sp_ar_sum[sub][m_name] += ar_grid

            if (pos + 1) % 150 == 0 or pos + 1 == len(test_ids):
                print(f"Gate 4 test eval {pos + 1}/{len(test_ids)}", flush=True)

    per_image_df = pd.DataFrame(records)
    diag_df = pd.DataFrame(diag_rows)
    bins_raw_df = pd.DataFrame(binned)

    # Strict machine-precision baseline check against historical frozen evaluation
    b3_ref_check = _compare_b3_test_reference(per_image_df, b3_test_reference_path)

    # Summary table matching comparison_order
    method_params = {
        "raw": 0,
        "b1_global_scale": 1,
        "b2_monotone_curve": 6,
        "b3_residual_v1": 15,
        **{name: count_parameters(ladder_models[name]) for name in LOCKED_LADDER_NAMES},
    }
    sum_df = (
        per_image_df.groupby("method", sort=False)[list(METRIC_NAMES)]
        .mean()
        .reindex(methods)
        .reset_index()
    )
    b3_row = sum_df.loc[sum_df["method"] == "b3_residual_v1", list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]
    sum_df["params"] = [method_params[m] for m in sum_df["method"]]
    sum_df["gain_vs_b3_abs_rel"] = b3_row[0] - sum_df["abs_rel"]
    sum_df["gain_vs_b3_rmse"] = b3_row[1] - sum_df["rmse"]
    sum_df["gain_vs_b3_delta1"] = sum_df["delta1"] - b3_row[2]
    sum_df = sum_df[
        ["method", "params", "abs_rel", "rmse", "delta1", "gain_vs_b3_abs_rel", "gain_vs_b3_rmse", "gain_vs_b3_delta1"]
    ]

    # Paired pairs list (consecutives + primary vs baselines + B3 vs every ladder stage)
    pairs: list[tuple[str, str]] = list(zip(methods, methods[1:]))
    for ref_m in ("raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"):
        if (ref_m, PRIMARY_SOTA_METHOD) not in pairs:
            pairs.append((ref_m, PRIMARY_SOTA_METHOD))
    for cand_m in LOCKED_LADDER_NAMES:
        if ("b3_residual_v1", cand_m) not in pairs:
            pairs.append(("b3_residual_v1", cand_m))

    boot_df = paired_scene_bootstrap(per_image_df, pairs)
    change_df = compute_change_counts(per_image_df, pairs)
    cases_df = select_representative_cases(per_image_df, diag_df, reference="b3_residual_v1", candidate=PRIMARY_SOTA_METHOD)

    bins_df = (
        bins_raw_df.groupby(["method", "lower_m", "upper_m"], sort=False)
        .agg(
            abs_rel=("abs_rel", "mean"),
            rmse=("rmse", "mean"),
            delta1=("delta1", "mean"),
            images=("index", "count"),
            pixels=("pixels", "sum"),
        )
        .reset_index()
    )
    method_rank = {m: i for i, m in enumerate(methods)}
    bins_df["_m_rank"] = bins_df["method"].map(method_rank)
    bins_df = bins_df.sort_values(["_m_rank", "lower_m"]).drop(columns=["_m_rank"]).reset_index(drop=True)

    oof_df = pd.read_csv(Path(cv_dir) / "oof_ablation_summary.csv")
    gap_df = build_generalization_gap(oof_df, sum_df)

    # Spatial residual summary JSON
    spatial_payload: dict[str, Any] = {
        "row_edges": row_edges,
        "col_edges": col_edges,
        "f_tilde_q25": f_tilde_q25,
        "f_tilde_q75": f_tilde_q75,
        "maps": {},
    }
    for sub in spatial_subsets:
        cnt = np.maximum(sp_cnt_sum[sub], 1.0)
        spatial_payload["maps"][sub] = {}
        for m_name in spatial_methods:
            log_map = sp_log_sum[sub][m_name] / cnt
            ar_map = sp_ar_sum[sub][m_name] / cnt
            spatial_payload["maps"][sub][m_name] = {
                "mean_log_residual_12x16": log_map.tolist(),
                "mean_abs_rel_12x16": ar_map.tolist(),
                "rms_spatial_log_bias": float(np.sqrt(np.mean(log_map ** 2))),
                "max_abs_spatial_log_bias": float(np.max(np.abs(log_map))),
            }

    runtime = benchmark_test_runtime(cache, test_ids, sota["primary_model"], b3, n_images=200)

    # Write all Gate 4 artifacts with explicit LF
    write_csv_lf(output_dir / "test_summary.csv", sum_df)
    write_csv_lf(output_dir / "test_per_image.csv", per_image_df)
    write_csv_lf(output_dir / "test_paired_bootstrap.csv", boot_df)
    write_csv_lf(output_dir / "test_depth_bins.csv", bins_df)
    write_csv_lf(output_dir / "test_change_counts.csv", change_df)
    write_csv_lf(output_dir / "test_representative_cases.csv", cases_df)
    write_csv_lf(output_dir / "generalization_gap.csv", gap_df)
    write_csv_lf(output_dir / "test_mechanism_diagnostics.csv", diag_df)
    write_json_lf(output_dir / "test_spatial_residuals.json", spatial_payload)
    write_json_lf(output_dir / "runtime_overhead.json", runtime)

    repo_root = Path(__file__).resolve().parents[1]
    receipt = {
        "schema": "GeoCalib_3D_test_receipt_v1",
        "complete": True,
        "one_shot_test_evaluation": True,
        "git_head_at_eval": _git_head(repo_root),
        "frozen_sota_path": str(frozen_sota_path),
        "frozen_sota_sha256": sota_sha,
        "b3_frozen_sha256": sota["b3_frozen_model_sha256"],
        "test_images": int(len(test_ids)),
        "test_scenes": int(per_image_df["scene"].nunique()),
        "b3_reference_check": b3_ref_check,
        "b3_reference_max_mean_error": float(b3_ref_check["max_mean_abs_error"]),
        "b3_reference_per_image_max_error": float(b3_ref_check["per_image_max_abs_error"]),
        "primary_method": PRIMARY_SOTA_METHOD,
        "primary_test_metrics": {
            k: float(sum_df.loc[sum_df["method"] == PRIMARY_SOTA_METHOD, k].iloc[0])
            for k in ("abs_rel", "rmse", "delta1", "gain_vs_b3_abs_rel", "gain_vs_b3_rmse", "gain_vs_b3_delta1")
        },
        "runtime_summary": runtime,
    }
    write_json_lf(output_dir / "test_evaluation_receipt.json", receipt)
    print(
        f"Gate 4 Test Complete: GeoCalib-3D AbsRel={receipt['primary_test_metrics']['abs_rel']:.6f} "
        f"(+{receipt['primary_test_metrics']['gain_vs_b3_abs_rel']:.6f} vs B3), "
        f"RMSE={receipt['primary_test_metrics']['rmse']:.6f}, "
        f"delta1={receipt['primary_test_metrics']['delta1']:.6f}, "
        f"B3 ref mean err={b3_ref_check['max_mean_abs_error']:.2e}",
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frozen-sota", type=Path, required=True)
    parser.add_argument("--b3-frozen", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--b3-test-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sota-sha256", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate_heldout_test(
        data_root=args.data_root,
        prediction_dir=args.predictions,
        frozen_sota_path=args.frozen_sota,
        b3_frozen_path=args.b3_frozen,
        cv_dir=args.cv_dir,
        b3_test_reference_path=args.b3_test_reference,
        output_dir=args.output,
        expected_sota_sha256=args.expected_sota_sha256,
        force=args.force,
    )


if __name__ == "__main__":
    main()
