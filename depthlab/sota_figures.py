"""CVPR publication-quality 300-DPI figures and verified manifest builder for GeoCalib-3D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .data import nyu_depth, nyu_image, open_nyu
from .frozen import frozen_predictions, load_frozen
from .metrics import CROP, valid_depth
from .sota_calibration import PRIMARY_SOTA_METHOD, apply_sota_model, load_frozen_sota
from .sota_io import file_sha256, load_nyu_rgb_cache, write_json_lf


DISPLAY_SHORT = {
    "raw": "Raw",
    "b1_global_scale": "B1",
    "b2_monotone_curve": "B2",
    "b3_residual_v1": "B3_prev",
    "S2_b2_geok_sp3d_ridge": "S2 Ridge",
    "J0_spline_bal111": "J0 Joint",
    "J1_geok_bal111_l01": "J1 +GeoK",
    "J2_geok_sp3d_bal111_l01": "J2 +Sp3D",
    "J3_geok_sp3d_slope_bal111_l01": "GeoCalib-3D",
}

PALETTE = {
    "raw": "#6C757D",
    "b1_global_scale": "#7B8CDE",
    "b2_monotone_curve": "#4E79A7",
    "b3_residual_v1": "#D95F02",
    "S2_b2_geok_sp3d_ridge": "#7570B3",
    "J0_spline_bal111": "#A6761D",
    "J1_geok_bal111_l01": "#66A61E",
    "J2_geok_sp3d_bal111_l01": "#1B9E77",
    "J3_geok_sp3d_slope_bal111_l01": "#0072B2",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    lw: float = 1.1,
    fontsize: float = 8.4,
    text_color: str = "#1A1A1A",
    bold: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.5 * w,
        y + 0.5 * h,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def _draw_arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = "#444444") -> None:
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=color,
        linewidth=1.15,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)


def plot_fig1_pipeline_overview(
    out_path: Path,
    frozen_sota: dict[str, Any],
    b3_frozen: dict[str, Any],
    diag_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    runtime: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(14.2, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.20)

    # Left: Pipeline Architecture Schematic
    ax_arch = fig.add_subplot(gs[0, 0])
    ax_arch.set_xlim(0.0, 13.0)
    ax_arch.set_ylim(0.0, 6.6)
    ax_arch.axis("off")
    ax_arch.set_title("(a) GeoCalib-3D Single-Pass Architecture (25 Learned Params, Zero GT K)", pad=8)

    _draw_box(ax_arch, 0.15, 2.65, 1.55, 1.35, "RGB Image\n(480x640)\nSingle Pass", "#F1F3F5", "#495057", fontsize=7.2, bold=True)
    _draw_box(ax_arch, 2.00, 2.55, 2.25, 1.55, "Frozen UniDepthV2\nViT-S Backbone\n-> Z_hat, K_hat\n(fx, fy, cx, cy)", "#E9ECEF", "#343A40", fontsize=7.2)
    _draw_arrow(ax_arch, (1.72, 3.32), (1.98, 3.32))

    # Three geometry / depth branches
    _draw_box(
        ax_arch,
        4.60,
        4.85,
        4.15,
        1.28,
        "1. Focal-Scale Spline h_f(f_tilde)\n5 free knots (anchor 1.14 -> 0)\nMonotone non-increasing",
        "#E7F1FA",
        "#0072B2",
        lw=1.3,
        fontsize=7.1,
    )
    _draw_box(
        ax_arch,
        4.60,
        2.68,
        4.15,
        1.62,
        "2. 4x4 Subcrop Geometry (14,980 px)\n2 Intrinsics + 8 Spatial-3D cues\n-> Scale Residual Delta s_geo\n(10 weights, tanh-bounded)",
        "#E6F5F0",
        "#1B9E77",
        lw=1.3,
        fontsize=7.1,
    )
    _draw_box(
        ax_arch,
        4.60,
        0.78,
        4.15,
        1.48,
        "3. Depth Spline & Slope Mod.\n6-knot monotone g(log Z_hat)\n+ Delta beta(f_tilde, sp3d)\n(4 slope weights, slope >= 0.05)",
        "#FFF3E0",
        "#D95F02",
        lw=1.3,
        fontsize=7.1,
    )

    _draw_arrow(ax_arch, (4.27, 3.65), (4.58, 5.48), "#0072B2")
    _draw_arrow(ax_arch, (4.27, 3.32), (4.58, 3.48), "#1B9E77")
    _draw_arrow(ax_arch, (4.27, 3.00), (4.58, 1.52), "#D95F02")

    _draw_box(
        ax_arch,
        9.15,
        2.32,
        3.65,
        2.05,
        "Joint Metric Head\nlog Z_cal = g(log Z_hat)\n+ clamp(h_f + Delta s_geo)\n+ Delta beta * (log Z_hat - 1)\n-> exp & clip[1e-3, 10] m",
        "#D0E6FA",
        "#005A8D",
        lw=1.6,
        fontsize=7.1,
        bold=True,
    )
    _draw_arrow(ax_arch, (8.77, 5.48), (9.13, 3.80), "#0072B2")
    _draw_arrow(ax_arch, (8.77, 3.48), (9.13, 3.37), "#1B9E77")
    _draw_arrow(ax_arch, (8.77, 1.52), (9.13, 2.90), "#D95F02")

    b3_row = test_sum_df.loc[test_sum_df["method"] == "b3_residual_v1"].iloc[0]
    geo_row = test_sum_df.loc[test_sum_df["method"] == PRIMARY_SOTA_METHOD].iloc[0]
    feat_ms = runtime["extract_sota_features_ms_per_image"]
    cpu_ms = runtime["geocalib3d_cpu_torch_f64_ms_per_image"]
    gpu_ms = runtime.get("geocalib3d_gpu_torch_f64_ms_per_image")
    b3_ms = runtime["b3_residual_v1_ms_per_image"]
    lat_txt = (
        f"Features: {feat_ms:.2f} ms | GeoCalib CPU f64: {cpu_ms:.2f} ms"
        + (f" | GPU f64: {gpu_ms:.2f} ms" if gpu_ms is not None else "")
        + f" (vs B3_prev: {b3_ms:.2f} ms)\nHeld-out Test (N=654): AbsRel {b3_row['abs_rel']:.4f} -> {geo_row['abs_rel']:.4f}, "
        f"RMSE {b3_row['rmse']:.4f} -> {geo_row['rmse']:.4f}m, delta1 {b3_row['delta1']:.4f} -> {geo_row['delta1']:.4f}"
    )
    _draw_box(ax_arch, 0.15, 0.08, 12.65, 0.52, lat_txt, "#F8F9FA", "#ADB5BD", fontsize=7.2)

    # Right column: Learned Focal Spline & Depth Spline
    gs_r = gs[0, 1].subgridspec(2, 1, hspace=0.45)

    # (b) Focal Spline
    ax_f = fig.add_subplot(gs_r[0, 0])
    fs = frozen_sota["primary_model"]["focal_spline"]
    fxk = np.asarray(fs["x_knots"], dtype=np.float64)
    fyk = np.asarray(fs["y_knots"], dtype=np.float64)
    f_grid = np.linspace(0.32, 1.38, 400)
    y_grid = np.interp(f_grid, fxk, fyk)

    ax_f_hist = ax_f.twinx()
    ax_f_hist.hist(
        diag_df["f_tilde"].to_numpy(),
        bins=28,
        range=(0.35, 1.35),
        color="#ADB5BD",
        alpha=0.28,
        edgecolor="none",
        density=True,
    )
    ax_f_hist.set_ylabel("Density", color="#6C757D", fontsize=8.0)
    ax_f_hist.tick_params(axis="y", colors="#6C757D", labelsize=7.5)
    ax_f_hist.spines["top"].set_visible(False)
    ax_f.set_zorder(ax_f_hist.get_zorder() + 1)
    ax_f.patch.set_visible(False)

    ax_f.axhline(0.0, color="#6C757D", ls="--", lw=0.9)
    ax_f.plot(f_grid, y_grid, color="#0072B2", lw=2.2, label="Learned h_f(f_tilde)")
    ax_f.scatter(fxk, fyk, color="#0072B2", s=34, zorder=5, edgecolor="white", linewidth=0.7)
    ax_f.scatter([1.14], [0.0], color="#D95F02", s=48, zorder=6, marker="D", label="Anchor (1.14, 0.0)")
    ax_f.annotate(
        f"+{fyk[0]:.3f}\n(+{np.exp(fyk[0]) - 1:.1%} scale)",
        xy=(fxk[0], fyk[0]),
        xytext=(fxk[0] + 0.05, fyk[0] - 0.14),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.8},
        fontsize=7.8,
    )
    ax_f.annotate(
        f"{fyk[-1]:.3f}\n({np.exp(fyk[-1]) - 1:+.1%})",
        xy=(fxk[-1], fyk[-1]),
        xytext=(fxk[-1] - 0.30, fyk[-1] - 0.01),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.8},
        fontsize=7.8,
    )
    ax_f.set_xlim(0.32, 1.40)
    ax_f.set_ylim(-0.24, 0.34)
    ax_f.set_xlabel("Predicted Normalized Focal Length  f_tilde = 0.5*(fx + fy) / 480")
    ax_f.set_ylabel("Log Scale Shift h_f")
    ax_f.set_title("(b) Learned Focal-Scale Spline (Anchored at Train Median f_tilde=1.14)")
    ax_f.grid(True, linestyle=":", alpha=0.45)
    ax_f.legend(loc="upper right", frameon=True, framealpha=0.9)

    # (c) Depth Spline
    ax_d = fig.add_subplot(gs_r[1, 0])
    ds = frozen_sota["primary_model"]["depth_spline"]
    dxk = np.asarray(ds["x_knots"], dtype=np.float64)
    dyk = np.asarray(ds["y_knots"], dtype=np.float64)
    b3_ds = b3_frozen["base_model"]
    bxk = np.asarray(b3_ds["x_knots"], dtype=np.float64)
    byk = np.asarray(b3_ds["y_knots"], dtype=np.float64)

    lp_grid = np.linspace(dxk[0], dxk[-1], 300)
    geo_y = np.interp(lp_grid, dxk, dyk)
    b3_y = np.interp(lp_grid, bxk, byk)
    ax_d.axhline(0.0, color="#6C757D", ls=":", lw=1.1, label="Identity (0 offset)")
    ax_d.plot(lp_grid, b3_y - lp_grid, color="#D95F02", ls="--", lw=1.7, label="B3 Base Offset (g_B3 - log Z)")
    ax_d.plot(lp_grid, geo_y - lp_grid, color="#0072B2", lw=2.2, label="GeoCalib-3D Offset (g_ours - log Z)")
    ax_d.scatter(dxk, dyk - dxk, color="#0072B2", s=28, zorder=5)
    ax_d.set_ylim(-0.46, 0.14)
    ax_d.set_xticks(dxk)
    ax_d.set_xticklabels([f"{np.exp(v):.1f}m" for v in dxk])
    ax_d.set_xlabel("Raw Predicted Depth Knot Position (m, log-spaced)")
    ax_d.set_ylabel("Depth Spline Log Offset")
    ax_d.set_title("(c) Learned Monotone Depth Spline Offset vs Raw Depth")
    ax_d.grid(True, linestyle=":", alpha=0.45)
    ax_d.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig2_ablation_and_ci(
    out_path: Path,
    oof_sum_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    oof_boot_df: pd.DataFrame,
    test_boot_df: pd.DataFrame,
) -> None:
    order = test_sum_df["method"].tolist()
    oof_t = oof_sum_df.set_index("method").reindex(order)
    test_t = test_sum_df.set_index("method").reindex(order)
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.6), gridspec_kw={"hspace": 0.56, "wspace": 0.28})

    metric_specs = [
        ("abs_rel", "AbsRel (lower better)", "(a) Ablation Progression: AbsRel", False),
        ("rmse", "RMSE (m, lower better)", "(b) Ablation Progression: RMSE", False),
        ("delta1", "delta1 Acc. (higher better)", "(c) Ablation Progression: delta1", True),
    ]
    for col, (m_key, ylabel, title, higher_better) in enumerate(metric_specs):
        ax = axes[0, col]
        y_oof = oof_t[m_key].to_numpy(dtype=np.float64)
        y_tst = test_t[m_key].to_numpy(dtype=np.float64)
        ax.plot(x, y_oof, "--o", color="#495057", mfc="white", lw=1.4, ms=5.5, label="5-Fold Train OOF (N=795)")
        ax.plot(x, y_tst, "-D", color="#0072B2", lw=2.0, ms=5.8, label="Held-out Test (N=654)")
        ax.axvline(3, color="#D95F02", ls=":", lw=1.0, alpha=0.75)

        b3_v = float(y_tst[3])
        geo_v = float(y_tst[-1])
        ax.annotate(
            f"B3: {b3_v:.4f}",
            xy=(3, b3_v),
            xytext=(3.18, b3_v + (0.008 if m_key == "abs_rel" else (0.014 if not higher_better else -0.016))),
            fontsize=7.8,
            color="#D95F02",
            fontweight="bold",
        )
        ax.annotate(
            f"Ours: {geo_v:.4f}",
            xy=(x[-1], geo_v),
            xytext=(x[-1] - 2.35, geo_v + (0.0045 if m_key == "abs_rel" else (0.009 if not higher_better else 0.008))),
            arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 0.8},
            fontsize=8.0,
            color="#0072B2",
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_SHORT[m] for m in order], fontsize=7.6, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        if col == 0:
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.8)

        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        ax.set_ylim(ymin - 0.16 * span, ymax + 0.16 * span)

    # Bottom row: Paired Scene-Cluster Bootstrap 95% CI forest plots
    refs = ["raw", "b1_global_scale", "b2_monotone_curve", "b3_residual_v1"]
    ref_labels = {
        "raw": "vs Raw",
        "b1_global_scale": "vs B1 Scalar",
        "b2_monotone_curve": "vs B2 Spline",
        "b3_residual_v1": "vs B3_prev (Ref.)",
    }
    forest_titles = [
        ("abs_rel", "AbsRel Gain (B - GeoCalib-3D)", "(d) Paired 10k-Boot 95% CI: AbsRel Gain"),
        ("rmse", "RMSE Reduction (m)", "(e) Paired 10k-Boot 95% CI: RMSE Gain"),
        ("delta1", "delta1 Gain (GeoCalib-3D - B)", "(f) Paired 10k-Boot 95% CI: delta1 Gain"),
    ]
    legend_handles = None
    for col, (m_key, xlabel, title) in enumerate(forest_titles):
        ax = axes[1, col]
        ax.axvline(0.0, color="#6C757D", ls="--", lw=1.0)
        y_pos = np.arange(len(refs))[::-1]
        max_hi = 0.0
        for row_i, ref in enumerate(refs):
            r_oof = oof_boot_df[
                (oof_boot_df["reference"] == ref)
                & (oof_boot_df["candidate"] == PRIMARY_SOTA_METHOD)
                & (oof_boot_df["metric"] == m_key)
            ].iloc[0]
            r_tst = test_boot_df[
                (test_boot_df["reference"] == ref)
                & (test_boot_df["candidate"] == PRIMARY_SOTA_METHOD)
                & (test_boot_df["metric"] == m_key)
            ].iloc[0]

            yo = y_pos[row_i] + 0.14
            yt = y_pos[row_i] - 0.14
            h_oof = ax.errorbar(
                r_oof["gain"],
                yo,
                xerr=[[r_oof["gain"] - r_oof["ci95_low"]], [r_oof["ci95_high"] - r_oof["gain"]]],
                fmt="s",
                color="#495057",
                mfc="white",
                ms=5.2,
                capsize=3,
                lw=1.3,
                label="Train OOF (249 scenes)",
            )
            h_tst = ax.errorbar(
                r_tst["gain"],
                yt,
                xerr=[[r_tst["gain"] - r_tst["ci95_low"]], [r_tst["ci95_high"] - r_tst["gain"]]],
                fmt="o",
                color="#0072B2" if ref == "b3_residual_v1" else "#1B9E77",
                ms=5.8,
                capsize=3.5,
                lw=1.8,
                label="Held-out Test (215 scenes)",
            )
            if col == 0 and row_i == len(refs) - 1:
                legend_handles = [h_oof, h_tst]
            ax.text(
                r_tst["ci95_high"] + 0.001,
                yt,
                f"+{r_tst['gain']:.4f} [{r_tst['ci95_low']:.4f}, {r_tst['ci95_high']:.4f}]",
                va="center",
                ha="left",
                fontsize=7.3,
                color="#005A8D" if ref == "b3_residual_v1" else "#1A1A1A",
                fontweight="bold" if ref == "b3_residual_v1" else "normal",
            )
            max_hi = max(max_hi, float(r_oof["ci95_high"]), float(r_tst["ci95_high"]))

        ax.set_yticks(y_pos)
        ax.set_yticklabels([ref_labels[r] for r in refs])
        ax.set_xlim(-0.003, max_hi * 1.92)
        ax.set_ylim(-0.6, len(refs) - 0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=["Train OOF (249 scenes, N=795)", "Held-out Test (215 scenes, N=654)"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=True,
            framealpha=0.95,
            fontsize=8.5,
        )

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig3_mechanism_diagnostics(
    out_path: Path,
    diag_df: pd.DataFrame,
    frozen_sota: dict[str, Any],
    spatial_data: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(14.4, 9.2))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.35)

    # Row 1: Scatter + binned mean +- SE BEFORE vs AFTER GeoCalib-3D
    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.22)
    ft = diag_df["f_tilde"].to_numpy(dtype=np.float64)
    eb3 = diag_df["median_log_err_b3"].to_numpy(dtype=np.float64)
    egeo = diag_df["median_log_err_geocalib3d"].to_numpy(dtype=np.float64)

    fs = frozen_sota["primary_model"]["focal_spline"]
    f_curve = np.linspace(0.35, 1.35, 250)
    inv_hf = -np.interp(f_curve, fs["x_knots"], fs["y_knots"])

    # Quantile edges for clean binned trend
    q_edges = np.quantile(ft, np.linspace(0.0, 1.0, 9))
    q_edges[0] -= 1e-6
    q_edges[-1] += 1e-6

    for col_i, (err_arr, color, title, show_hf) in enumerate(
        (
            (eb3, "#D95F02", "(a) Before Calibration (B3_prev): Scale Error vs Focal Length", True),
            (egeo, "#0072B2", "(b) After GeoCalib-3D (Ours): Flattened Focal-Scale Drift", False),
        )
    ):
        ax = fig.add_subplot(gs_top[0, col_i])
        ax.axhline(0.0, color="#6C757D", ls="--", lw=0.9)
        ax.scatter(ft, err_arr, s=16, alpha=0.28, color=color, edgecolors="none", label="654 Test Images")

        bx, by, bse = [], [], []
        for q0, q1 in zip(q_edges[:-1], q_edges[1:]):
            mask = (ft >= q0) & (ft < q1)
            if np.any(mask):
                bx.append(float(ft[mask].mean()))
                by.append(float(err_arr[mask].mean()))
                bse.append(float(err_arr[mask].std(ddof=1) / np.sqrt(mask.sum())))
        ax.errorbar(bx, by, yerr=bse, fmt="-o", color="#1A1A1A", mfc="white", ms=5.0, lw=1.6, capsize=3, label="Binned Mean +/- SE")
        if show_hf:
            ax.plot(f_curve, inv_hf, color="#0072B2", ls="-.", lw=1.8, label="Predicted Drift (-h_f(f_tilde))")

        r_val = float(np.corrcoef(ft, err_arr)[0, 1])
        sd_val = float(np.std(err_arr))
        mad_val = float(np.mean(np.abs(err_arr)))
        stat_txt = f"Pearson r = {r_val:+.3f}\nSD(log err) = {sd_val:.4f}\nMAE(log err) = {mad_val:.4f}"
        ax.text(
            0.03,
            0.06,
            stat_txt,
            transform=ax.transAxes,
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#ADB5BD", "alpha": 0.92},
        )
        ax.set_xlim(0.32, 1.38)
        ax.set_ylim(-0.62, 0.42)
        ax.set_xlabel("Predicted Normalized Focal Length  f_tilde = 0.5*(fx + fy) / 480")
        ax.set_ylabel("Per-Image Median Log Scale Error  log(Z_pred / Z_gt)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)

    # Row 2: 2D Image-Space Log-Residual Heatmaps (12x16 over Eigen crop)
    gs_bot = gs_outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.06], wspace=0.16)
    all_maps = spatial_data["maps"]["all"]
    heat_specs = [
        ("raw", "(c) Spatial Log-Bias Map: Raw"),
        ("b3_residual_v1", "(d) Spatial Log-Bias Map: B3_prev"),
        (PRIMARY_SOTA_METHOD, "(e) Spatial Log-Bias Map: GeoCalib-3D"),
    ]
    mats = [np.asarray(all_maps[m]["mean_log_residual_12x16"], dtype=np.float64) for m, _ in heat_specs]
    vlim = max(0.04, float(np.ceil(max(float(np.max(np.abs(a))) for a in mats) * 200.0) / 200.0))

    im_last = None
    for idx_m, ((m_key, htitle), mat) in enumerate(zip(heat_specs, mats)):
        ax = fig.add_subplot(gs_bot[0, idx_m])
        im_last = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
            extent=[41, 601, 471, 45],
        )
        rms_b = all_maps[m_key]["rms_spatial_log_bias"]
        max_b = all_maps[m_key]["max_abs_spatial_log_bias"]
        ax.set_title(f"{htitle}\nRMS bias={rms_b:.4f}, Max|bias|={max_b:.4f}", fontsize=9.6)
        ax.set_xlabel("Crop X (px)")
        if idx_m == 0:
            ax.set_ylabel("Crop Y (px)")
        else:
            ax.set_yticklabels([])

    cax = fig.add_subplot(gs_bot[0, 3])
    cb = fig.colorbar(im_last, cax=cax)
    cb.set_label("Mean log(Z_pred / Z_gt)  [Red: Over, Blue: Under]")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig4_depth_bins_and_learning_curve(
    out_path: Path,
    bins_df: pd.DataFrame,
    lc_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 8.6), gridspec_kw={"hspace": 0.42, "wspace": 0.24})

    # Top Row: Depth Bins on Held-out Test (N=654)
    bin_methods = ["raw", "b2_monotone_curve", "b3_residual_v1", PRIMARY_SOTA_METHOD]
    bin_labels = {"raw": "Raw", "b2_monotone_curve": "B2 Spline", "b3_residual_v1": "B3_prev", PRIMARY_SOTA_METHOD: "GeoCalib-3D (Ours)"}

    # Canonical bin order from raw rows
    raw_bins = bins_df[bins_df["method"] == "raw"].sort_values("lower_m")
    lowers = raw_bins["lower_m"].tolist()
    uppers = raw_bins["upper_m"].tolist()
    imgs_cnt = raw_bins["images"].tolist()
    x_labels = [
        f"[{lo:g}, {hi:g})m\n(n={int(ni)})" for lo, hi, ni in zip(lowers, uppers, imgs_cnt)
    ]
    x = np.arange(len(lowers))
    bar_w = 0.20

    for col_i, (m_key, ylabel, title, fmt_delta) in enumerate(
        (
            ("abs_rel", "AbsRel (lower better)", "(a) Held-out Test AbsRel Across Depth Ranges", "rel"),
            ("delta1", "delta1 Accuracy (higher better)", "(b) Held-out Test delta1 Accuracy Across Depth Ranges", "pp"),
        )
    ):
        ax = axes[0, col_i]
        b3_vals = (
            bins_df[bins_df["method"] == "b3_residual_v1"]
            .sort_values("lower_m")[m_key]
            .to_numpy(dtype=np.float64)
        )
        geo_vals = (
            bins_df[bins_df["method"] == PRIMARY_SOTA_METHOD]
            .sort_values("lower_m")[m_key]
            .to_numpy(dtype=np.float64)
        )
        for j, bm in enumerate(bin_methods):
            sub = bins_df[bins_df["method"] == bm].sort_values("lower_m")
            vals = sub[m_key].to_numpy(dtype=np.float64)
            offset = (j - 1.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, color=PALETTE[bm], label=bin_labels[bm], edgecolor="white", linewidth=0.5)
            if bm == PRIMARY_SOTA_METHOD:
                for b_idx, bar in enumerate(bars):
                    top = max(float(bar.get_height()), float(b3_vals[b_idx]))
                    if fmt_delta == "rel":
                        pct = 100.0 * (b3_vals[b_idx] - geo_vals[b_idx]) / b3_vals[b_idx]
                        lbl = f"-{pct:.1f}%"
                    else:
                        dpp = 100.0 * (geo_vals[b_idx] - b3_vals[b_idx])
                        lbl = f"+{dpp:.1f}pp"
                    ax.text(
                        bar.get_x() + 0.5 * bar.get_width(),
                        top + (0.008 if m_key == "abs_rel" else 0.012),
                        lbl,
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        fontweight="bold",
                        color="#005A8D",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        if m_key == "delta1":
            ax.set_ylim(0.45, 1.12)
            ax.legend(loc="upper center", ncol=4, frameon=True, framealpha=0.94, fontsize=7.3)
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0.0, ymax * 1.14)
            ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.6)

    # Bottom Row: Train-Only 5-Fold Learning Curve
    lc_methods = ["b1_global_scale", "b2_monotone_curve", "b3_residual_v1", PRIMARY_SOTA_METHOD]
    lc_labels = {
        "b1_global_scale": "B1 Global Scalar",
        "b2_monotone_curve": "B2 Monotone Curve",
        "b3_residual_v1": "B3_prev (Ref.)",
        PRIMARY_SOTA_METHOD: "GeoCalib-3D (Ours)",
    }
    fracs = sorted(lc_df["fraction"].unique())
    sub_b3 = lc_df[lc_df["method"] == "b3_residual_v1"].sort_values("fraction")
    xt_labels = [
        f"{int(round(f * 100))}%\n(~{int(round(sc))} sc.)"
        for f, sc in zip(sub_b3["fraction"], sub_b3["train_scenes_mean"])
    ]
    x_lc = np.array([10.0, 25.0, 50.0, 100.0])

    for col_i, (m_key, lo_k, hi_k, ylabel, title) in enumerate(
        (
            ("abs_rel", "ci95_lo_abs_rel", "ci95_hi_abs_rel", "5-Fold OOF AbsRel (lower better)", "(c) Sample-Efficiency Learning Curve: AbsRel"),
            ("delta1", "ci95_lo_delta1", "ci95_hi_delta1", "5-Fold OOF delta1 (higher better)", "(d) Sample-Efficiency Learning Curve: delta1"),
        )
    ):
        ax = axes[1, col_i]
        for lm in lc_methods:
            sub = lc_df[lc_df["method"] == lm].sort_values("fraction")
            yv = sub[m_key].to_numpy(dtype=np.float64)
            ylo = sub[lo_k].to_numpy(dtype=np.float64)
            yhi = sub[hi_k].to_numpy(dtype=np.float64)
            lw = 2.2 if lm == PRIMARY_SOTA_METHOD else 1.5
            marker = "D" if lm == PRIMARY_SOTA_METHOD else "o"
            ax.plot(x_lc, yv, f"-{marker}", color=PALETTE[lm], lw=lw, ms=5.5, label=lc_labels[lm])
            if lm in ("b3_residual_v1", PRIMARY_SOTA_METHOD):
                ax.fill_between(x_lc, ylo, yhi, color=PALETTE[lm], alpha=0.14)

        geo_10 = float(lc_df[(lc_df["method"] == PRIMARY_SOTA_METHOD) & (lc_df["fraction"] == 0.10)][m_key].iloc[0])
        b3_100 = float(lc_df[(lc_df["method"] == "b3_residual_v1") & (lc_df["fraction"] == 1.00)][m_key].iloc[0])
        ax.annotate(
            f"GeoCalib@10% ({geo_10:.4f})\nbeats B3@100% ({b3_100:.4f})",
            xy=(10.0, geo_10),
            xytext=(24.0, geo_10 + (-0.0015 if m_key == "abs_rel" else 0.003)),
            arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.9},
            fontsize=7.6,
            color="#005A8D",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#0072B2", "alpha": 0.88},
        )
        ax.set_xticks(x_lc)
        ax.set_xticklabels(xt_labels, fontsize=8.0)
        ax.set_xlabel("Fraction of Fitting Scenes Used per CV Fold (of 249 Train Scenes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper right" if m_key == "abs_rel" else "lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig5_qualitative_and_failures(
    out_path: Path,
    cases_df: pd.DataFrame,
    data_root: Path,
    prediction_dir: Path,
    frozen_sota: dict[str, Any],
    b3_frozen: dict[str, Any],
) -> None:
    cache = load_nyu_rgb_cache(prediction_dir, data_root, allowed_split="test", compute_hashes=False, verify_scenes=False)
    want_labels = ("largest_gain_1", "largest_gain_2", "p75_gain", "largest_regression_1")
    pretty_role = {
        "largest_gain_1": "Best Gain #1",
        "largest_gain_2": "Best Gain #2",
        "p75_gain": "Typical Gain (P75)",
        "largest_regression_1": "Failure Case (#1 Reg.)",
    }
    rows = [cases_df[cases_df["case_label"] == lbl].iloc[0] for lbl in want_labels]

    fig = plt.figure(figsize=(15.4, 10.6))
    gs = fig.add_gridspec(4, 6, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.05], wspace=0.09, hspace=0.36)

    col_titles = ("Input RGB (Crop)", "GT Depth (0.5-6m)", "Raw |Error|", "B3_prev |Error|", "GeoCalib-3D |Error|")
    err_vmax = 1.25
    cmap_err = matplotlib.colormaps.get_cmap("magma").copy()
    cmap_err.set_bad(color="#E9ECEF")
    cmap_gt = matplotlib.colormaps.get_cmap("turbo").copy()
    cmap_gt.set_bad(color="#E9ECEF")

    im_err_ref = None
    primary_model = frozen_sota["primary_model"]

    with open_nyu(data_root) as handle:
        for r_i, case in enumerate(rows):
            idx = int(case["index"])
            label = str(case["case_label"])
            is_fail = label == "largest_regression_1"

            rgb = nyu_image(handle, idx)[CROP]
            gt = nyu_depth(handle, idx)[CROP].astype(np.float64)
            valid = valid_depth(gt)
            raw_p = cache[idx].astype(np.float64)
            k_mat = cache.intrinsics(idx).astype(np.float64)

            b3_pred = frozen_predictions(raw_p, b3_frozen)["residual_scale"][CROP]
            geo_pred = apply_sota_model(raw_p, k_mat, primary_model, use_torch_fast=True)[CROP]
            raw_crop = raw_p[CROP]

            gt_vis = np.ma.masked_where(~valid, gt)
            err_raw = np.ma.masked_where(~valid, np.abs(raw_crop - gt))
            err_b3 = np.ma.masked_where(~valid, np.abs(b3_pred - gt))
            err_geo = np.ma.masked_where(~valid, np.abs(geo_pred - gt))

            panels = [
                ("rgb", rgb, f"{pretty_role[label]}\nidx={idx}, f~={case['f_tilde']:.3f}"),
                ("gt", gt_vis, f"Scene: {case['scene']}\nmed(GT)={np.median(gt[valid]):.2f}m"),
                ("err", err_raw, f"Raw: AR={case['raw_abs_rel']:.3f}, d1={case['raw_delta1']:.3f}"),
                ("err", err_b3, f"B3: AR={case['ref_abs_rel']:.3f}, d1={case['ref_delta1']:.3f}"),
                ("err", err_geo, f"Ours: AR={case['cand_abs_rel']:.3f}, d1={case['cand_delta1']:.3f}"),
            ]

            for c_i, (ptype, arr, sub_title) in enumerate(panels):
                ax = fig.add_subplot(gs[r_i, c_i])
                if ptype == "rgb":
                    ax.imshow(arr)
                elif ptype == "gt":
                    ax.imshow(arr, cmap=cmap_gt, vmin=0.5, vmax=6.0)
                else:
                    im_err_ref = ax.imshow(arr, cmap=cmap_err, vmin=0.0, vmax=err_vmax)

                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    if is_fail:
                        sp.set_edgecolor("#C92A2A")
                        sp.set_linewidth(1.8 if c_i == 4 else 1.2)
                    else:
                        sp.set_edgecolor("#ADB5BD")
                        sp.set_linewidth(0.8)

                full_title = (f"{col_titles[c_i]}\n{sub_title}" if r_i == 0 else sub_title)
                t_color = "#C92A2A" if (is_fail and c_i in (0, 4)) else ("#005A8D" if (not is_fail and c_i == 4) else "#1A1A1A")
                ax.set_title(
                    full_title,
                    fontsize=8.2,
                    color=t_color,
                    fontweight="bold" if (c_i in (0, 4)) else "normal",
                    pad=4,
                )

    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(im_err_ref, cax=cax, extend="max")
    cb.set_label("Absolute Depth Error |Z_pred - Z_gt| (m) [Shared Scale]", fontsize=9.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def build_part3b_sota_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Build SHA-256 manifest verifiable with `python -m depthlab.verify`."""
    repo_root = Path(repo_root).resolve()
    manifest_rel = Path(manifest_path).resolve().relative_to(repo_root).as_posix()

    candidates: list[Path] = []
    for rel in (
        "depthlab/sota_io.py",
        "depthlab/sota_diagnose.py",
        "depthlab/sota_calibration.py",
        "depthlab/sota_evaluate.py",
        "depthlab/sota_figures.py",
        "tests/test_part3b_sota.py",
        "docs/PART3B_SOTA.md",
    ):
        p = repo_root / rel
        if p.is_file():
            candidates.append(p)

    res_dir = repo_root / "results" / "part3b_sota"
    for p in sorted(list(res_dir.glob("*.csv")) + list(res_dir.glob("*.json"))):
        if p.resolve().relative_to(repo_root).as_posix() != manifest_rel:
            candidates.append(p)

    fig_dir = repo_root / "results" / "figures" / "part3b_sota"
    for p in sorted(fig_dir.glob("*.png")):
        candidates.append(p)

    entries = []
    for path in candidates:
        rel_str = path.resolve().relative_to(repo_root).as_posix()
        entries.append(
            {
                "path": rel_str,
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )

    payload = {
        "schema": "part3b_sota_manifest_v1",
        "file_count": len(entries),
        "files": entries,
    }
    write_json_lf(manifest_path, payload)
    print(f"Manifest written: {manifest_path} ({len(entries)} verified files)", flush=True)
    return payload


def generate_all_figures(
    data_root: Path,
    prediction_dir: Path,
    results_dir: Path,
    b3_frozen_path: Path,
    output_dir: Path,
) -> list[Path]:
    _configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    frozen_sota = load_frozen_sota(results_dir / "frozen_sota_model.json")
    b3_frozen = load_frozen(b3_frozen_path)

    oof_sum = pd.read_csv(results_dir / "oof_ablation_summary.csv")
    test_sum = pd.read_csv(results_dir / "test_summary.csv")
    oof_boot = pd.read_csv(results_dir / "oof_paired_bootstrap.csv")
    test_boot = pd.read_csv(results_dir / "test_paired_bootstrap.csv")
    diag_df = pd.read_csv(results_dir / "test_mechanism_diagnostics.csv")
    bins_df = pd.read_csv(results_dir / "test_depth_bins.csv")
    lc_df = pd.read_csv(results_dir / "oof_learning_curve.csv")
    cases_df = pd.read_csv(results_dir / "test_representative_cases.csv")
    runtime = json.loads((results_dir / "runtime_overhead.json").read_text(encoding="utf-8"))
    spatial_data = json.loads((results_dir / "test_spatial_residuals.json").read_text(encoding="utf-8"))

    p1 = output_dir / "fig1_pipeline_overview.png"
    p2 = output_dir / "fig2_ablation_and_ci.png"
    p3 = output_dir / "fig3_mechanism_diagnostics.png"
    p4 = output_dir / "fig4_depth_bins_and_learning_curve.png"
    p5 = output_dir / "fig5_qualitative_and_failures.png"

    plot_fig1_pipeline_overview(p1, frozen_sota, b3_frozen, diag_df, test_sum, runtime)
    plot_fig2_ablation_and_ci(p2, oof_sum, test_sum, oof_boot, test_boot)
    plot_fig3_mechanism_diagnostics(p3, diag_df, frozen_sota, spatial_data)
    plot_fig4_depth_bins_and_learning_curve(p4, bins_df, lc_df)
    plot_fig5_qualitative_and_failures(p5, cases_df, data_root, prediction_dir, frozen_sota, b3_frozen)

    for p in (p1, p2, p3, p4, p5):
        print(f"Generated {p} ({p.stat().st_size} bytes)", flush=True)
    return [p1, p2, p3, p4, p5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results/part3b_sota"))
    parser.add_argument("--b3-frozen", type=Path, default=Path("results/frozen_20261111/part3b_residual_20261111/final/frozen_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/part3b_sota"))
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not args.manifest_only:
        if args.data_root is None or args.predictions is None:
            parser.error("--data-root and --predictions are required unless --manifest-only is set")
        generate_all_figures(args.data_root, args.predictions, args.results_dir, args.b3_frozen, args.output_dir)
    if args.manifest_out is not None or args.manifest_only:
        target = args.manifest_out if args.manifest_out is not None else (args.results_dir / "manifest.json")
        build_part3b_sota_manifest(repo_root, target)


if __name__ == "__main__":
    main()
