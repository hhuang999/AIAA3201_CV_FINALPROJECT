"""300-DPI publication figures for Parts 1-3 and unified project SHA-256 manifest builder."""
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
from .geocalib_cache import load_geocalib_cache
from .geocalib_io import file_sha256, load_nyu_rgb_cache, write_json_lf
from .geocalib_model import (
    PRIMARY_GEO_METHOD,
    PRIMARY_MULTIPASS_METHOD,
    PRIMARY_SINGLE_PASS_METHOD,
    apply_geocalib_model,
    assemble_semantic_vector,
    load_frozen_geocalib,
)
from .metrics import CROP, valid_depth


SHORT_LABELS = {
    "raw": "Raw",
    "global_scale": "Global Scale",
    "monotone_spline": "Monotone Spline",
    "geo_focal_3d": "+Focal & 3D Geo",
    "geo_qknots_robust": "QKnots+Robust",
    "geo_inv_residual": "+InvResidual",
    "geosem_scale": "+SemScale",
    "geocalib_3d": "GeoCalib-3D (1x)",
    "flip_geo_only": "2x Flip+Geo",
    "flip_geo_gated": "2x Flip+Gate",
    "geocalib_3d_flip": "GeoCalib+Flip (2x)",
}

PALETTE = {
    "raw": "#6C757D",
    "global_scale": "#8E9AAF",
    "monotone_spline": "#D95F02",
    "geo_focal_3d": "#4E79A7",
    "geo_qknots_robust": "#A6761D",
    "geo_inv_residual": "#66A61E",
    "geosem_scale": "#1B9E77",
    "geocalib_3d": "#0072B2",
    "flip_geo_only": "#9467BD",
    "flip_geo_gated": "#8C564B",
    "geocalib_3d_flip": "#6A1B9A",
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
            "legend.fontsize": 8.0,
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


def plot_project_pipeline(out_path: Path, test_sum_df: pd.DataFrame) -> None:
    """Render clean full-project pipeline diagram spanning Parts 1, 2, and 3."""
    fig, ax = plt.subplots(figsize=(15.4, 6.2))
    ax.set_xlim(0.0, 15.6)
    ax.set_ylim(0.0, 6.4)
    ax.axis("off")
    ax.set_title("AIAA 3201 Depth Estimation Pipeline: Stereo Geometry, Monocular Metric Calibration & Video Consistency", pad=10, fontsize=11.5)

    # Row 1: Part 1
    _draw_box(ax, 0.2, 4.45, 3.0, 1.35, "Part 1: Middlebury Stereo\n15 TrainingQ Rectified Pairs\n(Quarter-Res + GT Disparity)", "#EEF3F7", "#5C7A99", fontsize=8.0, bold=True)
    _draw_box(ax, 3.65, 4.45, 3.55, 1.35, "Bidirectional SGBM Ablations\nreflect_3way / reflect_hh4 / unpad\n+ 1 px Left-Right Consistency", "#EEF3F7", "#5C7A99", fontsize=7.8)
    _draw_box(ax, 7.65, 4.45, 7.75, 1.35, "Geometric Metric Depth Conversion & Region Diagnostics\nreflect_3way: MAE = 1.3608 px (1.0508 nonoccluded), Cov = 74.33%, 54.3 ms/pair CPU\nExplicit baseline-focal disparity sensitivity quantification", "#E6F4EE", "#1B9E77", fontsize=7.7)
    _draw_arrow(ax, (3.23, 5.12), (3.62, 5.12))
    _draw_arrow(ax, (7.23, 5.12), (7.62, 5.12))

    # Row 2: Part 2
    _draw_box(ax, 0.2, 2.35, 3.0, 1.45, "Part 2: NYU V2 & DAVIS Video\n1,449 RGB-D Frames (No GT K)\n+ 70 Video Frames @ 24 FPS", "#F3EFF6", "#756BB1", fontsize=7.9, bold=True)
    _draw_box(ax, 3.65, 2.35, 3.55, 1.45, "Frozen Pretrained Backbones\nUniDepthV2 ViT-S -> Z_hat, K_hat\nDAV2 Small vs VDA Small (Video)", "#F3EFF6", "#756BB1", fontsize=7.7)
    _draw_box(ax, 7.65, 2.35, 7.75, 1.45, "Train Camera Diagnostic & Temporal Video Consistency Application\nCamera diag. (N=795 train): known K improves AbsRel 0.1179 -> 0.0971 (r = 0.523)\nVDA vs DAV2 side-by-side clip comparison & depth-aware background blur MP4", "#F3EFF6", "#756BB1", fontsize=7.6)
    _draw_arrow(ax, (3.23, 3.08), (3.62, 3.08))
    _draw_arrow(ax, (7.23, 3.08), (7.62, 3.08))

    # Row 3: Part 3
    gc_row = test_sum_df.loc[test_sum_df["method"] == PRIMARY_SINGLE_PASS_METHOD].iloc[0]
    _draw_box(ax, 0.2, 0.30, 3.0, 1.45, "Part 3: Train-Only Fit (795)\n5-Fold Scene-Disjoint CV (4 Seeds)\nSplits Hard-Guarded Before Eval", "#E7F1FA", "#0072B2", fontsize=7.9, bold=True)
    _draw_box(ax, 3.65, 0.30, 3.55, 1.45, "GeoCalib-3D Architecture\n1/Z_hat Monotone Spline + h_f(f_tilde)\n+ 3D Geo + Ridge Internal Semantics", "#E7F1FA", "#0072B2", fontsize=7.6)
    _draw_box(ax, 7.65, 0.30, 7.75, 1.45, f"Held-Out 654-Image NYU Test (215 Scenes, 46.9 ms/img End-to-End GPU)\nSingle-Pass GeoCalib-3D: AbsRel 0.1145 -> {gc_row['abs_rel']:.4f}, RMSE 0.3988 -> {gc_row['rmse']:.4f} m, d1 {gc_row['delta1']*100:.1f}%\nBeats Monotone Spline (0.1008) & +Focal & 3D Geo (0.0865) with 10k-Boot CI > 0", "#D0E6FA", "#005A8D", lw=1.5, fontsize=7.6, bold=True)
    _draw_arrow(ax, (3.23, 1.02), (3.62, 1.02))
    _draw_arrow(ax, (7.23, 1.02), (7.62, 1.02))

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_nyu_raw_examples(out_path: Path, data_root: Path, orig_preds: Path) -> None:
    """Render 3 representative NYU examples comparing Input RGB, GT Depth, Raw UniDepthV2, and Raw |Error|."""
    cache = load_nyu_rgb_cache(orig_preds, data_root, allowed_split="test", compute_hashes=False, verify_scenes=False)
    sample_ids = [0, 769, 1448]
    # Note: Verify whether 0, 769, 1448 are in test or use allowed_split="all" purely for Part 2 visualization
    cache_all = load_nyu_rgb_cache(orig_preds, data_root, allowed_split="all", compute_hashes=False, verify_scenes=False)
    del cache

    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(len(sample_ids), 7, width_ratios=[1.0, 1.0, 1.0, 0.045, 0.24, 1.0, 0.045], wspace=0.12, hspace=0.28)
    cmap_d = matplotlib.colormaps.get_cmap("turbo").copy()
    cmap_d.set_bad(color="#E9ECEF")
    cmap_e = matplotlib.colormaps.get_cmap("magma").copy()
    cmap_e.set_bad(color="#E9ECEF")

    im_d_ref, im_e_ref = None, None
    titles = ("RGB (Eigen Crop)", "GT Depth (0.5-6.0 m)", "Raw UniDepthV2 (1x RGB)", None, None, "Raw Abs. Error |Z_hat - Z_gt| (m)", None)
    with open_nyu(data_root) as handle:
        for r_i, idx in enumerate(sample_ids):
            rgb = nyu_image(handle, idx)[CROP]
            gt = nyu_depth(handle, idx)[CROP].astype(np.float64)
            valid = valid_depth(gt)
            raw = cache_all[idx].astype(np.float64)[CROP]
            ar = float(np.mean(np.abs(raw[valid] - gt[valid]) / gt[valid]))
            d1 = float(np.mean(np.maximum(raw[valid] / gt[valid], gt[valid] / raw[valid]) < 1.25))

            ax0 = fig.add_subplot(gs[r_i, 0])
            ax0.imshow(rgb)
            ax0.set_xticks([]); ax0.set_yticks([])
            ax0.set_title(f"{titles[0]}\nNYU idx {idx}" if r_i == 0 else f"NYU idx {idx}", fontsize=8.5, fontweight="bold")

            ax1 = fig.add_subplot(gs[r_i, 1])
            ax1.imshow(np.ma.masked_where(~valid, gt), cmap=cmap_d, vmin=0.5, vmax=6.0)
            ax1.set_xticks([]); ax1.set_yticks([])
            ax1.set_title(titles[1] if r_i == 0 else "Filled GT Depth", fontsize=8.5)

            ax2 = fig.add_subplot(gs[r_i, 2])
            im_d_ref = ax2.imshow(np.ma.masked_where(~valid, raw), cmap=cmap_d, vmin=0.5, vmax=6.0)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title(f"{titles[2]}\nAR={ar:.3f}, d1={d1:.3f}" if r_i == 0 else f"AR={ar:.3f}, d1={d1:.3f}", fontsize=8.5)

            ax4 = fig.add_subplot(gs[r_i, 5])
            err = np.ma.masked_where(~valid, np.abs(raw - gt))
            im_e_ref = ax4.imshow(err, cmap=cmap_e, vmin=0.0, vmax=1.25)
            ax4.set_xticks([]); ax4.set_yticks([])
            ax4.set_title(titles[5] if r_i == 0 else f"Mean |err|={float(np.mean(np.abs(raw[valid] - gt[valid]))):.3f} m", fontsize=8.5)

    cax_d = fig.add_subplot(gs[:, 3])
    fig.colorbar(im_d_ref, cax=cax_d).set_label("Metric Depth (m)")
    cax_e = fig.add_subplot(gs[:, 6])
    fig.colorbar(im_e_ref, cax=cax_e, extend="max").set_label("Absolute Error (m)")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig1_architecture_and_oracles(
    out_path: Path,
    oracles_df: pd.DataFrame,
    signal_r2_df: pd.DataFrame,
    oof_sum_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(15.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0.0, 13.4)
    ax.set_ylim(0.0, 7.2)
    ax.axis("off")
    ax.set_title("(a) GeoCalib-3D Dual-Track Architecture (Zero GT Intrinsics)", pad=8)

    ax.add_patch(FancyBboxPatch((0.08, 2.22), 13.2, 4.78, boxstyle="round,pad=0.05,rounding_size=0.16", fc="#F7FBFE", ec="#0072B2", lw=1.2, ls="-"))
    ax.text(0.28, 6.75, "Track A: Single-Pass Main Track (1x ViT-S Forward, 43.0 ms Backbone + 3.9 ms GPU / 5.8 ms CPU Post)", fontsize=7.6, fontweight="bold", color="#005A8D")

    _draw_box(ax, 0.22, 3.85, 1.50, 1.45, "RGB Image\n(480x640)\nSingle Pass", "#F1F3F5", "#495057", fontsize=7.0, bold=True)
    _draw_box(ax, 1.98, 3.65, 2.25, 1.85, "Frozen UniDepthV2\nViT-S Backbone\n-> Z_hat, K_hat\n+ Internal Feats", "#E9ECEF", "#343A40", fontsize=7.0)
    _draw_arrow(ax, (1.74, 4.58), (1.96, 4.58))

    _draw_box(
        ax, 4.55, 4.88, 4.35, 1.55,
        "1. Inverse-Depth & Focal Geometry\nMonotone 1/Z_hat Spline + h_f(f_tilde)\n+ Flip-Symmetric 3D Geo + Slope",
        "#E6F5F0", "#1B9E77", lw=1.2, fontsize=6.7,
    )
    _draw_box(
        ax, 4.55, 2.52, 4.35, 1.90,
        "2. Single-Pass Semantics (L2-Norm)\ncls (384) + patch_m/s (768) + df (512)\n-> Ridge Scale + 2D Field (v_tilde, r^2)\n(5018 raw -> 146.9 effective DoF)",
        "#E7F1FA", "#0072B2", lw=1.2, fontsize=6.6,
    )
    _draw_arrow(ax, (4.25, 4.95), (4.53, 5.60), "#1B9E77")
    _draw_arrow(ax, (4.25, 4.15), (4.53, 3.48), "#0072B2")

    sp_tst = test_sum_df.loc[test_sum_df["method"] == PRIMARY_SINGLE_PASS_METHOD].iloc[0]
    _draw_box(
        ax, 9.28, 3.42, 3.82, 2.38,
        f"GeoCalib-3D (1-Pass)\nSingle-Pass Winner\nTest AbsRel: {sp_tst['abs_rel']:.4f}\nTest RMSE: {sp_tst['rmse']:.4f} m\nTest delta1: {sp_tst['delta1']:.4f}",
        "#D0E6FA", "#005A8D", lw=1.6, fontsize=7.0, bold=True,
    )
    _draw_arrow(ax, (8.92, 5.60), (9.26, 4.95), "#1B9E77")
    _draw_arrow(ax, (8.92, 3.48), (9.26, 4.25), "#0072B2")

    ax.add_patch(FancyBboxPatch((0.08, 0.14), 13.2, 1.82, boxstyle="round,pad=0.05,rounding_size=0.16", fc="#FAF6FC", ec="#6A1B9A", lw=1.2, ls="--"))
    ax.text(0.28, 1.72, "Track B: Optional 2-Pass Flip-TTA Ablation (2x ViT-S Forward, 87.2 ms Backbone + 5.6 ms GPU Post)", fontsize=7.4, fontweight="bold", color="#6A1B9A")

    _draw_box(ax, 0.22, 0.30, 2.25, 1.22, "Horiz. Flip RGB\n-> 2nd ViT-S Pass\n(+44.1 ms)", "#F3E5F5", "#6A1B9A", fontsize=6.7)
    _draw_box(ax, 2.82, 0.30, 6.12, 1.22, "Un-Mirrored K2 + Learnable Blend w_flip\n+ Flip-Consistency (|f1-f2|, std d_log) + cls_sym (150.9 DoF)", "#F3E5F5", "#6A1B9A", fontsize=6.4)
    mp_tst = test_sum_df.loc[test_sum_df["method"] == PRIMARY_MULTIPASS_METHOD].iloc[0]
    _draw_box(ax, 9.28, 0.28, 3.82, 1.26, f"GeoCalib + Flip (2x)\nTest: {mp_tst['abs_rel']:.4f} / {mp_tst['rmse']:.4f} / {mp_tst['delta1']:.4f}", "#EBD8F5", "#6A1B9A", lw=1.4, fontsize=6.8, bold=True)
    _draw_arrow(ax, (2.49, 0.91), (2.80, 0.91), "#6A1B9A")
    _draw_arrow(ax, (8.96, 0.91), (9.26, 0.91), "#6A1B9A")

    gs_r = gs[0, 1].subgridspec(2, 1, hspace=0.52)

    ax_w = fig.add_subplot(gs_r[0, 0])
    oof_t = oof_sum_df.set_index("method")
    ora_t = oracles_df.set_index("method")
    wf_labels = ["Raw", "Monotone\nSpline", "+Focal &\n3D Geo", "GeoCalib\n(1x)", "GeoCalib\n+Flip (2x)", "Scalar\nOracle", "Spatial\nOracle"]
    wf_vals = [
        float(oof_t.loc["raw", "abs_rel"]),
        float(oof_t.loc["monotone_spline", "abs_rel"]),
        float(oof_t.loc[PRIMARY_GEO_METHOD, "abs_rel"]),
        float(oof_t.loc[PRIMARY_SINGLE_PASS_METHOD, "abs_rel"]),
        float(oof_t.loc[PRIMARY_MULTIPASS_METHOD, "abs_rel"]),
        float(ora_t.loc["geo_oracle_absrel_scalar", "abs_rel"]),
        float(ora_t.loc["geo_oracle_spatial_linear", "abs_rel"]),
    ]
    wf_colors = ["#6C757D", "#D95F02", "#4E79A7", "#0072B2", "#6A1B9A", "#ADB5BD", "#CED4DA"]
    bars = ax_w.bar(np.arange(len(wf_labels)), wf_vals, width=0.62, color=wf_colors, edgecolor="#333333", linewidth=0.5)
    for b, val in zip(bars, wf_vals):
        ax_w.text(b.get_x() + 0.5 * b.get_width(), val + 0.0022, f"{val:.4f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax_w.set_xticks(np.arange(len(wf_labels)))
    ax_w.set_xticklabels(wf_labels, fontsize=7.0)
    ax_w.set_ylim(0.0, 0.135)
    ax_w.set_ylabel("5-Fold Train OOF AbsRel")
    ax_w.set_title("(b) Residual Headroom & Oracle Ceiling (N=795 Train)")
    ax_w.grid(True, axis="y", linestyle=":", alpha=0.45)

    ax_r2 = fig.add_subplot(gs_r[1, 0])
    blocks = ["geo_sp3d_control", "confidence_stats", "flip_consistency", "depth_feat_stats", "cls_token", "patch_mean_std"]
    blk_short = ["Geo 3D\nControl", "Conf.\nStats", "Flip\nConsist.", "Depth\nFeats", "CLS\nToken", "Patch\nMean/Std"]
    best_r2 = signal_r2_df.sort_values("cv_r2_e_geo_med", ascending=False).groupby("feature_block").first()
    r2_med = [float(best_r2.loc[b, "cv_r2_e_geo_med"]) for b in blocks]
    r2_rad = [float(best_r2.loc[b, "cv_r2_e_geo_radial"]) for b in blocks]
    xb = np.arange(len(blocks))
    ax_r2.axhline(0.0, color="#343A40", lw=0.9, ls="--")
    b1 = ax_r2.bar(xb - 0.17, r2_med, width=0.33, color="#0072B2", label="Scale Residual e_geo_med")
    b2 = ax_r2.bar(xb + 0.17, r2_rad, width=0.33, color="#1B9E77", label="Radial Residual e_geo_rad")
    for bar, v in list(zip(b1, r2_med)) + list(zip(b2, r2_rad)):
        if v > 0.01:
            ax_r2.text(bar.get_x() + 0.5 * bar.get_width(), v + 0.0015, f"{v:+.3f}", ha="center", va="bottom", fontsize=6.8)
    ax_r2.set_xticks(xb)
    ax_r2.set_xticklabels(blk_short, fontsize=7.4)
    ax_r2.set_ylim(-0.016, 0.068)
    ax_r2.set_ylabel("5-Fold Scene-CV R^2")
    ax_r2.set_title("(c) Zero-Leakage Signal R^2 Predicting Post-Geo Residual")
    ax_r2.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_r2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.3)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig2_ablation_and_ci(
    out_path: Path,
    oof_sum_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    test_boot_df: pd.DataFrame,
) -> None:
    order = test_sum_df["method"].tolist()
    oof_t = oof_sum_df.set_index("method").reindex(order)
    test_t = test_sum_df.set_index("method").reindex(order)
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.3), gridspec_kw={"hspace": 0.58, "wspace": 0.28})

    metric_specs = [
        ("abs_rel", "AbsRel (lower better)", "(a) Self-Contained Ablation: AbsRel", False),
        ("rmse", "RMSE (m, lower better)", "(b) Self-Contained Ablation: RMSE", False),
        ("delta1", "delta1 Acc. (higher better)", "(c) Self-Contained Ablation: delta1", True),
    ]
    geo_pos = order.index(PRIMARY_GEO_METHOD)
    sp_pos = order.index(PRIMARY_SINGLE_PASS_METHOD)
    for col, (m_key, ylabel, title, higher_b) in enumerate(metric_specs):
        ax = axes[0, col]
        yo = oof_t[m_key].to_numpy(dtype=np.float64)
        yt = test_t[m_key].to_numpy(dtype=np.float64)
        ax.axvspan(-0.5, 2.5, color="#F8F9FA", alpha=0.9, zorder=0)
        ax.axvspan(2.5, 7.5, color="#EEF6FC", alpha=0.6, zorder=0)
        ax.axvspan(7.5, len(order) - 0.5, color="#F5EFFA", alpha=0.6, zorder=0)
        ax.axvline(2.5, color="#ADB5BD", ls="--", lw=0.9)
        ax.axvline(7.5, color="#9467BD", ls="--", lw=0.9)

        ax.plot(x, yo, "--o", color="#495057", mfc="white", lw=1.4, ms=5.0, label="5-Fold Train OOF (N=795)")
        ax.plot(x, yt, "-D", color="#0072B2", lw=2.0, ms=5.5, label="Held-out Test (N=654)")

        ymin, ymax = min(yo.min(), yt.min()), max(yo.max(), yt.max())
        span = ymax - ymin
        ax.set_ylim(ymin - 0.22 * span, ymax + 0.30 * span)

        top_y = ymax + 0.23 * span
        ax.text(1.0, top_y, "Depth-Only", ha="center", va="center", fontsize=7.0, color="#495057", fontweight="bold")
        ax.text(5.0, top_y, "Track A: 1-Pass", ha="center", va="center", fontsize=7.0, color="#005A8D", fontweight="bold")
        ax.text(9.0, top_y, "Track B: 2-Pass", ha="center", va="center", fontsize=7.0, color="#6A1B9A", fontweight="bold")

        geo_v = float(yt[geo_pos])
        sp_v = float(yt[sp_pos])
        if not higher_b:
            ax.annotate(
                f"Geo3D: {geo_v:.4f}",
                xy=(geo_pos, geo_v),
                xytext=(geo_pos - 0.3, geo_v - 0.14 * span),
                fontsize=7.0, color="#4E79A7", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#4E79A7", "lw": 0.75},
            )
            ax.annotate(
                f"Full: {sp_v:.4f}",
                xy=(sp_pos, sp_v),
                xytext=(sp_pos - 1.1, sp_v - 0.16 * span),
                fontsize=7.3, color="#005A8D", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.8},
            )
        else:
            ax.annotate(
                f"Geo3D: {geo_v:.4f}",
                xy=(geo_pos, geo_v),
                xytext=(geo_pos - 1.3, geo_v + 0.11 * span),
                fontsize=7.0, color="#4E79A7", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#4E79A7", "lw": 0.75},
            )
            ax.annotate(
                f"Full: {sp_v:.4f}",
                xy=(sp_pos, sp_v),
                xytext=(sp_pos - 1.2, sp_v + 0.11 * span),
                fontsize=7.3, color="#005A8D", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.8},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[m] for m in order], fontsize=7.1, rotation=34, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        if col == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.90), frameon=True, framealpha=0.92, fontsize=7.2)

    cands = [
        "geo_qknots_robust",
        "geo_inv_residual",
        "geosem_scale",
        "geocalib_3d",
        "flip_geo_only",
        "flip_geo_gated",
        "geocalib_3d_flip",
    ]
    forest_specs = [
        ("abs_rel", "Test AbsRel Gain vs +Focal & 3D Geo", "(d) Test 10k-Boot 95% CI: AbsRel Gain"),
        ("rmse", "Test RMSE Gain vs +Focal & 3D Geo (m)", "(e) Test 10k-Boot 95% CI: RMSE Gain"),
        ("delta1", "Test delta1 Gain vs +Focal & 3D Geo", "(f) Test 10k-Boot 95% CI: delta1 Gain"),
    ]
    yp = np.arange(len(cands))[::-1]
    for col, (m_key, xlabel, title) in enumerate(forest_specs):
        ax = axes[1, col]
        ax.axvline(0.0, color="#6C757D", ls="--", lw=1.0)
        max_hi, min_lo = 0.0, 0.0
        for row_i, cm in enumerate(cands):
            r = test_boot_df[
                (test_boot_df["reference"] == PRIMARY_GEO_METHOD)
                & (test_boot_df["candidate"] == cm)
                & (test_boot_df["metric"] == m_key)
            ].iloc[0]
            y_val = yp[row_i]
            col_c = PALETTE[cm]
            is_win = cm in (PRIMARY_SINGLE_PASS_METHOD, PRIMARY_MULTIPASS_METHOD)
            ax.errorbar(
                r["gain"], y_val,
                xerr=[[r["gain"] - r["ci95_low"]], [r["ci95_high"] - r["gain"]]],
                fmt="D" if is_win else "o",
                color=col_c, ms=6.2 if is_win else 5.0, capsize=3.5, lw=1.8 if is_win else 1.3,
            )
            max_hi = max(max_hi, float(r["ci95_high"]))
            min_lo = min(min_lo, float(r["ci95_low"]))

            def _f4(v: float) -> str:
                vv = 0.0 if abs(float(v)) < 5e-5 else float(v)
                return f"{vv:+.4f}" if vv != 0.0 else " 0.0000"

            ax.text(
                float(r["ci95_high"]) + 0.0003 * (1.0 if m_key != "rmse" else 2.2),
                y_val,
                f"{_f4(r['gain'])} [{_f4(r['ci95_low'])}, {_f4(r['ci95_high'])}]",
                va="center", ha="left", fontsize=6.8,
                color=col_c if is_win else "#212529",
                fontweight="bold" if is_win else "normal",
            )
        ax.set_yticks(yp)
        ax.set_yticklabels([SHORT_LABELS[c] for c in cands] if col == 0 else [""] * len(cands), fontsize=8.0)
        span_x = max_hi - min_lo
        ax.set_xlim(min_lo - 0.10 * span_x, max_hi + 1.15 * span_x)
        ax.set_ylim(-0.6, len(cands) - 0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig3_mechanism_and_spatial_bias(
    out_path: Path,
    diag_df: pd.DataFrame,
    spatial_data: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(15.4, 9.4))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.38)

    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.22)
    ft = diag_df["f_tilde"].to_numpy(dtype=np.float64)
    e_geo = diag_df["median_log_err_geo"].to_numpy(dtype=np.float64)
    e_sp = diag_df["median_log_err_sp"].to_numpy(dtype=np.float64)

    q_edges = np.quantile(ft, np.linspace(0.0, 1.0, 9))
    q_edges[0] -= 1e-6
    q_edges[-1] += 1e-6

    audit_ids_fail = (839, 832, 326)
    audit_ids_gain = (1150, 1151)

    for col_i, (err_arr, col_hex, title, is_full) in enumerate(
        (
            (e_geo, "#4E79A7", "(a) Geometry-Only (+Focal & 3D Geo): False Focal Alarms Over-Scaled", False),
            (e_sp, "#0072B2", "(b) Full GeoCalib-3D (1-Pass): Semantic Prior Suppresses False Alarms", True),
        )
    ):
        ax = fig.add_subplot(gs_top[0, col_i])
        ax.axhline(0.0, color="#6C757D", ls="--", lw=0.9)
        ax.axvspan(0.32, 0.55, color="#FFF5F5", alpha=0.75, zorder=0)
        ax.scatter(ft, err_arr, s=15, alpha=0.26, color=col_hex, edgecolors="none", label="654 Test Images")

        bx, by, bse = [], [], []
        for q0, q1 in zip(q_edges[:-1], q_edges[1:]):
            m = (ft >= q0) & (ft < q1)
            if np.any(m):
                bx.append(float(ft[m].mean()))
                by.append(float(err_arr[m].mean()))
                bse.append(float(err_arr[m].std(ddof=1) / np.sqrt(m.sum())))
        ax.errorbar(bx, by, yerr=bse, fmt="-o", color="#1A1A1A", mfc="white", ms=4.8, lw=1.5, capsize=3, label="Binned Mean +/- SE")

        sub_f = diag_df[diag_df["index"].isin(audit_ids_fail)]
        sub_g = diag_df[diag_df["index"].isin(audit_ids_gain)]
        col_y = "median_log_err_sp" if is_full else "median_log_err_geo"
        ax.scatter(sub_f["f_tilde"], sub_f[col_y], s=58, marker="X", color="#C92A2A", zorder=6, label="False Alarms (#839/#832/#326)")
        ax.scatter(sub_g["f_tilde"], sub_g[col_y], s=54, marker="*", color="#1B9E77", zorder=6, label="Wide-FoV Collapse (#1150/#1151)")

        r839 = diag_df[diag_df["index"] == 839].iloc[0]
        ax.annotate(
            f"#839 ({'sem=-0.098' if is_full else 'over +26%'})",
            xy=(r839["f_tilde"], r839[col_y]),
            xytext=(0.37, 0.34 if not is_full else r839[col_y] + 0.105),
            fontsize=7.3, color="#C92A2A", fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#C92A2A", "lw": 0.8},
        )
        r1151 = diag_df[diag_df["index"] == 1151].iloc[0]
        ax.annotate(
            f"#1151 ({'sem=+0.097' if is_full else 'geo'})",
            xy=(r1151["f_tilde"], r1151[col_y]),
            xytext=(0.56, r1151[col_y] - 0.14),
            fontsize=7.3, color="#0B724E", fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#1B9E77", "lw": 0.8},
        )

        r_val = float(np.corrcoef(ft, err_arr)[0, 1])
        sd_val = float(np.std(err_arr))
        mad_val = float(np.mean(np.abs(err_arr)))
        lt34 = diag_df[diag_df["f_tilde"] < 0.55]
        lt_ar = float(lt34["abs_rel_sp" if is_full else "abs_rel_geo"].mean())
        stat_txt = f"r = {r_val:+.4f}, SD = {sd_val:.4f}\nMAE(log) = {mad_val:.4f}\nf<0.55 (n=34) AbsRel = {lt_ar:.4f}"
        ax.text(0.68, 0.27, stat_txt, fontsize=7.6, bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#ADB5BD", "alpha": 0.94})

        ax.set_xlim(0.32, 1.38)
        ax.set_ylim(-0.46, 0.42)
        ax.set_xlabel("Predicted Normalized Focal Length  f_tilde = 0.5*(fx + fy) / 480")
        ax.set_ylabel("Median Log Scale Error  log(Z_pred / Z_gt)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.9)

    gs_bot = gs_outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.16)
    all_maps = spatial_data["maps"]["all"]
    heat_specs = [
        ("raw", "(c) Spatial Bias: Raw UniDepthV2"),
        (PRIMARY_GEO_METHOD, "(d) Spatial Bias: +Focal & 3D Geo"),
        (PRIMARY_SINGLE_PASS_METHOD, "(e) Spatial Bias: Full GeoCalib-3D (1x)"),
    ]
    mats = [np.asarray(all_maps[m]["mean_log_residual_12x16"], dtype=np.float64) for m, _ in heat_specs]
    vlim = 0.075

    im_last = None
    for idx_m, ((m_key, htitle), mat) in enumerate(zip(heat_specs, mats)):
        ax = fig.add_subplot(gs_bot[0, idx_m])
        im_last = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto", extent=[41, 601, 471, 45])
        rms_b = all_maps[m_key]["rms_spatial_log_bias"]
        max_b = all_maps[m_key]["max_abs_spatial_log_bias"]
        ax.set_title(f"{htitle}\nRMS bias={rms_b:.4f}, Max|bias|={max_b:.4f}", fontsize=9.2)
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


def plot_fig4_bins_lc_pareto(
    out_path: Path,
    bins_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    runtime: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.2), gridspec_kw={"hspace": 0.42, "wspace": 0.24})

    bin_methods = ["raw", "global_scale", "monotone_spline", PRIMARY_GEO_METHOD, PRIMARY_SINGLE_PASS_METHOD, PRIMARY_MULTIPASS_METHOD]
    bin_labels = {
        "raw": "Raw",
        "global_scale": "Global Scale",
        "monotone_spline": "Monotone Spline",
        PRIMARY_GEO_METHOD: "+Focal & 3D Geo",
        PRIMARY_SINGLE_PASS_METHOD: "GeoCalib-3D (1x)",
        PRIMARY_MULTIPASS_METHOD: "GeoCalib+Flip (2x)",
    }
    raw_bins = bins_df[bins_df["method"] == "raw"].sort_values("lower_m")
    lowers = raw_bins["lower_m"].tolist()
    uppers = raw_bins["upper_m"].tolist()
    imgs_cnt = raw_bins["images"].tolist()
    x_labels = [f"[{lo:g}, {hi:g})m\n(n={int(ni)})" for lo, hi, ni in zip(lowers, uppers, imgs_cnt)]
    x = np.arange(len(lowers))
    bar_w = 0.135

    for col_i, (m_key, ylabel, title, fmt_t) in enumerate(
        (
            ("abs_rel", "AbsRel (lower better)", "(a) Held-Out Test AbsRel by Depth Bin", "rel"),
            ("delta1", "delta1 Accuracy (higher better)", "(b) Held-Out Test delta1 by Depth Bin", "pp"),
        )
    ):
        ax = axes[0, col_i]
        geo_vals = bins_df[bins_df["method"] == PRIMARY_GEO_METHOD].sort_values("lower_m")[m_key].to_numpy(dtype=np.float64)
        sp_vals = bins_df[bins_df["method"] == PRIMARY_SINGLE_PASS_METHOD].sort_values("lower_m")[m_key].to_numpy(dtype=np.float64)
        for j, bm in enumerate(bin_methods):
            sub = bins_df[bins_df["method"] == bm].sort_values("lower_m")
            vals = sub[m_key].to_numpy(dtype=np.float64)
            offset = (j - 2.5) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, color=PALETTE[bm], label=bin_labels[bm], edgecolor="white", lw=0.4)
            if bm == PRIMARY_SINGLE_PASS_METHOD:
                for bi, bar in enumerate(bars):
                    top = max(float(bar.get_height()), float(geo_vals[bi]))
                    lbl = (f"{100*(sp_vals[bi]-geo_vals[bi])/geo_vals[bi]:+.1f}%" if fmt_t == "rel" else f"{100*(sp_vals[bi]-geo_vals[bi]):+.1f}pp")
                    ax.text(bar.get_x() + 0.5 * bar.get_width(), top + (0.008 if m_key == "abs_rel" else 0.011), lbl, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#005A8D")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        if m_key == "delta1":
            ax.set_ylim(0.45, 1.15)
            ax.legend(loc="upper center", ncol=3, frameon=True, framealpha=0.94, fontsize=6.7)
        else:
            _, ymax = ax.get_ylim()
            ax.set_ylim(0.0, ymax * 1.18)
            ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, fontsize=6.8)

    ax_lc = axes[1, 0]
    lc_methods = ["monotone_spline", PRIMARY_GEO_METHOD, PRIMARY_SINGLE_PASS_METHOD, PRIMARY_MULTIPASS_METHOD]
    sub_ref = lc_df[lc_df["method"] == PRIMARY_SINGLE_PASS_METHOD].sort_values("fraction")
    xt_labels = [f"{int(round(f*100))}%\n(~{int(round(sc))} sc.)" for f, sc in zip(sub_ref["fraction"], sub_ref["train_scenes_mean"])]
    x_lc = np.array([10.0, 25.0, 50.0, 100.0])

    for lm in lc_methods:
        sub = lc_df[lc_df["method"] == lm].sort_values("fraction")
        yv = sub["abs_rel"].to_numpy(dtype=np.float64)
        ylo = sub["ci95_lo_abs_rel"].to_numpy(dtype=np.float64)
        yhi = sub["ci95_hi_abs_rel"].to_numpy(dtype=np.float64)
        ax_lc.plot(x_lc, yv, "-D" if lm == PRIMARY_SINGLE_PASS_METHOD else "-o", color=PALETTE[lm], lw=2.1 if lm == PRIMARY_SINGLE_PASS_METHOD else 1.5, ms=5.2, label=SHORT_LABELS[lm])
        if lm in (PRIMARY_GEO_METHOD, PRIMARY_SINGLE_PASS_METHOD):
            ax_lc.fill_between(x_lc, ylo, yhi, color=PALETTE[lm], alpha=0.14)

    gc_25 = float(lc_df[(lc_df["method"] == PRIMARY_SINGLE_PASS_METHOD) & (lc_df["fraction"] == 0.25)]["abs_rel"].iloc[0])
    geo_100 = float(lc_df[(lc_df["method"] == PRIMARY_GEO_METHOD) & (lc_df["fraction"] == 1.00)]["abs_rel"].iloc[0])
    ax_lc.annotate(
        f"GeoCalib-3D @ 25% ({gc_25:.4f})\nbeats Geo-Only @ 100% ({geo_100:.4f})",
        xy=(25.0, gc_25), xytext=(42.0, gc_25 + 0.0052),
        arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.9},
        fontsize=7.3, color="#005A8D", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#0072B2", "alpha": 0.90},
    )
    ax_lc.set_xticks(x_lc)
    ax_lc.set_xticklabels(xt_labels, fontsize=7.8)
    ax_lc.set_xlabel("Fraction of Train Fitting Scenes per CV Fold (of 249)")
    ax_lc.set_ylabel("5-Fold OOF AbsRel (lower better)")
    ax_lc.set_title("(c) Sample-Efficiency Learning Curve (Train Only)")
    ax_lc.grid(True, linestyle=":", alpha=0.45)
    ax_lc.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.2)

    ax_p = axes[1, 1]
    t_sum = test_sum_df.set_index("method")
    bb1 = float(runtime["backbone_single_pass_ms_per_image"])
    pts_cpu = [
        ("Raw (1x, GPU)", bb1, float(t_sum.loc["raw", "abs_rel"]), PALETTE["raw"], "o"),
        ("Monotone Spline (1x, CPU-post)", float(runtime["end_to_end_spline_ms"]), float(t_sum.loc["monotone_spline", "abs_rel"]), PALETTE["monotone_spline"], "s"),
        ("+Focal & 3D Geo (1x, CPU-post)", float(runtime["end_to_end_geo_focal_3d_ms"]), float(t_sum.loc[PRIMARY_GEO_METHOD, "abs_rel"]), PALETTE["geo_focal_3d"], "^"),
        ("GeoCalib-3D (1x, CPU-post)", float(runtime["end_to_end_geocalib_3d_cpu_post_ms"]), float(t_sum.loc[PRIMARY_SINGLE_PASS_METHOD, "abs_rel"]), PALETTE[PRIMARY_SINGLE_PASS_METHOD], "D"),
        ("GeoCalib+Flip (2x, CPU-post)", float(runtime["end_to_end_geocalib_3d_flip_cpu_post_ms"]), float(t_sum.loc[PRIMARY_MULTIPASS_METHOD, "abs_rel"]), PALETTE[PRIMARY_MULTIPASS_METHOD], "P"),
    ]
    pts_gpu = [
        ("GeoCalib-3D (1x, GPU-post)", float(runtime["end_to_end_geocalib_3d_gpu_ms"]), float(t_sum.loc[PRIMARY_SINGLE_PASS_METHOD, "abs_rel"]), "#005A8D"),
        ("GeoCalib+Flip (2x, GPU-post)", float(runtime["end_to_end_geocalib_3d_flip_gpu_ms"]), float(t_sum.loc[PRIMARY_MULTIPASS_METHOD, "abs_rel"]), "#6A1B9A"),
    ]

    px_par = [pts_cpu[0][1], pts_cpu[1][1], pts_cpu[2][1], pts_cpu[3][1]]
    py_par = [pts_cpu[0][2], pts_cpu[1][2], pts_cpu[2][2], pts_cpu[3][2]]
    ax_p.plot(px_par, py_par, "--", color="#0072B2", lw=1.3, alpha=0.7, label="Single-Pass Pareto Frontier")

    for lbl, lx, ly, col_c, mk in pts_cpu:
        ax_p.scatter([lx], [ly], color=col_c, marker=mk, s=75 if "GeoCalib-3D (" in lbl else 58, zorder=5, edgecolor="white", lw=0.7)
        if "Spline" in lbl:
            xytext_p = (lx + 3.0, ly + 0.0020)
        elif "+Focal" in lbl:
            xytext_p = (56.0, 0.0925)
        elif "GeoCalib-3D (" in lbl:
            xytext_p = (58.0, 0.0858)
        elif "+Flip" in lbl:
            xytext_p = (88.0, 0.0890)
        else:
            xytext_p = (lx + 2.2, ly + 0.0012)
        ax_p.annotate(
            f"{lbl}\n({lx:.1f}ms, {ly:.4f})",
            xy=(lx, ly), xytext=xytext_p,
            arrowprops={"arrowstyle": "->", "color": col_c, "lw": 0.75} if "Raw" not in lbl else None,
            fontsize=6.8, color=col_c, fontweight="bold" if "GeoCalib-3D (" in lbl else "normal",
        )

    for lbl, lx, ly, col_c in pts_gpu:
        ax_p.scatter([lx], [ly], color=col_c, marker="*", s=125, zorder=6, label=f"{lbl}: {lx:.1f}ms")

    ax_p.set_xlim(38.0, 125.0)
    ax_p.set_ylim(0.079, 0.120)
    ax_p.set_xlabel("End-to-End Single-Image Latency (ms/img, RTX 4060 Backbone + Post)")
    ax_p.set_ylabel("Held-Out Test AbsRel (lower better)")
    ax_p.set_title("(d) Accuracy-Latency Pareto Frontier (Single-Pass Dominates 2-Pass)")
    ax_p.grid(True, linestyle=":", alpha=0.45)
    ax_p.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.9)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig5_qualitative_and_failure_cases(
    out_path: Path,
    cases_spec: list[tuple[int, str, bool]],
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    frozen_model_path: Path,
    diag_df: pd.DataFrame,
) -> None:
    cache = load_geocalib_cache(orig_preds, aux_dir, flip_dir, data_root, allowed_split="test", compute_hashes=False, verify_scenes=False)
    frozen = load_frozen_geocalib(frozen_model_path)
    m_raw = frozen["models"]["raw"]
    m_geo = frozen["models"][PRIMARY_GEO_METHOD]
    m_sp = frozen["models"][PRIMARY_SINGLE_PASS_METHOD]
    sm_sp = m_sp["semantic_branch"]["sem_mode"]

    fig = plt.figure(figsize=(15.8, 12.8))
    gs = fig.add_gridspec(len(cases_spec), 6, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.05], wspace=0.09, hspace=0.40)

    col_titles = ("Input RGB (Crop)", "GT Depth (0.5-6m)", "Raw |Error|", "+Focal & 3D Geo |Error|", "Full GeoCalib-3D |Error|")
    err_vmax = 1.25
    cmap_err = matplotlib.colormaps.get_cmap("magma").copy()
    cmap_err.set_bad(color="#E9ECEF")
    cmap_gt = matplotlib.colormaps.get_cmap("turbo").copy()
    cmap_gt.set_bad(color="#E9ECEF")

    im_err_ref = None
    diag_map = diag_df.set_index("index")

    with open_nyu(data_root) as handle:
        for r_i, (idx, role_label, is_reg) in enumerate(cases_spec):
            rgb = nyu_image(handle, idx)[CROP]
            gt = nyu_depth(handle, idx)[CROP].astype(np.float64)
            valid = valid_depth(gt)
            p1 = cache[idx].astype(np.float64)
            k1 = cache.intrinsics(idx).astype(np.float64)
            cs = cache.confidence_summary(idx)
            cls1 = cache.cls_token(idx)
            id_arr = np.array([idx], dtype=np.int64)
            pm = cache.patch_mean_for(id_arr)[0]
            ps = cache.patch_std_for(id_arr)[0]
            df = cache.depth_feat_stats_for(id_arr)[0]
            xs = assemble_semantic_vector(sm_sp, cls1, pm, ps, df)

            pred_raw = apply_geocalib_model(p1, k1, m_raw, use_torch_fast=True)[CROP]
            pred_geo = apply_geocalib_model(p1, k1, m_geo, use_torch_fast=True)[CROP]
            pred_sp = apply_geocalib_model(p1, k1, m_sp, conf_summary=cs, cached_sem=xs, use_torch_fast=True)[CROP]

            gt_vis = np.ma.masked_where(~valid, gt)
            err_raw = np.ma.masked_where(~valid, np.abs(pred_raw - gt))
            err_geo = np.ma.masked_where(~valid, np.abs(pred_geo - gt))
            err_sp = np.ma.masked_where(~valid, np.abs(pred_sp - gt))

            drow = diag_map.loc[idx]
            ar_raw = float(np.mean(np.abs(pred_raw[valid] - gt[valid]) / gt[valid]))
            d1_raw = float(np.mean(np.maximum(pred_raw[valid] / gt[valid], gt[valid] / pred_raw[valid]) < 1.25))
            ar_geo = float(np.mean(np.abs(pred_geo[valid] - gt[valid]) / gt[valid]))
            d1_geo = float(np.mean(np.maximum(pred_geo[valid] / gt[valid], gt[valid] / pred_geo[valid]) < 1.25))
            ar_sp = float(np.mean(np.abs(pred_sp[valid] - gt[valid]) / gt[valid]))
            d1_sp = float(np.mean(np.maximum(pred_sp[valid] / gt[valid], gt[valid] / pred_sp[valid]) < 1.25))

            panels = [
                ("rgb", rgb, f"{role_label}\nidx={idx}, f~={drow['f_tilde']:.3f}"),
                ("gt", gt_vis, f"Scene: {drow['scene']}\nsem_shift={drow['shift_sp_sem']:+.3f}"),
                ("err", err_raw, f"Raw: AR={ar_raw:.3f}, d1={d1_raw:.3f}"),
                ("err", err_geo, f"+Geo3D: AR={ar_geo:.3f}, d1={d1_geo:.3f}"),
                ("err", err_sp, f"Full (1x): AR={ar_sp:.3f}, d1={d1_sp:.3f}"),
            ]

            for c_i, (ptype, arr, stitle) in enumerate(panels):
                ax = fig.add_subplot(gs[r_i, c_i])
                if ptype == "rgb":
                    ax.imshow(arr)
                elif ptype == "gt":
                    ax.imshow(arr, cmap=cmap_gt, vmin=0.5, vmax=6.0)
                else:
                    im_err_ref = ax.imshow(arr, cmap=cmap_err, vmin=0.0, vmax=err_vmax)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    if is_reg:
                        sp.set_edgecolor("#C92A2A")
                        sp.set_linewidth(1.8 if c_i >= 3 else 1.1)
                    else:
                        sp.set_edgecolor("#0072B2" if c_i == 4 else "#ADB5BD")
                        sp.set_linewidth(1.5 if c_i == 4 else 0.8)

                full_t = f"{col_titles[c_i]}\n{stitle}" if r_i == 0 else stitle
                tcol = "#C92A2A" if (is_reg and c_i in (0, 4)) else ("#005A8D" if (not is_reg and c_i == 4) else "#1A1A1A")
                ax.set_title(full_t, fontsize=8.0, color=tcol, fontweight="bold" if c_i in (0, 4) else "normal", pad=4)

    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(im_err_ref, cax=cax, extend="max")
    cb.set_label("Absolute Depth Error |Z_pred - Z_gt| (m) [Shared Scale]", fontsize=9.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def build_project_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Build unified SHA-256 manifest for all delivered project source, doc, result, and figure files."""
    repo_root = Path(repo_root).resolve()
    manifest_rel = Path(manifest_path).resolve().relative_to(repo_root).as_posix()

    candidates: list[Path] = []
    for rel in ("README.md", "requirements-reproduction.txt", "requirements-linux-cu128.lock.txt"):
        p = repo_root / rel
        if p.is_file():
            candidates.append(p)
    for folder, exts in (
        ("docs", (".md",)),
        ("depthlab", (".py",)),
        ("scripts", (".sh",)),
        ("tests", (".py",)),
        ("results/part1_stereo", (".csv", ".json")),
        ("results/part2_depth_video", (".csv", ".json")),
        ("results/part3_geocalib", (".csv", ".json", ".md")),
        ("results/figures", (".png",)),
    ):
        base = repo_root / folder
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    if p.resolve().relative_to(repo_root).as_posix() != manifest_rel:
                        candidates.append(p)

    # Deduplicate cleanly while preserving deterministic order
    seen: set[Path] = set()
    entries = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            entries.append(
                {
                    "path": rp.relative_to(repo_root).as_posix(),
                    "bytes": int(rp.stat().st_size),
                    "sha256": file_sha256(rp),
                }
            )
    entries.sort(key=lambda e: e["path"])
    payload = {"schema": "aiaa3201_project_manifest", "file_count": len(entries), "files": entries}
    write_json_lf(manifest_path, payload)
    print(f"Unified manifest written: {manifest_path} ({len(entries)} verified files)", flush=True)
    return payload


def generate_all_figures(
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    results_dir: Path,
    oracles_csv: Path,
    signal_r2_csv: Path,
    output_dir: Path,
    figures_root: Path,
) -> list[Path]:
    _configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_root = Path(figures_root)
    (figures_root / "part2_nyu").mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    oracles_df = pd.read_csv(oracles_csv)
    signal_r2_df = pd.read_csv(signal_r2_csv)
    oof_sum_df = pd.read_csv(results_dir / "oof_ablation_summary.csv")
    test_sum_df = pd.read_csv(results_dir / "test_summary.csv")
    test_boot_df = pd.read_csv(results_dir / "test_paired_bootstrap.csv")
    diag_df = pd.read_csv(results_dir / "test_mechanism_diagnostics.csv")
    bins_df = pd.read_csv(results_dir / "test_depth_bins.csv")
    lc_df = pd.read_csv(results_dir / "oof_learning_curve.csv")
    runtime = json.loads((results_dir / "runtime_overhead.json").read_text(encoding="utf-8"))
    spatial_data = json.loads((results_dir / "test_spatial_residuals.json").read_text(encoding="utf-8"))

    p_pipe = figures_root / "pipeline.png"
    p_nyu_ex = figures_root / "part2_nyu" / "nyu_raw_examples.png"
    p1 = output_dir / "fig1_architecture_and_oracles.png"
    p2 = output_dir / "fig2_ablation_and_ci.png"
    p3 = output_dir / "fig3_mechanism_and_spatial_bias.png"
    p4 = output_dir / "fig4_depth_bins_learning_curve_pareto.png"
    p5 = output_dir / "fig5_qualitative_and_failure_cases.png"

    plot_project_pipeline(p_pipe, test_sum_df)
    plot_nyu_raw_examples(p_nyu_ex, data_root, orig_preds)
    plot_fig1_architecture_and_oracles(p1, oracles_df, signal_r2_df, oof_sum_df, test_sum_df)
    plot_fig2_ablation_and_ci(p2, oof_sum_df, test_sum_df, test_boot_df)
    plot_fig3_mechanism_and_spatial_bias(p3, diag_df, spatial_data)
    plot_fig4_bins_lc_pareto(p4, bins_df, lc_df, test_sum_df, runtime)

    cases_spec = [
        (839, "False Alarm #1 Fixed", False),
        (832, "False Alarm #2 Mitigated", False),
        (326, "False Alarm #3 Mitigated", False),
        (1151, "Wide-FoV Collapse Boosted", False),
        (167, "Worst New Regression", True),
    ]
    plot_fig5_qualitative_and_failure_cases(
        p5, cases_spec, data_root, orig_preds, aux_dir, flip_dir,
        results_dir / "frozen_geocalib_model.json", diag_df,
    )

    out_list = [p_pipe, p_nyu_ex, p1, p2, p3, p4, p5]
    for p in out_list:
        print(f"Generated {p} ({p.stat().st_size} bytes)", flush=True)
    return out_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--orig-preds", type=Path, default=Path("outputs/nyu_rgb"))
    parser.add_argument("--aux-dir", type=Path, default=Path("outputs/nyu_rgb_aux"))
    parser.add_argument("--flip-dir", type=Path, default=Path("outputs/nyu_rgb_flip"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/part3_geocalib"))
    parser.add_argument("--oracles-csv", type=Path, default=Path("results/part3_geocalib/diag_oracles.csv"))
    parser.add_argument("--signal-r2-csv", type=Path, default=Path("results/part3_geocalib/diag_signal_r2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/part3_geocalib"))
    parser.add_argument("--figures-root", type=Path, default=Path("results/figures"))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not args.manifest_only:
        generate_all_figures(
            args.data_root, args.orig_preds, args.aux_dir, args.flip_dir,
            args.results_dir, args.oracles_csv, args.signal_r2_csv,
            args.output_dir, args.figures_root,
        )
    if args.manifest_out is not None or args.manifest_only:
        target = args.manifest_out if args.manifest_out is not None else Path("results/manifest.json")
        build_project_manifest(repo_root, target)


if __name__ == "__main__":
    main()
