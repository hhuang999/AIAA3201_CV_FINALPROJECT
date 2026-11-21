"""CVPR publication-grade 300-DPI figures and SHA-256 manifest builder for GeoCalib-3D v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import nyu_depth, nyu_image, open_nyu
from .metrics import CROP, valid_depth
from .sota_calibration import apply_sota_model, load_frozen_sota
from .sota_figures import _configure_style, _draw_arrow, _draw_box
from .sota_io import file_sha256, write_json_lf
from .sota_v2_cache import load_v2_cache
from .sota_v2_calibration import apply_sota_v2_model, assemble_semantic_vector, load_frozen_sota_v2


SHORT_LABELS_V2 = {
    "raw": "Raw",
    "b3_residual_v1": "B3_prev",
    "v1_geocalib3d": "v1 GeoCalib",
    "D3_qknots_robust": "D3 QKnots",
    "N2_hybrid_log_inv_res": "N2 +InvRes",
    "S4_sem_all_r005": "S4 SemScale",
    "H6_geosem_inv_spline": "v2 (1-Pass)",
    "M1_v1_flip_sym_zero_retrain": "M1 v1+SymFlip",
    "M2_flip_geo_gated": "M2 FlipGeo",
    "M6_flip_inv_geosem": "v2+Flip (2-Pass)",
}

PALETTE_V2 = {
    "raw": "#6C757D",
    "b3_residual_v1": "#D95F02",
    "v1_geocalib3d": "#4E79A7",
    "D3_qknots_robust": "#A6761D",
    "N2_hybrid_log_inv_res": "#66A61E",
    "S4_sem_all_r005": "#1B9E77",
    "H6_geosem_inv_spline": "#0072B2",
    "M1_v1_flip_sym_zero_retrain": "#9467BD",
    "M2_flip_geo_gated": "#8C564B",
    "M6_flip_inv_geosem": "#6A1B9A",
}


def plot_fig1_v2_architecture_and_oracles(
    out_path: Path,
    oracles_df: pd.DataFrame,
    signal_r2_df: pd.DataFrame,
    oof_sum_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    runtime: dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(15.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.0], wspace=0.22)

    # Left: Dual-Track Architecture Schematic
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0.0, 13.4)
    ax.set_ylim(0.0, 7.2)
    ax.axis("off")
    ax.set_title("(a) GeoCalib-3D v2 Dual-Track Architecture (Zero GT K; Tracks Strictly Separated)", pad=8)

    # Track A outer container
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.08, 2.22), 13.2, 4.78, boxstyle="round,pad=0.05,rounding_size=0.16", fc="#F7FBFE", ec="#0072B2", lw=1.2, ls="-"))
    ax.text(0.28, 6.75, "Track A: Single-Pass Main Track (1x ViT-S Forward, 43.0 ms Backbone + 4.6 ms GPU / 10.3 ms CPU Post)", fontsize=7.6, fontweight="bold", color="#005A8D")

    _draw_box(ax, 0.22, 3.85, 1.50, 1.45, "RGB Image\n(480x640)\nSingle Pass", "#F1F3F5", "#495057", fontsize=7.0, bold=True)
    _draw_box(ax, 1.98, 3.65, 2.25, 1.85, "Frozen UniDepthV2\nViT-S Backbone\n-> Z_hat, K_hat\n+ Internal Feats", "#E9ECEF", "#343A40", fontsize=7.0)
    _draw_arrow(ax, (1.74, 4.58), (1.96, 4.58))

    _draw_box(
        ax, 4.55, 4.88, 4.35, 1.55,
        "1. Inverse-Depth & Focal Geometry\nMonotone 1/Z_hat Spline + h_f(f_tilde)\n+ v2_clean Geo (drop cx_off) + Slope",
        "#E6F5F0", "#1B9E77", lw=1.2, fontsize=6.7,
    )
    _draw_box(
        ax, 4.55, 2.52, 4.35, 1.90,
        "2. Single-Pass Semantics (L2-Norm)\ncls (384) + patch_m/s (768) + df (512)\n-> Ridge Scale + 2D Field (v_tilde, r^2)\n(5018 raw -> 146.9 effective DoF)",
        "#E7F1FA", "#0072B2", lw=1.2, fontsize=6.6,
    )
    _draw_arrow(ax, (4.25, 4.95), (4.53, 5.60), "#1B9E77")
    _draw_arrow(ax, (4.25, 4.15), (4.53, 3.48), "#0072B2")

    h6_tst = test_sum_df.loc[test_sum_df["method"] == "H6_geosem_inv_spline"].iloc[0]
    _draw_box(
        ax, 9.28, 3.42, 3.82, 2.38,
        f"GeoCalib-3D v2 (H6)\nSingle-Pass Winner\nTest AbsRel: {h6_tst['abs_rel']:.4f}\nTest RMSE: {h6_tst['rmse']:.4f} m\nTest delta1: {h6_tst['delta1']:.4f}",
        "#D0E6FA", "#005A8D", lw=1.6, fontsize=7.0, bold=True,
    )
    _draw_arrow(ax, (8.92, 5.60), (9.26, 4.95), "#1B9E77")
    _draw_arrow(ax, (8.92, 3.48), (9.26, 4.25), "#0072B2")

    # Track B dashed optional container
    ax.add_patch(FancyBboxPatch((0.08, 0.14), 13.2, 1.82, boxstyle="round,pad=0.05,rounding_size=0.16", fc="#FAF6FC", ec="#6A1B9A", lw=1.2, ls="--"))
    ax.text(0.28, 1.72, "Track B: Optional Multi-Pass Flip-TTA Extension (2x ViT-S Forward, 87.2 ms Backbone + 6.7 ms GPU Post)", fontsize=7.4, fontweight="bold", color="#6A1B9A")

    _draw_box(ax, 0.22, 0.30, 2.25, 1.22, "Horiz. Flip RGB\n-> 2nd ViT-S Pass\n(+44.1 ms)", "#F3E5F5", "#6A1B9A", fontsize=6.7)
    _draw_box(ax, 2.82, 0.30, 6.12, 1.22, "Un-Mirrored K2 + Learnable Blend w_flip\n+ Flip-Consistency (|f1-f2|, std d_log) + cls_sym (150.9 DoF)", "#F3E5F5", "#6A1B9A", fontsize=6.4)
    m6_tst = test_sum_df.loc[test_sum_df["method"] == "M6_flip_inv_geosem"].iloc[0]
    _draw_box(ax, 9.28, 0.28, 3.82, 1.26, f"v2 + Flip-TTA (M6)\nTest: {m6_tst['abs_rel']:.4f} / {m6_tst['rmse']:.4f} / {m6_tst['delta1']:.4f}", "#EBD8F5", "#6A1B9A", lw=1.4, fontsize=6.8, bold=True)
    _draw_arrow(ax, (2.49, 0.91), (2.80, 0.91), "#6A1B9A")
    _draw_arrow(ax, (8.96, 0.91), (9.26, 0.91), "#6A1B9A")

    # Right column: (b) Post-v1 Oracles Waterfall + (c) 5-fold CV R^2 Signal Predictability
    gs_r = gs[0, 1].subgridspec(2, 1, hspace=0.52)

    ax_w = fig.add_subplot(gs_r[0, 0])
    oof_t = oof_sum_df.set_index("method")
    ora_t = oracles_df.set_index("method")
    wf_labels = ["Raw", "B3_prev", "v1\nGeoCalib", "v2 (H6\n1-Pass)", "v2+Flip\n(M6 2x)", "v1 Scalar\nOracle", "v1 Spatial\nOracle"]
    wf_vals = [
        float(oof_t.loc["raw", "abs_rel"]),
        float(oof_t.loc["b3_residual_v1", "abs_rel"]),
        float(oof_t.loc["v1_geocalib3d", "abs_rel"]),
        float(oof_t.loc["H6_geosem_inv_spline", "abs_rel"]),
        float(oof_t.loc["M6_flip_inv_geosem", "abs_rel"]),
        float(ora_t.loc["v1_oracle_absrel_scalar", "abs_rel"]),
        float(ora_t.loc["v1_oracle_spatial_linear", "abs_rel"]),
    ]
    wf_colors = ["#6C757D", "#D95F02", "#4E79A7", "#0072B2", "#6A1B9A", "#ADB5BD", "#CED4DA"]
    bars = ax_w.bar(np.arange(len(wf_labels)), wf_vals, width=0.62, color=wf_colors, edgecolor="#333333", linewidth=0.5)
    for b, val in zip(bars, wf_vals):
        ax_w.text(b.get_x() + 0.5 * b.get_width(), val + 0.0022, f"{val:.4f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax_w.set_xticks(np.arange(len(wf_labels)))
    ax_w.set_xticklabels(wf_labels, fontsize=7.3)
    ax_w.set_ylim(0.0, 0.135)
    ax_w.set_ylabel("5-Fold Train OOF AbsRel")
    ax_w.set_title("(b) Post-v1 Residual Gap & Oracle Ceiling (N=795 Train)")
    ax_w.grid(True, axis="y", linestyle=":", alpha=0.45)

    ax_r2 = fig.add_subplot(gs_r[1, 0])
    blocks = ["v1_geo_sp3d_control", "confidence_stats", "flip_consistency", "depth_feat_stats", "cls_token", "patch_mean_std"]
    blk_short = ["v1 Geo\nControl", "Conf.\nStats", "Flip\nConsist.", "Depth\nFeats", "CLS\nToken", "Patch\nMean/Std"]
    best_r2 = signal_r2_df.sort_values("cv_r2_e_v1_med", ascending=False).groupby("feature_block").first()
    r2_med = [float(best_r2.loc[b, "cv_r2_e_v1_med"]) for b in blocks]
    r2_rad = [float(best_r2.loc[b, "cv_r2_e_v1_radial"]) for b in blocks]
    xb = np.arange(len(blocks))
    ax_r2.axhline(0.0, color="#343A40", lw=0.9, ls="--")
    b1 = ax_r2.bar(xb - 0.17, r2_med, width=0.33, color="#0072B2", label="Scale Residual e_v1_med")
    b2 = ax_r2.bar(xb + 0.17, r2_rad, width=0.33, color="#1B9E77", label="Radial Residual e_v1_rad")
    for bar, v in list(zip(b1, r2_med)) + list(zip(b2, r2_rad)):
        if v > 0.01:
            ax_r2.text(bar.get_x() + 0.5 * bar.get_width(), v + 0.0015, f"{v:+.3f}", ha="center", va="bottom", fontsize=6.8)
    ax_r2.set_xticks(xb)
    ax_r2.set_xticklabels(blk_short, fontsize=7.4)
    ax_r2.set_ylim(-0.016, 0.068)
    ax_r2.set_ylabel("5-Fold Scene-CV R^2")
    ax_r2.set_title("(c) Zero-Leakage Signal R^2 Predicting Post-v1 Residual")
    ax_r2.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax_r2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.3)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig2_v2_ablation_and_ci(
    out_path: Path,
    oof_sum_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    test_boot_df: pd.DataFrame,
) -> None:
    order = test_sum_df["method"].tolist()
    oof_t = oof_sum_df.set_index("method").reindex(order)
    test_t = test_sum_df.set_index("method").reindex(order)
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2), gridspec_kw={"hspace": 0.56, "wspace": 0.28})

    metric_specs = [
        ("abs_rel", "AbsRel (lower better)", "(a) Dual-Track Ablation: AbsRel", False),
        ("rmse", "RMSE (m, lower better)", "(b) Dual-Track Ablation: RMSE", False),
        ("delta1", "delta1 Acc. (higher better)", "(c) Dual-Track Ablation: delta1", True),
    ]
    for col, (m_key, ylabel, title, higher_b) in enumerate(metric_specs):
        ax = axes[0, col]
        yo = oof_t[m_key].to_numpy(dtype=np.float64)
        yt = test_t[m_key].to_numpy(dtype=np.float64)
        ax.axvspan(-0.5, 2.5, color="#F8F9FA", alpha=0.9, zorder=0)
        ax.axvspan(2.5, 6.5, color="#EEF6FC", alpha=0.6, zorder=0)
        ax.axvspan(6.5, 9.5, color="#F5EFFA", alpha=0.6, zorder=0)
        ax.axvline(2.5, color="#ADB5BD", ls="--", lw=0.9)
        ax.axvline(6.5, color="#9467BD", ls="--", lw=0.9)

        ax.plot(x, yo, "--o", color="#495057", mfc="white", lw=1.4, ms=5.0, label="5-Fold Train OOF (N=795)")
        ax.plot(x, yt, "-D", color="#0072B2", lw=2.0, ms=5.5, label="Held-out Test (N=654)")

        ymin, ymax = min(yo.min(), yt.min()), max(yo.max(), yt.max())
        span = ymax - ymin
        ax.set_ylim(ymin - 0.22 * span, ymax + 0.30 * span)

        top_y = ymax + 0.23 * span
        ax.text(1.0, top_y, "Baselines", ha="center", va="center", fontsize=7.2, color="#495057", fontweight="bold")
        ax.text(4.5, top_y, "Track A: 1-Pass", ha="center", va="center", fontsize=7.2, color="#005A8D", fontweight="bold")
        ax.text(8.0, top_y, "Track B: 2-Pass", ha="center", va="center", fontsize=7.2, color="#6A1B9A", fontweight="bold")

        # Annotate v1 and v2 (H6) cleanly without colliding with curves or axes
        v1_v = float(yt[2])
        h6_v = float(yt[6])
        if not higher_b:
            ax.annotate(
                f"v1: {v1_v:.4f}",
                xy=(2, v1_v),
                xytext=(2.1, v1_v - 0.14 * span),
                fontsize=7.2, color="#4E79A7", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#4E79A7", "lw": 0.75},
            )
            ax.annotate(
                f"v2: {h6_v:.4f}",
                xy=(6, h6_v),
                xytext=(4.9, h6_v - 0.15 * span),
                fontsize=7.4, color="#005A8D", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.8},
            )
        else:
            ax.annotate(
                f"v1: {v1_v:.4f}",
                xy=(2, v1_v),
                xytext=(0.7, v1_v + 0.11 * span),
                fontsize=7.2, color="#4E79A7", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#4E79A7", "lw": 0.75},
            )
            ax.annotate(
                f"v2: {h6_v:.4f}",
                xy=(6, h6_v),
                xytext=(4.6, h6_v + 0.10 * span),
                fontsize=7.4, color="#005A8D", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.8},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS_V2[m] for m in order], fontsize=7.3, rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        if col == 0:
            ax.legend(loc="center right", frameon=True, framealpha=0.92, fontsize=7.3)

    # Bottom row: Held-out Test 10k-Boot 95% CI vs GeoCalib-3D v1
    cands = [
        "D3_qknots_robust",
        "N2_hybrid_log_inv_res",
        "S4_sem_all_r005",
        "H6_geosem_inv_spline",
        "M1_v1_flip_sym_zero_retrain",
        "M2_flip_geo_gated",
        "M6_flip_inv_geosem",
    ]
    forest_specs = [
        ("abs_rel", "Test AbsRel Gain vs v1", "(d) Test 10k-Boot 95% CI: AbsRel Gain vs v1"),
        ("rmse", "Test RMSE Gain vs v1 (m)", "(e) Test 10k-Boot 95% CI: RMSE Gain vs v1"),
        ("delta1", "Test delta1 Gain vs v1", "(f) Test 10k-Boot 95% CI: delta1 Gain vs v1"),
    ]
    yp = np.arange(len(cands))[::-1]
    for col, (m_key, xlabel, title) in enumerate(forest_specs):
        ax = axes[1, col]
        ax.axvline(0.0, color="#6C757D", ls="--", lw=1.0)
        max_hi = 0.0
        min_lo = 0.0
        for row_i, cm in enumerate(cands):
            r = test_boot_df[
                (test_boot_df["reference"] == "v1_geocalib3d")
                & (test_boot_df["candidate"] == cm)
                & (test_boot_df["metric"] == m_key)
            ].iloc[0]
            y_val = yp[row_i]
            col_c = PALETTE_V2[cm]
            is_win = cm in ("H6_geosem_inv_spline", "M6_flip_inv_geosem")
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
                va="center", ha="left", fontsize=6.9,
                color=col_c if is_win else "#212529",
                fontweight="bold" if is_win else "normal",
            )
        ax.set_yticks(yp)
        ax.set_yticklabels([SHORT_LABELS_V2[c] for c in cands] if col == 0 else [""] * len(cands), fontsize=8.0)
        span_x = max_hi - min_lo
        ax.set_xlim(min_lo - 0.10 * span_x, max_hi + 1.08 * span_x)
        ax.set_ylim(-0.6, len(cands) - 0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig3_v2_mechanism_and_left_tail(
    out_path: Path,
    diag_df: pd.DataFrame,
    spatial_data: dict[str, Any],
    audit_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(15.4, 9.4))
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.38)

    gs_top = gs_outer[0].subgridspec(1, 2, wspace=0.22)
    ft = diag_df["f_tilde"].to_numpy(dtype=np.float64)
    ev1 = diag_df["median_log_err_v1"].to_numpy(dtype=np.float64)
    ev2 = diag_df["median_log_err_v2_sp"].to_numpy(dtype=np.float64)

    q_edges = np.quantile(ft, np.linspace(0.0, 1.0, 9))
    q_edges[0] -= 1e-6
    q_edges[-1] += 1e-6

    audit_ids_fail = (839, 832, 326)
    audit_ids_gain = (1150, 1151)

    for col_i, (err_arr, col_hex, title, is_v2) in enumerate(
        (
            (ev1, "#4E79A7", "(a) Before v2 (GeoCalib-3D v1): False Focal Alarms Over-Scaled", False),
            (ev2, "#0072B2", "(b) After v2 Single-Pass (H6): Semantic Gating Suppresses Alarms", True),
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

        # Highlight Round 1 audit cases
        sub_f = diag_df[diag_df["index"].isin(audit_ids_fail)]
        sub_g = diag_df[diag_df["index"].isin(audit_ids_gain)]
        col_y = "median_log_err_v2_sp" if is_v2 else "median_log_err_v1"
        ax.scatter(sub_f["f_tilde"], sub_f[col_y], s=58, marker="X", color="#C92A2A", zorder=6, label="R1 False Alarms (#839/#832/#326)")
        ax.scatter(sub_g["f_tilde"], sub_g[col_y], s=54, marker="*", color="#1B9E77", zorder=6, label="R1 True Collapse (#1150/#1151)")

        # Annotate #839 and #1151 cleanly
        r839 = diag_df[diag_df["index"] == 839].iloc[0]
        ax.annotate(
            f"#839 ({'sem=-0.098' if is_v2 else 'v1 over +26%'})",
            xy=(r839["f_tilde"], r839[col_y]),
            xytext=(0.43, r839[col_y] + (0.095 if is_v2 else 0.06)),
            fontsize=7.3, color="#C92A2A", fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#C92A2A", "lw": 0.8},
        )
        r1151 = diag_df[diag_df["index"] == 1151].iloc[0]
        ax.annotate(
            f"#1151 ({'sem=+0.097' if is_v2 else 'v1'})",
            xy=(r1151["f_tilde"], r1151[col_y]),
            xytext=(0.56, r1151[col_y] - 0.14),
            fontsize=7.3, color="#0B724E", fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#1B9E77", "lw": 0.8},
        )

        r_val = float(np.corrcoef(ft, err_arr)[0, 1])
        sd_val = float(np.std(err_arr))
        mad_val = float(np.mean(np.abs(err_arr)))
        lt34 = diag_df[diag_df["f_tilde"] < 0.55]
        lt_ar = float(lt34["abs_rel_v2_sp" if is_v2 else "abs_rel_v1"].mean())
        stat_txt = f"r = {r_val:+.4f}, SD = {sd_val:.4f}\nMAE(log) = {mad_val:.4f}\nf<0.55 (n=34) AbsRel = {lt_ar:.4f}"
        ax.text(0.60, 0.27, stat_txt, fontsize=7.6, bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#ADB5BD", "alpha": 0.94})

        ax.set_xlim(0.32, 1.38)
        ax.set_ylim(-0.46, 0.42)
        ax.set_xlabel("Predicted Normalized Focal Length  f_tilde = 0.5*(fx + fy) / 480")
        ax.set_ylabel("Median Log Scale Error  log(Z_pred / Z_gt)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=6.9)

    # Row 2: Spatial residual 12x16 heatmaps (v1 vs v2 Single-Pass H6 vs v2 Flip-TTA M6)
    gs_bot = gs_outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.16)
    all_maps = spatial_data["maps"]["all"]
    heat_specs = [
        ("v1_geocalib3d", "(c) Spatial Bias: GeoCalib-3D v1"),
        ("H6_geosem_inv_spline", "(d) Spatial Bias: v2 Single-Pass (H6)"),
        ("M6_flip_inv_geosem", "(e) Spatial Bias: v2 + Flip-TTA (M6)"),
    ]
    mats = [np.asarray(all_maps[m]["mean_log_residual_12x16"], dtype=np.float64) for m, _ in heat_specs]
    vlim = 0.036

    im_last = None
    for idx_m, ((m_key, htitle), mat) in enumerate(zip(heat_specs, mats)):
        ax = fig.add_subplot(gs_bot[0, idx_m])
        im_last = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto", extent=[41, 601, 471, 45])
        rms_b = all_maps[m_key]["rms_spatial_log_bias"]
        max_b = all_maps[m_key]["max_abs_spatial_log_bias"]
        ax.set_title(f"{htitle}\nRMS bias={rms_b:.4f}, Max|bias|={max_b:.4f}", fontsize=9.4)
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


def plot_fig4_v2_bins_lc_pareto(
    out_path: Path,
    bins_df: pd.DataFrame,
    lc_df: pd.DataFrame,
    test_sum_df: pd.DataFrame,
    runtime: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), gridspec_kw={"hspace": 0.42, "wspace": 0.24})

    # Top row: Depth Bins on Held-Out Test (N=654)
    bin_methods = ["raw", "b3_residual_v1", "v1_geocalib3d", "H6_geosem_inv_spline", "M6_flip_inv_geosem"]
    bin_labels = {
        "raw": "Raw",
        "b3_residual_v1": "B3_prev",
        "v1_geocalib3d": "v1",
        "H6_geosem_inv_spline": "v2 (1-Pass)",
        "M6_flip_inv_geosem": "v2+Flip (2x)",
    }
    raw_bins = bins_df[bins_df["method"] == "raw"].sort_values("lower_m")
    lowers = raw_bins["lower_m"].tolist()
    uppers = raw_bins["upper_m"].tolist()
    imgs_cnt = raw_bins["images"].tolist()
    x_labels = [f"[{lo:g}, {hi:g})m\n(n={int(ni)})" for lo, hi, ni in zip(lowers, uppers, imgs_cnt)]
    x = np.arange(len(lowers))
    bar_w = 0.16

    for col_i, (m_key, ylabel, title, fmt_t) in enumerate(
        (
            ("abs_rel", "AbsRel (lower better)", "(a) Held-Out Test AbsRel by Depth Bin", "rel"),
            ("delta1", "delta1 Accuracy (higher better)", "(b) Held-Out Test delta1 by Depth Bin", "pp"),
        )
    ):
        ax = axes[0, col_i]
        v1_vals = bins_df[bins_df["method"] == "v1_geocalib3d"].sort_values("lower_m")[m_key].to_numpy(dtype=np.float64)
        h6_vals = bins_df[bins_df["method"] == "H6_geosem_inv_spline"].sort_values("lower_m")[m_key].to_numpy(dtype=np.float64)
        for j, bm in enumerate(bin_methods):
            sub = bins_df[bins_df["method"] == bm].sort_values("lower_m")
            vals = sub[m_key].to_numpy(dtype=np.float64)
            offset = (j - 2.0) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, color=PALETTE_V2[bm], label=bin_labels[bm], edgecolor="white", lw=0.4)
            if bm == "H6_geosem_inv_spline":
                for bi, bar in enumerate(bars):
                    top = max(float(bar.get_height()), float(v1_vals[bi]))
                    lbl = (f"{100*(h6_vals[bi]-v1_vals[bi])/v1_vals[bi]:+.1f}%" if fmt_t == "rel" else f"{100*(h6_vals[bi]-v1_vals[bi]):+.1f}pp")
                    ax.text(bar.get_x() + 0.5 * bar.get_width(), top + (0.008 if m_key == "abs_rel" else 0.011), lbl, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#005A8D")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        if m_key == "delta1":
            ax.set_ylim(0.45, 1.13)
            ax.legend(loc="upper center", ncol=5, frameon=True, framealpha=0.94, fontsize=6.9)
        else:
            _, ymax = ax.get_ylim()
            ax.set_ylim(0.0, ymax * 1.15)
            ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92, fontsize=7.2)

    # Bottom Left (c): Train-only Learning Curve
    ax_lc = axes[1, 0]
    lc_methods = ["b3_residual_v1", "v1_geocalib3d", "H6_geosem_inv_spline", "M6_flip_inv_geosem"]
    sub_b3 = lc_df[lc_df["method"] == "b3_residual_v1"].sort_values("fraction")
    xt_labels = [f"{int(round(f*100))}%\n(~{int(round(sc))} sc.)" for f, sc in zip(sub_b3["fraction"], sub_b3["train_scenes_mean"])]
    x_lc = np.array([10.0, 25.0, 50.0, 100.0])

    for lm in lc_methods:
        sub = lc_df[lc_df["method"] == lm].sort_values("fraction")
        yv = sub["abs_rel"].to_numpy(dtype=np.float64)
        ylo = sub["ci95_lo_abs_rel"].to_numpy(dtype=np.float64)
        yhi = sub["ci95_hi_abs_rel"].to_numpy(dtype=np.float64)
        ax_lc.plot(x_lc, yv, "-D" if lm == "H6_geosem_inv_spline" else "-o", color=PALETTE_V2[lm], lw=2.1 if lm == "H6_geosem_inv_spline" else 1.5, ms=5.2, label=SHORT_LABELS_V2[lm])
        if lm in ("v1_geocalib3d", "H6_geosem_inv_spline"):
            ax_lc.fill_between(x_lc, ylo, yhi, color=PALETTE_V2[lm], alpha=0.14)

    v2_25 = float(lc_df[(lc_df["method"] == "H6_geosem_inv_spline") & (lc_df["fraction"] == 0.25)]["abs_rel"].iloc[0])
    v1_100 = float(lc_df[(lc_df["method"] == "v1_geocalib3d") & (lc_df["fraction"] == 1.00)]["abs_rel"].iloc[0])
    ax_lc.annotate(
        f"v2@25% ({v2_25:.4f})\nbeats v1@100% ({v1_100:.4f})",
        xy=(25.0, v2_25), xytext=(42.0, v2_25 + 0.0045),
        arrowprops={"arrowstyle": "->", "color": "#005A8D", "lw": 0.9},
        fontsize=7.6, color="#005A8D", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#0072B2", "alpha": 0.90},
    )
    ax_lc.set_xticks(x_lc)
    ax_lc.set_xticklabels(xt_labels, fontsize=7.8)
    ax_lc.set_xlabel("Fraction of Train Fitting Scenes per CV Fold (of 249)")
    ax_lc.set_ylabel("5-Fold OOF AbsRel (lower better)")
    ax_lc.set_title("(c) Sample-Efficiency Learning Curve (Train Only)")
    ax_lc.grid(True, linestyle=":", alpha=0.45)
    ax_lc.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.4)

    # Bottom Right (d): Accuracy-Latency Pareto Frontier
    ax_p = axes[1, 1]
    t_sum = test_sum_df.set_index("method")
    bb1 = float(runtime["backbone_single_pass_ms_per_image"])
    bb2 = float(runtime["backbone_two_pass_flip_ms_per_image"])

    pts_cpu = [
        ("Raw (1x, CPU-post)", bb1, float(t_sum.loc["raw", "abs_rel"]), PALETTE_V2["raw"], "o"),
        ("B3_prev (1x, CPU-post)", bb1 + float(runtime["b3_residual_v1_post_ms"]), float(t_sum.loc["b3_residual_v1", "abs_rel"]), PALETTE_V2["b3_residual_v1"], "s"),
        ("v1 GeoCalib (1x, CPU-post)", float(runtime["end_to_end_v1_ms"]), float(t_sum.loc["v1_geocalib3d", "abs_rel"]), PALETTE_V2["v1_geocalib3d"], "^"),
        ("v2 H6 (1x, CPU-post)", float(runtime["end_to_end_v2_single_pass_cpu_post_ms"]), float(t_sum.loc["H6_geosem_inv_spline", "abs_rel"]), PALETTE_V2["H6_geosem_inv_spline"], "D"),
        ("v2+Flip M6 (2x, CPU-post)", float(runtime["end_to_end_v2_flip_tta_cpu_post_ms"]), float(t_sum.loc["M6_flip_inv_geosem", "abs_rel"]), PALETTE_V2["M6_flip_inv_geosem"], "P"),
    ]
    pts_gpu = [
        ("v2 H6 (1x, GPU-post)", float(runtime["end_to_end_v2_single_pass_gpu_ms"]), float(t_sum.loc["H6_geosem_inv_spline", "abs_rel"]), "#005A8D"),
        ("v2+Flip M6 (2x, GPU-post)", float(runtime["end_to_end_v2_flip_tta_gpu_ms"]), float(t_sum.loc["M6_flip_inv_geosem", "abs_rel"]), "#6A1B9A"),
    ]

    # Pareto curve through Raw -> v1 -> v2_SP (CPU-post) -> note M6 is Pareto-dominated on Test AbsRel!
    px_par = [pts_cpu[0][1], float(runtime["end_to_end_v1_ms"]), float(runtime["end_to_end_v2_single_pass_cpu_post_ms"])]
    py_par = [pts_cpu[0][2], pts_cpu[2][2], pts_cpu[3][2]]
    ax_p.plot(px_par, py_par, "--", color="#0072B2", lw=1.3, alpha=0.7, label="Single-Pass Pareto Frontier")

    for lbl, lx, ly, col_c, mk in pts_cpu:
        ax_p.scatter([lx], [ly], color=col_c, marker=mk, s=75 if "H6" in lbl else 58, zorder=5, edgecolor="white", lw=0.7)
        if "B3" in lbl:
            xytext_p = (lx - 1.5, ly + 0.0028)
        elif "v1" in lbl:
            xytext_p = (55.5, 0.0928)
        elif "H6" in lbl:
            xytext_p = (59.0, 0.0860)
        elif "M6" in lbl:
            xytext_p = (89.5, 0.0890)
        else:
            xytext_p = (lx + 2.2, ly + 0.0012)
        ax_p.annotate(
            f"{lbl}\n({lx:.1f}ms, {ly:.4f})",
            xy=(lx, ly), xytext=xytext_p,
            arrowprops={"arrowstyle": "->", "color": col_c, "lw": 0.75} if ("v1" in lbl or "H6" in lbl or "M6" in lbl) else None,
            fontsize=6.9, color=col_c, fontweight="bold" if "H6" in lbl else "normal",
        )

    for lbl, lx, ly, col_c in pts_gpu:
        ax_p.scatter([lx], [ly], color=col_c, marker="*", s=125, zorder=6, label=f"{lbl}: {lx:.1f}ms")

    ax_p.set_xlim(38.0, 120.0)
    ax_p.set_ylim(0.079, 0.119)
    ax_p.set_xlabel("End-to-End Single-Image Latency (ms/img, RTX 4060 Backbone + Post)")
    ax_p.set_ylabel("Held-Out Test AbsRel (lower better)")
    ax_p.set_title("(d) Accuracy-Latency Pareto Frontier (Single-Pass Dominates 2-Pass)")
    ax_p.grid(True, linestyle=":", alpha=0.45)
    ax_p.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.1)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def plot_fig5_v2_round1_failures_and_qualitative(
    out_path: Path,
    cases_spec: list[tuple[int, str, bool]],
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    frozen_v1_path: Path,
    frozen_v2_path: Path,
    audit_df: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> None:
    cache = load_v2_cache(orig_preds, aux_dir, flip_dir, data_root, allowed_split="test", compute_hashes=False, verify_scenes=False)
    v1_frozen = load_frozen_sota(frozen_v1_path)
    v2_frozen = load_frozen_sota_v2(frozen_v2_path)
    win_sp = v2_frozen["primary_single_pass_method"]
    win_mp = v2_frozen["primary_multipass_method"]
    m_v1 = v1_frozen["primary_model"]
    m_sp = v2_frozen["models"][win_sp]
    m_mp = v2_frozen["models"][win_mp]
    sm_sp = m_sp["semantic_branch"]["sem_mode"]
    sm_mp = m_mp["semantic_branch"]["sem_mode"]

    fig = plt.figure(figsize=(15.8, 12.8))
    gs = fig.add_gridspec(len(cases_spec), 6, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.05], wspace=0.09, hspace=0.40)

    col_titles = ("Input RGB (Crop)", "GT Depth (0.5-6m)", "v1 GeoCalib |Error|", "v2 Single-Pass H6 |Error|", "v2 + Flip-TTA M6 |Error|")
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
            p2 = cache.depth_flip(idx).astype(np.float64)
            c1 = np.maximum(cache.confidence(idx).astype(np.float64), 1e-4)
            c2 = np.maximum(cache.confidence_flip(idx).astype(np.float64), 1e-4)
            k1 = cache.intrinsics(idx).astype(np.float64)
            k2 = cache.intrinsics_flip(idx).astype(np.float64)
            cs = cache.confidence_summary(idx)
            cls1 = cache.cls_token(idx)
            cls2 = cache.cls_token_flip(idx)
            id_arr = np.array([idx], dtype=np.int64)
            pm = cache.patch_mean_for(id_arr)[0]
            ps = cache.patch_std_for(id_arr)[0]
            df = cache.depth_feat_stats_for(id_arr)[0]

            xs = assemble_semantic_vector(sm_sp, cls1, pm, ps, df)
            xm = assemble_semantic_vector(sm_mp, cls1, pm, ps, df, cls2)

            pred_v1 = apply_sota_model(p1, k1, m_v1, use_torch_fast=True)[CROP]
            pred_sp = apply_sota_v2_model(p1, k1, m_sp, conf_summary=cs, cached_sem=xs, use_torch_fast=True)[CROP]
            pred_mp = apply_sota_v2_model(
                p1, k1, m_mp, flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2,
                conf_summary=cs, cls_orig=cls1, cls_flip=cls2, cached_sem=xm, use_torch_fast=True,
            )[CROP]

            gt_vis = np.ma.masked_where(~valid, gt)
            err_v1 = np.ma.masked_where(~valid, np.abs(pred_v1 - gt))
            err_sp = np.ma.masked_where(~valid, np.abs(pred_sp - gt))
            err_mp = np.ma.masked_where(~valid, np.abs(pred_mp - gt))

            drow = diag_map.loc[idx]
            ar_v1 = float(np.mean(np.abs(pred_v1[valid] - gt[valid]) / gt[valid]))
            d1_v1 = float(np.mean(np.maximum(pred_v1[valid] / gt[valid], gt[valid] / pred_v1[valid]) < 1.25))
            ar_sp = float(np.mean(np.abs(pred_sp[valid] - gt[valid]) / gt[valid]))
            d1_sp = float(np.mean(np.maximum(pred_sp[valid] / gt[valid], gt[valid] / pred_sp[valid]) < 1.25))
            ar_mp = float(np.mean(np.abs(pred_mp[valid] - gt[valid]) / gt[valid]))
            d1_mp = float(np.mean(np.maximum(pred_mp[valid] / gt[valid], gt[valid] / pred_mp[valid]) < 1.25))

            panels = [
                ("rgb", rgb, f"{role_label}\nidx={idx}, f~={drow['f_tilde']:.3f}"),
                ("gt", gt_vis, f"Scene: {drow['scene']}\nsem_shift={drow['shift_v2_sp_sem']:+.3f}"),
                ("err", err_v1, f"v1: AR={ar_v1:.3f}, d1={d1_v1:.3f}"),
                ("err", err_sp, f"v2(1x): AR={ar_sp:.3f}, d1={d1_sp:.3f}"),
                ("err", err_mp, f"v2(2x): AR={ar_mp:.3f}, d1={d1_mp:.3f}"),
            ]

            for c_i, (ptype, arr, stitle) in enumerate(panels):
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
                    if is_reg:
                        sp.set_edgecolor("#C92A2A")
                        sp.set_linewidth(1.8 if c_i >= 3 else 1.1)
                    else:
                        sp.set_edgecolor("#0072B2" if c_i == 3 else "#ADB5BD")
                        sp.set_linewidth(1.4 if c_i == 3 else 0.8)

                full_t = f"{col_titles[c_i]}\n{stitle}" if r_i == 0 else stitle
                tcol = "#C92A2A" if (is_reg and c_i in (0, 3, 4)) else ("#005A8D" if (not is_reg and c_i == 3) else "#1A1A1A")
                ax.set_title(full_t, fontsize=8.0, color=tcol, fontweight="bold" if c_i in (0, 3) else "normal", pad=4)

    cax = fig.add_subplot(gs[:, 5])
    cb = fig.colorbar(im_err_ref, cax=cax, extend="max")
    cb.set_label("Absolute Depth Error |Z_pred - Z_gt| (m) [Shared Scale]", fontsize=9.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def build_part3b_sota_v2_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Build complete SHA-256 manifest for Round 2 verifiable via `python -m depthlab.verify`."""
    repo_root = Path(repo_root).resolve()
    manifest_rel = Path(manifest_path).resolve().relative_to(repo_root).as_posix()

    candidates: list[Path] = []
    for rel in (
        "depthlab/sota_v2_cache.py",
        "depthlab/sota_v2_diagnose.py",
        "depthlab/sota_v2_calibration.py",
        "depthlab/sota_v2_evaluate.py",
        "depthlab/sota_v2_figures.py",
        "tests/test_part3b_sota_v2.py",
        "docs/PART3B_SOTA_V2.md",
    ):
        p = repo_root / rel
        if p.is_file():
            candidates.append(p)

    res_dir = repo_root / "results" / "part3b_sota_v2"
    for p in sorted(list(res_dir.glob("*.csv")) + list(res_dir.glob("*.json"))):
        if p.resolve().relative_to(repo_root).as_posix() != manifest_rel:
            candidates.append(p)

    fig_dir = repo_root / "results" / "figures" / "part3b_sota_v2"
    for p in sorted(fig_dir.glob("*.png")):
        candidates.append(p)

    entries = [
        {"path": p.resolve().relative_to(repo_root).as_posix(), "bytes": int(p.stat().st_size), "sha256": file_sha256(p)}
        for p in candidates
    ]
    payload = {"schema": "part3b_sota_v2_manifest", "file_count": len(entries), "files": entries}
    write_json_lf(manifest_path, payload)
    print(f"Round 2 Manifest written: {manifest_path} ({len(entries)} verified files)", flush=True)
    return payload


def generate_all_v2_figures(
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    results_dir: Path,
    frozen_v1_path: Path,
    oracles_csv: Path,
    signal_r2_csv: Path,
    output_dir: Path,
) -> list[Path]:
    _configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    oracles_df = pd.read_csv(oracles_csv)
    signal_r2_df = pd.read_csv(signal_r2_csv)
    oof_sum_df = pd.read_csv(results_dir / "oof_ablation_summary.csv")
    test_sum_df = pd.read_csv(results_dir / "test_summary.csv")
    test_boot_df = pd.read_csv(results_dir / "test_paired_bootstrap.csv")
    diag_df = pd.read_csv(results_dir / "test_mechanism_diagnostics.csv")
    audit_df = pd.read_csv(results_dir / "test_round1_case_audit.csv")
    bins_df = pd.read_csv(results_dir / "test_depth_bins.csv")
    lc_df = pd.read_csv(results_dir / "oof_learning_curve.csv")
    runtime = json.loads((results_dir / "runtime_overhead.json").read_text(encoding="utf-8"))
    spatial_data = json.loads((results_dir / "test_spatial_residuals.json").read_text(encoding="utf-8"))

    p1 = output_dir / "fig1_v2_architecture_and_oracles.png"
    p2 = output_dir / "fig2_v2_ablation_and_ci.png"
    p3 = output_dir / "fig3_v2_mechanism_and_left_tail.png"
    p4 = output_dir / "fig4_v2_depth_bins_learning_curve_pareto.png"
    p5 = output_dir / "fig5_v2_round1_failures_and_qualitative.png"

    plot_fig1_v2_architecture_and_oracles(p1, oracles_df, signal_r2_df, oof_sum_df, test_sum_df, runtime)
    plot_fig2_v2_ablation_and_ci(p2, oof_sum_df, test_sum_df, test_boot_df)
    plot_fig3_v2_mechanism_and_left_tail(p3, diag_df, spatial_data, audit_df)
    plot_fig4_v2_bins_lc_pareto(p4, bins_df, lc_df, test_sum_df, runtime)

    cases_spec = [
        (839, "R1 Failure #1 Fixed", False),
        (832, "R1 Failure #2 Mitigated", False),
        (326, "R1 Failure #3 Mitigated", False),
        (1151, "R1 High-Gain Boosted", False),
        (167, "New v2 Worst Reg.", True),
    ]
    plot_fig5_v2_round1_failures_and_qualitative(
        p5, cases_spec, data_root, orig_preds, aux_dir, flip_dir,
        frozen_v1_path, results_dir / "frozen_sota_v2_model.json", audit_df, diag_df,
    )

    for p in (p1, p2, p3, p4, p5):
        print(f"Generated {p} ({p.stat().st_size} bytes)", flush=True)
    return [p1, p2, p3, p4, p5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--orig-preds", type=Path, default=Path("outputs/nyu_rgb"))
    parser.add_argument("--aux-dir", type=Path, default=Path("outputs/nyu_rgb_aux"))
    parser.add_argument("--flip-dir", type=Path, default=Path("outputs/nyu_rgb_flip"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/part3b_sota_v2"))
    parser.add_argument("--frozen-v1", type=Path, default=Path("results/part3b_sota/frozen_sota_model.json"))
    parser.add_argument("--oracles-csv", type=Path, default=Path("../process_docs/part3b_sota_notes/round2/diag_v2_oracles.csv"))
    parser.add_argument("--signal-r2-csv", type=Path, default=Path("../process_docs/part3b_sota_notes/round2/diag_v2_signal_r2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/part3b_sota_v2"))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not args.manifest_only:
        generate_all_v2_figures(
            args.data_root, args.orig_preds, args.aux_dir, args.flip_dir,
            args.results_dir, args.frozen_v1, args.oracles_csv, args.signal_r2_csv, args.output_dir,
        )
    if args.manifest_out is not None or args.manifest_only:
        target = args.manifest_out if args.manifest_out is not None else (args.results_dir / "manifest.json")
        build_part3b_sota_v2_manifest(repo_root, target)


if __name__ == "__main__":
    main()
