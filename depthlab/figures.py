import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .calibration import apply_curve
from .cache import load_predictions
from .data import nyu_depth, nyu_image, open_nyu, splits
from .evaluate import corrected_predictions
from .metrics import CROP, METRIC_NAMES, valid_depth

LABELS = {"raw": "Raw RGB-only", "global_scale": "Global scalar",
          "monotone_curve": "Monotone curve", "residual_scale": "Residual scale"}
COLORS = ["#657786", "#d19c41", "#377e95", "#a44968"]


def pipeline(path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    def box(x, y, text, color):
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 1.0, boxstyle="round,pad=.12",
                                   facecolor=color, edgecolor="#8b939c", linewidth=.8))
        ax.text(x + 1.35, y + .5, text, ha="center", va="center", fontsize=10)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#59636e"})
    ax.text(.2, 7.6, "Stereo geometry, metric calibration, and video consistency", fontsize=17, weight="bold")
    rows = [(5.7, ["15 Middlebury pairs\nquarter-resolution calibration", "Bidirectional SGBM\n3-way / padding / HH4", "Disparity and metric depth\nMAE / coverage"], "#eef1f3"),
            (3.5, ["NYU RGB, 1449 images\no camera input", "UniDepthV2 ViT-S\nsaved raw depth + intrinsics", "795 training images\nscene-grouped calibration"], "#e8f0f3"),
            (1.0, ["70 video frames at 24 FPS\nshared 640 x 360 RGB", "Framewise and video models\nrelative depth only", "Fixed clip-wide display ranges\ncomparison + background blur"], "#f1edf2")]
    for y, texts, color in rows:
        for x, text in zip((.3, 4.0, 7.7), texts):
            box(x, y, text, color)
        arrow(3.15, y + .5, 3.85, y + .5)
        arrow(6.85, y + .5, 7.55, y + .5)
    box(11.0, 3.5, "654 held-out images\nraw vs frozen calibration", "#eaf0e7")
    arrow(10.55, 4.0, 10.85, 4.0)
    ax.text(7.7, 3.1, "Fit parameters committed before test evaluation", fontsize=9, color="#466043")
    ax.text(.2, .2, "Known-camera diagnosis is separate and uses training images only.", fontsize=10, color="#59636e")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def comparison(handle, raw, index, parameters, previous, current, title, path):
    rgb = nyu_image(handle, index)[CROP]
    gt = nyu_depth(handle, index)[CROP]
    outputs = corrected_predictions(raw[index].astype(np.float64), parameters)
    before, after = outputs[previous][CROP], outputs[current][CROP]
    valid = valid_depth(gt)
    errors = [np.where(valid, np.abs(np.clip(depth, .001, 10) - gt), np.nan)
              for depth in (before, after)]
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.8), layout="constrained")
    axes[0].imshow(rgb)
    for ax, depth in zip(axes[1:4], (gt, before, after)):
        im = ax.imshow(depth, cmap="viridis", vmin=0, vmax=10)
    for ax, error in zip(axes[4:], errors):
        err = ax.imshow(error, cmap="magma", vmin=0, vmax=1.5)
    for ax, label in zip(axes, ("RGB", "Filled target (m)", LABELS[previous], LABELS[current],
                               "Previous error (m)", "Current error (m)")):
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.colorbar(im, ax=list(axes[1:4]), shrink=.65)
    fig.colorbar(err, ax=list(axes[4:]), shrink=.65)
    fig.suptitle(f"{title} | NYU index {index} (zero-based), evaluation crop", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figures(args):
    output = args.results / "figures/nyu"
    output.mkdir(parents=True, exist_ok=True)
    params = json.loads((args.results / "calibration_parameters.json").read_text())
    summary = pd.read_csv(args.results / "test_summary.csv", index_col="method")
    methods = list(summary.index)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    for ax, metric, label in zip(axes, METRIC_NAMES, ["AbsRel (lower)", "RMSE in meters (lower)", "Delta1 (higher)"]):
        ax.bar(np.arange(len(methods)), summary[metric], color=COLORS[:len(methods)])
        ax.set_xticks(np.arange(len(methods)), [LABELS[m] for m in methods], rotation=28, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "test_scores.png", dpi=180)
    plt.close(fig)
    x = np.geomspace(.1, 15, 500)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.plot(x, x, "--", color=COLORS[0], label="Identity")
    ax.plot(x, params["global_scale"] * x, color=COLORS[1], label="Global scalar")
    ax.plot(x, apply_curve(x, params["monotone_curve"]), color=COLORS[2], label="Monotone curve")
    ax.set(xscale="log", yscale="log", xlabel="Raw predicted depth (m)", ylabel="Calibrated depth (m)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "calibration_curve.png", dpi=180)
    plt.close(fig)
    bins = pd.read_csv(args.results / "test_bins_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric in zip(axes, METRIC_NAMES):
        for color, method in zip(COLORS, methods):
            values = bins[bins.method == method].sort_values("lower_m")
            ax.plot(np.arange(len(values)), values[metric], "o-", color=color, label=LABELS[method])
        ax.set_xticks(np.arange(5), ["0-1", "1-2", "2-4", "4-6", "6-10"])
        ax.set(xlabel="Target depth range (m)", title=metric)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output / "depth_ranges.png", dpi=180)
    plt.close(fig)
    oof = pd.read_csv(args.results / "calibration_oof_summary.csv", index_col="method")
    cv_methods = ["raw", "global_scale", "monotone_curve"]
    if params["residual"] is not None:
        cv_methods.append(params["selected_method"])
    gains = []
    for method, cv_method in zip(methods[1:], cv_methods[1:]):
        for metric in METRIC_NAMES:
            direction = -1 if metric != "delta1" else 1
            gains.append({"method": method, "metric": metric,
                          "training_oof_gain": direction * (oof.loc[cv_method, metric] - oof.loc["raw", metric]),
                          "test_gain": direction * (summary.loc[method, metric] - summary.loc["raw", metric])})
    pd.DataFrame(gains).to_csv(args.results / "generalization_gains.csv", index=False)
    raw = load_predictions(args.predictions, required_ids=splits(args.data_root)["test"], expected_known_intrinsics=False, require_complete=True)
    with open_nyu(args.data_root) as handle:
        ids = splits(args.data_root)["test"]
        for i in (0, len(ids) // 2, len(ids) - 1):
            comparison(handle, raw, ids[i], params, "raw", methods[-1], "Representative test image",
                       output / f"representative_{ids[i]:04d}.png")
        cases = pd.read_csv(args.results / "test_cases.csv")
        for case in cases.itertuples():
            comparison(handle, raw, case.index, params, case.previous, case.current,
                       case.kind.replace("_", " "), output / f"{case.current}_{case.kind}_{case.index:04d}.png")
    diagnostic = pd.read_csv(args.results / "camera_diagnostic_per_image.csv")
    stats = json.loads((args.results / "camera_diagnostic.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    axes[0].hist(diagnostic.focal_ratio, bins=35, color=COLORS[2], alpha=.8)
    axes[0].axvline(1, color="#444444", linestyle="--", label="Known focal length")
    axes[0].axvline(stats["inverse_rgb_scale"], color=COLORS[3], linestyle=":", label="Inverse global scale")
    axes[0].set(xlabel="Estimated / known focal length", ylabel="Training images")
    axes[0].legend(fontsize=8)
    axes[1].scatter(diagnostic.focal_ratio, diagnostic.raw_over_target, s=9, alpha=.4, color=COLORS[2])
    axes[1].set(xlabel="Estimated / known focal length", ylabel="Median raw / target depth",
                title=f"Training only: Pearson r={stats['pearson_r']:.3f}, Spearman r={stats['spearman_r']:.3f}")
    fig.savefig(output / "camera_diagnostic.png", dpi=180)
    plt.close(fig)
    pipeline(args.results / "figures/pipeline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    figures(parser.parse_args())
