"""Descriptive training bias and previously saved-score uncertainty only.

This module never fits/selects a model, reads test GT, runs inference, or changes
the frozen primary. Every invocation uses a separate fresh evidence directory.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .cache import atomic_json, load_predictions, sha256
from .calibration_study import fresh_output, write_csv
from .data import nyu_depth, open_nyu, scene_names, splits
from .frozen import load_frozen
from .metrics import CROP, valid_depth

METRICS = ("abs_rel", "rmse", "d1")
EDGES = (.001, 1., 2., 4., 6., 10.)
QUANTILES = (0, .1, .25, .5, .75, .9, .95, 1)
QNAMES = ("min", "p10", "p25", "median", "p75", "p90", "p95", "max")

def describe(values):
    values = np.asarray(values,dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite scalar observations')
    return dict(mean=float(values.mean()),std=float(values.std()),
                **dict(zip(QNAMES,np.quantile(values,QUANTILES).tolist())))

def signed_bias_rows(prediction, truth, scale):
    p, g = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Expected positive finite paired valid training pixels")
    if not np.isfinite(g).all() or not valid_depth(g).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training GT pixels or frozen scale")
    rows = []
    for name, factor in (("raw", 1.), ("median", scale)):
        calibrated = np.clip(factor * p, .001, 10.)
        groups = [("all", np.ones(len(g), dtype=bool))] + [
            (f"{lo:g}-{hi:g}", (g >= lo) & (g < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        for depth_bin, within in groups:
            if not within.any():
                continue
            pp, gg = calibrated[within], g[within]
            rows.append({"method": name, "depth_bin": depth_bin, "pixels": int(within.sum()),
                         "signed_relative_error": float(np.mean((pp - gg) / gg)),
                         "signed_log_error": float(np.mean(np.log(pp / gg))),
                         "overprediction_fraction": float(np.mean(pp > gg)),
                         "median_gt_over_prediction": float(np.median(gg / pp))})
    return rows


def training_bias(data_root, predictions, frozen, output, smoke=False, reference_ratios=None):
    root, folder = Path(data_root), Path(predictions)
    ids = splits(root)["train"]
    if ids.tolist() != frozen["train_ids"]:
        raise ValueError("Frozen training identities differ from official training split")
    for filename, key in (("nyu_depth_v2_labeled.mat", "dataset_sha256"), ("splits.mat", "split_sha256")):
        if sha256(root / filename) != frozen[key]:
            raise ValueError("Frozen training provenance mismatch: " + key)
    cache = load_predictions(folder, required_ids=ids, expected_known_intrinsics=False, require_complete=True)
    scale = frozen["controls"]["median"]["scale"]
    if reference_ratios is not None:
        reference_ratios = np.asarray(reference_ratios, dtype=np.float64)
        if reference_ratios.shape != (795,) or not np.isfinite(reference_ratios).all():
            raise ValueError("Expected 795 finite reference training ratios")
    ratios, rows = [], []
    maximum = 0.
    with open_nyu(root) as handle:
        scenes = scene_names(handle)
        types = np.array(["".join(chr(int(c)) for c in handle[ref][:].ravel())
                          for ref in handle["sceneTypes"][:].ravel()])
        for pos, index in enumerate(ids[:3] if smoke else ids):
            raw = np.asarray(cache[int(index)], dtype=np.float64)[CROP]
            gt = nyu_depth(handle, int(index))[CROP].astype(np.float64)
            mask = valid_depth(gt)
            p, g = raw[mask], gt[mask]
            ratio = float(np.median(g / p))
            if reference_ratios is not None:
                maximum = max(maximum, abs(ratio - reference_ratios[pos]))
            ratios.append({"dataset_index": int(index), "scene_id": str(scenes[index]),
                           "scene_type": str(types[index]), "median_gt_over_raw": ratio})
            rows.extend({"dataset_index": int(index), "scene_id": str(scenes[index]), **row}
                        for row in signed_bias_rows(p, g, scale))
    if reference_ratios is not None and maximum > 1e-12:
        raise ValueError("Reference per-image training ratios differ")
    median_difference = abs(float(np.median([row["median_gt_over_raw"] for row in ratios])) - scale)
    if not smoke and median_difference > 1e-12:
        raise ValueError("Computed training median differs from the frozen control")
    summaries = []
    for method in ("raw", "median"):
        for depth_bin in ["all"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]:
            selected = [row for row in rows if row["method"] == method and row["depth_bin"] == depth_bin]
            if not selected:
                continue
            summaries.append({"method": method, "depth_bin": depth_bin, "images": len(selected),
                              "pixels": sum(row["pixels"] for row in selected),
                              **{key: float(np.mean([row[key] for row in selected])) for key in
                                 ("signed_relative_error", "signed_log_error", "overprediction_fraction")},
                              "median_of_image_median_gt_over_prediction": float(np.median([row["median_gt_over_prediction"] for row in selected]))})
    scenes = [{"scene_id": scene, "images": sum(row["scene_id"] == scene for row in ratios),
               **describe([row["median_gt_over_raw"] for row in ratios if row["scene_id"] == scene])}
              for scene in sorted({row["scene_id"] for row in ratios})]
    write_csv(Path(output) / "train_bias_per_image_bin.csv", rows)
    write_csv(Path(output) / "train_bias_by_depth.csv", summaries)
    write_csv(Path(output) / "train_per_image_scale_ratio.csv", ratios)
    write_csv(Path(output) / "train_scene_scale_ratio.csv", scenes)
    scene_medians = np.array([row["median"] for row in scenes])
    return {"images": len(ratios), "scenes": len(scenes), "frozen_median_scale": scale,
            "reference_ratio_max_difference": maximum if reference_ratios is not None else None,
            "full_training_median_difference": median_difference if not smoke else None,
            "scene_median_ratio_below_one": int((scene_medians < 1).sum()),
            "scene_median_ratio_above_one": int((scene_medians > 1).sum()),
            "per_image_ratio": describe([row["median_gt_over_raw"] for row in ratios]),
            "scope": "In-sample training-only diagnostics; no validation-gain or parameter-selection claim",
            "definitions": {"ratio": "Per-image median GT/raw before clipping; below one suggests image-level overestimation",
                            "signed_error": "Positive signed relative/log error means overprediction",
                            "aggregation": "Equal image means within each nonempty GT depth bin",
                            "warning": "A median ratio below one and a positive mean relative error summarize different aspects; neither means every pixel or scene is overpredicted"}}


def saved_metric_arrays(rows, expected_images=654):
    aliases = {"global_scale": "median", "monotone_curve": "spline_rmse"}
    grouped = {}
    for row in rows:
        name = aliases.get(row["method"], row["method"])
        index = int(row.get("dataset_index", row.get("index")))
        scene = row.get("scene_id", row.get("scene"))
        if not scene or index in grouped.setdefault(name, {}):
            raise ValueError("Missing scene or duplicate saved image")
        if "abs_rel" not in row or "rmse" not in row or ("d1" not in row and "delta1" not in row):
            raise ValueError("Missing saved metric column")
        values = [float(row["abs_rel"]), float(row["rmse"]), float(row.get("d1", row.get("delta1")))]
        grouped[name][index] = scene, values
    alternatives = [name for name in grouped if name not in ("raw", "median", "spline_rmse")]
    if len(alternatives) != 1 or set(grouped) != {"raw", "median", "spline_rmse", alternatives[0]}:
        raise ValueError("Expected exactly four already-frozen methods")
    primary = alternatives[0]
    names = ["raw", "median", "spline_rmse", primary]
    ids = sorted(grouped["raw"])
    if len(ids) != expected_images or any(set(grouped[name]) != set(ids) for name in names):
        raise ValueError("Expected matching complete saved image identities")
    if any(grouped[name][index][0] != grouped["raw"][index][0] for name in names for index in ids):
        raise ValueError("Saved scene identities differ")
    arrays = {name: np.array([grouped[name][index][1] for index in ids]) for name in names}
    if any(not np.isfinite(a).all() or np.any(a[:, :2] < 0) or np.any((a[:, 2] < 0) | (a[:, 2] > 1)) for a in arrays.values()):
        raise ValueError("Invalid saved metric values")
    return names, ids, np.array([grouped["raw"][index][0] for index in ids]), arrays


def summarize_saved(rows, expected_images=654, repeats=20000):
    names, ids, scenes, arrays = saved_metric_arrays(rows, expected_images)
    primary = names[-1]
    distributions, comparisons, intervals = [], [], {}
    for name in names:
        for col, metric in enumerate(METRICS):
            values = arrays[name][:, col]
            count = int(np.ceil(.1 * len(values)))
            worst = np.sort(values)[:count] if metric == "d1" else np.sort(values)[-count:]
            distributions.append({"method": name, "metric": metric, "images": len(ids), **describe(values),
                                  "worst_decile_mean": float(worst.mean()), "worst_direction": "low" if metric == "d1" else "high"})
    unique = np.unique(scenes)
    counts = np.array([(scenes == scene).sum() for scene in unique])
    rng = np.random.default_rng(20261111)
    # Bounded batches preserve the original fixed-model scene-bootstrap draws.
    pairs = [("median", "raw"), ("spline_rmse", "median"),
             (primary, "raw"), (primary, "median"), (primary, "spline_rmse")]
    gains = np.stack([arrays[reference] - arrays[candidate] for candidate, reference in pairs], axis=1)
    gains[:, :, 2] *= -1
    sums = np.stack([gains[scenes == scene].sum(0) for scene in unique])
    boot = np.empty((repeats, len(pairs), 3))
    for start in range(0, repeats, 100):
        draw = rng.integers(0, len(unique), (min(100, repeats - start), len(unique)))
        boot[start:start + len(draw)] = sums[draw].sum(1) / counts[draw].sum(1)[:, None, None]
    for pos, (candidate, reference) in enumerate(pairs):
        for col, metric in enumerate(METRICS):
            values = gains[:, pos, col]
            comparisons.append({"candidate": candidate, "reference": reference, "metric": metric, "images": len(ids),
                                "improved_images": int((values > 0).sum()), "worsened_images": int((values < 0).sum()),
                                "tied_images": int((values == 0).sum()), "improved_fraction": float(np.mean(values > 0)), **describe(values)})
        intervals[candidate + "_vs_" + reference] = {"gain": gains[:, pos].mean(0).tolist(),
                    "conditional_scene_95ci": np.percentile(boot[:, pos], [2.5, 97.5], axis=0).tolist(),
                    "one_sided_bonferroni_3_metrics_lower": np.percentile(boot[:, pos], 100 * .05 / 3, axis=0).tolist()}
    metadata = {"images": len(ids), "scenes": len(unique), "methods": names, "metrics": list(METRICS),
                "bootstrap_seed": 20261111, "bootstrap_repeats": repeats,
                "historical_test_exposure": True, "model_selection": False, "deployment_changed": False,
                "scope": "Previously saved scores only; no test GT reading, inference or rescoring",
                "uncertainty": "Fixed-model scene bootstrap, preserving image weighting. Three-metric adjustment is per comparison; five comparisons are descriptive, not a family-wide selection test. Does not include refitting uncertainty."}
    return distributions, comparisons, intervals, metadata


def plot_saved_cdf(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, ids, _, arrays = saved_metric_arrays(rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for col, (axis, metric) in enumerate(zip(axes, METRICS)):
        for name in names:
            axis.step(np.sort(arrays[name][:, col]), np.arange(1, len(ids) + 1) / len(ids), where="post", label=name)
        axis.set_xlabel(metric)
        axis.set_ylabel("Fraction of images at or below x")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Previously saved per-image distributions (descriptive)")
    figure.savefig(Path(output) / "saved_metric_cdf.png", dpi=180)
    figure.savefig(Path(output) / "saved_metric_cdf.pdf")
    plt.close(figure)


def run(args):
    output = fresh_output(args.output)
    if (args.data_root is None) != (args.predictions is None):
        raise ValueError("Training bias needs both --data-root and --predictions")
    if args.data_root is None and args.saved_scores is None:
        raise ValueError("Supply training data/predictions or already saved scores")
    if args.data_root is not None and args.frozen is None:
        raise ValueError("Training bias needs --frozen")
    protocol = {"scope": "Descriptive evidence only", "historical_test_exposure": True,
                "model_parameters_changed": False, "model_selection": False, "test_gt_read": False,
                "smoke": args.smoke_only, "bins": list(EDGES),
                "source_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                  ("calibration_analysis.py", "calibration_study.py", "frozen.py", "cache.py", "data.py", "metrics.py")}}
    if args.saved_scores:
        protocol["saved_scores_sha256"] = sha256(args.saved_scores)
    if args.frozen:
        protocol["frozen_sha256"] = sha256(args.frozen)
    if args.data_root:
        protocol["cache_manifest_sha256"] = sha256(Path(args.predictions) / "manifest.json")
    atomic_json(output / "protocol.json", protocol)
    summary = {"complete": True, "smoke": args.smoke_only, "protocol_sha256": sha256(output / "protocol.json"),
               "historical_test_exposure": True, "model_parameters_changed": False}
    if args.data_root:
        frozen = load_frozen(args.frozen, args.expected_frozen_sha256)
        reference = None
        if args.reference_ratios:
            reference = json.loads(Path(args.reference_ratios).read_text())["per_image_train_median_ratio"]
        summary["training_bias"] = training_bias(args.data_root, args.predictions, frozen, output, args.smoke_only, reference)
    if args.saved_scores:
        with Path(args.saved_scores).open() as stream:
            rows = list(csv.DictReader(stream))
        distributions, comparisons, intervals, metadata = summarize_saved(rows)
        write_csv(output / "saved_metric_distributions.csv", distributions)
        write_csv(output / "saved_paired_gain_distributions.csv", comparisons)
        atomic_json(output / "saved_uncertainty.json", {"metadata": metadata, "comparisons": intervals})
        summary["saved_distributions"] = metadata
        if args.plots:
            plot_saved_cdf(rows, output)
    atomic_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--expected-frozen-sha256")
    parser.add_argument("--reference-ratios", type=Path)
    parser.add_argument("--saved-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--plots", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
