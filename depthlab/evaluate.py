import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .calibration import SEED, apply_curve, apply_residual, log_features
from .data import nyu_depth, open_nyu, scene_names, splits
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, clip_depth, evaluate_image, metrics, valid_depth


def corrected_predictions(prediction, parameters):
    corrected = {"raw": prediction,
                 "global_scale": clip_depth(parameters["global_scale"] * prediction),
                 "monotone_curve": apply_curve(prediction, parameters["monotone_curve"])}
    if parameters["residual"] is not None:
        corrected["residual_scale"] = apply_residual(corrected["monotone_curve"],
                                                    log_features(prediction), parameters["residual"])
    return corrected


def analyze_pairs(frame, output):
    methods = list(frame.method.unique())
    tables = {name: frame[frame.method == name].set_index("index") for name in methods}
    counts, intervals, cases = [], [], []
    for previous, current in zip(methods, methods[1:]):
        before, after = tables[previous], tables[current]
        gain = (before[list(METRIC_NAMES)].values - after[list(METRIC_NAMES)].values) * [1, 1, -1]
        category = np.full(len(gain), "mixed_or_tied", dtype=object)
        category[(gain > 0).all(axis=1)] = "all_better"
        category[(gain < 0).all(axis=1)] = "all_worse"
        category[(gain == 0).all(axis=1)] = "unchanged"
        for name in ("all_better", "all_worse", "mixed_or_tied", "unchanged"):
            counts.append({"previous": previous, "current": current,
                           "category": name, "images": int((category == name).sum())})
        _, inverse = np.unique(before.scene, return_inverse=True)
        group_count = np.bincount(inverse)
        group_sum = np.stack([np.bincount(inverse, weights=gain[:, col]) for col in range(3)], axis=1)
        draws = np.random.default_rng(SEED).integers(0, len(group_count), size=(10000, len(group_count)))
        bootstrap = group_sum[draws].sum(axis=1) / group_count[draws].sum(axis=1)[:, None]
        for col, metric in enumerate(METRIC_NAMES):
            low, high = np.percentile(bootstrap[:, col], [2.5, 97.5])
            intervals.append({"previous": previous, "current": current, "metric": metric,
                              "gain": gain[:, col].mean(), "ci_low": low, "ci_high": high,
                              "scenes": len(group_count), "replicates": len(draws)})
        score = (gain / before[list(METRIC_NAMES)].mean().values).mean(axis=1)
        for kind, row in (("largest_gain", np.argmax(score)), ("largest_regression", np.argmin(score))):
            cases.append({"previous": previous, "current": current, "kind": kind,
                          "index": int(before.index[row]), "score": float(score[row]),
                          "scene": before.scene.iloc[row]})
    pd.DataFrame(counts).to_csv(output / "test_change_counts.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "test_paired_intervals.csv", index=False)
    pd.DataFrame(cases).to_csv(output / "test_cases.csv", index=False)


def evaluate(args):
    parameters = json.loads(args.parameters.read_text())
    predictions = np.load(args.predictions / "depth.npy", mmap_mode="r")
    indices = splits(args.data_root)["test"]
    args.output.mkdir(parents=True, exist_ok=True)
    handle = open_nyu(args.data_root)
    scenes = scene_names(handle)
    records, binned, timing = [], [], []
    for row, index in enumerate(indices):
        prediction = predictions[index].astype(np.float64)
        target = nyu_depth(handle, index)
        start = perf_counter()
        scaled = clip_depth(prediction * parameters["global_scale"])
        scalar_seconds = perf_counter() - start
        start = perf_counter()
        curved = apply_curve(prediction, parameters["monotone_curve"])
        curve_seconds = perf_counter() - start
        outputs = {"raw": prediction, "global_scale": scaled, "monotone_curve": curved}
        residual_seconds = 0.
        if parameters["residual"] is not None:
            start = perf_counter()
            outputs["residual_scale"] = apply_residual(curved, log_features(prediction), parameters["residual"])
            residual_seconds = perf_counter() - start
        timing.append({"index": index, "global_scale_seconds": scalar_seconds,
                       "monotone_curve_seconds": curve_seconds,
                       "residual_increment_seconds": residual_seconds})
        gt = target[CROP]
        valid = valid_depth(gt)
        for name, depth in outputs.items():
            records.append({"index": index, "scene": scenes[index], "method": name,
                            **dict(zip(METRIC_NAMES, evaluate_image(depth, target)))})
            for lower, upper in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                mask = valid & (gt >= lower) & (gt < upper)
                if mask.any():
                    binned.append({"index": index, "method": name, "lower_m": lower,
                                   "upper_m": upper, "pixels": mask.sum(),
                                   **dict(zip(METRIC_NAMES, metrics(depth[CROP], gt, mask)))})
        if row % 100 == 0:
            print("Test evaluation", row + 1, "/", len(indices), flush=True)
    handle.close()
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "test_per_image.csv", index=False)
    frame.groupby("method", sort=False)[list(METRIC_NAMES)].mean().to_csv(args.output / "test_summary.csv")
    bins = pd.DataFrame(binned)
    bins.to_csv(args.output / "test_bins_per_image.csv", index=False)
    bins.groupby(["method", "lower_m", "upper_m"])[list(METRIC_NAMES)].mean().to_csv(args.output / "test_bins_summary.csv")
    bins.groupby(["method", "lower_m", "upper_m"]).agg(images=("index", "count"), pixels=("pixels", "sum")).to_csv(args.output / "test_bins_counts.csv")
    pd.DataFrame(timing).to_csv(args.output / "postprocessing_per_image.csv", index=False)
    analyze_pairs(frame, args.output)
    print(frame.groupby("method", sort=False)[list(METRIC_NAMES)].mean(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=Path("results/calibration_parameters.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    evaluate(parser.parse_args())
