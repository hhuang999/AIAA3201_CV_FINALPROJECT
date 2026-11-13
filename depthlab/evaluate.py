import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .calibration import BOOTSTRAP_SEED, apply_curve, apply_residual, log_features
from .data import nyu_depth, open_nyu, scene_names, splits
from .cache import load_predictions, sha256
from .frozen import load_frozen, frozen_predictions
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, clip_depth, evaluate_image, metrics, valid_depth


def corrected_predictions(prediction, parameters):
    if parameters.get("schema") == "B_residual_v1":
        return frozen_predictions(prediction, parameters)
    corrected = {"raw": prediction,
                 "global_scale": clip_depth(parameters["global_scale"] * prediction),
                 "monotone_curve": apply_curve(prediction, parameters["monotone_curve"])}
    if parameters["residual"] is not None:
        corrected["residual_scale"] = apply_residual(corrected["monotone_curve"],
                                                    log_features(prediction), parameters["residual"])
    return corrected


def analyze_pairs(frame, output, prefix="test"):
    methods = list(frame.method.unique())
    tables = {name: frame[frame.method == name].set_index("index").sort_index() for name in methods}
    if frame.duplicated(["method", "index"]).any():
        raise ValueError("Duplicate method/image rows")
    if not np.isfinite(frame[list(METRIC_NAMES)].values).all():
        raise ValueError("Nonfinite metrics cannot be aggregated")
    counts, intervals, cases = [], [], []
    pairs = list(zip(methods, methods[1:]))
    if "residual_scale" in methods:
        pairs.extend((name, "residual_scale") for name in ("raw", "global_scale"))
    for previous, current in pairs:
        before, after = tables[previous], tables[current]
        if not before.index.equals(after.index):
            raise ValueError("Paired methods must have identical image IDs")
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
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(group_count), size=(10000, len(group_count)))
        bootstrap = group_sum[draws].sum(axis=1) / group_count[draws].sum(axis=1)[:, None]
        for col, metric in enumerate(METRIC_NAMES):
            low, high = np.percentile(bootstrap[:, col], [2.5, 97.5])
            intervals.append({"previous": previous, "current": current, "metric": metric,
                              "gain": gain[:, col].mean(), "ci_low": low, "ci_high": high,
                              "scenes": len(group_count), "replicates": len(draws)})
        score = (gain / np.maximum(before[list(METRIC_NAMES)].mean().values, 1e-12)).mean(axis=1)
        for kind, row in (("largest_gain", np.argmax(score)), ("largest_regression", np.argmin(score))):
            cases.append({"previous": previous, "current": current, "kind": kind,
                          "index": int(before.index[row]), "score": float(score[row]),
                          "scene": before.scene.iloc[row]})
    pd.DataFrame(counts).to_csv(output / f"{prefix}_change_counts.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / f"{prefix}_paired_intervals.csv", index=False)
    pd.DataFrame(cases).to_csv(output / f"{prefix}_cases.csv", index=False)


def _compare_reference(frame, reference, parameters):
    path = Path(reference)
    if path.is_dir():
        path = path / "per_image.csv"
    expected = pd.read_csv(path).rename(columns={"dataset_index": "index", "d1": "delta1"})
    aliases = {"median": "global_scale", "spline_rmse": "monotone_curve",
               parameters.get("primary", "residual"): "residual_scale"}
    expected["method"] = expected.method.replace(aliases)
    expected = expected[expected["index"].isin(frame["index"].unique())]
    keys = ["method", "index"]
    expected = expected.set_index(keys).sort_index()
    observed = frame.set_index(keys).sort_index()
    if not expected.index.equals(observed.index):
        raise ValueError("Reference method/image IDs do not match the replay")
    error = np.abs(expected[list(METRIC_NAMES)].values - observed[list(METRIC_NAMES)].values)
    if not np.isfinite(error).all() or np.max(error) > 1e-12:
        raise ValueError(f"Frozen per-image metrics differ: maximum absolute error {np.max(error)}")
    return {"reference_sha256": sha256(path), "rows": len(frame),
            "max_absolute_metric_error": float(np.max(error)), "tolerance": 1e-12}


def evaluate(args):
    split = getattr(args, "split", "test")
    limit = getattr(args, "limit", None)
    if limit is not None and (limit <= 0 or split != "train"):
        raise ValueError("Limited smoke evaluation is supported on training images only")
    parameters_digest = sha256(args.parameters)
    expected_digest = getattr(args, "expected_model_sha256", None)
    if expected_digest is not None and parameters_digest != expected_digest:
        raise ValueError("Calibration parameter SHA-256 differs from the expected digest")
    parameters = json.loads(args.parameters.read_text())
    frozen = parameters.get("schema") == "B_residual_v1"
    if frozen:
        parameters = load_frozen(args.parameters, expected_sha256=getattr(args, "expected_model_sha256", None))
    elif not parameters.get("complete") and not getattr(args, "allow_legacy_parameters", False):
        raise ValueError("Use completed train-only parameters or explicitly allow historical parameters")
    all_ids = splits(args.data_root)
    ids = all_ids[split][:limit]
    if frozen and not np.array_equal(np.asarray(parameters["train_ids"]), all_ids["train"]):
        raise ValueError("Frozen training IDs differ from the official split")
    if parameters.get("schema") == "depthlab_calibration_v2":
        if parameters.get("complete") is not True or parameters.get("test_read") is not False:
            raise ValueError("Completed calibration parameters must retain train-only provenance")
        if parameters.get("training_ids") != all_ids["train"].tolist():
            raise ValueError("Calibration training IDs differ from the official split")
        if parameters.get("training_images") != len(all_ids["train"]):
            raise ValueError("Calibration training image count differs from the official split")
        if any(key not in parameters for key in ("dataset_sha256", "split_sha256")):
            raise ValueError("Calibration parameters are missing data/split binding")
    data_hashes = {"nyu_mat_sha256": sha256(args.data_root / "nyu_depth_v2_labeled.mat"),
                   "splits_sha256": sha256(args.data_root / "splits.mat")}
    for key, actual in [("dataset_sha256", data_hashes["nyu_mat_sha256"]),
                        ("split_sha256", data_hashes["splits_sha256"])]:
        if key in parameters and parameters[key] != actual:
            raise ValueError(f"Calibration {key} does not match the data")
    predictions = load_predictions(args.predictions, required_ids=ids,
        expected_known_intrinsics=False,
        require_complete=not getattr(args, "allow_legacy_cache", False))
    metadata = predictions.metadata
    signature = metadata.get("signature", {})
    if "signature" in metadata and signature.get("data") != data_hashes:
        raise ValueError("Prediction cache is bound to different data or split")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evaluation output must be new or empty; preserve existing evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(Path(__file__).parent.parent)): sha256(p)
                     for p in [Path(__file__).with_name(name) for name in ("evaluate.py", "metrics.py", "frozen.py", "calibration.py", "cache.py", "data.py")]}
    identity = {"complete": False, "split": split, "ids": ids.tolist(),
        "parameters_sha256": parameters_digest, "data": data_hashes,
        "prediction_files": {p.name: sha256(p) for p in [args.predictions / "depth.npy",
                              args.predictions / "indices.npy"]},
        "source_sha256": source_hashes, "frozen_replay": frozen,
        "scope": "Fixed-parameter reproduction; no fit or model selection"}
    (args.output / "evaluation.json").write_text(json.dumps(identity, indent=2))
    reference_depth = getattr(args, "reference_depth", None)
    reference_array = np.load(reference_depth, mmap_mode="r", allow_pickle=False) if reference_depth else None
    if reference_array is not None and (split != "test" or reference_array.shape != (len(ids), 480, 640)):
        raise ValueError("Frozen reference depth must match the complete test split")
    changed_pixels, max_depth_error = 0, 0.
    records, binned, timing = [], [], []
    with open_nyu(args.data_root) as handle:
        scenes = scene_names(handle)
        for row, index in enumerate(ids):
            prediction = np.asarray(predictions[index], dtype=np.float64)
            target = nyu_depth(handle, index)
            start = perf_counter()
            outputs = corrected_predictions(prediction, parameters)
            timing.append({"index": int(index), "all_corrections_seconds": perf_counter() - start})
            if reference_array is not None:
                depth = outputs["residual_scale"]
                if not np.isfinite(reference_array[row]).all():
                    raise ValueError("Nonfinite frozen reference depth")
                difference = np.abs(depth - reference_array[row])
                if not np.isfinite(difference).all():
                    raise ValueError("Nonfinite frozen depth comparison")
                changed_pixels += int(np.count_nonzero(difference))
                max_depth_error = max(max_depth_error, float(np.max(difference)))
            gt = target[CROP]
            valid = valid_depth(gt)
            for name, depth in outputs.items():
                records.append({"index": int(index), "scene": scenes[index], "method": name,
                                **dict(zip(METRIC_NAMES, evaluate_image(depth, target)))})
                for lower, upper in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                    mask = valid & (gt >= lower) & (gt < upper)
                    if mask.any():
                        binned.append({"index": int(index), "method": name, "lower_m": lower,
                                       "upper_m": upper, "pixels": int(mask.sum()),
                                       **dict(zip(METRIC_NAMES, metrics(depth[CROP], gt, mask)))})
            if row % 100 == 0 or row == len(ids) - 1:
                print("Frozen replay" if frozen else "Evaluation", split, row + 1, "/", len(ids), flush=True)
    frame = pd.DataFrame(records)
    if not np.isfinite(frame[list(METRIC_NAMES)].values).all():
        raise ValueError("Nonfinite scores; no result can be declared complete")
    reference = getattr(args, "reference", None)
    if reference:
        identity["reference_metrics"] = _compare_reference(frame, reference, parameters)
    if reference_array is not None:
        identity["reference_depth"] = {"sha256": sha256(reference_depth),
            "changed_pixels": changed_pixels, "max_absolute_error": max_depth_error}
        if max_depth_error > 1e-12:
            raise ValueError(f"Frozen output depth differs by {max_depth_error}")
    frame.to_csv(args.output / f"{split}_per_image.csv", index=False)
    frame.groupby("method", sort=False)[list(METRIC_NAMES)].mean().to_csv(args.output / f"{split}_summary.csv")
    bins = pd.DataFrame(binned)
    bins.to_csv(args.output / f"{split}_bins_per_image.csv", index=False)
    bins.groupby(["method", "lower_m", "upper_m"])[list(METRIC_NAMES)].mean().to_csv(args.output / f"{split}_bins_summary.csv")
    bins.groupby(["method", "lower_m", "upper_m"]).agg(images=("index", "count"), pixels=("pixels", "sum")).to_csv(args.output / f"{split}_bins_counts.csv")
    pd.DataFrame(timing).to_csv(args.output / "postprocessing_per_image.csv", index=False)
    analyze_pairs(frame, args.output, prefix=split)
    identity.update(complete=True, images=len(ids))
    (args.output / "evaluation.json").write_text(json.dumps(identity, indent=2))
    print(frame.groupby("method", sort=False)[list(METRIC_NAMES)].mean(), flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--reference", type=Path, help="Saved per-image results for a fixed reproduction check")
    parser.add_argument("--reference-depth", type=Path, help="Optional original frozen output array")
    parser.add_argument("--allow-legacy-cache", action="store_true", help="Explicitly permit historical unbound caches")
    parser.add_argument("--allow-legacy-parameters", action="store_true")
    evaluate(parser.parse_args())
