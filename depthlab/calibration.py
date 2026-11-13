"""Training-only calibration; explicit refits are separate from frozen deployment."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .data import open_nyu, nyu_depth, scene_names, splits
from .frozen import apply_spline, depth_features
from .metrics import CROP, METRIC_NAMES, clip_depth, evaluate_image, valid_depth

SEED = 20261031
SAMPLE_SEED = 20261105
BOOTSTRAP_SEED = 20261111
SAMPLES_PER_IMAGE = 4096
KNOTS = np.log([0.3, 0.7, 1.5, 3, 6, 12])
RESIDUAL_LIMIT = .15


def candidate_specs():
    candidates = []
    for target in ("median_ratio", "rmse_ratio"):
        candidates.append(dict(name=f"constant_{target}", kind="constant", target=target,
                               shrinkage=1., max_log_scale=RESIDUAL_LIMIT))
        for lam in (.1, 1., 10.):
            for shrinkage in (.5, 1.):
                candidates.append(dict(name=f"residual_{target}_l{lam:g}_a{shrinkage:g}",
                                       kind="ridge", target=target, lam=lam,
                                       shrinkage=shrinkage, max_log_scale=RESIDUAL_LIMIT))
    return candidates


CANDIDATES = tuple(candidate_specs())


def fit_scale(ratios):
    ratios = np.asarray(ratios, dtype=np.float64)
    if not ratios.size or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("Expected finite positive training ratios")
    return float(np.median(ratios))


def log_features(depth):
    return depth_features(depth)


def curve_basis(depth):
    x = np.log(np.asarray(depth, dtype=np.float64))
    return np.stack([np.minimum(x - KNOTS[0], KNOTS[1] - KNOTS[0]),
                     *[np.clip(x - KNOTS[k], 0, KNOTS[k + 1] - KNOTS[k])
                       for k in range(1, 4)], np.maximum(x - KNOTS[4], 0)], axis=-1)


def apply_curve(depth, curve):
    if "x_knots" in curve:
        return apply_spline(depth, curve)
    # Read the rebuilt architecture's earlier offset/slope parameter artifacts.
    p = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Raw depth must be finite and positive")
    return clip_depth(np.exp(KNOTS[0] + curve["offset"] +
                             curve_basis(p) @ np.array(curve["slopes"])))


def fit_curve(samples, targets, scale):
    """Original spline_rmse CPU fit protocol, including operation order."""
    p = torch.tensor(np.asarray(samples), dtype=torch.float64)
    target = torch.tensor(np.asarray(targets), dtype=torch.float64)
    if (p.ndim != 2 or target.shape != p.shape or not bool(torch.isfinite(p).all()) or
            not bool(torch.isfinite(target).all()) or bool((p <= 0).any()) or
            bool((target <= 0).any()) or not .5 < scale < 2.):
        raise ValueError("Expected positive paired image samples and .5 < initial scale < 2")
    knots = torch.tensor(KNOTS, dtype=torch.float64)
    raw = torch.tensor([np.arctanh(np.log(scale) / np.log(2)), *([np.log(3. / 4.)] * 5)],
                       dtype=torch.float64, requires_grad=True)
    def loss_components(pred):
        # Retain the original stacked objective graph, including zero-weight
        # auxiliary terms, to preserve the historical optimizer arithmetic.
        relative = (pred - target) / target
        ar = (relative.square() + 1e-6).sqrt().sub(.001).mean()
        rmse = ((pred - target).square().mean(-1) + 1e-12).sqrt().mean()
        delta = torch.nn.functional.softplus(((pred.log() - target.log()).abs() - np.log(1.25)) / .02).mean() * .02
        return torch.stack([ar, rmse, delta])

    with torch.no_grad():
        reference = loss_components((p * scale).clamp(.001, 10.)).clamp_min(1e-9)
    optimizer = torch.optim.LBFGS([raw], max_iter=80, max_eval=120, history_size=20,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                  tolerance_change=1e-11)
    history = []

    def decoded():
        offset = np.log(2.) * raw[0].tanh()
        slopes = .25 + 1.75 * raw[1:].sigmoid()
        y = torch.cat([(knots[0] + offset).reshape(1),
                       knots[0] + offset + torch.cumsum(slopes * torch.diff(knots), dim=0)])
        return offset, y

    def closure():
        optimizer.zero_grad(set_to_none=True)
        _, y = decoded()
        lp = p.log()
        i = (torch.bucketize(lp.contiguous(), knots, right=True) - 1).clamp(0, len(knots) - 2)
        mapped = (y[i] + (lp - knots[i]) * (y[i + 1] - y[i]) / (knots[i + 1] - knots[i])).exp().clamp(.001, 10.)
        # The historical module decoded its regularizer separately from
        # forward; preserve that autograd accumulation order as well.
        _, regularization_y = decoded()
        slopes = torch.diff(regularization_y) / torch.diff(knots)
        terms = loss_components(mapped) / reference
        loss = terms[1] + (.01 * torch.diff(slopes).square().mean() + .001 * (slopes - 1).square().mean())
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError("Nonfinite calibration objective")
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    offset, y = decoded()
    return {"kind": "spline", "offset": float(offset.detach()),
            "slopes": (torch.diff(y) / torch.diff(knots)).detach().tolist(),
            "x_knots": KNOTS.tolist(), "y_knots": y.detach().tolist(),
            "iterations": int(optimizer.state[raw]["n_iter"]), "closure_calls": len(history),
            "objective_history": history}


def fit_residual(features, target, regularization, shrinkage):
    features, target = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 8 or target.shape != (len(features),) or
            not len(features) or not np.isfinite(features).all() or not np.isfinite(target).all() or
            not np.isfinite(regularization) or regularization <= 0 or
            not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("Expected finite N x 8 features, N targets and valid regularization/shrinkage")
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-8)
    x = (features - mean) / std
    intercept = target.mean()
    coefficient = np.linalg.solve(x.T @ x / len(x) + regularization * np.eye(8),
                                   x.T @ (target - intercept) / len(x))
    return {"kind": "ridge_residual", "mean": mean.tolist(), "std": std.tolist(),
            "intercept": float(intercept), "coefficient": coefficient.tolist(),
            "lambda": regularization, "shrinkage": shrinkage, "max_log_scale": RESIDUAL_LIMIT}


def fit_constant_residual(target):
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or not target.size or not np.isfinite(target).all():
        raise ValueError("Expected finite nonempty residual targets")
    return {"kind": "constant_residual", "intercept": float(target.mean()),
            "shrinkage": 1., "max_log_scale": RESIDUAL_LIMIT}


def residual_log_scale(features, model):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim not in (1, 2) or features.shape[-1] != 8 or not np.isfinite(features).all():
        raise ValueError("Expected finite residual features")
    if model.get("kind") == "constant_residual":
        prediction = np.full(features.shape[:-1], model["intercept"], dtype=np.float64)
    else:
        std = np.asarray(model["std"])
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Frozen feature standard deviations must be positive")
        prediction = model["intercept"] + ((features - np.array(model["mean"])) / std) @ np.array(model["coefficient"])
    return np.clip(model["shrinkage"] * prediction,
                   -model.get("max_log_scale", RESIDUAL_LIMIT), model.get("max_log_scale", RESIDUAL_LIMIT))


def apply_residual(curved, features, model):
    return clip_depth(curved * np.exp(residual_log_scale(features, model)))


def scene_folds(names, count=5):
    names = np.asarray(names)
    groups = np.unique(names)
    if count < 2 or len(groups) < count:
        raise ValueError("Expected at least one scene per fold")
    np.random.default_rng(SEED).shuffle(groups)
    return [np.flatnonzero(np.isin(names, part)) for part in np.array_split(groups, count)]


def prepare_training(handle, predictions, indices):
    features, ratios, samples, targets = [], [], [], []
    for index in indices:
        pred = np.asarray(predictions[index])
        gt = nyu_depth(handle, index)[CROP].astype(np.float64)
        p = pred[CROP].astype(np.float64)
        valid = valid_depth(gt)
        features.append(log_features(pred))
        ratios.append(np.median(gt[valid] / p[valid]))
        if valid.sum() < SAMPLES_PER_IMAGE:
            raise ValueError("Not enough valid pixels for fixed training sampling")
        choice = np.random.default_rng(SAMPLE_SEED + int(index)).choice(valid.sum(), SAMPLES_PER_IMAGE, replace=False)
        # Original preparation persisted float32 samples, then fitted float64.
        samples.append(p[valid][choice].astype(np.float32))
        targets.append(gt[valid][choice].astype(np.float32))
    return np.array(features), np.array(ratios), np.array(samples), np.array(targets)


def residual_targets(handle, predictions, indices, curve):
    output = []
    for index in indices:
        gt = nyu_depth(handle, index)[CROP].astype(np.float64)
        valid = valid_depth(gt)
        curved = apply_curve(predictions[index][CROP].astype(np.float64), curve)[valid]
        g = gt[valid]
        output.append([np.log(np.median(g / curved)),
                       np.log(np.dot(curved, g) / np.dot(curved, curved))])
    return np.array(output)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_training_partition(partition, all_scenes=None):
    """Check the official image partition and optional scene boundary, without GT."""
    train, test = (np.asarray(partition[name]) for name in ("train", "test"))
    for ids, count in ((train, 795), (test, 654)):
        if (ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer) or
                len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= 1449)):
            raise ValueError("Expected unique official 795/654 image identities within 1449 images")
    if np.intersect1d(train, test).size:
        raise ValueError("Official training/test image identities overlap")
    if all_scenes is not None:
        scenes = np.asarray(all_scenes)
        if (scenes.shape != (1449,) or len(np.unique(scenes[train])) != 249 or
                np.intersect1d(scenes[train], scenes[test]).size):
            raise ValueError("Expected 249 training scenes disjoint from official test scenes")
    return train


def validate_training_cache(data_root, predictions):
    """Bind a complete RGB-only cache to the current GT and split files."""
    root = Path(data_root)
    actual = {"nyu_mat_sha256": _sha(root / "nyu_depth_v2_labeled.mat"),
              "splits_sha256": _sha(root / "splits.mat")}
    signature = predictions.metadata.get("signature", {})
    if not isinstance(signature, dict) or signature.get("data") != actual:
        raise ValueError("Prediction cache data identity differs from the training dataset/split")
    return actual


def fit(data_root, prediction_folder, output):
    # Imported lazily so core calibration helpers remain usable independently.
    from .cache import load_predictions
    torch.set_num_threads(4)
    output, data_root, prediction_folder = Path(output), Path(data_root), Path(prediction_folder)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh calibration history directory; existing evidence cannot be overwritten")
    partition = splits(data_root)
    indices = validate_training_partition(partition)
    predictions = load_predictions(prediction_folder, required_ids=indices,
                                   expected_known_intrinsics=False, require_complete=True)
    data_hashes = validate_training_cache(data_root, predictions)
    output.mkdir(parents=True, exist_ok=True)
    with open_nyu(data_root) as handle:
        all_scenes = scene_names(handle)
        validate_training_partition(partition, all_scenes)
        scenes = all_scenes[indices]
        folds = scene_folds(scenes)
        features, ratios, samples, targets = prepare_training(handle, predictions, indices)
        protocol = {"scope": "official795_training_only", "historical_test_exposure": True,
                    "test_predictions_or_labels_read": False, "training_ids": indices.tolist(),
                    "scenes": scenes.tolist(), "folds": [[int(indices[r]) for r in f] for f in folds],
                    "fold_seed": SEED, "sample_seed": SAMPLE_SEED,
                    "sample_rule": "default_rng(sample_seed + dataset_index), 4096 valid pixels without replacement",
                    "samples_per_image": SAMPLES_PER_IMAGE, "candidates": candidate_specs(),
                    "optimizer": {"name": "LBFGS", "dtype": "float64", "max_iter": 80, "max_eval": 120,
                                  "history_size": 20, "tolerance_grad": 1e-9, "tolerance_change": 1e-11,
                                  "line_search_fn": "strong_wolfe"},
                    "selection_rule": "All three OOF mean gains >1e-6 and all three gains >0 in every scene fold; minimum eligible AbsRel",
                    "dataset_sha256": data_hashes["nyu_mat_sha256"],
                    "split_sha256": data_hashes["splits_sha256"],
                    "prediction_files_sha256": {p.name: _sha(p) for p in prediction_folder.iterdir()
                                                   if p.is_file() and p.name in ("depth.npy", "indices.npy", "intrinsics.npy", "runtime.json", "manifest.json")},
                    "source_sha256": {name: _sha(Path(__file__).parent / name) for name in
                                      ("calibration.py", "frozen.py", "cache.py", "metrics.py", "data.py")}}
        _write_json(output / "calibration_protocol.json", protocol)
        names = ["raw", "global_scale", "monotone_curve"] + [spec["name"] for spec in CANDIDATES]
        scores = np.full((len(indices), len(names), 3), np.nan)
        fold_id = np.empty(len(indices), dtype=int)
        fold_parameters = []
        for fold, validation in enumerate(folds):
            train = np.setdiff1d(np.arange(len(indices)), validation)
            start = perf_counter()
            scale = fit_scale(ratios[train])
            curve = fit_curve(samples[train], targets[train], scale)
            residual_y = residual_targets(handle, predictions, indices[train], curve)
            models = [fit_constant_residual(residual_y[:, int(spec["target"] == "rmse_ratio")])
                      if spec["kind"] == "constant" else
                      fit_residual(features[train], residual_y[:, int(spec["target"] == "rmse_ratio")], spec["lam"], spec["shrinkage"])
                      for spec in CANDIDATES]
            for row in validation:
                pred = predictions[indices[row]].astype(np.float64)
                gt = nyu_depth(handle, indices[row])
                curved = apply_curve(pred, curve)
                outputs = [pred, clip_depth(scale * pred), curved]
                outputs += [apply_residual(curved, features[row], model) for model in models]
                for col, depth in enumerate(outputs):
                    scores[row, col] = evaluate_image(depth, gt)
                fold_id[row] = fold
            fold_parameters.append({"fold": fold, "scale": scale, "curve": curve,
                                    "residual_models": dict(zip(names[3:], models)),
                                    "fit_ids": indices[train].tolist(), "validation_ids": indices[validation].tolist(),
                                    "fit_images": len(train), "validation_images": len(validation),
                                    "seconds": perf_counter() - start})
            print("Calibration fold", fold + 1, "seconds", round(perf_counter() - start, 1), flush=True)
        if not np.isfinite(scores).all():
            raise ArithmeticError("Nonfinite or incomplete out-of-fold metrics")
        rows = [{"index": int(index), "scene": scenes[row], "fold": int(fold_id[row]),
                 "method": method, **dict(zip(METRIC_NAMES, scores[row, col]))}
                for row, index in enumerate(indices) for col, method in enumerate(names)]
        pd.DataFrame(rows).to_csv(output / "calibration_oof_per_image.csv", index=False)
        overall = scores.mean(axis=0)
        direction = np.array([-1, -1, 1])
        gain = (scores[:, 3:] - scores[:, 2, None]) * direction
        unique, inverse = np.unique(scenes, return_inverse=True)
        count = np.bincount(inverse)
        sums = np.stack([gain[inverse == i].sum(0) for i in range(len(unique))])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap = np.empty((10000, len(CANDIDATES), 3))
        for start in range(0, 10000, 100):
            draw = rng.integers(0, len(unique), (100, len(unique)))
            bootstrap[start:start + 100] = sums[draw].sum(1) / count[draw].sum(1)[:, None, None]
        intervals = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        adjusted = np.percentile(bootstrap, 100 * .05 / (len(CANDIDATES) * 3), axis=0)
        selection, eligible, comparisons = [], [], {}
        for col, spec in enumerate(CANDIDATES, start=3):
            k = col - 3
            means = gain[:, k].mean(0)
            by_fold = np.array([gain[part, k].mean(0) for part in folds])
            accepted = bool(np.all(means > 1e-6) and np.all(by_fold > 0))
            selection.append({"method": names[col], "lambda": spec.get("lam"), "target": spec["target"],
                              "shrinkage": spec["shrinkage"], "eligible": accepted,
                              "simultaneous_confidence": bool(np.all(adjusted[k] > 0)),
                              **dict(zip(METRIC_NAMES, overall[col]))})
            comparisons[names[col]] = {"gain": means.tolist(), "fold_gains": by_fold.tolist(),
                                      "scene_95ci": intervals[:, k].tolist(), "simultaneous_lower": adjusted[k].tolist()}
            if accepted:
                eligible.append(col)
        best = min(eligible, key=lambda col: overall[col, 0]) if eligible else None
        scale = fit_scale(ratios)
        curve = fit_curve(samples, targets, scale)
        residual = None
        if best is not None:
            spec = CANDIDATES[best - 3]
            y = residual_targets(handle, predictions, indices, curve)[:, int(spec["target"] == "rmse_ratio")]
            residual = fit_constant_residual(y) if spec["kind"] == "constant" else fit_residual(features, y, spec["lam"], spec["shrinkage"])
        parameters = {"schema": "depthlab_calibration_v2", "complete": True, "global_scale": scale, "monotone_curve": curve, "residual": residual,
                      "selected_method": names[best] if best is not None else "monotone_curve",
                      "seed": SEED, "sample_seed": SAMPLE_SEED, "sample_count": SAMPLES_PER_IMAGE,
                      "training_images": len(indices), "training_scenes": len(unique),
                      "training_ids": indices.tolist(), "historical_test_exposure": True, "test_read": False,
                      "protocol_sha256": _sha(output / "calibration_protocol.json"),
                      "dataset_sha256": protocol["dataset_sha256"], "split_sha256": protocol["split_sha256"],
                      "prediction_files_sha256": protocol["prediction_files_sha256"],
                      "source_sha256": protocol["source_sha256"],
                      "selection": {"candidate": names[best] if best is not None else None,
                                    "candidate_spec": CANDIDATES[best - 3] if best is not None else None,
                                    "eligible_candidates": [names[col] for col in eligible],
                                    "all_candidates": selection, "test_read": False}}
        _write_json(output / "calibration_parameters.json", parameters)
        _write_json(output / "calibration_folds.json", fold_parameters)
        _write_json(output / "calibration_comparisons.json", comparisons)
        pd.DataFrame(selection).to_csv(output / "calibration_selection.csv", index=False)
        pd.DataFrame(overall, columns=METRIC_NAMES, index=pd.Index(names, name="method")).to_csv(output / "calibration_oof_summary.csv")
        print("Selected", parameters["selected_method"], "scale", scale, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh training-only study history directory")
    args = parser.parse_args()
    fit(args.data_root, args.predictions, args.output)
