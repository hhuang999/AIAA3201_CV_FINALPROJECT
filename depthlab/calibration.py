import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .data import open_nyu, nyu_depth, scene_names, splits
from .metrics import CROP, METRIC_NAMES, clip_depth, evaluate_image, valid_depth

SEED = 42
SAMPLES_PER_IMAGE = 4096
KNOTS = np.log([0.3, 0.7, 1.5, 3, 6, 12])
LOG_FLOOR = 1e-6
RESIDUAL_LIMIT = 0.15
CANDIDATES = tuple(itertools.product((0.1, 1., 10.), (0.5, 1.)))


def fit_scale(ratios):
    return float(np.median(ratios))


def log_features(depth):
    log = np.log(np.maximum(depth[CROP].astype(np.float64), LOG_FLOOR))
    gradient = (np.abs(np.diff(log, axis=0)).mean() +
                np.abs(np.diff(log, axis=1)).mean()) / 2
    return np.r_[np.percentile(log, [10, 25, 50, 75, 90]),
                 log.mean(), log.std(), gradient]


def curve_basis(depth):
    x = np.log(np.maximum(depth, LOG_FLOOR))
    return np.stack([np.minimum(x - KNOTS[0], KNOTS[1] - KNOTS[0]),
                     *[np.clip(x - KNOTS[k], 0, KNOTS[k + 1] - KNOTS[k])
                       for k in range(1, 4)], np.maximum(x - KNOTS[4], 0)], axis=-1)


def apply_curve(depth, curve):
    y = KNOTS[0] + curve["offset"] + curve_basis(depth) @ np.array(curve["slopes"])
    return clip_depth(np.exp(y))


def fit_curve(samples, targets, scale):
    basis = torch.from_numpy(curve_basis(samples))
    target = torch.from_numpy(targets)
    initial_offset = np.arctanh(np.clip(np.log(scale) / np.log(2), -.999999, .999999))
    initial_slope = np.log((1 - .25) / (2 - 1))
    raw = torch.tensor([initial_offset, *([initial_slope] * 5)],
                       dtype=torch.float64, requires_grad=True)
    normalizer = np.sqrt(np.mean((clip_depth(scale * samples) - targets)**2, axis=1)).mean()
    optimizer = torch.optim.LBFGS([raw], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        offset = np.log(2) * torch.tanh(raw[0])
        slopes = .25 + 1.75 * torch.sigmoid(raw[1:])
        mapped = torch.exp(KNOTS[0] + offset + basis @ slopes).clamp(.001, 10)
        loss = ((mapped - target).square().mean(dim=1).sqrt().mean() / normalizer
                + .01 * torch.diff(slopes).square().mean()
                + .001 * (slopes - 1).square().mean())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"offset": float((np.log(2) * torch.tanh(raw[0])).detach()),
            "slopes": (.25 + 1.75 * torch.sigmoid(raw[1:])).detach().tolist(),
            "iterations": int(optimizer.state[raw]["n_iter"])}


def fit_residual(features, target, regularization, shrinkage):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    x = (features - mean) / std
    intercept = target.mean()
    coefficient = np.linalg.solve(x.T @ x / len(x) + regularization * np.eye(8),
                                  x.T @ (target - intercept) / len(x))
    return {"mean": mean.tolist(), "std": std.tolist(), "intercept": float(intercept),
            "coefficient": coefficient.tolist(), "lambda": regularization,
            "shrinkage": shrinkage}


def residual_log_scale(features, model):
    x = (features - np.array(model["mean"])) / np.array(model["std"])
    return np.clip(model["shrinkage"] * (model["intercept"] +
                   x @ np.array(model["coefficient"])), -RESIDUAL_LIMIT, RESIDUAL_LIMIT)


def apply_residual(curved, features, model):
    return clip_depth(curved * np.exp(residual_log_scale(features, model)))


def scene_folds(names, count=5):
    groups = np.unique(names)
    np.random.default_rng(SEED).shuffle(groups)
    folds = [np.flatnonzero(np.isin(names, part)) for part in np.array_split(groups, count)]
    assert len(set.union(*[set(names[f]) for f in folds])) == sum(len(set(names[f])) for f in folds)
    return folds


def prepare_training(handle, predictions, indices):
    rng = np.random.default_rng(SEED)
    features, ratios, samples, targets = [], [], [], []
    for index in indices:
        pred = predictions[index]
        gt = nyu_depth(handle, index)[CROP].astype(np.float64)
        p = pred[CROP].astype(np.float64)
        valid = valid_depth(gt)
        ratios.append(np.median(gt[valid] / p[valid]))
        choice = rng.choice(valid.sum(), SAMPLES_PER_IMAGE, replace=False)
        samples.append(p[valid][choice])
        targets.append(gt[valid][choice])
        features.append(log_features(pred))
    return np.array(features), np.array(ratios), np.array(samples), np.array(targets)


def residual_targets(handle, predictions, indices, curve):
    output = []
    for index in indices:
        gt = nyu_depth(handle, index)[CROP]
        valid = valid_depth(gt)
        curved = apply_curve(predictions[index][CROP].astype(np.float64), curve)
        output.append(np.log(np.median(gt[valid] / curved[valid])))
    return np.array(output)


def fit(data_root, prediction_folder, output):
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    indices = splits(data_root)["train"]
    predictions = np.load(Path(prediction_folder) / "depth.npy", mmap_mode="r")
    handle = open_nyu(data_root)
    scenes = scene_names(handle)[indices]
    folds = scene_folds(scenes)
    features, ratios, samples, targets = prepare_training(handle, predictions, indices)
    names = ["raw", "global_scale", "monotone_curve"] + [f"residual_l{v:g}_s{s:g}" for v, s in CANDIDATES]
    scores = np.empty((len(indices), len(names), 3))
    fold_id = np.empty(len(indices), dtype=int)
    fold_parameters = []
    for fold, validation in enumerate(folds):
        train = np.setdiff1d(np.arange(len(indices)), validation)
        start = perf_counter()
        scale = fit_scale(ratios[train])
        curve = fit_curve(samples[train], targets[train], scale)
        residual_y = residual_targets(handle, predictions, indices[train], curve)
        models = [fit_residual(features[train], residual_y, v, s) for v, s in CANDIDATES]
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
                                "fit_images": len(train), "validation_images": len(validation),
                                "seconds": perf_counter() - start})
        print("Calibration fold", fold + 1, "seconds", round(perf_counter() - start, 1), flush=True)
    rows = [{"index": int(index), "scene": scenes[row], "fold": int(fold_id[row]),
             "method": method, **dict(zip(METRIC_NAMES, scores[row, col]))}
            for row, index in enumerate(indices) for col, method in enumerate(names)]
    pd.DataFrame(rows).to_csv(output / "calibration_oof_per_image.csv", index=False)
    overall = scores.mean(axis=0)
    per_fold = np.array([scores[part].mean(axis=0) for part in folds])
    direction = np.array([-1, -1, 1])
    eligible = []
    selection = []
    for col, candidate in enumerate(CANDIDATES, start=3):
        accepted = bool(np.all((overall[col] - overall[2]) * direction > 0)
                        and np.all((per_fold[:, col] - per_fold[:, 2]) * direction > 0))
        selection.append({"method": names[col], "lambda": candidate[0],
                          "shrinkage": candidate[1], "eligible": accepted,
                          **dict(zip(METRIC_NAMES, overall[col]))})
        if accepted:
            eligible.append(col)
    best = min(eligible, key=lambda col: overall[col, 0]) if eligible else None
    scale = fit_scale(ratios)
    curve = fit_curve(samples, targets, scale)
    residual = None
    if best is not None:
        v, s = CANDIDATES[best - 3]
        residual = fit_residual(features, residual_targets(handle, predictions, indices, curve), v, s)
    parameters = {"global_scale": scale, "monotone_curve": curve,
                  "residual": residual, "selected_method": names[best] if best is not None else "monotone_curve",
                  "seed": SEED, "sample_count": SAMPLES_PER_IMAGE,
                  "training_images": len(indices), "training_scenes": len(np.unique(scenes))}
    (output / "calibration_parameters.json").write_text(json.dumps(parameters, indent=2))
    (output / "calibration_folds.json").write_text(json.dumps(fold_parameters, indent=2))
    pd.DataFrame(selection).to_csv(output / "calibration_selection.csv", index=False)
    pd.DataFrame(overall, columns=METRIC_NAMES, index=pd.Index(names, name="method")).to_csv(output / "calibration_oof_summary.csv")
    print("Selected", parameters["selected_method"], "scale", scale, flush=True)
    handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    fit(args.data_root, args.predictions, args.output)
