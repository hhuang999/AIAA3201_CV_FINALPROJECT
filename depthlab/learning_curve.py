"""Optional training-scene learning curves; fixed deployment remains unchanged.

Reuses the full-data fold fits of a completed calibration_study. Smaller nested
scene subsets reproduce the original CPU float64 spline RMSE fitting protocol.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache import atomic_json, sha256
from .calibration_study import TrainingData, METRICS, fit_map, fresh_output, score_fold, write_csv

FRACTIONS = (.1, .25, .5, 1.)
SEED = 20261111


def nested_scene_subsets(assignment, scenes, fold):
    assignment, scenes = np.asarray(assignment), np.asarray(scenes)
    if assignment.shape != scenes.shape or assignment.ndim != 1:
        raise ValueError("Expected aligned scene/fold vectors")
    fitting = np.flatnonzero(assignment != fold)
    heldout = np.flatnonzero(assignment == fold)
    if not len(fitting) or not len(heldout) or set(scenes[fitting]) & set(scenes[heldout]):
        raise ValueError("Expected nonempty disjoint fitting/held-out scenes")
    shuffled = np.random.default_rng(SEED + fold).permutation(np.unique(scenes[fitting]))
    subsets = {fraction: fitting[np.isin(scenes[fitting], shuffled[:max(1, int(np.ceil(fraction * len(shuffled))))])]
               for fraction in FRACTIONS}
    return subsets, heldout


def load_base_study(path, data):
    path = Path(path)
    protocol = json.loads((path / "protocol.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    if not selection.get("complete") or selection.get("smoke") or selection.get("protocol_sha256") != sha256(path / "protocol.json"):
        raise ValueError("Expected completed non-smoke training discovery study")
    if (protocol.get("ids") != data.ids.tolist() or protocol.get("scenes") != data.scenes.tolist() or
            protocol.get("folds") != data.assignment.tolist() or protocol.get("test_read") is not False):
        raise ValueError("Base study has different training identities/scene folds")
    expected = data.fingerprint("learning_curve.py")
    for key in ("dataset_sha256", "split_sha256", "cache_manifest_sha256"):
        if protocol.get(key) != expected[key]:
            raise ValueError("Base study provenance mismatch: " + key)
    evidence = selection.get("fold_evidence")
    if not isinstance(evidence, list) or len(evidence) != 5:
        raise ValueError("Expected all five recorded base folds")
    models = []
    for fold in range(5):
        destination = path / f"fold{fold}"
        for name in ("completion.json", "per_image.csv", "models.json"):
            if evidence[fold].get(name) != sha256(destination / name):
                raise ValueError("Recorded base fold changed: " + name)
        receipt = json.loads((destination / "completion.json").read_text())
        fit, val = np.flatnonzero(data.assignment != fold), np.flatnonzero(data.assignment == fold)
        if (not receipt.get("complete") or receipt.get("smoke") or receipt.get("fold") != fold or
                receipt.get("fit_ids") != data.ids[fit].tolist() or receipt.get("validation_ids") != data.ids[val].tolist()):
            raise ValueError("Base fold receipt does not match fixed identities")
        models.append(json.loads((destination / "models.json").read_text()))
    return models


def load_legacy_base_study(path, frozen_path, data, expected_sha256=None):
    """Reuse migrated original folds with their original selection/hash chain.

    Explicit legacy freezing metadata is required. Cache bytes must match the
    frozen original train cache; no compatibility-by-filename assumption.
    """
    path, frozen_path = Path(path), Path(frozen_path)
    if expected_sha256 is not None and sha256(frozen_path) != expected_sha256:
        raise ValueError("Legacy base frozen SHA-256 mismatch")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("primary") != "spline_rmse" or frozen.get("test_read") is not False or frozen.get("train_ids") != data.ids.tolist():
        raise ValueError("Expected the original training-frozen spline base and identical training IDs")
    fingerprint = data.fingerprint("learning_curve.py")
    for key in ("dataset_sha256", "split_sha256"):
        if frozen.get(key) != fingerprint[key]:
            raise ValueError("Legacy base provenance mismatch: " + key)
    if sha256(data.prediction_root / "depth.npy") != frozen.get("train_cache_sha256"):
        raise ValueError("Legacy base requires the exact original training prediction cache")
    selection_path = path / "selection" / "selection.json"
    if sha256(selection_path) != frozen.get("selection_sha256"):
        raise ValueError("Legacy discovery selection changed after freezing")
    selection = json.loads(selection_path.read_text())
    evidence = selection.get("fold_evidence")
    if selection.get("test_read") is not False or not isinstance(evidence, list) or len(evidence) != 5:
        raise ValueError("Incomplete legacy training-fold evidence")
    models = []
    for fold in range(5):
        folder = path / "cv" / f"fold{fold}"
        receipt_keys = {"completion.json": "completion_sha256", "per_image.csv": "metrics_sha256", "models.json": "models_sha256"}
        if evidence[fold].get("fold") != fold or any(sha256(folder / name) != evidence[fold].get(key) for name, key in receipt_keys.items()):
            raise ValueError("Legacy fold artifact changed")
        receipt = json.loads((folder / "completion.json").read_text())
        fit, val = np.flatnonzero(data.assignment != fold), np.flatnonzero(data.assignment == fold)
        if (not receipt.get("complete") or receipt.get("fold") != fold or
                receipt.get("fit_ids") != data.ids[fit].tolist() or receipt.get("validation_ids") != data.ids[val].tolist()):
            raise ValueError("Legacy fold identities differ from restored scene assignment")
        models.append(json.loads((folder / "models.json").read_text()))
    return models


def summarize_learning(rows, fits, ids, scenes, repeats=10000):
    ids, scenes = np.asarray(ids), np.asarray(scenes)
    position = {int(index): pos for pos, index in enumerate(ids)}
    if len(position) != len(ids) or len(ids) != len(scenes) or repeats < 1:
        raise ValueError("Expected unique aligned image/scene identities")
    values = {(fraction, name): np.full((len(ids), 3), np.nan)
              for fraction in FRACTIONS for name in ("median", "spline_rmse")}
    for row in rows:
        key = float(row["fraction"]), row["method"]
        pos = position[int(row["dataset_index"])]
        if row["scene_id"] != scenes[pos] or np.isfinite(values[key][pos]).any():
            raise ValueError("Wrong or duplicate held-out identity")
        values[key][pos] = [float(row[k]) for k in METRICS]
    if any(not np.isfinite(a).all() for a in values.values()):
        raise ValueError("Incomplete/nonfinite learning-curve evidence")
    unique = np.unique(scenes)
    counts = np.array([(scenes == scene).sum() for scene in unique])
    draws = np.random.default_rng(SEED).integers(0, len(unique), (repeats, len(unique)))
    table, intervals = [], {}
    for fraction in FRACTIONS:
        selected = [fit for fit in fits if fit["fraction"] == fraction]
        if not selected:
            raise ValueError("Missing learning-curve fit receipt")
        gain = values[fraction, "median"] - values[fraction, "spline_rmse"]
        gain[:, 2] *= -1
        sums = np.stack([gain[scenes == scene].sum(0) for scene in unique])
        boot = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
        intervals[str(fraction)] = {"gain": gain.mean(0).tolist(),
                                   "conditional_scene_95ci": np.percentile(boot, [2.5, 97.5], axis=0).tolist()}
        for name in ("median", "spline_rmse"):
            table.append({"fraction": fraction, "method": name, "validation_images": len(ids),
                          "fit_images_mean": float(np.mean([fit["fit_images"] for fit in selected])),
                          "fit_images_min": min(fit["fit_images"] for fit in selected),
                          "fit_images_max": max(fit["fit_images"] for fit in selected),
                          "fit_scenes_mean": float(np.mean([fit["fit_scenes"] for fit in selected])),
                          **dict(zip(METRICS, values[fraction, name].mean(0).tolist()))})
    return table, intervals, values


def run(args):
    torch.set_num_threads(2)
    output = fresh_output(args.output)
    with TrainingData(args.data_root, args.predictions) as data:
        legacy = args.legacy_base_frozen is not None
        base_models = (load_legacy_base_study(args.base_study, args.legacy_base_frozen, data, args.expected_legacy_frozen_sha256)
                       if legacy else load_base_study(args.base_study, data))
        base_root = Path(args.base_study) / "cv" if legacy else Path(args.base_study)
        selection_root = Path(args.base_study) / "selection" if legacy else Path(args.base_study)
        protocol = data.fingerprint("learning_curve.py")
        protocol.update(fractions=list(FRACTIONS), seed=SEED, smoke=args.smoke_only,
                        fitting="Nested shuffled scene subsets, ceil(fraction * fitting scenes); all their images; full fold fits reused exactly",
                        model_selection=False, deployment_changed=False, planned_fit_trials=15,
                        base_selection_sha256=sha256(selection_root / "selection.json"))
        atomic_json(output / "protocol.json", protocol)
        all_rows, all_fits = [], []
        for fold in ([0] if args.smoke_only else range(5)):
            subsets, heldout = nested_scene_subsets(data.assignment, data.scenes, fold)
            if args.smoke_only:
                heldout = heldout[:2]
            destination = output / f"fold{fold}"
            destination.mkdir()
            rows, fits = [], []
            for fraction in ((.1,) if args.smoke_only else FRACTIONS):
                chosen = subsets[fraction]
                if args.smoke_only:
                    chosen = chosen[:8]
                started = time.perf_counter()
                median = {"kind": "scalar", "scale": float(np.median(data.ratios[chosen]))}
                if fraction == 1.:
                    if median != base_models[fold]["median"]:
                        raise ValueError("Full-data median changed from recorded base")
                    spline, trace = base_models[fold]["spline_rmse"], {"reused_exact_recorded_fit": True}
                else:
                    p = torch.tensor(data.samples[chosen], dtype=torch.float64)
                    gt = torch.tensor(data.targets[chosen], dtype=torch.float64)
                    spline, detail = fit_map("spline", "rmse", p, gt, median["scale"])
                    trace = {key: value for key, value in detail.items() if key != "history"}
                fits.append({"fold": fold, "fraction": fraction, "fit_ids": data.ids[chosen].tolist(),
                             "fit_images": len(chosen), "fit_scenes": len(np.unique(data.scenes[chosen])),
                             "median": median, "spline_rmse": spline, "optimization": trace,
                             "fit_seconds": time.perf_counter() - started})
                rows.extend(score_fold(data, heldout, {"median": median, "spline_rmse": spline}, fold,
                                       extra={"fraction": fraction, "fit_images": len(chosen),
                                              "fit_scenes": len(np.unique(data.scenes[chosen]))}))
                print("LEARNING_POINT", fold, fraction, len(chosen), flush=True)
            write_csv(destination / "per_image.csv", rows)
            atomic_json(destination / "fits.json", fits)
            atomic_json(destination / "completion.json", {"complete": True, "smoke": args.smoke_only,
                        "fold": fold, "heldout_ids": data.ids[heldout].tolist(), "test_read": False,
                        "base_model_sha256": sha256(base_root / f"fold{fold}" / "models.json")})
            all_rows.extend(rows)
            all_fits.extend(fits)
        if args.smoke_only:
            atomic_json(output / "completion.json", {"complete": True, "smoke": True, "test_read": False, "model_selection": False})
            return
        table, intervals, values = summarize_learning(all_rows, all_fits, data.ids, data.scenes)
        position = {int(index): pos for pos, index in enumerate(data.ids)}
        parity = 0.
        with (selection_root / "oof_per_image.csv").open() as stream:
            reference = [row for row in csv.DictReader(stream) if row["method"] in ("median", "spline_rmse")]
        if len(reference) != len(data.ids) * 2:
            raise ValueError("Missing paired full-training reference")
        for row in reference:
            pos = position[int(row["dataset_index"])]
            parity = max(parity, float(np.max(np.abs(values[1., row["method"]][pos] - [float(row[k]) for k in METRICS]))))
        if parity > 1e-10:
            raise ValueError("Full-training reference changed")
        write_csv(output / "learning_curve.csv", table)
        write_csv(output / "oof_per_image.csv", all_rows)
        atomic_json(output / "comparisons.json", intervals)
        atomic_json(output / "completion.json", {"complete": True, "smoke": False, "test_read": False,
                    "model_selection": False, "deployment_changed": False, "fit_trials": 15,
                    "full_fit_reused": 5, "full_oof_max_difference": parity,
                    "protocol_sha256": sha256(output / "protocol.json"),
                    "interpretation": "Exploratory training-only nested scene learning curve. Conditional scene intervals do not include refitting uncertainty."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--base-study", type=Path, required=True)
    parser.add_argument("--legacy-base-frozen", type=Path,
                        help="Original multiobjective frozen_models.json, enables hash-verified legacy cv/selection layout")
    parser.add_argument("--expected-legacy-frozen-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
