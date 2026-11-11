import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .calibration import fit_scale
from .data import nyu_depth, open_nyu, scene_names, splits
from .infer_nyu import RGB_INTRINSICS
from .metrics import CROP, METRIC_NAMES, evaluate_image, valid_depth


def diagnose(args):
    ids = splits(args.data_root)["train"]
    raw = np.load(args.predictions / "depth.npy", mmap_mode="r")
    known = np.load(args.known_predictions / "depth.npy", mmap_mode="r")
    intrinsics = np.load(args.predictions / "intrinsics.npy")
    assert np.array_equal(ids, np.load(args.known_predictions / "indices.npy"))
    handle = open_nyu(args.data_root)
    scenes = scene_names(handle)
    records, scores, ratios = [], [], {"rgb_only": [], "known_intrinsics": []}
    for row, index in enumerate(ids):
        gt = nyu_depth(handle, index)
        mask = valid_depth(gt[CROP])
        focal_ratio = .5 * (intrinsics[index, 0, 0] / RGB_INTRINSICS[0, 0]
                            + intrinsics[index, 1, 1] / RGB_INTRINSICS[1, 1])
        scale_ratio = np.median(raw[index][CROP][mask] / gt[CROP][mask])
        records.append({"index": index, "scene": scenes[index], "focal_ratio": focal_ratio,
                        "raw_over_target": scale_ratio})
        for method, depth in (("rgb_only", raw[index]), ("known_intrinsics", known[row])):
            scores.append({"index": index, "method": method,
                           **dict(zip(METRIC_NAMES, evaluate_image(depth, gt)))})
            ratios[method].append(np.median(gt[CROP][mask] / depth[CROP][mask]))
    handle.close()
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "camera_diagnostic_per_image.csv", index=False)
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(args.output / "camera_diagnostic_scores.csv", index=False)
    summary = score_frame.groupby("method")[list(METRIC_NAMES)].mean()
    summary["training_median_scale"] = [fit_scale(ratios[name]) for name in summary.index]
    summary.to_csv(args.output / "camera_diagnostic_summary.csv")
    pearson = pearsonr(frame.focal_ratio, frame.raw_over_target)
    spearman = spearmanr(frame.focal_ratio, frame.raw_over_target)
    report = {"images": len(ids), "split": "training only", "focal_ratio_mean": float(frame.focal_ratio.mean()),
              "focal_ratio_median": float(frame.focal_ratio.median()),
              "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
              "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
              "inverse_rgb_scale": 1 / fit_scale(ratios["rgb_only"])}
    (args.output / "camera_diagnostic.json").write_text(json.dumps(report, indent=2))
    print(summary, report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--known-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    diagnose(parser.parse_args())
