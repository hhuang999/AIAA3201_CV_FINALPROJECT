import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import read_pfm

BLOCK_SIZE = 5
CONFIGURATIONS = {"unpad_3way": (cv2.STEREO_SGBM_MODE_SGBM_3WAY, False),
                  "reflect_3way": (cv2.STEREO_SGBM_MODE_SGBM_3WAY, True),
                  "reflect_hh4": (cv2.STEREO_SGBM_MODE_HH4, True)}


def calibration(path):
    text = Path(path).read_text()
    values = {key: float(re.search(rf"^{key}=([^\n]+)", text, re.M)[1])
              for key in ("baseline", "doffs", "width", "height", "ndisp")}
    values["focal"] = float(re.search(r"cam0=\[([^ ]+)", text)[1])
    return values


def disparity_to_depth(disparity, focal, baseline_mm, doffs):
    return focal * (baseline_mm / 1000) / (disparity + doffs)


def geometric_valid(disparity):
    columns = np.arange(disparity.shape[1])[None, :]
    return (np.isfinite(disparity) & (disparity > 0) &
            (columns - disparity >= 0) & (columns - disparity < disparity.shape[1]))


def match_pair(left, right, calib, mode, padded):
    count = max(16, int(np.ceil(calib["ndisp"] / 16)) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=count, blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE**2, P2=32 * BLOCK_SIZE**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=mode)

    def compute(first, second):
        if padded:
            first = cv2.copyMakeBorder(first, 0, 0, count, 0, cv2.BORDER_REFLECT_101)
            second = cv2.copyMakeBorder(second, 0, 0, count, 0, cv2.BORDER_REFLECT_101)
        disparity = matcher.compute(np.ascontiguousarray(first),
                                    np.ascontiguousarray(second)).astype(np.float32) / 16
        return disparity[:, count:] if padded else disparity

    start = perf_counter()
    forward = compute(left, right)
    backward = compute(right[:, ::-1], left[:, ::-1])
    right_valid = geometric_valid(backward)[:, ::-1]
    backward = backward[:, ::-1]
    valid = geometric_valid(forward)
    x = np.floor(np.arange(left.shape[1])[None, :] - forward + 0.5).astype(int)
    inside = (x >= 0) & (x < left.shape[1])
    x = np.clip(x, 0, left.shape[1] - 1)
    rows = np.arange(left.shape[0])[:, None]
    valid &= inside & right_valid[rows, x] & (np.abs(forward - backward[rows, x]) <= 1)
    seconds = perf_counter() - start
    return forward, valid, seconds


def plot_scene(rgb, disparity, valid, depth, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    axes[0].imshow(rgb)
    images = [None,
              axes[1].imshow(np.ma.masked_where(~valid, disparity), cmap="magma"),
              axes[2].imshow(~valid, cmap="gray", vmin=0, vmax=1),
              axes[3].imshow(depth, cmap="viridis", vmin=0,
                             vmax=np.nanpercentile(depth, 98))]
    for ax, title in zip(axes, ["Left RGB", "Disparity (px)", "Invalid (white)", "Depth (m)"]):
        ax.set_title(title)
        ax.axis("off")
    for index in (1, 3):
        fig.colorbar(images[index], ax=axes[index], shrink=.75)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(data_root, output, limit=None, threads=4):
    cv2.setNumThreads(threads)
    output = Path(output)
    (output / "figures/stereo").mkdir(parents=True, exist_ok=True)
    folders = sorted((Path(data_root) / "MiddEval3/trainingQ").iterdir())[:limit]
    records = []
    for folder in folders:
        rgb = cv2.cvtColor(cv2.imread(str(folder / "im0.png")), cv2.COLOR_BGR2RGB)
        left = cv2.imread(str(folder / "im0.png"), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(folder / "im1.png"), cv2.IMREAD_GRAYSCALE)
        calib = calibration(folder / "calib.txt")
        assert left.shape == right.shape == (int(calib["height"]), int(calib["width"]))
        target = read_pfm(folder / "disp0GT.pfm")
        gt_valid = np.isfinite(target) & (target > 0)
        for name, (mode, padded) in CONFIGURATIONS.items():
            disparity, valid, seconds = match_pair(left, right, calib, mode, padded)
            intersection = valid & gt_valid
            records.append({"scene": folder.name, "configuration": name,
                            "mae_px": float(np.abs(disparity[intersection] - target[intersection]).mean()),
                            "valid_fraction": float(intersection.sum() / gt_valid.sum()),
                            "cpu_seconds": seconds, "threads": threads})
            if name == "reflect_hh4":
                depth = disparity_to_depth(disparity, calib["focal"], calib["baseline"], calib["doffs"])
                depth[~valid] = np.nan
                plot_scene(rgb, disparity, valid, depth, output / "figures/stereo" / f"{folder.name}.png")
        print("Stereo", folder.name, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "stereo_per_scene.csv", index=False)
    means = frame.groupby("configuration")[["mae_px", "valid_fraction", "cpu_seconds"]].mean()
    means.to_csv(output / "stereo_summary.csv")
    (output / "stereo_runtime.json").write_text(json.dumps({"opencv": cv2.__version__,
        "threads": threads, "scope": "forward and reverse matching plus consistency mask"}, indent=2))
    print(means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.data_root, args.output, args.limit)
