"""Frozen Part 1 stereo geometry inside the depthlab architecture.

The primary method is reflected-padding three-way SGBM with explicit 1 px
left/right consistency. Unpadded and HH4 runs remain descriptive ablations.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
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
                  "reflect_hh4": (cv2.STEREO_SGBM_MODE_HH4, True),
                  "reflect_hh": (cv2.STEREO_SGBM_MODE_HH, True)}
DEFAULT_CONFIGURATIONS = ("unpad_3way", "reflect_3way", "reflect_hh4")
PRIMARY_CONFIGURATION = "reflect_3way"
TIMING_SCOPE = ("Forward SGBM + reverse SGBM + explicit LR residual/mask; includes "
                "grayscale conversion, padding/cropping and original-bound rejection. "
                "Excludes RGB decoding, calibration/GT IO, metric/depth conversion and saving.")


@dataclass(frozen=True)
class SGBMConfig:
    block_size: int = BLOCK_SIZE
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    padding_mode: str = "none"
    mode: str = "SGBM_3WAY"

    def __post_init__(self):
        if self.block_size < 1 or self.block_size % 2 != 1:
            raise ValueError("block_size must be a positive odd integer")
        if self.uniqueness_ratio < 0 or self.speckle_window_size < 0 or self.speckle_range < 0:
            raise ValueError("uniqueness and speckle parameters must be nonnegative")
        if self.padding_mode not in ("none", "replicate", "reflect101"):
            raise ValueError("Unknown padding mode")
        if self.disp12_max_diff < 1:
            raise ValueError("Use an explicit positive OpenCV internal consistency threshold")
        if self.mode not in ("SGBM_3WAY", "HH4", "HH"):
            raise ValueError("Unknown SGBM aggregation mode")

    def metadata(self):
        return {**asdict(self), "P1": 8*self.block_size**2,
                "P2": 32*self.block_size**2, "input": "grayscale",
                "min_disparity": 0}


def calibration(path):
    text = Path(path).read_text()
    values = {}
    for key in ("baseline", "doffs", "width", "height", "ndisp"):
        match = re.search(rf"^{key}=([^\n]+)", text, re.M)
        if match is None:
            raise ValueError(f"{path} has no {key}")
        values[key] = float(match[1])
    camera = re.search(r"^cam0=\[([^ ]+)", text, re.M)
    if camera is None:
        raise ValueError(f"{path} has no cam0")
    values["focal"] = float(camera[1])
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"{path} has nonfinite calibration values")
    if min(values["focal"], values["baseline"], values["width"],
           values["height"], values["ndisp"]) <= 0:
        raise ValueError(f"{path} has nonpositive focal/baseline/dimensions/disparity range")
    for key in ("width", "height", "ndisp"):
        if values[key] != int(values[key]):
            raise ValueError(f"{path}: {key} must be an integer")
        values[key] = int(values[key])
    return values


def disparity_to_depth(disparity, focal, baseline_mm, doffs):
    """Preserve frozen float32 conversion and invalid-depth NaNs."""
    disparity = np.asarray(disparity)
    denominator = disparity.astype(np.float32) + np.float32(doffs)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = (np.isfinite(disparity) & (disparity > 0) &
             np.isfinite(denominator) & (denominator > 1e-3))
    depth[valid] = (focal * (baseline_mm / 1000.0)) / denominator[valid]
    return depth


def _num_disparities(ndisp):
    if ndisp < 1 or not np.isfinite(ndisp):
        raise ValueError("Calibration ndisp must be positive and finite")
    return max(16, int(np.ceil(ndisp / 16.0) * 16))


def _gray(image):
    # Reading PNG directly as grayscale changes rounding versus the frozen path.
    # Grayscale arrays remain accepted by the original depthlab match_pair API.
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 2:
        return image
    raise ValueError("Stereo inputs must be uint8 BGR or grayscale arrays")


def geometric_valid(disparity):
    if disparity.ndim != 2:
        raise ValueError("Disparity must be a 2D array")
    columns = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    correspondence = columns-disparity
    return (np.isfinite(disparity) & (disparity > 0) &
            (correspondence >= 0) & (correspondence <= disparity.shape[1]-1))


def stereo_sgbm(left, right, calib, config=SGBMConfig()):
    """Frozen grayscale conversion, matcher, padding and raw invalid sentinel."""
    gray_l, gray_r = _gray(left), _gray(right)
    if gray_l.shape != gray_r.shape:
        raise ValueError("Stereo images must have equal dimensions")
    if gray_l.dtype != np.uint8 or gray_r.dtype != np.uint8:
        raise ValueError("Stereo images must use uint8 pixels")
    disparities = _num_disparities(calib["ndisp"])
    if disparities >= gray_l.shape[1]:
        raise ValueError("Image width must exceed the calibration disparity search range")
    original_width = gray_l.shape[1]
    if config.padding_mode != "none":
        border = cv2.BORDER_REPLICATE if config.padding_mode == "replicate" else cv2.BORDER_REFLECT_101
        gray_l = cv2.copyMakeBorder(gray_l, 0, 0, disparities, 0, border)
        gray_r = cv2.copyMakeBorder(gray_r, 0, 0, disparities, 0, border)
    block = config.block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=disparities, blockSize=block,
        P1=8*block*block, P2=32*block*block,
        disp12MaxDiff=config.disp12_max_diff,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        mode=getattr(cv2, "STEREO_SGBM_MODE_"+config.mode))
    result = matcher.compute(np.ascontiguousarray(gray_l),
                             np.ascontiguousarray(gray_r)).astype(np.float32)/16.0
    if config.padding_mode != "none":
        result = result[:, disparities:disparities+original_width].copy()
        x = np.arange(original_width, dtype=np.float32)[None, :]
        correspondence = x-result
        outside = ((~np.isfinite(result)) | (correspondence < 0) |
                   (correspondence > original_width-1))
        result[outside] = -1.
    return result


def reverse_disparity(left, right, calib, config=SGBMConfig()):
    return np.fliplr(stereo_sgbm(np.fliplr(right), np.fliplr(left), calib, config))


def lr_residual(disparity_left, disparity_right):
    """Frozen np.rint sampling; out-of-image/nonpositive pairs are +infinity."""
    if disparity_left.shape != disparity_right.shape or disparity_left.ndim != 2:
        raise ValueError("Disparities must be equal-size 2D arrays")
    height, width = disparity_left.shape
    xs = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    finite = np.isfinite(disparity_left)
    location = xs-np.where(finite, disparity_left, 0)
    sampled_x = np.rint(np.clip(location, -1, width)).astype(np.int32)
    # Check the unrounded location: -0.25 must not become a valid pixel zero.
    inside = ((location >= 0) & (location <= width-1) &
              finite & (disparity_left > 0))
    sampled = disparity_right[np.arange(height)[:, None], np.clip(sampled_x, 0, width-1)]
    valid = inside & np.isfinite(sampled) & (sampled > 0)
    residual = np.full(disparity_left.shape, np.inf, dtype=np.float32)
    residual[valid] = np.abs(disparity_left[valid]-sampled[valid])
    return residual


def left_right_consistent(left, right, disparity_left, calib, tolerance=1.,
                          config=SGBMConfig()):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("LR tolerance must be finite and positive")
    return lr_residual(disparity_left, reverse_disparity(left, right, calib, config)) <= tolerance


def match_pair(left, right, calib, mode=None, padded=None, *, config=None,
               lr_tolerance=1., external_lr=True):
    """Keep the rebuilt positional API, with the frozen matcher/LR semantics."""
    if not np.isfinite(lr_tolerance) or lr_tolerance <= 0:
        raise ValueError("LR tolerance must be finite and positive")
    if config is None:
        modes = {getattr(cv2, "STEREO_SGBM_MODE_"+name): name
                 for name in ("SGBM_3WAY", "HH4", "HH")}
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY if mode is None else mode
        if mode not in modes:
            raise ValueError("Unknown SGBM aggregation mode")
        config = SGBMConfig(mode=modes[mode], padding_mode="reflect101" if padded else "none")
    start = perf_counter()
    forward = stereo_sgbm(left, right, calib, config)
    if external_lr:
        valid = left_right_consistent(left, right, forward, calib, lr_tolerance, config)
    else:
        valid = np.isfinite(forward) & (forward > 0)
    seconds = perf_counter()-start
    return forward, valid, seconds


def load_scene(folder):
    """Decode BGR first, preserving the frozen cvtColor rounding."""
    folder = Path(folder)
    left = cv2.imread(str(folder/"im0.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(folder/"im1.png"), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(f"Missing stereo images in {folder}")
    calib = calibration(folder/"calib.txt")
    target = read_pfm(folder/"disp0GT.pfm")
    expected = (calib["height"], calib["width"])
    if left.shape[:2] != expected or right.shape[:2] != expected or target.shape != expected:
        raise ValueError(f"{folder.name}: images/GT do not match calibration {expected}")
    label_path = folder/"mask0nocc.png"
    labels = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE) if label_path.exists() else None
    if labels is not None and (labels.shape != expected or not set(np.unique(labels)) <= {0, 128, 255}):
        raise ValueError(f"{folder.name}: unknown or wrong-sized Middlebury region labels")
    return left, right, target, labels, calib


def region_counts(pred, target, population, keep):
    valid = population & keep & np.isfinite(pred) & (pred > 0)
    errors = np.abs(pred[valid].astype(np.float64)-target[valid].astype(np.float64))
    denominator = int(population.sum())
    return {"gt_pixels": denominator, "retained_pixels": int(valid.sum()),
            "absolute_error_sum": float(errors.sum()),
            "good1_pixels": int(np.count_nonzero(errors <= 1.)),
            "mae_px": float(errors.mean()) if len(errors) else None,
            "coverage": float(valid.sum()/denominator) if denominator else None,
            "good1_yield": float(np.count_nonzero(errors <= 1.)/denominator) if denominator else None}


def _regions(target, labels):
    valid = np.isfinite(target) & (target > 0)
    result = {"all": valid}
    if labels is not None:
        result.update({name: valid & (labels == value) for name, value in
                       (("nonoccluded", 255), ("occluded", 128), ("unknown_label", 0))})
    return result


def plot_scene(rgb, disparity, valid, depth, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    axes[0].imshow(rgb)
    finite = depth[np.isfinite(depth)]
    maximum = float(np.percentile(finite, 98)) if finite.size else 1.
    images = [None,
              axes[1].imshow(np.ma.masked_where(~valid, disparity), cmap="magma"),
              axes[2].imshow(~valid, cmap="gray", vmin=0, vmax=1),
              axes[3].imshow(depth, cmap="viridis", vmin=0, vmax=maximum)]
    for ax, title in zip(axes, ["Left RGB", "Disparity (px)", "Invalid (white)", "Depth (m)"]):
        ax.set_title(title)
        ax.axis("off")
    for index in (1, 3):
        fig.colorbar(images[index], ax=axes[index], shrink=.75)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _configuration_plan(configuration, primary, block_size, uniqueness_ratio,
                        speckle_window_size, speckle_range, disp12_max_diff,
                        mode, padding_mode):
    common = dict(block_size=block_size, uniqueness_ratio=uniqueness_ratio,
                  speckle_window_size=speckle_window_size, speckle_range=speckle_range,
                  disp12_max_diff=disp12_max_diff)
    if mode is not None or padding_mode is not None:
        return {"custom": SGBMConfig(**common, mode=mode or "SGBM_3WAY",
                                     padding_mode=padding_mode or "reflect101")}, "custom"
    names = DEFAULT_CONFIGURATIONS if configuration == "all" else (configuration,)
    if any(name not in CONFIGURATIONS for name in names):
        raise ValueError("Unknown stereo configuration")
    modes = {getattr(cv2, "STEREO_SGBM_MODE_"+name): name
             for name in ("SGBM_3WAY", "HH4", "HH")}
    configs = {name: SGBMConfig(**common, mode=modes[CONFIGURATIONS[name][0]],
               padding_mode="reflect101" if CONFIGURATIONS[name][1] else "none")
               for name in names}
    if primary not in configs:
        primary = names[0]
    return configs, primary


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _region_summary(rows):
    result = []
    frame = pd.DataFrame(rows)
    for (configuration, region), group in frame.groupby(["configuration", "region"], sort=True):
        gt = int(group["gt_pixels"].sum())
        retained = int(group["retained_pixels"].sum())
        def mean(key):
            values = group[key].dropna()
            return float(values.mean()) if len(values) else None
        result.append({"configuration": configuration, "region": region,
            "scenes_with_gt": int((group["gt_pixels"] > 0).sum()),
            "scene_equal_mae_px": mean("mae_px"),
            "scene_equal_coverage": mean("coverage"),
            "scene_equal_good1_yield": mean("good1_yield"),
            "pooled_mae_px": float(group["absolute_error_sum"].sum()/retained) if retained else None,
            "pooled_coverage": retained/gt if gt else None,
            "pooled_good1_yield": float(group["good1_pixels"].sum()/gt) if gt else None,
            "gt_pixels": gt, "retained_pixels": retained})
    return result


def benchmark_cpu(pairs, configs, threads=4, passes=3, lr_tolerance=1., external_lr=True):
    """Preloaded RGB, full warmup, seeded scene permutations, alternating order."""
    if passes < 1:
        raise ValueError("Benchmark passes must be positive")
    cv2.setNumThreads(threads)
    counts = {}
    for name, cfg in configs.items():
        for scene, left, right, _, _, calib in pairs:
            _, valid, _ = match_pair(left, right, calib, config=cfg,
                lr_tolerance=lr_tolerance, external_lr=external_lr)
            counts[(name, scene.name)] = int(valid.sum())
    rng = np.random.default_rng(20261031)
    rows = []
    for repeat in range(passes):
        order = rng.permutation(len(pairs))
        methods = list(configs) if repeat % 2 == 0 else list(configs)[::-1]
        for name in methods:
            for index in order:
                scene, left, right, _, _, calib = pairs[int(index)]
                _, valid, seconds = match_pair(left, right, calib, config=configs[name],
                    lr_tolerance=lr_tolerance, external_lr=external_lr)
                count = int(valid.sum())
                if count != counts[(name, scene.name)]:
                    raise ValueError("Nondeterministic validity across benchmark passes")
                rows.append({"configuration": name, "pass": repeat, "scene": scene.name,
                             "milliseconds": seconds*1000, "valid_pixel_count": count})
    summary = {}
    for name in configs:
        selected = [row for row in rows if row["configuration"] == name]
        values = np.array([row["milliseconds"] for row in selected])
        summary[name] = {"mean_ms_per_pair": float(values.mean()),
                         "median_ms_per_pair": float(np.median(values)),
                         "p95_ms_per_pair": float(np.percentile(values, 95)),
                         "per_pass_mean_ms": [float(np.mean([row["milliseconds"]
                            for row in selected if row["pass"] == repeat])) for repeat in range(passes)]}
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    peak = None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if platform.system() != "Darwin" else 1024**2)
    except ImportError:
        pass
    return {"scene_count": len(pairs), "timed_passes_per_method": passes,
            "measurements_per_method": len(pairs)*passes, "seed": 20261031,
            "cpu_model": cpu, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "timing_scope": TIMING_SCOPE if external_lr else TIMING_SCOPE.replace(
                " + reverse SGBM + explicit LR residual/mask", " without external LR"),
            "schedule": "Preload RGB inputs; warm each scene/method once; identical seeded scene permutations; method order reverses on alternate passes; serial computation.",
            "memory_scope": "Current process high-water RSS, including prior run/figures and all configurations; not comparable to the historical isolated-worker memory experiment.",
            "peak_process_rss_mib": peak, "methods": summary, "measurements": rows}


def run(data_root, output, limit=None, threads=4, *, configuration="all",
        primary=PRIMARY_CONFIGURATION, block_size=5, uniqueness_ratio=10,
        speckle_window_size=100, speckle_range=2, disp12_max_diff=1,
        mode=None, padding_mode=None, lr_tolerance=1., external_lr=True,
        no_images=False, benchmark_passes=0):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive, or omitted for all scenes")
    if threads < 1 or benchmark_passes < 0:
        raise ValueError("threads must be positive and benchmark_passes nonnegative")
    if not np.isfinite(lr_tolerance) or lr_tolerance <= 0:
        raise ValueError("LR tolerance must be finite and positive")
    configs, primary = _configuration_plan(configuration, primary, block_size,
        uniqueness_ratio, speckle_window_size, speckle_range, disp12_max_diff, mode, padding_mode)
    cv2.setNumThreads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not no_images:
        (output/"figures/stereo").mkdir(parents=True, exist_ok=True)
    scene_root = Path(data_root)/"MiddEval3/trainingQ"
    folders = sorted(path for path in scene_root.iterdir()
                     if path.is_dir() and (path/"calib.txt").is_file())
    if not folders:
        raise ValueError("No Middlebury trainingQ stereo scenes found")
    available_scenes = len(folders)
    folders = folders[:limit]
    pairs = [(folder, *load_scene(folder)) for folder in folders]
    records, regions, paired, sensitivity, hashes = [], [], [], [], {}
    for folder, left, right, target, labels, calib in pairs:
        populations = _regions(target, labels)
        gt_valid = populations["all"]
        if not gt_valid.any():
            raise ValueError(f"{folder.name}: no finite positive ground-truth disparity")
        predictions = {}
        for name, cfg in configs.items():
            disparity, valid, seconds = match_pair(left, right, calib, config=cfg,
                lr_tolerance=lr_tolerance, external_lr=external_lr)
            depth = disparity_to_depth(disparity, calib["focal"], calib["baseline"], calib["doffs"])
            depth[~valid] = np.nan
            predictions[name] = (disparity, valid)
            destination = output/"arrays"/name
            destination.mkdir(parents=True, exist_ok=True)
            for suffix, array in (("disparity", disparity), ("valid", valid), ("depth", depth)):
                np.save(destination/f"{folder.name}_{suffix}.npy", array)
                if name == primary:
                    np.save(output/f"{folder.name}_{suffix}.npy", array)
            intersection = valid & gt_valid
            counts = region_counts(disparity, target, gt_valid, valid)
            records.append({"scene": folder.name, "configuration": name,
                "mae_px": float(np.abs(disparity[intersection]-target[intersection]).mean()) if intersection.any() else np.nan,
                "valid_fraction": float(intersection.sum()/gt_valid.sum()),
                "cpu_seconds": seconds, "threads": threads,
                "valid_pixels": counts["retained_pixels"], "gt_pixels": counts["gt_pixels"],
                "good1_yield": counts["good1_yield"], "primary": name == primary})
            for region, population in populations.items():
                regions.append({"scene": folder.name, "configuration": name, "region": region,
                                **region_counts(disparity, target, population, valid)})
            if name == primary and not no_images:
                plot_scene(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), disparity, valid, depth,
                           output/"figures/stereo"/f"{folder.name}.png")
        reference, reference_keep = predictions[primary]
        for name, (prediction, keep) in predictions.items():
            for region, population in populations.items():
                for kind, chosen in (("common", keep & reference_keep),
                                     ("added", keep & ~reference_keep),
                                     ("lost", reference_keep & ~keep)):
                    selected = population & chosen
                    paired.append({"scene": folder.name, "configuration": name,
                        "reference": primary, "region": region, "population": kind,
                        "pixels": int(selected.sum()),
                        "reference_mae_px": float(np.abs(reference[selected].astype(np.float64)-target[selected]).mean())
                            if selected.any() and kind != "added" else None,
                        "configuration_mae_px": float(np.abs(prediction[selected].astype(np.float64)-target[selected]).mean())
                            if selected.any() and kind != "lost" else None})
        for distance in (1., 2., 4., 8.):
            sensitivity.append({"scene": folder.name, "distance_m": distance,
                "depth_change_m_per_disparity_pixel": distance**2/(calib["focal"]*calib["baseline"]/1000.)})
        for name in ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm", "mask0nocc.png"):
            path = folder/name
            if path.exists():
                hashes[f"{folder.name}/{name}"] = _digest(path)
        print("Stereo", folder.name, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/"stereo_per_scene.csv", index=False)
    means = frame.groupby("configuration")[["mae_px", "valid_fraction", "cpu_seconds"]].mean()
    means.to_csv(output/"stereo_summary.csv")
    pd.DataFrame(regions).to_csv(output/"stereo_regions_per_scene.csv", index=False)
    pd.DataFrame(_region_summary(regions)).to_csv(output/"stereo_regions_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output/"stereo_paired_populations.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output/"disparity_sensitivity.csv", index=False)
    runtime = {"opencv": cv2.__version__, "numpy": np.__version__,
        "threads": threads, "scope": TIMING_SCOPE if external_lr else
            "Forward SGBM including grayscale conversion/padding/bound rejection; external LR disabled; excludes IO, metrics, depth conversion and saving.",
        "primary_configuration": primary, "configurations": {name: cfg.metadata() for name, cfg in configs.items()},
        "external_lr": external_lr, "lr_tolerance": lr_tolerance,
        "scene_count": len(folders), "available_scene_count": available_scenes,
        "complete_course_set": len(folders) == available_scenes == 15,
        "aggregation": "Equal scene weighting; MAE on valid-prediction/finite-positive-GT intersection; coverage denominator is all finite-positive GT.",
        "region_policy": "Official 255 nonoccluded,128 occluded,0 unknown; descriptive only; main denominator unchanged. Missing optional labels produce all-region diagnostics only.",
        "arrays": "arrays/<configuration>/<scene>_{disparity,valid,depth}.npy; root-level aliases are the primary method.",
        "source_sha256": _digest(__file__), "input_sha256": hashes}
    (output/"stereo_runtime.json").write_text(json.dumps(runtime, indent=2, allow_nan=False))
    if benchmark_passes:
        benchmark = benchmark_cpu(pairs, configs, threads, benchmark_passes, lr_tolerance, external_lr)
        pd.DataFrame(benchmark["measurements"]).to_csv(output/"stereo_cpu_benchmark.csv", index=False)
        (output/"stereo_cpu_benchmark.json").write_text(json.dumps(benchmark, indent=2, allow_nan=False))
    print(means)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-scenes", type=int, help="Legacy alias: zero means all scenes")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--configuration", choices=["all", *CONFIGURATIONS], default="all")
    parser.add_argument("--primary", choices=list(CONFIGURATIONS), default=PRIMARY_CONFIGURATION)
    parser.add_argument("--mode", choices=["SGBM_3WAY", "HH4", "HH"],
                        help="Run one custom configuration instead of the named comparison")
    parser.add_argument("--padding-mode", "--padding", choices=["none", "replicate", "reflect101"])
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--speckle-window-size", type=int, default=100)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--disp12-max-diff", type=int, default=1)
    parser.add_argument("--lr-tolerance", type=float, default=1.)
    parser.add_argument("--no-external-lr", action="store_true")
    parser.add_argument("--no-images", "--no-visualizations", action="store_true")
    parser.add_argument("--benchmark-passes", type=int, default=0,
                        help="Optional preloaded, warmed serial CPU benchmark; three passes matches the frozen schedule")
    args = parser.parse_args()
    if args.max_scenes is not None:
        if args.limit is not None or args.max_scenes < 0:
            parser.error("Use one scene limit; max-scenes must be nonnegative")
        args.limit = args.max_scenes or None
    run(args.data_root, args.output, args.limit, args.threads,
        configuration=args.configuration, primary=args.primary, block_size=args.block_size,
        uniqueness_ratio=args.uniqueness_ratio, speckle_window_size=args.speckle_window_size,
        speckle_range=args.speckle_range, disp12_max_diff=args.disp12_max_diff,
        mode=args.mode, padding_mode=args.padding_mode, lr_tolerance=args.lr_tolerance,
        external_lr=not args.no_external_lr, no_images=args.no_images,
        benchmark_passes=args.benchmark_passes)
