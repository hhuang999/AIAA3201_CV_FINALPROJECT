import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from .metrics import METRIC_NAMES


def summarize(args):
    runtime_folder = args.results / "runtime"
    runtime_folder.mkdir(parents=True, exist_ok=True)
    records = []
    folders = [("nyu_rgb", "UniDepthV2 RGB-only"), ("nyu_known", "UniDepthV2 known camera, training"),
               ("video_image", "Depth Anything V2 Small"), ("video_temporal", "Video Depth Anything Small")]
    for folder, name in folders:
        source = args.outputs / folder
        runtime = json.loads((source / "runtime.json").read_text())
        shutil.copyfile(source / "runtime.json", runtime_folder / f"{folder}.json")
        timing = source / ("inference_per_frame.csv" if folder.startswith("video") else "inference_per_image.csv")
        shutil.copyfile(timing, runtime_folder / f"{folder}_timing.csv")
        records.append({"model": name, "gpu": runtime["gpu"], "precision": runtime["precision"],
                        "network_hw": "x".join(map(str, runtime["network_hw"])),
                        "seconds_per_image_or_frame": runtime["mean_seconds"],
                        "peak_allocated_mib": runtime["peak_allocated_mib"],
                        "peak_reserved_mib": runtime["peak_reserved_mib"]})
    efficiency = pd.DataFrame(records)
    efficiency.to_csv(args.results / "efficiency_summary.csv", index=False)
    oof = pd.read_csv(args.results / "calibration_oof_per_image.csv")
    oof.groupby(["fold", "method"])[list(METRIC_NAMES)].mean().to_csv(
        args.results / "calibration_oof_per_fold.csv")
    timing = pd.read_csv(args.results / "postprocessing_per_image.csv")
    timing.drop(columns="index").mean().rename("mean_seconds").to_csv(
        args.results / "postprocessing_summary.csv")
    sections = []
    for title, name in [("Stereo: equal-weight mean over 15 scenes", "stereo_summary.csv"),
                        ("NYU: 654 held-out images, RGB only", "test_summary.csv"),
                        ("Scene-grouped training out-of-fold scores", "calibration_oof_summary.csv"),
                        ("Training-only camera diagnosis: 795 images", "camera_diagnostic_summary.csv")]:
        frame = pd.read_csv(args.results / name)
        sections.append(f"### {title}\n\n" + frame.to_markdown(index=False, floatfmt=".4f"))
    sections.append("### Efficiency\n\n" + efficiency.to_markdown(index=False, floatfmt=".4f"))
    sections.append("Times use each model's reported inference scope; offline video time is amortized over the 70 decoded frames. Peak memory is allocated GPU memory; reserved memory is also retained in the CSV and runtime JSON files.")
    params = json.loads((args.results / "calibration_parameters.json").read_text())
    if params["residual"] is None:
        sections.append("The final method is the monotone curve. No residual candidate passed all metrics in every fold. The residual stage is inactive; its recorded incremental runtime is zero.")
    changes = pd.read_csv(args.results / "figures/video/video_adjacent_change.csv")
    changes.groupby("method").mean(numeric_only=True).drop(columns="frame").to_csv(
        args.results / "video_change_summary.csv")
    sections.append("### Video: mean adjacent normalized-depth change\n\n" +
                    changes.groupby("method")["mean_adjacent_change"].mean().to_frame().to_markdown(floatfmt=".6f"))
    markdown = "\n\n".join(sections)
    (args.results / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    if args.readme is not None:
        content = args.readme.read_text(encoding="utf-8")
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        before = content.split(start)[0]
        after = content.split(end)[1]
        args.readme.write_text(before + start + "\n\n" + markdown + "\n\n" + end + after, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path)
    summarize(parser.parse_args())
