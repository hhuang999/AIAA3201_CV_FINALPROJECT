import argparse
import json
from pathlib import Path
import zipfile

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .infer_video import load_frames, load_video_predictions

FOCUS = 0.55
BLUR_LEVELS = 6
MAX_SIGMA = 9.
PALETTE = "inferno"


def normalize_clip(depth):
    depth = np.asarray(depth)
    if depth.ndim != 3 or min(depth.shape) <= 0 or not np.isfinite(depth).all():
        raise ValueError("Depth clip must be nonempty finite [T,H,W]")
    low, high = np.percentile(depth, [2, 98])
    return np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1), (float(low), float(high))


def background_blur(rgb, normalized):
    levels = [rgb.astype(np.float32)]
    levels += [cv2.GaussianBlur(rgb, (0, 0), float(sigma)).astype(np.float32)
               for sigma in np.linspace(0, MAX_SIGMA, BLUR_LEVELS)[1:]]
    strength = np.clip((FOCUS - normalized) / FOCUS, 0, 1) * (BLUR_LEVELS - 1)
    lower = np.floor(strength).astype(int)
    upper = np.minimum(lower + 1, BLUR_LEVELS - 1)
    weight = (strength - lower)[..., None]
    y, x = np.indices(normalized.shape)
    stack = np.stack(levels)
    return np.clip((1 - weight) * stack[lower, y, x] + weight * stack[upper, y, x], 0, 255).astype(np.uint8)


def labeled_panels(panels, labels, frame, total, fps):
    image = np.concatenate(panels, axis=1)
    canvas = cv2.copyMakeBorder(image, 36, 28, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    width = panels[0].shape[1]
    for i, label in enumerate(labels):
        cv2.putText(canvas, label, (i * width + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
    footer = f"Relative depth: independent clip-wide ranges; no shared metric scale. Frame {frame + 1}/{total}, {fps:g} FPS"
    cv2.putText(canvas, footer, (12, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def render(args):
    frames, info = load_frames(args.frames)
    image, image_manifest = load_video_predictions(args.image_predictions, frames, info, "image")
    video, video_manifest = load_video_predictions(args.video_predictions, frames, info, "video")
    if len(frames) < 2:
        raise ValueError("Rendering comparisons requires at least two frames")
    image_norm, image_bounds = normalize_clip(image)
    video_norm, video_bounds = normalize_clip(video)
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    bounds = {"image": image_bounds, "video": video_bounds,
              "percentiles": [2, 98], "frames": len(frames), "fps": info["fps"],
              "focus": FOCUS, "blur_levels": BLUR_LEVELS, "max_sigma_pixels": MAX_SIGMA,
              "decoded_frames_sha256": info["decoded_frames_sha256"],
              "image_depth_archive_sha256": image_manifest["depth_archive_sha256"],
              "video_depth_archive_sha256": video_manifest["depth_archive_sha256"]}
    (args.results / "video_display.json").write_text(json.dumps(bounds, indent=2))
    temporal = []
    for name, depth in (("framewise", image_norm), ("video", video_norm)):
        for i, change in enumerate(np.abs(np.diff(depth, axis=0)).mean(axis=(1, 2)), start=1):
            temporal.append({"method": name, "frame": i, "mean_adjacent_change": change})
    pd.DataFrame(temporal).to_csv(args.results / "video_adjacent_change.csv", index=False)
    comparison_path = args.output / "relative_depth_comparison.mp4"
    application_path = args.output / "depth_aware_blur.mp4"
    kwargs = {"fps": info["fps"], "codec": "libx264", "macro_block_size": 1,
              "pixelformat": "yuv420p", "ffmpeg_params": ["-crf", "18"]}
    palette = (np.asarray(matplotlib.colormaps[PALETTE].colors) * 255).astype(np.uint8)
    selected = {0, len(frames) // 2, len(frames) - 1}
    with imageio.get_writer(comparison_path, **kwargs) as comparison, imageio.get_writer(application_path, **kwargs) as application:
        for i, rgb in enumerate(frames):
            colored = [palette[(norm[i] * 255).astype(np.uint8)] for norm in (image_norm, video_norm)]
            panel = labeled_panels([rgb, *colored], ["RGB", "Framewise relative depth", "Video relative depth"],
                                   i, len(frames), info["fps"])
            comparison.append_data(panel)
            blurred = [background_blur(rgb, norm[i]) for norm in (image_norm, video_norm)]
            app = labeled_panels([rgb, *blurred], ["RGB", "Framewise background blur", "Video background blur"],
                                 i, len(frames), info["fps"])
            application.append_data(app)
            if i in selected:
                imageio.imwrite(args.results / f"video_frame_{i:02d}.png", panel)
                imageio.imwrite(args.results / f"application_frame_{i:02d}.png", app)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), layout="constrained")
    fig.suptitle(f"{len(frames)} frames, {info['fps']:g} FPS; independent fixed relative-depth ranges")
    for ax, depth, label in zip(axes, (image_norm, video_norm), ("Framewise", "Video")):
        ax.imshow(depth[:, depth.shape[1] // 2, :], cmap=PALETTE, vmin=0, vmax=1, aspect="auto")
        ax.set(title=f"{label}: center-row depth over time", xlabel="Horizontal pixel", ylabel="Frame")
    fig.savefig(args.results / "video_temporal_slices.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.8), layout="constrained")
    fig.suptitle(f"{len(frames)} frames, {info['fps']:g} FPS; adjacent changes include motion and flicker")
    for col, index in enumerate((min(15, len(frames) - 2), min(35, len(frames) - 2), len(frames) - 2)):
        for row, depth in enumerate((image_norm, video_norm)):
            axes[row, col].imshow(depth[index + 1] - depth[index], cmap="coolwarm", vmin=-.15, vmax=.15)
            axes[row, col].set_title(f"{'Framewise' if row == 0 else 'Video'}: {index} to {index + 1}")
            axes[row, col].axis("off")
    fig.savefig(args.results / "video_adjacent_frames.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), layout="constrained")
    middle = len(frames) // 2
    zoom = (slice(70, 250), slice(200, 450))
    for row, index in enumerate((max(middle - 1, 0), middle, min(middle + 1, len(frames) - 1))):
        axes[row, 0].imshow(frames[index][zoom])
        for col, depth in enumerate((image_norm, video_norm), start=1):
            axes[row, col].imshow(depth[index][zoom], cmap=PALETTE, vmin=0, vmax=1)
        for ax, label in zip(axes[row], ("RGB", "Framewise", "Video")):
            ax.set_title(f"{label}: frame {index}")
            ax.axis("off")
    fig.suptitle(f"{len(frames)} frames, {info['fps']:g} FPS; zoom x=200:450, y=70:250; fixed relative-depth ranges")
    fig.savefig(args.results / "video_boundary_zoom.png", dpi=180)
    plt.close(fig)
    with zipfile.ZipFile(args.output / "depth_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(comparison_path, comparison_path.name)
        archive.write(application_path, application_path.name)
    print("Rendered", len(frames), "frames at", info["fps"], "FPS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--image-predictions", type=Path, required=True)
    parser.add_argument("--video-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/delivery"))
    parser.add_argument("--results", type=Path, default=Path("results/figures/video"))
    render(parser.parse_args())
