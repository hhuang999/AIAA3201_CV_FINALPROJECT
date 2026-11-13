"""Same decoded RGB input and pinned Small-model video prediction caches."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch

from .cache import (array_sha256, atomic_json, cache_lock, sha256,
                    require_pinned_weights, source_identity, weights_identity)
from .infer_nyu import deterministic_setup, resolve_device
from .inference_backend import use_pytorch_attention

INPUT_SIZE = 518
MAX_FRAMES = 100
MAX_RESOLUTION = 640
REVISIONS = {"image": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
             "video": "4f5ae23172ba60fd7bc11ef671cca678842c7072"}
PINNED_WEIGHTS = {
    "image": "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
    "video": "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
}


def validate_frames(frames, info):
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or min(frames.shape[:3]) <= 0:
        raise ValueError("Decoded frames must be nonempty uint8 [T,H,W,3] RGB")
    fps = info.get("fps")
    if not isinstance(fps, (float, int)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    if info.get("frames") != len(frames) or info.get("decoded_hw") != list(frames.shape[1:3]):
        raise ValueError("Decoded RGB length/shape disagrees with its manifest")
    if info.get("decoded_frames_sha256") != array_sha256(frames):
        raise ValueError("Decoded RGB pixels differ from the recorded SHA256")
    if not info.get("input_sha256"):
        raise ValueError("Decoded frames have no source-video identity")


def load_frames(folder):
    folder = Path(folder)
    frames = np.load(folder / "frames.npy", mmap_mode="r", allow_pickle=False)
    info = json.loads((folder / "frames.json").read_text())
    validate_frames(frames, info)
    return frames, info


def validate_depth(depth, frames):
    if depth.shape != frames.shape[:3] or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise ValueError("Predicted depth must be finite float32 with the decoded frame count and shape")


def load_video_predictions(folder, frames, info, expected_method=None):
    """Read only complete archives explicitly bound to the same RGB pixels."""
    validate_frames(frames, info)
    folder = Path(folder)
    manifest = json.loads((folder / "prediction_manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("schema_version") != 1:
        raise ValueError("Video predictions lack a verified complete marker")
    if expected_method is not None and manifest.get("method") != expected_method:
        raise ValueError("Video prediction method differs")
    for key in ("decoded_frames_sha256", "input_sha256", "frames", "decoded_hw", "fps"):
        if manifest.get(key) != info.get(key):
            raise ValueError(f"Video prediction RGB identity mismatch: {key}")
    if manifest.get("depth_archive_sha256") != sha256(folder / "depth.npz"):
        raise ValueError("Video prediction archive SHA256 differs")
    if not all(key in manifest for key in ("source", "weights", "attention_backend", "preprocessing", "seed")):
        raise ValueError("Video prediction provenance is incomplete")
    with np.load(folder / "depth.npz", allow_pickle=False) as data:
        depth, indices, fps = data["depth"], data["frame_indices"], data["fps"]
    if not np.issubdtype(indices.dtype, np.integer) or not np.array_equal(indices, np.arange(len(frames))):
        raise ValueError("Video prediction frame IDs are missing, repeated or out of order")
    if fps.shape != () or not np.isfinite(fps) or abs(float(fps) - info["fps"]) > 1e-5:
        raise ValueError("Video prediction frame rate differs")
    validate_depth(depth, frames)
    return depth, manifest


def decode(path, output, limit):
    if limit <= 0:
        raise ValueError("Frame limit must be positive")
    reader = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not reader.isOpened():
            raise ValueError(f"Cannot open input video: {path}")
        fps = float(reader.get(cv2.CAP_PROP_FPS))
        source_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or source_count <= 0:
            raise ValueError("Video needs a finite positive frame rate and frame count")
        while len(frames) < min(limit, source_count):
            ok, frame = reader.read()
            if not ok:
                raise ValueError("Video ended before the recorded source frame count")
            h, w = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if max(h, w) > MAX_RESOLUTION:
                scale = MAX_RESOLUTION / max(h, w)
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
    finally:
        reader.release()
    if not frames:
        raise ValueError("Empty input video")
    frames = np.stack(frames)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("frames.npy", "frames.json")):
        raise ValueError("Decoded frame artifacts already exist; use a new output directory")
    np.save(output / "frames.npy", frames)
    atomic_json(output / "frames.json", {"frames": len(frames), "fps": fps, "source_frames": source_count,
        "original_hw": [h, w], "decoded_hw": list(frames.shape[1:3]), "input_sha256": sha256(path),
        "decoded_frames_sha256": array_sha256(frames), "requested_max_len": limit,
        "max_resolution": MAX_RESOLUTION, "decoder": "OpenCV VideoCapture; BGR to RGB; INTER_LINEAR resize"})
    print("Decoded", len(frames), "frames at", fps, "FPS", flush=True)


def save_depth_archive(folder, depth, fps):
    fd, temporary = tempfile.mkstemp(prefix=".depth.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, depth=depth, frame_indices=np.arange(len(depth)), fps=float(fps))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / "depth.npz")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def infer(args):
    for key in ("source", "weights", "frames"):
        if getattr(args, key) is None:
            raise ValueError(f"--{key} is required for inference")
    deterministic_setup()
    frames, info = load_frames(args.frames)
    source = source_identity(args.source, REVISIONS[args.method])
    weights = weights_identity(args.weights)
    require_pinned_weights(weights, {args.weights.name: PINNED_WEIGHTS[args.method]})
    args.output.mkdir(parents=True, exist_ok=True)
    with cache_lock(args.output / ".writer.lock"):
        if (args.output / "prediction_manifest.json").exists():
            _, manifest = load_video_predictions(args.output, frames, info, args.method)
            if manifest["source"] != source or manifest["weights"] != weights:
                raise ValueError("Existing video prediction source/checkpoint differs")
            print("Already complete; input and archive SHA256 verified without loading model", flush=True)
            return
        if (args.output / "depth.npz").exists():
            raise ValueError("Unbound prediction archive exists; use a new output directory")
        sys.path.insert(0, str(args.source.resolve()))
        backend = use_pytorch_attention("dav2" if args.method == "image" else "vda")
        if args.method == "image":
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        else:
            from video_depth_anything.video_depth import VideoDepthAnything
            model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=False)
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = model.to(device).eval()
        shape = {}
        hook = model.pretrained.patch_embed.register_forward_pre_hook(lambda module, inputs: shape.update(network_hw=list(inputs[0].shape[-2:])))
        fps = info["fps"]
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        def reset():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        try:
            with torch.inference_mode():
                if args.method == "image":
                    model.infer_image(cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR), INPUT_SIZE)
                    sync()
                    reset()
                    predictions, seconds = [], []
                    for frame in frames:
                        sync()
                        start = perf_counter()
                        predictions.append(model.infer_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), INPUT_SIZE))
                        sync()
                        seconds.append(perf_counter() - start)
                    depth = np.stack(predictions).astype(np.float32)
                    precision, warmup = "FP32", 1
                    scope = "synchronized infer_image including preprocessing and CPU output transfer; excludes decoding and encoding"
                else:
                    warmup = min(32, len(frames))
                    fp32 = device.type != "cuda"
                    model.infer_video_depth(frames[:warmup], fps, input_size=INPUT_SIZE, device=device.type, fp32=fp32)
                    sync()
                    reset()
                    start = perf_counter()
                    depth, output_fps = model.infer_video_depth(frames, fps, input_size=INPUT_SIZE, device=device.type, fp32=fp32)
                    sync()
                    if not np.isfinite(output_fps) or abs(float(output_fps) - fps) > 1e-5:
                        raise ValueError("Video model changed the input frame rate")
                    seconds = [(perf_counter() - start) / len(frames)] * len(frames)
                    depth = np.asarray(depth, dtype=np.float32)
                    precision = "FP32" if fp32 else "official FP16 autocast"
                    scope = "synchronized offline clip inference divided by frames, including window overlap, preprocessing and CPU output transfer"
        finally:
            hook.remove()
        validate_depth(depth, frames)
        save_depth_archive(args.output, depth, fps)
        pd.DataFrame({"frame": np.arange(len(frames)), "seconds": seconds}).to_csv(args.output / "inference_per_frame.csv", index=False)
        metadata = {"method": args.method, "source_revision": REVISIONS[args.method], "source": source, "weights": weights,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "torch": torch.__version__,
                    "cuda": torch.version.cuda, "precision": precision, "input_size": INPUT_SIZE, **shape,
                    **{key: info[key] for key in ("frames", "fps", "decoded_hw", "decoded_frames_sha256", "input_sha256")},
                    "mean_seconds": float(np.mean(seconds)), "attention_backend": backend, "seed": 0,
                    "preprocessing": "official infer_image / infer_video_depth; same decoded uint8 RGB; input_size=518",
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
                    "scope": scope, "warmup_frames": warmup}
        atomic_json(args.output / "runtime.json", metadata)
        atomic_json(args.output / "prediction_manifest.json", {**metadata, "schema_version": 1, "complete": True,
                    "depth_archive_sha256": sha256(args.output / "depth.npz")})
        print(metadata, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["decode", "image", "video"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=MAX_FRAMES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.method == "decode":
        if args.video is None:
            parser.error("--video is required for decode")
        decode(args.video, args.output, args.limit)
    else:
        infer(args)
