import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch

INPUT_SIZE = 518
MAX_FRAMES = 100
MAX_RESOLUTION = 640
REVISIONS = {"image": "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
             "video": "4f5ae23172ba60fd7bc11ef671cca678842c7072"}


def decode(path, output, limit):
    reader = cv2.VideoCapture(str(path))
    fps = reader.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < limit:
        ok, frame = reader.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = min(1, MAX_RESOLUTION / max(h, w))
        frames.append(cv2.cvtColor(cv2.resize(frame, (round(w * scale), round(h * scale))), cv2.COLOR_BGR2RGB))
    reader.release()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "frames.npy", np.stack(frames))
    (output / "frames.json").write_text(json.dumps({"frames": len(frames), "fps": fps,
        "original_hw": [h, w], "decoded_hw": list(frames[0].shape[:2])}, indent=2))
    print("Decoded", len(frames), "frames at", fps, "FPS", flush=True)


def infer(args):
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(4)
    shape = {}
    if args.method == "image":
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        encoder = model.pretrained
    else:
        from video_depth_anything.video_depth import VideoDepthAnything
        model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        encoder = model.pretrained
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    hook = encoder.patch_embed.register_forward_pre_hook(
        lambda module, inputs: shape.update(network_hw=list(inputs[0].shape[-2:])))
    frames = np.load(args.frames / "frames.npy")
    fps = json.loads((args.frames / "frames.json").read_text())["fps"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.method == "image":
        image = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
        for _ in range(5):
            model.infer_image(image, INPUT_SIZE)
        torch.cuda.synchronize()
        predictions, seconds = [], []
        torch.cuda.reset_peak_memory_stats()
        for frame in frames:
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            start = perf_counter()
            depth = model.infer_image(image, INPUT_SIZE)
            torch.cuda.synchronize()
            seconds.append(perf_counter() - start)
            predictions.append(depth)
        depth = np.stack(predictions)
        precision = "FP32"
        scope = "infer_image, including preprocessing and CPU output transfer; excludes decoding and encoding"
    else:
        model.infer_video_depth(frames[:5], fps, input_size=INPUT_SIZE, fp32=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        depth, _ = model.infer_video_depth(frames, fps, input_size=INPUT_SIZE, fp32=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        seconds = [elapsed / len(frames)] * len(frames)
        precision = "official FP16 autocast"
        scope = "offline clip inference divided by decoded frames, including window overlap, preprocessing and CPU transfer"
    np.savez_compressed(args.output / "depth.npz", depth=depth.astype(np.float32))
    pd.DataFrame({"frame": np.arange(len(frames)), "seconds": seconds}).to_csv(args.output / "inference_per_frame.csv", index=False)
    metadata = {"method": args.method, "source_revision": REVISIONS[args.method],
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "precision": precision, "input_size": INPUT_SIZE,
                **shape, "decoded_hw": list(frames.shape[1:3]), "frames": len(frames), "fps": fps,
                "mean_seconds": float(np.mean(seconds)),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "scope": scope, "warmup": "five frames; video model pads internally to a complete window"}
    (args.output / "runtime.json").write_text(json.dumps(metadata, indent=2))
    hook.remove()
    print(metadata, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["decode", "image", "video"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=MAX_FRAMES)
    args = parser.parse_args()
    if args.method == "decode":
        decode(args.video, args.output, args.limit)
    else:
        infer(args)
