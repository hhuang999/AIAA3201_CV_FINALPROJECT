"""Pinned UniDepth inference with compact dataset-ID aware resumable caches."""
import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch

from .cache import (atomic_json, cache_lock, chunk_record, load_predictions,
                    require_pinned_weights, sha256, source_identity, weights_identity, write_complete_manifest)
from .data import nyu_image, open_nyu, splits
from .inference_backend import use_pytorch_attention

RGB_INTRINSICS = np.array([[518.8579011745019, 0, 325.58244941119034],
                           [0, 519.4696111212749, 253.73616633400465],
                           [0, 0, 1]], dtype=np.float32)
UPSTREAM_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
CHECKPOINT_EVERY = 25
PINNED_WEIGHTS = {
    "config.json": "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe",
    "model.safetensors": "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648",
}


def deterministic_setup(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(2)


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def infer_with_camera(model, rgb, known_intrinsics):
    # Upstream infer mutates camera intrinsics: construct a fresh tensor each call.
    if known_intrinsics:
        return model.infer(rgb, torch.from_numpy(RGB_INTRINSICS.copy()).unsqueeze(0))
    return model.infer(rgb)


def commit_checkpoint(folder, depths, intrinsics, manifest):
    for name, array in (("depth.npy", depths), ("intrinsics.npy", intrinsics)):
        array.flush()
        with (folder / name).open("rb") as stream:
            import os
            os.fsync(stream.fileno())
    atomic_json(folder / "manifest.json", manifest)


def finalize_cache(folder, signature, runtime, records, network_hw=None):
    if len(records) != len(signature["indices"]) or [r.get("index") for r in records] != signature["indices"]:
        raise ValueError("Completed timing records disagree with cache indices")
    if any(not np.isfinite(r["seconds"]) or r["seconds"] <= 0 for r in records):
        raise ValueError("Invalid persisted inference timing")
    pd.DataFrame(records).to_csv(folder / "inference_per_image.csv", index=False)
    metadata = {**runtime, "network_hw": network_hw, "model": "UniDepthV2 ViT-S",
                "source_revision": UPSTREAM_REVISION, "rgb_input_hw": [480, 640],
                "known_intrinsics": signature["known_intrinsics"], "images": len(records),
                "mean_seconds": float(np.mean([r["seconds"] for r in records])), "warmup_images": 2,
                "scope": "synchronized infer call including preprocessing/input transfer; excludes output CPU transfer and file IO"}
    for key in ("peak_allocated_mib", "peak_reserved_mib"):
        values = [r[key] for r in records if r[key] is not None]
        metadata[key] = max(values) if values else None
    atomic_json(folder / "runtime.json", metadata)
    write_complete_manifest(folder, signature, runtime)


def infer(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    deterministic_setup()
    source = source_identity(args.source, UPSTREAM_REVISION)
    weights = weights_identity(args.weights)
    require_pinned_weights(weights, PINNED_WEIGHTS)
    data = {"nyu_mat_sha256": sha256(args.data_root / "nyu_depth_v2_labeled.mat"),
            "splits_sha256": sha256(args.data_root / "splits.mat")}
    args.output.mkdir(parents=True, exist_ok=True)
    with cache_lock(args.output / ".writer.lock"), open_nyu(args.data_root) as handle:
        indices = np.arange(len(handle["images"])) if args.split == "all" else splits(args.data_root)[args.split]
        indices = indices[:args.limit]
        if len(indices) == 0:
            raise ValueError("Empty inference split")
        signature = {"indices": indices.tolist(), "shape": [len(indices), 480, 640], "dtype": "float32",
                     "known_intrinsics": bool(args.known_intrinsics), "split": args.split,
                     "source": source, "weights": weights, "data": data, "seed": 0,
                     "backend": "upstream PyTorch attention branches; xformers disabled",
                     "preprocessing": "official infer; interpolation_mode=bilinear; RGB uint8 480x640",
                     "camera_policy": "fresh K tensor per call" if args.known_intrinsics else "RGB only"}
        old = None
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            cached = load_predictions(args.output, expected_known_intrinsics=args.known_intrinsics)
            old = cached.metadata
            if old.get("signature") != signature:
                raise ValueError("Cache identity differs; use a new output directory")
            if old.get("complete") is True:
                load_predictions(args.output, require_complete=True)
                print("Already complete; checksums verified without loading model", flush=True)
                return
            if old["completed"] == len(indices):
                finalize_cache(args.output, signature, old["runtime_signature"], old["records"], old.get("network_hw"))
                print("Recovered completed cache publication without loading model", flush=True)
                return
        elif any((args.output / name).exists() for name in ("depth.npy", "intrinsics.npy", "indices.npy", "runtime.json")):
            raise ValueError("Unbound/legacy artifacts exist; preserve them and use a new output directory")
        sys.path.insert(0, str(args.source.resolve()))
        from unidepth.models import UniDepthV2
        backend = use_pytorch_attention("unidepth")
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = UniDepthV2.from_pretrained(str(args.weights.resolve())).to(device).eval()
        model.interpolation_mode = "bilinear"
        runtime = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "attention_backend": backend, "shape_constraints": model.shape_constraints,
                   "precision": "official FP16 CUDA autocast / FP32 weights" if device.type == "cuda" else "FP32 CPU"}
        runtime = json.loads(json.dumps(runtime))
        if old is not None and old.get("runtime_signature") != runtime:
            raise ValueError("Partial cache runtime differs; use a new output directory")
        if old is None:
            np.save(args.output / "indices.npy", indices)
            depths = np.lib.format.open_memmap(args.output / "depth.npy", mode="w+", dtype=np.float32, shape=tuple(signature["shape"]))
            intrinsics = np.lib.format.open_memmap(args.output / "intrinsics.npy", mode="w+", dtype=np.float32, shape=(len(indices), 3, 3))
            old = {"schema_version": 1, "kind": "nyu_predictions", "signature": signature,
                   "completed": 0, "complete": False, "chunks": [], "records": [], "runtime_signature": runtime}
            commit_checkpoint(args.output, depths, intrinsics, old)
        else:
            depths = np.lib.format.open_memmap(args.output / "depth.npy", mode="r+")
            intrinsics = np.lib.format.open_memmap(args.output / "intrinsics.npy", mode="r+")
        start = old["completed"]
        records, chunks = list(old["records"]), list(old["chunks"])
        if len(records) != start or [r.get("index") for r in records] != indices[:start].tolist():
            raise ValueError("Persisted timing records disagree with committed indices")
        shape = {}
        hook = model.pixel_encoder.register_forward_pre_hook(lambda module, inputs: shape.update(network_hw=list(inputs[0].shape[-2:])))
        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        try:
            with torch.inference_mode():
                rgb = torch.from_numpy(nyu_image(handle, indices[start]).copy()).permute(2, 0, 1)
                for _ in range(2):
                    infer_with_camera(model, rgb, args.known_intrinsics)
                sync()
                chunk_start = start
                for row in range(start, len(indices)):
                    rgb = torch.from_numpy(nyu_image(handle, indices[row]).copy()).permute(2, 0, 1)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    sync()
                    began = perf_counter()
                    prediction = infer_with_camera(model, rgb, args.known_intrinsics)
                    sync()
                    seconds = perf_counter() - began
                    depth = prediction["depth"].squeeze().float().cpu().numpy()
                    intrinsic = prediction["intrinsics"].squeeze().float().cpu().numpy()
                    del prediction
                    if depth.shape != (480, 640):
                        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
                    if not np.isfinite(depth).all() or not (depth > 0).all() or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                        raise ValueError(f"Invalid model output at dataset index {indices[row]}")
                    depths[row], intrinsics[row] = depth, intrinsic
                    records.append({"index": int(indices[row]), "seconds": seconds,
                                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None})
                    if (row + 1) % CHECKPOINT_EVERY == 0 or row + 1 == len(indices):
                        stop = row + 1
                        chunks.append(chunk_record(depths, intrinsics, chunk_start, stop))
                        chunk_start = stop
                        old.update(completed=stop, chunks=chunks, records=records, **shape)
                        commit_checkpoint(args.output, depths, intrinsics, old)
                        print("NYU", args.split, stop, "/", len(indices), flush=True)
        finally:
            hook.remove()
        finalize_cache(args.output, signature, runtime, records, shape.get("network_hw"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--known-intrinsics", action="store_true")
    parser.add_argument("--device", default="cuda")
    infer(parser.parse_args())
