import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .data import nyu_depth, nyu_image, open_nyu, splits
from .metrics import METRIC_NAMES, evaluate_image

RGB_INTRINSICS = np.array([[518.8579011745019, 0, 325.58244941119034],
                           [0, 519.4696111212749, 253.73616633400465],
                           [0, 0, 1]], dtype=np.float32)
UPSTREAM_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"


def infer(args):
    sys.path.insert(0, str(args.source.resolve()))
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    torch.set_num_threads(4)
    model = UniDepthV2.from_pretrained(str(args.weights.resolve())).cuda().eval()
    shape = {}
    hook = model.pixel_encoder.register_forward_pre_hook(
        lambda module, inputs: shape.update(network_hw=list(inputs[0].shape[-2:])))
    handle = open_nyu(args.data_root)
    indices = np.arange(len(handle["images"])) if args.split == "all" else splits(args.data_root)[args.split]
    indices = indices[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "indices.npy", indices)
    depths = np.lib.format.open_memmap(args.output / "depth.npy", mode="w+", dtype=np.float32,
                                      shape=(len(indices), 480, 640))
    intrinsics = np.empty((len(indices), 3, 3), dtype=np.float32)
    rgb = torch.from_numpy(nyu_image(handle, indices[0]).copy()).permute(2, 0, 1)
    camera = Pinhole(K=torch.from_numpy(RGB_INTRINSICS).unsqueeze(0)) if args.known_intrinsics else None
    for _ in range(5):
        model.infer(rgb, camera=camera.clone() if camera is not None else None)
    torch.cuda.synchronize()
    records = []
    for row, index in enumerate(indices):
        rgb = torch.from_numpy(nyu_image(handle, index).copy()).permute(2, 0, 1)
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        prediction = model.infer(rgb, camera=camera.clone() if camera is not None else None)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        depths[row] = prediction["depth"].squeeze().float().cpu().numpy()
        intrinsics[row] = prediction["intrinsics"].squeeze().float().cpu().numpy()
        record = {"index": int(index), "seconds": seconds,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        if args.limit is not None and args.split == "train":
            record.update(dict(zip(METRIC_NAMES, evaluate_image(depths[row], nyu_depth(handle, index)))))
        records.append(record)
        if row % 50 == 0 or row == len(indices) - 1:
            print("NYU", args.split, row + 1, "/", len(indices), flush=True)
    depths.flush()
    np.save(args.output / "intrinsics.npy", intrinsics)
    pd.DataFrame(records).to_csv(args.output / "inference_per_image.csv", index=False)
    metadata = {"model": "UniDepthV2 ViT-S", "source_revision": UPSTREAM_REVISION,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "precision": "official internal FP16 autocast",
                "rgb_input_hw": [480, 640], **shape, "known_intrinsics": args.known_intrinsics,
                "images": len(indices), "mean_seconds": float(np.mean([r["seconds"] for r in records])),
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in records),
                "peak_reserved_mib": max(r["peak_reserved_mib"] for r in records),
                "warmup_images": 5, "scope": "infer call, including preprocessing and input transfer; excludes file IO and output transfer"}
    (args.output / "runtime.json").write_text(json.dumps(metadata, indent=2))
    hook.remove()
    handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--known-intrinsics", action="store_true")
    args = parser.parse_args()
    infer(args)
