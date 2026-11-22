"""Export a verified official split without repeating model inference."""
import argparse
import copy
from pathlib import Path

import numpy as np

from .cache import load_predictions, sha256, write_complete_manifest
from .data import splits


def export_subset(source, data_root, split, output):
    source, data_root, output = map(Path, (source, data_root, output))
    ids = splits(data_root)[split]
    cache = load_predictions(source, required_ids=ids, expected_known_intrinsics=False, require_complete=True)
    binding = {"nyu_mat_sha256": sha256(data_root / "nyu_depth_v2_labeled.mat"),
               "splits_sha256": sha256(data_root / "splits.mat")}
    if cache.metadata["signature"].get("data") != binding:
        raise ValueError("Prediction cache data binding differs from the requested split")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Subset output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "indices.npy", ids)
    shape = (len(ids), *cache.depths.shape[1:])
    depths = np.lib.format.open_memmap(output / "depth.npy", mode="w+", dtype=np.float32, shape=shape)
    for row, index in enumerate(ids):
        depths[row] = cache[index]
    depths.flush()
    if (source / "intrinsics.npy").exists():
        np.save(output / "intrinsics.npy", cache.intrinsics_for(ids))
    signature = copy.deepcopy(cache.metadata["signature"])
    signature.update(indices=ids.tolist(), shape=list(shape), split=split)
    manifest = write_complete_manifest(output, signature, cache.metadata.get("runtime_signature"),
        provenance={"operation": "original-ID subset; prediction values unchanged",
                    "parent_manifest_sha256": sha256(source / "manifest.json"),
                    "parent_depth_sha256": cache.metadata["files"]["depth.npy"]})
    print(split, len(ids), manifest["files"]["depth.npy"], flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_subset(args.predictions, args.data_root, args.split, args.output)
