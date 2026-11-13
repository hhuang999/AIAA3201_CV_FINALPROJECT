"""Prediction identity, atomic checkpoints and dataset-ID aware cache reads."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def cache_lock(path):
    # Server-side writers use an advisory lock released automatically on exit.
    import fcntl
    with Path(path).open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this cache") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_identity(source, expected_revision):
    source = Path(source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != expected_revision or dirty:
        raise ValueError("Upstream source differs from the pinned clean revision")
    return {"revision": revision, "tracked_source_clean": True}


def weights_identity(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError("Empty checkpoint directory")
    return {str(p.relative_to(path)) if path.is_dir() else p.name: sha256(p) for p in files}


def require_pinned_weights(identity, expected):
    for filename, digest in expected.items():
        if identity.get(filename) != digest:
            raise ValueError(f"Checkpoint differs from the pinned Small-model weights: {filename}")


def validate_indices(indices):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0:
        raise ValueError("Prediction indices must be a nonempty integer vector")
    if (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise ValueError("Prediction indices must be nonnegative and unique")
    return indices


def chunk_record(depths, intrinsics, start, stop):
    record = {"start": start, "stop": stop, "depth_sha256": array_sha256(depths[start:stop])}
    if intrinsics is not None:
        record["intrinsics_sha256"] = array_sha256(intrinsics[start:stop])
    return record


def verify_chunks(depths, intrinsics, manifest):
    completed = manifest.get("completed")
    if type(completed) is not int or not 0 <= completed <= len(depths):
        raise ValueError("Invalid committed row count")
    cursor = 0
    for record in manifest.get("chunks", []):
        start, stop = record["start"], record["stop"]
        if start != cursor or not start < stop <= completed:
            raise ValueError("Invalid or incomplete committed chunk coverage")
        if chunk_record(depths, intrinsics, start, stop) != record:
            raise ValueError("Committed prediction chunk checksum mismatch")
        block = depths[start:stop]
        if not np.isfinite(block).all() or not (block > 0).all():
            raise ValueError("Committed NYU predictions must be finite and positive")
        if intrinsics is not None and not np.isfinite(intrinsics[start:stop]).all():
            raise ValueError("Nonfinite predicted intrinsics")
        cursor = stop
    if cursor != completed:
        raise ValueError("Committed chunk coverage is incomplete")


def write_complete_manifest(folder, signature, runtime=None, chunk_size=25, provenance=None):
    """Bind verified arrays (including explicitly migrated frozen caches).

    Caller supplies independently verified input/source/checkpoint provenance;
    this helper never invents it or changes prediction arrays.
    """
    folder = Path(folder)
    indices = validate_indices(np.load(folder / "indices.npy", allow_pickle=False))
    depths = np.load(folder / "depth.npy", mmap_mode="r", allow_pickle=False)
    intrinsics = np.load(folder / "intrinsics.npy", mmap_mode="r", allow_pickle=False) if (folder / "intrinsics.npy").exists() else None
    _validate_shapes(indices, depths, intrinsics)
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    manifest = {"schema_version": 1, "kind": "nyu_predictions", "signature": signature,
                "completed": len(indices), "complete": True, "runtime_signature": runtime,
                "chunks": [chunk_record(depths, intrinsics, start, min(start + chunk_size, len(indices)))
                           for start in range(0, len(indices), chunk_size)]}
    if provenance is not None:
        manifest["migration_provenance"] = provenance
    manifest["files"] = {name: sha256(folder / name) for name in ("depth.npy", "indices.npy", "intrinsics.npy") if (folder / name).exists()}
    _verify_manifest(folder, indices, depths, intrinsics, manifest, require_complete=True)
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def _validate_shapes(indices, depths, intrinsics):
    if depths.ndim != 3 or len(depths) != len(indices) or min(depths.shape[1:]) <= 0 or depths.dtype != np.float32:
        raise ValueError("Depth cache must be compact float32 [indices,H,W]")
    if intrinsics is not None and (intrinsics.shape != (len(indices), 3, 3) or intrinsics.dtype != np.float32):
        raise ValueError("Intrinsics cache must be float32 [indices,3,3]")


def _verify_manifest(folder, indices, depths, intrinsics, manifest, require_complete):
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "nyu_predictions":
        raise ValueError("Unsupported NYU cache manifest")
    signature = manifest.get("signature", {})
    if signature.get("indices") != indices.tolist() or signature.get("shape") != list(depths.shape) or signature.get("dtype") != "float32":
        raise ValueError("Cache manifest disagrees with array indices/shape/dtype")
    if type(signature.get("known_intrinsics")) is not bool:
        raise ValueError("Cache camera identity is unrecorded")
    verify_chunks(depths, intrinsics, manifest)
    if require_complete:
        if manifest.get("complete") is not True or manifest["completed"] != len(indices):
            raise ValueError("Cache has no verified complete marker")
        for key in ("data", "source", "weights", "backend", "preprocessing", "seed"):
            if key not in signature or signature[key] is None:
                raise ValueError(f"Cache provenance is missing {key}")
        for name in ("depth.npy", "indices.npy", "intrinsics.npy"):
            if (folder / name).exists() and manifest.get("files", {}).get(name) != sha256(folder / name):
                raise ValueError(f"Complete cache file checksum mismatch: {name}")


class Predictions:
    def __init__(self, indices, depths, intrinsics, metadata):
        self.indices, self.depths, self.metadata = indices, depths, metadata
        self._intrinsics = intrinsics
        self._rows = {int(index): row for row, index in enumerate(indices)}

    def _row_indices(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Dataset IDs must be an integer vector")
        try:
            return [self._rows[int(index)] for index in ids]
        except KeyError as error:
            raise ValueError(f"Missing prediction for dataset index {error.args[0]}") from error

    def __getitem__(self, original_dataset_id):
        if np.isscalar(original_dataset_id):
            return self.depths[self._row_indices([original_dataset_id])[0]]
        return self.depths[self._row_indices(original_dataset_id)]

    def intrinsics_for(self, ids):
        if self._intrinsics is None:
            raise ValueError("This cache has no predicted intrinsics")
        return self._intrinsics[self._row_indices(ids)]


def load_predictions(folder, required_ids=None, expected_known_intrinsics=None, require_complete=False):
    folder = Path(folder)
    indices = validate_indices(np.load(folder / "indices.npy", allow_pickle=False))
    depths = np.load(folder / "depth.npy", mmap_mode="r", allow_pickle=False)
    intrinsics = np.load(folder / "intrinsics.npy", mmap_mode="r", allow_pickle=False) if (folder / "intrinsics.npy").exists() else None
    _validate_shapes(indices, depths, intrinsics)
    if (folder / "manifest.json").exists():
        metadata = json.loads((folder / "manifest.json").read_text())
        _verify_manifest(folder, indices, depths, intrinsics, metadata, require_complete)
        available = metadata["completed"]
        camera = metadata["signature"].get("known_intrinsics")
    else:
        if require_complete:
            raise ValueError("Legacy cache lacks a verified completion manifest")
        metadata = json.loads((folder / "runtime.json").read_text())
        available = len(indices)
        camera = metadata.get("known_intrinsics")
    if expected_known_intrinsics is not None and camera is not expected_known_intrinsics:
        raise ValueError("Prediction cache camera mode differs or is unrecorded")
    result = Predictions(indices[:available], depths[:available], None if intrinsics is None else intrinsics[:available], metadata)
    if required_ids is not None:
        result._row_indices(validate_indices(required_ids))
    return result
