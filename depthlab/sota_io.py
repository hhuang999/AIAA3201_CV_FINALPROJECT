"""Verified train-only/split-guarded I/O, provenance hashing, and LF helpers."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .data import open_nyu, scene_names, splits


def file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_lf(path: Path | str, text: str) -> None:
    """Write UTF-8 text using explicit LF newlines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text.replace("\r\n", "\n"))


def write_json_lf(path: Path | str, obj: object) -> None:
    """Write deterministic JSON using explicit LF newlines and finite numbers."""
    payload = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    write_text_lf(path, payload)


def write_csv_lf(path: Path | str, df: pd.DataFrame, index: bool = False) -> None:
    """Write a pandas DataFrame to CSV using explicit LF newlines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=index, lineterminator="\n")


def scene_folds_seeded(names: np.ndarray, count: int = 5, seed: int = 20261031) -> list[np.ndarray]:
    """Partition image rows into scene-disjoint folds; matches depthlab.calibration.scene_folds."""
    names_arr = np.asarray(names)
    groups = np.unique(names_arr)
    if count < 2 or len(groups) < count:
        raise ValueError("Expected at least one scene per fold")
    np.random.default_rng(seed).shuffle(groups)
    return [np.flatnonzero(np.isin(names_arr, part)) for part in np.array_split(groups, count)]


class VerifiedCache:
    """Split-guarded wrapper over outputs/nyu_rgb prediction arrays."""

    def __init__(
        self,
        prediction_dir: Path,
        data_root: Path,
        allowed_split: str | None = "train",
        compute_hashes: bool = False,
        verify_scenes: bool = True,
    ) -> None:
        self.prediction_dir = Path(prediction_dir)
        self.data_root = Path(data_root)
        if allowed_split not in ("train", "test", "all"):
            raise ValueError(f"Unsupported allowed_split: {allowed_split!r}")
        self.allowed_split = allowed_split

        partition = splits(self.data_root)
        self.train_ids = partition["train"]
        self.test_ids = partition["test"]
        if len(self.train_ids) != 795 or len(self.test_ids) != 654:
            raise ValueError("Unexpected NYU official split sizes")
        if np.intersect1d(self.train_ids, self.test_ids).size:
            raise ValueError("Training and test image IDs overlap")

        if allowed_split == "train":
            self.allowed_ids = self.train_ids
        elif allowed_split == "test":
            self.allowed_ids = self.test_ids
        else:
            self.allowed_ids = np.arange(1449, dtype=np.int64)
        self._allowed_set = set(int(i) for i in self.allowed_ids.tolist())

        indices_path = self.prediction_dir / "indices.npy"
        depth_path = self.prediction_dir / "depth.npy"
        intrinsics_path = self.prediction_dir / "intrinsics.npy"
        for required in (indices_path, depth_path, intrinsics_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing prediction cache file: {required}")

        indices = np.load(indices_path, allow_pickle=False)
        if indices.shape != (1449,) or not np.array_equal(indices, np.arange(1449, dtype=indices.dtype)):
            raise ValueError("Prediction indices.npy must equal exact 0..1448 ordering")
        self.indices = indices.astype(np.int64)

        self._depths = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        self._intrinsics = np.load(intrinsics_path, mmap_mode="r", allow_pickle=False)
        if self._depths.shape != (1449, 480, 640):
            raise ValueError(f"Unexpected depth.npy shape: {self._depths.shape}")
        if self._intrinsics.shape != (1449, 3, 3):
            raise ValueError(f"Unexpected intrinsics.npy shape: {self._intrinsics.shape}")

        for idx in self.allowed_ids:
            d = self._depths[int(idx)]
            if not np.isfinite(d).all() or not (d > 0).all():
                raise ValueError(f"Nonfinite or nonpositive depth prediction at index {int(idx)}")
            k = self._intrinsics[int(idx)]
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError(f"Invalid predicted intrinsics at index {int(idx)}")

        if verify_scenes:
            with open_nyu(self.data_root) as handle:
                all_scenes = scene_names(handle)
                if all_scenes.shape != (1449,):
                    raise ValueError(f"Unexpected NYU scene array shape: {all_scenes.shape}")
                train_scenes = np.unique(all_scenes[self.train_ids])
                test_scenes = np.unique(all_scenes[self.test_ids])
                if len(train_scenes) != 249 or len(test_scenes) != 215:
                    raise ValueError(
                        f"Expected 249 train / 215 test scenes, got {len(train_scenes)} / {len(test_scenes)}"
                    )
                if np.intersect1d(train_scenes, test_scenes).size:
                    raise ValueError("Train and test scenes overlap")

        self.hashes: dict[str, str] = {}
        if compute_hashes:
            self.hashes = {
                "nyu_mat_sha256": file_sha256(self.data_root / "nyu_depth_v2_labeled.mat"),
                "splits_sha256": file_sha256(self.data_root / "splits.mat"),
                "depth_npy_sha256": file_sha256(depth_path),
                "intrinsics_npy_sha256": file_sha256(intrinsics_path),
                "indices_npy_sha256": file_sha256(indices_path),
            }

    def _check_index(self, index: int) -> int:
        idx = int(index)
        if idx not in self._allowed_set:
            raise PermissionError(
                f"Index {idx} is outside allowed_split={self.allowed_split!r}; split isolation enforced"
            )
        return idx

    def __getitem__(self, index: int) -> np.ndarray:
        idx = self._check_index(index)
        return np.asarray(self._depths[idx], dtype=np.float32)

    def intrinsics(self, index: int) -> np.ndarray:
        idx = self._check_index(index)
        return np.asarray(self._intrinsics[idx], dtype=np.float32)

    def intrinsics_for(self, indices: np.ndarray) -> np.ndarray:
        idx_arr = np.asarray([self._check_index(i) for i in indices], dtype=np.int64)
        return np.asarray(self._intrinsics[idx_arr], dtype=np.float32)


def load_nyu_rgb_cache(
    prediction_dir: Path | str,
    data_root: Path | str,
    allowed_split: str | None = "train",
    compute_hashes: bool = False,
    verify_scenes: bool = True,
) -> VerifiedCache:
    """Load and verify the local NYU RGB-only UniDepthV2 prediction cache."""
    return VerifiedCache(
        prediction_dir=Path(prediction_dir),
        data_root=Path(data_root),
        allowed_split=allowed_split,
        compute_hashes=compute_hashes,
        verify_scenes=verify_scenes,
    )
