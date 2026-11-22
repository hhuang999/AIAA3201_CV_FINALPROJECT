"""Small fixtures for the Round 2 cache's manifest and row contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from depthlab import sota_v2_cache as cache_module


@pytest.fixture
def v2_caches(tmp_path, monkeypatch):
    data, original, aux, flip = [tmp_path / name for name in ("data", "original", "aux", "flip")]
    for folder in (data, original, aux, flip):
        folder.mkdir()

    n_images, image_hw = 2, (3, 4)
    monkeypatch.setattr(cache_module, "_NYU_IMAGES", n_images, raising=False)
    monkeypatch.setattr(cache_module, "_NYU_HW", image_hw, raising=False)
    indices = np.arange(n_images, dtype=np.int64)
    depth = np.ones((n_images, *image_hw), dtype=np.float32)
    intrinsics = np.broadcast_to(np.eye(3, dtype=np.float32), (n_images, 3, 3))
    for filename, value in (("indices.npy", indices), ("depth.npy", depth), ("intrinsics.npy", intrinsics)):
        np.save(original / filename, value)

    (data / "nyu_depth_v2_labeled.mat").write_bytes(b"synthetic RGB/depth identity")
    (data / "splits.mat").write_bytes(b"synthetic split identity")
    binding = {
        "nyu_mat_sha256": cache_module.file_sha256(data / "nyu_depth_v2_labeled.mat"),
        "splits_sha256": cache_module.file_sha256(data / "splits.mat"),
        "orig_depth_sha256": cache_module.file_sha256(original / "depth.npy"),
        "orig_intrinsics_sha256": cache_module.file_sha256(original / "intrinsics.npy"),
        "orig_indices_sha256": cache_module.file_sha256(original / "indices.npy"),
    }

    cls_tokens = np.ones((n_images, 384), dtype=np.float32)
    aux_arrays = {
        "indices.npy": indices,
        "cls_tokens.npy": cls_tokens,
        "patch_mean.npy": cls_tokens,
        "patch_std.npy": cls_tokens,
        "depth_feat_stats.npy": np.ones((n_images, 512), dtype=np.float32),
        "confidence_f16.npy": depth.astype(np.float16),
        "confidence_summary.npy": np.ones((n_images, len(cache_module.CONF_SUMMARY_NAMES)), dtype=np.float32),
    }
    flip_arrays = {
        "indices.npy": indices,
        "depth_flip.npy": depth,
        "intrinsics_flip.npy": intrinsics,
        "confidence_flip_f16.npy": depth.astype(np.float16),
        "cls_tokens_flip.npy": cls_tokens,
    }
    for folder, arrays in ((aux, aux_arrays), (flip, flip_arrays)):
        for filename, value in arrays.items():
            np.save(folder / filename, value)
        cache_module.write_json_lf(folder / "manifest_v2_cache.json", {
            "complete": True,
            **binding,
            "weights_sha256": cache_module.PINNED_WEIGHTS,
            "files_sha256": {filename: cache_module.file_sha256(folder / filename) for filename in arrays},
        })

    class Base(SimpleNamespace):
        def _check_index(self, index):
            if index not in self.allowed_ids:
                raise PermissionError("outside synthetic train split")
            return index

    def load_base(*args, allowed_split, compute_hashes, **kwargs):
        assert allowed_split == "train"
        hashes = {
            "nyu_mat_sha256": binding["nyu_mat_sha256"],
            "splits_sha256": binding["splits_sha256"],
            "depth_npy_sha256": binding["orig_depth_sha256"],
            "intrinsics_npy_sha256": binding["orig_intrinsics_sha256"],
            "indices_npy_sha256": binding["orig_indices_sha256"],
        }
        return Base(allowed_split=allowed_split, train_ids=indices[:1], test_ids=indices[1:],
                    allowed_ids=indices[:1], indices=indices, hashes=hashes if compute_hashes else {})

    monkeypatch.setattr(cache_module, "load_nyu_rgb_cache", load_base)
    monkeypatch.setattr(cache_module, "splits", lambda _: {"train": indices[:1], "test": indices[1:]})

    def unexpected_backbone(*args, **kwargs):
        pytest.fail("Existing caches must be checked without loading a backbone")

    monkeypatch.setattr(cache_module, "_load_local_unidepth", unexpected_backbone)
    monkeypatch.setattr(cache_module, "ensure_unidepth_weights", unexpected_backbone)
    return SimpleNamespace(root=tmp_path, data=data, original=original, aux=aux, flip=flip)


def edit_manifest(folder, update):
    path = folder / "manifest_v2_cache.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    update(manifest)
    cache_module.write_json_lf(path, manifest)


def replace_array(folder, filename, value):
    path = folder / filename
    np.save(path, value)
    edit_manifest(folder, lambda manifest: manifest["files_sha256"].update({filename: cache_module.file_sha256(path)}))


def reuse(caches):
    return cache_module.build_aux_and_flip_caches(
        caches.data, caches.root / "source", caches.root / "weights", caches.original, caches.aux, caches.flip,
    )


def read(caches, **kwargs):
    return cache_module.load_v2_cache(caches.original, caches.aux, caches.flip, caches.data, **kwargs)


def test_legacy_complete_cache_reuses_without_backbone(v2_caches):
    assert reuse(v2_caches)["aux_manifest"]["complete"] is True
    assert read(v2_caches, compute_hashes=True).hashes["flip_depth_sha256"] == cache_module.file_sha256(
        v2_caches.flip / "depth_flip.npy"
    )


def test_reuse_rejects_missing_array(v2_caches):
    (v2_caches.aux / "cls_tokens.npy").unlink()
    with pytest.raises(FileNotFoundError, match="cls_tokens.npy"):
        reuse(v2_caches)


def test_reuse_preserves_partial_cache_with_missing_manifest(v2_caches):
    (v2_caches.flip / "manifest_v2_cache.json").unlink()
    with pytest.raises(ValueError, match="incomplete Round 2 cache"):
        reuse(v2_caches)
    assert (v2_caches.flip / "depth_flip.npy").is_file()


@pytest.mark.parametrize("reader", [read, reuse])
def test_rejects_stale_original_intrinsics_binding(v2_caches, reader):
    edit_manifest(v2_caches.flip, lambda manifest: manifest.update(orig_intrinsics_sha256="0" * 64))
    with pytest.raises(ValueError, match="orig_intrinsics_sha256"):
        reader(v2_caches)


def test_reader_rejects_mismatched_recorded_source(v2_caches):
    edit_manifest(v2_caches.aux, lambda manifest: manifest.update(source={"revision": "different"}))
    with pytest.raises(ValueError, match="pinned clean revision"):
        read(v2_caches)


def test_reader_rejects_array_with_incorrect_checksum(v2_caches):
    path = v2_caches.aux / "patch_mean.npy"
    np.save(path, np.load(path) * 2)
    with pytest.raises(ValueError, match="patch_mean.npy"):
        read(v2_caches)


def test_reader_rejects_reordered_flip_indices_even_with_valid_checksum(v2_caches):
    replace_array(v2_caches.flip, "indices.npy", np.array([1, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="indices.npy"):
        read(v2_caches)


def test_reader_rejects_incorrect_feature_shape(v2_caches):
    replace_array(v2_caches.aux, "confidence_summary.npy", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="confidence_summary.npy"):
        read(v2_caches)


def test_reader_ignores_invalid_focal_outside_allowed_split(v2_caches):
    k = np.load(v2_caches.flip / "intrinsics_flip.npy")
    k[1, 0, 0] = np.nan
    replace_array(v2_caches.flip, "intrinsics_flip.npy", k)
    cache = read(v2_caches)
    assert cache.intrinsics_flip(0)[0, 0] == 1.0
    with pytest.raises(PermissionError, match="outside synthetic train split"):
        cache.intrinsics_flip(1)


@pytest.mark.parametrize("invalid_focal", [-1.0, np.nan])
def test_reader_rejects_invalid_focal_inside_allowed_split(v2_caches, invalid_focal):
    k = np.load(v2_caches.flip / "intrinsics_flip.npy")
    k[0, 1, 1] = invalid_focal
    replace_array(v2_caches.flip, "intrinsics_flip.npy", k)
    with pytest.raises(ValueError, match="intrinsics_flip.npy"):
        read(v2_caches)
