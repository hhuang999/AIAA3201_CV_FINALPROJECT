"""Synthetic Part 2 checks; no GPU, official data, video writing or rendering."""
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import pytest
import torch

from depthlab.cache import (array_sha256, atomic_json, chunk_record, load_predictions,
                            require_pinned_weights, write_complete_manifest)
from depthlab.infer_nyu import infer_with_camera, finalize_cache
from depthlab.infer_video import (load_frames, load_video_predictions, save_depth_archive,
                                 validate_depth)
from depthlab.inference_backend import MODULES, use_pytorch_attention
from depthlab.render_video import normalize_clip, background_blur


def signature(indices, shape):
    return {"indices": list(indices), "shape": list(shape), "dtype": "float32", "known_intrinsics": False,
            "data": {"synthetic": True}, "source": {"synthetic": True}, "weights": {"synthetic": True},
            "backend": "synthetic", "preprocessing": "synthetic", "seed": 0}


def nyu_cache(tmp_path, indices=(9, 2, 40), intrinsics=True):
    depths = np.stack([np.full((4, 5), i + 1, dtype=np.float32) for i in range(len(indices))])
    np.save(tmp_path / "depth.npy", depths)
    np.save(tmp_path / "indices.npy", np.asarray(indices))
    if intrinsics:
        np.save(tmp_path / "intrinsics.npy", np.stack([np.eye(3, dtype=np.float32)] * len(indices)))
    manifest = write_complete_manifest(tmp_path, signature(indices, depths.shape))
    return depths, manifest


def test_constant_clip_regression():
    with np.errstate(divide="raise", invalid="raise"):
        normalized, bounds = normalize_clip(np.ones((3, 4, 5), dtype=np.float32))
    np.testing.assert_array_equal(normalized, np.zeros_like(normalized))
    assert bounds == (1., 1.)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_normalization_rejects_nonfinite(invalid):
    with pytest.raises(ValueError, match="finite"):
        normalize_clip(np.full((3, 4, 5), invalid))


def test_blur_matches_frozen_uint8_gaussian_rule():
    rgb = np.random.default_rng(0).integers(0, 256, (6, 7, 3), dtype=np.uint8)
    # At normalized depth zero all pixels choose the strongest blur exactly.
    np.testing.assert_array_equal(background_blur(rgb, np.zeros((6, 7))),
                                  cv2.GaussianBlur(rgb, (0, 0), 9.))


def test_original_dataset_id_mapping(tmp_path):
    depths, _ = nyu_cache(tmp_path)
    cache = load_predictions(tmp_path, [40, 9], False, True)
    np.testing.assert_array_equal(cache[40], depths[2])
    np.testing.assert_array_equal(cache[[2, 40]], depths[[1, 2]])
    assert cache.intrinsics_for([40, 9]).shape == (2, 3, 3)
    with pytest.raises(ValueError, match="Missing"):
        load_predictions(tmp_path, [0])


def test_optional_intrinsics(tmp_path):
    nyu_cache(tmp_path, intrinsics=False)
    cache = load_predictions(tmp_path, require_complete=True)
    with pytest.raises(ValueError, match="no predicted intrinsics"):
        cache.intrinsics_for([9])


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        nyu_cache(tmp_path, (1, 1, 2))


def test_wrong_shape_rejected(tmp_path):
    nyu_cache(tmp_path)
    np.save(tmp_path / "depth.npy", np.ones((2, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="compact"):
        load_predictions(tmp_path)


def test_committed_depth_tampering_rejected(tmp_path):
    nyu_cache(tmp_path)
    depth = np.load(tmp_path / "depth.npy", mmap_mode="r+")
    depth[0, 0, 0] = 10
    depth.flush()
    with pytest.raises(ValueError, match="checksum"):
        load_predictions(tmp_path, require_complete=True)


def test_missing_chunk_rejected(tmp_path):
    _, manifest = nyu_cache(tmp_path)
    manifest["chunks"] = []
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="coverage"):
        load_predictions(tmp_path)


def test_camera_identity_rejected(tmp_path):
    nyu_cache(tmp_path)
    with pytest.raises(ValueError, match="camera"):
        load_predictions(tmp_path, expected_known_intrinsics=True)


def test_legacy_not_accepted_for_frozen_replay(tmp_path):
    nyu_cache(tmp_path)
    (tmp_path / "manifest.json").unlink()
    atomic_json(tmp_path / "runtime.json", {"known_intrinsics": False})
    assert len(load_predictions(tmp_path).indices) == 3
    with pytest.raises(ValueError, match="Legacy"):
        load_predictions(tmp_path, require_complete=True)


def test_partial_cache_only_exposes_committed_rows(tmp_path):
    depth, manifest = nyu_cache(tmp_path)
    intrinsic = np.load(tmp_path / "intrinsics.npy")
    manifest.update(completed=1, complete=False, chunks=[chunk_record(depth, intrinsic, 0, 1)])
    atomic_json(tmp_path / "manifest.json", manifest)
    assert load_predictions(tmp_path).indices.tolist() == [9]
    with pytest.raises(ValueError, match="Missing"):
        load_predictions(tmp_path, [2])
    with pytest.raises(ValueError, match="complete"):
        load_predictions(tmp_path, require_complete=True)


def test_completion_recovery_keeps_predictions(tmp_path):
    depths, manifest = nyu_cache(tmp_path)
    records = [{"index": i, "seconds": .2, "peak_allocated_mib": None, "peak_reserved_mib": None}
               for i in (9, 2, 40)]
    finalize_cache(tmp_path, manifest["signature"], {"gpu": "synthetic CPU"}, records, [4, 5])
    np.testing.assert_array_equal(load_predictions(tmp_path, require_complete=True).depths, depths)
    assert json.loads((tmp_path / "runtime.json").read_text())["network_hw"] == [4, 5]


def test_pinned_checkpoint_mismatch():
    with pytest.raises(ValueError, match="pinned Small-model"):
        require_pinned_weights({"weights.pth": "changed"}, {"weights.pth": "frozen"})


def test_unspecified_cuda_device_uses_current_visible_index(monkeypatch):
    import depthlab.infer_nyu as runner
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert runner.resolve_device("cuda") == torch.device("cuda:0")
    assert runner.resolve_device("cuda:2") == torch.device("cuda:2")
    assert runner.resolve_device("cpu") == torch.device("cpu")


def test_fractional_dataset_id_is_rejected(tmp_path):
    nyu_cache(tmp_path)
    with pytest.raises(ValueError, match="integer"):
        load_predictions(tmp_path)[2.9]


def test_writer_crash_resume_and_completed_no_model_load(tmp_path, monkeypatch):
    import depthlab.infer_nyu as runner
    class Handle(dict):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Hook:
        def remove(self):
            pass
    class Encoder:
        def register_forward_pre_hook(self, hook):
            return Hook()
    class Model:
        fail = True
        loads = 0
        @classmethod
        def from_pretrained(cls, path):
            cls.loads += 1
            return cls()
        def __init__(self):
            self.calls = 0
            self.shape_constraints = {"synthetic": True}
            self.pixel_encoder = Encoder()
        def to(self, device):
            return self
        def eval(self):
            return self
        def infer(self, rgb):
            assert torch.is_inference_mode_enabled()
            self.calls += 1
            if Model.fail and self.calls == 4:
                raise RuntimeError("synthetic interrupted writer")
            return {"depth": torch.ones((1, 1, 480, 640)), "intrinsics": torch.eye(3)[None]}
    monkeypatch.setitem(sys.modules, "unidepth.models", types.SimpleNamespace(UniDepthV2=Model))
    monkeypatch.setattr(runner, "source_identity", lambda *args: {"synthetic": True})
    monkeypatch.setattr(runner, "weights_identity", lambda *args: runner.PINNED_WEIGHTS.copy())
    monkeypatch.setattr(runner, "use_pytorch_attention", lambda *args: {"synthetic": True})
    monkeypatch.setattr(runner, "open_nyu", lambda *args: Handle(images=[0, 1]))
    monkeypatch.setattr(runner, "splits", lambda *args: {"train": np.array([7, 2])})
    monkeypatch.setattr(runner, "nyu_image", lambda *args: np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(runner, "CHECKPOINT_EVERY", 1)
    for name in ("nyu_depth_v2_labeled.mat", "splits.mat"):
        (tmp_path / name).write_text("synthetic hash fixture")
    args = types.SimpleNamespace(data_root=tmp_path, source=tmp_path, weights=tmp_path,
                                 output=tmp_path / "cache", split="train", limit=2,
                                 known_intrinsics=False, device="cpu")
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        runner.infer(args)
    partial = load_predictions(args.output)
    assert partial.indices.tolist() == [7]
    assert partial.metadata["completed"] == 1
    Model.fail = False
    runner.infer(args)
    complete = load_predictions(args.output, [7, 2], False, True)
    assert complete.indices.tolist() == [7, 2]
    assert Model.loads == 2
    runner.infer(args)
    assert Model.loads == 2


def test_known_camera_is_fresh_per_call():
    class Model:
        def infer(self, rgb, camera=None):
            if camera is None:
                return "RGB only"
            before = camera.clone()
            camera.mul_(0)
            return before
    first = infer_with_camera(Model(), None, True)
    second = infer_with_camera(Model(), None, True)
    torch.testing.assert_close(first, second)
    assert first[0, 0, 0] > 0
    assert infer_with_camera(Model(), None, False) == "RGB only"


@pytest.mark.parametrize("model_name", list(MODULES))
def test_fixed_attention_branch(monkeypatch, model_name):
    for name in MODULES[model_name]:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(XFORMERS_AVAILABLE=True))
    metadata = use_pytorch_attention(model_name)
    assert metadata["xformers_attention_enabled"] is False
    assert all(sys.modules[name].XFORMERS_AVAILABLE is False for name in MODULES[model_name])


def video_cache(tmp_path):
    from depthlab.cache import sha256
    frames = np.zeros((5, 4, 6, 3), dtype=np.uint8)
    info = {"frames": 5, "fps": 23.976, "decoded_hw": [4, 6], "input_sha256": "synthetic",
            "decoded_frames_sha256": array_sha256(frames)}
    np.save(tmp_path / "frames.npy", frames)
    atomic_json(tmp_path / "frames.json", info)
    depth = np.ones(frames.shape[:3], dtype=np.float32)
    save_depth_archive(tmp_path, depth, info["fps"])
    manifest = {**info, "schema_version": 1, "complete": True, "method": "image",
                "source": "synthetic", "weights": "synthetic", "attention_backend": "synthetic",
                "seed": 0, "preprocessing": "synthetic", "depth_archive_sha256": sha256(tmp_path / "depth.npz")}
    atomic_json(tmp_path / "prediction_manifest.json", manifest)
    return frames, info, manifest


def test_video_bound_archive(tmp_path):
    frames, info, _ = video_cache(tmp_path)
    loaded, _ = load_frames(tmp_path)
    np.testing.assert_array_equal(loaded, frames)
    depth, manifest = load_video_predictions(tmp_path, frames, info, "image")
    assert depth.shape == frames.shape[:3]
    with pytest.raises(ValueError, match="method"):
        load_video_predictions(tmp_path, frames, info, "video")


def test_video_same_input_rgb_mismatch(tmp_path):
    frames, info, _ = video_cache(tmp_path)
    frames[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="RGB pixels"):
        load_video_predictions(tmp_path, frames, info)


def test_video_fps_mismatch(tmp_path):
    frames, info, _ = video_cache(tmp_path)
    info["fps"] = 24.
    with pytest.raises(ValueError, match="fps"):
        load_video_predictions(tmp_path, frames, info)


def test_video_archive_tampering_rejected(tmp_path):
    frames, info, _ = video_cache(tmp_path)
    save_depth_archive(tmp_path, np.full(frames.shape[:3], 2, dtype=np.float32), info["fps"])
    with pytest.raises(ValueError, match="SHA256"):
        load_video_predictions(tmp_path, frames, info)


def test_video_shapes_and_nonfinite_rejected():
    frames = np.zeros((5, 4, 6, 3), dtype=np.uint8)
    for depth in (np.zeros((4, 4, 6), dtype=np.float32), np.full((5, 4, 6), np.nan, dtype=np.float32)):
        with pytest.raises(ValueError, match="finite float32"):
            validate_depth(depth, frames)


def test_render_rejects_mismatched_cache_before_output(tmp_path):
    from depthlab.render_video import render
    video_cache(tmp_path)
    args = types.SimpleNamespace(frames=tmp_path, image_predictions=tmp_path,
                                 video_predictions=tmp_path, output=tmp_path / "delivery",
                                 results=tmp_path / "figures")
    with pytest.raises(ValueError, match="method differs"):
        render(args)
    assert not args.output.exists()
    assert not args.results.exists()
