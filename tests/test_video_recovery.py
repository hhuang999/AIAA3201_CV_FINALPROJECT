"""CPU-only device binding and verified decoded-cache recovery regressions."""
import importlib
import json
from pathlib import Path
import types

import cv2
import numpy as np
import pytest
import torch

from depthlab import infer_video as runner
from depthlab.cache import atomic_json


@pytest.fixture
def dav2(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "third_party/Depth-Anything-V2"
    if not source.is_dir():
        pytest.skip("Pinned DAV2 checkout is required for preprocessing parity")
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("depth_anything_v2.dpt")


def test_cpu_binding_avoids_upstream_accelerator_selection_and_preserves_pixels(dav2, monkeypatch):
    class Model:
        image2tensor = dav2.DepthAnythingV2.image2tensor
    model = Model()
    rgb = np.random.default_rng(19).integers(0, 256, (18, 27, 3), dtype=np.uint8)
    # Establish exact official CPU preprocessing, without constructing a model.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    expected, expected_hw = model.image2tensor(rgb, 28)
    original_to = torch.Tensor.to
    destinations = []
    def guarded_to(tensor, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        destinations.append(str(target))
        if torch.device(target).type != "cpu":
            raise AssertionError("CPU inference attempted an accelerator allocation")
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "to", guarded_to)
    runner.bind_dav2_input_device(model, torch.device("cpu"))
    actual, actual_hw = model.image2tensor(rgb, 28)
    assert destinations == ["cpu"]
    assert actual.device.type == "cpu"
    assert actual_hw == expected_hw == (18, 27)
    assert torch.equal(actual, expected)


def test_cuda_binding_keeps_official_method_untouched():
    original = lambda *args: None
    model = types.SimpleNamespace(image2tensor=original)
    runner.bind_dav2_input_device(model, torch.device("cuda:2"))
    assert model.image2tensor is original


@pytest.fixture
def decoded(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic source identity; no encoded video")
    bgr = np.random.default_rng(3).integers(0, 256, (3, 8, 12, 3), dtype=np.uint8)
    class Reader:
        def __init__(self, *args):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 24., cv2.CAP_PROP_FRAME_COUNT: 3}[prop]
        def read(self):
            if self.index == len(bgr):
                return False, None
            frame = bgr[self.index].copy()
            self.index += 1
            return True, frame
        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoCapture", Reader)
    output = tmp_path / "frames"
    runner.decode(source, output, 2)
    return source, output


def test_decode_reuses_verified_cache_after_downstream_interruption(decoded, monkeypatch):
    source, output = decoded
    paths = [output / name for name in ("frames.npy", "frames.json")]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    def no_redecode(*args):
        raise AssertionError("Verified frames should be reused without decoding again")
    monkeypatch.setattr(cv2, "VideoCapture", no_redecode)
    # The preceding decode completed before a later baseline stage stopped.
    runner.decode(source, output, 2)
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before


@pytest.mark.parametrize("change", ["source", "limit", "resolution", "decoder", "count", "shape", "pixels", "dtype", "missing_manifest", "missing_array"])
def test_decode_rejects_changed_or_incomplete_cache(decoded, monkeypatch, change):
    source, output = decoded
    limit = 2
    manifest = json.loads((output / "frames.json").read_text())
    if change == "source":
        source.write_bytes(b"different source")
    elif change == "limit":
        limit = 3
    elif change == "resolution":
        monkeypatch.setattr(runner, "MAX_RESOLUTION", 320)
    elif change == "decoder":
        manifest["decoder"] = "other preprocessing"
    elif change == "count":
        manifest["source_frames"] = 1
    elif change == "shape":
        manifest["original_hw"] = [16, 24]
    elif change in ("pixels", "dtype"):
        frames = np.load(output / "frames.npy")
        if change == "pixels":
            frames[0, 0, 0, 0] ^= 1
        else:
            frames = frames.astype(np.float32)
        np.save(output / "frames.npy", frames)
    elif change == "missing_manifest":
        (output / "frames.json").unlink()
    elif change == "missing_array":
        (output / "frames.npy").unlink()
    if change in ("decoder", "count", "shape"):
        atomic_json(output / "frames.json", manifest)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
    with pytest.raises(ValueError):
        runner.decode(source, output, limit)
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()} == before
