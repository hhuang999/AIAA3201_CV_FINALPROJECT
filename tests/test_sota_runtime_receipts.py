import hashlib
import json
from itertools import count

import numpy as np
import pytest

from depthlab import sota_v2_evaluate as evaluation


class SmallCache:
    def __init__(self):
        self.depth = np.ones((2, 2))

    def __getitem__(self, index):
        return self.depth

    depth_flip = __getitem__
    confidence = __getitem__
    confidence_flip = __getitem__
    intrinsics = __getitem__
    intrinsics_flip = __getitem__

    def confidence_summary_for(self, indices):
        return np.ones((len(indices), 2))

    cls_tokens_for = confidence_summary_for
    cls_tokens_flip_for = confidence_summary_for
    patch_mean_for = confidence_summary_for
    patch_std_for = confidence_summary_for
    depth_feat_stats_for = confidence_summary_for


@pytest.fixture
def run_benchmark(monkeypatch):
    ticks = count()
    monkeypatch.setattr(evaluation, "perf_counter", lambda: next(ticks) / 1000)
    monkeypatch.setattr(evaluation, "apply_frozen", lambda depth, *_a, **_kw: depth)
    monkeypatch.setattr(evaluation, "apply_sota_model", lambda depth, *_a, **_kw: depth)
    monkeypatch.setattr(evaluation, "apply_sota_v2_model", lambda depth, *_a, **_kw: depth)
    monkeypatch.setattr(evaluation, "assemble_semantic_vector", lambda *_: None)
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: False)
    model = {"semantic_branch": {"sem_mode": "none"}}

    def run(aux, *, images=10, **kwargs):
        return evaluation.benchmark_v2_uncontended_runtime(
            SmallCache(), np.arange(images), model, model, {}, {}, aux, **kwargs,
        )

    return run


def write_receipt(path, duration, device="cuda", gpu_name="Some GPU"):
    path.write_text(json.dumps({
        "device": device,
        "gpu_name": gpu_name,
        "mean_seconds_per_image": duration,
    }), encoding="utf-8")
    return path


def test_reads_actual_aux_timing_and_leaves_missing_flip_unknown(tmp_path, run_benchmark):
    aux = write_receipt(tmp_path / "runtime_aux.json", 0.0125)
    runtime = run_benchmark(aux)

    assert runtime["backbone_single_pass_ms_per_image"] == pytest.approx(12.5)
    assert runtime["backbone_two_pass_flip_ms_per_image"] is None
    assert runtime["end_to_end_v1_ms"] == pytest.approx(12.6)
    assert runtime["end_to_end_v2_flip_tta_cpu_post_ms"] is None
    assert runtime["v2_flip_tta_cpu_post_ms"] == pytest.approx(0.1)
    assert runtime["backbone_timing_sources"]["flip"] is None


def test_uses_separate_flip_receipt_and_labels_composite_estimate(tmp_path, run_benchmark):
    aux = write_receipt(tmp_path / "runtime_aux.json", 0.0125)
    flip = write_receipt(tmp_path / "runtime_flip.json", 0.01825)
    runtime = run_benchmark(aux, flip_runtime_path=flip, images=2)

    assert runtime["benchmark_images"] == 2
    assert runtime["backbone_device"] == "cuda"
    assert runtime["backbone_single_pass_ms_per_image"] == pytest.approx(12.5)
    assert runtime["backbone_two_pass_flip_ms_per_image"] == pytest.approx(30.75)
    assert runtime["end_to_end_v2_single_pass_cpu_post_ms"] == pytest.approx(13)
    assert runtime["end_to_end_v2_flip_tta_cpu_post_ms"] == pytest.approx(31.25)
    assert runtime["end_to_end_timing_kind"] == "estimated_sum_of_separately_measured_components"
    assert runtime["postprocessing_timing_kind"] == "measured_cached_inputs"
    for key, path in (("aux", aux), ("flip", flip)):
        source = runtime["backbone_timing_sources"][key]
        assert source["path"] == str(path)
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert source["gpu_name"] == "Some GPU"


def test_missing_receipts_keep_measured_postprocessing(tmp_path, run_benchmark):
    runtime = run_benchmark(tmp_path / "runtime_aux.json")
    assert runtime["backbone_timing_sources"] == {"aux": None, "flip": None}
    assert runtime["backbone_device"] is None
    assert all(value is None for key, value in runtime.items() if key.endswith("_ms") and key.startswith("end_to_end_"))
    assert runtime["v2_single_pass_cpu_post_ms"] == pytest.approx(0.1)


@pytest.mark.parametrize(("images", "limit"), [(0, 10), (10, 0), (10, -1)])
def test_requires_positive_benchmark_size(tmp_path, run_benchmark, images, limit):
    with pytest.raises(ValueError, match="positive n_images.*at least one image"):
        run_benchmark(tmp_path / "runtime_aux.json", images=images, n_images=limit)


@pytest.mark.parametrize("duration", [0, -0.01, True, float("nan"), float("inf"), "0.0125"])
def test_rejects_invalid_component_time(tmp_path, run_benchmark, duration):
    aux = write_receipt(tmp_path / "runtime_aux.json", duration)
    with pytest.raises(ValueError, match=r"runtime_aux\.json.*mean_seconds_per_image"):
        run_benchmark(aux)


@pytest.mark.parametrize(("flip_device", "flip_gpu"), [("cpu", "CPU"), ("cuda", "A different GPU")])
def test_rejects_mismatched_backbone_devices(tmp_path, run_benchmark, flip_device, flip_gpu):
    aux = write_receipt(tmp_path / "runtime_aux.json", 0.0125)
    flip = write_receipt(tmp_path / "runtime_flip.json", 0.01825, flip_device, flip_gpu)
    with pytest.raises(ValueError, match=r"runtime_aux\.json.*runtime_flip\.json.*device"):
        run_benchmark(aux, flip_runtime_path=flip)


@pytest.mark.parametrize("post_gpu", ["Some GPU", "Another GPU"])
def test_gpu_composite_requires_matching_hardware(tmp_path, monkeypatch, run_benchmark, post_gpu):
    memory_events = []

    def reset(_device):
        memory_events.append("reset")

    def allocated(_device):
        memory_events.append("allocated")
        return 100 * memory_events.count("reset")

    def reserved(_device):
        memory_events.append("reserved")
        return 300 * memory_events.count("reset")

    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluation.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(evaluation.torch.cuda, "get_device_name", lambda _index: post_gpu)
    monkeypatch.setattr(evaluation.torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(evaluation.torch.cuda, "max_memory_allocated", allocated)
    monkeypatch.setattr(evaluation.torch.cuda, "max_memory_reserved", reserved)
    monkeypatch.setattr(
        evaluation, "_apply_v2_gpu_exact",
        lambda depth, *_a, **_kw: evaluation.torch.from_numpy(depth),
    )
    aux = write_receipt(tmp_path / "runtime_aux.json", 0.0125)
    flip = write_receipt(tmp_path / "runtime_flip.json", 0.01825)
    runtime = run_benchmark(aux, flip_runtime_path=flip, images=2)

    assert runtime["gpu_name"] == post_gpu
    assert runtime["v2_single_pass_gpu_post_ms"] == pytest.approx(0.5)
    assert runtime["v2_flip_tta_gpu_post_ms"] == pytest.approx(0.5)
    assert runtime["v2_single_pass_gpu_post_peak_allocated_bytes"] == 100
    assert runtime["v2_single_pass_gpu_post_peak_reserved_bytes"] == 300
    assert runtime["v2_flip_tta_gpu_post_peak_allocated_bytes"] == 200
    assert runtime["v2_flip_tta_gpu_post_peak_reserved_bytes"] == 600
    assert memory_events == ["reset", "allocated", "reserved", "reset", "allocated", "reserved"]
    assert "cached-input post-processing" in runtime["postprocessing_cuda_peak_scope"]
    if post_gpu == "Some GPU":
        assert runtime["end_to_end_v2_single_pass_gpu_ms"] == pytest.approx(13)
        assert runtime["end_to_end_v2_flip_tta_gpu_ms"] == pytest.approx(31.25)
    else:
        assert runtime["end_to_end_v2_single_pass_gpu_ms"] is None
        assert runtime["end_to_end_v2_flip_tta_gpu_ms"] is None
