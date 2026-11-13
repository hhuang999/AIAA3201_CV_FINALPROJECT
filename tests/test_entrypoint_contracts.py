"""Entrypoint contracts without inference or presentation generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from depthlab import figures, summarize

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("explicit", [False, True])
def test_shell_defaults_compose_and_explicit_paths_are_preserved(tmp_path, explicit):
    recorder = tmp_path / "python-recorder"
    recorder.write_text("#!" + sys.executable + "\nimport json,os,sys\nwith open(os.environ['RECORDER_LOG'], 'a') as out: out.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    recorder.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, METRIC_PYTHON=str(recorder), VIDEO_PYTHON=str(recorder), RECORDER_LOG=str(log))
    paths = ["data directory", "source directory", "output directory", "result directory"]
    baseline_args = paths if explicit else []
    evaluation_args = [paths[0], paths[2], paths[3]] if explicit else []
    subprocess.run(["bash", str(ROOT / "scripts/run_baselines.sh"), *baseline_args], cwd=tmp_path, env=env, check=True)
    subprocess.run(["bash", str(ROOT / "scripts/run_evaluation.sh"), *evaluation_args], cwd=tmp_path, env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    nyu = next(call for call in calls if call[1] == "depthlab.infer_nyu")
    evaluation = next(call for call in calls if call[1] == "depthlab.evaluate")
    def option(call, key): return call[call.index(key) + 1]
    assert option(nyu, "--output") == option(evaluation, "--predictions")
    expected_output = paths[2] if explicit else "outputs/reproduction"
    expected_results = paths[3] if explicit else "results/reproduction"
    assert option(nyu, "--output") == expected_output + "/nyu_rgb"
    assert option(evaluation, "--output") == expected_results + "/frozen_replay"
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "results").exists()


def historical_inputs(tmp_path):
    results, outputs, data, predictions = [tmp_path / name for name in ("results", "outputs", "data", "predictions")]
    results.mkdir()
    parameters = {"global_scale": .95, "monotone_curve": {"offset": 0., "slopes": [1.] * 5},
                  "residual": None, "selected_method": "monotone_curve", "seed": 42}
    (results / "calibration_parameters.json").write_text(json.dumps(parameters))
    for name in ("test_summary.csv", "test_bins_summary.csv", "calibration_oof_summary.csv", "test_cases.csv",
                 "camera_diagnostic_per_image.csv", "calibration_oof_per_image.csv", "postprocessing_per_image.csv",
                 "stereo_summary.csv", "camera_diagnostic_summary.csv"):
        (results / name).write_text("synthetic fixture\n")
    (results / "camera_diagnostic.json").write_text("{}")
    (results / "figures/video").mkdir(parents=True)
    (results / "figures/video/video_adjacent_change.csv").write_text("synthetic fixture\n")
    for folder in ("nyu_rgb", "nyu_known", "video_image", "video_temporal"):
        target = outputs / folder
        target.mkdir(parents=True)
        (target / "runtime.json").write_text("{}")
        (target / ("inference_per_frame.csv" if folder.startswith("video") else "inference_per_image.csv")).write_text("seconds\n1\n")
    data.mkdir()
    for name in ("nyu_depth_v2_labeled.mat", "splits.mat"): (data / name).write_bytes(b"fixture")
    predictions.mkdir()
    for name in ("depth.npy", "indices.npy", "manifest.json"): (predictions / name).write_bytes(b"fixture")
    return SimpleNamespace(results=results, outputs=outputs, data_root=data, predictions=predictions, readme=None), parameters


@pytest.mark.parametrize("module", [figures, summarize])
@pytest.mark.parametrize("kind", ["missing", "frozen"])
def test_presentation_wrong_directory_fails_before_any_write(tmp_path, monkeypatch, module, kind):
    args = SimpleNamespace(results=tmp_path / "replay", outputs=tmp_path / "outputs", data_root=tmp_path / "data",
                           predictions=tmp_path / "predictions", readme=None)
    if kind == "frozen":
        args.results.mkdir()
        (args.results / "calibration_parameters.json").write_text(json.dumps({"schema": "B_residual_v1", "complete": True}))
    before = snapshot(tmp_path)
    monkeypatch.setattr(figures.plt, "subplots", lambda *a, **k: pytest.fail("Plotting began before validation"))
    with pytest.raises(ValueError, match="historical|Historical"):
        getattr(module, module.__name__.rsplit(".", 1)[-1])(args)
    assert snapshot(tmp_path) == before
    assert not (args.results / "runtime").exists()
    assert not (args.results / "figures").exists()


@pytest.mark.parametrize("module,missing", [(figures, "test_cases.csv"), (summarize, "calibration_oof_per_image.csv")])
def test_presentation_missing_required_artifact_leaves_files_untouched(tmp_path, monkeypatch, module, missing):
    args, _ = historical_inputs(tmp_path)
    (args.results / missing).unlink()
    before = snapshot(tmp_path)
    monkeypatch.setattr(figures.plt, "subplots", lambda *a, **k: pytest.fail("Plotting began before validation"))
    with pytest.raises(ValueError, match="historical|Historical"):
        getattr(module, module.__name__.rsplit(".", 1)[-1])(args)
    assert snapshot(tmp_path) == before
    assert not (args.results / "runtime").exists()
    assert not (args.results / "figures/nyu").exists()


@pytest.mark.parametrize("content", ["# Current README\n", "<!-- results:start --><!-- results:start --><!-- results:end -->",
                                      "<!-- results:end --><!-- results:start -->"])
def test_readme_markers_fail_before_partial_summary(tmp_path, content):
    args, _ = historical_inputs(tmp_path)
    args.readme = tmp_path / "README.md"
    args.readme.write_text(content)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="marker"):
        summarize.summarize(args)
    assert snapshot(tmp_path) == before
    assert not (args.results / "runtime").exists()


@pytest.mark.parametrize("module", [figures, summarize])
def test_complete_historical_input_preflight_remains_available(tmp_path, module):
    args, parameters = historical_inputs(tmp_path)
    args.readme = tmp_path / "README.md"
    args.readme.write_text("Before\n<!-- results:start -->old<!-- results:end -->\nAfter")
    before = snapshot(tmp_path)
    assert module.validate_inputs(args) == parameters
    assert snapshot(tmp_path) == before
