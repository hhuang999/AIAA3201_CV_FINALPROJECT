"""Entrypoint contracts for shell pipelines without inference or figure generation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("explicit", [False, True])
def test_shell_defaults_compose_and_explicit_paths_are_preserved(tmp_path, explicit):
    recorder = tmp_path / "python-recorder"
    recorder.write_text(
        "#!" + sys.executable + "\nimport json,os,sys\n"
        "with open(os.environ['RECORDER_LOG'], 'a') as out: out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
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
    cache_build = next(call for call in calls if call[1] == "depthlab.geocalib_cache")
    evaluation = next(call for call in calls if call[1] == "depthlab.geocalib_evaluate")

    def option(call, key):
        return call[call.index(key) + 1]

    expected_output = paths[2] if explicit else "outputs"
    expected_eval_out = paths[3] if explicit else "results/part3_geocalib"
    assert option(nyu, "--output") == expected_output + "/nyu_rgb"
    assert option(cache_build, "--orig-preds") == option(nyu, "--output")
    assert option(evaluation, "--orig-preds") == option(nyu, "--output")
    assert option(evaluation, "--output") == expected_eval_out
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "results").exists()
