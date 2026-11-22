"""Regression checks for public help and project-only test discovery."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments", [
    ["-m", "depthlab.sota_calibration", "--help"],
    ["-m", "depthlab.sota_calibration", "learning-curve", "--help"],
    ["-m", "depthlab.sota_v2_calibration", "--help"],
    ["-m", "depthlab.sota_v2_cache", "build", "--help"],
    ["-m", "depthlab.sota_v2_evaluate", "--help"],
])
def test_public_help_requires_no_model_or_data(arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_default_collection_does_not_import_upstream_gpu_scripts(tmp_path):
    """An installed third-party repository must not expand our test suite."""
    shutil.copyfile(ROOT / "pytest.ini", tmp_path / "pytest.ini")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_project.py").write_text("def test_project():\n    pass\n")
    upstream = tmp_path / "third_party/Video-Depth-Anything/loss"
    upstream.mkdir(parents=True)
    (upstream / "test_loss.py").write_text(
        "raise RuntimeError('Unrelated upstream GPU script was imported')\n")
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "--collect-only",
                             "-q", "-p", "no:cacheprovider"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/test_project.py::test_project" in result.stdout
    assert "third_party" not in result.stdout
