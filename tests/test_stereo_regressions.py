"""Regressions for the frozen Part 1 geometry and artifact contract."""
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest

from depthlab import stereo


def texture_pair(seed=99):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (64, 160, 3), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    right[:, :-8] = left[:, 8:]
    return left, right


def test_invalid_disparities_never_become_metric_depth():
    disparity = np.array([[0., -1., 10., 30., np.inf, np.nan]], np.float32)
    actual = stereo.disparity_to_depth(disparity, 100., 200., 10.)
    np.testing.assert_allclose(actual[0, 2:4], [1., .5])
    assert actual.dtype == np.float32
    assert np.isnan(actual[0, [0, 1, 4, 5]]).all()


def test_reverse_halfpixel_lookup_keeps_frozen_numpy_rint_semantics():
    forward = np.ones((1, 32), np.float32)
    forward[0, :4] = [-1., .5, 1., 1.]
    reverse = np.ones((1, 32), np.float32)
    reverse[0, :4] = [.5, 2., 1., 1.]
    calls = iter([forward, reverse[:, ::-1]])
    class Matcher:
        def compute(self, first, second):
            return next(calls) * 16
    with patch.object(cv2, "StereoSGBM_create", return_value=Matcher()):
        _, valid, _ = stereo.match_pair(
            np.zeros((1, 32), np.uint8), np.zeros((1, 32), np.uint8),
            {"ndisp": 16}, cv2.STEREO_SGBM_MODE_SGBM_3WAY, False)
    # mapped x=.5 ties to zero under np.rint; reverse disparity is .5 there.
    assert valid[0, 1]


def test_lr_rejects_occluded_zero_and_unrounded_outside_locations():
    left = np.array([[.25, 1., 1., np.nan, np.inf, 1e30]], np.float32)
    right = np.array([[1., 0., 1., 1., 1., 1.]], np.float32)
    residual = stereo.lr_residual(left, right)
    assert np.isinf(residual[0, [0, 2, 3, 4, 5]]).all()
    assert residual[0, 1] == 0


@pytest.mark.parametrize("mode", ["SGBM_3WAY", "HH4", "HH"])
@pytest.mark.parametrize("padding", ["none", "replicate", "reflect101"])
def test_translated_rgb_texture_and_real_border_geometry(mode, padding):
    cv2.setNumThreads(1)
    left, right = texture_pair()
    calib = {"ndisp": 32}
    cfg = stereo.SGBMConfig(mode=mode, padding_mode=padding)
    disparity = stereo.stereo_sgbm(left, right, calib, cfg)
    reverse = stereo.reverse_disparity(left, right, calib, cfg)
    keep = stereo.lr_residual(disparity, reverse) <= 1.
    core = keep[8:-8, 32:-32]
    assert core.mean() > .95
    assert np.abs(disparity[8:-8, 32:-32][core] - 8).mean() < .15
    if padding != "none":
        y, x = np.nonzero(disparity > 0)
        assert ((x-disparity[y, x] >= 0) & (x-disparity[y, x] <= 159)).all()
        y, x = np.nonzero(reverse > 0)
        assert ((x+reverse[y, x] >= 0) & (x+reverse[y, x] <= 159)).all()


def test_color_decode_then_conversion_matches_frozen_inputs(tmp_path):
    left, right = texture_pair(0)
    scene = tmp_path / "scene"
    scene.mkdir()
    cv2.imwrite(str(scene/"im0.png"), left)
    cv2.imwrite(str(scene/"im1.png"), right)
    cv2.imwrite(str(scene/"mask0nocc.png"), np.full((64, 160), 255, np.uint8))
    (scene/"calib.txt").write_text(
        "cam0=[100 0 80; 0 100 32; 0 0 1]\nbaseline=200\ndoffs=0\nwidth=160\nheight=64\nndisp=32\n")
    with (scene/"disp0GT.pfm").open("wb") as stream:
        stream.write(b"Pf\n160 64\n-1.0\n")
        np.full((64,160), 8., dtype="<f4").tofile(stream)
    decoded_left, decoded_right, _, labels, _ = stereo.load_scene(scene)
    np.testing.assert_array_equal(decoded_left, left)
    np.testing.assert_array_equal(decoded_right, right)
    assert labels.shape == (64,160)
    cfg = stereo.SGBMConfig(padding_mode="reflect101")
    expected = stereo.stereo_sgbm(left, right, {"ndisp":32}, cfg)
    actual, _, _ = stereo.match_pair(decoded_left, decoded_right, {"ndisp":32},
        cv2.STEREO_SGBM_MODE_SGBM_3WAY, True)
    np.testing.assert_array_equal(actual, expected)


def test_region_coverage_counts_occlusions_without_changing_denominator():
    pred=np.array([1.,4.,-1.,np.nan])
    gt=np.array([2.,2.,3.,np.inf])
    keep=np.array([True,True,False,False])
    values=stereo.region_counts(pred,gt,np.isfinite(gt)&(gt>0),keep)
    assert values["gt_pixels"] == 3
    assert values["retained_pixels"] == 2
    assert values["mae_px"] == 1.5
    assert values["coverage"] == 2/3
    assert values["good1_yield"] == 1/3


def test_run_exports_primary_and_each_ablation_with_region_diagnostics(tmp_path, monkeypatch):
    import json
    scene = tmp_path/"data/MiddEval3/trainingQ/scene"
    scene.mkdir(parents=True)
    (scene/"calib.txt").write_text("synthetic fixture, loader supplied by test")
    left, right = texture_pair()
    target = np.full((64,160), 8., np.float32)
    labels = np.full(target.shape, 255, np.uint8)
    labels[:, :80] = 128
    calib = {"ndisp":32,"focal":100.,"baseline":200.,"doffs":0.,"width":160,"height":64}
    monkeypatch.setattr(stereo, "load_scene", lambda folder: (left,right,target,labels,calib))
    output = tmp_path/"results"
    frame = stereo.run(tmp_path/"data", output, limit=1, threads=1,
                       no_images=True, benchmark_passes=2)
    assert set(frame["configuration"]) == {"unpad_3way","reflect_3way","reflect_hh4"}
    assert frame.loc[frame["primary"],"configuration"].tolist() == ["reflect_3way"]
    for configuration in frame["configuration"]:
        for suffix in ("disparity","valid","depth"):
            assert (output/"arrays"/configuration/f"scene_{suffix}.npy").is_file()
    for suffix in ("disparity","valid","depth"):
        np.testing.assert_array_equal(np.load(output/f"scene_{suffix}.npy"),
            np.load(output/"arrays/reflect_3way"/f"scene_{suffix}.npy"))
    import pandas as pd
    regions = pd.read_csv(output/"stereo_regions_per_scene.csv")
    assert set(regions["region"]) == {"all","nonoccluded","occluded","unknown_label"}
    assert (regions.loc[regions["region"]=="all","gt_pixels"] == target.size).all()
    assert (output/"stereo_paired_populations.csv").is_file()
    assert (output/"disparity_sensitivity.csv").is_file()
    runtime = json.loads((output/"stereo_runtime.json").read_text())
    assert runtime["primary_configuration"] == "reflect_3way"
    assert runtime["external_lr"] is True
    assert runtime["complete_course_set"] is False
    benchmark = json.loads((output/"stereo_cpu_benchmark.json").read_text())
    assert benchmark["measurements_per_method"] == 2
    assert len(benchmark["measurements"]) == 6


def test_no_external_lr_ablation_preserves_matcher_knobs(monkeypatch):
    calls = []
    class Matcher:
        def compute(self, first, second):
            return np.full(first.shape, 8*16, np.int16)
    def create(**kwargs):
        calls.append(kwargs)
        return Matcher()
    monkeypatch.setattr(cv2, "StereoSGBM_create", create)
    def forbidden(*args, **kwargs):
        raise AssertionError("No-LR ablation must not execute the reverse matcher")
    monkeypatch.setattr(stereo, "reverse_disparity", forbidden)
    cfg = stereo.SGBMConfig(block_size=7, uniqueness_ratio=3,
        speckle_window_size=20, speckle_range=1, disp12_max_diff=2,
        padding_mode="replicate", mode="HH")
    left, right = texture_pair()
    disparity, keep, _ = stereo.match_pair(left,right,{"ndisp":32},
        config=cfg, external_lr=False, lr_tolerance=2.)
    assert len(calls) == 1
    assert calls[0]["blockSize"] == 7
    assert calls[0]["P1"] == 8*7**2 and calls[0]["P2"] == 32*7**2
    assert calls[0]["uniquenessRatio"] == 3
    assert calls[0]["speckleWindowSize"] == 20
    assert calls[0]["disp12MaxDiff"] == 2
    assert calls[0]["mode"] == cv2.STEREO_SGBM_MODE_HH
    np.testing.assert_array_equal(keep,np.isfinite(disparity)&(disparity>0))
    assert not keep[:,:8].any()


@pytest.mark.parametrize("kwargs", [
    {"block_size":2}, {"uniqueness_ratio":-1}, {"speckle_range":-1},
    {"padding_mode":"wrap"}, {"mode":"unknown"}, {"disp12_max_diff":0}])
def test_invalid_matcher_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        stereo.SGBMConfig(**kwargs)
