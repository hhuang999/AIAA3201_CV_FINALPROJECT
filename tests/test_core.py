import numpy as np

from depthlab.metrics import metrics
from depthlab.stereo import disparity_to_depth
from depthlab.calibration import apply_curve, apply_residual, fit_scale, scene_folds


def test_metric_conversion():
    np.testing.assert_allclose(disparity_to_depth(np.array([10., 20.]), 500, 100, 5),
                               [50 / 15, 2])


def test_depth_scores():
    gt = np.array([1., 2., 4.])
    np.testing.assert_allclose(metrics(gt, gt), [0, 0, 1])
    np.testing.assert_allclose(metrics(2 * gt, gt), [1, np.sqrt(7), 0])


def test_training_scale():
    assert fit_scale(np.array([.8, 1.2, .95, .9, 2.])) == .95


def test_monotone_curve():
    depth = np.geomspace(.01, 100, 200)
    curve = {"offset": -.2, "slopes": [.25, 2., .6, 1.1, .9]}
    assert np.all(np.diff(apply_curve(depth, curve)) >= 0)
    np.testing.assert_allclose(apply_curve(depth, {"offset": np.log(.9), "slopes": [1] * 5}),
                               np.clip(.9 * depth, .001, 10))


def test_residual_ratio_bounds():
    model = {"mean": [0] * 8, "std": [1] * 8, "intercept": 0,
             "coefficient": [1e3] * 8, "shrinkage": 1}
    ratios = np.array([apply_residual(np.array([3.]), np.full(8, sign), model)[0] / 3
                       for sign in (-1, 1)])
    np.testing.assert_allclose(ratios, np.exp([-.15, .15]))


def test_scene_separation():
    scenes = np.repeat([f"room_{i}" for i in range(19)], 3)
    folds = scene_folds(scenes)
    for i, first in enumerate(folds):
        for second in folds[i + 1:]:
            assert not set(scenes[first]) & set(scenes[second])
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(scenes)))
