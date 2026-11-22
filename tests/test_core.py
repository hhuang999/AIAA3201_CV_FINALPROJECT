import numpy as np

from depthlab.geocalib_io import scene_folds_seeded
from depthlab.geocalib_model import DEPTH_KNOTS, apply_spline, fit_scale
from depthlab.metrics import metrics
from depthlab.stereo import disparity_to_depth


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
    y_knots = (DEPTH_KNOTS + np.log(.9)).tolist()
    curve = {"x_knots": DEPTH_KNOTS.tolist(), "y_knots": y_knots}
    assert np.all(np.diff(apply_spline(depth, curve)) >= 0)
    np.testing.assert_allclose(apply_spline(depth, curve), np.clip(.9 * depth, .001, 10))


def test_scene_separation():
    scenes = np.repeat([f"room_{i}" for i in range(19)], 3)
    folds = scene_folds_seeded(scenes)
    for i, first in enumerate(folds):
        for second in folds[i + 1:]:
            assert not set(scenes[first]) & set(scenes[second])
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(len(scenes)))
