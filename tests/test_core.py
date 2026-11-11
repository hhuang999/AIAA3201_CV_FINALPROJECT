import numpy as np

from depthlab.metrics import metrics
from depthlab.stereo import disparity_to_depth


def test_metric_conversion():
    np.testing.assert_allclose(disparity_to_depth(np.array([10., 20.]), 500, 100, 5),
                               [50 / 15, 2])


def test_depth_scores():
    gt = np.array([1., 2., 4.])
    np.testing.assert_allclose(metrics(gt, gt), [0, 0, 1])
    np.testing.assert_allclose(metrics(2 * gt, gt), [1, np.sqrt(7), 0])
