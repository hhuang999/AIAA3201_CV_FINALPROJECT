import numpy as np
import pytest

from depthlab.metrics import metrics, evaluate_image


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_rejected_on_valid_pixels(bad):
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        metrics(np.array([bad, 2.]), np.array([1., 2.]))


def test_empty_valid_mask_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        metrics(np.ones(2), np.array([0., 10.]))


def test_mismatched_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        metrics(np.ones((2, 1)), np.ones(2))


def test_invalid_target_pixels_do_not_require_finite_predictions():
    result = metrics(np.array([np.nan, np.inf, 2.]), np.array([0., 10., 2.]))
    np.testing.assert_array_equal(result, [0., 0., 1.])


def test_delta_threshold_is_strict_and_depth_clipping_precedes_scoring():
    np.testing.assert_allclose(metrics(np.array([1.25]), np.array([1.])), [.25, .25, 0.])
    np.testing.assert_allclose(metrics(np.array([20.]), np.array([5.])), [1., 5., 0.])


def test_crop_and_equal_image_rmse_protocol():
    gt = np.ones((480, 640))
    pred = np.full((480, 640), np.nan)
    pred[45:471, 41:601] = 1.
    np.testing.assert_array_equal(evaluate_image(pred, gt), [0., 0., 1.])
