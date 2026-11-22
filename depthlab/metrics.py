import numpy as np

CROP = (slice(45, 471), slice(41, 601))
MIN_DEPTH = 0.001
MAX_DEPTH = 10.0
METRIC_NAMES = ("abs_rel", "rmse", "delta1")
DEPTH_BINS = (MIN_DEPTH, 1, 2, 4, 6, 10)


def valid_depth(target):
    return np.isfinite(target) & (target > MIN_DEPTH) & (target < MAX_DEPTH)


def clip_depth(prediction):
    return np.clip(prediction, MIN_DEPTH, MAX_DEPTH)


def metrics(prediction, target, mask=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = valid_depth(target) if mask is None else np.asarray(mask)
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Metric arrays must have equal shapes")
    if valid.dtype != np.bool_:
        raise ValueError("Metric mask must be boolean")
    if not np.any(valid):
        raise ValueError("Metric arrays require a nonempty GT mask")
    if not np.all(valid_depth(target[valid])):
        raise ValueError("Metric mask includes invalid target depths")
    if not np.all(np.isfinite(prediction[valid])):
        raise ValueError("Nonfinite predictions on valid evaluation pixels")
    p = clip_depth(prediction[valid])
    g = target[valid]
    return np.array([np.mean(np.abs(p - g) / g),
                     np.sqrt(np.mean((p - g) ** 2)),
                     np.mean(np.maximum(p / g, g / p) < 1.25)])


def evaluate_image(prediction, target):
    return metrics(prediction[CROP], target[CROP])
