"""Self-contained deployment of the previously selected, frozen Part 3B model.

The original artifact is retained intact. Its historical source hashes describe
its origin; optional whole-file SHA-256 verification binds the imported bytes.
These functions never fit, select a model, read GT, or import the old checkout.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

CROP = (slice(45, 471), slice(41, 601))


def depth_features(depth):
    """Use exactly the original positive raw-depth feature arithmetic."""
    p = np.asarray(depth, dtype=np.float64)
    if p.shape != (480, 640) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Expected finite positive raw 480x640 depth")
    lp = np.log(p[CROP])
    gradient = .5 * (np.mean(np.abs(np.diff(lp, axis=0))) +
                     np.mean(np.abs(np.diff(lp, axis=1))))
    return np.r_[np.quantile(lp, [.1, .25, .5, .75, .9]),
                 lp.mean(), lp.std(), gradient]


def apply_spline(depth, model):
    """Original log-space interpolation/extrapolation, then depth clipping."""
    p = np.asarray(depth, dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Raw depth must be finite and positive")
    x, y = np.asarray(model["x_knots"]), np.asarray(model["y_knots"])
    lp = np.log(p)
    i = np.clip(np.searchsorted(x, lp, side="right") - 1, 0, len(x) - 2)
    out = np.exp(y[i] + (lp - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]))
    return np.clip(out, .001, 10.)


def predict_residual_scale(model, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[-1] != 8 or not np.isfinite(x).all():
        raise ValueError("Expected finite features with shape (8,) or (N,8)")
    if model["kind"] == "constant_residual":
        prediction = np.full(x.shape[:-1], model["bias"], dtype=np.float64)
    else:
        prediction = (model["bias"] +
                      ((x - np.asarray(model["feature_mean"])) /
                       np.asarray(model["feature_std"])) @ np.asarray(model["coefficients"]))
    return np.exp(np.clip(model["shrinkage"] * prediction,
                          -model["max_log_scale"], model["max_log_scale"]))


def _vector(value, length, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ValueError("Invalid frozen " + name)
    return a


def _validate_frozen(frozen):
    """Validate a complete legacy artifact without changing any metadata.

    Historical source hashes are declarations, not hashes of this rewritten
    package. Call load_frozen(..., expected_sha256=...) to verify trusted bytes.
    """
    if not isinstance(frozen, dict) or frozen.get("schema") != "B_residual_v1" or frozen.get("complete") is not True:
        raise ValueError("Expected complete B_residual_v1 frozen artifact")
    required = ("primary", "candidate_spec", "base_model", "residual_model", "controls",
                "train_ids", "training_scenes", "source_sha256", "dataset_sha256",
                "split_sha256", "train_cache_sha256", "selection_sha256",
                "selection_protocol_sha256", "prepared_manifest_sha256", "base_frozen_sha256",
                "test_read", "historical_test_exposure", "final_decision")
    if any(k not in frozen for k in required):
        raise ValueError("Frozen artifact is missing selection/provenance metadata")
    for name in required:
        if name.endswith("sha256") and name != "source_sha256":
            if not isinstance(frozen[name], str) or re.fullmatch("[0-9a-f]{64}", frozen[name]) is None:
                raise ValueError("Invalid provenance digest: " + name)
    if (not isinstance(frozen["source_sha256"], dict) or not frozen["source_sha256"] or
            any(not isinstance(v, str) or re.fullmatch("[0-9a-f]{64}", v) is None
                for v in frozen["source_sha256"].values())):
        raise ValueError("Invalid historical source digests")
    ids = frozen["train_ids"]
    if (not isinstance(ids, list) or len(ids) != 795 or len(set(ids)) != 795 or
            any(type(i) is not int or not 0 <= i < 1449 for i in ids) or frozen["training_scenes"] != 249):
        raise ValueError("Expected 795 unique official training identities and 249 scenes")
    if frozen["test_read"] is not False or frozen["historical_test_exposure"] is not True:
        raise ValueError("Frozen selection must retain its historical exposure disclosure")
    base = frozen["base_model"]
    if base.get("kind") != "spline":
        raise ValueError("Expected frozen spline base")
    x = _vector(base.get("x_knots"), 6, "knots")
    y = _vector(base.get("y_knots"), 6, "ordinates")
    slopes = np.diff(y) / np.diff(x)
    if (not np.array_equal(x, np.log([.3, .7, 1.5, 3., 6., 12.])) or
            np.any(slopes < .25 - 1e-12) or np.any(slopes > 2 + 1e-12) or
            abs(y[0] - x[0]) > np.log(2) + 1e-12):
        raise ValueError("Frozen spline violates its declared family bounds")
    residual, spec = frozen["residual_model"], frozen["candidate_spec"]
    if spec.get("name") != frozen["primary"] or spec.get("target") not in ("median_ratio", "rmse_ratio"):
        raise ValueError("Frozen primary/candidate specification mismatch")
    kind = residual.get("kind")
    expected_kind = "ridge" if kind == "ridge_residual" else "constant" if kind == "constant_residual" else None
    if expected_kind is None or spec.get("kind") != expected_kind:
        raise ValueError("Unknown/mismatching frozen residual kind")
    shrinkage, limit, bias = (residual.get("shrinkage"), residual.get("max_log_scale"), residual.get("bias"))
    if (not np.isfinite([shrinkage, limit, bias]).all() or not 0 <= shrinkage <= 1 or
            limit != .15 or shrinkage != spec.get("shrinkage") or limit != spec.get("max_log_scale")):
        raise ValueError("Invalid or mismatching frozen residual bounds")
    canonical = (f"residual_{spec['target']}_l{spec.get('lam', 0):g}_a{shrinkage:g}"
                 if kind == "ridge_residual" else f"constant_{spec['target']}")
    if frozen["primary"] != canonical or shrinkage not in (.5, 1.) or (kind == "constant_residual" and shrinkage != 1.):
        raise ValueError("Frozen name is inconsistent with its candidate specification")
    if kind == "ridge_residual":
        _vector(residual.get("feature_mean"), 8, "feature mean")
        std = _vector(residual.get("feature_std"), 8, "feature standard deviation")
        _vector(residual.get("coefficients"), 8, "coefficients")
        lam = residual.get("ridge_lambda")
        if np.any(std <= 0) or lam not in (.1, 1., 10.) or lam != spec.get("lam"):
            raise ValueError("Invalid frozen ridge parameters")
    controls = frozen["controls"]
    if not isinstance(controls, dict) or set(controls) != {"raw", "median", "spline_rmse"} or controls["spline_rmse"] != base:
        raise ValueError("Frozen controls must retain the selected spline base")
    if controls["raw"] != {"kind": "scalar", "scale": 1.}:
        raise ValueError("Invalid raw control")
    median = controls["median"]
    if median.get("kind") != "scalar" or not np.isfinite(median.get("scale", np.nan)) or median["scale"] <= 0:
        raise ValueError("Invalid median control")
    return frozen


def validate_frozen(frozen):
    """Validate the unchanged artifact; malformed fields raise ValueError."""
    try:
        return _validate_frozen(frozen)
    except (TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError("Malformed frozen artifact") from exc


def load_frozen(path, expected_sha256=None):
    """Read the unchanged original JSON, optionally verify its whole-file hash."""
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen artifact SHA-256 mismatch")
    return validate_frozen(json.loads(payload))


def apply_frozen(raw_depth, frozen):
    """Apply the supplied already-selected candidate; no GT or reselection."""
    features = depth_features(raw_depth)
    base = apply_spline(raw_depth, frozen["base_model"])
    return np.clip(base * predict_residual_scale(frozen["residual_model"], features), .001, 10.)


def frozen_predictions(raw_depth, frozen):
    """Return old controls and residual under the rebuilt consumer names."""
    p = np.asarray(raw_depth, dtype=np.float64)
    features = depth_features(p)
    base = apply_spline(p, frozen["base_model"])
    return {"raw": np.clip(p, .001, 10.),
            "global_scale": np.clip(p * frozen["controls"]["median"]["scale"], .001, 10.),
            "monotone_curve": base,
            "residual_scale": np.clip(base * predict_residual_scale(frozen["residual_model"], features), .001, 10.)}
