"""Unit tests for Round 2 UniDepthV2 caching, split isolation, and post-v1 diagnosis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthlab.sota_calibration import CROP_SUB
from depthlab.sota_io import scene_folds_seeded
from depthlab.sota_v2_cache import (
    compute_confidence_summary,
    ensure_unidepth_weights,
    load_v2_cache,
)
from depthlab.sota_v2_diagnose import (
    cohens_d,
    compute_flip_features,
    roc_auc_mann_whitney,
    scene_cv_ridge_r2_pca,
)


def test_ensure_unidepth_weights_rejects_tampered_digest(tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "config.json").write_bytes(b"{}")
    (wdir / "model.safetensors").write_bytes(b"fake")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ensure_unidepth_weights(
            wdir,
            expected_hashes={"config.json": "0" * 64, "model.safetensors": "1" * 64},
        )


def test_confidence_summary_and_flip_features_sanity():
    rng = np.random.default_rng(42)
    conf1 = rng.uniform(0.5, 4.0, size=(480, 640))
    conf2 = conf1 * rng.uniform(0.95, 1.05, size=(480, 640))
    stats = compute_confidence_summary(conf1)
    assert stats.shape == (17,)
    assert np.isfinite(stats).all()

    p1 = rng.uniform(0.8, 5.0, size=(480, 640))
    p2 = p1 * 1.02
    k1 = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    k2 = np.array([[520.0, 0.0, 319.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])
    ff = compute_flip_features(
        p1,
        k1,
        p2,
        k2,
        conf_orig=conf1,
        conf_flip=conf2,
        cls_orig=np.ones(16),
        cls_flip=np.ones(16),
    )
    assert abs(ff["f_orig_tilde"] - 500.0 / 480.0) < 1e-12
    assert abs(ff["f_flip_tilde"] - 520.0 / 480.0) < 1e-12
    assert abs(ff["f_abs_diff"] - 20.0 / 480.0) < 1e-12
    assert abs(ff["cls_flip_cos"] - 1.0) < 1e-12
    assert np.isfinite(list(ff.values())).all()


def test_roc_auc_cohens_d_and_scene_cv_r2_pca():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([False, False, False, True, True, True])
    assert abs(roc_auc_mann_whitney(scores, labels) - 1.0) < 1e-12
    assert cohens_d(scores[labels], scores[~labels]) > 5.0

    # Fold-strict PCA + dual ridge check on high-dim synthetic signal
    rng = np.random.default_rng(13)
    n, d = 60, 128
    scenes = np.array([f"sc_{i // 3}" for i in range(n)])
    folds = scene_folds_seeded(scenes, count=5, seed=20261031)
    latent = rng.normal(size=(n, 3))
    proj = rng.normal(size=(3, d))
    x = latent @ proj + 0.05 * rng.normal(size=(n, d))
    y = 0.7 * latent[:, 0] - 0.5 * latent[:, 1] + 0.05 * rng.normal(size=n)

    r2, oof = scene_cv_ridge_r2_pca(x, y, folds, lam=0.1, pca_k=8)
    assert oof.shape == (n,) and np.isfinite(oof).all()
    assert r2 > 0.85


def test_v2_cache_split_guard_and_shapes_if_present():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"
    orig_dir = repo / "outputs" / "nyu_rgb"
    aux_dir = repo / "outputs" / "nyu_rgb_aux"
    flip_dir = repo / "outputs" / "nyu_rgb_flip"
    if not (aux_dir / "manifest_v2_cache.json").is_file() or not (flip_dir / "manifest_v2_cache.json").is_file():
        pytest.skip("Round 2 aux/flip caches not yet generated locally")

    cache = load_v2_cache(orig_dir, aux_dir, flip_dir, data_dir, allowed_split="train", compute_hashes=False, verify_scenes=False)
    assert len(cache.train_ids) == 795
    assert len(cache.test_ids) == 654

    tid = int(cache.train_ids[0])
    assert cache.confidence(tid).shape == (480, 640)
    assert cache.cls_token(tid).shape == (384,)
    assert cache.depth_flip(tid).shape == (480, 640)
    assert cache.intrinsics_flip(tid).shape == (3, 3)

    bad_id = int(cache.test_ids[0])
    with pytest.raises(PermissionError):
        _ = cache.confidence(bad_id)
    with pytest.raises(PermissionError):
        _ = cache.cls_token(bad_id)
    with pytest.raises(PermissionError):
        _ = cache.depth_flip(bad_id)
    with pytest.raises(PermissionError):
        _ = cache.intrinsics_flip(bad_id)
