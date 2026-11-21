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
from depthlab.sota_v2_calibration import (
    GeoCalibV2Spec,
    apply_sota_v2_model,
    assemble_semantic_matrix,
    assemble_semantic_vector,
    build_v2_candidate_specs,
    count_v2_parameters,
    extract_v2_geo_features,
    fit_joint_v2_model,
    load_frozen_sota_v2,
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


def test_v2_parametrizations_torch_numpy_parity_and_pca_dof():
    rng = np.random.default_rng(123)
    n, s_pts = 16, 64
    p1 = rng.uniform(0.6, 5.0, size=(n, 480, 640))
    p2 = p1 * rng.uniform(0.95, 1.05, size=(n, 480, 640))
    sp1 = rng.uniform(0.6, 5.0, size=(n, s_pts))
    sp2 = sp1 * 1.02
    w1 = np.full((n, s_pts), 0.52)
    gt = sp1 * rng.uniform(0.9, 1.1, size=(n, s_pts))
    rr = rng.integers(0, 426, size=(n, s_pts))
    cc = rng.integers(0, 560, size=(n, s_pts))
    k1 = np.tile(np.array([[520.0, 0.0, 322.0], [0.0, 520.0, 245.0], [0.0, 0.0, 1.0]]), (n, 1, 1))
    k2 = k1.copy()
    k2[:, 0, 2] = 639.0 - 322.0

    cls1 = rng.normal(size=(n, 384))
    cls2 = rng.normal(size=(n, 384))
    pm = rng.normal(size=(n, 384))
    ps = rng.normal(size=(n, 384))
    df = rng.normal(size=(n, 512))
    cs = rng.normal(size=(n, 17))
    feats = [
        extract_v2_geo_features(p1[i], k1[i], conf_summary=cs[i], flip_depth=p2[i], flip_intrinsics=k2[i])
        for i in range(n)
    ]

    x_mat = assemble_semantic_matrix("all_sp", cls1, pm, ps, df)
    x_vec0 = assemble_semantic_vector("all_sp", cls1[0], pm[0], ps[0], df[0])
    assert np.max(np.abs(x_mat[0] - x_vec0)) < 1e-12

    spec_map = {s.name: s for s in build_v2_candidate_specs()}
    c1_map = np.full((480, 640), 1.3)
    c2_map = np.full((480, 640), 1.1)

    for nm in (
        "D3_qknots_robust",
        "N1_inv_depth_spline",
        "N3_rational_depth",
        "H1_geosem_sp2d_r005_sp008",
        "H6_geosem_inv_spline",
        "M3_flip_geosem_sp2d_r005",
        "M4_flip_geosem_sp2d_conf_fgate",
        "M6_flip_inv_geosem",
    ):
        sp = spec_map[nm]
        xs = assemble_semantic_matrix(sp.sem_mode, cls1, pm, ps, df, cls2)
        m = fit_joint_v2_model(
            sp,
            sp1,
            gt,
            rr,
            cc,
            feats,
            np.ones(n),
            xs,
            samples_p2=(sp2 if sp.flip_blend_mode != "none" else None),
            samples_w1=(w1 if sp.flip_blend_mode == "conf" else None),
            max_iter=8,
        )
        raw_p, dof = count_v2_parameters(m)
        assert raw_p >= 22 and np.isfinite(dof)
        if sp.sem_mode != "none":
            assert 0.0 < dof < float(raw_p)

        x0 = xs[0] if xs.shape[1] else None
        out_t = apply_sota_v2_model(
            p1[0],
            k1[0],
            m,
            flip_depth=p2[0],
            flip_intrinsics=k2[0],
            conf_orig=c1_map,
            conf_flip=c2_map,
            conf_summary=cs[0],
            cached_sem=x0,
            use_torch_fast=True,
        )
        out_n = apply_sota_v2_model(
            p1[0],
            k1[0],
            m,
            flip_depth=p2[0],
            flip_intrinsics=k2[0],
            conf_orig=c1_map,
            conf_flip=c2_map,
            conf_summary=cs[0],
            cached_sem=x0,
            use_torch_fast=False,
        )
        assert float(np.max(np.abs(out_t - out_n))) < 1e-13

    # Verify PCA projection collapse mathematically matches uncollapsed projection
    pca_spec = spec_map["H3_geosem_sp2d_pca128"]
    m_pca = fit_joint_v2_model(pca_spec, sp1, gt, rr, cc, feats, np.ones(n), x_mat, max_iter=6)
    assert len(m_pca["semantic_branch"]["w_scale"]) == 1664
    raw_pca, dof_pca = count_v2_parameters(m_pca)
    assert raw_pca < 1664 and np.isfinite(dof_pca)


def test_frozen_v2_model_schema_and_lf_endings():
    repo = Path(__file__).resolve().parents[1]
    for p in list((repo / "depthlab").glob("sota_v2_*.py")) + [repo / "tests" / "test_part3b_sota_v2.py"]:
        assert b"\r" not in p.read_bytes(), f"CRLF detected in {p}"

    frozen_path = repo / "results" / "part3b_sota_v2" / "frozen_sota_v2_model.json"
    if frozen_path.is_file():
        payload = load_frozen_sota_v2(frozen_path)
        assert payload["test_read"] is False
        assert payload["primary_single_pass_raw_params"] > 0
        assert 0.0 < payload["primary_single_pass_effective_dof"] < payload["primary_single_pass_raw_params"]
        assert b"\r" not in frozen_path.read_bytes()

