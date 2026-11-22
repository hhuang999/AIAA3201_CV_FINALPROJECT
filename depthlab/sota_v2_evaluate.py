"""Gate 4 strictly one-shot held-out 654-image test evaluation and Round 1 case audit for GeoCalib-3D v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import nyu_depth, open_nyu, scene_names
from .frozen import apply_frozen, frozen_predictions, load_frozen
from .metrics import CROP, DEPTH_BINS, METRIC_NAMES, clip_depth, evaluate_image, metrics, valid_depth
from .sota_calibration import (
    _FULL_R2_T,
    _FULL_V_TILDE_T,
    PRIMARY_SOTA_METHOD,
    apply_sota_model,
    extract_sota_features,
    load_frozen_sota,
)
from .sota_evaluate import (
    _git_head,
    build_generalization_gap,
    compute_change_counts,
    paired_scene_bootstrap,
    select_representative_cases,
)
from .sota_io import file_sha256, write_csv_lf, write_json_lf
from .sota_v2_cache import load_v2_cache
from .sota_v2_calibration import (
    _std_z,
    apply_sota_v2_model,
    assemble_semantic_matrix,
    assemble_semantic_vector,
    count_v2_parameters,
    extract_v2_geo_features,
    load_frozen_sota_v2,
)


ROUND1_AUDIT_IDS = (839, 832, 326, 1150, 1151, 1097)


def _decompose_v2_shift_and_spatial(
    model: dict[str, Any],
    feats: dict[str, float],
    sem_vec: np.ndarray | None,
) -> dict[str, float]:
    """Extract exact scalar shift and 2D spatial bias terms for diagnostic interpretability."""
    if model.get("kind") != "geocalib_3d_v2":
        return {"focal": 0.0, "geo": 0.0, "sem": 0.0, "total": 0.0, "c_v": 0.0, "c_r2": 0.0}
    fs = model["focal_spline"]
    hf = float(np.interp(feats[fs["focal_key"]], fs["x_knots"], fs["y_knots"]))
    fg = model["focal_gate"]
    if fg["enabled"]:
        zg = _std_z(feats, fg)
        hf *= float(1.0 / (1.0 + np.exp(-(float(fg["bias"]) + float(zg @ np.asarray(fg["weights"], dtype=np.float64))))))
    sr = model["scale_residual"]
    z_sc = _std_z(feats, sr)
    geo_s = 0.40 * float(np.tanh(float(z_sc @ np.asarray(sr["weights"], dtype=np.float64)) / 0.40))
    sb = model["semantic_branch"]
    sem_s = 0.0
    c_v = 0.0
    c_r2 = 0.0
    if sb["raw_dims"] > 0 and sem_vec is not None:
        xc = np.asarray(sem_vec, dtype=np.float64) - np.asarray(sb["mean"], dtype=np.float64)
        cap = float(sb["scale_cap"])
        sem_s = cap * float(np.tanh(float(xc @ np.asarray(sb["w_scale"], dtype=np.float64)) / cap))
        sp2d = sb["spatial2d"]
        if sp2d["enabled"]:
            c_v = 0.14 * float(np.tanh(float(sp2d["a_v0"]) + float(xc @ np.asarray(sp2d["w_v"], dtype=np.float64))))
            c_r2 = 0.14 * float(np.tanh(float(sp2d["a_r0"]) + float(xc @ np.asarray(sp2d["w_r"], dtype=np.float64))))
    total = 0.50 * float(np.tanh((hf + geo_s + sem_s) / 0.50))
    return {"focal": hf, "geo": geo_s, "sem": sem_s, "total": total, "c_v": c_v, "c_r2": c_r2}


def _apply_v2_gpu_exact(
    p1_np: np.ndarray,
    k1_np: np.ndarray,
    model: dict[str, Any],
    dev: torch.device,
    flip_depth_np: np.ndarray | None = None,
    flip_intrinsics_np: np.ndarray | None = None,
    conf_orig_np: np.ndarray | None = None,
    conf_flip_np: np.ndarray | None = None,
    conf_sum_np: np.ndarray | None = None,
    cls_orig_np: np.ndarray | None = None,
    cls_flip_np: np.ndarray | None = None,
    pat_m_np: np.ndarray | None = None,
    pat_s_np: np.ndarray | None = None,
    df_np: np.ndarray | None = None,
) -> torch.Tensor:
    """Exact CUDA float64 forward matching pure NumPy within <= 1e-13 m."""
    feats = extract_v2_geo_features(
        p1_np, k1_np, conf_summary=conf_sum_np, flip_depth=flip_depth_np,
        flip_intrinsics=flip_intrinsics_np, conf_orig=conf_orig_np, conf_flip=conf_flip_np,
        cls_orig=cls_orig_np, cls_flip=cls_flip_np,
    )
    sb = model["semantic_branch"]
    sem_vec = (
        assemble_semantic_vector(sb["sem_mode"], cls_orig_np, pat_m_np, pat_s_np, df_np, cls_flip_np)
        if sb["raw_dims"] > 0
        else None
    )
    dec = _decompose_v2_shift_and_spatial(model, feats, sem_vec)

    dsm = model["depth_slope_mod"]
    z_sl = _std_z(feats, dsm)
    slope_lin = float(z_sl @ np.asarray(dsm["weights"], dtype=np.float64))
    if sem_vec is not None and sb["w_slope"]:
        xc = sem_vec - np.asarray(sb["mean"], dtype=np.float64)
        slope_lin += float(xc @ np.asarray(sb["w_slope"], dtype=np.float64))
    delta_b = 0.20 * float(np.tanh(slope_lin))

    p1_g = torch.from_numpy(np.ascontiguousarray(p1_np)).to(dev)
    ff = model["flip_fusion"]
    if ff["mode"] == "learnable":
        assert flip_depth_np is not None
        p2_g = torch.from_numpy(np.ascontiguousarray(flip_depth_np)).to(dev)
        wf = float(ff["w_flip"])
        pin_g = torch.exp(wf * torch.log(p1_g) + (1.0 - wf) * torch.log(p2_g))
    else:
        pin_g = p1_g

    lp_g = torch.log(pin_g)
    db = model["depth_branch"]
    xk = np.asarray(db["x_knots"], dtype=np.float64)
    yk = np.asarray(db["y_knots"], dtype=np.float64)
    slopes = np.diff(yk) / np.diff(xk)
    dx = np.diff(xk)
    if db["param"] == "inv_spline":
        invp_g = torch.reciprocal(pin_g)
        dspl = float(yk[0]) + float(slopes[0]) * torch.clamp(invp_g - float(xk[0]), max=float(dx[0]))
        for ki in range(1, len(slopes) - 1):
            dspl.add_(float(slopes[ki]) * torch.clamp(invp_g - float(xk[ki]), min=0.0, max=float(dx[ki])))
        dspl.add_(float(slopes[-1]) * torch.clamp(invp_g - float(xk[-2]), min=0.0))
        out_g = -torch.log(dspl.clamp_min_(0.04))
    else:
        out_g = float(yk[0]) + float(slopes[0]) * torch.clamp(lp_g - float(xk[0]), max=float(dx[0]))
        for ki in range(1, len(slopes) - 1):
            out_g.add_(float(slopes[ki]) * torch.clamp(lp_g - float(xk[ki]), min=0.0, max=float(dx[ki])))
        out_g.add_(float(slopes[-1]) * torch.clamp(lp_g - float(xk[-2]), min=0.0))

    out_g.add_(dec["total"])
    if delta_b != 0.0:
        out_g.add_(lp_g - float(dsm["log_depth_center"]), alpha=delta_b)
    if dec["c_v"] != 0.0 or dec["c_r2"] != 0.0:
        out_g.add_(_FULL_V_TILDE_T.to(dev), alpha=dec["c_v"])
        out_g.add_(_FULL_R2_T.to(dev) - float(sb["spatial2d"]["mean_r2_crop"]), alpha=dec["c_r2"])
    return torch.exp(out_g.clamp_(float(np.log(0.001)), float(np.log(10.0)))).clamp_(0.001, 10.0)


def _read_backbone_timing_receipt(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        record = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid runtime JSON") from exc
    if not isinstance(record, dict):
        raise ValueError(f"{path}: expected a runtime JSON object")

    seconds = record.get("mean_seconds_per_image")
    error = f"{path}: mean_seconds_per_image must be a positive finite duration"
    if type(seconds) not in (int, float):
        raise ValueError(error)
    try:
        mean_ms = float(seconds) * 1000.0
    except OverflowError as exc:
        raise ValueError(error) from exc
    if not math.isfinite(mean_ms) or mean_ms <= 0.0:
        raise ValueError(error)

    device = record.get("device")
    gpu_name = record.get("gpu_name")
    try:
        device_type = torch.device(device).type if isinstance(device, str) else None
    except RuntimeError as exc:
        raise ValueError(f"{path}: invalid backbone device {device!r}") from exc
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"{path}: expected a CPU or CUDA backbone device")
    if device_type == "cuda" and (not isinstance(gpu_name, str) or not gpu_name.strip()):
        raise ValueError(f"{path}: CUDA backbone device requires a gpu_name")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "device": device,
        "device_type": device_type,
        "gpu_name": gpu_name if device_type == "cuda" else None,
        "mean_ms_per_image": mean_ms,
    }


def _estimated_latency(backbone_ms: float | None, post_ms: float) -> float | None:
    return backbone_ms + post_ms if backbone_ms is not None else None


def benchmark_v2_uncontended_runtime(
    cache,
    test_ids: np.ndarray,
    sp_model: dict[str, Any],
    mp_model: dict[str, Any],
    v1_model: dict[str, Any],
    b3_frozen: dict[str, Any],
    aux_runtime_path: Path,
    n_images: int = 200,
    *,
    flip_runtime_path: Path | None = None,
) -> dict[str, Any]:
    """Measure cached-input post-processing and optionally estimate component sums."""
    if n_images <= 0 or not len(test_ids):
        raise ValueError("Timing requires a positive n_images and at least one image")
    aux_timing = _read_backbone_timing_receipt(aux_runtime_path)
    flip_timing = _read_backbone_timing_receipt(flip_runtime_path)
    bb_1pass_ms = aux_timing["mean_ms_per_image"] if aux_timing is not None else None
    bb_2pass_ms = None
    if aux_timing is not None and flip_timing is not None:
        if (aux_timing["device"], aux_timing["gpu_name"]) != (flip_timing["device"], flip_timing["gpu_name"]):
            raise ValueError(f"{aux_runtime_path} and {flip_runtime_path} must record the same backbone device")
        bb_2pass_ms = aux_timing["mean_ms_per_image"] + flip_timing["mean_ms_per_image"]
        if not math.isfinite(bb_2pass_ms):
            raise ValueError(f"{aux_runtime_path} and {flip_runtime_path} must have a finite combined duration")

    bench_ids = test_ids[: min(n_images, len(test_ids))]
    warmup_count = min(10, len(bench_ids))
    p1_l = [cache[int(i)].astype(np.float64) for i in bench_ids]
    p2_l = [cache.depth_flip(int(i)).astype(np.float64) for i in bench_ids]
    c1_l = [np.maximum(cache.confidence(int(i)).astype(np.float64), 1e-4) for i in bench_ids]
    c2_l = [np.maximum(cache.confidence_flip(int(i)).astype(np.float64), 1e-4) for i in bench_ids]
    k1_l = [cache.intrinsics(int(i)).astype(np.float64) for i in bench_ids]
    k2_l = [cache.intrinsics_flip(int(i)).astype(np.float64) for i in bench_ids]
    cs_l = cache.confidence_summary_for(bench_ids)
    cls1_l = cache.cls_tokens_for(bench_ids)
    cls2_l = cache.cls_tokens_flip_for(bench_ids)
    pm_l = cache.patch_mean_for(bench_ids)
    ps_l = cache.patch_std_for(bench_ids)
    df_l = cache.depth_feat_stats_for(bench_ids)

    sm_sp = sp_model["semantic_branch"]["sem_mode"]
    sm_mp = mp_model["semantic_branch"]["sem_mode"]

    # Parity check over first 20 test images
    max_diff_sp_cpu = 0.0
    max_diff_mp_cpu = 0.0
    for j in range(min(20, len(bench_ids))):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        s_t = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
        s_n = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
        max_diff_sp_cpu = max(max_diff_sp_cpu, float(np.max(np.abs(s_t - s_n))))

        m_t = apply_sota_v2_model(
            p1_l[j], k1_l[j], mp_model, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )
        m_n = apply_sota_v2_model(
            p1_l[j], k1_l[j], mp_model, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=False,
        )
        max_diff_mp_cpu = max(max_diff_mp_cpu, float(np.max(np.abs(m_t - m_n))))

    for j in range(warmup_count):
        _ = apply_frozen(p1_l[j], b3_frozen)
        _ = apply_sota_model(p1_l[j], k1_l[j], v1_model, use_torch_fast=True)
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        _ = apply_sota_v2_model(
            p1_l[j], k1_l[j], mp_model, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )

    t0 = perf_counter()
    for p in p1_l:
        _ = apply_frozen(p, b3_frozen)
    b3_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for p, k in zip(p1_l, k1_l):
        _ = apply_sota_model(p, k, v1_model, use_torch_fast=True)
    v1_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=True)
    v2_sp_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
        _ = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
    v2_sp_np_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    t0 = perf_counter()
    for j in range(len(bench_ids)):
        xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
        _ = apply_sota_v2_model(
            p1_l[j], k1_l[j], mp_model, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
            conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
            cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=True,
        )
    v2_mp_cpu_ms = (perf_counter() - t0) * 1000.0 / len(bench_ids)

    res: dict[str, Any] = {
        "benchmark_images": len(bench_ids),
        "uncontended_threads": torch.get_num_threads(),
        "postprocessing_timing_kind": "measured_cached_inputs",
        "end_to_end_timing_kind": "estimated_sum_of_separately_measured_components",
        "backbone_timing_sources": {"aux": aux_timing, "flip": flip_timing},
        "backbone_device": aux_timing["device"] if aux_timing is not None else None,
        "backbone_single_pass_ms_per_image": bb_1pass_ms,
        "backbone_two_pass_flip_ms_per_image": bb_2pass_ms,
        "b3_residual_v1_post_ms": float(b3_ms),
        "v1_geocalib3d_cpu_post_ms": float(v1_cpu_ms),
        "v2_single_pass_cpu_post_ms": float(v2_sp_cpu_ms),
        "v2_single_pass_numpy_post_ms": float(v2_sp_np_ms),
        "v2_flip_tta_cpu_post_ms": float(v2_mp_cpu_ms),
        "end_to_end_v1_ms": _estimated_latency(bb_1pass_ms, v1_cpu_ms),
        "end_to_end_v2_single_pass_cpu_post_ms": _estimated_latency(bb_1pass_ms, v2_sp_cpu_ms),
        "end_to_end_v2_flip_tta_cpu_post_ms": _estimated_latency(bb_2pass_ms, v2_mp_cpu_ms),
        "max_abs_diff_sp_cpu_vs_numpy": float(max_diff_sp_cpu),
        "max_abs_diff_mp_cpu_vs_numpy": float(max_diff_mp_cpu),
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        max_diff_sp_gpu = 0.0
        max_diff_mp_gpu = 0.0
        for j in range(warmup_count):
            xs = assemble_semantic_vector(sm_sp, cls1_l[j], pm_l[j], ps_l[j], df_l[j])
            g_sp = _apply_v2_gpu_exact(
                p1_l[j], k1_l[j], sp_model, dev, conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            ).cpu().numpy()
            n_sp = apply_sota_v2_model(p1_l[j], k1_l[j], sp_model, conf_summary=cs_l[j], cached_sem=xs, use_torch_fast=False)
            max_diff_sp_gpu = max(max_diff_sp_gpu, float(np.max(np.abs(g_sp - n_sp))))

            xm = assemble_semantic_vector(sm_mp, cls1_l[j], pm_l[j], ps_l[j], df_l[j], cls2_l[j])
            g_mp = _apply_v2_gpu_exact(
                p1_l[j], k1_l[j], mp_model, dev, flip_depth_np=p2_l[j], flip_intrinsics_np=k2_l[j],
                conf_orig_np=c1_l[j], conf_flip_np=c2_l[j], conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], cls_flip_np=cls2_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            ).cpu().numpy()
            n_mp = apply_sota_v2_model(
                p1_l[j], k1_l[j], mp_model, flip_depth=p2_l[j], flip_intrinsics=k2_l[j],
                conf_orig=c1_l[j], conf_flip=c2_l[j], conf_summary=cs_l[j],
                cls_orig=cls1_l[j], cls_flip=cls2_l[j], cached_sem=xm, use_torch_fast=False,
            )
            max_diff_mp_gpu = max(max_diff_mp_gpu, float(np.max(np.abs(g_mp - n_mp))))

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        t_g0 = perf_counter()
        for j in range(len(bench_ids)):
            _apply_v2_gpu_exact(
                p1_l[j], k1_l[j], sp_model, dev, conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            )
        torch.cuda.synchronize()
        sp_gpu_ms = (perf_counter() - t_g0) * 1000.0 / len(bench_ids)
        sp_peak_allocated = torch.cuda.max_memory_allocated(dev)
        sp_peak_reserved = torch.cuda.max_memory_reserved(dev)

        torch.cuda.reset_peak_memory_stats(dev)
        t_g1 = perf_counter()
        for j in range(len(bench_ids)):
            _apply_v2_gpu_exact(
                p1_l[j], k1_l[j], mp_model, dev, flip_depth_np=p2_l[j], flip_intrinsics_np=k2_l[j],
                conf_orig_np=c1_l[j], conf_flip_np=c2_l[j], conf_sum_np=cs_l[j],
                cls_orig_np=cls1_l[j], cls_flip_np=cls2_l[j], pat_m_np=pm_l[j], pat_s_np=ps_l[j], df_np=df_l[j],
            )
        torch.cuda.synchronize()
        mp_gpu_ms = (perf_counter() - t_g1) * 1000.0 / len(bench_ids)
        mp_peak_allocated = torch.cuda.max_memory_allocated(dev)
        mp_peak_reserved = torch.cuda.max_memory_reserved(dev)
        gpu_name = str(torch.cuda.get_device_name(dev))
        same_gpu_model = (
            aux_timing is not None
            and aux_timing["device_type"] == "cuda"
            and aux_timing["gpu_name"] == gpu_name
        )
        res["postprocessing_cuda_peak_scope"] = "process allocator peaks during cached-input post-processing"
        res["v2_single_pass_gpu_post_peak_allocated_bytes"] = sp_peak_allocated
        res["v2_single_pass_gpu_post_peak_reserved_bytes"] = sp_peak_reserved
        res["v2_flip_tta_gpu_post_peak_allocated_bytes"] = mp_peak_allocated
        res["v2_flip_tta_gpu_post_peak_reserved_bytes"] = mp_peak_reserved
        res["v2_single_pass_gpu_post_ms"] = float(sp_gpu_ms)
        res["v2_flip_tta_gpu_post_ms"] = float(mp_gpu_ms)
        res["end_to_end_v2_single_pass_gpu_ms"] = _estimated_latency(bb_1pass_ms if same_gpu_model else None, sp_gpu_ms)
        res["end_to_end_v2_flip_tta_gpu_ms"] = _estimated_latency(bb_2pass_ms if same_gpu_model else None, mp_gpu_ms)
        res["max_abs_diff_sp_gpu_vs_numpy"] = float(max_diff_sp_gpu)
        res["max_abs_diff_mp_gpu_vs_numpy"] = float(max_diff_mp_gpu)
        res["gpu_name"] = gpu_name
    return res


def evaluate_heldout_v2_test(
    data_root: Path,
    orig_preds: Path,
    aux_dir: Path,
    flip_dir: Path,
    frozen_v2_path: Path,
    b3_frozen_path: Path,
    v1_frozen_path: Path,
    v1_test_summary_path: Path,
    cv_dir: Path,
    output_dir: Path,
    expected_v2_sha256: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute strictly one-shot 654-test evaluation of frozen GeoCalib-3D v2 Dual-Track suite."""
    torch.set_num_threads(4)
    output_dir = Path(output_dir)
    summary_file = output_dir / "test_summary.csv"
    if summary_file.exists() and not force:
        raise FileExistsError(
            f"Refusing to re-run one-shot v2 test evaluation because {summary_file} already exists"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    v2_frozen = load_frozen_sota_v2(frozen_v2_path, expected_sha256=expected_v2_sha256)
    v2_sha = file_sha256(frozen_v2_path)
    b3 = load_frozen(b3_frozen_path, expected_sha256=v2_frozen["b3_frozen_model_sha256"])
    v1 = load_frozen_sota(v1_frozen_path, expected_sha256=v2_frozen["v1_frozen_sota_sha256"])

    cache = load_v2_cache(orig_preds, aux_dir, flip_dir, data_root, allowed_split="test", compute_hashes=True, verify_scenes=True)
    for k, expected_d in v2_frozen["cache_sha256"].items():
        if cache.hashes.get(k) != expected_d:
            raise ValueError(f"Cache SHA-256 mismatch for {k}")

    test_ids = cache.test_ids
    win_sp = v2_frozen["primary_single_pass_method"]
    win_mp = v2_frozen["primary_multipass_method"]
    methods = tuple(v2_frozen["locked_eval_order"])
    models_v2 = v2_frozen["models"]
    v2_eval_names = [m for m in methods if m not in ("raw", "b3_residual_v1", "v1_geocalib3d")]

    # Preload test-only inputs
    k1_all = cache.intrinsics_for(test_ids).astype(np.float64)
    k2_all = cache.intrinsics_flip_for(test_ids).astype(np.float64)
    cs_all = cache.confidence_summary_for(test_ids)
    cls1_all = cache.cls_tokens_for(test_ids)
    cls2_all = cache.cls_tokens_flip_for(test_ids)
    pm_all = cache.patch_mean_for(test_ids)
    ps_all = cache.patch_std_for(test_ids)
    df_all = cache.depth_feat_stats_for(test_ids)

    sem_mats = {
        sm: assemble_semantic_matrix(sm, cls1_all, pm_all, ps_all, df_all, cls2_all)
        for sm in ("none", "all_sp", "all_mp")
    }

    all_f_tilde = 0.5 * (k1_all[:, 0, 0] + k1_all[:, 1, 1]) / 480.0
    f_q25 = float(np.quantile(all_f_tilde, 0.25))
    f_q75 = float(np.quantile(all_f_tilde, 0.75))

    row_bins = np.array_split(np.arange(426), 12)
    col_bins = np.array_split(np.arange(560), 16)
    row_starts = np.array([int(b[0]) for b in row_bins], dtype=np.int64)
    col_starts = np.array([int(b[0]) for b in col_bins], dtype=np.int64)
    row_edges = [45 + int(b[0]) for b in row_bins] + [471]
    col_edges = [41 + int(b[0]) for b in col_bins] + [601]

    spatial_methods = ("raw", "b3_residual_v1", "v1_geocalib3d", win_sp, win_mp)
    spatial_subsets = ("all", "low_f_tilde", "high_f_tilde", "left_tail_lt055")
    sp_log_sum = {sub: {m: np.zeros((12, 16), dtype=np.float64) for m in spatial_methods} for sub in spatial_subsets}
    sp_cnt_sum = {sub: np.zeros((12, 16), dtype=np.float64) for sub in spatial_subsets}

    def _reduce_12x16(arr2d: np.ndarray) -> np.ndarray:
        return np.add.reduceat(np.add.reduceat(arr2d, row_starts, axis=0), col_starts, axis=1)

    records: list[dict[str, Any]] = []
    binned: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    audit_set = set(ROUND1_AUDIT_IDS)

    with open_nyu(data_root) as handle:
        scenes = scene_names(handle)
        for pos, idx in enumerate(test_ids):
            idx_int = int(idx)
            sc = str(scenes[idx_int])
            p1 = cache[idx_int].astype(np.float64)
            p2 = cache.depth_flip(idx_int).astype(np.float64)
            c1 = np.maximum(cache.confidence(idx_int).astype(np.float64), 1e-4)
            c2 = np.maximum(cache.confidence_flip(idx_int).astype(np.float64), 1e-4)
            k1 = k1_all[pos]
            k2 = k2_all[pos]
            gt_full = nyu_depth(handle, idx_int)
            gt_crop = gt_full[CROP]
            valid = valid_depth(gt_crop)

            f_outs = frozen_predictions(p1, b3)
            v1_feats = extract_sota_features(p1, k1, None, need_rgb=False)
            v2_feats = extract_v2_geo_features(
                p1, k1, cached_v1_feat=v1_feats, conf_summary=cs_all[pos],
                flip_depth=p2, flip_intrinsics=k2, conf_orig=c1, conf_flip=c2,
                cls_orig=cls1_all[pos], cls_flip=cls2_all[pos],
            )
            preds: dict[str, np.ndarray] = {
                "raw": f_outs["raw"],
                "b3_residual_v1": f_outs["residual_scale"],
                "v1_geocalib3d": apply_sota_model(p1, k1, v1["primary_model"], cached_features=v1_feats, use_torch_fast=True),
            }
            for nm in v2_eval_names:
                m_obj = models_v2[nm]
                sm = m_obj["semantic_branch"]["sem_mode"] if m_obj.get("kind") == "geocalib_3d_v2" else "none"
                x_row = sem_mats[sm][pos] if (sm in sem_mats and sem_mats[sm].shape[1]) else None
                preds[nm] = apply_sota_v2_model(
                    p1, k1, m_obj, flip_depth=p2, flip_intrinsics=k2,
                    conf_orig=c1, conf_flip=c2, conf_summary=cs_all[pos],
                    cls_orig=cls1_all[pos], cls_flip=cls2_all[pos],
                    cached_features=v2_feats, cached_sem=x_row, use_torch_fast=True,
                )

            img_scores: dict[str, tuple[float, float, float]] = {}
            for m_name in methods:
                d_pred = preds[m_name]
                ar, rmse, d1 = evaluate_image(d_pred, gt_full)
                img_scores[m_name] = (float(ar), float(rmse), float(d1))
                records.append({"index": idx_int, "scene": sc, "method": m_name, "abs_rel": float(ar), "rmse": float(rmse), "delta1": float(d1)})
                d_crop = d_pred[CROP]
                for lower, upper in zip(DEPTH_BINS, DEPTH_BINS[1:]):
                    mask = valid & (gt_crop >= lower) & (gt_crop < upper)
                    if mask.any():
                        b_ar, b_rmse, b_d1 = metrics(d_crop, gt_crop, mask)
                        binned.append({
                            "index": idx_int, "scene": sc, "method": m_name,
                            "lower_m": float(lower), "upper_m": float(upper),
                            "pixels": int(mask.sum()), "abs_rel": float(b_ar), "rmse": float(b_rmse), "delta1": float(b_d1),
                        })

            dec_sp = _decompose_v2_shift_and_spatial(models_v2[win_sp], v2_feats, sem_mats[models_v2[win_sp]["semantic_branch"]["sem_mode"]][pos])
            dec_mp = _decompose_v2_shift_and_spatial(models_v2[win_mp], v2_feats, sem_mats[models_v2[win_mp]["semantic_branch"]["sem_mode"]][pos])

            diag_rows.append({
                "index": idx_int,
                "scene": sc,
                "f_tilde": float(v2_feats["f_tilde"]),
                "f_flip_tilde": float(v2_feats["f2_tilde"]),
                "f_abs_diff": float(v2_feats["f_abs_diff"]),
                "fx": float(k1[0, 0]),
                "fy": float(k1[1, 1]),
                "log_span_y": float(v2_feats["log_span_y"]),
                "frac_far_4m": float(v2_feats["frac_far_4m"]),
                "median_log_err_raw": float(np.median(np.log(preds["raw"][CROP][valid] / gt_crop[valid]))),
                "median_log_err_b3": float(np.median(np.log(preds["b3_residual_v1"][CROP][valid] / gt_crop[valid]))),
                "median_log_err_v1": float(np.median(np.log(preds["v1_geocalib3d"][CROP][valid] / gt_crop[valid]))),
                "median_log_err_v2_sp": float(np.median(np.log(preds[win_sp][CROP][valid] / gt_crop[valid]))),
                "median_log_err_v2_mp": float(np.median(np.log(preds[win_mp][CROP][valid] / gt_crop[valid]))),
                "abs_rel_v1": img_scores["v1_geocalib3d"][0],
                "abs_rel_v2_sp": img_scores[win_sp][0],
                "abs_rel_v2_mp": img_scores[win_mp][0],
                "shift_v2_sp_total": dec_sp["total"],
                "shift_v2_sp_focal": dec_sp["focal"],
                "shift_v2_sp_geo": dec_sp["geo"],
                "shift_v2_sp_sem": dec_sp["sem"],
                "sp2d_v_v2_sp": dec_sp["c_v"],
                "sp2d_r2_v2_sp": dec_sp["c_r2"],
                "shift_v2_mp_total": dec_mp["total"],
            })

            if idx_int in audit_set:
                for m_lbl in ("raw", "b3_residual_v1", "v1_geocalib3d", win_sp, win_mp):
                    ar, rm, d1 = img_scores[m_lbl]
                    audit_rows.append({
                        "index": idx_int,
                        "scene": sc,
                        "group": "round1_failure" if idx_int in (839, 832, 326) else "round1_high_gain",
                        "method": m_lbl,
                        "f_tilde": float(v2_feats["f_tilde"]),
                        "f_flip_tilde": float(v2_feats["f2_tilde"]),
                        "log_span_y": float(v2_feats["log_span_y"]),
                        "frac_far_4m": float(v2_feats["frac_far_4m"]),
                        "abs_rel": ar,
                        "rmse": rm,
                        "delta1": d1,
                        "delta_vs_v1_abs_rel": img_scores["v1_geocalib3d"][0] - ar,
                        "delta_vs_v1_delta1": d1 - img_scores["v1_geocalib3d"][2],
                        "shift_v2_sp_focal": dec_sp["focal"],
                        "shift_v2_sp_geo": dec_sp["geo"],
                        "shift_v2_sp_sem": dec_sp["sem"],
                        "shift_v2_sp_total": dec_sp["total"],
                    })

            active_subs = ["all"]
            if v2_feats["f_tilde"] <= f_q25:
                active_subs.append("low_f_tilde")
            elif v2_feats["f_tilde"] >= f_q75:
                active_subs.append("high_f_tilde")
            if v2_feats["f_tilde"] < 0.55:
                active_subs.append("left_tail_lt055")
            cnt_grid = _reduce_12x16(valid.astype(np.float64))
            for sub in active_subs:
                sp_cnt_sum[sub] += cnt_grid
            gt_safe = np.where(valid, gt_crop, 1.0)
            for m_name in spatial_methods:
                pc = preds[m_name][CROP]
                lg_grid = _reduce_12x16(np.where(valid, np.log(pc / gt_safe), 0.0))
                for sub in active_subs:
                    sp_log_sum[sub][m_name] += lg_grid

            if (pos + 1) % 150 == 0 or pos + 1 == len(test_ids):
                print(f"Gate 4 v2 test eval {pos + 1}/{len(test_ids)}", flush=True)

    per_image_df = pd.DataFrame(records)
    diag_df = pd.DataFrame(diag_rows)
    bins_raw_df = pd.DataFrame(binned)
    audit_df = pd.DataFrame(audit_rows)

    # Assert exact < 1e-12 reproduction of Round 1 test_summary.csv
    r1_ref = pd.read_csv(v1_test_summary_path).set_index("method")
    obs_m = per_image_df.groupby("method")[list(METRIC_NAMES)].mean()
    max_r1_diff = 0.0
    for obs_k, r1_k in (("raw", "raw"), ("b3_residual_v1", "b3_residual_v1"), ("v1_geocalib3d", PRIMARY_SOTA_METHOD)):
        d = float(np.max(np.abs(obs_m.loc[obs_k].to_numpy(dtype=np.float64) - r1_ref.loc[r1_k, list(METRIC_NAMES)].to_numpy(dtype=np.float64))))
        max_r1_diff = max(max_r1_diff, d)
    if max_r1_diff > 1e-12:
        raise RuntimeError(f"Round 1 baseline/v1 exact parity violated: max_abs_diff={max_r1_diff}")

    # Build complexity & track metadata
    meta_map: dict[str, tuple[str, int, int, float]] = {
        "raw": ("single_pass", 1, 0, 0.0),
        "b3_residual_v1": ("single_pass", 1, 15, 15.0),
        "v1_geocalib3d": ("single_pass", 1, 25, 25.0),
    }
    oof_df = pd.read_csv(Path(cv_dir) / "oof_ablation_summary.csv")
    oof_t = oof_df.set_index("method")
    for nm in v2_eval_names:
        rp, dof = count_v2_parameters(models_v2[nm])
        tr = str(oof_t.loc[nm, "track"])
        bp = int(oof_t.loc[nm, "backbone_passes"])
        meta_map[nm] = (tr, bp, rp, dof)

    sum_df = per_image_df.groupby("method", sort=False)[list(METRIC_NAMES)].mean().reindex(methods).reset_index()
    b3_arr = sum_df.loc[sum_df["method"] == "b3_residual_v1", list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]
    v1_arr = sum_df.loc[sum_df["method"] == "v1_geocalib3d", list(METRIC_NAMES)].to_numpy(dtype=np.float64)[0]

    sum_df["track"] = [meta_map[m][0] for m in sum_df["method"]]
    sum_df["backbone_passes"] = [meta_map[m][1] for m in sum_df["method"]]
    sum_df["raw_params"] = [meta_map[m][2] for m in sum_df["method"]]
    sum_df["effective_dof"] = [meta_map[m][3] for m in sum_df["method"]]
    sum_df["gain_vs_b3_abs_rel"] = b3_arr[0] - sum_df["abs_rel"]
    sum_df["gain_vs_b3_rmse"] = b3_arr[1] - sum_df["rmse"]
    sum_df["gain_vs_b3_delta1"] = sum_df["delta1"] - b3_arr[2]
    sum_df["gain_vs_v1_abs_rel"] = v1_arr[0] - sum_df["abs_rel"]
    sum_df["gain_vs_v1_rmse"] = v1_arr[1] - sum_df["rmse"]
    sum_df["gain_vs_v1_delta1"] = sum_df["delta1"] - v1_arr[2]
    sum_df = sum_df[
        [
            "method", "track", "backbone_passes", "raw_params", "effective_dof",
            "abs_rel", "rmse", "delta1",
            "gain_vs_b3_abs_rel", "gain_vs_b3_rmse", "gain_vs_b3_delta1",
            "gain_vs_v1_abs_rel", "gain_vs_v1_rmse", "gain_vs_v1_delta1",
        ]
    ]

    # Paired bootstrap & win counts: vs B3, vs v1, and Track B vs Track A
    pairs: list[tuple[str, str]] = []
    for ref_m in ("b3_residual_v1", "v1_geocalib3d"):
        for cand_m in methods:
            if cand_m not in ("raw", ref_m) and not (ref_m == "v1_geocalib3d" and cand_m == "b3_residual_v1"):
                pairs.append((ref_m, cand_m))
    pairs.append((win_sp, win_mp))

    boot_df = paired_scene_bootstrap(per_image_df, pairs)
    change_df = compute_change_counts(per_image_df, pairs)

    # Representative cases (v2 Track A & B vs v1_geocalib3d)
    c_sp = select_representative_cases(per_image_df, diag_df, reference="v1_geocalib3d", candidate=win_sp)
    c_sp.insert(0, "comparison", f"{win_sp}_vs_v1")
    c_mp = select_representative_cases(per_image_df, diag_df, reference="v1_geocalib3d", candidate=win_mp)
    c_mp.insert(0, "comparison", f"{win_mp}_vs_v1")
    cases_df = pd.concat([c_sp, c_mp], ignore_index=True)

    bins_df = (
        bins_raw_df.groupby(["method", "lower_m", "upper_m"], sort=False)
        .agg(abs_rel=("abs_rel", "mean"), rmse=("rmse", "mean"), delta1=("delta1", "mean"), images=("index", "count"), pixels=("pixels", "sum"))
        .reset_index()
    )
    m_rank = {m: i for i, m in enumerate(methods)}
    bins_df["_r"] = bins_df["method"].map(m_rank)
    bins_df = bins_df.sort_values(["_r", "lower_m"]).drop(columns=["_r"]).reset_index(drop=True)

    # Build generalization gap table explicitly vs both B3 and v1
    gap_df = build_generalization_gap(oof_df, sum_df)
    oof_v1_arr = oof_t.loc["v1_geocalib3d", list(METRIC_NAMES)].to_numpy(dtype=np.float64)
    direction_v = np.array([-1.0, -1.0, 1.0])
    oof_gv1 = []
    test_gv1 = []
    for m_nm in gap_df["method"].tolist():
        o_v = oof_t.loc[m_nm, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        t_v = sum_df.set_index("method").loc[m_nm, list(METRIC_NAMES)].to_numpy(dtype=np.float64)
        oof_gv1.append((o_v - oof_v1_arr) * direction_v)
        test_gv1.append((t_v - v1_arr) * direction_v)
    og1 = np.asarray(oof_gv1)
    tg1 = np.asarray(test_gv1)
    gap_df["oof_gain_vs_v1_abs_rel"] = og1[:, 0]
    gap_df["test_gain_vs_v1_abs_rel"] = tg1[:, 0]
    gap_df["test_minus_oof_gain_v1_abs_rel"] = tg1[:, 0] - og1[:, 0]
    gap_df["oof_gain_vs_v1_rmse"] = og1[:, 1]
    gap_df["test_gain_vs_v1_rmse"] = tg1[:, 1]
    gap_df["test_minus_oof_gain_v1_rmse"] = tg1[:, 1] - og1[:, 1]
    gap_df["oof_gain_vs_v1_delta1"] = og1[:, 2]
    gap_df["test_gain_vs_v1_delta1"] = tg1[:, 2]
    gap_df["test_minus_oof_gain_v1_delta1"] = tg1[:, 2] - og1[:, 2]

    spatial_payload: dict[str, Any] = {
        "row_edges": row_edges,
        "col_edges": col_edges,
        "f_tilde_q25": f_q25,
        "f_tilde_q75": f_q75,
        "maps": {},
    }
    for sub in spatial_subsets:
        cnt = np.maximum(sp_cnt_sum[sub], 1.0)
        spatial_payload["maps"][sub] = {}
        for m_name in spatial_methods:
            log_map = sp_log_sum[sub][m_name] / cnt
            spatial_payload["maps"][sub][m_name] = {
                "mean_log_residual_12x16": log_map.tolist(),
                "rms_spatial_log_bias": float(np.sqrt(np.mean(log_map ** 2))),
                "max_abs_spatial_log_bias": float(np.max(np.abs(log_map))),
            }

    runtime = benchmark_v2_uncontended_runtime(
        cache, test_ids, models_v2[win_sp], models_v2[win_mp], v1["primary_model"], b3,
        aux_dir / "runtime_aux.json", n_images=200, flip_runtime_path=flip_dir / "runtime_flip.json",
    )

    write_csv_lf(output_dir / "test_summary.csv", sum_df)
    write_csv_lf(output_dir / "test_per_image.csv", per_image_df)
    write_csv_lf(output_dir / "test_paired_bootstrap.csv", boot_df)
    write_csv_lf(output_dir / "test_depth_bins.csv", bins_df)
    write_csv_lf(output_dir / "test_change_counts.csv", change_df)
    write_csv_lf(output_dir / "test_round1_case_audit.csv", audit_df)
    write_csv_lf(output_dir / "test_representative_cases.csv", cases_df)
    write_csv_lf(output_dir / "generalization_gap.csv", gap_df)
    write_csv_lf(output_dir / "test_mechanism_diagnostics.csv", diag_df)
    write_json_lf(output_dir / "test_spatial_residuals.json", spatial_payload)
    write_json_lf(output_dir / "runtime_overhead.json", runtime)

    repo_root = Path(__file__).resolve().parents[1]
    sp_row = sum_df[sum_df["method"] == win_sp].iloc[0]
    mp_row = sum_df[sum_df["method"] == win_mp].iloc[0]
    receipt = {
        "schema": "GeoCalib_3D_v2_test_receipt",
        "complete": True,
        "one_shot_test_evaluation": True,
        "git_head_at_eval": _git_head(repo_root),
        "frozen_sota_v2_path": str(frozen_v2_path),
        "frozen_sota_v2_sha256": v2_sha,
        "round1_exact_parity_max_abs_diff": float(max_r1_diff),
        "test_images": int(len(test_ids)),
        "test_scenes": int(per_image_df["scene"].nunique()),
        "track_a_winner": win_sp,
        "track_a_test_metrics": {k: float(sp_row[k]) for k in ("abs_rel", "rmse", "delta1", "gain_vs_v1_abs_rel", "gain_vs_v1_rmse", "gain_vs_v1_delta1", "gain_vs_b3_abs_rel")},
        "track_b_winner": win_mp,
        "track_b_test_metrics": {k: float(mp_row[k]) for k in ("abs_rel", "rmse", "delta1", "gain_vs_v1_abs_rel", "gain_vs_v1_rmse", "gain_vs_v1_delta1", "gain_vs_b3_abs_rel")},
        "runtime_summary": runtime,
    }
    write_json_lf(output_dir / "test_evaluation_receipt.json", receipt)
    print(
        f"Gate 4 v2 Test Complete:\n"
        f"  Track A ({win_sp}): AbsRel={sp_row['abs_rel']:.6f}, RMSE={sp_row['rmse']:.6f}, delta1={sp_row['delta1']:.6f} "
        f"(vs v1: {sp_row['gain_vs_v1_abs_rel']:+.6f} / {sp_row['gain_vs_v1_rmse']:+.6f} / {sp_row['gain_vs_v1_delta1']:+.6f})\n"
        f"  Track B ({win_mp}): AbsRel={mp_row['abs_rel']:.6f}, RMSE={mp_row['rmse']:.6f}, delta1={mp_row['delta1']:.6f} "
        f"(vs v1: {mp_row['gain_vs_v1_abs_rel']:+.6f} / {mp_row['gain_vs_v1_rmse']:+.6f} / {mp_row['gain_vs_v1_delta1']:+.6f})\n"
        f"  Round 1 Exact Parity Max Diff: {max_r1_diff:.2e}",
        flush=True,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--orig-preds", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--flip-dir", type=Path, required=True)
    parser.add_argument("--frozen-v2", type=Path, required=True)
    parser.add_argument("--b3-frozen", type=Path, required=True)
    parser.add_argument("--v1-frozen", type=Path, required=True)
    parser.add_argument("--v1-test-summary", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-v2-sha256", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    evaluate_heldout_v2_test(
        data_root=args.data_root,
        orig_preds=args.orig_preds,
        aux_dir=args.aux_dir,
        flip_dir=args.flip_dir,
        frozen_v2_path=args.frozen_v2,
        b3_frozen_path=args.b3_frozen,
        v1_frozen_path=args.v1_frozen,
        v1_test_summary_path=args.v1_test_summary,
        cv_dir=args.cv_dir,
        output_dir=args.output,
        expected_v2_sha256=args.expected_v2_sha256,
        force=args.force,
    )


if __name__ == "__main__":
    main()
