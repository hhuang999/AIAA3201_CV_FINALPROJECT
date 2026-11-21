"""Pinned UniDepthV2 auxiliary/flip-TTA cache builders and split-guarded Round 2 I/O."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
import types
from typing import Any

import numpy as np
import torch

from .data import nyu_image, open_nyu, splits
from .inference_backend import use_pytorch_attention
from .metrics import CROP
from .sota_io import VerifiedCache, file_sha256, load_nyu_rgb_cache, write_json_lf


PINNED_WEIGHTS: dict[str, str] = {
    "config.json": "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe",
    "model.safetensors": "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648",
}
UPSTREAM_REPO = "lpiccinelli/unidepth-v2-vits14"
UPSTREAM_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"

CONF_SUMMARY_NAMES: tuple[str, ...] = (
    "conf_mean",
    "conf_std",
    "conf_q05",
    "conf_q10",
    "conf_q25",
    "conf_q50",
    "conf_q75",
    "conf_q90",
    "conf_log_mean",
    "conf_log_std",
    "conf_top_mean",
    "conf_mid_mean",
    "conf_bot_mean",
    "conf_top_bot_diff",
    "conf_center_mean",
    "conf_periph_mean",
    "conf_center_periph_diff",
)

# Precomputed masks inside Eigen crop (426 x 560) for zone confidence statistics
_CROP_H, _CROP_W = 426, 560
_U_CROP, _V_CROP = np.meshgrid(
    (np.arange(41, 601, dtype=np.float64) - 319.5) / 320.0,
    (np.arange(45, 471, dtype=np.float64) - 239.5) / 240.0,
    indexing="xy",
)
_R2_CROP = _U_CROP ** 2 + _V_CROP ** 2
_ROW_THIRD = _CROP_H // 3
_TOP_SLICE = slice(0, _ROW_THIRD)
_MID_SLICE = slice(_ROW_THIRD, 2 * _ROW_THIRD)
_BOT_SLICE = slice(2 * _ROW_THIRD, _CROP_H)
_CENTER_MASK = _R2_CROP < 0.25
_PERIPH_MASK = _R2_CROP >= 0.50


def ensure_unidepth_weights(
    weights_dir: Path | str,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Ensure pinned UniDepthV2 ViT-S weights exist locally and pass SHA-256 verification."""
    target_dir = Path(weights_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = dict(PINNED_WEIGHTS if expected_hashes is None else expected_hashes)

    resolved_hashes: dict[str, str] = {}
    for fname, exp_sha in expected.items():
        fpath = target_dir / fname
        if not fpath.is_file():
            from huggingface_hub import hf_hub_download

            cached_file = Path(hf_hub_download(repo_id=UPSTREAM_REPO, filename=fname))
            shutil.copy2(cached_file, fpath)
        actual_sha = file_sha256(fpath)
        if actual_sha != exp_sha:
            raise ValueError(
                f"SHA-256 mismatch for {fpath}: expected {exp_sha}, got {actual_sha}"
            )
        resolved_hashes[fname] = actual_sha
    return resolved_hashes


def _install_unidepth_shims() -> None:
    """Install lightweight stdlib/torch shims for unused broken imports in third_party/UniDepth."""
    def _norm(t: torch.Tensor, mean: Any, std: Any, inplace: bool = False) -> torch.Tensor:
        m = torch.as_tensor(mean, dtype=t.dtype, device=t.device).view(-1, 1, 1)
        s = torch.as_tensor(std, dtype=t.dtype, device=t.device).view(-1, 1, 1)
        return t.sub_(m).div_(s) if inplace else (t - m) / s

    for mod_name in (
        "torchvision",
        "torchvision.transforms",
        "torchvision.transforms.functional",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2.functional",
    ):
        existing = sys.modules.get(mod_name)
        if existing is None or not hasattr(existing, "_depthlab_shim"):
            shim = types.ModuleType(mod_name)
            shim.normalize = _norm  # type: ignore[attr-defined]
            shim._depthlab_shim = True  # type: ignore[attr-defined]
            sys.modules[mod_name] = shim
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms"].functional = sys.modules["torchvision.transforms.functional"]  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms"].v2 = sys.modules["torchvision.transforms.v2"]  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms.v2"].functional = sys.modules["torchvision.transforms.v2.functional"]  # type: ignore[attr-defined]

    class _TimmShim(types.ModuleType):
        def __getattr__(self, item: str) -> Any:
            if item.startswith("__"):
                raise AttributeError(item)
            if item == "trunc_normal_":
                return torch.nn.init.trunc_normal_
            return type(item, (torch.nn.Module,), {})

    for mod_name in ("timm", "timm.models", "timm.models.layers", "timm.layers"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _TimmShim(mod_name)
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.ModuleType("wandb")


def _load_local_unidepth(source_dir: Path, weights_dir: Path, device_str: str = "cuda"):
    """Load pinned UniDepthV2 ViT-S cleanly using local PyTorch attention branches."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

    _install_unidepth_shims()
    src_resolved = str(Path(source_dir).resolve())
    if src_resolved not in sys.path:
        sys.path.insert(0, src_resolved)
    from unidepth.models import UniDepthV2

    backend = use_pytorch_attention("unidepth")
    device = torch.device(device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu")
    model = UniDepthV2.from_pretrained(str(Path(weights_dir).resolve())).to(device).eval()
    model.interpolation_mode = "bilinear"
    return model, device, backend


def compute_confidence_summary(conf_480x640: np.ndarray) -> np.ndarray:
    """Extract 17 summary statistics from UniDepthV2 pixel confidence over the Eigen crop."""
    c = np.asarray(conf_480x640[CROP], dtype=np.float64)
    c_pos = np.maximum(c, 1e-6)
    lc = np.log(c_pos)
    q = np.quantile(c, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    top_m = float(c[_TOP_SLICE].mean())
    mid_m = float(c[_MID_SLICE].mean())
    bot_m = float(c[_BOT_SLICE].mean())
    ctr_m = float(c[_CENTER_MASK].mean())
    per_m = float(c[_PERIPH_MASK].mean())
    return np.array(
        [
            float(c.mean()),
            float(c.std()),
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
            float(q[4]),
            float(q[5]),
            float(lc.mean()),
            float(lc.std()),
            top_m,
            mid_m,
            bot_m,
            top_m - bot_m,
            ctr_m,
            per_m,
            ctr_m - per_m,
        ],
        dtype=np.float32,
    )


def build_aux_and_flip_caches(
    data_root: Path,
    source_dir: Path,
    weights_dir: Path,
    orig_pred_dir: Path,
    aux_dir: Path,
    flip_dir: Path,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Generate fixed-weight original-RGB aux features and Horizontal-Flip TTA caches for 0..1448.

    Strictly RGB-only (never touches GT depths or test-split metrics/labels). A train-split-only
    parity check compares local GPU depth/intrinsics against orig_pred_dir on splits['train'] only.
    """
    data_root = Path(data_root)
    orig_pred_dir = Path(orig_pred_dir)
    aux_dir = Path(aux_dir)
    flip_dir = Path(flip_dir)

    mat_sha = file_sha256(data_root / "nyu_depth_v2_labeled.mat")
    splits_sha = file_sha256(data_root / "splits.mat")
    orig_depth_sha = file_sha256(orig_pred_dir / "depth.npy")
    orig_intr_sha = file_sha256(orig_pred_dir / "intrinsics.npy")
    orig_idx_sha = file_sha256(orig_pred_dir / "indices.npy")

    aux_manifest_path = aux_dir / "manifest_v2_cache.json"
    flip_manifest_path = flip_dir / "manifest_v2_cache.json"
    if aux_manifest_path.is_file() and flip_manifest_path.is_file():
        import json

        aux_m = json.loads(aux_manifest_path.read_text(encoding="utf-8"))
        flip_m = json.loads(flip_manifest_path.read_text(encoding="utf-8"))
        if (
            aux_m.get("complete") is True
            and flip_m.get("complete") is True
            and aux_m.get("nyu_mat_sha256") == mat_sha
            and flip_m.get("nyu_mat_sha256") == mat_sha
            and aux_m.get("orig_depth_sha256") == orig_depth_sha
        ):
            print("Round 2 aux & flip caches already complete and verified; skipping GPU rebuild.", flush=True)
            return {"aux_manifest": aux_m, "flip_manifest": flip_m}

    weight_hashes = ensure_unidepth_weights(weights_dir)
    model, device, backend = _load_local_unidepth(source_dir, weights_dir, device_str=device_str)

    aux_dir.mkdir(parents=True, exist_ok=True)
    flip_dir.mkdir(parents=True, exist_ok=True)

    train_ids = set(int(i) for i in splits(data_root)["train"].tolist())
    orig_depths = np.load(orig_pred_dir / "depth.npy", mmap_mode="r", allow_pickle=False)
    orig_intr = np.load(orig_pred_dir / "intrinsics.npy", mmap_mode="r", allow_pickle=False)

    n_total = 1449
    indices = np.arange(n_total, dtype=np.int64)
    np.save(aux_dir / "indices.npy", indices)
    np.save(flip_dir / "indices.npy", indices)

    captured_enc: dict[str, Any] = {}

    def _enc_hook(_mod: Any, _inp: Any, output: tuple[list[torch.Tensor], list[torch.Tensor]]) -> None:
        patch_list, cls_list = output
        captured_enc["patch_last"] = patch_list[-1].detach()
        captured_enc["cls_last"] = cls_list[-1].detach()

    hook_handle = model.pixel_encoder.register_forward_hook(_enc_hook)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        with open_nyu(data_root) as handle, torch.inference_mode():
            # Probe shapes cleanly using index 0 (single RGB image, zero GT read)
            probe_rgb = torch.from_numpy(np.ascontiguousarray(nyu_image(handle, 0))).permute(2, 0, 1)
            for _ in range(3):
                probe_out = model.infer(probe_rgb)
            _sync()

            d_emb = int(captured_enc["cls_last"].shape[-1])
            c_df = int(probe_out["depth_features"].shape[1])

            cls_tokens = np.empty((n_total, d_emb), dtype=np.float32)
            patch_mean = np.empty((n_total, d_emb), dtype=np.float32)
            patch_std = np.empty((n_total, d_emb), dtype=np.float32)
            depth_feat_stats = np.empty((n_total, 2 * c_df), dtype=np.float32)
            conf_summary = np.empty((n_total, len(CONF_SUMMARY_NAMES)), dtype=np.float32)

            conf_f16 = np.lib.format.open_memmap(
                aux_dir / "confidence_f16.npy", mode="w+", dtype=np.float16, shape=(n_total, 480, 640)
            )
            depth_flip = np.lib.format.open_memmap(
                flip_dir / "depth_flip.npy", mode="w+", dtype=np.float32, shape=(n_total, 480, 640)
            )
            intr_flip = np.lib.format.open_memmap(
                flip_dir / "intrinsics_flip.npy", mode="w+", dtype=np.float32, shape=(n_total, 3, 3)
            )
            conf_flip_f16 = np.lib.format.open_memmap(
                flip_dir / "confidence_flip_f16.npy", mode="w+", dtype=np.float16, shape=(n_total, 480, 640)
            )
            cls_tokens_flip = np.empty((n_total, d_emb), dtype=np.float32)

            t_aux: list[float] = []
            t_flip: list[float] = []
            train_depth_med_diffs: list[float] = []
            train_depth_mae_diffs: list[float] = []
            train_fx_rel_diffs: list[float] = []

            for idx in range(n_total):
                rgb_np = nyu_image(handle, idx)  # uint8 (480, 640, 3)

                # Pass 1: Original RGB (fixed-weight forward)
                rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_np)).permute(2, 0, 1)
                _sync()
                t0 = perf_counter()
                out1 = model.infer(rgb_t)
                _sync()
                t_aux.append(perf_counter() - t0)

                cls1 = captured_enc["cls_last"][0, 0].float().cpu().numpy()
                pat1 = captured_enc["patch_last"][0].float()  # (H_p, W_p, D)
                pm1 = pat1.mean(dim=(0, 1)).cpu().numpy()
                ps1 = pat1.std(dim=(0, 1), unbiased=False).cpu().numpy()
                df1 = out1["depth_features"][0].float()  # (C_df, H_df, W_df)
                df_stats = torch.cat(
                    [df1.mean(dim=(1, 2)), df1.std(dim=(1, 2), unbiased=False)], dim=0
                ).cpu().numpy()

                conf1_np = out1["confidence"][0, 0].float().cpu().numpy()
                cls_tokens[idx] = cls1
                patch_mean[idx] = pm1
                patch_std[idx] = ps1
                depth_feat_stats[idx] = df_stats
                conf_f16[idx] = np.clip(conf1_np, -65000.0, 65000.0).astype(np.float16)
                conf_summary[idx] = compute_confidence_summary(conf1_np)

                # Strictly train-only parity verification (never touch test indices)
                if idx in train_ids:
                    d1_np = out1["depth"][0, 0].float().cpu().numpy()
                    k1_np = out1["intrinsics"][0].float().cpu().numpy()
                    diff_d = np.abs(d1_np - np.asarray(orig_depths[idx], dtype=np.float32))
                    train_depth_med_diffs.append(float(np.median(diff_d)))
                    train_depth_mae_diffs.append(float(np.mean(diff_d)))
                    fx_ref = float(orig_intr[idx, 0, 0])
                    train_fx_rel_diffs.append(abs(float(k1_np[0, 0]) - fx_ref) / max(fx_ref, 1e-6))

                # Pass 2: Horizontal-Flipped RGB (un-flip spatial outputs cleanly)
                rgb_flip_np = np.ascontiguousarray(rgb_np[:, ::-1, :])
                rgb_flip_t = torch.from_numpy(rgb_flip_np).permute(2, 0, 1)
                _sync()
                t1 = perf_counter()
                out2 = model.infer(rgb_flip_t)
                _sync()
                t_flip.append(perf_counter() - t1)

                d2_np = np.ascontiguousarray(out2["depth"][0, 0].float().cpu().numpy()[:, ::-1])
                c2_np = np.ascontiguousarray(out2["confidence"][0, 0].float().cpu().numpy()[:, ::-1])
                k2_np = out2["intrinsics"][0].float().cpu().numpy()
                k2_np[0, 2] = 639.0 - k2_np[0, 2]  # mirror cx back to original image frame
                cls2 = captured_enc["cls_last"][0, 0].float().cpu().numpy()

                if not np.isfinite(d2_np).all() or np.any(d2_np <= 0):
                    raise ValueError(f"Nonfinite/nonpositive flip depth at index {idx}")

                depth_flip[idx] = d2_np
                intr_flip[idx] = k2_np
                conf_flip_f16[idx] = np.clip(c2_np, -65000.0, 65000.0).astype(np.float16)
                cls_tokens_flip[idx] = cls2

                if (idx + 1) % 250 == 0 or idx + 1 == n_total:
                    print(f"UniDepthV2 aux+flip cache progress: {idx + 1}/{n_total}", flush=True)

            conf_f16.flush()
            depth_flip.flush()
            intr_flip.flush()
            conf_flip_f16.flush()

            np.save(aux_dir / "cls_tokens.npy", cls_tokens)
            np.save(aux_dir / "patch_mean.npy", patch_mean)
            np.save(aux_dir / "patch_std.npy", patch_std)
            np.save(aux_dir / "depth_feat_stats.npy", depth_feat_stats)
            np.save(aux_dir / "confidence_summary.npy", conf_summary)
            write_json_lf(aux_dir / "confidence_summary_names.json", list(CONF_SUMMARY_NAMES))
            np.save(flip_dir / "cls_tokens_flip.npy", cls_tokens_flip)
    finally:
        hook_handle.remove()

    # Assert tight train-only parity vs existing outputs/nyu_rgb
    med_parity = float(np.median(train_depth_med_diffs))
    if med_parity > 0.02:
        raise RuntimeError(f"Train-only GPU parity check failed: median diff={med_parity:.4f} m > 0.02 m")

    parity_payload = {
        "scope": "splits['train'] only (N=795); zero test indices compared",
        "train_images": len(train_depth_med_diffs),
        "depth_median_of_per_image_median_abs_diff_m": med_parity,
        "depth_mean_of_per_image_mae_m": float(np.mean(train_depth_mae_diffs)),
        "depth_p95_of_per_image_mae_m": float(np.quantile(train_depth_mae_diffs, 0.95)),
        "fx_mean_rel_diff": float(np.mean(train_fx_rel_diffs)),
        "fx_p95_rel_diff": float(np.quantile(train_fx_rel_diffs, 0.95)),
    }
    write_json_lf(aux_dir / "parity_train_only.json", parity_payload)

    rt_aux = {
        "model": "UniDepthV2 ViT-S",
        "source_revision": UPSTREAM_REVISION,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "attention_backend": backend,
        "weights_sha256": weight_hashes,
        "images": n_total,
        "cls_dim": d_emb,
        "depth_feat_stats_dim": 2 * c_df,
        "mean_seconds_per_image": float(np.mean(t_aux[5:])),
        "median_seconds_per_image": float(np.median(t_aux[5:])),
    }
    rt_flip = {
        "model": "UniDepthV2 ViT-S Horizontal Flip",
        "source_revision": UPSTREAM_REVISION,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "images": n_total,
        "mean_seconds_per_image": float(np.mean(t_flip[5:])),
        "median_seconds_per_image": float(np.median(t_flip[5:])),
    }
    write_json_lf(aux_dir / "runtime_aux.json", rt_aux)
    write_json_lf(flip_dir / "runtime_flip.json", rt_flip)

    aux_files = (
        "indices.npy",
        "cls_tokens.npy",
        "patch_mean.npy",
        "patch_std.npy",
        "depth_feat_stats.npy",
        "confidence_f16.npy",
        "confidence_summary.npy",
    )
    flip_files = (
        "indices.npy",
        "depth_flip.npy",
        "intrinsics_flip.npy",
        "confidence_flip_f16.npy",
        "cls_tokens_flip.npy",
    )
    aux_manifest = {
        "complete": True,
        "nyu_mat_sha256": mat_sha,
        "splits_sha256": splits_sha,
        "orig_depth_sha256": orig_depth_sha,
        "orig_intrinsics_sha256": orig_intr_sha,
        "orig_indices_sha256": orig_idx_sha,
        "weights_sha256": weight_hashes,
        "files_sha256": {f: file_sha256(aux_dir / f) for f in aux_files},
    }
    flip_manifest = {
        "complete": True,
        "nyu_mat_sha256": mat_sha,
        "splits_sha256": splits_sha,
        "orig_depth_sha256": orig_depth_sha,
        "orig_intrinsics_sha256": orig_intr_sha,
        "orig_indices_sha256": orig_idx_sha,
        "weights_sha256": weight_hashes,
        "files_sha256": {f: file_sha256(flip_dir / f) for f in flip_files},
    }
    write_json_lf(aux_dir / "manifest_v2_cache.json", aux_manifest)
    write_json_lf(flip_dir / "manifest_v2_cache.json", flip_manifest)
    print(
        f"Aux+Flip caches built cleanly! Train depth parity median diff={med_parity:.6f}m, "
        f"aux={rt_aux['mean_seconds_per_image']*1000:.2f}ms/img, "
        f"flip={rt_flip['mean_seconds_per_image']*1000:.2f}ms/img",
        flush=True,
    )
    return {"aux_manifest": aux_manifest, "flip_manifest": flip_manifest, "parity": parity_payload}


class VerifiedV2Cache:
    """Split-guarded reader combining orig predictions, aux embeddings/confidence, and flip TTA."""

    def __init__(
        self,
        orig_pred_dir: Path | str,
        aux_dir: Path | str,
        flip_dir: Path | str,
        data_root: Path | str,
        allowed_split: str = "train",
        compute_hashes: bool = False,
        verify_scenes: bool = True,
    ) -> None:
        self.base: VerifiedCache = load_nyu_rgb_cache(
            orig_pred_dir,
            data_root,
            allowed_split=allowed_split,
            compute_hashes=compute_hashes,
            verify_scenes=verify_scenes,
        )
        self.aux_dir = Path(aux_dir)
        self.flip_dir = Path(flip_dir)
        self.allowed_split = self.base.allowed_split
        self.train_ids = self.base.train_ids
        self.test_ids = self.base.test_ids
        self.allowed_ids = self.base.allowed_ids

        for req in (
            self.aux_dir / "cls_tokens.npy",
            self.aux_dir / "patch_mean.npy",
            self.aux_dir / "patch_std.npy",
            self.aux_dir / "depth_feat_stats.npy",
            self.aux_dir / "confidence_f16.npy",
            self.aux_dir / "confidence_summary.npy",
            self.flip_dir / "depth_flip.npy",
            self.flip_dir / "intrinsics_flip.npy",
            self.flip_dir / "confidence_flip_f16.npy",
            self.flip_dir / "cls_tokens_flip.npy",
        ):
            if not req.is_file():
                raise FileNotFoundError(f"Missing Round 2 cache artifact: {req}")

        self._cls = np.load(self.aux_dir / "cls_tokens.npy", mmap_mode="r", allow_pickle=False)
        self._pat_m = np.load(self.aux_dir / "patch_mean.npy", mmap_mode="r", allow_pickle=False)
        self._pat_s = np.load(self.aux_dir / "patch_std.npy", mmap_mode="r", allow_pickle=False)
        self._df_stats = np.load(self.aux_dir / "depth_feat_stats.npy", mmap_mode="r", allow_pickle=False)
        self._conf = np.load(self.aux_dir / "confidence_f16.npy", mmap_mode="r", allow_pickle=False)
        self._conf_sum = np.load(self.aux_dir / "confidence_summary.npy", mmap_mode="r", allow_pickle=False)

        self._d_flip = np.load(self.flip_dir / "depth_flip.npy", mmap_mode="r", allow_pickle=False)
        self._k_flip = np.load(self.flip_dir / "intrinsics_flip.npy", mmap_mode="r", allow_pickle=False)
        self._c_flip = np.load(self.flip_dir / "confidence_flip_f16.npy", mmap_mode="r", allow_pickle=False)
        self._cls_flip = np.load(self.flip_dir / "cls_tokens_flip.npy", mmap_mode="r", allow_pickle=False)

        # Validate finiteness strictly ONLY on allowed_ids
        for idx in self.allowed_ids:
            i = int(idx)
            if not np.isfinite(self._cls[i]).all() or not np.isfinite(self._conf[i]).all():
                raise ValueError(f"Nonfinite aux values at allowed index {i}")
            df_i = self._d_flip[i]
            if not np.isfinite(df_i).all() or not (df_i > 0).all():
                raise ValueError(f"Nonfinite/nonpositive flip depth at allowed index {i}")

        self.hashes: dict[str, str] = dict(self.base.hashes)
        if compute_hashes:
            for tag, p in (
                ("aux_cls_tokens_sha256", self.aux_dir / "cls_tokens.npy"),
                ("aux_patch_mean_sha256", self.aux_dir / "patch_mean.npy"),
                ("aux_patch_std_sha256", self.aux_dir / "patch_std.npy"),
                ("aux_depth_feat_stats_sha256", self.aux_dir / "depth_feat_stats.npy"),
                ("aux_confidence_f16_sha256", self.aux_dir / "confidence_f16.npy"),
                ("aux_confidence_summary_sha256", self.aux_dir / "confidence_summary.npy"),
                ("flip_depth_sha256", self.flip_dir / "depth_flip.npy"),
                ("flip_intrinsics_sha256", self.flip_dir / "intrinsics_flip.npy"),
                ("flip_confidence_f16_sha256", self.flip_dir / "confidence_flip_f16.npy"),
                ("flip_cls_tokens_sha256", self.flip_dir / "cls_tokens_flip.npy"),
            ):
                self.hashes[tag] = file_sha256(p)

    def _guard(self, index: int) -> int:
        return self.base._check_index(index)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.base[index]

    def intrinsics(self, index: int) -> np.ndarray:
        return self.base.intrinsics(index)

    def intrinsics_for(self, indices: np.ndarray) -> np.ndarray:
        return self.base.intrinsics_for(indices)

    def confidence(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._conf[i], dtype=np.float32)

    def confidence_summary(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._conf_sum[i], dtype=np.float64)

    def confidence_summary_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._conf_sum[arr], dtype=np.float64)

    def cls_token(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._cls[i], dtype=np.float64)

    def cls_tokens_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._cls[arr], dtype=np.float64)

    def patch_mean_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._pat_m[arr], dtype=np.float64)

    def patch_std_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._pat_s[arr], dtype=np.float64)

    def depth_feat_stats_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._df_stats[arr], dtype=np.float64)

    def depth_flip(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._d_flip[i], dtype=np.float32)

    def intrinsics_flip(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._k_flip[i], dtype=np.float32)

    def intrinsics_flip_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._k_flip[arr], dtype=np.float64)

    def confidence_flip(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._c_flip[i], dtype=np.float32)

    def cls_token_flip(self, index: int) -> np.ndarray:
        i = self._guard(index)
        return np.asarray(self._cls_flip[i], dtype=np.float64)

    def cls_tokens_flip_for(self, indices: np.ndarray) -> np.ndarray:
        arr = [self._guard(i) for i in indices]
        return np.asarray(self._cls_flip[arr], dtype=np.float64)


def load_v2_cache(
    orig_pred_dir: Path | str,
    aux_dir: Path | str,
    flip_dir: Path | str,
    data_root: Path | str,
    allowed_split: str = "train",
    compute_hashes: bool = False,
    verify_scenes: bool = True,
) -> VerifiedV2Cache:
    """Instantiate split-guarded Round 2 cache reader."""
    return VerifiedV2Cache(
        orig_pred_dir=orig_pred_dir,
        aux_dir=aux_dir,
        flip_dir=flip_dir,
        data_root=data_root,
        allowed_split=allowed_split,
        compute_hashes=compute_hashes,
        verify_scenes=verify_scenes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bp = sub.add_parser("build", help="Download pinned UniDepthV2 ViT-S and build aux + flip-TTA caches")
    bp.add_argument("--data-root", type=Path, required=True)
    bp.add_argument("--source", type=Path, required=True)
    bp.add_argument("--weights", type=Path, required=True)
    bp.add_argument("--orig-preds", type=Path, required=True)
    bp.add_argument("--aux-out", type=Path, required=True)
    bp.add_argument("--flip-out", type=Path, required=True)
    bp.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.command == "build":
        build_aux_and_flip_caches(
            data_root=args.data_root,
            source_dir=args.source,
            weights_dir=args.weights,
            orig_pred_dir=args.orig_preds,
            aux_dir=args.aux_out,
            flip_dir=args.flip_out,
            device_str=args.device,
        )


if __name__ == "__main__":
    main()
