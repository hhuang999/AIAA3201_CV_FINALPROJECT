from argparse import Namespace
import h5py
import numpy as np
import pytest
from scipy.io import savemat


def make_subset_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    savemat(data / "splits.mat", {"trainNdxs": [[2], [4]], "testNdxs": [[1], [3]]})
    with h5py.File(data / "nyu_depth_v2_labeled.mat", "w") as handle:
        handle.create_dataset("depths", data=np.ones((4, 640, 480), dtype=np.float32))
        refs = handle.create_dataset("scenes", (1, 4), dtype=h5py.ref_dtype)
        for i in range(4):
            ds = handle.create_dataset(f"scene_{i}", data=np.array([ord(c) for c in f"room_{i}"], dtype=np.uint16))
            refs[0, i] = ds.ref
    return data


def make_camera_cache(folder, known, binding):
    from depthlab.cache import write_complete_manifest
    from depthlab.infer_nyu import RGB_INTRINSICS

    folder.mkdir()
    ids = np.array([1, 3])
    depth = np.stack([np.full((480, 640), v, np.float32) for v in ([1., 1.2] if known else [2., 3.])])
    np.save(folder / "indices.npy", ids)
    np.save(folder / "depth.npy", depth)
    intrinsics = np.stack([RGB_INTRINSICS.copy(), RGB_INTRINSICS.copy()])
    intrinsics[1, 0, 0] *= 2
    intrinsics[1, 1, 1] *= 2
    np.save(folder / "intrinsics.npy", intrinsics)
    write_complete_manifest(folder, {
        "indices": ids.tolist(), "shape": list(depth.shape),
        "dtype": "float32", "known_intrinsics": known, "data": binding,
        "source": {"synthetic": True}, "weights": {"synthetic": True},
        "backend": "synthetic", "preprocessing": "synthetic", "seed": 0,
    })


def test_diagnose_rejects_different_bound_data(tmp_path):
    from depthlab.diagnose import diagnose

    data = make_subset_data(tmp_path)
    raw, known = tmp_path / "raw", tmp_path / "known"
    make_camera_cache(raw, False, {"nyu_mat_sha256": "wrong", "splits_sha256": "wrong"})
    make_camera_cache(known, True, {"nyu_mat_sha256": "also-wrong", "splits_sha256": "also-wrong"})
    with pytest.raises(ValueError, match="data|split"):
        diagnose(Namespace(data_root=data, predictions=raw, known_predictions=known, output=tmp_path / "diag"))


def test_diagnose_preserves_existing_results(tmp_path):
    from depthlab.cache import sha256
    from depthlab.diagnose import diagnose

    data = make_subset_data(tmp_path)
    raw, known = tmp_path / "raw", tmp_path / "known"
    binding = {"nyu_mat_sha256": sha256(data / "nyu_depth_v2_labeled.mat"), "splits_sha256": sha256(data / "splits.mat")}
    make_camera_cache(raw, False, binding)
    make_camera_cache(known, True, binding)
    out = tmp_path / "diag"
    out.mkdir()
    marker = out / "camera_diagnostic.json"
    marker.write_text("previous evidence must survive")
    with pytest.raises(FileExistsError):
        diagnose(Namespace(data_root=data, predictions=raw, known_predictions=known, output=out))
    assert marker.read_text() == "previous evidence must survive"
