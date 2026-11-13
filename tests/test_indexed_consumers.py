from argparse import Namespace
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.io import savemat

from depthlab.evaluate import evaluate


def make_subset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    savemat(data / "splits.mat", {"trainNdxs": [[2], [4]], "testNdxs": [[1], [3]]})
    with h5py.File(data / "nyu_depth_v2_labeled.mat", "w") as handle:
        handle.create_dataset("depths", data=np.ones((4, 640, 480), dtype=np.float32))
        refs = handle.create_dataset("scenes", (1, 4), dtype=h5py.ref_dtype)
        for i in range(4):
            ds = handle.create_dataset(f"scene_{i}", data=np.array([ord(c) for c in f"room_{i}"], dtype=np.uint16))
            refs[0, i] = ds.ref
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    np.save(predictions / "indices.npy", np.array([2, 0]))
    np.save(predictions / "depth.npy", np.ones((2, 480, 640), dtype=np.float32))
    (predictions / "runtime.json").write_text(json.dumps({"known_intrinsics": False}))
    params = tmp_path / "parameters.json"
    params.write_text(json.dumps({"global_scale": 1., "monotone_curve": {"offset": 0., "slopes": [1.] * 5}, "residual": None}))
    return data, predictions, params


def test_evaluate_uses_original_ids_for_compact_shuffled_predictions(tmp_path):
    data, predictions, params = make_subset(tmp_path)
    output = tmp_path / "results"
    evaluate(Namespace(data_root=data, predictions=predictions, parameters=params, output=output,
                       split="test", limit=None, reference=None, reference_depth=None,
                       allow_legacy_cache=True, allow_legacy_parameters=True))
    import pandas as pd
    frame = pd.read_csv(output / "test_per_image.csv")
    assert sorted(frame["index"].unique()) == [0, 2]
    np.testing.assert_allclose(frame[["abs_rel", "rmse"]], 0, atol=1e-14)
    np.testing.assert_allclose(frame.delta1, 1)


def test_nonfinite_reference_depth_is_rejected(tmp_path, monkeypatch):
    import depthlab.evaluate as ev
    import pytest
    data, predictions, params = make_subset(tmp_path)
    reference = tmp_path / "bad_reference.npy"
    np.save(reference, np.full((2, 480, 640), np.nan))
    monkeypatch.setattr(ev, "corrected_predictions", lambda p, params: {
        name: p for name in ["raw", "global_scale", "monotone_curve", "residual_scale"]})
    args = Namespace(data_root=data, predictions=predictions, parameters=params,
                     output=tmp_path / "out", split="test", limit=None, reference=None,
                     reference_depth=reference, allow_legacy_cache=True, allow_legacy_parameters=True)
    with pytest.raises(ValueError, match="[Nn]onfinite|finite"):
        ev.evaluate(args)


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
    write_complete_manifest(folder, {"indices": ids.tolist(), "shape": list(depth.shape),
        "dtype": "float32", "known_intrinsics": known, "data": binding,
        "source": {"synthetic": True}, "weights": {"synthetic": True},
        "backend": "synthetic", "preprocessing": "synthetic", "seed": 0})


def test_diagnose_rejects_different_bound_data(tmp_path):
    import pytest
    from depthlab.diagnose import diagnose
    data, _, _ = make_subset(tmp_path)
    raw, known = tmp_path / "raw", tmp_path / "known"
    make_camera_cache(raw, False, {"nyu_mat_sha256": "wrong", "splits_sha256": "wrong"})
    make_camera_cache(known, True, {"nyu_mat_sha256": "also-wrong", "splits_sha256": "also-wrong"})
    with pytest.raises(ValueError, match="data|split"):
        diagnose(Namespace(data_root=data, predictions=raw, known_predictions=known, output=tmp_path / "diag"))


def test_diagnose_preserves_existing_results(tmp_path):
    import pytest
    from depthlab.diagnose import diagnose
    from depthlab.cache import sha256
    data, _, _ = make_subset(tmp_path)
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


def test_complete_cache_cannot_bypass_data_binding_with_empty_dict(tmp_path):
    import pytest
    from depthlab.cache import write_complete_manifest
    data, predictions, params = make_subset(tmp_path)
    ids = np.load(predictions / "indices.npy")
    depths = np.load(predictions / "depth.npy")
    write_complete_manifest(predictions, {"indices": ids.tolist(), "shape": list(depths.shape),
        "dtype": "float32", "known_intrinsics": False, "data": {}, "source": "synthetic",
        "weights": "synthetic", "backend": "synthetic", "preprocessing": "synthetic", "seed": 0})
    with pytest.raises(ValueError, match="data|split"):
        evaluate(Namespace(data_root=data, predictions=predictions, parameters=params,
            output=tmp_path / "out", split="test", limit=None, reference=None,
            reference_depth=None, allow_legacy_cache=False, allow_legacy_parameters=True))
