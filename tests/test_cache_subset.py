import numpy as np
import pytest
from scipy.io import savemat

from depthlab.cache import load_predictions, sha256, write_complete_manifest
from depthlab.cache_subset import export_subset


def test_subset_preserves_original_id_order_and_bound_predictions(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "nyu_depth_v2_labeled.mat").write_bytes(b"bound fixture identity")
    savemat(data / "splits.mat", {"trainNdxs": [4, 2], "testNdxs": [1, 3]})
    source = tmp_path / "all"
    source.mkdir()
    ids = np.array([3, 0, 1, 2])
    depths = np.stack([np.full((2, 3), i + 1, np.float32) for i in ids])
    np.save(source / "indices.npy", ids)
    np.save(source / "depth.npy", depths)
    signature = {"indices": ids.tolist(), "shape": list(depths.shape), "dtype": "float32",
        "known_intrinsics": False, "data": {"nyu_mat_sha256": sha256(data / "nyu_depth_v2_labeled.mat"),
        "splits_sha256": sha256(data / "splits.mat")}, "source": "synthetic", "weights": "synthetic",
        "backend": "synthetic", "preprocessing": "synthetic", "seed": 0}
    write_complete_manifest(source, signature)
    output = tmp_path / "train"
    export_subset(source, data, "train", output)
    loaded = load_predictions(output, required_ids=[1, 3], require_complete=True)
    np.testing.assert_array_equal(loaded.indices, [3, 1])
    np.testing.assert_array_equal(loaded[1], np.full((2, 3), 2.))
    with pytest.raises(FileExistsError):
        export_subset(source, data, "train", output)
