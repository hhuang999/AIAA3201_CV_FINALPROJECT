import numpy as np
import pytest
from scipy.io import savemat

from depthlab.data import splits


@pytest.mark.parametrize("train,test", [([0], [2]), ([1.5], [2]), ([1, 1], [2]), ([1], [1])])
def test_malformed_official_split_is_rejected(tmp_path, train, test):
    savemat(tmp_path / "splits.mat", {"trainNdxs": train, "testNdxs": test})
    with pytest.raises(ValueError):
        splits(tmp_path)


def test_matlab_ids_preserve_order(tmp_path):
    savemat(tmp_path / "splits.mat", {"trainNdxs": [4, 2], "testNdxs": [3, 1]})
    actual = splits(tmp_path)
    np.testing.assert_array_equal(actual["train"], [3, 1])
    np.testing.assert_array_equal(actual["test"], [2, 0])
