import json
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from depthlab.data import splits
from depthlab.evaluate import _compare_reference


@pytest.mark.parametrize("train,test", [([0],[2]),([1.5],[2]),([1,1],[2]),([1],[1])])
def test_malformed_official_split_is_rejected(tmp_path, train, test):
    savemat(tmp_path / "splits.mat", {"trainNdxs": train, "testNdxs": test})
    with pytest.raises(ValueError):
        splits(tmp_path)


def test_matlab_ids_preserve_order(tmp_path):
    savemat(tmp_path / "splits.mat", {"trainNdxs": [4,2], "testNdxs": [3,1]})
    actual = splits(tmp_path)
    np.testing.assert_array_equal(actual["train"], [3,1])
    np.testing.assert_array_equal(actual["test"], [2,0])


def test_replay_comparison_uses_ids_and_rejects_changed_metrics(tmp_path):
    records = [{"index": i, "method": "raw", "abs_rel": .1, "rmse": .2, "delta1": .9} for i in [7,2]]
    frame = pd.DataFrame(records)
    reference = tmp_path / "per_image.csv"
    frame.iloc[::-1].rename(columns={"index": "dataset_index", "delta1": "d1"}).to_csv(reference, index=False)
    result = _compare_reference(frame, reference, {})
    assert result["max_absolute_metric_error"] == 0
    frame.loc[0,"abs_rel"] += .001
    with pytest.raises(ValueError, match="differ"):
        _compare_reference(frame, reference, {})
