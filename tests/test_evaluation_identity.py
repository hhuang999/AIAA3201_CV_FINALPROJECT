import json
from argparse import Namespace

import pandas as pd
import pytest

from depthlab.cache import sha256
from depthlab.evaluate import evaluate
from test_indexed_consumers import make_subset


def prepared_run(tmp_path):
    data, predictions, path = make_subset(tmp_path)
    parameters = json.loads(path.read_text())
    parameters.update(schema="depthlab_calibration_v2", complete=True, test_read=False,
        training_ids=[1, 3], training_images=2,
        dataset_sha256=sha256(data / "nyu_depth_v2_labeled.mat"),
        split_sha256=sha256(data / "splits.mat"))
    path.write_text(json.dumps(parameters))
    args = Namespace(data_root=data, predictions=predictions, parameters=path,
        output=tmp_path / "evaluation", split="test", limit=None, reference=None,
        reference_depth=None, allow_legacy_cache=True, allow_legacy_parameters=False,
        expected_model_sha256=None)
    return args, parameters


def test_expected_digest_is_checked_for_completed_new_parameters(tmp_path):
    args, _ = prepared_run(tmp_path)
    args.expected_model_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate(args)
    assert not args.output.exists()


@pytest.mark.parametrize("change", ["test_ids", "missing_ids", "test_read", "missing_test_read",
                                    "missing_dataset", "missing_split", "wrong_count"])
def test_new_parameter_provenance_is_enforced_before_scoring(tmp_path, change):
    args, parameters = prepared_run(tmp_path)
    if change == "test_ids": parameters["training_ids"] = [0, 2]
    elif change == "missing_ids": del parameters["training_ids"]
    elif change == "test_read": parameters["test_read"] = True
    elif change == "missing_test_read": del parameters["test_read"]
    elif change == "missing_dataset": del parameters["dataset_sha256"]
    elif change == "missing_split": del parameters["split_sha256"]
    elif change == "wrong_count": parameters["training_images"] = 3
    args.parameters.write_text(json.dumps(parameters))
    with pytest.raises(ValueError, match="training|train-only|binding"):
        evaluate(args)
    assert not args.output.exists()


def test_valid_new_parameters_and_expected_digest_keep_same_scores(tmp_path):
    args, _ = prepared_run(tmp_path)
    args.expected_model_sha256 = sha256(args.parameters)
    result = evaluate(args)
    assert result["complete"] and result["parameters_sha256"] == args.expected_model_sha256
    frame = pd.read_csv(args.output / "test_per_image.csv")
    assert frame["abs_rel"].max() < 1e-14 and frame["rmse"].max() < 1e-14
    assert (frame["delta1"] == 1).all()
