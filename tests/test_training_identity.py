"""Training workflows must bind predictions and scene splits before sampling."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from depthlab.cache import sha256
import depthlab.calibration as calibration
import depthlab.calibration_study as study
import depthlab.calibration_analysis as analysis


class SamplingReached(RuntimeError):
    pass


@pytest.fixture
def identity(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "nyu_depth_v2_labeled.mat").write_bytes(b"synthetic dataset identity")
    (root / "splits.mat").write_bytes(b"synthetic split identity")
    hashes = {"nyu_mat_sha256": sha256(root / "nyu_depth_v2_labeled.mat"),
              "splits_sha256": sha256(root / "splits.mat")}
    partition = {"train": np.arange(795), "test": np.arange(795, 1449)}
    scenes = np.array([f"train{i % 249}" for i in range(795)] +
                      [f"test{i}" for i in range(654)])
    predictions = SimpleNamespace(metadata={"signature": {"data": dict(hashes)}})
    class SyntheticHandle:
        def close(self): pass
        def __getitem__(self, key):
            raise SamplingReached("No GT access is authorized by this identity test")
    handle = SyntheticHandle()
    for module in (calibration, study, analysis):
        monkeypatch.setattr(module, "splits", lambda *args: partition)
        monkeypatch.setattr(module, "scene_names", lambda *args: scenes)
    monkeypatch.setattr(calibration, "open_nyu", lambda *args: nullcontext(handle))
    monkeypatch.setattr(study, "open_nyu", lambda *args: handle)
    monkeypatch.setattr(analysis, "open_nyu", lambda *args: nullcontext(handle))
    monkeypatch.setattr("depthlab.cache.load_predictions", lambda *args, **kwargs: predictions)
    monkeypatch.setattr(study, "load_predictions", lambda *args, **kwargs: predictions)
    monkeypatch.setattr(analysis, "load_predictions", lambda *args, **kwargs: predictions)
    def sampling(*args, **kwargs):
        raise SamplingReached("No synthetic fit or GT access is authorized by this test")
    monkeypatch.setattr(calibration, "prepare_training", sampling)
    monkeypatch.setattr(study, "prepare_training", sampling)
    monkeypatch.setattr(analysis, "nyu_depth", sampling)
    frozen = {"train_ids": partition["train"].tolist(),
              "dataset_sha256": hashes["nyu_mat_sha256"], "split_sha256": hashes["splits_sha256"],
              "controls": {"median": {"scale": 1.}}}
    return SimpleNamespace(root=root, partition=partition, scenes=scenes, predictions=predictions,
                           frozen=frozen, output=tmp_path / "output")


def invoke(consumer, identity):
    if consumer == "residual":
        calibration.fit(identity.root, "unused", identity.output)
    elif consumer == "discovery":
        with study.TrainingData(identity.root, "unused"):
            pass
    else:
        analysis.training_bias(identity.root, "unused", identity.frozen, identity.output)


@pytest.mark.parametrize("consumer", ["residual", "discovery", "bias"])
@pytest.mark.parametrize("changed", ["nyu_mat_sha256", "splits_sha256", "missing"])
def test_training_rejects_cache_from_other_data_before_sampling(identity, consumer, changed):
    data = identity.predictions.metadata["signature"]["data"]
    if changed == "missing":
        identity.predictions.metadata["signature"].pop("data")
    else:
        data[changed] = "f" * 64
    with pytest.raises(ValueError, match="[Cc]ache.*data|[Pp]rediction.*data"):
        invoke(consumer, identity)


@pytest.mark.parametrize("consumer", ["residual", "discovery"])
@pytest.mark.parametrize("changed", ["scene_overlap", "wrong_scene_count", "out_of_range", "short_test",
                                     "invalid_train", "duplicate_train", "image_overlap"])
def test_training_rejects_unofficial_partitions_before_sampling(identity, consumer, changed):
    if changed == "scene_overlap":
        identity.scenes[795] = identity.scenes[0]
    elif changed == "wrong_scene_count":
        identity.scenes[:795] = [f"train{i % 248}" for i in range(795)]
    elif changed == "out_of_range":
        identity.partition["test"][-1] = 1449
    elif changed == "short_test":
        identity.partition["test"] = identity.partition["test"][:-1]
    elif changed == "invalid_train":
        identity.partition["train"][0] = -1
    elif changed == "duplicate_train":
        identity.partition["train"][0] = identity.partition["train"][1]
    else:
        identity.partition["test"][0] = identity.partition["train"][0]
    with pytest.raises(ValueError, match="[Ss]cene|identit|795|654|1449"):
        invoke(consumer, identity)


@pytest.mark.parametrize("consumer", ["residual", "discovery", "bias"])
def test_valid_training_identity_reaches_preparation(identity, consumer):
    with pytest.raises(SamplingReached):
        invoke(consumer, identity)
