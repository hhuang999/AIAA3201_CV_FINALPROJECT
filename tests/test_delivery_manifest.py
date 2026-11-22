import json
import pytest
from depthlab.verify import file_sha256, verify_manifest


def test_source_and_result_manifest_detects_changes(tmp_path):
    value = tmp_path / "model.json"
    value.write_text("frozen parameters")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": value.name, "sha256": file_sha256(value), "bytes": value.stat().st_size}]}))
    assert verify_manifest(manifest, tmp_path)["verified_files"] == 1
    value.write_text("different parameters")
    with pytest.raises(ValueError, match="changed"):
        verify_manifest(manifest, tmp_path)


def test_manifest_cannot_claim_external_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "../outside", "sha256": "0" * 64}]}))
    with pytest.raises(ValueError, match="external"):
        verify_manifest(manifest, tmp_path)
