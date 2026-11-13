"""Verify a delivered repository's source and result files against its manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest_path, root):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty file manifest")
    seen = set()
    for entry in entries:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            raise ValueError("Manifest contains an external or duplicate file path")
        seen.add(relative)
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed file: {relative}")
        if "bytes" in entry and path.stat().st_size != entry["bytes"]:
            raise ValueError(f"File size changed: {relative}")
    return {"complete": True, "verified_files": len(seen),
            "manifest_sha256": file_sha256(manifest_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/restoration_20261113/manifest.json"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify_manifest(args.manifest, args.root), indent=2))
