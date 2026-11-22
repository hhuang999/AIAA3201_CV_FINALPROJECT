import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from .data import nyu_depth, nyu_image, open_nyu, scene_names, splits


def prepare(root, output):
    for name in ("MiddEval3-data-Q.zip", "MiddEval3-GT0-Q.zip"):
        with zipfile.ZipFile(root / name) as archive:
            # Existing source data may be shared through symlinks. Do not rewrite it.
            missing = [item for item in archive.infolist()
                       if not item.is_dir() and not (root / item.filename).is_file()]
            for item in missing:
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Archive target is outside the requested data root")
                archive.extract(item, root)
    indices = splits(root)
    with open_nyu(root) as handle:
        assert handle["images"].shape == (1449, 3, 640, 480)
        assert handle["depths"].shape == (1449, 640, 480)
        if len(indices["train"]) != 795 or len(indices["test"]) != 654:
            raise ValueError("Expected the official 795/654 NYU split")
        if not np.array_equal(np.sort(np.r_[indices["train"], indices["test"]]), np.arange(1449)):
            raise ValueError("Official split must cover each labeled image once")
        names = scene_names(handle)
        train_names = np.unique(names[indices["train"]])
        test_names = np.unique(names[indices["test"]])
        if np.intersect1d(train_names, test_names).size:
            raise ValueError("Train and test scenes overlap")
        index = indices["train"][0]
        rgb, target = nyu_image(handle, index), nyu_depth(handle, index)
        assert rgb.shape == (480, 640, 3) and target.shape == (480, 640)
    reader = cv2.VideoCapture(str(root / "davis_rollercoaster.mp4"))
    if not reader.isOpened():
        raise ValueError("Video source cannot be decoded")
    report = {"train_images": len(indices["train"]), "test_images": len(indices["test"]),
              "train_scenes": len(train_names), "test_scenes": len(test_names),
              "scene_overlap": len(np.intersect1d(train_names, test_names)),
              "stereo_scenes": sorted(p.name for p in (root / "MiddEval3/trainingQ").iterdir()),
              "video_frames": int(reader.get(cv2.CAP_PROP_FRAME_COUNT)),
              "video_fps": reader.get(cv2.CAP_PROP_FPS),
              "video_width": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "video_height": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    reader.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/part2_depth_video/data_summary.json"))
    args = parser.parse_args()
    prepare(args.data_root, args.output)
