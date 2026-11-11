from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def splits(data_root):
    indices = loadmat(Path(data_root) / "splits.mat")
    return {name: indices[key].ravel().astype(int) - 1
            for name, key in [("train", "trainNdxs"), ("test", "testNdxs")]}


def scene_names(handle):
    return np.array(["".join(chr(int(c)) for c in handle[ref][:].ravel())
                     for ref in handle["scenes"][:].ravel()])


def nyu_image(handle, index):
    return handle["images"][index].transpose(2, 1, 0)


def nyu_depth(handle, index):
    return handle["depths"][index].T


def open_nyu(data_root):
    return h5py.File(Path(data_root) / "nyu_depth_v2_labeled.mat", "r")


def read_pfm(path):
    with open(path, "rb") as stream:
        assert stream.readline().strip() == b"Pf"
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        values = np.fromfile(stream, dtype="<f4" if scale < 0 else ">f4")
    return np.flipud(values.reshape(height, width)).copy()
