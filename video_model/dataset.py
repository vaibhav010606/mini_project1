"""UCSD Ped2 loading: frames, 5-frame clips and frame-level ground truth."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, PED2_DIR  # noqa: E402

SIZE = 128
CLIP = 5  # consecutive frames per sample


def _clip_dirs(split: str):
    base = PED2_DIR / split
    return sorted(p for p in base.iterdir() if p.is_dir() and re.fullmatch(rf"{split}\d{{3}}", p.name))


def _load_dir(d: Path, size: int = SIZE) -> np.ndarray:
    files = sorted(f for f in d.iterdir()
                   if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".bmp") and not f.name.startswith("."))
    frames = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(frames).astype(np.float32) / 255.0


def _labels_from_m(n_frames: list[int]):
    m_file = PED2_DIR / "Test" / "UCSDped2.m"
    if not m_file.exists():
        return None
    text = m_file.read_text(errors="ignore")
    entries = re.findall(r"gt_frame\s*=\s*\[([^\]]*)\]", text)  # one line per test video, in order
    if len(entries) != len(n_frames):
        return None
    labels = []
    for spec, n in zip(entries, n_frames):
        lab = np.zeros(n, dtype=np.int64)
        for a, b in re.findall(r"(\d+)\s*:\s*(\d+)", spec):
            lab[int(a) - 1:int(b)] = 1
        labels.append(lab)
    return labels


def _labels_from_masks(test_dirs, n_frames):
    labels = []
    for d, n in zip(test_dirs, n_frames):
        gt = d.parent / f"{d.name}_gt"
        files = sorted(f for f in gt.iterdir()
                       if f.suffix.lower() in (".bmp", ".png", ".tif") and not f.name.startswith("."))
        lab = np.array([int((cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0).any()) for f in files][:n])
        labels.append(lab)
    return labels


def load_ped2(size: int = SIZE, cache: bool = True):
    """Returns (train_videos, test_videos, test_labels), lists of arrays."""
    cache_file = DATA_DIR / f"ped2_{size}.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        return list(z["train"]), list(z["test"]), list(z["labels"])
    if not PED2_DIR.exists():
        raise FileNotFoundError(f"UCSD Ped2 not found at {PED2_DIR}. Download and extract the dataset first.")
    train = [_load_dir(d, size) for d in _clip_dirs("Train")]
    test_dirs = _clip_dirs("Test")
    test = [_load_dir(d, size) for d in test_dirs]
    n = [len(v) for v in test]
    labels = _labels_from_m(n) or _labels_from_masks(test_dirs, n)
    if cache:
        np.savez_compressed(cache_file, train=np.array(train, dtype=object), test=np.array(test, dtype=object),
                            labels=np.array(labels, dtype=object))
    return train, test, labels


def make_clips(video: np.ndarray, clip: int = CLIP) -> np.ndarray:
    """(N,H,W) -> (N-clip+1, clip, H, W) sliding windows."""
    idx = np.arange(clip)[None, :] + np.arange(len(video) - clip + 1)[:, None]
    return video[idx]


if __name__ == "__main__":
    tr, te, lab = load_ped2()
    print(f"train videos={len(tr)} frames={sum(len(v) for v in tr)}")
    print(f"test videos={len(te)} frames={sum(len(v) for v in te)} anomalous={sum(int(l.sum()) for l in lab)}")
