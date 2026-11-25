"""Create sample videos for testing the Upload tab of the web app (or detect.py).

    python samples/make_samples.py

Real (UCSD Ped2 test/train frames turned into .mp4 files, use mode "AI model" or "Auto"):
    ucsd_test004_cart.mp4      a small truck/cart drives through from 3.0 s
    ucsd_test001_bike.mp4      a cyclist appears from 6.0 s
    ucsd_test007.mp4           anomaly (dataset label) from 4.5 s
    ucsd_train001_normal.mp4   normal walking only: should give no alerts

Synthetic (generated top-down crowd, use mode "Crowd motion" or "Auto"):
    synthetic_panic.mp4        calm walking, then at 8 s everyone near one point runs away
    synthetic_calm.mp4         the same crowd walking calmly the whole time
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PED2_DIR  # noqa: E402

OUT = Path(__file__).resolve().parent


def writer(path: Path, fps: float, size):
    # MPEG-4 (mp4v) is always available in OpenCV; H.264 would need an extra OpenH264 DLL
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError("Cannot write mp4 with this OpenCV build")
    return w


def ucsd_to_mp4(split: str, vid: str, name: str):
    files = sorted(f for f in (PED2_DIR / split / vid).iterdir()
                   if f.suffix.lower() == ".tif" and not f.name.startswith("."))
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    vw = writer(OUT / name, 10.0, (w, h))
    for f in files:
        vw.write(cv2.imread(str(f), cv2.IMREAD_COLOR))
    vw.release()
    print(f"  {name}: {len(files)} frames, {w}x{h}, 10 fps")


def synthetic_crowd(name: str, panic: bool, seconds: float = 20.0, fps: int = 25, seed: int = 7):
    """Top-down view of a plaza. Scale: 1 pixel = 5 cm, so walking 1.2 m/s = 1 px per frame."""
    rng = np.random.default_rng(seed)
    W, H, N = 640, 400, 140
    floor = np.full((H, W, 3), 150, np.uint8)
    floor = cv2.add(floor, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))  # paving texture
    for x in range(0, W, 80):
        cv2.line(floor, (x, 0), (x, H), (135, 135, 135), 1)
    for y in range(0, H, 80):
        cv2.line(floor, (0, y), (W, y), (135, 135, 135), 1)

    pos = rng.uniform([20, 20], [W - 20, H - 20], (N, 2))
    target = rng.uniform([20, 20], [W - 20, H - 20], (N, 2))
    walk = rng.uniform(0.6, 1.3, N)          # px per frame  (~0.7-1.6 m/s)
    run = rng.uniform(3.0, 5.0, N)           # px per frame  (~3.7-6 m/s)
    colors = rng.integers(30, 220, (N, 3))
    centre = np.array([W * 0.55, H * 0.5])
    t_panic = int(8 * fps)
    delay = None

    vw = writer(OUT / name, fps, (W, H))
    for f in range(int(seconds * fps)):
        vec = target - pos
        d = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        step = vec / d * np.minimum(walk[:, None], d)
        reached = d[:, 0] < 3
        target[reached] = rng.uniform([20, 20], [W - 20, H - 20], (reached.sum(), 2))
        if panic and f >= t_panic:
            away = pos - centre
            r = np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
            if delay is None:
                delay = (r[:, 0] / 60.0) * fps * 0.5   # panic spreads outward in ~2-3 s
            fleeing = (f - t_panic) >= delay
            step[fleeing] = (away / r)[fleeing] * run[fleeing, None] + rng.normal(0, 0.3, (fleeing.sum(), 2))
        pos = np.clip(pos + step + rng.normal(0, 0.15, (N, 2)), 6, [W - 6, H - 6])

        frame = floor.copy()
        for (x, y), c in zip(pos.astype(int), colors):
            cv2.circle(frame, (x + 2, y + 2), 7, (90, 90, 90), -1)              # shadow
            cv2.circle(frame, (x, y), 7, tuple(int(v) for v in c), -1)          # shoulders
            cv2.circle(frame, (x, y), 3, (40, 30, 25), -1)                      # head
        noise = rng.normal(0, 3, frame.shape)                                   # camera noise
        vw.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    vw.release()
    print(f"  {name}: {int(seconds * fps)} frames, {W}x{H}, {fps} fps" + (", panic at 8.0 s" if panic else ""))


def main():
    print("Real UCSD Ped2 clips:")
    ucsd_to_mp4("Test", "Test004", "ucsd_test004_cart.mp4")
    ucsd_to_mp4("Test", "Test001", "ucsd_test001_bike.mp4")
    ucsd_to_mp4("Test", "Test007", "ucsd_test007.mp4")
    ucsd_to_mp4("Train", "Train001", "ucsd_train001_normal.mp4")
    print("Synthetic crowd clips:")
    synthetic_crowd("synthetic_panic.mp4", panic=True)
    synthetic_crowd("synthetic_calm.mp4", panic=False)
    print(f"Saved in {OUT}")


if __name__ == "__main__":
    main()
