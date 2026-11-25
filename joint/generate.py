"""Joint synthetic dataset: the SAME crowd event seen by a CCTV camera AND by the 5G network.

For every simulated event (normal / gathering / stampede):
  * the 5G simulator moves 800-2600 people across the 200 m x 200 m venue
  * 5G side : per-second network features from those people's phones (as in Model 2)
  * video   : a fixed overhead CCTV camera films an 80 m x 80 m part of the venue,
              one 96x96 grayscale frame per second, rendered from the same positions

Realistic hard cases so that neither source is enough on its own:
  * night (20% of events)          dark, noisy video with poor contrast
  * danger outside the camera view the hotspot is randomly placed; the camera covers only part of the venue
  * few phones visible (20%)       60% of people invisible to the operator (no phone / phone off / other operator)

    python joint/generate.py --episodes 100      (per scenario)
Output: data/joint_dataset.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA_DIR, FIG_DIR, plot_style  # noqa: E402
from network_model.simulator import FEATURES, SCENARIOS, run_episode  # noqa: E402

CAM_X0, CAM_Y0, CAM_SIZE = 40.0, 60.0, 80.0  # metres: camera sees x 40-120, y 60-140
RES = 96                                     # pixels per side (1.2 px per metre)
T = 180


def camera_sees(xy):
    return (xy[..., 0] >= CAM_X0) & (xy[..., 0] < CAM_X0 + CAM_SIZE) & \
           (xy[..., 1] >= CAM_Y0) & (xy[..., 1] < CAM_Y0 + CAM_SIZE)


def render_video(positions: np.ndarray, night: bool, rng: np.random.Generator) -> np.ndarray:
    """positions (T, n, 2) metres -> (T, RES, RES) uint8 overhead CCTV frames."""
    floor = 150 + gaussian_filter(rng.normal(0, 14, (RES, RES)), 1.2)  # paving texture
    floor[::12, :] -= 10
    floor[:, ::12] -= 10
    frames = np.empty((len(positions), RES, RES), np.uint8)
    for t, xy in enumerate(positions):
        inside = camera_sees(xy)
        px = ((xy[inside] - [CAM_X0, CAM_Y0]) / CAM_SIZE * RES).astype(int)
        px = np.clip(px, 0, RES - 1)
        occ = np.zeros((RES, RES))
        np.add.at(occ, (RES - 1 - px[:, 1], px[:, 0]), 1.0)  # image row 0 = far side
        people = gaussian_filter(occ, 0.7)
        img = floor - 55 * np.minimum(people, 2.5)
        if night:
            img = 40 + (img - 40) * 0.3 + rng.normal(0, 10, img.shape)
        else:
            img = img + rng.normal(0, 3, img.shape)
        frames[t] = np.clip(img, 0, 255).astype(np.uint8)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100, help="events per scenario")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    E = a.episodes * len(SCENARIOS)
    video = np.empty((E, T, RES, RES), np.uint8)
    net = np.empty((E, T, len(FEATURES)), np.float32)
    labels = np.empty((E, T), np.int64)
    meta = {k: np.empty(E) for k in ("scenario", "night", "low_phones", "hotspot_in_view", "n_people", "panic_t0")}
    t0 = time.time()
    e = 0
    for s_idx, scen in enumerate(SCENARIOS):
        for _ in range(a.episodes):
            night = rng.random() < 0.2
            low = rng.random() < 0.2
            feats, lab, _, info = run_episode(rng, scen, T=T, no_phone_p=0.6 if low else 0.05, track=True)
            video[e] = render_video(info["positions"], night, rng)
            net[e], labels[e] = feats, lab
            meta["scenario"][e] = s_idx
            meta["night"][e] = night
            meta["low_phones"][e] = low
            meta["hotspot_in_view"][e] = bool(camera_sees(np.asarray(info["hotspot"])))
            meta["n_people"][e] = info["n_people"]
            meta["panic_t0"][e] = info["panic_t0"] if info["panic_t0"] is not None else -1
            e += 1
        print(f"  {scen}: {a.episodes} events ({time.time() - t0:.0f}s)", flush=True)

    out = DATA_DIR / "joint_dataset.npz"
    np.savez(out, video=video, net=net, labels=labels, features=np.array(FEATURES), **meta)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.0f} MB): {E} events, video {video.shape}, 5G {net.shape}")
    print(f"  night={int(meta['night'].sum())}  few-phones={int(meta['low_phones'].sum())}  "
          f"hotspot outside camera={int((meta['hotspot_in_view'] == 0).sum())}")

    # preview: one frame per situation
    plt = plot_style()
    picks = []
    for name, cond in [("normal", (meta["scenario"] == 0) & (meta["night"] == 0)),
                       ("gathering (dense)", (meta["scenario"] == 1) & (meta["night"] == 0) & (meta["hotspot_in_view"] == 1)),
                       ("stampede", (meta["scenario"] == 2) & (meta["night"] == 0) & (meta["hotspot_in_view"] == 1)),
                       ("stampede at night", (meta["scenario"] == 2) & (meta["night"] == 1))]:
        idx = np.flatnonzero(cond)
        if len(idx):
            ei = int(idx[0])
            ti = int(meta["panic_t0"][ei] + 6) if meta["panic_t0"][ei] > 0 else (int(np.argmax(labels[ei])) or 150)
            picks.append((name, ei, ti))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.4))
    for ax, (name, ei, ti) in zip(np.atleast_1d(axes), picks):
        ax.imshow(video[ei, ti], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\nevent {ei}, t={ti}s", fontsize=9)
        ax.axis("off")
    fig.suptitle("Synthetic CCTV frames rendered from the same crowd as the 5G data")
    fig.savefig(FIG_DIR / "joint_video_preview.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
