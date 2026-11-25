"""One simulated crowd event -> CCTV video + 5G data of the same people -> 3 models -> result.json.

Runs in its own process (PyTorch/CUDA) so the API server never mixes CUDA with XGBoost.
Writes into --out:
    result.json   truth per second, probabilities of video-only / 5G-only / combined models,
                  5G density maps every 2 s, event info
    frames.jpg    all 180 CCTV frames as one sprite sheet (15 columns x 12 rows of 256x256), a clearer
                  display rendering of the same scene; the model itself analyses 96x96 frames
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import MODEL_DIR  # noqa: E402
from joint.generate import CAM_SIZE, CAM_X0, CAM_Y0, RES, T, camera_sees, render_video  # noqa: E402
from joint.train import CLIP, WIN, CrowdModel  # noqa: E402
from network_model.simulator import GNB, LABELS, NZ, VENUE, ZONE, run_episode  # noqa: E402
from network_model.train import summary_features  # noqa: E402

COLS = 15
DISPLAY = 256  # display resolution for people to watch (the model itself uses RES x RES)


def render_display(positions: np.ndarray, night: bool, rng: np.random.Generator) -> np.ndarray:
    """Clearer CCTV rendering of the same scene for the web page: people drawn as small figures."""
    ppm = DISPLAY / CAM_SIZE  # pixels per metre
    floor = np.full((DISPLAY, DISPLAY, 3), (168, 172, 176), np.float32)
    floor += rng.normal(0, 5, (DISPLAY, DISPLAY, 1))
    for k in range(0, DISPLAY, int(5 * ppm)):  # 5 m paving tiles
        floor[k, :] -= 18
        floor[:, k] -= 18
    floor = np.clip(floor, 0, 255).astype(np.uint8)
    n = positions.shape[1]
    colours = rng.integers(40, 230, (n, 3))
    out = np.empty((len(positions), DISPLAY, DISPLAY, 3), np.uint8)
    for t, xy in enumerate(positions):
        img = floor.copy()
        inside = np.flatnonzero(camera_sees(xy))
        px = (xy[inside, 0] - CAM_X0) * ppm
        py = DISPLAY - 1 - (xy[inside, 1] - CAM_Y0) * ppm
        for i, x, y in zip(inside, px.astype(int), py.astype(int)):
            cv2.circle(img, (x + 1, y + 1), 3, (120, 122, 125), -1)                 # shadow
            cv2.circle(img, (x, y), 3, tuple(int(c) for c in colours[i]), -1)       # shoulders
            cv2.circle(img, (x, y), 1, (35, 30, 28), -1)                            # head
        if night:
            img = img.astype(np.float32) * 0.33
            img[..., 0] += 12  # slight blue tint of night CCTV
            img = np.clip(img + rng.normal(0, 9, img.shape), 0, 255).astype(np.uint8)
        out[t] = img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="stampede", choices=["normal", "gathering", "stampede"])
    ap.add_argument("--people", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--night", action="store_true")
    ap.add_argument("--few-phones", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    times = tuple(range(0, T, 2))
    feats, labels, snaps, info = run_episode(np.random.default_rng(a.seed), a.scenario, T=T, n_people=a.people,
                                             snapshot_times=times, no_phone_p=0.6 if a.few_phones else 0.05, track=True)
    frames = render_video(info["positions"], a.night, np.random.default_rng(a.seed + 1))

    # sprite sheet of clearer display frames for the page (the model uses `frames` below)
    display = render_display(info["positions"], a.night, np.random.default_rng(a.seed + 2))
    rows = int(np.ceil(T / COLS))
    sheet = np.zeros((rows * DISPLAY, COLS * DISPLAY, 3), np.uint8)
    for t in range(T):
        r, c = divmod(t, COLS)
        sheet[r * DISPLAY:(r + 1) * DISPLAY, c * DISPLAY:(c + 1) * DISPLAY] = display[t]
    cv2.imwrite(str(out / "frames.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])

    # model inputs for every second t >= 30
    ends = np.arange(WIN, T + 1)
    wins = np.stack([feats[e - WIN:e] for e in ends]).astype(np.float32)
    netx_raw = summary_features(wins)[0].astype(np.float32)
    clips = np.stack([frames[e - CLIP:e] for e in ends])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probs = {}
    with torch.no_grad():
        for mode in ("video", "net", "fusion"):
            ck = torch.load(MODEL_DIR / f"joint_{mode}.pt", map_location=dev, weights_only=False)
            model = CrowdModel(mode, netx_raw.shape[1]).to(dev)
            model.load_state_dict(ck["state"])
            model.eval()
            netx = torch.tensor((netx_raw - ck["mu"]) / ck["sd"], dtype=torch.float32, device=dev)
            clip = torch.tensor(clips, device=dev)
            p = torch.softmax(model(clip, netx), 1).cpu().numpy()
            probs[mode] = [None] * (WIN - 1) + [[round(float(v), 3) for v in row] for row in p]

    maps = []
    for t in times:
        true_d, est_d, lab = snaps[t]
        maps.append(dict(t=t, est=np.round(est_d, 2).tolist(), true=np.round(true_d, 2).tolist()))
    hot = np.asarray(info["hotspot"])
    res = dict(
        seconds=T, labels=LABELS, venue=VENUE, zone=ZONE, grid=NZ, gnb=GNB.tolist(),
        camera=dict(x0=CAM_X0, y0=CAM_Y0, size=CAM_SIZE, res=DISPLAY, model_res=RES, cols=COLS),
        info=dict(scenario=a.scenario, n_people=info["n_people"], n_phones=info["n_phones"],
                  panic_t0=info["panic_t0"], hotspot=hot.round(1).tolist(),
                  hotspot_in_view=bool(camera_sees(hot)), night=a.night, few_phones=a.few_phones, seed=a.seed),
        truth=labels.tolist(), probs=probs, maps=maps,
    )
    (out / "result.json").write_text(json.dumps(res))
    print("ok")


if __name__ == "__main__":
    main()
