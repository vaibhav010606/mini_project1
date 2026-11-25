"""Upload a crowd video and get anomaly alerts.

    python detect.py --video my_crowd.mp4                    # auto mode
    python detect.py --video data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004 --mode ae
    python detect.py --video my_crowd.mp4 --mode motion

Modes
  ae      the trained autoencoder (Model 1). Accurate on UCSD-style scenes; on a scene it
          has never seen, everything looks unusual, so use it only for similar cameras.
  motion  general crowd-motion detector using optical flow. Learns "normal" motion from
          the first seconds of the video, then alerts when motion suddenly becomes much
          stronger (people running / scattering). Works on any FIXED camera.
  auto    'ae' for UCSD Ped2-sized frames, otherwise 'motion'.

Outputs (in results/detect/): annotated video, score graph, JSON with alert times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from common import MODEL_DIR, RESULTS, plot_style
from video_model.dataset import CLIP, SIZE, load_ped2, make_clips
from video_model.models import MODELS

OUT = RESULTS / "detect"
OUT.mkdir(parents=True, exist_ok=True)


def read_video(src: str, max_frames: int = 3000):
    p = Path(src)
    frames, fps = [], 25.0
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".tif", ".png", ".jpg", ".bmp")
                       and not f.name.startswith("."))
        frames = [cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files[:max_frames]]
        fps = 10.0
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    frames = [f for f in frames if f is not None]
    if len(frames) < CLIP + 1:
        raise SystemExit("Video too short.")
    return frames, fps


# ------------------------------------------------------------------ autoencoder mode
@torch.no_grad()
def raw_patch_scores(model, gray: np.ndarray, dev):
    clips = make_clips(gray)
    out = []
    for i in range(0, len(clips), 64):
        x = torch.tensor(clips[i:i + 64], device=dev)
        err = ((model(x) - x) ** 2).mean(1, keepdim=True)
        out.append(F.avg_pool2d(err, 16, 8).amax((1, 2, 3)).cpu().numpy())
    s = np.concatenate(out)
    return np.concatenate([np.repeat(s[:1], CLIP // 2), s, np.repeat(s[-1:], CLIP // 2)])


def ae_scores(frames, model_name="pred"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "pred":
        from video_model.predict_model import SIZE as size, PredUNet, video_errors

        model = PredUNet().to(dev)
        score_fn = lambda m, v: video_errors(m, v, dev)[1]  # noqa: E731  worst-patch prediction error
    else:
        size, model = SIZE, MODELS[model_name]().to(dev)
        score_fn = lambda m, v: raw_patch_scores(m, v, dev)  # noqa: E731
    model.load_state_dict(torch.load(MODEL_DIR / f"video_{model_name}.pt", map_location=dev))
    model.eval()
    calib = MODEL_DIR / f"video_{model_name}_calib.json"
    if calib.exists():
        c = json.loads(calib.read_text())
    else:  # threshold = what normal training clips almost never exceed
        train, _, _ = load_ped2(size)
        s = np.concatenate([gaussian_filter1d(score_fn(model, v), 2) for v in train])
        c = dict(p50=float(np.percentile(s, 50)), p995=float(np.percentile(s, 99.5)))
        calib.write_text(json.dumps(c))
    gray = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)
                     for f in frames]).astype(np.float32) / 255.0
    s = gaussian_filter1d(score_fn(model, gray), 2)
    # scale so that 1.0 = alert threshold
    return (s - c["p50"]) / (c["p995"] - c["p50"] + 1e-12), 1.0


# ------------------------------------------------------------------ motion mode
def motion_scores(frames, fps, ref_seconds=3.0):
    h, w = frames[0].shape[:2]
    scale = 320 / max(w, 1)
    size = (320, max(1, int(h * scale)))
    prev = cv2.cvtColor(cv2.resize(frames[0], size), cv2.COLOR_BGR2GRAY)
    energy = [0.0]
    for f in frames[1:]:
        g = cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2) * fps / 10.0  # comparable across frame rates
        moving = mag > 0.5
        # strong motion of many pixels = running crowd
        energy.append(float(np.percentile(mag, 95)) * (0.5 + moving.mean()))
        prev = g
    e = gaussian_filter1d(np.array(energy), max(1.0, fps / 5))
    e[0] = e[1]
    n_ref = max(5, int(ref_seconds * fps))
    ref = e[:n_ref]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826 + 0.05 * med + 1e-6
    z = (e - med) / mad
    return z / 6.0, 1.0  # 1.0 = six robust standard deviations above normal motion


# ------------------------------------------------------------------ output
def intervals(alert, fps, min_len):
    out, start = [], None
    for i, a in enumerate(np.r_[alert, False]):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                out.append((round(start / fps, 2), round(i / fps, 2)))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file or folder of frames")
    ap.add_argument("--mode", choices=["auto", "ae", "motion"], default="auto")
    ap.add_argument("--model", choices=["pred", *MODELS], default="pred",
                    help="video model in ae mode: pred = v2 future-frame prediction (best), big/student = v1 autoencoders")
    args = ap.parse_args()

    frames, fps = read_video(args.video)
    h, w = frames[0].shape[:2]
    mode = args.mode
    if mode == "auto":
        mode = "ae" if (w, h) in [(360, 240), (238, 158)] else "motion"
    print(f"{len(frames)} frames, {w}x{h}, {fps:.1f} fps -> mode: {mode}")

    score, thr = ae_scores(frames, args.model) if mode == "ae" else motion_scores(frames, fps)
    alert = score > thr
    ivs = intervals(alert, fps, min_len=max(2, int(0.3 * fps)))
    name = f"{Path(args.video).stem}_{mode}"
    res = dict(video=str(args.video), mode=mode, frames=len(frames), fps=fps,
               alert_intervals_s=ivs, alert_fraction=float(alert.mean()))
    (OUT / f"{name}_alerts.json").write_text(json.dumps(res, indent=2))

    vw = cv2.VideoWriter(str(OUT / f"{name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f, s, a in zip(frames, score, alert):
        f = f.copy()
        if a:
            cv2.rectangle(f, (0, 0), (w - 1, h - 1), (0, 0, 255), max(4, w // 80))
            cv2.putText(f, "ANOMALY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 600), (0, 0, 255), 2)
        cv2.putText(f, f"score {s:.2f}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 900),
                    (0, 255, 255), 1)
        vw.write(f)
    vw.release()

    plt = plot_style()
    t = np.arange(len(score)) / fps
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(t, score, color="#2f6feb", label="anomaly score")
    ax.axhline(thr, color="#cf222e", ls="--", lw=1, label="alert threshold")
    for a, b in ivs:
        ax.axvspan(a, b, color="#ffc9c9", zorder=0)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{name}: {mode} mode, {len(ivs)} alert period(s)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / f"{name}_scores.png")
    plt.close(fig)

    print(f"Alerts (seconds): {ivs if ivs else 'none'}")
    print(f"Saved: {OUT / (name + '_annotated.mp4')}\n       {OUT / (name + '_scores.png')}")


if __name__ == "__main__":
    main()
