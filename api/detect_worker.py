"""Runs video anomaly detection in its own process for the API server.

Keeping PyTorch/CUDA out of the server process avoids the CUDA + XGBoost/OpenMP crash
seen on Windows. Writes browser-friendly outputs into --out:
    result.json      scores per frame, threshold, alert intervals
    annotated.webm   video with red border + "ANOMALY" on alert frames (VP8, plays in browsers)
    peak.jpg         the most anomalous frame
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detect import ae_scores, intervals, motion_scores, read_video  # noqa: E402


def annotate(frame, score, alert):
    h, w = frame.shape[:2]
    f = frame.copy()
    if alert:
        cv2.rectangle(f, (0, 0), (w - 1, h - 1), (0, 0, 255), max(4, w // 80))
        cv2.putText(f, "ANOMALY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 600), (0, 0, 255), 2)
    cv2.putText(f, f"score {score:.2f}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 900), (0, 255, 255), 1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "ae", "motion"])
    ap.add_argument("--model", default="pred", choices=["pred", "big", "student"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, fps = read_video(a.video, max_frames=1500)
    h, w = frames[0].shape[:2]
    mode = a.mode
    if mode == "auto":
        mode = "ae" if (w, h) in [(360, 240), (238, 158)] else "motion"
    score, thr = ae_scores(frames, a.model) if mode == "ae" else motion_scores(frames, fps)
    alert = score > thr
    ivs = intervals(alert, fps, min_len=max(2, int(0.3 * fps)))

    vw = cv2.VideoWriter(str(out / "annotated.webm"), cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
    for fr, s, al in zip(frames, score, alert):
        vw.write(annotate(fr, float(s), bool(al)))
    vw.release()
    peak = int(np.argmax(score))
    cv2.imwrite(str(out / "peak.jpg"), annotate(frames[peak], float(score[peak]), bool(alert[peak])))

    res = dict(mode=mode, model=a.model if mode == "ae" else None, fps=float(fps), n_frames=len(frames),
               width=w, height=h, threshold=float(thr), scores=[round(float(s), 3) for s in score],
               alerts=ivs, alert_fraction=float(alert.mean()), peak_time=round(peak / fps, 2))
    (out / "result.json").write_text(json.dumps(res))
    print("ok")


if __name__ == "__main__":
    main()
