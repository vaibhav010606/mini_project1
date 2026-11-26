"""FastAPI backend for the React frontend.

    python -m uvicorn api.server:app --port 8000

Endpoints
    GET  /api/health        server is up
    GET  /api/test-videos   UCSD Ped2 videos you can pick, with true anomaly times
    POST /api/detect        upload a video or pick one -> Model 1 alerts      -> Video upload page
    POST /api/crowd         run a 5G crowd scenario -> Model 2 risk + heatmaps -> 5G crowd monitor
    POST /api/combined      one event -> CCTV video + 5G data -> combined model -> Combined monitor
    /files/...              static files from results/ (annotated videos, CCTV frames)

The Smart mall 3D and CCTV wall pages run entirely in the browser and need no endpoint.

Video detection runs in a separate process (api/detect_worker.py) so this server never loads
CUDA together with XGBoost (that combination can crash Python on Windows).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import MODEL_DIR, PED2_DIR, RESULTS  # noqa: E402
from network_model.simulator import FEATURES, GNB, LABELS, NZ, VENUE, WINDOW, ZONE, run_episode  # noqa: E402
from network_model.train import summary_features  # noqa: E402

WEB = RESULTS / "web"
UPLOADS = WEB / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}
MAX_UPLOAD_MB = 200

app = FastAPI(title="Crowd Anomaly 5G API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=RESULTS), name="files")


def clean(x):
    """Make values JSON-safe (numpy types, inf/nan -> None)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


@lru_cache(maxsize=None)
def xgb_model(name: str):
    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_DIR / name)
    return clf


# ------------------------------------------------------------------ basic
@app.get("/api/health")
def health():
    return {"ok": True}


@lru_cache(maxsize=1)
def _ped2_videos():
    from video_model.dataset import _labels_from_m

    def count(d):
        return sum(1 for f in d.iterdir() if f.suffix.lower() == ".tif" and not f.name.startswith("."))

    out = []
    test_dirs = sorted(d for d in (PED2_DIR / "Test").iterdir() if re.fullmatch(r"Test\d{3}", d.name))
    n = [count(d) for d in test_dirs]
    labels = _labels_from_m(n) or [np.zeros(k) for k in n]
    for d, k, lab in zip(test_dirs, n, labels):
        lab = np.asarray(lab)
        edges = np.flatnonzero(np.diff(np.r_[0, lab, 0]))
        ranges = [[round(a / 10, 1), round(b / 10, 1)] for a, b in zip(edges[::2], edges[1::2])]
        out.append(dict(id=d.name, split="Test", frames=k, fps=10, anomaly_s=ranges))
    for d in sorted(d for d in (PED2_DIR / "Train").iterdir() if re.fullmatch(r"Train\d{3}", d.name))[:4]:
        out.append(dict(id=d.name, split="Train", frames=count(d), fps=10, anomaly_s=[]))
    return out


@app.get("/api/test-videos")
def test_videos():
    if not PED2_DIR.exists():
        return []
    return _ped2_videos()


# ------------------------------------------------------------------ page 1: video detection
@app.post("/api/detect")
def detect(mode: Literal["auto", "ae", "motion"] = Form("auto"),
           model: Literal["pred", "big", "student"] = Form("pred"),
           video_id: Optional[str] = Form(None),
           file: Optional[UploadFile] = File(None)):
    truth = None
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"Unsupported file type {ext}. Use one of: {', '.join(sorted(VIDEO_EXT))}")
        path = UPLOADS / f"{uuid.uuid4().hex[:10]}{ext}"
        with path.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        if path.stat().st_size > MAX_UPLOAD_MB * 1e6:
            path.unlink()
            raise HTTPException(400, f"File larger than {MAX_UPLOAD_MB} MB")
        source = file.filename
    elif video_id:
        if not re.fullmatch(r"(Test|Train)\d{3}", video_id):
            raise HTTPException(400, "Unknown video id")
        path = PED2_DIR / ("Test" if video_id.startswith("Test") else "Train") / video_id
        if not path.exists():
            raise HTTPException(404, "Video not found")
        source = f"UCSD Ped2 {video_id}"
        truth = next((v["anomaly_s"] for v in _ped2_videos() if v["id"] == video_id), None)
    else:
        raise HTTPException(400, "Upload a video file or choose a test video")

    job = uuid.uuid4().hex[:10]
    out = WEB / job
    r = subprocess.run([sys.executable, str(ROOT / "api" / "detect_worker.py"), "--video", str(path),
                        "--mode", mode, "--model", model, "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not (out / "result.json").exists():
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise HTTPException(500, "Detection failed: " + " | ".join(tail))
    res = json.loads((out / "result.json").read_text())
    res.update(source=source, truth_anomaly_s=truth,
               video_url=f"/files/web/{job}/annotated.webm", peak_url=f"/files/web/{job}/peak.jpg")
    return clean(res)


# ------------------------------------------------------------------ page 2: 5G crowd monitor
class CrowdReq(BaseModel):
    scenario: Literal["normal", "gathering", "stampede"] = "stampede"
    n_people: int = Field(1800, ge=300, le=3000)
    seed: int = Field(0, ge=0, le=10_000)


@app.post("/api/crowd")
def crowd(req: CrowdReq):
    T, step = 180, 2
    times = tuple(range(0, T, step))
    feats, labels, snaps, info = run_episode(np.random.default_rng(req.seed), req.scenario, T=T,
                                             n_people=req.n_people, snapshot_times=times)
    win = np.stack([feats[e - WINDOW:e] for e in range(WINDOW, T + 1)])
    prob = xgb_model("network_xgb.json").predict_proba(summary_features(win)[0])
    risk = [None] * (WINDOW - 1) + [[round(float(p), 3) for p in row] for row in prob]
    frames = []
    for t in times:
        true_d, est_d, lab = snaps[t]
        hot = np.unravel_index(int(np.argmax(est_d)), est_d.shape)
        frames.append(dict(t=t, label=int(lab), est=np.round(est_d, 2), true=np.round(true_d, 2),
                           hotspot=[(hot[0] + 0.5) * ZONE, (hot[1] + 0.5) * ZONE]))
    return clean(dict(
        scenario=req.scenario, seconds=T, labels=LABELS, venue=VENUE, zone=ZONE, grid=NZ,
        gnb=GNB, info=dict(n_people=info["n_people"], n_phones=info["n_phones"], panic_t0=info["panic_t0"]),
        features={name: np.round(feats[:, i], 4) for i, name in enumerate(FEATURES)},
        truth=labels, risk=risk, frames=frames,
    ))


# ------------------------------------------------------------------ combined monitor (video + 5G)
class CombinedReq(BaseModel):
    scenario: Literal["normal", "gathering", "stampede"] = "stampede"
    n_people: int = Field(2000, ge=500, le=2600)
    seed: int = Field(0, ge=0, le=10_000)
    night: bool = False
    few_phones: bool = False


@app.post("/api/combined")
def combined(req: CombinedReq):
    job = uuid.uuid4().hex[:10]
    out = WEB / job
    cmd = [sys.executable, str(ROOT / "api" / "combined_worker.py"), "--scenario", req.scenario,
           "--people", str(req.n_people), "--seed", str(req.seed), "--out", str(out)]
    if req.night:
        cmd.append("--night")
    if req.few_phones:
        cmd.append("--few-phones")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not (out / "result.json").exists():
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise HTTPException(500, "Combined run failed: " + " | ".join(tail))
    res = json.loads((out / "result.json").read_text())
    res["frames_url"] = f"/files/web/{job}/frames.jpg"
    return res
