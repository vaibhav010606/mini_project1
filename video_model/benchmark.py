"""Measure how fast each video model runs on 'edge' and 'cloud' hardware.

Edge server stand-in : this laptop's CPU, limited to 4 threads
Cloud server stand-in: this laptop's NVIDIA GPU
Also measures the JPEG size of one camera frame (the 5G upload size).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, MODEL_DIR, PED2_DIR, RESULTS, plot_style  # noqa: E402
from video_model.models import MODELS, n_params  # noqa: E402

EDGE_THREADS = 4


@torch.no_grad()
def time_model(model, dev, iters, shape=(1, 5, 128, 128)):
    x = torch.rand(*shape, device=dev)
    for _ in range(10):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000.0


def frame_jpeg_kb():
    files = sorted((PED2_DIR / "Test" / "Test001").glob("*.tif"))[:20]
    sizes = {}
    for label, wh in [("ped2_native_360x240", None), ("hd_1280x720", (1280, 720))]:
        s = []
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if wh:
                img = cv2.resize(img, wh, interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            s.append(len(buf) / 1024)
        sizes[label] = float(np.mean(s)) if s else None
    return sizes


def main():
    torch.set_num_threads(EDGE_THREADS)
    metrics = json.loads((RESULTS / "video_metrics.json").read_text()) if (RESULTS / "video_metrics.json").exists() else {}
    from video_model.predict_model import PredUNet, SIZE as PRED_SIZE, T_IN

    rows = {}
    pred_path = MODEL_DIR / "video_pred.pt"
    if pred_path.exists():
        m = PredUNet()
        m.load_state_dict(torch.load(pred_path, map_location="cpu"))
        m.eval()
        cpu_ms = time_model(m, "cpu", 20, shape=(1, T_IN, PRED_SIZE, PRED_SIZE))
        gpu_ms = time_model(m.cuda(), "cuda", 200, shape=(1, T_IN, PRED_SIZE, PRED_SIZE)) if torch.cuda.is_available() else None
        rows["pred"] = dict(params=n_params(m), size_mb=pred_path.stat().st_size / 1e6, cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                            auc=metrics.get("pred", {}).get("auc", {}).get("patch_max_norm_smooth"))
        print(f"[pred] params={rows['pred']['params']:,}  size={rows['pred']['size_mb']:.1f}MB  "
              f"edge(CPU)={cpu_ms:.2f}ms  cloud(GPU)={gpu_ms:.2f}ms  AUC={rows['pred']['auc']}")
    for name, ctor in MODELS.items():
        path = MODEL_DIR / f"video_{name}.pt"
        if not path.exists():
            continue
        m = ctor()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        cpu_ms = time_model(m, "cpu", 50)
        gpu_ms = time_model(m.cuda(), "cuda", 300) if torch.cuda.is_available() else None
        rows[name] = dict(params=n_params(m), size_mb=path.stat().st_size / 1e6, cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                          auc=metrics.get(name, {}).get("auc", {}).get("patch_max_norm_smooth"))
        print(f"[{name}] params={rows[name]['params']:,}  size={rows[name]['size_mb']:.1f}MB  "
              f"edge(CPU)={cpu_ms:.2f}ms  cloud(GPU)={gpu_ms:.2f}ms  AUC={rows[name]['auc']}")
    out = dict(models=rows, frame_kb=frame_jpeg_kb(), edge_threads=EDGE_THREADS,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("frame sizes (KB):", out["frame_kb"])
    (RESULTS / "benchmark.json").write_text(json.dumps(out, indent=2))

    plt = plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    names = list(rows)
    cols = ["#8250df", "#2f6feb", "#1a7f37"][-len(names):]
    vals = [[rows[n]["auc"] or 0 for n in names], [rows[n]["size_mb"] for n in names]]
    for ax, v, t in zip(axes[:2], vals, ["Accuracy (AUC)", "Model size (MB)"]):
        b = ax.bar(names, v, color=cols)
        ax.bar_label(b, fmt="%.3g", fontsize=8)
        ax.set_title(t)
    xs = np.arange(len(names))
    b1 = axes[2].bar(xs - 0.18, [rows[n]["cpu_ms"] for n in names], 0.36, label="Edge (CPU)", color="#d4a72c")
    b2 = axes[2].bar(xs + 0.18, [rows[n]["gpu_ms"] or 0 for n in names], 0.36, label="Cloud (GPU)", color="#8250df")
    axes[2].bar_label(b1, fmt="%.1f", fontsize=8)
    axes[2].bar_label(b2, fmt="%.1f", fontsize=8)
    axes[2].set_xticks(xs, names)
    axes[2].set_title("Time per frame (ms)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Video models: accuracy vs size vs speed (pred = v2 cloud model, student = edge model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "video_benchmark.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
