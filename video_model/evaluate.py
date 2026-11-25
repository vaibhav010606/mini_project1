"""Score every UCSD Ped2 test frame and measure frame-level AUC.

Anomaly score of a frame:
  1. reconstruct the 5-frame clip centred on it
  2. squared error map, averaged over the 5 frames
  3. split into 16x16 patches; score = error of the worst patch (anomalies are small objects)
  4. per-video min-max normalisation and temporal smoothing (standard practice on Ped2)
AUC is also reported for simpler scores so the effect of each step is visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, MODEL_DIR, RESULTS, plot_style  # noqa: E402
from video_model.dataset import CLIP, load_ped2, make_clips  # noqa: E402
from video_model.models import MODELS  # noqa: E402

PATCH = 16
SMOOTH = 3.0


@torch.no_grad()
def frame_scores(model, video: np.ndarray, dev: str, batch: int = 64):
    """Returns (global_mse, patch_max, error_maps) per frame of the video."""
    clips = make_clips(video)
    g, p, maps = [], [], []
    for i in range(0, len(clips), batch):
        x = torch.tensor(clips[i:i + batch], device=dev)
        err = ((model(x) - x) ** 2).mean(1, keepdim=True)  # (B,1,H,W)
        g.append(err.mean((1, 2, 3)).cpu().numpy())
        p.append(F.avg_pool2d(err, PATCH, PATCH // 2).amax((1, 2, 3)).cpu().numpy())
        maps.append(err[:, 0].cpu().numpy())
    g, p, maps = np.concatenate(g), np.concatenate(p), np.concatenate(maps)
    pad = CLIP // 2  # clip i is centred on frame i+2; pad the edges
    padf = lambda a: np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])  # noqa: E731
    return padf(g), padf(p), padf(maps)


def normalise(s):
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return gaussian_filter1d(s, SMOOTH)


def evaluate(name: str, test, labels, dev):
    from sklearn.metrics import roc_auc_score, roc_curve

    model = MODELS[name]().to(dev)
    model.load_state_dict(torch.load(MODEL_DIR / f"video_{name}.pt", map_location=dev))
    model.eval()
    per_video, maps_all = [], []
    for v in test:
        g, p, maps = frame_scores(model, v, dev)
        per_video.append((g, p))
        maps_all.append(maps)
    y = np.concatenate(labels)
    s_global = np.concatenate([g for g, _ in per_video])
    s_patch = np.concatenate([p for _, p in per_video])
    final = [normalise(p) for _, p in per_video]
    s_final = np.concatenate(final)
    aucs = {
        "global_mse": roc_auc_score(y, s_global),
        "patch_max": roc_auc_score(y, s_patch),
        "patch_max_norm_smooth": roc_auc_score(y, s_final),
    }
    fpr, tpr, thr = roc_curve(y, s_final)
    best = np.argmax(tpr - fpr)
    return dict(aucs=aucs, threshold=float(thr[best]), tpr=float(tpr[best]), fpr=float(fpr[best]),
                roc=(fpr, tpr), scores=final, maps=maps_all)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, test, labels = load_ped2()
    out, res = {}, {}
    for name in MODELS:
        if not (MODEL_DIR / f"video_{name}.pt").exists():
            continue
        r = evaluate(name, test, labels, dev)
        res[name] = r
        out[name] = dict(auc=r["aucs"], threshold=r["threshold"], tpr_at_threshold=r["tpr"],
                         fpr_at_threshold=r["fpr"])
        a = r["aucs"]
        print(f"[{name}] AUC  global={a['global_mse']:.3f}  patch={a['patch_max']:.3f}  "
              f"patch+norm+smooth={a['patch_max_norm_smooth']:.3f}")
    (RESULTS / "video_metrics.json").write_text(json.dumps(out, indent=2))
    np.savez(RESULTS / "video_scores.npz", labels=np.concatenate(labels),
             video=np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)]),
             **{f"score_{k}": np.concatenate(v["scores"]) for k, v in res.items()})

    plt = plot_style()
    # ROC
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for name, r in res.items():
        fpr, tpr = r["roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['aucs']['patch_max_norm_smooth']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Video anomaly detection, UCSD Ped2")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "video_roc.png")
    plt.close(fig)

    # score curves for 4 test videos
    r = res.get("big") or next(iter(res.values()))
    pick = [0, 3, 5, 9]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    for ax, vi in zip(axes.ravel(), pick):
        lab = labels[vi]
        ax.fill_between(np.arange(len(lab)), 0, lab, color="#ffc9c9", step="mid", label="true anomaly")
        ax.plot(r["scores"][vi], color="#2f6feb", label="anomaly score")
        ax.axhline(r["threshold"], color="gray", ls="--", lw=0.8, label="alert threshold")
        ax.set_title(f"Test video {vi + 1:03d}", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Anomaly score over time (big model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "video_score_curves.png")
    plt.close(fig)

    # example frames with error heatmap
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for c, vi in enumerate([0, 3, 9]):
        fi = int(np.argmax(r["scores"][vi]))
        frame = test[vi][fi]
        err = r["maps"][vi][fi]
        axes[0, c].imshow(frame, cmap="gray")
        axes[0, c].set_title(f"Video {vi + 1:03d}, frame {fi + 1} (true: {'anomaly' if labels[vi][fi] else 'normal'})", fontsize=8)
        axes[1, c].imshow(frame, cmap="gray")
        axes[1, c].imshow(err / (err.max() + 1e-12), cmap="jet", alpha=0.45)
        axes[1, c].set_title("Reconstruction error (red = unusual)", fontsize=8)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.suptitle("Where the model sees the anomaly")
    fig.savefig(FIG_DIR / "video_examples.png")
    plt.close(fig)
    print("Saved video_metrics.json, video_scores.npz and figures.")


if __name__ == "__main__":
    main()
