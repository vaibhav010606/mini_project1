"""Model 1 v2: future-frame prediction U-Net (after Liu et al., CVPR 2018).

The network sees 4 consecutive frames and predicts the 5th. Normal motion is easy to
predict; a bike, skater or cart moving unusually is not, so prediction error marks anomalies.
Compared with v1 (reconstruction autoencoder) it also uses 256x256 frames (2x resolution)
and a gradient loss that keeps edges sharp.

    python video_model/predict_model.py train
    python video_model/predict_model.py eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, MODEL_DIR, RESULTS, plot_style, set_seed  # noqa: E402
from video_model.dataset import load_ped2  # noqa: E402

SIZE = 256
T_IN = 4
PATCH = 32
SMOOTH = 3.0


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class PredUNet(nn.Module):
    def __init__(self, t_in: int = T_IN, widths=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        c = t_in
        for w in widths:
            self.downs.append(_block(c, w))
            c = w
        self.mid = _block(widths[-1], widths[-1] * 2)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        c = widths[-1] * 2
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, 2))
            self.dec.append(_block(w * 2, w))
            c = w
        self.out = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return torch.sigmoid(self.out(x))


def gradient_loss(pred, target):
    def grads(img):
        return img[..., :, 1:] - img[..., :, :-1], img[..., 1:, :] - img[..., :-1, :]

    (px, py), (tx, ty) = grads(pred), grads(target)
    return (px.abs() - tx.abs()).abs().mean() + (py.abs() - ty.abs()).abs().mean()


def _index(videos):
    """(video, start) pairs for every 5-frame window."""
    return [(v, s) for v, vid in enumerate(videos) for s in range(len(vid) - T_IN)]


def train(epochs: int, batch: int = 8, lr: float = 2e-4):
    set_seed(42)
    dev = "cuda"
    train_v, _, _ = load_ped2(SIZE)
    vids = [torch.tensor(v, device=dev) for v in train_v]
    idx = _index(train_v)
    print(f"[pred] training windows={len(idx)} size={SIZE} device={dev}")
    model = PredUNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler()
    t0, losses = time.time(), []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(idx))
        tot = 0.0
        for i in range(0, len(perm), batch):
            sel = [idx[j] for j in perm[i:i + batch]]
            clip = torch.stack([vids[v][s:s + T_IN + 1] for v, s in sel])  # (B,5,H,W)
            if np.random.rand() < 0.5:
                clip = torch.flip(clip, dims=[3])
            x, y = clip[:, :T_IN], clip[:, T_IN:]
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(x)
            p = p.float()
            loss = F.mse_loss(p, y) + gradient_loss(p, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(sel)
        sched.step()
        losses.append(tot / len(idx))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep + 1:3d}/{epochs}  loss={losses[-1]:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), MODEL_DIR / "video_pred.pt")
    n_params = sum(p.numel() for p in model.parameters())
    (MODEL_DIR / "video_pred_train.json").write_text(json.dumps(
        dict(model="pred", params=n_params, epochs=epochs, final_loss=losses[-1], train_seconds=time.time() - t0)))
    print(f"Saved model ({n_params:,} params) in {time.time() - t0:.0f}s")


@torch.no_grad()
def video_errors(model, video: np.ndarray, dev, batch: int = 16):
    """Per-frame (full-frame PSNR score, worst-patch error, error maps). First T_IN frames padded."""
    v = torch.tensor(video, device=dev)
    psnr, patch, maps = [], [], []
    for i in range(0, len(video) - T_IN, batch):
        starts = range(i, min(i + batch, len(video) - T_IN))
        x = torch.stack([v[s:s + T_IN] for s in starts])
        y = torch.stack([v[s + T_IN:s + T_IN + 1] for s in starts])
        err = (model(x) - y) ** 2
        mse = err.mean((1, 2, 3))
        psnr.append((-10 * torch.log10(mse + 1e-10)).cpu().numpy())
        patch.append(F.avg_pool2d(err, PATCH, PATCH // 2).amax((1, 2, 3)).cpu().numpy())
        maps.append(err[:, 0].cpu().numpy())
    psnr, patch, maps = np.concatenate(psnr), np.concatenate(patch), np.concatenate(maps)
    pad = lambda a: np.concatenate([np.repeat(a[:1], T_IN, 0), a])  # noqa: E731
    return pad(psnr), pad(patch), pad(maps)


def norm_smooth(s):
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return gaussian_filter1d(s, SMOOTH)


def evaluate():
    from sklearn.metrics import roc_auc_score, roc_curve

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, test, labels = load_ped2(SIZE)
    model = PredUNet().to(dev)
    model.load_state_dict(torch.load(MODEL_DIR / "video_pred.pt", map_location=dev))
    model.eval()
    res = [video_errors(model, v, dev) for v in test]
    y = np.concatenate(labels)
    psnr_score = [norm_smooth(-p) for p, _, _ in res]  # low PSNR = anomaly
    patch_score = [norm_smooth(q) for _, q, _ in res]
    aucs = {
        "global_psnr_raw": roc_auc_score(y, np.concatenate([-p for p, _, _ in res])),
        "global_psnr_norm_smooth": roc_auc_score(y, np.concatenate(psnr_score)),
        "patch_max": roc_auc_score(y, np.concatenate([q for _, q, _ in res])),
        "patch_max_norm_smooth": roc_auc_score(y, np.concatenate(patch_score)),
    }
    final = patch_score  # same scoring rule as v1, chosen before seeing v2 results
    fpr, tpr, thr = roc_curve(y, np.concatenate(final))
    best = int(np.argmax(tpr - fpr))
    print("[pred] AUC  " + "  ".join(f"{k}={v:.3f}" for k, v in aucs.items()))

    mfile = RESULTS / "video_metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
    metrics["pred"] = dict(auc=aucs, threshold=float(thr[best]), tpr_at_threshold=float(tpr[best]),
                           fpr_at_threshold=float(fpr[best]))
    mfile.write_text(json.dumps(metrics, indent=2))
    sfile = RESULTS / "video_scores.npz"
    scores = dict(np.load(sfile)) if sfile.exists() else {"labels": y}
    scores["score_pred"] = np.concatenate(final)
    np.savez(sfile, **scores)

    plt = plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for name, key in [("v1 autoencoder (big)", "big"), ("v1 autoencoder (student)", "student")]:
        s = scores.get(f"score_{key}")
        if s is not None:
            f_, t_, _ = roc_curve(y, s)
            ax.plot(f_, t_, label=f"{name} (AUC={roc_auc_score(y, s):.3f})", alpha=0.8)
    ax.plot(fpr, tpr, label=f"v2 future-frame prediction (AUC={aucs['patch_max_norm_smooth']:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Video anomaly detection, UCSD Ped2")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(FIG_DIR / "video_roc.png")
    plt.close(fig)

    pick = [0, 3, 5, 9]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    for ax, vi in zip(axes.ravel(), pick):
        ax.fill_between(np.arange(len(labels[vi])), 0, labels[vi], color="#ffc9c9", step="mid", label="true anomaly")
        ax.plot(final[vi], color="#2f6feb", label="anomaly score")
        ax.axhline(thr[best], color="gray", ls="--", lw=0.8, label="alert threshold")
        ax.set_title(f"Test video {vi + 1:03d}", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Anomaly score over time (v2 future-frame prediction)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "video_score_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for c, vi in enumerate([0, 3, 9]):
        fi = int(np.argmax(final[vi]))
        frame, err = test[vi][fi], res[vi][2][fi]
        axes[0, c].imshow(frame, cmap="gray")
        axes[0, c].set_title(f"Video {vi + 1:03d}, frame {fi + 1} (true: {'anomaly' if labels[vi][fi] else 'normal'})", fontsize=8)
        axes[1, c].imshow(frame, cmap="gray")
        axes[1, c].imshow(err / (err.max() + 1e-12), cmap="jet", alpha=0.45)
        axes[1, c].set_title("Prediction error (red = unusual)", fontsize=8)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.suptitle("Where the model sees the anomaly (v2)")
    fig.savefig(FIG_DIR / "video_examples.png")
    plt.close(fig)
    print("Updated video_metrics.json, video_scores.npz and video figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    train(a.epochs) if a.action == "train" else evaluate()
