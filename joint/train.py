"""Combined crowd-risk model: CCTV video + 5G network data of the SAME event.

    python joint/train.py

Input for each moment t:
    video : the last 4 CCTV frames (96x96) up to t, plus their frame differences (motion)
    5G    : the last 30 s of the 9 network features, summarised (last/mean/max/min/std/slope)
Output: normal / dense / stampede at time t.

Three models are trained on exactly the same samples so they can be compared fairly:
    video-only   CNN on the frames
    5G-only      MLP on the network features
    combined     both branches -> fusion layer (trained end to end, with "modality dropout" so it
                 still works when one source is useless, e.g. dark video or few phones)
Split is by event (no event appears in both train and test).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import DATA_DIR, FIG_DIR, MODEL_DIR, RESULTS, plot_style, set_seed  # noqa: E402
from network_model.simulator import LABELS  # noqa: E402
from network_model.train import summary_features  # noqa: E402

CLIP, WIN, STRIDE = 4, 30, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ models
class VideoBranch(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        chans = [2 * CLIP - 1, 32, 64, 128, 128]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(256, out), nn.ReLU(inplace=True))

    def forward(self, clip_u8):  # (B, 4, H, W) uint8
        x = clip_u8.float() / 255.0
        x = x - x.mean((2, 3), keepdim=True)
        x = x / (x.std((1, 2, 3), keepdim=True) + 0.02)  # per-clip contrast normalisation (helps at night)
        x = torch.cat([x, x[:, 1:] - x[:, :-1]], 1)     # frames + motion
        h = self.conv(x)
        return self.fc(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))


class NetBranch(nn.Module):
    def __init__(self, n_in, out=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                 nn.Linear(128, out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.mlp(x)


class CrowdModel(nn.Module):
    """mode: 'video', 'net' or 'fusion'."""

    def __init__(self, mode, n_net):
        super().__init__()
        self.mode = mode
        self.video = VideoBranch() if mode in ("video", "fusion") else None
        self.net = NetBranch(n_net) if mode in ("net", "fusion") else None
        dim = (128 if self.video else 0) + (64 if self.net else 0)
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 3))

    def forward(self, clip, netx, drop_p=0.0):
        parts = []
        if self.video:
            hv = self.video(clip)
            parts.append(hv)
        if self.net:
            hn = self.net(netx)
            parts.append(hn)
        if self.training and drop_p > 0 and len(parts) == 2:
            # modality dropout: sometimes hide one source so the model learns to use the other alone
            r = torch.rand(len(hv), device=hv.device)
            parts[0] = parts[0] * (r >= drop_p).float()[:, None]
            parts[1] = parts[1] * ((r < drop_p) | (r >= 2 * drop_p)).float()[:, None]
        return self.head(torch.cat(parts, 1))


# ------------------------------------------------------------------ data
def build_windows(labels):
    E = labels.shape[0]
    ends = np.arange(WIN, labels.shape[1] + 1, STRIDE)
    ev = np.repeat(np.arange(E), len(ends))
    en = np.tile(ends, E)
    return ev, en, labels[ev, en - 1]


def split_events(scenario, seed=0):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(scenario):
        idx = rng.permutation(np.flatnonzero(scenario == s))
        n = len(idx)
        te += list(idx[: int(0.2 * n)])
        va += list(idx[int(0.2 * n): int(0.3 * n)])
        tr += list(idx[int(0.3 * n):])
    return np.array(tr), np.array(va), np.array(te)


class Data:
    def __init__(self):
        d = np.load(DATA_DIR / "joint_dataset.npz")
        self.meta = {k: d[k] for k in ("scenario", "night", "low_phones", "hotspot_in_view", "panic_t0")}
        self.video = torch.tensor(d["video"], device=DEV)  # uint8 (E, T, H, W)
        self.labels = d["labels"]
        self.ev, self.en, self.y = build_windows(self.labels)
        net = d["net"]
        wins = np.stack([net[e, t - WIN:t] for e, t in zip(self.ev, self.en)]).astype(np.float32)
        self.netx_raw = summary_features(wins)[0].astype(np.float32)
        self.tr_ev, self.va_ev, self.te_ev = split_events(self.meta["scenario"])

    def idx(self, events):
        return np.flatnonzero(np.isin(self.ev, events))

    def batch(self, ii, flip=False):
        clip_idx = torch.tensor(self.en[ii][:, None] - CLIP + np.arange(CLIP)[None], device=DEV)
        ev = torch.tensor(self.ev[ii], device=DEV)[:, None].expand(-1, CLIP)
        clip = self.video[ev, clip_idx]  # (B, 4, H, W)
        if flip:
            if np.random.rand() < 0.5:
                clip = torch.flip(clip, [3])
            if np.random.rand() < 0.5:
                clip = torch.flip(clip, [2])
        return clip, self.netx[ii], torch.tensor(self.y[ii], device=DEV)


# ------------------------------------------------------------------ training / evaluation
@torch.no_grad()
def predict(model, data, ii, bs=512):
    model.eval()
    out = []
    for i in range(0, len(ii), bs):
        clip, netx, _ = data.batch(ii[i:i + bs])
        out.append(torch.softmax(model(clip, netx), 1).cpu().numpy())
    return np.concatenate(out)


def train_model(mode, data, epochs=18, bs=64):
    from sklearn.metrics import f1_score

    set_seed(0)
    model = CrowdModel(mode, data.netx.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr, va = data.idx(data.tr_ev), data.idx(data.va_ev)
    w = torch.tensor(len(tr) / (3 * np.bincount(data.y[tr], minlength=3)), dtype=torch.float32, device=DEV)
    best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            clip, netx, y = data.batch(perm[i:i + bs], flip=True)
            loss = F.cross_entropy(model(clip, netx, drop_p=0.15), y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        f1 = f1_score(data.y[va], predict(model, data, va).argmax(1), average="macro")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  [{mode:6s}] best val macro-F1 = {best:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score

    pred = p.argmax(1)
    danger = y > 0
    return dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        danger_detected=float((pred[danger] > 0).mean()) if danger.any() else None,
        false_alarm=float((pred[~danger] > 0).mean()) if (~danger).any() else None,
        stampede_detected=float((pred[y == 2] == 2).mean()) if (y == 2).any() else None,
    )


def main():
    from sklearn.metrics import confusion_matrix

    set_seed(0)
    data = Data()
    mu = data.netx_raw[data.idx(data.tr_ev)].mean(0)
    sd = data.netx_raw[data.idx(data.tr_ev)].std(0) + 1e-6
    data.netx = torch.tensor((data.netx_raw - mu) / sd, device=DEV)
    print(f"windows={len(data.y)}  events train/val/test = {len(data.tr_ev)}/{len(data.va_ev)}/{len(data.te_ev)}")

    te = data.idx(data.te_ev)
    ev_te = data.ev[te]
    m = data.meta
    conditions = {
        "All test events": np.ones(len(te), bool),
        "Daytime, danger in view, phones normal": (m["night"][ev_te] == 0) & (m["hotspot_in_view"][ev_te] == 1) & (m["low_phones"][ev_te] == 0),
        "Night (dark video)": m["night"][ev_te] == 1,
        "Danger outside camera view": m["hotspot_in_view"][ev_te] == 0,
        "Few phones visible": m["low_phones"][ev_te] == 1,
    }
    results, probs = {}, {}
    for mode in ("video", "net", "fusion"):
        model = train_model(mode, data)
        torch.save({"state": model.state_dict(), "mode": mode, "mu": mu, "sd": sd}, MODEL_DIR / f"joint_{mode}.pt")
        p = predict(model, data, te)
        probs[mode] = p
        results[mode] = {c: scores(data.y[te][mask], p[mask]) for c, mask in conditions.items()}

    # stampede detection delay per test stampede event
    delays = {}
    for mode in probs:
        dl, missed = [], 0
        for e in np.unique(ev_te):
            if m["panic_t0"][e] < 0:
                continue
            rows = np.flatnonzero(ev_te == e)
            ends = data.en[te][rows]
            hit = ends[(probs[mode][rows].argmax(1) == 2) & (ends - 1 >= m["panic_t0"][e])]
            if len(hit):
                dl.append(float(hit.min() - 1 - m["panic_t0"][e]))
            else:
                missed += 1
        delays[mode] = dict(median_s=float(np.median(dl)) if dl else None, missed=missed, total=len(dl) + missed)

    names = {"video": "Video only", "net": "5G only", "fusion": "Combined (video + 5G)"}
    out = dict(conditions={c: int(mk.sum()) for c, mk in conditions.items()},
               test_events=dict(total=len(data.te_ev), night=int(m["night"][data.te_ev].sum()),
                                outside_view=int((m["hotspot_in_view"][data.te_ev] == 0).sum()),
                                few_phones=int(m["low_phones"][data.te_ev].sum())),
               results={names[k]: v for k, v in results.items()},
               stampede_delay={names[k]: v for k, v in delays.items()})
    (RESULTS / "joint_metrics.json").write_text(json.dumps(out, indent=2))

    print("\nMacro-F1 (normal / dense / stampede) on unseen test events:")
    print(f"  {'condition':42s}" + "".join(f"{names[k]:>24s}" for k in names))
    for c in conditions:
        print(f"  {c:42s}" + "".join(f"{results[k][c]['macro_f1']:>24.3f}" for k in names))
    print("Danger detected / false alarms (all test events):")
    for k in names:
        r = results[k]["All test events"]
        print(f"  {names[k]:24s} danger detected {r['danger_detected']:.1%}   false alarms {r['false_alarm']:.1%}   "
              f"stampede delay {delays[k]['median_s']} s, missed {delays[k]['missed']}/{delays[k]['total']}")

    # ---- figures
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    conds = list(conditions)
    xs = np.arange(len(conds))
    colors = {"video": "#d4a72c", "net": "#8250df", "fusion": "#1a7f37"}
    for j, k in enumerate(names):
        b = ax.bar(xs + (j - 1) * 0.27, [results[k][c]["macro_f1"] for c in conds], 0.27, label=names[k], color=colors[k])
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(xs, [c.replace(", ", ",\n") for c in conds], fontsize=8)
    ax.set_ylabel("macro-F1 (normal / dense / stampede)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Same events, three models: where each source fails and how combining helps")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(FIG_DIR / "joint_by_condition.png")
    plt.close(fig)

    cm = confusion_matrix(data.y[te], probs["fusion"].argmax(1), labels=[0, 1, 2])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Combined model (test events)")
    fig.savefig(FIG_DIR / "joint_confusion.png")
    plt.close(fig)
    print("Saved results/joint_metrics.json, models and figures.")


if __name__ == "__main__":
    main()
