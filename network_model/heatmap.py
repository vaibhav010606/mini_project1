"""Crowd density heatmaps and a stampede timeline from 5G network data.

1. Heatmaps: true crowd density vs the density the network estimates from phone
   positioning, at three moments of one stampede episode (calm, packed, panic).
2. Timeline: key network features over time with the XGBoost risk prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, MODEL_DIR, plot_style  # noqa: E402
from network_model.simulator import FEATURES, GNB, LABELS, VENUE, WINDOW, run_episode  # noqa: E402
from network_model.train import summary_features  # noqa: E402

SEED = 2024


def main():
    import xgboost as xgb

    plt = plot_style()
    _, _, _, info = run_episode(np.random.default_rng(SEED), "stampede", n_people=2200)
    p0 = info["panic_t0"]
    times = (40, p0 - 3, p0 + 6)
    feats, labels, snaps, info = run_episode(np.random.default_rng(SEED), "stampede",
                                             n_people=2200, snapshot_times=times)

    # ---------- heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    vmax = 3.0
    titles = ["Calm", "Packed (dense)", "Panic (stampede)"]
    for c, t in enumerate(times):
        true_d, est_d, lab = snaps[t]
        for r, (dmap, src) in enumerate([(true_d, "True"), (est_d, "Estimated from 5G network")]):
            ax = axes[r, c]
            im = ax.imshow(dmap.T, origin="lower", extent=[0, VENUE, 0, VENUE], cmap="RdYlGn_r",
                           vmin=0, vmax=vmax)
            ax.scatter(GNB[:, 0], GNB[:, 1], marker="^", c="black", s=30, label="5G cell")
            ax.grid(False)
            ax.set_title(f"{src}\n{titles[c]}, t={t}s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="people per m²  (≥2 = dangerous)")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Crowd density heatmap: ground truth vs 5G network estimate", fontsize=12)
    fig.savefig(FIG_DIR / "net_heatmaps.png")
    plt.close(fig)

    # ---------- timeline with model prediction
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_DIR / "network_xgb.json")
    T = len(labels)
    win = np.stack([feats[e - WINDOW:e] for e in range(WINDOW, T + 1)])
    prob = clf.predict_proba(summary_features(win)[0])
    tt = np.arange(WINDOW - 1, T)

    show = ["max_zone_density", "frac_fast", "handover_rate", "max_cell_load"]
    fig, axes = plt.subplots(len(show) + 1, 1, figsize=(9, 9), sharex=True)
    colors = {1: "#fff3b0", 2: "#ffc9c9"}
    for ax in axes:
        for k, col in colors.items():
            m = labels == k
            if m.any():
                for s in np.where(np.diff(np.r_[0, m.astype(int), 0]) == 1)[0]:
                    e = s + np.argmax(~m[s:]) if (~m[s:]).any() else T
                    ax.axvspan(s, e, color=col, zorder=0)
    for ax, name in zip(axes, show):
        ax.plot(feats[:, FEATURES.index(name)], color="#2f6feb")
        ax.set_ylabel(name, fontsize=8)
    axes[-1].plot(tt, prob[:, 1], label="P(dense)", color="#d4a72c")
    axes[-1].plot(tt, prob[:, 2], label="P(stampede)", color="#cf222e")
    axes[-1].set_ylabel("model risk", fontsize=8)
    axes[-1].set_ylim(-0.05, 1.05)
    axes[-1].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time (s)   [yellow = truly dense, red = stampede]")
    axes[0].set_title("One simulated stampede seen through 5G network data")
    fig.savefig(FIG_DIR / "net_timeline.png")
    plt.close(fig)
    print("Saved net_heatmaps.png and net_timeline.png")


if __name__ == "__main__":
    main()
