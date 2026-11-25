"""Fusion of Model 1 (video) and Model 2 (5G network) into one crowd-safety alert.

No public dataset has CCTV video and 5G network data of the SAME crowd, so we build a
paired evaluation set from the two test sets (a common practice for multimodal studies):

  * an "incident" (something safety-relevant happens) or "no incident" is drawn
  * VIDEO side: when there is an incident, the camera sees it with probability 0.75
    (otherwise it is dark, blocked or out of view and shows a normal-looking frame)
  * NETWORK side: when there is an incident, the crowd change is visible in the 5G data
    with probability 0.80 (otherwise few phones there / very local event)
  * scores come from the REAL test outputs of both trained models

Fusion methods compared: video only, network only, max rule, average, learned
logistic-regression fusion (trained on half, tested on the other half).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, RESULTS, plot_style  # noqa: E402

N = 6000
P_INCIDENT = 0.3
P_CAM_SEES = 0.75
P_NET_SEES = 0.80


def tpr_at_fpr(y, s, fpr_target=0.05):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(5)
    v = np.load(RESULTS / "video_scores.npz")
    n = np.load(RESULTS / "network_test_scores.npz")
    v_key = "score_pred" if "score_pred" in v.files else "score_big"  # best available video model
    v_score, v_lab = v[v_key], v["labels"]
    print(f"  video scores from: {v_key}")
    n_score = 1.0 - n["prob_xgb"][:, 0]  # P(dense or stampede)
    n_lab = (n["y"] > 0).astype(int)

    v_anom, v_norm = np.where(v_lab == 1)[0], np.where(v_lab == 0)[0]
    n_anom, n_norm = np.where(n_lab == 1)[0], np.where(n_lab == 0)[0]

    y = (rng.random(N) < P_INCIDENT).astype(int)
    cam_sees = rng.random(N) < P_CAM_SEES
    net_sees = rng.random(N) < P_NET_SEES
    vi = np.where((y == 1) & cam_sees, rng.choice(v_anom, N), rng.choice(v_norm, N))
    ni = np.where((y == 1) & net_sees, rng.choice(n_anom, N), rng.choice(n_norm, N))
    sv, sn = v_score[vi], n_score[ni]

    half = N // 2
    lr = LogisticRegression().fit(np.c_[sv, sn][:half], y[:half])
    s_lr = lr.predict_proba(np.c_[sv, sn])[:, 1]

    te = slice(half, N)
    methods = {
        "Video only (Model 1)": sv,
        "Network only (Model 2)": sn,
        "Fusion: max": np.maximum(sv, sn),
        "Fusion: average": (sv + sn) / 2,
        "Fusion: learned (LogReg)": s_lr,
    }
    res = {}
    for k, s in methods.items():
        res[k] = dict(auc=float(roc_auc_score(y[te], s[te])), tpr_at_5pct_fpr=tpr_at_fpr(y[te], s[te]))
    # incidents where the camera could not see: does the system still catch them?
    blind = (y == 1) & ~cam_sees
    blind_te = np.zeros(N, bool)
    blind_te[half:] = blind[half:]
    neg_te = np.zeros(N, bool)
    neg_te[half:] = y[half:] == 0
    for k, s in methods.items():
        thr = np.quantile(s[neg_te], 0.95)  # threshold giving 5% false alarms
        res[k]["caught_when_camera_blind"] = float((s[blind_te] > thr).mean())
    res["setup"] = dict(N=N, p_incident=P_INCIDENT, p_cam_sees=P_CAM_SEES, p_net_sees=P_NET_SEES,
                        logreg_weights=dict(video=float(lr.coef_[0, 0]), network=float(lr.coef_[0, 1])))
    (RESULTS / "fusion_results.json").write_text(json.dumps(res, indent=2))
    for k in methods:
        r = res[k]
        print(f"  {k:26s} AUC={r['auc']:.3f}  detect@5%FA={r['tpr_at_5pct_fpr']:.3f}  "
              f"camera-blind caught={r['caught_when_camera_blind']:.2f}")

    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in ["Video only (Model 1)", "Network only (Model 2)", "Fusion: learned (LogReg)"]:
        fpr, tpr, _ = roc_curve(y[te], methods[k][te])
        axes[0].plot(fpr, tpr, label=f"{k} (AUC={res[k]['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False alarm rate")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("Video + 5G network fusion")
    axes[0].legend(fontsize=8, loc="lower right")
    names = list(methods)
    xs = np.arange(len(names))
    b1 = axes[1].bar(xs - 0.2, [res[k]["tpr_at_5pct_fpr"] for k in names], 0.4, label="all incidents", color="#2f6feb")
    b2 = axes[1].bar(xs + 0.2, [res[k]["caught_when_camera_blind"] for k in names], 0.4,
                     label="camera blind", color="#cf222e")
    axes[1].bar_label(b1, fmt="%.2f", fontsize=7)
    axes[1].bar_label(b2, fmt="%.2f", fontsize=7)
    axes[1].set_xticks(xs, [k.replace(" (", "\n(").replace(": ", ":\n") for k in names], fontsize=7)
    axes[1].set_ylabel("detection rate at 5% false alarms")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Incidents caught")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_results.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
