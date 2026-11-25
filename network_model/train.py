"""Model 2: crowd-risk detection from 5G network data.

Input : last 30 s of per-second network observations (see simulator.FEATURES)
Output: normal / dense / stampede

Three detectors are compared:
  * rule baseline  - fixed thresholds on density and fast-moving fraction (tuned on train)
  * XGBoost        - on summary statistics of the 30 s window (main model)
  * LSTM           - on the raw 30 x 9 sequence
Splits are by episode, so no simulated crowd appears in both train and test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, FIG_DIR, MODEL_DIR, RESULTS, plot_style, set_seed  # noqa: E402
from network_model.simulator import FEATURES, LABELS, WINDOW  # noqa: E402

STATS = ["last", "last5", "mean", "max", "min", "std", "slope"]


def summary_features(X: np.ndarray):
    t = np.arange(X.shape[1], dtype=np.float32)
    t = (t - t.mean()) / ((t - t.mean()) ** 2).sum()
    parts = [X[:, -1], X[:, -5:].mean(1), X.mean(1), X.max(1), X.min(1), X.std(1),
             np.einsum("nwf,w->nf", X, t)]
    names = [f"{s}:{f}" for s in STATS for f in FEATURES]
    return np.concatenate(parts, 1), names


def rule_predict(X, dens_thr, fast_thr):
    dens = X[:, -1, FEATURES.index("max_zone_density")]
    fast = X[:, -1, FEATURES.index("frac_fast")]
    return np.where(fast > fast_thr, 2, np.where(dens > dens_thr, 1, 0))


def tune_rule(X, y):
    from sklearn.metrics import f1_score

    best = (-1, None)
    for d in np.arange(0.8, 3.01, 0.1):
        for f in np.arange(0.05, 0.61, 0.05):
            s = f1_score(y, rule_predict(X, d, f), average="macro")
            if s > best[0]:
                best = (s, (float(d), float(f)))
    return best[1]


def train_lstm(Xtr, ytr, Xva, yva, device):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    class Net(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.lstm = nn.LSTM(f, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    net = Net(Xtr.shape[2]).to(device)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    Xv = torch.tensor(Xva, device=device)
    best_f1, best_state = -1, None
    for ep in range(30):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv).argmax(1).cpu().numpy()
        f1 = f1_score(yva, pv, average="macro")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"  LSTM best val macro-F1 = {best_f1:.3f}")
    return net


def detection_delay(y_true, y_pred, episode):
    """Seconds between the first true stampede window and the first predicted one, per episode."""
    delays, missed = [], 0
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        yt, yp = y_true[idx], y_pred[idx]
        if not (yt == 2).any():
            continue
        first_true = np.argmax(yt == 2)
        hits = np.where(yp[first_true:] == 2)[0]
        if len(hits) == 0:
            missed += 1
        else:
            delays.append(hits[0] * 3)  # window stride is 3 s
    return delays, missed


def plot_confusion(cm, title, path):
    plt = plot_style()
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def main():
    import torch
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    set_seed(42)
    d = np.load(DATA_DIR / "network_dataset.npz")
    X, y, ep, sc = d["X"].astype(np.float32), d["y"], d["episode"], d["scenario"]
    print(f"Loaded {len(y)} windows from {len(np.unique(ep))} episodes")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, ep))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1)
    tr2, va = next(gss2.split(X[tr], y[tr], ep[tr]))
    tr_fit, va = tr[tr2], tr[va]

    results = {}

    # ---- rule baseline
    d_thr, f_thr = tune_rule(X[tr], y[tr])
    p_rule = rule_predict(X[te], d_thr, f_thr)
    results["rule"] = dict(accuracy=accuracy_score(y[te], p_rule),
                           macro_f1=f1_score(y[te], p_rule, average="macro"),
                           thresholds=dict(density=d_thr, frac_fast=f_thr))

    # ---- XGBoost
    F, names = summary_features(X)
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9,
                            colsample_bytree=0.8, objective="multi:softprob", num_class=3,
                            tree_method="hist", eval_metric="mlogloss", random_state=0)
    sw = (len(y[tr]) / (3 * np.bincount(y[tr], minlength=3)))[y[tr]]
    clf.fit(F[tr], y[tr], sample_weight=sw)
    prob_xgb = clf.predict_proba(F[te])
    p_xgb = prob_xgb.argmax(1)
    results["xgboost"] = dict(accuracy=accuracy_score(y[te], p_xgb),
                              macro_f1=f1_score(y[te], p_xgb, average="macro"),
                              report=classification_report(y[te], p_xgb, target_names=LABELS, output_dict=True))
    clf.save_model(MODEL_DIR / "network_xgb.json")

    # ---- LSTM
    mu = X[tr_fit].reshape(-1, X.shape[2]).mean(0)
    sd = X[tr_fit].reshape(-1, X.shape[2]).std(0) + 1e-6
    norm = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731
    # CPU on purpose: the LSTM is small, and using CUDA in the same process as XGBoost/
    # scikit-learn makes Python abort on exit on Windows (OpenMP runtime clash).
    device = "cpu"
    net = train_lstm(norm(X[tr_fit]), y[tr_fit], norm(X[va]), y[va], device)
    net.eval()
    with torch.no_grad():
        prob_lstm = torch.softmax(net(torch.tensor(norm(X[te]), device=device)), 1).cpu().numpy()
    p_lstm = prob_lstm.argmax(1)
    results["lstm"] = dict(accuracy=accuracy_score(y[te], p_lstm),
                           macro_f1=f1_score(y[te], p_lstm, average="macro"),
                           report=classification_report(y[te], p_lstm, target_names=LABELS, output_dict=True))
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd}, MODEL_DIR / "network_lstm.pt")

    # ---- stampede detection delay
    for name, pred in [("rule", p_rule), ("xgboost", p_xgb), ("lstm", p_lstm)]:
        delays, missed = detection_delay(y[te], pred, ep[te])
        results[name]["stampede_delay_median_s"] = float(np.median(delays)) if delays else None
        results[name]["stampede_episodes_missed"] = int(missed)
        results[name]["stampede_episodes_total"] = int(len(delays) + missed)

    for name in ["rule", "xgboost", "lstm"]:
        r = results[name]
        print(f"  {name:8s} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}  "
              f"stampede delay(median)={r['stampede_delay_median_s']}s  missed={r['stampede_episodes_missed']}/{r['stampede_episodes_total']}")

    with open(RESULTS / "network_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(RESULTS / "network_test_scores.npz", y=y[te], prob_xgb=prob_xgb, prob_lstm=prob_lstm,
             episode=ep[te], scenario=sc[te])

    # ---- figures
    plot_confusion(confusion_matrix(y[te], p_xgb), "Network model (XGBoost)", FIG_DIR / "net_confusion_xgb.png")
    plot_confusion(confusion_matrix(y[te], p_lstm), "Network model (LSTM)", FIG_DIR / "net_confusion_lstm.png")

    plt = plot_style()
    imp = clf.feature_importances_
    top = np.argsort(imp)[::-1][:12]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in top][::-1], imp[top][::-1], color="#2f6feb")
    ax.set_title("Most useful 5G network features (XGBoost)")
    ax.set_xlabel("Importance")
    fig.savefig(FIG_DIR / "net_feature_importance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    models = ["Rule baseline", "XGBoost", "LSTM"]
    acc = [results[k]["accuracy"] for k in ["rule", "xgboost", "lstm"]]
    f1 = [results[k]["macro_f1"] for k in ["rule", "xgboost", "lstm"]]
    xs = np.arange(3)
    b1 = ax.bar(xs - 0.18, acc, 0.36, label="Accuracy", color="#2f6feb")
    b2 = ax.bar(xs + 0.18, f1, 0.36, label="Macro F1", color="#1a7f37")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(xs, models)
    ax.set_ylim(0, 1.1)
    ax.set_title("Crowd-risk detection from 5G network data")
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "net_model_comparison.png")
    plt.close(fig)
    print("Saved metrics, scores, models and figures.")


if __name__ == "__main__":
    main()
