"""ML-based offloading: learn when to process at the 5G edge and when in the cloud.

1. Run the 5G MEC simulator under many random conditions, both paths each time.
2. Label = the path with the lower 95th-percentile alert delay.
3. Train XGBoost on what the system can observe (cameras, frame rate, public load,
   slicing on/off, edge server busy share, cloud RTT).
4. Compare on unseen conditions: always-edge, always-cloud, ML decision, perfect oracle.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, FIG_DIR, MODEL_DIR, RESULTS, plot_style  # noqa: E402
from edge_cloud.sim_5g_mec import Config, simulate  # noqa: E402

FEATS = ["n_cams", "fps", "bg_load", "slicing", "edge_other_load", "cloud_rtt_ms"]


def build(n_configs=500, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for i in range(n_configs):
        base = Config(
            n_cams=int(rng.integers(1, 81)),
            fps=float(rng.choice([5.0, 10.0, 15.0])),
            bg_load=float(rng.uniform(0, 0.95)),
            slicing=bool(rng.random() < 0.5),
            edge_other_load=float(rng.uniform(0, 0.8)),
            cloud_rtt_ms=float(rng.uniform(20, 100)),
            sim_s=10.0,
            seed=i,
        )
        e = simulate(replace(base, path="edge"))
        c = simulate(replace(base, path="cloud"))
        rows.append([base.n_cams, base.fps, base.bg_load, float(base.slicing), base.edge_other_load,
                     base.cloud_rtt_ms, e["p95_ms"], c["p95_ms"], e["mean_ms"], c["mean_ms"]])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_configs} conditions simulated ({time.time() - t0:.0f}s)")
    return np.array(rows)


def main():
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    cache = DATA_DIR / "offload_dataset.npy"
    if cache.exists():
        D = np.load(cache)
    else:
        D = build()
        np.save(cache, D)
    X = D[:, :6]
    e_p95, c_p95 = D[:, 6], D[:, 7]
    y = (c_p95 < e_p95).astype(int)  # 1 = cloud is better
    print(f"conditions={len(y)}  edge better={int((y == 0).sum())}  cloud better={int((y == 1).sum())}")

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
    clf.fit(X[idx_tr], y[idx_tr], sample_weight=np.abs(e_p95[idx_tr] - c_p95[idx_tr]) ** 0.5 + 1)
    pred = clf.predict(X[idx_te])
    clf.save_model(MODEL_DIR / "offload_xgb.json")

    cap = lambda a: np.minimum(a, 1000.0)  # noqa: E731  (cap overload cases at 1 s so means stay readable)
    pe, pc = cap(e_p95[idx_te]), cap(c_p95[idx_te])
    policies = {
        "Always edge": pe,
        "Always cloud": pc,
        "ML decides": np.where(pred == 1, pc, pe),
        "Oracle (best possible)": np.minimum(pe, pc),
    }
    res = {k: dict(mean_p95_ms=float(v.mean()), miss_rate=float((v > 100).mean())) for k, v in policies.items()}
    res["ml_accuracy"] = float((pred == y[idx_te]).mean())
    (RESULTS / "offload_results.json").write_text(json.dumps(res, indent=2))
    print(f"  ML decision accuracy = {res['ml_accuracy']:.3f}")
    for k in policies:
        print(f"  {k:24s} avg p95 delay = {res[k]['mean_p95_ms']:7.1f} ms   conditions over 100 ms = {res[k]['miss_rate']:.1%}")

    plt = plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(policies)
    ticks = [k.replace(" (", "\n(") for k in names]
    cols = ["#d4a72c", "#8250df", "#1a7f37", "#8c959f"]
    b = axes[0].bar(ticks, [res[k]["mean_p95_ms"] for k in names], color=cols)
    axes[0].bar_label(b, fmt="%.0f", fontsize=8)
    axes[0].set_ylabel("average 95th-pct delay (ms)")
    axes[0].set_title("Where to process: edge, cloud, or let ML decide")
    axes[0].tick_params(axis="x", labelsize=8)
    b = axes[1].bar(ticks, [res[k]["miss_rate"] * 100 for k in names], color=cols)
    axes[1].bar_label(b, fmt="%.0f%%", fontsize=8)
    axes[1].set_ylabel("% of conditions with p95 > 100 ms")
    axes[1].set_title("Deadline misses")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.savefig(FIG_DIR / "offload_policies.png")
    plt.close(fig)

    imp = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(5, 3))
    o = np.argsort(imp)
    ax.barh([FEATS[i] for i in o], imp[o], color="#2f6feb")
    ax.set_title("What drives the edge/cloud decision")
    fig.savefig(FIG_DIR / "offload_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
