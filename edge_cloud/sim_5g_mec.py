"""Discrete-event simulation: cameras -> 5G uplink -> MEC edge server OR remote cloud.

Path of one camera frame:
  1. 5G uplink   : shared cell uplink (default 250 Mbps). Competes with the crowd's own
                   phone traffic (background load). With NETWORK SLICING the safety
                   cameras get a URLLC slice = strict priority on the uplink scheduler;
                   without slicing everything shares one FIFO queue (eMBB best effort).
  2. transport   : edge = gNB -> co-located MEC host (~1 ms)
                   cloud = gNB -> 5G core -> internet -> data centre (RTT 20-100 ms)
  3. processing  : edge = lightweight student model on CPU, few workers
                   cloud = big model on GPU, many workers
                   (times come from video_model/benchmark.py)
  4. alert back  : small packet over the downlink

Same ideas as Simu5G (gNB, UEs, MEC host, slices), written in Python so it runs in minutes.
"""
from __future__ import annotations

import heapq
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG_DIR, RESULTS, plot_style  # noqa: E402

DEADLINE_MS = 100.0


def _bench_defaults():
    f = RESULTS / "benchmark.json"
    d = dict(edge_ms=4.0, edge_big_ms=40.0, cloud_ms=1.5, frame_kb=25.0)
    if f.exists():
        b = json.loads(f.read_text())
        m = b["models"]
        d["edge_ms"] = m.get("student", {}).get("cpu_ms", d["edge_ms"])
        d["edge_big_ms"] = m.get("big", {}).get("cpu_ms", d["edge_big_ms"])
        # cloud runs the most accurate model available (v2 prediction U-Net if trained)
        d["cloud_ms"] = m.get("pred", m.get("big", {})).get("gpu_ms", d["cloud_ms"])
        d["frame_kb"] = b["frame_kb"].get("ped2_native_360x240") or d["frame_kb"]
    return d


BENCH = _bench_defaults()


@dataclass
class Config:
    path: str = "edge"            # "edge" or "cloud"
    n_cams: int = 10
    fps: float = 10.0
    frame_kb: float = BENCH["frame_kb"]
    ul_mbps: float = 250.0
    bg_load: float = 0.5          # fraction of uplink used by the public's phones
    bg_chunk_kb: float = 64.0
    slicing: bool = True
    edge_proc_ms: float = BENCH["edge_ms"]
    edge_workers: int = 2
    edge_other_load: float = 0.0  # share of the edge server busy with other MEC apps
    edge_oneway_ms: float = 1.0
    cloud_proc_ms: float = BENCH["cloud_ms"] + 2.0  # + (de)serialisation overhead
    cloud_workers: int = 32
    cloud_rtt_ms: float = 50.0
    dl_ms: float = 1.0
    sim_s: float = 15.0
    warmup_s: float = 2.0
    seed: int = 0


def simulate(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    ev: list = []
    seq = 0

    def push(t, kind, data=None):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, data))
        seq += 1

    ul_bps = cfg.ul_mbps * 1e6
    q_cam, q_bg = deque(), deque()
    ul_busy = False
    proc_q = deque()
    if cfg.path == "edge":
        workers = cfg.edge_workers
        proc_ms = cfg.edge_proc_ms / max(1e-3, 1.0 - cfg.edge_other_load)
        oneway = lambda: cfg.edge_oneway_ms + rng.exponential(0.3)  # noqa: E731
    else:
        workers = cfg.cloud_workers
        proc_ms = cfg.cloud_proc_ms
        oneway = lambda: cfg.cloud_rtt_ms / 2 * rng.lognormal(0, 0.15)  # noqa: E731
    free = workers
    delays, created = [], 0
    end = cfg.sim_s

    for c in range(cfg.n_cams):
        push(rng.uniform(0, 1 / cfg.fps), "frame", c)
    bg_rate = cfg.bg_load * ul_bps / (cfg.bg_chunk_kb * 8192)
    if bg_rate > 0:
        push(rng.exponential(1 / bg_rate), "bg")

    def start_ul(t):
        nonlocal ul_busy
        if cfg.slicing:
            q = q_cam if q_cam else q_bg
        else:
            q = q_cam if (q_cam and (not q_bg or q_cam[0][0] <= q_bg[0][0])) else q_bg
        if not q:
            ul_busy = False
            return
        pkt = q.popleft()
        ul_busy = True
        push(t + pkt[1] * 8192 / ul_bps, "ul_done", pkt)

    def start_proc(t):
        nonlocal free
        while free > 0 and proc_q:
            pkt = proc_q.popleft()
            free -= 1
            push(t + proc_ms / 1000 * rng.uniform(0.9, 1.1), "proc_done", pkt)

    while ev:
        t, _, kind, data = heapq.heappop(ev)
        if t > end:
            break
        if kind == "frame":
            pkt = (t, cfg.frame_kb, True)
            q_cam.append(pkt)
            if cfg.warmup_s <= t <= end - 1.0:
                created += 1
            push(t + 1 / cfg.fps, "frame", data)
            if not ul_busy:
                start_ul(t)
        elif kind == "bg":
            q_bg.append((t, cfg.bg_chunk_kb, False))
            push(t + rng.exponential(1 / bg_rate), "bg")
            if not ul_busy:
                start_ul(t)
        elif kind == "ul_done":
            if data[2]:
                push(t + oneway() / 1000, "arrive", data)
            start_ul(t)
        elif kind == "arrive":
            proc_q.append(data)
            start_proc(t)
        elif kind == "proc_done":
            free += 1
            done = t + (oneway() + cfg.dl_ms) / 1000
            if cfg.warmup_s <= data[0] <= end - 1.0:
                delays.append((done - data[0]) * 1000)
            start_proc(t)

    d = np.array(delays) if delays else np.array([np.inf])
    lost = max(0, created - len(delays))  # frames stuck in queues = overload
    pad = np.concatenate([d, np.full(lost, (end - cfg.warmup_s) * 1000 / 2)])
    return dict(mean_ms=float(pad.mean()), p95_ms=float(np.percentile(pad, 95)),
                miss_rate=float((pad > DEADLINE_MS).mean()), completed=len(delays), created=created)


# ---------------------------------------------------------------- experiments
def exp_cameras():
    ns = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120]
    setups = {
        "Edge - lightweight model": Config(path="edge"),
        "Edge - big model (CPU)": Config(path="edge", edge_proc_ms=BENCH["edge_big_ms"]),
        "Cloud - accurate model (GPU)": Config(path="cloud"),
    }
    out = {}
    for name, base in setups.items():
        out[name] = [simulate(replace(base, n_cams=n, seed=n)) for n in ns]
    return ns, out


def exp_slicing():
    loads = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    out = {}
    for sl in (True, False):
        out["URLLC slice (priority)" if sl else "No slicing (shared)"] = [
            simulate(Config(path="edge", n_cams=20, bg_load=b, slicing=sl, seed=int(b * 100))) for b in loads]
    return loads, out


def main():
    plt = plot_style()
    print("Processing times used (ms):", {k: round(v, 2) for k, v in BENCH.items()})

    ns, res = exp_cameras()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, rows in res.items():
        axes[0].plot(ns, [r["mean_ms"] for r in rows], "o-", label=name, ms=4)
        axes[1].plot(ns, [r["miss_rate"] * 100 for r in rows], "o-", label=name, ms=4)
    axes[0].set_yscale("log")
    axes[0].axhline(DEADLINE_MS, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("number of cameras")
    axes[0].set_ylabel("mean end-to-end delay (ms, log)")
    axes[0].set_title("Alert delay vs number of cameras")
    axes[1].set_xlabel("number of cameras")
    axes[1].set_ylabel(f"% alerts later than {DEADLINE_MS:.0f} ms")
    axes[1].set_title("Deadline misses")
    axes[0].legend(fontsize=8)
    fig.savefig(FIG_DIR / "edge_vs_cloud_cameras.png")
    plt.close(fig)

    loads, sres = exp_slicing()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for name, rows in sres.items():
        ax.plot([l * 100 for l in loads], [r["p95_ms"] for r in rows], "o-", label=name, ms=4)
    ax.set_yscale("log")
    ax.axhline(DEADLINE_MS, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("uplink used by the public's phones (%)")
    ax.set_ylabel("95th percentile delay (ms, log)")
    ax.set_title("Effect of 5G network slicing (20 cameras, edge)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "slicing_effect.png")
    plt.close(fig)

    summary = dict(bench=BENCH, cameras=dict(n=ns, results=res), slicing=dict(bg_load=loads, results=sres))
    (RESULTS / "edge_cloud_results.json").write_text(json.dumps(summary, indent=2))
    for name, rows in res.items():
        print(f"  {name:26s} mean delay @1/10/40/80 cams: " +
              " / ".join(f"{rows[ns.index(k)]['mean_ms']:.1f}" for k in (1, 10, 40, 80)) + " ms")
    for name, rows in sres.items():
        print(f"  {name:26s} p95 @ 50%/90% public load: {rows[loads.index(0.5)]['p95_ms']:.1f} / "
              f"{rows[loads.index(0.9)]['p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()
