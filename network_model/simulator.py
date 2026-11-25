"""5G crowd simulator.

People carrying phones move across a 200 m x 200 m venue covered by a 3x3 grid of
5G small cells (gNBs, 3.5 GHz). Each second we record what the network side (gNB /
MEC host) can observe, using aggregate counts only, never individual tracks:

    total_phones        phones connected in the venue
    max_cell_phones     phones on the busiest cell
    max_zone_density    people per m^2 in the densest 10 m x 10 m zone, estimated from
                        network positioning (noisy, ~5 m error per axis)
    density_growth_10s  change of max_zone_density over the last 10 s
    mean_speed          mean phone speed from positioning (5 s displacement)
    frac_fast           fraction of phones moving faster than 3 m/s
    handover_rate       handovers per phone per second
    max_cell_load       active users on busiest cell / cell capacity
    mean_sinr_db        mean downlink SINR of connected phones

Three scenarios are simulated:
    normal     people wander; small, safe attractions (stalls, stage)
    gathering  a large share of the crowd packs into one hotspot -> dangerous density
    stampede   gathering, then sudden panic: people near the hotspot flee at 3-6 m/s

Ground-truth label per second: 0 normal, 1 dense (> 2 people/m^2 somewhere),
2 stampede (panic in progress).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR  # noqa: E402

VENUE = 200.0
ZONE = 10.0
NZ = int(VENUE / ZONE)
ZONE_AREA = ZONE * ZONE
GNB = np.array([[x, y] for x in (35.0, 100.0, 165.0) for y in (35.0, 100.0, 165.0)])
FC_GHZ = 3.5
TX_DBM = 30.0
NOISE_DBM = -174 + 10 * np.log10(20e6) + 9  # 20 MHz bandwidth, 9 dB noise figure
SHADOW_STD = 4.0
FADING_STD = 2.0
HYST_DB = 3.0
POS_ERR = 5.0
CELL_CAPACITY = 150  # simultaneously active users a small cell serves comfortably
PHONES_PER_PERSON = 1.05  # 5% none, 85% one, 10% two
DENSE_THR = 2.0
WINDOW = 30

LABELS = ["normal", "dense", "stampede"]
SCENARIOS = ["normal", "gathering", "stampede"]
FEATURES = [
    "total_phones", "max_cell_phones", "max_zone_density", "density_growth_10s",
    "mean_speed", "frac_fast", "handover_rate", "max_cell_load", "mean_sinr_db",
]


def pathloss_db(d: np.ndarray) -> np.ndarray:
    """3GPP UMi street-canyon (LOS) style path loss."""
    d = np.maximum(d, 1.0)
    return 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(FC_GHZ)


def zone_density(xy: np.ndarray, weight: float = 1.0) -> np.ndarray:
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=NZ, range=[[0, VENUE], [0, VENUE]])
    return h / ZONE_AREA / weight


def run_episode(rng: np.random.Generator, scenario: str, T: int = 180,
                n_people: int | None = None, snapshot_times=(), no_phone_p: float = 0.05, track: bool = False):
    """no_phone_p: share of people whose phone the operator cannot see (no phone, phone off, other operator).
    track: also return every person's position each second in info["positions"] (for rendering CCTV video)."""
    n = int(n_people or rng.integers(800, 2600))
    pos = rng.uniform(5, VENUE - 5, (n, 2))
    wander_target = rng.uniform(5, VENUE - 5, (n, 2))
    base_speed = rng.uniform(0.6, 1.5, n)
    hotspot = rng.uniform(60, 140, 2)

    if scenario == "normal":
        # busy but usually safe: stalls, stage; some crowds come close to the danger limit
        attract_frac, sigma = rng.uniform(0.0, 0.45), rng.uniform(7, 22)
    else:
        attract_frac, sigma = rng.uniform(0.4, 0.7), rng.uniform(5, 9)
    attracted = rng.random(n) < attract_frac
    spot = hotspot + rng.normal(0, sigma, (n, 2))
    t_start = rng.uniform(0, 70, n)

    # harmless rush (e.g. people hurrying to a train): fast movement that is NOT a stampede
    rush = scenario == "normal" and rng.random() < 0.4
    rushers = rng.random(n) < rng.uniform(0.08, 0.3) if rush else np.zeros(n, dtype=bool)
    rush_t0, rush_len = int(rng.integers(40, 140)), int(rng.integers(15, 40))
    rush_exit = np.array([rng.choice([2.0, VENUE - 2.0]), rng.uniform(20, VENUE - 20)])
    rush_speed = rng.uniform(2.0, 3.5, n)

    panic_t0 = int(rng.integers(100, 140)) if scenario == "stampede" else None
    panic_len = int(rng.integers(25, 41))
    panic_radius = rng.uniform(25, 45)
    flee_speed = rng.uniform(2.2, 5.0, n)
    flee_dir = None
    panic_delay = None

    q = float(no_phone_p)  # default 0.05 gives exactly [0.05, 0.85, 0.10]
    phones = rng.choice([0, 1, 2], size=n, p=[q, 0.85 * (1 - q) / 0.95, 0.10 * (1 - q) / 0.95])
    track_pos = np.zeros((T, n, 2), dtype=np.float32) if track else None
    owner = np.repeat(np.arange(n), phones)
    m = len(owner)
    shadow = rng.normal(0, SHADOW_STD, (m, len(GNB)))
    active = rng.random(m) < 0.3
    serving = None
    est_hist: list[np.ndarray] = []
    dens_hist: list[float] = []
    noise_lin = 10 ** (NOISE_DBM / 10)

    feats = np.zeros((T, len(FEATURES)), dtype=np.float32)
    labels = np.zeros(T, dtype=np.int64)
    snaps = {}

    for t in range(T):
        # ---------------- crowd movement (ground truth) ----------------
        true_dens = zone_density(pos)
        zi = np.clip((pos // ZONE).astype(int), 0, NZ - 1)
        local = true_dens[zi[:, 0], zi[:, 1]]
        slow = np.clip(1.0 - local / 5.0, 0.15, 1.0)

        goal = wander_target.copy()
        going = attracted & (t >= t_start)
        goal[going] = spot[going]
        vec = goal - pos
        dist = np.linalg.norm(vec, axis=1) + 1e-9
        step = np.minimum(base_speed * slow, dist)
        move = vec / dist[:, None] * step[:, None]

        reached = (dist < 1.0) & ~going
        wander_target[reached] = rng.uniform(5, VENUE - 5, (reached.sum(), 2))

        if rush and rush_t0 <= t < rush_t0 + rush_len:
            rv = rush_exit - pos[rushers]
            rd = np.linalg.norm(rv, axis=1, keepdims=True) + 1e-9
            rs = np.minimum(rush_speed[rushers] * slow[rushers], rd[:, 0])
            move[rushers] = rv / rd * rs[:, None]
        elif rush and t == rush_t0 + rush_len:
            wander_target[rushers] = rng.uniform(5, VENUE - 5, (rushers.sum(), 2))

        panicking = np.zeros(n, dtype=bool)
        in_panic = panic_t0 is not None and panic_t0 <= t < panic_t0 + panic_len
        if panic_t0 is not None and t == panic_t0 + panic_len:
            attracted[:] = False  # crowd has dispersed; everyone wanders again
            wander_target[:] = rng.uniform(5, VENUE - 5, (n, 2))
        if in_panic:
            if flee_dir is None:
                away = pos - hotspot
                r = np.linalg.norm(away, axis=1) + 1e-9
                flee_dir = away / r[:, None]
                panic_delay = np.where(r < panic_radius, 0, (r - panic_radius) / 8.0)  # panic spreads outward
                panic_delay[r > panic_radius + 50] = 1e9  # far away people do not react
            panicking = (t - panic_t0) >= panic_delay
            fs = flee_speed * np.clip(1.0 - local / 8.0, 0.5, 1.0)
            jitter = rng.normal(0, 0.15, (n, 2))
            move[panicking] = (flee_dir[panicking] + jitter[panicking]) * fs[panicking, None]

        pos = np.clip(pos + move + rng.normal(0, 0.1, (n, 2)), 0.5, VENUE - 0.5)
        if track:
            track_pos[t] = pos

        if in_panic:
            labels[t] = 2
        elif zone_density(pos).max() >= DENSE_THR:
            labels[t] = 1

        # ---------------- what the 5G network observes ----------------
        ppos = pos[owner]
        d = np.linalg.norm(ppos[:, None, :] - GNB[None, :, :], axis=2)
        rsrp = TX_DBM - pathloss_db(d) - shadow + rng.normal(0, FADING_STD, shadow.shape)
        best = rsrp.argmax(1)
        if serving is None:
            serving = best.copy()
            handovers = 0
        else:
            cur = rsrp[np.arange(m), serving]
            ho = rsrp[np.arange(m), best] > cur + HYST_DB
            handovers = int(ho.sum())
            serving[ho] = best[ho]

        lin = 10 ** (rsrp / 10)
        sig = lin[np.arange(m), serving]
        sinr = sig / (lin.sum(1) - sig + noise_lin)
        cell_phones = np.bincount(serving, minlength=len(GNB))
        cell_active = np.bincount(serving[active], minlength=len(GNB))

        est = ppos + rng.normal(0, POS_ERR, ppos.shape)
        est_hist.append(est)
        if len(est_hist) > 6:
            est_hist.pop(0)
        if len(est_hist) >= 6:
            spd = np.linalg.norm(est_hist[-1] - est_hist[0], axis=1) / 5.0
        else:
            spd = np.zeros(m)
        est_dens = zone_density(est, PHONES_PER_PERSON)
        mdens = float(est_dens.max())
        dens_hist.append(mdens)
        growth = mdens - dens_hist[-11] if len(dens_hist) > 10 else 0.0

        feats[t] = [
            m / 1000.0,
            cell_phones.max() / 1000.0,
            mdens,
            growth,
            spd.mean(),
            (spd > 3.0).mean(),
            handovers / max(m, 1),
            cell_active.max() / CELL_CAPACITY,
            10 * np.log10(sinr.mean()),
        ]
        if t in snapshot_times:
            snaps[t] = (zone_density(pos), est_dens, int(labels[t]))

    info = dict(n_people=n, n_phones=m, hotspot=hotspot, panic_t0=panic_t0)
    if track:
        info["positions"] = track_pos
    return feats, labels, snaps, info


def build_dataset(episodes_per_scenario: int = 150, T: int = 180, stride: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    X, y, ep, sc = [], [], [], []
    e = 0
    t0 = time.time()
    for s_idx, scen in enumerate(SCENARIOS):
        for _ in range(episodes_per_scenario):
            feats, labels, _, _ = run_episode(rng, scen, T)
            for end in range(WINDOW, T + 1, stride):
                X.append(feats[end - WINDOW:end])
                y.append(labels[end - 1])
                ep.append(e)
                sc.append(s_idx)
            e += 1
        print(f"  {scen}: {episodes_per_scenario} episodes done ({time.time() - t0:.0f}s)")
    return (np.stack(X), np.array(y), np.array(ep), np.array(sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=150, help="episodes per scenario")
    ap.add_argument("--T", type=int, default=180, help="seconds per episode")
    args = ap.parse_args()
    print(f"Simulating {args.episodes} episodes x {len(SCENARIOS)} scenarios ...")
    X, y, ep, sc = build_dataset(args.episodes, args.T)
    out = DATA_DIR / "network_dataset.npz"
    np.savez_compressed(out, X=X, y=y, episode=ep, scenario=sc, features=np.array(FEATURES))
    counts = {LABELS[k]: int((y == k).sum()) for k in range(3)}
    print(f"Saved {out}  windows={len(y)}  shape={X.shape}  labels={counts}")


if __name__ == "__main__":
    main()
