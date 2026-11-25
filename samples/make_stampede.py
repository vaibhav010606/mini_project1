"""Generate a synthetic CCTV-style stampede video for testing the motion detector.

    python samples/make_stampede.py      ->  samples/synthetic_stampede_cctv.mp4

A 40 m x 24 m plaza seen from an elevated camera. About 260 people: most stand packed in
front of a stage, others walk around. At 10 s panic starts at the stage: people nearby run
first, the panic spreads outward, everyone pushes toward the exits and a few people fall.
Crowd motion uses a simple social-force model (goal-seeking + pushing apart). The video is
clearly labelled as a synthetic simulation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "synthetic_stampede_cctv.mp4"
FPS, SECONDS, T_PANIC = 25, 26, 10.0
IMG_W, IMG_H = 960, 540
PLAZA_W, PLAZA_D = 40.0, 24.0
N = 260
TRIGGER = np.array([22.0, 18.5])
EXITS = np.array([[-3.0, 13.0], [43.0, 13.0], [20.0, -6.0]])  # left, right, towards camera

# ---------------------------------------------------------------- camera (perspective)
CAM = np.array([20.0, -9.0, 11.0])
PITCH = np.radians(32)
FWD = np.array([0.0, np.cos(PITCH), -np.sin(PITCH)])
UP = np.array([0.0, np.sin(PITCH), np.cos(PITCH)])
FOCAL, CX, CY = 700.0, IMG_W / 2, IMG_H / 2


def project(x, y, z=0.0):
    r = np.stack([np.asarray(x) - CAM[0], np.asarray(y) - CAM[1], np.full_like(np.asarray(x, float), z) - CAM[2]], -1)
    depth = r @ FWD
    u = CX + FOCAL * r[..., 0] / depth
    v = CY - FOCAL * (r @ UP) / depth
    return u, v, depth


def P(x, y, z=0.0):
    u, v, _ = project(np.array(float(x)), np.array(float(y)), z)
    return int(round(float(u))), int(round(float(v)))


# ---------------------------------------------------------------- static background
def background():
    img = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    _, v_far, _ = project(np.array(20.0), np.array(PLAZA_D))
    v_far = int(v_far)
    img[:v_far] = (120, 128, 138)  # building facade
    for wx in range(0, IMG_W, 70):
        for wy in range(12, v_far - 30, 45):
            cv2.rectangle(img, (wx + 12, wy), (wx + 50, wy + 26), (150, 160, 172), -1)
            cv2.rectangle(img, (wx + 12, wy), (wx + 50, wy + 26), (95, 100, 110), 1)
    plaza = np.array([P(-20, 0), P(60, 0), P(60, PLAZA_D), P(-20, PLAZA_D)], np.int32)
    cv2.fillPoly(img, [plaza], (158, 168, 176))
    for x in np.arange(-20, 61, 2.0):
        cv2.line(img, P(x, 0), P(x, PLAZA_D), (145, 154, 162), 1, cv2.LINE_AA)
    for y in np.arange(0, PLAZA_D + 0.1, 2.0):
        cv2.line(img, P(-20, y), P(60, y), (145, 154, 162), 1, cv2.LINE_AA)
    # stage at the back
    top = np.array([P(13, 21.5, 1.1), P(31, 21.5, 1.1), P(31, 24, 1.1), P(13, 24, 1.1)], np.int32)
    front = np.array([P(13, 21.5), P(31, 21.5), P(31, 21.5, 1.1), P(13, 21.5, 1.1)], np.int32)
    cv2.fillPoly(img, [front], (55, 45, 40))
    cv2.fillPoly(img, [top], (80, 70, 62))
    for x in (14, 30):
        cv2.line(img, P(x, 23.5, 1.1), P(x, 23.5, 4.5), (40, 40, 40), 3)
    cv2.line(img, P(14, 23.5, 4.5), P(30, 23.5, 4.5), (40, 40, 40), 3)
    return img


# ---------------------------------------------------------------- crowd simulation
def simulate(rng):
    dt = 1.0 / FPS
    crowd = rng.random(N) < 0.62
    pos = np.where(crowd[:, None],
                   np.c_[rng.normal(22, 5.0, N), rng.normal(17.5, 2.2, N)],
                   np.c_[rng.uniform(4, 36, N), rng.uniform(1, 20, N)])
    pos[:, 0] = np.clip(pos[:, 0], 2, 38)
    pos[:, 1] = np.clip(pos[:, 1], 0.5, 21)
    goal = pos.copy()
    goal[~crowd] = np.c_[rng.uniform(4, 36, (~crowd).sum()), rng.uniform(1, 20, (~crowd).sum())]
    walk = rng.uniform(1.0, 1.5, N)
    run = rng.uniform(3.5, 6.0, N)
    vel = np.zeros((N, 2))
    away = pos - TRIGGER
    dist = np.linalg.norm(away, axis=1)
    panic_at = T_PANIC + np.maximum(0, dist - 7.0) / 5.0 + rng.uniform(0, 0.4, N)
    fallen = np.zeros(N, bool)
    fall_dir = np.zeros(N)
    gone = np.zeros(N, bool)
    frames = []
    for k in range(int(SECONDS * FPS)):
        t = k * dt
        panicking = (t >= panic_at) & ~fallen & ~gone
        desired = np.zeros((N, 2))
        # calm: crowd members sway in place, walkers wander
        to_goal = goal - pos
        d = np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9
        calm = ~panicking & ~fallen & ~gone
        desired[calm] = (to_goal / d * np.minimum(walk[:, None], d))[calm]
        swaying = calm & crowd
        desired[swaying] = (to_goal / d * np.minimum(0.25, d))[swaying] + rng.normal(0, 0.15, (swaying.sum(), 2))
        arrived = calm & ~crowd & (d[:, 0] < 0.5)
        goal[arrived] = np.c_[rng.uniform(4, 36, arrived.sum()), rng.uniform(1, 20, arrived.sum())]
        # panic: run away from the trigger towards the best exit
        if panicking.any():
            a = pos[panicking] - TRIGGER
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
            ex = EXITS[None, :, :] - pos[panicking][:, None, :]
            exn = ex / (np.linalg.norm(ex, axis=2, keepdims=True) + 1e-9)
            best = np.argmax((exn * a[:, None, :]).sum(2), axis=1)
            dirv = 0.45 * a + 0.55 * exn[np.arange(len(best)), best]
            dirv /= np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-9
            desired[panicking] = dirv * run[panicking, None]
        # social force: push apart when closer than ~0.6 m (stronger in panic)
        active = ~gone
        diff = pos[:, None, :] - pos[None, :, :]
        dd = np.linalg.norm(diff, axis=2) + np.eye(N) * 99
        overlap = np.clip(0.62 - dd, 0, None) * active[None, :] * active[:, None]
        push = (diff / dd[..., None] * overlap[..., None]).sum(1) * np.where(panicking, 9.0, 4.0)[:, None]
        tau = np.where(panicking, 0.25, 0.6)[:, None]
        vel = vel + (desired - vel) * dt / tau + push * dt
        vel[fallen | gone] = 0
        # some people fall in the crush during the first seconds of panic
        if T_PANIC + 0.6 < t < T_PANIC + 4.0:
            crush = panicking & (overlap.sum(1) > 0.5) & (rng.random(N) < 0.012)
            if fallen.sum() + crush.sum() <= 8:
                fallen |= crush
                fall_dir[crush] = np.arctan2(vel[crush, 1], vel[crush, 0])
        pos = pos + vel * dt
        pos[:, 1] = np.clip(pos[:, 1], -8, 21.2)
        gone |= (pos[:, 0] < -2.5) | (pos[:, 0] > 42.5) | (pos[:, 1] < -5)
        frames.append((pos.copy(), vel.copy(), panicking.copy(), fallen.copy(), fall_dir.copy(), gone.copy()))
    return frames


# ---------------------------------------------------------------- drawing
def draw_person(img, x, y, vx, vy, phase, shirt, trousers, skin, running):
    u, v, depth = project(np.array(x), np.array(y))
    u, v, s = float(u), float(v), FOCAL / float(depth)  # s = pixels per metre here
    if not (-40 < u < IMG_W + 40 and 0 < v < IMG_H + 80):
        return
    # direction of motion on screen
    u2, v2, _ = project(np.array(x + vx * 0.3), np.array(y + vy * 0.3))
    mx, my = float(u2) - u, float(v2) - v
    n = np.hypot(mx, my) + 1e-6
    mx, my = mx / n, my / n
    speed = np.hypot(vx, vy)
    stride = (0.55 if running else 0.32) * min(1.0, speed / 1.2)
    lean = 0.25 if running else 0.04
    sw = np.sin(phase)
    t = lambda k: max(1, int(round(k * s)))  # noqa: E731
    cv2.ellipse(img, (int(u + 0.08 * s), int(v)), (t(0.32), t(0.1)), 0, 0, 360, (128, 136, 143), -1, cv2.LINE_AA)
    hip = (u, v - 0.9 * s)
    sh = (u + mx * lean * s, v - 1.42 * s)
    for sgn in (1, -1):
        fx = u + sgn * sw * stride * s * mx * 0.9
        fy = v + sgn * sw * stride * s * my * 0.3
        cv2.line(img, (int(hip[0] + sgn * 0.06 * s), int(hip[1])), (int(fx), int(fy)), trousers, t(0.13), cv2.LINE_AA)
    cv2.line(img, (int(hip[0]), int(hip[1])), (int(sh[0]), int(sh[1])), shirt, t(0.34), cv2.LINE_AA)
    arm = (0.5 if running else 0.25) * s
    for sgn in (1, -1):
        hx = sh[0] - sgn * sw * arm * mx + sgn * 0.17 * s
        hy = sh[1] + 0.55 * s - (0.25 * s if running else 0)
        cv2.line(img, (int(sh[0] + sgn * 0.15 * s), int(sh[1] + 0.05 * s)), (int(hx), int(hy)), shirt, t(0.09), cv2.LINE_AA)
    head = (int(sh[0] + mx * lean * 0.4 * s), int(sh[1] - 0.2 * s))
    cv2.circle(img, head, t(0.12), skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] - t(0.03)), (t(0.12), t(0.08)), 0, 180, 360, (35, 30, 30), -1, cv2.LINE_AA)


def draw_fallen(img, x, y, ang, shirt, trousers, skin):
    hx, hy = x + 0.85 * np.cos(ang), y + 0.85 * np.sin(ang)
    a = P(x, y, 0.12)
    b = P((x + hx) / 2, (y + hy) / 2, 0.12)
    c = P(hx, hy, 0.15)
    _, _, depth = project(np.array(x), np.array(y))
    s = FOCAL / float(depth)
    cv2.line(img, a, b, trousers, max(1, int(0.18 * s)), cv2.LINE_AA)
    cv2.line(img, b, c, shirt, max(1, int(0.3 * s)), cv2.LINE_AA)
    cv2.circle(img, c, max(1, int(0.12 * s)), skin, -1, cv2.LINE_AA)


def main():
    rng = np.random.default_rng(11)
    frames = simulate(rng)
    bg = background()
    shirts = rng.integers(30, 230, (N, 3))
    trousers = rng.choice([[40, 40, 45], [90, 60, 30], [60, 60, 70], [110, 100, 90], [30, 30, 80]], N)
    skins = rng.choice([[90, 130, 180], [70, 105, 150], [120, 160, 205], [50, 75, 110]], N)
    phase = rng.uniform(0, 2 * np.pi, N)
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
    noise_rng = np.random.default_rng(1)
    for k, (pos, vel, panicking, fallen, fdir, gone) in enumerate(frames):
        img = bg.copy()
        speed = np.hypot(vel[:, 0], vel[:, 1])
        phase += speed / FPS * (3.2 + 1.5 * panicking)
        order = np.argsort(-pos[:, 1])  # far people first
        for i in order:
            if gone[i]:
                continue
            col = tuple(int(c) for c in shirts[i])
            tr = tuple(int(c) for c in trousers[i])
            sk = tuple(int(c) for c in skins[i])
            if fallen[i]:
                draw_fallen(img, pos[i, 0], pos[i, 1], fdir[i], col, tr, sk)
            else:
                draw_person(img, pos[i, 0], pos[i, 1], vel[i, 0], vel[i, 1], phase[i], col, tr, sk, speed[i] > 2.5)
        t = k / FPS
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.clip(img + noise_rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
        cv2.rectangle(img, (0, 0), (IMG_W, 30), (0, 0, 0), -1)
        cv2.putText(img, f"CAM 03  CENTRAL PLAZA   2026-11-25  18:42:{int(t):02d}.{int((t % 1) * 100):02d}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(img, "SYNTHETIC SIMULATION", (IMG_W - 215, IMG_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    print(f"Saved {OUT}  ({len(frames)} frames, {SECONDS} s, panic at {T_PANIC:.0f} s, "
          f"{int(frames[-1][3].sum())} people fell, {int(frames[-1][5].sum())} left the plaza)")


if __name__ == "__main__":
    main()
