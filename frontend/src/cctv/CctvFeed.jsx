import { useEffect, useRef } from "react";
import { FLOOR_NAMES } from "../mall/cameras.js";

// Fake live CCTV feed: a small crowd simulation drawn in perspective on a canvas, with CCTV styling.
// Everything is generated in the browser; nothing is real footage.

const W = 480;
const H = 270;
const HZ = 96; // horizon line (far wall meets floor)
const FPS = 12;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// floor coordinates: u = 0..1 left to right, v = 0 (far wall) .. 1 (near camera)
function proj(u, v) {
  const y = HZ + v * (H - HZ);
  const spread = 0.52 + 0.58 * v;
  return [W / 2 + (u - 0.5) * W * 1.1 * spread, y, 0.32 + 0.78 * v];
}

const CLOTHES = ["#3d4a5c", "#7a3b3b", "#2f5d50", "#8a7a5a", "#4b4f8a", "#6b6b6b", "#a0522d", "#2b3a55", "#8c5a7a", "#557a95", "#c2a060", "#5e7d3a"];
const SKIN = ["#e0b993", "#c98f65", "#8d5a3b", "#f1d0b0", "#a8714f"];

// ---------------------------------------------------------------- background (drawn once per camera)
function drawBackground(ctx, cam, r) {
  // ceiling and far wall
  ctx.fillStyle = "#c9cfd6";
  ctx.fillRect(0, 0, W, HZ);
  ctx.fillStyle = "#b3bac3";
  ctx.fillRect(0, 0, W, 16);
  for (let x = 20; x < W; x += 70) {
    ctx.fillStyle = "#eef3f7";
    ctx.fillRect(x, 5, 36, 5);
  }
  // floor
  const [lx0, ly0] = proj(0, 0);
  const [rx0] = proj(1, 0);
  const [lx1, ly1] = proj(0, 1.05);
  const [rx1] = proj(1, 1.05);
  ctx.fillStyle = cam.floor === 0 ? "#e3ded4" : "#dcdfe3";
  ctx.beginPath();
  ctx.moveTo(lx0, ly0); ctx.lineTo(rx0, ly0); ctx.lineTo(rx1, ly1); ctx.lineTo(lx1, ly1);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = "#c8ccd2";
  ctx.fillRect(0, HZ, lx0 + 1, H);
  ctx.fillRect(rx0 - 1, HZ, W, H);
  // side walls fade into floor
  ctx.fillStyle = "#d3d8de";
  ctx.beginPath(); ctx.moveTo(0, HZ); ctx.lineTo(lx0, ly0); ctx.lineTo(lx1, ly1); ctx.lineTo(0, H); ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.moveTo(W, HZ); ctx.lineTo(rx0, ly0); ctx.lineTo(rx1, ly1); ctx.lineTo(W, H); ctx.closePath(); ctx.fill();
  // tile lines
  ctx.strokeStyle = "rgba(120,120,120,0.18)";
  ctx.lineWidth = 1;
  for (let u = 0; u <= 1.0001; u += 0.1) {
    const [a, b] = proj(u, 0); const [c, d] = proj(u, 1.05);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
  }
  for (let v = 0; v <= 1.05; v += 0.08) {
    const [a, b] = proj(0, v); const [c] = proj(1, v);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, b); ctx.stroke();
  }

  const shopfront = (x, w, name, color) => {
    ctx.fillStyle = "#8f98a3";
    ctx.fillRect(x, 20, w, HZ - 20);
    ctx.fillStyle = color;
    ctx.fillRect(x + 4, 22, w - 8, 15);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.fillText(name.toUpperCase(), x + w / 2, 33);
    ctx.fillStyle = "#a9c4d8";
    ctx.fillRect(x + 6, 41, w - 12, HZ - 43);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillRect(x + 10, 44, 6, HZ - 49);
    for (let k = 0; k < 3; k++) {
      ctx.fillStyle = "rgba(60,70,85,0.35)";
      ctx.fillRect(x + 18 + k * ((w - 36) / 3), 58, 8, HZ - 62);
    }
  };

  const signColors = ["#c0392b", "#1f6feb", "#2f855a", "#8e44ad", "#d35400", "#16a085"];
  switch (cam.scene) {
    case "corridor": {
      const shops = cam.shops || ["Shop", "Shop", "Shop"];
      const w = W / shops.length;
      shops.forEach((s, i) => shopfront(i * w + 3, w - 6, s, signColors[(i + cam.floor * 2) % signColors.length]));
      break;
    }
    case "storefront":
    case "entrance": {
      shopfront(0, 120, cam.scene === "entrance" ? "Pharmacy" : "Shop", "#5b6b7d");
      shopfront(W - 120, 120, cam.scene === "entrance" ? "Bank" : "Shop", "#5b6b7d");
      ctx.fillStyle = "#6f7a86";
      ctx.fillRect(124, 18, W - 248, HZ - 18);
      ctx.fillStyle = cam.scene === "entrance" ? "#12263f" : "#7b4bb7";
      ctx.fillRect(140, 22, W - 280, 18);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.fillText(cam.sign || "ENTRANCE", W / 2, 35);
      // glass doors with daylight behind them
      ctx.fillStyle = cam.scene === "entrance" ? "#eef6fb" : "#d9c9ef";
      ctx.fillRect(150, 44, W - 300, HZ - 44);
      ctx.strokeStyle = "#5c6670";
      ctx.lineWidth = 2;
      for (let k = 0; k <= 4; k++) {
        const x = 150 + k * ((W - 300) / 4);
        ctx.beginPath(); ctx.moveTo(x, 44); ctx.lineTo(x, HZ); ctx.stroke();
      }
      break;
    }
    case "foodcourt": {
      const stalls = ["PIZZA", "BIRYANI", "WRAPS", "JUICE", "NOODLES"];
      const w = W / stalls.length;
      stalls.forEach((s, i) => {
        ctx.fillStyle = "#7d6a5a";
        ctx.fillRect(i * w + 2, 20, w - 4, HZ - 20);
        ctx.fillStyle = signColors[i];
        ctx.fillRect(i * w + 6, 24, w - 12, 14);
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px Segoe UI, Arial";
        ctx.textAlign = "center";
        ctx.fillText(s, i * w + w / 2, 35);
        ctx.fillStyle = "#e8dccb";
        ctx.fillRect(i * w + 6, HZ - 22, w - 12, 20);
      });
      break;
    }
    case "cinema": {
      const posters = ["#2c3e50", "#8e2c2c", "#1f4e6e", "#5b2c6f", "#7d5a1e"];
      ctx.fillStyle = "#2b2f36";
      ctx.fillRect(0, 16, W, HZ - 16);
      posters.forEach((c, i) => {
        ctx.fillStyle = c;
        ctx.fillRect(20 + i * 92, 26, 60, 50);
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.fillRect(28 + i * 92, 64, 44, 4);
      });
      ctx.fillStyle = "#e6b84a";
      ctx.font = "bold 11px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.fillText("NOW SHOWING", W / 2, 23);
      break;
    }
    default: {
      // atrium / escalator: generic shop row on the far wall
      const names = cam.floor === 2 ? ["Kids Play", "VR Arcade", "Lounge", "Book Café"] : ["Jewellery", "Opticals", "Toys", "Gadgets"];
      const w = W / names.length;
      names.forEach((s, i) => shopfront(i * w + 3, w - 6, s, signColors[(i + 1) % signColors.length]));
    }
  }

  // floor furniture
  if (cam.scene === "atrium" && cam.ground) {
    const [x, y, s] = proj(0.5, 0.5);
    ctx.fillStyle = "#f7f7f7";
    ctx.beginPath(); ctx.ellipse(x, y, 46 * s, 14 * s, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#c7cdd4";
    ctx.fillRect(x - 46 * s, y - 16 * s, 92 * s, 16 * s);
    ctx.fillStyle = "#1f6feb";
    ctx.fillRect(x - 20 * s, y - 28 * s, 40 * s, 10 * s);
  }
  if (cam.scene === "atrium" && !cam.ground) {
    const corners = [proj(0.28, 0.3), proj(0.72, 0.3), proj(0.78, 0.62), proj(0.22, 0.62)];
    ctx.fillStyle = "#7f8893";
    ctx.beginPath(); corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.12)";
    ctx.fillRect(corners[0][0] + 10, corners[0][1] + 6, 60, 20);
    ctx.strokeStyle = "rgba(170,210,235,0.9)";
    ctx.lineWidth = 3;
    ctx.beginPath(); corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y - 8) : ctx.moveTo(x, y - 8))); ctx.closePath(); ctx.stroke();
  }
  if (cam.scene === "escalator") {
    const a = proj(0.12, 1.05), b = proj(0.3, 1.05), c = proj(0.5, 0.05), d = proj(0.42, 0.05);
    ctx.fillStyle = "#6b7480";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    for (let t = 0; t < 1; t += 0.06) {
      const p = lerp2(a, d, t), q = lerp2(b, c, t);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(170,210,235,0.9)";
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a[0], a[1] - 14); ctx.lineTo(d[0], d[1] - 6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(b[0], b[1] - 14); ctx.lineTo(c[0], c[1] - 6); ctx.stroke();
  }
  if (cam.scene === "foodcourt") {
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) {
      const [x, y, s] = proj(0.14 + i * 0.18, 0.28 + j * 0.27);
      ctx.fillStyle = "rgba(0,0,0,0.12)";
      ctx.beginPath(); ctx.ellipse(x, y + 3 * s, 22 * s, 7 * s, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#f4f1ea";
      ctx.beginPath(); ctx.ellipse(x, y - 12 * s, 20 * s, 7 * s, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#8b8f96";
      ctx.fillRect(x - 1.5 * s, y - 12 * s, 3 * s, 14 * s);
    }
  }
  if (cam.scene === "cinema") {
    const [x0, y0] = proj(0.2, 0.05);
    const [x1] = proj(0.8, 0.05);
    ctx.fillStyle = "#3a3f47";
    ctx.fillRect(x0, y0 - 16, x1 - x0, 18);
    ctx.fillStyle = "#e6b84a";
    ctx.fillRect(x0, y0 - 18, x1 - x0, 3);
    for (const v of [0.3, 0.5, 0.7]) {
      ctx.strokeStyle = "#b22222";
      ctx.lineWidth = 2;
      const a = proj(0.3, v), b = proj(0.7, v);
      ctx.beginPath(); ctx.moveTo(a[0], a[1] - 10 * a[2]); ctx.lineTo(b[0], b[1] - 10 * b[2]); ctx.stroke();
      for (const p of [a, b]) { ctx.fillStyle = "#c0c4ca"; ctx.fillRect(p[0] - 1.5, p[1] - 14 * p[2], 3, 14 * p[2]); }
    }
  }
  void r;
}

function lerp2(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

// ---------------------------------------------------------------- crowd surges (crowds that build, pack, then clear)
// Surge level 0..1 over the camera's cycle: quiet -> building -> packed -> clearing.
// Uses the wall clock, so a feed shows the same moment when it is enlarged or re-opened.
export function surgeLevel(cam, nowSec = Date.now() / 1000) {
  const s = cam.surge;
  if (!s) return 0;
  const p = (((nowSec + s.offset) % s.period) + s.period) % s.period / s.period;
  const smooth = (x) => x * x * (3 - 2 * x);
  if (p < 0.1) return 0; // quiet
  if (p < 0.35) return smooth((p - 0.1) / 0.25); // building up
  if (p < 0.7) return 1; // packed
  return 1 - smooth((p - 0.7) / 0.3); // clearing
}

function gauss(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
}

function surgeAgents(cam, r) {
  const s = cam.surge;
  return Array.from({ length: s.extra }, (_, i) => ({
    state: "surge", mode: "off", threshold: 0.05 + 0.9 * (i / s.extra) + r() * 0.03,
    home: [Math.min(0.97, Math.max(0.03, s.at[0] + gauss(r) * s.spread[0])),
           Math.min(1.02, Math.max(0.02, s.at[1] + gauss(r) * s.spread[1]))],
    color: CLOTHES[Math.floor(r() * CLOTHES.length)], skin: SKIN[Math.floor(r() * SKIN.length)],
    speed: 0.13 + r() * 0.07, phase: r() * 6, u: 0, v: 0,
  }));
}

// where a surge person walks in from (and leaves to): nearby doors or the side of the frame at the
// hotspot's depth, so crowds form within a few seconds
function surgeEntry(cam, r, home) {
  if (cam.scene === "entrance") return r() < 0.7 ? [0.42 + r() * 0.16, -0.02] : [r() < 0.5 ? -0.08 : 1.08, home[1] + (r() - 0.5) * 0.25];
  const side = r() < 0.5 ? -0.08 : 1.08;
  return [side, Math.min(1.05, Math.max(0.02, home[1] + (r() - 0.5) * 0.3))];
}

function stepSurge(cam, p, r, dt, level) {
  if (p.mode === "off") {
    if (level > p.threshold) {
      [p.u, p.v] = surgeEntry(cam, r, p.home);
      [p.tu, p.tv] = p.home;
      p.mode = "in";
    }
    return;
  }
  if ((p.mode === "in" || p.mode === "stay") && level < p.threshold - 0.04) {
    [p.tu, p.tv] = surgeEntry(cam, r, p.home);
    p.mode = "out";
  }
  if (p.mode === "stay") {
    // packed crowd: people shuffle on the spot
    p.u += (r() - 0.5) * 0.004;
    p.v += (r() - 0.5) * 0.003;
    p.u += (p.home[0] - p.u) * 0.05;
    p.v += (p.home[1] - p.v) * 0.05;
    p.phase += dt * 1.5;
    return;
  }
  const du = p.tu - p.u;
  const dv = p.tv - p.v;
  const d = Math.hypot(du, dv);
  if (d < 0.012) {
    if (p.mode === "in") p.mode = "stay";
    else p.mode = "off";
    return;
  }
  // slower when walking into a dense area
  const sp = p.speed * dt * (p.mode === "in" ? 1 - 0.5 * level : 1);
  p.u += (du / d) * Math.min(sp, d);
  p.v += (dv / d) * Math.min(sp, d) * 0.8;
  p.phase += dt * 9;
}

// ---------------------------------------------------------------- crowd simulation
function inVoid(cam, u, v) {
  return cam.scene === "atrium" && !cam.ground && u > 0.22 && u < 0.78 && v > 0.28 && v < 0.64;
}

function newTarget(cam, r, p) {
  switch (cam.scene) {
    case "corridor":
      return p.u < 0.5 ? [1.12, 0.1 + r() * 0.85] : [-0.12, 0.1 + r() * 0.85];
    case "entrance":
    case "storefront":
      return p.v < 0.3 ? [0.1 + r() * 0.8, 1.15] : [0.42 + r() * 0.16, -0.02];
    case "escalator":
      return [0.5 + r() * 0.45, 0.1 + r() * 0.85];
    default: {
      let u, v;
      do { u = 0.05 + r() * 0.9; v = 0.05 + r() * 0.9; } while (inVoid(cam, u, v) || (cam.ground && Math.hypot(u - 0.5, (v - 0.5) * 1.6) < 0.12));
      return [u, v];
    }
  }
}

function spawn(cam, r, initial) {
  const p = { color: CLOTHES[Math.floor(r() * CLOTHES.length)], skin: SKIN[Math.floor(r() * SKIN.length)],
              speed: 0.05 + r() * 0.05, phase: r() * 6, state: "walk" };
  if (cam.scene === "corridor") {
    p.u = initial ? r() : r() < 0.5 ? -0.1 : 1.1;
    p.v = 0.1 + r() * 0.85;
  } else if (cam.scene === "entrance" || cam.scene === "storefront") {
    const fromDoor = r() < 0.5;
    p.u = fromDoor ? 0.42 + r() * 0.16 : 0.1 + r() * 0.8;
    p.v = initial ? r() : fromDoor ? 0.0 : 1.12;
  } else {
    do { p.u = 0.05 + r() * 0.9; p.v = 0.05 + r() * 0.9; } while (inVoid(cam, p.u, p.v));
  }
  [p.tu, p.tv] = newTarget(cam, r, p);
  return p;
}

function buildCrowd(cam, r) {
  const people = [];
  const n = cam.crowd;
  if (cam.scene === "foodcourt") {
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) {
      if (r() < 0.75) {
        const seats = 1 + Math.floor(r() * 3);
        for (let k = 0; k < seats; k++) {
          const ang = (k / seats) * Math.PI * 2 + r();
          people.push({ state: "sit", u: 0.14 + i * 0.18 + Math.cos(ang) * 0.05, v: 0.28 + j * 0.27 + Math.sin(ang) * 0.04,
                        color: CLOTHES[Math.floor(r() * CLOTHES.length)], skin: SKIN[Math.floor(r() * SKIN.length)], phase: r() * 6 });
        }
      }
    }
  }
  if (cam.scene === "cinema") {
    let k = 0;
    for (const v of [0.24, 0.44, 0.64]) for (let i = 0; i < 6; i++) {
      people.push({ state: "queue", u: 0.32 + i * 0.07, v, base: 0.32 + i * 0.07, color: CLOTHES[k++ % CLOTHES.length],
                    skin: SKIN[k % SKIN.length], phase: r() * 6 });
    }
  }
  if (cam.scene === "escalator") {
    for (let i = 0; i < 7; i++) people.push({ state: "esc", t: i / 7, color: CLOTHES[i % CLOTHES.length], skin: SKIN[i % SKIN.length], phase: r() * 6 });
  }
  const walkers = Math.max(4, n - people.length);
  for (let i = 0; i < walkers; i++) people.push(spawn(cam, r, true));
  if (cam.surge) people.push(...surgeAgents(cam, r));
  return people;
}

function step(cam, people, r, dt, t, level) {
  for (const p of people) {
    if (p.state === "sit") continue;
    if (p.state === "surge") {
      stepSurge(cam, p, r, dt, level);
      continue;
    }
    if (p.state === "queue") {
      p.u = p.base + Math.sin(t * 0.4 + p.phase) * 0.004;
      continue;
    }
    if (p.state === "esc") {
      p.t += dt * 0.05;
      if (p.t > 1) p.t -= 1;
      continue;
    }
    const du = p.tu - p.u;
    const dv = p.tv - p.v;
    const d = Math.hypot(du, dv);
    if (d < 0.01) {
      if (p.u < -0.08 || p.u > 1.08 || p.v > 1.1 || p.v < 0.0) Object.assign(p, spawn(cam, r, false));
      else [p.tu, p.tv] = newTarget(cam, r, p);
      continue;
    }
    const sp = p.speed * dt;
    p.u += (du / d) * Math.min(sp, d);
    p.v += (dv / d) * Math.min(sp, d) * 0.8;
    if (inVoid(cam, p.u, p.v)) [p.tu, p.tv] = newTarget(cam, r, p);
    p.phase += dt * 9;
  }
}

function drawPerson(ctx, x, y, s, p, sitting) {
  const h = 34 * s;
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  ctx.beginPath(); ctx.ellipse(x, y, 7 * s, 2.5 * s, 0, 0, Math.PI * 2); ctx.fill();
  if (!sitting) {
    const sw = Math.sin(p.phase) * 3.5 * s;
    ctx.strokeStyle = "#2d3138";
    ctx.lineWidth = Math.max(1.2, 2.4 * s);
    ctx.beginPath(); ctx.moveTo(x - 1.5 * s, y - h * 0.45); ctx.lineTo(x - 1.5 * s + sw, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + 1.5 * s, y - h * 0.45); ctx.lineTo(x + 1.5 * s - sw, y); ctx.stroke();
  }
  const top = sitting ? y - h * 0.75 : y - h * 0.92;
  const bh = sitting ? h * 0.4 : h * 0.48;
  ctx.fillStyle = p.color;
  ctx.beginPath();
  ctx.roundRect(x - 5 * s, top + h * 0.14, 10 * s, bh, 3 * s);
  ctx.fill();
  ctx.fillStyle = p.skin;
  ctx.beginPath(); ctx.arc(x, top + h * 0.07, 4 * s, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#2a2522";
  ctx.beginPath(); ctx.arc(x, top + h * 0.05, 4 * s, Math.PI, 0); ctx.fill();
}

// ---------------------------------------------------------------- CCTV overlay textures
function makeNoise(seed) {
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(W, H);
  const r = rng(seed);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.floor(r() * 255);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
let NOISE = null;
let SCAN = null;
function overlays() {
  if (!NOISE) {
    NOISE = [makeNoise(1), makeNoise(2), makeNoise(3)];
    SCAN = document.createElement("canvas");
    SCAN.width = W; SCAN.height = H;
    const s = SCAN.getContext("2d");
    s.fillStyle = "rgba(0,0,0,0.13)";
    for (let y = 0; y < H; y += 3) s.fillRect(0, y, W, 1);
    const g = s.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.62);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0,0.55)");
    s.fillStyle = g;
    s.fillRect(0, 0, W, H);
  }
  return { noise: NOISE, scan: SCAN };
}

const pad = (n) => String(n).padStart(2, "0");

// ---------------------------------------------------------------- component
export default function CctvFeed({ cam, index, scale = 1, onCount }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const k = scale * (window.devicePixelRatio > 1 ? 1.5 : 1);
    canvas.width = W * k;
    canvas.height = H * k;
    const ctx = canvas.getContext("2d");
    const r = rng(1000 + index * 77);
    const bg = document.createElement("canvas");
    bg.width = W; bg.height = H;
    drawBackground(bg.getContext("2d"), cam, r);
    const people = buildCrowd(cam, r);
    const { noise, scan } = overlays();
    let raf;
    let last = performance.now();
    let acc = 0;
    let frame = 0;
    let lastCount = 0;
    let lastStatus = "normal";
    let lastReport = 0;
    const t0 = performance.now();

    const render = (now) => {
      raf = requestAnimationFrame(render);
      const dt = Math.min(0.2, (now - last) / 1000);
      last = now;
      acc += dt;
      if (acc < 1 / FPS) return;
      const stepDt = acc;
      acc = 0;
      frame++;
      const t = (now - t0) / 1000;
      const level = surgeLevel(cam);
      step(cam, people, r, stepDt, t, level);

      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.drawImage(bg, 0, 0);
      // people, far to near
      const drawn = people.filter((p) => !(p.state === "surge" && p.mode === "off")).map((p) => {
        if (p.state === "esc") {
          const a = proj(0.21 + 0.25 * p.t, 1.0 - 0.95 * p.t);
          return { p, x: a[0], y: a[1] - 6 * a[2], s: a[2], v: 1 - p.t };
        }
        const [x, y, s] = proj(p.u, p.v);
        return { p, x, y, s, v: p.v };
      }).filter((d) => d.x > -20 && d.x < W + 20 && d.y < H + 30);
      drawn.sort((a, b) => a.v - b.v);
      for (const d of drawn) drawPerson(ctx, d.x, d.y, d.s, d.p, d.p.state === "sit");

      // crowd density at the surge spot: people packed in, as a share of the crowd that gathers there
      let status = "normal";
      let packed = [];
      if (cam.surge) {
        packed = drawn.filter((d) => d.p.state === "surge" && d.p.mode === "stay");
        const share = packed.length / cam.surge.extra;
        status = share > 0.6 ? "crowded" : share > 0.25 ? "busy" : "normal";
      }
      const drawDensityBox = () => {
        if (status !== "crowded" || !packed.length) return;
        // detection box around the packed crowd, like an analytics overlay
        const xs = packed.map((d) => d.x);
        const ys = packed.map((d) => d.y);
        const x0 = Math.max(4, Math.min(...xs) - 10);
        const x1 = Math.min(W - 4, Math.max(...xs) + 10);
        const y0 = Math.max(26, Math.min(...ys) - 36);
        const y1 = Math.min(H - 26, Math.max(...ys) + 6);
        ctx.fillStyle = "rgba(255,59,48,0.10)";
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.strokeStyle = "#ff3b30";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
        const tag = `HIGH DENSITY · ${packed.length} people`;
        ctx.font = "bold 10px Consolas, monospace";
        const tw = ctx.measureText(tag).width + 10;
        ctx.fillStyle = "#ff3b30";
        ctx.fillRect(x0, y0 - 14, tw, 14);
        ctx.fillStyle = "#fff";
        ctx.textAlign = "left";
        ctx.fillText(tag, x0 + 5, y0 - 4);
      };

      // CCTV look: wash out colour, grain, scanlines, vignette, slight flicker
      ctx.globalCompositeOperation = "saturation";
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = "#808080";
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 0.1;
      ctx.drawImage(noise[frame % 3], 0, 0);
      ctx.globalAlpha = 1;
      ctx.drawImage(scan, 0, 0);
      if (frame % 47 === 0) {
        ctx.fillStyle = "rgba(255,255,255,0.06)";
        ctx.fillRect(0, 0, W, H);
      }
      drawDensityBox();

      // on-screen display
      const now2 = new Date();
      ctx.font = "bold 11px Consolas, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(6, 6, ctx.measureText(`${cam.id}  ${cam.name.toUpperCase()}`).width + 10, 16);
      ctx.fillStyle = "#f2f2f2";
      ctx.fillText(`${cam.id}  ${cam.name.toUpperCase()}`, 11, 18);
      const ts = `${now2.getFullYear()}-${pad(now2.getMonth() + 1)}-${pad(now2.getDate())} ${pad(now2.getHours())}:${pad(now2.getMinutes())}:${pad(now2.getSeconds())}`;
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(W - ctx.measureText(ts).width - 46, 6, ctx.measureText(ts).width + 40, 16);
      ctx.fillStyle = "#f2f2f2";
      ctx.fillText(ts, W - 11, 18);
      if (Math.floor(t * 1.5) % 2 === 0) {
        ctx.fillStyle = "#ff3b30";
        ctx.beginPath(); ctx.arc(W - ctx.measureText(ts).width - 32, 14, 4, 0, Math.PI * 2); ctx.fill();
      }
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(6, H - 22, 92, 16);
      ctx.fillStyle = "#f2f2f2";
      ctx.fillText(`FLOOR ${FLOOR_NAMES[cam.floor].toUpperCase()}`, 11, H - 10);
      const count = drawn.filter((d) => d.x > 0 && d.x < W && d.y > HZ - 10 && d.y < H + 10).length;
      ctx.textAlign = "right";
      ctx.fillStyle = status === "crowded" ? "rgba(200,30,30,0.85)" : status === "busy" ? "rgba(190,120,0,0.85)" : "rgba(0,0,0,0.45)";
      ctx.fillRect(W - 112, H - 22, 106, 16);
      ctx.fillStyle = "#f2f2f2";
      ctx.fillText(`PEOPLE: ${count}`, W - 11, H - 10);

      if (status !== "normal") {
        // warning banner at the top centre (blinks when crowded)
        const crowded = status === "crowded";
        if (!crowded || Math.floor(t * 2) % 2 === 0) {
          const msg = crowded ? "⚠ CROWDED" : "BUSY";
          ctx.font = "bold 13px Segoe UI, Arial";
          const bw = ctx.measureText(msg).width + 20;
          ctx.fillStyle = crowded ? "#e0241b" : "#d18a00";
          ctx.fillRect(W / 2 - bw / 2, 26, bw, 20);
          ctx.fillStyle = "#fff";
          ctx.textAlign = "center";
          ctx.fillText(msg, W / 2, 41);
        }
        if (crowded) {
          ctx.strokeStyle = "rgba(255,59,48,0.9)";
          ctx.lineWidth = 4;
          ctx.strokeRect(2, 2, W - 4, H - 4);
        }
      }

      if (onCount && (status !== lastStatus || (count !== lastCount && now - lastReport > 1000))) {
        lastCount = count;
        lastStatus = status;
        lastReport = now;
        onCount(cam.id, count, status);
      }
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [cam, index, scale, onCount]);

  return <canvas ref={ref} className="cctv-canvas" aria-label={`${cam.id} ${cam.name} live feed (simulated)`} />;
}
