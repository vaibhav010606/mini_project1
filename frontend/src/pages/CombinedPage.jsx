import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { postJSON } from "../api.js";
import { Heatmap, segments } from "./CrowdPage.jsx";

const MODELS = [
  { key: "fusion", name: "Combined (video + 5G)", color: "#1a7f37" },
  { key: "video", name: "Video only", color: "#b7791f" },
  { key: "net", name: "5G only", color: "#8250df" },
];
const BADGE = { normal: "green", dense: "amber", stampede: "red" };
const verdictOf = (p, labels) => (p ? labels[p.indexOf(Math.max(...p))] : null);

export default function CombinedPage() {
  const [scenario, setScenario] = useState("stampede");
  const [people, setPeople] = useState(2200);
  const [seed, setSeed] = useState(4);
  const [night, setNight] = useState(false);
  const [few, setFew] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [r, setR] = useState(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);

  async function run() {
    setBusy(true);
    setError("");
    setPlaying(false);
    try {
      const res = await postJSON("/api/combined", {
        scenario, n_people: Number(people), seed: Number(seed), night, few_phones: few,
      });
      setR(res);
      setT(0);
      setPlaying(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!playing || !r) return undefined;
    const id = setInterval(() => setT((x) => {
      if (x >= r.seconds - 1) { setPlaying(false); return x; }
      return x + 1;
    }), 110);
    return () => clearInterval(id);
  }, [playing, r]);

  return (
    <>
      <div className="page-head">
        <h1>Combined monitor: video + 5G</h1>
        <p className="muted">
          One crowd event, seen two ways at once: a CCTV camera films part of the venue while the 5G network senses the
          same people's phones everywhere. The combined model uses both to decide normal / dense / stampede.
        </p>
      </div>
      <div className="grid side">
        <div className="card">
          <h2>Event</h2>
          <div className="seg">
            {["normal", "gathering", "stampede"].map((s) => (
              <button key={s} className={scenario === s ? "on" : ""} onClick={() => setScenario(s)}>{s}</button>
            ))}
          </div>
          <label className="field" htmlFor="cp">People in the venue: {people}</label>
          <input id="cp" type="range" min="800" max="2600" step="100" value={people}
                 onChange={(e) => setPeople(e.target.value)} style={{ width: "100%" }} />
          <label className="field" htmlFor="cs">Random seed</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input id="cs" type="number" min="0" max="10000" value={seed} onChange={(e) => setSeed(e.target.value)} />
            <button className="btn" style={{ width: 110, marginTop: 0 }} onClick={() => setSeed(Math.floor(Math.random() * 10000))}>
              New event
            </button>
          </div>
          <div className="hint">The seed also decides where the hotspot is, so sometimes it is outside the camera view.</div>

          <h3 style={{ marginTop: 18 }}>Make it hard</h3>
          <label className="toggle">
            <input type="checkbox" checked={night} onChange={(e) => setNight(e.target.checked)} />
            <span><b>Night</b><span className="hint" style={{ display: "block", marginTop: 2 }}>Dark, noisy CCTV video: the camera barely sees people.</span></span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={few} onChange={(e) => setFew(e.target.checked)} />
            <span><b>Few phones visible</b><span className="hint" style={{ display: "block", marginTop: 2 }}>60% of people invisible to the operator (no phone, phone off, other network).</span></span>
          </label>

          <button className="btn" onClick={run} disabled={busy}>
            {busy ? <><span className="spinner" /> Simulating event + running 3 models…</> : "Run event"}
          </button>
          {error && <div className="error">{error}</div>}
          {r && <EventSummary r={r} />}
        </div>

        <div>
          {!r && !busy && (
            <div className="card empty">
              <p>Choose an event and press <b>Run event</b>.</p>
              <p className="small">You will see the CCTV video and the 5G density map of the same crowd side by side, and what each model decides every second.</p>
            </div>
          )}
          {busy && !r && <div className="card empty"><span className="spinner dark" /><p>Simulating the crowd, filming it and running the models… about 10 s.</p></div>}
          {r && <Monitor r={r} t={t} setT={setT} playing={playing} setPlaying={setPlaying} />}
        </div>
      </div>
    </>
  );
}

function EventSummary({ r }) {
  const i = r.info;
  const firstStampede = (key) => {
    const idx = r.probs[key].findIndex((p) => p && p[2] === Math.max(...p));
    return idx >= 0 ? idx : null;
  };
  const caught = (key) => {
    const danger = r.truth.map((y, k) => [y, k]).filter(([y, k]) => y > 0 && r.probs[key][k]);
    if (!danger.length) return null;
    return danger.filter(([, k]) => verdictOf(r.probs[key][k], r.labels) !== "normal").length / danger.length;
  };
  const falseAlarm = (key) => {
    const calm = r.truth.map((y, k) => [y, k]).filter(([y, k]) => y === 0 && r.probs[key][k]);
    return calm.filter(([, k]) => verdictOf(r.probs[key][k], r.labels) !== "normal").length / calm.length;
  };
  return (
    <div style={{ marginTop: 16 }}>
      <h3>This event</h3>
      <table className="simple">
        <tbody>
          <tr><td>People / visible phones</td><td>{i.n_people} / {i.n_phones}</td></tr>
          <tr><td>Hotspot in camera view</td><td>{i.hotspot_in_view ? "yes" : <b style={{ color: "var(--red)" }}>no: camera cannot see it</b>}</td></tr>
          <tr><td>Panic started</td><td>{i.panic_t0 !== null ? `${i.panic_t0} s` : "no panic"}</td></tr>
        </tbody>
      </table>
      <table className="simple" style={{ marginTop: 10 }}>
        <thead><tr><th>Model</th><th>Danger caught</th><th>False alarms</th>{i.panic_t0 !== null && <th>Stampede at</th>}</tr></thead>
        <tbody>
          {MODELS.map((m) => {
            const c = caught(m.key);
            const fs = firstStampede(m.key);
            return (
              <tr key={m.key} className={m.key === "fusion" ? "hl" : ""}>
                <td>{m.name}</td>
                <td>{c === null ? "–" : `${Math.round(c * 100)}%`}</td>
                <td>{`${Math.round(falseAlarm(m.key) * 100)}%`}</td>
                {i.panic_t0 !== null && <td>{fs === null ? "missed" : `${fs} s`}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Monitor({ r, t, setT, playing, setPlaying }) {
  const truth = r.labels[r.truth[t]];
  const map = r.maps[Math.min(r.maps.length - 1, Math.floor(t / 2))];
  const est = map.est;
  const maxD = Math.max(...est.flat());
  const simLike = { grid: r.grid, venue: r.venue, gnb: r.gnb };

  return (
    <>
      <div className="stats">
        {MODELS.map((m) => {
          const v = verdictOf(r.probs[m.key][t], r.labels);
          return (
            <div key={m.key} className="stat" style={m.key === "fusion" ? { borderColor: "var(--green)", borderWidth: 2 } : null}>
              <div className="v" style={{ fontSize: m.key === "fusion" ? 22 : 18 }}>
                {v ? <span className={`badge ${BADGE[v]}`} style={{ fontSize: m.key === "fusion" ? 18 : 14 }}>{v.toUpperCase()}</span> : <span className="muted small">warming up</span>}
              </div>
              <div className="k">{m.name}</div>
            </div>
          );
        })}
        <div className="stat">
          <div className="v"><span className={`badge ${BADGE[truth]}`} style={{ fontSize: 14 }}>{truth}</span></div>
          <div className="k">true state · {t} s</div>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>📷 CCTV camera {r.info.night && <span className="badge gray">night</span>}</h3>
          <CctvFrame r={r} t={t} />
          <p className="hint">
            Overhead camera covering the blue box on the map ({r.camera.size} m × {r.camera.size} m). Synthetic video rendered from the
            simulated people; shown here at {r.camera.res}×{r.camera.res} for viewing, while the model analyses a {r.camera.model_res}×{r.camera.model_res} version.
          </p>
        </div>
        <div className="card">
          <h3>📡 5G network view {r.info.few_phones && <span className="badge gray">few phones</span>}</h3>
          <Heatmap grid={est} sim={simLike} hotspot={maxD >= 1 ? hotspotOf(est, r) : null} camera={r.camera} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 10 }}>
          <button className="btn" style={{ width: 90, marginTop: 0 }} onClick={() => {
            if (t >= r.seconds - 1) setT(0);
            setPlaying(!playing);
          }}>{playing ? "Pause" : "Play"}</button>
          <input type="range" min="0" max={r.seconds - 1} value={t} style={{ flex: 1 }}
                 onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
          <span className="small muted" style={{ width: 52 }}>{t} s</span>
        </div>
        <h3>Danger probability over time (P(dense) + P(stampede))</h3>
        <DangerChart r={r} t={t} onPick={(s) => { setPlaying(false); setT(s); }} />
      </div>
    </>
  );
}

function hotspotOf(est, r) {
  let best = [0, 0];
  let v = -1;
  est.forEach((col, ix) => col.forEach((d, iy) => { if (d > v) { v = d; best = [ix, iy]; } }));
  return [(best[0] + 0.5) * r.zone, (best[1] + 0.5) * r.zone];
}

function CctvFrame({ r, t }) {
  const ref = useRef(null);
  const img = useRef(null);
  const [ready, setReady] = useState(false);
  const size = 384;

  useEffect(() => {
    setReady(false);
    const im = new Image();
    im.onload = () => { img.current = im; setReady(true); };
    im.src = r.frames_url;
  }, [r.frames_url]);

  useEffect(() => {
    if (!ready) return;
    const ctx = ref.current.getContext("2d");
    const { res, cols } = r.camera;
    const sx = (t % cols) * res;
    const sy = Math.floor(t / cols) * res;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img.current, sx, sy, res, res, 0, 0, size, size);
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(0, 0, size, 22);
    ctx.fillStyle = "#fff";
    ctx.font = "12px Consolas, monospace";
    ctx.fillText(`CAM 01   t = ${String(t).padStart(3, " ")} s`, 8, 15);
  }, [ready, t, r]);

  return <canvas ref={ref} width={size} height={size} style={{ width: "100%", maxWidth: size, display: "block", margin: "0 auto", borderRadius: 6, background: "#000" }} />;
}

function DangerChart({ r, t, onPick }) {
  const data = useMemo(() => r.truth.map((_, k) => {
    const row = { t: k };
    MODELS.forEach((m) => {
      const p = r.probs[m.key][k];
      row[m.key] = p ? +(p[1] + p[2]).toFixed(3) : null;
    });
    return row;
  }), [r]);
  return (
    <>
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={data} margin={{ top: 6, right: 14, bottom: 0, left: -14 }}
                   onClick={(e) => e && e.activeLabel !== undefined && onPick(Number(e.activeLabel))}>
          <CartesianGrid stroke="#eef1f4" />
          <XAxis dataKey="t" type="number" domain={[0, r.seconds - 1]} tickCount={10} fontSize={11} />
          <YAxis domain={[0, 1]} fontSize={11} />
          {segments(r.truth, 1).map(([a, b]) => <ReferenceArea key={`d${a}`} x1={a} x2={b} fill="#f0c83c" fillOpacity={0.25} />)}
          {segments(r.truth, 2).map(([a, b]) => <ReferenceArea key={`s${a}`} x1={a} x2={b} fill="#d1242f" fillOpacity={0.15} />)}
          <ReferenceLine y={0.5} stroke="#8c959f" strokeDasharray="4 4" />
          <ReferenceLine x={t} stroke="#1b2430" />
          <Tooltip formatter={(v) => (v === null ? "–" : v.toFixed(2))} labelFormatter={(x) => `${x} s`} />
          {MODELS.slice().reverse().map((m) => (
            <Line key={m.key} dataKey={m.key} name={m.name} stroke={m.color} dot={false}
                  strokeWidth={m.key === "fusion" ? 3 : 1.6} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="legend">
        {MODELS.map((m) => <span key={m.key}><i style={{ background: m.color }} />{m.name}</span>)}
        <span><i style={{ background: "#f0c83c", opacity: 0.5 }} />truly dense</span>
        <span><i style={{ background: "#d1242f", opacity: 0.3 }} />true stampede</span>
        <span>above the dashed line = alert · click to jump</span>
      </div>
    </>
  );
}
