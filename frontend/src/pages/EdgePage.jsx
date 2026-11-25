import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { ms, postJSON } from "../api.js";

const DEFAULTS = { n_cams: 20, fps: 10, bg_load: 0.5, slicing: true, edge_other_load: 0, cloud_rtt_ms: 50 };
const PRESETS = [
  { name: "Normal day", desc: "10 cameras, light phone traffic", v: { n_cams: 10, fps: 10, bg_load: 0.3, slicing: true, edge_other_load: 0, cloud_rtt_ms: 50 } },
  { name: "Big festival", desc: "80 cameras, busy network", v: { n_cams: 80, fps: 10, bg_load: 0.7, slicing: true, edge_other_load: 0.2, cloud_rtt_ms: 50 } },
  { name: "Network jammed", desc: "everyone on their phones, no slicing", v: { n_cams: 20, fps: 10, bg_load: 0.92, slicing: false, edge_other_load: 0, cloud_rtt_ms: 50 } },
  { name: "Edge server busy", desc: "edge shared with other apps", v: { n_cams: 40, fps: 15, bg_load: 0.5, slicing: true, edge_other_load: 0.7, cloud_rtt_ms: 30 } },
];

export default function EdgePage() {
  const [cfg, setCfg] = useState(DEFAULTS);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const reqId = useRef(0);

  useEffect(() => {
    const id = ++reqId.current;
    const timer = setTimeout(async () => {
      setBusy(true);
      setError("");
      try {
        const r = await postJSON("/api/edge", { ...cfg, sweep: true });
        if (id === reqId.current) setRes(r); // ignore answers to older settings
      } catch (e) {
        if (id === reqId.current) setError(e.message);
      } finally {
        if (id === reqId.current) setBusy(false);
      }
    }, 450);
    return () => clearTimeout(timer);
  }, [cfg]);

  const set = (k) => (e) => setCfg({ ...cfg, [k]: e.target.type === "checkbox" ? e.target.checked : Number(e.target.value) });

  return (
    <>
      <div className="page-head">
        <h1>Edge vs cloud over 5G</h1>
        <p className="muted">
          Where should the AI run? Change the situation and see how long an alert takes when the video is processed at
          the 5G edge server (next to the tower) or in a faraway cloud, and what the ML decision model would pick.
        </p>
      </div>
      <div className="grid side">
        <div className="card">
          <h2>Situation</h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {PRESETS.map((p) => (
              <button key={p.name} className="preset" title={p.desc} onClick={() => setCfg(p.v)}>
                <b>{p.name}</b><span>{p.desc}</span>
              </button>
            ))}
          </div>

          <Slider label="Cameras" value={cfg.n_cams} min={1} max={150} step={1} onChange={set("n_cams")}
                  show={`${cfg.n_cams}`} />
          <Slider label="Frames per second per camera" value={cfg.fps} min={1} max={30} step={1} onChange={set("fps")}
                  show={`${cfg.fps} fps`} />
          <Slider label="5G uplink used by the public's phones" value={cfg.bg_load} min={0} max={0.95} step={0.05}
                  onChange={set("bg_load")} show={`${Math.round(cfg.bg_load * 100)}%`} />
          <Slider label="Edge server busy with other apps" value={cfg.edge_other_load} min={0} max={0.8} step={0.05}
                  onChange={set("edge_other_load")} show={`${Math.round(cfg.edge_other_load * 100)}%`} />
          <Slider label="Cloud round-trip time" value={cfg.cloud_rtt_ms} min={10} max={150} step={5}
                  onChange={set("cloud_rtt_ms")} show={`${cfg.cloud_rtt_ms} ms`} />

          <label className="toggle">
            <input type="checkbox" checked={cfg.slicing} onChange={set("slicing")} />
            <span>
              <b>5G network slicing (URLLC)</b>
              <span className="hint" style={{ display: "block", marginTop: 2 }}>
                Safety cameras get their own priority lane, so the public's phone traffic cannot slow them down.
              </span>
            </span>
          </label>
          {error && <div className="error">{error}</div>}
        </div>

        <div>{res ? <Results res={res} cfg={cfg} busy={busy} /> : (
          <div className="card empty"><span className="spinner dark" /><p>Simulating the 5G network…</p></div>
        )}</div>
      </div>
    </>
  );
}

function Slider({ label, value, min, max, step, onChange, show }) {
  return (
    <>
      <label className="field" style={{ display: "flex", justifyContent: "space-between" }}>
        <span>{label}</span><span style={{ color: "var(--blue)" }}>{show}</span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={onChange} style={{ width: "100%" }} />
    </>
  );
}

function PathCard({ title, sub, r, faster, deadline, proc }) {
  const late = r.p95_ms > deadline;
  return (
    <div className="card" style={{ borderColor: faster ? "var(--green)" : undefined, borderWidth: faster ? 2 : 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {faster && <span className="badge green">faster</span>}
      </div>
      <p className="hint" style={{ marginTop: 2 }}>{sub}</p>
      <div style={{ fontSize: 32, fontWeight: 700, color: late ? "var(--red)" : "var(--ink)", marginTop: 6 }}>{ms(r.p95_ms)}</div>
      <div className="small muted">95% of alerts arrive within this time</div>
      <table className="simple" style={{ marginTop: 10 }}>
        <tbody>
          <tr><td>Average delay</td><td>{ms(r.mean_ms)}</td></tr>
          <tr><td>Alerts later than {deadline} ms</td><td style={{ color: r.miss_rate > 0.05 ? "var(--red)" : undefined }}>{(r.miss_rate * 100).toFixed(1)}%</td></tr>
          <tr><td>AI time per frame</td><td>{proc.toFixed(1)} ms</td></tr>
          <tr><td>Frames handled</td><td>{r.completed} / {r.created}</td></tr>
        </tbody>
      </table>
    </div>
  );
}

function Results({ res, cfg, busy }) {
  const data = res.sweep.map((s) => ({ ...s, edge_p95: Math.max(1, s.edge_p95), cloud_p95: Math.max(1, s.cloud_p95) }));
  const best = res.best;
  return (
    <div style={{ opacity: busy ? 0.6 : 1, transition: "opacity .2s" }}>
      <div className="grid two">
        <PathCard title="⚡ Edge (MEC)" sub="Server next to the 5G tower · small fast model · 2 workers"
                  r={res.edge} faster={best === "edge"} deadline={res.deadline_ms} proc={res.processing_ms.edge} />
        <PathCard title="☁️ Cloud" sub={`Data centre ${cfg.cloud_rtt_ms} ms away · most accurate model (v2) on GPU`}
                  r={res.cloud} faster={best === "cloud"} deadline={res.deadline_ms} proc={res.processing_ms.cloud} />
      </div>

      <div className="card decision" style={{ marginTop: 18 }}>
        <div>
          <div className="small muted">Actually faster here</div>
          <div className="big-word">{best === "edge" ? "⚡ Edge" : "☁️ Cloud"}</div>
        </div>
        <div>
          <div className="small muted">ML decision model picked</div>
          <div className="big-word">{res.ml_choice === "edge" ? "⚡ Edge" : "☁️ Cloud"}</div>
        </div>
        <div>
          <div className="small muted">Was the ML right?</div>
          <div className="big-word" style={{ color: res.ml_correct ? "var(--green)" : "var(--red)" }}>
            {res.ml_correct ? "✓ Correct" : "✗ Wrong"}
          </div>
        </div>
        {busy && <span className="spinner dark" style={{ width: 20, height: 20 }} />}
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h3>What happens as more cameras are added</h3>
        <p className="hint">Same settings, number of cameras from 1 to 120. Log scale: each grid line is 10× slower.</p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 8, right: 20, bottom: 18, left: 4 }}>
            <CartesianGrid stroke="#eef1f4" />
            <XAxis dataKey="n_cams" type="number" domain={[0, 120]} ticks={[1, 20, 40, 60, 80, 100, 120]}
                   label={{ value: "number of cameras", position: "insideBottom", offset: -8, fontSize: 12 }} />
            <YAxis scale="log" domain={[1, 10000]} ticks={[1, 10, 100, 1000, 10000]} allowDataOverflow
                   tickFormatter={(v) => (v >= 1000 ? `${v / 1000}s` : `${v}ms`)} width={50} />
            <ReferenceLine y={res.deadline_ms} stroke="#d1242f" strokeDasharray="5 4"
                           label={{ value: "100 ms deadline", position: "insideTopRight", fontSize: 11, fill: "#d1242f" }} />
            <ReferenceLine x={cfg.n_cams} stroke="#1b2430" label={{ value: "now", position: "top", fontSize: 11 }} />
            <Tooltip formatter={(v) => ms(v)} labelFormatter={(n) => `${n} cameras`} />
            <Legend verticalAlign="top" height={28} />
            <Line dataKey="edge_p95" name="Edge (95% of alerts)" stroke="#b7791f" strokeWidth={2} isAnimationActive={false} />
            <Line dataKey="cloud_p95" name="Cloud (95% of alerts)" stroke="#8250df" strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="ml-strip">
          <span className="small muted">ML picks:</span>
          {res.sweep.map((s) => (
            <span key={s.n_cams} className={`badge ${s.ml === "edge" ? "amber" : "gray"}`} title={`${s.n_cams} cameras`}>
              {s.n_cams}: {s.ml === "edge" ? "edge" : "cloud"}
            </span>
          ))}
        </div>
      </div>

      <p className="hint" style={{ marginTop: 12 }}>
        Each camera sends a {res.frame_kb.toFixed(1)} KB JPEG frame over a shared 250 Mbps 5G uplink. Delay = upload over 5G
        + travel to the server + waiting in the queue + AI processing + alert back. Simulated with a discrete-event model of
        the 5G network, using the AI speeds measured on this laptop.
      </p>
    </div>
  );
}
