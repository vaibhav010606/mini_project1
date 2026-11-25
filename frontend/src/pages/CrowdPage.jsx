import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { postJSON } from "../api.js";

const LABEL_STYLE = { normal: "green", dense: "amber", stampede: "red" };
const SCENARIO_HELP = {
  normal: "People wander around; small safe crowds and sometimes a harmless rush (e.g. hurrying to a train).",
  gathering: "A large part of the crowd packs into one spot until it becomes dangerously dense.",
  stampede: "The crowd packs into one spot, then panics and flees at 2–5 m/s.",
};
const DENSITY_MAX = 3; // people per m² at the top of the colour scale
const DANGER = 2; // people per m² considered dangerous

// green -> yellow -> red
export function densityColor(v) {
  const x = Math.max(0, Math.min(1, v / DENSITY_MAX));
  const stops = [[26, 127, 55], [240, 200, 60], [209, 36, 47]];
  const [a, b, f] = x < 0.5 ? [stops[0], stops[1], x / 0.5] : [stops[1], stops[2], (x - 0.5) / 0.5];
  return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * f)).join(",")})`;
}

export function segments(arr, value) {
  const out = [];
  let start = null;
  arr.forEach((v, i) => {
    if (v === value && start === null) start = i;
    if (v !== value && start !== null) { out.push([start, i - 1]); start = null; }
  });
  if (start !== null) out.push([start, arr.length - 1]);
  return out;
}

export default function CrowdPage() {
  const [scenario, setScenario] = useState("stampede");
  const [people, setPeople] = useState(1800);
  const [seed, setSeed] = useState(3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sim, setSim] = useState(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState("est");

  async function run() {
    setBusy(true);
    setError("");
    setPlaying(false);
    try {
      const r = await postJSON("/api/crowd", { scenario, n_people: Number(people), seed: Number(seed) });
      setSim(r);
      setIdx(0);
      setPlaying(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!playing || !sim) return undefined;
    const id = setInterval(() => {
      setIdx((i) => {
        if (i >= sim.frames.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, 220);
    return () => clearInterval(id);
  }, [playing, sim]);

  return (
    <>
      <div className="page-head">
        <h1>5G crowd monitor</h1>
        <p className="muted">
          Model 2 watches the crowd through the 5G network: how many phones each cell sees, how they move and where they
          gather. Only zone counts are used, never individual people.
        </p>
      </div>
      <div className="grid side">
        <div className="card">
          <h2>Scenario</h2>
          <div className="seg">
            {["normal", "gathering", "stampede"].map((s) => (
              <button key={s} className={scenario === s ? "on" : ""} onClick={() => setScenario(s)}>{s}</button>
            ))}
          </div>
          <div className="hint">{SCENARIO_HELP[scenario]}</div>

          <label className="field" htmlFor="people">People in the venue: {people}</label>
          <input id="people" type="range" min="500" max="2600" step="100" value={people}
                 onChange={(e) => setPeople(e.target.value)} style={{ width: "100%" }} />

          <label className="field" htmlFor="seed">Random seed</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input id="seed" type="number" min="0" max="10000" value={seed} onChange={(e) => setSeed(e.target.value)} />
            <button className="btn" style={{ width: 110, marginTop: 0 }} onClick={() => setSeed(Math.floor(Math.random() * 10000))}>
              New crowd
            </button>
          </div>
          <div className="hint">Same seed = same crowd. Change it to get a different crowd and hotspot.</div>

          <button className="btn" onClick={run} disabled={busy}>
            {busy ? <><span className="spinner" /> Simulating 3 minutes of crowd…</> : "Run simulation"}
          </button>
          {error && <div className="error">{error}</div>}
          {sim && <Summary sim={sim} />}
        </div>

        <div>
          {!sim && !busy && (
            <div className="card empty">
              <p>Pick a scenario and press <b>Run simulation</b>.</p>
              <p className="small">You will see the venue as the 5G network sees it, second by second, with Model 2's verdict.</p>
            </div>
          )}
          {busy && !sim && (
            <div className="card empty"><span className="spinner dark" /><p>Simulating people, phones and 5G cells…</p></div>
          )}
          {sim && (
            <Monitor sim={sim} idx={idx} setIdx={setIdx} playing={playing} setPlaying={setPlaying} view={view} setView={setView} />
          )}
        </div>
      </div>
    </>
  );
}

function Summary({ sim }) {
  const risk = sim.risk;
  const firstStampede = risk.findIndex((p) => p && p[2] === Math.max(...p));
  const firstDense = risk.findIndex((p) => p && p[1] === Math.max(...p));
  const panic = sim.info.panic_t0;
  return (
    <div style={{ marginTop: 16 }}>
      <h3>Summary</h3>
      <table className="simple">
        <tbody>
          <tr><td>People / phones</td><td>{sim.info.n_people} / {sim.info.n_phones}</td></tr>
          <tr><td>Panic started</td><td>{panic !== null ? `${panic} s` : "no panic"}</td></tr>
          <tr><td>Model: first "dense"</td><td>{firstDense >= 0 ? `${firstDense} s` : "never"}</td></tr>
          <tr><td>Model: first "stampede"</td><td>{firstStampede >= 0 ? `${firstStampede} s` : "never"}</td></tr>
          {panic !== null && firstStampede >= 0 && (
            <tr><td>Detection delay</td><td><b>{firstStampede - panic >= 0 ? "+" : ""}{firstStampede - panic} s</b></td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function Monitor({ sim, idx, setIdx, playing, setPlaying, view, setView }) {
  const frame = sim.frames[idx];
  const t = frame.t;
  const p = sim.risk[t];
  const verdict = p ? sim.labels[p.indexOf(Math.max(...p))] : null;
  const truth = sim.labels[sim.truth[t]];
  const grid = view === "est" ? frame.est : frame.true;
  const maxD = Math.max(...grid.flat());

  return (
    <>
      <div className="stats">
        <div className={`stat ${verdict === "stampede" ? "bad" : verdict === "normal" ? "good" : ""}`}>
          <div className="v" style={verdict === "dense" ? { color: "var(--amber)" } : null}>
            {verdict ? verdict.toUpperCase() : "warming up"}
          </div>
          <div className="k">{verdict ? "Model 2 verdict" : "model needs 30 s of data"}</div>
        </div>
        <div className="stat">
          <div className="v"><span className={`badge ${LABEL_STYLE[truth]}`} style={{ fontSize: 16 }}>{truth}</span></div>
          <div className="k">true state (simulator)</div>
        </div>
        <div className="stat">
          <div className="v">{maxD.toFixed(2)}</div>
          <div className="k">densest zone, people/m²</div>
        </div>
        <div className="stat">
          <div className="v">{t} s</div>
          <div className="k">of {sim.seconds} s</div>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Venue density map</h3>
            <div className="seg" style={{ width: 230 }}>
              <button className={view === "est" ? "on" : ""} onClick={() => setView("est")}>5G estimate</button>
              <button className={view === "true" ? "on" : ""} onClick={() => setView("true")}>True</button>
            </div>
          </div>
          <Heatmap grid={grid} sim={sim} hotspot={maxD >= 1 ? frame.hotspot : null} />
          <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 12 }}>
            <button className="btn" style={{ width: 90, marginTop: 0 }} onClick={() => {
              if (idx >= sim.frames.length - 1) setIdx(0);
              setPlaying(!playing);
            }}>
              {playing ? "Pause" : "Play"}
            </button>
            <input type="range" min="0" max={sim.frames.length - 1} value={idx} style={{ flex: 1 }}
                   onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)); }} />
          </div>
        </div>
        <div className="card">
          <h3>Model 2 risk over time</h3>
          <RiskChart sim={sim} t={t} onPick={(s) => { setPlaying(false); setIdx(Math.min(sim.frames.length - 1, Math.round(s / 2))); }} />
          <h3 style={{ marginTop: 14 }}>5G network signals</h3>
          <Signal sim={sim} t={t} name="max_zone_density" label="densest zone (people/m²)" color="#d1242f" />
          <Signal sim={sim} t={t} name="frac_fast" label="share of fast-moving phones" color="#1f6feb" />
          <Signal sim={sim} t={t} name="handover_rate" label="handovers per phone per second" color="#8250df" />
        </div>
      </div>
    </>
  );
}

export function Heatmap({ grid, sim, hotspot, camera }) {
  const ref = useRef(null);
  const size = 420;
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const n = sim.grid;
    const c = size / n;
    ctx.clearRect(0, 0, size, size);
    for (let ix = 0; ix < n; ix++) {
      for (let iy = 0; iy < n; iy++) {
        ctx.fillStyle = densityColor(grid[ix][iy]);
        ctx.fillRect(ix * c, size - (iy + 1) * c, c + 0.5, c + 0.5); // y axis points up
      }
    }
    const px = (m) => (m / sim.venue) * size;
    if (camera) {
      // CCTV camera field of view
      ctx.strokeStyle = "#1f6feb";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([7, 5]);
      ctx.strokeRect(px(camera.x0), size - px(camera.y0 + camera.size), px(camera.size), px(camera.size));
      ctx.setLineDash([]);
      ctx.fillStyle = "#1f6feb";
      ctx.font = "bold 12px Segoe UI, sans-serif";
      ctx.fillText("CAMERA VIEW", px(camera.x0) + 5, size - px(camera.y0 + camera.size) + 15);
    }
    if (hotspot) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px(hotspot[0]), size - px(hotspot[1]), c * 1.6, 0, Math.PI * 2);
      ctx.stroke();
    }
    sim.gnb.forEach(([x, y]) => {
      const cx = px(x);
      const cy = size - px(y);
      ctx.fillStyle = "#111";
      ctx.beginPath();
      ctx.moveTo(cx, cy - 8);
      ctx.lineTo(cx - 7, cy + 5);
      ctx.lineTo(cx + 7, cy + 5);
      ctx.closePath();
      ctx.fill();
    });
  }, [grid, sim, hotspot, camera]);

  return (
    <>
      <canvas ref={ref} width={size} height={size} style={{ width: "100%", maxWidth: size, display: "block", margin: "0 auto", borderRadius: 6 }}
              aria-label="Crowd density heatmap" />
      <div style={{ marginTop: 8 }}>
        <div style={{ height: 10, borderRadius: 3, background: `linear-gradient(90deg, ${densityColor(0)}, ${densityColor(1.5)}, ${densityColor(3)})` }} />
        <div style={{ display: "flex", justifyContent: "space-between" }} className="small muted">
          <span>0</span><span>1</span><span>{DANGER} = dangerous</span><span>{DENSITY_MAX}+ people/m²</span>
        </div>
        <div className="legend">
          <span>▲ 5G cell (gNB)</span><span>◯ hotspot</span><span>200 m × 200 m venue, 10 m zones</span>
        </div>
      </div>
    </>
  );
}

function RiskChart({ sim, t, onPick }) {
  const data = useMemo(
    () => sim.risk.map((p, i) => ({ t: i, dense: p ? p[1] : null, stampede: p ? p[2] : null })),
    [sim],
  );
  const last = sim.truth.length - 1;
  return (
    <>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -18 }}
                   onClick={(e) => e && e.activeLabel !== undefined && onPick(Number(e.activeLabel))}>
          <CartesianGrid stroke="#eef1f4" />
          <XAxis dataKey="t" type="number" domain={[0, last]} tickCount={7} fontSize={11} />
          <YAxis domain={[0, 1]} fontSize={11} />
          {segments(sim.truth, 1).map(([a, b]) => (
            <ReferenceArea key={`d${a}`} x1={a} x2={b} fill="#f0c83c" fillOpacity={0.25} />
          ))}
          {segments(sim.truth, 2).map(([a, b]) => (
            <ReferenceArea key={`s${a}`} x1={a} x2={b} fill="#d1242f" fillOpacity={0.15} />
          ))}
          <ReferenceLine x={t} stroke="#1b2430" />
          <Tooltip formatter={(v) => (v === null ? "–" : v.toFixed(2))} labelFormatter={(x) => `${x} s`} />
          <Line dataKey="dense" name="P(dense)" stroke="#b7791f" dot={false} strokeWidth={2} isAnimationActive={false} />
          <Line dataKey="stampede" name="P(stampede)" stroke="#d1242f" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="legend">
        <span><i style={{ background: "#b7791f" }} />P(dense)</span>
        <span><i style={{ background: "#d1242f" }} />P(stampede)</span>
        <span><i style={{ background: "#f0c83c", opacity: 0.5 }} />truly dense</span>
        <span><i style={{ background: "#d1242f", opacity: 0.3 }} />true stampede</span>
        <span>click to jump</span>
      </div>
    </>
  );
}

function Signal({ sim, t, name, label, color }) {
  const data = useMemo(() => sim.features[name].map((v, i) => ({ t: i, v })), [sim, name]);
  return (
    <div style={{ marginTop: 6 }}>
      <div className="small muted">{label}: <b style={{ color: "var(--ink)" }}>{sim.features[name][t].toFixed(3)}</b></div>
      <ResponsiveContainer width="100%" height={56}>
        <LineChart data={data} margin={{ top: 4, right: 10, bottom: 0, left: 10 }}>
          <XAxis dataKey="t" type="number" domain={[0, sim.truth.length - 1]} hide />
          <YAxis hide domain={["auto", "auto"]} />
          <ReferenceLine x={t} stroke="#1b2430" />
          <Line dataKey="v" stroke={color} dot={false} strokeWidth={1.6} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
