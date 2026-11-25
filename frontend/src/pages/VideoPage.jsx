import { useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmt, getJSON, postForm } from "../api.js";

const MODES = {
  auto: "Auto: autoencoder for UCSD-style video, motion for anything else",
  ae: "AI model (Model 1): learned what normal looks like for the UCSD walkway camera",
  motion: "Crowd motion: works on any fixed camera; alerts when the crowd suddenly moves much faster",
};
const MODELS = {
  pred: "v2 future-frame prediction (AUC 0.974, best)",
  big: "v1 autoencoder (AUC 0.836)",
  student: "v1 edge model (AUC 0.777, fastest)",
};

export default function VideoPage() {
  const [videos, setVideos] = useState([]);
  const [source, setSource] = useState("ped2");
  const [videoId, setVideoId] = useState("Test004");
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("auto");
  const [model, setModel] = useState("pred");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const timer = useRef(null);

  useEffect(() => {
    getJSON("/api/test-videos").then(setVideos).catch((e) => setError(`Cannot reach the backend: ${e.message}`));
  }, []);

  const selected = videos.find((v) => v.id === videoId);
  const canRun = !busy && (source === "ped2" ? Boolean(videoId) : Boolean(file));

  async function run() {
    setBusy(true);
    setError("");
    setResult(null);
    setElapsed(0);
    const t0 = Date.now();
    timer.current = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    const fd = new FormData();
    fd.append("mode", mode);
    fd.append("model", model);
    if (source === "ped2") fd.append("video_id", videoId);
    else fd.append("file", file);
    try {
      setResult(await postForm("/api/detect", fd));
    } catch (e) {
      setError(e.message);
    } finally {
      clearInterval(timer.current);
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Video anomaly detection</h1>
        <p className="muted">Model 1 watches CCTV video and flags anything that does not look like normal crowd movement.</p>
      </div>
      <div className="grid side">
        <div className="card">
          <h2>Input</h2>
          <div className="seg" role="tablist">
            <button className={source === "ped2" ? "on" : ""} onClick={() => setSource("ped2")}>UCSD test video</button>
            <button className={source === "upload" ? "on" : ""} onClick={() => setSource("upload")}>Upload your own</button>
          </div>

          {source === "ped2" ? (
            <>
              <label className="field" htmlFor="vid">Video</label>
              <select id="vid" value={videoId} onChange={(e) => setVideoId(e.target.value)}>
                {videos.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.id} · {(v.frames / v.fps).toFixed(0)} s · {v.split === "Train" ? "normal only" : "has anomaly"}
                  </option>
                ))}
              </select>
              {selected && (
                <div className="hint">
                  {selected.anomaly_s.length
                    ? `True anomaly: ${selected.anomaly_s.map(([a, b]) => `${a}–${b} s`).join(", ")}`
                    : "Normal video (from the training set): there should be no alerts."}
                </div>
              )}
            </>
          ) : (
            <>
              <label className="field">Video file</label>
              <label className={file ? "drop has" : "drop"}>
                <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0] || null)} />
                {file ? `${file.name} (${(file.size / 1e6).toFixed(1)} MB)` : "Click to choose a video (.mp4, .avi, .mov…)"}
              </label>
              <div className="hint">Use a fixed camera. Shaky phone videos confuse both modes.</div>
            </>
          )}

          <label className="field" htmlFor="mode">Detection mode</label>
          <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="auto">Auto</option>
            <option value="ae">AI model (Model 1)</option>
            <option value="motion">Crowd motion</option>
          </select>
          <div className="hint">{MODES[mode]}</div>

          <label className="field" htmlFor="model">AI model</label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)} disabled={mode === "motion"}>
            {Object.entries(MODELS).map(([k, v]) => (
              <option key={k} value={k}>{v}</option>
            ))}
          </select>

          <button className="btn" onClick={run} disabled={!canRun}>
            {busy ? <><span className="spinner" /> Processing… {elapsed.toFixed(0)} s</> : "Detect anomalies"}
          </button>
          {error && <div className="error">{error}</div>}
        </div>

        <div>
          {!result && !busy && (
            <div className="card empty">
              <p>Choose a video and press <b>Detect anomalies</b>.</p>
              <p className="small">Results appear here: the annotated video, the anomaly score over time and the alert times.</p>
            </div>
          )}
          {busy && (
            <div className="card empty">
              <span className="spinner dark" />
              <p>Running the model on every frame… usually 10–20 seconds.</p>
            </div>
          )}
          {result && <Result r={result} />}
        </div>
      </div>
    </>
  );
}

function Result({ r }) {
  const data = useMemo(() => r.scores.map((s, i) => ({ t: +(i / r.fps).toFixed(2), score: s })), [r]);
  const truth = r.truth_anomaly_s || [];
  const first = r.alerts[0]?.[0];
  const delay = truth.length && first !== undefined ? first - truth[0][0] : null;
  const tEnd = data[data.length - 1].t;
  const clip = (x) => Math.min(x, tEnd); // shaded areas must stay inside the x-axis or the chart drops them
  const yMin = Math.floor(Math.min(0, ...r.scores));
  const yMax = Math.ceil(Math.max(r.threshold * 1.5, ...r.scores));

  return (
    <>
      <div className="stats">
        <div className={r.alerts.length ? "stat bad" : "stat good"}>
          <div className="v">{r.alerts.length ? "ANOMALY" : "Normal"}</div>
          <div className="k">{r.alerts.length} alert period{r.alerts.length === 1 ? "" : "s"}</div>
        </div>
        <div className="stat">
          <div className="v">{first !== undefined ? `${first} s` : "–"}</div>
          <div className="k">first alert</div>
        </div>
        {truth.length > 0 && (
          <div className="stat">
            <div className="v">{delay !== null ? `${delay >= 0 ? "+" : ""}${delay.toFixed(1)} s` : "missed"}</div>
            <div className="k">vs true start ({truth[0][0]} s)</div>
          </div>
        )}
        <div className="stat">
          <div className="v">{(r.alert_fraction * 100).toFixed(0)}%</div>
          <div className="k">of frames flagged</div>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Annotated video</h3>
          <video key={r.video_url} src={r.video_url} controls autoPlay muted loop />
          <p className="hint">
            {r.source} · {r.n_frames} frames · {r.fps} fps · mode: {r.mode === "ae" ? `AI model (${r.model})` : "crowd motion"}
          </p>
        </div>
        <div className="card">
          <h3>Most unusual moment ({r.peak_time} s)</h3>
          <img className="frame" src={r.peak_url} alt="Frame with the highest anomaly score" />
          <h3 style={{ marginTop: 16 }}>Alert periods</h3>
          {r.alerts.length ? (
            <table className="simple">
              <thead><tr><th>From</th><th>To</th><th>Length</th></tr></thead>
              <tbody>
                {r.alerts.map(([a, b]) => (
                  <tr key={a}><td>{a} s</td><td>{b} s</td><td>{(b - a).toFixed(1)} s</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted small">No alerts: the whole video looks normal to the model.</p>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 18 }}>
        <h3>Anomaly score over time</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#eef1f4" />
            <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tickCount={10}
                   label={{ value: "time (s)", position: "insideBottom", offset: -8, fontSize: 12 }} />
            <YAxis domain={[yMin, yMax]} allowDecimals={false} width={40} />
            {r.alerts.map(([a, b]) => (
              <ReferenceArea key={`a${a}`} x1={a} x2={clip(b)} y1={yMin} y2={yMax} fill="#d1242f" fillOpacity={0.12} />
            ))}
            {truth.map(([a, b]) => (
              <ReferenceArea key={`t${a}`} x1={a} x2={clip(b)} y1={yMax - (yMax - yMin) * 0.05} y2={yMax}
                             fill="#1b2430" fillOpacity={0.55} />
            ))}
            <ReferenceLine y={r.threshold} stroke="#d1242f" strokeDasharray="5 4" />
            <Tooltip formatter={(v) => fmt(v, 2)} labelFormatter={(t) => `${t} s`} />
            <Line type="monotone" dataKey="score" stroke="#1f6feb" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="legend">
          <span><i style={{ background: "#1f6feb" }} />anomaly score</span>
          <span><i style={{ background: "#d1242f", opacity: 0.7 }} />alert threshold / alert period</span>
          {truth.length > 0 && <span><i style={{ background: "#1b2430", opacity: 0.55 }} />true anomaly (dataset label)</span>}
        </div>
      </div>
    </>
  );
}
