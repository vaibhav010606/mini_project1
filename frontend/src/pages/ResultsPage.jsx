import { useEffect, useState } from "react";
import { fmt, getJSON, ms, pct } from "../api.js";

const VNAME = { pred: "v2 future-frame prediction (final, cloud)", big: "v1 autoencoder", student: "v1 student (edge)" };
const EDGE = "Edge - lightweight model";
const CLOUD = "Cloud - accurate model (GPU)";

export default function ResultsPage() {
  const [d, setD] = useState(null);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(null);

  useEffect(() => {
    getJSON("/api/results").then(setD).catch((e) => setError(`Cannot load results: ${e.message}`));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!d) return <div className="card empty"><span className="spinner dark" /><p>Loading results…</p></div>;

  const fig = (name) => d.figures[name];
  const Fig = ({ name, caption }) =>
    fig(name) ? (
      <figure className="fig" onClick={() => setZoom({ src: fig(name), caption })}>
        <img src={fig(name)} alt={caption} loading="lazy" />
        <figcaption>{caption} <span className="muted">· click to enlarge</span></figcaption>
      </figure>
    ) : null;

  const v = d.video || {};
  const net = d.network || {};
  const fu = d.fusion || {};
  const off = d.offload || {};
  const cams = d.edge_cloud?.cameras;
  const camAt = (setup, n) => cams?.results?.[setup]?.[cams.n.indexOf(n)]?.mean_ms;
  const sl = d.edge_cloud?.slicing;
  const sliceAt = (name, load) => sl?.results?.[name]?.[sl.bg_load.indexOf(load)]?.p95_ms;
  const videoBest = v.pred?.auc?.patch_max_norm_smooth ?? v.big?.auc?.patch_max_norm_smooth;
  const jm = d.joint || {};
  const JMODELS = ["Video only", "5G only", "Combined (video + 5G)"];
  const jall = (m) => jm.results?.[m]?.["All test events"];

  return (
    <>
      <div className="page-head">
        <h1>Results</h1>
        <p className="muted">All numbers come from the saved results of the trained models and simulations (the same files the report is built from).</p>
      </div>

      <div className="stats kpi-row">
        <Kpi v={fmt(videoBest)} k="Video model AUC (UCSD Ped2)" />
        <Kpi v={pct(net.xgboost?.accuracy)} k="5G network model accuracy" />
        <Kpi v={fmt(jall("Combined (video + 5G)")?.macro_f1)} k="Combined video + 5G model (macro-F1)" />
        <Kpi v={`${ms(camAt(EDGE, 10))} vs ${ms(camAt(CLOUD, 10))}`} k="Alert delay, edge vs cloud (10 cams)" small />
        <Kpi v={pct(off.ml_accuracy)} k="ML edge/cloud decision correct" />
      </div>

      <Section n="1" title="Model 1: video anomaly detection" note="Real data: UCSD Ped2 (16 normal training videos, 12 test videos).">
        <table className="simple">
          <thead><tr><th>Model</th><th>AUC whole frame</th><th>AUC worst patch</th><th>AUC final</th><th>Edge CPU</th><th>Cloud GPU</th><th>Size</th></tr></thead>
          <tbody>
            {["pred", "big", "student"].filter((m) => v[m]).map((m) => {
              const a = v[m].auc;
              const b = d.benchmark?.models?.[m] || {};
              return (
                <tr key={m}>
                  <td>{VNAME[m]}</td>
                  <td>{fmt(a.global_psnr_raw ?? a.global_mse)}</td>
                  <td>{fmt(a.patch_max)}</td>
                  <td><b>{fmt(a.patch_max_norm_smooth)}</b></td>
                  <td>{ms(b.cpu_ms)}</td>
                  <td>{ms(b.gpu_ms)}</td>
                  <td>{b.size_mb ? `${b.size_mb.toFixed(1)} MB` : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="figs">
          <Fig name="video_roc" caption="ROC curves: v2 clearly beats v1" />
          <Fig name="video_examples" caption="Where the model sees the anomaly" />
          <Fig name="video_score_curves" caption="Anomaly score over time on 4 test videos" />
          <Fig name="video_benchmark" caption="Accuracy vs size vs speed" />
        </div>
      </Section>

      <Section n="2" title="Model 2: crowd risk from 5G network data" note="Simulated data: 450 crowd episodes over a 9-cell 5G venue (real operator data is private).">
        <table className="simple">
          <thead><tr><th>Detector</th><th>Accuracy</th><th>Macro-F1</th><th>Stampede detection delay (median)</th><th>Stampedes missed</th></tr></thead>
          <tbody>
            {[["rule", "Rule baseline"], ["xgboost", "XGBoost (final)"], ["lstm", "LSTM"]].filter(([k]) => net[k]).map(([k, label]) => (
              <tr key={k}>
                <td>{label}</td><td>{pct(net[k].accuracy)}</td><td><b>{fmt(net[k].macro_f1)}</b></td>
                <td>{net[k].stampede_delay_median_s} s</td><td>{net[k].stampede_episodes_missed} / {net[k].stampede_episodes_total}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="figs">
          <Fig name="net_heatmaps" caption="Density map: true vs estimated from 5G" />
          <Fig name="net_timeline" caption="One stampede seen through 5G data" />
          <Fig name="net_confusion_xgb" caption="XGBoost confusion matrix" />
          <Fig name="net_feature_importance" caption="Most useful network signals" />
        </div>
      </Section>

      {jm.results && (
        <Section n="3" title="Combined model: video + 5G of the same event"
                 note={`Joint simulated data: each event is filmed by a CCTV camera and sensed by the 5G network at the same time. ${jm.test_events?.total} unseen test events (${jm.test_events?.night} at night, ${jm.test_events?.outside_view} with the danger outside the camera view, ${jm.test_events?.few_phones} with few phones visible).`}>
          <table className="simple">
            <thead><tr><th>Test condition (macro-F1)</th>{JMODELS.map((m) => <th key={m}>{m}</th>)}</tr></thead>
            <tbody>
              {Object.keys(jm.conditions || {}).map((c) => (
                <tr key={c}>
                  <td>{c}</td>
                  {JMODELS.map((m) => (
                    <td key={m}>{m.startsWith("Combined") ? <b>{fmt(jm.results[m][c].macro_f1)}</b> : fmt(jm.results[m][c].macro_f1)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <table className="simple" style={{ marginTop: 12 }}>
            <thead><tr><th>All test events</th>{JMODELS.map((m) => <th key={m}>{m}</th>)}</tr></thead>
            <tbody>
              <tr><td>Dangerous moments detected</td>{JMODELS.map((m) => <td key={m}>{pct(jall(m)?.danger_detected)}</td>)}</tr>
              <tr><td>False alarms</td>{JMODELS.map((m) => <td key={m}>{pct(jall(m)?.false_alarm)}</td>)}</tr>
              <tr><td>Stampede detection delay (median)</td>{JMODELS.map((m) => <td key={m}>{jm.stampede_delay?.[m]?.median_s} s</td>)}</tr>
              <tr><td>Stampedes missed</td>{JMODELS.map((m) => <td key={m}>{jm.stampede_delay?.[m]?.missed} / {jm.stampede_delay?.[m]?.total}</td>)}</tr>
            </tbody>
          </table>
          <p className="hint" style={{ marginTop: 8 }}>
            Night has only {jm.test_events?.night} test events, so its macro-F1 is unreliable; the combined model still detects
            {" "}{pct(jm.results["Combined (video + 5G)"]["Night (dark video)"]?.danger_detected)} of dangerous moments at night with
            {" "}{pct(jm.results["Combined (video + 5G)"]["Night (dark video)"]?.false_alarm)} false alarms.
          </p>
          <div className="figs">
            <Fig name="joint_by_condition" caption="Each source has a blind spot; combined covers it" />
            <Fig name="joint_video_preview" caption="CCTV frames rendered from the same crowd as the 5G data" />
            <Fig name="joint_confusion" caption="Combined model confusion matrix" />
          </div>
        </Section>
      )}

      <Section n="3b" title="Earlier baseline: score-level fusion" note={`Separately trained models, outputs paired artificially: the camera sees ${pct(fu.setup?.p_cam_sees, 0)} of incidents, the network ${pct(fu.setup?.p_net_sees, 0)}. Superseded by the combined model above.`}>
        <table className="simple">
          <thead><tr><th>Method</th><th>AUC</th><th>Detected at 5% false alarms</th><th>Caught when camera blind</th></tr></thead>
          <tbody>
            {Object.entries(fu).filter(([k]) => k !== "setup").map(([k, r]) => (
              <tr key={k} className={k.includes("learned") ? "hl" : ""}>
                <td>{k}</td><td><b>{fmt(r.auc)}</b></td><td>{pct(r.tpr_at_5pct_fpr)}</td><td>{pct(r.caught_when_camera_blind)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="figs one"><Fig name="fusion_results" caption="Earlier baseline: score-level fusion" /></div>
      </Section>

      <Section n="4" title="Edge vs cloud over 5G, and network slicing" note="Discrete-event 5G simulation using the AI speeds measured on this laptop.">
        <table className="simple">
          <thead><tr><th>Cameras</th><th>Edge, small model</th><th>Cloud, v2 model</th></tr></thead>
          <tbody>
            {[1, 10, 40, 80, 120].filter((n) => cams?.n.includes(n)).map((n) => (
              <tr key={n}><td>{n}</td><td>{ms(camAt(EDGE, n))}</td><td>{ms(camAt(CLOUD, n))}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="small" style={{ marginTop: 10 }}>
          Slicing, 20 cameras at 90% public load (95% of alerts): <b>{ms(sliceAt("URLLC slice (priority)", 0.9))}</b> with a URLLC slice vs{" "}
          <b>{ms(sliceAt("No slicing (shared)", 0.9))}</b> without.
        </p>
        <div className="figs">
          <Fig name="edge_vs_cloud_cameras" caption="Delay as cameras are added" />
          <Fig name="slicing_effect" caption="URLLC slice protects safety traffic" />
        </div>
      </Section>

      <Section n="5" title="ML decision: edge or cloud?" note="500 random network situations, 30% held out for testing.">
        <table className="simple">
          <thead><tr><th>Policy</th><th>Average 95th-percentile delay</th><th>Situations over 100 ms</th></tr></thead>
          <tbody>
            {Object.entries(off).filter(([, r]) => typeof r === "object").map(([k, r]) => (
              <tr key={k} className={k === "ML decides" ? "hl" : ""}><td>{k}</td><td>{ms(r.mean_p95_ms)}</td><td>{pct(r.miss_rate)}</td></tr>
            ))}
          </tbody>
        </table>
        <div className="figs">
          <Fig name="offload_policies" caption="ML comes close to the best possible choice" />
          <Fig name="offload_importance" caption="What drives the decision" />
        </div>
      </Section>

      <div className="card notes">
        <h3>What is real and what is simulated</h3>
        <ul>
          <li><b>Real:</b> UCSD Ped2 video, the trained video models and their accuracy, and the AI speeds measured on this laptop.</li>
          <li><b>Simulated:</b> the 5G network data for Model 2, the 5G edge/cloud network (Python, same concepts as Simu5G) and network slicing.</li>
          <li><b>Simulated joint events:</b> the combined model sees CCTV video and 5G data of the same simulated events; the video is rendered from simulated people, because no public dataset has CCTV and 5G data of the same crowd.</li>
          <li><b>Paired synthetically (earlier baseline only):</b> the score-level fusion combines real outputs of two separately trained models.</li>
          <li><b>Privacy:</b> the network model only uses counts per zone; no individual is tracked.</li>
        </ul>
      </div>

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)} role="dialog" aria-label={zoom.caption}>
          <img src={zoom.src} alt={zoom.caption} />
          <div>{zoom.caption} · click anywhere to close</div>
        </div>
      )}
    </>
  );
}

function Kpi({ v, k, small }) {
  return (
    <div className="stat">
      <div className="v" style={small ? { fontSize: 17 } : null}>{v}</div>
      <div className="k">{k}</div>
    </div>
  );
}

function Section({ n, title, note, children }) {
  return (
    <section className="card result-section">
      <h2><span className="num">{n}</span>{title}</h2>
      {note && <p className="hint" style={{ marginTop: -6, marginBottom: 12 }}>{note}</p>}
      {children}
    </section>
  );
}
