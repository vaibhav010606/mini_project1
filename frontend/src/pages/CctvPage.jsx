import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { currentUser } from "../auth.js";
import CctvFeed from "../cctv/CctvFeed.jsx";
import { emailReady, sendCameraAlert } from "../cctv/emailAlert.js";
import { CAMERAS, FLOOR_NAMES } from "../mall/cameras.js";

const pad = (n) => String(n).padStart(2, "0");
const COOLDOWN_MS = 30_000; // one email per camera every 30 s, to protect the Formspree monthly quota

// "📧 Email alert" button: emails this camera's location and crowd level through Formspree
function EmailButton({ cam, status, people, state, onState, big = false }) {
  const s = state || {};
  const coolingDown = s.state === "sent" && Date.now() - s.at < COOLDOWN_MS;
  const label =
    s.state === "sending" ? "Sending…"
      : coolingDown ? `✓ Email sent ${s.time}`
        : s.state === "error" ? "⚠ Retry email"
          : "📧 Email alert";

  async function send(e) {
    e.stopPropagation();
    if (s.state === "sending" || coolingDown) return;
    if (!emailReady()) {
      onState(cam.id, { state: "error", error: "Email is not set up yet (no Formspree form ID)." });
      return;
    }
    onState(cam.id, { state: "sending" });
    try {
      await sendCameraAlert(cam, status, people, currentUser()?.username);
      const t = new Date();
      onState(cam.id, { state: "sent", at: Date.now(), time: `${pad(t.getHours())}:${pad(t.getMinutes())}` });
    } catch (err) {
      onState(cam.id, { state: "error", error: err.message });
    }
  }

  return (
    <button type="button" className={`email-btn ${big ? "big" : ""} ${s.state || ""} ${coolingDown ? "done" : ""}`}
            onClick={send} disabled={s.state === "sending"} title={s.error || `Email the ${cam.name} alert`}>
      {label}
    </button>
  );
}

function useClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export default function CctvPage() {
  const [floor, setFloor] = useState("all");
  const [focus, setFocus] = useState(null);
  const [counts, setCounts] = useState({});
  const [statuses, setStatuses] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [emails, setEmails] = useState({});
  const onEmailState = useCallback((id, st) => setEmails((e) => ({ ...e, [id]: st })), []);
  const prevStatus = useRef({});
  const now = useClock();

  const onCount = useCallback((id, n, status = "normal") => {
    setCounts((c) => (c[id] === n ? c : { ...c, [id]: n }));
    const before = prevStatus.current[id] || "normal";
    if (status !== before) {
      prevStatus.current = { ...prevStatus.current, [id]: status };
      setStatuses((s) => ({ ...s, [id]: status }));
      const cam = CAMERAS.find((c) => c.id === id);
      const t = new Date();
      setAlerts((a) => [{ key: `${id}-${t.getTime()}`, id, name: cam?.name, status, people: n,
                          time: `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}` }, ...a].slice(0, 8));
    }
  }, []);
  const crowdedNow = Object.values(statuses).filter((s) => s === "crowded").length;
  const busyNow = Object.values(statuses).filter((s) => s === "busy").length;
  const cams = useMemo(() => CAMERAS.filter((c) => floor === "all" || c.floor === floor), [floor]);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const busiest = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  const busiestCam = busiest && CAMERAS.find((c) => c.id === busiest[0]);
  const focused = focus && CAMERAS.find((c) => c.id === focus);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && setFocus(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <div className="page-head">
        <h1>CCTV control room</h1>
        <p className="muted">
          Live view of all 12 mall cameras, as the control-room operator sees them. Click a camera to enlarge it.
          These feeds are simulated in the browser (no real footage).
        </p>
      </div>

      <div className="cctv-room">
        <div className="cctv-bar">
          <div className="cctv-clock">
            {pad(now.getHours())}:{pad(now.getMinutes())}:{pad(now.getSeconds())}
            <span>{now.toDateString()}</span>
          </div>
          <div className="cctv-kpis">
            <div><b className="ok">{CAMERAS.length}/{CAMERAS.length}</b><span>cameras online</span></div>
            <div><b><i className="rec-dot" />REC</b><span>all cameras recording</span></div>
            <div><b>{total || "–"}</b><span>people in view</span></div>
            <div><b className={crowdedNow ? "bad" : "ok"}>{crowdedNow}</b><span>crowded now{busyNow ? ` · ${busyNow} busy` : ""}</span></div>
            <div><b>{busiestCam ? busiestCam.id : "–"}</b><span>{busiestCam ? `busiest: ${busiestCam.name}` : "busiest camera"}</span></div>
          </div>
          <div className="seg cctv-seg">
            {[["all", "All floors"], [0, "Ground"], [1, "First"], [2, "Second"]].map(([v, l]) => (
              <button key={l} className={floor === v ? "on" : ""} onClick={() => setFloor(v)}>{l}</button>
            ))}
          </div>
        </div>

        {focused ? (
          <div className="cctv-focus">
            <div className="cctv-main">
              <CctvFeed cam={focused} index={CAMERAS.indexOf(focused)} scale={2} onCount={onCount} />
              <div className="cctv-info">
                <div>
                  <div className="cctv-id">{focused.id}</div>
                  <div className="cctv-name">{focused.name}</div>
                  <div className="cctv-meta">{FLOOR_NAMES[focused.floor]} floor · position ({focused.pos[0]} m, {focused.pos[1]} m)</div>
                </div>
                <div className="cctv-info-stats">
                  <div><b>{counts[focused.id] ?? "–"}</b><span>people in view</span></div>
                  <div>
                    <b className={statuses[focused.id] === "crowded" ? "bad" : statuses[focused.id] === "busy" ? "warn" : "ok"}>
                      {(statuses[focused.id] || "normal").toUpperCase()}
                    </b>
                    <span>crowd level</span>
                  </div>
                  <div><b>1080p · 12 fps</b><span>stream</span></div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <EmailButton cam={focused} status={statuses[focused.id] || "normal"} people={counts[focused.id]}
                               state={emails[focused.id]} onState={onEmailState} big />
                  <button className="logout" onClick={() => setFocus(null)}>Back to wall (Esc)</button>
                </div>
              </div>
              {emails[focused.id]?.state === "error" && <div className="email-error">{emails[focused.id].error}</div>}
            </div>
            <div className="cctv-strip">
              {cams.filter((c) => c.id !== focused.id).map((c) => (
                <button key={c.id} className="cctv-tile small" onClick={() => setFocus(c.id)}>
                  <CctvFeed cam={c} index={CAMERAS.indexOf(c)} scale={0.6} onCount={onCount} />
                  <span className="cctv-label">{c.id} · {c.name}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="cctv-grid">
            {cams.map((c) => {
              const st = statuses[c.id] || "normal";
              const em = emails[c.id];
              return (
                <div key={c.id} role="button" tabIndex={0} className={`cctv-tile ${st} ${em?.state ? "has-email" : ""}`}
                     onClick={() => setFocus(c.id)} onKeyDown={(e) => e.key === "Enter" && setFocus(c.id)}
                     title={`Enlarge ${c.id} ${c.name}`}>
                  <CctvFeed cam={c} index={CAMERAS.indexOf(c)} onCount={onCount} />
                  <EmailButton cam={c} status={st} people={counts[c.id]} state={em} onState={onEmailState} />
                  <span className="cctv-label">
                    <i className="live-dot" /> {c.id} · {c.name}
                    {st !== "normal" ? <b className={`st-badge ${st}`}>{st === "crowded" ? "CROWDED" : "BUSY"}</b> : <em>{FLOOR_NAMES[c.floor]}</em>}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        <div className="cctv-alerts">
          <div className="cctv-alerts-head">Crowd alerts <span>crowds build and clear on their own in this simulation</span></div>
          {alerts.length === 0 ? (
            <div className="cctv-alert muted-light">No alerts yet. Crowds build up at the entrance, food court, escalators and cinema over about a minute.</div>
          ) : alerts.map((a) => (
            <button key={a.key} className={`cctv-alert ${a.status}`} onClick={() => setFocus(a.id)}>
              <span className="mono">{a.time}</span>
              <b>{a.id}</b> {a.name}
              <span className={`st-badge ${a.status}`}>{a.status === "normal" ? "CLEARED" : a.status.toUpperCase()}</span>
              <span className="muted-light">{a.people} people in view</span>
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
