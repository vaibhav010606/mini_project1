import { useState } from "react";
import { DEMO_USER, login } from "../auth.js";

const FEATURES = [
  ["📷", "Video AI", "Spots unusual movement on CCTV"],
  ["📡", "5G crowd sensing", "Feels crowd density through the network"],
  ["⚡", "Edge alerts", "Combined decision in milliseconds"],
];

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function submit(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    // short pause so it feels like a real sign-in
    setTimeout(() => {
      const user = login(username, password);
      setBusy(false);
      if (user) onLogin(user);
      else setError("Wrong username or password. Use the demo account shown below.");
    }, 500);
  }

  function fillDemo() {
    setUsername(DEMO_USER.username);
    setPassword(DEMO_USER.password);
    setError("");
  }

  return (
    <div className="login-wrap">
      <section className="login-brand">
        <div className="brand" style={{ marginBottom: 40 }}>
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-title">Crowd Safety AI</div>
            <div className="brand-sub">Video + 5G network anomaly detection at the edge</div>
          </div>
        </div>
        <h1>Detect crowd danger before it becomes a stampede</h1>
        <p className="login-lead">
          A control-room dashboard that combines CCTV video and 5G network data in one AI model, running at the 5G edge.
        </p>
        <ul className="login-features">
          {FEATURES.map(([icon, title, desc]) => (
            <li key={title}>
              <span className="login-icon" aria-hidden="true">{icon}</span>
              <span><b>{title}</b><br /><span className="muted-light">{desc}</span></span>
            </li>
          ))}
        </ul>
      </section>

      <section className="login-panel">
        <form className="login-card" onSubmit={submit} noValidate>
          <h2>Operator sign in</h2>
          <p className="muted small" style={{ marginBottom: 18 }}>Sign in to open the crowd monitoring dashboard.</p>

          <label className="field" htmlFor="lu">Username</label>
          <input id="lu" className="input" autoComplete="username" value={username}
                 onChange={(e) => setUsername(e.target.value)} placeholder="Enter username" autoFocus />

          <label className="field" htmlFor="lp">Password</label>
          <div className="pw-row">
            <input id="lp" className="input" type={show ? "text" : "password"} autoComplete="current-password"
                   value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter password" />
            <button type="button" className="pw-toggle" onClick={() => setShow(!show)} aria-label={show ? "Hide password" : "Show password"}>
              {show ? "Hide" : "Show"}
            </button>
          </div>

          {error && <div className="error" role="alert">{error}</div>}

          <button className="btn" type="submit" disabled={busy || !username || !password}>
            {busy ? <><span className="spinner" /> Signing in…</> : "Sign in"}
          </button>

          <div className="demo-box">
            <div>
              <div className="small"><b>Demo account</b></div>
              <div className="small muted">
                Username <code>{DEMO_USER.username}</code> · Password <code>{DEMO_USER.password}</code>
              </div>
            </div>
            <button type="button" className="demo-fill" onClick={fillDemo}>Use demo</button>
          </div>
          <p className="hint" style={{ marginTop: 12 }}>
            Demonstration login only: there are no real accounts and nothing is sent to a server.
          </p>
        </form>
      </section>
    </div>
  );
}
