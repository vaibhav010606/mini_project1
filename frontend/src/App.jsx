import { useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { currentUser, logout } from "./auth.js";
import CombinedPage from "./pages/CombinedPage.jsx";
import CrowdPage from "./pages/CrowdPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import CctvPage from "./pages/CctvPage.jsx";
import MallPage from "./pages/MallPage.jsx";
import VideoPage from "./pages/VideoPage.jsx";

const PAGES = [
  { path: "/mall", label: "Smart mall 3D", element: <MallPage /> },
  { path: "/cctv", label: "CCTV wall", element: <CctvPage /> },
  { path: "/combined", label: "Combined (video + 5G)", element: <CombinedPage /> },
  { path: "/", label: "Video detection", element: <VideoPage /> },
  { path: "/crowd", label: "5G crowd monitor", element: <CrowdPage /> },
];

export default function App() {
  const [user, setUser] = useState(currentUser);

  if (!user) return <LoginPage onLogin={setUser} />;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <div className="brand-title">Crowd Safety AI</div>
            <div className="brand-sub">Video + 5G network anomaly detection at the edge</div>
          </div>
        </div>
        <nav>
          {PAGES.map((p) => (
            <NavLink key={p.path} to={p.path} end className={({ isActive }) => (isActive ? "nav active" : "nav")}>
              {p.label}
            </NavLink>
          ))}
        </nav>
        <div className="user-chip">
          <span className="user-avatar" aria-hidden="true">{user.username[0].toUpperCase()}</span>
          <span>{user.username}</span>
          <button className="logout" onClick={() => { logout(); setUser(null); }}>Log out</button>
        </div>
      </header>
      <main className="page">
        <Routes>
          {PAGES.map((p) => (
            <Route key={p.path} path={p.path} element={p.element} />
          ))}
        </Routes>
      </main>
    </>
  );
}
