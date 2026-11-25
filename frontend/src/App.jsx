import { NavLink, Route, Routes } from "react-router-dom";
import CombinedPage from "./pages/CombinedPage.jsx";
import CrowdPage from "./pages/CrowdPage.jsx";
import EdgePage from "./pages/EdgePage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import VideoPage from "./pages/VideoPage.jsx";

const PAGES = [
  { path: "/combined", label: "Combined (video + 5G)", element: <CombinedPage /> },
  { path: "/", label: "Video detection", element: <VideoPage /> },
  { path: "/crowd", label: "5G crowd monitor", element: <CrowdPage /> },
  { path: "/edge", label: "Edge vs cloud", element: <EdgePage /> },
  { path: "/results", label: "Results", element: <ResultsPage /> },
];

export default function App() {
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
