import { useMemo, useState } from "react";
import Mall3D from "../mall/Mall3D.jsx";
import { CATEGORIES, FLOORS, area } from "../mall/layout.js";

const VIEWS = [["all", "All floors"], [0, "Ground"], [1, "First"], [2, "Second"]];
const PRESETS = [["overview", "Overview"], ["top", "Top view"], ["entrance", "Entrance"], ["atrium", "Atrium"],
                 ["exterior", "Outside"]];

export default function MallPage() {
  const [view, setView] = useState("all");
  const [exploded, setExploded] = useState(false);
  const [labels, setLabels] = useState(true);
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [preset, setPreset] = useState(null);
  const [night, setNight] = useState(false);
  const [building, setBuilding] = useState("cutaway");

  const stats = useMemo(() => {
    const units = FLOORS.flatMap((f) => f.units);
    return { shops: units.length, gla: units.reduce((s, u) => s + area(u.rect), 0) };
  }, []);

  const hoveredName = hovered ? hovered.split(":")[1] : null;

  return (
    <>
      <div className="page-head">
        <h1>Smart mall: 3D digital twin</h1>
        <p className="muted">
          A 3D model of the mall for the control-room operator. Drag to rotate, scroll to zoom, right-drag to pan. Click a
          shop for details. The live crowd heatmap will be added on top of this model next.
        </p>
      </div>

      <div className="grid side">
        <div className="card">
          <h2>View</h2>
          <label className="field">Floor</label>
          <div className="seg">
            {VIEWS.map(([v, l]) => (
              <button key={l} className={view === v ? "on" : ""} onClick={() => { setView(v); setSelected(null); }}>{l}</button>
            ))}
          </div>
          <div className="hint">Choosing a floor hides the floors above it so you can look inside.</div>

          <label className="field">Building</label>
          <div className="seg">
            <button className={building === "cutaway" ? "on" : ""} onClick={() => setBuilding("cutaway")}>See inside</button>
            <button className={building === "full" ? "on" : ""} onClick={() => setBuilding("full")}>Full building</button>
          </div>
          <div className="hint">"Full building" shows the outside walls, roof and the SMART MALL sign.</div>

          <label className="field">Time of day</label>
          <div className="seg">
            <button className={!night ? "on" : ""} onClick={() => setNight(false)}>Day</button>
            <button className={night ? "on" : ""} onClick={() => setNight(true)}>Night</button>
          </div>

          <label className="toggle">
            <input type="checkbox" checked={exploded} onChange={(e) => setExploded(e.target.checked)} />
            <span><b>Exploded view</b><span className="hint" style={{ display: "block", marginTop: 2 }}>Pull the floors apart to see every level.</span></span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />
            <span><b>Shop names</b></span>
          </label>

          <label className="field">Camera</label>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            {PRESETS.map(([k, l]) => (
              <button key={k} className="preset" onClick={() => { if (k === "exterior") setBuilding("full"); setPreset({ name: k, t: Date.now() }); }}><b>{l}</b></button>
            ))}
          </div>

          <h3 style={{ marginTop: 18 }}>Legend</h3>
          <div className="mall-legend">
            {Object.values(CATEGORIES).map((c) => (
              <span key={c.label}><i style={{ background: c.color }} />{c.label}</span>
            ))}
          </div>

          {selected ? (
            <div className="shop-card">
              <div className="small muted">{FLOORS[selected.floorId].name}</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{selected.name}</div>
              <div className="small" style={{ marginTop: 4 }}>
                <span className="badge" style={{ background: CATEGORIES[selected.cat].color, color: "#1b2430" }}>
                  {CATEGORIES[selected.cat].label}
                </span>
                <span className="muted" style={{ marginLeft: 8 }}>{area(selected.rect).toLocaleString()} mÂ²</span>
              </div>
              <div className="hint" style={{ marginTop: 6 }}>Crowd numbers for this shop will appear here once the heatmap is added.</div>
            </div>
          ) : (
            <p className="hint" style={{ marginTop: 14 }}>{hoveredName ? `Pointing at: ${hoveredName}` : "Click any shop to see its details."}</p>
          )}
        </div>

        <div>
          <div className="stats">
            <div className="stat"><div className="v">3</div><div className="k">floors</div></div>
            <div className="stat"><div className="v">{stats.shops}</div><div className="k">shops and anchors</div></div>
            <div className="stat"><div className="v">{(stats.gla / 1000).toFixed(1)}k mÂ²</div><div className="k">shop floor area</div></div>
            <div className="stat"><div className="v">4 + 2</div><div className="k">escalators + lifts</div></div>
          </div>
          <div className="card mall-canvas">
            <Mall3D view={view} exploded={exploded} showLabels={labels} hovered={hovered} selected={selected?.id}
                    onHover={setHovered} onSelect={setSelected} preset={preset} night={night} building={building} />
          </div>
        </div>
      </div>
    </>
  );
}
