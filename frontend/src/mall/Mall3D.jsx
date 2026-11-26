import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Edges, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { ATRIUM, CATEGORIES, ESCALATORS, FLOORS, LIFTS, MALL } from "./layout.js";

// Realistic smart-mall digital twin, generated entirely in code (no downloaded models or textures).
const HALF_W = MALL.w / 2;
const HALF_D = MALL.d / 2;
const FH = MALL.floorHeight;
const SLAB = MALL.slab;
const ROOM_H = FH - SLAB; // 5.6 m floor-to-ceiling
const GLASS_H = 4.0; // shopfront glass height
const TOP = FLOORS.length - 1;

export const CAMERA_PRESETS = {
  overview: { pos: [92, 84, -112], target: [0, 4, 0] },
  top: { pos: [0, 150, -0.01], target: [0, 0, 0] },
  entrance: { pos: [0, 9, -78], target: [0, 6, -10] },
  atrium: { pos: [30, 26, -26], target: [0, 6, 0] },
  exterior: { pos: [78, 30, -98], target: [0, 9, -10] },
};

export function floorY(i, exploded) {
  return i * (FH + (exploded ? 7 : 0));
}

// ---------------------------------------------------------------- small helpers
function rngFrom(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  let s = h >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function shade(hex, f) {
  const c = new THREE.Color(hex);
  return `#${c.multiplyScalar(f).getHexString()}`;
}

// ---------------------------------------------------------------- procedural textures (canvas)
const TEX = {};
function canvasTex(key, w, h, draw, repeat) {
  if (TEX[key]) return TEX[key];
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(repeat[0], repeat[1]);
  }
  TEX[key] = t;
  return t;
}

const marbleTex = () => canvasTex("marble", 1024, 1024, (ctx, w, h) => {
  const r = rngFrom("marble");
  ctx.fillStyle = "#eeeae3";
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 5000; i++) {
    const g = 170 + Math.floor(r() * 50);
    ctx.fillStyle = `rgba(${g},${g - 6},${g - 14},0.07)`;
    ctx.fillRect(r() * w, r() * h, 2 + r() * 7, 2 + r() * 7);
  }
  for (let i = 0; i < 26; i++) {
    ctx.strokeStyle = `rgba(135,125,115,${0.05 + r() * 0.12})`;
    ctx.lineWidth = 0.6 + r() * 2.4;
    ctx.beginPath();
    let x = r() * w;
    let y = r() * h;
    ctx.moveTo(x, y);
    for (let k = 0; k < 4; k++) {
      const nx = x + (r() - 0.5) * 420;
      const ny = y + (r() - 0.5) * 420;
      ctx.quadraticCurveTo(x + (r() - 0.5) * 300, y + (r() - 0.5) * 300, nx, ny);
      x = nx;
      y = ny;
    }
    ctx.stroke();
  }
  ctx.strokeStyle = "#d3ccbf";
  ctx.lineWidth = 5;
  for (const p of [0, w / 2]) {
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(w, p); ctx.stroke();
  }
}, [1 / 2.4, 1 / 2.4]);

const ceilingTex = () => canvasTex("ceiling", 256, 256, (ctx, w, h) => {
  ctx.fillStyle = "#f4f4f1";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#dcdcd6";
  ctx.lineWidth = 3;
  for (const p of [0, w / 2]) {
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(w, p); ctx.stroke();
  }
}, [1 / 1.2, 1 / 1.2]);

const noiseTex = (key, base, spread, repeat) => canvasTex(key, 512, 512, (ctx, w, h) => {
  const r = rngFrom(key);
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 9000; i++) {
    const v = Math.floor((r() - 0.5) * spread);
    ctx.fillStyle = `rgba(${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${v > 0 ? 255 : 0},${Math.abs(v) / 255})`;
    ctx.fillRect(r() * w, r() * h, 2, 2);
  }
}, repeat);

function signTex(text, bg, fg = "#ffffff") {
  return canvasTex(`sign|${text}|${bg}|${fg}`, 1024, 192, (ctx, w, h) => {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    let size = 104;
    ctx.font = `700 ${size}px Segoe UI, Arial`;
    while (ctx.measureText(text).width > w * 0.88 && size > 40) {
      size -= 4;
      ctx.font = `700 ${size}px Segoe UI, Arial`;
    }
    ctx.fillStyle = fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, w / 2, h / 2 + 4);
  });
}

const logoTex = () => canvasTex("logo", 2048, 320, (ctx, w, h) => {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "800 210px Segoe UI, Arial";
  ctx.fillText("SMART MALL", w / 2, h / 2 + 10);
});

// shared materials (created once, reused by many meshes)
const MAT = {};
function mats() {
  if (MAT.glass) return MAT;
  MAT.glass = new THREE.MeshStandardMaterial({ color: "#d4e8f4", transparent: true, opacity: 0.26, roughness: 0.04,
                                               metalness: 0.3, depthWrite: false, envMapIntensity: 1.4 });
  MAT.alu = new THREE.MeshStandardMaterial({ color: "#8f98a3", roughness: 0.35, metalness: 0.8 });
  MAT.dark = new THREE.MeshStandardMaterial({ color: "#2b3038", roughness: 0.5, metalness: 0.4 });
  MAT.wall = new THREE.MeshStandardMaterial({ color: "#f2efe9", roughness: 0.85 });
  MAT.white = new THREE.MeshStandardMaterial({ color: "#fbfbf9", roughness: 0.5 });
  MAT.wood = new THREE.MeshStandardMaterial({ color: "#b48a62", roughness: 0.55 });
  MAT.stone = new THREE.MeshStandardMaterial({ color: "#d9d3c8", roughness: 0.75 });
  MAT.leaf = new THREE.MeshStandardMaterial({ color: "#4e8a4a", roughness: 0.9, flatShading: true });
  MAT.leaf2 = new THREE.MeshStandardMaterial({ color: "#3f7a42", roughness: 0.9, flatShading: true });
  MAT.trunk = new THREE.MeshStandardMaterial({ color: "#7a5a3e", roughness: 0.9 });
  MAT.hit = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false });
  return MAT;
}

// ---------------------------------------------------------------- floating name labels (optional)
const labelCache = new Map();
function labelTexture(text, dot, big = false) {
  const key = `${text}|${dot}|${big}`;
  if (labelCache.has(key)) return labelCache.get(key);
  const font = big ? "bold 64px Segoe UI, Arial" : "600 40px Segoe UI, Arial";
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = font;
  const tw = ctx.measureText(text).width;
  const padX = big ? 36 : 26;
  const dotW = dot ? 34 : 0;
  c.width = Math.ceil(tw + padX * 2 + dotW);
  c.height = big ? 104 : 68;
  ctx.font = font;
  ctx.fillStyle = big ? "rgba(18,38,63,0.92)" : "rgba(255,255,255,0.95)";
  ctx.beginPath();
  ctx.roundRect(0, 0, c.width, c.height, c.height / 2);
  ctx.fill();
  if (dot) {
    ctx.fillStyle = dot;
    ctx.beginPath();
    ctx.arc(padX + 10, c.height / 2, 11, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = big ? "#ffffff" : "#1b2430";
  ctx.textBaseline = "middle";
  ctx.fillText(text, padX + dotW, c.height / 2 + 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const out = { tex, aspect: c.width / c.height };
  labelCache.set(key, out);
  return out;
}

function Label({ text, position, dot, height = 2.4, big = false }) {
  const { tex, aspect } = useMemo(() => labelTexture(text, dot, big), [text, dot, big]);
  return (
    // constant size on screen (does not grow huge when the camera is close)
    <sprite position={position} scale={[height * 0.0105 * aspect, height * 0.0105, 1]} renderOrder={10}>
      <spriteMaterial map={tex} transparent depthWrite={false} depthTest={false} sizeAttenuation={false} />
    </sprite>
  );
}

// ---------------------------------------------------------------- environment and lights
function SceneEnv({ night }) {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = env;
    return () => {
      scene.environment = null;
      env.dispose();
      pmrem.dispose();
    };
  }, [gl, scene]);
  useEffect(() => {
    scene.environmentIntensity = night ? 0.12 : 0.7;
    scene.fog = new THREE.Fog(night ? "#0a1220" : "#dce8f3", 220, 520);
  }, [night, scene]);
  return null;
}

function Lights({ night, visibleFloors, exploded }) {
  return (
    <>
      <hemisphereLight args={night ? ["#3a4a6a", "#0a0d14", 0.45] : ["#e8f2ff", "#b5a68f", 0.9]} />
      <directionalLight position={night ? [-60, 90, 40] : [85, 120, -75]} intensity={night ? 0.35 : 2.8}
                        color={night ? "#a9bcff" : "#fff3df"} castShadow={!night}
                        shadow-mapSize={[2048, 2048]} shadow-bias={-0.0004} shadow-normalBias={0.04}
                        shadow-camera-left={-120} shadow-camera-right={120} shadow-camera-top={120} shadow-camera-bottom={-120}
                        shadow-camera-far={400} />
      {night && visibleFloors.map((f) => {
        const y = floorY(f.id, exploded) + 4.6;
        return [[-16, 0], [16, 0], [0, -11], [0, 11]].map(([x, z]) => (
          <pointLight key={`${f.id}${x}${z}`} position={[x, y, z]} color="#ffdcae" intensity={160} distance={30} decay={1.6} />
        ));
      })}
      {night && [[-10, -HALF_D - 10], [10, -HALF_D - 10]].map(([x, z]) => (
        <pointLight key={`p${x}`} position={[x, 7, z]} color="#ffd59a" intensity={260} distance={38} decay={1.6} />
      ))}
    </>
  );
}

// ---------------------------------------------------------------- floor slab, ceiling and ceiling lights
function slabShape(withHole, flip) {
  const s = new THREE.Shape();
  s.moveTo(-HALF_W, -HALF_D);
  s.lineTo(HALF_W, -HALF_D);
  s.lineTo(HALF_W, HALF_D);
  s.lineTo(-HALF_W, HALF_D);
  s.closePath();
  if (withHole) {
    const [x0, z0, x1, z1] = ATRIUM;
    const k = flip ? 1 : -1;
    const h = new THREE.Path();
    h.moveTo(x0, k * z1);
    h.lineTo(x1, k * z1);
    h.lineTo(x1, k * z0);
    h.lineTo(x0, k * z0);
    h.closePath();
    s.holes.push(h);
  }
  return s;
}

function Slab({ y, withHole, roof = false }) {
  const geom = useMemo(() => new THREE.ExtrudeGeometry(slabShape(withHole, false), { depth: SLAB, bevelEnabled: false }), [withHole]);
  const materials = useMemo(() => [
    roof ? new THREE.MeshStandardMaterial({ map: noiseTex("roof", "#b9bec4", 40, [1 / 6, 1 / 6]), roughness: 0.95 })
      : new THREE.MeshStandardMaterial({ map: marbleTex(), roughness: 0.22, metalness: 0.05, envMapIntensity: 0.9 }),
    new THREE.MeshStandardMaterial({ color: "#cfd3d8", roughness: 0.8 }),
  ], [roof]);
  return (
    <mesh geometry={geom} material={materials} rotation={[-Math.PI / 2, 0, 0]} position={[0, y - SLAB, 0]} receiveShadow castShadow />
  );
}

const LIGHT_SPOTS = (() => {
  const pts = [];
  for (let x = -21; x <= 21; x += 3.5) {
    for (let z = -14.5; z <= 14.5; z += 3.5) {
      const inAtrium = Math.abs(x) < ATRIUM[2] + 0.8 && Math.abs(z) < ATRIUM[3] + 0.8;
      if (!inAtrium) pts.push([x, z]);
    }
  }
  return pts;
})();

function Ceiling({ y, night }) {
  // underside of the slab at height y (its top surface), seen from the floor below
  const geom = useMemo(() => new THREE.ShapeGeometry(slabShape(true, true)), []);
  const lightsRef = useRef();
  useLayoutEffect(() => {
    const m = new THREE.Matrix4();
    LIGHT_SPOTS.forEach(([x, z], i) => {
      m.makeRotationX(Math.PI / 2).setPosition(x, y - SLAB - 0.04, z);
      lightsRef.current.setMatrixAt(i, m);
    });
    lightsRef.current.instanceMatrix.needsUpdate = true;
  }, [y]);
  return (
    <group>
      <mesh geometry={geom} rotation={[Math.PI / 2, 0, 0]} position={[0, y - SLAB - 0.02, 0]}>
        <meshStandardMaterial map={ceilingTex()} roughness={0.9} />
      </mesh>
      <instancedMesh ref={lightsRef} args={[null, null, LIGHT_SPOTS.length]}>
        <planeGeometry args={[1.4, 0.6]} />
        <meshStandardMaterial color="#ffffff" emissive="#fff4e2" emissiveIntensity={night ? 2.2 : 0.9} side={THREE.DoubleSide} />
      </instancedMesh>
    </group>
  );
}

// ---------------------------------------------------------------- shops
function frameOf([x0, z0, x1, z1]) {
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;
  if (z0 >= 15.9) return { cx, cz, W: x1 - x0, D: z1 - z0, rot: 0 }; // north row, front faces the centre (-z)
  if (z1 <= -15.9) return { cx, cz, W: x1 - x0, D: z1 - z0, rot: Math.PI }; // south row
  if (x1 <= -22.9) return { cx, cz, W: z1 - z0, D: x1 - x0, rot: -Math.PI / 2 }; // west wing
  return { cx, cz, W: z1 - z0, D: x1 - x0, rot: Math.PI / 2 }; // east wing
}

function Interior({ unit, w, d, night }) {
  const r = rngFrom(unit.name);
  const cat = CATEGORIES[unit.cat].color;
  const items = [];
  const box = (k, pos, size, mat) => items.push(
    <mesh key={k} position={pos} castShadow receiveShadow material={mat}><boxGeometry args={size} /></mesh>);
  const M = mats();
  if (unit.cat === "anchor") {
    const rows = Math.floor((d - 4) / 2.6);
    for (let i = 0; i < rows; i++) box(`r${i}`, [0, 0.9, -d / 2 + 3.2 + i * 2.6], [w * 0.72, 1.8, 0.8], M.white);
    for (let i = 0; i < 4; i++) box(`c${i}`, [-w / 2 + 2 + i * 1.6, 0.55, -d / 2 + 1.4], [1.2, 1.1, 0.7], M.dark);
  } else if (unit.cat === "entertainment") {
    box("scr", [0, 2.6, d / 2 - 0.3], [w * 0.75, 2.4, 0.08],
        new THREE.MeshStandardMaterial({ color: "#1d2a4d", emissive: "#4c7dff", emissiveIntensity: night ? 1.4 : 0.6 }));
    for (let i = 0; i < 3; i++) box(`m${i}`, [-w / 3 + i * (w / 3), 0.9, 0], [1.4, 1.8, 1.0], M.dark);
  } else if (unit.cat === "food") {
    box("ctr", [0, 0.55, d / 2 - 1.4], [w * 0.7, 1.1, 0.9], M.wood);
    for (let i = 0; i < 3; i++) {
      const x = -w / 3 + i * (w / 3);
      items.push(
        <group key={`t${i}`} position={[x, 0, -0.5]}>
          <mesh position={[0, 0.75, 0]} castShadow material={M.white}><cylinderGeometry args={[0.55, 0.55, 0.06, 20]} /></mesh>
          <mesh position={[0, 0.37, 0]} material={M.alu}><cylinderGeometry args={[0.05, 0.05, 0.74, 8]} /></mesh>
        </group>);
    }
  } else {
    // shelves along the walls + display tables (+ mannequins for fashion)
    box("sl", [-w / 2 + 0.55, 1.0, 0.6], [0.6, 2.0, d * 0.62], M.white);
    box("sr", [w / 2 - 0.55, 1.0, 0.6], [0.6, 2.0, d * 0.62], M.white);
    const n = w > 13 ? 3 : 2;
    for (let i = 0; i < n; i++) box(`d${i}`, [-w / 4 + (i * w) / (2 * (n - 1 || 1)), 0.45, -0.4 + r() * 1.5], [1.5, 0.9, 0.9], M.wood);
    if (unit.cat === "fashion") {
      for (let i = 0; i < 3; i++) {
        const x = -1.6 + i * 1.6;
        items.push(
          <group key={`mq${i}`} position={[x, 0, -d / 2 + 1.1]}>
            <mesh position={[0, 0.9, 0]} castShadow><cylinderGeometry args={[0.22, 0.18, 1.1, 10]} />
              <meshStandardMaterial color={shade(cat, 0.75)} roughness={0.6} /></mesh>
            <mesh position={[0, 1.6, 0]} material={M.white}><sphereGeometry args={[0.15, 12, 10]} /></mesh>
            <mesh position={[0, 0.2, 0]} material={M.alu}><cylinderGeometry args={[0.04, 0.04, 0.4, 6]} /></mesh>
          </group>);
      }
    }
  }
  return <group>{items}</group>;
}

function Shop({ unit, y, floorId, hovered, selected, onHover, onSelect, night }) {
  const { cx, cz, W, D, rot } = frameOf(unit.rect);
  const w = W - 0.3;
  const d = D - 0.3;
  const cat = CATEGORIES[unit.cat].color;
  const id = `${floorId}:${unit.name}`;
  const active = hovered === id || selected === id;
  const M = mats();
  const big = unit.cat === "anchor" || w > 20;
  const door = big ? 6 : 2.4;
  const panels = useMemo(() => {
    const out = [];
    const sides = [[-w / 2 + 0.15, -door / 2], [door / 2, w / 2 - 0.15]];
    for (const [a, b] of sides) {
      const n = Math.max(1, Math.round((b - a) / 2.4));
      const pw = (b - a) / n;
      for (let i = 0; i < n; i++) out.push(a + pw * (i + 0.5));
      out.pw = pw; // same width for every panel on both sides (symmetric)
    }
    return out;
  }, [w, door]);
  const signMat = useMemo(() => {
    const t = signTex(unit.name.toUpperCase(), shade(cat, 0.62));
    return new THREE.MeshStandardMaterial({ map: t, emissiveMap: t, emissive: "#ffffff", emissiveIntensity: night ? 0.9 : 0.12, roughness: 0.4 });
  }, [unit.name, cat, night]);
  const events = {
    onPointerOver: (e) => { e.stopPropagation(); onHover(id); document.body.style.cursor = "pointer"; },
    onPointerOut: () => { onHover(null); document.body.style.cursor = "auto"; },
    onClick: (e) => { e.stopPropagation(); onSelect({ ...unit, floorId, id }); },
  };
  const signW = Math.min(w * 0.75, big ? 16 : 8.5);

  return (
    <group position={[cx, y, cz]} rotation={[0, rot, 0]}>
      {/* walls */}
      <mesh position={[-w / 2 + 0.1, ROOM_H / 2, 0]} castShadow receiveShadow material={M.wall}><boxGeometry args={[0.2, ROOM_H, d]} /></mesh>
      <mesh position={[w / 2 - 0.1, ROOM_H / 2, 0]} castShadow receiveShadow material={M.wall}><boxGeometry args={[0.2, ROOM_H, d]} /></mesh>
      <mesh position={[0, ROOM_H / 2, d / 2 - 0.1]} castShadow receiveShadow material={M.wall}><boxGeometry args={[w, ROOM_H, 0.2]} /></mesh>
      {/* interior floor and back feature wall in the brand colour */}
      <mesh position={[0, 0.02, 0.1]} receiveShadow>
        <boxGeometry args={[w - 0.4, 0.04, d - 0.4]} />
        <meshStandardMaterial color={shade(cat, 1.02)} roughness={0.55} emissive={cat} emissiveIntensity={night ? 0.12 : 0} />
      </mesh>
      <mesh position={[0, 2.5, d / 2 - 0.22]}>
        <boxGeometry args={[w * 0.7, 2.8, 0.04]} />
        <meshStandardMaterial color={cat} emissive={cat} emissiveIntensity={night ? 0.55 : 0.06} roughness={0.5} />
      </mesh>
      <Interior unit={unit} w={w} d={d} night={night} />
      {/* glass shopfront with aluminium frames and an open door */}
      {panels.map((x, i) => (
        <mesh key={`g${i}`} position={[x, GLASS_H / 2, -d / 2 + 0.05]} material={M.glass}>
          <boxGeometry args={[panels.pw - 0.06, GLASS_H, 0.04]} />
        </mesh>
      ))}
      {panels.map((x, i) => (
        <mesh key={`m${i}`} position={[x - panels.pw / 2, GLASS_H / 2, -d / 2 + 0.05]} material={M.alu}>
          <boxGeometry args={[0.07, GLASS_H, 0.1]} />
        </mesh>
      ))}
      <mesh position={[w / 2 - 0.15, GLASS_H / 2, -d / 2 + 0.05]} material={M.alu}><boxGeometry args={[0.07, GLASS_H, 0.1]} /></mesh>
      <mesh position={[0, GLASS_H, -d / 2 + 0.05]} material={M.alu}><boxGeometry args={[w - 0.2, 0.1, 0.12]} /></mesh>
      {/* fascia with the shop sign, and bulkhead up to the ceiling */}
      <mesh position={[0, GLASS_H + 0.6, -d / 2 + 0.02]} castShadow material={M.dark}><boxGeometry args={[w, 1.1, 0.3]} /></mesh>
      <mesh position={[0, GLASS_H + 0.6, -d / 2 - 0.135]} rotation={[0, Math.PI, 0]} material={signMat}>
        <planeGeometry args={[signW, signW * 0.1875 > 0.9 ? 0.9 : signW * 0.1875]} />
      </mesh>
      <mesh position={[0, (GLASS_H + 1.15 + ROOM_H) / 2, -d / 2 + 0.05]} material={M.wall}>
        <boxGeometry args={[w, ROOM_H - GLASS_H - 1.15, 0.12]} />
      </mesh>
      {/* invisible hit box for hover/click + highlight */}
      <mesh position={[0, ROOM_H / 2, 0]} material={M.hit} {...events}><boxGeometry args={[w, ROOM_H, d]} /></mesh>
      {active && (
        <mesh position={[0, ROOM_H / 2, 0]}>
          <boxGeometry args={[w + 0.1, ROOM_H + 0.1, d + 0.1]} />
          <meshBasicMaterial color={cat} transparent opacity={selected === id ? 0.28 : 0.16} depthWrite={false} />
          <Edges color={selected === id ? "#12263f" : "#ffffff"} />
        </mesh>
      )}
    </group>
  );
}

function FoodCourt({ unit, y, floorId, hovered, selected, onHover, onSelect, night }) {
  const [x0, z0, x1, z1] = unit.rect;
  const cx = (x0 + x1) / 2;
  const cz = (z0 + z1) / 2;
  const M = mats();
  const id = `${floorId}:${unit.name}`;
  const active = hovered === id || selected === id;
  const stalls = ["PIZZA", "BIRYANI", "WRAPS", "JUICE", "NOODLES"];
  const tables = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 5; j++) tables.push([x0 + 7 + i * 5, z0 + 3.5 + j * 5.6]);
  const events = {
    onPointerOver: (e) => { e.stopPropagation(); onHover(id); document.body.style.cursor = "pointer"; },
    onPointerOut: () => { onHover(null); document.body.style.cursor = "auto"; },
    onClick: (e) => { e.stopPropagation(); onSelect({ ...unit, floorId, id }); },
  };
  return (
    <group>
      <mesh position={[cx, y + 0.02, cz]} receiveShadow>
        <boxGeometry args={[x1 - x0 - 0.3, 0.04, z1 - z0 - 0.3]} />
        <meshStandardMaterial color="#c9a782" roughness={0.6} />
      </mesh>
      {stalls.map((s, i) => {
        const z = z0 + 3 + i * 6;
        const t = signTex(s, ["#c0392b", "#d35400", "#2f855a", "#8e44ad", "#1f6feb"][i]);
        return (
          <group key={s} position={[x0 + 1.6, y, z]}>
            <mesh position={[0, 0.55, 0]} castShadow material={M.wood}><boxGeometry args={[1.4, 1.1, 4.8]} /></mesh>
            <mesh position={[-0.9, 2.2, 0]} castShadow material={M.wall}><boxGeometry args={[0.6, 4.4, 5]} /></mesh>
            <mesh position={[-0.55, 3.6, 0]} rotation={[0, Math.PI / 2, 0]}>
              <planeGeometry args={[4.4, 0.85]} />
              <meshStandardMaterial map={t} emissiveMap={t} emissive="#ffffff" emissiveIntensity={night ? 0.9 : 0.12} />
            </mesh>
          </group>
        );
      })}
      {tables.map(([tx, tz], k) => (
        <group key={k} position={[tx, y, tz]}>
          <mesh position={[0, 0.75, 0]} castShadow material={M.white}><cylinderGeometry args={[0.75, 0.75, 0.06, 24]} /></mesh>
          <mesh position={[0, 0.37, 0]} material={M.alu}><cylinderGeometry args={[0.06, 0.06, 0.74, 8]} /></mesh>
          {[0, 1, 2, 3].map((c) => {
            const a = (c * Math.PI) / 2;
            return (
              <mesh key={c} position={[Math.cos(a) * 1.15, 0.45, Math.sin(a) * 1.15]} castShadow material={M.dark}>
                <boxGeometry args={[0.45, 0.9, 0.45]} />
              </mesh>
            );
          })}
        </group>
      ))}
      <mesh position={[cx, y + ROOM_H / 2, cz]} material={M.hit} {...events}><boxGeometry args={[x1 - x0, ROOM_H, z1 - z0]} /></mesh>
      {active && (
        <mesh position={[cx, y + 0.08, cz]}>
          <boxGeometry args={[x1 - x0, 0.1, z1 - z0]} />
          <meshBasicMaterial color={CATEGORIES[unit.cat].color} transparent opacity={0.35} depthWrite={false} />
          <Edges color="#12263f" />
        </mesh>
      )}
    </group>
  );
}

// ---------------------------------------------------------------- atrium, circulation, furniture
function AtriumRailing({ y }) {
  const [x0, z0, x1, z1] = ATRIUM;
  const M = mats();
  const rh = 1.1;
  const sides = [
    { pos: [(x0 + x1) / 2, z0], size: [x1 - x0, 0.06] },
    { pos: [(x0 + x1) / 2, z1], size: [x1 - x0, 0.06] },
    { pos: [x0, (z0 + z1) / 2], size: [0.06, z1 - z0] },
    { pos: [x1, (z0 + z1) / 2], size: [0.06, z1 - z0] },
  ];
  return sides.map((s, i) => (
    <group key={i}>
      <mesh position={[s.pos[0], y + rh / 2, s.pos[1]]} material={M.glass}><boxGeometry args={[s.size[0], rh, s.size[1]]} /></mesh>
      <mesh position={[s.pos[0], y + rh, s.pos[1]]} material={M.alu}>
        <boxGeometry args={[s.size[0] + 0.08, 0.07, s.size[1] + 0.08]} />
      </mesh>
      {/* slab edge band */}
      <mesh position={[s.pos[0], y - SLAB / 2, s.pos[1]]} material={M.white}>
        <boxGeometry args={[s.size[0] + 0.04, SLAB + 0.02, s.size[1] + 0.04]} />
      </mesh>
    </group>
  ));
}

function Escalator({ esc, exploded }) {
  const M = mats();
  const y0 = floorY(esc.from, exploded);
  const y1 = floorY(esc.from + 1, exploded);
  const xs = esc.reverse ? [esc.x1, esc.x0] : [esc.x0, esc.x1];
  const dx = xs[1] - xs[0];
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  const ang = Math.atan2(dy, dx);
  return (
    <group position={[(xs[0] + xs[1]) / 2, (y0 + y1) / 2, esc.z]} rotation={[0, 0, ang]}>
      <mesh castShadow material={M.alu}><boxGeometry args={[len, 0.5, 1.5]} /></mesh>
      <mesh position={[0, 0.27, 0]} material={M.dark}><boxGeometry args={[len, 0.04, 1.05]} /></mesh>
      {[-0.68, 0.68].map((o) => (
        <group key={o}>
          <mesh position={[0, 0.75, o]} material={M.glass}><boxGeometry args={[len, 0.95, 0.05]} /></mesh>
          <mesh position={[0, 1.25, o]} material={M.dark}><boxGeometry args={[len + 0.4, 0.08, 0.1]} /></mesh>
        </group>
      ))}
    </group>
  );
}

function Lifts({ exploded, topVisible }) {
  const M = mats();
  const top = floorY(topVisible, exploded) + FH;
  return LIFTS.map(([x, z]) => (
    <group key={x}>
      <mesh position={[x, top / 2, z]} material={M.glass}><cylinderGeometry args={[1.5, 1.5, top, 32, 1, true]} /></mesh>
      {[0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((a) => (
        <mesh key={a} position={[x + Math.cos(a) * 1.5, top / 2, z + Math.sin(a) * 1.5]} material={M.alu}>
          <boxGeometry args={[0.1, top, 0.1]} />
        </mesh>
      ))}
      <mesh position={[x, floorY(1, exploded) + 1.3, z]} castShadow material={M.white}><cylinderGeometry args={[1.2, 1.2, 2.5, 24]} /></mesh>
    </group>
  ));
}

function Columns({ y }) {
  const pts = [];
  for (const x of [-21, -7, 7, 21]) for (const z of [-15, 15]) pts.push([x, z]);
  for (const [x, z] of [[-14, -7], [14, -7], [-14, 7], [14, 7]]) pts.push([x, z]);
  const M = mats();
  return pts.map(([x, z]) => (
    <mesh key={`${x},${z}`} position={[x, y + ROOM_H / 2, z]} castShadow material={M.white}>
      <cylinderGeometry args={[0.45, 0.45, ROOM_H, 20]} />
    </mesh>
  ));
}

function Planter({ x, z, y, tall = false }) {
  const M = mats();
  return (
    <group position={[x, y, z]}>
      <mesh position={[0, 0.4, 0]} castShadow material={M.stone}><boxGeometry args={[1.6, 0.8, 1.6]} /></mesh>
      {tall ? (
        <>
          <mesh position={[0, 1.6, 0]} material={M.trunk}><cylinderGeometry args={[0.09, 0.12, 1.8, 8]} /></mesh>
          <mesh position={[0, 2.9, 0]} castShadow material={M.leaf}><icosahedronGeometry args={[1.1, 1]} /></mesh>
        </>
      ) : (
        <>
          <mesh position={[-0.25, 1.15, 0.1]} castShadow material={M.leaf}><icosahedronGeometry args={[0.6, 1]} /></mesh>
          <mesh position={[0.3, 1.05, -0.2]} castShadow material={M.leaf2}><icosahedronGeometry args={[0.5, 1]} /></mesh>
        </>
      )}
    </group>
  );
}

function Bench({ x, z, y, rot = 0 }) {
  const M = mats();
  return (
    <group position={[x, y, z]} rotation={[0, rot, 0]}>
      <mesh position={[0, 0.45, 0]} castShadow material={M.wood}><boxGeometry args={[2.2, 0.1, 0.6]} /></mesh>
      {[-0.9, 0.9].map((o) => <mesh key={o} position={[o, 0.22, 0]} material={M.alu}><boxGeometry args={[0.08, 0.44, 0.5]} /></mesh>)}
    </group>
  );
}

function Kiosk({ x, z, y, name, color, night }) {
  const M = mats();
  const t = signTex(name, color);
  return (
    <group position={[x, y, z]}>
      <mesh position={[0, 0.55, 0]} castShadow material={M.white}><boxGeometry args={[2.6, 1.1, 1.4]} /></mesh>
      <mesh position={[0, 1.12, 0]} material={M.wood}><boxGeometry args={[2.7, 0.06, 1.5]} /></mesh>
      {[[-1.25, -0.65], [1.25, -0.65], [-1.25, 0.65], [1.25, 0.65]].map(([a, b]) => (
        <mesh key={`${a}${b}`} position={[a, 1.7, b]} material={M.alu}><boxGeometry args={[0.06, 1.2, 0.06]} /></mesh>
      ))}
      <mesh position={[0, 2.35, 0]} castShadow material={M.dark}><boxGeometry args={[2.9, 0.12, 1.7]} /></mesh>
      {[-0.86, 0.86].map((o) => (
        <mesh key={o} position={[0, 2.6, o]} rotation={[0, o < 0 ? Math.PI : 0, 0]}>
          <planeGeometry args={[2.6, 0.45]} />
          <meshStandardMaterial map={t} emissiveMap={t} emissive="#ffffff" emissiveIntensity={night ? 0.9 : 0.12} />
        </mesh>
      ))}
    </group>
  );
}

function CorridorFurniture({ floorId, y, night }) {
  const planters = [[-10, -11.5], [10, -11.5], [-10, 11.5], [10, 11.5], [-18.5, -6], [18.5, -6], [-18.5, 6], [18.5, 6]];
  const benches = [[-5, -11.5], [5, -11.5], [-5, 11.5], [5, 11.5]];
  const kiosks = [
    [["Phone Cases", "#1f6feb"], ["Coffee Cart", "#7b4a2b"]],
    [["Watches", "#8e44ad"], ["Ice Cream", "#e8487a"]],
    [["Tickets", "#c0392b"], ["Popcorn", "#d68910"]],
  ][floorId];
  return (
    <group>
      {planters.map(([x, z], i) => <Planter key={`p${i}`} x={x} z={z} y={y} tall={i < 4} />)}
      {benches.map(([x, z], i) => <Bench key={`b${i}`} x={x} z={z} y={y} />)}
      <Kiosk x={-16} z={-11.2} y={y} name={kiosks[0][0]} color={kiosks[0][1]} night={night} />
      <Kiosk x={16} z={11.2} y={y} name={kiosks[1][0]} color={kiosks[1][1]} night={night} />
    </group>
  );
}

function InfoDesk({ y, showLabels }) {
  const M = mats();
  return (
    <group>
      <mesh position={[0, y + 0.55, 0]} castShadow material={M.white}><cylinderGeometry args={[2.2, 2.3, 1.1, 40]} /></mesh>
      <mesh position={[0, y + 1.13, 0]} material={M.wood}><cylinderGeometry args={[2.35, 2.35, 0.06, 40]} /></mesh>
      <mesh position={[0, y + 0.06, 0]} receiveShadow material={M.stone}><cylinderGeometry args={[3.2, 3.2, 0.1, 48]} /></mesh>
      {showLabels && <Label text="Information" position={[0, y + 2.8, 0]} dot="#1f6feb" height={2} />}
    </group>
  );
}

function Skylight({ y }) {
  const M = mats();
  const [x0, z0, x1, z1] = ATRIUM;
  const beams = [];
  for (let x = x0; x <= x1 + 0.01; x += 3.5) beams.push(<mesh key={`x${x}`} position={[x, y + 1.2, 0]} material={M.alu}><boxGeometry args={[0.14, 0.2, z1 - z0]} /></mesh>);
  for (let z = z0; z <= z1 + 0.01; z += 3.5) beams.push(<mesh key={`z${z}`} position={[0, y + 1.2, z]} material={M.alu}><boxGeometry args={[x1 - x0, 0.2, 0.14]} /></mesh>);
  return (
    <group>
      <mesh position={[0, y + 1.25, 0]} material={M.glass}><boxGeometry args={[x1 - x0, 0.05, z1 - z0]} /></mesh>
      {beams}
      {[[x0, z0], [x1, z0], [x0, z1], [x1, z1]].map(([x, z]) => (
        <mesh key={`${x}${z}`} position={[x, y + 0.6, z]} material={M.alu}><boxGeometry args={[0.2, 1.2, 0.2]} /></mesh>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------- building envelope
// a wall run along one side, split into bands (stone spandrel, glass, panel) with gaps
function FacadeRun({ axis, fixed, from, to, y, night, gaps = [], solid }) {
  const M = mats();
  const glassMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#5f7f96", transparent: true, opacity: 0.62, roughness: 0.06, metalness: 0.7, envMapIntensity: 1.6,
    emissive: "#ffc98a", emissiveIntensity: night ? 0.35 : 0, depthWrite: false,
  }), [night]);
  const panelMat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#ebe6de", roughness: 0.7 }), []);
  const segs = [];
  let a = from;
  for (const [g0, g1] of [...gaps].sort((p, q) => p[0] - q[0])) {
    if (g0 > a) segs.push([a, g0]);
    a = Math.max(a, g1);
  }
  if (a < to) segs.push([a, to]);
  const place = (s0, s1, y0, y1, mat, depth, key) => {
    const len = s1 - s0;
    const c = (s0 + s1) / 2;
    const pos = axis === "x" ? [c, (y0 + y1) / 2, fixed] : [fixed, (y0 + y1) / 2, c];
    const size = axis === "x" ? [len, y1 - y0, depth] : [depth, y1 - y0, len];
    return <mesh key={key} position={pos} material={mat} castShadow={mat !== glassMat} receiveShadow><boxGeometry args={size} /></mesh>;
  };
  const out = [];
  segs.forEach(([s0, s1], i) => {
    if (!solid) {
      out.push(place(s0, s1, y, y + ROOM_H, M.glass, 0.08, `c${i}`));
      return;
    }
    out.push(place(s0, s1, y - SLAB, y + 0.9, M.stone, 0.5, `s${i}`));
    out.push(place(s0, s1, y + 0.9, y + 4.3, glassMat, 0.12, `g${i}`));
    out.push(place(s0, s1, y + 4.3, y + ROOM_H, panelMat, 0.45, `p${i}`));
    for (let f = s0; f <= s1 + 0.01; f += 6) {
      const pos = axis === "x" ? [f, y + 2.6, fixed] : [fixed, y + 2.6, f];
      out.push(<mesh key={`f${i}${f}`} position={pos} material={M.alu}><boxGeometry args={axis === "x" ? [0.18, 3.4, 0.5] : [0.5, 3.4, 0.18]} /></mesh>);
    }
  });
  return <group>{out}</group>;
}

function Envelope({ floor, y, solid, night }) {
  const entranceGap = floor.id === 0 ? [[-5, 5]] : [];
  return (
    <group>
      <FacadeRun axis="x" fixed={-HALF_D} from={-HALF_W} to={HALF_W} y={y} night={night} solid={solid} gaps={entranceGap} />
      <FacadeRun axis="x" fixed={HALF_D} from={-HALF_W} to={HALF_W} y={y} night={night} solid={solid} />
      <FacadeRun axis="z" fixed={-HALF_W} from={-HALF_D} to={HALF_D} y={y} night={night} solid={solid} />
      <FacadeRun axis="z" fixed={HALF_W} from={-HALF_D} to={HALF_D} y={y} night={night} solid={solid} />
    </group>
  );
}

function GrandEntrance({ solid, night, showLabels }) {
  // glass entrance in the gap between the two ground-floor shops at x -5..5
  const M = mats();
  const h = ROOM_H;
  const z = -HALF_D;
  const mullions = [];
  for (let x = -5; x <= 5.01; x += 2) mullions.push(<mesh key={x} position={[x, h / 2, z]} material={M.alu}><boxGeometry args={[0.12, h, 0.2]} /></mesh>);
  return (
    <group>
      <mesh position={[0, h / 2, z]} material={M.glass}><boxGeometry args={[10, h, 0.1]} /></mesh>
      {mullions}
      <mesh position={[0, 3.2, z]} material={M.alu}><boxGeometry args={[10, 0.12, 0.2]} /></mesh>
      {/* canopy */}
      <mesh position={[0, 4.6, z - 4]} castShadow material={M.dark}><boxGeometry args={[16, 0.35, 8]} /></mesh>
      <mesh position={[0, 4.4, z - 4]}>
        <boxGeometry args={[15, 0.05, 7.4]} />
        <meshStandardMaterial color="#ffffff" emissive="#fff1d6" emissiveIntensity={night ? 1.2 : 0.2} />
      </mesh>
      {[-7, 7].map((x) => <mesh key={x} position={[x, 2.3, z - 7.6]} castShadow material={M.alu}><cylinderGeometry args={[0.22, 0.22, 4.6, 16]} /></mesh>)}
      {/* revolving door drum */}
      <mesh position={[0, 1.3, z + 0.9]} material={M.glass}><cylinderGeometry args={[1.6, 1.6, 2.6, 32, 1, true]} /></mesh>
      <mesh position={[0, 2.65, z + 0.9]} material={M.alu}><cylinderGeometry args={[1.7, 1.7, 0.15, 32]} /></mesh>
      {showLabels && !solid && <Label text="Main entrance" position={[0, 8.5, z - 4]} big height={2.6} />}
    </group>
  );
}

function Roof({ exploded, night }) {
  const M = mats();
  const y = floorY(TOP, exploded) + FH;
  const logo = logoTex();
  return (
    <group>
      <Slab y={y} withHole roof />
      {/* parapet */}
      {[[0, -HALF_D, MALL.w, 0.3], [0, HALF_D, MALL.w, 0.3], [-HALF_W, 0, 0.3, MALL.d], [HALF_W, 0, 0.3, MALL.d]].map(([x, z, w, d], i) => (
        <mesh key={i} position={[x, y + 0.6, z]} castShadow material={M.stone}><boxGeometry args={[w, 1.2, d]} /></mesh>
      ))}
      {/* rooftop units */}
      {[[-30, 15], [-24, 15], [26, -18], [32, -18], [30, 18]].map(([x, z], i) => (
        <group key={i} position={[x, y, z]}>
          <mesh position={[0, 1.1, 0]} castShadow material={M.alu}><boxGeometry args={[4, 2.2, 3]} /></mesh>
          <mesh position={[0, 2.25, 0]} material={M.dark}><cylinderGeometry args={[0.9, 0.9, 0.12, 20]} /></mesh>
        </group>
      ))}
      {/* SMART MALL sign above the entrance */}
      <mesh position={[0, y + 2.6, -HALF_D - 0.4]} castShadow material={M.dark}><boxGeometry args={[30, 4.2, 0.6]} /></mesh>
      <mesh position={[0, y + 2.6, -HALF_D - 0.72]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[27, 4.2]} />
        <meshStandardMaterial map={logo} transparent emissiveMap={logo} emissive="#ffffff" emissiveIntensity={night ? 1.6 : 0.25} />
      </mesh>
    </group>
  );
}

// ---------------------------------------------------------------- site: plaza, roads, parking, lights, trees
function useInstances(ref, list, toMatrix, colors) {
  useLayoutEffect(() => {
    if (!ref.current) return;
    const m = new THREE.Matrix4();
    list.forEach((item, i) => {
      toMatrix(m, item);
      ref.current.setMatrixAt(i, m);
      if (colors) ref.current.setColorAt(i, new THREE.Color(colors[i]));
    });
    ref.current.instanceMatrix.needsUpdate = true;
    if (colors && ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true;
  }, [ref, list, toMatrix, colors]);
}

const SITE = (() => {
  const r = rngFrom("site");
  const stalls = [];
  const cars = [];
  const carColors = [];
  const palette = ["#d8dde3", "#1f2329", "#9aa3ad", "#8b1e1e", "#1d3f7a", "#e8e8e8", "#3c4a3a", "#5a5f66", "#b7892f"];
  for (const side of [-1, 1]) {
    for (const cx of [60, 74, 88, 102]) {
      for (let z = -33; z <= 33; z += 2.8) {
        const x = side * cx;
        stalls.push([x, z]);
        if (r() < 0.62) {
          cars.push([x + side * (r() - 0.5) * 0.3, z + 1.4, r() < 0.5 ? 0 : Math.PI]);
          carColors.push(palette[Math.floor(r() * palette.length)]);
        }
      }
    }
  }
  const lamps = [];
  for (let x = -110; x <= 110; x += 20) lamps.push([x, -HALF_D - 21, 0]);
  for (const side of [-1, 1]) for (const cx of [67, 95]) for (const z of [-24, 0, 24]) lamps.push([side * cx, z, 0]);
  const trees = [];
  for (let x = -100; x <= 100; x += 12) if (Math.abs(x) > 16) trees.push([x, -HALF_D - 15 + (r() - 0.5) * 2, 0.8 + r() * 0.5]);
  for (let x = -60; x <= 60; x += 10) trees.push([x, HALF_D + 9 + (r() - 0.5) * 3, 0.8 + r() * 0.5]);
  for (const side of [-1, 1]) for (let z = -40; z <= 40; z += 10) trees.push([side * 52, z, 0.7 + r() * 0.4]);
  return { stalls, cars, carColors, lamps, trees };
})();

function Site({ night }) {
  const M = mats();
  const stallRef = useRef();
  const bodyRef = useRef();
  const cabinRef = useRef();
  const poleRef = useRef();
  const headRef = useRef();
  const trunkRef = useRef();
  const crownRef = useRef();
  const e = new THREE.Euler();
  const q = new THREE.Quaternion();
  const v = new THREE.Vector3();
  const one = new THREE.Vector3(1, 1, 1);
  useInstances(stallRef, SITE.stalls, (m, [x, z]) => m.makeRotationX(-Math.PI / 2).setPosition(x, -0.38, z), null);
  useInstances(bodyRef, SITE.cars, (m, [x, z, rot]) => m.compose(v.set(x, 0.12, z), q.setFromEuler(e.set(0, rot, 0)), one), SITE.carColors);
  useInstances(cabinRef, SITE.cars, (m, [x, z, rot]) => m.compose(v.set(x - Math.cos(rot) * 0.2, 0.86, z), q.setFromEuler(e.set(0, rot, 0)), one), SITE.carColors);
  useInstances(poleRef, SITE.lamps, (m, [x, z]) => m.makeTranslation(x, 3.6, z), null);
  useInstances(headRef, SITE.lamps, (m, [x, z]) => m.makeTranslation(x, 7.3, z), null);
  useInstances(trunkRef, SITE.trees, (m, [x, z, s]) => m.compose(v.set(x, 1.4 * s - 0.4, z), q.identity(), v.clone().set(s, s, s)), null);
  useInstances(crownRef, SITE.trees, (m, [x, z, s]) => m.compose(v.set(x, 3.8 * s - 0.4, z), q.identity(), v.clone().set(s, s, s)), null);

  // plane UVs run 0..1, so each texture repeats about once every 8 m
  const grass = night ? noiseTex("grassN", "#2f4a33", 60, [88, 88]) : noiseTex("grass", "#86a971", 60, [88, 88]);
  const road = noiseTex("road", "#4a4f56", 50, [32, 1.5]);
  const asphalt = noiseTex("asphalt", "#4a4f56", 50, [7, 9.5]);
  const paving = noiseTex("paving", "#cfcac0", 30, [26, 20]);
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.45, 0]} receiveShadow>
        <planeGeometry args={[700, 700]} />
        <meshStandardMaterial map={grass} roughness={1} />
      </mesh>
      {/* paved surround and entrance plaza */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.42, -4]} receiveShadow>
        <planeGeometry args={[MALL.w + 14, MALL.d + 22]} />
        <meshStandardMaterial map={paving} roughness={0.85} />
      </mesh>
      {/* road and car parks */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.41, -HALF_D - 28]} receiveShadow>
        <planeGeometry args={[260, 12]} />
        <meshStandardMaterial map={road} roughness={0.95} />
      </mesh>
      {Array.from({ length: 22 }, (_, i) => (
        <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[-126 + i * 12, -0.39, -HALF_D - 28]}>
          <planeGeometry args={[5, 0.3]} /><meshStandardMaterial color="#f2f2ee" />
        </mesh>
      ))}
      {[-1, 1].map((side) => (
        <mesh key={side} rotation={[-Math.PI / 2, 0, 0]} position={[side * 81, -0.41, 0]} receiveShadow>
          <planeGeometry args={[58, 76]} />
          <meshStandardMaterial map={asphalt} roughness={0.95} />
        </mesh>
      ))}
      <instancedMesh ref={stallRef} args={[null, null, SITE.stalls.length]}>
        <planeGeometry args={[5, 0.12]} /><meshStandardMaterial color="#f2f2ee" />
      </instancedMesh>
      <instancedMesh ref={bodyRef} args={[null, null, SITE.cars.length]} castShadow>
        <boxGeometry args={[4.3, 0.9, 1.85]} /><meshStandardMaterial roughness={0.35} metalness={0.6} />
      </instancedMesh>
      <instancedMesh ref={cabinRef} args={[null, null, SITE.cars.length]} castShadow>
        <boxGeometry args={[2.3, 0.6, 1.65]} /><meshStandardMaterial roughness={0.2} metalness={0.5} />
      </instancedMesh>
      <instancedMesh ref={poleRef} args={[null, null, SITE.lamps.length]} material={M.alu}>
        <cylinderGeometry args={[0.12, 0.16, 8, 8]} />
      </instancedMesh>
      <instancedMesh ref={headRef} args={[null, null, SITE.lamps.length]}>
        <boxGeometry args={[1.1, 0.25, 0.5]} />
        <meshStandardMaterial color="#f5f5f0" emissive="#ffd9a0" emissiveIntensity={night ? 3 : 0.1} />
      </instancedMesh>
      <instancedMesh ref={trunkRef} args={[null, null, SITE.trees.length]} material={M.trunk}>
        <cylinderGeometry args={[0.22, 0.3, 2.8, 8]} />
      </instancedMesh>
      <instancedMesh ref={crownRef} args={[null, null, SITE.trees.length]} material={M.leaf} castShadow>
        <icosahedronGeometry args={[2.3, 1]} />
      </instancedMesh>
    </group>
  );
}

function CameraRig({ preset }) {
  const { camera, controls } = useThree();
  useEffect(() => {
    if (!controls || !preset) return;
    const p = CAMERA_PRESETS[preset.name];
    camera.position.set(...p.pos);
    controls.target.set(...p.target);
    controls.update();
  }, [preset, camera, controls]);
  return null;
}

// ---------------------------------------------------------------- whole scene
export default function Mall3D({ view, exploded, showLabels, hovered, selected, onHover, onSelect, preset,
                                 night = false, building = "cutaway" }) {
  const visibleFloors = FLOORS.filter((f) => view === "all" || f.id <= view);
  const topVisible = visibleFloors[visibleFloors.length - 1].id;
  const labelFloor = view === "all" ? TOP : view;
  const solid = building === "full";
  const showRoof = solid && topVisible === TOP;

  return (
    <Canvas shadows="soft" dpr={[1, 2]} camera={{ position: CAMERA_PRESETS.overview.pos, fov: 42, near: 0.5, far: 1500 }}
            gl={{ antialias: true }} onPointerMissed={() => onSelect(null)}
            style={{ background: night ? "linear-gradient(#03060d, #111d33 70%, #1b2a44)" : "linear-gradient(#9cc4ea, #dbe9f5 65%, #eef3f7)" }}>
      <SceneEnv night={night} />
      <Lights night={night} visibleFloors={visibleFloors} exploded={exploded} />
      <Site night={night} />
      {visibleFloors.map((f) => {
        const y = floorY(f.id, exploded);
        return (
          <group key={f.id}>
            <Slab y={y} withHole={f.id > 0} />
            {f.id > 0 && <Ceiling y={y} night={night} />}
            <Envelope floor={f} y={y} solid={solid} night={night} />
            <Columns y={y} />
            {f.id > 0 && <AtriumRailing y={y} />}
            {f.id === 0 && <InfoDesk y={y} showLabels={showLabels && labelFloor === 0} />}
            <CorridorFurniture floorId={f.id} y={y} night={night} />
            {f.units.map((u) => (u.open
              ? <FoodCourt key={u.name} unit={u} y={y} floorId={f.id} hovered={hovered} selected={selected}
                           onHover={onHover} onSelect={onSelect} night={night} />
              : <Shop key={u.name} unit={u} y={y} floorId={f.id} hovered={hovered} selected={selected}
                      onHover={onHover} onSelect={onSelect} night={night} />))}
            {showLabels && !showRoof && f.id === labelFloor && f.units.map((u) => {
              const [x0, z0, x1, z1] = u.rect;
              return <Label key={u.name} text={u.name} dot={CATEGORIES[u.cat].color}
                            position={[(x0 + x1) / 2, y + ROOM_H + 1.4, (z0 + z1) / 2]} />;
            })}
            <Label text={f.short === "G" ? "Ground" : `Floor ${f.short}`} position={[HALF_W + 5, y + 2.5, -HALF_D - 5]} big height={2.6} />
          </group>
        );
      })}
      <GrandEntrance solid={solid} night={night} showLabels={showLabels} />
      {showRoof && <Skylight y={floorY(TOP, exploded) + FH - SLAB} />}
      {showRoof && <Roof exploded={exploded} night={night} />}
      {ESCALATORS.filter((e) => view === "all" || e.from + 1 <= view).map((e, i) => <Escalator key={i} esc={e} exploded={exploded} />)}
      <Lifts exploded={exploded} topVisible={topVisible} />
      <OrbitControls makeDefault target={CAMERA_PRESETS.overview.target} maxPolarAngle={Math.PI / 2.06}
                     minDistance={10} maxDistance={360} enableDamping />
      <CameraRig preset={preset} />
    </Canvas>
  );
}
