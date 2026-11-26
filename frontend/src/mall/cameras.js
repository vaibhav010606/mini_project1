// The 12 CCTV cameras of the smart mall (4 per floor). Positions use the same metres as layout.js,
// so they can be shown on the 3D model and linked to the crowd heatmap later.
// area = plain-language location used in email alerts.
// scene = which animated view the fake feed draws; crowd = how many people are usually in view.
// surge = a crowd that builds up at one spot, stays packed for a while, then clears, on a repeating cycle:
//   at [u, v] on the camera's floor (u 0..1 left-right, v 0 far .. 1 near), spread [su, sv],
//   extra = how many extra people join, period = cycle length (s), offset = start of the cycle (s).

export const CAMERAS = [
  { id: "CAM-01", name: "Main entrance", area: "the main entrance (south side of the mall)",  floor: 0, pos: [0, -24], scene: "entrance", crowd: 26, sign: "MAIN ENTRANCE",
    surge: { at: [0.5, 0.32], spread: [0.13, 0.12], extra: 55, period: 70, offset: 0 } },
  { id: "CAM-02", name: "Ground atrium", area: "the central atrium, next to the information desk",  floor: 0, pos: [0, 0], scene: "atrium", crowd: 30, ground: true },
  { id: "CAM-03", name: "Hypermarket entry", area: "the Hypermarket entrance (west wing)",  floor: 0, pos: [-22, 0], scene: "storefront", crowd: 22, sign: "HYPERMARKET" },
  { id: "CAM-04", name: "North corridor (G)", area: "the north corridor outside Fashion Hub, Footwear and Electronics",  floor: 0, pos: [0, 14], scene: "corridor", crowd: 18, shops: ["Fashion Hub", "Footwear", "Electronics"] },

  { id: "CAM-05", name: "Food court", area: "the Food court (west wing)",  floor: 1, pos: [-34, 0], scene: "foodcourt", crowd: 34,
    surge: { at: [0.32, 0.16], spread: [0.16, 0.08], extra: 45, period: 60, offset: 20 } },
  { id: "CAM-06", name: "South corridor (1)", area: "the south corridor outside Menswear, Womenswear and Ethnic Wear",  floor: 1, pos: [0, -14], scene: "corridor", crowd: 16, shops: ["Menswear", "Womenswear", "Ethnic Wear"] },
  { id: "CAM-07", name: "Escalators G-1", area: "the escalators between the Ground and First floor (central atrium)",  floor: 1, pos: [-7, -4], scene: "escalator", crowd: 14,
    surge: { at: [0.3, 0.8], spread: [0.12, 0.1], extra: 35, period: 55, offset: 38 } },
  { id: "CAM-08", name: "Department store L1", area: "the Department Store entrance (east wing)",  floor: 1, pos: [22, 0], scene: "storefront", crowd: 16, sign: "DEPARTMENT STORE" },

  { id: "CAM-09", name: "Cinema lobby", area: "the Cinema lobby and ticket counter (west wing)",  floor: 2, pos: [-30, 0], scene: "cinema", crowd: 24,
    surge: { at: [0.5, 0.18], spread: [0.15, 0.08], extra: 45, period: 80, offset: 50 } },
  { id: "CAM-10", name: "Game zone", area: "the Game Zone entrance (east wing)",  floor: 2, pos: [30, 0], scene: "storefront", crowd: 20, sign: "GAME ZONE" },
  { id: "CAM-11", name: "Restaurant row", area: "the restaurant row (north side): Pizza, Chinese and Indian Kitchen",  floor: 2, pos: [0, 14], scene: "corridor", crowd: 22, shops: ["Pizza", "Chinese", "Indian Kitchen"] },
  { id: "CAM-12", name: "Second floor atrium", area: "the walkway around the atrium railing",  floor: 2, pos: [0, 7], scene: "atrium", crowd: 18 },
];

export const FLOOR_NAMES = ["Ground", "First", "Second"];
