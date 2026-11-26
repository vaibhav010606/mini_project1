// Smart mall layout (metres). x = east-west, z = north-south (negative z = south, main entrance side).
// Footprint 90 m x 56 m, 3 floors, a central atrium open through all upper floors.
// Shop rectangles are [x0, z0, x1, z1]. The same layout will be used later for the crowd heatmap.

export const MALL = { w: 90, d: 56, floorHeight: 6, slab: 0.4 };
export const ATRIUM = [-14, -7, 14, 7]; // open void on floors 1 and 2
export const CORRIDOR = [-23, -16, 23, 16]; // walkway band around the atrium

export const CATEGORIES = {
  fashion: { label: "Fashion", color: "#f4a6b8" },
  electronics: { label: "Electronics", color: "#8fb8ef" },
  food: { label: "Food & drink", color: "#f6be7b" },
  anchor: { label: "Anchor store", color: "#b9a4e6" },
  entertainment: { label: "Entertainment", color: "#f08a8a" },
  services: { label: "Services", color: "#93d3a8" },
  leisure: { label: "Kids & leisure", color: "#7fd1cf" },
};

const NORTH = [[-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45]];
const SOUTH = [[-45, -30], [-30, -15], [-15, -5], [5, 15], [15, 30], [30, 45]];
const SOUTH_UPPER = [[-45, -30], [-30, -15], [-15, 0], [0, 15], [15, 30], [30, 45]];

function row(xs, z0, z1, shops) {
  return xs.map(([x0, x1], i) => ({ rect: [x0, z0, x1, z1], ...shops[i] }));
}

export const FLOORS = [
  {
    id: 0,
    name: "Ground floor",
    short: "G",
    units: [
      ...row(NORTH, 16, 28, [
        { name: "Fashion Hub", cat: "fashion" }, { name: "Footwear", cat: "fashion" },
        { name: "Electronics", cat: "electronics" }, { name: "Pharmacy", cat: "services" },
        { name: "Café", cat: "food" }, { name: "Bank & ATM", cat: "services" },
      ]),
      ...row(SOUTH, -28, -16, [
        { name: "Books", cat: "leisure" }, { name: "Watches", cat: "fashion" },
        { name: "Mobile Store", cat: "electronics" }, { name: "Cosmetics", cat: "fashion" },
        { name: "Kids Wear", cat: "fashion" }, { name: "Florist", cat: "services" },
      ]),
      { name: "Hypermarket", cat: "anchor", rect: [-45, -15, -23, 15] },
      { name: "Department Store", cat: "anchor", rect: [23, -15, 45, 15] },
    ],
    features: [
      { type: "entrance", name: "Main entrance", rect: [-5, -28, 5, -16] },
      { type: "desk", name: "Information desk", rect: [-3, -2, 3, 2] },
    ],
  },
  {
    id: 1,
    name: "First floor",
    short: "1",
    units: [
      ...row(NORTH, 16, 28, [
        { name: "Sports", cat: "fashion" }, { name: "Home Decor", cat: "services" },
        { name: "Toys", cat: "leisure" }, { name: "Jewellery", cat: "fashion" },
        { name: "Opticals", cat: "services" }, { name: "Gadgets", cat: "electronics" },
      ]),
      ...row(SOUTH_UPPER, -28, -16, [
        { name: "Menswear", cat: "fashion" }, { name: "Womenswear", cat: "fashion" },
        { name: "Ethnic Wear", cat: "fashion" }, { name: "Bags", cat: "fashion" },
        { name: "Perfumes", cat: "fashion" }, { name: "Salon", cat: "services" },
      ]),
      { name: "Food Court", cat: "food", rect: [-45, -15, -23, 15], open: true },
      { name: "Department Store L1", cat: "anchor", rect: [23, -15, 45, 15] },
    ],
    features: [],
  },
  {
    id: 2,
    name: "Second floor",
    short: "2",
    units: [
      ...row(NORTH, 16, 28, [
        { name: "Pizza", cat: "food" }, { name: "Chinese", cat: "food" },
        { name: "Indian Kitchen", cat: "food" }, { name: "Burgers", cat: "food" },
        { name: "Desserts", cat: "food" }, { name: "Juice Bar", cat: "food" },
      ]),
      ...row(SOUTH_UPPER, -28, -16, [
        { name: "Kids Play", cat: "leisure" }, { name: "Fitness Studio", cat: "leisure" },
        { name: "Book Café", cat: "food" }, { name: "Music Store", cat: "electronics" },
        { name: "VR Arcade", cat: "entertainment" }, { name: "Lounge", cat: "services" },
      ]),
      { name: "Cinema (6 screens)", cat: "entertainment", rect: [-45, -15, -23, 15] },
      { name: "Game Zone", cat: "entertainment", rect: [23, -15, 45, 15] },
    ],
    features: [],
  },
];

// Escalators cross the atrium: [x0, z, x1] going up from floor `from` to `from + 1`.
export const ESCALATORS = [
  { from: 0, x0: -11, x1: -3, z: -4.2 },
  { from: 0, x0: 3, x1: 11, z: 4.2, reverse: true },
  { from: 1, x0: 3, x1: 11, z: -4.2 },
  { from: 1, x0: -11, x1: -3, z: 4.2, reverse: true },
];
export const LIFTS = [[-19, 0], [19, 0]];

export const area = ([x0, z0, x1, z1]) => Math.abs((x1 - x0) * (z1 - z0));
