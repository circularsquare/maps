// Buses follow one-way streets. Run with: npx vite-node src/map/roadOneWay.test.ts
import { joinRoads, RoadGraph, type Snap } from "./roadGraph";
import { decodeRoads, type RoadLines } from "./roadTiles";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function graph(lines: number[][], ways: number[]): RoadGraph {
  const g = new RoadGraph();
  g.oneWay = true;
  const rl: RoadLines = {
    extent: 4096,
    lines: lines.map((l) => Int32Array.from(l)),
    weights: lines.map(() => 1),
    ways,
  };
  g.addTile(0, 0, rl);
  return g;
}

function snapOn(g: RoadGraph, edge: number, t: number): Snap {
  const a = g.ea[edge], b = g.eb[edge];
  return { edge, t, x: g.gx[a] + t * (g.gx[b] - g.gx[a]), y: g.gy[a] + t * (g.gy[b] - g.gy[a]), d: 0 };
}

/** Eastbound one-way with a two-way block to the south, so the way back exists. */
const block = [
  [100, 100, 900, 100],
  [100, 100, 100, 800],
  [100, 800, 900, 800],
  [900, 800, 900, 100],
];

function wrongWay(g: RoadGraph, edge: number, pts: number[]) {
  const y = g.gy[g.ea[edge]];
  for (let i = 2; i + 1 < pts.length; i += 2) {
    const x0 = pts[i - 2], y0 = pts[i - 1], x1 = pts[i], y1 = pts[i + 1];
    if (Math.abs(y0 - y) < 1 && Math.abs(y1 - y) < 1) assert(x1 + 1e-6 >= x0, `ran the one-way backward (${x0} to ${x1})`);
  }
}

const east = graph(block, [1, 0, 0, 0]);
const forward = east.path(snapOn(east, 0, 0.2), snapOn(east, 0, 0.8), 1e6);
assert(forward && forward.pts.length === 4, "with the one-way, the bus stays on it");
const back = east.path(snapOn(east, 0, 0.8), snapOn(east, 0, 0.2), 1e6);
assert(back && back.pts.length > 4, "against the one-way, the bus goes round");
assert(forward.m < back.m, "the long way round is longer");
wrongWay(east, 0, back.pts);
const off = graph(block, [1, 0, 0, 0]);
off.oneWay = false;
assert(off.path(snapOn(off, 0, 0.8), snapOn(off, 0, 0.2), 1e6)?.pts.length === 4, "with the setting off the one-way is run backward");

const dead = graph([[100, 100, 900, 100]], [1]);
assert(dead.path(snapOn(dead, 0, 0.8), snapOn(dead, 0, 0.2), 1e6) === null, "against a one-way with no other road, no path");
const tram = dead.turnPath([snapOn(dead, 0, 0.8)], [snapOn(dead, 0, 0.2)], () => 0, 1e6, 0, 0, Math.PI, 0);
assert(tram && tram.pts.length >= 4, "a tram still follows the road against a one-way street");

const both = graph([[100, 100, 900, 100]], [0]);
const rev = both.path(snapOn(both, 0, 0.8), snapOn(both, 0, 0.2), 1e6);
assert(rev && rev.pts.length === 4, "a two-way street still runs back");

const west = graph(block, [-1, 0, 0, 0]);
const westbound = west.path(snapOn(west, 0, 0.8), snapOn(west, 0, 0.2), 1e6);
assert(westbound && westbound.pts.length === 4, "oneway -1 runs against the drawn line");
const eastbound = west.path(snapOn(west, 0, 0.2), snapOn(west, 0, 0.8), 1e6);
assert(eastbound && eastbound.pts.length > 4, "oneway -1 refuses the drawn direction");
wrongWay(west, 0, eastbound.pts.map((_, i, a) => (i % 2 === 0 ? -a[i] : a[i])));

const joined = joinRoads({ extent: 4096, lines: [Int32Array.from([100, 100, 900, 100])], weights: [1], ways: [-1] }, 1);
assert(joined.ways?.length === 1 && joined.ways[0] === -1, "joining roads keeps the street's way");

function varint(n: number): number[] {
  const o: number[] = [];
  let v = n;
  while (v > 127) {
    o.push((v & 127) | 128);
    v = Math.floor(v / 128);
  }
  o.push(v);
  return o;
}
function lenField(field: number, body: number[]): number[] {
  return [...varint((field << 3) | 2), ...varint(body.length), ...body];
}
function text(s: string): number[] {
  return [...new TextEncoder().encode(s)];
}
function sintValue(n: number): number[] {
  const z = n < 0 ? -n * 2 - 1 : n * 2;
  return lenField(4, [...varint((6 << 3) | 0), ...varint(z)]);
}
const geom = [...varint(9), 0, 0, ...varint(10), ...varint(20), 0];
function feature(value: number): number[] {
  const tags = [...varint(0), ...varint(0), ...varint(1), ...varint(value)];
  return lenField(2, [...varint((3 << 3) | 0), ...varint(2), ...lenField(2, tags), ...lenField(4, geom)]);
}
const layer = [
  ...lenField(1, text("transportation")),
  ...lenField(3, text("class")),
  ...lenField(3, text("oneway")),
  ...lenField(4, lenField(1, text("primary"))),
  ...sintValue(1),
  ...sintValue(-1),
  ...feature(1),
  ...feature(2),
];
const roads = decodeRoads(new Uint8Array(lenField(3, layer)).buffer);
assert(roads.lines.length === 2 && roads.ways?.[0] === 1 && roads.ways?.[1] === -1, `decoded ways ${roads.ways?.join(",")}`);

console.log("road oneway ok");
