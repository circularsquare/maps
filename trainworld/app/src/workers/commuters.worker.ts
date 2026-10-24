// Commuter bubbles for the whole city (T-097, SPEC 8), built off the main thread: one worker,
// started when the bubbles are first shown.
//
// - Once per city: each cell's position in local units, and its group at each aggregation level,
//   the cell's H3 parent at resolutions 8, 7 and 6 (and the cell itself at 9). A parent's index is
//   the cell's with the digits below its resolution set to 7, so grouping needs no H3 library.
// - Per demand answer and end (homes or jobs): commuters by mode summed into every level, each
//   bubble at its commuters' weighted centre, as discs ready to upload (map/bubbles.ts).

import { BUBBLE_ALPHA } from "../game/palette";
import { aggregate, biggestFirst, discs, recolour, reorder } from "../map/bubbles";
import { toLocal, type Anchor } from "../map/geo";
import { BUBBLE_DENSITY, BUBBLE_RES, type BubbleLevel, type FromCommuters, type Grouping, type ToCommuters } from "./commutersProtocol";

import { M_PER_DEG, M_PER_DEG_LAT, makeFrame, type FrameSpec } from "../game/frame";

let cellX = new Float32Array(0), cellY = new Float32Array(0);
/** each cell's longitude and latitude, for placing it from any draw anchor (T-317) */
let cellLng = new Float64Array(0), cellLat = new Float64Array(0);
/** the draw anchor cellX/cellY (and so everything sent) are relative to; the main thread moves it */
let cur: Anchor | null = null;
let levels: Grouping[] = [];

/** Measure every cell from draw anchor `a` (map/geo.ts, T-317). */
function place(a: Anchor) {
  cur = a;
  const n = cellLng.length;
  cellX = new Float32Array(n);
  cellY = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const [x, y] = toLocal(cellLng[i], cellLat[i], a);
    cellX[i] = x;
    cellY[i] = y;
  }
}

function setCells(xy: Float32Array, h3: BigUint64Array, spec: FrameSpec, a: Anchor) {
  const frame = makeFrame(spec);
  const n = xy.length / 2;
  cellLng = new Float64Array(n);
  cellLat = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [lng, lat] = frame.toLonLat(xy[i * 2], xy[i * 2 + 1]);
    cellLng[i] = lng;
    cellLat[i] = lat;
  }
  place(a);
  bucketCells();
  levels = BUBBLE_RES.map((res) => {
    const groupOf = new Uint32Array(n);
    let count = 0;
    if (res === 9) for (let i = 0; i < n; i++) groupOf[i] = count++;
    else {
      const fill = (1n << BigInt(3 * (15 - res))) - 1n;
      // clearing the resolution field groups T-044's merged res-8 cells with res-9 neighbours
      const unres = ~(0xfn << 52n);
      const ids = new Map<bigint, number>();
      for (let i = 0; i < n; i++) {
        const key = (h3[i] | fill) & unres;
        let g = ids.get(key);
        if (g === undefined) ids.set(key, (g = count++));
        groupOf[i] = g;
      }
    }
    // members of each group
    const off = new Uint32Array(count + 1);
    for (let i = 0; i < n; i++) off[groupOf[i] + 1]++;
    for (let g = 0; g < count; g++) off[g + 1] += off[g];
    const at = off.slice(0, count);
    const members = new Uint32Array(n);
    for (let i = 0; i < n; i++) members[at[groupOf[i]]++] = i;
    return { res, groupOf, n: count, off, members };
  });
}

// ---- the colour smoothed among nearby cells (T-900) ----
// The model gives every cell of a station catchment's subzone the same rail share and every cell
// of a zone the same walk share, so neighbouring bubbles came out in blocks of one colour with
// sharp edges. For the colour only, each cell takes its neighbours' mode mix weighted by their
// trips times a Gaussian of the distance; sizes, hovers and clicks keep the real figures.
/** the kernel's spread and cutoff, metres */
const SMOOTH_SIGMA_M = 500;
const SMOOTH_CUT_M = 1250;
/** cells in buckets of SMOOTH_CUT_M square: their metres and the bucket each starts */
let mx = new Float32Array(0), my = new Float32Array(0);
let byBucket = new Uint32Array(0);
const buckets = new Map<number, [number, number]>();
const bucketKey = (bx: number, by: number) => bx * 1048576 + by;

function bucketCells() {
  const n = cellLng.length;
  let lat0 = 0;
  for (let i = 0; i < n; i++) lat0 += cellLat[i];
  lat0 = n ? lat0 / n : 0;
  const kx = M_PER_DEG * Math.cos((lat0 * Math.PI) / 180), ky = M_PER_DEG_LAT;
  mx = new Float32Array(n);
  my = new Float32Array(n);
  let x0 = Infinity, y0 = Infinity;
  for (let i = 0; i < n; i++) (x0 = Math.min(x0, cellLng[i] * kx)), (y0 = Math.min(y0, cellLat[i] * ky));
  const key = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    mx[i] = cellLng[i] * kx - x0;
    my[i] = cellLat[i] * ky - y0;
    key[i] = bucketKey(Math.floor(mx[i] / SMOOTH_CUT_M), Math.floor(my[i] / SMOOTH_CUT_M));
  }
  byBucket = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => key[a] - key[b]);
  buckets.clear();
  for (let s = 0; s < n; ) {
    let e = s + 1;
    while (e < n && key[byBucket[e]] === key[byBucket[s]]) e++;
    buckets.set(key[byBucket[s]], [s, e]);
    s = e;
  }
}

function smooth(train: Float32Array, walk: Float32Array, drive: Float32Array): [Float32Array, Float32Array, Float32Array] {
  const n = train.length;
  const t = new Float32Array(n), w = new Float32Array(n), d = new Float32Array(n);
  const inv = -1 / (2 * SMOOTH_SIGMA_M * SMOOTH_SIGMA_M), cut2 = SMOOTH_CUT_M * SMOOTH_CUT_M;
  for (let c = 0; c < n; c++) {
    const all = train[c] + walk[c] + drive[c];
    if (!(all > 0)) continue;
    const bx = Math.floor(mx[c] / SMOOTH_CUT_M), by = Math.floor(my[c] / SMOOTH_CUT_M);
    let st = 0, sw = 0, sd = 0;
    for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
      const b = buckets.get(bucketKey(bx + ox, by + oy));
      if (!b) continue;
      for (let k = b[0]; k < b[1]; k++) {
        const j = byBucket[k];
        const dx = mx[j] - mx[c], dy = my[j] - my[c], d2 = dx * dx + dy * dy;
        if (d2 > cut2) continue;
        const g = Math.exp(d2 * inv);
        (st += g * train[j]), (sw += g * walk[j]), (sd += g * drive[j]);
      }
    }
    const s = st + sw + sd;
    if (!(s > 0)) (t[c] = train[c]), (w[c] = walk[c]), (d[c] = drive[c]);
    else (t[c] = (all * st) / s), (w[c] = (all * sw) / s), (d[c] = (all * sd) / s);
  }
  return [t, w, d];
}

function build(end: "home" | "work", train: Float32Array, walk: Float32Array, drive: Float32Array) {
  const totals: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < train.length; i++) {
    totals[0] += train[i];
    totals[1] += walk[i];
    totals[2] += drive[i];
  }
  const [st, sw, sd] = smooth(train, walk, drive);
  const out: BubbleLevel[] = levels.map((g) => {
    const a = aggregate(g, cellX, cellY, null, train, walk, drive);
    const order = biggestFirst(a);
    const s = reorder(a, order);
    // the same cells and groups, so the same groups in the same order
    const sh = reorder(aggregate(g, cellX, cellY, null, st, sw, sd), order);
    const l: BubbleLevel = { discs: discs(s, 1 / BUBBLE_DENSITY[end], BUBBLE_ALPHA, 0, cur!), group: s.group, train: s.train, walk: s.walk, drive: s.drive };
    if (sh.group.length === s.group.length) l.shade = { train: sh.train, walk: sh.walk, drive: sh.drive };
    return l;
  });
  recolour(out);
  return { levels: out, totals, anchor: cur! };
}

self.onmessage = (e: MessageEvent<ToCommuters>) => {
  const msg = e.data;
  const t0 = performance.now();
  const post = (m: FromCommuters, transfer: Transferable[]) => (self as unknown as Worker).postMessage(m, transfer);
  if (msg.kind === "cells") {
    setCells(msg.xy, msg.h3, msg.frame, msg.anchor);
    // copies: the worker keeps its own for every build
    const ready: FromCommuters = { kind: "ready", ms: performance.now() - t0, cellX: cellX.slice(), cellY: cellY.slice(), anchor: cur!, levels: levels.map((l) => ({ ...l, groupOf: l.groupOf.slice(), off: l.off.slice(), members: l.members.slice() })) };
    post(ready, [ready.cellX.buffer, ready.cellY.buffer, ...ready.levels.flatMap((l) => [l.groupOf.buffer, l.off.buffer, l.members.buffer])]);
    return;
  }
  if (msg.kind === "anchor") {
    // a version already in force is not measured again
    if (!cur || msg.anchor.version !== cur.version) place(msg.anchor);
    const moved: FromCommuters = { kind: "moved", id: msg.id, anchor: cur!, cellX: cellX.slice(), cellY: cellY.slice() };
    post(moved, [moved.cellX.buffer, moved.cellY.buffer]);
    return;
  }
  if (msg.kind === "sum") {
    const out: FromCommuters = { kind: "summed", id: msg.id, levels: levels.map((g) => aggregate(g, cellX, cellY, msg.cells, msg.train, msg.walk, msg.drive)), anchor: cur!, ms: 0 };
    out.ms = performance.now() - t0;
    post(out, out.levels.flatMap((l) => [l.group.buffer, l.x.buffer, l.y.buffer, l.train.buffer, l.walk.buffer, l.drive.buffer]));
    return;
  }
  if (msg.kind === "build") {
    const r = build(msg.end, msg.train, msg.walk, msg.drive);
    const out: FromCommuters = { kind: "bubbles", id: msg.id, end: msg.end, ...r, ms: performance.now() - t0 };
    post(out, r.levels.flatMap((l) => [l.discs.buffer, l.group.buffer, l.train.buffer, l.walk.buffer, l.drive.buffer, ...(l.shade ? [l.shade.train.buffer, l.shade.walk.buffer, l.shade.drive.buffer] : [])]));
  }
};
