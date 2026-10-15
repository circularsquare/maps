// Commuter bubbles for the whole city (T-097, SPEC 8), built off the main thread: one worker,
// started when the bubbles are first shown.
//
// - Once per city: each cell's position in local units, and its group at each aggregation level,
//   the cell's H3 parent at resolutions 8, 7 and 6 (and the cell itself at 9). A parent's index is
//   the cell's with the digits below its resolution set to 7, so grouping needs no H3 library.
// - Per demand answer and end (homes or jobs): commuters by mode summed into every level, each
//   bubble at its commuters' weighted centre, as discs ready to upload (map/bubbles.ts).

import { BUBBLE_ALPHA } from "../game/palette";
import { aggregate, biggestFirst, discs, reorder } from "../map/bubbles";
import { toLocal, type Anchor } from "../map/geo";
import { BUBBLE_DENSITY, BUBBLE_RES, type BubbleLevel, type FromCommuters, type Grouping, type ToCommuters } from "./commutersProtocol";

import { makeFrame, type FrameSpec } from "../game/frame";

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

function build(end: "home" | "work", train: Float32Array, walk: Float32Array, drive: Float32Array) {
  const totals: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < train.length; i++) {
    totals[0] += train[i];
    totals[1] += walk[i];
    totals[2] += drive[i];
  }
  const out: BubbleLevel[] = levels.map((g) => {
    const a = aggregate(g, cellX, cellY, null, train, walk, drive);
    const s = reorder(a, biggestFirst(a));
    return { discs: discs(s, 1 / BUBBLE_DENSITY[end], BUBBLE_ALPHA, 1, cur!), group: s.group, train: s.train, walk: s.walk, drive: s.drive };
  });
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
    post(out, r.levels.flatMap((l) => [l.discs.buffer, l.group.buffer, l.train.buffer, l.walk.buffer, l.drive.buffer]));
  }
};
