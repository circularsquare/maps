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
import { toLocal } from "../map/geo";
import { BUBBLE_DENSITY, BUBBLE_RES, type BubbleLevel, type FromCommuters, type Grouping, type ToCommuters } from "./commutersProtocol";

const R_EARTH = 6371008.8;
const DEG = 180 / Math.PI;

let cellX = new Float32Array(0), cellY = new Float32Array(0);
let levels: Grouping[] = [];

function setCells(xy: Float32Array, h3: BigUint64Array, origin: { lon: number; lat: number }) {
  const n = xy.length / 2;
  cellX = new Float32Array(n);
  cellY = new Float32Array(n);
  const cosO = Math.cos(origin.lat / DEG);
  for (let i = 0; i < n; i++) {
    const lng = origin.lon + (xy[i * 2] / (R_EARTH * cosO)) * DEG;
    const lat = origin.lat + (xy[i * 2 + 1] / R_EARTH) * DEG;
    const [x, y] = toLocal(lng, lat);
    cellX[i] = x;
    cellY[i] = y;
  }
  levels = BUBBLE_RES.map((res) => {
    const groupOf = new Uint32Array(n);
    let count = 0;
    if (res === 9) for (let i = 0; i < n; i++) groupOf[i] = count++;
    else {
      const fill = (1n << BigInt(3 * (15 - res))) - 1n;
      const ids = new Map<bigint, number>();
      for (let i = 0; i < n; i++) {
        const key = h3[i] | fill;
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
    return { discs: discs(s, 1 / BUBBLE_DENSITY[end], BUBBLE_ALPHA, 1), group: s.group, train: s.train, walk: s.walk, drive: s.drive };
  });
  return { levels: out, totals };
}

self.onmessage = (e: MessageEvent<ToCommuters>) => {
  const msg = e.data;
  const t0 = performance.now();
  const post = (m: FromCommuters, transfer: Transferable[]) => (self as unknown as Worker).postMessage(m, transfer);
  if (msg.kind === "cells") {
    setCells(msg.xy, msg.h3, msg.origin);
    // copies: the worker keeps its own for every build
    const ready: FromCommuters = { kind: "ready", ms: performance.now() - t0, cellX: cellX.slice(), cellY: cellY.slice(), levels: levels.map((l) => ({ ...l, groupOf: l.groupOf.slice(), off: l.off.slice(), members: l.members.slice() })) };
    post(ready, [ready.cellX.buffer, ready.cellY.buffer, ...ready.levels.flatMap((l) => [l.groupOf.buffer, l.off.buffer, l.members.buffer])]);
    return;
  }
  if (msg.kind === "sum") {
    const out: FromCommuters = { kind: "summed", id: msg.id, levels: levels.map((g) => aggregate(g, cellX, cellY, msg.cells, msg.train, msg.walk, msg.drive)), ms: 0 };
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
