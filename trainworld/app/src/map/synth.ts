// Synthetic stress network around New York (the T-002 spike's), for `?synth=1&lines=500&trains=10000`.
// Deterministic. Debug only: the dock keeps showing the real world while the map shows this.
// Each line has one constant-speed phase per run; the same trips are used for every hour.

import { LINE_PALETTE, rgb } from "../game/palette";
import { MERC_K } from "./geo";
import type { NetworkBuffers, Strokes } from "./network";

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Hubs in real km east/north of the origin: Midtown, Lower Manhattan, Downtown Brooklyn,
// Long Island City, Jamaica, Newark, Jersey City, the Bronx hub.
const HUBS: [number, number][] = [
  [0.5, 2.0], [-1.0, -3.3], [1.2, -4.2], [3.0, 1.8], [16.0, 0.0], [-16.5, -4.0], [-4.0, -2.5], [3.5, 11.0],
];

export function makeSynth(lineCount: number, trainCount: number, seed = 7): NetworkBuffers & { totalKm: number; trips: Float32Array } {
  const rnd = mulberry32(seed);
  const lines: number[][] = [];
  for (let li = 0; li < lineCount; li++) {
    const kind = li % 10 < 3 ? "radial" : li % 10 < 6 ? "crosstown" : "walk";
    let x: number, y: number, heading: number, lengthKm: number, wiggle: number;
    if (kind === "radial") {
      const hub = HUBS[Math.floor(rnd() * HUBS.length)];
      x = hub[0] + (rnd() - 0.5) * 3;
      y = hub[1] + (rnd() - 0.5) * 3;
      heading = rnd() * Math.PI * 2;
      lengthKm = 10 + rnd() * 30;
      wiggle = 0.04;
    } else if (kind === "crosstown") {
      const r = Math.sqrt(rnd()) * 25;
      const a = rnd() * Math.PI * 2;
      x = Math.cos(a) * r;
      y = Math.sin(a) * r;
      heading = a + Math.PI / 2 + (rnd() - 0.5) * 0.8;
      lengthKm = 5 + rnd() * 20;
      wiggle = 0.03;
    } else {
      const r = Math.sqrt(rnd()) * 30;
      const a = rnd() * Math.PI * 2;
      x = Math.cos(a) * r;
      y = Math.sin(a) * r;
      heading = rnd() * Math.PI * 2;
      lengthKm = 5 + rnd() * 35;
      wiggle = 0.12;
    }
    const step = 0.25;
    const n = Math.max(2, Math.round(lengthKm / step) + 1);
    const pts: number[] = [];
    let turn = 0;
    for (let i = 0; i < n; i++) {
      // local y grows southward (mercator), so north is -y
      pts.push(x * 1000 * MERC_K, -y * 1000 * MERC_K);
      turn = turn * 0.85 + (rnd() - 0.5) * wiggle;
      heading += turn;
      x += Math.cos(heading) * step;
      y += Math.sin(heading) * step;
    }
    lines.push(pts);
  }

  const seg: number[] = [], colour: number[] = [], samples: number[] = [];
  const lineTable = new Float32Array(lineCount * 4);
  const lineColours = new Uint8Array(lineCount * 4);
  const meta = new Float32Array(lineCount * 6 * 4);
  const phases = new Float32Array(lineCount * 2 * 4);
  const trips: number[] = [];
  const perLine = Math.max(1, Math.round(trainCount / lineCount));
  let totalLen = 0;
  lines.forEach((p, li) => {
    for (let i = 0; i + 3 < p.length; i += 2) {
      seg.push(p[i], p[i + 1], p[i + 2], p[i + 3]);
      colour.push(li);
    }
    const cum = [0];
    for (let i = 2; i < p.length; i += 2) cum.push(cum[cum.length - 1] + Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]));
    const len = cum[cum.length - 1];
    const lenM = len / MERC_K;
    totalLen += len;
    const step = 50;
    const count = Math.max(2, Math.ceil(lenM / step) + 1);
    const start = samples.length / 2;
    let sg = 0;
    for (let k = 0; k < count; k++) {
      const d = Math.min(len, k * step * MERC_K);
      while (sg < cum.length - 2 && cum[sg + 1] < d) sg++;
      const t = (d - cum[sg]) / Math.max(1e-9, cum[sg + 1] - cum[sg]);
      samples.push(p[sg * 2] + (p[sg * 2 + 2] - p[sg * 2]) * t, p[sg * 2 + 1] + (p[sg * 2 + 3] - p[sg * 2 + 1]) * t);
    }
    lineTable.set([start, count, lenM, step], li * 4);
    lineColours.set([...rgb(LINE_PALETTE[li % LINE_PALETTE.length]), 255], li * 4);
    // One constant-speed phase per run (12-30 m/s), both runs; trips cover the hour.
    const v = 12 + rnd() * 18, tripS = lenM / v;
    for (let run = 0; run < 2; run++) {
      phases.set([0, 0, v, 0], (li * 2 + run) * 4);
      meta.set([li * 2 + run, 1, tripS, lenM], ((li * 2 + run) * 3) * 4);
    }
    const hw = (2 * tripS) / perLine;
    for (let dep = -tripS - rnd() * hw, k = 0; dep < 3600; dep += hw, k++) trips.push((li * 2 + (k % 2)) * 3, dep);
  });
  const n = colour.length;
  const strokes: Strokes = {
    seg: new Float32Array(seg), colour: new Uint32Array(colour), edge: new Uint32Array(n).fill(0xffffffff),
    level: new Float32Array(n), flags: new Float32Array(n), dist: new Float32Array(n), count: n,
  };
  const empty: Strokes = { seg: new Float32Array(0), colour: new Uint32Array(0), edge: new Uint32Array(0), level: new Float32Array(0), flags: new Float32Array(0), dist: new Float32Array(0), count: 0 };
  return {
    track: empty,
    lines: strokes,
    lineIds: lines.map((_, i) => String(i)),
    lineColours,
    stations: new Float32Array(0),
    stationIds: [],
    stationCount: 0,
    junctions: new Float32Array(0),
    junctionIds: [],
    phases,
    meta,
    samples: new Float32Array(samples),
    lineTable,
    edgePts: new Map(),
    trips: new Float32Array(trips),
    totalKm: totalLen / MERC_K / 1000,
  };
}
