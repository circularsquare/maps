// T-1567: the start menu's logo, "trains" over "world" in joined-up, slanted script built as a small
// network the renderer draws as it draws the game's (after Anita's sketch in the track tool): each
// stroke a line with its stations, overlapping boxes where strokes join or cross (an interchange),
// and one train a line with the game's stop and start. No sim: the lines, stations and timetables are
// made here as the renderer's buffers (NetworkBuffers), as synth.ts makes its stress network.
//
// Letters are drawn upright as straight runs joined by curves of one radius (CURVE_R; a loop or a
// bowl may be wider), then stretched and slanted together, so the script keeps an even curve.
// Where a curve would be tighter the track reverses instead (a switchback: the train stops with its
// front at the end of a short stub and runs back out) or one stroke branches off another.

import { LINE_PALETTE, rgb } from "../game/palette";
import { makeAnchor, pxPerUnitAt, unitPerPxAt } from "./geo";
import type { NetworkBuffers, Strokes } from "./network";
import { PLATFORM_HALF_M, platformPoints, platformShape } from "./stationGeometry";

/** The zoom the logo is drawn at: boxes as the game draws them close in. */
export const LOGO_ZOOM = 15;
/** Metres a design px spans at LOGO_ZOOM (on the equator a local unit is a metre). */
const M = unitPerPxAt(LOGO_ZOOM);
const ANCHOR = makeAnchor(0, 0, -1);

/** platform length, m: longer than a train (TRAIN_HALF_M) */
const PLATFORM_M = 74;
/** half a train as the logo draws it, m: one pill, renderer.ts's shortest at LOGO_ZOOM (9 px wide,
 * 2.2 / 0.75 times as long) */
const TRAIN_HALF_M = (0.5 * 9 * 2.2) / 0.75 / pxPerUnitAt(LOGO_ZOOM);
/** running: top speed m/s, acceleration and braking m/s², the stop at a station s, at a switchback,
 * at a terminal s */
const V = 14, ACC = 0.9, DWELL = 10, REVERSE = 6, TURN = 16;
/** the forward slant (x per y) and the upward stretch applied to the upright letters */
const SLANT = 0.22, STRETCH = 0.9;
/** design px a number in the letters below spans */
const K = 1.5;
/** every curve's radius in the upright letters, their units */
const CURVE_R = 12;

type P = [number, number];
/** a corner point: its radius when not CURVE_R, or a switchback (the track ends there and the train
 * reverses) */
type C = [number, number] | [number, number, { r?: number; rev?: true }];
/** A stroke: its corner points (upright, y up from its row's baseline) or a circle, where trains stop
 * besides its ends (points near it), its palette colour. */
interface Stroke { pts?: C[]; circle?: [number, number, number]; stops?: P[]; colour: number }

/** The two rows, x-height 56, ascenders to 96; "world" set right so "trains" sits over its "wor".
 * Stations are the ends and the stops listed. */
const ROWS: { strokes: Stroke[]; dx: number }[] = [
  { dx: 0, strokes: [
    // the lead-in turning up the t's stem to a switchback at its top, back down past the fork and
    // on along the baseline into the r
    { pts: [[-70, 0], [10, 0, { r: 30 }], [10, 96, { rev: true }], [10, 0], [40, 0], [40, 56], [64, 56]], stops: [[10, 68]], colour: 1 },
    // the t's bar, across the stem
    { pts: [[-12, 68], [34, 68]], colour: 0 },
    // the a: round its bowl, up its stem and back, on along the baseline, up the i and back
    { pts: [[114, 52], [78, 52, { r: 18 }], [78, 0, { r: 18 }], [114, 0, { r: 18 }], [114, 53, { rev: true }], [114, 0], [144, 0], [144, 56, { rev: true }], [144, 0], [162, 0]], stops: [[144, 30]], colour: 2 },
    // the i's dot
    { circle: [146, 80, 10], colour: 11 },
    // from the i's foot the n (as wide as the o), then the s squiggling up
    { pts: [[150, 0], [176, 0], [176, 56], [214, 56], [214, 0], [250, 0], [250, 28], [226, 28, { r: 14 }], [226, 56, { r: 14 }], [252, 56]], stops: [[195, 56]], colour: 4 },
  ] },
  { dx: 78, strokes: [
    // the w, its middle a switchback
    { pts: [[0, 56], [0, 0], [27, 0], [27, 44, { rev: true }], [27, 0], [54, 0], [54, 56]], colour: 7 },
    // from the w's foot along the baseline, round the o's loop and on into the r
    { pts: [[50, 0], [120, 0, { r: 18 }], [120, 52, { r: 18 }], [80, 52, { r: 18 }], [80, 0, { r: 18 }], [140, 0], [140, 56], [164, 56]], stops: [[100, 52]], colour: 0 },
    // the l, along the baseline, round the d's bowl and up its stem
    { pts: [[180, 96], [180, 0], [242, 0], [242, 52, { r: 18 }], [206, 52, { r: 18 }], [206, 0, { r: 18 }], [242, 0], [244, 96]], stops: [[196, 0], [211, 42]], colour: 5 },
    // the exclamation mark: its stem, and a dot like the i's
    { pts: [[272, 96], [272, 30]], colour: 1 },
    { circle: [272, 12, 9], colour: 11 },
  ] },
];

/** The corner points with each corner turned on a circle of its radius (shortened where the legs are
 * too short), sampled every few degrees; a switchback's point stays a point. Returns the polyline and
 * the switchbacks' indices in it. */
function rounded(pts: C[]): { p: P[]; revs: number[] } {
  const n = pts.length;
  const rOf = (i: number) => (pts[i][2]?.rev ? 0 : pts[i][2]?.r ?? CURVE_R);
  const dir = (a: C, b: C): P => { const l = Math.hypot(b[0] - a[0], b[1] - a[1]); return [(b[0] - a[0]) / l, (b[1] - a[1]) / l]; };
  // each corner's turn and the leg it wants for its curve, cut back where two share too short a leg
  const turn = new Array(n).fill(0), want = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const u0 = dir(pts[i - 1], pts[i]), u1 = dir(pts[i], pts[i + 1]);
    turn[i] = Math.acos(Math.max(-1, Math.min(1, u0[0] * u1[0] + u0[1] * u1[1])));
    want[i] = rOf(i) > 0 && turn[i] > 1e-3 && turn[i] < Math.PI - 1e-3 ? rOf(i) * Math.tan(turn[i] / 2) : 0;
  }
  const t = want.slice();
  for (let i = 0; i + 1 < n; i++) {
    const leg = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
    if (t[i] + t[i + 1] > leg) { const f = leg / (t[i] + t[i + 1]); t[i] *= f; t[i + 1] *= f; }
  }
  const out: P[] = [[pts[0][0], pts[0][1]]], revs: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    const [bx, by] = pts[i];
    if (t[i] <= 0) { if (pts[i][2]?.rev) revs.push(out.length); out.push([bx, by]); continue; }
    const u0 = dir(pts[i - 1], pts[i]), u1 = dir(pts[i], pts[i + 1]);
    const rr = t[i] / Math.tan(turn[i] / 2);
    const p0: P = [bx - u0[0] * t[i], by - u0[1] * t[i]];
    const left = u0[0] * u1[1] - u0[1] * u1[0] > 0;
    const nx = left ? -u0[1] : u0[1], ny = left ? u0[0] : -u0[0];
    const ox = p0[0] + nx * rr, oy = p0[1] + ny * rr;
    const a0 = Math.atan2(p0[1] - oy, p0[0] - ox), steps = Math.max(2, Math.ceil(turn[i] / (Math.PI / 36)));
    for (let k = 0; k <= steps; k++) {
      const a = a0 + (left ? 1 : -1) * turn[i] * (k / steps);
      out.push([ox + rr * Math.cos(a), oy + rr * Math.sin(a)]);
    }
  }
  out.push([pts[n - 1][0], pts[n - 1][1]]);
  return { p: out, revs };
}

/** A polyline with its distances along. */
class Path {
  cum: number[] = [0];
  constructor(readonly p: P[], readonly closed: boolean) {
    for (let i = 1; i < p.length; i++) this.cum.push(this.cum[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
  }
  get length() { return this.cum[this.cum.length - 1]; }
  at(d: number): P {
    const L = this.length;
    if (this.closed) d = ((d % L) + L) % L;
    else d = Math.max(0, Math.min(L, d));
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < d) i++;
    const f = (d - this.cum[i - 1]) / Math.max(1e-9, this.cum[i] - this.cum[i - 1]);
    return [this.p[i - 1][0] + (this.p[i][0] - this.p[i - 1][0]) * f, this.p[i - 1][1] + (this.p[i][1] - this.p[i - 1][1]) * f];
  }
  /** the distance along of the point nearest `q` */
  near(q: P): number {
    let best = 0, bestD = Infinity;
    for (let i = 1; i < this.p.length; i++) {
      const [ax, ay] = this.p[i - 1], [bx, by] = this.p[i];
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((q[0] - ax) * dx + (q[1] - ay) * dy) / l2)) : 0;
      const e = Math.hypot(ax + dx * t - q[0], ay + dy * t - q[1]);
      if (e < bestD - 1e-9) [bestD, best] = [e, this.cum[i - 1] + t * Math.sqrt(l2)];
    }
    return best;
  }
}

/** upright units between the rows' baselines; design px round the drawing */
const ROW_STEP = 84, MARGIN = 22;

export interface LogoNetwork extends NetworkBuffers {
  trips: Float32Array;
  ends: Float32Array;
  /** the drawing's size, design px, and its middle in mercator 0..1 (the camera's centre) */
  w: number;
  h: number;
  mx: number;
  my: number;
  /** the timetable's period, s: every line's round trip divides it, so time can wrap there */
  period: number;
}

/** Periods every round trip is made to divide (divisors of an hour). */
const CYCLES = [60, 72, 75, 80, 90, 100, 120, 144, 150, 180, 200, 225, 240, 300, 360, 400, 450, 600, 720, 900, 1200, 1800, 3600];

/** A stop: where the train arrives and where it leaves from, m along its run; at a switchback it
 * leaves from the stub's far side (the same place on the ground, run back), else the two are one. */
type Stop = { arrive: number; leave: number; rev: boolean };

/** One run's phase table: a train starts from rest at each stop, runs up to V and brakes to the next;
 * braking holds it there until the next phase (rendererShaders.ts POS_GLSL). Returns the phases and
 * the time it reaches its last stop. */
function runPhases(stops: Stop[]): { ph: number[]; t: number } {
  const ph: number[] = [];
  let t = 0;
  for (let k = 0; k + 1 < stops.length; k++) {
    const s0 = stops[k].leave, len = stops[k + 1].arrive - s0;
    const reach = (V * V) / ACC;
    if (len >= reach) {
      const ta = V / ACC, cruise = (len - reach) / V;
      ph.push(t, s0, 0, ACC, t + ta, s0 + reach / 2, V, 0, t + ta + cruise, s0 + len - reach / 2, V, -ACC);
      t += 2 * ta + cruise;
    } else {
      const vp = Math.sqrt(ACC * Math.max(len, 0.01)), ta = vp / ACC;
      ph.push(t, s0, 0, ACC, t + ta, s0 + len / 2, vp, -ACC);
      t += 2 * ta;
    }
    if (k + 2 < stops.length) t += stops[k + 1].rev ? REVERSE : DWELL;
  }
  return { ph, t };
}

/** The logo's network; deterministic. */
export function logoNetwork(): LogoNetwork {
  // upright units to design px, stretched and slanted, y up from row 0's baseline
  const place = (p: P, dx: number, row: number): P => {
    const y = p[1] * STRETCH;
    return [K * (p[0] + dx + SLANT * y), K * (y - row * ROW_STEP)];
  };
  const hD = TRAIN_HALF_M / M;
  const strokes = ROWS.flatMap((row, ri) => row.strokes.map((s) => {
    let pl: P[], revs: number[] = [], closed = false;
    if (s.circle) {
      const [cx, cy, r] = s.circle;
      pl = Array.from({ length: 73 }, (_, k): P => [cx + r * Math.sin((k * Math.PI) / 36), cy - r * Math.cos((k * Math.PI) / 36)]);
      closed = true;
    } else ({ p: pl, revs } = rounded(s.pts!));
    const path = new Path(pl.map((p) => place(p, row.dx, ri)), closed);
    const len = path.length;
    const stops: Stop[] = [];
    const at = (d: number) => stops.push({ arrive: d, leave: d, rev: false });
    // a terminal's train stops at the middle of its platform, which runs back from the track's end
    const endD = PLATFORM_M / 2 / M;
    if (closed) { at(0); at(len); } else { at(endD); at(len - endD); }
    for (const p of s.stops ?? []) at(path.near(place(p, row.dx, ri)));
    for (const i of revs) stops.push({ arrive: path.cum[i] - hD, leave: path.cum[i] + hD, rev: true });
    stops.sort((a, b) => a.arrive - b.arrive);
    return { path, stops, colour: s.colour };
  }));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of strokes) for (const [x, y] of s.path.p) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const w = x1 - x0 + 2 * MARGIN, h = y1 - y0 + 2 * MARGIN;
  // to local units (metres, y down), the drawing's top left at (0, 0)
  const toLocal = (p: P): P => [(p[0] - x0 + MARGIN) * M, (y1 - p[1] + MARGIN) * M];

  const seg: number[] = [], colour: number[] = [], dist: number[] = [], samples: number[] = [];
  const lineTable: number[] = [], lineColours: number[] = [], lineCars: number[] = [], lineIds: string[] = [];
  const phases: number[] = [], meta: number[] = [], trips: number[] = [], ends: number[] = [];
  const stations: number[] = [], axis: number[] = [], shape: number[] = [], stLine: number[] = [], stIds: string[] = [];
  const STEP = 2;
  /** a station `d` m along a local polyline, its platform found from the track either side as the
   * game's are */
  const addStation = (path: Path, d: number) => {
    const way = (sign: number) => {
      const out: number[] = [];
      for (let k = 0; k <= 30; k++) {
        const e = d + sign * k * 2;
        if (!path.closed && (e < 0 || e > path.length)) break;
        out.push(...path.at(e), 0);
      }
      return Float32Array.from(out);
    };
    const sh = platformShape([way(1), way(-1)].filter((p) => p.length >= 6), 0, PLATFORM_M / 2);
    const at = path.at(d);
    stations.push(at[0], at[1], 4);
    axis.push(sh.ax, sh.ay);
    shape.push(sh.mx, sh.my, sh.phi, PLATFORM_HALF_M);
    stLine.push(...Array.from(platformPoints(sh)));
    stIds.push(String(stIds.length));
  };
  strokes.forEach((s, li) => {
    const pl = s.path.p.map(toLocal);
    const path = new Path(pl, s.path.closed);
    const len = path.length;
    for (let i = 1; i < pl.length; i++) {
      seg.push(pl[i - 1][0], pl[i - 1][1], pl[i][0], pl[i][1]);
      colour.push(li);
      dist.push(path.cum[i - 1]);
    }
    // run 0's samples every STEP m
    const first = samples.length / 2, count = Math.ceil(len / STEP) + 1;
    for (let k = 0; k < count; k++) samples.push(...path.at(Math.min(len, k * STEP)));
    lineTable.push(first, count, len, STEP);
    lineColours.push(...rgb(LINE_PALETTE[s.colour]), 255);
    lineCars.push(1);
    lineIds.push(String(li));
    const stops = s.stops.map((x) => ({ arrive: x.arrive * M, leave: x.leave * M, rev: x.rev }));
    stops.forEach((x, k) => {
      if (x.rev || (s.path.closed && k === stops.length - 1)) return;
      const end = !s.path.closed && (k === 0 || k === stops.length - 1);
      // a terminal's node is the track's end, its platform running back along the track from there
      addStation(path, end ? (k === 0 ? 0 : len) : x.arrive);
    });
    // the timetable: one train, there and back, run 1 from the far end
    const r0 = runPhases(stops);
    const r1 = runPhases(stops.map((x) => ({ arrive: len - x.leave, leave: len - x.arrive, rev: x.rev })).reverse());
    const tripS = Math.max(r0.t, r1.t);
    const cycle = CYCLES.find((c) => c >= 2 * (tripS + TURN)) ?? 3600;
    for (const [run, r] of [[0, r0], [1, r1]] as const) {
      const prof = (li * 2 + run) * 3;
      meta[prof * 4] = phases.length / 4;
      meta[prof * 4 + 1] = r.ph.length / 4;
      meta[prof * 4 + 2] = r.t;
      meta[prof * 4 + 3] = len;
      phases.push(...r.ph);
    }
    // the train leaves at a different moment on each line, so they are not all in step
    const off = ((li * 0.37) % 1) * cycle;
    for (let dep = off - 2 * cycle; dep < 3600 + cycle; dep += cycle) {
      trips.push(li * 6, dep);
      ends.push(dep + cycle / 2);
      trips.push(li * 6 + 3, dep + cycle / 2);
      ends.push(dep + cycle);
    }
  });
  for (let i = 0; i < strokes.length * 6 * 4; i++) meta[i] ??= 0;
  // interchanges: a station another's box overlaps
  const S = stations;
  for (let i = 0; i < S.length / 3; i++)
    for (let j = 0; j < S.length / 3; j++)
      if (i !== j && Math.hypot(S[i * 3] - S[j * 3], S[i * 3 + 1] - S[j * 3 + 1]) < PLATFORM_M) S[i * 3 + 2] = Math.round(S[i * 3 + 2]) | 1;

  const nSeg = colour.length;
  const empty: Strokes = { seg: new Float32Array(0), colour: new Uint32Array(0), edge: new Uint32Array(0), level: new Float32Array(0), flags: new Float32Array(0), dist: new Float32Array(0), count: 0 };
  const lines: Strokes = { seg: new Float32Array(seg), colour: new Uint32Array(colour), edge: new Uint32Array(nSeg).fill(0xffffffff), level: new Float32Array(nSeg), flags: new Float32Array(nSeg), dist: new Float32Array(dist), count: nSeg };
  const stationCount = stIds.length;
  return {
    track: empty, lines, lineIds, lineColours: new Uint8Array(lineColours),
    stations: new Float32Array(stations), stationAxis: new Float32Array(axis), stationShape: new Float32Array(shape),
    stationLine: new Float32Array(stLine), stationIds: stIds, stationCount, stationLength: new Float32Array(stationCount).fill(PLATFORM_M),
    junctions: new Float32Array(0), junctionIds: [],
    phases: new Float32Array(phases), meta: new Float32Array(meta), samples: new Float32Array(samples),
    lineTable: new Float32Array(lineTable), lineCars: new Float32Array(lineCars), edgePts: new Map(), anchor: ANCHOR,
    trips: new Float32Array(trips), ends: new Float32Array(ends),
    w, h, mx: ANCHOR.merc[0] + (w / 2) * M / 40075016.686, my: ANCHOR.merc[1] + (h / 2) * M / 40075016.686, period: 3600,
  };
}
