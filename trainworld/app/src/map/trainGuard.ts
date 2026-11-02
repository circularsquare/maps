// T-1965: trains are drawn from their own line's timetable, so a faster train would be drawn
// through a slower one ahead of it on the same track. The guard holds such a train back on screen:
// each frame, for the trains in or near the view on lines that share track with another line, a
// train may come no closer than a small gap behind the back of the train ahead on its track, nor
// onto a merge another train is on. Held trains are drawn late in their own run (`lag`, game
// seconds, per trip of the renderer's buffer), so they keep their stops and speeds, and once clear
// they make the lag up at LAG_DECAY. Lengths are the trains' own: zoomed out, where a train is drawn
// longer, lines sharing track are drawn side by side (T-1799). Display only: run times, boarding and
// demand go by the timetable (notes/T-1965.md says what is a lie). Flat crossings are off
// (`crossings`): on the real save's 340 they held hundreds of trains for many minutes (T-1966).

import { MARK_STRIDE, M_CROSSING } from "./markData";
import { profileS, type NetworkBuffers } from "./network";

/** Room left between a held train's front and the back of the train ahead, m. */
export const GUARD_GAP_M = 30;
/** A held train makes up its lag at this many seconds a game second: drawn up to 30% faster than
 * its timetable until it is back on it. */
const LAG_DECAY = 0.3;
/** No train is held further behind its timetable than this, s: past it the timetable is too far
 * from what can be drawn (a pattern the following loss does not foresee) and it is let through. */
const LAG_MAX = 1200;
/** A clock step longer than this (a load, a jump of the clock) starts every train on its timetable. */
const JUMP_S = 900;
/** Trains this far outside the view, CSS px, are guarded too, so one coming in is already in place. */
const VIEW_MARGIN_PX = 150;
/** Local units (about a metre) within which a flat crossing's mark is on an edge's centre line. */
const CROSS_TOL = 3;
const CELL = 256;

/** What the guard needs of the frame: the time and the view. */
export interface GuardFrame {
  /** game seconds, absolute */
  time: number;
  /** the trip buffer's epoch */
  epoch: number;
  trips: Float32Array;
  /** per trip, when it stops being drawn - epoch (T-206) */
  ends: Float32Array | null;
  /** per trip, the departure - epoch its train takes next, NaN none (T-245) */
  nexts: Float32Array | null;
  /** CSS px of a point (local units; the answer may be reused by the next call), null to guard every
   * train wherever it is */
  project: ((x: number, y: number) => readonly [number, number]) | null;
  /** the view, CSS px */
  w: number;
  h: number;
}

export class TrainGuard {
  /** per trip of the buffer: game seconds it is drawn behind its timetable */
  lag = new Float32Array(0);
  /** per trip: its offset along its run as last drawn, NaN when not guarded last frame */
  private prevS = new Float64Array(0);
  private trips: Float32Array | null = null;
  private epoch = NaN;
  private net: NetworkBuffers | null = null;
  /** lines worth guarding: sharing a track (an edge in one direction) with another line (or with
   * `crossings`, crossing one at a flat crossing) */
  private guarded = new Uint8Array(0);
  /** per edge: its flat crossings (crossing id, metres from node a), lazily */
  private crossOn = new Map<number, number[]>();
  private crossCells: Map<number, number[]> | null = null;
  /** the game time of the last update: the lags hold for the frame drawn at it */
  lastTime = NaN;
  /** the next trip of each line, found by its departure: line * 1e7 + round(dep * 100) */
  private nextIndex = new Map<number, number>();
  /** how many trips were guarded the last frame (checks) */
  guardedCount = 0;
  heldCount = 0;
  /** per line: its run 0 samples' box (x0, y0, x1, y1, local units), for guarded lines */
  private boxes: Float32Array | null = null;
  /** per line, this frame: 1 its box is in or near the view, 0 not, 2 not looked at yet */
  private seen = new Uint8Array(0);
  /** trains also wait for each other at flat crossings (off: T-1966); read on a network change */
  crossings = false;

  setNetwork(n: NetworkBuffers) {
    if (n === this.net) return;
    this.net = n;
    this.crossOn.clear();
    this.crossCells = null;
    this.boxes = null;
    const T = n.pathTable, E = n.pathEntry;
    const lines = (n.lineTable.length / 4) | 0;
    this.guarded = new Uint8Array(lines);
    if (!T || !E) return;
    // crossings: their marks, by cell
    const marks = n.marks;
    let cells: Map<number, number[]> | null = null;
    if (marks && this.crossings) for (let k = 0; k + MARK_STRIDE <= marks.length; k += MARK_STRIDE) {
      if (marks[k + 2] !== M_CROSSING) continue;
      cells ??= new Map();
      const key = cellKey(marks[k], marks[k + 1]);
      const v = cells.get(key);
      if (v) v.push(k / MARK_STRIDE);
      else cells.set(key, [k / MARK_STRIDE]);
    }
    this.crossCells = cells;
    // a track key is shared when two lines use it
    const owner = new Map<number, number>();
    const shared = new Set<number>();
    for (let l = 0; l < lines && 2 * l + 1 < T.length; l++) {
      const f = T[2 * l], c = T[2 * l + 1];
      for (let k = f; k < f + c; k++) {
        const p = E[k];
        for (const key of [p, p ^ 1]) {
          const o = owner.get(key);
          if (o === undefined) owner.set(key, l);
          else if (o !== l) shared.add(key);
        }
      }
    }
    for (let l = 0; l < lines && 2 * l + 1 < T.length; l++) {
      const f = T[2 * l], c = T[2 * l + 1];
      if ((n.lineCars?.[l] ?? 8) < 1) continue; // a bus runs on roads
      for (let k = f; k < f + c && !this.guarded[l]; k++) {
        if (shared.has(E[k]) || shared.has(E[k] ^ 1) || (cells && this.crossingsOn(E[k] >> 1, n.pathLen![k]).length)) this.guarded[l] = 1;
      }
    }
    const box = (this.boxes = new Float32Array(lines * 4));
    for (let l = 0; l < lines; l++) {
      if (!this.guarded[l]) continue;
      const first = n.lineTable[l * 4], count = n.lineTable[l * 4 + 1];
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let k = first; k < first + count; k++) {
        const x = n.samples[k * 2], y = n.samples[k * 2 + 1];
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
      box.set([x0, y0, x1, y1], l * 4);
    }
  }

  /** The flat crossings on edge `e` (crossing id, metres from its node a, pairs), from its points. */
  private crossingsOn(e: number, lenM: number): number[] {
    let out = this.crossOn.get(e);
    if (out) return out;
    out = [];
    const n = this.net!, cells = this.crossCells, pts = n.edgePts.get(e), marks = n.marks;
    if (cells && pts && marks && pts.length >= 6) {
      let total = 0;
      for (let k = 0; k + 5 < pts.length; k += 3) total += Math.hypot(pts[k + 3] - pts[k], pts[k + 4] - pts[k + 1]);
      let at = 0;
      const seen = new Set<number>();
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const x0 = pts[k], y0 = pts[k + 1], dx = pts[k + 3] - x0, dy = pts[k + 4] - y0, len = Math.hypot(dx, dy);
        const cx0 = Math.floor((Math.min(x0, x0 + dx) - CROSS_TOL) / CELL), cx1 = Math.floor((Math.max(x0, x0 + dx) + CROSS_TOL) / CELL);
        const cy0 = Math.floor((Math.min(y0, y0 + dy) - CROSS_TOL) / CELL), cy1 = Math.floor((Math.max(y0, y0 + dy) + CROSS_TOL) / CELL);
        for (let cx = cx0; cx <= cx1; cx++) for (let cy = cy0; cy <= cy1; cy++) {
          for (const id of cells.get(cx * 1_000_003 + cy) ?? []) {
            if (seen.has(id)) continue;
            const mx = marks[id * MARK_STRIDE], my = marks[id * MARK_STRIDE + 1];
            const t = len > 1e-9 ? Math.max(0, Math.min(1, ((mx - x0) * dx + (my - y0) * dy) / (len * len))) : 0;
            if (Math.hypot(x0 + t * dx - mx, y0 + t * dy - my) > CROSS_TOL) continue;
            seen.add(id);
            out.push(id, total > 0 ? ((at + t * len) / total) * lenM : 0);
          }
        }
        at += len;
      }
    }
    this.crossOn.set(e, out);
    return out;
  }

  /** The trip buffer changed: each trip keeps its lag and last place by its profile and departure. */
  private remap(f: GuardFrame) {
    const n = f.trips.length / 2;
    const lag = new Float32Array(n), prevS = new Float64Array(n).fill(NaN);
    const old = this.trips;
    if (old && this.lag.length) {
      const was = new Map<string, number>();
      for (let i = 0; i < this.lag.length; i++) if (this.lag[i] > 0 || !Number.isNaN(this.prevS[i])) was.set(`${old[i * 2]}|${Math.round((old[i * 2 + 1] + this.epoch) * 100)}`, i);
      if (was.size) for (let i = 0; i < n; i++) {
        const j = was.get(`${f.trips[i * 2]}|${Math.round((f.trips[i * 2 + 1] + f.epoch) * 100)}`);
        if (j !== undefined) { lag[i] = this.lag[j]; prevS[i] = this.prevS[j]; }
      }
    }
    this.lag = lag;
    this.prevS = prevS;
    this.trips = f.trips;
    this.epoch = f.epoch;
    this.nextIndex.clear();
    for (let i = 0; i < n; i++) {
      const line = Math.floor(f.trips[i * 2] / 6);
      if (this.guarded[line]) this.nextIndex.set(line * 1e7 + Math.round(f.trips[i * 2 + 1] * 100), i);
    }
  }

  /**
   * Work out every guarded trip's lag for the frame; true when `lag` changed. Candidates are the
   * trips of guarded lines drawn in or near the view.
   */
  update(f: GuardFrame): boolean {
    const n = this.net;
    if (!n || !n.pathTable || !n.pathEntry || !n.pathFrom || !n.pathLen) return false;
    if (f.trips !== this.trips || f.epoch !== this.epoch) this.remap(f);
    const dt = f.time - this.lastTime;
    this.lastTime = f.time;
    const lag = this.lag, prevS = this.prevS;
    let changed = false;
    if (!(dt >= 0 && dt <= JUMP_S)) {
      // a jump of the clock: everyone back on their timetable
      for (let i = 0; i < lag.length; i++) if (lag[i]) { lag[i] = 0; changed = true; }
      prevS.fill(NaN);
    }
    const decay = dt >= 0 && dt <= JUMP_S ? LAG_DECAY * dt : 0;
    const uTime = f.time - f.epoch;
    const T = n.pathTable, E = n.pathEntry, PF = n.pathFrom, PL = n.pathLen, meta = n.meta;
    // the candidates: trip, offset along its run now, half its length
    const cand: number[] = [], cs: number[] = [], half: number[] = [];
    const tauOf: number[] = [];
    const nTrips = f.trips.length / 2;
    const lo = -VIEW_MARGIN_PX, hiX = f.w + VIEW_MARGIN_PX, hiY = f.h + VIEW_MARGIN_PX;
    // a guarded line whose box is off the view has no train to look at (1 in view, 0 off, 2 not
    // worked out yet this frame)
    const seen = this.seen.length === this.guarded.length ? this.seen.fill(2) : (this.seen = new Uint8Array(this.guarded.length).fill(2));
    const box = this.boxes;
    for (let i = 0; i < nTrips; i++) {
      const p = f.trips[i * 2], line = Math.floor(p / 6);
      const l0 = lag[i];
      const li = l0 > decay ? l0 - decay : 0;
      if (this.guarded[line] && seen[line] === 2) {
        seen[line] = 1;
        if (f.project && box) {
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (const [bx, by] of [[box[line * 4], box[line * 4 + 1]], [box[line * 4 + 2], box[line * 4 + 1]], [box[line * 4], box[line * 4 + 3]], [box[line * 4 + 2], box[line * 4 + 3]]]) {
            const q = f.project(bx, by);
            x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]); y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]);
          }
          // (a corner behind the camera projects to nonsense: kept)
          if (Number.isFinite(x0 + x1 + y0 + y1) && (x1 < lo || x0 > hiX || y1 < lo || y0 > hiY)) seen[line] = 0;
        }
      }
      if (!this.guarded[line] || !seen[line]) {
        if (l0) { lag[i] = li; changed = true; }
        prevS[i] = NaN;
        continue;
      }
      const dep = f.trips[i * 2 + 1];
      const tau = uTime - dep - li;
      const end = f.ends ? f.ends[i] : -Infinity;
      if (meta[p * 4 + 1] < 0.5 || tau < 0 || (tau > meta[p * 4 + 2] && uTime > end)) {
        if (l0 !== li) { lag[i] = li; changed = true; }
        prevS[i] = NaN;
        continue;
      }
      const s = profileS(n, p, tau);
      if (f.project) {
        // the nearest run 0 sample is near enough for the view
        const run = Math.floor(p / 3) % 2, d = run === 0 ? s : meta[p * 4 + 3] - s;
        const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
        const si = first + Math.max(0, Math.min(count - 1, Math.round(d / step)));
        const q = f.project(n.samples[si * 2], n.samples[si * 2 + 1]);
        const px = q[0], py = q[1];
        if (!(px >= lo && px <= hiX && py >= lo && py <= hiY)) {
          if (l0 !== li) { lag[i] = li; changed = true; }
          prevS[i] = NaN;
          continue;
        }
      }
      cand.push(i);
      cs.push(s);
      half.push(Math.min(Math.max(1, n.lineCars?.[line] ?? 8) * 20, meta[p * 4 + 3]) / 2);
      tauOf.push(tau);
      if (l0 !== li) { lag[i] = li; changed = true; }
    }
    this.guardedCount = cand.length;
    this.heldCount = 0;
    if (cand.length < 2) {
      if (cand.length) prevS[cand[0]] = cs[0];
      return changed;
    }
    const gapM = GUARD_GAP_M;
    // Each candidate's place on every track (an edge in its direction of travel) and flat crossing
    // from its back (where it was or is, the further back) to just past its front: the coordinate
    // on a track is from the edge's first node in the direction of travel, on a crossing from the
    // crossing, so a candidate's coordinate there is its offset along its run plus `off`.
    const regs = new Map<number, number[]>();
    const sNew = cs.slice();
    for (let c = 0; c < cand.length; c++) {
      const i = cand[c], p = f.trips[i * 2], line = Math.floor(p / 6), run = Math.floor(p / 3) % 2;
      const L = meta[p * 4 + 3], first = T[line * 2], count = T[line * 2 + 1];
      const was = prevS[i];
      const a = Math.min(cs[c], Number.isNaN(was) ? cs[c] : was) - half[c], b = cs[c] + half[c] + gapM + 10;
      // the entries in run 0's offsets [d0, d1]
      const d0 = run === 0 ? a : L - b, d1 = run === 0 ? b : L - a;
      let k = upper(PF, first, first + count, d0) - 1;
      if (k < first) k = first;
      for (; k < first + count && PF[k] <= d1; k++) {
        const len = PL[k];
        if (PF[k] + len < d0) continue;
        const ent = E[k];
        const key = run === 0 ? ent : ent ^ 1;
        // where the entry starts along this run
        const start = run === 0 ? PF[k] : L - PF[k] - len;
        push(regs, key * 2, c, -start, len, -1);
        const xs = this.crossings ? this.crossingsOn(ent >> 1, len) : NO_CROSSINGS;
        for (let q = 0; q < xs.length; q += 2) {
          // along this run's direction of travel
          const along = (key & 1) === 0 ? xs[q + 1] : len - xs[q + 1];
          if (start + along < a - 1 || start + along > b + 1) continue;
          push(regs, -(xs[q] + 1) * 2 - 1, c, -(start + along), 0, ent >> 1);
        }
      }
    }
    // order each place's candidates by where their fronts were (no overtaking on screen), then hold
    // each behind the backs of those ahead that are on it and have not cleared it; at a crossing,
    // only those on the other route (trains passing each other on one double track do not meet)
    const lists: number[][] = [];
    for (const v of regs.values()) {
      if (v.length < 2 * REG) continue;
      const m = v.length / REG, order = Array.from({ length: m }, (_, j) => j);
      const fr = order.map((j) => {
        const c = v[j * REG], was = prevS[cand[c]];
        return (Number.isNaN(was) ? cs[c] : Math.min(was, cs[c])) + v[j * REG + 1] + half[c];
      });
      order.sort((x, y) => fr[y] - fr[x] || v[x * REG] - v[y * REG]);
      const out: number[] = [];
      for (const j of order) for (let q = 0; q < REG; q++) out.push(v[j * REG + q]);
      lists.push(out);
    }
    // where a train on it has its back, clamped to the place's start; Infinity when it holds no one
    const holds = (c: number, off: number, zone: number) => {
      const back = sNew[c] + off - half[c];
      return back < zone && back + 2 * half[c] > 0 ? Math.max(back, 0) : Infinity;
    };
    for (let pass = 0; pass < 6; pass++) {
      let moved = false;
      for (const v of lists) {
        let m = Infinity;
        for (let j = 0; j < v.length; j += REG) {
          const c = v[j], off = v[j + 1], zone = v[j + 2], edge = v[j + 3];
          if (edge >= 0) {
            m = Infinity;
            for (let q = 0; q < j; q += REG) if (v[q + 3] !== edge) m = Math.min(m, holds(v[q], v[q + 1], v[q + 2]));
          }
          if (m < Infinity) {
            const cap = m - gapM - off - half[c];
            if (sNew[c] > cap + 1e-6) {
              sNew[c] = cap;
              moved = true;
            }
          }
          if (edge < 0) m = Math.min(m, holds(c, off, zone));
        }
      }
      if (!moved) break;
    }
    // a held candidate's lag: the time in its run when it was at its new place
    for (let c = 0; c < cand.length; c++) {
      const i = cand[c], p = f.trips[i * 2];
      if (sNew[c] < cs[c] - 1e-3) {
        const tau = timeAt(n, p, Math.max(0, sNew[c]), tauOf[c]);
        const l = Math.min(LAG_MAX, uTime - f.trips[i * 2 + 1] - tau);
        if (l > lag[i]) { lag[i] = l; changed = true; }
        this.heldCount++;
      }
      prevS[i] = sNew[c] < cs[c] - 1e-3 ? profileS(n, p, uTime - f.trips[i * 2 + 1] - lag[i]) : cs[c];
      // still running when its train is due out again: its next trip leaves that much later
      if (lag[i] > 0 && f.ends && f.nexts) {
        const dep = f.trips[i * 2 + 1], arrive = dep + meta[p * 4 + 2] + lag[i], next = f.nexts[i];
        if (arrive > f.ends[i] && Number.isFinite(next)) {
          const j = this.nextIndex.get(Math.floor(p / 6) * 1e7 + Math.round(next * 100));
          if (j !== undefined && lag[j] < Math.min(LAG_MAX, arrive - next)) { lag[j] = Math.min(LAG_MAX, arrive - next); changed = true; }
        }
      }
    }
    return changed;
  }
}

const NO_CROSSINGS: number[] = [];
const cellKey =(x: number, y: number) => Math.floor(x / CELL) * 1_000_003 + Math.floor(y / CELL);

/** Numbers per registration: candidate, offset to its coordinate there, the place's length (0 a
 * crossing), the edge it crosses on (-1 on a track). */
const REG = 4;

function push(regs: Map<number, number[]>, key: number, c: number, off: number, zone: number, edge: number) {
  const v = regs.get(key);
  if (v) {
    // a run over the same edge twice (out and back on one track) counts its nearer pass
    for (let j = 0; j < v.length; j += REG) if (v[j] === c) return;
    v.push(c, off, zone, edge);
  } else regs.set(key, [c, off, zone, edge]);
}

/** The first index in [lo, hi) whose value is above `x`. */
function upper(a: Float32Array, lo: number, hi: number, x: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The latest trip time, no later than `tauMax`, at which profile `p` is at offset `s` or short
 * of it: the inverse of `profileS`. */
export function timeAt(n: NetworkBuffers, p: number, s: number, tauMax: number): number {
  const ph = n.phases, first = n.meta[p * 4], count = n.meta[p * 4 + 1];
  if (!count) return tauMax;
  // the last phase starting at or before s
  let lo = first, hi = first + count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ph[mid * 4 + 1] <= s) lo = mid;
    else hi = mid - 1;
  }
  const t = ph[lo * 4], s0 = ph[lo * 4 + 1], v = ph[lo * 4 + 2], a = ph[lo * 4 + 3];
  const ds = Math.max(0, s - s0);
  let tt: number;
  if (v === 0 && a === 0) {
    // standing there: until the next phase starts
    tt = lo + 1 < first + count ? ph[(lo + 1) * 4] - t : 0;
  } else if (Math.abs(a) < 1e-9) tt = ds / v;
  else {
    const disc = v * v + 2 * a * ds;
    tt = disc <= 0 ? (a < 0 ? v / -a : 0) : (-v + Math.sqrt(disc)) / a;
  }
  return Math.min(tauMax, t + Math.max(0, tt));
}
