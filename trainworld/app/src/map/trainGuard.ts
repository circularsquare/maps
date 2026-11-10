// T-1965: trains are drawn from their own line's timetable, so a faster train would be drawn
// through a slower one ahead of it on the same track. The guard holds such a train back on screen:
// each frame, for the trains in or near the view on lines that share track with another line, a
// train may come no closer than a small gap behind the back of the train ahead on its track, nor
// onto a merge another train is on. Held trains are drawn late in their own run (`lag`, game
// seconds, per trip of the renderer's buffer), so they keep their stops and speeds, and once clear
// they make the lag up at LAG_DECAY. Lengths are the trains' own: zoomed out, where a train is drawn
// longer, lines sharing track are drawn side by side (T-1799). Display only: run times, boarding and
// demand go by the timetable (notes/T-1965.md says what is a lie).
//
// T-1966: trains also wait for each other at flat crossings and at junction moves that cross (the
// track model's own conflicts, `TrackApi.guard_conflicts`). There the rule is by time, not by where
// trains are drawn in a frame: each train's time over the conflict point comes from its timetable
// (its lag included), and one reaching the point while a train on a crossing route is over it, or
// is due over it first, stands short of it until that train has cleared. Only zoomed in (the
// renderer's `CONFLICT_ZOOM`), where two trains at a crossing read apart; further out they are drawn
// through each other. The timetable already carries the average wait (track-and-service.md 6.2);
// this only draws one.
//
// T-2233: at a station with passing tracks a train that stops is drawn on the outer track and one that
// does not on the running track beside it, so there neither holds the other: the one place order on a
// track may change. A faster line stopping there stops on the running track beside it (T-2248).

import { PASS_OUTER, PASS_STRIDE, carsOf, profileS, type NetworkBuffers } from "./network";
import { perfOff } from "../perfFlags";

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
/** A train waiting at a conflict stands with its front this far short of it, m (sim `HOLD_MARGIN_M`). */
export const CONFLICT_BEFORE_M = 20;
/** A train holds a conflict until its back is this far past it, m. */
const CONFLICT_AFTER_M = 10;
/** The longest one wait at a conflict, s: past it the train is let over (drawn through whatever is
 * there), so a jam drawn on screen cannot grow without end. */
export const CONFLICT_WAIT_MAX = 240;
/** How far past the hold point a train may stand and still be short of it, m: a held train is
 * drawn there, and one further on last frame is over the conflict. */
const HOLD_TOL_M = 1;
/** path entries of one move in one key: in * MOVE_KEY + out */
const MOVE_KEY = 2 ** 26;
/** A train stopping at a passing station is on its outer track, out of the way, once its back is
 * this share of the outer track's lead short of the platform (drawn 90% of the way out), until its
 * front is past the lead. */
const PASS_IN = 0.2;

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
  /** zoomed in far enough for trains to wait at conflicts (renderer `CONFLICT_ZOOM`) */
  close?: boolean;
}

export class TrainGuard {
  /** per trip of the buffer: game seconds it is drawn behind its timetable */
  lag = new Float32Array(0);
  /** per trip: its offset along its run as last drawn, NaN when not guarded last frame */
  private prevS = new Float64Array(0);
  /** per trip: the conflict it is waiting at (-1 none) and since when (game s - epoch) */
  private waitRes = new Int32Array(0);
  private waitFrom = new Float64Array(0);
  /** conflicts were guarded last frame */
  private waiting = false;
  private trips: Float32Array | null = null;
  private epoch = NaN;
  private net: NetworkBuffers | null = null;
  /** lines worth guarding: sharing a track (an edge in one direction) with another line; with
   * conflicts, also those passing a flat crossing or a crossing junction move */
  private guardedTrack = new Uint8Array(0);
  private guardedAll = new Uint8Array(0);
  private guarded = new Uint8Array(0);
  /** per edge: its flat crossings, (crossing, metres... as a share of the edge from node a) pairs */
  private crossOn = new Map<number, number[]>();
  /** per move through a junction (in entry * MOVE_KEY + out entry): its index */
  private moveIdx = new Map<number, number>();
  /** per move: its conflict (flat crossings first, then junction nodes) */
  private moveRes: number[] = [];
  /** crossing move pairs, both ways: a * moves + b */
  private movePairs = new Set<number>();
  private nCross = 0;
  /** per profile and conflict offset: the trip times its front reaches the hold point and its back
   * clears (`tauKey`), for the phase tables it was worked out on */
  private tauCache = new Map<number, number>();
  private tauPhases: Float32Array | null = null;
  /** the game time of the last update: the lags hold for the frame drawn at it */
  lastTime = NaN;
  /** the next trip of each line, found by its departure: line * 1e7 + round(dep * 100) */
  private nextIndex = new Map<number, number>();
  /** how many trips were guarded the last frame (checks) */
  guardedCount = 0;
  heldCount = 0;
  /** the last frame: trips standing at a conflict, and the longest of their waits so far, s (checks) */
  conflictHeld = 0;
  conflictWaitMax = 0;
  /** since the last `resetCounts`: the longest wait at a conflict, s, waits ended, and trains let
   * over at CONFLICT_WAIT_MAX (checks) */
  conflictLongest = 0;
  conflictWaits = 0;
  conflictWaitSum = 0;
  conflictLetThrough = 0;
  /** per line: its run 0 samples' box (x0, y0, x1, y1, local units), for guarded lines */
  private boxes: Float32Array | null = null;
  /** per line, this frame: 1 its box is in or near the view, 0 not, 2 not looked at yet */
  private seen = new Uint8Array(0);
  /** checks: when an array, each train let over at CONFLICT_WAIT_MAX is noted in it */
  log: unknown[] | null = null;
  /** T-2233, per line: the passing stations where it stands on the outer track (offset along run 0,
   * half the platform, node, the leads before and after, as `NetworkBuffers.passing`), and those it is
   * on the running track at, running through or stopping beside a slower line (T-2248) */
  private passStops = new Map<number, number[]>();
  private passThrough = new Map<number, Set<number>>();
  /** since the last `resetCounts`: train pairs found beside each other at a passing station, summed
   * over frames (checks) */
  passedBy = 0;
  /** trains also wait for each other at flat crossings and crossing junction moves (T-1966) */
  crossings = !perfOff("crossWait");

  resetCounts() {
    this.conflictLongest = 0;
    this.conflictWaits = 0;
    this.conflictWaitSum = 0;
    this.conflictLetThrough = 0;
    this.passedBy = 0;
  }

  setNetwork(n: NetworkBuffers) {
    if (n === this.net) return;
    this.net = n;
    this.passStops.clear();
    this.passThrough.clear();
    const P = n.passing;
    if (P) for (let k = 0; k + PASS_STRIDE <= P.length; k += PASS_STRIDE) {
      const line = P[k];
      if (P[k + 4] === PASS_OUTER) {
        const at = [P[k + 1], P[k + 2], P[k + 3], P[k + 5], P[k + 6]];
        const v = this.passStops.get(line);
        if (v) v.push(...at);
        else this.passStops.set(line, at);
      } else {
        const v = this.passThrough.get(line);
        if (v) v.add(P[k + 3]);
        else this.passThrough.set(line, new Set([P[k + 3]]));
      }
    }
    this.crossOn.clear();
    this.moveIdx.clear();
    this.moveRes = [];
    this.movePairs.clear();
    this.tauCache.clear();
    this.boxes = null;
    const T = n.pathTable, E = n.pathEntry;
    const lines = (n.lineTable.length / 4) | 0;
    this.guardedTrack = new Uint8Array(lines);
    this.guardedAll = new Uint8Array(lines);
    this.guarded = this.guardedTrack;
    if (!T || !E) return;
    this.readConflicts(n.conflicts);
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
      for (let k = f; k < f + c; k++) {
        if (shared.has(E[k]) || shared.has(E[k] ^ 1)) { this.guardedTrack[l] = this.guardedAll[l] = 1; break; }
        if (this.crossOn.has(E[k] >> 1) || (k + 1 < f + c && (this.moveIdx.has(E[k] * MOVE_KEY + E[k + 1]) || this.moveIdx.has((E[k + 1] ^ 1) * MOVE_KEY + (E[k] ^ 1))))) this.guardedAll[l] = 1;
      }
    }
    const box = (this.boxes = new Float32Array(lines * 4));
    for (let l = 0; l < lines; l++) {
      if (!this.guardedAll[l]) continue;
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

  /** `ClockState.conflicts`: the flat crossings by edge, and the junction moves with their pairs. */
  private readConflicts(c: Float64Array | undefined) {
    this.nCross = 0;
    if (!c || !c.length) return;
    let k = 0;
    const nx = c[k++];
    for (let x = 0; x < nx; x++, k += 4) {
      for (const [e, at] of [[c[k], c[k + 1]], [c[k + 2], c[k + 3]]]) {
        const v = this.crossOn.get(e);
        if (v) v.push(x, at);
        else this.crossOn.set(e, [x, at]);
      }
    }
    this.nCross = nx;
    const np = c[k++] ?? 0;
    const nodeRes = new Map<number, number>();
    const pairs: number[] = [];
    const move = (node: number, p: number, q: number) => {
      const key = p * MOVE_KEY + q;
      let m = this.moveIdx.get(key);
      if (m === undefined) {
        m = this.moveRes.length;
        this.moveIdx.set(key, m);
        let r = nodeRes.get(node);
        if (r === undefined) nodeRes.set(node, (r = nx + nodeRes.size));
        this.moveRes.push(r);
      }
      return m;
    };
    for (let j = 0; j < np; j++, k += 5) pairs.push(move(c[k], c[k + 1], c[k + 2]), move(c[k], c[k + 3], c[k + 4]));
    const M = this.moveRes.length;
    for (let j = 0; j < pairs.length; j += 2) {
      this.movePairs.add(pairs[j] * M + pairs[j + 1]);
      this.movePairs.add(pairs[j + 1] * M + pairs[j]);
    }
  }

  /** The trip buffer changed: each trip keeps its lag and last place by its profile and departure. */
  private remap(f: GuardFrame) {
    const n = f.trips.length / 2;
    const lag = new Float32Array(n), prevS = new Float64Array(n).fill(NaN);
    const waitRes = new Int32Array(n).fill(-1), waitFrom = new Float64Array(n);
    const old = this.trips;
    if (old && this.lag.length) {
      const was = new Map<string, number>();
      for (let i = 0; i < this.lag.length; i++) if (this.lag[i] > 0 || !Number.isNaN(this.prevS[i])) was.set(`${old[i * 2]}|${Math.round((old[i * 2 + 1] + this.epoch) * 100)}`, i);
      if (was.size) for (let i = 0; i < n; i++) {
        const j = was.get(`${f.trips[i * 2]}|${Math.round((f.trips[i * 2 + 1] + f.epoch) * 100)}`);
        if (j !== undefined) {
          lag[i] = this.lag[j];
          prevS[i] = this.prevS[j];
          waitRes[i] = this.waitRes[j];
          waitFrom[i] = this.waitFrom[j] + this.epoch - f.epoch;
        }
      }
    }
    this.lag = lag;
    this.prevS = prevS;
    this.waitRes = waitRes;
    this.waitFrom = waitFrom;
    this.trips = f.trips;
    this.epoch = f.epoch;
    this.nextIndex.clear();
    for (let i = 0; i < n; i++) {
      const line = Math.floor(f.trips[i * 2] / 6);
      if (this.guardedAll[line]) this.nextIndex.set(line * 1e7 + Math.round(f.trips[i * 2 + 1] * 100), i);
    }
  }

  /**
   * Work out every guarded trip's lag for the frame; true when `lag` changed. Candidates are the
   * trips of guarded lines drawn in or near the view.
   */
  update(f: GuardFrame): boolean {
    const n = this.net;
    if (!n || !n.pathTable || !n.pathEntry || !n.pathFrom || !n.pathLen) return false;
    const conflicts = this.crossings && f.close !== false;
    this.guarded = conflicts ? this.guardedAll : this.guardedTrack;
    if (f.trips !== this.trips || f.epoch !== this.epoch) this.remap(f);
    if (!conflicts && this.waiting) { this.waitRes.fill(-1); this.waiting = false; }
    this.waiting = conflicts;
    const dt = f.time - this.lastTime;
    this.lastTime = f.time;
    const lag = this.lag, prevS = this.prevS;
    let changed = false;
    if (!(dt >= 0 && dt <= JUMP_S)) {
      // a jump of the clock: everyone back on their timetable
      for (let i = 0; i < lag.length; i++) if (lag[i]) { lag[i] = 0; changed = true; }
      prevS.fill(NaN);
      this.waitRes.fill(-1);
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
      half.push(Math.min(Math.max(1, carsOf(n.lineCars?.[line] ?? 8)) * 20, meta[p * 4 + 3]) / 2);
      tauOf.push(tau);
      if (l0 !== li) { lag[i] = li; changed = true; }
    }
    this.guardedCount = cand.length;
    this.heldCount = 0;
    this.conflictHeld = 0;
    this.conflictWaitMax = 0;
    if (cand.length < 2) {
      if (cand.length) prevS[cand[0]] = cs[0];
      return changed;
    }
    if (conflicts && (this.crossOn.size || this.moveRes.length)) {
      if (this.holdAtConflicts(f, cand, cs, half, tauOf)) changed = true;
    }
    const gapM = GUARD_GAP_M;
    // Each candidate's place on every track (an edge in its direction of travel) from its back
    // (where it was or is, the further back) to just past its front: the coordinate on a track is
    // from the edge's first node in the direction of travel, so a candidate's coordinate there is
    // its offset along its run plus `off`.
    const regs = new Map<number, number[]>();
    const sNew = cs.slice();
    for (let c = 0; c < cand.length; c++) {
      const i = cand[c], p = f.trips[i * 2], line = Math.floor(p / 6), run = Math.floor(p / 3) % 2;
      // on a line here for its conflicts alone (it shares no track), a train standing out its turn
      // at the end of its run holds no one: the next one, late from a wait, would queue behind it
      // there and pass its lateness on to the one after, and so on (lags to the 20-minute cap)
      if (!this.guardedTrack[line] && tauOf[c] >= meta[p * 4 + 2]) continue;
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
        const key = run === 0 ? E[k] : E[k] ^ 1;
        // where the entry starts along this run
        const start = run === 0 ? PF[k] : L - PF[k] - len;
        push(regs, key, c, -start, len);
      }
    }
    // order each track's candidates by where their fronts were (no overtaking on screen), then hold
    // each behind the backs of those ahead that are on it and have not cleared it
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
    // where a train on it has its back, clamped to the track's start; Infinity when it holds no one
    const holds = (c: number, off: number, zone: number) => {
      const back = sNew[c] + off - half[c];
      return back < zone && back + 2 * half[c] > 0 ? Math.max(back, 0) : Infinity;
    };
    // T-2233: the passing station each candidate stands at on its outer track (-1 none), and a pair
    // where one is there and the other runs through it, so is beside it
    const outer = this.passStops.size ? this.outerAt(f, cand, cs, half) : null;
    const beside = (a: number, b: number) => {
      const p = f.trips[cand[a] * 2], q = f.trips[cand[b] * 2];
      return (outer![a] >= 0 && !!this.passThrough.get(Math.floor(q / 6))?.has(outer![a])) || (outer![b] >= 0 && !!this.passThrough.get(Math.floor(p / 6))?.has(outer![b]));
    };
    const besideIn = outer ? lists.map((v) => {
      for (let j = 0; j < v.length; j += REG) if (outer[v[j]] >= 0) return true;
      return false;
    }) : null;
    for (let pass = 0; pass < 6; pass++) {
      let moved = false;
      lists.forEach((v, li) => {
        let m = Infinity;
        for (let j = 0; j < v.length; j += REG) {
          const c = v[j], off = v[j + 1], zone = v[j + 2];
          let mc = m;
          if (besideIn?.[li]) {
            // each train ahead holds it unless the two are beside each other
            mc = Infinity;
            for (let q = 0; q < j; q += REG) {
              if (beside(c, v[q])) { if (pass === 0) this.passedBy++; continue; }
              mc = Math.min(mc, holds(v[q], v[q + 1], v[q + 2]));
            }
          }
          if (mc < Infinity) {
            const cap = mc - gapM - off - half[c];
            if (sNew[c] > cap + 1e-6) {
              sNew[c] = cap;
              moved = true;
            }
          }
          m = Math.min(m, holds(c, off, zone));
        }
      });
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

  /**
   * T-1966: the waits at conflicts. Each candidate whose body is over a conflict, or whose front
   * reached its hold point this frame, has its time there from its timetable and lag: [A, B], from
   * its front at the hold point to its back clear. One already over it holds it until B. The rest
   * go in the order of their A: one goes at A when no train on a crossing route holds the point
   * then, else at the end of that train's B, standing at the hold point meanwhile (its lag grows),
   * and a later train on a crossing route waits behind its turn too, so no one waits for ever.
   * Updates `cs` and `tauOf` of those it holds; true when a lag changed.
   */
  private holdAtConflicts(f: GuardFrame, cand: number[], cs: number[], half: number[], tauOf: number[]): boolean {
    const n = this.net!, T = n.pathTable!, E = n.pathEntry!, PF = n.pathFrom!, PL = n.pathLen!, meta = n.meta;
    const lag = this.lag, prevS = this.prevS, waitRes = this.waitRes, waitFrom = this.waitFrom;
    if (this.tauPhases !== n.phases) { this.tauCache.clear(); this.tauPhases = n.phases; }
    const uTime = f.time - f.epoch;
    // per conflict: candidate, its route there (the edge at a crossing, the move at a junction), its
    // offset along its run
    const regs = new Map<number, number[]>();
    const reg = (res: number, c: number, route: number, sx: number) => {
      const v = regs.get(res);
      if (v) v.push(c, route, sx);
      else regs.set(res, [c, route, sx]);
    };
    for (let c = 0; c < cand.length; c++) {
      const i = cand[c], p = f.trips[i * 2], line = Math.floor(p / 6), run = Math.floor(p / 3) % 2;
      const L = meta[p * 4 + 3], first = T[line * 2], count = T[line * 2 + 1];
      const was = prevS[i], s = cs[c], h = half[c];
      // a conflict at sx concerns it from its front at the hold point (sx - h - before) to its back
      // clear (sx + h + after), since where it was last frame; and one whose hold point it reaches
      // before its back clears one it is at (so it does not go over that one only to stop on it)
      const a = Math.min(s, Number.isNaN(was) ? s : was) - h - CONFLICT_AFTER_M;
      const b = s + 3 * h + 2 * CONFLICT_BEFORE_M + CONFLICT_AFTER_M + 1;
      const d0 = run === 0 ? a : L - b, d1 = run === 0 ? b : L - a;
      let k = upper(PF, first, first + count, d0) - 1;
      if (k < first) k = first;
      for (; k < first + count && PF[k] <= d1; k++) {
        const len = PL[k];
        if (PF[k] + len < d0) continue;
        const ent = E[k];
        const start = run === 0 ? PF[k] : L - PF[k] - len;
        const xs = this.crossOn.get(ent >> 1);
        if (xs) for (let q = 0; q < xs.length; q += 2) {
          const fromA = xs[q + 1] * len, along = ((run === 0 ? ent : ent ^ 1) & 1) === 0 ? fromA : len - fromA;
          const sx = start + along;
          if (sx > a && sx <= b) reg(xs[q], c, ent >> 1, sx);
        }
        // the node at the end of this entry along run 0
        if (k + 1 < first + count) {
          const sx = run === 0 ? PF[k + 1] : L - PF[k + 1];
          if (sx > a && sx <= b) {
            const m = this.moveIdx.get(run === 0 ? E[k] * MOVE_KEY + E[k + 1] : (E[k + 1] ^ 1) * MOVE_KEY + (E[k] ^ 1));
            if (m !== undefined) reg(this.moveRes[m], c, m, sx);
          }
        }
      }
    }
    // candidate `c`'s time over the conflict at `sx` along its run, game s - epoch: its front at the
    // hold point to its back clear, or to the end of its stand at the run's end when still over it
    // then (it is drawn standing there until `ends`)
    const span = (c: number, sx: number): [number, number] => {
      const i = cand[c], p = f.trips[i * 2], [tIn, tOut] = this.tauAt(n, p, sx, half[c]);
      const base = f.trips[i * 2 + 1] + lag[i];
      return [base + tIn, f.ends && tOut >= meta[p * 4 + 2] - 1e-3 ? Math.max(base + tOut, f.ends[i]) : base + tOut];
    };
    let changed = false;
    const M = this.moveRes.length;
    const grants: number[] = []; // candidate, route, A, B
    const asks: number[] = []; // index into v, A
    // per candidate this frame: the first conflict it has to wait at, and when it reaches it
    const mustWait = new Map<number, [number, number]>();
    // candidates already over a conflict: they take their turn first at the next, so none is made
    // to stop over one for a train that reached the next after it
    const over = new Set<number>();
    // over the conflict at `sx` already (or past its hold point) last frame, or from the start of its
    // run (its train stood there before it, the last trip's end). One first looked at this frame
    // (come into view, zoomed in to, after a clock jump) takes its turn by its timetable like the
    // rest, so two found over one point are parted.
    const overIt = (c: number, sx: number) => prevS[cand[c]] > sx - half[c] - CONFLICT_BEFORE_M + 2 * HOLD_TOL_M || this.tauAt(n, f.trips[cand[c] * 2], sx, half[c])[0] === 0;
    for (const v of regs.values()) for (let j = 0; j < v.length; j += 3) if (overIt(v[j], v[j + 2])) over.add(v[j]);
    // again while a pass held someone: one held at a conflict ahead stands longer over one behind
    // it, and the trains at that one see it so
    for (let pass = 0, moved = true; moved && pass < 3; pass++) {
      moved = false;
      this.conflictHeld = 0;
      this.conflictWaitMax = 0;
      for (const [res, v] of regs) {
        if (v.length < 6) continue;
        const crossing = res < this.nCross;
        const conflict = (x: number, y: number) => (crossing ? x !== y : this.movePairs.has(x * M + y));
        let any = false;
        for (let j = 3; j < v.length && !any; j += 3) for (let q = 0; q < j && !any; q += 3) any = conflict(v[j + 1], v[q + 1]);
        if (!any) continue;
        grants.length = 0;
        asks.length = 0;
        for (let j = 0; j < v.length; j += 3) {
          const c = v[j], sx = v[j + 2], i = cand[c];
          const [A, B] = span(c, sx);
          if (overIt(c, sx)) {
            // it keeps it until its back is clear
            grants.push(c, v[j + 1], -Infinity, B);
            if (waitRes[i] === res) waitRes[i] = -1;
          } else asks.push(j, A);
        }
        if (!asks.length) continue;
        const order = Array.from({ length: asks.length / 2 }, (_, q) => q);
        const first = (q: number) => (over.has(v[asks[q * 2]]) ? 0 : 1);
        order.sort((x, y) => first(x) - first(y) || asks[x * 2 + 1] - asks[y * 2 + 1] || asks[x * 2] - asks[y * 2]);
        for (const q of order) {
          const j = asks[q * 2], c = v[j], route = v[j + 1], sx = v[j + 2], i = cand[c], p = f.trips[i * 2];
          const [A, B] = span(c, sx), occ = B - A;
          const elsewhere = mustWait.get(c);
          if (elsewhere && elsewhere[0] !== res && elsewhere[1] < B) {
            // it has to wait at another conflict before its back clears this one: it stands short of
            // this one too (so it does not block it while it waits) and takes no turn here; one
            // found long over it (come into view there) is left where it is
            if (A < uTime && uTime - A <= CONFLICT_WAIT_MAX && lag[i] <= 2 * CONFLICT_WAIT_MAX) {
              lag[i] = Math.min(LAG_MAX, lag[i] + uTime - A);
              tauOf[c] = uTime - f.trips[i * 2 + 1] - lag[i];
              cs[c] = profileS(n, p, tauOf[c]);
              changed = moved = true;
            }
            continue;
          }
          // it goes once every train on a crossing route ahead of it in the order has cleared (none
          // slips in before one waiting, even where it would clear in time)
          let go = A;
          for (let g = 0; g < grants.length; g += 4) {
            if (grants[g + 3] > go && grants[g] !== c && conflict(route, grants[g + 1])) go = grants[g + 3];
          }
          if (go <= A + 1e-6) {
            grants.push(c, route, A, A + occ);
            if (waitRes[i] === res) this.endWait(i, A);
            if (elsewhere?.[0] === res) mustWait.delete(c);
            continue;
          }
          if (!elsewhere || A < elsewhere[1]) {
            mustWait.set(c, [res, A]);
            if (elsewhere?.[0] !== res) moved = true;
          }
          // a wait runs from when it first had to stand, whichever conflicts it stands for since
          if (waitRes[i] < 0) waitFrom[i] = A;
          waitRes[i] = res;
          // (a lag past twice the cap: waits chained over several conflicts, let over too)
          if (go - waitFrom[i] > CONFLICT_WAIT_MAX || lag[i] > 2 * CONFLICT_WAIT_MAX) {
            // too long: let over now, drawn through what is there (waitFrom -Infinity: let over, so it
            // keeps going until it is over the point)
            grants.push(c, route, A, A + occ);
            if (waitFrom[i] !== -Infinity) {
              this.conflictLetThrough++;
              const w = A - waitFrom[i];
              if (w > this.conflictLongest) this.conflictLongest = w;
              if (this.log) {
                const by: number[] = [];
                for (let g = 0; g < grants.length - 4; g += 4) if (grants[g + 3] > A && conflict(route, grants[g + 1])) by.push(f.trips[cand[grants[g]] * 2], +(grants[g + 3] - uTime).toFixed(0), +(cs[grants[g]]).toFixed(0));
                this.log.push({ res, p, s: +cs[c].toFixed(0), sx: +sx.toFixed(0), wait: +(go - waitFrom[i]).toFixed(0), by });
              }
              waitFrom[i] = -Infinity;
            }
            if (mustWait.get(c)?.[0] === res) mustWait.delete(c);
            continue;
          }
          // it goes at `go`: late by go - A, or standing at the hold point until then
          const extra = Math.min(go, uTime) - A;
          if (extra > 0) {
            const l = Math.min(LAG_MAX, lag[i] + extra);
            if (l - lag[i] > 1e-3) moved = true;
            if (l !== lag[i]) { lag[i] = l; changed = true; }
            tauOf[c] = uTime - f.trips[i * 2 + 1] - lag[i];
            cs[c] = profileS(n, p, tauOf[c]);
          }
          grants.push(c, route, go, go + occ);
          if (go <= uTime) {
            this.endWait(i, go);
            if (mustWait.get(c)?.[0] === res) mustWait.delete(c);
          } else if (A <= uTime) {
            this.conflictHeld++;
            this.conflictWaitMax = Math.max(this.conflictWaitMax, uTime - waitFrom[i]);
          }
        }
      }
    }
    // a wait no conflict holds any more (one let over keeps going until it is over the point)
    for (let c = 0; c < cand.length; c++) {
      const i = cand[c];
      if (waitRes[i] >= 0 && waitFrom[i] !== -Infinity && !mustWait.has(c)) waitRes[i] = -1;
    }
    return changed;
  }

  /** Per candidate, the passing station whose outer track it is on (`PASS_IN`), -1 none (T-2233). */
  private outerAt(f: GuardFrame, cand: number[], cs: number[], half: number[]): Int32Array {
    const meta = this.net!.meta, out = new Int32Array(cand.length).fill(-1);
    for (let c = 0; c < cand.length; c++) {
      const p = f.trips[cand[c] * 2], st = this.passStops.get(Math.floor(p / 6));
      if (!st) continue;
      const run0 = Math.floor(p / 3) % 2 === 0, L = meta[p * 4 + 3], s = cs[c], h = half[c];
      for (let k = 0; k < st.length; k += 5) {
        // the outer track's leads in its travel direction: in before the platform, out after it
        const at = run0 ? st[k] : L - st[k], ph = st[k + 1], [lin, lout] = run0 ? [st[k + 3], st[k + 4]] : [st[k + 4], st[k + 3]];
        if (s - h >= at - ph - PASS_IN * lin && s + h <= at + ph + lout) { out[c] = st[k + 2]; break; }
      }
    }
    return out;
  }

  /** Trip `i`'s wait ends: it goes over at `at` (game s - epoch). */
  private endWait(i: number, at: number) {
    const w = at - this.waitFrom[i];
    if (this.waitRes[i] >= 0 && w > 0 && w < Infinity) {
      this.conflictWaits++;
      this.conflictWaitSum += w;
      if (w > this.conflictLongest) this.conflictLongest = w;
    }
    this.waitRes[i] = -1;
  }

  /** Trip times when profile `p`'s front reaches the hold point of a conflict `sx` along its run,
   * and when its back clears it. */
  private tauAt(n: NetworkBuffers, p: number, sx: number, h: number): readonly [number, number] {
    const key = p * 2 ** 27 + Math.round(sx * 2);
    const tIn = this.tauCache.get(key * 2), tOut = this.tauCache.get(key * 2 + 1);
    if (tIn !== undefined && tOut !== undefined) return [tIn, tOut];
    const dur = n.meta[p * 4 + 2];
    const sIn = sx - h - CONFLICT_BEFORE_M, sOut = sx + h + CONFLICT_AFTER_M;
    // over it from the start of the run: from the trip's start, not when it leaves its first stop.
    // A timetabled hold for this conflict stands at the hold point (sim `HOLD_MARGIN_M`, to float32's
    // precision): HOLD_TOL_M keeps it out of the time over it
    const a = sIn <= profileS(n, p, 0) + HOLD_TOL_M ? 0 : timeAt(n, p, sIn + HOLD_TOL_M, dur);
    // a back still over it where the run ends stands there until the trip ends
    const b = sOut >= profileS(n, p, dur) - HOLD_TOL_M ? dur : timeAt(n, p, sOut, dur);
    this.tauCache.set(key * 2, a);
    this.tauCache.set(key * 2 + 1, b);
    return [a, b];
  }
}

/** Numbers per registration on a track: candidate, offset to its coordinate there, the track's length. */
const REG = 3;

function push(regs: Map<number, number[]>, key: number, c: number, off: number, zone: number) {
  const v = regs.get(key);
  if (v) {
    // a run over the same edge twice (out and back on one track) counts its nearer pass
    for (let j = 0; j < v.length; j += REG) if (v[j] === c) return;
    v.push(c, off, zone);
  } else regs.set(key, [c, off, zone]);
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
