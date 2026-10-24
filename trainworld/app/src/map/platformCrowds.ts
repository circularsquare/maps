// The riders waiting on each platform (T-164, T-219), worked out on the CPU from the boarding
// account's hour for the renderer's crowd fills (map/renderer.ts) and the hover tags.

import { platformRoom, QUEUE_STRIDE, waitingAt, type BoardHour } from "../game/boarding";
import { lineAt, type NetworkBuffers } from "./network";

export class PlatformCrowds {
  /** per station: how full its + and - side are (0-1), whether that side is full (T-219); 4 each */
  readonly crowds: Float32Array;
  /** per station, zoomed out (T-228): its dot's fullest platform side (0-1) and whether a side is
   * full; 2 each */
  readonly dotCrowds: Float32Array;
  /** per station: riders waiting on its + and - side */
  private wait: Float32Array;
  /** per station: riders waiting at it */
  private dotWait: Float32Array;
  /** per platform of the hour: its station's index in the station buffer (-1 none) and its side;
   * per station which sides have platforms (1 the + side, 2 the - side) */
  private placed: { station: Int32Array; side: Int8Array; sides: Uint8Array; board: BoardHour; net: NetworkBuffers } | null = null;
  /** the time and hour the crowds were last worked out for */
  private at = { t: NaN, board: null as BoardHour | null };
  private any = false;

  constructor(stationCount: number) {
    const n = Math.max(1, stationCount);
    this.crowds = new Float32Array(n * 4);
    this.wait = new Float32Array(n * 2);
    this.dotCrowds = new Float32Array(n * 2);
    this.dotWait = new Float32Array(n);
  }

  /** Each platform's station in the station buffer, for the hour last worked out; null before one. */
  get stationOfQueue(): Int32Array | null {
    return this.placed?.station ?? null;
  }

  /** The riders waiting on each station's two sides at `t` s into the trip buffer's hour (T-164):
   * each platform's crowd is what the last train to leave it left plus who has arrived since; drawn
   * as a share of the room to stand on that side (T-219), the whole platform's when only one side
   * has trains. Worked out at most once a game second, only while drawn; `changed` is called with
   * the arrays when they were. False when nobody waits anywhere. */
  update(n: NetworkBuffers, b: BoardHour, t: number, changed: (crowds: Float32Array, dotCrowds: Float32Array) => void): boolean {
    if (!this.placed || this.placed.board !== b || this.placed.net !== n) this.placed = place(n, b);
    if (this.at.board === b && Math.abs(t - this.at.t) < 1) return this.any;
    const c = this.crowds, w = this.wait;
    c.fill(0);
    w.fill(0);
    const { station, side, sides } = this.placed;
    let any = false;
    for (let q = 0; q < b.queueCount; q++) {
      const s = station[q];
      if (s < 0) continue;
      const waiting = waitingAt(b, q, t);
      const k = side[q] > 0 ? 0 : 1;
      w[s * 2 + k] += waiting;
      if (waiting >= 1) any = true;
    }
    for (let s = 0; s < n.stationCount; s++) {
      if (!sides[s]) continue;
      const room = platformRoom(n.stationLength?.[s] ?? 0);
      if (sides[s] === 3) {
        c[s * 4] = Math.min(1, w[s * 2] / room);
        c[s * 4 + 1] = Math.min(1, w[s * 2 + 1] / room);
        c[s * 4 + 2] = c[s * 4] >= 1 ? 1 : 0;
        c[s * 4 + 3] = c[s * 4 + 1] >= 1 ? 1 : 0;
      } else {
        // trains on one side only (a terminal): the whole platform is theirs
        const f = Math.min(1, (w[s * 2] + w[s * 2 + 1]) / (2 * room)), bad = f >= 1 ? 1 : 0;
        c.set([f, f, bad, bad], s * 4);
      }
    }
    // each dot's (T-228): its station's platforms
    const dc = this.dotCrowds, dw = this.dotWait;
    dc.fill(0);
    dw.fill(0);
    for (let s = 0; s < n.stationCount; s++) {
      if (!sides[s]) continue;
      dc[s * 2] = Math.max(c[s * 4], c[s * 4 + 1]);
      if (c[s * 4 + 2] > 0 || c[s * 4 + 3] > 0) dc[s * 2 + 1] = 1; // a full side
      dw[s] = w[s * 2] + w[s * 2 + 1];
    }
    changed(c, dc);
    this.at = { t, board: b };
    this.any = any;
    return any;
  }

  /** The share of the room to stand that the platform side of queue `q` of the trip buffer holds,
   * as last worked out (0-1); null when the queue has no platform on `net`. */
  queueShare(q: number, net: NetworkBuffers): number | null {
    const o = this.placed;
    if (!o || o.net !== net || q < 0 || q >= o.station.length || o.station[q] < 0) return null;
    return this.crowds[o.station[q] * 4 + (o.side[q] > 0 ? 0 : 1)];
  }

  /** Station `i`'s riders waiting at its platforms and its fullest platform side's share of the
   * room there; null for a station that is not there. */
  dot(i: number): { waiting: number; share: number } | null {
    if (i < 0 || i >= this.dotWait.length) return null;
    return { waiting: this.dotWait[i], share: this.dotCrowds[i * 2] };
  }
}

/** Each platform's station in the station buffer, and its side: + where its trains travel along
 * the station's drawn platform axis, - against it; and per station the sides with platforms. */
function place(n: NetworkBuffers, b: BoardHour) {
  const index = new Map<number, number>();
  n.stationIds.forEach((id, i) => index.set(Number(id), i));
  const station = new Int32Array(b.queueCount), side = new Int8Array(b.queueCount), sides = new Uint8Array(Math.max(1, n.stationCount));
  for (let q = 0; q < b.queueCount; q++) {
    const base = q * QUEUE_STRIDE;
    const s = index.get(b.queues[base]) ?? -1;
    station[q] = s;
    if (s < 0) continue;
    const line = b.queues[base + 1], run = b.queues[base + 2], d = b.queues[base + 3];
    const [ax, ay] = lineAt(n, line, d - 5), [bx, by] = lineAt(n, line, d + 5);
    const dot = ((bx - ax) * (n.stationAxis?.[s * 2] ?? 1) + (by - ay) * (n.stationAxis?.[s * 2 + 1] ?? 0)) * (run === 1 ? -1 : 1);
    side[q] = dot >= 0 ? 1 : -1;
    sides[s] |= side[q] > 0 ? 1 : 2;
  }
  return { station, side, sides, board: b, net: n };
}
