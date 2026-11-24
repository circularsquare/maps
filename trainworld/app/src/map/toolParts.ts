// The constants, types and small helpers the map tools share (T-2374, moved out of tools.ts).

import { t as tr } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";
import { distance } from "../game/format";
import { type RouteEnd } from "../workers/protocol";
import { toLngLat } from "./geo";

/** the touch control bar's gap from the point it serves, so the finger does not cover it (px) */
export const TOUCH_GAP_PX = 44;
export type LngLatish = { lng: number; lat: number };

// T-1820: the station tool's reach for track, 1.5 times the earlier 10 (it snaps to track only, T-1870)
export const SNAP_EDGE_PX = 15;
// T-1895: the track tool's, smaller, since a click near track more often means open ground than
// a branch; a node (a loose end to carry on from) keeps more
export const TRACK_NODE_PX = 12;
export const TRACK_EDGE_PX = 8;
/** T-3204: the dot for the track a pointer is on: zoomed out the two tracks are within this of the centre line, and nearer than this the pointer picks neither; the dot sits at least this far off */
export const DEAD_PX = 3, DOT_MIN_PX = 6;
/** T-2404: the pointer this near a segment whose state is held (twice the node snap, so the state
 * lands before the pointer reaches what it could snap to), unless this near the start */
export const HELD_NEAR_PX = 2 * TRACK_NODE_PX, HELD_START_PX = 3 * TRACK_NODE_PX;
/** T-2271: the station tool's trips-near bar is empty at 1,000 trips and full at 1M, log scaled
 * (New York's lines have 30k-1M within 1 km); its ticks (style.css `.size-bar`) are 10k and 100k */
export const NEAR_BAR_LO = 1e3, NEAR_BAR_HI = 1e6;
/** open ground this near track (and further than TRACK_EDGE_PX) draws beside it (T-1685) */
export const BESIDE_PX = 40;
/** tracks this much further from the pointer than the nearest still count as overlapping it (T-469) */
export const OVERLAP_PX = 4;

/** `join` (T-3137): with single track picked, the point (local units) on the track of double track there that the
 * pointer picks (`Tools.joinPoint`) */
export type Snap =
  | { kind: "node"; node: number; lx: number; ly: number; join?: [number, number]; dot?: [number, number]; merge?: true }
  | { kind: "edge"; edge: number; lx: number; ly: number; join?: [number, number]; dot?: [number, number] }
  | { kind: "free"; lx: number; ly: number };

export type Spot = Exclude<Snap, { kind: "free" }>;

/** where a station would go for the candidates `key` (null: no ghost), and whether a click there
 * is taken; `held`: the ghost as it was, until the worker answers */
export interface Verdict { key: string; spot: Spot | null; ok: boolean; why: string; held?: boolean }

/** this many candidates under the pointer, not yet tried, and the ghost waits for the worker (T-695) */
export const HOLD_AT = 7;
/** the worker's answer for a track tried this close to the pointer's spot stands for it (T-695) */
export const KNOWN_PX = 16;
export const LAYOUT_ISSUES = new Set(["PlatformTooLong", "PlatformStation", "PlatformJunction", "PlatformJoint", "PlatformCrossing", "StationPorts", "StationJunction"]);
/** track drawn beside track, centre to centre (sim `BESIDE_SPACING`, T-1895) */
export const BESIDE_SPACING_M = 12;
/** each of double track's two tracks from its centre line once drawn apart, m (renderer.ts `LANE_M`, sim `LANE_HALF_M`) */
export const LANE_HALF_M = 2;

export const lengthOf = (pts: ArrayLike<number>) => {
  let l = 0;
  for (let k = 0; k + 5 < pts.length; k += 3) l += Math.hypot(pts[k + 3] - pts[k], pts[k + 4] - pts[k + 1]);
  return l;
};

export interface Placed {
  end: RouteEnd;
  /** where it is in degrees: local units go stale when the draw anchor moves (T-1020) */
  lng: number;
  lat: number;
  /** the real network's point the snap starts from here (T-727d), or null */
  ring: [number, number] | null;
  /** the piece of track the start is beside (T-1685), or null */
  beside?: number | null;
  /** T-3137: the track of double track the start joins (`Snap.join`) */
  join?: [number, number];
  /** T-3260: the way track leaves here (local units, unit length), for Ctrl's straight line: the end's heading
   * of the segment placed last, else the first leg of a ghost from here; none from a start on open ground */
  dir?: [number, number];
}

/** The longest piece of free-drawn track one click lays, straight-line km between its ends (T-1020:
 * a piece from London to Shanghai took the clock worker 3-11 s a preview, and every pointer move
 * asked for one). */
export const MAX_PIECE_KM = 500;
/** a click or ghost waiting this long on the worker shows the spinner in the tag, ms (T-1039) */
export const WAIT_MS = 300;
export const tooFarText = () => tr(mobileDevice() ? "tools.tooFarTouch" : "tools.tooFar", { max: distance(MAX_PIECE_KM * 1000) });

/** An x, y, z polyline less the shares `fromA` and `fromB` of its length at its two ends. */
export function trimPolyline(pts: Float32Array, fromA: number, fromB: number): Float32Array {
  const n = pts.length / 3, cum = [0];
  for (let i = 1; i < n; i++) cum.push(cum[i - 1] + Math.hypot(pts[i * 3] - pts[i * 3 - 3], pts[i * 3 + 1] - pts[i * 3 - 2]));
  const total = cum[n - 1], s0 = fromA * total, s1 = (1 - fromB) * total;
  const at = (s: number) => {
    let i = 1;
    while (i < n - 1 && cum[i] < s) i++;
    const t = cum[i] > cum[i - 1] ? (s - cum[i - 1]) / (cum[i] - cum[i - 1]) : 0;
    return [0, 1, 2].map((k) => pts[(i - 1) * 3 + k] + (pts[i * 3 + k] - pts[(i - 1) * 3 + k]) * t);
  };
  const out = [...at(s0)];
  for (let i = 0; i < n; i++) if (cum[i] > s0 && cum[i] < s1) out.push(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
  out.push(...at(s1));
  return Float32Array.from(out);
}

export function placedAt(end: RouteEnd, lx: number, ly: number, ring: [number, number] | null): Placed {
  const [lng, lat] = toLngLat(lx, ly);
  return { end, lng, lat, ring };
}


/** T-4143: the stand-in ghost before a run's first answer, x, y, z points from the start (sx, sy) to the pointer (ex, ey):
 * straight, or from a start with a heading (`dir`, unit length) the circular arc that leaves along it and ends there
 * (straight when the pointer is behind the start, where no such arc looks like track); heights from z0 to z1, m. */
export function standInPts(sx: number, sy: number, ex: number, ey: number, dir: [number, number] | undefined, z0: number, z1: number): Float32Array {
  const dx = ex - sx, dy = ey - sy;
  let n = 2, at = (t: number): [number, number] => [sx + t * dx, sy + t * dy];
  if (dir) {
    const [ux, uy] = dir, ang = Math.atan2(ux * dy - uy * dx, ux * dx + uy * dy);
    if (Math.abs(ang) > 1e-3 && Math.abs(ang) < Math.PI / 2) {
      // a chord at `ang` to the heading turns the arc through twice that, on a circle to that side
      const r = Math.hypot(dx, dy) / (2 * Math.sin(Math.abs(ang))), sg = Math.sign(ang);
      const cx = sx - uy * r * sg, cy = sy + ux * r * sg, ox = sx - cx, oy = sy - cy;
      n = 17;
      at = (t) => {
        const c = Math.cos(2 * ang * t), s = Math.sin(2 * ang * t);
        return [cx + ox * c - oy * s, cy + ox * s + oy * c];
      };
    }
  }
  const pts = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1), [x, y] = at(t);
    pts.set([x, y, z0 + t * (z1 - z0)], k * 3);
  }
  return pts;
}

/** A refusal's tag content and redness, with `name` above it in the ordinary ink when there is one. */
export function withName(name: string | null, why: string): [string | HTMLDivElement[], boolean] {
  if (!name) return [why, true];
  const row = (s: string, cls = "") => Object.assign(document.createElement("div"), { textContent: s, className: cls });
  return [[row(name), row(why, "bad")], false];
}
