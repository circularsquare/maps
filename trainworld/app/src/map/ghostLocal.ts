// T-4210: the track tool's ghost of a plain tangent segment (the run going on from the end just placed, onto open
// ground), laid on the page by the track model's own code (`ghost_tangent`, sim/src/track/wasm.rs), so the ghost is the
// segment the worker will build in the frame the pointer moves, never a guess bent to the pointer. The worker still
// answers each ghost with its checks and price; where water, a ramp or other track change the route, its answer
// differs and the tool falls back to bending that (map/tools.ts `localOk`).

import type { PreviewAnswer } from "../workers/clockClient";
import { lngLatToPack, packToLngLat } from "../game/coords";
import { getWorldFrame } from "../game/cities";
import { toLngLat, toLocal } from "./geo";

type Sim = typeof import("../wasm/track/sim");
let sim: Sim | null = null;
let loading: Promise<void> | null = null;

/** Load the track model's code on the page (once, on the track tool's first use; about 25 ms and 1 MB). */
export function warmGhostLocal() {
  loading ??= import("../wasm/track/sim").then(async (m) => {
    await m.default();
    sim = m;
  }).catch(() => { loading = null; });
}

export const ghostLocalReady = () => !!sim;

/** How far apart on screen, px, a ghost laid here and the worker's answer for the same spot may be and still be one route */
export const LOCAL_PX = 2;

/** Whether polylines `a` and `b` (x, y, z local units) lie within `px` of each other on screen, both ways. */
export function sameLine(a: Float32Array, b: Float32Array, screen: (x: number, y: number) => [number, number], px: number): boolean {
  const pa = toScreen(a, screen), pb = toScreen(b, screen);
  return within(pa, pb, px) && within(pb, pa, px);
}
function toScreen(p: Float32Array, screen: (x: number, y: number) => [number, number]): number[] {
  const o: number[] = [];
  for (let k = 0; k + 2 < p.length; k += 3) o.push(...screen(p[k], p[k + 1]));
  return o;
}
/** every point of `a` within `px` of the polyline `b` (x, y pairs) */
function within(a: number[], b: number[], px: number): boolean {
  for (let i = 0; i < a.length; i += 2) {
    let best = Infinity;
    for (let k = 0; k + 3 < b.length && best > px; k += 2) {
      const x0 = b[k], y0 = b[k + 1], dx = b[k + 2] - x0, dy = b[k + 3] - y0, l2 = dx * dx + dy * dy;
      const t = l2 ? Math.max(0, Math.min(1, ((a[i] - x0) * dx + (a[i + 1] - y0) * dy) / l2)) : 0;
      best = Math.min(best, Math.hypot(x0 + t * dx - a[i], y0 + t * dy - a[i + 1]));
    }
    if (best > px) return false;
  }
  return true;
}

/** What a tangent ghost needs besides its ends: the drawing tool's settings. */
export interface TangentSpec { level: number; design: number; hsr: boolean; single: boolean }

const pack = (lx: number, ly: number) => lngLatToPack(...toLngLat(lx, ly));
const local = (x: number, y: number) => toLocal(...packToLngLat(x, y));

/** The ghost from (sx, sy) leaving along unit `dir` to (ex, ey), local units, as a preview answer with no figures of
 * its own (`ok`, cost and the rest are the caller's), and the way it leaves its end (local unit vector); null when
 * the code is not loaded yet or the route would be refused. */
export function ghostTangent(sx: number, sy: number, dir: [number, number], ex: number, ey: number, s: TangentSpec): { a: PreviewAnswer; end: [number, number] } | null {
  if (!sim) return null;
  const p0 = pack(sx, sy), p1 = pack(ex, ey);
  // the heading in the track model's plane: a step along `dir` a thousandth of the way, carried over
  const e = Math.max(1e-6, 1e-3 * Math.hypot(ex - sx, ey - sy));
  const q = pack(sx + dir[0] * e, sy + dir[1] * e);
  const th = Math.atan2(q.y - p0.y, q.x - p0.x);
  const out = sim.ghost_tangent(p0.x, p0.y, th, p1.x, p1.y, s.level, getWorldFrame().kind === "merc", s.design, 0, s.hsr);
  if (out.length < 10) return null;
  const n = out[4], b = 5 + n, nv = out[b + 3];
  const verts = new Float64Array(nv * 4);
  for (let i = 0; i < nv; i++) {
    const k = b + 5 + i * 4, [x, y] = local(out[k], out[k + 1]);
    verts.set([x, y, out[k + 2], out[k + 3]], i * 4);
  }
  const p = b + 5 + nv * 4, pts = new Float32Array(out.length - p);
  for (let k = p; k + 2 < out.length; k += 3) {
    const [x, y] = local(out[k], out[k + 1]);
    pts.set([x, y, out[k + 2]], k - p);
  }
  const speeds = Float32Array.from(out.subarray(5, b));
  let arc: Float64Array | undefined;
  if (Number.isFinite(out[3])) {
    const [x, y] = local(out[1], out[2]);
    arc = new Float64Array([x, y, out[3], 0]);
  }
  // the end heading, back in local units
  const [lx, ly] = [pts[pts.length - 3], pts[pts.length - 2]];
  const d = Math.max(1e-6, Math.hypot(p1.x - p0.x, p1.y - p0.y) * 1e-3);
  const [fx, fy] = local(out[out.length - 3] + Math.cos(out[0]) * d, out[out.length - 2] + Math.sin(out[0]) * d);
  const ux = fx - lx, uy = fy - ly, l = Math.hypot(ux, uy) || 1;
  const a: PreviewAnswer = { ok: true, cost: NaN, lengthM: out[b + 2], mult: NaN, issues: [], verts, pts, speeds, lanes: new Uint8Array(speeds.length).fill(s.single ? 0 : 255), arc };
  return { a, end: [ux / l, uy / l] };
}
