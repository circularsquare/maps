// Each station's platform centre line in world units (T-1859), for the complexes (game/complexes.ts):
// the line its box is drawn along (map/stationGeometry.ts `platformShape`, map/network.ts), taken over
// the track model's own polylines, so two stations whose boxes touch are one complex whatever the
// curve, and a dead end's platform lies along its track from the node as drawn.

import type { TrackApi } from "../wasm/track/sim";
import { PLATFORM_POINTS, platformPoints, platformShape } from "../map/stationGeometry";

/** Every second point of the drawn line's: 7 points, 6 stretches. */
const STEP = 2;

interface Kept {
  platform: number;
  x: number;
  y: number;
  k: number;
  ways: Float64Array[];
  line: Float32Array;
}

export class PlatformLines {
  /** per edge, its polyline (x, y, z triples) from its a end and from its b end */
  private edgePts = new Map<number, [Float64Array, Float64Array]>();
  private ends = new Map<number, [number, number]>();
  /** per node, its edges: most lines first, then by id, as map/network.ts orders a platform's ways */
  private at = new Map<number, number[]>();
  private onEdge = new Map<number, number>();
  private kept = new Map<number, Kept>();

  forgetEdge(id: number) {
    this.edgePts.delete(id);
  }

  clear() {
    this.edgePts.clear();
    this.kept.clear();
  }

  /** How many lines run over each edge, from the latest line rows. */
  setLines(lines: readonly { ok: boolean; path: number[]; planPath?: number[] }[]) {
    const on = new Map<number, number>();
    for (const l of lines) {
      if (!l.ok) continue;
      // T-3474: a line's whole plan, which the map draws
      for (const e of new Set((l.planPath ?? l.path).map((p) => p >> 1))) on.set(e, (on.get(e) ?? 0) + 1);
    }
    this.onEdge = on;
  }

  /** The edges at each node, from the state's edge rows. */
  sync(edges: readonly { id: number; a: number; b: number }[]) {
    const at = new Map<number, number[]>();
    this.ends.clear();
    for (const { id, a, b } of edges) {
      this.ends.set(id, [a, b]);
      for (const n of [a, b]) {
        const v = at.get(n);
        if (v) v.push(id);
        else at.set(n, [id]);
      }
    }
    const on = this.onEdge;
    for (const v of at.values()) if (v.length > 1) v.sort((p, q) => (on.get(q) ?? 0) - (on.get(p) ?? 0) || p - q);
    this.at = at;
    for (const n of this.kept.keys()) if (!at.has(n)) this.kept.delete(n);
  }

  /** Station `node`'s platform line, x, y offsets from (x, y); `k` world units per ground metre there.
   * The same array while nothing it depends on changed; undefined for a node with no track. */
  of(a: TrackApi, node: number, x: number, y: number, platform: number, k: number): Float32Array | undefined {
    const ids = this.at.get(node);
    if (!ids?.length || platform <= 0) return undefined;
    const ways = ids.map((id) => this.way(a, node, id));
    const kept = this.kept.get(node);
    if (kept && kept.platform === platform && kept.x === x && kept.y === y && kept.k === k
      && kept.ways.length === ways.length && kept.ways.every((w, i) => w === ways[i])) return kept.line;
    const line = PlatformLines.lineOf(ways, x, y, platform, k);
    if (line) this.kept.set(node, { platform, x, y, k, ways, line });
    return line;
  }

  /** T-4214: `of` for the model as an edit just left it, before the next state's rows are synced: the edges `changed`
   * since (their ends now, null when gone) in place of those kept. Nothing is kept from it (the edit's patch, clockPatch.ts). */
  peekOf(a: TrackApi, node: number, x: number, y: number, platform: number, k: number, changed: ReadonlyMap<number, readonly [number, number] | null>): Float32Array | undefined {
    if (platform <= 0) return undefined;
    const ids = (this.at.get(node) ?? []).filter((id) => !changed.has(id));
    for (const [id, ends] of changed) if (ends && (ends[0] === node || ends[1] === node)) ids.push(id);
    if (!ids.length) return undefined;
    const on = this.onEdge;
    ids.sort((p, q) => (on.get(q) ?? 0) - (on.get(p) ?? 0) || p - q);
    const ways = ids.map((id) => {
      const ends = changed.get(id);
      if (!ends) return this.way(a, node, id);
      const [f, r] = PlatformLines.both(a, id);
      return ends[0] === node ? f : r;
    });
    return PlatformLines.lineOf(ways, x, y, platform, k);
  }

  /** the platform line along `ways` (the first is the platform's track), as offsets from (x, y) */
  private static lineOf(ways: Float64Array[], x: number, y: number, platform: number, k: number): Float32Array | undefined {
    const first = ways[0];
    if (first.length < 6) return undefined;
    const sh = platformShape(ways, 0, platform / 2 * k);
    const pts = platformPoints(sh);
    const ox = first[0] - x, oy = first[1] - y;
    const line = new Float32Array(2 * Math.ceil(PLATFORM_POINTS / STEP));
    for (let i = 0, o = 0; i < PLATFORM_POINTS; i += STEP, o += 2) {
      line[o] = pts[i * 2] + ox;
      line[o + 1] = pts[i * 2 + 1] + oy;
    }
    return line;
  }

  /** an edge's polyline from its a end and from its b end */
  private static both(a: TrackApi, id: number): [Float64Array, Float64Array] {
    const f = Float64Array.from(a.edge_render(id)), n = f.length / 3, r = new Float64Array(f.length);
    for (let i = 0; i < n; i++) r.set(f.subarray((n - 1 - i) * 3, (n - i) * 3), i * 3);
    return [f, r];
  }

  private way(a: TrackApi, node: number, id: number): Float64Array {
    let p = this.edgePts.get(id);
    if (!p) this.edgePts.set(id, (p = PlatformLines.both(a, id)));
    return this.ends.get(id)?.[0] === node ? p[0] : p[1];
  }
}
