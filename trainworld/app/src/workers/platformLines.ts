// Each station's platform centre line in world units (T-1859), for the complexes (game/complexes.ts):
// the line its box is drawn along (map/stationGeometry.ts `platformShape`, map/network.ts), taken over
// the track model's own polylines, so two stations whose boxes touch are one complex whatever the
// curve, and a dead end's platform lies along its track from the node as drawn.

import type { TrackApi } from "../wasm/track/sim";
import { PLATFORM_POINTS, platformPoints, platformShape } from "../map/stationGeometry";

/** what `setLines` reads of a line row */
type Line = { id: number; ok: boolean; path: number[]; planPath?: number[] };

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
  /** T-4291: the edges each line counts on in `onEdge`, and the edges whose count moved since the last `sync` (the ways at
   * their ends are ordered again there) */
  private lineEdges: Map<number, Set<number>> | null = null;
  private countMoved = new Set<number>();

  /** T-4291: how many edge ends node `n` has, as last synced (an edge from it back to it counts twice) */
  endsAt(n: number): number | undefined {
    return this.at.get(n)?.length;
  }

  forgetEdge(id: number) {
    this.edgePts.delete(id);
  }

  clear() {
    this.edgePts.clear();
    this.kept.clear();
  }

  /** How many lines run over each edge, from the latest line rows. T-4291: `changed`, the lines whose rows are other objects
   * or gone since the last call (`now` gives a line's row), counted again alone; null counts every line. */
  setLines(lines: readonly Line[], changed?: ReadonlySet<number> | null, now?: (id: number) => Line | undefined) {
    const edgesOf = (l: Line | undefined) => (l?.ok ? new Set((l.planPath ?? l.path).map((p) => p >> 1)) : undefined);
    if (changed && now && this.lineEdges) {
      const on = this.onEdge;
      for (const id of changed) {
        const was = this.lineEdges.get(id), is = edgesOf(now(id));
        if (was) for (const e of was) if (!is?.has(e)) { const n = on.get(e)! - 1; if (n) on.set(e, n); else on.delete(e); this.countMoved.add(e); }
        if (is) for (const e of is) if (!was?.has(e)) { on.set(e, (on.get(e) ?? 0) + 1); this.countMoved.add(e); }
        if (is) this.lineEdges.set(id, is); else this.lineEdges.delete(id);
      }
      return;
    }
    const on = new Map<number, number>(), by = new Map<number, Set<number>>();
    for (const l of lines) {
      // T-3474: a line's whole plan, which the map draws
      const is = edgesOf(l);
      if (!is) continue;
      by.set(l.id, is);
      for (const e of is) on.set(e, (on.get(e) ?? 0) + 1);
    }
    this.onEdge = on;
    this.lineEdges = by;
    this.countMoved.clear();
    this.resortAll = true;
  }
  /** T-4291: every node's ways to be ordered again at the next `sync` (the counts were worked out whole) */
  private resortAll = true;

  /** The edges at each node, from the state's edge rows. T-4291: `changed`, the edges whose rows are other objects or gone
   * since the last call (`now` gives an edge's row), laid in alone, and the ways ordered again only at their ends and at the
   * ends of edges whose lines' count moved; null lays every edge. */
  sync(edges: readonly { id: number; a: number; b: number }[], changed?: ReadonlySet<number> | null, now?: (id: number) => { a: number; b: number } | undefined) {
    if (changed && now && this.ends.size && !this.resortAll) return this.syncChanged(changed, now);
    this.resortAll = false;
    this.countMoved.clear();
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

  /** T-4291's check (`?patch=check`): what differs from a whole `sync` of `edges` at the counts held, after `sync`; and
   * with `lines`, the counts from a whole `setLines`, after `setLines` */
  differs(edges: readonly { id: number; a: number; b: number }[], lines?: readonly Line[]): string[] {
    const bad: string[] = [];
    if (lines) {
      const f = new PlatformLines();
      f.setLines(lines);
      if (f.onEdge.size !== this.onEdge.size || [...f.onEdge].some(([e, n]) => this.onEdge.get(e) !== n)) bad.push("counts");
      return bad;
    }
    const f = new PlatformLines();
    f.onEdge = this.onEdge;
    f.sync(edges);
    if (f.at.size !== this.at.size || [...f.at].some(([n, v]) => { const w = this.at.get(n); return !w || w.join() !== v.join(); })) bad.push("ways");
    if (f.ends.size !== this.ends.size || [...f.ends].some(([e, v]) => { const w = this.ends.get(e); return !w || w[0] !== v[0] || w[1] !== v[1]; })) bad.push("ends");
    return bad;
  }

  private syncChanged(changed: ReadonlySet<number>, now: (id: number) => { a: number; b: number } | undefined) {
    const at = this.at, nodes = new Set<number>();
    for (const id of changed) {
      const was = this.ends.get(id), is = now(id);
      if (was && is && was[0] === is.a && was[1] === is.b) continue;
      if (was) {
        for (const n of new Set(was)) { const v = at.get(n); if (v) at.set(n, v.filter((e) => e !== id)); nodes.add(n); }
        this.ends.delete(id);
      }
      if (is) {
        this.ends.set(id, [is.a, is.b]);
        // (an edge from a node back to it is listed there twice, as `sync` lists it)
        for (const n of [is.a, is.b]) { const v = at.get(n); if (v) v.push(id); else at.set(n, [id]); nodes.add(n); }
      }
    }
    for (const e of this.countMoved) { const en = this.ends.get(e); if (en) { nodes.add(en[0]); nodes.add(en[1]); } }
    this.countMoved.clear();
    const on = this.onEdge;
    for (const n of nodes) {
      const v = at.get(n);
      if (!v?.length) { at.delete(n); this.kept.delete(n); continue; }
      if (v.length > 1) v.sort((p, q) => (on.get(q) ?? 0) - (on.get(p) ?? 0) || p - q);
    }
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
