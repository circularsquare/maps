// The roads a bus route follows (T-1465, notes/T-1445.md "Stage 2"). Between two clicked points the
// route takes the shortest path over the basemap's zoom 14 roads and motorways (`roadGraph.ts`), counting road length
// as it is; a leg with no such path is refused (T-2451: no road near a point, no path under the detour
// cap, or too long to search), and the tool keeps it out of a route. The tiles are fetched here,
// straight from the basemap's own tile server, because the map holds only the zoom it is showing; they
// are kept decoded, and the graph is built from them as a leg needs.

import type { Map as MlMap } from "maplibre-gl";
import { EXTENT, LINK, MOTORWAY, RoadGraph, TRUNK, WORLD, gxOf, gyOf, joinRoads, latOf, lngOf, metresPerUnit, thin, thinFlagged, type Snap } from "./roadGraph";
import { BUS_CLASSES, decodeRoads, type RoadKeep, type RoadLines } from "./roadTiles";

/** The roads a bus drives: the bus classes, and motorways (T-2469: most long bridges and tunnels are
 * motorways, and their approaches too), which a click lands on only where no street is near. An
 * unnamed-kind service road open to all may be a short link between streets (`LINK`, T-2882). */
const busRoad: RoadKeep = (cls, brunnel, service, access) =>
  cls === "trunk" ? TRUNK : BUS_CLASSES.has(cls) ? 1 : cls === "motorway" ? MOTORWAY : cls === "service" && !service && !access && !brunnel ? LINK : 0;

/** `busRoad` as a MapLibre filter on the basemap's `transportation` lines (T-2883), for drawing the roads
 * a bus may use; built from the same classes, so the two cannot drift apart. Without `motorways`, the
 * streets a tram follows (T-2998: `RoadGraph` keeps a street route off motorways). */
export const busRoadFilter = (motorways = true): unknown[] => [
  "any",
  ["in", ["get", "class"], ["literal", motorways ? [...BUS_CLASSES, "motorway"] : [...BUS_CLASSES]]],
  ["all", ["==", ["get", "class"], "service"], ["!", ["has", "service"]], ["!", ["has", "access"]], ["!", ["has", "brunnel"]]],
];

/** A click this near a road (ground metres) is moved onto it. */
export const SNAP_M = 100;
/** A road path longer than this many times the straight line, and `DETOUR_M` more than it, is not used:
 * a short leg may still go round a block or to the next crossing. */
export const DETOUR = 3;
const DETOUR_M = 1000;
/** A free end landing on a road other than the nearest pays this many metres of road for each metre it
 * is from the click (`snapCost`), so the nearest reachable road wins unless another saves far more road;
 * at most `MAX_ALTS` other roads are tried. */
const SNAP_PEN = 3;
const MAX_ALTS = 400;
/** A tram end's streets tried (T-2927), nearest first. */
const TRAM_ALTS = 12;
/** The route is thinned to within this (ground metres) of the road. */
const THIN_M = 2.5;
/** Tiles beyond which a leg is not searched (refused as `long`). */
const MAX_TILES = 100;
/** Tiles the graph holds at most; past it the graph is rebuilt from the decoded tiles. */
const GRAPH_TILES = 220;
/** Decoded tiles kept. */
const KEEP_TILES = 500;
const PARALLEL = 8;
const RETRY_MS = 15000;

export interface Pt {
  lng: number;
  lat: number;
}

/** Why a leg has no road: a point more than `SNAP_M` from any road, no road path under the detour cap,
 * or the leg too long to search. */
export type BusRefusal = "far" | "path" | "long";

/** A stretch of route between two clicked points. */
export interface Leg {
  /** longitude, latitude pairs from the first point (moved onto its road) to the last */
  pts: number[];
  /** it follows roads */
  road: boolean;
  /** the tiles it needs are loaded; false while it waits, when `pts` is the straight line */
  ready: boolean;
  /** road metres (the straight line's length when it is straight) */
  m: number;
  /** per stretch between points of `pts`, on a trunk road (T-2904); absent for none */
  fast?: boolean[];
  /** ready but with no road: why, and `pts` is the straight line */
  refused?: BusRefusal;
}

type Tile = { rl: RoadLines | null; at: number } | "loading";

/** The basemap's tile URL template for the layer holding the roads; null while the style is not in. */
export async function roadTileTemplate(map: MlMap): Promise<string | null> {
  try {
    const style = map.getStyle();
    const layer = style?.layers.find((l) => (l as { "source-layer"?: string })["source-layer"] === "transportation");
    const id = layer && "source" in layer ? (layer.source as string) : null;
    if (!id) return null;
    const live = map.getSource(id) as { tiles?: string[]; url?: string } | undefined;
    const spec = style.sources[id] as { tiles?: string[]; url?: string } | undefined;
    const tiles = live?.tiles ?? spec?.tiles;
    if (tiles?.length) return tiles[0];
    const url = live?.url ?? spec?.url;
    if (!url) return null;
    const json = await (await fetch(url)).json();
    return json.tiles?.[0] ?? null;
  } catch {
    return null;
  }
}

export class BusRoads {
  private graph = new RoadGraph();
  private tiles = new Map<number, Tile>();
  private queue: number[] = [];
  private active = 0;
  private template: Promise<string | null> | null = null;
  private legs = new Map<string, Leg>();
  private snaps = new Map<string, Snap | null>();
  private cachedVersion = -1;
  private notify = false;
  /** the search's cost, for the check: calls, total and slowest, ms */
  stats = { legs: 0, ms: 0, max: 0, tiles: 0 };

  constructor(private map: MlMap, private onLoad: () => void) {}

  // ---------------------------------------------------------------- tiles

  private url(): Promise<string | null> {
    if (this.template) return this.template;
    const p = roadTileTemplate(this.map);
    // a style that is not there yet is asked again
    void p.then((u) => {
      if (!u) this.template = null;
    });
    return (this.template = p);
  }

  private pump() {
    while (this.active < PARALLEL && this.queue.length) {
      const key = this.queue.shift()!;
      this.active++;
      void this.load(key).finally(() => {
        this.active--;
        this.pump();
      });
    }
  }

  private async load(key: number) {
    const tx = Math.floor(key / 16384), ty = key % 16384;
    let rl: RoadLines | null = null;
    try {
      const tpl = await this.url();
      if (tpl) {
        const r = await fetch(tpl.replace("{z}", "14").replace("{x}", String(tx)).replace("{y}", String(ty)));
        if (r.status === 200) rl = joinRoads(decodeRoads(await r.arrayBuffer(), busRoad), metresPerUnit((ty + 0.5) * EXTENT));
        else if (r.status === 404 || r.status === 204) rl = { extent: EXTENT, lines: [] };
      }
    } catch {
      rl = null;
    }
    this.stats.tiles++;
    this.tiles.set(key, { rl, at: Date.now() });
    if (this.tiles.size > KEEP_TILES) {
      for (const k of this.tiles.keys()) {
        if (this.tiles.size <= KEEP_TILES * 0.8) break;
        if (this.tiles.get(k) !== "loading") this.tiles.delete(k);
      }
    }
    if (!this.notify) {
      this.notify = true;
      queueMicrotask(() => {
        this.notify = false;
        this.onLoad();
      });
    }
  }

  /** Make sure the tiles over a grid box are loading or loaded; true when every one is in the graph.
   * A tile that failed counts as empty for `RETRY_MS`. */
  private need(x0: number, y0: number, x1: number, y1: number): boolean {
    const tx0 = Math.floor(x0 / EXTENT), tx1 = Math.floor(x1 / EXTENT), ty0 = Math.floor(y0 / EXTENT), ty1 = Math.floor(y1 / EXTENT);
    const keys: number[] = [];
    let all = true;
    for (let tx = tx0; tx <= tx1; tx++)
      for (let ty = ty0; ty <= ty1; ty++) {
        if (ty < 0 || ty >= WORLD / EXTENT) continue;
        const key = RoadGraph.tileKey(tx, ty);
        keys.push(key);
        const t = this.tiles.get(key);
        if (t === undefined || (t !== "loading" && !t.rl && Date.now() - t.at > RETRY_MS)) {
          this.tiles.set(key, "loading");
          this.queue.push(key);
          all = false;
        } else if (t === "loading") all = false;
      }
    this.pump();
    if (!all) return false;
    let missing = keys.filter((k) => !this.graph.has(Math.floor(k / 16384), k % 16384));
    if (missing.length && this.graph.tileCount + missing.length > GRAPH_TILES) {
      this.graph.clear();
      missing = keys;
    }
    for (const k of missing) {
      const t = this.tiles.get(k);
      this.graph.addTile(Math.floor(k / 16384), k % 16384, t && t !== "loading" && t.rl ? t.rl : { extent: EXTENT, lines: [] });
    }
    return true;
  }

  private fresh() {
    if (this.cachedVersion === this.graph.version) return;
    this.cachedVersion = this.graph.version;
    this.legs.clear();
    this.snaps.clear();
  }

  // ---------------------------------------------------------------- legs

  /** Start loading the roads around a point (the pointer, before any leg needs them); true when they are. */
  look(p: Pt): boolean {
    const x = gxOf(p.lng), y = gyOf(p.lat), pad = 600 / metresPerUnit(y);
    return this.need(x - pad, y - pad, x + pad, y + pad);
  }

  private snapAt(x: number, y: number, mpu: number): Snap | null {
    const key = `${x},${y}`;
    if (this.snaps.has(key)) return this.snaps.get(key)!;
    const s = this.graph.snap(x, y, SNAP_M / mpu);
    if (this.snaps.size > 400) this.snaps.clear();
    this.snaps.set(key, s);
    return s;
  }

  /** Where a click lands: on the nearest road within `SNAP_M`, else where it is; null while the tiles
   * load. */
  corner(p: Pt): Pt | null {
    if (!this.look(p)) return null;
    this.fresh();
    const x = gxOf(p.lng), y = gyOf(p.lat), s = this.snapAt(x, y, metresPerUnit(y));
    return s ? { lng: lngOf(s.x), lat: latOf(s.y) } : p;
  }

  /** T-2927: where a tram click lands: on the nearest street (no motorway) within `SNAP_M`, null where
   * there is none; undefined while the tiles load. */
  street(p: Pt): Pt | null | undefined {
    if (!this.look(p)) return undefined;
    this.fresh();
    const x = gxOf(p.lng), y = gyOf(p.lat), s = this.graph.snap(x, y, SNAP_M / metresPerUnit(y), true);
    return s ? { lng: lngOf(s.x), lat: latOf(s.y) } : null;
  }

  /** T-2927: the streets a tram follows between two points (`RoadGraph.turnPath`), as lng, lat pairs
   * from the street point by `a` to the one by `b`. `refused` as a bus leg's; not `ready` while tiles load.
   * `from`: a point the track ran from to reach `a`, where the route carries on from it (T-2997). */
  tramLeg(a: Pt, b: Pt, turnM: number, minTurn: number, maxTurn: number, backM: number, from: Pt | null = null): Leg & { turns: number; states: number; ms: number } {
    const ax = gxOf(a.lng), ay = gyOf(a.lat), bx = gxOf(b.lng), by = gyOf(b.lat);
    const mpu = metresPerUnit((ay + by) / 2);
    const straight = Math.hypot(ax - bx, ay - by) * mpu;
    const none = (ready: boolean, refused?: BusRefusal) => ({ pts: [a.lng, a.lat, b.lng, b.lat], road: false, ready, m: straight, refused, turns: 0, states: 0, ms: 0 });
    const pad = Math.max(600, 0.25 * straight) / mpu;
    const x0 = Math.min(ax, bx) - pad, x1 = Math.max(ax, bx) + pad, y0 = Math.min(ay, by) - pad, y1 = Math.max(ay, by) + pad;
    const tiles = (Math.floor(x1 / EXTENT) - Math.floor(x0 / EXTENT) + 1) * (Math.floor(y1 / EXTENT) - Math.floor(y0 / EXTENT) + 1);
    if (tiles > MAX_TILES) return none(true, "long");
    if (!this.need(x0, y0, x1, y1)) return none(false);
    this.fresh();
    const t0 = performance.now();
    // the nearest streets by each end (the nearest alone can be a stub joined to nothing)
    // (one a road: a segment touching one taken, as a curved road's next one does, is skipped)
    const near = (x: number, y: number) => {
      const out: Snap[] = [], seen = new Set<number>(), g = this.graph;
      for (const s of g.snapsWithin(x, y, SNAP_M / mpu, true)) {
        const a = g.ea[s.edge], b = g.eb[s.edge], touches = seen.has(a) || seen.has(b);
        seen.add(a).add(b);
        if (touches) continue;
        out.push(s);
        if (out.length >= TRAM_ALTS) break;
      }
      return out;
    };
    const sa = near(ax, ay), sb = near(bx, by);
    if (!sa.length || !sb.length) return none(true, "far");
    const direct = Math.hypot(sa[0].x - sb[0].x, sa[0].y - sb[0].y) * mpu;
    let dir: [number, number] | null = null;
    if (from) {
      const dx = ax - gxOf(from.lng), dy = ay - gyOf(from.lat), l = Math.hypot(dx, dy);
      if (l > 0) dir = [dx / l, dy / l];
    }
    const path = this.graph.turnPath(sa, sb, (c) => SNAP_PEN * c.d * mpu, Math.max(direct * DETOUR, direct + DETOUR_M) + 20 * turnM, turnM, minTurn, maxTurn, backM, dir);
    const ms = performance.now() - t0;
    if (!path) return { ...none(true, "path"), ms };
    const pts = thin(path.pts, 1 / mpu), ll: number[] = [];
    for (let i = 0; i < pts.length; i += 2) ll.push(lngOf(pts[i]), latOf(pts[i + 1]));
    return { pts: ll, road: true, ready: true, m: path.m, turns: path.turns, states: path.states, ms };
  }

  /** A road within `SNAP_M` of a point: true or false, null while the tiles load. */
  near(p: Pt): boolean | null {
    if (!this.look(p)) return null;
    this.fresh();
    const x = gxOf(p.lng), y = gyOf(p.lat);
    return this.snapAt(x, y, metresPerUnit(y)) !== null;
  }

  /** The route between two clicked points: along the roads when a road path under the detour cap joins
   * the roads they land on, else `refused` (straight). Not `ready` while tiles load (straight
   * meanwhile); the callback given at construction runs when they have. The `free` end (a point being
   * placed or moved) may land on another road within `SNAP_M` where the nearest has no path (T-2467),
   * each metre further from the click costing `SNAP_PEN` of road; the leg's first or last point is
   * where it landed. */
  leg(a: Pt, b: Pt, free: "a" | "b" | null = null): Leg {
    const ax = gxOf(a.lng), ay = gyOf(a.lat), bx = gxOf(b.lng), by = gyOf(b.lat);
    const mpu = metresPerUnit((ay + by) / 2);
    const line = (pa: [number, number], pb: [number, number], ready: boolean, refused?: BusRefusal): Leg => ({
      pts: [lngOf(pa[0]), latOf(pa[1]), lngOf(pb[0]), latOf(pb[1])],
      road: false,
      ready,
      m: Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * mpu,
      refused,
    });
    const straight = Math.hypot(ax - bx, ay - by) * mpu;
    const pad = Math.max(600, 0.25 * straight) / mpu;
    const x0 = Math.min(ax, bx) - pad, x1 = Math.max(ax, bx) + pad, y0 = Math.min(ay, by) - pad, y1 = Math.max(ay, by) + pad;
    const tiles = (Math.floor(x1 / EXTENT) - Math.floor(x0 / EXTENT) + 1) * (Math.floor(y1 / EXTENT) - Math.floor(y0 / EXTENT) + 1);
    if (tiles > MAX_TILES) return line([ax, ay], [bx, by], true, "long");
    if (!this.need(x0, y0, x1, y1)) return line([ax, ay], [bx, by], false);
    this.fresh();
    const key = `${a.lng},${a.lat},${b.lng},${b.lat},${free}`;
    const hit = this.legs.get(key);
    if (hit) return hit;
    const t0 = performance.now();
    const sa = this.snapAt(ax, ay, mpu), sb = this.snapAt(bx, by, mpu);
    const pa: [number, number] = sa ? [sa.x, sa.y] : [ax, ay], pb: [number, number] = sb ? [sb.x, sb.y] : [bx, by];
    let out = line(pa, pb, true, sa && sb ? "path" : "far");
    if (sa && sb) {
      const direct = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * mpu;
      const cap = Math.max(direct * DETOUR, direct + DETOUR_M);
      let path = this.graph.path(sa, sb, cap);
      if (!path && free) {
        // the nearest road has no path: another road near the free end may (searched from the fixed one)
        const [fixed, near, fx, fy] = free === "b" ? [sa, sb, bx, by] : [sb, sa, ax, ay];
        const alts = this.graph.snapsWithin(fx, fy, SNAP_M / mpu).slice(1, 1 + MAX_ALTS);
        const got = alts.length ? this.graph.path(fixed, near, cap + SNAP_PEN * SNAP_M, alts, (c) => SNAP_PEN * this.graph.snapCost(c) * mpu) : null;
        if (got && free === "a") {
          const r: number[] = [];
          for (let i = got.pts.length - 2; i >= 0; i -= 2) r.push(got.pts[i], got.pts[i + 1]);
          got.pts = r;
          got.fast?.reverse();
        }
        path = got;
      }
      if (path) {
        const thinned = path.fast ? thinFlagged(path.pts, path.fast, THIN_M / mpu) : { pts: thin(path.pts, THIN_M / mpu), flags: undefined };
        const pts = thinned.pts;
        const ll: number[] = [];
        for (let i = 0; i < pts.length; i += 2) ll.push(lngOf(pts[i]), latOf(pts[i + 1]));
        out = { pts: ll, road: true, ready: true, m: path.m, fast: thinned.flags?.some(Boolean) ? thinned.flags : undefined };
      }
    }
    const ms = performance.now() - t0;
    this.stats.legs++;
    this.stats.ms += ms;
    this.stats.max = Math.max(this.stats.max, ms);
    if (this.legs.size > 300) this.legs.clear();
    this.legs.set(key, out);
    return out;
  }
}
