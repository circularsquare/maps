// The roads a bus route follows (T-1465, notes/T-1445.md "Stage 2"). Between two clicked points the
// route takes the shortest path over the basemap's zoom 14 roads (`roadGraph.ts`), counting road length
// as it is, and stays straight where that path is over 3 times the straight line or no road is near a
// point. The tiles are fetched here, straight from the basemap's own tile server, because the map holds
// only the zoom it is showing; they are kept decoded, and the graph is built from them as a leg needs.

import type { Map as MlMap } from "maplibre-gl";
import { EXTENT, RoadGraph, WORLD, gxOf, gyOf, latOf, lngOf, metresPerUnit, thin, type Snap } from "./roadGraph";
import { decodeRoads, type RoadLines } from "./roadTiles";

/** A click this near a road (ground metres) is moved onto it. */
export const SNAP_M = 100;
/** A road path longer than this many times the straight line is not used. */
export const DETOUR = 3;
/** The route is thinned to within this (ground metres) of the road. */
const THIN_M = 2.5;
/** Tiles beyond which a leg is left straight (a leg this long is not a bus leg). */
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
}

type Tile = { rl: RoadLines | null; at: number } | "loading";

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
    const p = (async () => {
      try {
        const style = this.map.getStyle();
        const layer = style?.layers.find((l) => (l as { "source-layer"?: string })["source-layer"] === "transportation");
        const id = layer && "source" in layer ? (layer.source as string) : null;
        if (!id) return null;
        const live = this.map.getSource(id) as { tiles?: string[]; url?: string } | undefined;
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
    })();
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
        if (r.status === 200) rl = decodeRoads(await r.arrayBuffer());
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

  /** The route between two clicked points: along the roads when a road path under `DETOUR` times the
   * straight line joins the roads they land on, else straight. Not `ready` while tiles load (straight
   * meanwhile); the callback given at construction runs when they have. */
  leg(a: Pt, b: Pt): Leg {
    const ax = gxOf(a.lng), ay = gyOf(a.lat), bx = gxOf(b.lng), by = gyOf(b.lat);
    const mpu = metresPerUnit((ay + by) / 2);
    const line = (pa: [number, number], pb: [number, number], ready: boolean): Leg => ({
      pts: [lngOf(pa[0]), latOf(pa[1]), lngOf(pb[0]), latOf(pb[1])],
      road: false,
      ready,
      m: Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * mpu,
    });
    const straight = Math.hypot(ax - bx, ay - by) * mpu;
    const pad = Math.max(600, 0.25 * straight) / mpu;
    const x0 = Math.min(ax, bx) - pad, x1 = Math.max(ax, bx) + pad, y0 = Math.min(ay, by) - pad, y1 = Math.max(ay, by) + pad;
    const tiles = (Math.floor(x1 / EXTENT) - Math.floor(x0 / EXTENT) + 1) * (Math.floor(y1 / EXTENT) - Math.floor(y0 / EXTENT) + 1);
    if (tiles > MAX_TILES) return line([ax, ay], [bx, by], true);
    if (!this.need(x0, y0, x1, y1)) return line([ax, ay], [bx, by], false);
    this.fresh();
    const key = `${a.lng},${a.lat},${b.lng},${b.lat}`;
    const hit = this.legs.get(key);
    if (hit) return hit;
    const t0 = performance.now();
    const sa = this.snapAt(ax, ay, mpu), sb = this.snapAt(bx, by, mpu);
    const pa: [number, number] = sa ? [sa.x, sa.y] : [ax, ay], pb: [number, number] = sb ? [sb.x, sb.y] : [bx, by];
    let out = line(pa, pb, true);
    if (sa && sb) {
      const direct = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) * mpu;
      const path = this.graph.path(sa, sb, Math.max(direct * DETOUR, 1));
      if (path) {
        const pts = thin(path.pts, THIN_M / mpu);
        const ll: number[] = [];
        for (let i = 0; i < pts.length; i += 2) ll.push(lngOf(pts[i]), latOf(pts[i + 1]));
        out = { pts: ll, road: true, ready: true, m: path.m };
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
