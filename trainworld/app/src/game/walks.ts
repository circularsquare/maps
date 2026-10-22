// Walking minutes between stations and from a place to its closest stations (T-312), on the same
// walking barrier the demand model routes walks on (T-275, sim/src/demand/routes.rs): the pack's
// water raster at 50 m less the OSM bridges a person may walk over. The search below is that
// file's `WalkGrid::field` and `Field::eff`, in the same arithmetic, run on the main thread from
// the pack's `walk_row` and `walk_x` (about 250 KB a city, fetched once with a Range request) so
// a hover or an inspector never waits for a demand worker mid-solve. The metres are the demand
// model's walk; the minutes are those metres x its 1.3 detour at its 6 km/h (`Params::walk_mps`,
// T-371; change them together).

import { signal } from "@preact/signals";
import { cityById, type CityId } from "./cities";
import { makeFrame } from "./frame";
import type { StationInput } from "./types";
import { zoneAt } from "./zones";
import { dataUrl } from "./dataBase";

const ORTH = 70, DIAG = 99, INF = 0x7fffffff, RING = 256;
const OCTILE_MAX = 1.0824;
const SNAP_PX = 10;

/** The demand model's walking pace (T-371, T-389) and its detour over the routed metres. */
export const WALK_KMH = 6;
export const WALK_DETOUR = 1.3;

/** The stations a place lists, and how far it looks for them: first 4 km (past the demand
 * model's 3 km walk), then 10 km when fewer than three are reachable. */
const BOUNDS = [4000, 10000];
const SHOWN = 3;

/** Bumped when a city's walking barrier arrives, so a list worked out without it is worked out again. */
export const walkGridEpoch = signal(0);

interface Grid { x0: number; y0: number; cell: number; w: number; h: number; rowOff: Uint32Array; xs: Uint16Array }

const grids = new Map<CityId, Grid | null>();
const loading = new Map<CityId, Promise<void>>();

function loadGrid(id: CityId) {
  if (loading.has(id)) return;
  const p = (async () => {
    const base = dataUrl(id);
    const head = await (await fetch(base + ".json")).json();
    const wb = head.routes?.walk;
    const row = head.arrays?.find((a: { name: string }) => a.name === "walk_row");
    const xs = head.arrays?.find((a: { name: string }) => a.name === "walk_x");
    if (!wb || !row || !xs) return void grids.set(id, null);
    const start = row.offset as number, end = xs.offset + xs.count * 2;
    const res = await fetch(base + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` } });
    const all = await res.arrayBuffer();
    const at = res.status === 206 ? start : 0;
    const buf = all.slice(row.offset - at, end - at);
    grids.set(id, {
      x0: wb.origin_m[0], y0: wb.origin_m[1], cell: wb.cell_m, w: wb.size[0], h: wb.size[1],
      rowOff: new Uint32Array(buf, 0, row.count),
      xs: new Uint16Array(buf, xs.offset - row.offset, xs.count),
    });
  })().catch((err) => {
    console.warn("walking barrier unavailable", id, err);
    grids.set(id, null);
  }).then(() => { walkGridEpoch.value++; });
  loading.set(id, p);
}

// ---------------------------------------------------------------- the search

/** One place's walk search: costs to every pixel of a window round the barrier. */
interface Field { x0: number; y0: number; cell: number; w: number; sc: number; sr: number; d: Uint32Array; bar: Uint8Array }

/** Pixel (c, r) if walkable, else the walkable pixels within SNAP_PX at most two steps further
 * than the nearest, each with its cost from (c, r). */
function nearestWalkable(bar: Uint8Array, w: number, c: number, r: number): [number, number][] {
  if (bar[r * w + c] === 0) return [[r * w + c, 0]];
  const v: [number, number][] = [];
  for (let dr = -SNAP_PX; dr <= SNAP_PX; dr++)
    for (let dc = -SNAP_PX; dc <= SNAP_PX; dc++) {
      const nc = c + dc, nr = r + dr;
      if (nc < 0 || nr < 0 || nc >= w || nr >= w || bar[nr * w + nc] !== 0) continue;
      const a = Math.min(Math.abs(dc), Math.abs(dr)), b = Math.max(Math.abs(dc), Math.abs(dr));
      v.push([nr * w + nc, DIAG * a + ORTH * (b - a)]);
    }
  const m = v.reduce((x, e) => Math.min(x, e[1]), INF);
  return v.filter((e) => e[1] <= m + 2 * ORTH);
}

/** The walk from (x, y) (pack metres) out to about `maxM`; null where the window holds no barrier
 * (every walk is then the straight line). */
function field(g: Grid, x: number, y: number, maxM: number): Field | null {
  const rad = Math.ceil(maxM * OCTILE_MAX / g.cell) + 3;
  const sc = Math.floor((x - g.x0) / g.cell), sr = Math.floor((y - g.y0) / g.cell);
  const c0 = sc - rad, r0 = sr - rad, w = 2 * rad + 1;
  const bar = new Uint8Array(w * w);
  let any = false;
  for (let r = Math.max(0, r0); r < Math.min(g.h, r0 + w); r++) {
    const a = g.rowOff[r], b = g.rowOff[r + 1];
    for (let k = a; k + 1 < b; k += 2) {
      const s = Math.max(g.xs[k], c0), e = Math.min(g.xs[k + 1], c0 + w);
      if (s < e) {
        bar.fill(1, (r - r0) * w + (s - c0), (r - r0) * w + (e - c0));
        any = true;
      }
    }
  }
  if (!any) return null;
  const d = new Uint32Array(w * w).fill(INF);
  const limit = Math.floor(maxM * OCTILE_MAX / g.cell * ORTH) + 2 * DIAG;
  const buckets: number[][] = Array.from({ length: RING }, () => []);
  const start = nearestWalkable(bar, w, rad, rad);
  if (!start.length) return null;
  let cur = start.reduce((m, e) => Math.min(m, e[1]), INF), pending = 0;
  for (const [i, v] of start) if (v < d[i]) { d[i] = v; buckets[v % RING].push(i); pending++; }
  const steps: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  while (pending > 0 && cur <= limit) {
    const q = buckets[cur % RING];
    let i: number | undefined;
    while ((i = q.pop()) !== undefined) {
      pending--;
      if (d[i] !== cur) continue;
      const c = i % w, r = (i - c) / w;
      for (const [dc, dr] of steps) {
        const nc = c + dc, nr = r + dr;
        if (nc < 0 || nr < 0 || nc >= w || nr >= w) continue;
        const j = nr * w + nc;
        if (bar[j] !== 0) continue;
        let step = ORTH;
        if (dc !== 0 && dr !== 0) {
          if (bar[r * w + nc] !== 0 || bar[nr * w + c] !== 0) continue; // no cutting a corner
          step = DIAG;
        }
        const v = cur + step;
        if (v < d[j]) { d[j] = v; buckets[v % RING].push(j); pending++; }
      }
    }
    cur++;
  }
  return { x0: g.x0 + c0 * g.cell, y0: g.y0 + r0 * g.cell, cell: g.cell, w, sc: rad, sr: rad, d, bar };
}

/** Walking metres to (x, y), `straight` metres away: the straight line times how much longer the
 * search's path is than the free path between the same two pixels. Null: cut off by water. */
function eff(f: Field, x: number, y: number, straight: number): number | null {
  const c = Math.floor((x - f.x0) / f.cell), r = Math.floor((y - f.y0) / f.cell);
  if (c < 1 || r < 1 || c >= f.w - 1 || r >= f.w - 1) return null;
  const i = r * f.w + c;
  let g = f.bar[i] === 0 ? f.d[i] : INF;
  if (f.bar[i] !== 0) {
    const near = nearestWalkable(f.bar, f.w, c, r);
    if (!near.length) return straight;
    for (const [j, v] of near) if (f.d[j] !== INF) g = Math.min(g, f.d[j] + v);
  }
  if (g === INF) return null;
  const a = Math.abs(c - f.sc), b = Math.abs(r - f.sr);
  const free = DIAG * Math.min(a, b) + ORTH * (Math.max(a, b) - Math.min(a, b));
  return free === 0 || g <= free ? straight : straight * g / free;
}

// ---------------------------------------------------------------- what the interface asks

/** Great-circle metres. */
export function metresBetween(lng0: number, lat0: number, lng1: number, lat1: number): number {
  const R = Math.PI / 180, dy = (lat1 - lat0) * R, dx = (lng1 - lng0) * R * Math.cos((lat0 + lat1) * R / 2);
  return 6371008.8 * Math.hypot(dx, dy);
}

/** Walking minutes for a walk of `metres` along the routed path. */
export const walkMinutes = (metres: number) => metres * WALK_DETOUR / (WALK_KMH / 3.6) / 60;

/** "7 min", "1 min" at the least. */
export const minLabel = (metres: number) => `${Math.max(1, Math.round(walkMinutes(metres)))} min`;

const fields = new Map<string, Field | null>();
function fieldAt(city: CityId, g: Grid, x: number, y: number, bound: number): Field | null {
  const key = `${city}|${x.toFixed(0)}|${y.toFixed(0)}|${bound}`;
  if (fields.has(key)) {
    const f = fields.get(key)!;
    fields.delete(key); // most recent last
    fields.set(key, f);
    return f;
  }
  const f = field(g, x, y, bound);
  fields.set(key, f);
  if (fields.size > 24) fields.delete(fields.keys().next().value!);
  return f;
}

export interface Near {
  station: StationInput;
  /** the walk along the routed path, metres; null when water cuts it off */
  metres: number | null;
  /** the straight line, metres */
  straight: number;
}

/**
 * The `n` stations (of `stations`, minus `exclude`) with the shortest walk from (lng, lat), nearest
 * first. A station the water cuts off from the place is listed ("no walk") only to fill up the
 * list when fewer than `n` can be walked to within 10 km. A name is listed once (the real network
 * has a station per platform group, and one name can stand at several), nearest first; `own` is a
 * name left out altogether, the station asking's.
 */
export function closestStations(lng: number, lat: number, stations: readonly StationInput[], exclude: ReadonlySet<string> = new Set(), n = SHOWN, own: string | null = null): Near[] {
  walkGridEpoch.value; // read, so a component re-renders when the barrier arrives
  const key = (s: StationInput) => s.name.trim().toLowerCase();
  const ownKey = own === null ? null : own.trim().toLowerCase();
  const once = (list: Near[]) => {
    const seen = new Set<string>();
    return list.filter((r) => !seen.has(key(r.station)) && !!seen.add(key(r.station)));
  };
  const all = stations
    .filter((s) => !exclude.has(s.id) && key(s) !== ownKey)
    .map((s) => ({ station: s, straight: metresBetween(lng, lat, s.lng, s.lat) }))
    .sort((a, b) => a.straight - b.straight);
  if (!all.length) return [];
  const city = zoneAt(lng, lat);
  if (!grids.has(city)) loadGrid(city);
  const g = grids.get(city) ?? null;
  const o = g ? makeFrame({ kind: "equirect", ...cityById(city).origin }) : null;
  const [px, py] = o ? o.fromLonLat(lng, lat) : [0, 0];
  let walked: Near[] = [], cut: Near[] = [];
  for (const bound of BOUNDS) {
    const near = all.filter((s) => s.straight <= bound);
    const f = g && o ? fieldAt(city, g, px, py, bound) : null;
    walked = []; cut = [];
    for (const s of near) {
      if (!f || !o) { walked.push({ ...s, metres: s.straight }); continue; }
      const [sx, sy] = o.fromLonLat(s.station.lng, s.station.lat);
      const m = eff(f, sx, sy, Math.hypot(sx - px, sy - py));
      (m === null ? cut : walked).push({ ...s, metres: m });
    }
    walked = once(walked.sort((a, b) => a.metres! - b.metres!));
    const names = new Set(walked.map((r) => key(r.station)));
    cut = once(cut).filter((r) => !names.has(key(r.station)));
    // done when n are walkable and the n-th is no further than the bound looked at (a station
    // beyond it is further in a straight line, so further to walk)
    if (walked.length >= n && walked[n - 1].metres! <= bound) break;
  }
  return [...walked.slice(0, n), ...cut.slice(0, Math.max(0, n - walked.length))];
}
