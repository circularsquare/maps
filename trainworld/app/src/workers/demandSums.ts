// T-1208, T-1210: demand's sums that need no WASM, done in the sums worker (demandSums.worker.ts)
// and on the main thread only where it fails: a solve's views of each kind of day, and a
// selection's commuters and longer trips and a station's from the workers' answers. Pure functions
// of their inputs, so the figures are the same wherever they run.

import type { DriveInput } from "../game/drives";
import { makeFrame, type FrameSpec } from "../game/frame";
import { EVERYDAY_KEYS, membersOf, PURPOSES, shownKey, type PurposeFilter, type PurposeKey } from "../game/purposes";
import { PERIODS } from "../game/periods";
import type { CityId } from "../game/cities";
import { BACK_DAY, DEMAND_DAYS, OUT_DAY, type DemandDay } from "../game/days";
import type { Corridors, DemandView, FarEnd, Flows, IntercityDemand, IntercitySelection, IntercityStation, LineDemand, PlaceFlows, PlaceTrips, PurposeDemand, StationDemand, ZoneTrips } from "../game/demand";
import type { DealtCells } from "../game/farChunks";
import type { FromDemand } from "./demandProtocol";
import { marketPartsOn } from "./marketParts";
import { mergeEveryday } from "./selectionMerge";

/** The everyday kinds (`EVERYDAY_KEYS` order) whose outings cross zone lines, in the order of
 * `IntercityApi.selection`'s blocks (sim `E_CROSS`). */
export const CROSS_KINDS = [0, 1, 2, 3, 5];

/** The zone count of a longer-trip answer of `len` numbers, with `m` market places. For M markets a
 * selection has 1+8*M header numbers, 4 per zone of flows, 3 per zone of x, y, market, then (T-276)
 * the commutes between zones: 4 per market and 4 per zone, then (T-1229) 6 per zone for each of the
 * everyday kinds that cross zone lines (`CROSS_KINDS`: legs, rail and walking, starting and ending),
 * then (T-378) the trips with places that have no zone, starting and ending; a station has 6*M
 * header numbers then 5 per zone. */
export function intercityZones(kind: "icSel" | "icStation", len: number, m: number): number {
  return kind === "icSel" ? (len - 3 - 12 * m) / (11 + 6 * CROSS_KINDS.length) : (len - 6 * m) / 5;
}
/** Where the zones' own x, y, market start: the same on every worker, so never added up. */
export function intercityZoneXY(kind: "icSel" | "icStation", len: number, m: number): number {
  const nz = intercityZones(kind, len, m);
  return kind === "icSel" ? 1 + 8 * m + 4 * nz : 6 * m + 2 * nz;
}
/** Add a worker's longer-trip answer into `sums`: everything but the matched share (a selection's
 * first number) and the zones' x, y, market. */
export function addIntercity(kind: "icSel" | "icStation", sums: Float64Array, v: Float64Array, m: number) {
  const xy = intercityZoneXY(kind, sums.length, m), end = xy + 3 * intercityZones(kind, sums.length, m);
  for (let k = kind === "icSel" ? 1 : 0; k < sums.length; k++) if (k < xy || k >= end) sums[k] += v[k];
}
/** Every worker's answer added up, in the order given (the workers' period order). The first is
 * added into, so the caller hands over arrays it no longer needs. */
export function sumParts(kind: "icSel" | "icStation", parts: Float64Array[], m: number): Float64Array {
  const sums = parts[0];
  for (let i = 1; i < parts.length; i++) addIntercity(kind, sums, parts[i], m);
  return sums;
}

/** A fingerprint of an answer's exact numbers: stations the demand model merged answer with the
 * same numbers, so a station complex counts them once (T-146). */
export function signature(a: Float32Array | Float64Array): string {
  const bits = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4);
  let h = 0x811c9dc5;
  for (let i = 0; i < bits.length; i++) h = Math.imul(h ^ bits[i], 0x01000193);
  return `${a.length}:${h >>> 0}`;
}

/** What turns the market's numbers into the shown ones: the market's frame and the game's (a point
 * in the market frame as a world point is cities.ts `marketToWorld`), the market's place count, and
 * for a selection the place it lies in (-1 none), each place's share of its longer trips not by rail
 * that fly and that drive on the corridor with that place (2 per place, NaN where the corridor has
 * none; null before any are known or with no place of its own: all by car), and each place's
 * registry centre as lon, lat (NaN for a place with no zone). */
export interface SumsCtx {
  market: FrameSpec;
  world: FrameSpec;
  places: number;
  own?: number;
  modes?: Float64Array | null;
  centres?: Float64Array;
}

function toWorld(ctx: SumsCtx): (x: number, y: number) => { x: number; y: number } {
  const from = makeFrame(ctx.market), to = makeFrame(ctx.world);
  return (x, y) => {
    if (from === to) return { x, y };
    const [lon, lat] = from.toLonLat(x, y);
    const [px, py] = to.fromLonLat(lon, lat);
    return { x: px, y: py };
  };
}

/** The longer trips of a bubble selection from its summed answer `v` (game/demand.ts
 * `IntercitySelection`, T-118), less its version. */
export function selectionFlows(v: Float64Array, ctx: SumsCtx): Omit<IntercitySelection, "version"> {
  const marketXY = toWorld(ctx), world = makeFrame(ctx.world);
  // per place, 8 numbers after the matched share: starts (all, rail, x, y) then ends; then per
  // far zone (all, rail) of starts and of ends, then zone x, y, market (not summed)
  const P = ctx.places;
  const head = 1 + 8 * P, nz = intercityZones("icSel", v.length, P), xy = intercityZoneXY("icSel", v.length, P);
  // T-276: the commutes between zones after the zones' x, y: per place, then per zone, the legs of
  // commuters living in the selection (by where they work) and of those working there (by where
  // they live), all and rail. Starts take the first, ends the second.
  const tail = xy + 3 * nz;
  // T-1229: then the everyday outings across a zone line: per kind of `CROSS_KINDS`, per far zone,
  // (legs, rail legs, walking legs) of the residents' outings and of those arriving, as the
  // commutes' blocks are
  const etail = tail + 4 * P + 4 * nz;
  const own = ctx.own ?? -1, modes = ctx.modes ?? null, centres = ctx.centres;
  // T-378: the longer trips not by rail by the corridor's shares (`[by plane, by car]`); commuters drive
  const split = (p: number, rest: number): [number, number] => {
    if (own < 0 || !modes || Number.isNaN(modes[2 * p])) return [0, rest];
    return [rest * modes[2 * p], rest * modes[2 * p + 1]];
  };
  const centre = (p: number): { x: number; y: number } | null => {
    if (!centres || Number.isNaN(centres[2 * p])) return null;
    const [x, y] = world.fromLonLat(centres[2 * p], centres[2 * p + 1]);
    return { x, y };
  };
  const flows = (dir: number, longer: boolean, kinds: number[]): PlaceFlows => {
    const places: PlaceTrips[] = [];
    const zones: ZoneTrips[] = [];
    // the everyday outings of `kinds` by far zone, then by place
    const cz = new Map<number, [number, number, number]>();
    for (const ki of kinds) {
      for (let z = 0; z < nz; z++) {
        const o = etail + ki * 6 * nz + 6 * z + 3 * dir;
        if (!(v[o] > 0)) continue;
        const e = cz.get(z) ?? cz.set(z, [0, 0, 0]).get(z)!;
        e[0] += v[o]; e[1] += v[o + 1]; e[2] += Math.max(0, v[o + 2]);
      }
    }
    const cp = new Map<number, [number, number, number, number, number]>();
    for (const [z, e] of cz) {
      const pl = v[xy + 3 * z + 2];
      const x = cp.get(pl) ?? cp.set(pl, [0, 0, 0, 0, 0]).get(pl)!;
      x[0] += e[0]; x[1] += e[1]; x[2] += e[2]; x[3] += e[0] * v[xy + 3 * z]; x[4] += e[0] * v[xy + 3 * z + 1];
    }
    let all = 0, rail = 0, air = 0, car = 0, walk = 0;
    for (let p = 0; p < P; p++) {
      const o = 1 + 8 * p + 4 * dir;
      const ca = longer ? v[tail + 4 * p + 2 * dir] ?? 0 : 0, cr = longer ? v[tail + 4 * p + 2 * dir + 1] ?? 0 : 0;
      const lo = longer ? v[o] : 0, lr = longer ? v[o + 1] : 0;
      const x = cp.get(p);
      const a = lo + ca + (x?.[0] ?? 0), r = lr + cr + (x?.[1] ?? 0), w = x?.[2] ?? 0;
      const [pa, pc] = split(p, lo - lr);
      const c = pc + ca - cr + Math.max(0, (x?.[0] ?? 0) - (x?.[1] ?? 0) - w);
      all += a; rail += r; air += pa; car += c; walk += w;
      if (a > 0) {
        const at = lo > 0 ? marketXY(v[o + 2] / lo, v[o + 3] / lo) : x ? marketXY(x[3] / x[0], x[4] / x[0]) : centre(p) ?? marketXY(0, 0);
        places.push({ place: p, all: a, rail: r, air: pa, car: c, ...(w > 0 ? { walk: w } : {}), ...at });
      }
    }
    for (let z = 0; z < nz; z++) {
      const c = tail + 4 * P + 4 * z + 2 * dir;
      const e = cz.get(z);
      const a = (longer ? v[head + 4 * z + 2 * dir] + (v[c] ?? 0) : 0) + (e?.[0] ?? 0);
      if (a > 0) zones.push({ ...marketXY(v[xy + 3 * z], v[xy + 3 * z + 1]), place: v[xy + 3 * z + 2], gid: z, all: a, rail: (longer ? v[head + 4 * z + 2 * dir + 1] + (v[c + 1] ?? 0) : 0) + (e?.[1] ?? 0), ...(e && e[2] > 0 ? { walk: e[2] } : {}) });
    }
    return { all, rail, air, car, ...(walk > 0 ? { walk } : {}), elsewhere: longer ? v[v.length - 2 + dir] ?? 0 : 0, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
  };
  const every = CROSS_KINDS.map((_, i) => i);
  // the filter's far end for each shown purpose: its kinds' outings alone (school counts university)
  const byPurpose = new Map<PurposeKey, { starts: PlaceFlows; ends: PlaceFlows }>();
  for (const key of new Set(CROSS_KINDS.map((k) => shownKey(EVERYDAY_KEYS[k])))) {
    const ks = every.filter((i) => shownKey(EVERYDAY_KEYS[CROSS_KINDS[i]]) === key);
    const starts = flows(0, false, ks), ends = flows(1, false, ks);
    if (starts.all > 0 || ends.all > 0) byPurpose.set(key, { starts, ends });
  }
  return { starts: flows(0, true, every), ends: flows(1, true, every), matched: v[0], byPurpose };
}

/** A station's longer-trip rail riders (game/demand.ts `IntercityStation`, T-118) from each member
 * station's summed answer (null: none), each demand station once (T-146); null with none. */
export function stationFlows(answers: (Float64Array | null)[], ctx: SumsCtx): Pick<IntercityStation, "starts" | "ends"> | null {
  let v: Float64Array | null = null;
  const seen = new Set<string>();
  for (const a of answers) {
    if (!a) continue;
    const sig = signature(a);
    if (seen.has(sig)) continue;
    seen.add(sig);
    if (!v) v = a.slice();
    else addIntercity("icStation", v, a, ctx.places);
  }
  if (!v) return null;
  const marketXY = toWorld(ctx);
  const n = ctx.places;
  // rows: counts by place for starts then ends, then those times x and y (starts x, y, ends x, y);
  // then per far zone the riders starting, then ending; then zone x, y, market (not summed)
  const head = 6 * n, nz = (v.length - head) / 5, xy = intercityZoneXY("icStation", v.length, n);
  const flows = (dir: number): PlaceFlows => {
    const places: PlaceTrips[] = [];
    let rail = 0;
    for (let p = 0; p < n; p++) {
      const c = v![dir * n + p];
      rail += c;
      if (c > 0) places.push({ place: p, all: c, rail: c, air: 0, car: 0, ...marketXY(v![(2 + 2 * dir) * n + p] / c, v![(3 + 2 * dir) * n + p] / c) });
    }
    const zones: ZoneTrips[] = [];
    for (let z = 0; z < nz; z++) {
      const c = v![head + dir * nz + z];
      if (c > 0) zones.push({ ...marketXY(v![xy + 3 * z], v![xy + 3 * z + 1]), place: v![xy + 3 * z + 2], gid: z, all: c, rail: c });
    }
    return { all: rail, rail, air: 0, car: 0, elsewhere: 0, places: places.sort((a, b) => b.all - a.all), zones: zones.sort((a, b) => b.all - a.all) };
  };
  return { starts: flows(0), ends: flows(1) };
}

/** Each worker's `purposes` answer to a selection (`flowsAt`): its periods, the commutes' timing and
 * the everyday far end packed (demandProtocol `everyday`). */
export interface PurposePart { mask: number; work: Float32Array | null; everyday: Float32Array | null; names: string[] }

/** Where the commuters of a set of cells go (game/demand.ts `Flows`, T-098), less its version and
 * end, from the far end packed as `flows` answers (`p`) and every worker's `purposes` answer; null
 * where an answer is malformed or a period is missing. */
export function flowsBody(p: Float32Array, purposeParts: PurposePart[]): Omit<Flows, "version" | "end"> | null {
  purposeParts.sort((a, b) => a.mask - b.mask);
  const n = p[6];
  const at = (k: number) => p.subarray(7 + k * n, 7 + (k + 1) * n);
  const result: Omit<Flows, "version" | "end"> = {
    rail: 2*p[0],
    walk: 2*p[1],
    drive: 2*p[2],
    cutRail: 2*p[3],
    cutWalk: 2*p[4],
    cutDrive: 2*p[5],
    cells: Uint32Array.from(at(0)),
    farRail: Float32Array.from(at(1), v => 2*v),
    farWalk: Float32Array.from(at(2), v => 2*v),
    farDrive: Float32Array.from(at(3), v => 2*v),
    purposes: [],
  };
  // each purpose on its own for the map's filter (T-158): work is the commute answer as it is
  const { rail, walk, drive, cutRail, cutWalk, cutDrive, cells: farCells, farRail, farWalk, farDrive } = result;
  const byPurpose = new Map<PurposeKey, FarEnd>([["work", { rail, walk, drive, cutRail, cutWalk, cutDrive, cells: farCells, farRail, farWalk, farDrive }]]);
  result.byPurpose = byPurpose;
  let work: Float32Array | null = null;
  let purposeNames: string[] | null = null;
  // T-285: each worker's everyday answer is packed (per purpose kind 30 timing values, m, m far
  // cells ascending, their rail, walk and drive legs), only the far cells the selection reaches
  const everyday: Float32Array[] = [];
  let mask = 0;
  for (const reply of purposeParts) {
    if (!reply.work) return null;
    if (purposeNames && JSON.stringify(purposeNames) !== JSON.stringify(reply.names)) return null;
    purposeNames = reply.names;
    mask |= reply.mask;
    if (!work) work = reply.work.slice();
    else for (let k=0;k<work.length;k++) work[k] += reply.work[k];
    if (reply.everyday) everyday.push(reply.everyday);
  }
  if (mask !== (1 << PERIODS.length) - 1 || !work) return null;
  const purpose = (name: string, values: Float32Array) => ({name,
    outward: [0,1,2].map(m => Array.from(values.subarray(m*5,m*5+5))),
    returning: [0,1,2].map(m => Array.from(values.subarray(15+m*5,20+m*5))),
  });
  result.purposes.push(purpose("Work", work));
  if (everyday.length) {
    const kinds = purposeNames!.length;
    if (!kinds) return null;
    // per shown purpose (school counts university too, T-192): far-end legs by mode, and totals
    const merged = mergeEveryday(everyday, kinds, (kind) => (EVERYDAY_KEYS[kind] ? shownKey(EVERYDAY_KEYS[kind]) : null), result, [result.rail, result.walk, result.drive]);
    if (!merged) return null;
    merged.timing.forEach((t, kind) => result.purposes.push(purpose(purposeNames![kind], t)));
    for (const [key, { far, total }] of merged.shown)
      byPurpose.set(key as PurposeKey, { rail: total[0], walk: total[1], drive: total[2], cutRail: 0, cutWalk: 0, cutDrive: 0, ...far });
    Object.assign(result, merged.far);
    [result.rail, result.walk, result.drive] = merged.legs;
  }
  return result;
}

// ---------------------------------------------------------------- the views of each kind of day (T-1208)

/** The parts of a period's result (`FromDemand` "result") the day views read. */
export type DayResult = Pick<Extract<FromDemand, { kind: "result" }>, "summary" | "citySummary" | "cityParts" | "intercitySummary" | "intercityParts" | "corridorKey" | "corridorShares" | "corridorRail" | "elsewhere" | "seg" | "board" | "alight" | "arrive" | "loadOfCrush" | "fareX" | "xferIn" | "parts" | "fareSense"> & { corridorDay?: [number, number] };
/** A result's fields `DayResult` names, without the rest (the corridor table and modes are large and
 * read elsewhere). */
export function dayResultOf(r: DayResult): DayResult {
  const { summary, citySummary, cityParts, intercitySummary, intercityParts, corridorKey, corridorShares, corridorRail, elsewhere, seg, board, alight, arrive, loadOfCrush, fareX, xferIn, parts, fareSense } = r;
  return { summary, citySummary, cityParts, intercitySummary, intercityParts, corridorKey, corridorShares, corridorRail, elsewhere, seg, board, alight, arrive, loadOfCrush, fareX, xferIn, parts, fareSense };
}

/** What a solve's views are worked out from: its version, the round published and whether it is
 * the last, per line the first route node and stop count (`Solve.layout`), the period results, the
 * load factors of each kind of day (`day_factors()`, sim/src/demand/days.rs) and each period's
 * busiest hour, the zones every period covered (registry order), the market's place count, and
 * which days. `noCorridors`: the market's corridors left out. */
export interface DayInput {
  version: number;
  round: number;
  final: boolean;
  layout: { id: string; o: number; n: number; stops: string[] }[];
  res: DayResult[];
  dayTable: number[];
  peakHourFactor: number[];
  solved: CityId[];
  places: number;
  days: DemandDay[];
  noCorridors: boolean;
}

/** A kind of day's corridors less the table they index (`Corridors.table`, kept on the main thread
 * by its key); null for none (`NO_CORRIDORS`). */
export type CorridorRail = Omit<Corridors, "table"> & { key: string };
/** A kind of day's views as `DayDemand`, before the zones open bare join the cities and the
 * corridors their table. */
export interface DayViews { view: DemandView; cities: Map<CityId, DemandView>; intercity: Omit<IntercityDemand, "corridors"> & { corridors: CorridorRail | null } }

/** The purposes in `day_factors()`'s order (sim/src/demand/days.rs): work, the everyday layer's
 * kinds (`EVERYDAY_KEYS`), the market's longer trips. */
const PURPOSE_INDEX = Object.fromEntries((["work", ...EVERYDAY_KEYS, "longer"] as PurposeKey[]).map((k, i) => [k, i])) as Record<PurposeKey, number>;
const PURPOSE_INDEX_COUNT = 2 + EVERYDAY_KEYS.length;

/** A kind of day's load factor for purpose `key` in period `q` in `dayTable`. */
export function dayFactor(dayTable: ArrayLike<number>, day: DemandDay, key: string, q: number): number {
  const p = PURPOSE_INDEX[key as PurposeKey] ?? 0;
  return dayTable[(day * PURPOSE_INDEX_COUNT + p) * PERIODS.length + q] ?? 1;
}

/**
 * The lines' and stations' riders by trip purpose (T-158), from every period's `parts`, laid out
 * as the totals are. A purpose no period reports is left out. Keyed by the shown purposes: school
 * counts university too (T-192, `membersOf`).
 */
function purposeDemand(layout: DayInput["layout"], res: DayResult[]): Map<PurposeKey, PurposeDemand> {
  const out = new Map<PurposeKey, PurposeDemand>();
  for (const { key } of PURPOSES) {
    const members = membersOf(key);
    if (!res.some((r) => members.some((m) => r.parts?.keys.includes(m)))) continue;
    const lines: PurposeDemand["lines"] = new Map(), stations: PurposeDemand["stations"] = new Map();
    for (const { id, o, n, stops } of layout) {
      const boardings = stops.map(() => 0);
      const loads: Float32Array[] = [];
      res.forEach((r) => {
        const seg = new Float32Array(2 * (n - 1));
        loads.push(seg);
        const p = r.parts;
        if (!p) return;
        for (const m of members) {
          const at = p.keys.indexOf(m);
          if (at < 0) continue;
          const nr = p.seg.length / p.keys.length, base = at * nr;
          for (let k = 0; k < n; k++) {
            const a = base + o + k, b = base + o + 2 * n - 1 - k;
            boardings[k] += p.board[a] + p.board[b];
            let sd = stations.get(stops[k]);
            if (!sd) stations.set(stops[k], (sd = { boardings: 0, alightings: 0 }));
            sd.boardings += p.board[a] + p.board[b];
            sd.alightings += p.alight[a] + p.alight[b];
          }
          for (let k = 0; k < n - 1; k++) {
            seg[k] += p.seg[base + o + k];
            seg[n - 1 + k] += p.seg[base + o + n + k];
          }
        }
      });
      lines.set(id, { ridersPerDay: boardings.reduce((x, y) => x + y, 0), boardings, loads });
    }
    out.set(key, { lines, stations });
  }
  return out;
}

/**
 * One period's result as it is on another kind of day (T-260): each purpose's riders times its
 * load factor, the totals their sum, the mode split from each city's purposes the same way, the
 * market's trips and corridors by its own factor. The busiest-hour load of crush follows the
 * segment's riders. Route and mode choices are the weekday's (docs/demand.md 4.2).
 */
function dayResult(dayTable: ArrayLike<number>, r: DayResult, q: number, day: DemandDay): DayResult {
  const p = r.parts;
  if (!p || !dayTable.length) return r;
  const factor = (key: string) => dayFactor(dayTable, day, key, q);
  const nr = r.seg.length;
  const seg = new Float32Array(nr), board = new Float32Array(nr), alight = new Float32Array(nr);
  const ps = new Float32Array(p.seg.length), pb = new Float32Array(p.board.length), pa = new Float32Array(p.alight.length);
  // T-1373: a rush day's longer trips lean out of big places going out and back coming back
  const sign = day === OUT_DAY ? 1 : day === BACK_DAY ? -1 : 0;
  const lean = sign && p.lean?.length === 3 * nr ? p.lean : null;
  p.keys.forEach((key, i) => {
    const f = factor(key);
    const l = key === "longer" ? lean : null;
    for (let j = 0, o = i * nr; j < nr; j++, o++) {
      if (l) {
        ps[o] = Math.max(0, p.seg[o] + sign * l[j]) * f; pb[o] = Math.max(0, p.board[o] + sign * l[nr + j]) * f; pa[o] = Math.max(0, p.alight[o] + sign * l[2 * nr + j]) * f;
      } else {
        ps[o] = p.seg[o] * f; pb[o] = p.board[o] * f; pa[o] = p.alight[o] * f;
      }
      seg[j] += ps[o]; board[j] += pb[o]; alight[j] += pa[o];
    }
  });
  const loadOfCrush = r.loadOfCrush.map((v, j) => (r.seg[j] > 0 ? (v * seg[j]) / r.seg[j] : 0));
  // the fare curve's extra follows the riders on the stretch where it is counted (T-1516)
  const fareX = r.fareX?.map((v, j) => (r.seg[j] > 0 ? (v * seg[j]) / r.seg[j] : 0));
  // T-1255: the riders coming to a platform follow its boardings
  const arrive = r.arrive?.map((v, j) => (r.board[j] > 0 ? (v * board[j]) / r.board[j] : v));  const citySummary: Record<CityId, number[]> = {};
  for (const [id, s0] of Object.entries(r.citySummary)) {
    const s = s0.slice(), cp = r.cityParts?.[id];
    if (cp) {
      s[1] = s[2] = s[3] = 0;
      for (let i = 0; i * 3 < cp.length; i++) {
        const f = factor(p.keys[i]);
        s[1] += cp[3 * i] * f; s[2] += cp[3 * i + 1] * f; s[3] += cp[3 * i + 2] * f;
      }
    }
    citySummary[id] = s;
  }
  const fl = factor("longer"), fw = factor("work");
  const intercitySummary = r.intercitySummary.slice();
  const ip = r.intercityParts;
  // T-276: longer trips by their factor, commutes between zones by work's
  if (ip && ip.length >= 4) { intercitySummary[1] = ip[0] * fl + ip[2] * fw; intercitySummary[2] = ip[1] * fl + ip[3] * fw; }
  else { intercitySummary[1] *= fl; intercitySummary[2] *= fl; }
  // T-1175, T-1229: everyday outings across a zone line, each purpose's by its own factor
  const intercityParts = ip && marketPartsOn(ip, fl, fw, factor);
  if (intercityParts && intercityParts.length >= 7) { intercitySummary[1] += intercityParts[4]; intercitySummary[2] += intercityParts[5]; intercitySummary[3] = intercityParts[6]; }
  const elsewhere = r.elsewhere?.map((v) => v * fl);
  const summary = r.summary.slice();
  summary[1] = summary[2] = summary[3] = 0;
  for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3]) summary[k] += v[k] ?? 0;
  // T-1300: the fare sensitivity follows the rail trips, local and the market's
  const railOf = (s: Record<CityId, number[]>) => Object.values(s).reduce((a, v) => a + (v[2] ?? 0), 0);
  const by = (day: number, raw: number) => (raw > 0 ? day / raw : 1);
  const fareSense = r.fareSense && [r.fareSense[0] * by(railOf(citySummary), railOf(r.citySummary)), r.fareSense[1] * by(intercitySummary[2] ?? 0, r.intercitySummary[2] ?? 0), r.fareSense[2], r.fareSense[3]] as [number, number, number, number];
  // the corridors keep the weekday's numbers and their factors, applied as `dayCorridors` adds them (T-789)
  return { ...r, fareSense, seg, board, alight, arrive, loadOfCrush, fareX, citySummary, intercitySummary, intercityParts, corridorDay: [fl, fw], elsewhere, summary, parts: { keys: p.keys, seg: ps, board: pb, alight: pa } };
}

/**
 * T-1008: a kind of day's corridors from its period results, less the market's table (sent once,
 * kept by its key): the day's scales, the periods' shares times their day factors, and the pairs
 * with rail added up over the periods. Work in the pairs with rail only.
 */
function dayCorridors(res: DayResult[]): CorridorRail | null {
  const key = res.find((r) => r.corridorKey)?.corridorKey;
  if (!key) return null;
  let longerScale = 0, commuteScale = 0;
  const rail = new Map<number, [number, number]>();
  for (const r of res) {
    // every period of one solve has the same market; a period of another would index other rows
    if (r.corridorKey !== key) {
      if (r.corridorKey) console.error("demand: periods with different corridor tables", key, r.corridorKey);
      continue;
    }
    const [fl, fw] = r.corridorDay ?? [1, 1];
    if (r.corridorShares) { longerScale += r.corridorShares[0] * fl; commuteScale += r.corridorShares[1] * fw; }
    const v = r.corridorRail;
    for (let i = 0; v && i < v.length; i += 3) {
      const j = v[i], a = rail.get(j);
      if (a) { a[0] += v[i + 1] * fl; a[1] += v[i + 2] * fw; } else rail.set(j, [v[i + 1] * fl, v[i + 2] * fw]);
    }
  }
  const railAt = Uint32Array.from(rail.keys()).sort();
  const longerRail = new Float64Array(railAt.length), commuteRail = new Float64Array(railAt.length);
  railAt.forEach((j, k) => { const [l, c] = rail.get(j)!; longerRail[k] = l; commuteRail[k] = c; });
  return { key, longerScale, commuteScale, railAt, longerRail, commuteRail };
}

/** The views of one kind of day from its period results. */
function dayViewsOf(input: DayInput, day: DemandDay, res: DayResult[]): DayViews {
  let trips = 0, rail = 0, walk = 0;
  const byPeriod: [string, number][] = [];
  res.forEach((r, q) => {
    const [, t, ra, wa] = r.summary;
    trips += t;
    rail += ra;
    walk += wa;
    byPeriod.push([PERIODS[q].name, t > 0 ? (100 * ra) / t : 0]);
  });
  const lines = new Map<string, LineDemand>();
  const stations = new Map<string, StationDemand>();
  const st = (id: string) => {
    let v = stations.get(id);
    if (!v) stations.set(id, (v = { boardings: 0, alightings: 0 }));
    return v;
  };
  // T-1530: the day's block of each period's transfers in (`xferIn`, every kind of day's)
  const withXfer = res.length > 0 && res.every((r) => r.xferIn?.length === DEMAND_DAYS.length * r.board.length);
  for (const { id, o, n, stops } of input.layout) {
    const boardings = stops.map(() => 0);
    let fullest = 0;
    const loads: Float32Array[] = [], board: Float32Array[] = [], alight: Float32Array[] = [], arrive: Float32Array[] = [];
    const fareX: number[] = [];
    res.forEach((r, q) => {
      let fx = 0;
      if (r.fareX) for (let k = o; k < o + 2 * n; k++) fx += r.fareX[k];
      fareX.push(fx);
      for (let k = 0; k < n; k++) {
        // stop k is route node o + k on run 0 and o + n + (n - 1 - k) on run 1
        const a = o + k, b = o + 2 * n - 1 - k;
        boardings[k] += r.board[a] + r.board[b];
        const sd = st(stops[k]);
        sd.boardings += r.board[a] + r.board[b];
        sd.alightings += r.alight[a] + r.alight[b];
        if (withXfer) { const x = day * r.board.length; sd.transfers = (sd.transfers ?? 0) + r.xferIn![x + a] + r.xferIn![x + b]; }
      }
      const seg = new Float32Array(2 * (n - 1));
      for (let k = 0; k < n - 1; k++) {
        seg[k] = r.seg[o + k];
        seg[n - 1 + k] = r.seg[o + n + k];
      }
      loads.push(seg);
      // the boarding account's input (T-164): boardings and alightings at each route node
      board.push(r.board.slice(o, o + 2 * n));
      alight.push(r.alight.slice(o, o + 2 * n));
      arrive.push((r.arrive ?? r.board).slice(o, o + 2 * n));
      if (PERIODS[q].demand === "high")
        for (let k = 0; k < 2 * n; k++) fullest = Math.max(fullest, r.loadOfCrush[o + k]);
    });
    lines.set(id, { ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: Math.round(100 * fullest), boardings, loads, board, alight, arrive, fareX });
  }
  const pct = (v: number) => (trips > 0 ? (100 * v) / trips : 0);
  const network: DemandView = {
    version: input.version,
    day,
    round: input.round,
    final: input.final,
    split: { train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk)) },
    trainShareByPeriod: byPeriod,
    peakHourFactor: input.peakHourFactor,
    tripsPerDay: trips,
    railTripsPerDay: rail,
    lines,
    stations,
    purposes: purposeDemand(input.layout, res),
  };
  // only the cities this solve covered (T-149: a city not loaded has no view, and its controls wait)
  const cities = new Map<CityId, DemandView>(input.solved.map((id) => {
    let trips = 0, rail = 0, walk = 0;
    const trainShareByPeriod: [string, number][] = res.map((r, q) => {
      const [, t, ra, wa] = r.citySummary[id]; trips += t; rail += ra; walk += wa;
      return [PERIODS[q].name, t > 0 ? 100 * ra / t : 0];
    });
    const pct = (n: number) => trips > 0 ? 100 * n / trips : 0;
    return [id, {...network, tripsPerDay: trips, railTripsPerDay: rail, trainShareByPeriod, split: {train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk))}}];
  }));
  // the regional layer between places: longer trips and (T-276) commutes between zones together
  let interTrips = 0, interRail = 0, commuteTrips = 0, commuteRail = 0;
  const elsewhere = Array.from({ length: input.places }, () => 0);
  for (const r of res) {
    interTrips += r.intercitySummary[1]; interRail += r.intercitySummary[2];
    if (r.intercityParts && r.intercityParts.length >= 4) { commuteTrips += r.intercityParts[2]; commuteRail += r.intercityParts[3]; }
    r.elsewhere?.forEach((v, p) => { if (p < elsewhere.length) elsewhere[p] += v; });
  }
  // T-1300: the day's fare sensitivity, with the rail trips it is of; packed with the view's skeleton
  if (res.every((r) => r.fareSense)) {
    const sum = (i: number) => res.reduce((a, r) => a + r.fareSense![i], 0);
    const reach = (i: number, j: number) => { const d = sum(i); return d < 0 ? res.reduce((a, r) => a + r.fareSense![i] * (r.fareSense![j] ?? 1), 0) / d : 1; };
    network.fareSense = { local: sum(0), market: sum(1), localReach: reach(0, 2), marketReach: reach(1, 3), marketRail: interRail };
  }
  return { view: network, cities, intercity: { version: input.version, round: input.round, tripsPerDay: interTrips, railTripsPerDay: interRail, commuteTripsPerDay: commuteTrips, commuteRailPerDay: commuteRail, corridors: input.noCorridors ? null : dayCorridors(res), elsewhere } };
}

/** T-260: each asked kind of day's views: the weekday is the solve itself; a weekend is worked out
 * from it, purpose by purpose. */
export function dayViews(input: DayInput): DayViews[] {
  return input.days.map((day) => dayViewsOf(input, day, day === 0 ? input.res : input.res.map((r, q) => dayResult(input.dayTable, r, q, day))));
}

// ---------------------------------------------------------------- the day views packed (T-1208)

/** `DayViews` in a few flat arrays, which a message carries at the cost of a copy: a structured clone
 * of the views' tens of thousands of maps, objects and small arrays took longer to read on the main
 * thread than adding them up there. Every number keeps its exact value (doubles in `f64`, the views'
 * Float32Arrays in `f32`, unpacked as views into it: never transfer their buffers); `idx` numbers
 * the stations of each station map in its order, by `stationIds`. */
export interface PackedDays {
  f32: Float32Array;
  f64: Float64Array;
  idx: Int32Array;
  stationIds: string[];
  days: {
    view: Omit<DemandView, "lines" | "stations" | "purposes">;
    stations: number;
    /** each purpose's key and its station count, in the map's order */
    purposes: [PurposeKey, number][];
    cities: [CityId, Pick<DemandView, "tripsPerDay" | "railTripsPerDay" | "trainShareByPeriod" | "split">][];
    intercity: DayViews["intercity"];
  }[];
}

/** `views` packed; `layout` and the period count `periods` give every array's length. */
export function packDays(views: DayViews[], layout: DayInput["layout"], periods: number): PackedDays {
  const f32: number[] = [], f64: number[] = [], idx: number[] = [], stationIds: string[] = [];
  const index = new Map<string, number>();
  const put32 = (a: Float32Array) => { for (let i = 0; i < a.length; i++) f32.push(a[i]); };
  const put64 = (a: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) f64.push(a[i]); };
  const stationsOf = (m: Map<string, StationDemand>) => {
    for (const [id, v] of m) {
      let k = index.get(id);
      if (k === undefined) index.set(id, (k = stationIds.length)), stationIds.push(id);
      idx.push(k);
      f64.push(v.boardings, v.alightings, v.transfers ?? -1);
    }
    return m.size;
  };
  const days = views.map((d) => {
    const { lines, stations, purposes, ...view } = d.view;
    for (const { id } of layout) {
      const l = lines.get(id)!;
      f64.push(l.ridersPerDay, l.fullestPct);
      put64(l.boardings);
      put64(l.fareX!);
      for (let q = 0; q < periods; q++) { put32(l.loads[q]); put32(l.board![q]); put32(l.alight![q]); put32(l.arrive![q]); }
    }
    const nStations = stationsOf(stations);
    const ps: [PurposeKey, number][] = [];
    for (const [key, p] of purposes ?? []) {
      for (const { id } of layout) {
        const l = p.lines.get(id)!;
        f64.push(l.ridersPerDay);
        put64(l.boardings);
        for (let q = 0; q < periods; q++) put32(l.loads[q]);
      }
      ps.push([key, stationsOf(p.stations)]);
    }
    const cities = [...d.cities].map(([id, c]): PackedDays["days"][number]["cities"][number] => [id, { tripsPerDay: c.tripsPerDay, railTripsPerDay: c.railTripsPerDay, trainShareByPeriod: c.trainShareByPeriod, split: c.split }]);
    return { view, stations: nStations, purposes: ps, cities, intercity: d.intercity };
  });
  return { f32: Float32Array.from(f32), f64: Float64Array.from(f64), idx: Int32Array.from(idx), stationIds, days };
}

/** `packDays`' views as they were. */
export function unpackDays(p: PackedDays, layout: DayInput["layout"], periods: number): DayViews[] {
  let a = 0, b = 0, c = 0;
  const { f32, f64, idx, stationIds } = p;
  // views into the one buffer: a copy each cost more than the rest of the unpacking
  const take32 = (n: number) => f32.subarray(a, (a += n));
  const take64 = (n: number) => { const out: number[] = new Array(n); for (let i = 0; i < n; i++) out[i] = f64[b++]; return out; };
  const stationsOf = (n: number) => {
    const m = new Map<string, StationDemand>();
    for (let i = 0; i < n; i++) {
      const t = f64[b + 2];
      m.set(stationIds[idx[c++]], t >= 0 ? { boardings: f64[b], alightings: f64[b + 1], transfers: t } : { boardings: f64[b], alightings: f64[b + 1] });
      b += 3;
    }
    return m;
  };
  return p.days.map((d) => {
    const lines = new Map<string, LineDemand>();
    for (const { id, n } of layout) {
      const ridersPerDay = f64[b++], fullestPct = f64[b++];
      const boardings = take64(n), fareX = take64(periods);
      const loads: Float32Array[] = [], board: Float32Array[] = [], alight: Float32Array[] = [], arrive: Float32Array[] = [];
      for (let q = 0; q < periods; q++) { loads.push(take32(2 * (n - 1))); board.push(take32(2 * n)); alight.push(take32(2 * n)); arrive.push(take32(2 * n)); }
      lines.set(id, { ridersPerDay, fullestPct, boardings, loads, board, alight, arrive, fareX });
    }
    const stations = stationsOf(d.stations);
    const purposes = new Map<PurposeKey, PurposeDemand>();
    for (const [key, nStations] of d.purposes) {
      const pl: PurposeDemand["lines"] = new Map();
      for (const { id, n } of layout) {
        const ridersPerDay = f64[b++];
        const boardings = take64(n);
        const loads: Float32Array[] = [];
        for (let q = 0; q < periods; q++) loads.push(take32(2 * (n - 1)));
        pl.set(id, { ridersPerDay, boardings, loads });
      }
      purposes.set(key, { lines: pl, stations: stationsOf(nStations) });
    }
    const view: DemandView = { ...d.view, lines, stations, purposes };
    // the network's alone, as `dayViewsOf` gives it
    const { fareSense: _, ...base } = view;
    const cities = new Map<CityId, DemandView>(d.cities.map(([id, x]) => [id, { ...base, tripsPerDay: x.tripsPerDay, railTripsPerDay: x.railTripsPerDay, trainShareByPeriod: x.trainShareByPeriod, split: x.split }]));
    return { view, cities, intercity: d.intercity };
  });
}

// ---------------------------------------------------------------- the sums worker's messages

export type ToSums =
  /** a bubble selection's longer trips: every worker's answer, in period order */
  | { kind: "icSel"; id: number; parts: Float64Array[]; ctx: SumsCtx }
  /** a station's or complex's: per member station, every worker's answer */
  | { kind: "icStation"; id: number; members: Float64Array[][]; ctx: SumsCtx }
  /** a bubble selection's commuters and everyday far end (`flowsBody`), dealt for the display under
   * every filter with the city's cells (`cells.key`; their `xy` the first time) */
  | { kind: "flows"; id: number; packed: Float32Array; parts: PurposePart[]; cells: { key: string; xy?: Float32Array } | null }
  /** a solve's views of each kind of day */
  | { kind: "days"; id: number; input: DayInput }
  /** T-1435: a zone's drive arcs at every level (game/drives.ts `driveLevels`) */
  | { kind: "drives"; id: number; input: DriveInput };
export type SumsAsk = ToSums extends infer M ? (M extends unknown ? Omit<M, "id"> : never) : never;
export interface FromSums { id: number; result: unknown; error?: string }
/** A "flows" answer: the far end, and per filter its far end dealt (`dealtCells`; null for none) */
export interface FlowsAnswer { body: Omit<Flows, "version" | "end">; dealt: [PurposeFilter, DealtCells | null][] | null }
