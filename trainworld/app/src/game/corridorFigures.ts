// T-2136: what the views show of the market's corridors, from the market's book and the rows the
// workers send for a place or two (game/demand.ts `rowsOf`). Each returns null while its rows are
// asked; a computed or component that calls it works out again when they land.

import { CITIES } from "./cities";
import { shareOf } from "./corridorShares";
import { askEverydayPartners, eachCorridor, INTERCITY_PLACES, rowsLanded, rowsOf, type CorridorRows, type Corridors } from "./demand";

const PLACES = INTERCITY_PLACES.length;
/** T-789: the city view draws zones only, so it keeps the places that hold a zone (`Z` of them),
 * not every place of the market. */
const ZONE_PLACES = [...new Set(CITIES.map((c) => c.place).filter((p) => p >= 0 && p < PLACES))];
export const Z = ZONE_PLACES.length;
const zoneIndex = new Int32Array(PLACES).fill(-1);
ZONE_PLACES.forEach((p, i) => (zoneIndex[p] = i));
/** a place's index among the zone places, -1 for none */
export const zoneOf = (place: number) => (place >= 0 && place < PLACES ? zoneIndex[place] : -1);
/** each corridor endpoint that is a zone place (`zoneOf`), -1 for none; a corridor within one
 * place or with a place past the list counts nowhere */
const ends = (from: number, to: number): [number, number] =>
  from === to || from >= PLACES || to >= PLACES ? [-1, -1] : [zoneOf(from), zoneOf(to)];

/** The market's trips a day for the city view: each zone place's with every other place of the
 * market (`totals`, [trips, rail, air] at `zoneOf` index * 3), all modes, by train and (T-378) by
 * plane; and a zone place's with each other zone place (`zoneRow`). T-3503: `totals` take each zone
 * place's everyday outings across its lines too (`everyday`), and `walk` their walking. */
export interface MarketPairs {
  c: Corridors; totals: Float64Array; walk: Float64Array;
  /** each zone place's everyday outings across its lines, [legs, rail, walking] at `zoneOf` index * 3 (what `totals` and `walk` took) */
  everyday: Float64Array;
  rows: Map<number, Float64Array>;
  /** a selected zone place's everyday outings by partner place (`askEverydayPartners`), `undefined` once asked, null if unanswered */
  partners: Map<number, Float64Array | null | undefined>;
}
/** the width of a `zoneRowOf` row's entry for a zone place: trips, rail, air, walking */
export const ROW = 4;
export function marketPairs(c: Corridors, everyday: number[] = []): MarketPairs {
  const b = c.book.totals, totals = new Float64Array(Z * 3), walk = new Float64Array(Z), eve = new Float64Array(Z * 3);
  for (let i = 0; i + 4 < everyday.length; i += 5) {
    const z = zoneOf(everyday[i]);
    if (z < 0) continue;
    totals[3 * z] += everyday[i + 1];
    totals[3 * z + 1] += everyday[i + 2];
    walk[z] += everyday[i + 3];
    eve[3 * z] += everyday[i + 1];
    eve[3 * z + 1] += everyday[i + 2];
    eve[3 * z + 2] += everyday[i + 3];
  }
  if (b.length) for (let z = 0; z < Z; z++) {
    const o = 3 * ZONE_PLACES[z];
    totals[3 * z] += b[o] * c.longerScale + b[o + 1] * c.commuteScale;
    totals[3 * z + 2] = b[o + 2] * c.longerScale;
  }
  // the pairs with rail: their rail, and their longer trips by rail no longer by plane
  for (let k = 0; k < c.railAt.length; k++) {
    const r = c.longerRail[k] + c.commuteRail[k], f = c.longerRail[k] * shareOf(c.railAir[k]);
    for (const z of ends(c.railFrom[k], c.railTo[k])) if (z >= 0) { totals[3 * z + 1] += r; totals[3 * z + 2] -= f; }
  }
  return { c, totals, walk, everyday: eve, rows: new Map(), partners: new Map() };
}
/** the last answer for each zone place's everyday partners, shown while a newer result's is asked */
const lastPartners = new Map<number, Float64Array>();
/**
 * Zone place `za`'s trips with each other zone place, [trips, rail, air, walking] at `zoneOf` index * ROW: the longer
 * trips and commutes between the zones, and (T-4610) the everyday outings across their lines, which the market holds
 * only as one total a place and the workers give by partner (asked with the rows, once a selection). The partners'
 * outings are scaled to the place's `everyday` total, so the kind of day and the period weights are the totals'.
 * Null while the rows, or the first partners, are asked.
 */
export function zoneRowOf(p: MarketPairs, za: number): Float64Array | null {
  const kept = p.rows.get(za);
  if (kept) return kept;
  const r = rowsOf(p.c, ZONE_PLACES[za]);
  if (!r) return null;
  if (!p.partners.has(za)) {
    p.partners.set(za, undefined);
    void askEverydayPartners(ZONE_PLACES[za]).then((v) => {
      p.partners.set(za, v);
      if (v) lastPartners.set(za, v);
      rowsLanded.value++;
    });
  }
  const asked = p.partners.get(za), partners = asked === undefined ? lastPartners.get(za) : asked;
  if (asked === undefined && !partners) return null;
  const row = new Float64Array(Z * ROW);
  eachCorridor(p.c, r, (from, to) => { const [a, b] = ends(from, to); return a >= 0 && b >= 0 && (a === za || b === za); }, (from, to, v) => {
    const [a, b] = ends(from, to), o = ROW * (a === za ? b : a);
    row[o] += v.trips; row[o + 1] += v.rail; row[o + 2] += v.air;
  });
  if (partners) {
    let all = 0, rail = 0, walk = 0;
    for (let i = 0; i + 3 < partners.length; i += 4) { all += partners[i + 1]; rail += partners[i + 2]; walk += partners[i + 3]; }
    const [fa, fr, fw] = [all, rail, walk].map((s, k) => (s > 0 ? p.everyday[3 * za + k] / s : 0));
    for (let i = 0; i + 3 < partners.length; i += 4) {
      const z = zoneOf(partners[i]);
      if (z < 0 || z === za) continue;
      row[ROW * z] += partners[i + 1] * fa;
      row[ROW * z + 1] += partners[i + 2] * fr;
      row[ROW * z + 3] += partners[i + 3] * fw;
    }
  }
  // (the last answer stands in while a newer one is asked, and is not kept)
  if (asked !== undefined) p.rows.set(za, row);
  return row;
}
/** `zoneRowOf` for a place, from the corridors alone (checks). */
export const zoneRow = (c: Corridors, place: number): Float64Array | null => zoneRowOf(marketPairs(c), zoneOf(place));
/** `marketPairs`, for checks (`waitFor` takes it as it is). */
export const marketTotals = (c: Corridors, everyday?: number[]): MarketPairs => marketPairs(c, everyday);

export interface PlaceTrips { all: number; rail: number; air: number; none: number }
/** The city view's To other places: `place`'s trips with each other place, both ways, biggest first. */
export function placeFigures(c: Corridors, place: number): [number, PlaceTrips][] | null {
  const r = rowsOf(c, place);
  if (!r) return null;
  const byPlace = new Map<number, PlaceTrips>();
  eachCorridor(c, r, (from, to) => from !== to && (from === place || to === place), (from, to, v) => {
    const other = from === place ? to : from;
    const p = byPlace.get(other) ?? { all: 0, rail: 0, air: 0, none: 0 };
    p.all += v.trips;
    p.rail += v.rail;
    p.air += v.air;
    p.none += v.none;
    byPlace.set(other, p);
  });
  return [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
}

export interface PairTrips { trips: number; rail: number; air: number; car: number; none: number }
/** Two places' longer trips a day both ways as the solve split them (the corridor's, commutes
 * between the zones left out); null with none, undefined while the rows are asked. */
export function pairFigures(c: Corridors, a: number, b: number): PairTrips | null | undefined {
  if (a === b) return null;
  const r = rowsOf(c, a);
  if (!r) return undefined;
  let out: PairTrips | null = null;
  eachCorridor(c, r, (x, y) => (x === a && y === b) || (x === b && y === a), (_, __, f) => {
    const trips = f.trips - f.commutes, rail = f.rail - f.commuteRail;
    if (trips > 0) out = { trips, rail, air: f.air, car: Math.max(0, f.car - (f.commutes - f.commuteRail - f.commuteNone)), none: f.none - f.commuteNone };
  });
  return out;
}

/** T-378: the shares of the longer trips not by rail that fly and that drive on each corridor with
 * place `own`, 2 per place (`SumsCtx.modes`): NaN where the table has no row (a corridor with no
 * trips, whose bubbles carry none either, T-2122) or the shares are NaN (near places on two
 * landmasses). */
export function ownModesFrom(r: CorridorRows, own: number): Float64Array {
  const out = new Float64Array(2 * PLACES).fill(NaN);
  for (let i = 0; i < r.n; i++) {
    const p = r.from[i] === own ? r.to[i] : r.to[i] === own ? r.from[i] : -1;
    if (p >= 0 && p < PLACES) (out[2 * p] = shareOf(r.air[i])), (out[2 * p + 1] = shareOf(r.car[i]));
  }
  return out;
}
export const ownModesOf = (c: Corridors, own: number): Float64Array | null => { const r = rowsOf(c, own); return r && ownModesFrom(r, own); };

/** For checks: `f()` once it is no longer null (or undefined, with `nullDone`), asking again as rows land. */
export async function waitFor<T>(f: () => T | null | undefined, nullDone = false, ms = 30000): Promise<T> {
  const end = performance.now() + ms;
  for (;;) {
    const v = f();
    if (v !== undefined && (v !== null || nullDone)) return v as T;
    if (performance.now() > end) throw new Error("rows did not come");
    await new Promise((r) => setTimeout(r, 20));
  }
}
