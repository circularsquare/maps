// T-3740: fare zones. The player gives stations a fare area and a zone number (the track model keeps each
// station's code, `StationData::fare`); the fares keep the areas and their prices. T-3758: a ride on a line set to
// zone fares (the default; `LineData::fare_dist` off) between two stations of one area is a zone ride; zone rides
// one after another in one area pay one price, for the zones from the lowest to the highest of that area's stations
// they stop at or pass, and the other rides pay the distance curve on their km, the base once (notes/T-3740.md,
// DECISIONS 2026-11-19).

import { FARE_MAX, fareAt, type FareArea, type FareNode, type Fares } from "./fareCurve";

/** Zones an area can have, so prices for 1 to this many zones. */
export const MAX_ZONES = 12;
/** A new area's prices: the curve's fares at this many km times the zone count (5, 10, 15 km ...). */
export const START_KM_PER_ZONE = 5;

/** A station's code for zone `zone` of area `id`; 0 is none. */
export const zoneCode = (id: number, zone: number): number => (id << 8) | zone;
export const areaOfCode = (code: number): number => code >>> 8;
export const zoneOfCode = (code: number): number => code & 255;

const price = (v: number) => (Number.isFinite(v) ? Math.min(FARE_MAX, Math.max(0, Math.round(v * 1e4) / 1e4)) : 0);

/** Prices as kept: 1 to `MAX_ZONES`, each held to 0-`FARE_MAX` and to a hundredth of a cent, never falling. */
export function tidyPrices(prices: readonly number[]): number[] {
  const out: number[] = [];
  for (const v of prices.slice(0, MAX_ZONES)) out.push(Math.max(price(v), out[out.length - 1] ?? 0));
  return out.length ? out : [0];
}

/** Areas as kept: whole positive ids, each once, a name, tidy prices; anything else dropped. */
export function tidyAreas(list: unknown): FareArea[] {
  if (!Array.isArray(list)) return [];
  const out: FareArea[] = [], seen = new Set<number>();
  for (const a of list) {
    const id = a?.id;
    if (!Number.isInteger(id) || id < 1 || id >= 1 << 23 || seen.has(id) || !Array.isArray(a.prices)) continue;
    seen.add(id);
    out.push({ id, name: typeof a.name === "string" ? a.name.slice(0, 80) : "", prices: tidyPrices(a.prices.filter((v: unknown) => typeof v === "number")), ...(a.off === true ? { off: true } : {}) });
  }
  return out;
}

/** `f`'s areas and next id as a save or an edit gives them, tidied; nothing when there are none. */
export function zonesOf(f: { zones?: unknown; zoneNext?: unknown }): Pick<Fares, "zones" | "zoneNext"> {
  const zones = tidyAreas(f.zones);
  const most = zones.reduce((m, a) => Math.max(m, a.id + 1), 1);
  const next = Math.max(most, Number.isInteger(f.zoneNext) ? (f.zoneNext as number) : 1);
  return { ...(zones.length ? { zones } : {}), ...(next > 1 ? { zoneNext: next } : {}) };
}

/** A new area's prices on curve `nodes`: its fares at 5, 10, 15 km and so on, so painting zones keeps fares near. */
export function startPrices(nodes: FareNode[], zones = MAX_ZONES): number[] {
  return tidyPrices(Array.from({ length: zones }, (_, k) => Math.round(fareAt(nodes, START_KM_PER_ZONE * (k + 1)) * 100) / 100));
}

/** An area's prices as charged: never below the base fare, held as they are read, as the transfer fare is (a base
 * raised later lifts them). Curve units. */
export function chargedPrices(f: { nodes?: FareNode[] }, a: FareArea): number[] {
  const base = f.nodes?.[0]?.[1] ?? 0;
  return a.prices.map((v) => Math.max(v, base));
}

/** What a demand population is given (`set_fare_zones`), all empty for no zones. */
export interface DemandZones { st: Uint32Array; off: Uint32Array; usd: Float32Array; dist: Uint8Array; passOff: Uint32Array; pass: Uint32Array }

/**
 * What a demand population is given (`set_fare_zones`): per station of `codes` (the stations' codes) its code with
 * the area's place among `f.zones` in place of its id (0 for none or a gone area), and the areas' prices one after
 * another times `scale` (its country's, `countryFareScales` rate; 1 for the market) with their offsets. T-3758: per
 * route node `lines.dist` (1 where its line pays the distance fare) and the codes of the stations passed
 * (`lines.pass[lines.passOff[r]..]`), those codes as the stations' and only the ones of an area kept.
 */
export function demandZones(f: { nodes?: FareNode[]; zones?: FareArea[] }, codes: Uint32Array | undefined, scale: number, lines: { dist?: Uint8Array; passOff?: Uint32Array; pass?: Uint32Array } = {}): DemandZones {
  const areas = f.zones ?? [];
  const none = { st: new Uint32Array(0), off: new Uint32Array(0), usd: new Float32Array(0), dist: new Uint8Array(0), passOff: new Uint32Array(0), pass: new Uint32Array(0) };
  if (!codes || !areas.length) return none;
  const at = new Map(areas.map((a, i) => [a.id, i + 1]));
  const recode = (c: number) => { const k = at.get(areaOfCode(c)); return k && zoneOfCode(c) ? zoneCode(k, zoneOfCode(c)) : 0; };
  const st = Uint32Array.from(codes, recode);
  if (!st.some((c) => c)) return none;
  const off = [0], usd: number[] = [];
  for (const a of areas) { for (const v of chargedPrices(f, a)) usd.push(v * scale); off.push(usd.length); }
  let passOff = new Uint32Array(0), pass = new Uint32Array(0);
  if (lines.pass?.length && lines.passOff) {
    const o = new Uint32Array(lines.passOff.length), kept: number[] = [];
    for (let r = 0; r + 1 < lines.passOff.length; r++) {
      for (let i = lines.passOff[r]; i < lines.passOff[r + 1]; i++) { const c = recode(lines.pass[i]); if (c) kept.push(c); }
      o[r + 1] = kept.length;
    }
    if (kept.length) { passOff = o; pass = Uint32Array.from(kept); }
  }
  return { st, off: Uint32Array.from(off), usd: Float32Array.from(usd), dist: lines.dist ?? new Uint8Array(0), passOff, pass };
}
