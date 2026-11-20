// The clock worker's fares: a new game's, a save's, a line's and a station's (T-2374: moved out of clock.worker.ts).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { CITIES, cityById, cityWithin, HOME_CITY, isCity } from "../game/cities";
import { startNodes } from "../game/fareDefault";
import { areaOfCode, zonesOf } from "../game/fareZones";
import { capTransfer, faresOf, simpleNodes, START_FARE, START_NODES, tidyHsr, tidyLineFare, tidyTransfer, type Fares } from "../game/fareCurve";
import { EVERY, FARE_BASELINE, FARE_E, incomeScale, keepOwn, OLD_FARE_E, oldCurveFactor, oldIncomeScale, scaleFromOld, START, startFromOld, tidyOwn } from "../game/income";
import { type NewGameSpec, type SaveHeader } from "./saveFile";
import type { EditOp } from "./protocol";
import { firstBuiltAt } from "./clockMain";
import { api, fares, putFares } from "./clockState";

const START_FARES: Fares = { ...faresOf({ nodes: START_NODES }, START_FARE), scales: {}, baseline: FARE_BASELINE, basis: "usd" };
/** T-1906: a new game's fares, round ones where it starts (`spec.start`, else the home city); every
 * other country charges them by its income over that country's until the player sets a scale. */
export function startFares(spec?: NewGameSpec): Fares {
  const country = (spec?.start ? CITIES.find((x) => x.id === spec.start)?.country : undefined) ?? cityById(HOME_CITY).country;
  return { ...faresOf({ nodes: startNodes(country) }, START_FARE), scales: { [START]: incomeScale(country) }, baseline: FARE_BASELINE, basis: "usd", income: FARE_E };
}
/** A save's fares. One from before T-1277 kept New York's fares, so its curve is divided by the home
 * country's scale; one from before T-1521 (no `basis`) is turned to the market-rate rule with every
 * fare weighing as it did (game/income.ts). */
export function loadFares(f: SaveHeader["fares"] | undefined): Fares {
  if (!f) return { ...START_FARES };
  const old = f.basis !== "usd";
  const home = cityById(HOME_CITY).country;
  const k = !old ? 1 : oldCurveFactor(home) / (f.baseline ? 1 : oldIncomeScale(home));
  const g = k === 1 ? f : { base: f.base * k, perKm: f.perKm * k, nodes: f.nodes?.map(([km, v]) => [km, v * k] as [number, number]) };
  const scales: Record<string, number> = {};
  for (const [c, v] of Object.entries(f.scales ?? {})) if (Number.isFinite(v)) scales[c] = old ? scaleFromOld(c, v) : v;
  // T-2670: a start key from before is its country's income scale under T-1521's power; the same
  // country's now (the US's is the same 11.6, so a game started there charges as it did)
  if (scales[START] !== undefined && (f.income ?? OLD_FARE_E) !== FARE_E && scales[START] > 0) scales[START] = startFromOld(scales[START]);
  // T-2021: each country's own multiple is kept to the nearest 0.01x (the start and every-country keys are the default's, not a multiple);
  // T-2679: once there is a reference country, scales are kept as set (multiples of its fares)
  const ref = typeof f.ref === "string" && f.ref ? { ref: f.ref } : {};
  for (const c of Object.keys(scales)) if (c !== START && c !== EVERY) scales[c] = ref.ref ? keepOwn(scales[c]) : tidyOwn(c, scales[c], scales);
  // T-1626: a save from before has none and charges the default
  const hsr = typeof f.hsr === "number" && Number.isFinite(f.hsr) ? { hsr: tidyHsr(f.hsr) } : {};
  const ferry = typeof f.ferry === "number" && Number.isFinite(f.ferry) ? { ferry: tidyHsr(f.ferry) } : {};
  const bus = typeof f.bus === "number" && Number.isFinite(f.bus) ? { bus: tidyHsr(f.bus) } : {};
  // T-1801: a save from before has none: changes are free
  const transfer = tidyTransfer(f.transfer ?? 0);
  // T-3567: lines' own multiples; a save from before has none, every line at 1
  const own: Record<string, number> = {};
  for (const [id, v] of Object.entries(f.lines ?? {})) if (typeof v === "number" && Number.isFinite(v) && tidyLineFare(v) !== 1) own[id] = tidyLineFare(v);
  const lines = Object.keys(own).length ? { lines: own } : {};
  // T-2559: an old save's base and rate are taken from its rounded nodes (what the slot writes next), so the first reload's network is the later ones'
  // T-3739: a save from before the market-rate rule had three nodes (two before T-1143): any other count is its line
  const curve = faresOf(old && g.nodes ? { ...g, nodes: simpleNodes(g.nodes) } : g, START_FARES);
  const fare = old || !f.nodes ? faresOf({ nodes: curve.nodes }, START_FARES) : curve;
  // T-3740: fare areas; a save from before has none
  return capTransfer({ ...fare, scales, baseline: FARE_BASELINE, basis: "usd", income: FARE_E, ...hsr, ...ferry, ...bus, ...(transfer > 0 ? { transfer } : {}), ...ref, ...lines, ...zonesOf(f), ...cityFares(f) });
}
/** T-3766: `f` without the list of multiples it does not charge by: by country no city's, by city no country's own
 * (`faresIn` reads only the start and every-country keys then). */
function cityFaresInUse<F extends Fares>(f: F): F {
  if (!f.byCity) {
    if (!f.cityScales) return f;
    const { cityScales: _, ...rest } = f;
    return rest as F;
  }
  const scales: Record<string, number> = {};
  for (const k of [START, EVERY]) if (f.scales?.[k] !== undefined) scales[k] = f.scales[k];
  return { ...f, scales };
}
/** T-3766: fares by city and the cities' own multiples, as kept (a save from before has neither: fares by country). */
export function cityFares(f: { byCity?: unknown; cityScales?: unknown }): Pick<Fares, "byCity" | "cityScales"> {
  const own: Record<string, number> = {};
  if (f.cityScales && typeof f.cityScales === "object") for (const [c, v] of Object.entries(f.cityScales)) if (typeof v === "number" && Number.isFinite(v) && isCity(c)) own[c] = keepOwn(v);
  return { ...(f.byCity === true ? { byCity: true } : {}), ...(Object.keys(own).length ? { cityScales: own } : {}) };
}
/** T-3567: line `line`'s own fare multiple set to `mul` (1 clears it), its quad-track twin's with it; false for
 * no such line. */
export function setLineFare(a: TrackApi, line: number, mul: number): boolean {
  const own = { ...fares.lines }, v = tidyLineFare(mul), li = a.lines_info();
  let found = false;
  // the hidden twins of the line (T-3333), run on its other pair of tracks, charge as it does
  const twins: number[] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i] === line) found = true;
    else if (a.line_twin_of(li[i]) === line) twins.push(li[i]);
  }
  if (!found) return false;
  for (const id of [line, ...twins]) { if (v === 1) delete own[String(id)]; else own[String(id)] = v; }
  const { lines: _, ...rest } = fares;
  putFares(Object.keys(own).length ? { ...rest, lines: own } : rest);
  return true;
}
/** T-3567: the lines' own fare multiples kept to lines that exist, each twin at its line's; called as the network
 * is sent to demand, so a deleted line's number never carries a multiple to a later line. Integer keys keep
 * ascending order in any object, so two of the same multiples compare equal as JSON. */
export function tidyLineFares(a: TrackApi): void {
  if (!fares.lines && !goneLineFares.size) return;
  const li = a.lines_info(), own: Record<string, number> = {}, had = fares.lines ?? {};
  for (let i = 0; i < li.length; i += 22) {
    const id = li[i], key = String(a.line_twin_of(id) >= 0 ? a.line_twin_of(id) : id), v = had[key] ?? goneLineFares.get(key);
    if (v !== undefined) own[String(id)] = v;
  }
  // T-3581: a deleted line's multiple stays in hand for an undo that brings the line back
  for (const id of Object.keys(own)) goneLineFares.delete(id);
  for (const [id, v] of Object.entries(had)) if (!(id in own)) goneLineFares.set(id, v);
  if (JSON.stringify(own) === JSON.stringify(had)) return;
  const { lines: _, ...rest } = fares;
  putFares(Object.keys(own).length ? { ...rest, lines: own } : rest);
}
/** T-3581: the own fare multiples of deleted lines, by id, until an undo brings one back or a new line takes its
 * id; never saved. */
export const goneLineFares = new Map<string, number>();
/** T-3581: a new line starts at 1.00x, whatever a deleted line of its id had. */
export function newLineFare(op: EditOp, line: number) {
  if (op.op !== "addLine" && op.op !== "addBusLine" && op.op !== "addFerryLine" && op.op !== "copyLine") return;
  const id = String(line);
  goneLineFares.delete(id);
  if (fares.lines?.[id] === undefined) return;
  const { [id]: _, ...own } = fares.lines;
  const { lines: __, ...rest } = fares;
  putFares(Object.keys(own).length ? { ...rest, lines: own } : rest);
}
/** T-2679: the fares as demand reads them, without the reference country, which changes no fare. */
export function demandFares(): Fares {
  const { ref: _, zones, zoneNext: __, ...all } = fares;
  // T-3766: nor does the list of multiples not in use (the cities' with fares by country, the countries' own by city)
  const rest = cityFaresInUse(all);
  if (!zones) return rest;
  // T-3740: nor do an area's name, the next id or an area no station is in, so none of them starts a solve
  const codes = api?.station_fares() ?? [], used = new Set<number>();
  for (let k = 1; k < codes.length; k += 2) used.add(areaOfCode(codes[k]));
  const kept = zones.filter((a) => !a.off && used.has(a.id)).map((a) => ({ id: a.id, name: "", prices: a.prices }));
  return kept.length ? { ...rest, zones: kept } : rest;
}
/** T-3740: stations whose code names an area the fares no longer have lose it (on no undo history: removing an
 * area is a setting), so the model and the save never hold a removed area's code; after an area goes, an undo or
 * redo, and a load. How many changed. */
export function tidyStationFares(a: TrackApi): number {
  return a.tidy_station_fares(Uint32Array.from(fares.zones ?? [], (z) => z.id));
}

/** T-2679: a game without a reference country takes its first built station's country, once the
 * zones' outlines can name it. Not marked for saving: a reload works out the same one. */
export function pinFareRef() {
  if (fares.ref || !firstBuiltAt) return;
  const z = cityWithin(firstBuiltAt[0], firstBuiltAt[1]);
  const c = z ? cityById(z).country : undefined;
  if (c) putFares({ ...fares, ref: c });
}
