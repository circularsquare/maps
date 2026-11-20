// T-3695: the transit share split by the kind of line ridden (ferry, tram, train, high speed, bus), for the mode
// split bars while the setting is on. The demand bubbles keep their three colours.
//
// A zone's split is its stops' boardings by the kind of line boarded, from the shown day's figures: a boarding is
// one leg, so a trip changing between kinds counts once on each (by leg count; a trip's own legs are not traced,
// DECISIONS 2026-11-20). Trips within a zone count every purpose but the longer trips between places; trips to
// other places count those alone.

import type { DemandView } from "./demand";
import type { CityId } from "./cities";
import { stationZone } from "./listScope";
import { lineKind } from "./newHeadway";
import { display } from "./ui";
import { linesById, stopsById, world } from "./world";
import { metresBetween } from "./walks";
import { zonesLoaded } from "./zones";
import { t } from "../i18n";

export type TransitMode = "ferry" | "tram" | "train" | "hsr" | "bus";
/** In bar order: Anita's teal to blueish purple (T-3695). */
export const TRANSIT_MODES: readonly TransitMode[] = ["ferry", "tram", "train", "hsr", "bus"];
/** The same in both themes and under every bubble scheme; train is the 3-color scheme's transit blue. At least 15
 * apart in OKLab (x100) from each other and 17 from walking's green (notes/T-3695.md). */
export const TRANSIT_COLOURS: Record<TransitMode, string> = {
  ferry: "#0e8c96",
  tram: "#72c2f2",
  train: "#1e3ce6",
  hsr: "#1d2a8c",
  bus: "#9145d2",
};
export function transitName(m: TransitMode): string {
  switch (m) {
    case "ferry": return t("transitModes.ferry");
    case "tram": return t("transitModes.tram");
    case "train": return t("transitModes.train");
    case "hsr": return t("transitModes.hsr");
    case "bus": return t("transitModes.bus");
  }
}

/** Each kind's share of the transit trips, 0-1, summing to 1. */
export type TransitShares = Record<TransitMode, number>;

/** The setting is on. */
export const transitOn = (): boolean => display.value.transitModes;

/** The kind of line `id` is; high speed is the line's train kind (T-817), whatever track it runs on. */
export function lineMode(id: string): TransitMode | null {
  switch (lineKind(id)) {
    case "ferry": return "ferry";
    case "bus": return "bus";
    case "tram": return "tram";
    case "fast": return "hsr";
    case "rail": return "train";
    default: return null;
  }
}

type Sums = Map<CityId, number[]>;
/** per day's lines (every zone's view shares them): the world they were placed with and each zone's boardings by
 * kind, within and to other places */
const memo = new WeakMap<object, { at: string; within: Sums; between: Sums }>();

function sumsOf(v: DemandView): { within: Sums; between: Sums } | null {
  const w = world.value;
  if (!w) return null;
  const at = `${w.version}|${zonesLoaded.value}`;
  const had = memo.get(v.lines);
  if (had?.at === at) return had;
  const byStop = stopsById(w), byLine = linesById(w);
  const longer = v.purposes?.get("longer")?.lines;
  const within: Sums = new Map(), between: Sums = new Map();
  const add = (m: Sums, zone: CityId, k: number, n: number) => {
    let a = m.get(zone);
    if (!a) m.set(zone, (a = [0, 0, 0, 0, 0]));
    a[k] += n;
  };
  for (const [id, d] of v.lines) {
    const mode = lineMode(id);
    if (!mode) continue;
    const k = TRANSIT_MODES.indexOf(mode);
    const stops = d.stops?.length === d.boardings.length ? d.stops : byLine.get(id)?.stops;
    if (!stops || stops.length !== d.boardings.length) continue;
    const far = longer?.get(id)?.boardings;
    for (let j = 0; j < stops.length; j++) {
      const s = byStop.get(stops[j]);
      const zone = s ? stationZone(s) : null;
      if (!zone) continue;
      const x = far?.[j] ?? 0;
      add(within, zone, k, Math.max(0, d.boardings[j] - x));
      if (x > 0) add(between, zone, k, x);
    }
  }
  const out = { at, within, between };
  memo.set(v.lines, out);
  return out;
}

const sharesOf = (a: number[] | undefined): TransitShares | null => {
  const all = a ? a.reduce((s, x) => s + x, 0) : 0;
  if (!a || !(all > 0)) return null;
  return Object.fromEntries(TRANSIT_MODES.map((m, k) => [m, a[k] / all])) as TransitShares;
};

/** Zone `zone`'s transit by kind on view `v`'s day: its trips within it, or (`between`) its trips to other places;
 * null with the setting off, or no boardings at its stops. */
export function zoneTransit(v: DemandView | null | undefined, zone: CityId | null | undefined, between = false): TransitShares | null {
  if (!transitOn() || !v || !zone) return null;
  const s = sumsOf(v);
  return s ? sharesOf((between ? s.between : s.within).get(zone)) : null;
}

/** Several zones' transit trips together: each zone's shares weighted by its transit trips `rail`. */
export function sumTransit(parts: Iterable<{ shares: TransitShares | null; rail: number }>): TransitShares | null {
  const a = [0, 0, 0, 0, 0];
  for (const { shares, rail } of parts) if (shares && rail > 0) TRANSIT_MODES.forEach((m, k) => (a[k] += shares[m] * rail));
  return sharesOf(a);
}

/** A trip's transit by kind of line (the trip explorer): each listed route by its share of the trip's riders, and
 * within it each ride by its length, station to station in a straight line (T-3695: a trip of several kinds is
 * shared by leg length). Null with the setting off or no ride. */
export function tripTransit(routes: readonly { share: number; legs: readonly { kind: "ride" | "walk"; line: string; stations: string[] }[] }[]): TransitShares | null {
  const w = world.value;
  if (!transitOn() || !w) return null;
  const byStop = stopsById(w), a = [0, 0, 0, 0, 0];
  for (const r of routes) {
    const km = [0, 0, 0, 0, 0];
    for (const l of r.legs) {
      const mode = l.kind === "ride" ? lineMode(l.line) : null;
      if (!mode) continue;
      let m = 0;
      for (let j = 1; j < l.stations.length; j++) {
        const p = byStop.get(l.stations[j - 1]), q = byStop.get(l.stations[j]);
        if (p && q) m += metresBetween(p.lng, p.lat, q.lng, q.lat);
      }
      km[TRANSIT_MODES.indexOf(mode)] += m;
    }
    const all = km.reduce((s, x) => s + x, 0);
    if (all > 0) km.forEach((x, k) => (a[k] += (r.share * x) / all));
  }
  return sharesOf(a);
}

/** The trips within zones `ids` together (the world's bar, the Cities tab's first row); null with the setting off. */
export function citiesTransit(ids: Iterable<CityId>, views: ReadonlyMap<CityId, DemandView>): TransitShares | null {
  if (!transitOn()) return null;
  return sumTransit([...ids].map((id) => { const v = views.get(id); return { shares: zoneTransit(v, id), rail: v?.railTripsPerDay ?? 0 }; }));
}

/** The parts transit amount `n` shows as: one per kind with any, in bar order, their amounts summing to `n`. */
export function transitParts(n: number, shares: TransitShares): { mode: TransitMode; name: string; n: number; colour: string }[] {
  return TRANSIT_MODES.filter((m) => shares[m] > 0).map((m) => ({ mode: m, name: transitName(m), n: n * shares[m], colour: TRANSIT_COLOURS[m] }));
}
