// T-3740 stage Z2: the Fare zones tool's state and its edits. The tool gives stations the picked fare area and zone
// number (`paintZone` 0 takes them out of any); a click or a loop is one `stationFares` edit, so one recalculation
// of trips and one Ctrl+Z. A station is painted with its whole complex: riders see a complex as one station, and
// demand merges its stations into one (the first zone among them). The areas live in the fares (`fareZones`, a
// setting, not undone). notes/T-3740.md, docs/ui-panels.md "Advanced fares".

import { signal } from "@preact/signals";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";
import { cityById, isCity, viewedZone } from "./cities";
import { complexOf } from "./complexView";
import type { FareArea } from "./fareCurve";
import { areaOfCode, MAX_ZONES, startPrices, tidyPrices, zoneCode, zoneOfCode } from "./fareZones";
import { moneyState } from "./money";
import { tool } from "./ui";
import { edit, stationsById, world } from "./world";

/** The area the tool paints into; null until one is picked. */
export const paintArea = signal<number | null>(null);
/** The zone number the tool gives, 1 to `MAX_ZONES`; 0 takes stations out of their fare zone. */
export const paintZone = signal(1);

export const fareAreas = (): FareArea[] => moneyState.value?.fares.zones ?? [];

/** A station's code as it counts: 0 when its area is gone (Z3 tidies those) or it has none. */
export function liveCode(code: number | undefined, areas: readonly FareArea[]): number {
  if (!code || !zoneOfCode(code)) return 0;
  const id = areaOfCode(code);
  return areas.some((a) => a.id === id) ? code : 0;
}

/** The area the tool paints into now: the one picked if it is still there, else the first. */
export function currentArea(areas = fareAreas()): FareArea | null {
  const id = paintArea.value;
  return areas.find((a) => a.id === id) ?? areas[0] ?? null;
}

/** Start the tool on area `id`. */
export function startPainting(id: number) {
  paintArea.value = id;
  if (paintZone.peek() < 1) paintZone.value = 1;
  tool.value = "fares";
}

function sendAreas(zones: FareArea[], zoneNext?: number) {
  return edit({ op: "fareZones", zones, ...(zoneNext ? { zoneNext } : {}) });
}

/** A new area named for the city in view, its prices the curve's fares at 5, 10, 15 km ...; its id, or null. */
export async function newArea(): Promise<number | null> {
  const f = moneyState.peek()?.fares;
  if (!f) return null;
  const areas = f.zones ?? [];
  const id = Math.max(f.zoneNext ?? 1, ...areas.map((a) => a.id + 1));
  const z = viewedZone.peek(), c = isCity(z) ? cityById(z!) : null;
  const name = c ? t("fareZonePaint.areaName", { city: tPlace(c.id, c.name) }) : t("fareZonePaint.areaNameNone", { n: areas.length + 1 });
  const a = await sendAreas([...areas, { id, name, prices: startPrices(f.nodes) }], id + 1);
  return a.ok ? id : null;
}

export function renameArea(id: number, name: string) {
  const areas = fareAreas();
  const v = name.trim();
  if (!v || !areas.some((a) => a.id === id && a.name !== v)) return;
  void sendAreas(areas.map((a) => (a.id === id ? { ...a, name: v } : a)));
}

/** Area `id`'s price for `k` zones (curve units), the later ones lifted so prices never fall. */
export function setZonePrice(id: number, k: number, v: number) {
  const areas = fareAreas();
  const a = areas.find((x) => x.id === id);
  if (!a || !(v >= 0) || k < 1 || k > MAX_ZONES) return;
  const prices = a.prices.slice();
  while (prices.length < k) prices.push(prices[prices.length - 1] ?? v);
  prices[k - 1] = v;
  // a lower price may not fall under an earlier one: those come down with it
  for (let i = k - 2; i >= 0; i--) prices[i] = Math.min(prices[i], v);
  void sendAreas(areas.map((x) => (x.id === id ? { ...x, prices: tidyPrices(prices) } : x)));
}

/** Switch area `id` off or on; its paint, name and prices stay. */
export function setAreaOff(id: number, off: boolean) {
  const areas = fareAreas();
  if (!areas.some((a) => a.id === id && !!a.off !== off)) return;
  void sendAreas(areas.map((a) => (a.id === id ? (off ? { ...a, off: true } : { id: a.id, name: a.name, prices: a.prices }) : a)));
}

export function removeArea(id: number) {
  void sendAreas(fareAreas().filter((a) => a.id !== id));
}

/** The highest zone number painted in area `id` (0 for none). */
export function topZone(id: number): number {
  let top = 0;
  for (const s of world.value?.save.stations ?? []) if (s.fare && areaOfCode(s.fare) === id) top = Math.max(top, zoneOfCode(s.fare));
  return top;
}

/** Stations `nums` (track model node ids) with their complexes into `code` (0: out of any): one edit, none when no
 * station's code would change. True when an edit went. */
export async function setStationCodes(nums: Iterable<number>, code: number): Promise<boolean> {
  const w = world.peek();
  if (!w) return false;
  const byId = stationsById(w), byNum = new Map(w.save.stations.map((s) => [s.num, s]));
  const nodes: number[] = [], seen = new Set<number>();
  for (const n of nums) {
    const s = byNum.get(n);
    if (!s) continue;
    for (const m of complexOf(s.id)) {
      const o = byId.get(m);
      if (!o || seen.has(o.num)) continue;
      seen.add(o.num);
      if ((o.fare ?? 0) !== code) nodes.push(o.num);
    }
  }
  if (!nodes.length) return false;
  const a = await edit({ op: "stationFares", nodes, codes: nodes.map(() => code) });
  return a.ok;
}

/** The tool's click or loop: stations `nums` into the picked area and zone. */
export function paintStations(nums: Iterable<number>): Promise<boolean> {
  const a = currentArea(), z = paintZone.peek();
  if (!a) return Promise.resolve(false);
  return setStationCodes(nums, z > 0 ? zoneCode(a.id, z) : 0);
}
