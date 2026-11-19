// T-3740 stage Z2: the Fare zones tool's state and its edits. The tool gives stations the picked fare area and zone
// number (`paintZone` 0 takes them out of it); a click or a loop is one `stationFares` edit, so one recalculation
// of trips and one Ctrl+Z. A station is painted with its whole complex: riders see a complex as one station, and
// demand merges its stations into one (the first zone among them). The areas live in the fares (`fareZones`, a
// setting, not undone). notes/T-3740.md, docs/ui-panels.md "Advanced fares".

import { signal } from "@preact/signals";
import { t } from "../i18n";
import { tPlace } from "../i18n/places";
import { cityById, cityWithin, isCity, viewedZone, type CityId } from "./cities";
import { complexOf } from "./complexView";
import type { FareArea } from "./fareCurve";
import { areaOfCode, MAX_ZONES, startPrices, tidyPrices, zoneCode, zoneOfCode } from "./fareZones";
import { live } from "../map/live";
import { moneyState } from "./money";
import type { StationId, StationInput } from "./types";
import { stationZone } from "./listScope";
import { ask, tool } from "./ui";
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

/** A new area's name: "{city} fare zones", then "{city} fare zones 2", 3 ... (or "Fare area {n}" with no city in
 * view), never one an area in the game has, so an area made twice by mistake reads as a second one. */
export function newAreaName(areas: readonly FareArea[], city: string | null): string {
  const taken = new Set(areas.map((a) => a.name.trim()));
  for (let n = city ? 1 : areas.length + 1; ; n++) {
    const name = city
      ? n === 1 ? t("fareZonePaint.areaName", { city }) : t("fareZonePaint.areaNameNth", { city, n })
      : t("fareZonePaint.areaNameNone", { n });
    if (!taken.has(name)) return name;
  }
}

/** The city a new area is named for: the one under the map's centre, or the viewed one before the map exists; when
 * the player has stations and none is in it (a game opened with the camera still at the default view), the city
 * holding most of their stations. */
function namingCity(): CityId | null {
  const m = live.map?.getCenter();
  const view = m ? cityWithin(m.lng, m.lat) : viewedZone.peek();
  const count = new Map<CityId, number>();
  for (const s of world.peek()?.save.stations ?? []) {
    const z = stationZone(s);
    if (z && isCity(z)) count.set(z, (count.get(z) ?? 0) + 1);
  }
  if (view && (!count.size || count.has(view))) return view;
  let best: CityId | null = view, top = 0;
  for (const [z, n] of count) if (n > top) { best = z; top = n; }
  return best;
}

/** A new area named for the city it is made in, its prices the curve's fares at 5, 10, 15 km ...; its id, or null. */
export async function newArea(): Promise<number | null> {
  const f = moneyState.peek()?.fares;
  if (!f) return null;
  const areas = f.zones ?? [];
  const id = Math.max(f.zoneNext ?? 1, ...areas.map((a) => a.id + 1));
  const z = namingCity(), c = isCity(z) ? cityById(z!) : null;
  const name = newAreaName(areas, c ? tPlace(c.id, c.name) : null);
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

/** The code a station or complex (`members`) shows and is painted as: its first member's live code, so a complex
 * whose stations were painted apart and then joined reads as one area (the panel, the tint, the tool). */
export const complexCode = (members: readonly StationInput[], areas: readonly FareArea[]): number =>
  members.map((s) => liveCode(s.fare, areas)).find((c) => c) ?? 0;

/** Stations `nums` (track model node ids) as complexes, each once, in the world as it is now. */
function complexesOf(nums: Iterable<number>): StationInput[][] {
  const w = world.peek();
  if (!w) return [];
  const byId = stationsById(w), byNum = new Map(w.save.stations.map((s) => [s.num, s]));
  const out: StationInput[][] = [], seen = new Set<number>();
  for (const n of nums) {
    const s = byNum.get(n);
    if (!s || seen.has(s.num)) continue;
    const members: StationInput[] = [];
    for (const m of complexOf(s.id)) {
      const o = byId.get(m);
      if (!o || seen.has(o.num)) continue;
      seen.add(o.num);
      members.push(o);
    }
    if (members.length) out.push(members);
  }
  return out;
}

/** Complexes `groups` into `code`: one edit, none when no station's code would change. True when an edit went. */
async function setCodes(groups: readonly StationInput[][], code: number): Promise<boolean> {
  const nodes = groups.flat().filter((o) => (o.fare ?? 0) !== code).map((o) => o.num);
  if (!nodes.length) return false;
  const a = await edit({ op: "stationFares", nodes, codes: nodes.map(() => code) });
  return a.ok;
}

/** Stations `nums` with their complexes into `code` (0: out of any). True when an edit went. */
export function setStationCodes(nums: Iterable<number>, code: number): Promise<boolean> {
  return setCodes(complexesOf(nums), code);
}

/** Linking station `from` to `to` makes one complex, which is in one fare area: `from`'s, or the only one either has.
 * `ask` is true when both have an area and the areas differ (the link then waits for the player's answer); `apply` writes the code onto both complexes once the link is made. Null when nothing is painted. */
export function linkFarePlan(from: StationId, to: StationId): { ask: boolean; apply: () => Promise<boolean> } | null {
  const w = world.peek();
  if (!w) return null;
  const byId = stationsById(w), a = byId.get(from), b = byId.get(to);
  if (!a || !b) return null;
  const ga = complexesOf([a.num])[0] ?? [a], gb = complexesOf([b.num])[0] ?? [b], areas = fareAreas();
  const ca = complexCode(ga, areas), cb = complexCode(gb, areas), code = ca || cb;
  if (!code || ca === cb) return null;
  const ask = !!ca && !!cb && areaOfCode(ca) !== areaOfCode(cb);
  return { ask, apply: () => setCodes([ga, gb], code) };
}

/** Of stations `nums`, those the tool would change now (none: only stations of the picked area, or of a removed one,
 * stay as they are otherwise); a loop rings just these. */
export function paintChanges(nums: readonly number[]): Set<number> {
  const a = currentArea(), out = new Set<number>(nums);
  if (!a || paintZone.peek()) return out;
  const areas = fareAreas(), keep = new Set<number>();
  for (const g of complexesOf(nums)) {
    const c = complexCode(g, areas);
    if (c ? areaOfCode(c) !== a.id : !g.some((o) => o.fare)) for (const o of g) keep.add(o.num);
  }
  for (const n of keep) out.delete(n);
  return out;
}

/** The tool's click or loop: stations `nums` into the picked area and zone. A station is in one area at most, so
 * one in another area moves only once the player says so (Cancel paints the rest); "none" takes out only stations
 * of the picked area (and those of a removed one), leaving other areas' alone. One edit either way. */
export function paintStations(nums: Iterable<number>): Promise<boolean> {
  const list = [...nums];
  const a = currentArea(), z = paintZone.peek();
  if (!a) return Promise.resolve(false);
  const split = () => {
    const areas = fareAreas(), mine: StationInput[][] = [], other: StationInput[][] = [];
    for (const g of complexesOf(list)) {
      const c = complexCode(g, areas);
      (c && areaOfCode(c) !== a.id ? other : mine).push(g);
    }
    return { mine, other };
  };
  const now = split();
  if (!z) return setCodes(now.mine, 0);
  const code = zoneCode(a.id, z);
  if (!now.other.length) return setCodes(now.mine, code);
  return new Promise((done) => {
    ask.value = {
      text: t("fareZonePaint.moveAsk"), yes: t("fareZonePaint.move"), no: t("toolNotes.cancel"),
      run: () => { const s = split(); void setCodes([...s.mine, ...s.other], code).then(done); },
      stay: () => void setCodes(split().mine, code).then(done),
    };
  });
}
