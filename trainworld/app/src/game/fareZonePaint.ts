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
import { areaOfCode, MAX_ZONES, START_KM_PER_ZONE, startPrices, tidyPrices, zoneCode, zoneOfCode } from "./fareZones";
import { unitMetres } from "./format";
import { groundM } from "../map/geo";
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
function setCodes(groups: readonly StationInput[][], code: number): Promise<boolean> {
  return setEach(groups.map((g) => [g, code]));
}

/** Each complex into its own code, all in one edit, none when no station's code would change. */
async function setEach(list: readonly (readonly [readonly StationInput[], number])[]): Promise<boolean> {
  const nodes: number[] = [], codes: number[] = [];
  for (const [g, code] of list) for (const o of g) if ((o.fare ?? 0) !== code) { nodes.push(o.num); codes.push(code); }
  if (!nodes.length) return false;
  const a = await edit({ op: "stationFares", nodes, codes });
  return a.ok;
}

// ---------------------------------------------------------------- set zones by distance (rings)

/** Ring widths offered, in the player's unit (km or mi). */
export const RING_STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30];
/** The next station click sets zones by distance from it ("Set zones by distance" pressed). */
export const ringsArmed = signal(false);
/** The ring width picked, in the player's unit; null for the step nearest `START_KM_PER_ZONE` km. */
export const ringWidth = signal<number | null>(null);

export function ringWidthShown(): number {
  const v = ringWidth.value;
  if (v !== null) return v;
  const want = (START_KM_PER_ZONE * 1000) / unitMetres();
  return RING_STEPS.reduce((best, s) => (Math.abs(s - want) < Math.abs(best - want) ? s : best));
}

/** A complex's (or lone station's) middle, lng/lat. */
function middle(g: readonly StationInput[]): [number, number] {
  let lng = 0, lat = 0;
  for (const s of g) { lng += s.lng; lat += s.lat; }
  return [lng / g.length, lat / g.length];
}

export interface RingPlan {
  /** the centre complex's middle, lng/lat */
  at: [number, number];
  /** ring width, ground m */
  widthM: number;
  /** each complex the rings reach, with its zone (1 at the centre, one more each `widthM` out) */
  zones: { group: StationInput[]; zone: number }[];
  /** the highest zone given */
  top: number;
}

/** Zones by distance from station `centre` (its complex's middle): a complex `d` ground m from it gets zone
 * floor(d / width) + 1; those past the last zone are left as they are, so a far city is never pulled in. */
export function ringPlan(centre: number, widthM = ringWidthShown() * unitMetres()): RingPlan | null {
  const w = world.peek();
  if (!w || !(widthM > 0)) return null;
  const groups = complexesOf(w.save.stations.map((s) => s.num));
  const home = groups.find((g) => g.some((s) => s.num === centre));
  if (!home) return null;
  const at = middle(home), zones: RingPlan["zones"] = [];
  let top = 0;
  for (const g of groups) {
    const [lng, lat] = middle(g), zone = Math.floor(groundM(at[0], at[1], lng, lat) / widthM) + 1;
    if (zone > MAX_ZONES) continue;
    zones.push({ group: g, zone });
    top = Math.max(top, zone);
  }
  return { at, widthM, zones, top };
}

/** The rings helper's click: stations around `centre` into the tool's area by distance, one edit (one Ctrl+Z). As
 * a loop does, a complex in another area moves only once the player says so. */
export function paintRings(centre: number): Promise<boolean> {
  const a = currentArea();
  ringsArmed.value = false;
  if (!a) return Promise.resolve(false);
  const split = () => {
    const plan = ringPlan(centre), areas = fareAreas(), mine: [StationInput[], number][] = [], other: [StationInput[], number][] = [];
    for (const { group, zone } of plan?.zones ?? []) {
      const c = complexCode(group, areas);
      (c && areaOfCode(c) !== a.id ? other : mine).push([group, zoneCode(a.id, zone)]);
    }
    return { mine, other };
  };
  const now = split();
  if (!now.other.length) return setEach(now.mine);
  return new Promise((done) => {
    ask.value = {
      text: t("fareZonePaint.moveAsk"), yes: t("fareZonePaint.move"), no: t("toolNotes.cancel"),
      run: () => { const s = split(); void setEach([...s.mine, ...s.other]).then(done); },
      stay: () => void setEach(split().mine).then(done),
    };
  });
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
