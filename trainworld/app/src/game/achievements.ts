// T-2124 (notes/T-2124.md; docs/ui-panels.md "Achievements"): goals for the whole game. Four tiered
// ones, each a ladder of tiers, and firsts earned once. The page works out the figures (network ones as
// the world changes, demand ones from final results on the average day, the best line's profit each
// game hour) and posts each step met to the clock worker, which keeps it in the save with the game
// second it landed. Never taken back. A sandbox earns none. An older save works its out quietly
// once its first final result is in: everything met then is recorded and nothing is toasted.

import { computed, effect, signal } from "@preact/signals";
import { CITIES, cityById, type CityId } from "./cities";
import { complexes } from "./complexView";
import { continentLabel, continentOf } from "./continents";
import { AVERAGE_DAY, WEEKDAY_LONG } from "./days";
import { demandByDay, demandUpdating, eachCorridor, lineFiguresIn, roughZones, type Corridors } from "./demand";
import { bigDistance, clockTime, dayOf, money, speed } from "./format";
import { t, tCountry, tList, tNumber } from "../i18n";
import { tPlace } from "../i18n/places";

import { stationZone } from "./listScope";
import { moneyState } from "./money";
import { readToday, runningToday } from "./today";
import { weekdayOf } from "./days";
import { stationsById, world } from "./world";
import { RIDES_PER_PERSON, sumModes } from "./worldTotals";
import { zonesLoaded } from "./zones";
import type { LineInput, LineStats, StationInput, WorldView } from "./types";
import { gamesLoaded, postAchievements, queryClock } from "../workers/clockClient";
import type { Achieved } from "../workers/saveFile";

export type Detail = Omit<Achieved, "at">;

/** Three significant figures, trailing zeros dropped: 17.2M, 669k, 1B, 300M. */
export function short(n: number): string {
  const [u, s] = n >= 999.5e6 ? [1e9, "B"] : n >= 999.5 ? (n >= 999.5e3 ? [1e6, "M"] : [1e3, "k"]) : [1, ""];
  return String(Number((n / u).toPrecision(3))) + s;
}

export type TieredId = "cities" | "trips" | "station" | "line";
export interface Tiered { id: TieredId; name: string; tiers: number[] }

/** Anita, 2026-11-03: the first tier not had with a first line, the top one only by going far. */
export const TIERED: Tiered[] = [
  { id: "cities", get name() { return t("achievements.nameCities"); }, tiers: [2, 10, 25, 100] },
  { id: "trips", get name() { return t("achievements.nameTrips"); }, tiers: [1e6, 10e6, 100e6, 300e6, 1e9] },
  { id: "station", get name() { return t("achievements.nameStation"); }, tiers: [500e3, 1e6, 2e6, 5e6] },
  { id: "line", get name() { return t("achievements.nameLine"); }, tiers: [500e3, 1e6, 3e6, 10e6] },
];
export const tierKey = (id: TieredId, tier: number) => `${id}:${tier}`;

/** A tier's step in words: with what earned it (`a`), else the rule ("at one station"); `toast`, under
 * the achievement's name, where "served" would repeat it. */
export function tierText(id: TieredId, tier: number, a?: Achieved | Detail, toast = false): string {
  if (id === "cities") return t(toast ? "achievements.citiesToast" : "achievements.citiesServed", { n: tier });
  if (id === "trips") return t("achievements.tripsDay", { n: short(tier) });
  if (id === "station") return t("achievements.stationBoardings", { n: short(tier), of: a?.of ?? t("achievements.oneStation") });
  return t("achievements.lineRiders", { n: short(tier), of: a?.of ?? t("achievements.oneLine") });
}

/** The firsts in the panel's order (Sea crossing comes with T-2150). */
export type CorridorId = "allJapan" | "orient" | "transcon" | "siberian" | "andean" | "cape" | "ghan" | "silk";
export type NationCode = "JP" | "US" | "ES" | "GB" | "IT" | "DE" | "KR" | "TW" | "CN" | "ID" | "MX" | "BR" | "AU" | "TR";
export type FirstId = "border" | "continents" | "long" | "pearls" | "money" | "bullet" | "circle" | "change" | "city" | "intercity" | "sleeper"
  | CorridorId | "europe" | `nation:${NationCode}`;
export interface First { id: FirstId; name: string; rule?: () => string; text: (a: Achieved | Detail) => string;
  /** the figure now, for the hover of a step not yet earned; "" while unknown */
  now?: () => string;
  /** T-3037: not listed until earned */
  hidden?: boolean }

export const LONG_M = 5_000_000;
export const PEARLS_M = 3_000_000;
export const MONEY_USD = 200e6;
export const BULLET_KMH = 300;
export const CIRCLE_STOPS = 10;
export const CHANGES_PER_TRIP = 1;
export const CHANGE_MIN_TRIPS = 100e3;
/** % of a city's trips within it by train (the city's 20% goal), and of its trips to other places */
export const CITY_SHARE = 20;
export const INTERCITY_SHARE = 20;
/** trips a day to and from other places a city needs before its share counts */
export const INTERCITY_MIN_TRIPS = 100e3;
export const SLEEPER_S = 12 * 3600;
/** riders a day who must be on one train that long */
export const SLEEPER_RIDERS = 1;

/** T-3037: the share of a region's rides a day that trains carry, as a percent, that earns its achievement.
 * Japan's 10 and Europe's 5 are Anita's. The rest are about twice what one strong metro reaches: the real
 * New York save carries 1.2% of the US's rides, the Tokyo save's Seoul and Busan 5.1% of Korea's (its Japan
 * 3.1%), so a country needs several cities of that size, more where it is small (notes/T-3037.md). The countries
 * are those with the most play (Anita's playtime map). `people`: the region's population, the zone populations
 * summed (checks/nation-people.mjs), at `RIDES_PER_PERSON` rides each, as the world's share is (T-1301). */
export const EUROPE_SHARE = 5, EUROPE_PEOPLE = 709_948_408;
export const NATIONS: { cc: NationCode; share: number; people: number }[] = [
  { cc: "JP", share: 10, people: 126_117_819 }, { cc: "US", share: 2, people: 327_534_599 }, { cc: "ES", share: 3, people: 45_649_934 },
  { cc: "GB", share: 4, people: 66_408_930 }, { cc: "IT", share: 3, people: 59_366_251 }, { cc: "DE", share: 4, people: 83_346_335 },
  { cc: "KR", share: 8, people: 51_205_458 }, { cc: "TW", share: 5, people: 23_752_850 }, { cc: "CN", share: 1, people: 1_412_121_555 },
  { cc: "ID", share: 2, people: 267_744_927 }, { cc: "MX", share: 2, people: 126_014_694 }, { cc: "BR", share: 2, people: 201_987_873 },
  { cc: "AU", share: 4, people: 24_664_377 }, { cc: "TR", share: 2, people: 87_720_246 },
];

/** A built station of a running line, as a corridor's ends read it. */
interface Where { zone: CityId | null; cc: string | undefined; lng: number; lat: number }
/** Hokkaido: Japan from the Tsugaru Strait north (Honshu's Shimokita reaches 41.55N east of 140.7E). */
const inHokkaido = (p: Where) => p.cc === "JP" && p.lat >= 41.3 && !(p.lng >= 140.7 && p.lat < 41.65);
/** Shikoku: its box, minus the Hiroshima and Okayama coast and the Shimanami islands (Honshu, north of 34.12N west of 133.2E). */
const inShikoku = (p: Where) => p.cc === "JP" && p.lng >= 132 && p.lng <= 134.85 && p.lat >= 32.7 && p.lat <= 34.42 && !(p.lng < 133.2 && p.lat > 34.12);
const WEST_EUROPE = new Set("GB IE FR BE NL LU DE CH AT LI MC ES PT IT AD".split(" "));
/** Russia from Irkutsk (104.3E) east. */
const EAST_RUSSIA_LNG = 104;
const zoneIs = (id: CityId) => (p: Where) => p.zone === id;
export const CORRIDORS: { id: CorridorId; from: (p: Where) => boolean; to: (p: Where) => boolean }[] = [
  { id: "allJapan", from: inHokkaido, to: inShikoku },
  { id: "orient", from: zoneIs("par"), to: zoneIs("ist") },
  { id: "transcon", from: zoneIs("nyc"), to: zoneIs("snf") },
  { id: "siberian", from: (p) => WEST_EUROPE.has(p.cc ?? ""), to: (p) => p.cc === "RU" && p.lng >= EAST_RUSSIA_LNG },
  { id: "andean", from: zoneIs("san"), to: zoneIs("bue") },
  { id: "cape", from: zoneIs("cap"), to: zoneIs("cai") },
  { id: "ghan", from: zoneIs("ade"), to: zoneIs("daw") },
  { id: "silk", from: zoneIs("ist"), to: zoneIs("xia") },
];
const corridorName: Record<CorridorId, () => string> = {
  allJapan: () => t("achievements.nameAllJapan"), orient: () => t("achievements.nameOrient"), transcon: () => t("achievements.nameTranscon"),
  siberian: () => t("achievements.nameSiberian"), andean: () => t("achievements.nameAndean"), cape: () => t("achievements.nameCape"),
  ghan: () => t("achievements.nameGhan"), silk: () => t("achievements.nameSilk"),
};

/** A record keeps the city's Latin name; shown in the language now set. */
const cityShown = (name: string | undefined) => { const c = name ? CITIES.find((x) => x.name === name) : undefined; return c ? tPlace(c.id, c.name) : name; };
const shareText = (v: number | undefined) => decimals(v ?? 0, 1) + "%";
const pct = (v: number) => `${Math.floor(v)}%`;
const decimals = (v: number, dp: number) => tNumber(v, { minimumFractionDigits: dp, maximumFractionDigits: dp });
/** The countries of a Over-the-border record: new ones store ISO codes, older saves the English names. */
const placeName = (s: string) => (/^[A-Z]{2}$/.test(s) ? tCountry(s) : s);
export const FIRSTS: First[] = [
  { id: "border", get name() { return t("achievements.nameBorder"); }, rule: () => t("achievements.ruleBorder"), text: (a) => t("achievements.textBorder", { of: a.of, places: tList((a.in ?? []).map(placeName)) }) },
  { id: "continents", get name() { return t("achievements.nameContinents"); }, rule: () => t("achievements.ruleContinents"), text: (a) => t("achievements.textContinents", { of: a.of, places: tList((a.in ?? []).map(continentLabel)) }) },
  { id: "long", get name() { return t("achievements.nameLong"); }, rule: () => t("achievements.ruleLong", { length: bigDistance(LONG_M) }), text: (a) => t("achievements.textLong", { of: a.of, length: bigDistance(a.v ?? 0) }) },
  { id: "pearls", get name() { return t("achievements.namePearls"); }, rule: () => t("achievements.rulePearls", { length: bigDistance(PEARLS_M) }), text: (a) => t("achievements.textPearls", { of: a.of, length: bigDistance(a.v ?? 0) }) },
  { id: "money", get name() { return t("achievements.nameMoney"); }, rule: () => t("achievements.ruleMoney", { amount: money(MONEY_USD) }), text: (a) => t("achievements.textMoney", { of: a.of, amount: money(a.v ?? 0) }) },
  { id: "bullet", get name() { return t("achievements.nameBullet"); }, rule: () => t("achievements.ruleBullet", { speed: speed(BULLET_KMH) }), text: (a) => t("achievements.textBullet", { of: a.of, speed: speed(a.v ?? 0) }) },
  { id: "circle", get name() { return t("achievements.nameCircle"); }, rule: () => t("achievements.ruleCircle", { n: CIRCLE_STOPS }), text: (a) => t("achievements.textCircle", { of: a.of, n: a.v ?? 0 }) },
  { id: "change", get name() { return t("achievements.nameChange"); }, rule: () => t("achievements.ruleChange", { trips: short(CHANGE_MIN_TRIPS) }), text: (a) => t("achievements.textChange", { n: decimals(a.v ?? 0, 2) }),
    now: () => { const v = demandFigures.value?.transfersPerTrip; return v === undefined ? "" : t("achievements.nowChange", { n: decimals(v, 2) }); } },
  { id: "city", get name() { return t("achievements.nameCity"); }, rule: () => t("achievements.ruleCity", { share: CITY_SHARE }), text: (a) => t("achievements.textCity", { share: pct(a.v ?? 0), of: cityShown(a.of) }) },
  { id: "intercity", get name() { return t("achievements.nameIntercity"); }, rule: () => t("achievements.ruleIntercity", { share: INTERCITY_SHARE }), text: (a) => t("achievements.textIntercity", { share: pct(a.v ?? 0), of: cityShown(a.of) }) },
  { id: "sleeper", get name() { return t("achievements.nameSleeper"); }, rule: () => t("achievements.ruleSleeper", { hours: SLEEPER_S / 3600 }), text: (a) => t("achievements.textSleeper", { hours: decimals(a.v ?? 0, 1), of: a.of }) },
  // T-3037: hidden until earned, so no rule to show
  ...CORRIDORS.map((c): First => ({ id: c.id, hidden: true, get name() { return corridorName[c.id](); }, text: (a) => t("achievements.textCorridor", { from: cityShown(a.of), to: cityShown(a.in?.[0]) }) })),
  { id: "europe", hidden: true, get name() { return t("achievements.nameEurope"); }, text: (a) => t("achievements.textShare", { share: shareText(a.v), place: continentLabel("Europe") }) },
  ...NATIONS.map((n): First => ({ id: `nation:${n.cc}`, hidden: true, get name() { return t("achievements.nameNation", { cc: n.cc, country: tCountry(n.cc) }); }, text: (a) => t("achievements.textShare", { share: shareText(a.v), place: tCountry(n.cc) }) })),
];

/** "Monday, day 406, at 14:08" on the game clock. */
export function when(at: number): string {
  const m = Math.floor(at / 60), d = dayOf(m);
  return t("achievements.when", { weekday: WEEKDAY_LONG[weekdayOf(d)], day: d, time: clockTime(m) });
}

/** Rail lines with trains set at some level; a blueprint, a line with no trains and buses never count. */
function runningLines(w: WorldView): { l: LineInput; st: LineStats }[] {
  const out: { l: LineInput; st: LineStats }[] = [];
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (st && !st.bus && st.status === "running" && (l.tph.high > 0 || l.tph.medium > 0 || l.tph.low > 0)) out.push({ l, st });
  }
  return out;
}

/** What the figures read of the world: the stations and the running lines. The same object while
 * neither changes (clockClient.ts keeps a station's, line's and line's stats' objects while their rows
 * stay), so an edit to track alone leaves the figures as they were (T-2403). */
interface AchWorld { stations: readonly StationInput[]; byId: Map<string, StationInput>; running: { l: LineInput; st: LineStats }[] }
let lastAch: AchWorld | null = null;
const achWorld = computed<AchWorld | null>(() => {
  const w = world.value;
  if (!w) return (lastAch = null);
  const running = runningLines(w), was = lastAch;
  if (was && was.stations === w.save.stations && was.running.length === running.length && was.running.every((x, i) => x.l === running[i].l && x.st === running[i].st)) return was;
  return (lastAch = { stations: w.save.stations, byId: stationsById(w), running });
});

/** The figures that come from the network alone. */
export const networkFigures = computed(() => {
  const w = achWorld.value;
  void zonesLoaded.value; // boundaries arriving re-place the stations
  if (!w) return null;
  const byId = w.byId;
  const served = new Set<CityId>();
  const firsts = new Map<FirstId, Detail>();
  let long: Detail | null = null, bullet: Detail | null = null, circle: Detail | null = null;
  for (const { l, st } of w.running) {
    const countries = new Set<string>(), continents = new Set<string>();
    for (const id of l.stops) {
      const s = byId.get(id);
      if (!s?.built) continue;
      const z = stationZone(s);
      if (!z) continue;
      served.add(z);
      const cc = cityById(z)?.country;
      if (!cc) continue;
      countries.add(cc);
      const c = continentOf(cc, s.lng, s.lat);
      if (c) continents.add(c);
    }
    if (countries.size > 1 && !firsts.has("border")) firsts.set("border", { of: l.name, in: [...countries] });
    if (continents.size > 1 && !firsts.has("continents")) firsts.set("continents", { of: l.name, in: [...continents] });
    const m = st.km * 1000;
    if (m >= LONG_M && (!long || m > long.v!)) long = { of: l.name, v: m };
    const top = Math.max(0, ...st.stretches.map((x) => x.reached));
    if (top >= BULLET_KMH && (!bullet || top > bullet.v!)) bullet = { of: l.name, v: top };
    const stops = l.stops.length - 1;
    if (st.loop === "circle" && !st.bus && stops >= CIRCLE_STOPS && (!circle || stops > circle.v!)) circle = { of: l.name, v: stops };
  }
  if (long) firsts.set("long", long);
  if (bullet) firsts.set("bullet", bullet);
  if (circle) firsts.set("circle", circle);
  return { cities: served.size, firsts, running: w.running.length };
});

/** T-3037: the corridors with service. Lines are joined where they share a station or a complex (the
 * places a rider changes at); a corridor is met when one such network holds a built station at each end.
 * Worked out when the running lines or the stations change, not with the clock. */
export const corridorFigures = computed(() => {
  const w = achWorld.value;
  void zonesLoaded.value;
  const cx = complexes.value;
  if (!w) return null;
  const up = new Map<string, string>();
  const find = (k: string): string => { let r = k; while (up.get(r) !== r) r = up.get(r)!; up.set(k, r); return r; };
  const key = (id: string) => { const c = cx.of.get(id); return c === undefined ? id : "c" + c; };
  const stops: { name: string; k: string; at: Where }[] = [];
  for (const { l } of w.running) {
    let first: string | null = null;
    for (const id of l.stops) {
      const s = w.byId.get(id);
      if (!s?.built) continue;
      const k = key(id);
      if (!up.has(k)) up.set(k, k);
      if (first === null) first = k; else up.set(find(k), find(first));
      const zone = stationZone(s);
      stops.push({ name: s.name || (zone ? cityById(zone).name : ""), k, at: { zone, cc: zone ? cityById(zone)?.country : undefined, lng: s.lng, lat: s.lat } });
    }
  }
  const ends = new Map<string, { from?: string; to?: string }>();
  const out = new Map<FirstId, Detail>();
  for (const c of CORRIDORS) {
    ends.clear();
    for (const { name, k, at } of stops) {
      const root = find(k), e = ends.get(root) ?? {};
      if (!e.from && c.from(at)) e.from = name;
      if (!e.to && c.to(at)) e.to = name;
      ends.set(root, e);
      if (e.from && e.to) { out.set(c.id, { of: e.from, in: [e.to] }); break; }
    }
  }
  return { firsts: out };
});

/** The longest a rider is sure to stay on one of the line's trains, s, among riders at least
 * `SLEEPER_RIDERS` a day: of the riders on stretch b, all but those who boarded at the stops since
 * stop a were already aboard leaving a (the line's loads and boardings say no more of who rides how far). */
function longestRide(l: LineInput, st: LineStats, loads: Float32Array[], board: Float32Array[] | undefined): number {
  const n = l.stops.length, t0 = st.fromStartS;
  if (!board || n < 2 || t0.length !== n || t0[n - 1] < SLEEPER_S) return 0;
  if (loads.some((p) => p.length !== 2 * (n - 1)) || board.some((p) => p.length !== 2 * n) || board.length !== loads.length) return 0;
  const T = t0[n - 1];
  let best = 0;
  for (let r = 0; r < 2; r++) {
    const t = (k: number) => (r === 0 ? t0[k] : T - t0[n - 1 - k]);
    // boardings up to each stop of this run, per period
    const upTo = board.map((p) => { const s = new Float64Array(n); let a = 0; for (let k = 0; k < n; k++) s[k] = a += p[r * n + k]; return s; });
    for (let a = 0; a < n - 1; a++) {
      for (let b = a; b < n - 1; b++) {
        const dur = t(b + 1) - t(a);
        if (dur < SLEEPER_S || dur <= best) continue;
        let sure = 0;
        for (let p = 0; p < loads.length; p++) sure += Math.max(0, loads[p][r * (n - 1) + b] - (upTo[p][b] - upTo[p][a]));
        if (sure >= SLEEPER_RIDERS) best = dur;
      }
    }
  }
  return best;
}

/** Each place's trips to and from other places and those by rail, over the corridors touching any of
 * `places`, kept for the last corridors and places (a demand result's, T-2403). */
let corridorMemo: { corr: Corridors; key: string; sums: Map<number, { all: number; rail: number }> } | null = null;
function corridorSums(corr: Corridors, places: ReadonlyMap<number, CityId>): Map<number, { all: number; rail: number }> {
  const key = [...places.keys()].sort((a, b) => a - b).join(",");
  if (corridorMemo?.corr === corr && corridorMemo.key === key) return corridorMemo.sums;
  const sums = new Map<number, { all: number; rail: number }>();
  const add = (p: number, all: number, rail: number) => {
    if (!places.has(p)) return;
    const e = sums.get(p) ?? { all: 0, rail: 0 };
    e.all += all; e.rail += rail;
    sums.set(p, e);
  };
  eachCorridor(corr, (a, b) => a !== b && (places.has(a) || places.has(b)), (j, x) => { add(corr.table.from[j], x.trips, x.rail); add(corr.table.to[j], x.trips, x.rail); });
  corridorMemo = { corr, key, sums };
  return sums;
}

/** The figures from demand: final results only, on the average day, never a rough zone's. */
export const demandFigures = computed(() => {
  const w = achWorld.value;
  void zonesLoaded.value;
  const d = demandByDay.value[AVERAGE_DAY];
  const v = d?.view;
  if (!w || !v?.final) return null;
  const rough = roughZones.value, byId = w.byId;
  const run = w.running;
  const firsts = new Map<FirstId, Detail>();
  // Trips a day: the Cities tab's World row, coarse zones included (a big game holds most zones
  // coarse, and a total of millions is past the noise the star's rule guards against)
  const built = new Set<CityId>();
  for (const s of w.stations) if (s.built) { const z = stationZone(s); if (z) built.add(z); }
  const views = [...built].map((id) => d.cities.get(id)).filter((x) => x?.final);
  const trips = sumModes(views)?.rides.train ?? 0;
  // busiest station, by complex
  const cx = complexes.value;
  const byCx = new Map<string, { n: number; top: number; name: string }>();
  let transfers = 0;
  for (const [sid, sd] of v.stations) {
    const s = byId.get(sid);
    if (!s) continue;
    transfers += sd.transfers ?? 0;
    const c = cx.of.get(sid), k = c === undefined ? sid : "c" + c;
    const e = byCx.get(k) ?? { n: 0, top: -1, name: s.name };
    e.n += sd.boardings;
    if (sd.boardings > e.top) { e.top = sd.boardings; e.name = s.name; }
    byCx.set(k, e);
  }
  let station: Detail | null = null;
  for (const e of byCx.values()) if (!station || e.n > station.v!) station = { of: e.name || undefined, v: e.n };
  let line: Detail | null = null, sleeper: Detail | null = null;
  // whether the result is this network's: every running line has figures for its stops (after a load
  // the last game's result can stand until the new one's lands)
  let covered = true;
  for (const { l, st } of run) {
    const dm = v.lines.get(l.id);
    if (!dm || !lineFiguresIn(v, l.id, l.stops.length)) { covered = false; continue; }
    if (!line || dm.ridersPerDay > line.v!) line = { of: l.name, v: dm.ridersPerDay };
    const s = longestRide(l, st, dm.loads, dm.board);
    if (s >= SLEEPER_S && (!sleeper || s / 3600 > sleeper.v!)) sleeper = { of: l.name, v: s / 3600 };
  }
  if (sleeper) firsts.set("sleeper", sleeper);
  if (v.railTripsPerDay >= CHANGE_MIN_TRIPS && transfers / v.railTripsPerDay >= CHANGES_PER_TRIP) firsts.set("change", { v: transfers / v.railTripsPerDay });
  // a city's own trips (its 20% goal) and its trips to and from other places
  let city: Detail | null = null, inter: Detail | null = null;
  const places = new Map<number, CityId>();
  for (const id of built) {
    const c = cityById(id);
    if (!c) continue;
    places.set(c.place, id);
    const cv = d.cities.get(id);
    if (cv?.final && !rough.has(id) && cv.split.train >= CITY_SHARE && (!city || cv.split.train > city.v!)) city = { of: c.name, v: cv.split.train };
  }
  const corr = d.intercity?.corridors;
  if (corr && places.size) {
    for (const [p, e] of corridorSums(corr, places)) {
      if (e.all < INTERCITY_MIN_TRIPS) continue;
      const share = 100 * e.rail / e.all;
      if (share >= INTERCITY_SHARE && (!inter || share > inter.v!)) inter = { of: cityById(places.get(p)!)?.name, v: share };
    }
  }
  if (city) firsts.set("city", city);
  if (inter) firsts.set("intercity", inter);
  // T-3037: the share of each region's rides that trains carry (built zones' final views)
  const railIn = new Map<string, number>();
  let railEurope = 0;
  for (const id of built) {
    const c = cityById(id), cv = d.cities.get(id);
    if (!c?.country || !cv?.final) continue;
    railIn.set(c.country, (railIn.get(c.country) ?? 0) + cv.railTripsPerDay);
    if (continentOf(c.country, c.center[0], c.center[1]) === "Europe") railEurope += cv.railTripsPerDay;
  }
  const shareOf = (rail: number, people: number) => (100 * rail) / (RIDES_PER_PERSON * people);
  if (shareOf(railEurope, EUROPE_PEOPLE) >= EUROPE_SHARE) firsts.set("europe", { v: shareOf(railEurope, EUROPE_PEOPLE) });
  for (const n of NATIONS) { const v = shareOf(railIn.get(n.cc) ?? 0, n.people); if (v >= n.share) firsts.set(`nation:${n.cc}`, { of: n.cc, v }); }
  const transfersPerTrip = v.railTripsPerDay > 0 ? transfers / v.railTripsPerDay : undefined;
  return { trips, station, line, firsts, covered, transfersPerTrip };
});

/** The best line's profit over the past 24 hours (the Lines tab's Money figure), asked of the clock
 * worker as each game hour turns and with each new demand result; undefined until the first answer. */
export const bestProfit = signal<Detail | null | undefined>(undefined);
/** the longest line (m) of at least PEARLS_M that is in profit over the same 24 hours */
export const longProfitable = signal<Detail | null>(null);
/** the average day's view the last answer was asked after */
let profitView: unknown = null;
let profitHour = NaN, profitGame = -1, profitAsked: unknown = null;
effect(() => {
  const m = moneyState.value, g = gamesLoaded.value, dv = demandByDay.value[AVERAGE_DAY]?.view ?? null;
  if (g !== profitGame) { profitGame = g; profitHour = NaN; profitAsked = profitView = null; bestProfit.value = undefined; longProfitable.value = null; }
  const hour = m?.hours.at(-1)?.hour;
  if (!m || m.mode === "sandbox" || hour === undefined || (hour === profitHour && dv === profitAsked)) return;
  profitHour = hour;
  profitAsked = dv;
  void queryClock({ what: "today" }).then((raw) => {
    const w = world.peek();
    if (!raw || !w || gamesLoaded.peek() !== g) return;
    const t = readToday(raw);
    let best: Detail | null = null, pearl: Detail | null = null;
    for (const { l, st } of runningLines(w)) {
      const d = t.get(l.id);
      if (!d) continue;
      const net = d.fares - runningToday(l, st) - st.upkeep;
      if (!best || net > best.v!) best = { of: l.name, v: net };
      const m = st.km * 1000;
      if (net > 0 && m >= PEARLS_M && (!pearl || m > pearl.v!)) pearl = { of: l.name, v: m };
    }
    profitView = dv;
    longProfitable.value = pearl;
    bestProfit.value = best;
  });
});

/** Whether the world has come since the last game was loaded (the last game's stands until then). */
const worldSinceLoad = signal(false);
let worldAtLoad: unknown = null;
effect(() => { void gamesLoaded.value; worldAtLoad = world.peek(); worldSinceLoad.value = false; });
effect(() => { const w = world.value; if (w && w !== worldAtLoad) worldSinceLoad.value = true; });

/** Every step met now, by key, with what met it, and whether every figure is in for a quiet working-out. */
export const metNow = computed(() => {
  const nf = networkFigures.value, df = demandFigures.value, bp = bestProfit.value, cf = corridorFigures.value;
  const out: Record<string, Detail> = {};
  if (!nf) return { met: out, ready: false };
  const tier = (id: TieredId, v: number | undefined, d: Detail) => {
    for (const t of TIERED.find((x) => x.id === id)!.tiers) if (v !== undefined && v >= t) out[tierKey(id, t)] = d;
  };
  tier("cities", nf.cities, { v: nf.cities });
  for (const [k, d] of nf.firsts) out[k] = d;
  if (cf) for (const [k, d] of cf.firsts) out[k] = d;
  if (df) {
    tier("trips", df.trips, { v: df.trips });
    if (df.station) tier("station", df.station.v, df.station);
    if (df.line) tier("line", df.line.v, df.line);
    for (const [k, d] of df.firsts) out[k] = d;
  }
  if (bp && bp.v! >= MONEY_USD) out.money = bp;
  if (longProfitable.value) out.pearls = longProfitable.value;
  // an older save's quiet working-out waits for the loaded game's own world, this network's own final
  // (a cached result shown ahead of it is solved again, `demandUpdating`) and the profit asked after it
  const settled = !!df?.covered && bp !== undefined && profitView === demandByDay.value[AVERAGE_DAY]?.view;
  return { met: out, ready: worldSinceLoad.value && !demandUpdating.value && (nf.running === 0 || settled) };
});

/** How many steps there are, and how many this game has earned. */
export const STEP_KEYS = new Set([...TIERED.flatMap((t) => t.tiers.map((x) => tierKey(t.id, x))), ...FIRSTS.map((f) => f.id)]);
const HIDDEN_KEYS = new Set<string>(FIRSTS.filter((f) => f.hidden).map((f) => f.id));
/** The steps listed: the hidden ones count only once earned. */
export const STEPS = computed(() => STEP_KEYS.size - HIDDEN_KEYS.size + Object.keys(moneyState.value?.achievements ?? {}).filter((k) => HIDDEN_KEYS.has(k)).length);
export const earned = computed(() => Object.keys(moneyState.value?.achievements ?? {}).filter((k) => STEP_KEYS.has(k)).length);

// Post the steps met that the worker has not recorded; an older save's once its figures are all in.
let posted = new Set<string>(), postedGame = -1;
effect(() => {
  const g = gamesLoaded.value;
  if (g !== postedGame) { postedGame = g; posted = new Set(); }
  const m = moneyState.value;
  if (!m || m.mode === "sandbox") return;
  const { met, ready } = metNow.value;
  const had = m.achievements;
  if (!had) {
    if (!ready || posted.has("")) return;
    posted.add("");
    postAchievements(met);
    return;
  }
  const got: Record<string, Detail> = {};
  for (const [k, d] of Object.entries(met)) if (!(k in had) && !posted.has(k)) { got[k] = d; posted.add(k); }
  if (Object.keys(got).length) postAchievements(got);
});
