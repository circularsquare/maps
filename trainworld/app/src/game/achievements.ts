// T-2124 (notes/T-2124.md; docs/ui-panels.md "Achievements"): goals for the whole game. Four tiered
// ones, each a bar of tiers, and firsts earned once. The page works out the figures (network ones as
// the world changes, demand ones from final results on the average day, the best line's profit each
// game hour) and posts each step met to the clock worker, which keeps it in the save with the game
// second it landed. Never taken back. A sandbox earns none. An older save works its out quietly
// once its first final result is in: everything met then is recorded and nothing is toasted.

import { computed, effect, signal } from "@preact/signals";
import { cityById, type CityId } from "./cities";
import { complexes } from "./complexView";
import { continentOf } from "./continents";
import { AVERAGE_DAY } from "./days";
import { demandByDay, demandUpdating, eachCorridor, lineFiguresIn, roughZones } from "./demand";
import { andList, bigDistance, clockTime, dayOf, money, speed } from "./format";
import { countryName } from "./holidays";
import { stationZone } from "./listScope";
import { moneyState } from "./money";
import { readToday, runningToday } from "./today";
import { weekdayOf } from "./days";
import { stationsById, world } from "./world";
import { sumModes } from "./worldTotals";
import { zonesLoaded } from "./zones";
import type { LineInput, LineStats, WorldView } from "./types";
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
  { id: "cities", name: "Cities served", tiers: [2, 10, 25, 100] },
  { id: "trips", name: "Trips a day", tiers: [1e6, 10e6, 100e6, 300e6, 1e9] },
  { id: "station", name: "Busiest station", tiers: [500e3, 1e6, 2e6, 5e6] },
  { id: "line", name: "Busiest line", tiers: [500e3, 1e6, 3e6, 10e6] },
];
export const tierKey = (id: TieredId, tier: number) => `${id}:${tier}`;

/** A tier's step in words: with what earned it (`a`), else the rule ("at one station"); `toast`, under
 * the achievement's name, where "served" would repeat it. */
export function tierText(id: TieredId, tier: number, a?: Achieved | Detail, toast = false): string {
  if (id === "cities") return toast ? `${tier} cities` : `${tier} cities served`;
  if (id === "trips") return `${short(tier)} train trips/day`;
  if (id === "station") return `${short(tier)} boardings/day at ${a?.of ?? "one station"}`;
  return `${short(tier)} riders/day on ${a?.of ?? "one line"}`;
}

/** The firsts in the panel's order (Sea crossing comes with T-2150). */
export type FirstId = "border" | "continents" | "long" | "money" | "bullet" | "circle" | "change" | "city" | "intercity" | "sleeper";
export interface First { id: FirstId; name: string; rule: () => string; text: (a: Achieved | Detail) => string }

export const LONG_M = 1_000_000;
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

const pct = (v: number) => `${Math.floor(v)}%`;
export const FIRSTS: First[] = [
  { id: "border", name: "Over the border", rule: () => "A line with stops in two countries", text: (a) => `${a.of} stops in ${andList(a.in ?? [])}` },
  { id: "continents", name: "Bicontinental", rule: () => "A line with stops on two continents", text: (a) => `${a.of} stops in ${andList(a.in ?? [])}` },
  { id: "long", name: "Longest road", rule: () => `A line ${bigDistance(LONG_M)} long`, text: (a) => `${a.of} is ${bigDistance(a.v ?? 0)} long` },
  { id: "money", name: "Money train", rule: () => `A line with ${money(MONEY_USD)} profit in a day`, text: (a) => `${a.of} made ${money(a.v ?? 0)} profit today` },
  { id: "bullet", name: "Bullet train", rule: () => `A train at ${speed(BULLET_KMH)}`, text: (a) => `${a.of} reaches ${speed(a.v ?? 0)}` },
  { id: "circle", name: "Round and round", rule: () => `A circle line of ${CIRCLE_STOPS} stops`, text: (a) => `${a.of} is a circle of ${a.v} stops` },
  { id: "change", name: "Change here", rule: () => `One change of trains per trip on average, with ${short(CHANGE_MIN_TRIPS)} trips/day`, text: (a) => `${(a.v ?? 0).toFixed(2)} changes per trip` },
  { id: "city", name: "Train town", rule: () => `A city with ${CITY_SHARE}% of its trips by train`, text: (a) => `${pct(a.v ?? 0)} of trips within ${a.of} go by train` },
  { id: "intercity", name: "Grounded", rule: () => `A city with ${INTERCITY_SHARE}% of its trips to other places by train`, text: (a) => `${pct(a.v ?? 0)} of trips between ${a.of} and other places go by train` },
  { id: "sleeper", name: "Sleeper", rule: () => `A rider on one train for ${SLEEPER_S / 3600} hours`, text: (a) => `Riders spend ${(a.v ?? 0).toFixed(1)} hours on ${a.of}` },
];

const WEEKDAY_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
/** "Monday, day 406, at 14:08" on the game clock. */
export function when(at: number): string {
  const t = Math.floor(at / 60), d = dayOf(t);
  return `${WEEKDAY_LONG[weekdayOf(d)]}, day ${d}, at ${clockTime(t)}`;
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

/** The figures that come from the network alone. */
export const networkFigures = computed(() => {
  const w = world.value;
  void zonesLoaded.value; // boundaries arriving re-place the stations
  if (!w) return null;
  const byId = stationsById(w);
  const served = new Set<CityId>();
  const firsts = new Map<FirstId, Detail>();
  let long: Detail | null = null, bullet: Detail | null = null, circle: Detail | null = null;
  for (const { l, st } of runningLines(w)) {
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
    if (countries.size > 1 && !firsts.has("border")) firsts.set("border", { of: l.name, in: [...countries].map(countryName) });
    if (continents.size > 1 && !firsts.has("continents")) firsts.set("continents", { of: l.name, in: [...continents] });
    const m = st.km * 1000;
    if (m >= LONG_M && (!long || m > long.v!)) long = { of: l.name, v: m };
    const top = Math.max(0, ...st.stretches.map((x) => x.reached));
    if (top >= BULLET_KMH && (!bullet || top > bullet.v!)) bullet = { of: l.name, v: top };
    const stops = l.stops.length - 1;
    if (st.loop === "circle" && stops >= CIRCLE_STOPS && (!circle || stops > circle.v!)) circle = { of: l.name, v: stops };
  }
  if (long) firsts.set("long", long);
  if (bullet) firsts.set("bullet", bullet);
  if (circle) firsts.set("circle", circle);
  return { cities: served.size, firsts, running: runningLines(w).length };
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

/** The figures from demand: final results only, on the average day, never a rough zone's. */
export const demandFigures = computed(() => {
  const w = world.value;
  void zonesLoaded.value;
  const d = demandByDay.value[AVERAGE_DAY];
  const v = d?.view;
  if (!w || !v?.final) return null;
  const rough = roughZones.value, byId = stationsById(w);
  const run = runningLines(w);
  const firsts = new Map<FirstId, Detail>();
  // Trips a day: the Cities tab's World row, coarse zones included (a big game holds most zones
  // coarse, and a total of millions is past the noise the star's rule guards against)
  const built = new Set<CityId>();
  for (const s of w.save.stations) if (s.built) { const z = stationZone(s); if (z) built.add(z); }
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
    const sums = new Map<number, { all: number; rail: number }>();
    const add = (p: number, all: number, rail: number) => {
      if (!places.has(p)) return;
      const e = sums.get(p) ?? { all: 0, rail: 0 };
      e.all += all; e.rail += rail;
      sums.set(p, e);
    };
    eachCorridor(corr, (a, b) => a !== b && (places.has(a) || places.has(b)), (j, x) => { add(corr.table.from[j], x.trips, x.rail); add(corr.table.to[j], x.trips, x.rail); });
    for (const [p, e] of sums) {
      if (e.all < INTERCITY_MIN_TRIPS) continue;
      const share = 100 * e.rail / e.all;
      if (share >= INTERCITY_SHARE && (!inter || share > inter.v!)) inter = { of: cityById(places.get(p)!)?.name, v: share };
    }
  }
  if (city) firsts.set("city", city);
  if (inter) firsts.set("intercity", inter);
  return { trips, station, line, firsts, covered };
});

/** The best line's profit over the past 24 hours (the Lines tab's Money figure), asked of the clock
 * worker as each game hour turns and with each new demand result; undefined until the first answer. */
export const bestProfit = signal<Detail | null | undefined>(undefined);
/** the average day's view the last answer was asked after */
let profitView: unknown = null;
let profitHour = NaN, profitGame = -1, profitAsked: unknown = null;
effect(() => {
  const m = moneyState.value, g = gamesLoaded.value, dv = demandByDay.value[AVERAGE_DAY]?.view ?? null;
  if (g !== profitGame) { profitGame = g; profitHour = NaN; profitAsked = profitView = null; bestProfit.value = undefined; }
  const hour = m?.hours.at(-1)?.hour;
  if (!m || m.mode === "sandbox" || hour === undefined || (hour === profitHour && dv === profitAsked)) return;
  profitHour = hour;
  profitAsked = dv;
  void queryClock({ what: "today" }).then((raw) => {
    const w = world.peek();
    if (!raw || !w || gamesLoaded.peek() !== g) return;
    const t = readToday(raw);
    let best: Detail | null = null;
    for (const { l, st } of runningLines(w)) {
      const d = t.get(l.id);
      if (!d) continue;
      const net = d.fares - runningToday(l, st) - st.upkeep;
      if (!best || net > best.v!) best = { of: l.name, v: net };
    }
    profitView = dv;
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
  const nf = networkFigures.value, df = demandFigures.value, bp = bestProfit.value;
  const out: Record<string, Detail> = {};
  if (!nf) return { met: out, ready: false };
  const tier = (id: TieredId, v: number | undefined, d: Detail) => {
    for (const t of TIERED.find((x) => x.id === id)!.tiers) if (v !== undefined && v >= t) out[tierKey(id, t)] = d;
  };
  tier("cities", nf.cities, { v: nf.cities });
  for (const [k, d] of nf.firsts) out[k] = d;
  if (df) {
    tier("trips", df.trips, { v: df.trips });
    if (df.station) tier("station", df.station.v, df.station);
    if (df.line) tier("line", df.line.v, df.line);
    for (const [k, d] of df.firsts) out[k] = d;
  }
  if (bp && bp.v! >= MONEY_USD) out.money = bp;
  // an older save's quiet working-out waits for the loaded game's own world, this network's own final
  // (a cached result shown ahead of it is solved again, `demandUpdating`) and the profit asked after it
  const settled = !!df?.covered && bp !== undefined && profitView === demandByDay.value[AVERAGE_DAY]?.view;
  return { met: out, ready: worldSinceLoad.value && !demandUpdating.value && (nf.running === 0 || settled) };
});

/** How many steps there are, and how many this game has earned. */
export const STEP_KEYS = new Set([...TIERED.flatMap((t) => t.tiers.map((x) => tierKey(t.id, x))), ...FIRSTS.map((f) => f.id)]);
export const STEPS = STEP_KEYS.size;
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
