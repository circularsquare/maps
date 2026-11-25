// What the city's demand came to, as the UI reads it (T-026; SPEC 4, 8). Written only by
// workers/demandClient.ts when a solve lands; components read the signals and the lookups below
// and never write them. The last finished result stays until a newer one replaces it (SPEC 1).
//
// Ids are the UI's: a line or station id as a string (the clock worker's numeric ids go through
// String()).

import { t } from "../i18n";
import { shareOf } from "./corridorShares";
import { computed, signal } from "@preact/signals";
import { activeCity, CITIES, cityById, REGISTRY, viewedZone, type CityId } from "./cities";
import { carryLoads, type StopKey } from "./carryLoads";
import { DEMAND_DAYS, type DemandDay } from "./days";
import type { TripLengths } from "./tripLengths";
import type { Fares } from "./fareCurve";
import { shownDay } from "./ui";
import type { PurposeFilter, PurposeKey } from "./purposes";
import { splitOf, type ModeSplit } from "./types";

export interface LineDemand {
  /** boardings onto the line a day, both directions, transfers included */
  ridersPerDay: number;
  /** fullest train at high demand: the worst segment's busiest-hour load in a high-demand
   * period, % of the crush capacity of the trains running then. A forecast average, not shown to
   * the player since T-230: the line panel shows the boarding account's trains instead */
  fullestPct: number;
  /** boardings a day at each stop, both directions, in the line's stop order */
  boardings: number[];
  /** riders per segment and period, `loads[period][segment]`: run 0's segments (stop k to
   * k + 1) and then run 1's in its own travel order; periods as game/periods.ts */
  loads: Float32Array[];
  /** boardings and alightings per period at each route node, `board[period][node]`: run 0's
   * stops, then run 1's, each in its travel order (2 x stops); the boarding account's input (T-164) */
  board?: Float32Array[];
  alight?: Float32Array[];
  /** T-1255: the riders coming to each platform, laid out as `board` and at least it; those beyond
   * it would ride if the trains had room, and a full platform turns them away */
  arrive?: Float32Array[];
  /** per period, US$: what the fare curve adds beyond its first stretch's base and rate on the
   * trips that ride this line, by km (T-1143, T-1516, the sim's `fare_extra`); absent or 0 while the curve is a
   * straight line */
  fareX?: number[];
  /** T-3578: per period, US$: what the lines' own multiples add past one base on the trips riding this line, by
   * km (the sim's `base_extra`); income adds the network's to the base fares `baseTrips` shares out */
  baseX?: number[];
  /** T-324: from the free-flow estimate of a solve still running. A line new or changed since the
   * shown final result gets these first figures early, and the final round replaces them. */
  estimate?: boolean;
  /** T-1419: the stops (station ids) the figures above are for, so they can be carried onto the
   * line's stops after an edit (`lineLoadsFor`); absent from a cached result */
  stops?: string[];
  /** T-1492: the trains an hour (T-3542: in each of the five periods) and cars the figures were
   * solved with; absent from a cached result */
  tph?: number[];
  cars?: number;
  /** T-2266, T-2286: the fares the figures were solved with, which its riders pay
   * (game/fareInEffect.ts); absent from a cached result */
  solvedFares?: Fares;
}

export interface StationDemand {
  boardings: number;
  alightings: number;
  /** of the boardings, riders changing lines here (T-1530), from the solve's results; absent where
   * the results carried none (a cached result written before, the purposes' stations) */
  transfers?: number;
}

export interface DemandView {
  /** the demand client's solve number this came from */
  version: number;
  /** the day it is for (T-260, T-464): a Saturday, a Sunday and the average are worked out from the weekday's solve */
  day: DemandDay;
  /** crowding rounds every period has had: 0 = the free-flow first estimate */
  round: number;
  /** the solve's last round: its figures are final. Earlier rounds are shown only as estimates,
   * where nothing final is shown yet (T-232); a re-solve after an edit keeps the last final
   * result until its own final round */
  final: boolean;
  /** Mode split, % of legs: network-wide here, all modelled local purposes in cityDemandViews. */
  split: ModeSplit;
  trainShareByPeriod: [string, number][];
  /** per period (game/periods.ts order), the busiest hour's riders against the period's mean
   * hour (1.4 in the peaks): segment loads / period hours x this = riders in the busiest hour */
  peakHourFactor: number[];
  tripsPerDay: number;
  railTripsPerDay: number;
  lines: Map<string, LineDemand>;
  stations: Map<string, StationDemand>;
  /** The same lines and stations by trip purpose, for the map's purpose filter (T-158); the
   * purposes add up to `lines` and `stations`. Absent purposes carried no riders. Keyed by the
   * shown purposes (game/purposes.ts `PURPOSES`): "school" counts university too (T-192). */
  purposes?: Map<PurposeKey, PurposeDemand>;
  /** T-1300: rail trips a day per curve dollar dearer on every fare (negative), the local layers'
   * and the market's; each one's reach (its sensitivity-weighted mean trip km over its
   * rider-weighted one); and the market's rail trips a day (the rest of `railTripsPerDay` is local) */
  fareSense?: { local: number; market: number; localReach: number; marketReach: number; marketRail: number };
}

/** One purpose's share of `DemandView.lines` and `.stations` (T-158). */
export interface PurposeDemand {
  /** per line id: boardings a day and riders per segment and period, as `LineDemand` */
  lines: Map<string, { ridersPerDay: number; boardings: number[]; loads: Float32Array[] }>;
  stations: Map<string, StationDemand>;
}

/** T-1279: whether `dv` has riders for the line as its stops are now; until it does, the boarding
 * account runs its trains empty, so their crowding is not known (the map draws them striped). */
export function lineFiguresIn(dv: DemandView | null, lineId: string, stops: number): boolean {
  const loads = dv?.lines.get(lineId)?.loads;
  return !!loads?.length && stops >= 2 && loads.every((p) => p.length === 2 * (stops - 1));
}

/** A line's riders per segment and period as the map draws them: one purpose's, or all. With
 * `now`, the line's stops as they are, and figures for other stops than those, the figures carried
 * onto `now` (T-1419, `carryLoads`) while demand solves for the edit; `key` makes a station and its
 * complex one place. */
export function lineLoadsFor(dv: DemandView | null, lineId: string, purpose: PurposeFilter = "all", now?: readonly string[], key?: StopKey): Float32Array[] | undefined {
  if (!dv) return undefined;
  const d = dv.lines.get(lineId);
  // a purpose with no riders anywhere has no entry: zeros, not "unknown"
  const loads = purpose === "all" ? d?.loads : dv.purposes?.get(purpose)?.lines.get(lineId)?.loads ?? d?.loads.map((p) => new Float32Array(p.length));
  if (!now || !d?.stops || !loads?.length || loads.every((p) => p.length === 2 * (now.length - 1))) return loads;
  const id = now.join(",");
  const had = carried.get(loads);
  if (had?.now === id) return had.out ?? undefined;
  const out = carryLoads(loads, d.stops, now, key);
  carried.set(loads, { now: id, out });
  return out ?? undefined;
}
const carried = new WeakMap<Float32Array[], { now: string; out: Float32Array[] | null }>();

/** The regional layer (T-117, T-276): trips a day include the commutes between zones; each
 * corridor's trips and rail are both parts together, `commutes` and `commuteRail` the commute legs;
 * T-378: `air` and `car` the rest by plane and by car, `elsewhere` each place's longer trips with
 * places that have no zone, each way. */
export interface IntercityDemand { version: number; round: number; tripsPerDay: number; railTripsPerDay: number; commuteTripsPerDay?: number; commuteRailPerDay?: number; corridors: Corridors; elsewhere?: number[];
  /** T-3503: each place's everyday outings across a zone line with other places, five numbers a
   * place with any (place, legs with an end there, rail legs, walking legs, and (T-2216) legs between two
   * lands that no rail takes, which have no way) */
  everyday?: number[] }
/**
 * The market's corridors with trips (T-789: millions of place pairs; 8.5M on the New York save in
 * 2026-11), the part no network or period changes. T-2136: the page keeps no table of them; it has
 * each market's book, and asks the workers for a few places' rows when a view needs them (`rowsOf`).
 * One book serves every result of a market (T-1008).
 */
export interface CorridorBook {
  key: string; n: number;
  /** per market place, 3 numbers over the rows between it and another place: their longer trips,
   * commute legs, and longer trips times the share flying, before a kind of day's scales */
  totals: Float64Array;
  /** T-3509: per market place, 2 numbers from its own row (between its own zones): its longer trips
   * (32-bit, as the rows hold them) and commute legs, before a kind of day's scales (`withinOf`) */
  own: Float64Array;
  /** T-2216: per market place, the share of its own row's trips between zones on two lands (island zones),
   * which have no car; 0 for most places */
  apart: Float64Array;
}
/** Some of a market's table rows (`rowsOf`), corridor ascending: the corridor (`Corridors.railAt`'s
 * numbers), its two places, its longer trips and commute legs a day before a kind of day's scales,
 * and the shares of its longer trips not by rail that fly and that drive (T-378). */
export interface CorridorRows {
  key: string; n: number; at: Uint32Array; from: Uint32Array; to: Uint32Array;
  /** 32-bit (T-3493: the world's total moves by 2e-10 of itself) */
  longer: Float32Array;
  commute: Float64Array;
  /** in steps of 1 / SHARE_ONE (`shareOf`, game/corridorShares.ts) */
  air: Uint16Array; car: Uint16Array;
}
/** A kind of day's corridors: the book's longer trips and commute legs times `longerScale` and
 * `commuteScale`, and the table rows with rail (`railAt`, by corridor, ascending) their longer trips
 * and commute legs by rail, their two places and share flying (`shareOf`). */
export interface Corridors { book: CorridorBook; longerScale: number; commuteScale: number; railAt: Uint32Array; longerRail: Float64Array; commuteRail: Float64Array; railFrom: Uint32Array; railTo: Uint32Array; railAir: Uint16Array }

export const NO_CORRIDORS: Corridors = {
  book: { key: "", n: 0, totals: new Float64Array(0), own: new Float64Array(0), apart: new Float64Array(0) },
  longerScale: 0, commuteScale: 0, railAt: new Uint32Array(0), longerRail: new Float64Array(0), commuteRail: new Float64Array(0), railFrom: new Uint32Array(0), railTo: new Uint32Array(0), railAir: new Uint16Array(0),
};
const NO_ROWS: CorridorRows = { key: "", n: 0, at: new Uint32Array(0), from: new Uint32Array(0), to: new Uint32Array(0), longer: new Float32Array(0), commute: new Float64Array(0), air: new Uint16Array(0), car: new Uint16Array(0) };

/** One corridor's figures a day: all trips (longer and commute legs), by rail, by plane, by car,
 * (T-2216) with no way, and the commute legs alone. */
export interface CorridorFigures { trips: number; rail: number; air: number; car: number; none: number; commutes: number; commuteRail: number; commuteNone: number }

/** Row `i` of `r`'s figures; `k` its place in `railAt` (-1 none; looked up when left out). */
export function corridorAt(c: Corridors, r: CorridorRows, i: number, k = railIndex(c, r.at[i])): CorridorFigures {
  const tl = r.longer[i] * c.longerScale, cm = r.commute[i] * c.commuteScale;
  const lr = k >= 0 ? c.longerRail[k] : 0, cr = k >= 0 ? c.commuteRail[k] : 0, rest = tl - lr;
  const air = shareOf(r.air[i]), car = shareOf(r.car[i]);
  // T-2216: what is left of the longer trips after the plane and the car has no way but rail (two landmasses
  // with no plane, or zones on two lands); commutes never fly, so with no car at all they have none either
  const noneL = air + car <= 1 ? rest * (1 - air - car) : 0;
  const noneC = r.car[i] === 0 && !(air > 0.999) ? cm - cr : 0;
  return { trips: tl + cm, rail: lr + cr, air: rest * air, car: rest * car + cm - cr - noneC, none: noneL + noneC, commutes: cm, commuteRail: cr, commuteNone: noneC };
}

/** Corridor `j`'s place in `railAt`, -1 when it has no rail. */
export const railIndex = (c: Corridors, j: number): number => sortedIndex(c.railAt, j);

/** `j`'s place in the ascending `a`, -1 when it is not there. */
export function sortedIndex(a: Uint32Array, j: number): number {
  let lo = 0, hi = a.length - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (a[m] === j) return m;
    if (a[m] < j) lo = m + 1; else hi = m - 1;
  }
  return -1;
}

/** Every row of `r` whose places `keep` accepts, in table order, with its figures. */
export function eachCorridor(c: Corridors, r: CorridorRows, keep: (from: number, to: number) => boolean, f: (from: number, to: number, v: CorridorFigures) => void) {
  const { from, to, n } = r, at = c.railAt;
  for (let i = 0, k = 0; i < n; i++) {
    while (k < at.length && at[k] < r.at[i]) k++;
    if (keep(from[i], to[i])) f(from[i], to[i], corridorAt(c, r, i, k < at.length && at[k] === r.at[i] ? k : -1));
  }
}

/** T-2136: where rows come from (the demand client's `corridorRows`: a solving worker that holds
 * the market); null with none of the book's market open there. */
let rowsSource: ((key: string, place: number) => Promise<CorridorRows | null>) | null = null;
export const setRowsSource = (f: typeof rowsSource) => { rowsSource = f; };
/** bumped as asked rows land, so the views that read `rowsOf` work out again */
export const rowsLanded = signal(0);
/** the places' rows asked for lately, by book key and place; a place's are about 80 kB on the New York
 * save (one place an ask, so the page never holds more than ROWS_KEPT places' rows) */
const rowsKept = new Map<string, { rows: CorridorRows | null; failedFor?: Corridors; asking?: Promise<CorridorRows | null> }>();
const ROWS_KEPT = 48;
/**
 * The table rows of `c`'s market touching `place`, or null while they are asked of the workers
 * (`rowsLanded` then moves); asked again for a newer result, or once a worker sends the book
 * (`rowsCanCome`), if no worker had them. Reads `rowsLanded`, so a computed or component reading it
 * works out again when they land.
 */
export function rowsOf(c: Corridors, place: number): CorridorRows | null {
  void rowsLanded.value;
  if (!c.book.key) return NO_ROWS;
  const id = `${c.book.key}|${place}`;
  const e = rowsKept.get(id);
  if (e) {
    rowsKept.delete(id);
    rowsKept.set(id, e);
    if (e.rows || e.asking || e.failedFor === c) return e.rows;
  }
  void askRows(c, id, place);
  return null;
}
/** `rowsOf` once they are in (null if no worker has them). */
export function rowsNow(c: Corridors, place: number): Promise<CorridorRows | null> {
  const got = rowsOf(c, place);
  if (got || !c.book.key) return Promise.resolve(got);
  return rowsKept.get(`${c.book.key}|${place}`)?.asking ?? Promise.resolve(null);
}
function askRows(c: Corridors, id: string, place: number): Promise<CorridorRows | null> {
  const e: { rows: CorridorRows | null; failedFor?: Corridors; asking?: Promise<CorridorRows | null> } = { rows: null };
  rowsKept.set(id, e);
  while (rowsKept.size > ROWS_KEPT) rowsKept.delete(rowsKept.keys().next().value!);
  e.asking = (rowsSource ? rowsSource(c.book.key, place) : Promise.resolve(null)).catch(() => null).then((rows) => {
    e.asking = undefined;
    e.rows = rows;
    if (!rows) e.failedFor = c;
    rowsLanded.value++;
    return rows;
  });
  return e.asking;
}
/** The rows held of the book with `key`, `first`'s first, at most `most` (the demand cache's). */
export function heldRows(key: string, first: number, most: number): { place: number; rows: CorridorRows }[] {
  const out: { place: number; rows: CorridorRows }[] = [];
  for (const [id, e] of rowsKept) {
    const [k, p] = id.split("|");
    if (k === key && e.rows) out.push({ place: Number(p), rows: e.rows });
  }
  out.sort((a, b) => Number(b.place === first) - Number(a.place === first));
  return out.slice(0, most);
}
/** Rows kept with a cached result (`heldRows`), as if asked and answered. */
export function seedRows(got: { place: number; rows: CorridorRows }[]) {
  for (const { place, rows } of got) {
    const id = `${rows.key}|${place}`;
    if (!rowsKept.get(id)?.rows) rowsKept.set(id, { rows });
  }
  while (rowsKept.size > ROWS_KEPT) rowsKept.delete(rowsKept.keys().next().value!);
  if (got.length) rowsLanded.value++;
}
/** A worker has sent the book with `key` (it holds that market now): the rows no worker had for it
 * are asked again by the views that want them (a cached result's, shown before any worker held it). */
export function rowsCanCome(key: string) {
  let any = false;
  for (const [id, e] of rowsKept) if (!e.rows && !e.asking && id.startsWith(key + "|")) { rowsKept.delete(id); any = true; }
  if (any) rowsLanded.value++;
}

/** One kind of day's results (T-260, game/days.ts): the network's view, each city's, the market's.
 * Written together by workers/demandClient.ts. */
export interface DayDemand {
  view: DemandView | null;
  cities: Map<CityId, DemandView>;
  intercity: IntercityDemand | null;
}
const noDay = (): DayDemand => ({ view: null, cities: new Map(), intercity: null });
const firstDays = DEMAND_DAYS.map(noDay);
/** The newest finished solve for each day, `DemandDay` order: weekday, Saturday, Sunday, average: the figures shown,
 * with an edit's quick figures from a side job over the main pool's while it lasts (T-4220). */
export const demandByDay = signal<DayDemand[]>(firstDays);
/** T-4307: the main pool's figures alone, `demandByDay` without a side job's over them. Whatever is kept (money, the
 * boarding account's riders, points, achievements, stars) reads these, so quick figures never feed a save. */
export const mainDemandByDay = signal<DayDemand[]>(firstDays);

/** The newest finished solve on the day shown (`shownDay`); null until the first lands. */
export const demandView = computed<DemandView | null>(() => demandByDay.value[shownDay.value]?.view ?? null);
/** Each zone's trips within it on the day shown (`zoneViews`), separate from network-wide totals. */
export const cityDemandViews = computed<Map<CityId, DemandView>>(() => zoneViews(demandByDay.value[shownDay.value]));

/** T-3509: each kind of day's rail row within each place (`railFrom === railTo`), by place */
const withinRail = new WeakMap<Corridors, Map<number, number>>();
/** T-3509: the market's legs a day between two 6 km zones of place `place` (its longer trips 25 km and
 * more apart, and the long commutes that stay in it), and those by rail; null with no market result.
 * T-2136: from the book's own-row sums and the rail rows, no table rows asked. */
export function withinOf(ic: IntercityDemand | null | undefined, place: number): { trips: number; rail: number; none: number } | null {
  const c = ic?.corridors;
  const own = c?.book.own;
  if (!c || !own || place < 0 || 2 * place + 1 >= own.length) return null;
  const trips = own[2 * place] * c.longerScale + own[2 * place + 1] * c.commuteScale;
  if (!(trips > 0)) return null;
  let rail = withinRail.get(c);
  if (!rail) {
    rail = new Map();
    for (let k = 0; k < c.railAt.length; k++) if (c.railFrom[k] === c.railTo[k]) rail.set(c.railFrom[k], k);
    withinRail.set(c, rail);
  }
  const k = rail.get(place);
  const r = k === undefined ? 0 : c.longerRail[k] + c.commuteRail[k];
  // T-2216: the share between zones on two lands has no car, so what its rail leaves has no way (the rail
  // is taken as spread over the row evenly)
  const apart = c.book.apart?.[place] ?? 0; // (a book cached before this has none)
  return { trips, rail: Math.min(r, trips), none: apart * Math.max(0, trips - r) };
}

/** T-3509 (Anita, 2026-11-18: "yes, all trips within should count"): zone `id`'s view on day `d` with
 * the market's trips between its own far-apart parts (`withinOf`) added to its local ones: its Trips
 * within, the figure its train share, goals and rides read. Driving unless by rail, as its bubbles
 * count them; the per-period shares stay the local trips'. */
export function zoneView(d: DayDemand | undefined, id: CityId): DemandView | undefined {
  const v = d?.cities.get(id);
  const w = v && withinOf(d?.intercity, cityById(id)?.place ?? -1);
  if (!v || !w) return v;
  const trips = v.tripsPerDay + w.trips, rail = v.railTripsPerDay + w.rail;
  const walk = (v.tripsPerDay * v.split.walk) / 100, none = (v.tripsPerDay * (v.split.none ?? 0)) / 100 + w.none;
  return { ...v, tripsPerDay: trips, railTripsPerDay: rail, split: splitOf(trips, rail, walk, none) };
}
/** T-3509: day `d` has the market's result (its corridor book), so `zoneView` holds the trips between
 * a zone's own parts. A load shows final local figures before it: a goal waits for it. */
export const marketIn = (d: DayDemand | undefined): boolean => (d?.intercity?.corridors?.book.n ?? 0) > 0;
/** `zoneView` of every zone with a view on day `d`. */
export function zoneViews(d: DayDemand | undefined): Map<CityId, DemandView> {
  const out = new Map<CityId, DemandView>();
  if (d) for (const id of d.cities.keys()) out.set(id, zoneView(d, id)!);
  return out;
}
/** Distinct, synthetic intercity trips: never included in local commute denominators. */
export const intercityDemand = computed<IntercityDemand | null>(() => demandByDay.value[shownDay.value]?.intercity ?? null);
/** A solve is running for a newer network than the one `demandView` shows (or, T-328, the shown
 * figures are the cache's and the same network is being solved again). */
export const demandUpdating = signal(false);
/** T-2658: the world version (`WorldView.version`) as the newest network reached demand, and as the
 * network of the whole final shown did: what the player sees at a world version is solved once the
 * second reaches it. -1 before any. */
export const networkWorld = signal(-1);
export const finalWorld = signal(-1);
/** T-2409: the game minute (world time) of the oldest edit no final result covers yet, or null */
export const demandBehindFrom = signal<number | null>(null);
/** How far that solve has got: results published so far (the free-flow estimate, then each
 * crowding round) of all it will publish. */
export const demandStep = signal({ done: 0, of: 4 });
/** T-2729: an edit's solve, whose results land at each stage's end: every solving worker's last `work`
 * (demandProtocol.ts) and when it came (performance.now()), or null before each has sent one (and for
 * other solves) */
export type WorkPart = { stage: 0 | 1; busy: boolean; done: [number, number]; of: [number, number]; unitMs: [number, number]; at: number;
  /** T-3599 (demandProtocol.ts `work`) */
  left: [number, number]; blind: [number, number]; cur: number | null; curSize: number; sizeMs: [number, number]; blindLate?: [number, number]; viewBlindLate?: [number, number]; curLate?: boolean; sizeMsLate?: [number, number]; ran: [number, number]; ranExp: [number, number]; ranN: [number, number]; viewLeft: [number, number]; viewBlind: [number, number]; curView: boolean; leftQ: number[]; ranQ: number[]; expQ: number[]; nQ: number[]; curQ: number };
export const demandWork = signal<WorkPart[] | null>(null);
/** T-3599: a newer network waits for the running edit's solve to end (T-1734), and that solve was sent at
 * `sentAt` (performance.now()); null otherwise */
export const editHeld = signal<{ sentAt: number } | null>(null);
/** T-3925: counts the edits' solves stopped part way by a newer network: the calculating bar starts again */
export const solveRestarted = signal(0);
/** before any unit has been measured, the time over which the unit in hand goes 63% of the way to 90%
 * of a unit, ms: it slows rather than stops, since a first unit can take 15 s (New York's real save) */
const UNIT_MS = 4000;
/** The share of stage `stage` done at `now`: units finished, and the unit in hand by the time it has
 * taken against the stage's mean (to 90% of it); null where the stage has no units. */
export function workShare(parts: WorkPart[], stage: 0 | 1, now: number): number | null {
  let done = 0, of = 0;
  for (const p of parts) {
    of += p.of[stage];
    done += p.done[stage];
    const t = now - p.at, mean = p.unitMs[stage];
    if (p.busy && p.stage === stage) done += mean ? Math.min(0.9, t / mean) : 0.9 * (1 - Math.exp(-t / UNIT_MS));
  }
  return of > 0 ? Math.min(1, done / of) : null;
}
/** T-1904: how many zones the running solve covers (a load's, or an edit's whole solve), or 0 for
 * a solve of the zone in view alone (a rough solve, an edit's local solve): the status tag says
 * which, since a whole solve of a big game takes much longer */
export const demandScope = signal(0);
/** T-392: the zones the running solve has still to finish (each cluster's land as it is solved);
 * null until the workers say, then read as every zone */
export const demandSolving = signal<Set<CityId> | null>(null);
/** T-1223: the zones whose shown figures are from a coarse solve (notes/T-1206.md): rough, and
 * never a star's */
export const roughZones = signal<ReadonlySet<CityId>>(new Set());
/** T-1223: the zone in view, shown rough, being worked out in detail (its fine solve not shown yet) */
export const refiningZone = signal<CityId | null>(null);
/** T-2180: the player's edits are being solved for the edit note (game/editNote.ts); zones keep their
 * levels of detail meanwhile, so the note compares like with like */
export const editsSettling = signal(false);
/** T-3427: a network from the player's edits waits for the game to run before it is solved ("Recalculate
 * trips while paused" off), so the figures shown are from before those edits */
export const demandHeld = signal(false);
/** T-2045: the zones shown from their own solve during a game load (the focus pool), true once it
 * has its last round: settled, which counts as final until the load's whole solve replaces it */
export const focusZones = signal<ReadonlyMap<CityId, boolean>>(new Map());
/** T-2453: counts the times a view's questions change where they go while the shown version and round
 * stay (a load's whole solve sent, a round of a cluster solved ahead or of the focus pool shown): a
 * panel that asks once per shown round asks again */
export const demandAskTurn = signal(0);
/** A solve is running that will still change the figures of the zone in view. */
export const viewUpdating = computed(() => demandUpdating.value && !focusZones.value.get(activeCity.value) && (demandSolving.value?.has(activeCity.value) ?? true));
/** T-421: the figures shown for the zone in view are final: the shown result is, or (T-402) the
 * zone's cluster has its final while other clusters still solve and the view is not yet the solve's */
export const viewFinal = computed(() => {
  const v = demandView.value;
  return !!v && (v.final || (!viewUpdating.value && cityDemandViews.value.has(activeCity.value)));
});

/** T-1851: the trips shown for the zone in view are being recomputed (rough, early, from before an
 * edit, or its detailed solve on the way). It drives the status tag (never the clock's cap, T-2622), and
 * is false once the zone in view is final while other zones still solve. */
export const viewRecomputing = computed(() => !!refiningZone.value || !viewFinal.value || viewUpdating.value);

/** The mode-split bar's numbers: the viewed zone's own figures (a solve's, or the bare pass's for a
 * zone no station reaches), null until it has them and over dead space (T-318). */
export const modeSplit = computed<ModeSplit | null>(() => !viewedZone.value ? null : cityDemandViews.value.get(activeCity.value)?.split ?? null);

/** T-342: the same split in rides (trip legs) a day, `all` the total of the three. It is what
 * `modeSplit` is a percentage of (the zone's modelled local legs, `DemandView.tripsPerDay`); null
 * until the zone has a solved or bare view, and over dead space. */
export const modeRides = computed<(ModeSplit & { all: number }) | null>(() => {
  const v = !viewedZone.value ? null : cityDemandViews.value.get(activeCity.value);
  return v ? ridesOf(v) : null;
});

/** A view's split in rides (trip legs) a day, `all` their total: its `split` times its trips. */
export function ridesOf(v: DemandView): ModeSplit & { all: number } {
  const all = v.tripsPerDay;
  return { all, train: v.railTripsPerDay, walk: all * v.split.walk / 100, drive: all * v.split.drive / 100, none: all * (v.split.none ?? 0) / 100 };
}

export const lineDemand = (id: string): LineDemand | null => demandView.value?.lines.get(id) ?? null;
export const stationDemand = (id: string): StationDemand | null => demandView.value?.stations.get(id) ?? null;

// --- Data for the demand views (T-078 for T-084; notes/T-078.md) --------------------------------
//
// Asked for on demand, not pushed with every solve: the workers compute them from the solve that
// `demandView` shows and transfer typed arrays (no per-cell objects). Units are commuters a day:
// one person making a trip to work and one home. Every per-cell array is in the city pack's cell
// order (`CityCells`). A query resolves to null while there is no solve yet, or when the workers
// have moved on to a newer network than the one shown; ask again when `demandView` changes.

/** The city's cells and zones. Static for a city: ask once. */
export interface CityCells {
  /** number of cells */
  n: number;
  /** x, y per cell in the loaded game's world frame (game/coords.ts turns them into lng/lat) */
  xy: Float32Array;
  /** H3 index (resolution 9) of each cell */
  h3: BigUint64Array;
  /** gravity zone of each cell (2 km squares, the unit of where people work) */
  zone: Uint32Array;
  /** zone side in metres, and x, y of each zone's centre (2 per zone) */
  zoneM: number;
  zoneXY: Float32Array;
}

/** Local trip legs a day per cell by mode, home and activity ends, both directions. */
export interface CellModes {
  /** the `DemandView.version` these belong to */
  version: number;
  /** and its day (T-260, T-464) */
  day: DemandDay;
  /** living in the cell: by rail, walking the whole way, driving (car, bus, taxi) */
  homeRail: Float32Array;
  homeWalk: Float32Array;
  homeDrive: Float32Array;
  /** activity destinations (work plus implemented everyday purposes), the same three */
  workRail: Float32Array;
  workWalk: Float32Array;
  workDrive: Float32Array;
  /** T-2214, T-2215: trips no way serves (between two land parts, not by rail), "No route", at
   * each end; in none of the three above */
  homeNone: Float32Array;
  workNone: Float32Array;
}

/** Cells and the commuters from each, most first. */
export interface CellRiders {
  cells: Uint32Array;
  riders: Float32Array;
  total: number;
}

/** Zones and commuters, most first (at most `top` of them). */
export interface ZoneRiders {
  zones: Uint32Array;
  riders: Float32Array;
}

/**
 * One station's riders (`stationRiders(id)`). Stations whose platforms are within 100 m of each
 * other are one station for demand (SPEC 4.4), so a pair of them answers the same. Asked for a
 * station complex (T-146, `members`), the answer is the sum over its distinct demand stations.
 */
export interface StationRiders {
  version: number;
  stationId: string;
  /** the stations summed: `members.join(",")`, just the station's id outside complex view */
  scope: string;
  /** commuters living in each cell who use this station at the home end (its catchment) */
  home: CellRiders;
  /** commuters working in each cell who use this station at the work end */
  work: CellRiders;
  /** the home riders (commutes) by how they reach the station: on foot, or by a ride (car, bus,
   * bike or lift, T-371); commuters a day, adding up to `home.total` */
  walked: number;
  rode: number;
  /** where the home riders work, by zone */
  homeRidersWorkIn: ZoneRiders;
  /** where the work riders live, by zone */
  workRidersLiveIn: ZoneRiders;
}

/**
 * Where a set of cells' commuters go (`flows(end, cells)`, T-098 for the bubbles, notes/T-098.md):
 * everyone living (end "home") or working (end "work") in the selected cells, every mode, spread
 * over the cells at the other end. Units as `CellModes`: commuters a day.
 */
export interface Flows {
  version: number;
  end: "home" | "work";
  /** the selected commuters by mode; the same as `cellModes` summed over the selected cells */
  rail: number;
  walk: number;
  drive: number;
  /** cells at the other end with commuters from (or to) the selection, pack order */
  cells: Uint32Array;
  /** per far-end cell, commuters a day */
  farRail: Float32Array;
  farWalk: Float32Array;
  farDrive: Float32Array;
  /** commuters in the far cells left out (the smallest, together 1% of the selection), by mode:
   * the far cells plus these add up to rail, walk, drive */
  cutRail: number;
  cutWalk: number;
  cutDrive: number;
  /** Same local trip legs as the bubbles, separated by activity and departure direction. */
  purposes: PurposeTrips[];
  /** The far end and totals of each local purpose on its own, for the map's purpose filter
   * (T-158); the fields mean what the ones above mean. Absent: the selection has none of it. */
  byPurpose?: Map<PurposeKey, FarEnd>;
}

export type FarEnd = Pick<Flows, "rail" | "walk" | "drive" | "cells" | "farRail" | "farWalk" | "farDrive" | "cutRail" | "cutWalk" | "cutDrive">;

export interface PurposeTrips {
  name: string;
  /** Five departure periods per mode: rail, walk, drive. */
  outward: number[][];
  returning: number[][];
}

/** T-3242: a place's name is its zone's shown name (zone-names.json, T-825), so the long-trip lists
 * name it as the map does; the market's own name is the pipeline's, kept for a place with no zone. */
const SHOWN = new Map(CITIES.filter(c => c.place >= 0).map(c => [c.place, c.name]));
/** Registered places in pipeline/intercity.py order: the indices in the arrays below. */
export const INTERCITY_PLACES = REGISTRY.market.places.map((p, i) => SHOWN.get(i) ?? p.name);
/** The places as the player reads them. `own` is the place the selection itself lies in: the trips
 * that stay within it are all 40 km or longer (the market has none shorter). */
export function intercityPlaceName(place: number, own = -1): string {
  const p = REGISTRY.market.places[place];
  const shown = SHOWN.get(place);
  const name = !p ? "" : shown !== undefined && p.regionName.startsWith(p.name) ? shown + p.regionName.slice(p.name.length) : p.regionName;
  return place === own ? t("demand.otherPlaces", { name }) : name;
}

/** Longer trips (non-commute, T-117) to or from one place, a day. */
export interface PlaceTrips {
  /** index into INTERCITY_PLACES */
  place: number;
  /** all modes, and by rail; T-378: by plane and by car (a station's: 0) */
  all: number;
  rail: number;
  air: number;
  car: number;
  /** T-1229: walked the whole way, the everyday outings across a zone line only (absent: none) */
  walk?: number;
  /** where the trips' far ends are, mean position in the world frame (NaN with no trips) */
  x: number;
  y: number;
}

/** Longer trips to or from one far 6 km zone, a day (T-120): drawn as one bubble. */
export interface ZoneTrips {
  /** Exact market membership from the demand pack, not inferred from a zone's centroid. */
  place: number;
  /** the zone's global id in the market (T-866: the trip explorer asks about it) */
  gid: number;
  /** the zone's own position, pack metres (as `PlaceTrips.x`) */
  x: number;
  y: number;
  /** all modes, and by rail (a station's: both the riders); T-1229: and walking */
  all: number;
  rail: number;
  walk?: number;
}

/** One direction of a selection's or a station's longer trips, by place at the far end. Since
 * T-1229 a selection's also hold its everyday outings across a zone line (`walk` only theirs). */
export interface PlaceFlows {
  all: number;
  rail: number;
  air: number;
  car: number;
  walk?: number;
  /** T-378: longer trips with places that have no zone (not in `all`) */
  elsewhere: number;
  /** the places with trips, biggest (all modes) first */
  places: PlaceTrips[];
  /** the far zones with trips, biggest first; their sums equal the places' */
  zones: ZoneTrips[];
}

/**
 * A bubble selection's longer trips (`intercitySelection(end, cells)`, T-118): the market's own
 * trips from the 6 km zones the selected cells lie in, each zone's share given to the 600 m squares
 * by people (starts) or attraction (ends) and the selected part of every square counted. Units:
 * trips a day, and `rail` the part by train. Not commuters.
 */
export interface IntercitySelection {
  version: number;
  /** trips starting in the selected cells, by where they end */
  starts: PlaceFlows;
  /** trips ending in the selected cells, by where they start */
  ends: PlaceFlows;
  /** share of the selected squares that the market has sites for (1 = all) */
  matched: number;
  /** T-1229: each shown purpose's everyday outings across a zone line alone, which the purpose
   * filter draws as the far end (the longer trips and the commutes have no entry) */
  byPurpose?: Map<PurposeKey, { starts: PlaceFlows; ends: PlaceFlows }>;
}

/** A station's longer-trip rail riders (`intercityStation(id)`): starting here, ending here, by place. */
export interface IntercityStation {
  version: number;
  stationId: string;
  /** as `StationRiders.scope` */
  scope: string;
  starts: PlaceFlows;
  ends: PlaceFlows;
}

/** Riders changing from one line at one station to another line at another (or the same) station
 * a day, every period and purpose (T-176). Both ends name a snapshot station and line; a walking
 * transfer has two different stations. */
export interface TransferPair {
  fromStation: string;
  fromLine: string;
  toStation: string;
  toLine: string;
  riders: number;
}

/** The transfers at a station complex (`complexTransfers(members)`): every pair with either end at
 * a member, both directions of a line merged. */
export interface ComplexTransfers {
  version: number;
  round: number;
  /** as `StationRiders.scope`: the members asked about, joined by commas */
  scope: string;
  pairs: TransferPair[];
}

export interface DemandQueries {
  transfers(members: string[]): Promise<ComplexTransfers | null>;
  platformTransfers(): Promise<Float32Array | null>;
  cityCells(city?: CityId): Promise<CityCells | null>;
  cellModes(purpose?: PurposeFilter, city?: CityId): Promise<CellModes | null>;
  stationRiders(stationId: string, top?: number, members?: string[]): Promise<StationRiders | null>;
  flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null>;
  intercitySelection(end: "home" | "work", cells: Uint32Array, still?: () => boolean): Promise<IntercitySelection | null>;
  intercityStation(stationId: string, members?: string[]): Promise<IntercityStation | null>;
  tripLengths(zone?: CityId): Promise<TripLengths | null>;
  legsKey(zone?: CityId): string;
}

let queries: DemandQueries | null = null;
/** Set by workers/demandClient.ts when the pool starts. */
export function setDemandQueries(q: DemandQueries) {
  queries = q;
}
/** The city's cells (null until the pool has opened the city). `city`: another zone's than the
 * focused one (a nearby zone's bubbles, T-276); it waits for that zone to open. */
export const cityCells = (city?: CityId): Promise<CityCells | null> => queries?.cityCells(city) ?? Promise.resolve(null);
/** T-2365: a zone's local trips a day by length on the day shown (game/tripLengths.ts); null while unknown. */
export const zoneTripLengths = (zone?: CityId): Promise<TripLengths | null> => queries?.tripLengths(zone) ?? Promise.resolve(null);
/** Local trip legs a day per cell by mode, for the solve `demandView` shows. One 4.9 MB answer in
 * New York. `purpose`: one purpose's legs only (T-158); longer trips have no local legs (zeros).
 * `city`: a nearby zone's (T-276), not cached; null while that zone is not in the solve shown. */
export const cellModes = (purpose: PurposeFilter = "all", city?: CityId): Promise<CellModes | null> => queries?.cellModes(purpose, city) ?? Promise.resolve(null);
/** T-2508: which solve `cellModes` would read the zone's legs from now (bubbles built from another are stale) */
export const legsKey = (city?: CityId): string => queries?.legsKey(city) ?? "";
/** A station's catchment and where its riders go, for the solve `demandView` shows; `top` zones
 * per direction (default 20). A few kB; ask on click. `members`: a station complex's stations,
 * summed (T-146). */
export const stationRiders = (stationId: string, top?: number, members?: string[]): Promise<StationRiders | null> =>
  queries?.stationRiders(stationId, top, members) ?? Promise.resolve(null);
/** Where the commuters living (or working) in `cells` (pack cell ids) work (or live), by mode, for
 * the solve `demandView` shows. Asked on click; null while there is no solve or the workers have
 * moved on to a newer network. */
export const flows = (end: "home" | "work", cells: Uint32Array): Promise<Flows | null> =>
  queries?.flows(end, cells) ?? Promise.resolve(null);
/** The longer (non-commute) trips of the people living or working in `cells`, by place, for the
 * solve `demandView` shows. Null while there is no solve or the workers moved on, or once `still` says
 * the selection is no longer wanted (T-2746). */
export const intercitySelection = (end: "home" | "work", cells: Uint32Array, still?: () => boolean): Promise<IntercitySelection | null> =>
  queries?.intercitySelection(end, cells, still) ?? Promise.resolve(null);
/** A station's longer-trip riders by place, or a complex's (`members`). Null as above. */
export const intercityStation = (stationId: string, members?: string[]): Promise<IntercityStation | null> =>
  queries?.intercityStation(stationId, members) ?? Promise.resolve(null);
/** Transfers a day at a station complex's members (T-176), for the solve `demandView` shows. A few
 * kB; asked when the complex map shows. Null while there is no solve or the workers moved on. */
export const complexTransfers = (members: string[]): Promise<ComplexTransfers | null> =>
  queries?.transfers(members) ?? Promise.resolve(null);
/** Every change between platforms a day, for the solve `demandView` shows (T-554): per pair the
 * line, run and stop (in the run's order) riders leave, the same where they board, and riders, as
 * `TrackApi.set_transfers` reads them. Null while there is no solve or the workers moved on. */
export const platformTransfers = (): Promise<Float32Array | null> => queries?.platformTransfers() ?? Promise.resolve(null);
