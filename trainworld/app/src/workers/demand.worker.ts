// Shared-world demand worker (T-116/T-117). Owns New York local commutes,
// Philadelphia local commutes and the separate regional market. Each assigned period
// advances all populations from one frozen total load vector at every crowding round.
// Network messages supersede work between populations; queries require the same
// version AND refinement round across all this worker's periods; one marked `wait` about a round
// not reached yet is answered when it is (T-327).
//
// Packs open as they arrive (T-149): `init` starts the WASM (and may carry packs), each `open`
// adds city packs or the regional market. A solve covers the cities open when it starts.
//
// One worker of the pool has no periods (`aux`, T-119): it only sets the network up. It is never
// busy, so the questions that need no period's result (the cells, a bubble's commuters by mode
// from the sums it is handed) are answered at once, not after a busy worker's current pass.
//
// T-325: the aux worker also holds the zones no station reaches, opened `bare`: their commute and
// everyday layers on an empty network, solved once for every period with no rail. Their bubbles
// and city figures come from there; no solve covers them, and no solving worker opens them.
//
// T-301: `close` frees packs (and the market) the page no longer needs; one the solve in hand
// covers goes when that solve ends.

import init, { DemandApi, EverydayApi, IntercityApi, MarketPacks, day_factors, demand_periods, everyday_warm_up, last_panic, release_skims, wasm_heap_bytes, wasm_heap_peak, wasm_memory_bytes } from "../wasm/demand/sim"; // demand only (T-051)
import { CITIES, cityById, cityToLonLat, cityToMarket, cityToWorld, HOME_CITY, lonLatToCity, marketWorld, setWorldFrame, widenBox, worldToCity, worldToLonLat, zoneBox, type CityId } from "../game/cities";
import { loadAttractions, type Attraction } from "../game/attractions";
import { EVERYDAY_KEYS, EVERYDAY_NAMES } from "../game/purposes";
import { PERIODS } from "../game/periods";
import { DEMAND_DAYS } from "../game/days";
import { curveKnots, scaledKnots } from "../game/fareCurve";
import { roundedDefault } from "../game/fareDefault";
import { FARE_BASELINE, fareFactors, fareScale, fareWeight } from "../game/income";
import type { CorridorTable } from "../game/demand";
import { arriveOf, comeMax, MARKET_TRIP_LAYOUT, TRIP_LAYOUT, type BareFigures, type DemandNetwork, type FromDemand, type Heap, type MarketPack, type MarketRows, type OpenRejected, type ToDemand } from "./demandProtocol";
import { cellBox, cityReach, clusterView, reachView, splitClusters, type Box, type Cluster, type PopView } from "./demandClusters";

/** WASM started (and the market opened if `init` carried it): `open` and solves wait for this. */
let started = false;
/** init and open in arrival order, each after the one before (T-149: packs arrive one by one) */
let opening: Promise<void> = Promise.resolve();
let aux = false;
let warmed = false;
const cities = new Map<CityId, DemandApi>();
/** the cities the network in hand was set on (`current`'s job): only these answer queries */
let jobCities = new Set<CityId>();
const everyday = new Map<CityId, EverydayApi>();
const EVERYDAY_PURPOSES = EVERYDAY_NAMES;
/** T-377: the zones' market packs open here, and `intercity` the market composed of all of them
 * (composed again whenever one opens or closes); it answers every question's all-mode part */
let marketPacks: MarketPacks | null = null;
/** T-776: the world market's rows until `marketPacks` is made from them */
let marketRows: MarketRows | null = null;
/** T-279: the routed drives that came with them */
let marketDrives: Uint8Array | null = null;
let intercity: IntercityApi | null = null;
/** T-409: the market's parts by `Cluster.marketKey`, one per cluster the market reaches, each on its
 * cluster's network alone (`IntercityApi.beside`). A part keeps the composition it was made from
 * while its key lasts (its zones' packs are in it); a part whose key is gone is reused for a new
 * key when it is of the newest composition, the rest are freed */
const marketParts = new Map<string, IntercityApi>();
/** the market's open 6 km zones' world ids and boxes (`IntercityApi.zone_boxes`), which its
 * clusters are joined by */
let marketZones: Float64Array | null = null;
let mine: number[] = [];
/** T-1607: the periods whose market this worker solves in a solve with the barrier, when `share`
 * (the pool sets some market apart from its period's cities, which stay in `mine`) */
let mkt: number[] = [];
let shareMarkets = false;
/** the periods the solve in hand has this worker solve anything of: `mine`, and with `share` `mkt` */
let units: number[] = [];
/** T-1607: the other side's loads of a shared period, by `${version}:${cluster}:${period}:${round}` */
const inbox = new Map<string, Float32Array>();
/** T-1607: market parts last solved sharing (their shared periods held by two workers) */
const sharedParts = new WeakSet<IntercityApi>();
let pending: Extract<ToDemand, { kind: "solve" }> | null = null;
let busy = false;
/** the version whose network `api` holds (queries about any other get nulls) */
let current = -1;
/** the round each of this worker's periods has reached; -1 before a result, -2 while a stage runs */
const roundsHeld: number[] = [];
/**
 * What `subSums` answers, taken the moment all this worker's periods have a round, per city and
 * round (T-119). The bubbles ask for it right after a round is published, when this worker is
 * already part way into the next one; the live state would be a mixture of two rounds by then, so
 * the answer is kept from when it was whole.
 */
const sumsByRound = new Map<string, { mask: number; sums: Float32Array; everyday: Float32Array | null }>();
type Sums = { mask: number; sums: Float32Array; everyday: Float32Array | null };
/** T-1904: the kind of day the page shows and the zones whose bubbles it draws (`view`) */
let viewDay = 0;
let viewZones = new Set<CityId>();
/**
 * A city's bubble sums for `rounds` of a solve, kept for `subSums` and sent unasked with the last
 * round (`kept`: the weekday's, already worked out). The weekday's always (other zones' bubbles and
 * the cache read them); the day shown as well for the zones on screen, since a day other than a
 * weekday is otherwise read from the live state, which is a round ahead by the time it is asked
 * (T-1904: the average day's bubbles waited a whole round, the cached result's for the final).
 */
function keepSums(version: number, rounds: number[], id: CityId, a: DemandApi, kept?: Sums) {
  const days = viewDay !== 0 && viewZones.has(id) ? [0, viewDay] : [0];
  for (const day of days) {
    const k: Sums = day === 0 ? kept ?? { mask: a.solved_mask(), sums: a.sub_sums(), everyday: everyday.get(id)?.cell_modes() ?? null }
      : { mask: a.solved_mask(), sums: a.sub_sums_day(day), everyday: everyday.get(id)?.cell_modes_day(day) ?? null };
    for (const r of rounds) sumsByRound.set(day === 0 ? `${id}:${r}` : `${id}:${r}:${day}`, k);
    const copy = k.sums.slice();
    const extra = k.everyday?.slice() ?? null;
    sendMessage({ kind: "sums", version, round: rounds[rounds.length - 1], city: id, mask: k.mask, sums: copy, everyday: extra, ...(day ? { day } : {}) }, [copy.buffer, ...(extra ? [extra.buffer] : [])]);
  }
}
/** the round the solve in hand is working on or last finished */
let stageRound = -1;
/** T-327: questions marked `wait` about a round (or version) this worker has not reached yet,
 * answered the moment it has, before it starts the next round */
let waiting: Query[] = [];
type Query = Exclude<ToDemand, { kind: "init" | "open" | "solve" | "go" | "whole" | "close" | "corridorsAgain" | "debugPanic" | "periods" | "first" | "view" | "loads" | "pause" }>;
/** T-327: the round every worker may start, by version and group (a solve marked `barrier` waits
 * for it) */
let go = { version: -1, group: 0, round: 0 };
const goAllows = (version: number, group: number, round: number) =>
  go.version > version || (go.version === version && (go.group > group || (go.group === group && go.round >= round)));
/** T-402: the solve in hand's dirty clusters by group: a solve with the barrier steps the zone in
 * view's cluster alone through its rounds first, then the rest together. `stageGroup` is the group
 * being solved; `ready` the clusters set up on the network in hand */
const groupOf = new Map<JobCluster, number>();
let stageGroup = 0;
const ready = new Set<JobCluster>();
/** the group the city's cluster is solved in (undefined: kept, or in no cluster) */
const cityGroup = (city: CityId) => { const c = jobClusters.find((c) => c.cities.includes(city)); return c ? groupOf.get(c) : undefined; };
/** `day_factors()`, read once */
let dayTable: ReturnType<typeof day_factors> | null = null;
/** a paused round boundary's resume (a `go` or a newer network) */
let wake: (() => void) | null = null;
/** the questions whose answer reads the periods' state at a round (the others read none) */
const BY_ROUND = new Set<Query["kind"]>(["station", "flowRail", "purposes", "icSel", "icStation", "transfers", "trip", "icTrip", "driveRail"]);
type Population = DemandApi | EverydayApi | IntercityApi;
/** T-1607: which of a cluster's populations a worker solves a period of: all, the cities' alone
 * (the market's in another worker) or the market's alone */
type Side = "all" | "cities" | "market";
const onSide = (side: Side, a: Population) => side === "all" || (side === "market") === (a instanceof IntercityApi);
/** T-326: the network in hand's clusters, each with its populations (cities, their everyday
 * layers, the market: the order their loads are added in), and the whole network's route nodes */
/** `pops` in the order their figures are added up; `solveOrder` the same populations in the order
 * they are set up and solved (it changes no figure) */
/** `reuse`: this worker's periods (bit q) kept from the last solve, set when it is set up (T-124) */
/** `part`: the market's part for the cluster's network (T-409), among `pops` */
/** `views`: the network each population is set on (T-756: a city's stations within its reach, the
 * market's the cluster's) and its maps to the whole network's stations and route nodes */
type JobCluster = Cluster & { pops: Population[]; solveOrder: Population[]; reuse: number; part: IntercityApi | null; views: Map<Population, PopView> };
const viewOf = (c: JobCluster, p: Population) => c.views.get(p)!;
/** T-1255: a population's boardings per period, on its own route nodes, from its last solve: the
 * free-flow pass's, and the last crowding round's before the cap. */
const boards = new WeakMap<Population, { free: (Float32Array | undefined)[]; rounds: (Float32Array | undefined)[] }>();
const boardsOf = (p: Population) => {
  let b = boards.get(p);
  if (!b) boards.set(p, (b = { free: [], rounds: [] }));
  return b;
};
/** T-1733: per period and round r, a population's loads as the round after r was priced on (its
 * segment loads, boardings and alightings on its own route nodes, one array; the last round's before
 * the cap), from its last solve. A local solve crowds the touched populations on the others' loads
 * round by round as the whole solve did, so with nothing changed it repeats the whole solve: priced
 * on the others' final loads instead, a 3-round solve with the market frozen took another path and
 * moved lines by 6-16% on all of Japan. */
const trail = new WeakMap<Population, (Float32Array | undefined)[][]>();
function keepTrail(p: Population, q: number, r: number) {
  let t = trail.get(p);
  if (!t) trail.set(p, (t = []));
  const seg = p.seg(q), board = p.board(q), alight = p.alight(q), nr = seg.length, v = new Float32Array(3 * nr);
  v.set(seg); v.set(board.subarray(0, nr), nr); v.set(alight.subarray(0, nr), 2 * nr);
  if (r === 0) t[q] = [];
  (t[q] ??= [])[r] = v;
}
/** T-756: whole network route node -> a view's, or -1 (made when first asked) */
const nodeLocals = new WeakMap<PopView, Int32Array>();
function nodeLocal(v: PopView, nodes: number): Int32Array {
  let m = nodeLocals.get(v);
  if (!m) {
    m = new Int32Array(nodes).fill(-1);
    v.nodeGlobal.forEach((g, k) => (m![g] = k));
    nodeLocals.set(v, m);
  }
  return m;
}
let jobClusters: JobCluster[] = [];
let jobNodes = 0;
/** T-326: the cluster `sig` each population holds every one of this worker's periods solved for,
 * to the last round */
const held = new Map<Population, string>();
/** T-124: per population and period, the cluster's `psig` it holds that period solved for, to the
 * last round, in a solve where every population of the cluster solved it too */
const heldQ = new Map<Population, string[]>();
/** T-1609: a period solved alone against its cluster's frozen loads (`heldQ` psig plus this), which
 * a later solve may refine against again but never keeps as a whole solve's */
const LOCAL = "~local";
/** T-1609: per cluster `sig` and period, the last whole solve's loads before the capacity fit over
 * those after it, where they differ (route nodes of the whole network, the three blocks as `step`'s
 * totals): the frozen loads a local refine crowds on are the populations' capped loads times this */
const capRatio = new Map<string, ({ idx: Uint32Array; v: Float32Array } | null)[]>();
function keepCapRatio(sig: string, q: number, before: Float32Array, after: Float32Array) {
  const idx: number[] = [], v: number[] = [];
  for (let k = 0; k < before.length; k++) if (after[k] > 0 && before[k] !== after[k]) { idx.push(k); v.push(before[k] / after[k]); }
  const list = capRatio.get(sig) ?? [];
  capRatio.delete(sig);
  list[q] = idx.length ? { idx: Uint32Array.from(idx), v: Float32Array.from(v) } : null;
  capRatio.set(sig, list);
  while (capRatio.size > 32) capRatio.delete(capRatio.keys().next().value!);
}
/** T-1862: per population, the cluster it holds whole-solved (`sig` and `psig` as `held` and `heldQ`
 * hold them, re-keyed by a keep within noise), its `shape`, and the network it was solved on */
const heldOn = new WeakMap<Population, { sig: string; psig: string[]; shape: string; net: DemandNetwork }>();
/** T-1862: the settle moves lines no edit touched by float noise (0.12 s and 0.08% seen, T-1850);
 * one train on a line of under 500 an hour is past 0.2%. Span is compared exactly */
const NOISE_RUN_S = 0.5;
const NOISE_TRAINS = 0.002;
/** T-1862: run times and trains an hour of `now` within noise of `was` (same shape), span the same */
function withinNoise(was: DemandNetwork, now: DemandNetwork): boolean {
  const s0 = was.span ?? new Float32Array(0), s1 = now.span ?? new Float32Array(0);
  if (was.times.length !== now.times.length || was.tph.length !== now.tph.length || s0.length !== s1.length) return false;
  for (let k = 0; k < s0.length; k++) if (s0[k] !== s1[k]) return false;
  for (let k = 0; k < was.times.length; k++) if (!(Math.abs(was.times[k] - now.times[k]) <= NOISE_RUN_S)) return false;
  for (let k = 0; k < was.tph.length; k++) if (!(Math.abs(was.tph[k] - now.tph[k]) <= NOISE_TRAINS * Math.abs(was.tph[k]))) return false;
  return true;
}
const VIEW_TOLERANCE = 0.01;
/** T-1733: trains an hour (and the share of riders with a train due, which follows them) move by a
 * few percent on far lines when a line is built (all of Japan: 2.7% on an Osaka line for a Tokyo
 * line, which made Osaka a third of the local solve); within this a zone keeps its figures until
 * the whole solve */
const TRAINS_TOLERANCE = 0.05;
/** T-1764: `span` is a share (0 to 1) in steps of 0.05, one train more or less in a period's day; a
 * single step, 0.05000001 in float32, is the same view, so a far line's step does not touch its zones */
const SPAN_TOLERANCE = 0.06;
/** T-1733: the page's `whole` for this version: the edit's local results are shown (and a moment
 * has passed for the next edit), so the whole solve may start */
let wholeGo = -1;
/** T-1733: at most this long waiting for `whole` (a worker that never sends its local result) */
const WHOLE_WAIT_MS = 300_000;
/** T-1733: a route node with no counterpart in the network now (a write there is dropped) */
const NONE = 0xffffffff;
/** T-1733: the network the last solve started on, to map route nodes from it to the next one's */
type NetIds = { lines: string[]; stations: string[]; lineN: Uint32Array; stops: Uint32Array; tph: Float32Array; times: Float32Array };
let lastNet: NetIds | null = null;
/** T-1735: per market part, the network (`lastNet`) it holds solved, untouched since */
const marketNet = new Map<IntercityApi, NetIds>();
/** T-1733: the view each city population holds every one of its periods solved on, untouched since
 * (a population set on a network again is dropped from it) */
const cityOn = new Map<Population, PopView>();
/** T-1733: a market part's route nodes on the network now, for the network it holds solved, untouched
 * since; remapped as each solve starts */
const marketOn = new Map<IntercityApi, Uint32Array>();
/** T-1607, T-1733: the cities' side of a shared period keeps the market's last loads there (the other
 * worker's), whole-network route nodes, remapped as each solve starts */
const sharedMkt = new Map<IntercityApi, (Float32Array | undefined)[]>();
/** T-1733: the same per round as each was priced on (`trail`), round 0 to the last */
const sharedTrail = new Map<IntercityApi, (Float32Array[] | undefined)[]>();
/** T-1733: route node of network `a` -> route node of network `b` (-1: none), by line id and stop
 * station id (the k-th visit of a station on a run to the k-th) */
function nodeMap(a: NonNullable<typeof lastNet>, b: NonNullable<typeof lastNet>): Int32Array {
  const out = new Int32Array(2 * a.stops.length).fill(-1);
  const at = new Map<string, [number, number]>();
  let o = 0;
  b.lines.forEach((id, l) => { at.set(id, [o, b.lineN[l]]); o += b.lineN[l]; });
  let oa = 0;
  a.lines.forEach((id, l) => {
    const n = a.lineN[l], hit = at.get(id);
    if (hit) {
      const [ob, nb] = hit;
      for (let dir = 0; dir < 2; dir++) {
        const where = new Map<string, number>(), seen = new Map<string, number>();
        for (let j = 0; j < nb; j++) {
          const s = b.stations[b.stops[ob + (dir ? nb - 1 - j : j)]], k = seen.get(s) ?? 0;
          seen.set(s, k + 1);
          where.set(`${s}#${k}`, j);
        }
        seen.clear();
        for (let i = 0; i < n; i++) {
          const s = a.stations[a.stops[oa + (dir ? n - 1 - i : i)]], k = seen.get(s) ?? 0;
          seen.set(s, k + 1);
          const j = where.get(`${s}#${k}`);
          if (j !== undefined) out[2 * oa + dir * n + i] = 2 * ob + dir * nb + j;
        }
      }
    }
    oa += n;
  });
  return out;
}
/** T-1733: whether two views give a population the same inputs (so its state solved on one holds
 * for the other); their route nodes in the whole network may differ */
/** each value replaced by the order it first appears in */
function canonical(a: Uint32Array): Uint32Array {
  const seen = new Map<number, number>();
  return a.map((v) => { let k = seen.get(v); if (k === undefined) seen.set(v, (k = seen.size)); return k; });
}
function sameView(x: PopView, y: PopView): boolean {
  return viewDiff(x, y) === null;
}
function viewDiff(x: PopView, y: PopView): string | null {
  if (x === y) return null;
  if (x.pads.length !== y.pads.length || x.pads.some((v, k) => v !== y.pads[k])) return "pads";
  const a = x.net as unknown as Record<string, unknown>, b = y.net as unknown as Record<string, unknown>;
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    let u = a[k], v = b[k];
    // a station of no complex is numbered past every complex (`packService`): the same grouping
    // under other numbers is the same input
    if (k === "complex" && u instanceof Uint32Array && v instanceof Uint32Array) [u, v] = [canonical(u), canonical(v)];
    if (ArrayBuffer.isView(u) && ArrayBuffer.isView(v)) {
      const p = u as unknown as ArrayLike<number>, w = v as unknown as ArrayLike<number>;
      if (p.length !== w.length) return `${k} length ${p.length} ${w.length}`;
      // run times, trains an hour and the like move by float noise and by small knock-on changes
      // when a line is added beside them (0.1% seen): within VIEW_TOLERANCE they count as the same
      const tol = !(u instanceof Float32Array) || k === "stXY" || k === "catchXY" ? 0 : k === "tph" ? TRAINS_TOLERANCE : k === "span" ? SPAN_TOLERANCE : VIEW_TOLERANCE;
      for (let i = 0; i < p.length; i++) if (p[i] !== w[i] && Math.abs(p[i] - w[i]) > tol * Math.max(1, Math.abs(p[i]))) return `${k}[${i}] ${p[i]} ${w[i]}`;
    } else if (JSON.stringify(u) !== JSON.stringify(v)) return `${k} ${JSON.stringify(u)?.slice(0, 80)} ${JSON.stringify(v)?.slice(0, 80)}`;
  }
  return null;
}
/** T-1735: a window's reach by rail from the stations an edit changed, at most (`set_window`); the
 * stations it may hold bound it first on a big market */
const WINDOW_REACH_MIN = 180;
/** T-1735: the most of the market's searched stations a window holds, or `WINDOW_FLOOR` stations
 * (`set_window`). Its pass costs about that share of a whole pass, and the local solve two such
 * passes a period (all of Japan's new 675 km express at free flow: 83% of its market riders at 8%,
 * 88% at 10%); a search costs less on a smaller network, and China's longer trips come to an express
 * from further round its stops (a 2,400 km one: 22% of them at 9%, 89% at half its stations) */
const WINDOW_SHARE = 0.09;
const WINDOW_FLOOR = 800;
/** T-1735: an edit whose longest changed run (`editedStations`' `minutes`) is shorter leaves the
 * market frozen: a city line's market riders are a few percent of its riders (China's new metro
 * lines 1.5-2%), and the window's market set on the network was most of China's local solve */
const WINDOW_EDIT_MIN = 20;
/** T-1735: the stations network `b` changed from `a` (a stop added or dropped there, or a hop's run
 * time changed), as masks over `b`'s and `a`'s stations; `map` is `nodeMap(a, b)`.
 * `minutes`: the longest run between a line's first and last changed stops, the window's reach (a
 * pair goes out of its way for a new service only about as far as the service itself goes) */
function editedStations(a: NonNullable<typeof lastNet>, b: NonNullable<typeof lastNet>, map: Int32Array): { now: Uint8Array; was: Uint8Array; minutes: number } {
  const now = new Uint8Array(b.stations.length), was = new Uint8Array(a.stations.length);
  const atB = new Map(b.stations.map((id, k) => [id, k]));
  const offs = (n: Uint32Array) => { const o = [0]; for (const v of n) o.push(o[o.length - 1] + v); return o; };
  const oa = offs(a.lineN), ob = offs(b.lineN);
  // a route node's stop, by its place in the line's two runs
  const stopOf = (n: Uint32Array, o: number[], l: number, k: number) => { const len = n[l], i = k - 2 * o[l]; return o[l] + (i < len ? i : 2 * len - 1 - i); };
  const lineOf = (o: number[], k: number) => { let lo = 0, hi = o.length - 2; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (2 * o[m] <= k) lo = m; else hi = m - 1; } return lo; };
  const hit = new Uint8Array(2 * b.stops.length);
  // per line, the first and last stop changed (indices into `stops`), each network's
  const spanA = new Map<number, [number, number]>(), spanB = new Map<number, [number, number]>();
  const widen = (m: Map<number, [number, number]>, l: number, i: number) => { const s = m.get(l); m.set(l, s ? [Math.min(s[0], i), Math.max(s[1], i)] : [i, i]); };
  const mark = (stA: number, la: number, ia: number) => { was[stA] = 1; widen(spanA, la, ia); const k = atB.get(a.stations[stA]); if (k !== undefined) now[k] = 1; };
  for (let k = 0; k < map.length; k++) {
    const la = lineOf(oa, k), sa = a.stops[stopOf(a.lineN, oa, la, k)];
    if (map[k] < 0) { mark(sa, la, stopOf(a.lineN, oa, la, k)); continue; }
    hit[map[k]] = 1;
    const m = map[k], lb = lineOf(ob, m), ia = stopOf(a.lineN, oa, la, k), ib = stopOf(b.lineN, ob, lb, m);
    // this hop's run time past VIEW_TOLERANCE (trains an hour do not count: a train-count edit moves
    // the market's riders by under 1% on lines, T-1650, and an edit's knock-on trains on whole far
    // lines would open a window for a city line)
    const off = (x: number, y: number, tol: number) => x !== y && Math.abs(x - y) > tol * Math.max(1, Math.abs(x));
    let changed = false;
    // (times: per line, two directions by three levels of a value per stop)
    const ta = 6 * oa[la], tb = 6 * ob[lb], na = a.lineN[la], nb = b.lineN[lb];
    for (let r = 0; r < 6 && !changed; r++) changed = off(a.times[ta + r * na + (ia - oa[la])], b.times[tb + r * nb + (ib - ob[lb])], VIEW_TOLERANCE);
    if (changed) { mark(sa, la, ia); widen(spanB, lb, ib); }
  }
  for (let m = 0; m < hit.length; m++) if (!hit[m]) { const lb = lineOf(ob, m), ib = stopOf(b.lineN, ob, lb, m); now[b.stops[ib]] = 1; widen(spanB, lb, ib); }
  // (run times are seconds, the first row each hop's on from its stop one way at the first level)
  const run = (n: NonNullable<typeof lastNet>, o: number[], spans: Map<number, [number, number]>) => {
    let most = 0;
    for (const [l, [i, j]] of spans) { let s = 0; for (let k = i; k < j; k++) s += n.times[6 * o[l] + (k - o[l])] || 0; most = Math.max(most, s); }
    return most / 60;
  };
  return { now, was, minutes: Math.min(WINDOW_REACH_MIN, Math.max(run(a, oa, spanA), run(b, ob, spanB))) };
}
/** T-1733: carry the route-node state kept across solves (`marketOn`, `sharedMkt`, `capRatio`) to a
 * new network; without both networks' ids it is dropped */
function remapKept(next: NonNullable<typeof lastNet> | null) {
  const prev = lastNet;
  lastNet = next;
  if (!prev || !next) { marketOn.clear(); sharedMkt.clear(); sharedTrail.clear(); capRatio.clear(); marketNet.clear(); return; }
  const same = prev.lines.join("\n") === next.lines.join("\n") && prev.stations.join("\n") === next.stations.join("\n")
    && prev.lineN.length === next.lineN.length && prev.lineN.every((v, k) => v === next.lineN[k]) && prev.stops.length === next.stops.length && prev.stops.every((v, k) => v === next.stops[k]);
  if (same) return;
  const map = nodeMap(prev, next), N0 = map.length, N1 = 2 * next.stops.length;
  for (const [p, g] of marketOn) marketOn.set(p, g.map((v) => (v === NONE || map[v] < 0 ? NONE : map[v])));
  const move = (t: Float32Array) => {
    const out = new Float32Array(3 * N1);
    for (let k = 0; k < N0; k++) { const m = map[k]; if (m >= 0) for (let b = 0; b < 3; b++) out[b * N1 + m] = t[b * N0 + k]; }
    return out;
  };
  for (const [p, list] of sharedMkt) sharedMkt.set(p, list.map((t) => t && move(t)));
  for (const [p, list] of sharedTrail) sharedTrail.set(p, list.map((rs) => rs?.map(move)));
  for (const [sig, list] of capRatio) capRatio.set(sig, list.map((r) => {
    if (!r) return r;
    const idx: number[] = [], v: number[] = [];
    for (let i = 0; i < r.idx.length; i++) { const b = Math.floor(r.idx[i] / N0), m = map[r.idx[i] % N0]; if (m >= 0) { idx.push(b * N1 + m); v.push(r.v[i]); } }
    return idx.length ? { idx: Uint32Array.from(idx), v: Float32Array.from(v) } : null;
  }));
}
/** T-1733: a population set on a network again no longer holds what it solved */
function disturbed(p: Population) {
  cityOn.delete(p);
  heldOn.delete(p);
  trail.delete(p);
  if (p instanceof IntercityApi) { marketOn.delete(p); sharedMkt.delete(p); sharedTrail.delete(p); marketNet.delete(p); }
}
const boxes = new Map<CityId, Box>();
/** T-392: the clusters of the last few networks, so a question about a shown result whose cluster
 * this worker still holds is answered at once, whatever else is being solved. `railed`: the network
 * has stations and lines (a population set on it then has subzones). */
/** `markets` (T-1607): the periods whose market this worker solved for it */
interface Version { clusters: JobCluster[]; stations: number; railed: boolean; nodes: number; rounds: number; markets: number[] }
const maskOf = (qs: number[]) => qs.reduce((m, q) => m | (1 << q), 0);
const versions = new Map<number, Version>();
const KEEP_VERSIONS = 6;
/** every population of the cluster holds its network solved to the last round */
const isHeld = (c: JobCluster) => c.pops.length > 0 && c.pops.every((p) => held.get(p) === c.sig);
/** T-325, the aux worker: each solving worker's periods as a bit mask, in worker order */
let groupMasks: number[] = [];
/** T-325, the aux worker: zones open bare, apart from `cities` (a solve never covers these) */
const bare = new Map<CityId, { a: DemandApi; e: EverydayApi | null; figures: BareFigures }>();

function sendMessage(msg: FromDemand, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/**
 * T-1174: a Rust panic (or an allocation the memory cannot grow for) aborts the WASM call as a trap,
 * never unwinding it, so the borrow wasm-bindgen took on the object stays held and every later call
 * on it throws "recursive use of an object". No call here awaits inside WASM, so two calls never
 * overlap: that error always means an earlier call aborted. The instance is then not to be trusted
 * (its allocator may be mid-change too): the worker stops and the page replaces it.
 */
let dead = false;
function isPanic(err: unknown): boolean {
  return err instanceof WebAssembly.RuntimeError || (err instanceof Error && /recursive use of an object|null pointer passed to rust|attempted to take ownership of Rust value/.test(err.message));
}
function died(err: unknown) {
  if (dead) return;
  dead = true;
  let panic = "", memBytes = 0;
  try { panic = last_panic(); memBytes = wasm_memory_bytes(); } catch { /* the instance is past reading */ }
  sendMessage({ kind: "dead", error: String(err), panic, memBytes });
}
/** T-1205: `done`'s peak is its solve's (reset as each job starts) */
const heap = (): Heap => [wasm_heap_bytes(), wasm_heap_peak(false)];
/** T-1205: the periods the pool dealt this worker when it shrank, taken up between solves; nothing
 * kept is trusted after (a period dealt anew was never solved here) */
let dealt: Extract<ToDemand, { kind: "periods" }> | null = null;
let solvedOnce = false;
function deal() {
  mine = dealt!.periods;
  mkt = dealt!.markets ?? mine;
  shareMarkets = !!dealt!.share;
  groupMasks = (dealt!.groups ?? []).map((g) => g.reduce((m, q) => m | (1 << q), 0));
  dealt = null;
  held.clear();
  heldQ.clear();
}
/** `debugPanic` (dev server only): the next step of a solve panics */
let panicInSolve = false;
/** `debugPanic` now: a panic swallowed (as a pack's rejection once swallowed one), then the next
 * call on the object finds its borrow held: the pair of errors T-1174 saw, in that order */
function panicNow() {
  const a = cities.values().next().value ?? bare.values().next().value?.a;
  if (!a) throw new Error("no pack is open");
  try { a.debug_panic(); } catch { /* swallowed */ }
  a.info();
}
/** A panic ends the worker (`died`); any other error is `failed`, as before. */
function failed(err: unknown) {
  if (dead) return;
  if (isPanic(err)) died(err);
  else sendMessage({ kind: "failed", error: String(err) });
}

/** Yield to the event loop so a waiting message (a newer network) is delivered; while `pause` holds
 * this worker (T-2045), wait there until it lets go or a newer solve comes. */
async function yieldNow(): Promise<void> {
  await new Promise<void>((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(0);
  });
  while (pauseGate && !pending) await pauseGate;
}
/** T-2045: the focus pool solving the zone in view first: this pool's solving and opening wait */
let pauseGate: Promise<void> | null = null;
let pauseEnd: (() => void) | null = null;
function setPause(on: boolean) {
  if (on && !pauseGate) pauseGate = new Promise<void>((r) => (pauseEnd = r));
  else if (!on && pauseGate) { pauseEnd?.(); pauseGate = pauseEnd = null; }
}
/** wake the waiters (a newer solve came); still paused for the rest */
function nudgePause() {
  if (!pauseGate) return;
  pauseEnd?.();
  pauseGate = new Promise<void>((r) => (pauseEnd = r));
}

/**
 * Open city packs and, on a solving worker, the regional market. Returns each opened city's info
 * (DemandApi.info(), with its everyday purposes folded into its trips and walk share). Any pack
 * carrying the everyday arrays gets the layer (every registered city since T-142); a pack
 * without them stays commute-only.
 */
function openPacks(packs: { id: CityId; header: string; bin: ArrayBuffer; level?: number }[], market: MarketPack[] | undefined, isBare = false, rejected: OpenRejected[] = []) {
  // T-492: a pack that does not open is left out and reported; the pool carries on
  const reject = (id: CityId, isMarket: boolean, err: unknown) => rejected.push({ id, market: isMarket, error: String(err) });
  for (const pack of packs) {
    try {
      if (isBare) {
        if (!bare.has(pack.id)) bare.set(pack.id, openBare(pack));
        continue;
      }
      const level = aux ? 0 : pack.level ?? 0;
      if (cities.has(pack.id)) {
        // (T-301: reopened before its close took effect: it stays)
        if (cities.get(pack.id)!.level() === level) { closeLater.ids.delete(pack.id); reopenLater.delete(pack.id); continue; }
        // T-1223: at another level; the solve in hand keeps the one it set up
        if (busy && jobCities.has(pack.id)) { reopenLater.set(pack.id, pack); continue; }
        dropCity(pack.id);
      }
      openCity(pack, level);
    } catch (err) {
      if (isPanic(err)) throw err;
      reject(pack.id, false, err);
    }
  }
  if (market?.length && !aux) {
    if (!marketPacks) {
      if (!marketRows) {
        for (const m of market) reject(m.id, true, "the market's rows have not arrived");
        return opened(packs, isBare, rejected);
      }
      marketPacks = new MarketPacks(marketWorld(typeof marketRows === "string" ? marketRows : new TextDecoder().decode(marketRows)));
      try {
        if (marketDrives) marketPacks.set_drives(marketDrives);
      } catch (err) {
        if (isPanic(err)) throw err;
        console.warn("market drives:", err); // the rule's drives everywhere
      }
      marketRows = marketDrives = null;
    }
    const have = new Set(marketPacks.places());
    const added: CityId[] = [];
    for (const m of market) {
      closeLater.markets.delete(m.id);
      if (have.has(MARKET_OF[m.id])) continue;
      try {
        // T-180: the page sends the pack's bytes; decoding here keeps it off the main thread
        marketPacks.add(typeof m.pack === "string" ? m.pack : new TextDecoder().decode(m.pack));
        added.push(m.id);
        sendMessage({ kind: "packOpened", id: m.id, market: true }); // T-1195
      } catch (err) {
        if (isPanic(err)) throw err;
        reject(m.id, true, err);
      }
    }
    if (added.length) {
      try {
        recompose();
      } catch (err) {
        if (isPanic(err)) throw err;
        // composing checks each pack against the registry's row for its zone (the zone count) and
        // names no pack: the ones just added are taken out again, the market stays as it was
        for (const id of added) { marketPacks.remove(MARKET_OF[id]); reject(id, true, err); }
        recompose();
      }
    }
  }
  return opened(packs, isBare, rejected);
}

/** A city's commutes and, on a solving worker, its everyday layer, at `level` (T-1222). */
function openCity(pack: { id: CityId; header: string; bin: ArrayBuffer }, level: number) {
  const bin = new Uint8Array(pack.bin);
  const a = DemandApi.at_level(pack.header, bin, level);
  // T-399: the everyday layer shares the commutes' cells, zones, gravity and routes
  let e: EverydayApi | null = null;
  try {
    if (!aux && hasEveryday(pack.header)) e = EverydayApi.beside_with(a, pack.header, bin, attractionsIn(pack.id));
  } catch (err) {
    a.free();
    throw err;
  }
  cities.set(pack.id, a);
  if (e) everyday.set(pack.id, e);
  sendMessage({ kind: "packOpened", id: pack.id }); // T-1195
}

/** T-1223: zones to open again at another level once the solve in hand, which covers them, ends */
const reopenLater = new Map<CityId, { id: CityId; header: string; bin: ArrayBuffer; level?: number }>();

/** Each city pack that opened, with its figures (`info`). */
function opened(packs: { id: CityId }[], isBare: boolean, rejected: OpenRejected[]) {
  const failed = new Set(rejected.filter((r) => !r.market).map((r) => r.id));
  if (!isBare) for (const id of reopenLater.keys()) failed.add(id);
  return Object.fromEntries(packs.filter(({ id }) => !failed.has(id)).map(({ id }) => {
    const b = isBare ? bare.get(id) : undefined;
    const info = Array.from((b?.a ?? cities.get(id)!).info());
    // Auxiliary worker only serves commute allocation; compute workers report all local demand
    // (and so does the aux worker for a zone it holds bare, T-325).
    const extra = (b ? b.e : everyday.get(id))?.info();
    if (extra) {
      let outings = 0, walking = 0;
      for (let k=0;k<extra.length;k+=3) { outings += extra[k]; walking += extra[k]*extra[k+1]; }
      info[3] = (info[2]*info[3]+walking)/(info[2]+outings);
      info[2] += outings;
    }
    return [id, info];
  })) as Record<CityId, number[]>;
}

/** T-263: the world's attractions, loaded with the WASM */
let attractions: Attraction[] = [];
/** The attractions in or near a zone's box, as `EverydayApi.beside_with` takes them: x, y in the
 * pack's frame and visitors a year (the layer keeps those within reach of one of its cells). */
function attractionsIn(id: CityId): Float32Array {
  const [w, s, e, n] = widenBox(zoneBox(cityById(id)), 1000);
  const out: number[] = [];
  for (const t of attractions) {
    if (t.lng < w || t.lng > e || t.lat < s || t.lat > n) continue;
    const { x, y } = lonLatToCity(t.lng, t.lat, id);
    out.push(x, y, t.visitors);
  }
  return Float32Array.from(out);
}

const hasEveryday = (header: string) => JSON.parse(header).arrays.some((a: {name: string}) => a.name === "errand_jobs");

/**
 * T-325: a zone no station reaches, opened bare. With no station within reach every trip walks or
 * drives, so its layers are set on an empty network and solved once for every period: the same
 * figures a solve of any network gives it (no subzone, no rail), worked out the same way as
 * `run` does per period (`citySummary`, `cityParts`).
 */
function openBare(pack: { id: CityId; header: string; bin: ArrayBuffer }) {
  const bin = new Uint8Array(pack.bin);
  const a = new DemandApi(pack.header, bin);
  let e: EverydayApi | null = null;
  try {
    e = hasEveryday(pack.header) ? EverydayApi.beside_with(a, pack.header, bin, attractionsIn(pack.id)) : null;
    return solveBare(a, e);
  } catch (err) {
    e?.free();
    a.free();
    throw err;
  }
}

function solveBare(a: DemandApi, e: EverydayApi | null) {
  const f0 = new Float32Array(0), u0 = new Uint32Array(0);
  a.set_network(f0, u0, u0, f0, f0, f0);
  e?.set_network(f0, u0, u0, f0, f0, f0);
  const keys = ["work", ...EVERYDAY_KEYS.slice(0, Math.min(EVERYDAY_KEYS.length, e?.purpose_count() ?? 0))];
  const figures: BareFigures = { summary: [], parts: [], keys };
  for (let q = 0; q < PERIODS.length; q++) {
    a.solve(q);
    e?.solve(q);
    const own = Array.from(a.summary(q)), s = own.slice();
    const v = e?.summary(q);
    if (v) for (const k of [1, 2, 3, 5, 10, 11, 12]) s[k] += v[k] ?? 0;
    const p = new Array(3 * keys.length).fill(0);
    for (let k = 0; k < 3; k++) p[k] = own[1 + k] ?? 0;
    const kt = e?.kind_totals(q);
    if (kt) for (let k = 0; k < Math.min(kt.length, 3 * (keys.length - 1)); k++) p[3 + k] = kt[k];
    figures.summary.push(s);
    figures.parts.push(p);
  }
  return { a, e, figures };
}

/** T-301: closes of packs the solve in hand covers, applied once it has ended */
const closeLater = { ids: new Set<CityId>(), markets: new Set<CityId>() };

/** T-377: the market of every open market pack, in place of the last. That one is freed unless a
 * part still holds it (its zones' packs are in it; it carries on until its key goes). T-1198: freed
 * before the new one is composed, so its space (about 450 MB with the world's 2,567 places) takes the
 * new one rather than the worker's memory growing by a second; and a part composed of the same packs
 * (a selected zone's market opened, then closed) serves again, composing nothing. */
function recompose() {
  const old = intercity;
  marketZones = null;
  const key = marketPacks?.places().join(",") ?? "";
  if (old && ![...marketParts.values()].includes(old)) { intercity = null; forget([old]); }
  const same = key ? [...marketParts.values()].find((p) => composedOf.get(p) === key) : undefined;
  intercity = same ?? (key ? marketPacks!.market() : null);
  if (intercity && !same) composedOf.set(intercity, key);
}
/** T-1198: the market packs (places, in order) each composition and its parts were made of */
const composedOf = new WeakMap<IntercityApi, string>();

/** T-301: take populations out of every kept network's clusters, then free them. */
function forget(pops: Population[]) {
  const gone = new Set(pops);
  for (const cs of [jobClusters, ...[...versions.values()].map((v) => v.clusters)]) for (const c of cs) {
    if (c.part && gone.has(c.part)) c.part = null;
    c.pops = c.pops.filter((p) => !gone.has(p));
    c.solveOrder = c.solveOrder.filter((p) => !gone.has(p));
  }
  for (const p of gone) { held.delete(p); heldQ.delete(p); disturbed(p); p.free(); }
}

function dropCity(id: CityId) {
  const a = cities.get(id);
  if (!a) return;
  const e = everyday.get(id);
  cities.delete(id);
  everyday.delete(id);
  boxes.delete(id);
  if (!bare.has(id)) squares.delete(id);
  forget(e ? [e, a] : [a]);
}

/** T-377: close zones' market packs; with none left, the market and all its parts go. */
function dropMarkets(ids: CityId[]) {
  if (!marketPacks) return;
  let changed = false;
  for (const id of ids) changed = marketPacks.remove(MARKET_OF[id]) || changed;
  if (!changed) return;
  if (marketPacks.places().length) return recompose();
  const parts = [...new Set([...(intercity ? [intercity] : []), ...marketParts.values()])];
  marketParts.clear();
  intercity = null;
  marketZones = null;
  forget(parts);
}

function closePacks(msg: Extract<ToDemand, { kind: "close" }>) {
  if (msg.bare) {
    for (const id of msg.ids) {
      const b = bare.get(id);
      if (!b) continue;
      bare.delete(id);
      b.e?.free();
      b.a.free();
      if (!cities.has(id)) squares.delete(id);
    }
  } else {
    for (const id of msg.ids) { reopenLater.delete(id); if (busy && jobCities.has(id)) closeLater.ids.add(id); else dropCity(id); }
    if (msg.markets?.length) { if (busy) for (const id of msg.markets) closeLater.markets.add(id); else dropMarkets(msg.markets); }
  }
  sendMessage({ kind: "closed", memBytes: wasm_memory_bytes(), heap: heap() });
}

function closeDeferred() {
  for (const id of closeLater.ids) dropCity(id);
  closeLater.ids.clear();
  if (closeLater.markets.size) dropMarkets([...closeLater.markets]);
  closeLater.markets.clear();
  if (!reopenLater.size) return;
  const packs = [...reopenLater.values()];
  reopenLater.clear();
  const t0 = performance.now(), rejected: OpenRejected[] = [];
  for (const pack of packs) {
    dropCity(pack.id);
    try { openCity(pack, pack.level ?? 0); } catch (err) { if (isPanic(err)) throw err; rejected.push({ id: pack.id, market: false, error: String(err) }); }
  }
  sendMessage({ kind: "opened", cityInfo: opened(packs, false, rejected), markets: [], openMs: performance.now() - t0, warmMs: 0, memBytes: wasm_memory_bytes(), heap: heap(), ...(rejected.length ? { rejected } : {}) });
}

/** T-889: `open`s not yet taken up, each with the count of `close`s that came before it. Opens
 * that arrive together are opened together: composing the market costs about as much for one pack
 * as for forty, so a big game's packs, composed one by one, took 0.7-0.9 s each. */
const openQueue: { msg: Extract<ToDemand, { kind: "open" }>; epoch: number }[] = [];
let closeEpoch = 0;

let oneByOne = false;

function openQueued(epoch: number): Promise<void> {
  // a macrotask, so the opens already posted are delivered and join this one; T-1533: not while a
  // rough solve of the zone in view runs
  return (oneByOne ? Promise.resolve() : new Promise<void>((r) => setTimeout(r, 0))).then(() => roughRun ?? undefined).then(async () => { while (pauseGate) await pauseGate; }).then(() => {
    let first = openQueue[0];
    // taken up by an earlier batch, or after a close that comes before it in the chain
    if (!first || first.epoch !== epoch) return;
    // T-1533: city packs before market packs: a rough solve needs no market, and composing a big
    // game's market (hundreds of packs) holds the worker up for a minute
    if (!first.msg.packs.length && !aux) {
      let j = openQueue.findIndex((q) => q.epoch === epoch && q.msg.packs.length > 0 && !!q.msg.first);
      if (j < 0) j = openQueue.findIndex((q) => q.epoch === epoch && q.msg.packs.length > 0);
      if (j > 0) { openQueue.unshift(...openQueue.splice(j, 1)); first = openQueue[0]; }
    }
    const isBare = !!first.msg.bare && aux;
    // T-1533: only market packs open together (one composition for them all); a city pack opens on
    // its own, so the zone in view's, moved ahead (`first`), waits for one pack at most
    let n = 1;
    while (!oneByOne && !aux && !first.msg.packs.length && n < openQueue.length && openQueue[n].epoch === epoch && !openQueue[n].msg.packs.length) n++;
    const batch = openQueue.splice(0, n).map((q) => q.msg);
    if (!started || dead) return;
    try {
      const t0 = performance.now();
      for (const m of batch) if (m.rows && !marketPacks) [marketRows, marketDrives] = [m.rows, m.drives ?? null];
      const packs = batch.flatMap((m) => m.packs), market = batch.flatMap((m) => m.intercity ?? []);
      const rejected: OpenRejected[] = [];
      const cityInfo = openPacks(packs, market, isBare, rejected);
      const openMs = performance.now() - t0;
      const warmMs = warmUp();
      const figures = isBare ? Object.fromEntries(packs.filter(({ id }) => bare.has(id)).map(({ id }) => [id, bare.get(id)!.figures])) : undefined;
      const markets = market.map((m) => m.id).filter((id) => !rejected.some((r) => r.market && r.id === id));
      sendMessage({ kind: "opened", cityInfo, markets, openMs, warmMs, memBytes: wasm_memory_bytes(), heap: heap(), ...(figures ? { bare: figures } : {}), ...(rejected.length ? { rejected } : {}) });
    } catch (err) {
      failed(err);
    }
    startRough();
  });
}

/** T-1533: a rough solve of the zone in view running; the opens waiting wait for it */
let roughRun: Promise<void> | null = null;
let roughEnd: (() => void) | null = null;
function setRough(on: boolean) {
  if (on && !roughRun) roughRun = new Promise<void>((r) => (roughEnd = r));
  else if (!on && roughRun) { roughEnd?.(); roughRun = roughEnd = null; }
}
const roughReady = (job: Extract<ToDemand, { kind: "solve" }>) => !!job.only?.every((id) => cities.has(id));
/** A rough solve starts as soon as its zones are open here, ahead of the opens still waiting. */
function startRough() {
  if (pending?.only && started && !busy && !dead && roughReady(pending)) void run();
}

/** Warm-up solves on the first city a worker opens, so V8 has optimised the kernel before the first real one. */
function warmUp(): number {
  if (warmed || !cities.size) return 0;
  warmed = true;
  return [...cities.values()].reduce((sum, a) => sum + a.warm_up(), 0);
}

onmessage = (e: MessageEvent<ToDemand>) => {
  const msg = e.data;
  if (dead) return;
  if (msg.kind === "init") {
    opening = opening.then(async () => {
      try {
        const t0 = performance.now();
        [, attractions] = await Promise.all([init(), loadAttractions()]);
        const t1 = performance.now();
        aux = !!msg.aux;
        oneByOne = !!msg.oneByOne;
        mine = msg.periods;
        mkt = msg.markets ?? mine;
        shareMarkets = !!msg.share;
        groupMasks = (msg.groups ?? []).map((g) => g.reduce((m, q) => m | (1 << q), 0));
        // T-173: one worker opens a small synthetic everyday layer first, so the browser has
        // optimised that code before the real cities open in every worker at once
        // (150: 22,500 cells, 900 zones; about 150 ms, mostly while the packs download)
        const everydayWarmMs = msg.warmEveryday ? everyday_warm_up(150) : 0;
        if (msg.rows && !marketPacks) marketRows = msg.rows;
        const cityInfo = openPacks(msg.packs, msg.intercity);
        const t2 = performance.now();
        const warmMs = warmUp() + everydayWarmMs;
        started = true;
        sendMessage({ kind: "ready", cityInfo, wasmMs: t1 - t0, openMs: t2 - t1, warmMs, info: cityInfo[HOME_CITY], periods: Array.from(demand_periods()), days: Array.from(day_factors()), memBytes: wasm_memory_bytes(), heap: heap() });
      } catch (err) {
        failed(err);
      }
    });
  } else if (msg.kind === "open") {
    // T-149: packs arrive one by one, the viewed city's first; a worker busy solving opens them
    // between populations (its job keeps the cities it started with); T-889: those waiting together
    // open together
    const epoch = closeEpoch;
    // T-1533: the zone in view's pack goes ahead of the opens waiting since the last close
    let at = openQueue.length;
    if (msg.first) while (at > 0 && openQueue[at - 1].epoch === epoch && !openQueue[at - 1].msg.first) at--;
    openQueue.splice(at, 0, { msg, epoch });
    opening = opening.then(() => openQueued(epoch));
  } else if (msg.kind === "close") {
    closeEpoch++;
    // in the opening chain, so a close and a reopen of the same pack keep their order
    opening = opening.then(() => {
      if (!started || dead) return;
      try { closePacks(msg); } catch (err) { failed(err); }
    });
  } else if (msg.kind === "solve") {
    // (a `freshParts` not yet taken up carries over to the solve that supersedes it)
    pending = pending?.freshParts && !msg.freshParts ? { ...msg, freshParts: true } : msg;
    wake?.();
    nudgePause();
    if (msg.only) startRough();
    else void opening.then(() => { if (started && !busy) run(); });
  } else if (msg.kind === "whole") {
    wholeGo = Math.max(wholeGo, msg.version);
    wake?.();
  } else if (msg.kind === "go") {
    const group = msg.group ?? 0;
    if (!goAllows(msg.version, group, msg.round)) go = { version: msg.version, group, round: msg.round };
    wake?.();
  } else if (msg.kind === "pause") {
    setPause(msg.on);
  } else if (msg.kind === "view") {
    viewDay = msg.day;
    viewZones = new Set(msg.zones);
  } else if (msg.kind === "first") {
    const i = openQueue.findIndex((q) => q.msg.packs.some((p) => p.id === msg.id));
    if (i > 0) {
      const [e] = openQueue.splice(i, 1);
      e.msg.first = true;
      let at = i;
      while (at > 0 && openQueue[at - 1].epoch === e.epoch && !openQueue[at - 1].msg.first) at--;
      openQueue.splice(at, 0, e);
    }
  } else if (msg.kind === "corridorsAgain") {
    sentTables.clear();
  } else if (msg.kind === "periods") {
    dealt = msg;
    if (!busy) deal();
  } else if (msg.kind === "loads") {
    inbox.set(`${msg.version}:${msg.cluster}:${msg.period}:${msg.round}`, msg.loads);
    wake?.();
  } else if (msg.kind === "debugPanic") {
    if (!import.meta.env.DEV) return;
    if (msg.when === "solve") panicInSolve = true;
    else try { panicNow(); } catch (err) { failed(err); }
  } else {
    query(msg);
  }
};

/**
 * Each local cell's 600 m square (the longer-trip market's endpoint sites are built from the same
 * cells, pipeline/intercity.py), and the commuters living and working in each square, so a
 * selection of cells can be turned into the share of every square it holds (T-118). Built once
 * per city on the first longer-trip query.
 */
interface Squares {
  key: Int32Array;
  total: Map<number, [number, number]>;
}
const SQUARE_M = 600, KEY_SPAN = 4096, KEY_OFFSET = 2048;
const squares = new Map<CityId, Squares>();
function squaresOf(id: CityId, a: DemandApi): Squares {
  let s = squares.get(id);
  if (s) return s;
  const xy = a.cell_xy(), w = a.cell_commuters(), n = xy.length / 2;
  const key = new Int32Array(n), total = new Map<number, [number, number]>();
  for (let c = 0; c < n; c++) {
    const p = cityToMarket(xy[2 * c], xy[2 * c + 1], id);
    const k = (Math.floor(p.x / SQUARE_M) + KEY_OFFSET) * KEY_SPAN + Math.floor(p.y / SQUARE_M) + KEY_OFFSET;
    key[c] = k;
    const t = total.get(k) ?? [0, 0];
    t[0] += w[c]; t[1] += w[n + c];
    total.set(k, t);
  }
  squares.set(id, (s = { key, total }));
  return s;
}
/** A selection's squares: their centres (world metres) and the share of each that is selected. */
function selectionSquares(id: CityId, a: DemandApi, end: number, cells: Uint32Array) {
  const s = squaresOf(id, a), n = s.key.length;
  const w = a.cell_commuters();
  const picked = new Map<number, number>();
  for (const c of cells) if (c < n) picked.set(s.key[c], (picked.get(s.key[c]) ?? 0) + w[end * n + c]);
  const xy = new Float32Array(2 * picked.size), share = new Float32Array(picked.size);
  let i = 0;
  for (const [k, sel] of picked) {
    const tot = s.total.get(k)![end];
    xy[2 * i] = (Math.floor(k / KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    xy[2 * i + 1] = ((k % KEY_SPAN) - KEY_OFFSET + 0.5) * SQUARE_M;
    share[i] = tot > 0 ? Math.min(1, sel / tot) : 1;
    i++;
  }
  return { xy, share };
}
/** A zone's place in the world market (T-377: every zone has one). */
const MARKET_OF: Record<CityId, number> = Object.fromEntries(CITIES.map(c => [c.id, c.place]));
const CITY_ORDER = new Map(CITIES.map((c, i) => [c.id, i]));

/** The demand views' queries (T-078). They run between periods, never inside a solve. */
function query(msg: Query) {
  // (T-1533: and the bubbles' sums read from the live state: a weekend's, one purpose's)
  const byRound = BY_ROUND.has(msg.kind) || msg.kind === "purposeModes" || (msg.kind === "subSums" && ((msg as { day?: number }).day ?? 0) !== 0);
  if ((msg as { wait?: boolean }).wait && msg.kind !== "cells" && byRound && !aux) {
    const { version, round } = msg;
    // T-402: a zone in a later group of the solve in hand waits for that group's round
    const g = version === current ? cityGroup(msg.city) ?? stageGroup : stageGroup;
    const behind = version > current || (version === current && busy && (g > stageGroup || (g === stageGroup && (stageRound < round || (stageRound === round && !units.every((q) => roundsHeld[q] === round))))));
    if (behind) return void waiting.push(msg);
  }
  // A city the network in hand was not set on (opened since, T-149) answers as if not opened.
  // (cells: a zone open bare answers too, T-325)
  // T-392: the city's cluster in the network asked about; while this worker holds it solved to the
  // last round, it answers for that network at once, whatever it is solving meanwhile
  const at = msg.kind === "cells" ? undefined : versions.get(msg.version);
  const cl = at?.clusters.find((c) => c.cities.includes(msg.city));
  const kept = !!cl && isHeld(cl);
  const a = msg.kind === "cells" ? cities.get(msg.city) ?? bare.get(msg.city)?.a : jobCities.has(msg.city) || cl ? cities.get(msg.city) : undefined;
  type WithoutCity<T> = T extends {city: CityId} ? Omit<T, "city"> : T;
  // T-285: when the question was taken up (epoch ms) and how long it took, for ?debug=1's timings
  const qStart = performance.timeOrigin + performance.now();
  const sendMessage = (reply: WithoutCity<FromDemand>, transfer: Transferable[] = []) => {
    postMessage({ ...reply, city: msg.city, qStart, qMs: performance.timeOrigin + performance.now() - qStart }, { transfer });
  };
  // T-402: the rounds this worker holds are its group's; a cluster of another group has none of them
  const atRound = (c: JobCluster | undefined) => msg.kind !== "cells" && msg.version === current && (!c || (groupOf.get(c) ?? stageGroup) === stageGroup) && units.every(q => roundsHeld[q] === msg.round);
  const live = !!a && msg.kind !== "cells" && (atRound(cl) || (kept && msg.round === at!.rounds));
  // the network's subzones, whatever round (cellModes, flows)
  const sameNet = msg.kind !== "cells" && ((msg.version === current && (!cl || !groupOf.has(cl) || ready.has(cl))) || kept);
  // the regional market's state at the round asked about (its clusters may not be the city's):
  // every part of it for a selection, the station's own for a station
  const clusterLive = (c: JobCluster) => msg.kind !== "cells" && (atRound(c) || (isHeld(c) && msg.round === at!.rounds));
  const mcs = at?.clusters.filter((c) => c.part) ?? [];
  const marketLive = mcs.length > 0 && mcs.every(clusterLive);
  // T-260, T-464: the day asked about (0 weekday, 1 Saturday, 2 Sunday, 3 average); every answer weighs its periods by it
  const day = (msg as { day?: number }).day ?? 0;
  try {
    if (msg.kind === "cells") {
      // not open here (a focus change raced the client's wait): an empty answer, never a pool failure
      if (!a) return sendMessage({ kind: "cells", id: msg.id, xy: new Float32Array(0), h3: new BigUint64Array(0), zone: new Uint32Array(0), zoneXY: new Float32Array(1) });
      setWorldFrame(msg.frame);
      const xy = a.cell_xy(), h3 = a.cell_h3(), zone = a.cell_zone(), zoneXY = a.zone_xy();
      for (let k = 0; k < xy.length; k += 2) {
        const p = cityToWorld(xy[k], xy[k + 1], msg.city); xy[k] = p.x; xy[k + 1] = p.y;
      }
      for (let k = 1; k < zoneXY.length; k += 2) {
        const p = cityToWorld(zoneXY[k], zoneXY[k + 1], msg.city); zoneXY[k] = p.x; zoneXY[k + 1] = p.y;
      }
      sendMessage({ kind: "cells", id: msg.id, xy, h3, zone, zoneXY }, [xy.buffer, h3.buffer, zone.buffer, zoneXY.buffer]);
    } else if (msg.kind === "subSums" && day !== 0) {
      // T-260: a weekend's sums are not kept per round (nobody asks while the weekday shows); they
      // are worked out from the state now, so only while every period holds the round asked about
      // (T-1904: or from the copy kept for the zones on screen when the round was whole)
      const copy = a && msg.version === current ? sumsByRound.get(`${msg.city}:${msg.round}:${day}`) : undefined;
      const sums = copy ? copy.sums.slice() : live ? a!.sub_sums_day(day) : null;
      const extra = copy ? copy.everyday?.slice() ?? null : live ? everyday.get(msg.city)?.cell_modes_day(day) ?? null : null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: copy ? copy.mask : live ? a!.solved_mask() : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "subSums") {
      // from the copy taken when the round was whole, not from the state now (see sumsByRound)
      // (or, for a cluster held to its last round, from its state, which has not moved since)
      const copy = a && msg.version === current ? sumsByRound.get(`${msg.city}:${msg.round}`) : undefined;
      const sums = copy ? copy.sums.slice() : live ? a!.sub_sums() : null;
      const extra = copy ? copy.everyday?.slice() ?? null : live ? everyday.get(msg.city)?.cell_modes() ?? null : null;
      sendMessage({ kind: "subSums", id: msg.id, version: msg.version, mask: copy ? copy.mask : live ? a!.solved_mask() : 0, sums, everyday: extra }, [...(sums ? [sums.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "cellModes") {
      // reads the network's subzones and the sums it is given, not any period's state, so it holds
      // for the version whatever round this worker is in
      const modes = a && sameNet ? a.cell_modes_day(msg.sums, day) : null;
      sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: modes && modes.length ? modes : null }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "station") {
      const packed = live ? stationRiders(at!, cl, a!, msg.station, day) : null;
      sendMessage({ kind: "station", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "flowRail") {
      const sums = live ? a!.flow_rail_day(msg.end, msg.cells, day) : null;
      // empty is a real answer on a network with no running line (no subzones, T-099)
      sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, sums }, sums ? [sums.buffer] : []);
    } else if (msg.kind === "purposes") {
      const work = live ? a!.purpose_timing_day(msg.end, msg.cells, day) : null;
      const layer = live ? everyday.get(msg.city) : undefined;
      // T-285: packed, only the far cells the selection's samples reach (demandProtocol `everyday`)
      const extra = layer?.selection_sparse_day(msg.end, msg.cells, day) ?? null;
      const names = EVERYDAY_PURPOSES.slice(0, layer?.purpose_count() ?? 0);
      sendMessage({ kind: "purposes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, work, everyday: extra, names }, [...(work ? [work.buffer] : []), ...(extra ? [extra.buffer] : [])]);
    } else if (msg.kind === "flows") {
      // reads the network's subzones and the sums it is given, not any period's state (as cellModes)
      const packed = a && sameNet ? a.flows_day(msg.end, msg.cells, msg.sums, day) : null;
      sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icSel") {
      // T-336: a zone open bare in the aux worker comes with its squares (this worker does not hold
      // the zone)
      const ok = marketLive && (!!msg.sq || !!a) && !!intercity;
      const sq = !ok ? null : msg.sq ?? selectionSquares(msg.city, a!, msg.end, msg.cells);
      // T-409, T-377: the all-mode trips and the zones once, from the market of every open pack,
      // then every part's rail
      let packed: Float64Array | null = null;
      if (sq) {
        const place = MARKET_OF[msg.city];
        packed = Float64Array.from(intercity!.selection_all_day(place, sq.xy, sq.share, day, maskOf(at?.markets ?? mine)));
        for (const c of mcs) {
          const v = c.part!.selection_rail_day(place, sq.xy, sq.share, day);
          for (let i = 0; i < v.length; i++) packed[i] += v[i];
        }
      }
      const mask = ok ? mcs.reduce((m, c) => m & c.part!.solved_mask(), 31) : 0;
      sendMessage({ kind: "icSel", id: msg.id, version: msg.version, mask, packed }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "icStation") {
      const sc = mcs.find((c) => (c.stLocal[msg.station] ?? -1) >= 0);
      const ok = live && (!sc || clusterLive(sc)) && !!intercity;
      const packed = ok ? Float64Array.from(marketStationFlows(at!, sc, intercity!, msg.station, day)) : null;
      sendMessage({ kind: "icStation", id: msg.id, version: msg.version, mask: ok ? (sc?.part ?? intercity!).solved_mask() : 0, packed: packed && packed.length ? packed : null }, packed ? [packed.buffer] : []);
    } else if (msg.kind === "purposeModes") {
      // one everyday purpose's legs (T-158); a city without that purpose answers empty, not null
      const layer = live ? everyday.get(msg.city) : undefined;
      const modes = !live ? null : layer && msg.purpose < layer.purpose_count() ? layer.cell_modes_of_day(msg.purpose, day) : new Float32Array(0);
      sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, modes }, modes ? [modes.buffer] : []);
    } else if (msg.kind === "transfers") {
      const pairs = live ? transfersAt(at!, msg.version === current && units.every(q => roundsHeld[q] === msg.round), msg.nodes, day) : null;
      sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, pairs }, pairs ? [pairs.buffer] : []);
    } else if (msg.kind === "trip") {
      // T-195: the trip as a commute, or as an everyday purpose (T-586), each period this worker
      // holds; the cells back in world metres
      // (T-979: a zone open bare is answered by the aux worker's copy, every period solved on no network)
      const b = aux ? bare.get(msg.city) : undefined;
      const api = b ? b.a : live ? a! : null;
      const answers: Float64Array[] = [], periods: number[] = [];
      if (api) {
        const o = worldToCity(msg.from[0], msg.from[1], msg.city), d = worldToCity(msg.to[0], msg.to[1], msg.city);
        const c = b ? undefined : cl;
        const k = msg.purpose ?? -1, layer = b ? b.e : everyday.get(msg.city);
        for (const q of b ? PERIODS.map((_, i) => i) : mine) {
          const v = k < 0 ? api.trip(q, o.x, o.y, d.x, d.y) : layer && k < layer.purpose_count() ? layer.trip(k, q, o.x, o.y, d.x, d.y) : new Float64Array(35);
          if (v[0] === 1) {
            const p0 = cityToWorld(v[27], v[28], msg.city), p1 = cityToWorld(v[29], v[30], msg.city);
            v[27] = p0.x; v[28] = p0.y; v[29] = p1.x; v[30] = p1.y;
            if (c) tripNodesGlobal(v, viewOf(c, api).nodeGlobal);
          }
          answers.push(v);
          periods.push(q);
        }
      }
      sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: api ? api.solved_mask() : 0, periods, answers: api ? answers : null }, answers.map((v) => v.buffer));
    } else if (msg.kind === "icTrip") {
      // T-866: each period from the market part with the most rail (each part's rail is its own
      // network's); with no part, an answer with no rail
      const ok = marketLive || (mcs.length === 0 && !!intercity && live);
      // T-885: a near place or 6 km zone instead of cells
      const near = ok ? msg.near : undefined;
      const sq = !ok || near ? null : msg.sq ?? (a ? selectionSquares(msg.city, a, msg.end, msg.cells) : null);
      const go = !!near || !!sq;
      const answers: Float64Array[] = [], periods: number[] = [];
      if (go) {
        const place = MARKET_OF[msg.city];
        for (const q of at?.markets ?? mine) {
          let best: Float64Array | null = null;
          for (const c of mcs) {
            const v = near ? c.part!.market_trip_between(q, day, near.place, near.gid, msg.end, msg.far, msg.farGid) : c.part!.market_trip(q, day, place, sq!.xy, sq!.share, msg.end, msg.far, msg.farGid);
            if (v[0] !== 1) continue;
            if (!best || v[3] + v[20] + v[23] > best[3] + best[20] + best[23]) { marketTripNodesGlobal(v, viewOf(c, c.part!).nodeGlobal); best = v; }
          }
          if (!best) { best = new Float64Array(MARKET_TRIP_LAYOUT.HEAD); best[0] = 1; best[14] = 2; best[13] = 1; best[6] = best[7] = best[24] = -1; }
          answers.push(best);
          periods.push(q);
        }
      }
      const mask = !go ? 0 : mcs.length ? mcs.reduce((m, c) => m & c.part!.solved_mask(), 31) : maskOf(at?.markets ?? mine);
      sendMessage({ kind: "icTrip", id: msg.id, version: msg.version, mask, periods, answers: go ? answers : null }, answers.map((v) => v.buffer));
    } else if (msg.kind === "bare") {
      // T-325: no network in it, so any version and round; the everyday legs per solving worker's
      // periods, as each would have sent them, so the main thread adds them up as it does those
      const b = bare.get(msg.city);
      const commute = !b ? null : msg.purpose < 0 ? b.a.cell_modes_day(new Float32Array(0), day) : new Float32Array(0);
      const e = msg.purpose === -2 ? null : b?.e;
      const parts = !e ? [] : groupMasks.map((m) => msg.purpose === -1 ? e.cell_modes_mask_day(m, day) : msg.purpose < e.purpose_count() ? e.cell_modes_of_mask_day(msg.purpose, m, day) : new Float32Array(0));
      sendMessage({ kind: "bare", id: msg.id, commute, masks: parts.length ? groupMasks : [], parts }, [...(commute ? [commute.buffer] : []), ...parts.map((p) => p.buffer)]);
    } else if (msg.kind === "bareFlows") {
      // T-336: a bare zone's selection with no rail: the commuters' far end as `flows` gives it
      // (no subzones, so no rail sums), and per solving worker's periods what its `purposes` would
      const b = bare.get(msg.city);
      const packed = b ? b.a.flows_day(msg.end, msg.cells, new Float32Array(0), day) : null;
      const work = b ? groupMasks.map((m) => b.a.purpose_timing_mask_day(msg.end, msg.cells, m, day)) : [];
      const e = b?.e;
      const extra = e ? groupMasks.map((m) => e.selection_sparse_mask_day(msg.end, msg.cells, m, day)) : [];
      const names = EVERYDAY_PURPOSES.slice(0, e?.purpose_count() ?? 0);
      const out = packed && packed.length ? packed : null;
      sendMessage({ kind: "bareFlows", id: msg.id, packed: out, masks: b ? groupMasks : [], work, everyday: extra, names }, [...(out ? [out.buffer] : []), ...work.map((v) => v.buffer), ...extra.map((v) => v.buffer)]);
    } else if (msg.kind === "bareSquares") {
      // T-336: the solving workers hold the market but not the zone: its squares come from here
      const b = bare.get(msg.city);
      const sq = b ? selectionSquares(msg.city, b.a, msg.end, msg.cells) : null;
      sendMessage({ kind: "bareSquares", id: msg.id, xy: sq?.xy ?? null, share: sq?.share ?? null });
    } else if (msg.kind === "driveRail") {
      const commute = live ? a!.drive_rail_day(day) : null;
      const everydayRail = live ? everyday.get(msg.city)?.drive_rail_day(day) ?? null : null;
      sendMessage({ kind: "driveRail", id: msg.id, version: msg.version, mask: live ? a!.solved_mask() : 0, commute, everyday: everydayRail }, [...(commute ? [commute.buffer] : []), ...(everydayRail ? [everydayRail.buffer] : [])]);
    } else if (msg.kind === "driveBase") {
      // no network in it: a zone open bare answers from the aux worker's copy (T-325)
      const b = bare.get(msg.city);
      const api = cities.get(msg.city) ?? b?.a;
      const layer = everyday.get(msg.city) ?? b?.e;
      const base = !api ? null : msg.part === 0 ? api.drive_base_day(day) : layer?.drive_base_day(day) ?? null;
      let squares: Float64Array | null = null;
      if (api && base) {
        squares = Float64Array.from(api.drive_squares());
        const n = squares[2];
        for (let i = 0; i < n; i++) [squares[3 + i], squares[3 + n + i]] = cityToLonLat(squares[3 + i], squares[3 + n + i], msg.city);
      }
      sendMessage({ kind: "driveBase", id: msg.id, squares, base }, [...(squares ? [squares.buffer] : []), ...(base ? [base.buffer] : [])]);
    }
  } catch (err) {
    // a failed query answers with nothing; the pool itself carries on (after a panic, with this
    // worker replaced)
    if (isPanic(err)) return died(err);
    console.error("demand query:", err);
    if (msg.kind === "subSums" || msg.kind === "station") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, ...(msg.kind === "subSums" ? { sums: null, everyday: null } : { packed: null }) } as FromDemand);
    else if (msg.kind === "purposes") sendMessage({kind: "purposes", id: msg.id, version: msg.version, mask: 0, work: null, everyday: null, names: []});
    else if (msg.kind === "cellModes") sendMessage({ kind: "cellModes", id: msg.id, version: msg.version, modes: null });
    else if (msg.kind === "flowRail") sendMessage({ kind: "flowRail", id: msg.id, version: msg.version, mask: 0, sums: null });
    else if (msg.kind === "flows") sendMessage({ kind: "flows", id: msg.id, version: msg.version, packed: null });
    else if (msg.kind === "icSel" || msg.kind === "icStation") sendMessage({ kind: msg.kind, id: msg.id, version: msg.version, mask: 0, packed: null });
    else if (msg.kind === "purposeModes") sendMessage({ kind: "purposeModes", id: msg.id, version: msg.version, mask: 0, modes: null });
    else if (msg.kind === "transfers") sendMessage({ kind: "transfers", id: msg.id, version: msg.version, mask: 0, pairs: null });
    else if (msg.kind === "trip") sendMessage({ kind: "trip", id: msg.id, version: msg.version, mask: 0, periods: [], answers: null });
    else if (msg.kind === "icTrip") sendMessage({ kind: "icTrip", id: msg.id, version: msg.version, mask: 0, periods: [], answers: null });
    else if (msg.kind === "bare") sendMessage({ kind: "bare", id: msg.id, commute: null, masks: [], parts: [] });
    else if (msg.kind === "bareFlows") sendMessage({ kind: "bareFlows", id: msg.id, packed: null, masks: [], work: [], everyday: [], names: [] });
    else if (msg.kind === "bareSquares") sendMessage({ kind: "bareSquares", id: msg.id, xy: null, share: null });
    else if (msg.kind === "driveRail") sendMessage({ kind: "driveRail", id: msg.id, version: msg.version, mask: 0, commute: null, everyday: null });
    else if (msg.kind === "driveBase") sendMessage({ kind: "driveBase", id: msg.id, squares: null, base: null });
  }
}

/** T-327: ask the waiting questions again; those still ahead of this worker wait on. */
function answerWaiting() {
  const w = waiting;
  waiting = [];
  for (const m of w) query(m);
}

/**
 * Transfers with either end at one of `nodes` (T-176): every population of the job, every period
 * this worker holds, each (arriving, leaving) route node pair once. Asked for one station complex
 * on selection, so the answer is a few thousand numbers, never the whole network's.
 */
function transfersAt(at: Version, allLive: boolean, nodes: Uint32Array, day: number): Float32Array {
  const sum = new Map<number, number>(), jobNodes = at.nodes;
  for (const c of at.clusters) {
    // T-392: a cluster mid-solve has no figures for this network yet (T-402: nor one of a later group)
    if (!isHeld(c) && !(allLive && (groupOf.get(c) ?? stageGroup) === stageGroup)) continue;
    for (const p of c.pops) {
      const view = viewOf(c, p), local = nodeLocal(view, jobNodes);
      const sel = new Uint8Array(view.nodeGlobal.length);
      let any = false;
      for (const r of nodes) {
        const l = r < jobNodes ? local[r] : -1;
        if (l >= 0) (sel[l] = 1), (any = true);
      }
      if (!any) continue;
      const v = p.transfers_day(sel, day);
      for (let k = 0; k + 2 < v.length; k += 3) {
        const key = view.nodeGlobal[v[k]] * jobNodes + view.nodeGlobal[v[k + 1]];
        sum.set(key, (sum.get(key) ?? 0) + v[k + 2]);
      }
    }
  }
  const out = new Float32Array(3 * sum.size);
  let o = 0;
  for (const [key, riders] of sum) {
    out[o++] = Math.floor(key / jobNodes); out[o++] = key % jobNodes; out[o++] = riders;
  }
  return out;
}

/**
 * T-326: a station's riders from one city, on its own network (T-756). A station outside it
 * gets the answer a population set on the whole network gives for a station it cannot reach: no
 * riders, every zone listed (`DemandApi.station_riders_day`).
 */
function stationRiders(at: Version, c: JobCluster | undefined, a: DemandApi, station: number, day: number): Float32Array {
  const view = c?.views.get(a);
  const ls = view && view.net.lineN.length ? view.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return a.station_riders_day(ls, day);
  return at.railed && station < at.stations ? new Float32Array(4 + 2 * a.info()[1]) : new Float32Array(0);
}
/** The same for the market's longer trips (`IntercityApi.station_flows_day`), from the part of
 * the station's cluster `c` (T-409); `m` answers for a station in no market cluster. */
function marketStationFlows(at: Version, c: JobCluster | undefined, m: IntercityApi, station: number, day: number): Float64Array {
  const ls = c ? c.stLocal[station] ?? -1 : -1;
  if (ls >= 0) return c!.part!.station_flows_day(ls, day);
  return station < at.stations ? m.no_station_flows() : new Float64Array(0);
}

/**
 * T-1008: the market's corridor table (game/demand.ts `CorridorTable`): the pairs with longer trips
 * or commutes, which depend on neither the period nor the network (most of the market's place
 * pairs have none), with a key of all of it. `row` maps a market corridor to its row, -1 none.
 * Made once per market part; the main thread is sent each key once (`sentTables`), since its tens
 * of MB in every period's result cost it a frame or more each (T-790). T-1204: the parts and
 * compositions of one world share their corridors (`corridors_id`), so the table is made once for
 * them all (it was tens of MB a part, and as many again to make). T-2122: the worker keeps only the
 * rows' corridors (`idx`, ascending, which a period's rail is matched against); the columns go to the
 * main thread as made and are made again if it asks (`corridorsAgain`), so no whole-world array
 * (12.7M corridors in 2026-11, against 3.9M rows) is held or copied here.
 */
interface MadeTable { key: string; idx: Uint32Array; cols: CorridorTable | null }
let madeTable: { id: number; made: MadeTable } | null = null;
const sentTables = new Set<string>();
function tableColumns(p: IntercityApi): CorridorTable {
  const pairs = p.corridor_rows_pairs(), longer = p.corridor_rows_trips(), base = p.corridor_rows_commutes(), modes = p.corridor_rows_modes();
  const n = longer.length, hasM = modes.length === 2 * n;
  const t: CorridorTable = { key: "", n, from: new Uint32Array(n), to: new Uint32Array(n), longer, commutes: base.length === n ? base : new Float64Array(n), air: new Float32Array(n), car: new Float32Array(n) };
  for (let j = 0; j < n; j++) {
    t.from[j] = pairs[2 * j]; t.to[j] = pairs[2 * j + 1];
    // T-378: before any shares are known, all by car
    if (hasM) { t.air[j] = modes[2 * j]; t.car[j] = modes[2 * j + 1]; } else t.car[j] = 1;
  }
  // FNV-1a over the bits of every column
  let h = 0x811c9dc5;
  for (const a of [t.from, t.to, t.longer, t.commutes, t.air, t.car]) {
    const w = new Uint32Array(a.buffer, a.byteOffset, a.byteLength / 4);
    for (let i = 0; i < w.length; i++) h = Math.imul(h ^ w[i], 0x01000193);
  }
  t.key = `${n}:${(h >>> 0).toString(36)}`;
  return t;
}
function corridorTable(p: IntercityApi): MadeTable {
  const id = p.corridors_id();
  if (madeTable?.id === id) return madeTable.made;
  const cols = tableColumns(p);
  const made = { key: cols.key, idx: p.corridor_rows(), cols };
  madeTable = { id, made };
  return made;
}
/** The row of corridor `c` in `idx`, -1 none. */
function rowOf(idx: Uint32Array, c: number): number {
  let lo = 0, hi = idx.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1, v = idx[mid];
    if (v === c) return mid;
    if (v < c) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

/** T-1008: period `q`'s corridor figures over the market's parts: the table, the period's shares
 * (the first part's that solved it; the trips are the same in every part) and the pairs with rail,
 * each part's own network's added up, by table row, ascending. */
function marketCorridors(parts: IntercityApi[], q: number) {
  if (!parts.length) return null;
  const made = corridorTable(parts[0]);
  let shares: number[] = [];
  const rail = new Map<number, [number, number]>();
  for (const p of parts) {
    const s = p.corridor_shares(q);
    if (!s.length) continue;
    if (!shares.length) shares = Array.from(s);
    const v = p.corridor_rail(q);
    for (let i = 0; i < v.length; i += 3) {
      const j = rowOf(made.idx, v[i]);
      if (j < 0) continue;
      const r = rail.get(j);
      if (r) { r[0] += v[i + 1]; r[1] += v[i + 2]; } else rail.set(j, [v[i + 1], v[i + 2]]);
    }
  }
  const rows = [...rail.keys()].sort((a, b) => a - b);
  const out = new Float64Array(3 * rows.length);
  rows.forEach((j, k) => { const [r, c] = rail.get(j)!; out[3 * k] = j; out[3 * k + 1] = r; out[3 * k + 2] = c; });
  // (transferred: made again for a `corridorsAgain`)
  let table: CorridorTable | undefined;
  if (!sentTables.has(made.key)) {
    table = made.cols ?? tableColumns(parts[0]);
    made.cols = null;
    made.key = table.key;
    sentTables.add(table.key);
  }
  return { corridorKey: made.key, corridorTable: table, corridorShares: shares.length >= 2 ? [shares[0], shares[1]] as [number, number] : undefined, corridorRail: out };
}

/**
 * T-409: the market's figures in period q over its parts (`jobClusters` order). Each part's rail
 * is its own network's; the trips, pairs and corridor trips need no network and are the same in
 * every part, so they are taken once. One part gives its own figures as they are.
 */
function marketFigures(parts: IntercityApi[], q: number) {
  const of = (read: (p: IntercityApi) => Float64Array, rail: (i: number) => boolean) => {
    let out: number[] = [];
    for (const p of parts) {
      const v = read(p);
      if (!v.length) continue;
      if (!out.length) out = Array.from(v);
      else for (let i = 0; i < v.length; i++) if (rail(i)) out[i] += v[i];
    }
    return out;
  };
  // round, trips, rail, 0, ms, pairs, segments over seats, then the worst loads: the largest part's;
  // then the fare sensitivity (T-1300)
  const summary = of((p) => p.summary(q), (i) => i === 2 || i === 4 || i === 6 || i >= 10);
  if (parts.length > 1 && summary.length) for (const i of [7, 8, 9]) summary[i] = Math.max(...parts.map((p) => p.summary(q)[i] ?? 0));
  // the longer trips, the commutes, the everyday pairs (T-1175), then (T-1229) the pairs by purpose,
  // four each: only the rail is a part's own
  const ps = of((p) => p.part_summary(q), (i) => (i < 6 ? i % 2 === 1 : i >= 8 && (i - 8) % 4 === 1));
  // T-1175: the everyday pairs' walkers are every pair's with no rail (the same in every part) less
  // those each part's rail takes, in all and for each purpose
  if (parts.length > 1 && ps.length >= 7) {
    const each = parts.map((p) => p.part_summary(q)).filter((v) => v.length >= 7);
    for (const [at, base] of [[6, 7], ...Array.from({ length: Math.max(0, (ps.length - 8) >> 2) }, (_, k) => [8 + 4 * k + 2, 8 + 4 * k + 3])]) {
      const b = each[0][base] ?? each[0][at];
      ps[at] = Math.max(0, b - each.reduce((s, v) => s + (b - v[at]), 0));
    }
    if (summary.length) summary[3] = ps[6];
  }
  return {
    summary,
    parts: ps,
    ...marketCorridors(parts, q),
    // T-378: the same in every part
    elsewhere: parts.length ? Array.from(parts[0].elsewhere(q)) : [],
  };
}
/** A `DemandApi.trip` answer's route nodes (trip.ts `PAIR_LEN`, `LEG_LEN`) from its cluster's
 * network to the whole network's, the ones the client reads. */
function tripNodesGlobal(v: Float64Array, nodeGlobal: Uint32Array) {
  const { HEAD, PAIR_LEN, LEG_LEN } = TRIP_LAYOUT;
  const g = (i: number) => { if (v[i] >= 0) v[i] = nodeGlobal[v[i]]; };
  const np = v[24], nr = v[25];
  for (let k = 0; k < np; k++) { g(HEAD + k * PAIR_LEN + 2); g(HEAD + k * PAIR_LEN + 3); }
  let o = HEAD + np * PAIR_LEN;
  for (let k = 0; k < nr; k++) {
    const n = v[o + 2];
    o += 3;
    for (let i = 0; i < n; i++, o += LEG_LEN) { g(o + 1); g(o + 2); }
  }
}

/** The same for an `IntercityApi.market_trip` answer (T-866: head 24 since T-880, pairs of 7, legs of 12). */
function marketTripNodesGlobal(v: Float64Array, nodeGlobal: Uint32Array) {
  const { HEAD, PAIR_LEN } = MARKET_TRIP_LAYOUT, { LEG_LEN } = TRIP_LAYOUT;
  const g = (i: number) => { if (v[i] >= 0) v[i] = nodeGlobal[v[i]]; };
  const np = v[11], nr = v[12];
  for (let k = 0; k < np; k++) { g(HEAD + k * PAIR_LEN); g(HEAD + k * PAIR_LEN + 1); }
  let o = HEAD + np * PAIR_LEN;
  for (let k = 0; k < nr; k++) {
    const n = v[o + 2];
    o += 3;
    for (let i = 0; i < n; i++, o += LEG_LEN) { g(o + 1); g(o + 2); }
  }
}

/**
 * A period's `seg`, `board` and `alight` split by trip purpose (T-158, demandProtocol `parts`):
 * work is every city's commutes, then the everyday layer's purposes, then the regional market's
 * longer trips. They add up to the result's totals; crowding is never split. Each cluster's route
 * nodes go to their place in the whole network's (T-326).
 */
function purposeParts(q: number, nr: number, clusters: typeof jobClusters, regional: boolean, pick: (a: Population) => boolean = () => true) {
  const days = clusters.flatMap((c) => c.pops.filter((p): p is EverydayApi => p instanceof EverydayApi));
  const kinds = Math.min(EVERYDAY_KEYS.length, Math.max(0, ...days.map((e) => e.purpose_count())));
  const keys: string[] = ["work", ...EVERYDAY_KEYS.slice(0, kinds)];
  if (regional) keys.push("longer");
  const seg = new Float32Array(keys.length * nr), board = new Float32Array(keys.length * nr), alight = new Float32Array(keys.length * nr);
  // T-1373: the market's longer trips times their lean (`IntercityApi.lean_loads`), seg, board and
  // alight one after another
  let lean: Float32Array | undefined;
  for (const c of clusters) {
    // T-756: each population's route nodes go through its own view
    let map = c.nodeGlobal, m = map.length;
    const to = (p: Population) => { map = viewOf(c, p).nodeGlobal; m = map.length; };
    const add = (out: Float32Array, at: number, v: ArrayLike<number>, from = 0) => {
      if (v.length < from + m) return; // nothing solved for this population (no network)
      for (let r = 0; r < m; r++) out[at * nr + map[r]] += v[from + r];
    };
    for (const a of c.pops) if (a instanceof DemandApi && pick(a)) { to(a); add(seg, 0, a.seg(q)); add(board, 0, a.board(q)); add(alight, 0, a.alight(q)); }
    for (const e of c.pops) {
      if (!(e instanceof EverydayApi) || !pick(e)) continue;
      to(e);
      const v = e.part_loads(q), k1 = Math.min(kinds, e.purpose_count());
      if (v.length !== 3 * e.purpose_count() * m) continue;
      for (let k = 0; k < k1; k++) { add(seg, 1 + k, v, 3 * k * m); add(board, 1 + k, v, (3 * k + 1) * m); add(alight, 1 + k, v, (3 * k + 2) * m); }
    }
    if (regional && c.part && pick(c.part)) {
      const at = keys.length - 1, part = c.part;
      to(part);
      add(seg, at, part.seg(q)); add(board, at, part.board(q)); add(alight, at, part.alight(q));
      // T-276: the regional layer's commutes between zones are work: moved from longer trips to it
      const v = part.commute_loads(q);
      const move = (v: Float32Array, from: number, to: number) => {
        for (let r = 0; r < m; r++) {
          const g = map[r];
          seg[to * nr + g] += v[from + r]; seg[at * nr + g] -= v[from + r];
          board[to * nr + g] += v[from + m + r]; board[at * nr + g] -= v[from + m + r];
          alight[to * nr + g] += v[from + 2 * m + r]; alight[at * nr + g] -= v[from + 2 * m + r];
        }
      };
      if (v.length === 3 * m) move(v, 0, 0);
      // T-1175: its everyday outings across a zone line are their purposes', a block each
      const e = part.everyday_loads(q);
      if (e.length && e.length % (3 * m) === 0) for (let k = 0; k < Math.min(kinds, e.length / (3 * m)); k++) move(e, 3 * k * m, 1 + k);
      const l = part.lean_loads(q);
      if (l.length === 3 * m) {
        lean ??= new Float32Array(3 * nr);
        for (let k = 0; k < 3; k++) for (let r = 0; r < m; r++) lean[k * nr + map[r]] += l[k * m + r];
      }
    }
  }
  return { keys, seg, board, alight, ...(lean ? { lean } : {}) };
}

/**
 * T-409: a market part for every market cluster's key: the part that had the key, else one of the
 * newest composition whose key is gone (set up afresh), the composed market itself first, else a
 * new one beside it. Parts left over are freed (never the composed market itself).
 */
function assignParts(regional: IntercityApi, split: Cluster[]) {
  const keys = [...new Set(split.filter((c) => c.market).map((c) => c.marketKey))];
  const spare = [...marketParts].filter(([k]) => !keys.includes(k)).map(([k, p]) => (marketParts.delete(k), p));
  const used = new Set(marketParts.values());
  const usable = spare.filter((p) => p !== regional && regional.same_market(p));
  if (!used.has(regional)) usable.unshift(regional);
  for (const k of keys) {
    if (marketParts.has(k)) continue;
    let p = usable.shift();
    if (!p) { p = regional.beside(); composedOf.set(p, composedOf.get(regional) ?? ""); }
    marketParts.set(k, p);
    used.add(p);
  }
  for (const p of new Set(spare)) {
    if (used.has(p) || p === regional) continue;
    held.delete(p);
    heldQ.delete(p);
    p.free();
  }
}

async function run() {
  busy = true;
  try {
    while (pending && started && !dead) {
      // T-1533: a rough solve waits until its zones are open here (`startRough`)
      if (pending.only && !roughReady(pending)) break;
      setRough(!!pending.only);
      closeDeferred();
      if (dealt) deal();
      wasm_heap_peak(true);
      let job = pending;
      pending = null;
      // (a worker's fresh market parts are for the next whole solve, not a rough one)
      if (job.only && job.freshParts) { freshCarry = true; job = { ...job, freshParts: false }; }
      else if (!job.only && freshCarry) { freshCarry = false; job = { ...job, freshParts: true }; }
      const n = job.net;
      remapKept(job.ids ? { lines: job.ids.lines, stations: job.ids.stations, lineN: n.lineN, stops: n.stops, tph: n.tph, times: n.times } : null);
      setWorldFrame(n.frame);
      let setupMs = 0;
      // the cities open now: a pack opened during this job waits for the next one (T-149)
      // in the registry's order, not the order the packs happened to open in: the figures are added
      // up in it, so the same network gives the same figures to the bit however the packs arrived
      const byRegistry = <T,>(list: [CityId, T][]) => list.sort((x, y) => CITY_ORDER.get(x[0])! - CITY_ORDER.get(y[0])!);
      // T-1533: a rough solve covers its zones alone
      const only = job.only ? new Set(job.only) : null;
      const jobLocal = byRegistry([...cities].filter(([id]) => !only || only.has(id)));
      const jobEveryday = byRegistry([...everyday].filter(([id]) => cities.has(id) && (!only || only.has(id))));
      jobCities = new Set(jobLocal.map(([id]) => id));
      // T-1174: after a worker is replaced, every worker composes its market afresh from the packs
      // open now, as the replacement did, so all hold the same market and corridor table
      // (T-1205: not in a worker started since, whose one composition is already from every pack open:
      // a second, fresh corridors and all, left a replacement's memory about 30% above its need)
      if (job.freshParts && marketPacks && solvedOnce && !only) {
        const old = [...new Set([...marketParts.values(), ...(intercity ? [intercity] : [])])];
        marketParts.clear();
        intercity = null;
        marketZones = null;
        forget(old);
        marketPacks.forget_corridors();
        recompose();
      }
      const regional = only ? null : intercity;
      // T-1607: in a solve with the barrier the pool may set a period's market apart from its
      // cities (`mkt`, `mine`): each side solves its populations from the same summed loads, the
      // two trading their loads every round through the page, so the figures are one worker's
      const sharing = shareMarkets && !!job.barrier && !only && !!regional;
      const marketQs = sharing ? mkt : mine;
      units = [...new Set([...mine, ...marketQs])].sort((a, b) => a - b);
      for (const k of [...inbox.keys()]) if (Number(k.slice(0, k.indexOf(":"))) < job.version) inbox.delete(k);
      // T-326: each cluster on its own network; one this worker has solved on the same network
      // keeps its populations' results, untouched
      for (const [id, a] of jobLocal) if (!boxes.has(id)) boxes.set(id, cellBox(a.cell_xy()));
      if (regional && !marketZones) marketZones = regional.zone_boxes();
      const split = splitClusters(n, jobLocal.map(([id]) => ({ id, box: boxes.get(id) ?? null })), regional ? marketZones : null, job.rounds, job.whole, job.oneMarket);
      if (regional) assignParts(regional, split);
      const partOf = (c: Cluster) => (c.market && regional ? [marketParts.get(c.marketKey)!] : []);
      jobClusters = split.map((c) => ({ ...c, part: partOf(c)[0] ?? null, pops: [
        ...c.cities.map((id) => cities.get(id)!),
        ...jobEveryday.filter(([id]) => c.cities.includes(id)).map(([, e]) => e),
        ...partOf(c),
      ], solveOrder: [
        // the order the packs opened in (the viewed zone's first); the order changes no figure. Set
        // up in registry order instead, the real save's first round-0 solve on each worker is ~1 s
        // slower in WASM (T-405, notes/T-405.md; the skim cache no longer depends on the order)
        ...[...cities.keys()].filter((id) => c.cities.includes(id)).map((id) => cities.get(id)!),
        ...[...everyday.keys()].filter((id) => c.cities.includes(id) && jobCities.has(id)).map((id) => everyday.get(id)!),
        ...partOf(c),
      ], reuse: 0, views: new Map<Population, PopView>() }));
      // T-756: a city's layers solve on the stations within its reach (`?perfOff=reach`: on its
      // whole cluster); the market's longer trips on the whole cluster
      for (const c of jobClusters) {
        const whole = clusterView(c);
        for (const id of c.cities) {
          const v = job.wide || job.whole ? whole : reachView(n, c, cityReach(n, id, boxes.get(id) ?? null));
          c.views.set(cities.get(id)!, v);
          const e = everyday.get(id);
          if (e && c.pops.includes(e)) c.views.set(e, v);
        }
        if (c.part) c.views.set(c.part, whole);
      }
      // T-1862: a cluster whose populations hold it whole-solved on the same shape, its run times and
      // trains an hour within noise of those it was solved on, is re-keyed to this network unsolved
      // (one last solved sharing stays dirty below, kept apart by T-1733's path with nothing fresh)
      let noiseKept = 0;
      if (!job.exactKeep && !job.fresh && !job.freshParts && !sharing && !only && mine.length) {
        const all = maskOf(mine), near = new Map<DemandNetwork, boolean>();
        for (const c of jobClusters) {
          const was = c.pops.length ? held.get(c.pops[0]) : undefined;
          if (was === undefined || was === c.sig) continue;
          const sh = c.part ? sharedMkt.get(c.part) : undefined;
          const keep = c.pops.every((p) => {
            const on = heldOn.get(p), h = heldQ.get(p), m = p.solved_mask();
            if (!on || held.get(p) !== was || on.sig !== was || on.shape !== c.shape) return false;
            if (!mine.every((q) => h?.[q] === on.psig[q])) return false;
            if (p === c.part ? !mine.every((q) => m & (1 << q) || sh?.[q]) : (m & all) !== all) return false;
            if (!near.has(on.net)) near.set(on.net, withinNoise(on.net, c.net));
            return near.get(on.net)!;
          });
          if (!keep) continue;
          for (const p of c.pops) {
            held.set(p, c.sig);
            const h = heldQ.get(p)!, on = heldOn.get(p)!;
            for (const q of units) h[q] = c.psig[q];
            on.sig = c.sig;
            on.psig = c.psig;
          }
          const r = capRatio.get(was);
          if (r) { capRatio.delete(was); capRatio.set(c.sig, r); }
          noiseKept++;
        }
        if (import.meta.env.DEV && noiseKept) console.info(`T1862 v${job.version}: ${noiseKept} clusters kept within noise of the network they solved on`);
      }
      // (sharing, every market cluster is solved, since both sides of a period must step it; and
      // not sharing, every one last solved sharing, whose shared periods are held apart)
      const dirty = jobClusters.filter((c) => (!!c.part && (sharing || sharedParts.has(c.part))) || c.pops.some((p) => held.get(p) !== c.sig));
      // which of a cluster's populations this worker solves period q of
      const sideOf = (c: JobCluster, q: number): Side | null => {
        const ci = mine.includes(q), mk = marketQs.includes(q);
        if (!c.part) return ci ? "all" : null;
        return ci && mk ? "all" : ci ? "cities" : mk ? "market" : null;
      };
      const shared = (q: number) => mine.includes(q) !== marketQs.includes(q);
      // the other side's loads of a shared period after round r (null: a newer network came)
      const partner = async (c: JobCluster, q: number, r: number): Promise<Float32Array | null> => {
        const key = `${job.version}:${jobClusters.indexOf(c)}:${q}:${r}`;
        while (!inbox.has(key) && !pending) {
          await new Promise<void>((res) => (wake = res));
          wake = null;
        }
        return pending ? null : inbox.get(key)!;
      };
      // T-393: the cluster of the city in view first
      if (job.first) {
        const i = dirty.findIndex((c) => c.cities.includes(job.first!));
        if (i > 0) dirty.unshift(...dirty.splice(i, 1));
      }
      // T-392: after an edit (no barrier) the clusters are solved one after another, each to its last
      // round, and each but the last is sent as a cluster result the moment it is done; then the
      // solve's results. With the barrier (the shown estimates step through the rounds) they all go
      // round by round together, in groups (T-402): the city in view's cluster alone first, sent as
      // cluster results round by round, then the rest, whose results are the solve's.
      // T-1609: a cluster whose network is the one its populations hold solved, and whose only new
      // populations are zones opened again at another level (`fresh`), solves those alone against
      // the others' loads frozen from the last solve (`localSolve`)
      const localOf = new Map<JobCluster, Population[]>();
      if (!job.barrier && !job.fresh && !job.noLocal && !only && mine.length) {
        const all = maskOf(mine);
        for (const c of dirty) {
          // T-1733: a cluster last solved with its market set apart (T-1607) is kept that way: its
          // market's shared periods are the other worker's, their loads here in `sharedMkt`; one
          // dirty only for that solves nothing
          const apart = !!c.part && sharedParts.has(c.part), sh = c.part ? sharedMkt.get(c.part) : undefined;
          const fresh = c.pops.filter((p) => held.get(p) !== c.sig);
          if ((!fresh.length && !apart) || fresh.length === c.pops.length || fresh.some((p) => p instanceof IntercityApi)) continue;
          const holds = (p: Population) => (p === c.part ? mine.every((q) => p.solved_mask() & (1 << q) || sh?.[q]) : (p.solved_mask() & all) === all);
          const frozen = c.pops.every((p) => fresh.includes(p) || (holds(p) && mine.every((q) => { const h = heldQ.get(p)?.[q]; return h === c.psig[q] || h === c.psig[q] + LOCAL; })));
          if (frozen) localOf.set(c, fresh);
        }
        // (kept apart only if every such cluster is: one solved whole joins its market's periods
        // back to its cities' worker, and the results below then split no period)
        if (jobClusters.some((c) => c.part && sharedParts.has(c.part) && !localOf.has(c))) for (const c of [...localOf.keys()]) if (c.part && sharedParts.has(c.part)) localOf.delete(c);
      }
      // T-1733: the periods split between workers by T-1607 that stay split in this solve: this
      // worker's cities' periods whose market is the other's (`cq`), and the markets it holds of the
      // other's periods (`mq`); their results go as the two sides' parts
      const apartKept = !job.barrier && shareMarkets && [...localOf.keys()].some((c) => c.part && sharedParts.has(c.part));
      const cq = apartKept ? mine.filter((q) => !mkt.includes(q)) : [], mq = apartKept ? mkt.filter((q) => !mine.includes(q)) : [];
      // T-1733: after an edit, a cluster whose market part and untouched cities still hold what they
      // last solved (the cities on the same inputs, `sameView`) first solves only the cities the edit
      // touched against those frozen loads; that result is sent, then the cluster is solved whole
      // as usual. `market`: the part's route nodes on this network; `sig`: the cluster it last solved
      const editOf = new Map<JobCluster, { fresh: Population[]; market: Uint32Array | null; sig: string | undefined }>();
      if (!job.barrier && !job.fresh && !job.noLocal && !only && mine.length) {
        const all = maskOf(mine);
        for (const c of dirty) {
          if (localOf.has(c)) continue;
          let market: Uint32Array | null = null;
          if (c.part) {
            const g = marketOn.get(c.part), m = c.part.solved_mask(), sh = sharedMkt.get(c.part);
            if (!g || !mine.every((q) => m & (1 << q) || sh?.[q])) continue;
            market = g;
          }
          const intact = (p: Population) => cityOn.has(p) && (p.solved_mask() & all) === all && sameView(cityOn.get(p)!, viewOf(c, p));
          const fresh = c.pops.filter((p) => !(p instanceof IntercityApi) && !intact(p));
          // (something must stay frozen: the market part, or a city; on all of Japan the cities are a fifth of a solve)
          if (!fresh.length || (!c.part && fresh.length === c.pops.length)) continue;
          const kept = c.part ?? c.pops.find((p) => !fresh.includes(p))!;
          editOf.set(c, { fresh, market, sig: held.get(kept) });
        }
      }
      const early = job.barrier ? [] : dirty;
      // (T-2045: not in a solve of `only` zones, whose clusters are all the zone in view's)
      const ahead = job.barrier && !job.whole && !only && dirty.length > 1 && !!job.first && dirty[0].cities.includes(job.first);
      // T-2045: the lines a solve of `only` zones gives figures, so the page puts in only those
      const onlyLines = only ? () => ({ onlyLines: Uint32Array.from(new Set(jobClusters.flatMap((c) => [...c.lines]))) }) : () => ({});
      const groups = !job.barrier ? [] : ahead ? [[dirty[0]], dirty.slice(1)] : [dirty];
      groupOf.clear();
      ready.clear();
      for (const c of early) groupOf.set(c, 0);
      groups.forEach((cs, g) => { for (const c of cs) groupOf.set(c, g); });
      stageGroup = 0;
      let periodsKept = 0, periodsSolved = 0;
      const setUpPops = (c: JobCluster, pops: Population[]) => {
        for (const p of pops) {
          const m = viewOf(c, p).net;
          held.delete(p);
          disturbed(p);
          p.set_complexes(m.complex);
          p.set_upgrades(m.ups ?? new Uint8Array(0));
          p.set_passages(m.passages ?? new Uint32Array(0));
          p.set_span(m.span ?? new Float32Array(0));
          // T-1277: riders weigh fares by their country's income and pay the curve times its scale;
          // the market takes the curve as set, at the baseline country's weight, and (T-1286) each
          // trip's at its origin's scale over its income's
          let scale = 1, rounded: ReturnType<typeof roundedDefault> = null;
          if (p instanceof IntercityApi) {
            p.set_fare_weight(fareWeight(FARE_BASELINE));
            p.set_fare_factors(JSON.stringify(fareFactors(m.fare.scales)));
            p.set_sampled(!job.exactMarket);
            // T-377: the market takes stations as longitude and latitude, each zone in its own frame
            const ll = new Float64Array(m.stXY.length);
            for (let k = 0; k < ll.length; k += 2) [ll[k], ll[k + 1]] = worldToLonLat(m.stXY[k], m.stXY[k + 1]);
            // T-1464: the bus corridors' stretches, as the stations
            const cxy = m.catchXY ?? new Float32Array(0), cll = new Float64Array(cxy.length);
            for (let k = 0; k < cll.length; k += 2) [cll[k], cll[k + 1]] = worldToLonLat(cxy[k], cxy[k + 1]);
            p.set_bus_ll(m.catchN ?? new Uint32Array(0), cll);
            setupMs += p.set_network_ll(ll, m.lineN, m.stops, m.times, m.tph, m.cars);
          } else {
            if (p instanceof DemandApi) p.set_chunked(!job.exactMarket);
            const city = c.cities.find((id) => cities.get(id) === p || everyday.get(id) === p)!;
            const country = cityById(city).country;
            p.set_fare_weight(fareWeight(country));
            scale = fareScale(country, m.fare.scales);
            rounded = roundedDefault(m.fare, country); // T-1662
            const xy = m.stXY.slice();
            for (let k = 0; k < xy.length; k += 2) {
              const w = worldToCity(xy[k], xy[k + 1], city);
              xy[k] = w.x; xy[k + 1] = w.y;
            }
            const cxy = m.catchXY ? m.catchXY.slice() : new Float32Array(0);
            for (let k = 0; k < cxy.length; k += 2) {
              const w = worldToCity(cxy[k], cxy[k + 1], city);
              cxy[k] = w.x; cxy[k + 1] = w.y;
            }
            p.set_bus(m.catchN ?? new Uint32Array(0), cxy);
            setupMs += p.set_network(xy, m.lineN, m.stops, m.times, m.tph, m.cars);
          }
          p.set_fares(rounded ? rounded.base : m.fare.base * scale, rounded ? rounded.perKm : m.fare.perKm * scale, m.hopKm);
          p.set_fare_mul(m.fareW ?? new Float32Array(0));
          p.set_fare_curve(rounded ? curveKnots(rounded) : scaledKnots(m.fare, scale));
          // T-1801: the market's trips change lines free (its fares weigh by income group)
          if (!(p instanceof IntercityApi)) p.set_transfer_fare((m.fare.transfer ?? 0) * scale);
        }
      };
      const setUp = (c: JobCluster) => {
        setUpPops(c, c.solveOrder);
        // T-124: a period is kept only if every population kept it (its inputs are unchanged: the
        // Rust side) and each holds it from a whole solve of this cluster's period (`psig`): the
        // crowding is the whole cluster's
        // T-1607: a period this worker does not solve a population's side of is held by another
        // worker (or by none): dropped, so no answer counts it twice. A shared period is never
        // kept, since one side always lacks it here
        for (const p of c.pops) {
          const own = p instanceof IntercityApi ? marketQs : mine, m = p.solved_mask();
          for (let q = 0; q < PERIODS.length; q++) if (m & (1 << q) && !own.includes(q)) p.forget_period(q);
        }
        if (c.part) sharedParts.delete(c.part);
        let reuse = job.fresh ? 0 : maskOf(units);
        for (const p of c.pops) {
          reuse &= p.solved_mask();
          const h = heldQ.get(p);
          for (const q of units) if (h?.[q] !== c.psig[q]) reuse &= ~(1 << q);
        }
        for (const p of c.pops) {
          const h = heldQ.get(p) ?? [];
          for (const q of units) if (!(reuse & (1 << q))) h[q] = "";
          heldQ.set(p, h);
        }
        c.reuse = reuse;
        ready.add(c);
        for (const q of units) if (reuse & (1 << q)) periodsKept++; else periodsSolved++;
      };
      const solved =(c: JobCluster) => {
        for (const p of c.pops) {
          held.set(p, c.sig);
          const h = heldQ.get(p)!;
          for (const q of units) h[q] = c.psig[q];
          heldOn.set(p, { sig: c.sig, psig: c.psig, shape: c.shape, net: c.net });
          if (p instanceof IntercityApi) { marketOn.set(p, viewOf(c, p).nodeGlobal); const net = lastNet as NetIds | null; if (net) marketNet.set(p, net); }
          else cityOn.set(p, viewOf(c, p));
        }
        if (sharing && c.part) sharedParts.add(c.part);
      };
      jobNodes = 2 * n.stops.length;
      versions.set(job.version, { clusters: jobClusters, stations: n.stXY.length / 2, railed: n.stXY.length > 0 && n.lineN.length > 0, nodes: jobNodes, rounds: job.rounds, markets: marketQs });
      for (const v of versions.keys()) if (versions.size > KEEP_VERSIONS && v !== job.version) versions.delete(v);
      const citiesOf = (c: JobCluster, f: Population[]) => c.cities.filter((id) => f.includes(cities.get(id)!) || (everyday.has(id) && f.includes(everyday.get(id)!)));
      const freshCities = (c: JobCluster) => citiesOf(c, localOf.get(c)!);
      sendMessage({ kind: "plan", version: job.version, solving: dirty.map((c) => (localOf.has(c) ? freshCities(c) : c.cities)), kept: jobClusters.filter((c) => !dirty.includes(c)).flatMap((c) => c.cities), ...(ahead ? { ahead: dirty[0].cities } : {}), local: editOf.size });
      current = job.version;
      roundsHeld.fill(-1);
      // T-1609: in a solve where every cluster solved is a local refine, a zone it does not solve
      // keeps the last solve's bubbles' sums (working out all of China's took 5-6 s)
      const keptSums = new Map<CityId, { mask: number; sums: Float32Array; everyday: Float32Array | null }>();
      if (localOf.size && localOf.size === dirty.length) {
        const fresh = new Set([...localOf.keys()].flatMap(freshCities));
        for (const [id] of jobLocal) { const k = sumsByRound.get(`${id}:${job.rounds}`); if (k && !fresh.has(id)) keptSums.set(id, k); }
      }
      sumsByRound.clear();
      stageRound = -1;
      answerWaiting(); // an older version's get nulls; this one's wait on
      let stale = false;
      // the clusters' loads, all their populations', per route node of the whole network
      // (`pick`: only those populations, T-1607)
      const sumAll = (cs: JobCluster[], read: (a: Population) => Float32Array, out = new Float32Array(jobNodes), at = 0, pick?: (a: Population) => boolean): Float32Array => {
        for (const c of cs) {
          for (const a of c.pops) { if (pick && !pick(a)) continue; const map = viewOf(c, a).nodeGlobal, v = read(a); for (let k = 0; k < map.length; k++) out[at + map[k]] += v[k] ?? 0; }
        }
        return out;
      };
      // T-1607: a shared period's sums as one worker adds them: the cities' (every population
      // before the market), then the market's, which is one population, so the bits are the same
      const bothSides = (side: Side, own: Float32Array, other: Float32Array): Float32Array => {
        const [t, m] = side === "cities" ? [own, other] : [other.slice(), own];
        for (let k = 0; k < t.length; k++) t[k] += m[k];
        return t;
      };
      // the cluster's loads on a population's own route nodes; at a pad, the riders aboard as
      // its boardings (`PopView.pads`)
      const ownOfIn = (c: JobCluster, a: Population, total: Float32Array) => {
        const N = jobNodes, { nodeGlobal: map, pads } = viewOf(c, a), nr = map.length, own = new Float32Array(3 * nr);
        for (let k = 0; k < nr; k++) { const g = map[k]; own[k] = total[g]; own[nr + k] = total[N + g]; own[2 * nr + k] = total[2 * N + g]; }
        for (const k of pads) { own[nr + k] = own[k]; own[2 * nr + k] = 0; }
        return own;
      };
      // One round of period q for the cluster. Periods are independent. Every population of a
      // cluster advances from one frozen load vector of the whole cluster, never from whichever
      // population happened to run first. False if a newer network came meanwhile.
      const step = async (c: JobCluster, round: number, q: number, skip?: Set<Population>): Promise<boolean> => {
        // the cluster's segment loads, boardings and alightings per route node: the crowding and
        // the waits for full trains (T-188) the round is solved with
        const N = jobNodes;
        // T-1607: the populations this worker solves of a shared period, and the other side's loads
        const side = sideOf(c, q) ?? "all", here = (a: Population) => onSide(side, a);
        const ownLoads = () => {
          const t = new Float32Array(3 * N);
          sumAll([c], a => a.seg(q), t, 0, here); sumAll([c], a => a.board(q), t, N, here); sumAll([c], a => a.alight(q), t, 2 * N, here);
          return t;
        };
        // the loads after round r: this worker's own once it has finished r
        const loads = async (r: number): Promise<Float32Array | null> => {
          if (side === "all") return ownLoads();
          const other = await partner(c, q, r);
          return other && bothSides(side, ownLoads(), other);
        };
        const ownOf = (a: Population, total: Float32Array) => ownOfIn(c, a, total);
        const total = round > 0 ? await loads(round - 1) : null;
        if (round > 0 && !total) return false;
        if (total) for (const a of c.pops) if (here(a)) keepTrail(a, q, round - 1);
        for (const a of c.solveOrder) {
          if (import.meta.env.DEV && panicInSolve && a instanceof DemandApi) { panicInSolve = false; a.debug_panic(); }
          if (!here(a) || (total === null && skip?.has(a))) continue;
          if (total === null) a.solve(q);
          else if (!a.crowd_with_external(q, ownOf(a, total))) throw new Error("shared crowding layout disagrees");
          // a population's pass is up to a second of one call: let a waiting query or a newer
          // network in between
          await yieldNow();
          if (pending) return false;
        }
        // T-1255: the free-flow pass's boardings, the riders who would come if the trains had room
        if (total === null) for (const a of c.pops) { if (!here(a)) continue; const b = boardsOf(a); b.free[q] = a.board(q); b.rounds[q] = undefined; }
        if (side !== "all") {
          const t = ownLoads();
          sendMessage({ kind: "loads", version: job.version, cluster: jobClusters.indexOf(c), period: q, round, side, loads: t }, [t.buffer]);
        }
        if (round === job.rounds && round > 0) {
          // T-1070, T-1255: the final loads, every population's, held to the riders the trains take
          // within the wait for a train with room, with everyone who comes to the platforms
          const final = await loads(round);
          if (!final) return false;
          for (const a of c.pops) if (here(a)) { boardsOf(a).rounds[q] = a.board(q); keepTrail(a, q, round); }
          let come: Float32Array;
          if (side === "all") come = comeOf([c], q);
          else {
            // (the other side's boardings of the free flow and of this round, as its loads sent them)
            const [free, rounds] = comeRaw([c], q, here), f0 = (await partner(c, q, 0))!, fr = (await partner(c, q, round))!;
            come = comeMax(bothSides(side, free, f0.subarray(N, 2 * N)), bothSides(side, rounds, fr.subarray(N, 2 * N)));
          }
          for (const a of c.solveOrder) {
            if (!here(a)) continue;
            const own = ownOf(a, final), { nodeGlobal: map, pads } = viewOf(c, a), nr = map.length, four = new Float32Array(4 * nr);
            four.set(own);
            for (let k = 0; k < nr; k++) four[3 * nr + k] = Math.max(come[map[k]], own[nr + k]);
            for (const k of pads) four[3 * nr + k] = own[nr + k];
            if (!a.cap_with_external(q, four)) throw new Error("shared capacity layout disagrees");
          }
          if (side === "all") keepCapRatio(c.sig, q, final, ownLoads());
        }
        return true;
      };
      // T-1538: period `to`'s first figures estimated from `from`'s free flow, a few ms a population
      // (`DemandApi::estimate`); the market, which has none, is solved, and goes in `done` so the
      // step that solves `to` later skips it. False if a population could not estimate.
      const estimate = (c: JobCluster, from: number, to: number, done: Set<Population>): boolean => {
        for (const a of c.solveOrder) {
          if (a instanceof IntercityApi) { a.solve(to); done.add(a); }
          else if (!a.estimate(from, to)) return false;
        }
        for (const a of c.pops) { const b = boardsOf(a); b.free[to] = a.board(to); b.rounds[to] = undefined; }
        return true;
      };
      let estimated = false;
      // T-1255: per route node of the whole network, the riders coming to the platforms of clusters
      // `cs` in period q: the free-flow pass's boardings, at least the rounds' (before the cap)
      const comeRaw = (cs: JobCluster[], q: number, pick?: (a: Population) => boolean): [Float32Array, Float32Array] =>
        [sumAll(cs, (a) => boardsOf(a).free[q] ?? a.board(q), undefined, 0, pick), sumAll(cs, (a) => boardsOf(a).rounds[q] ?? a.board(q), undefined, 0, pick)];
      const comeOf = (cs: JobCluster[], q: number, pick?: (a: Population) => boolean): Float32Array => comeMax(...comeRaw(cs, q, pick));
      // T-1530: the transfers in at each route node of the whole network, every kind of day: a
      // population's weekday figures times the factor its `transfers_day` scales its pairs by
      const xferDays = (cs: JobCluster[], q: number, pick?: (a: Population) => boolean): Float32Array => {
        const N = jobNodes, D = DEMAND_DAYS.length, out = new Float32Array(D * N);
        const table = dayTable ??= day_factors(), K = 2 + EVERYDAY_KEYS.length, P = PERIODS.length;
        const f = (day: number, p: number) => (day === 0 ? 1 : table[(day * K + p) * P + q] ?? 1);
        for (const c of cs)
          for (const a of c.pops) {
            if (pick && !pick(a)) continue;
            const v = a.transfers_in(q);
            if (!v.length) continue;
            const map = viewOf(c, a).nodeGlobal;
            const kinds = a instanceof EverydayApi ? a.kind_totals(q) : null;
            for (let day = 0; day < D; day++) {
              let k = 1;
              if (a instanceof DemandApi) k = f(day, 0);
              else if (a instanceof IntercityApi) k = f(day, K - 1);
              else if (kinds && day !== 0) {
                // the purposes' transfers are kept together: their rail riders' mean factor
                let rail = 0, weighted = 0;
                for (let i = 0; 3 * i + 1 < kinds.length; i++) { rail += kinds[3 * i + 1]; weighted += kinds[3 * i + 1] * f(day, 1 + i); }
                k = rail > 0 ? weighted / rail : 0;
              }
              for (let j = 0, o = day * N; j < map.length; j++) out[o + map[j]] += v[j] * k;
            }
          }
        return out;
      };
      // period q's figures over the clusters `cs` (all of them for a result), as the state stands.
      // `part` (T-1543): this worker's part of the period's origins, its market left out of a period
      // it does not hold (`noMarket`)
      // (T-1607 `part.side`: one side of a shared period, its populations alone, with the boardings
      // `arrive` is worked out from)
      // (T-1733 `extra`: loads to add, three blocks as `step`'s totals: a shared period's market loads,
      // which the other worker holds, all counted as longer trips)
      // (T-1735 `delta`: loads to add by purpose, `marketBlocks`' blocks: a window's pairs' new loads
      // less their old)
      const figures =(q: number, round: number, ms: number, cs: JobCluster[], part?: { at: [number, number]; noMarket: boolean; side?: Side }, extra?: Float32Array, delta?: Float32Array[]) => {
        const side = part?.side, pick = side ? (a: Population) => onSide(side, a) : undefined;
        // in the job's order: the summary adds the cities up in it
        const ids = side === "market" ? [] : jobLocal.map(([id]) => id).filter((id) => cs.some((c) => c.cities.includes(id)));
        // the market's figures are all its parts' as they stand, whichever clusters are asked for
        const hasMarket = !!regional && cs.some((c) => c.market);
        const market = hasMarket && !part?.noMarket ? marketFigures(jobClusters.flatMap((c) => c.part ? [c.part] : []), q) : null;
        const seg = sumAll(cs, a => a.seg(q), undefined, 0, pick), board = sumAll(cs, a => a.board(q), undefined, 0, pick), alight = sumAll(cs, a => a.alight(q), undefined, 0, pick), loadOfCrush = sumAll(cs, a => a.load_of_crush(q), undefined, 0, pick), fareX = sumAll(cs, a => a.fare_extra(q), undefined, 0, pick), xferIn = xferDays(cs, q, pick);
        if (extra) for (let k = 0; k < jobNodes; k++) { seg[k] += extra[k]; board[k] += extra[jobNodes + k]; alight[k] += extra[2 * jobNodes + k]; }
        // (less a window's old loads, which can pass a node's own)
        if (delta?.[0]) { const d = delta[0]; for (let k = 0; k < jobNodes; k++) { seg[k] = Math.max(0, seg[k] + d[k]); board[k] = Math.max(0, board[k] + d[jobNodes + k]); alight[k] = Math.max(0, alight[k] + d[2 * jobNodes + k]); } }
        const [free, rounds] = comeRaw(cs, q, pick);
        const come: [Float32Array, Float32Array] | undefined = side ? [free.slice(), rounds.slice()] : undefined;
        const arrive = arriveOf(comeMax(free, rounds), rounds, board);
        const citySummary = Object.fromEntries(ids.map((id) => [id, Array.from(cities.get(id)!.summary(q))])) as Record<CityId, number[]>;
        for (const [id, a] of jobEveryday) if (citySummary[id]) { const v = a.summary(q); for (const k of [1, 2, 3, 5, 10, 11, 12]) citySummary[id][k] += v[k] ?? 0; }
        const intercitySummary = market ? market.summary : [];
        const summary = [round, 0, 0, 0, ms, 0, 0, 0, 0, 0];
        for (const v of [...Object.values(citySummary), intercitySummary]) for (const k of [1, 2, 3, 5]) summary[k] += v[k] ?? 0;
        // T-1300: rail trips per curve dollar dearer on every fare, local and the market's (each
        // population's per perceived minute, summary slot 10, times its minutes a curve dollar), and
        // how far the trips' sensitivity reaches as against their riders: the sensitivity-weighted
        // mean km over the rider-weighted one (slots 11 and 12)
        const perDollar = (id: CityId) => { const c = cityById(id).country; return fareWeight(c) * fareScale(c, n.fare.scales); };
        const reach = (d: number, dKm: number, rail: number, km: number) => (d < 0 && km > 0 && rail > 0 ? (dKm / d) / (km / rail) : 1);
        const ls = ids.reduce((a, id) => {
          const v = citySummary[id], w = perDollar(id);
          return [a[0] + (v[10] ?? 0) * w, a[1] + (v[11] ?? 0) * w, a[2] + (v[2] ?? 0), a[3] + (v[12] ?? 0)];
        }, [0, 0, 0, 0]);
        const im = intercitySummary;
        const fareSense: [number, number, number, number] = [ls[0], (im[10] ?? 0) * fareWeight(FARE_BASELINE), reach(ls[0], ls[1], ls[2], ls[3]), reach(im[10] ?? 0, im[11] ?? 0, im[2] ?? 0, im[12] ?? 0)];
        const parts = purposeParts(q, jobNodes, cs, hasMarket, pick);
        if (extra && hasMarket) {
          const at = (parts.keys.length - 1) * jobNodes;
          for (let k = 0; k < jobNodes; k++) { parts.seg[at + k] += extra[k]; parts.board[at + k] += extra[jobNodes + k]; parts.alight[at + k] += extra[2 * jobNodes + k]; }
        }
        // (the commutes to work, each everyday purpose to its own, the rest to the longer trips, as `purposeParts`)
        if (delta?.[0] && hasMarket) {
          const N = jobNodes, [all, com, ...ev] = delta, last = parts.keys.length - 1, kinds = Math.min(ev.length, last - 1);
          const into = [parts.seg, parts.board, parts.alight];
          for (let b = 0; b < 3; b++) for (let k = 0; k < N; k++) {
            let rest = all[b * N + k];
            if (com) { into[b][k] += com[b * N + k]; rest -= com[b * N + k]; }
            for (let i = 0; i < kinds; i++) { into[b][(1 + i) * N + k] += ev[i][b * N + k]; rest -= ev[i][b * N + k]; }
            into[b][last * N + k] += rest;
          }
        }
        // T-260: each city's trips, rail and walked trips by purpose, `parts.keys` order less the
        // market, so the main thread can work out a weekend's mode split
        const local = parts.keys.length - (hasMarket ? 1 : 0);
        const cityParts = Object.fromEntries(ids.map((id) => {
          const v = new Array(3 * local).fill(0), s = cities.get(id)!.summary(q);
          for (let k = 0; k < 3; k++) v[k] = s[1 + k] ?? 0;
          const e = everyday.get(id)?.kind_totals(q);
          if (e) for (let k = 0; k < Math.min(e.length, 3 * (local - 1)); k++) v[3 + k] = e[k];
          return [id, v];
        })) as Record<CityId, number[]>;
        const msg = { version: job.version, period: q, summary, fareSense, citySummary, cityParts, intercitySummary, intercityParts: market ? market.parts : [], corridorKey: market?.corridorKey, corridorTable: market?.corridorTable, corridorShares: market?.corridorShares, corridorRail: market?.corridorRail, elsewhere: market?.elsewhere, seg, board, alight, arrive, loadOfCrush, fareX, xferIn, parts, ms, setupMs, memBytes: wasm_memory_bytes(), ...(part ? { part: part.at, fareRaw: ls as [number, number, number, number] } : {}), ...(come ? { come } : {}) };
        const t = market?.corridorTable;
        const big = [...(come ? come.map((a) => a.buffer) : []), market?.corridorRail?.buffer, ...(t ? [t.from, t.to, t.longer, t.commutes, t.air, t.car].map((a) => a.buffer) : [])].filter((b): b is ArrayBuffer => !!b);
        return { msg, transfer: [seg.buffer, board.buffer, alight.buffer, arrive.buffer, loadOfCrush.buffer, fareX.buffer, xferIn.buffer, parts.seg.buffer, parts.board.buffer, parts.alight.buffer, ...(parts.lean ? [parts.lean.buffer] : []), ...big] };
      };
      // T-1543: a free-flow round shown before it is solved (a rough solve's, the zone in view's
      // preview after an edit) is one period solved in parts of its origins, a part in each solving
      // worker, every period estimated from it and the parts added up on the page. Each worker
      // then solves its own periods as before, so nothing after the shown round changes.
      const share = !job.exact0 && job.part && job.part[1] > 1 && mine.length ? job.part : null;
      const SPLIT_FROM = 0;
      // The parts' round for clusters `cs`, sent as results (with `cluster` when given). Null when
      // not possible here (a period kept, a population that cannot estimate), false if a newer
      // network came. The market is solved whole in this worker's own periods only (`done`: the
      // periods whose market is solved, for the round's own step to skip). `sums`: the cities whose
      // bubbles' sums go with it.
      const splitRound = async (cs: JobCluster[], cluster: JobCluster | null, sums: [CityId, DemandApi][], done?: Map<number, Set<Population>>, figCs = cs): Promise<boolean | null> => {
        if (!share || cs.some((c) => c.reuse)) return null;
        const t0 = performance.now();
        const own = (a: Population): a is DemandApi | EverydayApi => !(a instanceof IntercityApi);
        const all = PERIODS.map((_, q) => q);
        // the populations' own periods are solved whole next; the others go
        const drop = () => {
          for (const c of cs) for (const a of c.pops) {
            if (!own(a)) continue;
            a.set_part(0, 1);
            for (const q of all) if (!mine.includes(q)) { a.forget_period(q); boardsOf(a).free[q] = undefined; }
          }
        };
        let ok = true;
        try {
          for (const c of cs) for (const a of c.solveOrder) {
            if (!own(a)) continue;
            a.set_part(share[0], share[1]);
            a.solve(SPLIT_FROM);
            await yieldNow();
            if (pending) return false;
          }
          for (const c of cs) for (const a of c.solveOrder) for (let q = 0; q < PERIODS.length && ok; q++) if (own(a) && q !== SPLIT_FROM) ok = a.estimate(SPLIT_FROM, q);
          if (!ok) return null;
          for (const c of cs) {
            if (!c.part) continue;
            for (const q of mine) {
              c.part.solve(q);
              if (done) { const d = done.get(q) ?? new Set<Population>(); d.add(c.part); done.set(q, d); }
              await yieldNow();
              if (pending) return false;
            }
          }
        } finally {
          // (a part of the origins is never left in a population; on success the figures are read first)
          if (!ok || pending) drop();
        }
        for (const c of cs) for (const a of c.pops) { const b = boardsOf(a); for (const q of all) { b.free[q] = a.board(q); b.rounds[q] = undefined; } }
        for (const [id, a] of sums) keepSums(job.version, [0], id, a);
        const ms = performance.now() - t0;
        for (const q of all) {
          const { msg, transfer } = figures(q, 0, ms, figCs, { at: share, noMarket: !mine.includes(q) });
          sendMessage({ kind: "result", ...msg, ...(cluster ? { cluster: { cities: cluster.cities, market: cluster.market, lines: cluster.lines.slice() } } : onlyLines()) }, transfer);
        }
        drop();
        return true;
      };
      // T-1609: the populations `fresh` of cluster c solved alone, each of this worker's periods
      // through every round, crowding on the other populations' loads (T-1733: round by round as
      // their last solve was priced, `trail`) plus their own, then held to the trains' room
      // as `step` holds them. The others keep their figures: the result is theirs with the fresh
      // populations' in place of the ones they replace. False if a newer network came.
      const sharedOf = (c: JobCluster, q: number, shared?: (Float32Array | undefined)[]) => (c.part && !(c.part.solved_mask() & (1 << q)) ? shared?.[q] : undefined);
      // T-1735: population a's period-q loads added to `out` (three blocks per route node of the whole network)
      const addOwn = (c: JobCluster, a: Population, q: number, out: Float32Array) => {
        const N = jobNodes, map = viewOf(c, a).nodeGlobal, s = a.seg(q), b = a.board(q), l = a.alight(q);
        for (let k = 0; k < map.length; k++) { const g = map[k]; out[g] += s[k]; out[N + g] += b[k]; out[2 * N + g] += l[k]; }
      };
      // T-1735: blocks of three per route node `map` gives (NONE: dropped) onto the whole network's
      const toWhole = (v: ArrayLike<number>, map: Uint32Array): Float32Array[] => {
        const N = jobNodes, m = map.length, out: Float32Array[] = [];
        // (a view with no route nodes, as after its cluster's last line goes, carries nothing; its
        // zero stride would loop until the worker's heap ran out, T-2071)
        if (m === 0) return out;
        for (let at = 0; at + 3 * m <= v.length; at += 3 * m) {
          const t = new Float32Array(3 * N);
          for (let b = 0; b < 3; b++) for (let k = 0; k < m; k++) if (map[k] !== NONE) t[b * N + map[k]] += v[at + b * m + k];
          out.push(t);
        }
        return out;
      };
      // T-1735: a market population's loads in `window_loads`' blocks: every pair's, the commutes', each everyday purpose's
      const marketBlocks = (c: JobCluster, a: IntercityApi, q: number): Float32Array[] => {
        const map = viewOf(c, a).nodeGlobal, all = new Float32Array(3 * jobNodes);
        addOwn(c, a, q, all);
        return [all, toWhole(a.commute_loads(q), map)[0] ?? new Float32Array(3 * jobNodes), ...toWhole(a.everyday_loads(q), map)];
      };
      // (T-1733: `ratioSig` the cluster the frozen populations last solved; `shared` a shared period's
      // market loads, for a period the market part here does not hold)
      // (T-1735 `win`: the market's pairs with an end in a window solved too, `sub` set on the
      // network as the cluster's market and solved, in the periods `less` gives, in one pass at the
      // crowding of the touched cities' last round; the result's market less `less(q)`, the window's
      // pairs' old loads, plus `sub`'s)
      const localSolve = async (c: JobCluster, fresh: Population[], ratioSig = c.sig, shared?: (Float32Array | undefined)[], win?: { sub: IntercityApi; less: (q: number) => Float32Array | null }): Promise<boolean> => {
        const N = jobNodes, isFresh = (a: Population) => fresh.includes(a), order = c.solveOrder.filter(isFresh);
        setUpPops(c, win ? [...order, win.sub] : order);
        for (const p of win ? [...fresh, win.sub] : fresh) {
          const m = p.solved_mask();
          for (let q = 0; q < PERIODS.length; q++) if (m & (1 << q)) p.forget_period(q);
          heldQ.set(p, []);
        }
        c.reuse = 0;
        ready.add(c);
        for (const q of mine) {
          const sh = sharedOf(c, q, shared), shTrail = sh && c.part ? sharedTrail.get(c.part)?.[q] : undefined;
          // the others' loads as each round was priced on (`trail`), when every one holds them;
          // else as they ended, before the cap (`capRatio`)
          const others = c.pops.filter((a) => !isFresh(a) && !(sh && a === c.part));
          const trails = others.map((a) => trail.get(a)?.[q]);
          const onTrail = trails.every((t, i) => {
            const nr = viewOf(c, others[i]).nodeGlobal.length;
            for (let r = 0; r <= job.rounds; r++) if (t?.[r]?.length !== 3 * nr) return false;
            return true;
          });
          let ended: Float32Array | null = null;
          if (!onTrail) {
            const frozen = new Float32Array(3 * N), notFresh = (a: Population) => !isFresh(a);
            sumAll([c], (a) => a.seg(q), frozen, 0, notFresh); sumAll([c], (a) => a.board(q), frozen, N, notFresh); sumAll([c], (a) => a.alight(q), frozen, 2 * N, notFresh);
            const r = capRatio.get(ratioSig)?.[q];
            if (r) for (let i = 0; i < r.idx.length; i++) frozen[r.idx[i]] *= r.v[i];
            ended = frozen;
          }
          const frozenAt = (round: number) => {
            const out = ended ? new Float32Array(ended) : new Float32Array(3 * N);
            if (!ended) others.forEach((a, i) => {
              const v = trails[i]![round]!, map = viewOf(c, a).nodeGlobal, nr = map.length;
              for (let b = 0; b < 3; b++) for (let k = 0; k < nr; k++) out[b * N + map[k]] += v[b * nr + k];
            });
            const m = sh && (shTrail?.[round] ?? sh);
            if (m) for (let k = 0; k < out.length; k++) out[k] += m[k];
            return out;
          };
          const total = (round: number) => {
            const t = frozenAt(round);
            sumAll([c], (a) => a.seg(q), t, 0, isFresh); sumAll([c], (a) => a.board(q), t, N, isFresh); sumAll([c], (a) => a.alight(q), t, 2 * N, isFresh);
            return t;
          };
          for (let round = 0; round <= job.rounds; round++) {
            const t = round > 0 ? total(round - 1) : null;
            if (t) for (const a of fresh) keepTrail(a, q, round - 1);
            for (const a of order) {
              if (t === null) a.solve(q);
              else if (!a.crowd_with_external(q, ownOfIn(c, a, t))) throw new Error("local crowding layout disagrees");
              await yieldNow();
              if (pending) return false;
            }
            if (t === null) for (const a of fresh) { const b = boardsOf(a); b.free[q] = a.board(q); b.rounds[q] = undefined; }
          }
          let final = job.rounds > 0 ? total(job.rounds) : null;
          // T-1735: the window's pairs at the crowding the cities ended on (their old loads standing
          // in for their own), then swapped for their old loads in the totals the cap takes
          const less = win?.less(q) ?? null, sub = less ? win!.sub : null;
          if (sub) {
            const at = final ?? total(0);
            if (!sub.solve_crowded(q, ownOfIn(c, sub, at))) throw new Error("window crowding layout disagrees");
            await yieldNow();
            if (pending) return false;
            const b = boardsOf(sub);
            b.free[q] = b.rounds[q] = sub.board(q);
            if (final) {
              for (let k = 0; k < final.length; k++) final[k] -= less![k];
              addOwn(c, sub, q, final);
              // (the window's old loads, from one pass at the last crowding, can pass a node's frozen ones)
              for (let k = 0; k < final.length; k++) if (final[k] < 0) final[k] = 0;
            }
          }
          if (final) {
            for (const a of fresh) { boardsOf(a).rounds[q] = a.board(q); keepTrail(a, q, job.rounds); }
            const come = comeOf([c], q);
            for (const a of sub ? [...order, sub] : order) {
              const own = ownOfIn(c, a, final), { nodeGlobal: map, pads } = viewOf(c, a), nr = map.length, four = new Float32Array(4 * nr);
              four.set(own);
              for (let k = 0; k < nr; k++) four[3 * nr + k] = Math.max(come[map[k]], own[nr + k]);
              for (const k of pads) four[3 * nr + k] = own[nr + k];
              if (!a.cap_with_external(q, four)) throw new Error("local capacity layout disagrees");
            }
          }
          release_skims();
          periodsSolved++;
        }
        return true;
      };
      // T-1733: every edited cluster's touched cities first, the zone in view's cluster first
      // (`early`'s order), each sent as a local cluster result; the market part read through its
      // route nodes on this network (`marketOn`) meanwhile
      for (const c of early) {
        const e = editOf.get(c);
        if (!e) continue;
        const t0 = performance.now(), whole = c.part ? viewOf(c, c.part) : null, shared = c.part ? sharedMkt.get(c.part) : undefined;
        if (c.part && e.market) c.views.set(c.part, { ...whole!, nodeGlobal: e.market, pads: new Uint32Array(0) });
        // T-1735: the market's pairs with an end in a zone a station changed since the market's last
        // whole solve serves, solved with the touched cities on the window's stations alone; the
        // frozen market less those pairs' old loads (`less`)
        const part = c.part, base = part && e.market && lastNet && !job.noWindow ? marketNet.get(part) : undefined;
        const ed = base ? editedStations(base, lastNet!, nodeMap(base, lastNet!)) : null;
        const sub = ed && ed.minutes >= WINDOW_EDIT_MIN ? part!.beside() : null;
        const lessOf = new Map<number, Float32Array[]>();
        if (sub) {
          c.views.set(sub, whole!);
          sub.set_window(ed!.now, part!.zones_at(ed!.was), ed!.minutes, WINDOW_SHARE, WINDOW_FLOOR);
        }
        const win = sub ? {
          sub,
          less: (q: number) => {
            if (!(part!.solved_mask() & (1 << q))) return null;
            const zones = sub.window_zones();
            if (!zones.length) return null;
            const blocks = toWhole(part!.window_loads(q, zones), e.market!);
            if (!blocks.length) return null;
            lessOf.set(q, blocks);
            return blocks[0];
          },
        } : undefined;
        try {
          if (!(await localSolve(c, e.fresh, e.sig, shared, win))) { stale = true; break; }
          for (const p of e.fresh) cityOn.set(p, viewOf(c, p));
          const touched = citiesOf(c, e.fresh);
          for (const q of mine) {
            const less = lessOf.get(q);
            const delta = sub && less ? marketBlocks(c, sub, q).map((v, i) => { const t = v.slice(), l = less[i]; if (l) for (let k = 0; k < t.length; k++) t[k] -= l[k]; return t; }) : undefined;
            if (import.meta.env.DEV && delta) { let a = 0, b = 0; for (let k = jobNodes; k < 2 * jobNodes; k++) { a += less![0][k]; b += delta[0][k] + less![0][k]; } console.info(`T1735 v${job.version} period ${q}: the window's pairs board ${Math.round(b)} on the new network, ${Math.round(a)} on the old`); }
            const { msg, transfer } = figures(q, job.rounds, 0, [c], undefined, sharedOf(c, q, shared), delta);
            sendMessage({ kind: "result", ...msg, cluster: { cities: touched, market: false, lines: c.lines.slice(), local: true } }, transfer);
          }
          if (import.meta.env.DEV) console.info(`T1733 v${job.version} local solve of ${touched.length} zones (${touched.slice(0, 8).join(",")}) of ${c.cities.length} after an edit${sub ? `, the market's pairs of ${sub.window_zones().length} of its zones` : ""}: ${Math.round(performance.now() - t0)} ms`);
        } finally {
          if (whole) c.views.set(c.part!, whole);
          if (sub) { c.views.delete(sub); held.delete(sub); heldQ.delete(sub); disturbed(sub); sub.free(); }
        }
      }
      // T-1733: the whole solve sets every population on the new network, after which no later edit
      // can be solved locally until it ends. So it waits for the page's `whole`: every worker's local
      // result is in, and a moment has passed for the next edit (stations built, then the line)
      if (editOf.size && !stale) {
        const until = performance.now() + WHOLE_WAIT_MS;
        while (!pending && wholeGo < job.version && performance.now() < until) await new Promise<void>((r) => { wake = r; setTimeout(r, until - performance.now()); });
        wake = null;
        if (pending) stale = true;
      }
      for (const c of early) {
        if (stale) break;
        const fresh = localOf.get(c);
        if (fresh) {
          const t0 = performance.now(), shared = c.part ? sharedMkt.get(c.part) : undefined;
          if (fresh.length && !(await localSolve(c, fresh, c.sig, shared))) { stale = true; break; }
          for (const p of c.pops) {
            held.set(p, c.sig);
            if (!fresh.includes(p)) continue;
            cityOn.set(p, viewOf(c, p));
            const h = heldQ.get(p)!;
            for (const q of mine) h[q] = c.psig[q] + LOCAL;
          }
          if (import.meta.env.DEV) console.info(`T1609 local refine of ${freshCities(c).join(",")} in a cluster of ${c.cities.length} zones: ${Math.round(performance.now() - t0)} ms`);
          if (c === early[early.length - 1]) break;
          // (a period whose market is the other worker's: its loads added, the market's own figures
          // left as shown)
          for (const q of mine) {
            const extra = sharedOf(c, q, shared);
            const { msg, transfer } = figures(q, job.rounds, 0, [c], undefined, extra);
            sendMessage({ kind: "result", ...msg, cluster: { cities: c.cities, market: c.market && !(c.part && sharedParts.has(c.part)), lines: c.lines.slice() } }, transfer);
          }
          continue;
        }
        setUp(c);
        // T-1533: the zone in view's cluster's free-flow figures at once, for the lines new since
        // the shown result (T-324), rather than at the cluster's last round
        // (T-1733: none after its local result)
        const preview = job.rounds > 0 && c === early[0] && !!job.first && c.cities.includes(job.first) && !editOf.has(c);
        const sendPreview = () => {
          for (const q of mine) {
            const { msg, transfer } = figures(q, 0, 0, [c]);
            sendMessage({ kind: "result", ...msg, cluster: { cities: c.cities, market: c.market, lines: c.lines.slice() } }, transfer);
          }
        };
        for (let round = 0; round <= job.rounds && !stale; round++) {
          let previewed = false, parted = false;
          const done = new Map<number, Set<Population>>();
          if (preview && round === 0) {
            const r = await splitRound([c], c, [], done);
            if (r === false) { stale = true; break; }
            previewed = parted = !!r;
          }
          const todo = mine.filter((q) => !(c.reuse & (1 << q)));
          for (const q of todo) {
            await yieldNow();
            if (pending || !(await step(c, round, q, done.get(q)))) { stale = true; break; }
            release_skims();
            // T-1538: the other periods estimated from the first one's free flow, so the preview
            // goes now; they are solved after it, replacing the estimates
            if (preview && round === 0 && !previewed && !job.exact0 && todo.length > 1) {
              const rest = todo.filter((r) => r !== q);
              if (rest.every((r) => { const d = new Set<Population>(); done.set(r, d); return estimate(c, q, r, d); })) { sendPreview(); previewed = true; }
            }
          }
          // (after the parts, this worker's own periods too: the page takes them for any period whose
          // parts did not all come, as when another worker kept a period and sent no part)
          if (!stale && round === 0 && preview && (!previewed || parted)) sendPreview();
        }
        if (stale) break;
        solved(c);
        // the last one's figures go with the solve's results, below
        if (c === early[early.length - 1]) break;
        for (const q of mine) {
          const { msg, transfer } = figures(q, job.rounds, 0, [c]);
          sendMessage({ kind: "result", ...msg, cluster: { cities: c.cities, market: c.market, lines: c.lines.slice() } }, transfer);
        }
      }
      if (!stale && !job.barrier) {
        // Every cluster is solved to its last round, so the solve's result for every round is its
        // final figures, worked out once: after an edit only the final is shown, and a line new
        // since the shown result takes these at round 0 (T-324).
        stageRound = job.rounds;
        for (const q of mine) roundsHeld[q] = job.rounds;
        if (mine.length) {
          const rounds = Array.from({ length: job.rounds + 1 }, (_, r) => r);
          for (const [id, a] of jobLocal) keepSums(job.version, rounds, id, a, keptSums.get(id));
        }
        answerWaiting();
        for (const q of [...mine, ...mq]) {
          const t0 = performance.now();
          // (T-1733: a period kept split, as the barrier's solve sent it, T-1607)
          const split = cq.includes(q) || mq.includes(q) ? { at: (mine.includes(q) ? [0, 2] : [1, 2]) as [number, number], noMarket: mine.includes(q), side: (mine.includes(q) ? "cities" : "market") as Side } : undefined;
          const { msg } = figures(q, job.rounds, 0, jobClusters, split);
          msg.ms = msg.summary[4] = performance.now() - t0;
          for (let r = 0; r <= job.rounds; r++) {
            const lean = msg.parts.lean?.slice();
            const m = { ...msg, summary: [r, ...msg.summary.slice(1)], seg: msg.seg.slice(), board: msg.board.slice(), alight: msg.alight.slice(), arrive: msg.arrive.slice(), loadOfCrush: msg.loadOfCrush.slice(), parts: { keys: msg.parts.keys, seg: msg.parts.seg.slice(), board: msg.parts.board.slice(), alight: msg.parts.alight.slice(), ...(lean ? { lean } : {}) }, setupMs: r === 0 ? setupMs : 0 };
            sendMessage({ kind: "result", ...m }, [m.seg.buffer, m.board.buffer, m.alight.buffer, m.arrive.buffer, m.loadOfCrush.buffer, m.parts.seg.buffer, m.parts.board.buffer, m.parts.alight.buffer, ...(lean ? [lean.buffer] : [])]);
          }
        }
        setupMs = 0;
      }
      // with the barrier only: without it the results were sent just above
      for (let g = 0; g < groups.length && !stale; g++) {
        const late = groups[g];
        // a group solved ahead sends its cluster's figures alone; the last group's results are the
        // solve's, with the groups before it held at their last round
        const alone = g < groups.length - 1 ? late[0] : null;
        const sumsFor = jobLocal.filter(([id]) => alone ? alone.cities.includes(id) : !groups.slice(0, g).some((cs) => cs.some((c) => c.cities.includes(id))));
        stageGroup = g;
        stageRound = -1;
        roundsHeld.fill(-1);
        for (const c of late) setUp(c);
        for (let round = 0; round <= job.rounds && !stale; round++) {
          // T-327: a worker ahead holds its last round until every worker has finished it
          while (round > 0 && units.length && !pending && !goAllows(job.version, g, round)) {
            await new Promise<void>((r) => (wake = r));
            wake = null;
          }
          if (pending) { stale = true; break; }
          // T-1543: a rough solve's free flow in parts first; the opens waiting go on once it is out,
          // and this worker's own round 0 follows (its results only stand in for missing parts)
          let parted = false;
          if (round === 0 && only) {
            const r = await splitRound(late, alone, sumsFor, undefined, alone ? [alone] : jobClusters);
            if (r === false) { stale = true; break; }
            parted = !!r;
            if (parted) setRough(false);
          }
          stageRound = round;
          for (const q of units) {
            await yieldNow();
            if (pending) { stale = true; break; }
            const t0 = performance.now();
            // The period is part way between rounds until every population has had it; a query in
            // between (the yields below) must not read it.
            roundsHeld[q] = -2;
            // T-1538: a rough solve's free flow is the worker's first period alone, its others
            // estimated from it
            const from = mine.find((r) => !(late.some((c) => c.reuse & (1 << r))));
            const est = round === 0 && !!only && !job.exact0 && from !== undefined && from !== q;
            for (const c of late) {
              if (c.reuse & (1 << q) || !sideOf(c, q)) continue;
              if (est && estimate(c, from, q, new Set())) { estimated = true; continue; }
              if (!(await step(c, round, q))) { stale = true; break; }
            }
            release_skims();
            if (stale) break;
            roundsHeld[q] = round;
            // (after the parts, round 0's sums are theirs: every worker's must be of the same kind)
            if (q === mine[mine.length - 1] && !parted) {
              // sent unasked, ahead of this round's last result: the client holds every worker's by
              // the time it publishes the round, and the bubbles need no query to a worker that is
              // busy with the next round
              for (const [id, a] of sumsFor) keepSums(job.version, [round], id, a);
            }
            const { msg, transfer } = figures(q, round, performance.now() - t0, alone ? [alone] : jobClusters, sharing && shared(q) ? { at: mine.includes(q) ? [0, 2] : [1, 2], noMarket: mine.includes(q), side: mine.includes(q) ? "cities" : "market" } : undefined);
            // T-327: ahead of the result, so the client holds every worker's answer for a round by
            // the time it publishes that round
            if (q === units[units.length - 1]) answerWaiting();
            if (alone) sendMessage({ kind: "result", ...msg, cluster: { cities: alone.cities, market: alone.market, lines: alone.lines.slice() } }, transfer);
            else {
              sendMessage({ kind: "result", ...msg, ...onlyLines() }, transfer);
              setupMs = 0;
            }
          }
          // T-1533: a rough solve's estimate is out: the opens waiting go on between its populations
          if (only && round === 0) setRough(false);
        }
        // (a rough solve from estimates is never kept for a whole solve to reuse; T-2045: in the focus
        // pool, which has no whole solve, it is, so a zone's questions are answered after the next)
        if (!stale && (!estimated || job.keep)) for (const c of late) solved(c);
      }
      if (!stale) {
        solvedOnce = true;
        // T-1733: a shared period's market loads from the other side, for a local solve after an edit
        if (sharing) jobClusters.forEach((c, i) => {
          if (!c.part || !dirty.includes(c)) return;
          for (const q of mine) {
            const t = shared(q) ? inbox.get(`${job.version}:${i}:${q}:${job.rounds}`) : undefined;
            if (!t) continue;
            const list = sharedMkt.get(c.part) ?? [];
            list[q] = t.slice();
            sharedMkt.set(c.part, list);
            const rounds: Float32Array[] = [];
            for (let r = 0; r <= job.rounds; r++) { const v = inbox.get(`${job.version}:${i}:${q}:${r}`); if (v) rounds.push(v.slice()); }
            const trails = sharedTrail.get(c.part) ?? [];
            trails[q] = rounds.length === job.rounds + 1 ? rounds : undefined;
            sharedTrail.set(c.part, trails);
          }
        });
        sendMessage({ kind: "done", version: job.version, memBytes: wasm_memory_bytes(), heap: heap(), clusters: [jobClusters.length - dirty.length, dirty.length], periods: [periodsKept, periodsSolved], ...(localOf.size ? { local: [...localOf.keys()].flatMap(freshCities) } : {}) });
      }
    }
  } catch (err) {
    failed(err);
  }
  busy = false;
  setRough(false);
  if (!dead) closeDeferred();
}
let freshCarry = false;
