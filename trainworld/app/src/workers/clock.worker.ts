// The clock worker (SPEC 6.5, 7; T-025): owns the network (the track model's `TrackApi`, T-040),
// applies edits, runs profiles and the capacity pass, and sends the main thread what it draws:
// track polylines, keyframes (shared phase tables) and each game hour's trips (the hourly
// re-base, SPEC 7). After a service change it also builds the network snapshot demand runs on,
// debounced 0.5 s (SPEC 6.5). notes/T-025.md.
//
// Every edit answers with an "edited" message (and, if it applied, a new "state"); the state
// holds the whole network. Fine for a city; per-tile updates are follow-up work (notes/T-025.md).

import init, { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { anchor, makeAnchor, setAnchor, toLocal } from "../map/geo";
import { CITIES, cityWithin, HOME_CITY, NEW_GAME_FRAME, WORLD_ID, zoneBox, zonesTouching } from "../game/cities";
import { frameKindName, frameKindNumber, makeFrame, mercK, type Frame, type FrameSpec } from "../game/frame";
import { PERIODS } from "../game/periods";
import { SHADES } from "../game/shadeTable";
import { DAY_KINDS } from "../game/days";
import { shiftAt, timeZoneTable } from "../game/localTime";
import { between, lagged, runLag, strengths, table, type PerHour } from "../game/moneyTiming";
import { networkFromState, newNetCache, runsTrains } from "../map/network";
import { buildComplexes } from "../game/complexes";
import { decodeSave, encodeSave, isNortheastSave, trackFrameKind, readStored, SAVE_FORMAT, START_CASH, START_CLOCK, writeStored, listSlots, writeSlot, renameSlot, deleteSlot, leaseSlot, ORIGINAL_SLOT, listBackups, writeBackup, isBackupOf, nextGameName, newGameName, slotName, type GameMode, type NewGameSpec, type SlotBackup, type SlotLease, type SlotSummary, type StoredSave, type DayLedger, type HourLedger, type SaveHeader } from "./saveFile";
import type { ClockEdge, ClockLine, ClockNode, ClockState, EditOp, FromClock, MoneyState, NetworkSnapshot, PiInput, RegionNow, RouteEnd, StationAt, ToClock } from "./protocol";

/** Run 0 of each line is resampled this often (m) for the renderer's line texture. */
const SAMPLE_STEP = 50;
const SNAPSHOT_DEBOUNCE_MS = 500;
/** Whole trains on a new line across both directions, high/medium/low demand. */
const NEW_LINE_TRAINS: [number, number, number] = [0, 0, 0];
/** Default fare curve (T-028): US$ a ride plus US$ a km. */
const START_FARES = { base: 1.5, perKm: 0.1 };
/** Autosave (T-029): this long after the last one if anything changed, wall ms; at once after a
 * construct or other paid edit, and when the page is hidden. */
const AUTOSAVE_MS = 60_000;
/** T-744: an autosave this long after the slot's newest backup is also written as a backup, wall ms */
let backupMs = 5 * 60_000;
/** Days of money kept in the ledger, and hours in the hourly one (T-447). */
const LEDGER_DAYS = 30;
const LEDGER_HOURS = 48;
const LEVEL_OF = { high: 0, medium: 1, low: 2 } as const;

let api: TrackApi | null = null;
let city = HOME_CITY;
/** the frame of the loaded game (T-125): new games are in the world frame, saves made before it in
 * the legacy one; read from the track model after every load */
let frame: Frame = makeFrame(NEW_GAME_FRAME);
function frameSpec(): FrameSpec { return { kind: frame.kind, lon: frame.lon, lat: frame.lat }; }
function syncFrame() {
  const o = api!.frame_origin();
  frame = makeFrame({ kind: frameKindName(api!.frame_kind()), lon: o[0], lat: o[1] });
}
let version = 0;
let snapVersion = 0;
let epoch = NaN;
let snapTimer = 0;
/** the last edit's time by step, ms (T-229): sent with its answer for `?debug=1` */
let timing: Record<string, number> = {};

// Money and the clock (T-028). The main thread's clock is the game's; it tells this worker each
// time it is set, paused or changes speed, and the worker settles fares and running costs one game
// hour at a time from that.
let fares = { ...START_FARES };
let ledger: DayLedger[] = [];
let hourLedger: HourLedger[] = [];
/** fare income per game hour in each period, US$M (from the main thread's demand results), per
 * kind of day (T-260, game/days.ts) */
let incomePerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
/** the same per line id (T-293), for each line's fares taken today */
let lineIncome = new Map<number, number[][]>();
/** train rides per game hour as `incomePerHour` (T-565), and per line id */
let ridesPerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
let lineRides = new Map<number, number[][]>();
/** T-736: the boarding account's in-period shape strengths per kind of day and period, and the lag
 * (s) money lands after riders set out (game/moneyTiming.ts) */
const peakC: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
let moneyLag = 0;
const fareHour = (per: number[][] | undefined, tl: number) => lagged(table(per), peakC, tl, moneyLag);
const rideHour = (per: number[][] | undefined, tl: number) => between(table(per), peakC, tl, tl + 3600);
const costHour = (run: number[], tl: number) => lagged(((_k, p) => run[LEVEL_OF[PERIODS[p].demand]]) as PerHour, null, tl, moneyLag);
/** T-531: each region's ledgers, kept as the network's (a zone id; "" outside every zone or not yet
 * placed, so the regions add up to the network) */
let regions = new Map<string, { days: DayLedger[]; hours: HourLedger[] }>();
/** T-531: each running line's fares and running cost by region, [region, share] (game/regionMoney.ts) */
let lineShares = new Map<number, { fare: [string, number][]; ride: [string, number][]; run: [string, number][] }>();
/** T-684: each running line's points per game hour by region and kind of day and period (game/points.ts) */
let linePoints = new Map<number, [string, number[][]][]>();
/** T-685: each region's star, the game second it was earned */
let stars = new Map<string, number>();
/** T-706: each line's station note and the city it was chosen in */
let lineNotes = new Map<string, [number, string]>();
/** T-531: paid edits' charges, held under "" until the main thread says how each divides */
const heldCharges = new Map<number, { day: number; hour: number; build: number; trains: number }>();
/** Wall ms on the clock both threads read (`ToClock` "clock"): Date.now() drifts from it, and the
 * worker running ahead of the main thread's clock pulled money time back across midnight (T-488). */
const wallNow = () => performance.timeOrigin + performance.now();
let clk = { t: START_CLOCK, rate: 0, at: wallNow() };
/** Loads so far (T-715): a "clock" message the main thread sent before it saw the latest still
 * carries the previous game's time, and settling to it charged the gap's running costs and then
 * held the money until the clock caught up. */
let game = 0;
/** T-284: money is settled up to this game second, not by whole hours, so the cash moves as fares
 * land. `partHour` is the hour being settled and `partFare`, `partCost` the shares of its fares and
 * running cost already in the cash (each reaches 1 as the hour ends, so an hour still pays its whole
 * fare and cost). */
let settledT = START_CLOCK;
let partHour = Math.floor(START_CLOCK / 3600);
let partFare = (START_CLOCK % 3600) / 3600, partCost = partFare;
/** per game hour (absolute index), the share of its fares paid by each second into it (3601
 * values from 0 to 1), from the boarding account's trains leaving platforms with riders; an hour
 * without one pays evenly (T-284) */
const hourShape = new Map<number, Float32Array>();
/** the cash and today's fares last sent in a "cash" message */
let sentCash = NaN, sentFares = NaN;
/** something to autosave; when the last autosave was written */
let dirty = false;
let savedAt = Date.now();
let saveInFlight: Promise<boolean> | null = null;
let activeSlot = ORIGINAL_SLOT;
let activeLease: SlotLease | null = null;
/** the start menu's new game (T-677): its slot's name, given at its first write */
let newSlotName: { key: string; name: string } | null = null;
/** the active slot while nothing is stored under it (a new game's, T-677) */
let unwritten: string | null = null;
let storageReady = true;
/** `?new=1`, `?synth=1`, `?demandTest=1` (T-320): no slot is written until "Save as a copy" gives
 * the game a slot of its own; a `?new=1` game has no active slot ("") and reads none */
let testPage = false;
let saveFailure: string | undefined;
/** while the game is being swapped for another slot's (T-743): nothing autosaves */
let switching = false;
/** T-728: the game's mode, start zone and opened zones, kept in the save's header (`opened` absent
 * in a save from before it) */
let mode: GameMode = "normal";
let startZone: string | undefined;
let opened: string[] | undefined = [];
function setMode(m: GameMode) {
  mode = m;
  api!.set_unlimited(m === "sandbox");
}
const slotChanges = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel("anitabuilder-save-slots");
slotChanges?.addEventListener("message", () => { if (api) void publishSlots(); });

async function publishSlots(error?: string) {
  let slots: Awaited<ReturnType<typeof listSlots>> = [];
  let backups: SlotBackup[] = [];
  try {
    slots = await listSlots(); storageReady = true;
    if (activeSlot) backups = await listBackups(activeSlot);
  } catch { storageReady = false; }
  const writable = !!activeLease && storageReady && !saveFailure;
  const reason = saveFailure || (!storageReady ? "Browser storage is unavailable. Save to a file to keep this game."
    : !activeLease ? testPage ? "Autosave is off in this test page. Save a copy to keep the game."
      : navigator.locks ? "Autosave is off here because this slot was opened in another tab. Save to a new slot, or close the other tab and reload."
      : "This browser cannot protect save slots. Save to a file to keep this game." : undefined);
  send({ kind: "saveSlots", slots, active: activeSlot, writable, at: slots.find(s => s.key === activeSlot)?.at ?? 0, reason, error, backups });
}

async function flushBeforeSwitch() {
  if (dirty && !activeLease) throw new Error("Save this game to a new slot before leaving it. You can also download a file and reload this page.");
  if (!await autosave()) throw new Error("This game could not be saved. Save to a file before leaving it.");
}

/** T-743: New game, Load a file and a backup's restore each put the game in a slot of its own,
 * named `name`; the game being played is saved first and keeps its slot. `load` replaces the game
 * (throwing changes nothing). A new game's slot is written once something changes, a loaded one's
 * at once. A test page has no slot to keep: the game is replaced in the page and owns none. */
async function intoNewSlot(name: string, load: () => Promise<unknown> | void, writeNow: boolean) {
  if (testPage) {
    await load();
    activeLease?.release();
    activeSlot = "";
    activeLease = null;
    dirty = writeNow;
    await publishSlots();
    return;
  }
  await flushBeforeSwitch();
  const key = "slot-" + crypto.randomUUID();
  const lease = await leaseSlot(key);
  switching = true;
  try {
    await load();
  } catch (err) { lease?.release(); throw err; } finally { switching = false; }
  activeLease?.release();
  activeSlot = key;
  activeLease = lease;
  unwritten = key;
  newSlotName = { key, name };
  saveFailure = undefined;
  dirty = writeNow;
  savedAt = Date.now();
  if (!writeNow || !await autosave()) await publishSlots();
  slotChanges?.postMessage({ key });
}
/** the empty network's save, for a new game (keeps the water mask) */
let emptyTrack: Uint8Array | null = null;
let moneyParams = [2.5, 2, 100, 9];

function send(msg: FromClock, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/** world metres -> lng/lat in the loaded game's frame (game/frame.ts) */
function packLngLat(x: number, y: number): [number, number] {
  return frame.toLonLat(x, y);
}
/** world metres -> local units, relative to the draw anchor the main thread last sent (T-307) */
function packLocal(x: number, y: number): [number, number] {
  const [lng, lat] = packLngLat(x, y);
  return toLocal(lng, lat);
}
function anchorSpec() {
  const a = anchor();
  return { lng: a.lng, lat: a.lat, version: a.version };
}

// ---------------------------------------------------------------- water masks on demand (T-302)
//
// A zone's mask is fetched only when the view comes near its box or track is drawn or loaded within
// it (docs/zones.md 6). The rule that keeps building exact: every zone whose box meets existing track
// has its mask installed. An edit or preview that would make new track is first worked out, and
// if its geometry meets a zone whose mask is not in yet, it waits for the mask and is worked out
// again, so ground over water is never accepted because a mask was late.

/** the packs' folder (the init message's `pack` names one pack in it) */
let packBase = "";
interface WaterRaster { cell: number; x0: number; y0: number; w: number; h: number; rows: Uint32Array; runs: Uint16Array; lon: number; lat: number }
const waterFetches = new Map<string, Promise<WaterRaster | null>>();
/** zones whose masks are in the track model (a zone with no mask counts once its head said so) */
const masksIn = new Set<string>();
/** a point of track this near a zone's box needs its mask: the sampled geometry's spacing to spare */
const TRACK_MARGIN_M = 2_000;
/** the view this near a zone's box fetches its mask ahead, when the view is under VIEW_SPAN_M wide */
const VIEW_MARGIN_M = 20_000;
const VIEW_SPAN_M = 400_000;

function fetchWater(id: string): Promise<WaterRaster | null> {
  let p = waterFetches.get(id);
  if (!p) {
    const url = new URL(id, packBase).href;
    p = (async () => {
      const head = await (await fetch(url + ".json")).json();
      const row = head.arrays.find((a: any) => a.name === "water_row");
      const xs = head.arrays.find((a: any) => a.name === "water_x");
      if (!head.water || !row || !xs) return null;
      const start = row.offset, end = xs.offset + xs.count * 2;
      const res = await fetch(url + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (!res.ok) throw new Error(`${res.status} ${url}.bin`);
      const buf = await res.arrayBuffer();
      const base = res.status === 206 ? start : 0;
      const w = head.water;
      return {
        cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1],
        rows: new Uint32Array(buf, row.offset - base, row.count), runs: new Uint16Array(buf, xs.offset - base, xs.count),
        lon: head.origin.lon, lat: head.origin.lat,
      };
    })();
    waterFetches.set(id, p);
    // a failed fetch is tried again by the next edit or view that needs it
    p.catch((err) => {
      if (waterFetches.get(id) === p) waterFetches.delete(id);
      send({ kind: "error", message: `water mask for ${id} not loaded: ${err}` });
    });
  }
  return p;
}

/** Install the masks of `ids` (fetching those not fetched); resolves once each is in or failed. The
 * raster is laid out in the pack's own frame; the track model puts it in the game's (T-125). */
async function ensureMasks(ids: Iterable<string>): Promise<void> {
  const todo = [...new Set(ids)].filter((id) => !masksIn.has(id));
  if (!todo.length) return;
  const got = await Promise.all(todo.map((id) => fetchWater(id).then((r) => ({ r }), () => null)));
  todo.forEach((id, k) => {
    const g = got[k];
    if (!g || masksIn.has(id)) return;
    masksIn.add(id);
    const r = g.r;
    if (r) api!.add_pack_water(r.cell, r.x0, r.y0, r.w, r.h, r.rows, r.runs, r.lon, r.lat);
  });
}

/** The zones within TRACK_MARGIN_M of any of these world-metre points (x, y every `stride`). */
function zonesOfPoints(xy: ArrayLike<number>, stride = 2, start = 0): string[] {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (let k = start; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    if (lon < w) w = lon; if (lon > e) e = lon;
    if (lat < s) s = lat; if (lat > n) n = lat;
  }
  if (!(e >= w)) return [];
  const near = zonesTouching(w, s, e, n, TRACK_MARGIN_M);
  // one box over points far apart (a long route, a whole save) meets zones none of them is near:
  // test those point by point
  if (near.length <= 1) return near.map((c) => c.id);
  const boxes = near.map((c) => {
    const [bw, bs, be, bn] = zoneBox(c);
    const dy = TRACK_MARGIN_M / 111_320, dx = dy / Math.cos(Math.min(85, Math.max(Math.abs(bs), Math.abs(bn))) * Math.PI / 180);
    return { id: c.id, w: bw - dx, s: bs - dy, e: be + dx, n: bn + dy };
  });
  const hit = new Set<string>();
  for (let k = start; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    for (const b of boxes) if (lon >= b.w && lon <= b.e && lat >= b.s && lat <= b.n) hit.add(b.id);
  }
  return [...hit];
}

/** Work out a preview; while its geometry meets zones whose masks are not in, install them and work it
 * out again. Returns the last result. */
async function withMasks(run: () => Float64Array, points: (out: Float64Array) => number[]): Promise<Float64Array> {
  for (;;) {
    const out = run();
    const need = zonesOfPoints(points(out)).filter((id) => !masksIn.has(id));
    if (!need.length) return out;
    await ensureMasks(need);
    // a mask that could not be fetched (reported) leaves the edit as it was without it
    if (need.every((id) => !masksIn.has(id))) return run();
  }
}

/** world x, y of a route or edge preview's polyline (`preview_route`'s layout) */
function routePoints(out: Float64Array): number[] {
  const nv = out[3], p0 = 5 + nv * 4, xy: number[] = [];
  for (let k = p0; k + 2 < out.length; k += 3) xy.push(out[k], out[k + 1]);
  return xy;
}
/** the same for a node drag's preview (`preview_move_node`) */
function nodePoints(out: Float64Array): number[] {
  const n = out[2], xy: number[] = [];
  let i = 3;
  for (let e = 0; e < n; e++) {
    i += out[i] * 4 + 1;
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) xy.push(out[i], out[i + 1]);
  }
  return xy;
}

/** Before a load (T-737): the masks of the zones the save's nodes and curve points fall in, so the
 * network is derived once (a load is about 3 s in WASM for Japan's 709 lines). `masksForLoaded`
 * still catches a zone only the drawn curves reach. */
async function masksBeforeLoad(net: Uint8Array) {
  if (trackFrameKind(net) !== NEW_GAME_FRAME.kind) return;
  const pts = TrackApi.save_points(net);
  if (!pts.length) return; // undecodable: the load says so
  const shown = frame;
  frame = makeFrame(NEW_GAME_FRAME);
  const missing = zonesOfPoints(pts).filter((id) => !masksIn.has(id));
  frame = shown;
  if (!missing.length) return;
  api!.load(emptyTrack!);
  await ensureMasks(missing);
}

/** After a network is loaded: the masks of every zone its track meets, then the network opened again
 * over them if any was missing (each mask installed on a loaded network would rework all of it). */
async function masksForLoaded(net: Uint8Array) {
  const a = api!;
  syncFrame();
  const xy: number[] = [];
  const ni = a.nodes_info();
  for (let i = 0; i < ni.length; i += 10) xy.push(ni[i + 1], ni[i + 2]);
  const ei = a.edges_info();
  for (let i = 0; i < ei.length; i += 10) {
    const r = a.edge_render(ei[i]);
    for (let k = 0; k + 2 < r.length; k += 3) xy.push(r[k], r[k + 1]);
  }
  const missing = zonesOfPoints(xy).filter((id) => !masksIn.has(id));
  if (!missing.length) return;
  a.load(emptyTrack!);
  await ensureMasks(missing);
  if (!a.load(net)) throw new Error("This save is damaged.");
}

// ---------------------------------------------------------------- state for the main thread

/** `TrackApi.line_auto`: a headway in seconds per level, or null for a line with counts by hand */
function autoOf(v: ArrayLike<number>): [number, number, number] | null {
  return v.length === 3 ? [v[0], v[1], v[2]] : null;
}

function hex(c: number) {
  return "#" + c.toString(16).padStart(6, "0");
}

// What the renderer draws, per edge and per line, in local units, kept between edits: an edit
// re-renders only the edges and lines it dirtied (T-063). Cleared when a game is loaded.
const edgeDrawn = new Map<number, Float32Array>();
const edgeSpeedDrawn = new Map<number, Float32Array>();
const lineDrawn = new Map<number, Float32Array>();
function forget(all = false) {
  if (all) {
    edgeDrawn.clear();
    edgeSpeedDrawn.clear();
    lineDrawn.clear();
    return;
  }
  for (const e of api!.dirty_edges()) { edgeDrawn.delete(e); edgeSpeedDrawn.delete(e); }
  for (const l of api!.dirty_lines()) lineDrawn.delete(l);
}

/** The track model's mark rows (`crossings_info`, `preview_marks`) with x, y moved to local units. */
function markPoints(rows: Float64Array, stride: number): Float32Array {
  const out = new Float32Array(rows.length);
  for (let k = 0; k < rows.length; k += stride) {
    const [x, y] = packLocal(rows[k], rows[k + 1]);
    out.set([x, y], k);
    for (let j = 2; j < stride; j++) out[k + j] = rows[k + j];
  }
  return out;
}

function buildState(): ClockState {
  const a = api!;
  const ei = a.edges_info();
  const offEdges = new Set<number>(a.off_edges());
  const crossoverIds = a.crossover_edges(), crossoverPartners = a.crossover_partners();
  const crossovers = new Map<number, number>(Array.from(crossoverIds, (id, k) => [id, crossoverPartners[k] === 0xffffffff ? -1 : crossoverPartners[k]]));
  const offStations = new Set<number>(a.off_stations());
  const edges: ClockEdge[] = [];
  const drawn: Float32Array[] = [];
  const off: number[] = [0];
  let total = 0;
  for (let i = 0; i < ei.length; i += 10) {
    const id = ei[i];
    edges.push({
      id, a: ei[i + 1], b: ei[i + 2], tracks: ei[i + 3] as 1 | 2, built: ei[i + 4] === 1, off: offEdges.has(id), crossover: crossovers.has(id), partner: crossovers.get(id) ?? -1, lengthM: ei[i + 5], cost: ei[i + 6],
      vminKmh: ei[i + 7], levelMin: ei[i + 8], levelMax: ei[i + 9],
    });
    let d = edgeDrawn.get(id);
    if (!d) {
      const r = a.edge_render(id);
      d = new Float32Array(r.length);
      for (let k = 0; k < r.length; k += 3) {
        const [x, y] = packLocal(r[k], r[k + 1]);
        d[k] = x;
        d[k + 1] = y;
        d[k + 2] = r[k + 2];
      }
      edgeDrawn.set(id, d);
      edgeSpeedDrawn.set(id, a.edge_render_speeds(id));
    }
    drawn.push(d);
    total += d.length;
    off.push(total / 3);
  }
  const pts = new Float32Array(total);
  const edgeSpeed = new Float32Array(total / 3);
  edges.forEach((e, i) => edgeSpeed.set(edgeSpeedDrawn.get(e.id) ?? [], off[i]));
  drawn.reduce((o, d) => (pts.set(d, o), o + d.length), 0);
  const ni = a.nodes_info();
  const nodes: ClockNode[] = [];
  for (let i = 0; i < ni.length; i += 10) {
    const [lng, lat] = packLngLat(ni[i + 1], ni[i + 2]);
    nodes.push({
      id: ni[i], x: ni[i + 1], y: ni[i + 2], lng, lat, level: ni[i + 3], ports: ni[i + 4], builtPorts: ni[i + 5], flying: ni[i + 6] === 1,
      platform: ni[i + 7], stationBuilt: ni[i + 8] === 1, stationOff: offStations.has(ni[i]), name: ni[i + 7] > 0 ? a.node_name(ni[i]) : "", heading: ni[i + 9],
    });
  }
  const li = a.lines_info();
  const lines: ClockLine[] = [];
  for (let i = 0; i < li.length; i += 22) {
    const id = li[i];
    const ok = li[i + 2] === 1, running = li[i + 3] === 1;
    const t3 = (k: number) => [li[i + k], li[i + k + 1], li[i + k + 2]] as [number, number, number];
    lines.push({
      id, name: a.line_name(id), letter: a.line_symbol(id), colour: hex(li[i + 1]), ok, running, broken: li[i + 4] === 1, tph: t3(5), cars: li[i + 21] || li[i + 8], carsSet: li[i + 8],
      lengthM: li[i + 9], roundTrip: t3(10), trains: t3(13), auto: autoOf(a.line_auto(id)), delay: t3(16), dwell: li[i + 19], turnaround: li[i + 20],
      stops: [...a.line_stops(id)], path: [...a.line_path(id)],
      fromStart: [0, 1, 2].map((lev) => [...a.stop_times(id, 0, lev)]),
      stopS: [[...a.line_stop_s(id, 0)], [...a.line_stop_s(id, 1)]],
      buildCost: ok && !running ? a.line_build_cost(id) : 0,
      trainCost: ok && !running ? a.line_train_cost(id) : 0,
      shift: a.line_shift(id),
      // in the state, not asked for after it, so the line panel never draws new stops with old gaps (T-648)
      passed: [...a.passed_stations(id)],
    });
  }
  // Keyframes: phase tables, profile meta, and run 0 of every line resampled.
  const meta = a.profile_meta();
  const nLines = meta.length / 24;
  const lineTable = new Float32Array(Math.max(1, nLines) * 4);
  const runs: Float32Array[] = [];
  let nSamples = 0;
  for (const l of lines) {
    if (!l.ok) continue;
    let d = lineDrawn.get(l.id);
    if (!d) {
      const p = a.run_polyline(l.id, 0, SAMPLE_STEP);
      d = new Float32Array(p.length);
      for (let k = 0; k < p.length; k += 2) d.set(packLocal(p[k], p[k + 1]), k);
      lineDrawn.set(l.id, d);
    }
    runs.push(d);
    lineTable.set([nSamples, d.length / 2, l.lengthM, SAMPLE_STEP], l.id * 4);
    nSamples += d.length / 2;
  }
  const samples = new Float32Array(nSamples * 2);
  runs.reduce((o, d) => (samples.set(d, o), o + d.length), 0);
  const markers = [0, 1, 2].map((lev) => {
    const m = a.markers(lev);
    const out: ClockState["markers"][number] = [];
    for (let k = 0; k < m.length; k += 7) {
      const [x, y] = packLocal(m[k], m[k + 1]);
      out.push({ x, y, rho: m[k + 2], delayS: m[k + 3], kind: m[k + 4], tph: m[k + 5], held: m[k + 6] > 0 });
    }
    return out;
  });
  const crossings = markPoints(a.crossings_info(), 5);
  return {
    version: ++version,
    markers,
    fullTrack: [0, 1, 2].map((lev) => a.full_track(lev)),
    crossings,
    cash: a.cash(),
    builtCost: a.built_cost(),
    blueprintCost: a.blueprint_cost(),
    blueprintItems: Array.from(a.blueprint_cost_items()),
    canUndo: a.can_undo(),
    canRedo: a.can_redo(),
    edges,
    nodes,
    lines,
    links: Array.from(a.station_links()),
    render: {
      edgePts: pts,
      edgeSpeed,
      edgeOff: new Uint32Array(off),
      phases: a.phases(),
      meta,
      samples,
      lineTable,
    },
    money: moneyState(),
    anchor: anchorSpec(),
  };
}

// ---------------------------------------------------------------- money (T-028)

function gameNow(): number {
  return clk.t + Math.max(0, (wallNow() - clk.at) / 1000) * clk.rate;
}

/** Money's now: the clock, or the time settled up to if the clock has been set back behind it, so
 * a charge or the day shown never lands on a day already closed (T-488). */
function moneyNow(): number {
  return Math.max(gameNow(), settledT);
}

/** The entry for `key` in a ledger kept in order of `at`, made if missing (in its place, so a day or
 * hour is never listed twice, T-488); the ledger keeps the newest `cap`. */
function entryIn<T extends { fares: number; running: number; trains: number; build: number; rides?: number }>(list: T[], at: (e: T) => number, key: number, make: () => T, cap: number): T {
  let i = list.length;
  while (i > 0 && at(list[i - 1]) > key) i--;
  if (i > 0 && at(list[i - 1]) === key) return list[i - 1];
  const e = make();
  list.splice(i, 0, e);
  if (list.length > cap) list.splice(0, list.length - cap);
  return e;
}

/** Game day `day`'s ledger entry; the ledger keeps LEDGER_DAYS days. */
function dayEntry(day: number): DayLedger {
  return entryIn(ledger, (d) => d.day, day, () => ({ day, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS);
}

/** The hour's entry in the hourly ledger (T-447); it keeps LEDGER_HOURS hours. */
function hourEntry(hour: number): HourLedger {
  return entryIn(hourLedger, (d) => d.hour, hour, () => ({ hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS);
}

/** T-531: a region's entries for game day `day` and hour `hour`, made if missing. */
function regionEntries(zone: string, day: number, hour: number): [DayLedger, HourLedger] {
  let r = regions.get(zone);
  if (!r) regions.set(zone, (r = { days: [], hours: [] }));
  return [
    entryIn(r.days, (d) => d.day, day, () => ({ day, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS),
    entryIn(r.hours, (d) => d.hour, hour, () => ({ hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS),
  ];
}

/** T-531: every region's today and this hour that has an entry today. */
function regionNow(): RegionNow {
  const day = Math.floor(moneyNow() / 86400), hour = Math.floor(moneyNow() / 3600), out: RegionNow = {};
  for (const [k, r] of regions) {
    const d = r.days[r.days.length - 1], h = r.hours[r.hours.length - 1];
    if (d?.day === day) out[k] = { today: { ...d }, hour: h?.hour === hour ? { ...h } : { hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 } };
  }
  return out;
}

/** T-531: the running lines' money by region for settling: each line's fare income tables and its
 * running cost per game hour at each demand level (as `running_cost`), with its local time. */
function regionLines() {
  const a = api!, li = a.lines_info(), unit = moneyParams[2] / 1e6;
  const costs: { shift: number; run: number[]; shares: [string, number][] }[] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue;
    const id = li[i], km = li[i + 9] / 1000, cars = li[i + 21] || li[i + 8];
    costs.push({ shift: a.line_shift(id), run: [0, 1, 2].map((l) => li[i + 5 + l] * 2 * km * (moneyParams[3] + moneyParams[1] * cars) * unit), shares: lineShares.get(id)?.run ?? [] });
  }
  const fares = [...lineIncome].map(([id, per]) => ({ shift: a.line_shift(id), per, shares: lineShares.get(id)?.fare ?? [] }));
  const rides = [...lineRides].map(([id, per]) => ({ shift: a.line_shift(id), per, shares: lineShares.get(id)?.ride ?? [] }));
  return { costs, fares, rides };
}

/** T-531: hour `t`'s fares (US$M), running cost (US$M) and train rides (T-565) in each region. A
 * line not yet placed counts under "". */
function regionHour(t: number, rl: ReturnType<typeof regionLines>): Map<string, [number, number, number]> {
  const out = new Map<string, [number, number, number]>();
  const spread = (shares: [string, number][], v: number, k: 0 | 1 | 2) => {
    if (!v) return;
    for (const [z, s] of shares.length ? shares : [["", 1] as [string, number]]) {
      let e = out.get(z);
      if (!e) out.set(z, (e = [0, 0, 0]));
      e[k] += v * s;
    }
  };
  for (const f of rl.fares) spread(f.shares, fareHour(f.per, t + f.shift), 0);
  for (const f of rl.rides) spread(f.shares, rideHour(f.per, t + f.shift), 2);
  for (const c of rl.costs) spread(c.shares, costHour(c.run, t + c.shift), 1);
  return out;
}

/** T-531: a region's saved ledgers, as `mergeLedger` makes the network's. */
function loadRegions(saved: SaveHeader["regions"]) {
  regions = new Map(Object.entries(saved ?? {}).map(([k, r]) => [k, { days: mergeLedger(r.ledger, (d) => d.day), hours: mergeLedger(r.hours, (d) => d.hour) }]));
  heldCharges.clear();
}

function savedRegions(copy: <T extends object>(e: T) => T = (e) => ({ ...e })): NonNullable<SaveHeader["regions"]> {
  return Object.fromEntries([...regions].map(([k, r]) => [k, { ledger: r.days.map(copy), hours: r.hours.map(copy) }]));
}

/** An entry as the save keeps it (T-567): money to the dollar and rides to a tenth, which the
 * charts cannot tell from the full figures and which gzips to about 60% of them. */
function rounded<T extends object>(e: T): T {
  const out: Record<string, number> = { ...e } as Record<string, number>;
  for (const k of ["fares", "running", "trains", "build"]) if (typeof out[k] === "number") out[k] = Math.round(out[k] * 1e6) / 1e6;
  if (typeof out.rides === "number") out.rides = Math.round(out.rides * 10) / 10;
  if (typeof out.points === "number") out.points = Math.round(out.points);
  return out as T;
}

/** A saved ledger in order with one entry per day or hour: saves written before T-488 can list a
 * day twice, the second time empty, which merging adds back where it belongs. */
function mergeLedger<T extends { fares: number; running: number; trains: number; build: number; rides?: number; points?: number }>(list: T[] | undefined, at: (e: T) => number): T[] {
  const by = new Map<number, T>();
  for (const e of list ?? []) {
    const k = at(e), d = by.get(k);
    if (!d) by.set(k, { ...e, rides: e.rides ?? 0, points: e.points ?? 0 });
    else { d.fares += e.fares; d.running += e.running; d.trains += e.trains; d.build += e.build; d.rides = (d.rides ?? 0) + (e.rides ?? 0); d.points = (d.points ?? 0) + (e.points ?? 0); }
  }
  return [...by.values()].sort((a, b) => at(a) - at(b));
}

/** A load settles again from the save's clock, so what the loaded ledgers hold after it leaves them
 * and its net comes back into the cash; edits' charges stay, on the clock's day and hour. Such
 * entries come from a clock pulled back (T-488, minutes) or, before T-715, a clock of the game open
 * before (up to a year of running costs with no fares, charged again by each such load). The day
 * ledger keeps 30 days, so the running cost is worked out again on the loaded network up to where the
 * hour ledger ends, times the whole days kept over the same worked out again. Returns US$M. */
function unsettleAfter(clockT: number): number {
  const D = Math.floor(clockT / 86400), H = Math.floor(clockT / 3600);
  const last = hourLedger[hourLedger.length - 1];
  if (!last || last.hour <= H) return 0;
  const groups = localGroups();
  const costAt = (h: number) => groups.reduce((s, g) => s + costHour(g.running, h * 3600 + g.shift), 0);
  const replay = (from: number, to: number) => {
    let c = 0;
    for (let t = from; t < to;) {
      const h = Math.floor(t / 3600), e = Math.min(to, (h + 1) * 3600);
      c += costAt(h) * (e - t) / 3600;
      t = e;
    }
    return c;
  };
  const whole = ledger.filter((d) => d.day > D && d.day < Math.floor(last.hour / 24));
  const kept = whole.reduce((s, d) => s + d.running, 0), again = whole.reduce((s, d) => s + replay(d.day * 86400, (d.day + 1) * 86400), 0);
  const times = kept > 0 && again > 0 ? kept / again : 1;
  const full = costAt(last.hour) * times;
  const end = last.hour * 3600 + 3600 * (full > 0 ? Math.min(1, last.running / full) : 1);
  const back = times * replay(Math.max(clockT, end - 86400 * 366), end); // settle's year at most
  const trim = (days: DayLedger[], hours: HourLedger[]) => {
    const later = days.filter((d) => d.day > D), laterH = hours.filter((e) => e.hour > H);
    const today = laterH.filter((e) => Math.floor(e.hour / 24) === D);
    const fares = [...later, ...today].reduce((s, e) => s + e.fares, 0);
    const keep = { days: days.filter((d) => d.day <= D), hours: hours.filter((e) => e.hour <= H) };
    const d = keep.days.find((x) => x.day === D);
    if (d) for (const e of today) { d.fares -= e.fares; d.running -= e.running; d.rides = (d.rides ?? 0) - (e.rides ?? 0); d.points = (d.points ?? 0) - (e.points ?? 0); }
    const carry = <T extends DayLedger | HourLedger>(list: T[], from: T[], at: (e: T) => number, key: number, make: () => T, cap: number) => {
      const trains = from.reduce((s, e) => s + e.trains, 0), build = from.reduce((s, e) => s + e.build, 0);
      if (trains || build) { const e = entryIn(list, at, key, make, cap); e.trains += trains; e.build += build; }
    };
    carry(keep.days, later, (x) => x.day, D, () => ({ day: D, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS);
    carry(keep.hours, laterH, (x) => x.hour, H, () => ({ hour: H, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS);
    return { ...keep, fares };
  };
  const net = trim(ledger, hourLedger);
  ledger = net.days;
  hourLedger = net.hours;
  for (const [k, r] of regions) { const t = trim(r.days, r.hours); regions.set(k, { days: t.days, hours: t.hours }); }
  return back - net.fares;
}

function moneyState(): MoneyState {
  const a = api!;
  dayEntry(Math.floor(moneyNow() / 86400));
  hourEntry(Math.floor(moneyNow() / 3600));
  return {
    cash: a.cash(),
    mode,
    fleet: a.fleet(),
    carsNeeded: a.cars_needed(),
    carPrice: moneyParams[0],
    runCostCarKm: moneyParams[1],
    runCostTrainKm: moneyParams[3],
    economy: moneyParams[2],
    fares: { ...fares },
    runningPerHour: [0, 1, 2].map((l) => a.running_cost(l)) as [number, number, number],
    ledger: ledger.map((d) => ({ ...d })),
    hours: hourLedger.map((d) => ({ ...d })),
    regions: savedRegions(),
    stars: Object.fromEntries(stars),
    notes: Object.fromEntries(lineNotes),
  };
}

function sendMoney() {
  const m = moneyState();
  sentCash = m.cash;
  sentFares = m.ledger[m.ledger.length - 1]?.fares ?? NaN;
  send({ kind: "money", money: m });
}

/** T-284: the cash and today's ledger, if either moved since last sent. */
function sendCash() {
  const cash = api!.cash(), today = dayEntry(Math.floor(moneyNow() / 86400)), hour = hourEntry(Math.floor(moneyNow() / 3600));
  if (cash === sentCash && today.fares === sentFares) return;
  sentCash = cash;
  sentFares = today.fares;
  send({ kind: "cash", cash, today: { ...today }, hour: { ...hour }, regions: regionNow() });
}

/** T-309: the running lines grouped by the local time they run on, each group's running cost per
 * demand level and its fare income per kind of day and period. With one local time (every game in
 * the Northeast and Toronto) the network's totals, as before. */
function localGroups(): { shift: number; running: number[]; income: number[][]; rides: number[][] }[] {
  const a = api!;
  const shifts = [...a.running_shifts()];
  if (shifts.length <= 1) return [{ shift: shifts[0] ?? 0, running: [0, 1, 2].map((l) => a.running_cost(l)), income: incomePerHour, rides: ridesPerHour }];
  return shifts.map((s) => {
    const sum = (zero: number[][], by: Map<number, number[][]>) => {
      const out = zero.map((p) => p.map(() => 0));
      for (const [id, per] of by) {
        if (a.line_shift(id) !== s) continue;
        per.forEach((row, k) => row.forEach((v, p) => { if (out[k]) out[k][p] += v; }));
      }
      return out;
    };
    return { shift: s, running: [0, 1, 2].map((l) => a.running_cost_in(l, s)), income: sum(incomePerHour, lineIncome), rides: sum(ridesPerHour, lineRides) };
  });
}

/** Fares in and running costs out up to now (T-284: to the game second, not the whole hour). An
 * hour's fare (`fareHour`: the period's income shaped as the riders arrive, landing as rides end)
 * is paid as its trains pick up riders (`hourShape`), its running cost (`costHour`) evenly; by the
 * hour's end both are paid in full. Returns whether an hour ended. */
function settle(): boolean {
  if (!api) return false;
  const now = gameNow();
  if (now - settledT > 86400 * 366) { // a year at most in one go
    settledT = Math.floor((now - 86400 * 366) / 3600) * 3600;
    partHour = settledT / 3600;
    partFare = partCost = 0;
  }
  if (now <= settledT) return false;
  const groups = localGroups(), rl = regionLines();
  const pl = [...linePoints].map(([id, regs]) => ({ shift: api!.line_shift(id), regs }));
  let cash = api.cash(), ended = false;
  while (settledT < now) {
    const h = Math.floor(settledT / 3600);
    if (h !== partHour) { partHour = h; partFare = partCost = 0; }
    const t = h * 3600, end = Math.min(now, t + 3600), full = end >= t + 3600;
    // T-309: each local time's lines at their own period and day; T-260: the hour's own kind of
    // day, the weekday's until a weekend's has come; T-736: shaped and lagged (fareHour)
    let fare = 0, cost = 0, rides = 0;
    for (const g of groups) {
      const tl = t + g.shift;
      fare += fareHour(g.income, tl);
      rides += rideHour(g.rides, tl);
      cost += costHour(g.running, tl);
    }
    const wf = full ? 1 : shareAt(h, end - t), wc = full ? 1 : (end - t) / 3600;
    const df = Math.max(0, wf - partFare), dc = Math.max(0, wc - partCost);
    partFare = Math.max(partFare, wf);
    partCost = Math.max(partCost, wc);
    cash += fare * df - cost * dc;
    const d = dayEntry(Math.floor(t / 86400));
    d.fares += fare * df;
    d.running += cost * dc;
    d.rides = (d.rides ?? 0) + rides * df;
    const e = hourEntry(h);
    e.fares += fare * df;
    e.running += cost * dc;
    e.rides = (e.rides ?? 0) + rides * df;
    for (const [z, [f, c, n]] of regionHour(t, rl)) {
      for (const r of regionEntries(z, d.day, h)) { r.fares += f * df; r.running += c * dc; r.rides = (r.rides ?? 0) + n * df; }
    }
    // T-684: points land as fares do, each line's at its own local period and kind of day
    if (df > 0) for (const { shift, regs } of pl) {
      for (const [z, per] of regs) {
        const v = fareHour(per, t + shift) * df;
        if (!(v > 0)) continue;
        d.points = (d.points ?? 0) + v;
        e.points = (e.points ?? 0) + v;
        for (const r of regionEntries(z, d.day, h)) r.points = (r.points ?? 0) + v;
      }
    }
    settledT = full ? t + 3600 : end;
    if (full) { ended = true; hourShape.delete(h); }
  }
  api.set_cash(cash);
  dirty = true;
  return ended;
}

/** T-446: every line's past 24 hours (`TrackApi.past_day`, `today`'s layout), then each listed
 * line's fares taken in them, US$M, in the same order: the hour now by the share already paid, as
 * `settle` pays the network's, and the hour a day ago by the rest. */
function today(): Float64Array {
  if (settle()) sendMoney();
  return packDays(lineDaysAt(Math.min(gameNow(), settledT), false, undefined, true));
}

/** One line's day, in `TrackApi.today`'s per-line layout, with its fares (US$M). */
interface LineDay { id: number; block: ArrayLike<number>; fare: number }

/** Every line's block of `TrackApi.today(now)` with its fares taken: each hour since the line's
 * local midnight (T-309), at its local period and day, by the share already paid (`closed`: the
 * day is over, every hour paid whole). `shift` keeps the lines on one local time. `past`: the past
 * 24 hours instead (T-446). */
function lineDaysAt(now: number, closed: boolean, shift?: number, past = false): LineDay[] {
  const a = api!;
  const v = past ? a.past_day(now) : a.today(now), count = v[0], hNow = Math.floor(now / 3600);
  const out: LineDay[] = [];
  let i = 1;
  for (let k = 0; k < count; k++) {
    const id = v[i], n = v[i + 1], spans = v[i + 6], len = 15 + 2 * spans + n;
    const sh = a.line_shift(id);
    if (shift === undefined || sh === shift) {
      const per = lineIncome.get(id);
      let f = 0;
      const h0 = past ? hNow - 24 : Math.floor((now + sh) / 86400) * 24 - sh / 3600;
      const wNow = closed ? 1 : hNow === partHour ? partFare : 0;
      if (per) for (let h = h0; h <= hNow; h++) {
        const t = h * 3600 + sh, w = h === hNow ? wNow : past && h === h0 ? 1 - wNow : 1;
        f += fareHour(per, t) * w;
      }
      out.push({ id, block: v.subarray(i, i + len), fare: f });
    }
    i += len;
  }
  return out;
}

/** The flat array the panels read (game/today.ts `readToday`): the line count, each line's block,
 * then each line's fares. */
function packDays(items: LineDay[]): Float64Array {
  const len = items.reduce((s, d) => s + d.block.length, 1);
  const out = new Float64Array(len + items.length);
  out[0] = items.length;
  let i = 1;
  for (const d of items) { out.set(d.block, i); i += d.block.length; }
  items.forEach((d, k) => { out[i + k] = d.fare; });
  return out;
}

/** The share of hour `h`'s fares paid by `s` seconds into it (T-284). */
function shareAt(h: number, s: number): number {
  const c = hourShape.get(h);
  if (!c) return s / 3600;
  return c[Math.max(0, Math.min(3600, Math.floor(s)))];
}

/** T-284: when hour `epoch`'s fares are paid, from its slice of the boarding account
 * (game/boarding.ts `readBoardHour` has the layout): each train leaving a platform in the hour
 * takes the riders waiting less those it leaves behind, and the hour's fares are paid in
 * proportion as they board. Only the timing within the hour comes from here. */
function shapeHour(epoch: number, b: Float32Array) {
  const h = Math.round(epoch / 3600);
  for (const k of hourShape.keys()) if (k < partHour - 1) hourShape.delete(k);
  const nt = b[0], nl = b[1], nq = b[2], ne = b[3], S = 12;
  const q0 = 5 + 2 * nt + nl, e0 = q0 + nq * S, p0 = e0 + ne * 2;
  const bins = new Float64Array(3600);
  let total = 0;
  for (let q = 0; q < nq; q++) {
    const Q = q0 + q * S, rate = b[Q + 4], first = b[Q + 8], n = b[Q + 9];
    let p = b[Q + 10];
    const pEnd = p + b[Q + 11];
    for (let j = first + 1; j < first + n; j++) {
      const t = b[e0 + j * 2];
      const tPrev = b[e0 + (j - 1) * 2], leftPrev = b[e0 + (j - 1) * 2 + 1];
      // riders changing from other lines' trains since the train before (T-554)
      let changed = 0;
      for (; p < pEnd && b[p0 + p * 2] <= t; p++) if (b[p0 + p * 2] > tPrev) changed += b[p0 + p * 2 + 1];
      if (!(t >= 0 && t < 3600)) continue;
      const w = Math.max(0, leftPrev + rate * Math.max(0, t - tPrev) + changed - b[e0 + j * 2 + 1]);
      bins[Math.floor(t)] += w;
      total += w;
    }
  }
  if (!(total > 0)) { hourShape.delete(h); return; }
  const c = new Float32Array(3601);
  let acc = 0;
  for (let s = 0; s < 3600; s++) { c[s] = acc / total; acc += bins[s]; }
  c[3600] = 1;
  hourShape.set(h, c);
}

// ---------------------------------------------------------------- saves (T-029)

function header(): SaveHeader {
  return { game: "anitabuilder", format: SAVE_FORMAT, city, region: WORLD_ID, cities: CITIES.map(c => c.id), clock: gameNow(), cash: api!.cash(), fleet: api!.fleet(), fares: { ...fares }, ledger: ledger.map(rounded), hours: hourLedger.map(rounded), regions: savedRegions(rounded), stars: Object.fromEntries(stars), notes: Object.fromEntries(lineNotes), mode, start: startZone, opened: opened && [...opened] };
}

/** The game as file bytes. The network and header are taken at once, before anything awaits. */
function saveBytes(): Promise<Uint8Array> {
  settle();
  return encodeSave(header(), api!.save());
}

/** T-241: the saves list's picture of the network and its figures. Read at once, before anything
 * awaits (call it straight after saveBytes); a failure leaves the picture out, never the save. */
// T-251: drawn at twice the size the saves panel shows it (192 x 128), for sharp lines on high-density screens
const THUMB_W = 384, THUMB_H = 256, THUMB_PAD = 16;
/** the theme the page is in (main.ts sends it): a thumbnail is drawn on that theme's land (T-543) */
let thumbShade: "light" | "dark" = "light";
async function slotSummary(): Promise<SlotSummary> {
  const a = api!;
  // stations and the cities they are in
  const ni = a.nodes_info();
  const byCity = new Map<string, number>();
  let stations = 0;
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] <= 0) continue;
    stations++;
    const [lng, lat] = packLngLat(ni[i + 1], ni[i + 2]);
    const c = cityWithin(lng, lat); // T-318: a station over dead space is in no city
    if (c) byCity.set(c, (byCity.get(c) ?? 0) + 1);
  }
  const cities = [...byCity].sort((x, y) => y[1] - x[1]).map(([c]) => c);
  const li = a.lines_info();
  const summary: SlotSummary = { cities, lines: li.length / 22, stations, cash: a.cash(), mode };
  try {
    if (typeof OffscreenCanvas === "undefined") return summary;
    // track (grey) and each line's route (its colour), in local units: what the renderer draws
    const track: ArrayLike<number>[] = [];
    const ei = a.edges_info();
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i];
      let d: ArrayLike<number> | undefined = edgeDrawn.get(id);
      if (!d) {
        const r = a.edge_render(id), out = new Float32Array(r.length);
        for (let k = 0; k < r.length; k += 3) out.set([...packLocal(r[k], r[k + 1]), r[k + 2]], k);
        d = out;
      }
      const xy = new Float32Array((d.length / 3) * 2);
      for (let k = 0, j = 0; k < d.length; k += 3, j += 2) { xy[j] = d[k]; xy[j + 1] = d[k + 1]; }
      track.push(xy);
    }
    const routes: { colour: string; xy: ArrayLike<number> }[] = [];
    for (let i = 0; i < li.length; i += 22) {
      if (li[i + 2] !== 1) continue;
      const id = li[i];
      let d: ArrayLike<number> | undefined = lineDrawn.get(id);
      if (!d) {
        const p = a.run_polyline(id, 0, SAMPLE_STEP), out = new Float32Array(p.length);
        for (let k = 0; k < p.length; k += 2) out.set(packLocal(p[k], p[k + 1]), k);
        d = out;
      }
      routes.push({ colour: hex(li[i + 1]), xy: d });
    }
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const xy of [...track, ...routes.map((r) => r.xy)])
      for (let k = 0; k < xy.length; k += 2) {
        x0 = Math.min(x0, xy[k]); x1 = Math.max(x1, xy[k]);
        y0 = Math.min(y0, xy[k + 1]); y1 = Math.max(y1, xy[k + 1]);
      }
    if (!(x1 >= x0)) return summary; // nothing built or drawn
    // at least 3 km across, so a single short stretch is not blown up to fill the picture
    const span = Math.max(x1 - x0, (y1 - y0) * THUMB_W / THUMB_H, 3000);
    const s = (THUMB_W - 2 * THUMB_PAD) / span;
    const ox = THUMB_W / 2 - ((x0 + x1) / 2) * s, oy = THUMB_H / 2 - ((y0 + y1) / 2) * s;
    const canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
    const ctx = canvas.getContext("2d")!;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.fillStyle = SHADES[thumbShade].land;
    ctx.fillRect(0, 0, THUMB_W, THUMB_H);
    const stroke =(xy: ArrayLike<number>) => {
      ctx.beginPath();
      let lx = NaN, ly = NaN;
      for (let k = 0; k < xy.length; k += 2) {
        const px = ox + xy[k] * s, py = oy + xy[k + 1] * s;
        if (k > 0 && k < xy.length - 2 && Math.hypot(px - lx, py - ly) < 0.8) continue;
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        lx = px; ly = py;
      }
      ctx.stroke();
    };
    ctx.strokeStyle = SHADES[thumbShade].thumbTrack;
    ctx.lineWidth = 2.5;
    track.forEach(stroke);
    ctx.lineWidth = 4.5;
    for (const r of routes) { ctx.strokeStyle = r.colour; stroke(r.xy); }
    const png = new Uint8Array(await (await canvas.convertToBlob({ type: "image/png" })).arrayBuffer());
    let bin = "";
    for (let k = 0; k < png.length; k += 0x8000) bin += String.fromCharCode(...png.subarray(k, k + 0x8000));
    summary.thumb = "data:image/png;base64," + btoa(bin);
  } catch { /* no picture */ }
  return summary;
}

function autosave(): Promise<boolean> {
  if (saveInFlight) return saveInFlight.then(ok => dirty && ok ? autosave() : ok);
  if (!api || !dirty) return Promise.resolve(true);
  if (!activeLease || switching) return Promise.resolve(false);
  // T-677: a new game whose slot was never written is not kept while its network is empty
  if (unwritten === activeSlot && !api.nodes_info().length && !api.lines_info().length) {
    dirty = false;
    return Promise.resolve(true);
  }
  const key = activeSlot; // an in-flight write must never move to the next slot
  dirty = false;
  saveInFlight = (async () => {
    try {
      const day = Math.floor(gameNow() / 86400);
      const t0 = performance.now();
      const bytesP = saveBytes(), summaryP = slotSummary();
      const bytes = await bytesP;
      const encodeMs = performance.now() - t0;
      const summary = await summaryP;
      const at = Date.now();
      const name = newSlotName?.key === key ? newSlotName.name : undefined;
      await writeSlot(key, { bytes, at, day, city, ...summary, name });
      if (name) newSlotName = null;
      if (unwritten === key) unwritten = null;
      const backups = await listBackups(key).catch(() => null);
      if (backups && !(at - (backups[0]?.at ?? -Infinity) < backupMs)) await writeBackup(key, { bytes, at, day, city, ...summary });
      savedAt = at;
      saveFailure = undefined;
      send({ kind: "saved", at, bytes: bytes.length, encodeMs, totalMs: performance.now() - t0 });
      await publishSlots();
      slotChanges?.postMessage({ key });
      return true;
    } catch {
      dirty = true;
      savedAt = Date.now(); // retry on the next autosave interval, rather than every tick
      saveFailure = "The browser could not save this game. Save to a file to keep your changes.";
      await publishSlots(saveFailure);
      return false;
    } finally { saveInFlight = null; }
  })();
  return saveInFlight;
}

async function manageSlot(msg: Extract<ToClock, { kind: "saveSlot" }>) {
  try {
    if (msg.action === "refresh") { await publishSlots(); return; }
    if (msg.action === "create") {
      const key = "slot-" + crypto.randomUUID();
      const lease = await leaseSlot(key);
      if (!lease) throw new Error("This browser cannot protect save slots. Save to a file instead.");
      try {
        const bytesP = saveBytes(), summaryP = slotSummary();
        const bytes = await bytesP;
        await writeSlot(key, { bytes, at: Date.now(), day: Math.floor(gameNow() / 86400), city, name: msg.name?.trim() || "Saved game", ...await summaryP });
        // Finish earlier writes before releasing the old slot's guard. The new copy is already safe.
        if (activeLease) await autosave();
        activeLease?.release();
        activeSlot = key;
        activeLease = lease;
        saveFailure = undefined;
        dirty = true;
        savedAt = Date.now();
      } catch (err) { lease.release(); throw err; }
    } else if (msg.action === "open") {
      const key = msg.key;
      if (!key || key === activeSlot) { await publishSlots(); return; }
      await flushBeforeSwitch();
      const lease = await leaseSlot(key);
      try {
        // Read after acquiring the guard: the previous owner may have saved just before release.
        const stored = await readStored(key);
        if (!stored) throw new Error("That save slot no longer exists.");
        const decoded = await decodeSave(stored.bytes);
        if (!isNortheastSave(decoded.header)) throw new Error("This save is for another region.");
        switching = true;
        const converted = await loadGame(stored.bytes, "file").finally(() => { switching = false; });
        activeLease?.release();
        activeSlot = key;
        activeLease = lease;
        saveFailure = undefined;
        dirty = false;
        savedAt = stored.at;
        if (converted) await saveConverted(key, stored);
      } catch (err) { lease?.release(); throw err; }
    } else if (msg.action === "restore") {
      const key = msg.key, from = activeSlot;
      if (!key || !isBackupOf(key, from)) throw new Error("That backup no longer exists.");
      const backup = await readStored(key), slot = await readStored(from);
      if (!backup) throw new Error("That backup no longer exists.");
      const time = new Date(backup.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      await intoNewSlot(`${slotName(from, slot)} at ${time}`, () => loadGame(backup.bytes, "file"), true);
      return;
    } else {
      const key = msg.key;
      if (!key) throw new Error("Choose a save slot first.");
      if (msg.action === "delete" && key === activeSlot) throw new Error("Open another slot before deleting this one.");
      const temporary = key === activeSlot ? null : await leaseSlot(key);
      if (key === activeSlot ? !activeLease : !temporary) throw new Error("That slot is open in another tab. Close it before changing this save.");
      try {
        if (msg.action === "rename") await renameSlot(key, msg.name ?? "");
        else await deleteSlot(key);
      } finally { temporary?.release(); }
    }
    await publishSlots();
    slotChanges?.postMessage({ key: activeSlot });
  } catch (err) { await publishSlots((err as Error).message); }
}

/** whether the game just loaded was converted from an older frame (T-308), for the "loaded" message */
let convertedOnLoad = false;

/** Replace the game with a save's. Throws a player-readable reason, changing nothing. A save whose
 * track is not in the world frame is converted once, through longitude and latitude, and loaded in
 * it (T-308); the return value says so, and the caller then writes the slot again in the new
 * format. Money, fares and the ledger load as saved, but for anything settled after the clock
 * (`unsettleAfter`). With `fresh`, the save is a real network a new game starts from (T-731): only
 * its network, fleet and fares are kept, the rest is a new game's. */
async function loadGame(bytes: Uint8Array, from: "autosave" | "file" | "new", fresh?: NewGameSpec): Promise<boolean> {
  const decoded = await decodeSave(bytes);
  const { track } = decoded;
  const h: SaveHeader = fresh
    ? { ...decoded.header, clock: startClock(fresh), cash: START_CASH, ledger: [], hours: [], regions: undefined, stars: undefined, notes: undefined, ...newGameHeader(fresh) }
    : decoded.header;
  if (!isNortheastSave(h)) throw new Error("This save is for another region.");
  let net = track, converted = false;
  if (trackFrameKind(track) !== NEW_GAME_FRAME.kind) {
    const c = TrackApi.convert_save_frame(track, frameKindNumber(NEW_GAME_FRAME.kind));
    // an unreadable network converts to nothing: let the load below say so
    if (c.length) { net = c; converted = true; }
  }
  await masksBeforeLoad(net);
  if (!api!.load(net)) throw new Error("This save is damaged.");
  await masksForLoaded(net);
  api!.set_fleet(h.fleet ?? 0);
  fares = { ...START_FARES, ...h.fares };
  ledger = mergeLedger(h.ledger, (d) => d.day);
  hourLedger = mergeLedger(h.hours, (d) => d.hour);
  loadRegions(h.regions);
  stars = new Map(Object.entries(h.stars ?? {}));
  lineNotes = new Map(Object.entries(h.notes ?? {}));
  api!.set_cash(h.cash + unsettleAfter(h.clock));
  setMode(h.mode === "sandbox" ? "sandbox" : "normal");
  startZone = h.start;
  opened = h.opened;
  convertedOnLoad = converted;
  restart(h.clock, from);
  return converted;
}

/** A new game's clock (T-728): the first time from day 1 07:00 on the game clock at which it is
 * 07:00 in the start city, so every start opens on its own morning peak (Tokyo: Tue 2 07:00 there). */
function startClock(spec?: NewGameSpec): number {
  const c = spec?.start ? CITIES.find((x) => x.id === spec.start) : undefined;
  const ahead = c ? shiftAt(c.center[0], c.center[1]) : 0;
  return START_CLOCK + (((-ahead % 86400) + 86400) % 86400);
}

/** A new game's mode, start and opened zones (T-728). */
function newGameHeader(spec?: NewGameSpec): Pick<SaveHeader, "mode" | "start" | "opened"> {
  return { mode: spec?.mode ?? "normal", start: spec?.start, opened: spec?.start ? [spec.start] : [] };
}
function sendOpened() {
  send({ kind: "opened", zones: opened ? [...opened] : null });
}
function setNewGame(spec?: NewGameSpec) {
  const g = newGameHeader(spec);
  setMode(g.mode!);
  startZone = g.start;
  opened = g.opened;
}

/** After a slot's save was converted on load: keep the old-format copy under a key the saves list
 * does not show, then write the slot in the new format. */
async function saveConverted(key: string, old: StoredSave) {
  await writeStored("old-frame-" + key, old);
  dirty = true;
  await autosave();
}

/** A new game: empty, or from `spec.real`'s network (T-731). */
async function newGame(spec?: NewGameSpec) {
  if (spec?.real) {
    await loadGame(new Uint8Array(spec.real), "new", spec);
    return;
  }
  setNewGame(spec);
  api!.load(emptyTrack!);
  api!.set_cash(START_CASH);
  api!.set_fleet(0);
  fares = { ...START_FARES };
  ledger = [];
  hourLedger = [];
  loadRegions(undefined);
  stars = new Map();
  lineNotes = new Map();
  restart(startClock(spec), "new");
}

/** After a load or a new game: the clock, fresh state, a new snapshot for demand. */
function restart(t: number, from: "autosave" | "file" | "new") {
  clk = { t, rate: clk.rate, at: wallNow() };
  game++;
  // T-284: the loaded hour counts as paid up to the clock, evenly (its trains' timing is not known yet)
  settledT = t;
  partHour = Math.floor(t / 3600);
  partFare = partCost = (t - partHour * 3600) / 3600;
  hourShape.clear();
  incomePerHour =DAY_KINDS.map(() => PERIODS.map(() => 0)); // the main thread sends the new network's once demand lands
  ridesPerHour = DAY_KINDS.map(() => PERIODS.map(() => 0));
  lineIncome = new Map(); // T-715: the old game's per line, which regions' fares were paid from
  lineRides = new Map();
  linePoints = new Map();
  api!.clear_riders(); // and its riders (T-164)
  forget(true);
  syncFrame();
  sendOpened();
  send({ kind: "loaded", from, clock: t, frame: frameSpec(), converted: convertedOnLoad || undefined, game });
  convertedOnLoad = false;
  sendState();
  clearTimeout(snapTimer);
  send({ kind: "snapshot", snapshot: buildSnapshot() });
}

/** the last build's pieces of the renderer's buffers (T-496) */
const netCache = newNetCache();
/** the lines with trains at the last full build */
let runningLines = "";

/** `service`: only the service changed (a train count, T-229): the renderer's buffers are not built
 * again, which was about half the edit's time on New York's real network. */
function sendState(service = false) {
  const t0 = performance.now();
  const s = buildState();
  const t1 = performance.now();
  moneyLag = runLag(s.lines.filter((l) => l.ok && l.running).map((l) => ({ tph: l.tph[1], lengthM: l.lengthM, cars: l.cars, oneWayS: l.roundTrip[1] / 2 - l.turnaround })));
  // a line gaining its first train or losing its last changes the draw order (T-533)
  const running = s.lines.filter((l) => l.ok && runsTrains(l)).map((l) => l.id).join();
  if (service && running === runningLines) {
    s.service = true;
    s.ms = [t1 - t0, 0];
    timing.state = t1 - t0;
    const r = s.render;
    postState(s, [r.edgePts.buffer, r.edgeOff.buffer, r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer, ...(r.edgeSpeed ? [r.edgeSpeed.buffer] : [])] as Transferable[]);
    sendTrips();
    return;
  }
  // The renderer's buffers are built here too, so the main thread only uploads them (T-063).
  runningLines = running;
  const n = networkFromState(s, netCache);
  s.net = n;
  s.ms = [t1 - t0, performance.now() - t1];
  timing.state = t1 - t0;
  timing.buffers = performance.now() - t1;
  const r = s.render;
  const bufs = new Set<ArrayBufferLike>([r.edgePts.buffer, r.edgeOff.buffer, r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
  if (r.edgeSpeed) bufs.add(r.edgeSpeed.buffer);
  for (const st of [n.track, n.lines]) for (const v of [st.seg, st.colour, st.edge, st.level, st.flags, st.dist, st.speed, st.lane]) if (v) bufs.add(v.buffer);
  for (const v of [n.lineColours, n.lineCars, n.stations, n.stationAxis, n.stationShape, n.junctions, n.marks]) if (v) bufs.add(v.buffer);
  postState(s, [...bufs] as Transferable[]);
  sendTrips();
}

/** the edges and nodes of the last state posted */
let postedEdges: ClockEdge[] = [], postedNodes: ClockNode[] = [];

/** Rows of plain fields, field for field. */
function sameRows<T extends object>(a: T[], b: T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as Record<string, unknown>, y = b[i] as Record<string, unknown>;
    for (const k in x) if (x[k] !== y[k]) return false;
  }
  return true;
}

/** Post a state, leaving out the edges and nodes when they are as last posted (T-648): copying
 * them to the main thread and back into objects was about half of the state's ~12 ms trip there on
 * New York's real network, and a line edit changes neither. */
function postState(s: ClockState, transfer: Transferable[]) {
  const sameEdges = sameRows(s.edges, postedEdges), sameNodes = sameRows(s.nodes, postedNodes);
  postedEdges = s.edges;
  postedNodes = s.nodes;
  send({ kind: "state", state: { ...s, edges: sameEdges ? [] : s.edges, nodes: sameNodes ? [] : s.nodes, sameEdges, sameNodes } }, transfer);
}

/** Lines of the boarding account worked out per step: a few ms on New York's real network, so an
 * edit or the money tick never waits long behind new riders (T-164, notes/T-164.md). */
const BOARDING_STEP_LINES = 3;
let boardingTimer = 0;
/** time spent stepping since the last hour sent, and its longest step (`?debug=1`) */
let prep = { ms: 0, max: 0 };
function stepBoarding() {
  clearTimeout(boardingTimer);
  if (!api) return;
  const t0 = performance.now();
  const more = api.boarding_step(BOARDING_STEP_LINES);
  const ms = performance.now() - t0;
  prep = { ms: prep.ms + ms, max: Math.max(prep.max, ms) };
  if (more) boardingTimer = setTimeout(stepBoarding, 0) as unknown as number;
  else sendTrips();
}

function sendTrips() {
  if (!api || !Number.isFinite(epoch)) return;
  api.set_now(gameNow());
  timing.trips = 0;
  timing.board = 0;
  for (const e of [epoch, epoch + 3600]) {
    // per trip: profile, departure, end and the departure its train takes next, relative to the
    // epoch; the end (T-206) and the next (T-245) go apart
    const tT = performance.now();
    const raw = api.trips(e), n = raw.length / 4;
    timing.trips += performance.now() - tT;
    const trips = new Float32Array(n * 2), ends = new Float32Array(n), nexts = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      trips[i * 2] = raw[i * 4];
      trips[i * 2 + 1] = raw[i * 4 + 1];
      ends[i] = raw[i * 4 + 2];
      nexts[i] = raw[i * 4 + 3];
    }
    // The boarding account's hour (T-164), once demand's riders are here: each trip's loads and
    // the platforms' crowds. The account is worked out again only for lines whose service or
    // riders changed.
    if (api.has_riders()) {
      const t0 = performance.now();
      const board = api.boarding_hour(e);
      shapeHour(e, board); // T-284: when this hour's fares land, read before the slice is sent
      timing.board += performance.now() - t0;
      send({ kind: "trips", version, epoch: e, trips, ends, nexts, board, boardMs: performance.now() - t0, redone: api.boarding_redone(), prepMs: prep.ms, prepMax: prep.max }, [trips.buffer, ends.buffer, nexts.buffer, board.buffer]);
      prep = { ms: 0, max: 0 };
    } else send({ kind: "trips", version, epoch: e, trips, ends, nexts }, [trips.buffer, ends.buffer, nexts.buffer]);
  }
}

// ---------------------------------------------------------------- the snapshot demand runs on

function buildSnapshot(): NetworkSnapshot {
  const a = api!;
  const ni = a.nodes_info();
  const stations: NetworkSnapshot["stations"] = [];
  const geo: { id: string; x: number; y: number; heading: number; length: number }[] = [];
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] > 0 && ni[i + 8] === 1) {
      stations.push({ id: ni[i], name: a.node_name(ni[i]), x: ni[i + 1], y: ni[i + 2], level: ni[i + 3] as any, platform: ni[i + 7] });
      geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
    }
  }
  // T-161: each constructed station's complex, among constructed stations only (a blueprint
  // platform carries nobody, so it joins nothing), with the player's links (T-197)
  const fl = a.station_links(), links: [string, string][] = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = buildComplexes(geo, links, (y) => (frame.kind === "merc" ? mercK(y) : 1));
  stations.forEach((s) => (s.complex = cx.of.get(String(s.id))));
  const li = a.lines_info();
  const lines: NetworkSnapshot["lines"] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue; // running lines only
    const id = li[i];
    const hopS: number[][][] = [], dwellS: number[][][] = [];
    for (let run = 0; run < 2; run++) {
      hopS.push([]);
      dwellS.push([]);
      for (let lev = 0; lev < 3; lev++) {
        const arr = a.stop_times(id, run, lev), dep = a.stop_departures(id, run, lev);
        hopS[run].push(Array.from({ length: arr.length - 1 }, (_, k) => arr[k + 1] - dep[k]));
        dwellS[run].push(Array.from(arr, (t, k) => (k === 0 || k === arr.length - 1 ? 0 : dep[k] - t)));
      }
    }
    lines.push({
      id, name: a.line_name(id), stops: [...a.line_stops(id)], tph: [li[i + 5], li[i + 6], li[i + 7]], hopS, dwellS,
      hopKm: [0, 1].map((run) => { const s = a.line_stop_s(id, run); return Array.from({ length: s.length - 1 }, (_, k) => (s[k + 1] - s[k]) / 1000); }),
      turnaroundS: li[i + 20], cars: li[i + 21],
    });
  }
  return { version: ++snapVersion, city, frame: frameSpec(), stations, lines, fare: { ...fares } };
}

function scheduleSnapshot() {
  clearTimeout(snapTimer);
  snapTimer = setTimeout(() => send({ kind: "snapshot", snapshot: buildSnapshot() }), SNAPSHOT_DEBOUNCE_MS) as unknown as number;
}

// ---------------------------------------------------------------- edits

function endArr(e: RouteEnd): number[] {
  if (e.kind === "free") return [0, 0, e.x, e.y, e.level];
  if (e.kind === "node") return [1, e.node, 0, 0, 0];
  return [2, e.edge, e.x, e.y, 0];
}
function freeEnds(at: Extract<StationAt, { kind: "free" }>): Float64Array {
  return new Float64Array([0, 0, at.x0, at.y0, at.level, 0, 0, at.x1, at.y1, at.level]);
}
function pisArr(p: PiInput[]): Float64Array {
  return new Float64Array(p.flatMap((v) => [v.x, v.y, v.radius ?? 0, v.level]));
}
function colourInt(c: string) {
  return parseInt(c.slice(1), 16);
}

function apply(op: EditOp): { r: number; line?: number; node?: number } {
  const a = api!;
  switch (op.op) {
    case "route": {
      a.set_route_tangent(!!op.through);
      const r = a.add_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : op.quad ? 4 : 2, false);
      return { r, node: r === 0 ? a.route_end() : undefined };
    }
    case "edgePis":
      return { r: a.set_edge_pis(op.edge, pisArr(op.pis)) };
    case "moveNode":
      return { r: a.move_node(op.node, op.x, op.y) };
    case "moveNodes":
      return { r: a.move_nodes(new Uint32Array(op.nodes), op.dx, op.dy) };
    case "edgeLevel":
      return { r: a.set_edge_level(op.edge, op.level) };
    case "removeEdge":
      return { r: a.delete_edge(op.edge) };
    case "flying":
      return { r: a.set_flying(op.node, op.flying) };
    case "addStation":
      return op.at.kind === "node"
        ? { r: a.add_station(1, op.at.node, 0, 0, op.platform, op.name) }
        : op.at.kind === "edge"
          ? { r: a.add_station(2, op.at.edge, op.at.x, op.at.y, op.platform, op.name) }
          : { r: a.add_free_station(freeEnds(op.at), op.at.quad ? 4 : 2, op.platform, op.name) };
    case "crossover":
      return { r: a.add_crossover(op.edge, op.x, op.y) };
    case "doubleUp":
      return { r: a.double_up(op.edge) };
    case "stationProps":
      return { r: a.set_station_props(op.node, op.platform, op.name) };
    case "removeStation":
      return { r: a.remove_station(op.node) };
    case "construct":
      return { r: a.construct(new Uint32Array(op.edges), new Uint32Array(op.stations)) };
    case "constructAll":
      return { r: a.construct_all() };
    case "blueprintOff":
      return { r: a.set_blueprint_off(new Uint32Array(op.edges), new Uint32Array(op.stations), op.off) };
    case "constructLine":
      return { r: a.construct_line(op.line) };
    case "addLine": {
      const id = a.new_line_id();
      const [h, m, l] = NEW_LINE_TRAINS;
      return { r: a.set_line_with_trains(id, new Uint32Array(op.stops), h, m, l, 30, 180, 0, colourInt(op.colour), op.name || a.next_line_name()), line: id };
    }
    case "copyLine": {
      const id = a.new_line_id();
      const existing = new Set<string>();
      const info = a.lines_info();
      for (let i = 0; i < info.length; i += 22) existing.add(a.line_name(info[i]));
      const base = op.name?.trim() || `${a.line_name(op.line)} copy`;
      let name = base, n = 2;
      while (existing.has(name)) name = `${base} ${n++}`;
      return { r: a.copy_line(op.line, id, name), line: id };
    }
    case "setStops":
      return { r: a.set_stops(op.line, new Uint32Array(op.stops)) };
    case "insertStop": {
      // one message instead of a query and then setStops: a round trip less, and no stale stops
      const stops = a.insert_stop(op.line, op.station);
      return stops.length ? { r: a.set_stops(op.line, Uint32Array.from(stops)) } : { r: INSERT_REFUSED };
    }
    case "setTrainCount":
      return { r: a.set_train_counts(op.line, ...op.trains) };
    case "setHeadway":
      return { r: op.seconds ? a.set_auto_headway(op.line, ...op.seconds) : a.clear_auto_headway(op.line) };
    case "lineLook":
      return { r: op.letter === undefined
        ? a.set_line_look(op.line, op.name, colourInt(op.colour))
        : a.set_line_badge(op.line, op.name, colourInt(op.colour), op.letter.trim()) };
    case "removeLine":
      return { r: a.remove_line(op.line) };
    case "lineCars":
      return { r: a.set_line_cars(op.line, Math.max(0, Math.min(255, Math.round(op.cars)))) };
    case "fares":
      fares = { base: Math.max(0, op.base), perKm: Math.max(0, op.perKm) };
      return { r: 0 };
    case "link":
      return { r: op.on ? a.link_stations(op.a, op.b) : a.unlink_stations(op.a, op.b) };
    case "undo":
      return { r: a.undo() };
    case "redo":
      return { r: a.redo() };
  }
}

/** `apply`'s answer for an insertStop with nowhere to go (the track model's own codes are >= 0) */
const INSERT_REFUSED = -1;

function issues(): string[] {
  return api!.issue_kinds().split("\n").filter(Boolean);
}

/** The "doubleUp" query's answer (protocol.ts). */
function doubleUpAnswer(edge: number): Float64Array {
  const price = api!.preview_double_up(edge);
  if (price >= 0) return new Float64Array([price]);
  const kinds = api!.issue_kinds();
  return new Float64Array([-1, ...Array.from(kinds, (c) => c.charCodeAt(0))]);
}

// ---------------------------------------------------------------- messages

let ready: Promise<void> | null = null;

/** the zones whose masks the view fetches ahead (T-302) */
let viewZones: string[] = [];
function viewMasks(): Promise<void> {
  return api && viewZones.length ? ensureMasks(viewZones) : Promise.resolve();
}

let messages = Promise.resolve();
onmessage = (e: MessageEvent<ToClock>) => {
  const msg = e.data;
  if (msg.kind === "view") {
    // outside the queue, so the fetch starts at once; a mask is installed between messages
    const midLat = (msg.s + msg.n) / 2, wide = (msg.e - msg.w) * 111_320 * Math.cos(midLat * Math.PI / 180) > VIEW_SPAN_M;
    const cx = (msg.w + msg.e) / 2;
    viewZones = (wide ? zonesTouching(cx, midLat, cx, midLat, VIEW_MARGIN_M) : zonesTouching(msg.w, msg.s, msg.e, msg.n, VIEW_MARGIN_M)).map((c) => c.id);
    for (const id of viewZones) if (!masksIn.has(id)) void fetchWater(id).catch(() => {});
    void viewMasks();
    return;
  }
  if (msg.kind === "shade") {
    thumbShade = msg.shade;
    return;
  }
  messages =messages.then(() => processMessage(msg)).catch(err => send({ kind: "error", message: String(err) }));
};
async function processMessage(msg: ToClock) {
  if (msg.kind === "init") {
    // World identity and coordinates never follow the map's focused city.
    city = HOME_CITY;
    packBase = msg.pack;
    ready = (async () => {
      const t0 = performance.now();
      await init();
      api = new TrackApi(NEW_GAME_FRAME.lon, NEW_GAME_FRAME.lat, true);
      // T-309: each region's local time, kept across loads
      api.set_time_zones(timeZoneTable());
      api.set_empty_frame(frameKindNumber(NEW_GAME_FRAME.kind));
      syncFrame();
      api.clear_water();
      api.set_cash(START_CASH);
      emptyTrack = api.save();
      moneyParams = [...api.money_params()];
      void viewMasks(); // the view sent before the track model was up
      send({ kind: "ready", initMs: performance.now() - t0, frame: frameSpec() });
      // The last game, unless the page asked for a new one (T-029).
      testPage = !!(msg.fresh || msg.testPage);
      if (msg.backupMs && msg.backupMs > 0) backupMs = msg.backupMs;
      if (msg.fresh) {
        // T-320: a new game for a check owns no slot, so nothing it does can reach a saved game
        activeSlot = "";
        activeLease = null;
      } else if (msg.newSlot) {
        activeSlot = "slot-" + crypto.randomUUID();
        newSlotName = { key: activeSlot, name: msg.newSlot };
        activeLease = await leaseSlot(activeSlot);
      } else {
        activeSlot = msg.slot?.startsWith("slot-") ? msg.slot : ORIGINAL_SLOT;
        if (activeSlot !== ORIGINAL_SLOT && !await readStored(activeSlot)) activeSlot = ORIGINAL_SLOT;
        // a stress page reads the slot but does not take it, so it never writes it
        activeLease = msg.testPage ? null : await leaseSlot(activeSlot);
      }
      await publishSlots();
      const stored = msg.fresh || msg.newSlot ? null : await readStored(activeSlot);
      if (!stored && activeSlot) unwritten = activeSlot;
      if (msg.fresh || msg.newSlot) {
        // T-728, T-731: the start menu's choice; a real network is written to its slot at once
        if (msg.game?.real) {
          try {
            await newGame(msg.game);
            dirty = true;
            void autosave();
            return;
          } catch (err) {
            send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The real network could not be loaded. " + (err as Error).message, frame: frameSpec(), game });
          }
        }
        const empty = msg.game && { ...msg.game, real: undefined };
        // a start on another clock than New York's starts again at its own morning
        if (startClock(empty) !== START_CLOCK) { await newGame(empty); return; }
        setNewGame(empty);
      }
      if (stored) {
        try {
          if (await loadGame(stored.bytes, "autosave") && activeLease) await saveConverted(activeSlot, stored);
          return;
        } catch (err) {
          // Keep the unreadable save aside, and its slot as it is: the new game gets a slot of its own (T-743).
          await writeStored("unreadable-" + stored.at, stored);
          if (activeLease) {
            activeLease.release();
            activeSlot = "slot-" + crypto.randomUUID();
            newSlotName = { key: activeSlot, name: nextGameName(await listSlots().catch(() => [])) };
            unwritten = activeSlot;
            activeLease = await leaseSlot(activeSlot);
            await publishSlots();
          }
          send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The last game could not be loaded. " + (err as Error).message, frame: frameSpec(), game });
        }
      }
      sendOpened();
      sendState();
      send({ kind: "snapshot", snapshot: buildSnapshot() });
    })();
    setInterval(tick, 1000);
    setInterval(liveMoney, LIVE_MONEY_MS);
    return;
  }
  await ready;
  if (!api) return;
  try {
    if (msg.kind === "saveSlot") return await manageSlot(msg);
    if (msg.kind === "import" || msg.kind === "export" || msg.kind === "newGame") return await whole(msg);
    await handle(msg);
  } catch (err) {
    // A Rust panic: report it rather than leave the main thread waiting on an answer.
    send({ kind: "error", message: String(err) });
    if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: ["Internal"], charge: 0 });
    if (msg.kind === "previewStation") send({ kind: "stationCheck", seq: msg.seq, ok: false, issues: ["Internal"], cost: 0 });
    if (msg.kind === "previewCrossover") send({ kind: "crossoverCheck", seq: msg.seq, ok: false, issues: ["Internal"], cost: 0, parts: [] });
    if (msg.kind === "preview" || msg.kind === "previewEdge" || msg.kind === "previewNode" || msg.kind === "previewNodes") send({ kind: "preview", seq: msg.seq, ok: false, cost: 0, lengthM: 0, mult: 0, issues: ["Internal"], verts: new Float64Array(0), pts: new Float32Array(0) });
  }
};

/** Once a wall second: autosave when due. */
function tick() {
  if (!api) return;
  if (dirty && Date.now() - savedAt >= AUTOSAVE_MS) void autosave();
}

/** T-284: how often the cash is settled and sent while the clock runs, wall ms: 15 game seconds at
 * normal speed, 4 game minutes at the fastest. */
const LIVE_MONEY_MS = 250;
function liveMoney() {
  if (!api || !(clk.rate > 0)) return;
  if (settle()) sendMoney();
  else sendCash();
}

/** Messages that replace or copy the whole game. */
async function whole(msg: Extract<ToClock, { kind: "import" | "export" | "newGame" }>) {
  if (msg.kind === "export") {
    const bytes = await saveBytes();
    send({ kind: "exported", bytes, day: Math.floor(gameNow() / 86400) }, [bytes.buffer]);
  } else {
    try {
      const name = msg.name?.trim();
      if (msg.kind === "import") await intoNewSlot(name || "Loaded game", () => loadGame(new Uint8Array(msg.bytes), "file"), true);
      else {
        const spec = msg.game;
        await intoNewSlot(name || newGameName(await listSlots().catch(() => []), spec ?? { mode: "normal" }), () => newGame(spec), !!spec?.real);
      }
    } catch (err) {
      // "file": the main thread keeps the game it has
      send({ kind: "loaded", from: "file", clock: gameNow(), frame: frameSpec(), error: (err as Error).message, game });
    }
  }
}

/** Before an edit that makes new track: the masks its geometry needs (worked out as its preview). */
async function masksFor(op: EditOp) {
  const a = api!;
  if (op.op === "route") await withMasks(() => (a.set_route_tangent(!!op.through), a.preview_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : op.quad ? 4 : 2, false)), routePoints);
  else if (op.op === "edgePis") await withMasks(() => a.preview_edge(op.edge, pisArr(op.pis)), routePoints);
  else if (op.op === "moveNode") await withMasks(() => a.preview_move_node(op.node, op.x, op.y), nodePoints);
  else if (op.op === "moveNodes") await withMasks(() => a.preview_move_nodes(new Uint32Array(op.nodes), op.dx, op.dy), nodePoints);
}

async function handle(msg: ToClock) {
  if (!api) return;
  if (settle()) sendMoney();
  // an edit's new departures start now; trains already running keep their places (T-206)
  api.set_now(gameNow());
  if (msg.kind === "opened") {
    opened = [...msg.zones];
    dirty = true;
  } else if (msg.kind === "sandbox") {
    if (mode === "sandbox") return;
    setMode("sandbox");
    dirty = true;
    sendMoney();
    void autosave(); // the slot's card shows the mark at once
  } else if (msg.kind === "edit") {
    if (msg.op.op === "fares") {
      apply(msg.op);
      dirty = true;
      sendMoney();
      scheduleSnapshot();
      send({ kind: "edited", seq: msg.seq, ok: true, issues: [], charge: 0 });
      return;
    }
    if (msg.op.op === "link") {
      // T-197: a setting outside the undo history, like fares; the complexes change, so the
      // drawing (links in the state) and demand (the snapshot's complexes) both follow
      const { r } = apply(msg.op);
      if (r === 0) {
        dirty = true;
        sendState();
        scheduleSnapshot();
      }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : msg.op.on ? issues() : [], charge: 0 });
      return;
    }
    await masksFor(msg.op);
    const tA = performance.now();
    const { r, line, node } = apply(msg.op);
    const tB = performance.now();
    timing = {};
    const charge = r === 0 ? api.last_charge() : 0;
    // The new state first, so the world is up to date when the edit's answer arrives.
    if (r === 0) {
      dirty = true;
      if (charge > 0) {
        const d = dayEntry(Math.floor(moneyNow() / 86400));
        const e = hourEntry(Math.floor(moneyNow() / 3600));
        const trains = api.last_train_charge();
        d.trains += trains;
        d.build += charge - trains;
        e.trains += trains;
        e.build += charge - trains;
        for (const x of regionEntries("", d.day, e.hour)) { x.trains += trains; x.build += charge - trains; }
        heldCharges.set(msg.seq, { day: d.day, hour: e.hour, build: charge - trains, trains });
      }
      forget();
      const demand = api.dirty_demand_lines();
      sendState(msg.op.op === "setTrainCount" || msg.op.op === "setHeadway");
      if (demand.length || msg.op.op === "undo" || msg.op.op === "redo") scheduleSnapshot();
    }
    send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : r === INSERT_REFUSED ? ["InsertStop"] : issues(), charge, line, node, ms: { apply: tB - tA, ...timing, total: performance.now() - tA } });
    // Paid for something: keep it (T-029, autosave on construct).
    if (charge > 0) void autosave();
  } else if (msg.kind === "clock") {
    if (msg.game === game) clk = { t: msg.t, rate: msg.rate, at: msg.at };
  } else if (msg.kind === "income") {
    incomePerHour = msg.perHour.map((p) => p.slice(0, PERIODS.length));
    lineIncome = new Map((msg.lines ?? []).map((l) => [l.line, l.perHour]));
    ridesPerHour = msg.rides.map((p) => p.slice(0, PERIODS.length));
    lineRides = new Map((msg.lines ?? []).map((l) => [l.line, l.rides]));
  } else if (msg.kind === "regionShares") {
    lineShares = new Map(msg.lines.map((l) => [l.line, { fare: l.fare, ride: l.ride, run: l.run }]));
  } else if (msg.kind === "points") {
    linePoints = new Map(msg.lines.map((l) => [l.line, l.regions]));
  } else if (msg.kind === "star") {
    if (!stars.has(msg.zone)) {
      stars.set(msg.zone, Math.floor(gameNow()));
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "lineNotes") {
    let added = false;
    for (const [id, v] of Object.entries(msg.notes)) if (!lineNotes.has(id)) { lineNotes.set(id, v); added = true; }
    if (added) {
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "chargeShares") {
    // T-531: move a paid edit's charge from "" to the regions it was spent in
    const c = heldCharges.get(msg.seq);
    heldCharges.delete(msg.seq);
    if (c) {
      const move = (shares: [string, number][], v: number, k: "build" | "trains") => {
        if (!v || !shares.length) return;
        for (const x of regionEntries("", c.day, c.hour)) x[k] -= v;
        for (const [z, s] of shares) for (const x of regionEntries(z, c.day, c.hour)) x[k] += v * s;
      };
      move(msg.build, c.build, "build");
      move(msg.trains, c.trains, "trains");
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "riders") {
    // a new demand result: the boarding account follows it (T-164), a few lines at a time, one
    // kind of day's riders a message (T-260)
    if (peakC[msg.day ?? 0]) peakC[msg.day ?? 0] = strengths(msg.packed.subarray(0, PERIODS.length));
    api.set_riders(msg.packed, msg.day ?? 0);
    stepBoarding();
  } else if (msg.kind === "transfers") {
    // T-554: when riders changing lines reach their next platform; the account is unchanged
    api.set_transfers(msg.packed);
    sendTrips();
  } else if (msg.kind === "flush") {
    void autosave();
  } else if (msg.kind === "query") {
    const q = msg.q;
    const data =
      q.what === "insertStop" ? api.insert_stop(q.line, q.station)
      : q.what === "stopTiers" ? api.stop_tiers(q.line)
      : q.what === "passedStations" ? api.passed_stations(q.line)
      : q.what === "junction" ? api.junction_info(q.node)
      : q.what === "edgePis" ? api.edge_pis(q.edge)
      : q.what === "flyover" ? new Float64Array([api.flyover_cost(q.node)])
      : q.what === "doubleUp" ? doubleUpAnswer(q.edge)
      : q.what === "lineAccount" ? api.line_account(q.line, q.day ?? 0)
      : q.what === "linePeriods" ? api.line_periods(q.line, q.day)
      : q.what === "today" ? today()
      : q.what === "lineHeld" ? api.line_held(q.line)
      : q.what === "lineMarks" ? api.line_marks(q.line)
      : api.line_delays(q.line, q.level);
    send({ kind: "answer", seq: msg.seq, data }, [data.buffer]);
  } else if (msg.kind === "previewStation") {
    const at = msg.at;
    const r = at.kind === "node" ? api.preview_station(1, at.node, 0, 0, msg.platform)
      : at.kind === "edge" ? api.preview_station(2, at.edge, at.x, at.y, msg.platform)
      : api.preview_free_station(freeEnds(at), at.quad ? 4 : 2, msg.platform);
    send({ kind: "stationCheck", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), cost: r === 0 ? api.station_cost() : 0 });
  } else if (msg.kind === "previewCrossover") {
    const out = api.preview_crossover(msg.edge, msg.x, msg.y);
    const ok = out[0] === 1;
    const parts: Float32Array[] = [];
    for (let k = 2; k < out.length; ) {
      const n = out[k++], pts = new Float32Array(n * 3);
      for (let j = 0; j < n; j++, k += 3) {
        const [x, y] = packLocal(out[k], out[k + 1]);
        pts.set([x, y, out[k + 2]], j * 3);
      }
      parts.push(pts);
    }
    send({ kind: "crossoverCheck", seq: msg.seq, ok, issues: ok ? [] : issues(), cost: out[1], parts }, parts.map((p) => p.buffer));
  } else if (msg.kind === "previewNode") {
    const a = api;
    sendNodePreview(msg.seq, await withMasks(() => a.preview_move_node(msg.node, msg.x, msg.y), nodePoints));
  } else if (msg.kind === "previewNodes") {
    const a = api;
    sendNodePreview(msg.seq, await withMasks(() => a.preview_move_nodes(new Uint32Array(msg.nodes), msg.dx, msg.dy), nodePoints));
  } else if (msg.kind === "preview" || msg.kind === "previewEdge") {
    const a = api;
    const out = await withMasks(() => msg.kind === "preview"
      ? (a.set_route_tangent(!!msg.through), a.preview_route(new Float64Array([...endArr(msg.from), ...endArr(msg.to)]), pisArr(msg.pis), msg.single ? 1 : msg.quad ? 4 : 2, false))
      : a.preview_edge(msg.edge, pisArr(msg.pis)), routePoints);
    let arc: Float64Array | undefined;
    const ra = msg.kind === "preview" ? a.route_arc() : null;
    if (ra?.length === 3) {
      const [x, y] = packLocal(ra[0], ra[1]);
      arc = new Float64Array([x, y, ra[2], 0]);
    }
    const ok = out[0] === 1, nv = out[3];
    const verts = new Float64Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      const [x, y] = packLocal(out[5 + i * 4], out[6 + i * 4]);
      verts.set([x, y, out[7 + i * 4], out[8 + i * 4]], i * 4);
    }
    const p0 = 5 + nv * 4;
    const pts = new Float32Array(out.length - p0);
    for (let k = p0; k < out.length; k += 3) {
      const [x, y] = packLocal(out[k], out[k + 1]);
      pts.set([x, y, out[k + 2]], k - p0);
    }
    // Explicit shape also permits type-checking before the track WASM declarations regenerate.
    const speeds = (api as TrackApi & { preview_render_speeds(): Float32Array }).preview_render_speeds();
    if (speeds.length !== Math.max(0, pts.length / 3 - 1)) throw new Error("Preview speed/point length mismatch");
    const marks = markPoints((api as TrackApi & { preview_marks(): Float64Array }).preview_marks(), 3);
    send({ kind: "preview", seq: msg.seq, ok, cost: out[1], lengthM: out[2], mult: out[4], issues: ok ? [] : issues(), verts, pts, speeds, arc, anchor: anchor().version, marks }, [verts.buffer, pts.buffer, speeds.buffer, marks.buffer]);
  } else if (msg.kind === "epoch") {
    if (msg.epoch !== epoch) {
      epoch = msg.epoch;
      sendTrips();
    }
  } else if (msg.kind === "anchor") {
    // the view travelled far (T-307): every drawn position again, relative to the new anchor
    if (msg.version !== anchor().version) {
      setAnchor(makeAnchor(msg.lng, msg.lat, msg.version));
      forget(true);
      sendState();
    }
  }
}

/** A node drag's preview (`TrackApi.preview_move_node`, T-093): every edge at the node, its
 * vertices one after another and its polyline as one part of `pts`. */
function sendNodePreview(seq: number, out: Float64Array) {
  const ok = out[0] === 1, n = out[2];
  const verts: number[] = [], pts: number[] = [], parts: number[] = [0];
  let i = 3;
  for (let e = 0; e < n; e++) {
    const nv = out[i++];
    for (let k = 0; k < nv; k++, i += 4) verts.push(...packLocal(out[i], out[i + 1]), out[i + 2], out[i + 3]);
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) pts.push(...packLocal(out[i], out[i + 1]), out[i + 2]);
    parts.push(pts.length / 3);
  }
  const v = new Float64Array(verts), p = new Float32Array(pts), pa = new Uint32Array(parts);
  const marks = markPoints((api as TrackApi & { preview_marks(): Float64Array }).preview_marks(), 3);
  send({ kind: "preview", seq, ok, cost: out[1], lengthM: 0, mult: 0, issues: ok ? [] : issues(), verts: v, pts: p, parts: pa, anchor: anchor().version, marks }, [v.buffer, p.buffer, pa.buffer, marks.buffer]);
}
