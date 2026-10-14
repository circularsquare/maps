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
import { CITIES, cityNear, HOME_CITY, NEW_GAME_FRAME, WORLD_ID } from "../game/cities";
import { frameKindName, frameKindNumber, makeFrame, mercK, type Frame, type FrameSpec } from "../game/frame";
import { PERIODS, periodAt } from "../game/periods";
import { DAY_KINDS, dayKindAt } from "../game/days";
import { networkFromState } from "../map/network";
import { buildComplexes } from "../game/complexes";
import { decodeSave, encodeSave, isNortheastSave, readStored, SAVE_FORMAT, START_CLOCK, writeStored, listSlots, writeSlot, renameSlot, deleteSlot, leaseSlot, ORIGINAL_SLOT, type SlotLease, type SlotSummary, type DayLedger, type SaveHeader } from "./saveFile";
import type { ClockEdge, ClockLine, ClockNode, ClockState, EditOp, FromClock, MoneyState, NetworkSnapshot, PiInput, RouteEnd, ToClock } from "./protocol";

/** Run 0 of each line is resampled this often (m) for the renderer's line texture. */
const SAMPLE_STEP = 50;
const SNAPSHOT_DEBOUNCE_MS = 500;
/** SPEC 6.4: US$6B to start, in US$M. */
const START_CASH = 6000;
/** Whole trains on a new line across both directions, high/medium/low demand. */
const NEW_LINE_TRAINS: [number, number, number] = [0, 0, 0];
/** Default fare curve (T-028): US$ a ride plus US$ a km. */
const START_FARES = { base: 1.5, perKm: 0.1 };
/** Autosave (T-029): this long after the last one if anything changed, wall ms; at once after a
 * construct or other paid edit, and when the page is hidden. */
const AUTOSAVE_MS = 60_000;
/** Days of money kept in the ledger. */
const LEDGER_DAYS = 30;
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
/** fare income per game hour in each period, US$M (from the main thread's demand results), per
 * kind of day (T-260, game/days.ts) */
let incomePerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
/** the same per line id (T-293), for each line's fares taken today */
let lineIncome = new Map<number, number[][]>();
let clk ={ t: START_CLOCK, rate: 0, at: Date.now() };
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
let storageReady = true;
let saveFailure: string | undefined;
const slotChanges = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel("anitabuilder-save-slots");
slotChanges?.addEventListener("message", () => { if (api) void publishSlots(); });

async function publishSlots(error?: string) {
  let slots: Awaited<ReturnType<typeof listSlots>> = [];
  try { slots = await listSlots(); storageReady = true; } catch { storageReady = false; }
  const writable = !!activeLease && storageReady && !saveFailure;
  const reason = saveFailure || (!storageReady ? "Browser storage is unavailable. Save to a file to keep this game."
    : !activeLease ? navigator.locks ? "Autosave is off here because this slot was opened in another tab. Save to a new slot, or close the other tab and reload."
      : "This browser cannot protect save slots. Save to a file to keep this game." : undefined);
  send({ kind: "saveSlots", slots, active: activeSlot, writable, at: slots.find(s => s.key === activeSlot)?.at ?? 0, reason, error });
}

async function flushBeforeSwitch() {
  if (dirty && !activeLease) throw new Error("Save this game to a new slot before opening another slot. You can also download a file and reload this page.");
  if (!await autosave()) throw new Error("This game could not be saved. Save to a file before opening another slot.");
}
/** the empty network's save, for a new game (keeps the water mask) */
let emptyTrack: Uint8Array | null = null;
let moneyParams = [2.5, 6, 100];

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

async function loadWater(packUrl: string): Promise<{ ok: boolean; population: number; jobs: number }> {
  const head = await (await fetch(packUrl + ".json")).json();
  const pop = head.totals?.pop ?? 0, jobs = head.totals?.jobs ?? 0;
  const row = head.arrays.find((a: any) => a.name === "water_row");
  const xs = head.arrays.find((a: any) => a.name === "water_x");
  if (!head.water || !row || !xs) return { ok: false, population: pop, jobs };
  const start = row.offset, end = xs.offset + xs.count * 2;
  const res = await fetch(packUrl + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` } });
  const buf = await res.arrayBuffer();
  const base = res.status === 206 ? start : 0;
  const rows = new Uint32Array(buf, row.offset - base, row.count);
  const runs = new Uint16Array(buf, xs.offset - base, xs.count);
  const w = head.water;
  // The raster is laid out in the pack's own frame; the track model puts it in the game's (T-125).
  api!.add_pack_water(w.cell_m, w.origin_m[0], w.origin_m[1], w.size[0], w.size[1], rows, runs, head.origin.lon, head.origin.lat);
  return { ok: true, population: pop, jobs };
}

// ---------------------------------------------------------------- state for the main thread

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

function buildState(): ClockState {
  const a = api!;
  const ei = a.edges_info();
  const edges: ClockEdge[] = [];
  const drawn: Float32Array[] = [];
  const off: number[] = [0];
  let total = 0;
  for (let i = 0; i < ei.length; i += 10) {
    const id = ei[i];
    edges.push({
      id, a: ei[i + 1], b: ei[i + 2], tracks: ei[i + 3] as 1 | 2, built: ei[i + 4] === 1, lengthM: ei[i + 5], cost: ei[i + 6],
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
      platform: ni[i + 7], stationBuilt: ni[i + 8] === 1, name: ni[i + 7] > 0 ? a.node_name(ni[i]) : "", heading: ni[i + 9],
    });
  }
  const li = a.lines_info();
  const lines: ClockLine[] = [];
  for (let i = 0; i < li.length; i += 22) {
    const id = li[i];
    const ok = li[i + 2] === 1, running = li[i + 3] === 1;
    const t3 = (k: number) => [li[i + k], li[i + k + 1], li[i + k + 2]] as [number, number, number];
    lines.push({
      id, name: a.line_name(id), letter: a.line_symbol(id), colour: hex(li[i + 1]), ok, running, broken: li[i + 4] === 1, tph: t3(5), cars: li[i + 21] || li[i + 8],
      lengthM: li[i + 9], roundTrip: t3(10), trains: t3(13), delay: t3(16), dwell: li[i + 19], turnaround: li[i + 20],
      stops: [...a.line_stops(id)], path: [...a.line_path(id)],
      fromStart: [0, 1, 2].map((lev) => [...a.stop_times(id, 0, lev)]),
      stopS: [[...a.line_stop_s(id, 0)], [...a.line_stop_s(id, 1)]],
      buildCost: ok && !running ? a.line_build_cost(id) : 0,
      trainCost: ok && !running ? a.line_train_cost(id) : 0,
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
    for (let k = 0; k < m.length; k += 5) {
      const [x, y] = packLocal(m[k], m[k + 1]);
      out.push({ x, y, rho: m[k + 2], delayS: m[k + 3], kind: m[k + 4] });
    }
    return out;
  });
  return {
    version: ++version,
    markers,
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
  return clk.t + Math.max(0, (Date.now() - clk.at) / 1000) * clk.rate;
}

/** Today's ledger entry (game day `day`), made if missing; the ledger keeps LEDGER_DAYS days. */
function dayEntry(day: number): DayLedger {
  let d = ledger[ledger.length - 1];
  if (!d || d.day !== day) {
    d = { day, fares: 0, running: 0, trains: 0, build: 0 };
    ledger.push(d);
    if (ledger.length > LEDGER_DAYS) ledger.splice(0, ledger.length - LEDGER_DAYS);
  }
  return d;
}

function moneyState(): MoneyState {
  const a = api!;
  dayEntry(Math.floor(gameNow() / 86400));
  return {
    cash: a.cash(),
    fleet: a.fleet(),
    carsNeeded: a.cars_needed(),
    carPrice: moneyParams[0],
    runCostCarKm: moneyParams[1],
    economy: moneyParams[2],
    fares: { ...fares },
    runningPerHour: [0, 1, 2].map((l) => a.running_cost(l)) as [number, number, number],
    ledger: ledger.map((d) => ({ ...d })),
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
  const cash = api!.cash(), today = dayEntry(Math.floor(gameNow() / 86400));
  if (cash === sentCash && today.fares === sentFares) return;
  sentCash = cash;
  sentFares = today.fares;
  send({ kind: "cash", cash, today: { ...today } });
}

/** Fares in and running costs out up to now (T-284: to the game second, not the whole hour). An
 * hour's fare (the main thread's income for its period and kind of day) is paid as its trains
 * pick up riders (`hourShape`), its running cost evenly; by the hour's end both are paid in full,
 * so a day's totals are what whole-hour settling gave. Returns whether an hour ended. */
function settle(): boolean {
  if (!api) return false;
  const now = gameNow();
  if (now - settledT > 86400 * 366) { // a year at most in one go
    settledT = Math.floor((now - 86400 * 366) / 3600) * 3600;
    partHour = settledT / 3600;
    partFare = partCost = 0;
  }
  if (now <= settledT) return false;
  const running = [0, 1, 2].map((l) => api!.running_cost(l));
  let cash = api.cash(), ended = false;
  while (settledT < now) {
    const h = Math.floor(settledT / 3600);
    if (h !== partHour) { partHour = h; partFare = partCost = 0; }
    const t = h * 3600, end = Math.min(now, t + 3600), full = end >= t + 3600;
    const p = periodAt(t);
    // T-260: the hour's own kind of day; the weekday's until a weekend's has come
    const fare = (incomePerHour[dayKindAt(t)] ?? incomePerHour[0])?.[p] ?? 0, cost = running[LEVEL_OF[PERIODS[p].demand]];
    const wf = full ? 1 : shareAt(h, end - t), wc = full ? 1 : (end - t) / 3600;
    const df = Math.max(0, wf - partFare), dc = Math.max(0, wc - partCost);
    partFare = Math.max(partFare, wf);
    partCost = Math.max(partCost, wc);
    cash += fare * df - cost * dc;
    const d = dayEntry(Math.floor(t / 86400));
    d.fares += fare * df;
    d.running += cost * dc;
    settledT = full ? t + 3600 : end;
    if (full) { ended = true; hourShape.delete(h); }
  }
  api.set_cash(cash);
  dirty = true;
  return ended;
}

/** T-293: every line's day so far (`TrackApi.today`, its layout), then each listed line's fares
 * taken today, US$M, in the same order: its fare for each hour since midnight, the hour now by the
 * share already paid, as `settle` pays the network's. */
function today(): Float64Array {
  const a = api!;
  if (settle()) sendMoney();
  const now = Math.min(gameNow(), settledT);
  const v = a.today(now);
  const count = v[0];
  const h0 = Math.floor(now / 86400) * 24, hNow = Math.floor(now / 3600);
  // the share of each hour since midnight paid so far
  const shares: { t: number; w: number }[] = [];
  for (let h = h0; h <= hNow; h++) shares.push({ t: h * 3600, w: h < hNow ? 1 : h === partHour ? partFare : 0 });
  const fares: number[] = [];
  let i = 1;
  for (let k = 0; k < count; k++) {
    const id = v[i], n = v[i + 1], spans = v[i + 6];
    const per = lineIncome.get(id);
    let f = 0;
    if (per) for (const s of shares) f += ((per[dayKindAt(s.t)] ?? per[0])?.[periodAt(s.t)] ?? 0) * s.w;
    fares.push(f);
    i += 7 + 2 * spans + n;
  }
  const out = new Float64Array(v.length + fares.length);
  out.set(v);
  out.set(fares, v.length);
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
  const nt = b[0], nl = b[1], nq = b[2];
  const q0 = 4 + 2 * nt + nl, e0 = q0 + nq * 10;
  const bins = new Float64Array(3600);
  let total = 0;
  for (let q = 0; q < nq; q++) {
    const Q = q0 + q * 10, rate = b[Q + 4], first = b[Q + 8], n = b[Q + 9];
    for (let j = first + 1; j < first + n; j++) {
      const t = b[e0 + j * 2];
      if (!(t >= 0 && t < 3600)) continue;
      const tPrev = b[e0 + (j - 1) * 2], leftPrev = b[e0 + (j - 1) * 2 + 1];
      const w = Math.max(0, leftPrev + rate * Math.max(0, t - tPrev) - b[e0 + j * 2 + 1]);
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
  return { game: "anitabuilder", format: SAVE_FORMAT, city, region: WORLD_ID, cities: CITIES.map(c => c.id), clock: gameNow(), cash: api!.cash(), fleet: api!.fleet(), fares: { ...fares }, ledger: ledger.map((d) => ({ ...d })) };
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
    const c = cityNear(lng, lat);
    byCity.set(c, (byCity.get(c) ?? 0) + 1);
  }
  const cities = [...byCity].sort((x, y) => y[1] - x[1]).map(([c]) => c);
  const li = a.lines_info();
  const summary: SlotSummary = { cities, lines: li.length / 22, stations, cash: a.cash() };
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
    const stroke = (xy: ArrayLike<number>) => {
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
    ctx.strokeStyle = "#b9b4ad";
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
  if (!activeLease) return Promise.resolve(false);
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
      await writeSlot(key, { bytes, at, day, city, ...summary });
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
        await loadGame(stored.bytes, "file");
        activeLease?.release();
        activeSlot = key;
        activeLease = lease;
        saveFailure = undefined;
        dirty = false;
        savedAt = stored.at;
      } catch (err) { lease?.release(); throw err; }
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

/** Replace the game with a save's. Throws a player-readable reason, changing nothing. */
async function loadGame(bytes: Uint8Array, from: "autosave" | "file") {
  const { header: h, track } = await decodeSave(bytes);
  if (!isNortheastSave(h)) throw new Error("This save is for another region.");
  if (!api!.load(track)) throw new Error("This save is damaged.");
  api!.set_cash(h.cash);
  api!.set_fleet(h.fleet ?? 0);
  fares = { ...START_FARES, ...h.fares };
  ledger = (h.ledger ?? []).map((d) => ({ ...d }));
  restart(h.clock, from);
}

function newGame() {
  api!.load(emptyTrack!);
  api!.set_cash(START_CASH);
  api!.set_fleet(0);
  fares = { ...START_FARES };
  ledger = [];
  restart(START_CLOCK, "new");
}

/** After a load or a new game: the clock, fresh state, a new snapshot for demand. */
function restart(t: number, from: "autosave" | "file" | "new") {
  clk = { t, rate: clk.rate, at: Date.now() };
  // T-284: the loaded hour counts as paid up to the clock, evenly (its trains' timing is not known yet)
  settledT = t;
  partHour = Math.floor(t / 3600);
  partFare = partCost = (t - partHour * 3600) / 3600;
  hourShape.clear();
  incomePerHour = DAY_KINDS.map(() => PERIODS.map(() => 0)); // the main thread sends the new network's once demand lands
  api!.clear_riders(); // and its riders (T-164)
  forget(true);
  syncFrame();
  send({ kind: "loaded", from, clock: t, frame: frameSpec() });
  sendState();
  clearTimeout(snapTimer);
  send({ kind: "snapshot", snapshot: buildSnapshot() });
}

/** `service`: only the service changed (a train count, T-229): the renderer's buffers are not built
 * again, which was about half the edit's time on New York's real network. */
function sendState(service = false) {
  const t0 = performance.now();
  const s = buildState();
  const t1 = performance.now();
  if (service) {
    s.service = true;
    s.ms = [t1 - t0, 0];
    timing.state = t1 - t0;
    const r = s.render;
    send({ kind: "state", state: s }, [r.edgePts.buffer, r.edgeOff.buffer, r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer, ...(r.edgeSpeed ? [r.edgeSpeed.buffer] : [])] as Transferable[]);
    sendTrips();
    return;
  }
  // The renderer's buffers are built here too, so the main thread only uploads them (T-063).
  const n = networkFromState(s);
  s.net = n;
  s.ms = [t1 - t0, performance.now() - t1];
  timing.state = t1 - t0;
  timing.buffers = performance.now() - t1;
  const r = s.render;
  const bufs = new Set<ArrayBufferLike>([r.edgePts.buffer, r.edgeOff.buffer, r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
  if (r.edgeSpeed) bufs.add(r.edgeSpeed.buffer);
  for (const st of [n.track, n.lines]) for (const v of [st.seg, st.colour, st.edge, st.level, st.flags, st.dist, st.speed]) if (v) bufs.add(v.buffer);
  for (const v of [n.lineColours, n.lineCars, n.stations, n.stationAxis, n.stationShape, n.junctions]) if (v) bufs.add(v.buffer);
  send({ kind: "state", state: s }, [...bufs] as Transferable[]);
  sendTrips();
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
function pisArr(p: PiInput[]): Float64Array {
  return new Float64Array(p.flatMap((v) => [v.x, v.y, v.radius ?? 0, v.level]));
}
function colourInt(c: string) {
  return parseInt(c.slice(1), 16);
}

function apply(op: EditOp): { r: number; line?: number } {
  const a = api!;
  switch (op.op) {
    case "route":
      return { r: a.add_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : op.quad ? 4 : 2, false) };
    case "edgePis":
      return { r: a.set_edge_pis(op.edge, pisArr(op.pis)) };
    case "moveNode":
      return { r: a.move_node(op.node, op.x, op.y) };
    case "edgeLevel":
      return { r: a.set_edge_level(op.edge, op.level) };
    case "removeEdge":
      return { r: a.delete_edge(op.edge) };
    case "flying":
      return { r: a.set_flying(op.node, op.flying) };
    case "addStation":
      return op.at.kind === "node"
        ? { r: a.add_station(1, op.at.node, 0, 0, op.platform, op.name) }
        : { r: a.add_station(2, op.at.edge, op.at.x, op.at.y, op.platform, op.name) };
    case "stationProps":
      return { r: a.set_station_props(op.node, op.platform, op.name) };
    case "removeStation":
      return { r: a.remove_station(op.node) };
    case "construct":
      return { r: a.construct(new Uint32Array(op.edges), new Uint32Array(op.stations)) };
    case "constructAll":
      return { r: a.construct_all() };
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
      const base = op.name?.trim() || `${a.line_name(op.line)} ${op.express ? "express" : "copy"}`;
      let name = base, n = 2;
      while (existing.has(name)) name = `${base} ${n++}`;
      return { r: op.express ? a.express_copy(op.line, id, name) : a.copy_line(op.line, id, name), line: id };
    }
    case "setStops":
      return { r: a.set_stops(op.line, new Uint32Array(op.stops)) };
    case "setTrainCount":
      return { r: a.set_train_counts(op.line, ...op.trains) };
    case "lineLook":
      return { r: op.letter === undefined
        ? a.set_line_look(op.line, op.name, colourInt(op.colour))
        : a.set_line_badge(op.line, op.name, colourInt(op.colour), op.letter.trim()) };
    case "removeLine":
      return { r: a.remove_line(op.line) };
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

function issues(): string[] {
  return api!.issue_kinds().split("\n").filter(Boolean);
}

// ---------------------------------------------------------------- messages

let ready: Promise<void> | null = null;

let messages = Promise.resolve();
onmessage = (e: MessageEvent<ToClock>) => {
  const msg = e.data;
  messages = messages.then(() => processMessage(msg)).catch(err => send({ kind: "error", message: String(err) }));
};
async function processMessage(msg: ToClock) {
  if (msg.kind === "init") {
    // World identity and coordinates never follow the map's focused city.
    city = HOME_CITY;
    ready = (async () => {
      const t0 = performance.now();
      await init();
      api = new TrackApi(NEW_GAME_FRAME.lon, NEW_GAME_FRAME.lat, true);
      api.set_empty_frame(frameKindNumber(NEW_GAME_FRAME.kind));
      syncFrame();
      let water = { ok: false, population: 0, jobs: 0 };
      try {
        api.clear_water();
        const packs = msg.packs ?? CITIES.map(c => new URL(c.id, msg.pack).href);
        const loaded = await Promise.allSettled(packs.map(pack => loadWater(pack)));
        const results = loaded.flatMap(r => r.status === "fulfilled" ? [r.value] : []);
        water = { ok: loaded.length === results.length && results.every(r => r.ok), population: results[0]?.population ?? 0, jobs: results[0]?.jobs ?? 0 };
        const failure = loaded.find(r => r.status === "rejected");
        if (failure?.status === "rejected") send({ kind: "error", message: "Some regional water coverage could not be loaded: " + failure.reason });
      } catch (err) {
        send({ kind: "error", message: "city pack not loaded, building without the water mask: " + err });
      }
      api.set_cash(START_CASH);
      emptyTrack = api.save();
      moneyParams = [...api.money_params()];
      send({ kind: "ready", initMs: performance.now() - t0, frame: frameSpec(), water: water.ok, population: water.population, jobs: water.jobs });
      // The last game, unless the page asked for a new one (T-029).
      activeSlot = msg.slot?.startsWith("slot-") ? msg.slot : ORIGINAL_SLOT;
      if (activeSlot !== ORIGINAL_SLOT && !await readStored(activeSlot)) activeSlot = ORIGINAL_SLOT;
      activeLease = await leaseSlot(activeSlot);
      await publishSlots();
      const stored = msg.fresh ? null : await readStored(activeSlot);
      if (stored) {
        try {
          await loadGame(stored.bytes, "autosave");
          return;
        } catch (err) {
          // Keep the unreadable save aside rather than overwrite it with the new game.
          await writeStored("unreadable-" + stored.at, stored);
          send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The last game could not be loaded. " + (err as Error).message, frame: frameSpec() });
        }
      }
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
    handle(msg);
  } catch (err) {
    // A Rust panic: report it rather than leave the main thread waiting on an answer.
    send({ kind: "error", message: String(err) });
    if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: ["Internal"], charge: 0 });
    if (msg.kind === "previewStation") send({ kind: "stationCheck", seq: msg.seq, ok: false, issues: ["Internal"] });
    if (msg.kind === "preview" || msg.kind === "previewEdge" || msg.kind === "previewNode") send({ kind: "preview", seq: msg.seq, ok: false, cost: 0, lengthM: 0, mult: 0, issues: ["Internal"], verts: new Float64Array(0), pts: new Float32Array(0) });
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
  } else if (msg.kind === "import") {
    try {
      await loadGame(new Uint8Array(msg.bytes), "file");
    } catch (err) {
      send({ kind: "loaded", from: "file", clock: gameNow(), frame: frameSpec(), error: (err as Error).message });
      return;
    }
    dirty = true;
    await autosave();
  } else {
    newGame();
    dirty = true;
    await autosave();
  }
}

function handle(msg: ToClock) {
  if (!api) return;
  if (settle()) sendMoney();
  // an edit's new departures start now; trains already running keep their places (T-206)
  api.set_now(gameNow());
  if (msg.kind === "edit") {
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
    const tA = performance.now();
    const { r, line } = apply(msg.op);
    const tB = performance.now();
    timing = {};
    const charge = r === 0 ? api.last_charge() : 0;
    // The new state first, so the world is up to date when the edit's answer arrives.
    if (r === 0) {
      dirty = true;
      if (charge > 0) {
        const d = dayEntry(Math.floor(gameNow() / 86400));
        const trains = api.last_train_charge();
        d.trains += trains;
        d.build += charge - trains;
      }
      forget();
      const demand = api.dirty_demand_lines();
      sendState(msg.op.op === "setTrainCount");
      if (demand.length || msg.op.op === "undo" || msg.op.op === "redo") scheduleSnapshot();
    }
    send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), charge, line, ms: { apply: tB - tA, ...timing, total: performance.now() - tA } });
    // Paid for something: keep it (T-029, autosave on construct).
    if (charge > 0) void autosave();
  } else if (msg.kind === "clock") {
    clk = { t: msg.t, rate: msg.rate, at: msg.at };
  } else if (msg.kind === "income") {
    incomePerHour = msg.perHour.map((p) => p.slice(0, PERIODS.length));
    lineIncome = new Map((msg.lines ?? []).map((l) => [l.line, l.perHour]));
  } else if (msg.kind === "riders") {
    // a new demand result: the boarding account follows it (T-164), a few lines at a time, one
    // kind of day's riders a message (T-260)
    api.set_riders(msg.packed, msg.day ?? 0);
    stepBoarding();
  } else if (msg.kind === "flush") {
    void autosave();
  } else if (msg.kind === "query") {
    const q = msg.q;
    const data =
      q.what === "insertStop" ? api.insert_stop(q.line, q.station)
      : q.what === "stopTiers" ? api.stop_tiers(q.line)
      : q.what === "junction" ? api.junction_info(q.node)
      : q.what === "edgePis" ? api.edge_pis(q.edge)
      : q.what === "flyover" ? new Float64Array([api.flyover_cost(q.node)])
      : q.what === "lineAccount" ? api.line_account(q.line, q.day ?? 0)
      : q.what === "today" ? today()
      : api.line_delays(q.line, q.level);
    send({ kind: "answer", seq: msg.seq, data }, [data.buffer]);
  } else if (msg.kind === "previewStation") {
    const at = msg.at;
    const r = at.kind === "node" ? api.preview_station(1, at.node, 0, 0, msg.platform) : api.preview_station(2, at.edge, at.x, at.y, msg.platform);
    send({ kind: "stationCheck", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues() });
  } else if (msg.kind === "previewNode") {
    sendNodePreview(msg.seq, api.preview_move_node(msg.node, msg.x, msg.y));
  } else if (msg.kind === "preview" || msg.kind === "previewEdge") {
    const out =
      msg.kind === "preview"
        ? api.preview_route(new Float64Array([...endArr(msg.from), ...endArr(msg.to)]), pisArr(msg.pis), msg.single ? 1 : msg.quad ? 4 : 2, false)
        : api.preview_edge(msg.edge, pisArr(msg.pis));
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
    send({ kind: "preview", seq: msg.seq, ok, cost: out[1], lengthM: out[2], mult: out[4], issues: ok ? [] : issues(), verts, pts, speeds, anchor: anchor().version }, [verts.buffer, pts.buffer, speeds.buffer]);
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
  send({ kind: "preview", seq, ok, cost: out[1], lengthM: 0, mult: 0, issues: ok ? [] : issues(), verts: v, pts: p, parts: pa, anchor: anchor().version }, [v.buffer, p.buffer, pa.buffer]);
}
