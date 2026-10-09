// The clock worker (SPEC 6.5, 7; T-025): owns the network (the track model's `TrackApi`, T-040),
// applies edits, runs profiles and the capacity pass, and sends the main thread what it draws:
// track polylines, keyframes (shared phase tables) and each game hour's trips (the hourly
// re-base, SPEC 7). After a service change it also builds the network snapshot demand runs on,
// debounced 0.5 s (SPEC 6.5). notes/T-025.md.
//
// Every edit answers with an "edited" message (and, if it applied, a new "state"); the state
// holds the whole network. Fine for a city; per-tile updates are follow-up work (notes/T-025.md).

import init, { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { toLocal } from "../map/geo";
import { PERIODS, periodAt } from "../game/periods";
import { networkFromState } from "../map/network";
import { decodeSave, encodeSave, readStored, SAVE_FORMAT, START_CLOCK, writeStored, type DayLedger, type SaveHeader } from "./saveFile";
import type { ClockEdge, ClockLine, ClockNode, ClockState, EditOp, FromClock, MoneyState, NetworkSnapshot, PiInput, RouteEnd, ToClock } from "./protocol";

const R = 6371008.8;
/** Run 0 of each line is resampled this often (m) for the renderer's line texture. */
const SAMPLE_STEP = 50;
const SNAPSHOT_DEBOUNCE_MS = 500;
/** SPEC 6.4: US$6B to start, in US$M. */
const START_CASH = 6000;
/** Default schedule for a new line: trains an hour at high, medium, low demand. */
const NEW_LINE_TPH: [number, number, number] = [12, 8, 4];
/** Default fare curve (T-028): US$ a ride plus US$ a km. */
const START_FARES = { base: 1.5, perKm: 0.1 };
/** Autosave (T-029): this long after the last one if anything changed, wall ms; at once after a
 * construct or other paid edit, and when the page is hidden. */
const AUTOSAVE_MS = 60_000;
/** Days of money kept in the ledger. */
const LEDGER_DAYS = 30;
const LEVEL_OF = { high: 0, medium: 1, low: 2 } as const;

let api: TrackApi | null = null;
let city = "nyc";
let origin = { lon: -73.985, lat: 40.758 };
let version = 0;
let snapVersion = 0;
let epoch = NaN;
let snapTimer = 0;

// Money and the clock (T-028). The main thread's clock is the game's; it tells this worker each
// time it is set, paused or changes speed, and the worker settles fares and running costs one game
// hour at a time from that.
let fares = { ...START_FARES };
let ledger: DayLedger[] = [];
/** fare income per game hour in each period, US$M (from the main thread's demand results) */
let incomePerHour = PERIODS.map(() => 0);
let clk = { t: START_CLOCK, rate: 0, at: Date.now() };
/** game hours before this one are settled */
let settledHour = Math.floor(START_CLOCK / 3600);
/** something to autosave; when the last autosave was written */
let dirty = false;
let savedAt = Date.now();
let saving = false;
/** the empty network's save, for a new game (keeps the water mask) */
let emptyTrack: Uint8Array | null = null;
let moneyParams = [2.5, 6, 100];

function send(msg: FromClock, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/** pack metres -> lng/lat (the pack's equirectangular frame, notes/T-004.md) */
function packLngLat(x: number, y: number): [number, number] {
  const r = 180 / Math.PI;
  return [origin.lon + (x / (R * Math.cos(origin.lat / r))) * r, origin.lat + (y / R) * r];
}
function packLocal(x: number, y: number): [number, number] {
  const [lng, lat] = packLngLat(x, y);
  return toLocal(lng, lat);
}

async function loadWater(packUrl: string): Promise<{ ok: boolean; population: number; jobs: number }> {
  const head = await (await fetch(packUrl + ".json")).json();
  const pop = head.totals?.pop ?? 0, jobs = head.totals?.jobs ?? 0;
  origin = { lon: head.origin.lon, lat: head.origin.lat };
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
  api!.set_water(w.cell_m, w.origin_m[0], w.origin_m[1], w.size[0], w.size[1], rows, runs);
  return { ok: true, population: pop, jobs };
}

// ---------------------------------------------------------------- state for the main thread

function hex(c: number) {
  return "#" + c.toString(16).padStart(6, "0");
}

// What the renderer draws, per edge and per line, in local units, kept between edits: an edit
// re-renders only the edges and lines it dirtied (T-063). Cleared when a game is loaded.
const edgeDrawn = new Map<number, Float32Array>();
const lineDrawn = new Map<number, Float32Array>();
function forget(all = false) {
  if (all) {
    edgeDrawn.clear();
    lineDrawn.clear();
    return;
  }
  for (const e of api!.dirty_edges()) edgeDrawn.delete(e);
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
    }
    drawn.push(d);
    total += d.length;
    off.push(total / 3);
  }
  const pts = new Float32Array(total);
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
      id, name: a.line_name(id), colour: hex(li[i + 1]), ok, running, broken: li[i + 4] === 1, tph: t3(5), cars: li[i + 21] || li[i + 8],
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
    canUndo: a.can_undo(),
    canRedo: a.can_redo(),
    edges,
    nodes,
    lines,
    render: {
      edgePts: pts,
      edgeOff: new Uint32Array(off),
      phases: a.phases(),
      meta,
      samples,
      lineTable,
    },
    money: moneyState(),
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
  send({ kind: "money", money: moneyState() });
}

/** Fares in and running costs out for every whole game hour up to now. */
function settle(): boolean {
  if (!api) return false;
  const now = Math.floor(gameNow() / 3600);
  if (now - settledHour > 24 * 366) settledHour = now - 24 * 366; // a year at most in one go
  if (now <= settledHour) return false;
  const running = [0, 1, 2].map((l) => api!.running_cost(l));
  let cash = api.cash();
  for (; settledHour < now; settledHour++) {
    const t = settledHour * 3600;
    const p = periodAt(t);
    const fare = incomePerHour[p] ?? 0, cost = running[LEVEL_OF[PERIODS[p].demand]];
    cash += fare - cost;
    const d = dayEntry(Math.floor(t / 86400));
    d.fares += fare;
    d.running += cost;
  }
  api.set_cash(cash);
  dirty = true;
  return true;
}

// ---------------------------------------------------------------- saves (T-029)

function header(): SaveHeader {
  return { game: "anitabuilder", format: SAVE_FORMAT, city, clock: gameNow(), cash: api!.cash(), fleet: api!.fleet(), fares: { ...fares }, ledger: ledger.map((d) => ({ ...d })) };
}

/** The game as file bytes. The network and header are taken at once, before anything awaits. */
function saveBytes(): Promise<Uint8Array> {
  settle();
  return encodeSave(header(), api!.save());
}

async function autosave() {
  if (!api || saving || !dirty) return;
  saving = true;
  dirty = false;
  try {
    const day = Math.floor(gameNow() / 86400);
    const t0 = performance.now();
    const bytes = await saveBytes();
    const encodeMs = performance.now() - t0;
    savedAt = Date.now();
    if (await writeStored("autosave", { bytes, at: savedAt, day })) send({ kind: "saved", at: savedAt, bytes: bytes.length, encodeMs, totalMs: performance.now() - t0 });
  } catch (err) {
    send({ kind: "error", message: "autosave failed: " + err });
  } finally {
    saving = false;
  }
}

/** Replace the game with a save's. Throws a player-readable reason, changing nothing. */
async function loadGame(bytes: Uint8Array, from: "autosave" | "file") {
  const { header: h, track } = await decodeSave(bytes);
  if (h.city !== city) throw new Error("This save is for another city.");
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
  settledHour = Math.floor(t / 3600);
  incomePerHour = PERIODS.map(() => 0); // the main thread sends the new network's once demand lands
  forget(true);
  send({ kind: "loaded", from, clock: t });
  sendState();
  clearTimeout(snapTimer);
  send({ kind: "snapshot", snapshot: buildSnapshot() });
}

function sendState() {
  const t0 = performance.now();
  const s = buildState();
  const t1 = performance.now();
  // The renderer's buffers are built here too, so the main thread only uploads them (T-063).
  const n = networkFromState(s);
  s.net = n;
  s.ms = [t1 - t0, performance.now() - t1];
  const r = s.render;
  const bufs = new Set<ArrayBufferLike>([r.edgePts.buffer, r.edgeOff.buffer, r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
  for (const st of [n.track, n.lines]) for (const v of [st.seg, st.colour, st.edge, st.level, st.flags, st.dist, st.offset]) if (v) bufs.add(v.buffer);
  for (const v of [n.lineColours, n.stations, n.junctions, n.sampleOff]) if (v) bufs.add(v.buffer);
  send({ kind: "state", state: s }, [...bufs] as Transferable[]);
  sendTrips();
}

function sendTrips() {
  if (!api || !Number.isFinite(epoch)) return;
  for (const e of [epoch, epoch + 3600]) {
    const trips = api.trips(e);
    send({ kind: "trips", version, epoch: e, trips }, [trips.buffer]);
  }
}

// ---------------------------------------------------------------- the snapshot demand runs on

function buildSnapshot(): NetworkSnapshot {
  const a = api!;
  const ni = a.nodes_info();
  const stations: NetworkSnapshot["stations"] = [];
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] > 0 && ni[i + 8] === 1)
      stations.push({ id: ni[i], name: a.node_name(ni[i]), x: ni[i + 1], y: ni[i + 2], level: ni[i + 3] as any, platform: ni[i + 7] });
  }
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
      turnaroundS: li[i + 20], cars: li[i + 21],
    });
  }
  return { version: ++snapVersion, city, stations, lines, fare: { ...fares } };
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
      return { r: a.add_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : 2, false) };
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
      const [h, m, l] = NEW_LINE_TPH;
      return { r: a.set_line(id, new Uint32Array(op.stops), h, m, l, 30, 180, 0, colourInt(op.colour), op.name), line: id };
    }
    case "setStops":
      return { r: a.set_stops(op.line, new Uint32Array(op.stops)) };
    case "setFrequency":
      return { r: a.set_schedule(op.line, ...op.tph) };
    case "lineLook":
      return { r: a.set_line_look(op.line, op.name, colourInt(op.colour)) };
    case "removeLine":
      return { r: a.remove_line(op.line) };
    case "fares":
      fares = { base: Math.max(0, op.base), perKm: Math.max(0, op.perKm) };
      return { r: 0 };
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

onmessage = async (e: MessageEvent<ToClock>) => {
  const msg = e.data;
  if (msg.kind === "init") {
    city = msg.city;
    ready = (async () => {
      const t0 = performance.now();
      await init();
      api = new TrackApi(origin.lon, origin.lat, true);
      let water = { ok: false, population: 0, jobs: 0 };
      try {
        water = await loadWater(msg.pack);
        // The network's frame is the pack's: start again on its origin, then lay the mask.
        if (Math.abs(origin.lon + 73.985) > 1e-9 || Math.abs(origin.lat - 40.758) > 1e-9) {
          api.free();
          api = new TrackApi(origin.lon, origin.lat, true);
          await loadWater(msg.pack);
        }
      } catch (err) {
        send({ kind: "error", message: "city pack not loaded, building without the water mask: " + err });
      }
      api.set_cash(START_CASH);
      emptyTrack = api.save();
      moneyParams = [...api.money_params()];
      send({ kind: "ready", initMs: performance.now() - t0, origin, water: water.ok, population: water.population, jobs: water.jobs });
      // The last game, unless the page asked for a new one (T-029).
      const stored = msg.fresh ? null : await readStored("autosave");
      if (stored) {
        try {
          await loadGame(stored.bytes, "autosave");
          return;
        } catch (err) {
          // Keep the unreadable save aside rather than overwrite it with the new game.
          await writeStored("unreadable-" + stored.at, stored);
          send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The last game could not be loaded. " + (err as Error).message });
        }
      }
      sendState();
      send({ kind: "snapshot", snapshot: buildSnapshot() });
    })();
    setInterval(tick, 1000);
    return;
  }
  await ready;
  if (!api) return;
  try {
    if (msg.kind === "import" || msg.kind === "export" || msg.kind === "newGame") return await whole(msg);
    handle(msg);
  } catch (err) {
    // A Rust panic: report it rather than leave the main thread waiting on an answer.
    send({ kind: "error", message: String(err) });
    if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: ["Internal"], charge: 0 });
    if (msg.kind === "preview" || msg.kind === "previewEdge" || msg.kind === "previewNode") send({ kind: "preview", seq: msg.seq, ok: false, cost: 0, lengthM: 0, mult: 0, issues: ["Internal"], verts: new Float64Array(0), pts: new Float32Array(0) });
  }
};

/** Once a wall second: settle money while the clock runs, autosave when due. */
function tick() {
  if (!api) return;
  if (clk.rate > 0 && settle()) sendMoney();
  if (dirty && Date.now() - savedAt >= AUTOSAVE_MS) void autosave();
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
      send({ kind: "loaded", from: "file", clock: gameNow(), error: (err as Error).message });
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
  if (msg.kind === "edit") {
    if (msg.op.op === "fares") {
      apply(msg.op);
      dirty = true;
      sendMoney();
      send({ kind: "edited", seq: msg.seq, ok: true, issues: [], charge: 0 });
      return;
    }
    const { r, line } = apply(msg.op);
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
      sendState();
      if (demand.length || msg.op.op === "undo" || msg.op.op === "redo") scheduleSnapshot();
    }
    send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), charge, line });
    // Paid for something: keep it (T-029, autosave on construct).
    if (charge > 0) void autosave();
  } else if (msg.kind === "clock") {
    clk = { t: msg.t, rate: msg.rate, at: msg.at };
  } else if (msg.kind === "income") {
    incomePerHour = msg.perHour.slice(0, PERIODS.length);
  } else if (msg.kind === "flush") {
    void autosave();
  } else if (msg.kind === "query") {
    const q = msg.q;
    const data =
      q.what === "junction" ? api.junction_info(q.node)
      : q.what === "edgePis" ? api.edge_pis(q.edge)
      : q.what === "flyover" ? new Float64Array([api.flyover_cost(q.node)])
      : api.line_delays(q.line, q.level);
    send({ kind: "answer", seq: msg.seq, data }, [data.buffer]);
  } else if (msg.kind === "previewNode") {
    sendNodePreview(msg.seq, api.preview_move_node(msg.node, msg.x, msg.y));
  } else if (msg.kind === "preview" || msg.kind === "previewEdge") {
    const out =
      msg.kind === "preview"
        ? api.preview_route(new Float64Array([...endArr(msg.from), ...endArr(msg.to)]), pisArr(msg.pis), msg.single ? 1 : 2, false)
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
    send({ kind: "preview", seq: msg.seq, ok, cost: out[1], lengthM: out[2], mult: out[4], issues: ok ? [] : issues(), verts, pts }, [verts.buffer, pts.buffer]);
  } else if (msg.kind === "epoch") {
    if (msg.epoch !== epoch) {
      epoch = msg.epoch;
      sendTrips();
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
  send({ kind: "preview", seq, ok, cost: out[1], lengthM: 0, mult: 0, issues: ok ? [] : issues(), verts: v, pts: p, parts: pa }, [v.buffer, p.buffer, pa.buffer]);
}
