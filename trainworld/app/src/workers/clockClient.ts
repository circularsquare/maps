// Main-thread side of the clock worker (T-025). Never waits on it: requests go out, answers
// arrive as messages. Each "state" becomes the `world` snapshot (game/world.ts) and the
// renderer's buffers; "trips" go to the renderer; network snapshots go to whoever registered
// with `onNetworkSnapshot` (the demand pool, T-026).

import { effect, signal } from "@preact/signals";
import { clock, SPEEDS } from "../game/clock";
import { HOME_CITY, setWorldFrame } from "../game/cities";
import { holdForGameLoad } from "../game/loading";
import { incomeByDay, lastSaved, liveMoney, setMoneyState } from "../game/money";
import { demandByDay } from "../game/demand";
import { accountVersion, packRiders } from "../game/boarding";
import { lineDraft, selection, tool } from "../game/ui";
import { say, setEditSink, world } from "../game/world";
import type { LineInput, LineStats, StationInput, TrackEdge, TrackNode, WorldView } from "../game/types";
import type { ClockState, EditOp, FromClock, NetworkSnapshot, PiInput, RouteEnd, StationAt, ToClock } from "./protocol";
import { anchorVersion } from "../map/anchorVersion";
import { anchor, anchorWanted, EARTH_C, HOME_ANCHOR, makeAnchor, setAnchor, type Anchor } from "../map/geo";

export const saveSlotsState = signal<{ slots: import("./saveFile").SaveSlot[]; active: string; writable: boolean; at: number; reason?: string; error?: string }>({ slots: [], active: "autosave", writable: true, at: 0 });
export function manageSaveSlot(action: "refresh" | "create" | "open" | "rename" | "delete", key?: string, name?: string) {
  current?.post({ kind: "saveSlot", action, key, name });
}

export const clockWorkerStatus = signal("loading");
/** The newest network snapshot (what demand runs on); null until the worker has started. */
export const networkSnapshot = signal<NetworkSnapshot | null>(null);

export interface EditAnswer {
  ok: boolean;
  issues: string[];
  /** US$M paid */
  charge: number;
  /** the new line's id, for addLine */
  line?: number;
  /** a route's end node, for drawing on from it (T-418) */
  node?: number;
}

export interface PreviewAnswer {
  ok: boolean;
  cost: number;
  lengthM: number;
  /** cost multiplier: the track's price over its length at the base price (T-085) */
  mult: number;
  issues: string[];
  verts: Float64Array;
  pts: Float32Array;
  speeds?: Float32Array;
  /** where each alignment starts in `pts` (triples) when there are several (a node drag) */
  parts?: Uint32Array;
  /** a tangent route's circle (T-418): centre x, y (local units), ground radius m, 0 */
  arc?: Float64Array;
}

/** A crossover preview (T-338): its cost US$M or why not, and its diagonals' centre lines. */
export type CrossoverAnswer = Pick<Extract<FromClock, { kind: "crossoverCheck" }>, "ok" | "issues" | "cost" | "parts">;

const snapshotListeners: ((s: NetworkSnapshot) => void)[] = [];
/** Call `f` with every network snapshot from now on (and the newest one, if there is one). */
export function onNetworkSnapshot(f: (s: NetworkSnapshot) => void) {
  snapshotListeners.push(f);
  if (networkSnapshot.peek()) f(networkSnapshot.peek()!);
}


function letterOf(name: string): string {
  const m = /^Line (\S{1,2})$/.exec(name);
  if (m) return m[1];
  const c = name.trim()[0] ?? "?";
  return c.toUpperCase();
}

/** Flat node pairs (the clock worker's `links`) as station id pairs (T-197). */
function pairsOf(flat: number[]): [string, string][] {
  const out: [string, string][] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) out.push([String(flat[i]), String(flat[i + 1])]);
  return out;
}

/** The UI's world view from the worker's state. */
function toWorld(s: ClockState, city: { population: number; jobs: number }): WorldView {
  const stations: StationInput[] = s.nodes
    .filter((n) => n.platform > 0)
    .map((n) => ({
      id: String(n.id), num: n.id, name: n.name, lng: n.lng, lat: n.lat, x: n.x, y: n.y, level: n.level as any, length: n.platform, built: n.stationBuilt, off: n.stationOff, heading: n.heading,
    }));
  const lines: LineInput[] = s.lines.map((l) => ({
    id: String(l.id), num: l.id, letter: l.letter || letterOf(l.name), symbol: l.letter, name: l.name, colour: l.colour, stops: l.stops.map(String),
    tph: { high: l.tph[0], medium: l.tph[1], low: l.tph[2] },
  }));
  const lineStats: Record<string, LineStats> = {};
  const edgeLines = new Map<number, string[]>();
  for (const l of s.lines) {
    for (const p of new Set(l.path.map((p) => p >> 1))) {
      const v = edgeLines.get(p) ?? [];
      v.push(String(l.id));
      edgeLines.set(p, v);
    }
    lineStats[String(l.id)] = {
      status: l.running ? "running" : l.ok || (!l.broken && l.stops.length < 2) ? "planned" : "broken",
      km: l.lengthM / 1000,
      fromStartS: l.fromStart[0] ?? [],
      fromStartByLevel: l.fromStart,
      roundTripS: l.roundTrip[0],
      roundTripByLevel: l.roundTrip,
      trainsNeeded: l.trains,
      delayS: l.delay,
      cars: l.cars,
      buildCost: l.buildCost * 1e6,
      trainCost: l.trainCost * 1e6,
      stopS: l.stopS,
      dwellS: l.dwell,
      turnaroundS: l.turnaround,
      shift: l.shift,
    };
  }
  const edges: TrackEdge[] = s.edges.map((e) => ({
    id: e.id, a: e.a, b: e.b, tracks: e.tracks, built: e.built, off: e.off, crossover: e.crossover, partner: e.partner, lengthM: e.lengthM, cost: e.cost * 1e6, vminKmh: e.vminKmh,
    levelMin: e.levelMin as any, levelMax: e.levelMax as any, lines: edgeLines.get(e.id) ?? [],
  }));
  const nodes = new Map<number, TrackNode>(
    s.nodes.map((n) => [n.id, { id: n.id, x: n.x, y: n.y, lng: n.lng, lat: n.lat, level: n.level as any, ports: n.ports, builtPorts: n.builtPorts, flying: n.flying, platform: n.platform }]),
  );
  return {
    version: s.version,
    capacity: s.markers,
    save: { version: 0, city: HOME_CITY, clock: clock.now(), cash: s.cash * 1e6, stations, lines, links: pairsOf(s.links ?? []) },
    lineStats,
    edges,
    nodes,
    city: { name: "New York", population: city.population, jobs: city.jobs, split: { train: 0, walk: 0, drive: 100 }, trainShareByPeriod: [] },
    money: {
      builtValue: s.builtCost * 1e6, blueprintCost: s.blueprintCost * 1e6,
      blueprintItems: Array.from({ length: s.blueprintItems.length / 7 }, (_, i) => {
        const r = s.blueprintItems.slice(i * 7, i * 7 + 7);
        return { kind: r[0], level: r[1] as any, tracks: r[2], wet: !!r[3], ramp: !!r[4], quantity: r[5], cost: r[6] * 1e6 };
      }),
    },
    canUndo: s.canUndo,
    canRedo: s.canRedo,
  };
}

let current: ClockClient | null = null;

/** Download the game as a file (T-029). */
export function exportGame() {
  current?.post({ kind: "export" });
}
/** Load a save file the player picked. */
export async function importGame(file: File) {
  const bytes = await file.arrayBuffer();
  current?.post({ kind: "import", bytes }, [bytes]);
}
/** Start again in an empty city. */
export function newGame() {
  current?.post({ kind: "newGame" });
}
/** Ask the clock worker an inspector question (T-031); null before the worker is there. */
export function queryClock(q: Extract<ToClock, { kind: "query" }>["q"]): Promise<Float64Array | null> {
  return current ? current.query(q) : Promise.resolve(null);
}

export class ClockClient {
  private worker: Worker;
  private seq = 0;
  private pending = new Map<number, (a: EditAnswer) => void>();
  private stationBusy = false;
  private stationNext: { at: StationAt; platform: number; cb: (ok: boolean, issues: string[]) => void } | null = null;
  private stationCb: ((ok: boolean, issues: string[]) => void) | null = null;
  private crossBusy = false;
  private crossNext: { edge: number; x: number; y: number; cb: (a: CrossoverAnswer) => void } | null = null;
  private crossCb: ((a: CrossoverAnswer) => void) | null = null;
  private previewBusy = false;
  private previewNext: { msg: (seq: number) => ToClock; cb: (a: PreviewAnswer) => void } | null = null;
  private previewCb: ((a: PreviewAnswer) => void) | null = null;
  private city = { population: 0, jobs: 0 };
  private epoch = NaN;
  private sawSlots = false;
  /** the last game load: where it came from and whether its save was converted to the world frame
   * (T-308; read by `?debug=1` checks, the player is told nothing) */
  lastLoad: { from: "autosave" | "file" | "new"; converted: boolean } | null = null;
  /** the renderer's hooks, set by main.ts */
  onState: (s: ClockState) => void = () => {};
  /** `board`: the boarding account's hour (T-164, game/boarding.ts), once demand has riders */
  onTrips: (epoch: number, version: number, trips: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array) => void = () => {};
  /** the boarding account's last hours: ms in the worker for the hour, lines worked out again
   * before it and the ms that took in steps (all, longest), bytes; newest last, at most 40
   * (`?debug=1`, notes/T-164.md) */
  boardLog: { ms: number; redone: number; bytes: number; prepMs: number; prepMax: number }[] = [];
  /** the last edits' time in the worker by step, and the main thread's for the state they sent
   * (T-229, `?debug=1`); newest last, at most 40 */
  editLog: { worker?: Record<string, number>; stateMs?: number }[] = [];
  private lastStateMs = 0;
  /** Draw anchors (T-307, map/geo.ts) by version: the one asked for last, and the ones states and
   * previews may still arrive in. `anchor()` switches when the worker's state built at a new one
   * arrives, so the drawn network and every local-unit conversion change together. */
  private anchors = new Map<number, Anchor>([[HOME_ANCHOR.version, HOME_ANCHOR]]);
  private wantedAnchor: Anchor = HOME_ANCHOR;
  /** called after `anchor()` moved, before the state built at it is handed over (main.ts) */
  onAnchor: (a: Anchor) => void = () => {};

  /** The view moved (MapLibre's moveend): ask for a new draw anchor when it has travelled far. */
  followView(lng: number, lat: number, zoom: number) {
    const to = anchorWanted(lng, lat, zoom, this.wantedAnchor);
    if (!to) return;
    const a = makeAnchor(to[0], to[1], this.wantedAnchor.version + 1);
    this.anchors.set(a.version, a);
    for (const v of this.anchors.keys()) if (v < a.version - 8 && v !== anchor().version) this.anchors.delete(v);
    this.wantedAnchor = a;
    this.post({ kind: "anchor", lng: a.lng, lat: a.lat, version: a.version });
  }

  /** The view's bounds after a move (T-302): the worker fetches the water masks near it ahead. */
  followBounds(w: number, s: number, e: number, n: number) {
    this.post({ kind: "view", w, s, e, n });
  }

  /** The anchor of `version`, as the worker reported it. */
  private anchorOf(spec: { lng: number; lat: number; version: number }): Anchor {
    let a = this.anchors.get(spec.version);
    if (!a) this.anchors.set(spec.version, (a = makeAnchor(spec.lng, spec.lat, spec.version)));
    return a;
  }

  /** A preview's positions, relative to anchor `version`, moved to the current anchor's local
   * units in place (an anchor change and a preview crossed). */
  private toCurrentAnchor(version: number | undefined, verts: Float64Array, pts: Float32Array) {
    const from = version === undefined ? undefined : this.anchors.get(version), to = anchor();
    if (!from || from === to) return;
    const dx = (from.merc[0] - to.merc[0]) * EARTH_C, dy = (from.merc[1] - to.merc[1]) * EARTH_C;
    for (let i = 0; i < verts.length; i += 4) { verts[i] += dx; verts[i + 1] += dy; }
    for (let i = 0; i < pts.length; i += 3) { pts[i] += dx; pts[i + 1] += dy; }
  }

  constructor(cityId = HOME_CITY) {
    this.worker = new Worker(new URL("./clock.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<FromClock>) => this.receive(e.data);
    this.worker.onerror = (e) => (clockWorkerStatus.value = "failed to load: " + (e.message ?? ""));
    const q = new URLSearchParams(location.search);
    const fresh = q.get("new") === "1";
    // T-320: a check or stress page never writes over a saved game
    this.testPage = fresh || q.get("synth") === "1" || q.get("demandTest") === "1";
    this.cityId = cityId;
    let slot = "autosave";
    try { slot = localStorage.getItem("anitabuilder-save-slot") || slot; } catch { /* storage unavailable */ }
    this.post({ kind: "init", city: cityId, pack: new URL("packs/" + cityId, document.baseURI).href, fresh, testPage: this.testPage, slot });
    setEditSink((op) => this.edit(op));
    current = this;
    // The worker settles money from the clock (T-028, T-284): tell it each time the
    // clock is set, paused, resumed or sped up.
    const sendClock = () => this.post({ kind: "clock", t: clock.now(), rate: clock.ticking ? SPEEDS[clock.speed.peek()] : 0, at: Date.now() });
    clock.onChange(sendClock);
    sendClock();
    // Fare income from the riders (game/money.ts), whenever riders, fares or lines change.
    // One table per kind of day (T-260): the worker settles each hour by its own day's.
    let lastIncome = "";
    effect(() => {
      const days = incomeByDay.value;
      const per = days.map((i) => i.perHour);
      // T-293: each line's too, for its fares taken today
      const ids = new Set(days.flatMap((i) => [...i.lines.keys()]));
      const lines = [...ids].map((id) => ({ line: Number(id), perHour: days.map((i) => i.lines.get(id)?.perHour ?? (per[0] ?? []).map(() => 0)) }));
      const key = JSON.stringify([per, lines]);
      if (key !== lastIncome) {
        lastIncome = key;
        this.post({ kind: "income", perHour: per, lines });
      }
    });
    // Demand's riders for the boarding account (T-164): each new result, every kind of day's
    // (T-260); the trains and platforms of each hour read their own day's.
    effect(() => {
      demandByDay.value.forEach((d, day) => {
        const packed = d.view ? packRiders(d.view) : null;
        if (packed) this.post({ kind: "riders", packed, day }, [packed.buffer]);
      });
    });
    // Hidden or closing: write the autosave now (T-029).
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && this.post({ kind: "flush" }));
    addEventListener("pagehide", () => this.post({ kind: "flush" }));
  }

  private cityId: string;
  /** `?new=1`, `?synth=1` or `?demandTest=1`: no autosave until a copy is saved (T-320) */
  private testPage: boolean;

  post(msg: ToClock, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  private receive(msg: FromClock) {
    switch (msg.kind) {
      case "ready":
        clockWorkerStatus.value = `ready in ${msg.initMs.toFixed(0)} ms`;
        setWorldFrame(msg.frame);
        break;
      case "error":
        console.warn("clock worker:", msg.message);
        break;
      case "state": {
        const t0 = performance.now();
        // built at another draw anchor (T-307): switch to it first, so everything rebuilt from
        // this state converts with the anchor its positions are relative to
        const a = msg.state.anchor ? this.anchorOf(msg.state.anchor) : anchor();
        const moved = a !== anchor();
        if (moved) setAnchor(a);
        if (msg.state.net) msg.state.net.anchor = a;
        setMoneyState(msg.state.money);
        world.value = toWorld(msg.state, this.city);
        this.onState(msg.state);
        if (moved) {
          anchorVersion.value = a.version; // the demand views rebuild at it (T-317)
          this.onAnchor(a);
        }
        this.lastStateMs = performance.now() - t0;
        break;
      }
      case "money":
        setMoneyState(msg.money);
        break;
      case "cash": // T-284: fares landing, running costs draining
        liveMoney.value = { cash: msg.cash, today: msg.today, hour: msg.hour };
        break;
      case "loaded":
        // the game now in the worker may be in the other frame (T-125); first, before anything reads positions
        setWorldFrame(msg.frame);
        this.lastLoad = { from: msg.from, converted: !!msg.converted };
        if (msg.error) {
          say(msg.error, true);
          if (msg.from === "file") break;
        }
        // a different game: its clock waits for its demand (T-149); the snapshot follows
        holdForGameLoad();
        // Ids are renumbered by a load: nothing old may stay selected or half drawn.
        selection.value = null;
        tool.value = "select";
        lineDraft.value = { line: null, stops: [] };
        clock.set(msg.clock);
        if (msg.from === "file") say(`Game loaded, day ${Math.floor(msg.clock / 86400)}.`);
        if (msg.from === "new" && !msg.error) say("New game started.");
        break;
      case "exported": {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([msg.bytes as BlobPart], { type: "application/octet-stream" }));
        a.download = `anitabuilder-${this.cityId}-day-${msg.day}.save`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
        break;
      }
      case "saveSlots": {
        const before = saveSlotsState.value;
        saveSlotsState.value = { slots: msg.slots, active: msg.active, writable: msg.writable, at: msg.at, reason: msg.reason, error: msg.error };
        lastSaved.value = msg.at;
        // a test page does not make its slot the one the next page opens, until it changes (T-320)
        if (msg.active && ((!this.sawSlots && !this.testPage) || before.active !== msg.active)) {
          try { localStorage.setItem("anitabuilder-save-slot", msg.active); } catch { /* storage unavailable */ }
        }
        this.sawSlots = true;
        if (msg.error) say(msg.error, true);
        else if (!msg.writable && before.writable) say(msg.reason || "This slot is open in another tab. Autosave is off here.", true);
        break;
      }
      case "saved":
        lastSaved.value = msg.at;
        break;
      case "answer": {
        const f = this.queries.get(msg.seq);
        this.queries.delete(msg.seq);
        f?.(msg.data);
        break;
      }
      case "trips":
        if (msg.board) {
          this.boardLog.push({ ms: msg.boardMs ?? 0, redone: msg.redone ?? 0, bytes: msg.board.byteLength, prepMs: msg.prepMs ?? 0, prepMax: msg.prepMax ?? 0 });
          if (this.boardLog.length > 40) this.boardLog.shift();
          accountVersion.value++;
        }
        this.onTrips(msg.epoch, msg.version, msg.trips, msg.board, msg.ends, msg.nexts);
        break;
      case "edited": {
        const f = this.pending.get(msg.seq);
        this.pending.delete(msg.seq);
        this.editLog.push({ worker: msg.ms, stateMs: this.lastStateMs });
        if (this.editLog.length > 40) this.editLog.shift();
        f?.({ ok: msg.ok, issues: msg.issues, charge: msg.charge, line: msg.line, node: msg.node });
        break;
      }
      case "preview": {
        this.previewBusy = false;
        this.toCurrentAnchor(msg.anchor, msg.verts, msg.pts);
        if (msg.arc) this.toCurrentAnchor(msg.anchor, msg.arc, new Float32Array(0));
        this.previewCb?.({ ok: msg.ok, cost: msg.cost, lengthM: msg.lengthM, mult: msg.mult, issues: msg.issues, verts: msg.verts, pts: msg.pts, speeds: msg.speeds, parts: msg.parts, arc: msg.arc });
        const n = this.previewNext;
        this.previewNext = null;
        if (n) this.ask(n.msg, n.cb);
        break;
      }
      case "stationCheck": {
        this.stationBusy = false;
        this.stationCb?.(msg.ok, msg.issues);
        const n = this.stationNext;
        this.stationNext = null;
        if (n) this.checkStation(n.at, n.platform, n.cb);
        break;
      }
      case "crossoverCheck": {
        this.crossBusy = false;
        this.crossCb?.(msg);
        const n = this.crossNext;
        this.crossNext = null;
        if (n) this.checkCrossover(n.edge, n.x, n.y, n.cb);
        break;
      }
      case "snapshot":
        networkSnapshot.value = msg.snapshot;
        for (const f of snapshotListeners) f(msg.snapshot);
        break;
    }
  }

  private queries = new Map<number, (d: Float64Array) => void>();
  query(q: Extract<ToClock, { kind: "query" }>["q"]): Promise<Float64Array> {
    const seq = ++this.seq;
    this.post({ kind: "query", seq, q });
    return new Promise((res) => this.queries.set(seq, res));
  }

  /** Send an edit; the promise resolves with the worker's answer. */
  edit(op: EditOp): Promise<EditAnswer> {
    const seq = ++this.seq;
    this.post({ kind: "edit", seq, op });
    return new Promise((res) => this.pending.set(seq, res));
  }

  /** Ask what a route would cost. One request in flight; newer ones replace a waiting one. */
  preview(req: { from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean }, cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "preview", seq, ...req }), cb);
  }

  /** What giving blueprint track new PIs would do (T-064); shares the one-in-flight queue. */
  previewEdge(edge: number, pis: PiInput[], cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "previewEdge", seq, edge, pis }), cb);
  }

  /** What moving a blueprint node to pack metres (x, y) would do (T-093); the same queue. */
  previewNode(node: number, x: number, y: number, cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "previewNode", seq, node, x, y }), cb);
  }

  /** Whether a station could be placed here (T-311): the worker tries `addStation` and rolls it
   * back. One request in flight; a newer one replaces a waiting one. */
  checkStation(at: StationAt, platform: number, cb: (ok: boolean, issues: string[]) => void) {
    if (this.stationBusy) {
      this.stationNext = { at, platform, cb };
      return;
    }
    this.stationBusy = true;
    this.stationCb = cb;
    this.post({ kind: "previewStation", seq: ++this.seq, at, platform });
  }

  /** What a crossover on track `edge` near pack metres (x, y) would cost, or why not (T-338), with
   * its two diagonals to draw. One request in flight; a newer one replaces a waiting one. */
  checkCrossover(edge: number, x: number, y: number, cb: (a: CrossoverAnswer) => void) {
    if (this.crossBusy) {
      this.crossNext = { edge, x, y, cb };
      return;
    }
    this.crossBusy = true;
    this.crossCb = cb;
    this.post({ kind: "previewCrossover", seq: ++this.seq, edge, x, y });
  }

  /** Drop a preview waiting to be sent (a drag was cancelled or finished). */
  dropPreview() {
    this.previewNext = null;
  }

  private ask(msg: (seq: number) => ToClock, cb: (a: PreviewAnswer) => void) {
    if (this.previewBusy) {
      this.previewNext = { msg, cb };
      return;
    }
    this.previewBusy = true;
    this.previewCb = cb;
    this.post(msg(++this.seq));
  }

  /** The render epoch the renderer is drawing (start of the game hour). */
  setEpoch(epoch: number) {
    if (epoch === this.epoch) return;
    this.epoch = epoch;
    this.post({ kind: "epoch", epoch });
  }
}
