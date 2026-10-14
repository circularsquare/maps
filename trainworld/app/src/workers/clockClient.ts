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
import type { ClockState, EditOp, FromClock, NetworkSnapshot, PiInput, RouteEnd, ToClock } from "./protocol";

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
}

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
      id: String(n.id), num: n.id, name: n.name, lng: n.lng, lat: n.lat, x: n.x, y: n.y, level: n.level as any, length: n.platform, built: n.stationBuilt, heading: n.heading,
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
    };
  }
  const edges: TrackEdge[] = s.edges.map((e) => ({
    id: e.id, a: e.a, b: e.b, tracks: e.tracks, built: e.built, lengthM: e.lengthM, cost: e.cost * 1e6, vminKmh: e.vminKmh,
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
  private previewBusy = false;
  private previewNext: { msg: (seq: number) => ToClock; cb: (a: PreviewAnswer) => void } | null = null;
  private previewCb: ((a: PreviewAnswer) => void) | null = null;
  private city = { population: 0, jobs: 0 };
  private epoch = NaN;
  private sawSlots = false;
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

  constructor(cityId = HOME_CITY) {
    this.worker = new Worker(new URL("./clock.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<FromClock>) => this.receive(e.data);
    this.worker.onerror = (e) => (clockWorkerStatus.value = "failed to load: " + (e.message ?? ""));
    const fresh = new URLSearchParams(location.search).get("new") === "1";
    this.cityId = cityId;
    let slot = "autosave";
    try { slot = localStorage.getItem("anitabuilder-save-slot") || slot; } catch { /* storage unavailable */ }
    this.post({ kind: "init", city: cityId, pack: new URL("packs/" + cityId, document.baseURI).href, fresh, slot });
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

  post(msg: ToClock, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  private receive(msg: FromClock) {
    switch (msg.kind) {
      case "ready":
        clockWorkerStatus.value = `ready in ${msg.initMs.toFixed(0)} ms${msg.water ? "" : ", no water mask"}`;
        setWorldFrame(msg.frame);
        this.city = { population: msg.population, jobs: msg.jobs };
        break;
      case "error":
        console.warn("clock worker:", msg.message);
        break;
      case "state": {
        const t0 = performance.now();
        setMoneyState(msg.state.money);
        world.value = toWorld(msg.state, this.city);
        this.onState(msg.state);
        this.lastStateMs = performance.now() - t0;
        break;
      }
      case "money":
        setMoneyState(msg.money);
        break;
      case "cash": // T-284: fares landing, running costs draining
        liveMoney.value = { cash: msg.cash, today: msg.today };
        break;
      case "loaded":
        // the game now in the worker may be in the other frame (T-125); first, before anything reads positions
        setWorldFrame(msg.frame);
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
        if (!this.sawSlots || before.active !== msg.active) {
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
        f?.({ ok: msg.ok, issues: msg.issues, charge: msg.charge, line: msg.line });
        break;
      }
      case "preview": {
        this.previewBusy = false;
        this.previewCb?.({ ok: msg.ok, cost: msg.cost, lengthM: msg.lengthM, mult: msg.mult, issues: msg.issues, verts: msg.verts, pts: msg.pts, speeds: msg.speeds, parts: msg.parts });
        const n = this.previewNext;
        this.previewNext = null;
        if (n) this.ask(n.msg, n.cb);
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
  preview(req: { from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean }, cb: (a: PreviewAnswer) => void) {
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
