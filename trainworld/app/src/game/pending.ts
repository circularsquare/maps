// T-4145: the edits waiting to show (notes/T-4145.md). Each is drawn and shown from the click, from what the page can
// tell (`ahead`), and armed by the worker's word for it (workers/clockWord.ts), whose parts replace the guess's. Message
// order says what a state holds: every state the worker sends after an edit's word was built after that edit was applied
// (its follow-ups start again behind a new edit, and a stop click's rows go before the next edit's apply). So an armed
// entry's map part goes with the first network taken that arrived after its word, its panel part with the first rows or
// state, its cost with the first money, cash or state. A refused edit's entry goes in the frame its answer lands.

import { signal } from "@preact/signals";
import type { EdgeLook, EditOp, EditWord, StationBox } from "../workers/protocol";
import type { NetworkBuffers, Strokes } from "../map/network";
import type { RawStations } from "../map/lineAhead";

/** What an edit shows ahead of the worker: every part optional (notes/T-4145.md). */
export interface Show {
  /** a rail line's strokes draw these edges (null: the line is gone) */
  lines?: Map<number, Iterable<number> | null>;
  /** edges not drawn, track and lines */
  hideEdges?: Iterable<number>;
  /** edges drawn with a new look */
  edges?: Map<number, EdgeLook>;
  /** station node ids not drawn */
  hideStations?: Iterable<number>;
  stations?: StationBox[];
  /** line colours, 0-255 */
  colours?: Map<number, [number, number, number]>;
  /** panel fields by record ("line:12", "station:40", "edge:7") */
  fields?: Map<string, Record<string, unknown>>;
  /** cash taken at the click (the quote), the cash's units */
  cost?: number;
  /** strokes and stations drawn as they are, and lines whose network strokes are not drawn (an undo's, from the network
   * before the edit) */
  track?: Strokes[];
  lineStrokes?: Strokes[];
  hideLines?: Iterable<number>;
  rawStations?: RawStations[];
}

/** The edits drawn ahead, composed: every entry's map part, later edits over earlier ones. */
export interface Composed {
  lines: Map<number, Set<number> | null>;
  hideEdges: Set<number>;
  edges: Map<number, EdgeLook>;
  hideStations: Set<number>;
  stations: StationBox[];
  colours: Map<number, [number, number, number]>;
  track: Strokes[];
  lineStrokes: Strokes[];
  hideLines: Set<number>;
  rawStations: RawStations[];
}

interface Entry {
  /** the edit's seq, -1 while unkeyed (a guess not sent yet, or a try refused) */
  seq: number;
  show: Show;
  /** the arrival order of its word, -1 before it */
  armed: number;
  /** which parts still show */
  map: boolean;
  panel: boolean;
  money: boolean;
  /** when its edit was clicked or sent: entries are drawn in this order */
  t: number;
  /** when it was made or last sent, for a guess never answered */
  at: number;
  tries: boolean;
  /** T-4114's per-line guesses (map/aheadShow.ts `showAhead`) */
  line?: number;
}

/** A guess never answered (a worker that restarted, an edit that never went) goes after this, ms */
const GUESS_MAX_MS = 15_000;

let entries: Entry[] = [];
/** bumped on every change, so panels reading `shown` and `cashShown` redraw */
export const pendingVersion = signal(0);
/** when each edit went, by seq (for a word that comes with no entry made at the click) */
const sentAt = new Map<number, number>();
let sending: Entry | null = null;
let sink: ((c: Composed, build: boolean) => void) | null = null;
let sweep = 0;

/** map/aheadShow.ts: where the composed map parts go (`build` false: a network is about to be set, which builds them). */
export function setAheadSink(f: (c: Composed, build: boolean) => void) { sink = f; }

function changed(mapToo: boolean, build = true) {
  pendingVersion.value++;
  if (mapToo) sink?.(compose(), build);
  if (!sweep && entries.some((e) => e.armed < 0)) sweep = window.setTimeout(() => {
    sweep = 0;
    const now = performance.now(), was = entries.length;
    entries = entries.filter((e) => e.armed >= 0 || now - e.at < GUESS_MAX_MS);
    if (entries.length !== was) changed(true);
    else if (entries.some((e) => e.armed < 0)) changed(false);
  }, GUESS_MAX_MS);
}

const hasMap = (s: Show) => !!(s.lines?.size || s.hideEdges || s.edges?.size || s.hideStations || s.stations?.length || s.colours?.size || s.track?.length || s.lineStrokes?.length || s.hideLines || s.rawStations?.length);

function compose(): Composed {
  const c: Composed = { lines: new Map(), hideEdges: new Set(), edges: new Map(), hideStations: new Set(), stations: [], colours: new Map(), track: [], lineStrokes: [], hideLines: new Set(), rawStations: [] };
  for (const e of [...entries].sort((a, b) => a.t - b.t)) {
    if (!e.map) continue;
    const s = e.show;
    for (const [l, edges] of s.lines ?? []) c.lines.set(l, edges ? new Set(edges) : null);
    for (const x of s.hideEdges ?? []) c.hideEdges.add(x);
    for (const [x, look] of s.edges ?? []) c.edges.set(x, { ...c.edges.get(x), ...look });
    for (const x of s.hideStations ?? []) c.hideStations.add(x);
    c.stations.push(...(s.stations ?? []));
    for (const [l, rgb] of s.colours ?? []) c.colours.set(l, rgb);
    c.track.push(...(s.track ?? []));
    c.lineStrokes.push(...(s.lineStrokes ?? []));
    for (const x of s.hideLines ?? []) c.hideLines.add(x);
    c.rawStations.push(...(s.rawStations ?? []));
  }
  return c;
}

/** A handle on one edit shown ahead (`ahead`). */
export interface Ahead {
  /** `f` sends the edit (it must call `edit()` before it first awaits): the entry is keyed by that edit's seq, and
   * moves to the new one when called again (another try) */
  send<T>(f: () => Promise<T>): Promise<T>;
  /** taken away now */
  drop(): void;
  /** the guess replaced */
  set(show: Show): void;
}

/** Show an edit at the click. `tries`: a refusal leaves it shown, unkeyed, until `drop()` or the next `send` (an edit
 * sent several ways until one is taken). */
export function ahead(show: Show, opts: { tries?: boolean } = {}): Ahead {
  const now = performance.now();
  const e: Entry = { seq: -1, show, armed: -1, map: true, panel: true, money: true, t: now, at: now, tries: !!opts.tries };
  entries.push(e);
  changed(hasMap(show));
  return {
    send<T>(f: () => Promise<T>): Promise<T> {
      if (!entries.includes(e)) return f();
      sending = e;
      e.seq = -1;
      e.at = performance.now();
      try { return f(); } finally {
        sending = null;
        if (e.seq < 0 && import.meta.env.DEV) console.warn("T-4145: an edit shown ahead was not sent at once; it goes in 15 s");
      }
    },
    drop() { drop(e); },
    set(s: Show) {
      if (!entries.includes(e)) return;
      e.show = s;
      changed(true);
    },
  };
}

function drop(e: Entry) {
  const was = entries.length;
  entries = entries.filter((x) => x !== e);
  if (entries.length !== was) changed(e.map && hasMap(e.show));
}

// ---------------------------------------------------------------- from clockClient

/** An edit went to the worker as `seq`: an `ahead` being sent takes it. `quiet`: no other edit was unanswered. */
export function pendingSent(seq: number, op: EditOp, quiet: boolean) {
  const now = performance.now();
  sentAt.set(seq, now);
  if (sentAt.size > 256) sentAt.delete(sentAt.keys().next().value!);
  if (sending) { sending.seq = seq; sending.at = now; sending = null; }
  undoSent(seq, op, quiet);
}

// ---------------------------------------------------------------- undo (stage 0b)

/** The newest edit on the undo history while nothing else was waiting, with the network drawn before it: an undo of it
 * right after is drawn at its click from that network (`undoAhead`). `word`: its word's arrival, -1 before;
 * `netIn`: a network holding it is drawn. */
let last: { seq: number; was: NetworkBuffers; ok: boolean; word: number; netIn: boolean } | null = null;
let netNow: () => NetworkBuffers | null = () => null;
let undoShow: ((was: NetworkBuffers, now: NetworkBuffers) => Show) | null = null;
let undoWarm: ((n: NetworkBuffers) => void) | null = null;
/** map/aheadShow.ts: the network drawn now, the undo's show from the network before an edit and the one now, and that
 * comparison's work on one network done ahead */
export function setUndoSource(net: () => NetworkBuffers | null, show: (was: NetworkBuffers, now: NetworkBuffers) => Show, warm: (n: NetworkBuffers) => void) { netNow = net; undoShow = show; undoWarm = warm; }
/** both networks an undo would compare, worked out in idle moments (on a big save a few hundred ms at the click otherwise) */
function warmLater(l: NonNullable<typeof last>) {
  const idle = (f: () => void) => ("requestIdleCallback" in window ? requestIdleCallback(f, { timeout: 3000 }) : setTimeout(f, 200));
  idle(() => { if (last === l) undoWarm?.(l.was); idle(() => { const n = netNow(); if (last === l && n) undoWarm?.(n); }); });
}
/** edits outside the track model's undo history: the page keeps the fares' own (game/fareUndo.ts), loans have none */
const offHistory = (op: EditOp) => op.op.startsWith("fare") || op.op === "stationFares" || op.op === "borrow" || op.op === "repay";
/** edits outside that history that change what is drawn: an undo after one takes back an older edit than it shows */
const drawnOffHistory = (op: EditOp) => op.op === "link" || op.op === "addBuilding" || op.op === "reshapeBuilding" || op.op === "renameBuilding" || op.op === "removeBuilding";

function undoSent(seq: number, op: EditOp, quiet: boolean) {
  if (op.op === "undo" || op.op === "redo") {
    const l = last, now = netNow();
    last = null;
    if (op.op !== "undo" || !l || !l.ok || !l.netIn || !quiet || !now || now === l.was || !undoShow) return;
    if (entries.some((e) => e.armed >= 0 && e.map)) return;
    const t = performance.now();
    entries.push({ seq, show: undoShow(l.was, now), armed: -1, map: true, panel: false, money: false, t, at: t, tries: false });
    changed(true);
    return;
  }
  if (offHistory(op)) return;
  const was = netNow();
  last = !drawnOffHistory(op) && quiet && was && !entries.some((e) => e.armed >= 0 && e.map) ? { seq, was, ok: false, word: -1, netIn: false } : null;
}

/** The worker's word for `seq`, the `order`th message to arrive. */
export function pendingWord(seq: number, word: EditWord, order: number) {
  if (last?.seq === seq) last.word = order;
  let e = entries.find((x) => x.seq === seq);
  const fromWord = wordShow(word);
  if (!e) {
    if (!hasMap(fromWord) && fromWord.cost === undefined) return;
    const t = sentAt.get(seq) ?? performance.now();
    e = { seq, show: {}, armed: -1, map: true, panel: true, money: true, t, at: t, tries: false };
    entries.push(e);
  }
  const drawn = hasMap(e.show) || hasMap(fromWord);
  e.show = { ...e.show, ...fromWord };
  e.armed = order;
  // T-4114: the line tool's guesses for a line the word speaks for, made before this edit went, are replaced by it
  const sent = sentAt.get(seq) ?? e.t;
  if (word.lines?.length) {
    const ls = new Set(word.lines.map((l) => l.line));
    entries = entries.filter((x) => !(x.line !== undefined && x.armed < 0 && ls.has(x.line) && x.t <= sent));
  }
  // (a word with nothing to draw, a cost alone, leaves the map as it is)
  changed(drawn);
}

function wordShow(w: EditWord): Show {
  const s: Show = {};
  if (w.lines?.length) s.lines = new Map(w.lines.map((l) => [l.line, l.edges]));
  if (w.hideEdges?.length) s.hideEdges = w.hideEdges;
  if (w.edges?.length) s.edges = new Map(w.edges.map(({ edge, ...look }) => [edge, look]));
  if (w.hideStations?.length) s.hideStations = w.hideStations;
  if (w.stations?.length) s.stations = w.stations;
  if (w.colours?.length) s.colours = new Map(w.colours.map((c) => [c.line, c.rgb]));
  if (w.charge !== undefined) s.cost = w.charge;
  return s;
}

/** `seq`'s answer: refused, its entry goes now (or, tried several ways, waits unkeyed for the next try); taken and with
 * no word yet (a worker that sent none), armed now. */
export function pendingAnswer(seq: number, ok: boolean, order: number) {
  const e = entries.find((x) => x.seq === seq);
  sentAt.delete(seq);
  if (last?.seq === seq) { if (ok) last.ok = true; else last = null; }
  if (!e) return;
  if (ok) {
    if (e.armed < 0) { e.armed = order; changed(false); }
    return;
  }
  if (e.tries) { e.seq = -1; e.at = performance.now(); return; }
  drop(e);
}

/** A state, network or money taken that arrived as the `order`th message: the parts it holds of every edit whose word came
 * before it go. `net`: a network is about to be set (it builds what is drawn ahead, so this only updates the model). */
export function pendingTaken(order: number, what: { net?: boolean; rows?: boolean; money?: boolean }): Map<number, Set<number> | null> {
  let map = false, any = false;
  /** the lines the cleared entries drew, the newest's edges each, for the dev check that the network holds them */
  const held = new Map<number, Set<number> | null>();
  if (what.net && last && !last.netIn && last.word >= 0 && last.word < order) { last.netIn = true; warmLater(last); }
  for (const e of [...entries].sort((a, b) => a.t - b.t)) {
    if (e.armed < 0 || e.armed > order) continue;
    if (what.net && e.map) {
      e.map = false;
      map = any = true;
      for (const [l, edges] of e.show.lines ?? []) held.set(l, edges ? new Set(edges) : null);
    }
    if (what.rows && e.panel) { e.panel = false; any = true; }
    if (what.money && e.money) { e.money = false; any = true; }
  }
  // a line drawn ahead looks as the world says (running or not, fast), which a new line gets with its rows
  const redraw = !!what.rows && !what.net && entries.some((e) => e.map && e.show.lines?.size);
  if (!any && !redraw) return held;
  entries = entries.filter((e) => e.map || e.panel || e.money);
  changed(map || redraw, !what.net);
  return held;
}

/** A new game or a worker started again: nothing waits. */
export function pendingClear() {
  sentAt.clear();
  last = null;
  if (!entries.length) return;
  entries = [];
  changed(true);
}

/** T-4114: the line tool's and the line panel's guess for a line's strokes, one per line (map/aheadShow.ts). */
export function guessLine(line: number, edges: Iterable<number> | null) {
  entries = entries.filter((x) => !(x.line === line && x.armed < 0));
  const now = performance.now();
  entries.push({ seq: -1, show: { lines: new Map([[line, edges]]) }, armed: -1, map: true, panel: false, money: false, t: now, at: now, tries: false, line });
  changed(true);
}
/** T-4114: that line's edit refused: its guess goes. */
export function dropLineGuess(line: number) {
  const e = [...entries].reverse().find((x) => x.line === line && x.armed < 0);
  if (e) drop(e);
}

// ---------------------------------------------------------------- what panels read

/** The value an edit not yet in the world gives `record`'s `field` (the newest), else `value`. */
export function shown<T>(record: string, field: string, value: T): T {
  void pendingVersion.value;
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i], f = e.panel ? e.show.fields?.get(record) : undefined;
    if (f && field in f) return f[field] as T;
  }
  return value;
}

/** The cash less what edits not yet in the money cost (their quote, then the worker's charge). */
export function cashShown(cash: number): number {
  void pendingVersion.value;
  let c = cash;
  for (const e of entries) if (e.money && e.show.cost) c -= e.show.cost;
  return c;
}

/** Whether any edit is waiting to show (undo reads it). */
export function anyPending(): boolean {
  return entries.length > 0;
}
