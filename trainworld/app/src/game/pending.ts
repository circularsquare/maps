// T-4145: the edits waiting to show (notes/T-4145.md). Each is drawn and shown from the click, from what the page can
// tell (`ahead`), and armed by the worker's word for it (workers/clockWord.ts), whose parts replace the guess's. Message
// order says what a state holds: every state the worker sends after an edit's word was built after that edit was applied
// (its follow-ups start again behind a new edit, and a stop click's rows go before the next edit's apply). So an armed
// entry's map part goes with the first network taken that arrived after its word, its panel part with the first rows or
// state, its cost with the first money, cash or state. A refused edit's entry goes in the frame its answer lands.
// T-4170, one path: an edit's entry is made in one place, when it is sent (`pendingSent`, which every edit passes), so an
// edit has at most one. What it shows is read from its op (`setOpGuess`: game/editGuess.ts, or the undo's from the
// networks), joined with what the caller drew at the click with `ahead()` (a ghost's box, a quoted price).

import { signal } from "@preact/signals";
import type { EdgeLook, EditOp, EditWord, StationBox } from "../workers/protocol";
import { strokesFromPieces, type NetworkBuffers, type Strokes } from "../map/network";
import type { RawStations } from "../map/lineAhead";
import type { BusAhead } from "../map/busAhead";

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
  /** cash taken at the click (the quote), the cash's units: only a figure the worker will charge exactly, else none */
  cost?: number;
  /** what the edit costs at least, where its price is not known exactly (the cash's units): one the cash cannot pay is
   * refused, so nothing is shown for it */
  least?: number;
  /** strokes and stations drawn as they are, and lines whose network strokes are not drawn (an undo's, from the network
   * before the edit) */
  track?: Strokes[];
  lineStrokes?: Strokes[];
  hideLines?: Iterable<number>;
  rawStations?: RawStations[];
  /** bus and ferry lines drawn on a new route in the bus layer (map/busAhead.ts), until the rows that hold them: a part
   * of the panel's kind */
  buses?: Map<number, BusAhead>;
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
  buses: Map<number, BusAhead>;
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
  /** when it was made or last sent, then when it was armed: a guess never answered, or an entry no state ever clears,
   * goes after a while (`changed`) */
  at: number;
  tries: boolean;
}

/** A guess never sent (a caller that never sent its edit) goes after this, ms */
const GUESS_MAX_MS = 15_000;
/** A guess sent and not yet answered goes after this, ms: a backstop (a worker that restarts clears every entry), far
 * above an edit that waits on its ground or masks downloading */
const ANSWER_MAX_MS = 120_000;
/** An armed entry whose network, rows or money never came goes after this, ms: a backstop (the word says when no
 * network follows), so a slip shows the network as it is rather than a stand-in for ever. Far above the slowest
 * network after an edit seen on Asia (about 6 s). */
const ARMED_MAX_MS = 30_000;

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
  if (!sweep && entries.length) sweep = window.setTimeout(() => {
    sweep = 0;
    const now = performance.now(), was = entries.length;
    entries = entries.filter((e) => {
      const keep = now - e.at < (e.armed >= 0 ? ARMED_MAX_MS : e.seq >= 0 && sentAt.has(e.seq) ? ANSWER_MAX_MS : GUESS_MAX_MS);
      if (!keep && e.armed >= 0) { swept++; console.warn("T-4170: an edit's entry outlived its states", e.seq, { map: e.map, panel: e.panel, money: e.money }); }
      return keep;
    });
    if (entries.length !== was) changed(true);
    else if (entries.length) changed(false);
  }, GUESS_MAX_MS);
}

/** T-4170: what a check reads: entries that ever shared a seq (none, by the one path), armed entries the backstop took */
let doubled = 0, swept = 0;

/** (an empty list draws nothing, and an entry with only that would wait for a network that may never come) */
const some = (x: Iterable<unknown> | undefined) => !!x && !x[Symbol.iterator]().next().done;
/** what the network draws (an entry waits for the network that holds it) */
const hasDrawn = (s: Show) => !!(s.lines?.size || some(s.hideEdges) || s.edges?.size || some(s.hideStations) || s.stations?.length || s.colours?.size || s.track?.length || s.lineStrokes?.length || some(s.hideLines) || s.rawStations?.length);
/** anything drawn at the click: the network's, or the bus layer's */
const hasMap = (s: Show) => hasDrawn(s) || !!s.buses?.size;

function compose(): Composed {
  const c: Composed = { lines: new Map(), hideEdges: new Set(), edges: new Map(), hideStations: new Set(), stations: [], colours: new Map(), track: [], lineStrokes: [], hideLines: new Set(), rawStations: [], buses: new Map() };
  for (const e of [...entries].sort((a, b) => a.t - b.t)) {
    // (the bus layer's routes come from the world, so they go with the rows, not the network)
    if (e.panel) for (const [l, b] of e.show.buses ?? []) c.buses.set(l, b);
    if (!e.map) continue;
    const s = e.show;
    for (const [l, edges] of s.lines ?? []) c.lines.set(l, edges ? new Set(edges) : null);
    // (an edge hidden by a later edit is not drawn by an earlier one's look or line)
    for (const x of s.hideEdges ?? []) {
      c.hideEdges.add(x);
      c.edges.delete(x);
      for (const edges of c.lines.values()) edges?.delete(x);
    }
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
      if (sending && import.meta.env.DEV) console.warn("T-4170: an edit shown ahead was sent inside another's send");
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
  if (entries.length !== was) changed((e.map && hasDrawn(e.show)) || (e.panel && !!e.show.buses?.size));
}

// ---------------------------------------------------------------- from clockClient

/** Two shows as one: `b`'s parts over `a`'s where both say something of the same record, lists joined. */
export function mergeShow(a: Show, b: Show): Show {
  const out: Show = { ...a, ...b };
  const maps = ["lines", "edges", "colours", "buses"] as const;
  for (const k of maps) if (a[k] && b[k]) (out as Record<string, unknown>)[k] = new Map<number, unknown>([...a[k]!, ...b[k]!]);
  if (a.fields && b.fields) {
    const f = new Map(a.fields);
    for (const [r, v] of b.fields) f.set(r, { ...f.get(r), ...v });
    out.fields = f;
  }
  const lists = ["hideEdges", "hideStations", "stations", "track", "lineStrokes", "hideLines", "rawStations"] as const;
  for (const k of lists) if (a[k] && b[k]) (out as Record<string, unknown>)[k] = [...(a[k] as Iterable<unknown>), ...(b[k] as Iterable<unknown>)];
  return out;
}

/** T-4097: edits sent so far: a read of the track model asked after the nth is of the model holding the first n */
let sentCount = 0;
export const editsSentCount = () => sentCount;

let opGuess: ((op: EditOp) => Show | null) | null = null;
/** the cash now and whether every price is paid (a sandbox), the cash's units */
let cashNow: () => { cash: number; sandbox: boolean } | null = () => null;
/** map/aheadShow.ts: where the cash is read (game/money.ts) */
export function setCashSource(f: () => { cash: number; sandbox: boolean } | null) { cashNow = f; }
/** map/aheadShow.ts: what an edit shows at its click, read from its op (game/editGuess.ts) */
export function setOpGuess(f: (op: EditOp) => Show | null) { opGuess = f; }

/** An edit went to the worker as `seq` (every edit passes here once): its one entry is made, from the `ahead` being sent
 * if there is one, with what its op says joined in. `quiet`: no other edit was unanswered. */
export function pendingSent(seq: number, op: EditOp, quiet: boolean) {
  const now = performance.now();
  sentAt.set(seq, now);
  if (sentAt.size > 256) sentAt.delete(sentAt.keys().next().value!);
  let guess: Show | null = null;
  if (op.op === "undo" || op.op === "redo") guess = undoGuess(op, quiet);
  else {
    noteEdit(seq, op, quiet);
    try { guess = opGuess?.(op) ?? null; } catch (err) { console.error("T-4170: an edit's guess failed", err); }
  }
  const e = sending;
  sending = null;
  if (e) {
    e.seq = seq;
    e.at = now;
    if (guess) e.show = mergeShow(guess, e.show);
    if (!priced(e.show, quiet, e)) { drop(e); return; }
    if (guess) changed(hasMap(guess));
  } else if (guess && priced(guess, quiet)) {
    entries.push({ seq, show: guess, armed: -1, map: true, panel: true, money: true, t: now, at: now, tries: false });
    changed(hasMap(guess));
  }
  if (entries.filter((x) => x.seq === seq).length > 1) { doubled++; console.error("T-4170: two entries for one edit", seq, op.op); }
  // (after the guess, which reads whether any edit went since its read)
  sentCount++;
}

/** Whether an edit's show can stand, its price made right first: a price is shown only for an edit sent with no other
 * unanswered (another's could change it, and the worker's word gives the exact one a moment later), and an edit the cash
 * cannot pay (the worker refuses it) shows nothing. */
function priced(s: Show, quiet: boolean, self?: Entry): boolean {
  if (!quiet) { delete s.cost; return true; }
  const need = s.cost ?? s.least ?? 0, m = cashNow();
  if (!(need > 0) || !m || m.sandbox) return true;
  let cash = m.cash;
  for (const x of entries) if (x !== self && x.money && x.show.cost) cash -= x.show.cost;
  return need <= cash + 1e-6;
}

/** Armed by its word or its answer: a part it has nothing for is done already (a panel edit's map, a free edit's cost),
 * so it waits only for the states that hold what it shows. False when nothing is left. */
function arm(e: Entry, order: number, noNet = false): boolean {
  e.armed = order;
  e.at = performance.now();
  // (no network follows a service edit or a rename: whatever it guessed on the map is not what the network will draw)
  e.map = !noNet && hasDrawn(e.show);
  e.panel = !!(e.show.fields?.size || e.show.buses?.size);
  e.money = !!e.show.cost;
  return e.map || e.panel || e.money;
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

/** An undo's show at its click: what the newest edit changed, from the network before it (null when that cannot be told) */
function undoGuess(op: EditOp, quiet: boolean): Show | null {
  const l = last, now = netNow();
  last = null;
  undoMiss = op.op !== "undo" ? "redo" : !l ? "no edit kept" : !l.ok ? "not answered" : !l.netIn ? "its network not in" : !quiet ? "another edit out" : !now || now === l.was || !undoShow ? "no network" : now.anchor?.version !== l.was.anchor?.version ? "the draw anchor moved" : entries.some((e) => e.armed >= 0 && e.map) ? "another edit drawn ahead" : "";
  return undoMiss ? null : undoShow!(l!.was, now!);
}
/** why the last undo was not drawn at its click, "" when it was (for checks) */
let undoMiss = "";

/** An edit on the undo history: kept with the network drawn before it, when nothing else is waiting. */
function noteEdit(seq: number, op: EditOp, quiet: boolean) {
  if (offHistory(op)) return;
  const was = netNow();
  last = !drawnOffHistory(op) && quiet && was && !entries.some((e) => e.armed >= 0 && e.map) ? { seq, was, ok: false, word: -1, netIn: false } : null;
}

/** The worker's word for `seq`, the `order`th message to arrive; `noNet`: no network follows it. */
export function pendingWord(seq: number, word: EditWord, order: number, noNet = false) {
  // (an edit an undo would not take back whole: none is drawn at the undo's click)
  if (last?.seq === seq) { if (word.noUndo) last = null; else last.word = order; }
  let e = entries.find((x) => x.seq === seq);
  const fromWord = wordShow(word);
  // (a one-way word names no end nodes: the guess's are the same edges')
  if (fromWord.edges && e?.show.edges) for (const [id, look] of fromWord.edges) { const ends = e.show.edges.get(id)?.ends; if (look.oneWay !== undefined && ends) look.ends = ends; }
  if (!e) {
    if (!hasMap(fromWord) && !fromWord.cost) return;
    const t = sentAt.get(seq) ?? performance.now();
    e = { seq, show: {}, armed: -1, map: true, panel: true, money: true, t, at: t, tries: false };
    entries.push(e);
  }
  const drawn = hasMap(e.show) || hasMap(fromWord);
  e.show = { ...e.show, ...fromWord };
  if (!arm(e, order, noNet)) drop(e);
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
  if (w.pieces?.length) s.track = [strokesFromPieces(w.pieces)];
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
    if (e.armed < 0) { if (arm(e, order)) changed(false); else drop(e); }
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
    if (what.rows && e.panel) {
      e.panel = false;
      any = true;
      if (e.show.buses?.size) map = true;
    }
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

/** T-4170: the entries waiting now and how many times two shared a seq, for checks (checks/t4170-one-path.mjs). */
export function pendingNow() {
  const now = performance.now();
  return { doubled, swept, entries: entries.map((e) => ({ seq: e.seq, armed: e.armed, map: e.map, panel: e.panel, money: e.money, tries: e.tries, ageMs: Math.round(now - e.at) })), undo: last && { seq: last.seq, ok: last.ok, word: last.word, netIn: last.netIn, sameNet: last.was === netNow() }, undoMiss };
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
