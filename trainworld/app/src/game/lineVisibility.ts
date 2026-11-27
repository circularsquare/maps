// T-3420: lines the player hid on the map (the Lines tab's hide mode) and the track levels and kinds of line and
// track (T-4400) hidden in Settings' Show. A view setting only: hidden lines still run, hidden track still carries
// trains. Hidden lines are saved with the game (the header's `hidden`, kept by the clock worker outside the undo
// history, as the hand order is); hidden levels and kinds last while the page is open, so a reload never opens on a
// map missing a level. map/visibility.ts is what the renderer and the picks read.

import { effect, signal } from "@preact/signals";
import type { Level, WorldView } from "./types";
import type { TransitMode } from "./transitModes";
import { modeOf } from "./vehicle";
import { edgeMode, MODE_BIT } from "../map/visibility";

/** Line ids not drawn. */
export const hiddenLines = signal<ReadonlySet<string>>(new Set());
/** Track levels not drawn. */
export const hiddenLevels = signal<ReadonlySet<Level>>(new Set());
/** T-4400: the kinds of line not drawn, with their track and stations (Settings' Show modes), while the page is open
 * as the levels are. */
export const hiddenModes = signal<ReadonlySet<TransitMode>>(new Set());
/** Blueprints not drawn (Settings' Show blueprints), remembered in this browser. */
const BLUEPRINTS_KEY = "tw.hideBlueprints";
export const hideBlueprints = signal<boolean>((() => { try { return localStorage.getItem(BLUEPRINTS_KEY) === "1"; } catch { return false; } })());
effect(() => { const on = hideBlueprints.value; try { localStorage.setItem(BLUEPRINTS_KEY, on ? "1" : "0"); } catch { /* kept for this page only */ } });
/** T-4002: the same for a viewed community network, which opens with blueprints hidden whatever `hideBlueprints` says; the
 * player's own setting is left as it was for their own games. */
export const hideBlueprintsView = signal(true);
/** Hide mode: the Lines tab shows each row's box. Leaving it shows every line again, so lines stay hidden only while it is on. */
export const visToggles = signal(false);

let sink: ((lines: string[]) => void) | null = null;
/** Where a change goes (the clock worker, for the save). */
export function setHiddenLinesSink(f: (lines: string[]) => void) { sink = f; }

/** The game's hidden lines as the clock worker has them: on a load, and after a new line reuses an id. A game
 * with lines hidden opens in hide mode, the only place they can be shown again. */
export function takeHiddenLines(lines: readonly string[]) {
  hiddenLines.value = new Set(lines);
  if (lines.length) visToggles.value = true;
}

/** Hide or show lines `ids`. */
export function setLinesHidden(ids: readonly string[], hide: boolean) {
  const was = hiddenLines.peek(), next = new Set(was);
  for (const id of ids) {
    if (hide) next.add(id);
    else next.delete(id);
  }
  if (next.size === was.size && [...next].every((id) => was.has(id))) return;
  hiddenLines.value = next;
  sink?.([...next]);
}

/** Show every line. */
export function showAllLines() { setLinesHidden([...hiddenLines.peek()], false); }

/** Hide a shown level, show a hidden one. */
export function toggleLevel(l: Level) {
  const next = new Set(hiddenLevels.peek());
  if (next.has(l)) next.delete(l);
  else next.add(l);
  hiddenLevels.value = next;
}

/** Show every level. */
export function showAllLevels() { if (hiddenLevels.peek().size) hiddenLevels.value = new Set(); }

/** `hiddenLevels` as the shaders' bits, one per level from -3. */
export const levelBits = (s: ReadonlySet<Level>) => [...s].reduce<number>((m, l) => m | (1 << (l + 3)), 0);

/** Hide a shown kind, show a hidden one. */
export function toggleMode(m: TransitMode) {
  const next = new Set(hiddenModes.peek());
  if (next.has(m)) next.delete(m);
  else next.add(m);
  hiddenModes.value = next;
}

/** Show every kind. */
export function showAllModes() { if (hiddenModes.peek().size) hiddenModes.value = new Set(); }

/** Show every level and kind of line. */
export function showAllShown() { showAllLevels(); showAllModes(); }

/** `hiddenModes` as `MODE_BIT` bits. */
export const modeBits = (s: ReadonlySet<TransitMode>) => [...s].reduce<number>((m, x) => m | MODE_BIT[x], 0);

/** T-4400: each node's kinds of track, and each edge's, as `MODE_BIT` bits (kept while the edges are) */
const trackModeMemo = new WeakMap<object, { nodes: Map<number, number>; edges: Map<number, number> }>();
export function trackModes(w: WorldView) {
  let m = trackModeMemo.get(w.edges);
  if (!m) {
    m = { nodes: new Map(), edges: new Map() };
    for (const e of w.edges) {
      const b = edgeMode(e);
      m.edges.set(e.id, b);
      m.nodes.set(e.a, (m.nodes.get(e.a) ?? 0) | b);
      m.nodes.set(e.b, (m.nodes.get(e.b) ?? 0) | b);
    }
    trackModeMemo.set(w.edges, m);
  }
  return m;
}

/** T-4400: each station's kinds as `MODE_BIT` bits: its own (a port the ferries', a BRT station the buses', a tram stop
 * the trams', a train station its track's, train or high speed) and those of the lines stopping there, so a station
 * shared with a kind still shown stays */
const stationModeMemo = new WeakMap<WorldView, Map<string, number>>();
export function stationModes(w: WorldView): Map<string, number> {
  let m = stationModeMemo.get(w);
  if (m) return m;
  m = new Map();
  const nodes = trackModes(w).nodes, rail = MODE_BIT.train | MODE_BIT.hsr;
  for (const s of w.save.stations) m.set(s.id, s.port ? MODE_BIT.ferry : s.brt ? MODE_BIT.bus : s.tram ? MODE_BIT.tram : (nodes.get(s.num) ?? 0) & rail || MODE_BIT.train);
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st) continue;
    const b = MODE_BIT[modeOf(st)];
    for (const id of l.stops) { const v = m.get(id); if (v !== undefined) m.set(id, v | b); }
    if (st.backStops) for (const id of st.backStops) { const v = m.get(id); if (v !== undefined) m.set(id, v | b); }
  }
  stationModeMemo.set(w, m);
  return m;
}
