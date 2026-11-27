// T-3420: what the player hid on the map, as the renderer and the picks read it: lines (the Lines tab's
// toggles, game/lineVisibility.ts), track levels (Settings' Show levels) and kinds of line and track (Show modes,
// T-4400). A view setting only: hidden lines still run, hidden track still carries them. What is hidden is not
// drawn and is skipped by picks, snaps and the select tool (T-3560); a ramp fades out toward a hidden level.

import type { Level } from "../game/types";
import type { TransitMode } from "../game/transitModes";
import { F_BUSWAY, F_HSR, F_TRAM } from "./network";

/** T-4400: a kind of line and track's bit in `hidden.modes`. */
export const MODE_BIT: Readonly<Record<TransitMode, number>> = { train: 1, hsr: 2, tram: 4, bus: 8, ferry: 16 };

/** Line numbers not drawn (a hidden line's twin too), the levels not drawn as bits (`1 << (level + 3)`), the kinds
 * of line and track not drawn (`MODE_BIT`), and by id a station's and a node's level and kinds (`MODE_BIT`: a
 * station's own and its lines', a node's track's) and an edge's kind, for the picks, marks and station boxes. `v`
 * counts changes. */
export const hidden = {
  lines: new Set<number>(),
  levels: 0,
  modes: 0,
  stationLevel: (_id: string): Level | undefined => undefined,
  nodeLevel: (_id: number): Level | undefined => undefined,
  stationModes: (_id: string): number => 0,
  nodeModes: (_id: number): number => 0,
  edgeModes: (_id: number): number => 0,
  stationBuilt: (_id: string): boolean | undefined => undefined,
  nodeBuiltPorts: (_id: number): number | undefined => undefined,
  /** Settings' Show blueprints is off: every blueprint piece (track, lines, stations, BRT lanes) is not drawn or picked */
  blueprints: false,
  v: 0,
};

/** How the station and node lookups are answered (main.ts, from the world in hand). */
export interface HiddenLookups {
  stationLevel: (id: string) => Level | undefined;
  nodeLevel: (id: number) => Level | undefined;
  stationModes: (id: string) => number;
  nodeModes: (id: number) => number;
  edgeModes: (id: number) => number;
  stationBuilt: (id: string) => boolean | undefined;
  nodeBuiltPorts: (id: number) => number | undefined;
}

const blueprintHooks = new Set<() => void>();
/** `f` runs when blueprints are hidden or shown (the DOM marks, which no signal moves). */
export const onBlueprintsHidden = (f: () => void) => void blueprintHooks.add(f);

/** Hide or show every blueprint (view only); true when it changed. */
export function setBlueprintsHidden(on: boolean) {
  if (on === hidden.blueprints) return false;
  hidden.blueprints = on;
  hidden.v++;
  for (const f of blueprintHooks) f();
  return true;
}

export function setHidden(lines: Set<number>, levels: number, modes: number, look: HiddenLookups) {
  const same = levels === hidden.levels && modes === hidden.modes && lines.size === hidden.lines.size && [...lines].every((l) => hidden.lines.has(l));
  Object.assign(hidden, look);
  if (same) return false;
  hidden.lines = lines;
  hidden.levels = levels;
  hidden.modes = modes;
  hidden.v++;
  return true;
}

/** How much of a height (in levels, a ramp's between two) is drawn: 1 on a shown level, 0 on a hidden one, and
 * between them along a ramp, so a ramp fades out toward a hidden level. */
export function levelShown(z: number, mask = hidden.levels): number {
  if (!mask) return 1;
  const c = Math.max(-3, Math.min(3, z)), lo = Math.floor(c), f = c - lo;
  const on = (l: number) => (l > 3 || (mask & (1 << (l + 3))) === 0 ? 1 : 0);
  return on(lo) * (1 - f) + on(lo + 1) * f;
}

/** Whether a whole level is hidden. */
export const levelHidden = (l: number) => (hidden.levels & (1 << (Math.max(-3, Math.min(3, Math.round(l))) + 3))) !== 0;

/** T-4400: whether every kind in `bits` (`MODE_BIT`) is hidden; false for none known. */
export const modesHidden = (bits: number) => bits !== 0 && (bits & ~hidden.modes) === 0;
/** A track stroke's kind (its network.ts flags) as its `MODE_BIT`: a busway is the buses' */
export const strokeMode = (flags: number) => ((flags & F_BUSWAY) !== 0 ? MODE_BIT.bus : (flags & F_TRAM) !== 0 ? MODE_BIT.tram : (flags & F_HSR) !== 0 ? MODE_BIT.hsr : MODE_BIT.train);
/** A track edge's kind as its `MODE_BIT` (`TrackEdge.kind` 4 a busway, 2 and 3 tram track) */
export const edgeMode = (e: { hsr?: boolean; kind?: number }) => (e.kind === 4 ? MODE_BIT.bus : e.kind ? MODE_BIT.tram : e.hsr ? MODE_BIT.hsr : MODE_BIT.train);

/** Whether station `id` is hidden: blueprint hidden, on a hidden level, or every kind of line and track at it hidden. */
export function stationOff(id: string, built?: boolean): boolean {
  if (hidden.blueprints && (built ?? hidden.stationBuilt(id)) === false) return true;
  if (hidden.levels) { const l = hidden.stationLevel(id); if (l !== undefined && levelHidden(l)) return true; }
  return !!hidden.modes && modesHidden(hidden.stationModes(id));
}
/** Whether node `id` (a junction, a track end, a flyover) is hidden: blueprint-only while blueprints are hidden, on a hidden level, or all its track of hidden kinds. */
export function nodeOff(id: number, builtPorts?: number): boolean {
  if (hidden.blueprints && (builtPorts ?? hidden.nodeBuiltPorts(id)) === 0) return true;
  if (hidden.levels) { const l = hidden.nodeLevel(id); if (l !== undefined && levelHidden(l)) return true; }
  return !!hidden.modes && modesHidden(hidden.nodeModes(id));
}
/** Whether track edge `e` is hidden: blueprint hidden, wholly on hidden levels, or of a hidden kind. */
export const edgeOff = (e: { hsr?: boolean; kind?: number; levelMin: number; levelMax: number; built?: boolean }) =>
  (hidden.blueprints && e.built === false) || (!!hidden.levels && levelHidden(e.levelMin) && levelHidden(e.levelMax)) || (hidden.modes & edgeMode(e)) !== 0;
/** Whether a Park and Ride lot (named for station `stationId`) is hidden. */
export const lotOff = (stationId: string, built?: boolean): boolean => stationOff(stationId, built);

/** Whether a stroke segment with `levelCode` `code` (map/network.ts) is hidden: both its ends past half faded. */
export function codeHidden(code: number): boolean {
  if (!hidden.levels) return false;
  const q0 = Math.floor(code / 256), q1 = code - q0 * 256;
  return Math.max(levelShown(q0 / 32 - 4), levelShown(q1 / 32 - 4)) < 0.5;
}

/** Per stroke set (by its `seg` array), which segments a pick passes over: those not drawn. */
export const pickSkip = new WeakMap<Float32Array, (i: number) => boolean>();

/** The shaders' share of a height drawn (as `levelShown`), from `u_level_hide`. */
export const LEVEL_SHOWN_GLSL = `float levelShown(float z) {
  if (u_level_hide == 0) return 1.0;
  float c = clamp(z, -3.0, 3.0), lo = floor(c);
  int a = int(lo) + 3, b = min(a + 1, 6);
  float sa = ((u_level_hide >> a) & 1) != 0 ? 0.0 : 1.0;
  float sb = ((u_level_hide >> b) & 1) != 0 ? 0.0 : 1.0;
  return mix(sa, sb, c - lo);
}`;
