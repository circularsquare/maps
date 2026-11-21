// T-3420: what the player hid on the map, as the renderer and the picks read it: lines (the Lines tab's
// toggles, game/lineVisibility.ts) and track levels (Settings' Show levels). A view setting only:
// hidden lines still run, hidden track still carries them. A switched-off level is drawn faint
// (LEVEL_FADE) and skipped by picks, snaps and the select tool (T-3560); a hidden line is not drawn.

import type { Level } from "../game/types";

/** Line numbers not drawn (a hidden line's twin too), the levels not drawn as bits (`1 << (level + 3)`),
 * and a station's and a node's level by id, for the picks and the station boxes. `v` counts changes. */
export const hidden = {
  lines: new Set<number>(),
  levels: 0,
  stationLevel: (_id: string): Level | undefined => undefined,
  nodeLevel: (_id: number): Level | undefined => undefined,
  /** Settings' Show blueprints is off: every blueprint piece (track, lines, stations, BRT lanes) is not drawn or picked */
  blueprints: false,
  v: 0,
};

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

export function setHidden(lines: Set<number>, levels: number, stationLevel: (id: string) => Level | undefined, nodeLevel: (id: number) => Level | undefined) {
  const same = levels === hidden.levels && lines.size === hidden.lines.size && [...lines].every((l) => hidden.lines.has(l));
  hidden.stationLevel = stationLevel;
  hidden.nodeLevel = nodeLevel;
  if (same) return false;
  hidden.lines = lines;
  hidden.levels = levels;
  hidden.v++;
  return true;
}

/** T-3560: a level switched off is drawn this opaque, and cannot be picked. */
export const LEVEL_FADE = 0.08;

/** How opaque a height (in levels, a ramp's between two) is drawn: 1 on a shown level, `LEVEL_FADE` on a
 * switched-off one, and between them along a ramp, so a ramp fades toward a switched-off level. */
export function levelShown(z: number, mask = hidden.levels): number {
  if (!mask) return 1;
  const c = Math.max(-3, Math.min(3, z)), lo = Math.floor(c), f = c - lo;
  const on = (l: number) => (l > 3 || (mask & (1 << (l + 3))) === 0 ? 1 : LEVEL_FADE);
  return on(lo) * (1 - f) + on(lo + 1) * f;
}

/** Whether a whole level is hidden. */
export const levelHidden = (l: number) => (hidden.levels & (1 << (Math.max(-3, Math.min(3, Math.round(l))) + 3))) !== 0;

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
  float sa = ((u_level_hide >> a) & 1) != 0 ? ${LEVEL_FADE.toFixed(2)} : 1.0;
  float sb = ((u_level_hide >> b) & 1) != 0 ? ${LEVEL_FADE.toFixed(2)} : 1.0;
  return mix(sa, sb, c - lo);
}`;
