// The elevation and speed limit in the map's tags (T-526): the track hover and the track tool's
// placing tag write them the same way, an icon then the value on its display colour.

import { levelLabel, moneyShort, plural, speed } from "../game/format";
import { bulletNode, isBusLine, isFerryLine } from "../ui/bullet";
import { idleLines } from "../game/lineRunning";
import { inkOn, LEVEL_COLOURS, rgb, speedColour } from "../game/palette";
import { TERRAIN_COLOURS, TERRAIN_NAMES, TERRAIN_SURCHARGES, WATER_SURCHARGE } from "../game/terrain";
import { guideVerbose } from "../game/ui";

const WATER_COLOUR = "#7fb3e0";
import type { Level, LineInput } from "../game/types";

/** An up and down arrow: elevation. */
const ELEVATION_PATH = "M8 1.8v12.4M5 4.8l3-3 3 3M5 11.2l3 3 3-3";
/** Two chevrons pointing right: speed limit. */
const SPEED_PATH = "M3.5 3.8L7.7 8l-4.2 4.2M8.5 3.8L12.7 8l-4.2 4.2";

/** A value with its colour behind it at `alpha`, dark text on top (T-411). */
export function chip(text: string, colour: string | null, alpha = 0.5): HTMLSpanElement {
  const s = document.createElement("span");
  s.textContent = text;
  if (colour) {
    const [r, g, b] = rgb(colour);
    s.className = "tag-chip";
    s.style.setProperty("--chip-bg",`rgba(${r},${g},${b},${alpha})`);
    s.style.padding = "0 3px";
    s.style.borderRadius = "2px";
  }
  return s;
}

const COST_YELLOW = "#f2c200", COST_GREEN = "#8fd97a";
/** A cost multiplier's colour and strength (T-411, T-1159): light yellow at 1x (Anita, T-2271),
 * growing above it, light green growing below it down to 0.5x. The track tool's tag and the Build
 * tab's price. */
export function costTint(mult: number): [string, number] | null {
  if (mult < 0.995) return [COST_GREEN, 0.3 + 0.5 * Math.min(1, (1 - mult) / 0.5)];
  return [COST_YELLOW, 0.2 + 0.6 * Math.min(1, Math.max(0, mult - 1) / 1.5)];
}
/** `costTint` as a CSS background. */
export function costBackground(mult: number): string | undefined {
  const t = costTint(mult);
  if (!t) return undefined;
  const [r, g, b] = rgb(t[0]);
  return `rgba(${r},${g},${b},${t[1]})`;
}

function icon(path: string, name: string): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "tag-icon");
  svg.setAttribute("aria-label", name);
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", path);
  svg.append(p);
  return svg;
}

/** The icon and chip for an elevation (a drawn stroke's level is rounded, as the shader colours it). */
export function elevationFact(level: number, ...also: number[]): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(ELEVATION_PATH, "Level"));
  for (const v of [level, ...also]) {
    const l = Math.max(-3, Math.min(3, Math.round(v))) as Level;
    s.append(chip(levelLabel(l), LEVEL_COLOURS[l]));
  }
  return s;
}

/** The icon and chip for a speed limit, on the Speed display's colour. */
export function speedFact(kmh: number): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(SPEED_PATH, "Speed limit"), chip(speed(kmh), speedColour(kmh)));
  return s;
}

/** The Speed view's train speed (T-1271): words, since the chip beside it is a limit. */
export function trainSpeedFact(kmh: number): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append("Fastest train ", chip(speed(kmh), speedColour(kmh)));
  return s;
}

/** A mountain: the ground's terrain class (T-1270). */
const TERRAIN_PATH = "M1.5 13.5L6 6l2.4 3.4L10.6 6.5l3.9 7z";

/** Two waves: water. */
const WATER_PATH = "M1.5 6.5q1.6-1.6 3.25 0t3.25 0 3.25 0 3.25 0M1.5 10.5q1.6-1.6 3.25 0t3.25 0 3.25 0 3.25 0";

/** The icon and a chip for the roughest terrain class a route crosses where it is charged (T-1394),
 * with what it adds to the multiplier (T-1890), from the preview's metres per class
 * (`TrackApi.preview_ground`, flat to mountains); null on flat ground only. */
export function terrainFact(ground: ArrayLike<number> | undefined): HTMLSpanElement | null {
  if (!ground) return null;
  const k = [3, 2, 1].find((c) => (ground[c] ?? 0) >= 1);
  if (k === undefined) return null;
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(TERRAIN_PATH, "Terrain"), chip(`${TERRAIN_NAMES[k]} +${TERRAIN_SURCHARGES[k].toFixed(1)}x`, TERRAIN_COLOURS[k], 0.6));
  return s;
}

/** The icon and what water adds (any level) where a route crosses water (T-1887, the preview's
 * fifth figure); null on land only. */
export function waterFact(ground: ArrayLike<number> | undefined): HTMLSpanElement | null {
  if (!ground || (ground[4] ?? 0) < 1) return null;
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(WATER_PATH, "Water"), chip(`Water +${WATER_SURCHARGE}x`, WATER_COLOUR, 0.6));
  return s;
}

/** A road with its centre line dashed. */
const ROAD_PATH = "M5 1.5L3.5 14.5M11 1.5l1.5 13M8 2v2.2M8 6.9v2.2M8 11.8V14";
const ROAD_COLOUR = "#c9b28a";

/** The icon and a chip for the roads a route crosses at ground level and what they add (T-1929, the
 * preview's sixth and seventh figures: weighted count, US$M); null where it crosses none. */
export function roadFact(ground: ArrayLike<number> | undefined): HTMLSpanElement | null {
  const n = ground?.[5] ?? 0;
  if (!ground || n < 1) return null;
  const s = document.createElement("span");
  s.className = "tag-fact";
  // T-2263: the tutorial's longer wording
  s.append(icon(ROAD_PATH, "Roads"), chip(`${plural(n, guideVerbose.peek() ? "road crossing" : "crossing")} +${moneyShort((ground[6] ?? 0) * 1e6)}`, ROAD_COLOUR, 0.6));
  return s;
}

/** One line bullet: the markup and CSS of ui/widgets.tsx's Chip (a tag is plain DOM, not Preact). */
export function lineBullet(l: LineInput, small = false): HTMLSpanElement {
  const c = document.createElement("span");
  c.className = (small ? "chip sm" : "chip") + (idleLines.peek().has(l.id) ? " idle" : "");
  c.style.setProperty("--c", l.colour);
  c.style.setProperty("--ct", inkOn(l.colour));
  c.append(bulletNode(l.letter, small ? "sm" : "", isBusLine(l.id), isFerryLine(l.id)));
  return c;
}

/** A row of line bullets. */
export function lineBullets(lines: LineInput[]): HTMLDivElement {
  const d = document.createElement("div");
  d.className = "tag-lines";
  d.append(...lines.map((l) => lineBullet(l)));
  return d;
}

/** One line of the tag holding `facts`. */
export function factRow(...facts: HTMLElement[]): HTMLDivElement {
  const d = document.createElement("div");
  d.className = "tag-facts";
  d.append(...facts);
  return d;
}
