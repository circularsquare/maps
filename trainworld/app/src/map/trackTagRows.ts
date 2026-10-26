// The elevation and speed limit in the map's tags (T-526): the track hover and the track tool's
// placing tag write them the same way, an icon then the value on its display colour.

import { levelLabel, speed } from "../game/format";
import { bulletNode } from "../ui/bullet";
import { inkOn, LEVEL_COLOURS, rgb, speedColour } from "../game/palette";
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
    s.style.background = `rgba(${r},${g},${b},${alpha})`;
    s.style.padding = "0 3px";
    s.style.borderRadius = "2px";
  }
  return s;
}

const COST_YELLOW = "#f2c200", COST_GREEN = "#8fd97a";
/** A cost multiplier's colour and strength (T-411, T-1159): yellow growing above 1x, light green
 * growing below it down to 0.5x, none at 1x. The track tool's tag and the Build tab's price. */
export function costTint(mult: number): [string, number] | null {
  if (mult > 1.005) return [COST_YELLOW, 0.2 + 0.6 * Math.min(1, (mult - 1) / 1.5)];
  if (mult < 0.995) return [COST_GREEN, 0.3 + 0.5 * Math.min(1, (1 - mult) / 0.5)];
  return null;
}
/** `costTint` as a CSS background, or undefined at 1x. */
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
  s.append(icon(ELEVATION_PATH, "Elevation"));
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

/** One line bullet: the markup and CSS of ui/widgets.tsx's Chip (a tag is plain DOM, not Preact). */
export function lineBullet(l: LineInput, small = false): HTMLSpanElement {
  const c = document.createElement("span");
  c.className = small ? "chip sm" : "chip";
  c.style.setProperty("--c", l.colour);
  c.style.setProperty("--ct", inkOn(l.colour));
  c.append(bulletNode(l.letter, small ? "sm" : ""));
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
