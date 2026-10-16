// The elevation and speed limit in the map's tags (T-526): the track hover and the track tool's
// placing tag write them the same way, an icon then the value on its display colour.

import { levelLabel, speed } from "../game/format";
import { LEVEL_COLOURS, rgb, speedColour } from "../game/palette";
import type { Level } from "../game/types";

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
export function elevationFact(level: number): HTMLSpanElement {
  const l = Math.max(-3, Math.min(3, Math.round(level))) as Level;
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(ELEVATION_PATH, "Elevation"), chip(levelLabel(l), LEVEL_COLOURS[l]));
  return s;
}

/** The icon and chip for a speed limit, on the Max speed display's colour. */
export function speedFact(kmh: number): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(SPEED_PATH, "Speed limit"), chip(speed(kmh), speedColour(kmh)));
  return s;
}

/** One line of the tag holding `facts`. */
export function factRow(...facts: HTMLElement[]): HTMLDivElement {
  const d = document.createElement("div");
  d.className = "tag-facts";
  d.append(...facts);
  return d;
}
