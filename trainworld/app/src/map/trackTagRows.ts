// The elevation and speed limit in the map's tags (T-526): the track hover and the track tool's
// placing tag write them the same way, an icon then the value on its display colour.

import { t, tRich } from "../i18n";
import type { ComponentChild, ComponentChildren } from "preact";
import { distance, levelLabel, moneyShort, speed } from "../game/format";
import { bulletNode, isBusLine, isFerryLine } from "../ui/bullet";
import { describeIssues } from "../game/issues";
import { idleLines } from "../game/lineRunning";
import { busSpeedColour, ferrySpeedColour, inkOn, LEVEL_COLOURS, rgb, speedColour } from "../game/palette";
import { BASE_M_PER_KM, trackMult } from "../ui/trackPrice";
import { TERRAIN_COLOURS, TERRAIN_NAMES, TERRAIN_SURCHARGES, WATER_SURCHARGE } from "../game/terrain";
import { guideVerbose, highSpeed, trackType, type TrackType } from "../game/ui";
import type { Level, LineInput } from "../game/types";
import type { PreviewAnswer } from "../workers/clockClient";
import { onScreen, placeAt } from "./screenPlace";

const WATER_COLOUR = "#7fb3e0";

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
 * growing above it, light green growing below it down to 0.5x, at most 0.5 strong
 * (T-2744: 0.8 there outshone 1x, most in dark mode). The track tool's tag and the Build tab's price. */
export function costTint(mult: number): [string, number] | null {
  if (mult < 0.995) return [COST_GREEN, 0.3 + 0.2 * Math.min(1, (1 - mult) / 0.5)];
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
  s.append(icon(ELEVATION_PATH, t("trackTagRows.level")));
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
  s.append(icon(SPEED_PATH, t("trackTagRows.speedLimit")), chip(speed(kmh), speedColour(kmh)));
  return s;
}

/** The Speed view's train speed (T-1271): words, since the chip beside it is a limit; a bus or ferry
 * line's says its own vehicle (T-2452). */
export function trainSpeedFact(kmh: number, vehicle = "train"): HTMLSpanElement {
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(...kids(tRich("trackTagRows.fastest", { vehicle, speed: chip(speed(kmh), vehicle === "train" ? speedColour(kmh) : vehicle === "boat" ? ferrySpeedColour(kmh) : busSpeedColour(kmh)) as unknown as ComponentChild })));
  return s;
}

/** A message's children as DOM pieces: its text and the nodes passed in as values. */
export function kids(c: ComponentChildren): (string | Node)[] {
  return ([c] as unknown[]).flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false).map((x) => (typeof x === "object" ? (x as Node) : String(x)));
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
  s.append(icon(TERRAIN_PATH, t("trackTagRows.terrain")), chip(t("trackTagRows.terrainAdds", { terrain: TERRAIN_NAMES[k], mult: TERRAIN_SURCHARGES[k].toFixed(1) }), TERRAIN_COLOURS[k], 0.6));
  return s;
}

/** The icon and what water adds (any level) where a route crosses water (T-1887, the preview's
 * fifth figure); null on land only. */
export function waterFact(ground: ArrayLike<number> | undefined): HTMLSpanElement | null {
  if (!ground || (ground[4] ?? 0) < 1) return null;
  const s = document.createElement("span");
  s.className = "tag-fact";
  s.append(icon(WATER_PATH, t("trackTagRows.water")), chip(t("trackTagRows.waterAdds", { mult: WATER_SURCHARGE }), WATER_COLOUR, 0.6));
  return s;
}

/** A road with its centre line dashed. */
const ROAD_PATH = "M5 1.5L3.5 14.5M11 1.5l1.5 13M8 2v2.2M8 6.9v2.2M8 11.8V14";
const ROAD_COLOUR = "#c9b28a";
/** T-2743: the crossings chip by what they make cheapest for the stretch at ground level: a viaduct
 * (tan), a bored tunnel (light red-brown), else at grade, faded the further the crossings are from
 * paying for the cheaper of the two (`ROAD_FADE` of its strength gone with none) */
const ROAD_UNDER = "#c98a76", ROAD_FADE = 0.75;

/** T-2743: the cheapest way to build the ground-level stretch of a route that crosses roads costing
 * `crossM` US$M: at grade with them, a viaduct (cut and cover costs the same) or a bored tunnel,
 * from the levels' multipliers (trackPrice.ts) plus the route's mean terrain surcharge, which bored
 * tunnels do not pay; `share` the crossings over what the cheaper of the two adds. `groundKm` the
 * stretch's length. */
export function crossingCheapest(crossM: number, groundKm: number, ground: ArrayLike<number>, tt: TrackType): { best: "ground" | "up" | "down"; share: number } {
  let m = 0, ms = 0;
  for (let k = 0; k < 4; k++) { m += ground[k] ?? 0; ms += (ground[k] ?? 0) * TERRAIN_SURCHARGES[k]; }
  const s = m > 0 ? ms / m : 0;
  const up = trackMult(1, tt) - trackMult(0, tt), down = trackMult(-2, tt) - trackMult(0, tt) - s * trackMult(1, tt);
  const extra = BASE_M_PER_KM * groundKm * Math.min(up, down);
  if (extra > 0 && crossM <= extra) return { best: "ground", share: crossM / extra };
  return { best: down < up ? "down" : "up", share: 1 };
}

/** the crossings chip's colour and strength (T-2743): at grade the tan thins toward the tag's own
 * background, which is near white in the light theme and dark in the dark one */
export function crossingColour(c: ReturnType<typeof crossingCheapest>): [string, number] {
  const a = 0.6;
  return c.best === "down" ? [ROAD_UNDER, a] : c.best === "up" ? [ROAD_COLOUR, a] : [ROAD_COLOUR, a * (1 - ROAD_FADE * (1 - c.share))];
}

/** The share of a preview's length at ground level, from its points (x, y, height m triples); 1
 * with none */
function groundShare(pts: ArrayLike<number>): number {
  let all = 0, at = 0;
  for (let i = 3; i + 2 < pts.length; i += 3) {
    const d = Math.hypot(pts[i] - pts[i - 3], pts[i + 1] - pts[i - 2]);
    all += d;
    if (Math.round((pts[i + 2] + pts[i - 1]) / 16) === 0) at += d;
  }
  return all > 0 ? at / all : 1;
}

/** The icon and a chip for the roads a route crosses at ground level and what they add (T-1929, the
 * preview's sixth and seventh figures: weighted count, US$M); null where it crosses none. `groundKm`
 * the route's length at ground level, for the chip's colour (T-2743). */
export function roadFact(ground: ArrayLike<number> | undefined, groundKm: number): HTMLSpanElement | null {
  const n = ground?.[5] ?? 0;
  if (!ground || n < 1) return null;
  const [colour, alpha] = crossingColour(crossingCheapest(ground[6] ?? 0, groundKm, ground, trackType.peek()));
  const s = document.createElement("span");
  s.className = "tag-fact";
  // T-2263: the tutorial's longer wording
  s.append(icon(ROAD_PATH, t("trackTagRows.roads")), chip(t("trackTagRows.roadsAdd", { verbose: guideVerbose.peek(), count: Number(n.toPrecision(3)), cost: moneyShort((ground[6] ?? 0) * 1e6) }), colour, alpha));
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

/** The M1 train's top speed, km/h: straight track's limit (track-and-service.md 6.1). */
const TOP_KMH = 160;
/** High-speed track's straight limit, km/h (the sim's `V_HSR`, T-817). */
const HSR_KMH = 320;

/**
 * The track tool's tag (T-411, T-526): length, cost multiplier and price; then the elevation being
 * placed and the route's lowest speed limit. The multiplier is on `costTint`'s colour. When the
 * segment could not be built ending here, why first.
 */
export function placementTag(pv: PreviewAnswer, level: number): HTMLDivElement[] {
  const div = (...kids: (string | Node)[]) => {
    const d = document.createElement("div");
    d.append(...kids);
    return d;
  };
  const out: HTMLDivElement[] = [];
  if (!pv.ok) {
    const why = div(describeIssues(pv.issues) || t("trackTagRows.cannotPlace"));
    why.style.color = "var(--bad)"; // style.css's `.tool-tag.bad`, for this line only
    out.push(why);
  }
  const nv = pv.verts.length / 4;
  // the preview's limits already carry the track type's cap (T-995: high-speed track's is 320)
  let top = Infinity;
  if (pv.speeds?.length) {
    for (const v of pv.speeds) if (v >= 0) top = Math.min(top, v);
  } else for (let i = 1; i < nv - 1; i++) if (pv.verts[i * 4 + 2] > 0) top = Math.min(top, pv.verts[i * 4 + 3]);
  if (!Number.isFinite(top)) top = highSpeed.peek() ? HSR_KMH : TOP_KMH;
  if (pv.ok) {
    const tint = costTint(pv.mult);
    out.push(div(...kids(tRich("trackTagRows.lengthCost", { length: distance(pv.lengthM), mult: chip(`${pv.mult.toFixed(2)}x`, tint?.[0] ?? null, tint?.[1]) as unknown as ComponentChild, cost: moneyShort(pv.cost * 1e6) }))));
  }
  const [first, ...more] = placedLevels(pv.pts, level);
  out.push(factRow(elevationFact(first, ...more), speedFact(top)));
  const ground = [terrainFact(pv.ground), waterFact(pv.ground), roadFact(pv.ground, (pv.lengthM / 1000) * groundShare(pv.pts))].filter((f) => f !== null);
  if (ground.length) out.push(factRow(...ground));
  return out;
}

/** The elevations a segment passes, in order and each once: where it starts, the top of a
 * bridge (T-656) or the bottom of a dip, where it ends (T-707: a ramp names both its levels).
 * `pts` x, y, height (m) triples; `level` when there are none. */
function placedLevels(pts: ArrayLike<number>, level: number): number[] {
  const n = pts.length / 3;
  if (n < 1) return [level];
  const a = pts[2] / 8, b = pts[(n - 1) * 3 + 2] / 8;
  let hi = -Infinity, lo = Infinity;
  for (let i = 2; i < pts.length; i += 3) {
    hi = Math.max(hi, pts[i] / 8);
    lo = Math.min(lo, pts[i] / 8);
  }
  const seq = [a];
  if (hi > Math.max(a, b) + 0.25) seq.push(hi);
  if (lo < Math.min(a, b) - 0.25) seq.push(lo);
  seq.push(b);
  return [...new Set(seq.map(Math.round))];
}

/** A curve of at least this radius, m, is too gentle to slow a train and gets no label. */
const GENTLE_M = 5000;
/** at most this many curve labels on the screen at once */
const MAX_CURVE_LABELS = 300;

/** Whether a curve of `radius` m (0: none) with speed limit `kmh` is labelled. T-1898: a wide curve
 * on high-speed track is labelled while it holds trains under 320. */
export function curveLabelled(radius: number, kmh: number): boolean {
  return radius > 0 && (radius < GENTLE_M || (kmh > 160.5 && kmh < 319.5));
}

/** A curve's label: its middle in local units, and the curve's speed limit in km/h. */
export interface CurveLabel {
  x: number;
  y: number;
  kmh: number;
}

/**
 * One label per curve from a preview's vertices (x, y, radius, km/h each): a curve fitted as
 * several PIs (one turning more than a right angle) has the same radius at consecutive vertices
 * and gets one label, at its middle one.
 */
export function curveLabels(verts: Float64Array): CurveLabel[] {
  const out: CurveLabel[] = [];
  const nv = verts.length / 4;
  for (let i = 1; i < nv - 1; ) {
    const r = verts[i * 4 + 2];
    let j = i + 1;
    while (j < nv - 1 && r > 0 && Math.abs(verts[j * 4 + 2] - r) < 0.5) j++;
    const kmh = verts[i * 4 + 3];
    if (curveLabelled(r, kmh)) {
      const m = (i + j - 1) >> 1;
      out.push({ x: verts[m * 4], y: verts[m * 4 + 1], kmh });
    }
    i = j;
  }
  return out;
}

/** A curve label's content (T-417; the radius left out, T-530): the speed limit on the speed colour,
 * as the track tool's tag colours the speed. Set only when it changes, since the labels are placed
 * every frame. */
export function setCurveLabel(el: HTMLElement, kmh: number) {
  const key = String(kmh);
  if (el.dataset.curve === key && el.firstChild) return;
  el.dataset.curve = key;
  el.replaceChildren(chip(speed(kmh), speedColour(kmh)));
}

/**
 * Place curve labels on a `w` by `h` screen with `take(i)` giving the element for the i-th shown
 * one; returns how many are shown. Only those on screen, none over another: a snapped route can
 * have thousands of curves (T-727d), and zoomed out a long route keeps a few. `w` and `h` are the
 * overlay's size, since reading the container's would lay the page out again after this frame's
 * writes (T-1394).
 */
export function placeCurveLabels(labels: readonly CurveLabel[], toScreen: (x: number, y: number) => [number, number], w: number, h: number, take: (i: number) => HTMLElement): number {
  const shown: [number, number][] = [];
  for (const c of labels) {
    const [x, y] = toScreen(c.x, c.y);
    if (!onScreen(x, y, w, h, 40)) continue;
    if (shown.some(([u, v]) => Math.abs(u - x) < 64 && Math.abs(v - y) < 22)) continue;
    if (shown.length >= MAX_CURVE_LABELS) break;
    shown.push([x, y]);
    const el = take(shown.length - 1);
    setCurveLabel(el, c.kmh);
    placeAt(el, x, y, 8, 6);
  }
  return shown.length;
}

/** The curve labels of one tool, their elements made in `box` as needed. */
export class CurveLabelPool {
  private els: HTMLDivElement[] = [];

  constructor(private box: HTMLElement) {}

  /** Show `labels` (see `placeCurveLabels`) and hide the elements left over. */
  place(labels: readonly CurveLabel[], toScreen: (x: number, y: number) => [number, number], w: number, h: number) {
    const n = placeCurveLabels(labels, toScreen, w, h, (i) => {
      let l = this.els[i];
      if (!l) {
        l = this.els[i] = document.createElement("div");
        l.className = "pi-label";
        this.box.appendChild(l);
      }
      l.style.display = "";
      return l;
    });
    for (let i = n; i < this.els.length; i++) this.els[i].style.display = "none";
  }
}
