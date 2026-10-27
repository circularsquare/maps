// T-705, T-745: a line bullet drawn as one SVG, its circle (or pill) and its symbol together, the
// symbol's ink box centred in the circle. HTML text sits by the font's line box, so it read off
// centre (T-705). What is drawn, measured on the rendered pixels (checks/t745-bullets.mjs):
// - the ink box comes from a canvas (actualBoundingBox) at MEASURE_PX, scaled down: Chrome rounds
//   those values to whole pixels at the size measured (a 12 px "L" reads 9 high for 8.46);
// - the circle is drawn in the SVG rather than as the span's background: Chrome snaps a box's
//   background and an SVG's origin to whole device pixels separately, which moved the symbol up to a
//   pixel sideways against the circle;
// - text baselines snap to whole device pixels too, so the baseline is put on one (from the device
//   pixel ratio and the CSS zoom the bullet is drawn at) and the circle moved with it, less than half
//   a device pixel.
// Measured once per font, symbol, size and scale. Used by ui/widgets.tsx's Chip and, as plain DOM,
// by map/trackTagRows.ts's lineBullet.

import { signal } from "@preact/signals";
import { world } from "../game/world";

export type BulletSize = "" | "sm" | "lg";
/** The chip's font size, height and side padding by size, as style.css sets them. */
const FONT_PX: Record<BulletSize, number> = { "": 12, sm: 12, lg: 18 };
const HEIGHT: Record<BulletSize, number> = { "": 20, sm: 16, lg: 30 };
const PAD: Record<BulletSize, number> = { "": 5, sm: 4, lg: 7 };
const MEASURE_PX = 240;

/** Bumped when the page font, a face's loading or the device pixel ratio changes: components
 * reading it draw again. */
export const bulletEpoch = signal(0);

export interface BulletText {
  /** the bullet's width (a circle for one narrow symbol, else a pill), which is the SVG's */
  w: number;
  /** the bullet's height, which is the SVG's */
  h: number;
  /** the text origin's x, so the ink is centred across `w` */
  x: number;
  /** the baseline's y, on a whole device pixel */
  y: number;
  /** the circle's top, within half a device pixel of 0, so the ink is centred in it */
  top: number;
}

let ctx: CanvasRenderingContext2D | null = null;
let family = "";
const cache = new Map<string, BulletText | null>();
/** fonts and symbols asked of `document.fonts.load` once, so a face that never loads asks once */
const loading = new Set<string>();

function forget() {
  family = "";
  cache.clear();
  bulletEpoch.value++;
}
function watchRatio() {
  matchMedia(`(resolution: ${devicePixelRatio}dppx)`).addEventListener("change", () => {
    forget();
    watchRatio();
  }, { once: true });
}
/** T-1021: a face that loads drops only the symbols with a character in its range. A big game's CJK
 * place names load dozens of font subsets, and each cleared every symbol and drew every chip again. */
function faceLoaded(e: Event) {
  const faces = (e as Event & { fontfaces?: readonly FontFace[] }).fontfaces;
  if (!faces) return forget();
  const ranges: [number, number][] = [];
  for (const f of faces) {
    for (const part of (f.unicodeRange || "U+0-10FFFF").split(",")) {
      const m = /U\+([0-9A-F?]+)(?:-([0-9A-F]+))?/i.exec(part.trim());
      if (!m) continue;
      const lo = parseInt(m[1].replace(/\?/g, "0"), 16), hi = m[2] ? parseInt(m[2], 16) : parseInt(m[1].replace(/\?/g, "F"), 16);
      ranges.push([lo, hi]);
    }
  }
  const hits = (text: string) => { for (const ch of text) { const c = ch.codePointAt(0)!; if (ranges.some(([lo, hi]) => c >= lo && c <= hi)) return true; } return false; };
  let dropped = false;
  for (const key of cache.keys()) if (hits(key.split("|").slice(3).join("|"))) { cache.delete(key); dropped = true; }
  if (dropped) bump();
}
/** one redraw for the symbols measured in a fallback whose faces came in together */
let bumpSoon = false;
function bump() {
  if (bumpSoon) return;
  bumpSoon = true;
  setTimeout(() => { bumpSoon = false; bulletEpoch.value++; }, 50);
}
if (typeof document !== "undefined") {
  document.addEventListener("tw-fontchange", forget);
  document.fonts?.addEventListener?.("loadingdone", faceLoaded);
  if (typeof matchMedia !== "undefined") watchRatio();
}

/** The placement of `text` drawn at CSS `zoom`, or null where it cannot be measured (the caller
 * then uses plain text). */
export function bulletText(text: string, size: BulletSize, zoom = 1): BulletText | null {
  if (typeof document === "undefined" || !text) return null;
  ctx ??= document.createElement("canvas").getContext("2d");
  if (!ctx) return null;
  family ||= getComputedStyle(document.body).fontFamily;
  const scale = devicePixelRatio * zoom;
  const key = `${family}|${size}|${scale}|${text}`;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  const font = `700 ${MEASURE_PX}px ${family}`;
  ctx.font = font;
  ctx.textBaseline = "alphabetic";
  const m = ctx.measureText(text);
  const k = FONT_PX[size] / MEASURE_PX, h = HEIGHT[size];
  const w = Math.max(h, m.width * k + 2 * PAD[size]);
  const y = h / 2 + ((m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2) * k;
  const snapped = scale > 0 ? Math.round(y * scale) / scale : y;
  const v: BulletText = { w, h, x: w / 2 - ((m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2) * k, y: snapped, top: snapped - y };
  const ok = [v.w, v.x, v.y, v.top].every(Number.isFinite) && m.width > 0;
  // a face still loading would be measured in the fallback: kept until it loads, then measured again
  // (T-1021: a CJK symbol measured once a draw while its face loaded, 0.5 ms each)
  if (document.fonts && !document.fonts.check(font, text)) {
    const p = `${font}|${text}`;
    if (!loading.has(p)) {
      loading.add(p);
      const again = () => { for (const k of cache.keys()) if (k.endsWith(`|${text}`)) cache.delete(k); bump(); };
      void document.fonts.load(font, text).then(again, () => undefined);
    }
  }
  cache.set(key, ok ? v : null);
  return ok ? v : null;
}

/** `bulletText` when it is measured already; else null (the caller draws plain text for now) and it is
 * measured in idle time, and `bulletEpoch` bumped once done. T-1021: a big save's Lines tab measured
 * every new symbol in the load's longest task, about 0.5-1 ms a CJK symbol. */
export function bulletTextSoon(text: string, size: BulletSize, zoom = 1): BulletText | null {
  if (typeof document === "undefined" || !text) return null;
  family ||= getComputedStyle(document.body).fontFamily;
  const hit = cache.get(`${family}|${size}|${devicePixelRatio * zoom}|${text}`);
  if (hit !== undefined) return hit;
  queued.set(`${size}|${zoom}|${text}`, [text, size, zoom]);
  if (!measuring) {
    measuring = true;
    const run = (d?: IdleDeadline) => {
      const until = performance.now() + 8;
      let done = 0;
      for (const [k, [t, s, z]] of queued) {
        if (done++ && (d && !d.didTimeout ? d.timeRemaining() < 1 : performance.now() > until)) break;
        queued.delete(k);
        bulletText(t, s, z);
      }
      if (queued.size) next(run);
      else { measuring = false; bump(); }
    };
    next(run);
  }
  return null;
}
const queued = new Map<string, [string, BulletSize, number]>();
let measuring = false;
const next = (f: (d?: IdleDeadline) => void) => (typeof requestIdleCallback === "function" ? requestIdleCallback(f, { timeout: 500 }) : setTimeout(f, 16));

/** T-1464, T-1475: a bus line's bullet is a rounded square, a rail line's a circle (or pill): the
 * corner radius of a bus bullet as a share of its side, which the map's end squares use too. */
export const BUS_CORNER = 0.18;
export const bulletRadius = (h: number, square: boolean) => (square ? h * BUS_CORNER : h / 2);

/** Whether line `id` is a bus line, read as the world stands now. */
export const isBusLine = (id: string): boolean => !!world.peek()?.lineStats[id]?.bus;

const SVG = "http://www.w3.org/2000/svg";

/** The bullet as a DOM node (for the map's tags, which are not Preact and not zoomed); `square` for
 * a bus line's (`isBusLine`). */
export function bulletNode(text: string, size: BulletSize, square = false): Node {
  const b = bulletText(text, size);
  if (!b) {
    const s = document.createElement("span");
    s.textContent = text;
    return s;
  }
  const svg = document.createElementNS(SVG, "svg");
  svg.style.width = b.w + "px";
  svg.style.height = b.h + "px";
  const r = document.createElementNS(SVG, "rect");
  for (const [a, v] of [["y", b.top], ["width", b.w], ["height", b.h], ["rx", bulletRadius(b.h, square)]] as const) r.setAttribute(a, String(v));
  const t = document.createElementNS(SVG, "text");
  t.setAttribute("x", String(b.x));
  t.setAttribute("y", String(b.y));
  t.textContent = text;
  svg.append(r, t);
  return svg;
}
