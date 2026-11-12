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
// by map/trackTagRows.ts's lineBullet. Its outline is the line's shape (T-3319, `bulletDraw`).

import { signal } from "@preact/signals";
import type { LineShape } from "../game/lineShape";

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
  /** the ink box's width and half its height, which a diamond is sized round (T-3319) */
  iw: number;
  ih: number;
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
    // i18n-ignore (CSS)
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
  // i18n-ignore (CSS)
  const font = `700 ${MEASURE_PX}px ${family}`;
  ctx.font = font;
  ctx.textBaseline = "alphabetic";
  const m = ctx.measureText(text);
  const k = FONT_PX[size] / MEASURE_PX, h = HEIGHT[size];
  const w = Math.max(h, m.width * k + 2 * PAD[size]);
  const y = h / 2 + ((m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2) * k;
  const snapped = scale > 0 ? Math.round(y * scale) / scale : y;
  const v: BulletText = { w, h, x: w / 2 - ((m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2) * k, y: snapped, top: snapped - y,
    iw: (m.actualBoundingBoxRight + m.actualBoundingBoxLeft) * k, ih: ((m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) / 2) * k };
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

/** T-1835: the span of bold text at `fontPx` that is centred in a pill: from `up` above the baseline to
 * `down` below it, px. Latin text is its capital height (a descender, the y of "Keiyo", hangs below
 * the middle and the letters' bodies stay in it); other scripts are their own ink box. null where
 * unmeasured. The pill (GroupPill) puts the baseline where this box is centred, measured in the page
 * (the dock is zoomed, which rounds the line box, so the font's metrics cannot place it). */
export function inkBox(text: string, fontPx: number): { up: number; down: number } | null {
  if (typeof document === "undefined" || !text) return null;
  ctx ??= document.createElement("canvas").getContext("2d");
  if (!ctx) return null;
  family ||= getComputedStyle(document.body).fontFamily;
  // i18n-ignore (CSS)
  ctx.font = `700 ${MEASURE_PX}px ${family}`;
  ctx.textBaseline = "alphabetic";
  const k = fontPx / MEASURE_PX;
  const latin = /^[ -~À-ɏ]+$/.test(text);
  const m = ctx.measureText(latin ? "H" : text);
  const up = m.actualBoundingBoxAscent * k, down = latin ? 0 : m.actualBoundingBoxDescent * k;
  return Number.isFinite(up + down) ? { up, down } : null;
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

/** T-1464, T-1475: a square bullet's corner radius as a share of its side, which the map's bus end
 * squares use too. */
export const BUS_CORNER = 0.18;
/** T-3319: a square's side and a diamond's height as shares of a circle's, so the three look one size
 * side by side (a square of the circle's height looks bigger, a diamond of it smaller); the diamond
 * reaches past the bullet's box above and below. */
const SQUARE_SIDE = 0.94, DIAMOND_SIDE = 1.14;
/** T-2194: a pill is at least this many times as wide as it is tall, fully rounded. */
const PILL_W = 1.5;

/** A bullet as drawn in its shape: the SVG's size, the text's origin, and the outline, a rounded rect or
 * a path whose stroke (in the fill colour) rounds its corners. */
export interface BulletDraw {
  w: number;
  h: number;
  x: number;
  y: number;
  rect?: { y: number; w: number; h: number; r: number };
  path?: { d: string; stroke: number };
}

export function bulletDraw(b: BulletText, shape: LineShape): BulletDraw {
  const { h, top } = b;
  if (shape === "square") {
    const s = SQUARE_SIDE * h, w = Math.max(s, b.w - (h - s));
    return { w, h, x: b.x + (w - b.w) / 2, y: b.y, rect: { y: top + (h - s) / 2, w, h: s, r: s * BUS_CORNER } };
  }
  if (shape === "diamond") {
    // a symbol too wide for the diamond stretches it into a hexagon with the same slopes
    const d = DIAMOND_SIDE * h, r = 0.06 * h, w = Math.max(d, b.iw + 2 * (b.ih + 0.12 * h));
    const cy = top + h / 2, a = d / 2 - r, xs = [r, d / 2, w - d / 2, w - r].map((v) => v.toFixed(2));
    const up = (cy - a).toFixed(2), down = (cy + a).toFixed(2), mid = cy.toFixed(2);
    return { w, h, x: b.x + (w - b.w) / 2, y: b.y, path: { d: `M${xs[0]} ${mid}L${xs[1]} ${up}L${xs[2]} ${up}L${xs[3]} ${mid}L${xs[2]} ${down}L${xs[1]} ${down}Z`, stroke: 2 * r } };
  }
  const w = shape === "pill" ? Math.max(b.w, PILL_W * h) : b.w;
  return { w, h, x: b.x + (w - b.w) / 2, y: b.y, rect: { y: top, w, h, r: h / 2 } };
}

const SVG = "http://www.w3.org/2000/svg";

/** The bullet as a DOM node (for the map's tags and hovers, which are not Preact and not zoomed). */
export function bulletNode(text: string, size: BulletSize, shape: LineShape = "circle"): Node {
  const b0 = bulletText(text, size);
  if (!b0) {
    const s = document.createElement("span");
    s.textContent = text;
    return s;
  }
  const b = bulletDraw(b0, shape);
  const svg = document.createElementNS(SVG, "svg");
  svg.style.width = b.w + "px";
  svg.style.height = b.h + "px";
  let outline: Element;
  if (b.path) {
    outline = document.createElementNS(SVG, "path");
    outline.setAttribute("d", b.path.d);
    outline.setAttribute("stroke-width", String(b.path.stroke));
  } else {
    outline = document.createElementNS(SVG, "rect");
    const q = b.rect!;
    for (const [a, v] of [["y", q.y], ["width", q.w], ["height", q.h], ["rx", q.r]] as const) outline.setAttribute(a, String(v));
  }
  const t = document.createElementNS(SVG, "text");
  t.setAttribute("x", String(b.x));
  t.setAttribute("y", String(b.y));
  t.textContent = text;
  svg.append(outline, t);
  return svg;
}

