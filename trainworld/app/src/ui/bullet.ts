// T-705: a line bullet's text, drawn as SVG text with its ink box centred in the circle. HTML text
// sits by the font's line box (ascent and descent differ per font, and the browser rounds them at
// the UI scale), so it read off centre. Here the ink box is measured on a canvas for the page's
// current font (actualBoundingBox values, once per font, symbol and size) and the baseline is
// placed from it, so no line box is involved. Used by ui/widgets.tsx's Chip and, as plain DOM, by
// map/trackTagRows.ts's lineBullet.

import { signal } from "@preact/signals";

export type BulletSize = "" | "sm" | "lg";
/** The chip's font size and height by size, as style.css sets them. */
const FONT_PX: Record<BulletSize, number> = { "": 12, sm: 12, lg: 18 };
const HEIGHT: Record<BulletSize, number> = { "": 20, sm: 16, lg: 30 };

/** Bumped when the page font changes or finishes loading: components reading it draw again. */
export const bulletEpoch = signal(0);

export interface BulletText {
  /** the text's advance width, which is the SVG's width */
  w: number;
  /** the text origin's x, so the ink is centred across `w` */
  x: number;
  /** the chip's height, which is the SVG's */
  h: number;
  /** the baseline's y, which centres the ink vertically */
  y: number;
}

let ctx: CanvasRenderingContext2D | null = null;
let family = "";
const cache = new Map<string, BulletText | null>();

function forget() {
  family = "";
  cache.clear();
  bulletEpoch.value++;
}
if (typeof document !== "undefined") {
  document.addEventListener("tw-fontchange", forget);
  document.fonts?.addEventListener?.("loadingdone", forget);
}

/** The placement of `text`, or null where it cannot be measured (the caller then uses plain text). */
export function bulletText(text: string, size: BulletSize): BulletText | null {
  if (typeof document === "undefined" || !text) return null;
  ctx ??= document.createElement("canvas").getContext("2d");
  if (!ctx) return null;
  family ||= getComputedStyle(document.body).fontFamily;
  const key = `${family}|${size}|${text}`;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  ctx.font = `700 ${FONT_PX[size]}px ${family}`;
  ctx.textBaseline = "alphabetic";
  const m = ctx.measureText(text);
  const v: BulletText = {
    w: m.width,
    x: m.width / 2 - (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2,
    h: HEIGHT[size],
    y: HEIGHT[size] / 2 + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2,
  };
  const ok = [v.w, v.x, v.y].every(Number.isFinite) && v.w > 0;
  cache.set(key, ok ? v : null);
  return ok ? v : null;
}

const SVG = "http://www.w3.org/2000/svg";

/** The bullet's text as a DOM node (for the map's tags, which are not Preact). */
export function bulletTextNode(text: string, size: BulletSize): Node {
  const b = bulletText(text, size);
  if (!b) {
    const s = document.createElement("span");
    s.textContent = text;
    return s;
  }
  const svg = document.createElementNS(SVG, "svg");
  svg.style.width = b.w + "px";
  svg.style.height = b.h + "px";
  const t = document.createElementNS(SVG, "text");
  t.setAttribute("x", b.x.toFixed(2));
  t.setAttribute("y", b.y.toFixed(2));
  t.textContent = text;
  svg.append(t);
  return svg;
}
