// T-705: where a line bullet's text sits so its ink, not the font's line box, is centred in the
// circle. Every bullet (ui/widgets.tsx's Chip, map/trackTagRows.ts's lineBullet) moves its text by
// `bulletShift`. The ink box is measured on a canvas for the page's current font and the baseline
// on a probe laid out with the chip's own CSS (a font's ascent and descent differ, and Nunito's
// kana come from Zen Maru Gothic), once per font, text and size.

import { signal } from "@preact/signals";

export type BulletSize = "" | "sm" | "lg";
/** The chip's font size by size, as style.css sets it. */
const FONT_PX: Record<BulletSize, number> = { "": 12, sm: 12, lg: 18 };

/** Bumped when the page font changes or finishes loading: components reading it draw again. */
export const bulletEpoch = signal(0);

let ctx: CanvasRenderingContext2D | null = null;
let family = "";
const baselines = new Map<string, number>();
const shifts = new Map<string, string>();

function forget() {
  family = "";
  baselines.clear();
  shifts.clear();
  bulletEpoch.value++;
}
if (typeof document !== "undefined") {
  document.addEventListener("tw-fontchange", forget);
  document.fonts?.addEventListener?.("loadingdone", forget);
}

/** The baseline's distance below the chip's centre for text of this size in the page font, px. */
function baselineBelowCentre(px: number): number {
  const hit = baselines.get(family + px);
  if (hit !== undefined) return hit;
  const chip = document.createElement("span");
  chip.className = "chip " + (px > 12 ? "lg" : "");
  chip.style.cssText = "position:absolute;left:-999px;top:0;visibility:hidden";
  const t = document.createElement("span");
  const mark = document.createElement("i");
  mark.style.cssText = "display:inline-block;width:0;height:0";
  t.append("x", mark);
  chip.append(t);
  document.body.append(chip);
  const c = chip.getBoundingClientRect();
  const m = mark.getBoundingClientRect();
  chip.remove();
  const v = m.bottom - (c.top + c.height / 2);
  baselines.set(family + px, v);
  return v;
}

/** The CSS `transform` that centres `text`'s ink in a bullet of this size. */
export function bulletShift(text: string, size: BulletSize): string {
  if (typeof document === "undefined" || !text) return "none";
  ctx ??= document.createElement("canvas").getContext("2d");
  if (!ctx) return "none";
  family ||= getComputedStyle(document.body).fontFamily;
  const px = FONT_PX[size];
  const key = `${family}|${size}|${text}`;
  const hit = shifts.get(key);
  if (hit !== undefined) return hit;
  ctx.font = `700 ${px}px ${family}`;
  ctx.textBaseline = "alphabetic";
  const m = ctx.measureText(text);
  const x = m.width / 2 - (m.actualBoundingBoxRight - m.actualBoundingBoxLeft) / 2;
  const inkMid = baselineBelowCentre(px) - (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2;
  const v = Number.isFinite(x) && Number.isFinite(inkMid) ? `translate(${x.toFixed(2)}px,${(-inkMid).toFixed(2)}px)` : "none";
  shifts.set(key, v);
  return v;
}
