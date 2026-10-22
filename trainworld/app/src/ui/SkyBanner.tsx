// The top bar's sky (T-861, T-882): a little scene of the next day at the place in view, 6 hours
// back and 18 ahead, sliding left as game time passes: flat sky colours by the sun's height, stars
// at night, the sun or moon peeking in from the top at now, grass and a track along the bottom
// with a small train running on it, and the demand level of each hour in a 2 px band under it. Drawn into one canvas from the clock's own `now()` on animation frames, never through
// Preact, and only when it has moved a fraction of a pixel, so it costs next to nothing between
// game minutes. A holiday in a country with the player's stations (T-862, game/holidays.ts) is a
// ribbon along the sky's top over its day, with a tick where it starts.

import { useEffect, useRef } from "preact/hooks";
import { clock } from "../game/clock";
import { holidaysNear, type HolidaySpan } from "../game/holidays";
import { CLOCK_UTC_OFFSET, viewShift } from "../game/localTime";
import { viewCentre } from "../game/listScope";
import { PERIODS, periodAt } from "../game/periods";
import { sunElevation } from "../game/sun";
import { perfOff } from "../perfFlags";
import type { Demand } from "../game/types";

const HOURS = 24, NOW_AT = 0.25;
/** css px: the scene (sky, grass, track), gap, band (the canvas is their sum high) */
const SKY = 18, GAP = 1, BAND = 2;
/** Where no view centre is known yet: New York. */
const HOME: [number, number] = [-74.0, 40.7];

/** the holiday ribbon's colour, both themes: apart from every sky and band colour */
export const HOLIDAY_COLOUR = "#ff5fa2";

type RGB = [number, number, number];
const hex = (s: string): RGB => {
  const m = /#([0-9a-f]{6})/i.exec(s);
  const n = m ? parseInt(m[1], 16) : 0x808080;
  return [n >> 16, (n >> 8) & 255, n & 255];
};
const mix = (a: RGB, b: RGB, f: number): RGB => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];

/** T-882 (Anita): flat, cute colours by the sun's elevation (degrees): dark blue night, a short pink
 * and warm sunrise or sunset, light blue day. One colour a column. */
const STOPS: [number, RGB][] = ([
  [-90, "#2c3d78"],
  [-6, "#2c3d78"],
  [-2.5, "#b986b6"],
  [1, "#ffb37e"],
  [5, "#9fd8f7"],
  [90, "#9fd8f7"],
] as const).map(([e, c]) => [e, hex(c)]);

function skyAt(e: number): RGB {
  let i = 1;
  while (i < STOPS.length - 1 && e > STOPS[i][0]) i++;
  const [e0, c0] = STOPS[i - 1], [e1, c1] = STOPS[i];
  return mix(c0, c1, Math.min(1, Math.max(0, (e - e0) / (e1 - e0))));
}
/** 0 at night to 1 by day: how lit the ground is */
const daylight = (e: number) => Math.min(1, Math.max(0, (e + 5) / 10));

const GROUND: [RGB, RGB] = [hex("#2f5d4a"), hex("#7cc46c")];
const TUFT: [RGB, RGB] = [hex("#25503e"), hex("#5cae56")];
const RAIL: [RGB, RGB] = [hex("#1c2733"), hex("#6a5a50")];
const css = (c: RGB) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

/** Scene rows, css px from the top: grass from GROUND_Y, the rail at RAIL_Y, tufts up to TUFT_H
 * above the grass. Scenery repeats every so many css px of time, so it slides with the sky. */
const GROUND_Y = 13, RAIL_Y = 15.5, TUFT_EVERY = 4, SLEEPER_EVERY = 3, STAR_EVERY = 7;
/** The train crosses the strip once every this many game seconds, left to right. */
const TRAIN_LAP = 3 * 3600, TRAIN_LEN = 13;

interface Colours { band: Record<Demand, RGB>; dark: boolean; bg: RGB }
function readColours(): Colours {
  const cs = getComputedStyle(document.documentElement);
  return {
    band: { high: hex(cs.getPropertyValue("--dl-high")), medium: hex(cs.getPropertyValue("--dl-medium")), low: hex(cs.getPropertyValue("--dl-low")) },
    dark: document.documentElement.dataset.shade === "dark",
    bg: hex(cs.getPropertyValue("--bg")),
  };
}

function startSky(cv: HTMLCanvasElement): () => void {
  const ctx = cv.getContext("2d")!;
  let img: ImageData | null = null, colours = readColours(), last = NaN, lastKey = "", raf = 0;
  let lastMarks: readonly HolidaySpan[] = holidaysNear.peek();
  const resize = (w: number, h: number) => {
    if (w < 1 || h < 1 || (img && w === cv.width && h === cv.height)) return;
    cv.width = w; cv.height = h;
    img = ctx.createImageData(w, h);
    lastKey = "";
    draw();
  };
  const ro = new ResizeObserver(([e]) => {
    const box = e.devicePixelContentBoxSize?.[0];
    resize(box ? box.inlineSize : Math.round(e.contentRect.width * devicePixelRatio), box ? box.blockSize : Math.round(e.contentRect.height * devicePixelRatio));
  });
  try { ro.observe(cv, { box: "device-pixel-content-box" }); } catch { ro.observe(cv); }
  const mo = new MutationObserver(() => { colours = readColours(); lastKey = ""; draw(); });
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-shade", "data-theme"] });

  function draw() {
    if (!img) return;
    const w = img.width, h = img.height, d = img.data;
    const shift = viewShift.peek(), [lon, lat] = viewCentre.peek() ?? HOME, marks = holidaysNear.peek();
    const now = clock.now() + shift, secPx = HOURS * 3600 / w, nowX = Math.round(NOW_AT * w);
    const key = `${w} ${h} ${shift} ${lon} ${lat}`;
    // a fifth of a device pixel's movement of the train (the fastest thing) is below what shows
    if (key === lastKey && Math.abs(now - last) < Math.min(secPx, TRAIN_LAP / w) * 0.2 && marks === lastMarks) return;
    lastKey = key; last = now; lastMarks = marks;
    const off = CLOCK_UTC_OFFSET + shift / 3600, scale = h / (SKY + GAP + BAND);
    const skyRows = Math.round(SKY * scale), bandFrom = h - Math.max(1, Math.round(BAND * scale));
    const groundRow = Math.round(GROUND_Y * scale), railRow = Math.floor(RAIL_Y * scale), railRows = Math.max(1, Math.round(scale));
    const dim = colours.dark ? 0.15 : 0;
    const shade = (c: RGB) => (dim ? mix(c, colours.bg, dim) : c);
    const lit = (pair: [RGB, RGB], l: number) => shade(mix(pair[0], pair[1], l));
    const elevAt = (x: number) => sunElevation(now + (x - nowX) * secPx, lon, lat, off);
    for (let x = 0; x < w; x++) {
      const t = now + (x + 0.5 - nowX) * secPx, e = sunElevation(t, lon, lat, off), l = daylight(e);
      const sky = shade(skyAt(e)), ground = lit(GROUND, l), rail = lit(RAIL, l);
      for (let y = 0; y < skyRows; y++) {
        const c = y < groundRow ? sky : y >= railRow && y < railRow + railRows ? rail : ground, i = (y * w + x) * 4;
        d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
      }
      for (let y = skyRows; y < bandFrom; y++) d[(y * w + x) * 4 + 3] = 0;
      // four samples across the pixel, so a change of level slides rather than steps
      let r = 0, g = 0, b = 0;
      for (let s = 0; s < 4; s++) {
        const c = colours.band[PERIODS[periodAt(t + (s - 1.5) / 4 * secPx)].demand];
        r += c[0]; g += c[1]; b += c[2];
      }
      for (let y = bandFrom; y < h; y++) {
        const i = (y * w + x) * 4;
        d[i] = r / 4; d[i + 1] = g / 4; d[i + 2] = b / 4; d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.save();
    ctx.beginPath(); ctx.rect(0, 0, w, skyRows); ctx.clip();
    // scenery fixed to times, so it slides with the sky: k-th item at time k * every css px
    const each = (everyCss: number, f: (x: number, k: number) => void) => {
      const step = everyCss * scale * secPx, k0 = Math.floor((now - nowX * secPx) / step) - 1;
      for (let k = k0; ; k++) {
        const x = nowX + (k * step - now) / secPx;
        if (x > w + 4 * scale) break;
        f(x, k);
      }
    };
    const hash = (k: number) => { const s = Math.sin(k * 12.9898) * 43758.5453; return s - Math.floor(s); };
    // stars where it is night
    each(STAR_EVERY, (x, k) => {
      const l = daylight(elevAt(x));
      if (l > 0.3) return;
      ctx.fillStyle = `rgba(255,250,225,${(0.9 * (1 - l / 0.3)).toFixed(2)})`;
      ctx.fillRect(x + hash(k) * 4 * scale, (1 + hash(k + 0.5) * (GROUND_Y - 5)) * scale, scale, scale);
    });
    // grass: triangles on the ground's edge, two heights
    each(TUFT_EVERY, (x, k) => {
      const tall = (k & 1 ? 1.6 : 2.6) * scale, base = groundRow;
      ctx.fillStyle = css(lit(TUFT, daylight(elevAt(x))));
      ctx.beginPath(); ctx.moveTo(x - 1.5 * scale, base); ctx.lineTo(x, base - tall); ctx.lineTo(x + 1.5 * scale, base); ctx.fill();
    });
    // sleepers under the rail
    each(SLEEPER_EVERY, (x) => {
      ctx.fillStyle = css(lit(RAIL, daylight(elevAt(x))));
      ctx.fillRect(x, railRow + railRows, scale * 0.8, scale);
    });
    ctx.fillStyle = HOLIDAY_COLOUR;
    for (const m of marks) {
      const x0 = nowX + (m.t0 - now) / secPx, x1 = nowX + (m.t1 - now) / secPx;
      if (x1 <= 0 || x0 >= w) continue;
      ctx.fillRect(x0, 0, x1 - x0, 2 * scale);
      ctx.fillRect(x0 - scale / 2, 0, scale, groundRow);
    }
    // the sun, or the moon, peeking in from the top at now
    const e = sunElevation(now, lon, lat, off);
    if (e > -3) {
      ctx.fillStyle = css(shade(hex("#ffd84d")));
      ctx.beginPath(); ctx.arc(nowX, -0.5 * scale, 5 * scale, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = css(shade(hex("#fff1a8")));
      ctx.beginPath(); ctx.arc(nowX, -0.5 * scale, 3 * scale, 0, 2 * Math.PI); ctx.fill();
    } else {
      ctx.fillStyle = css(shade(hex("#f6f0d4")));
      ctx.beginPath(); ctx.arc(nowX, 0, 4.5 * scale, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = css(shade(skyAt(e)));
      ctx.beginPath(); ctx.arc(nowX + 2.2 * scale, -1 * scale, 4 * scale, 0, 2 * Math.PI); ctx.fill();
    }
    // a small train running left to right along the track, a lap every TRAIN_LAP game seconds
    const len = TRAIN_LEN * scale, tx = ((now / TRAIN_LAP) % 1 + 1) % 1 * (w + len) - len;
    const top = railRow - 3.5 * scale, l = daylight(elevAt(tx + len / 2));
    ctx.fillStyle = css(shade(mix(hex("#9c3a33"), hex("#e2574a"), l)));
    ctx.beginPath(); ctx.roundRect(tx, top, len, 3.5 * scale, [scale, 1.6 * scale, scale * 0.4, scale * 0.4]); ctx.fill();
    ctx.fillStyle = l > 0.5 ? "#fff6e0" : "#ffd98a";
    for (let i = 0; i < 3; i++) ctx.fillRect(tx + (1.5 + i * 3.4) * scale, top + 0.8 * scale, 2 * scale, 1.2 * scale);
    ctx.restore();
  }
  const frame = () => { raf = requestAnimationFrame(frame); draw(); };
  // ?perfOff=clock (T-045, measurement only): drawn once, never moving
  if (!perfOff("clock")) raf = requestAnimationFrame(frame);
  return () => { cancelAnimationFrame(raf); ro.disconnect(); mo.disconnect(); };
}

export function SkyBanner({ skyTip, bandTip }: { skyTip: string; bandTip: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => startSky(ref.current!), []);
  return (
    <div class="sky">
      <canvas ref={ref} class="sky-canvas" />
      <i class="sky-hit" data-tip={skyTip} />
      <i class="band-hit" data-tip={bandTip} />
    </div>
  );
}
