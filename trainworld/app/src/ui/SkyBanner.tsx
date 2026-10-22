// The top bar's sky (T-861): a strip of the next day's sky at the place in view, 6 hours back and
// 18 ahead, sliding left as game time passes, with the demand level of each hour in a 2 px band
// under it. Drawn into one canvas from the clock's own `now()` on animation frames, never through
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
/** css px: sky, gap, band (the canvas is their sum high) */
const SKY = 12, GAP = 1, BAND = 2;
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

/** Sky by the sun's elevation (degrees): the colour at the top of the strip and at its horizon. */
const STOPS: [number, RGB, RGB][] = ([
  [-90, "#0b1026", "#101834"],
  [-18, "#0b1026", "#121a3a"],
  [-10, "#1b2350", "#35326a"],
  [-4, "#384078", "#b75d7c"],
  [0, "#5868a6", "#f0925a"],
  [4, "#7a9fd6", "#f6c07e"],
  [10, "#6eb0e6", "#bcdcf0"],
  [30, "#58a4e4", "#a8d4f2"],
  [90, "#58a4e4", "#a8d4f2"],
] as const).map(([e, a, b]) => [e, hex(a), hex(b)]);

function skyAt(e: number): [RGB, RGB] {
  let i = 1;
  while (i < STOPS.length - 1 && e > STOPS[i][0]) i++;
  const [e0, t0, b0] = STOPS[i - 1], [e1, t1, b1] = STOPS[i];
  const f = Math.min(1, Math.max(0, (e - e0) / (e1 - e0)));
  return [mix(t0, t1, f), mix(b0, b1, f)];
}

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
    // a fifth of a device pixel's movement is below what shows
    if (key === lastKey && Math.abs(now - last) < secPx * 0.2 && marks === lastMarks) return;
    lastKey = key; last = now; lastMarks = marks;
    const off = CLOCK_UTC_OFFSET + shift / 3600, scale = h / (SKY + GAP + BAND);
    const skyRows = Math.round(SKY * scale), bandFrom = h - Math.max(1, Math.round(BAND * scale));
    const dim = colours.dark ? 0.22 : 0;
    for (let x = 0; x < w; x++) {
      const t = now + (x + 0.5 - nowX) * secPx;
      let [top, bot] = skyAt(sunElevation(t, lon, lat, off));
      if (dim) { top = mix(top, colours.bg, dim); bot = mix(bot, colours.bg, dim); }
      for (let y = 0; y < skyRows; y++) {
        const c = mix(top, bot, skyRows > 1 ? y / (skyRows - 1) : 1), i = (y * w + x) * 4;
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
    ctx.fillStyle = HOLIDAY_COLOUR;
    for (const m of marks) {
      const x0 = nowX + (m.t0 - now) / secPx, x1 = nowX + (m.t1 - now) / secPx;
      if (x1 <= 0 || x0 >= w) continue;
      ctx.fillRect(x0, 0, x1 - x0, 2 * scale);
      ctx.fillRect(x0 - scale / 2, 0, scale, skyRows);
    }
    // now: a thin line, and the sun on it while it is up, higher as it climbs
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillRect(nowX - Math.max(1, Math.round(scale / 2)), 0, Math.max(1, Math.round(scale)), skyRows);
    const e = sunElevation(now, lon, lat, off), rad = 3 * scale;
    if (e > -4) {
      const f = Math.min(1, (e + 4) / 44), y = skyRows + rad * 0.6 - f * (skyRows - rad * 0.4);
      ctx.fillStyle = colours.dark ? "#f0cf6a" : "#ffd84d";
      ctx.beginPath(); ctx.arc(nowX, y, rad, 0, 2 * Math.PI); ctx.fill();
    }
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
