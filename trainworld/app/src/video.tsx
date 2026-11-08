// T-2774: `?video=1`, a view for recording the launch video with a screen recorder. game/ui.ts sets the
// display for this page only (line colours, no names or labels, no bubbles, platforms filling, trains
// unfilled, sound on) and main.ts leaves out the boundaries, the loading grey and the dark realms; this
// hides every panel and the pointer, and runs the camera.
//   `&at=lat,lng,zoom` the start camera (default: wherever the game opens)
//   `&end=lat,lng,zoom` where the zoom out ends (default: the start's centre, zoom 2.4 on the globe, 3 flat)
//   `&hold=5` seconds at the start, `&secs=15` seconds of zooming out
//   `&pace=2` all of it, game clock included, twice as slow (for a recording sped up afterwards)
//   `&spin=1.5` degrees a second the globe turns from a zoom level before the end, on until Esc (0 none; globe only)
// Keys: V runs it (the hold, the zoom out, then the start menu's logo fading in); Esc goes back to the start.
// The tiles along the way are fetched first (the camera flies the path once and comes back), and the
// map is told to keep them.
import { render } from "preact";
import type { Map as MlMap } from "maplibre-gl";
import { StartLogo } from "./ui/StartLogo";
import { clock } from "./game/clock";
import { display } from "./game/ui";
import { world } from "./game/world";

type Cam = { lng: number; lat: number; zoom: number };

const q = new URLSearchParams(location.search);
const triple = (k: string): Cam | null => {
  const v = (q.get(k) ?? "").split(",").map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? { lat: v[0], lng: v[1], zoom: v[2] } : null;
};
const num = (k: string, d: number) => { const v = Number(q.get(k)); return q.has(k) && Number.isFinite(v) && v >= 0 ? v : d; };
/** `&pace=2`: everything twice as slow, game and camera, for a recording sped up 2x afterwards */
const PACE = Math.max(0.1, num("pace", 1));
const HOLD = num("hold", 5);
const SECS = num("secs", 15);
/** seconds the zoom rate takes to build up, and to die away at the end */
const EASE_IN = 2.5, EASE_OUT = 1.2;
/** the share of the zoom out after which the centre may drift to `&end`'s */
const DRIFT_FROM = 0.8;
/** the logo's fade: seconds long, starting this long before the zoom stops (Anita, 2026-11-07) */
const FADE = 0.9, FADE_LEAD = 0.5;
/** degrees of longitude a second the globe turns at the end (`&spin=`, 0 none), and the seconds it takes to get up to that */
const SPIN = num("spin", 1.5), SPIN_EASE = 3;
/** degrees turned `u` seconds after the turning starts */
const spun = (u: number) => SPIN * (u < SPIN_EASE ? u / 2 - (SPIN_EASE / (2 * Math.PI)) * Math.sin((Math.PI * u) / SPIN_EASE) : SPIN_EASE / 2 + u - SPIN_EASE);

const CSS = `
html.tw-video, html.tw-video * { cursor: none !important; outline: none !important; }
html.tw-video #dock-root, html.tw-video #topbar-root, html.tw-video #tools-root, html.tw-video #debug-root,
html.tw-video .maplibregl-control-container { display: none !important; }
html.tw-video #map > *:not(.maplibregl-canvas-container):not(canvas):not(.tw-video-keep) { display: none !important; }
html.tw-video body > *:not(#game):not(.tw-video-logo) { display: none !important; }
html.tw-video.tw-video-run #map { pointer-events: none; }
.tw-video-logo { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; z-index: 99999; pointer-events: none; opacity: 0; }
`;

export function startVideo(map: MlMap) {
  const style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);
  document.documentElement.classList.add("tw-video");
  const at = triple("at"), end = triple("end");
  if (at) map.jumpTo({ center: [at.lng, at.lat], zoom: at.zoom });

  // tiles the camera has passed are kept, whatever their zoom (MapLibre keeps 5 zoom levels by default)
  const keepTiles = () => {
    // MapLibre resizes each cache from this on its next update
    const ms = (map as unknown as { style?: { tileManagers?: Record<string, { _maxTileCacheZoomLevels: number | null }> } }).style?.tileManagers ?? {};
    for (const m of Object.values(ms)) m._maxTileCacheZoomLevels = 40;
  };
  // the basemap's country and region lines
  const noBoundaries = () => {
    for (const l of map.getStyle()?.layers ?? []) if (/boundary|admin/i.test(l.id) && map.getLayoutProperty(l.id, "visibility") !== "none") map.setLayoutProperty(l.id, "visibility", "none");
  };
  map.on("styledata", noBoundaries);
  if (map.isStyleLoaded()) noBoundaries();

  const cam = (): Cam => ({ lng: map.getCenter().lng, lat: map.getCenter().lat, zoom: map.getZoom() });
  const endOf = (s: Cam): Cam => end ?? { lng: s.lng, lat: s.lat, zoom: display.peek().projection === "globe" ? 2.4 : 3 };
  /** the camera `t` seconds into the zoom out */
  const along = (s: Cam, e: Cam, t: number): Cam => {
    const D = Math.max(SECS, EASE_IN + EASE_OUT), a = EASE_IN, b = EASE_OUT;
    const vmax = 1 / (D - (a + b) / 2);
    const tt = Math.min(Math.max(t, 0), D);
    let p: number;
    if (tt < a) p = vmax * (tt / 2 - (a / (2 * Math.PI)) * Math.sin((Math.PI * tt) / a));
    else if (tt <= D - b) p = vmax * (a / 2 + (tt - a));
    else { const u = tt - (D - b); p = vmax * (a / 2 + (D - b - a) + u / 2 + (b / (2 * Math.PI)) * Math.sin((Math.PI * u) / b)); }
    p = Math.min(1, p);
    const d = p <= DRIFT_FROM ? 0 : (p - DRIFT_FROM) / (1 - DRIFT_FROM), k = d * d * (3 - 2 * d);
    return { zoom: s.zoom + (e.zoom - s.zoom) * p, lng: s.lng + (e.lng - s.lng) * k, lat: s.lat + (e.lat - s.lat) * k };
  };
  const put = (c: Cam) => map.jumpTo({ center: [c.lng, c.lat], zoom: c.zoom });

  let start: Cam | null = null, fetched: string | null = null, gen = 0, busy = false;
  let pre: Promise<void> = Promise.resolve();
  const key = (c: Cam) => [c.lng, c.lat, c.zoom].map((v) => v.toFixed(5)).join();
  const idle = () => new Promise<void>((res) => {
    const t = setTimeout(res, 8000);
    map.once("idle", () => { clearTimeout(t); res(); });
    map.triggerRepaint();
  });
  /** flies the path in steps of half a zoom, waiting at each for its tiles, then back to the start */
  const prefetch = async (s: Cam, g: number) => {
    keepTiles();
    const e = endOf(s), n = Math.ceil(Math.abs(s.zoom - e.zoom) * 2);
    for (let i = 0; i <= n && g === gen; i++) {
      // the zoom's steps, with the centre where the run has it at that zoom
      const p = i / n, z = s.zoom + (e.zoom - s.zoom) * p;
      const d = p <= DRIFT_FROM ? 0 : (p - DRIFT_FROM) / (1 - DRIFT_FROM), k = d * d * (3 - 2 * d);
      put({ zoom: z, lng: s.lng + (e.lng - s.lng) * k, lat: s.lat + (e.lat - s.lat) * k });
      await idle();
    }
    // the globe's side that turns into view (a minute's turning, and more)
    if (SPIN > 0 && display.peek().projection === "globe") {
      for (let d = 30; d <= Math.min(180, SPIN * 70) && g === gen; d += 30) { put({ ...e, lng: e.lng - d }); await idle(); }
    }
    if (g !== gen) return;
    put(s);
    await idle();
    fetched = key(s);
  };

  let logo: HTMLDivElement | null = null;
  const dropLogo = () => { if (logo) { render(null, logo); logo.remove(); logo = null; } };
  const run = async () => {
    if (busy) return;
    busy = true;
    try { await pre; } catch { /* fetched again below */ } finally { busy = false; }
    const g = ++gen;
    dropLogo();
    start = cam();
    if (fetched !== key(start)) await prefetch(start, g);
    if (g !== gen) return;
    document.documentElement.classList.add("tw-video-run");
    if (PACE === 1) clock.setSpeed(0); // normal speed
    else clock.setMultiple(1 / PACE);
    const s = start, e = endOf(s), D = Math.max(SECS, EASE_IN + EASE_OUT);
    // on the globe, from a zoom level before the end, the Earth turns west to east and keeps turning
    let spinFrom = Infinity;
    if (SPIN > 0 && display.peek().projection === "globe") for (let t = 0; t <= D; t += 0.01) if (along(s, e, t).zoom <= e.zoom + 1) { spinFrom = t; break; }
    // the logo is set up unseen while the camera holds still: setting it up mid-zoom cost a 100 ms frame
    logo = document.createElement("div");
    logo.className = "tw-video-logo";
    document.body.appendChild(logo);
    render(<StartLogo />, logo);
    const t0 = performance.now();
    const frame = () => {
      if (g !== gen) return;
      const t = (performance.now() - t0) / 1000 / PACE - HOLD;
      if (t > 0) {
        const c = along(s, e, t);
        if (t > spinFrom) c.lng -= spun(t - spinFrom);
        put(c);
      }
      if (logo) logo.style.opacity = String(Math.min(1, Math.max(0, (t - D + FADE_LEAD) / FADE)));
      if (t < D - FADE_LEAD + FADE + 0.1 ||spinFrom < Infinity) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };
  const reset = () => {
    gen++;
    dropLogo();
    document.documentElement.classList.remove("tw-video-run");
    if (start) put(start);
  };

  window.addEventListener("keydown", (ev) => {
    const el = ev.target as HTMLElement | null;
    if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
    if (ev.key === "v" || ev.key === "V") void run();
    else if (ev.key === "Escape") reset();
    else return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
  }, { capture: true });

  // the path from `&at` is fetched as soon as the game is on screen
  if (at) {
    const wait = setInterval(() => {
      if (!world.peek() || !map.isStyleLoaded()) return;
      clearInterval(wait);
      start = cam();
      pre = prefetch(start, gen);
    }, 500);
  }
  (window as unknown as { twVideo: unknown }).twVideo = { run, reset, along };
}
