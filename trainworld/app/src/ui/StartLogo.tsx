// T-1567: the start menu's logo (`?logo=1`), "trains" over "world" as a little network with trains running, drawn by the
// game's own NetworkRenderer into a canvas of its own (map/logoNetwork.ts builds the network). The
// camera is a flat one of the logo's own, as the complex map's snapshot has (overlay.ts); the whole
// drawing is scaled to the screen by drawing it at a smaller or larger pixel ratio, so lines, boxes and
// trains keep their proportions at LOGO_ZOOM.

import { useEffect, useRef } from "preact/hooks";
import { NetworkRenderer, type DrawOptions } from "../map/renderer";
import { LOGO_ZOOM, logoNetwork } from "../map/logoNetwork";

/** game seconds a second: the trains' pace in the menu */
const TIME_SCALE = 7;
/** the logo's largest scale on a wide screen, of its design size */
const MAX_SCALE = 1.05;
const OPTIONS: DrawOptions = { showTrains: true, byLevel: false, selLine: -1, selEdge: -1, selStation: -1, flatMarks: false, hollowTrains: true };

/** what the checks read (`window.twLogo`): frames drawn, their draws' ms on the CPU, and a stop for
 * measuring the menu without the logo */
export const logoStats = { frames: 0, drawMs: 0, setupMs: 0, firstAt: 0, stop: () => {} };
(window as unknown as { twLogo: typeof logoStats }).twLogo = logoStats;

export function StartLogo() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const s0 = performance.now();
    const canvas = ref.current;
    // no stencil: the renderer then draws no track through the station boxes, so they are plain white
    const gl = canvas?.getContext("webgl2", { antialias: true, premultipliedAlpha: true, alpha: true, stencil: false });
    if (!canvas || !gl) return;
    const r = new NetworkRenderer();
    try { r.init(gl); } catch { return; }
    const net = logoNetwork();
    r.setNetwork(net);
    r.setTrips(0, 1, net.trips, undefined, net.ends);
    let scale = 1, raf = 0;
    const t0 = performance.now();
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const fit = () => {
      const avail = Math.max(120, Math.min(window.innerWidth - 32, 1400));
      scale = Math.min(MAX_SCALE, avail / net.w);
      const dpr = window.devicePixelRatio || 1;
      const w = net.w * scale, h = net.h * scale;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    };
    const draw = () => {
      raf = still ? 0 : requestAnimationFrame(draw);
      if (gl.isContextLost()) return;
      const a = performance.now();
      const dpr = (window.devicePixelRatio || 1) * scale;
      const W = canvas.width, H = canvas.height;
      // Mercator 0..1 to clip space at LOGO_ZOOM, device px at `dpr`, centred on the drawing
      const S = 512 * 2 ** LOGO_ZOOM * dpr, kx = (2 * S) / W, ky = (2 * S) / H;
      const merc = [kx, 0, 0, 0, 0, -ky, 0, 0, 0, 0, 1, 0, -net.mx * kx, net.my * ky, 0, 1];
      // a still frame starts part way through the timetable, so every line has a train out
      const time = (300 + ((a - t0) / 1000) * TIME_SCALE) % net.period;
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      r.draw({ merc, globe: null, plane: [0, 0, 0, 1], t: 0 }, time, LOGO_ZOOM, dpr, OPTIONS);
      logoStats.frames++;
      logoStats.drawMs += performance.now() - a;
    };
    fit();
    logoStats.setupMs = performance.now() - s0;
    logoStats.firstAt = performance.now();
    const onResize = () => { fit(); if (still) draw(); };
    window.addEventListener("resize", onResize);
    draw();
    logoStats.stop = () => cancelAnimationFrame(raf);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);
  return <canvas ref={ref} class="start-logo" role="img" aria-label="trainsworld" />;
}
