// Faces rising from platforms as riders get off a train (T-686): a happy one, or a red unhappy one
// when the platform's riders waited long (map/stationLife.ts decides). Two pre-drawn sprites on a
// 2D canvas of their own over the network, moved on the map's own projection. The canvas is
// drawn only while a face is alive: an animation frame loop starts with the first face and stops
// with the last, so with none there is no work at all. Motion by design, like the trains.

import type { Map as MlMap } from "maplibre-gl";
import type { Overlay } from "./overlay";

/** Faces alive at once, all platforms together. */
const MAX_ALIVE = 36;
/** Faces alive at once on one platform. */
export const MAX_PER_PLATFORM = 5;
/** A face's life, ms, and how far it rises, CSS px. */
const LIFE_MS = 2000;
const RISE_PX = 30;

const SPRITE = 48;

interface Face {
  station: string;
  lng: number;
  lat: number;
  /** metres east and north of the station's lng, lat */
  dx: number;
  dy: number;
  born: number;
  red: boolean;
  /** sideways sway, px */
  sway: number;
}

function sprite(red: boolean): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = SPRITE;
  const g = c.getContext("2d")!;
  const m = SPRITE / 2;
  const fill = red ? "#f48a8c" : "#ffe2a0", edge = red ? "#d4545e" : "#e2b45e", ink = red ? "#7c2a36" : "#74513c";
  g.lineCap = "round";
  g.beginPath();
  g.arc(m, m, m - 4, 0, Math.PI * 2);
  g.fillStyle = fill;
  g.fill();
  g.lineWidth = 3;
  g.strokeStyle = edge;
  g.stroke();
  g.fillStyle = ink;
  g.strokeStyle = ink;
  g.lineWidth = 3;
  if (red) {
    // slanted brows, round eyes, a small frown
    g.beginPath();
    g.moveTo(m - 13, m - 11); g.lineTo(m - 5, m - 7);
    g.moveTo(m + 13, m - 11); g.lineTo(m + 5, m - 7);
    g.stroke();
    for (const x of [-8, 8]) { g.beginPath(); g.arc(m + x, m - 1, 2.8, 0, Math.PI * 2); g.fill(); }
    g.beginPath();
    g.arc(m, m + 15, 7, Math.PI * 1.15, Math.PI * 1.85);
    g.stroke();
  } else {
    for (const x of [-8, 8]) { g.beginPath(); g.arc(m + x, m - 4, 2.8, 0, Math.PI * 2); g.fill(); }
    g.beginPath();
    g.arc(m, m + 3, 9, Math.PI * 0.15, Math.PI * 0.85);
    g.stroke();
    g.fillStyle = "rgba(244,130,130,0.45)";
    for (const x of [-14, 14]) { g.beginPath(); g.arc(m + x, m + 4, 3.6, 0, Math.PI * 2); g.fill(); }
  }
  return c;
}

export class Faces {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private sprites: [HTMLCanvasElement, HTMLCanvasElement] | null = null;
  private faces: Face[] = [];
  private raf = 0;

  constructor(private map: MlMap, private overlay: Overlay) {
    map.on("resize", () => this.canvas && this.size());
  }

  /** Faces alive on one platform now. */
  on(station: string): number {
    let n = 0;
    for (const f of this.faces) if (f.station === station) n++;
    return n;
  }

  get alive() {
    return this.faces.length;
  }

  /** `count` faces from the platform of `station` (centre `lng`, `lat`, `length` m along `heading`),
   * staggered over the next stretch of the train's stop. */
  spawn(station: string, lng: number, lat: number, length: number, heading: number, count: number, red: boolean) {
    const room = Math.min(count, MAX_PER_PLATFORM - this.on(station), MAX_ALIVE - this.faces.length);
    if (room <= 0) return;
    this.ensure();
    const now = performance.now();
    const half = Math.max(10, length * 0.4);
    for (let i = 0; i < room; i++) {
      const along = (Math.random() * 2 - 1) * half, across = (Math.random() * 2 - 1) * 2;
      const dx = Math.cos(heading) * along - Math.sin(heading) * across, dy = Math.sin(heading) * along + Math.cos(heading) * across;
      this.faces.push({ station, lng, lat, dx, dy, born: now + i * 160 + Math.random() * 220, red, sway: (Math.random() * 2 - 1) * 5 });
    }
    if (!this.raf) this.raf = requestAnimationFrame(() => this.frame());
  }

  /** Drop every face (the switch turned off, or the view left the platforms' zoom). */
  clear() {
    this.faces = [];
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.wipe();
  }

  private ensure() {
    if (this.canvas) return;
    const c = document.createElement("canvas");
    c.className = "overlay faces";
    this.overlay.canvas.after(c);
    this.canvas = c;
    this.ctx = c.getContext("2d");
    this.sprites = [sprite(false), sprite(true)];
    this.size();
  }

  private size() {
    const c = this.canvas!, dpr = this.map.getPixelRatio(), m = this.map.getCanvas();
    c.width = Math.round(m.clientWidth * dpr);
    c.height = Math.round(m.clientHeight * dpr);
    c.style.width = m.clientWidth + "px";
    c.style.height = m.clientHeight + "px";
  }

  private wipe() {
    if (this.canvas && this.ctx) {
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  private frame() {
    this.raf = 0;
    const ctx = this.ctx, sp = this.sprites;
    if (!ctx || !sp || !this.canvas) return;
    const now = performance.now(), dpr = this.map.getPixelRatio(), zoom = this.map.getZoom();
    this.faces = this.faces.filter((f) => now - f.born < LIFE_MS);
    this.wipe();
    if (!this.faces.length) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // a little larger the closer the view, from 14 px at the platforms' zoom to 20 px
    const size = 14 + 6 * Math.min(1, Math.max(0, (zoom - 14) / 3));
    // metres to degrees at the platform's latitude
    for (const f of this.faces) {
      const u = (now - f.born) / LIFE_MS;
      if (u < 0) continue;
      const lat = f.lat + f.dy / 110574, lng = f.lng + f.dx / (111320 * Math.cos((f.lat * Math.PI) / 180));
      const p = this.map.project([lng, lat]);
      const ease = 1 - (1 - u) * (1 - u);
      ctx.globalAlpha = Math.min(1, u * 8) * (u < 0.55 ? 1 : 1 - (u - 0.55) / 0.45) * 0.95;
      const x = p.x + f.sway * Math.sin(u * Math.PI * 2), y = p.y - RISE_PX * ease - 4;
      ctx.drawImage(sp[f.red ? 1 : 0], x - size / 2, y - size / 2, size, size);
    }
    ctx.globalAlpha = 1;
    this.raf = requestAnimationFrame(() => this.frame());
  }
}
