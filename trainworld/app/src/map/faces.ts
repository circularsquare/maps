// Faces rising from platforms as riders get off a train (T-686): a happy one, a tired one when the
// platform's riders waited long, a sad one when it is crowded (map/stationLife.ts decides), a few
// with a rare look (T-1154). Sprites drawn once per look, on a 2D canvas of their own over the network, moved on the map's own projection. The canvas is
// drawn only while a face is alive: an animation frame loop starts with the first face and stops
// with the last, so with none there is no work at all. Motion by design, like the trains.

import type { Map as MlMap } from "maplibre-gl";
import { sizeCanvasLike } from "./gl";
import type { Overlay } from "./overlay";

/** Faces alive at once, all platforms together. */
const MAX_ALIVE = 36;
/** Faces alive at once on one platform. */
export const MAX_PER_PLATFORM = 5;
/** A face's life, ms, and how far it rises, CSS px. */
const LIFE_MS = 2000;
const RISE_PX = 30;

/** A sprite's side, px, with room for a hat; a face's size on the map is that of a 48 px sprite
 * (T-686's), so a sprite is drawn SPRITE / 48 times it. */
const SPRITE = 64;
const SPRITE_SCALE = SPRITE / 48;

interface Face {
  station: string;
  lng: number;
  lat: number;
  /** metres east and north of the station's lng, lat */
  dx: number;
  dy: number;
  born: number;
  look: Look;
  /** sideways sway, px */
  sway: number;
}

export type Mood = "happy" | "tired" | "sad";

// No red: a red face reads as angry (T-802). Tired is a drained yellow, sad is blue.
const LOOK: Record<Mood, { fill: string; edge: string; ink: string }> = {
  happy: { fill: "#ffe2a0", edge: "#e2b45e", ink: "#74513c" },
  tired: { fill: "#e4dcc0", edge: "#b3a77c", ink: "#5f5640" },
  sad: { fill: "#b9d5f2", edge: "#6f9ed0", ink: "#2f4d78" },
};

/** T-1154: rare looks, each drawn once per face at spawn and kept for its life. An expression
 * replaces the mood's eyes and mouth (crying and dead only on sad faces, flat and sleepy only on the
 * others); a hat or sunglasses goes over any face but a dead or sleepy one's eyes; a child is drawn
 * at CHILD_SCALE. */
export type Expression = "" | "flat" | "sleepy" | "crying" | "dead";
export type Accessory = "" | "bucket" | "cap" | "shades";
export interface Look {
  mood: Mood;
  expr: Expression;
  acc: Accessory;
  child: boolean;
}
export const EXPR_CHANCE: Record<Exclude<Expression, "">, number> = { flat: 0.04, sleepy: 0.04, crying: 0.08, dead: 0.03 };
export const ACC_CHANCE: Record<Exclude<Accessory, "">, number> = { bucket: 0.025, cap: 0.025, shades: 0.025 };
export const CHILD_CHANCE = 0.06;
export const CHILD_SCALE = 0.72;

export function pickLook(mood: Mood, rand: () => number = Math.random): Look {
  let expr: Expression = "", acc: Accessory = "";
  const re = rand(), ra = rand();
  const exprs: Exclude<Expression, "">[] = mood === "sad" ? ["crying", "dead"] : ["flat", "sleepy"];
  let t = 0;
  for (const e of exprs) if (!expr && re < (t += EXPR_CHANCE[e])) expr = e;
  t = 0;
  for (const a of ["bucket", "cap", "shades"] as const) if (!acc && ra < (t += ACC_CHANCE[a])) acc = a;
  if (acc === "shades" && (expr === "dead" || expr === "sleepy" || expr === "crying")) acc = "";
  return { mood, expr, acc, child: rand() < CHILD_CHANCE };
}

export const lookKey = (l: Look) => `${l.mood}|${l.expr}|${l.acc}`;

const HATS = {
  bucket: { fill: "#a9d8b8", edge: "#5f9e76" },
  cap: { fill: "#f4a983", edge: "#c9714d" },
};

/** One face, its centre at SPRITE/2 across and 4 px under SPRITE/2 down, leaving room above for a
 * hat. */
export function sprite(look: Look | Mood): HTMLCanvasElement {
  const l: Look = typeof look === "string" ? { mood: look, expr: "", acc: "", child: false } : look;
  const { mood, expr, acc } = l;
  const c = document.createElement("canvas");
  c.width = c.height = SPRITE;
  const g = c.getContext("2d")!;
  const m = SPRITE / 2, mx = m, my = m + 4;
  const { fill, edge, ink } = LOOK[mood];
  g.lineCap = "round";
  g.lineJoin = "round";
  g.beginPath();
  g.arc(mx, my, FACE_R, 0, Math.PI * 2);
  g.fillStyle = fill;
  g.fill();
  g.lineWidth = 3;
  g.strokeStyle = edge;
  g.stroke();
  g.fillStyle = ink;
  g.strokeStyle = ink;
  g.lineWidth = 3;
  if (expr) drawExpression(g, expr, mx, my, ink, acc);
  else drawMood(g, mood, mx, my, edge, ink, acc === "shades");
  if (acc === "shades") drawShades(g, mx, my);
  else if (acc) drawHat(g, acc, mx, my);
  return c;
}

const FACE_R = 20;

function drawShades(g: CanvasRenderingContext2D, mx: number, my: number) {
  g.fillStyle = "#33313f";
  g.strokeStyle = "#33313f";
  g.lineWidth = 2;
  for (const x of [-8, 8]) {
    g.beginPath();
    g.ellipse(mx + x, my - 4, 6.4, 4.6, 0, 0, Math.PI * 2);
    g.fill();
  }
  g.beginPath(); g.moveTo(mx - 2, my - 5); g.lineTo(mx + 2, my - 5); g.stroke();
  g.beginPath(); g.moveTo(mx - 14, my - 5); g.lineTo(mx - 19, my - 7); g.moveTo(mx + 14, my - 5); g.lineTo(mx + 19, my - 7); g.stroke();
  g.fillStyle = "rgba(255,255,255,0.7)";
  for (const x of [-10, 6]) { g.beginPath(); g.ellipse(mx + x, my - 6, 1.6, 1.1, -0.5, 0, Math.PI * 2); g.fill(); }
}

function drawHat(g: CanvasRenderingContext2D, kind: "bucket" | "cap", mx: number, my: number) {
  const { fill, edge } = HATS[kind];
  g.fillStyle = fill;
  g.strokeStyle = edge;
  g.lineWidth = 2.4;
  if (kind === "bucket") {
    // a rounded crown over a wide drooping brim
    g.beginPath();
    g.moveTo(mx - 15, my - 13);
    g.lineTo(mx - 12, my - 25);
    g.quadraticCurveTo(mx, my - 30, mx + 12, my - 25);
    g.lineTo(mx + 15, my - 13);
    g.closePath();
    g.fill(); g.stroke();
    g.beginPath();
    g.moveTo(mx - 23, my - 9);
    g.quadraticCurveTo(mx, my - 19, mx + 23, my - 9);
    g.quadraticCurveTo(mx, my - 13, mx - 23, my - 9);
    g.closePath();
    g.fill(); g.stroke();
    g.strokeStyle = "rgba(255,255,255,0.55)";
    g.beginPath(); g.moveTo(mx - 13, my - 16.5); g.quadraticCurveTo(mx, my - 19.5, mx + 13, my - 16.5); g.stroke();
  } else {
    // a dome hugging the head, a bill out to the right and a button on top
    g.beginPath();
    g.moveTo(mx - 19, my - 10);
    g.bezierCurveTo(mx - 19, my - 30, mx + 19, my - 30, mx + 19, my - 10);
    g.quadraticCurveTo(mx, my - 14, mx - 19, my - 10);
    g.closePath();
    g.fill(); g.stroke();
    g.beginPath();
    g.moveTo(mx + 12, my - 12);
    g.quadraticCurveTo(mx + 26, my - 14, mx + 29, my - 9);
    g.quadraticCurveTo(mx + 22, my - 7, mx + 14, my - 9);
    g.closePath();
    g.fill(); g.stroke();
    g.fillStyle = edge;
    g.beginPath(); g.arc(mx, my - 25, 2.2, 0, Math.PI * 2); g.fill();
  }
}

function drawExpression(g: CanvasRenderingContext2D, expr: Exclude<Expression, "">, mx: number, my: number, ink: string, acc: Accessory) {
  const shades = acc === "shades";
  if (expr === "flat") {
    // expressionless: dash eyes and a straight mouth
    if (!shades) {
      g.beginPath();
      for (const x of [-8, 8]) { g.moveTo(mx + x - 3.5, my - 4); g.lineTo(mx + x + 3.5, my - 4); }
      g.stroke();
    }
    g.beginPath(); g.moveTo(mx - 7, my + 9); g.lineTo(mx + 7, my + 9); g.stroke();
  } else if (expr === "sleepy") {
    // closed eyes curving down, a small round mouth, a z where no hat is in the way
    g.lineWidth = 2.6;
    for (const x of [-8, 8]) { g.beginPath(); g.arc(mx + x, my - 5, 4, Math.PI * 0.15, Math.PI * 0.85); g.stroke(); }
    g.beginPath(); g.ellipse(mx, my + 9, 2.6, 3.2, 0, 0, Math.PI * 2); g.fill();
    if (!acc) {
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(mx + 15, my - 22); g.lineTo(mx + 21, my - 22); g.lineTo(mx + 15, my - 16); g.lineTo(mx + 21, my - 16);
      g.stroke();
    }
  } else if (expr === "crying") {
    // brows up, eyes squeezed shut, a wailing mouth and two streams of tears
    g.lineWidth = 2.4;
    g.beginPath();
    g.moveTo(mx - 14, my - 9); g.lineTo(mx - 5, my - 13);
    g.moveTo(mx + 14, my - 9); g.lineTo(mx + 5, my - 13);
    g.stroke();
    g.lineWidth = 2.6;
    for (const x of [-8, 8]) { g.beginPath(); g.arc(mx + x, my - 1, 3.6, Math.PI * 1.15, Math.PI * 1.85); g.stroke(); }
    g.fillStyle = "#eef7ff";
    for (const x of [-9, 9]) {
      g.beginPath();
      g.moveTo(mx + x - 1.8, my);
      g.lineTo(mx + x - 2.4, my + 14);
      g.quadraticCurveTo(mx + x, my + 17, mx + x + 2.4, my + 14);
      g.lineTo(mx + x + 1.8, my);
      g.closePath();
      g.fill();
    }
    g.fillStyle = ink;
    g.beginPath(); g.ellipse(mx, my + 10, 5, 4, 0, 0, Math.PI * 2); g.fill();
  } else {
    // cartoon knocked out: X eyes and a tongue poking out of a wobbly mouth
    g.lineWidth = 2.6;
    g.beginPath();
    for (const x of [-8, 8]) {
      g.moveTo(mx + x - 3.4, my - 7.4); g.lineTo(mx + x + 3.4, my - 0.6);
      g.moveTo(mx + x + 3.4, my - 7.4); g.lineTo(mx + x - 3.4, my - 0.6);
    }
    g.stroke();
    g.fillStyle = "#f29bab";
    g.beginPath(); g.ellipse(mx + 3, my + 11.5, 3.4, 4, 0, 0, Math.PI * 2); g.fill();
    g.beginPath();
    g.moveTo(mx - 7, my + 9); g.quadraticCurveTo(mx - 3.5, my + 6.5, mx, my + 9); g.quadraticCurveTo(mx + 3.5, my + 11.5, mx + 7, my + 9);
    g.stroke();
  }
}

/** The mood's own face about (x, y); its eyes left out under sunglasses. */
function drawMood(g: CanvasRenderingContext2D, mood: Mood, x: number, y: number, edge: string, ink: string, shades: boolean) {
  if (mood === "tired") {
    // heavy lids over half-closed eyes, bags under them, a flat mouth
    if (!shades) {
      for (const dx of [-8, 8]) {
        g.beginPath(); g.arc(x + dx, y - 3, 3.6, 0, Math.PI); g.fill();
        g.lineWidth = 2.6;
        g.beginPath(); g.moveTo(x + dx - 5.5, y - 3); g.lineTo(x + dx + 5.5, y - 3); g.stroke();
      }
      g.lineWidth = 1.8;
      g.strokeStyle = edge;
      for (const dx of [-8, 8]) { g.beginPath(); g.arc(x + dx, y + 1, 5, Math.PI * 0.2, Math.PI * 0.8); g.stroke(); }
      g.strokeStyle = ink;
    }
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(x - 6, y + 13); g.lineTo(x + 6, y + 13);
    g.stroke();
  } else if (mood === "sad") {
    // brows raised at the inner ends, round eyes, a small frown and a tear
    if (!shades) {
      g.lineWidth = 2.4;
      g.beginPath();
      g.moveTo(x - 14, y - 6); g.lineTo(x - 5, y - 11);
      g.moveTo(x + 14, y - 6); g.lineTo(x + 5, y - 11);
      g.stroke();
      for (const dx of [-8, 8]) { g.beginPath(); g.arc(x + dx, y - 1, 2.8, 0, Math.PI * 2); g.fill(); }
    }
    g.lineWidth = 3;
    g.beginPath();
    g.arc(x, y + 17, 7, Math.PI * 1.2, Math.PI * 1.8);
    g.stroke();
    g.fillStyle = "#ffffff";
    g.beginPath();
    g.ellipse(x + 11, y + 7, 2.4, 3.4, 0, 0, Math.PI * 2);
    g.fill();
  } else {
    if (!shades) for (const dx of [-8, 8]) { g.beginPath(); g.arc(x + dx, y - 4, 2.8, 0, Math.PI * 2); g.fill(); }
    g.beginPath();
    g.arc(x, y + 3, 9, Math.PI * 0.15, Math.PI * 0.85);
    g.stroke();
    g.fillStyle = "rgba(244,130,130,0.45)";
    for (const dx of [-14, 14]) { g.beginPath(); g.arc(x + dx, y + 4, 3.6, 0, Math.PI * 2); g.fill(); }
  }
}

export class Faces {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  /** sprites by `lookKey`, drawn the first time a look is needed */
  private sprites = new Map<string, HTMLCanvasElement>();
  private faces: Face[] = [];
  private raf = 0;

  constructor(private map: MlMap, private overlay: Overlay) {
    map.on("resize", () => this.canvas && this.size());
    overlay.afterMapFrame.push(() => {
      if (this.faces.length) this.draw();
    });
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
  spawn(station: string, lng: number, lat: number, length: number, heading: number, count: number, mood: Mood) {
    const room = Math.min(count, MAX_PER_PLATFORM - this.on(station), MAX_ALIVE - this.faces.length);
    if (room <= 0) return;
    this.ensure();
    const now = performance.now();
    const half = Math.max(10, length * 0.4);
    for (let i = 0; i < room; i++) {
      const along = (Math.random() * 2 - 1) * half, across = (Math.random() * 2 - 1) * 2;
      const dx = Math.cos(heading) * along - Math.sin(heading) * across, dy = Math.sin(heading) * along + Math.cos(heading) * across;
      this.faces.push({ station, lng, lat, dx, dy, born: now + i * 160 + Math.random() * 220, look: pickLook(mood), sway: (Math.random() * 2 - 1) * 5 });
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
    this.size();
  }

  private size() {
    sizeCanvasLike(this.canvas!, this.map);
  }

  private wipe() {
    if (this.canvas && this.ctx) {
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  /** The animation loop: a draw on each frame while faces are alive, but on MapLibre's frames from
   * `afterMapFrame`, so the faces move with the camera of the frame they are shown in (T-1418). */
  private frame() {
    this.raf = 0;
    if (!this.overlay.mapFrameQueued()) this.draw();
    if (this.faces.length) this.raf = requestAnimationFrame(() => this.frame());
  }

  private draw() {
    const ctx = this.ctx, sp = this.sprites;
    if (!ctx || !this.canvas) return;
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
      const key = lookKey(f.look);
      let img = sp.get(key);
      if (!img) sp.set(key, (img = sprite(f.look)));
      const s = size * SPRITE_SCALE * (f.look.child ? CHILD_SCALE : 1);
      // the face sits 4 sprite px under the sprite's centre
      ctx.drawImage(img, x - s / 2, y - s / 2 - (4 * s) / SPRITE, s, s);
    }
    ctx.globalAlpha = 1;
  }
}
