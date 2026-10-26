// Where people drive (T-1435, docs/ui-demand-views.md): the biggest car flows as soft arcs on a
// canvas of their own, over the demand bubbles and under the station ridership circles. Width is
// car trips a day, opacity the minutes a straight-line train would save them (game/drives.ts).
// Zoomed in, the focused zone's squares (the workers' tables, put together in the sums worker once
// per demand result); zoomed out, the market's corridors between zones. Redrawn with each MapLibre
// frame; the arcs in sight are picked again then, never per train frame.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity, CITIES, cityById } from "../game/cities";
import { cityDemandViews, demandView, driveArcs, eachCorridor, intercityDemand, type Corridors } from "../game/demand";
import { ARC, trainGain, corridorDriveMin, type DriveLevel } from "../game/drives";
import { forecast } from "../game/format";
import { EARTH_R, M_PER_DEG, M_PER_DEG_LAT } from "../game/frame";
import { loadPlacesAt, places } from "../game/placeList";
import { display, shade, shownDay, shownDayPhrase, stopEditing } from "../game/ui";
import { addFinder } from "./demandHover";
import { kAtLat, pxPerUnitAt, toLocal } from "./geo";
import { sizeCanvasLike } from "./gl";
import { CITY_MARKS_MIN_ZOOM, OVERVIEW_ZOOM } from "./intercityView";
import type { Overlay } from "./overlay";

/** Arcs drawn at most, the biggest in sight by car trips times minutes saved. */
const MAX_ARCS = 60;
/** The finest grouping whose squares are at least this wide on screen, CSS px. */
const SQUARE_PX = 110;
/** Width in CSS px: the square root of car trips a day over this, between the two bounds. */
const WIDTH_PER_ROOT = 45, WIDTH_MIN = 1, WIDTH_MAX = 14;
/** Opacity from `ALPHA_MIN` at no gain up to `ALPHA_MAX` at `FULL_GAIN_MIN` minutes or more. */
const ALPHA_MIN = 0.18, ALPHA_MAX = 0.7, FULL_GAIN_MIN = 30;
/** How far the arc bows out, as a share of its length. */
const BOW = 0.16;
/** How far short of its ends' centres an arc stops, CSS px: a share of the square's width on screen
 * up to a cap, and a fixed gap at a zone's centre zoomed out. */
const GAP_SHARE = 0.15, GAP_MAX_PX = 36, CORRIDOR_GAP_PX = 5;
/** Corridors longer than this are left out: the market routes drives within 1,500 km
 * (sim intercity/drives.rs), and a car's share beyond is the rule's alone. */
const CORRIDOR_MAX_KM = 1500;
/** A question that came back empty is asked again this often, this many times. */
const RETRY_MS = 3000, RETRIES = 10;

export const DRIVE_COLOURS = { light: "#c86400", dark: "#ffad45" };

const arcWidth = (car: number) => Math.min(WIDTH_MAX, Math.max(WIDTH_MIN, Math.sqrt(car) / WIDTH_PER_ROOT));
const arcAlpha = (gain: number) => ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * Math.min(1, gain / FULL_GAIN_MIN);

/** The corridors between zones as arcs (`ARC` floats, ends in longitude and latitude), by car trips
 * times minutes saved, the biggest first; both ways of a pair together, trips within a zone left
 * out. Each arc's two registry places beside it in `places`. */
function corridorArcs(c: Corridors): { arcs: Float64Array; places: Int32Array } {
  const centre = new Map<number, [number, number]>();
  for (const z of CITIES) if (z.place >= 0 && !centre.has(z.place)) centre.set(z.place, z.center);
  const sum = new Map<number, [number, number, number]>();
  const { from, to } = c.table;
  eachCorridor(c, (a, b) => a !== b && centre.has(a) && centre.has(b), (j, v) => {
    const a = Math.min(from[j], to[j]), b = Math.max(from[j], to[j]), key = a * 65536 + b, e = sum.get(key);
    if (e) e[2] += v.car;
    else sum.set(key, [a, b, v.car]);
  });
  const list: [number, number, number, number][] = [];
  for (const [a, b, car] of sum.values()) {
    if (!(car > 0)) continue;
    const km = greatCircleKm(centre.get(a)!, centre.get(b)!), gain = trainGain(corridorDriveMin(km), km);
    if (gain > 0 && km <= CORRIDOR_MAX_KM) list.push([a, b, car, gain]);
  }
  list.sort((p, q) => q[2] * q[3] - p[2] * p[3]);
  const top = list.slice(0, 2500), arcs = new Float64Array(top.length * ARC), pl = new Int32Array(top.length * 2);
  top.forEach(([a, b, car, gain], i) => {
    const pa = centre.get(a)!, pb = centre.get(b)!;
    arcs.set([pa[0], pa[1], pb[0], pb[1], car, gain], i * ARC);
    pl[2 * i] = a;
    pl[2 * i + 1] = b;
  });
  return { arcs, places: pl };
}

function greatCircleKm(a: [number, number], b: [number, number]): number {
  const r = Math.PI / 180, p1 = a[1] * r, p2 = b[1] * r;
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(((b[0] - a[0]) * r) / 2) ** 2;
  return (2 * EARTH_R * Math.asin(Math.min(1, Math.sqrt(h)))) / 1000;
}

/** How far a place's name reaches, by OSM kind: an arc's end takes the place nearest relative to
 * its reach, so a neighbourhood beside it beats a city's label further off. */
const REACH: Record<string, number> = { city: 15000, borough: 8000, town: 5000, suburb: 4000, quarter: 3000, village: 3000, neighbourhood: 2000, hamlet: 1500 };

/** Names for a point, best first (at most three), from the place lists loaded. */
function namesAt(lng: number, lat: number): string[] {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), out: { name: string; f: number }[] = [];
  for (const list of places.peek().values())
    for (const p of list) {
      const dx = (p.lng - lng) * kx, dy = (p.lat - lat) * M_PER_DEG_LAT;
      if (Math.abs(dx) > 20000 || Math.abs(dy) > 20000) continue;
      const f = Math.hypot(dx, dy) / (REACH[p.kind] ?? 2000);
      if (f <= 1.5) out.push({ name: p.name, f });
    }
  return out.sort((a, b) => a.f - b.f).slice(0, 3).map((o) => o.name);
}

/** An arc as last drawn: its ends and bow in CSS px, width, and what it stands for. */
interface Placed { x0: number; y0: number; cx: number; cy: number; x1: number; y1: number; w: number; car: number; gain: number; a: [number, number]; b: [number, number]; names?: [string, string] }

export class DriveArcs {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private placed: Placed[] = [];
  private cssW = 0;
  private cssH = 0;
  private dirty = true;
  private levels: DriveLevel[] | null = null;
  private levelsCity = "";
  private corridors: { arcs: Float64Array; places: Int32Array } | null = null;
  private ask = 0;

  constructor(private map: MlMap, private overlay: Overlay, below: HTMLCanvasElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay drives";
    below.after(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    this.size();
    map.on("resize", () => {
      this.size();
      this.draw();
    });
    overlay.afterMapFrame.push(() => this.draw());
    // the place lists name the arcs' ends; loaded for the view when it is on
    map.on("moveend", () => this.loadNames());
    effect(() => {
      const on = display.value.drives;
      const dv = demandView.value;
      const city = activeCity.value;
      shownDay.value;
      cityDemandViews.value; // the zone in view opened
      if (city !== this.levelsCity) this.levels = null;
      if (!on || !dv) return this.invalidate();
      this.loadNames();
      const ask = ++this.ask;
      // null: the workers moved on (the next demandView asks again) or the zone is not open yet,
      // asked again a little later
      const go = (tries: number) => void driveArcs().then((v) => {
        if (ask !== this.ask) return;
        if (!v) return void (tries > 0 && setTimeout(() => go(tries - 1), RETRY_MS));
        this.levels = v;
        this.levelsCity = city;
        this.invalidate();
      });
      go(RETRIES);
    });
    effect(() => {
      const r = intercityDemand.value;
      this.corridors = display.value.drives && r ? corridorArcs(r.corridors) : null;
      this.invalidate();
    });
    effect(() => {
      shade.value;
      stopEditing.value;
      shownDayPhrase.value;
      this.invalidate();
    });
    addFinder(15, (p) => {
      const at = this.map.project([p.lng, p.lat]);
      const hit = this.at(at.x, at.y);
      if (!hit) return null;
      const [na, nb] = hit.names ?? this.name(hit);
      return [`${forecast(hit.car)} drives ${shownDayPhrase.peek()} between ${na} and ${nb}`, `About ${Math.max(1, Math.round(hit.gain))} min faster by train`];
    });
  }

  private loadNames() {
    if (!display.peek().drives || this.map.getZoom() <= OVERVIEW_ZOOM) return;
    const c = this.map.getCenter();
    void loadPlacesAt(c.lng, c.lat).catch(() => undefined);
  }

  /** Names for a zone arc's two ends, different from each other. */
  private name(p: Placed): [string, string] {
    const zone = cityById(activeCity.peek()).name;
    const a = namesAt(p.a[0], p.a[1]), b = namesAt(p.b[0], p.b[1]);
    const na = a[0] ?? zone;
    const nb = b.find((n) => n !== na) ?? (na === zone ? "elsewhere in the zone" : zone);
    return [na, nb];
  }

  private size() {
    [this.cssW, this.cssH] = sizeCanvasLike(this.canvas, this.map);
  }

  /** The figures changed: draw again on the next animation frame. */
  invalidate() {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => this.dirty && this.draw());
  }

  draw() {
    this.dirty = false;
    const g = this.ctx, k = this.canvas.width / Math.max(1, this.cssW);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.placed = [];
    const zoom = this.map.getZoom();
    if (!display.peek().drives || stopEditing.peek() || zoom < CITY_MARKS_MIN_ZOOM || !this.overlay.cam) return;
    let arcs: Float64Array | null = null, pl: Int32Array | null = null, gap = CORRIDOR_GAP_PX;
    if (zoom <= OVERVIEW_ZOOM) {
      arcs = this.corridors?.arcs ?? null;
      pl = this.corridors?.places ?? null;
    } else if (this.levels?.length) {
      const lat = this.map.getCenter().lat, perM = pxPerUnitAt(zoom) * kAtLat(lat);
      const level = this.levels.find((l) => l.sideM * perM >= SQUARE_PX) ?? this.levels[this.levels.length - 1];
      arcs = level.arcs;
      // a square's arcs all end at its centre: stopped short of it, so the busiest square is an open
      // hub rather than a knot
      gap = Math.min(GAP_MAX_PX, GAP_SHARE * level.sideM * perM);
    }
    if (!arcs) return;
    const s0: [number, number] = [NaN, NaN], s1: [number, number] = [NaN, NaN];
    const W = this.cssW, H = this.cssH;
    for (let i = 0; i < arcs.length / ARC && this.placed.length < MAX_ARCS; i++) {
      const o = i * ARC, a: [number, number] = [arcs[o], arcs[o + 1]], b: [number, number] = [arcs[o + 2], arcs[o + 3]];
      const la = toLocal(a[0], a[1]), lb = toLocal(b[0], b[1]);
      this.overlay.toScreenInto(la[0], la[1], s0);
      this.overlay.toScreenInto(lb[0], lb[1], s1);
      if (Number.isNaN(s0[0]) || Number.isNaN(s1[0])) continue;
      // the western end first, so an arc always bows the same way whichever end the data names first
      let [x0, y0, x1, y1] = s0[0] <= s1[0] ? [s0[0], s0[1], s1[0], s1[1]] : [s1[0], s1[1], s0[0], s0[1]];
      const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
      if (len < 4 * gap || len < 4) continue;
      const cx = (x0 + x1) / 2 + dy * BOW, cy = (y0 + y1) / 2 - dx * BOW;
      // each end moved towards the bow along the curve's own direction there
      const h0 = Math.hypot(cx - x0, cy - y0), h1 = Math.hypot(cx - x1, cy - y1);
      [x0, y0, x1, y1] = [x0 + ((cx - x0) * gap) / h0, y0 + ((cy - y0) * gap) / h0, x1 + ((cx - x1) * gap) / h1, y1 + ((cy - y1) * gap) / h1];
      const w = arcWidth(arcs[o + 4]);
      // one end at least in sight: arcs only passing over the view are left for a view of their own
      const inside = (x: number, y: number) => x >= 0 && y >= 0 && x <= W && y <= H;
      if (!inside(x0, y0) && !inside(x1, y1)) continue;
      const p: Placed = { x0, y0, cx, cy, x1, y1, w, car: arcs[o + 4], gain: arcs[o + 5], a, b };
      if (pl) p.names = [placeName(pl[2 * i]), placeName(pl[2 * i + 1])];
      this.placed.push(p);
    }
    g.setTransform(k, 0, 0, k, 0, 0);
    g.strokeStyle = DRIVE_COLOURS[shade.peek()];
    g.lineCap = "round";
    // the biggest last, on top
    for (let i = this.placed.length - 1; i >= 0; i--) {
      const p = this.placed[i];
      g.globalAlpha = arcAlpha(p.gain);
      g.lineWidth = p.w;
      g.beginPath();
      g.moveTo(p.x0, p.y0);
      g.quadraticCurveTo(p.cx, p.cy, p.x1, p.y1);
      g.stroke();
    }
    g.globalAlpha = 1;
  }

  /** The arc under a point (CSS px in the map) as last drawn, the biggest of those it is on. */
  at(px: number, py: number): Placed | null {
    for (const p of this.placed) {
      const reach = p.w / 2 + 3;
      let qx = p.x0, qy = p.y0;
      for (let s = 1; s <= 24; s++) {
        const t = s / 24, u = 1 - t;
        const x = u * u * p.x0 + 2 * u * t * p.cx + t * t * p.x1, y = u * u * p.y0 + 2 * u * t * p.cy + t * t * p.y1;
        if (segmentDistance(px, py, qx, qy, x, y) <= reach) return p;
        qx = x;
        qy = y;
      }
    }
    return null;
  }
}

/** A registry place's name: its zone's. */
const placeNames = new Map<number, string>();
function placeName(place: number): string {
  if (!placeNames.size) for (const c of CITIES) if (c.place >= 0 && !placeNames.has(c.place)) placeNames.set(c.place, c.name);
  return placeNames.get(place) ?? "another place";
}

function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}
