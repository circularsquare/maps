// Snapping the track tool to the real network (T-727d, notes/T-727.md), called from tools.ts while
// the snap switch is on. Where a click would snap: the nearest drawn real line within 12 px (the
// layer's own rendered features, so only what is seen). With the segment's start within reach of
// real track and the pointer resting near real track for 120 ms, the router worker
// (workers/realRoute.worker.ts) finds the route and fits it; its real line is drawn as the ghost,
// then the fitted track's preview replaces it with the usual tag. Nothing joining the two is a red
// ghost and "No real track joins these points"; a pointer away from real track (or where there is no
// graph) is an ordinary segment, which tools.ts draws as before.

import type { Map as MlMap } from "maplibre-gl";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import type { PiInput, RouteEnd } from "../workers/protocol";
import type { FromRouter, ToRouter } from "../workers/realRoute.worker";
import type { Level } from "../game/types";
import { buildLevel, trackCount } from "../game/ui";
import { dataUrl } from "../game/dataBase";
import type { Overlay } from "./overlay";
import { strokesFromPreview } from "./network";
import { realReach } from "./realNetwork";
import { EARTH_C, toLocal } from "./geo";

/** the snap radius, px; and the pointer's rest before a route is searched, ms */
const SNAP_PX = 12;
const REST_MS = 120;
/** a pointer this far from the spot asked about is a new spot, px */
const SAME_PX = 4;
/** a search or fit taking longer shows the spinner in the tag, ms */
const SLOW_MS = 500;
/** the fit's tolerance, ground m (DECISIONS 2026-10-19) */
const TOL_M = 25;
/** graph and tiles can differ by a few metres; the router looks this much further, m */
const SLACK_M = 20;
/** the reach's vertices are kept this far apart, px (T-839) */
const REACH_STEP_PX = 1.5;
/** a point this near a covered region's outline counts as covered, degrees (T-913) */
const COVER_PAD = 0.05;

function ringBox(r: Float64Array): [number, number, number, number] {
  let w = 180, s = 90, e = -180, n = -90;
  for (let k = 0; k < r.length; k += 2) (w = Math.min(w, r[k])), (e = Math.max(e, r[k])), (s = Math.min(s, r[k + 1])), (n = Math.max(n, r[k + 1]));
  return [w, s, e, n];
}

function inRing(r: Float64Array, x: number, y: number) {
  let inside = false;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
    const yi = r[i + 1], yj = r[j + 1];
    if (yi > y !== yj > y && x < ((r[j] - r[i]) * (y - yi)) / (yj - yi) + r[i]) inside = !inside;
  }
  return inside;
}

/** Whether an edge of ring r passes within d degrees of (x, y), longitude scaled to ground. */
function nearRing(r: Float64Array, x: number, y: number, d: number) {
  const k = Math.cos((y * Math.PI) / 180), d2 = d * d;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
    const ax = (r[j] - x) * k, ay = r[j + 1] - y, bx = (r[i] - x) * k, by = r[i + 1] - y;
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
    if ((ax + t * dx) ** 2 + (ay + t * dy) ** 2 < d2) return true;
  }
  return false;
}

/** The route the second click places: its real ends (lng, lat) and the fitted corners between. */
export interface RealRoute { first: [number, number]; last: [number, number]; pis: PiInput[] }

/** What a click places: a route, a refusal ("No real track joins these points"), or nothing to do
 * with the real network (an ordinary segment). */
export type RealClick = { kind: "route"; route: RealRoute } | { kind: "refuse" } | null;

interface Spot {
  seq: number;
  px: number;
  py: number;
  lng: number;
  lat: number;
  t0: number;
  phase: "rest" | "search" | "fit" | "preview" | "done" | "none" | "off";
  line?: Float64Array;
  route?: RealRoute;
  preview?: PreviewAnswer;
  waiters: (() => void)[];
}

export class RealSnap {
  private worker: Worker | null = null;
  private seq = 0;
  /** the segment's start, if a drawn real line was within reach of it: lng, lat, radius m */
  private from: { lng: number; lat: number; r: number } | null = null;
  private spot: Spot | null = null;
  private timer = 0;
  /** the ring: where a click at the pointer would snap to real track */
  ringAt: [number, number] | null = null;
  /** timings of the last route, for checks (`window.tw.tools.real.last`) */
  last: { searchMs: number; states: number; fitMs: number; lengthM: number; corners: number; previewMs: number; worst: number } | null = null;

  /** the reach's last answer, for checks: the worker's ms, from asking to the answer, chains reachable and drawn */
  reachLast: { workerMs: number; ms: number; chains: number; drawn: number; points: number } | null = null;
  private reachSeq = 0;
  private reachAt = 0;

  constructor(
    private map: MlMap,
    private overlay: Overlay,
    private client: ClockClient,
    /** the route request (ends, corners, track) for a real route, as tools.ts builds them */
    private request: (r: RealRoute) => { from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean },
    /** show a preview answer as the drawing tool's own (null: none), with the request it answers */
    private show: (a: PreviewAnswer | null, req?: object) => void,
    private redraw: () => void,
  ) {
    map.on("moveend", () => {
      this.sendView();
      if (this.from) this.askReach();
    });
  }

  /** The real track reachable from the segment's start, for the part of the map in view and a
   * quarter of it around (T-839). */
  private askReach() {
    const f = this.from;
    if (!f) {
      if (realReach.peek() || this.reachAt) this.post({ kind: "reach", seq: ++this.reachSeq, a: null, ra: 0, box: [0, 0, 0, 0], step: 0 });
      this.reachAt = 0;
      realReach.value = null;
      return;
    }
    const b = this.map.getBounds(), pw = (b.getEast() - b.getWest()) / 4, ph = (b.getNorth() - b.getSouth()) / 4;
    this.reachAt = performance.now();
    this.post({
      kind: "reach", seq: ++this.reachSeq, a: [f.lng, f.lat], ra: f.r,
      box: [b.getWest() - pw, b.getSouth() - ph, b.getEast() + pw, b.getNorth() + ph],
      step: (REACH_STEP_PX * 360) / (512 * 2 ** this.map.getZoom()),
    });
  }

  private router(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL("../workers/realRoute.worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (e: MessageEvent<FromRouter>) => this.answer(e.data);
      this.post({ kind: "init", page: document.baseURI, graphs: dataUrl("realgraph/") });
    }
    return this.worker;
  }

  private post(m: ToRouter) {
    this.router().postMessage(m);
  }

  /** The view, so the regions in it are fetched before a click needs them. */
  sendView() {
    if (!this.on) return;
    const b = this.map.getBounds();
    this.post({ kind: "view", w: b.getWest(), s: b.getSouth(), e: b.getEast(), n: b.getNorth() });
  }

  private on = false;
  /** The snap switch with the track tool in hand. */
  setOn(on: boolean) {
    if (on === this.on) return;
    this.on = on;
    if (on) this.sendView();
    else (this.ringAt = null), this.reset();
  }

  /** Ground metres per screen px at a latitude. */
  private perPx(lat: number) {
    return (EARTH_C * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** this.map.getZoom());
  }

  /** The outlines of the regions that have a graph, from the router; null until it answers (T-913). */
  private cover: { r: Float64Array; box: [number, number, number, number] }[] | "all" | null = null;
  private coverCache = new Map<number, boolean>();

  /** Whether a real-network graph covers (lng, lat): inside a covered region's outline or within
   * COVER_PAD of one (outlines are coarse at coasts and borders). Kept per 0.01 degree cell. */
  covers(lng: number, lat: number): boolean {
    const c = this.cover;
    if (c === "all") return true;
    if (!c) return false;
    const key = (Math.floor(lng * 100) + 18000) * 20000 + Math.floor(lat * 100) + 9000;
    let v = this.coverCache.get(key);
    if (v === undefined) {
      v = c.some(({ r, box }) => lng >= box[0] && lng <= box[2] && lat >= box[1] && lat <= box[3] && inRing(r, lng, lat))
        || c.some(({ r, box }) => lng >= box[0] - COVER_PAD && lng <= box[2] + COVER_PAD && lat >= box[1] - COVER_PAD && lat <= box[3] + COVER_PAD && nearRing(r, lng, lat, COVER_PAD));
      this.coverCache.set(key, v);
    }
    return v;
  }

  /** The nearest point on a drawn real line within SNAP_PX of screen point (px, py), lng and lat;
   * null too where no graph covers it, so the track tool draws an ordinary segment there (T-913). */
  ring(px: number, py: number): [number, number] | null {
    const at = this.nearestDrawn(px, py);
    return at && this.covers(at[0], at[1]) ? at : null;
  }

  private nearestDrawn(px: number, py: number): [number, number] | null {
    const layers = this.map.getStyle().layers.filter((l) => l.id.startsWith("tw-real-")).map((l) => l.id);
    if (!layers.length) return null;
    const fs = this.map.queryRenderedFeatures([[px - SNAP_PX, py - SNAP_PX], [px + SNAP_PX, py + SNAP_PX]], { layers });
    let best = SNAP_PX * SNAP_PX, at: [number, number] | null = null;
    for (const f of fs) {
      const g = f.geometry;
      const lines = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : [];
      for (const l of lines) {
        let p0 = this.map.project(l[0] as [number, number]);
        for (let i = 1; i < l.length; i++) {
          const p1 = this.map.project(l[i] as [number, number]);
          const dx = p1.x - p0.x, dy = p1.y - p0.y, l2 = dx * dx + dy * dy;
          const t = l2 ? Math.max(0, Math.min(1, ((px - p0.x) * dx + (py - p0.y) * dy) / l2)) : 0;
          const x = p0.x + t * dx, y = p0.y + t * dy, d = (x - px) ** 2 + (y - py) ** 2;
          if (d < best) {
            best = d;
            const ll = this.map.unproject([x, y]);
            at = [ll.lng, ll.lat];
          }
          p0 = p1;
        }
      }
    }
    return at;
  }

  /** The pointer moved (screen px): the ring, and the route to it searched once it rests. */
  move(px: number, py: number) {
    this.ringAt = this.on ? this.ring(px, py) : null;
    if (!this.from || this.held) return;
    const s = this.spot;
    if (s && Math.hypot(s.px - px, s.py - py) < SAME_PX) return;
    this.drop();
    if (!this.ringAt) return;
    this.spot = { seq: ++this.seq, px, py, lng: this.ringAt[0], lat: this.ringAt[1], t0: 0, phase: "rest", waiters: [] };
    this.show(null);
    this.timer = setTimeout(() => this.ask(), REST_MS) as unknown as number;
  }

  /** The first click of a segment, at (lng, lat), or null when no real line was within reach. */
  begin(at: [number, number] | null) {
    this.drop();
    this.from = at ? { lng: at[0], lat: at[1], r: SNAP_PX * this.perPx(at[1]) + SLACK_M } : null;
    this.askReach();
  }

  /** The route between this segment's clicks is the real network's, not an ordinary segment's. */
  get active() {
    return !!this.from && !!this.spot && this.spot.phase !== "off";
  }

  /** The tag's rows while a real route is being found or was not found (null: the usual tag). */
  tag(): { text: string; bad: boolean; spin: boolean } | null {
    const s = this.spot;
    if (!this.active || !s) return null;
    if (s.phase === "none") return { text: "No real track joins these points", bad: true, spin: false };
    if (s.phase === "done") return null;
    return { text: "", bad: false, spin: s.t0 > 0 && performance.now() - s.t0 > SLOW_MS };
  }

  /** The second click at (px, py): waits for the route if it is still being found. */
  async click(px: number, py: number): Promise<RealClick> {
    if (!this.from) return null;
    this.move(px, py);
    const s = this.spot;
    if (!s || s.phase === "off") return null;
    if (s.phase === "rest") {
      clearTimeout(this.timer);
      this.ask();
    }
    if (s.phase !== "done" && s.phase !== "none") {
      // held while the click waits: moving the pointer meanwhile does not drop it
      this.held = true;
      await new Promise<void>((r) => s.waiters.push(r));
      this.held = false;
    }
    if (this.spot !== s || (s.phase as Spot["phase"]) === "off") return null;
    // a fitted route the track model refuses goes to the edit, which says why
    if (s.phase === "none" || !s.route) return { kind: "refuse" };
    return { kind: "route", route: s.route };
  }

  private held = false;

  /** The elevation or the track count changed: the route at the pointer fitted again. */
  again() {
    const s = this.spot;
    if (!s || !this.from || this.held) return;
    this.drop();
    this.spot = { ...s, seq: ++this.seq, t0: 0, phase: "rest", line: undefined, route: undefined, preview: undefined, waiters: [] };
    this.ask();
  }

  /** The segment is placed or dropped. */
  reset() {
    this.from = null;
    this.drop();
    this.askReach();
  }

  private drop() {
    clearTimeout(this.timer);
    const s = this.spot;
    this.spot = null;
    if (s) {
      // a click waiting on it gets nothing (an ordinary segment)
      s.phase = "off";
      for (const w of s.waiters.splice(0)) w();
      this.show(null);
    }
  }

  private ask() {
    const s = this.spot, f = this.from;
    if (!s || !f || s.phase !== "rest") return;
    s.phase = "search";
    s.t0 = performance.now();
    setTimeout(() => this.redraw(), SLOW_MS + 20);
    this.post({ kind: "route", seq: s.seq, a: [f.lng, f.lat], b: [s.lng, s.lat], ra: f.r, rb: SNAP_PX * this.perPx(s.lat) + SLACK_M, tracks: trackCount.peek(), level: buildLevel.peek(), tol: TOL_M });
  }

  private finish(s: Spot, phase: "done" | "none" | "off") {
    s.phase = phase;
    for (const w of s.waiters.splice(0)) w();
    this.redraw();
  }

  private answer(m: FromRouter) {
    if (m.kind === "cover") {
      this.cover = m.all ? "all" : m.rings.map((r) => ({ r, box: ringBox(r) }));
      this.coverCache.clear();
      return;
    }
    if (m.kind === "reach") {
      if (m.seq !== this.reachSeq || !this.from) return;
      realReach.value = { xy: m.xy, off: m.off, cls: m.cls };
      this.reachLast = { workerMs: m.ms, ms: performance.now() - this.reachAt, chains: m.chains, drawn: m.cls.length, points: m.xy.length / 2 };
      return;
    }
    const s = this.spot;
    if (!s || s.seq !== m.seq) return;
    if (m.kind === "line") {
      this.last = { searchMs: m.ms, states: m.states, fitMs: 0, lengthM: m.lengthM, corners: 0, previewMs: 0, worst: 0 };
      if (!m.line) {
        if (m.why === "off") {
          this.finish(s, "off");
          this.show(null);
          return;
        }
        // red, straight from the start to the pointer
        const [ax, ay] = toLocal(this.from!.lng, this.from!.lat), [bx, by] = toLocal(s.lng, s.lat);
        this.overlay.renderer.setPreview(strokesFromPreview(new Float32Array([ax, ay, 0, bx, by, 0]), true, undefined, true));
        this.overlay.invalidate();
        this.finish(s, "none");
        return;
      }
      s.phase = "fit";
      s.line = m.line;
      this.overlay.renderer.setPreview(strokesFromPreview(this.ghostPts(m.line), false, undefined, true));
      this.overlay.invalidate();
      this.redraw();
      return;
    }
    if (!m.pis || !s.line) {
      this.finish(s, "none");
      return;
    }
    const pis: PiInput[] = [];
    for (let k = 0; k + 3 < m.pis.length; k += 4) pis.push({ x: m.pis[k], y: m.pis[k + 1], level: m.pis[k + 3] as Level, ...(m.pis[k + 2] > 0 ? { radius: m.pis[k + 2] } : {}) });
    const L = s.line.length;
    s.route = { first: [s.line[0], s.line[1]], last: [s.line[L - 2], s.line[L - 1]], pis };
    s.phase = "preview";
    if (this.last) Object.assign(this.last, { fitMs: m.ms, corners: pis.length, worst: m.worst });
    const t = performance.now();
    const req = this.request(s.route);
    this.client.preview(req, (a) => {
      if (this.spot !== s) return;
      s.preview = a;
      if (this.last) this.last.previewMs = performance.now() - t;
      this.show(a, req);
      this.finish(s, "done");
    });
  }

  /** The real line in local units for the ghost, a vertex kept only half a pixel or more from the last. */
  private ghostPts(line: Float64Array): Float32Array {
    const step = (0.5 * EARTH_C) / (512 * 2 ** this.map.getZoom());
    const [x0, y0] = toLocal(line[0], line[1]);
    const out = [x0, y0, 0];
    let lx = x0, ly = y0;
    for (let k = 2; k < line.length; k += 2) {
      const [x, y] = toLocal(line[k], line[k + 1]);
      if (k + 2 < line.length && Math.hypot(x - lx, y - ly) < step) continue;
      out.push(x, y, 0);
      lx = x, ly = y;
    }
    return new Float32Array(out);
  }
}
