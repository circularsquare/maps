// T-2927: the track tool's snap to streets with Tram picked (notes/T-2921.md 6), called from tools.ts
// while it is on (the switch, flipped while Ctrl is held). With the pointer resting for REST_MS, the
// streets between the segment's start and the pointer are searched with a cost for each turn
// (`BusRoads.tramLeg`, `RoadGraph.turnPath`), drawn as the ghost, and fitted as tram corners in the
// router worker (`TrackApi.trace_kind`) at the first tolerance of TOLS that fits; the fitted track's
// preview then replaces the ghost with the usual tag. A pointer with no street within reach is an
// ordinary segment (tools.ts draws it, off-street tram track); no street path, or no fit, is a red
// ghost and a refusal.

import { t } from "../i18n";
import type { Map as MlMap } from "maplibre-gl";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import type { PiInput, RouteAsk } from "../workers/protocol";
import type { FromRouter, ToRouter } from "../workers/realRoute.worker";
import type { Level } from "../game/types";
import { getWorldFrame } from "../game/cities";
import type { Overlay } from "./overlay";
import { strokesFromPreview } from "./network";
import { toLocal, unitPerPxAt } from "./geo";
import { BusRoads, type BusRefusal } from "./busRoads";

/** the pointer's rest before the streets are searched, ms */
const REST_MS = 60;
/** a pointer this far from the spot asked about is a new spot, px */
const SAME_PX = 4;
/** a search or fit taking longer shows the spinner in the tag, ms */
const SLOW_MS = 300;
/** the fit's tolerances in turn, ground m: the street's line within the first that fits */
const TOLS = [10, 20, 40];
/** a fit whose worst point is further than this many times its tolerance, and than WORST_M, is not
 * taken. Where it leaves the street it is tram track, priced so (`set_route_street`); WORST_M is a
 * 25 m curve round the sharpest corner the search takes (BACK_DEG, 40 m from the corner) and a bit */
const WORST_SHARE = 1.5;
const WORST_M = 45;
/** the search's cost for a turn sharper than TURN_DEG where streets meet, m; never turning back
 * more than BACK_DEG at a junction, or within BACK_M of street (across a divided road's median) */
export const TURN_M = 200;
const TURN_DEG = 30;
const BACK_DEG = 135;
const BACK_M = 60;
/** the track kind street snapped track is placed as (`TrackKind::StreetTram`) */
const STREET_TRAM = 2;

/** The route the second click places: its ends (lng, lat), the fitted corners between and the street's
 * line they were fitted to (lng, lat pairs, flat). */
export interface StreetRoute { first: [number, number]; last: [number, number]; pis: PiInput[]; line: number[] }

/** What a click places: a route, a refusal with its text, or nothing to do with streets (an ordinary segment). */
export type StreetClick = { kind: "route"; route: StreetRoute } | { kind: "refuse"; text: string } | null;

interface Spot {
  seq: number;
  px: number;
  py: number;
  /** the pointer's point, and whether it is on tram track the route ends on */
  lng: number;
  lat: number;
  onTrack: boolean;
  t0: number;
  phase: "rest" | "search" | "fit" | "preview" | "done" | "none" | "off";
  why?: string;
  line?: Float64Array;
  tol: number;
  route?: StreetRoute;
  waiters: (() => void)[];
}

const REFUSED: Record<Exclude<BusRefusal, "far">, () => string> = { path: () => t("busDraw.tagPath"), long: () => t("busDraw.tagLong") };

export class StreetSnap {
  private worker: Worker | null = null;
  private seq = 0;
  private from: [number, number] | null = null;
  private spot: Spot | null = null;
  private timer = 0;
  private held = false;
  private on = false;
  readonly roads: BusRoads;
  /** where a first click at the pointer would land on a street */
  ringAt: [number, number] | null = null;
  /** the last route's figures, for checks (`window.tw.tools.street.last`) */
  last: { searchMs: number; states: number; turns: number; lengthM: number; fitMs: number; tol: number; corners: number; worst: number; previewMs: number } | null = null;

  constructor(
    private map: MlMap,
    private overlay: Overlay,
    private client: ClockClient,
    /** the route request for a street route, as tools.ts builds it */
    private request: (r: StreetRoute) => RouteAsk,
    /** show a preview answer as the drawing tool's own (null: none), with the request it answers */
    private show: (a: PreviewAnswer | null, req?: object) => void,
    private redraw: () => void,
    /** the elevation a route is laid at */
    private level: () => Level,
  ) {
    this.roads = new BusRoads(map, () => this.landed());
  }

  private router(): Worker {
    if (!this.worker) {
      this.worker = new Worker(new URL("../workers/realRoute.worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (e: MessageEvent<FromRouter>) => this.answer(e.data);
    }
    return this.worker;
  }

  private post(m: ToRouter) {
    this.router().postMessage(m);
  }

  /** The switch (flipped by Ctrl) with Tram in the track tool's hand. */
  setOn(on: boolean) {
    if (on === this.on) return;
    this.on = on;
    if (!on) this.ringAt = null;
    this.drop();
  }

  get isOn() {
    return this.on;
  }

  /** where the route at the pointer is (checks) */
  get phase() {
    return this.spot?.phase ?? null;
  }

  /** Where a first click at (lng, lat) goes: onto the street within reach, or where it is. */
  landing(lng: number, lat: number): [number, number] {
    const p = this.on ? this.roads.street({ lng, lat }) : null;
    return p ? [p.lng, p.lat] : [lng, lat];
  }

  /** The segment's start (lng, lat), or null when there is none. */
  begin(at: [number, number] | null) {
    this.drop();
    this.from = at;
  }

  /** The route between this segment's clicks follows streets. */
  get active() {
    return this.on && !!this.from && !!this.spot && this.spot.phase !== "off";
  }

  /** The pointer moved (screen px) to (lng, lat); `onTrack` where it is on tram track to end on. */
  move(px: number, py: number, lng: number, lat: number, onTrack: boolean) {
    if (!this.on) return;
    if (!this.from) {
      const p = this.roads.street({ lng, lat });
      this.ringAt = p ? [p.lng, p.lat] : null;
      return;
    }
    this.ringAt = null;
    if (this.held) return;
    const s = this.spot;
    if (s && Math.hypot(s.px - px, s.py - py) < SAME_PX && s.onTrack === onTrack) return;
    this.drop();
    this.spot = { seq: ++this.seq, px, py, lng, lat, onTrack, t0: 0, phase: "rest", tol: 0, waiters: [] };
    this.show(null);
    this.timer = setTimeout(() => this.ask(), REST_MS) as unknown as number;
  }

  /** The tag's rows while a street route is being found or was refused (null: the usual tag). */
  tag(): { text: string; bad: boolean; spin: boolean } | null {
    const s = this.spot;
    if (!this.active || !s) return null;
    if (s.phase === "none") return { text: s.why ?? "", bad: true, spin: false };
    if (s.phase === "done") return null;
    return { text: "", bad: false, spin: s.t0 > 0 && performance.now() - s.t0 > SLOW_MS };
  }

  /** The second click: waits for the route if it is still being found. */
  async click(px: number, py: number, lng: number, lat: number, onTrack: boolean): Promise<StreetClick> {
    if (!this.on || !this.from) return null;
    this.move(px, py, lng, lat, onTrack);
    const s = this.spot;
    if (!s || s.phase === "off") return null;
    if (s.phase === "rest") {
      clearTimeout(this.timer);
      this.ask();
    }
    if (s.phase !== "done" && s.phase !== "none" && (s.phase as Spot["phase"]) !== "off") {
      this.held = true;
      await new Promise<void>((r) => s.waiters.push(r));
      this.held = false;
    }
    if (this.spot !== s || (s.phase as Spot["phase"]) === "off") return null;
    if (s.phase === "none" || !s.route) return { kind: "refuse", text: s.why ?? t("tools.streetsTight") };
    return { kind: "route", route: s.route };
  }

  /** The elevation changed: the route at the pointer fitted again. */
  again() {
    const s = this.spot;
    if (!s || !this.from || this.held) return;
    this.drop();
    this.spot = { ...s, seq: ++this.seq, t0: 0, phase: "rest", line: undefined, route: undefined, why: undefined, tol: 0, waiters: [] };
    this.ask();
  }

  /** The segment is placed or dropped. */
  reset() {
    this.from = null;
    this.drop();
  }

  /** Street tiles landed: a search waiting on them goes ahead. */
  private landed() {
    const s = this.spot;
    if (s && s.phase === "rest" && s.t0 > 0) this.ask(true);
  }

  private drop() {
    clearTimeout(this.timer);
    const s = this.spot;
    this.spot = null;
    if (s) {
      if (s.phase === "fit") this.post({ kind: "drop", seq: s.seq });
      else if (s.phase === "preview") this.client.dropPreview();
      s.phase = "off";
      for (const w of s.waiters.splice(0)) w();
      this.show(null);
    }
  }

  private finish(s: Spot, phase: "done" | "none" | "off", why?: string) {
    s.phase = phase;
    s.why = why;
    for (const w of s.waiters.splice(0)) w();
    this.redraw();
  }

  /** The red straight from the start to the pointer. */
  private redLine(s: Spot) {
    const [ax, ay] = toLocal(this.from![0], this.from![1]), [bx, by] = toLocal(s.lng, s.lat);
    this.overlay.renderer.setPreview(strokesFromPreview(new Float32Array([ax, ay, 0, bx, by, 0]), true, undefined, true, undefined, 255, false, true));
    this.overlay.invalidate();
  }

  private ask(retry = false) {
    const s = this.spot, f = this.from;
    if (!s || !f || (s.phase !== "rest")) return;
    if (!retry) s.t0 = performance.now();
    const leg = this.roads.tramLeg({ lng: f[0], lat: f[1] }, { lng: s.lng, lat: s.lat }, TURN_M, (TURN_DEG * Math.PI) / 180, (BACK_DEG * Math.PI) / 180, BACK_M);
    // tiles still landing: asked again when they have (`landed`)
    if (!leg.ready) {
      setTimeout(() => this.redraw(), SLOW_MS + 20);
      return;
    }
    this.last = { searchMs: leg.ms, states: leg.states, turns: leg.turns, lengthM: leg.m, fitMs: 0, tol: 0, corners: 0, worst: 0, previewMs: 0 };
    if (leg.refused === "far") {
      // no street by one end: an ordinary segment
      this.finish(s, "off");
      this.show(null);
      return;
    }
    if (leg.refused) {
      this.redLine(s);
      this.finish(s, "none", REFUSED[leg.refused]());
      return;
    }
    const line = [f[0], f[1], ...leg.pts];
    if (s.onTrack) line.push(s.lng, s.lat);
    s.line = new Float64Array(line);
    this.overlay.renderer.setPreview(strokesFromPreview(this.ghostPts(s.line), false, undefined, true, undefined, 255, false, true));
    this.overlay.invalidate();
    this.fit(s);
  }

  private fit(s: Spot) {
    s.phase = "fit";
    this.redraw();
    const fr = getWorldFrame();
    this.post({ kind: "trace", seq: s.seq, line: s.line!.slice(), tol: TOLS[s.tol], track: STREET_TRAM, level: this.level(), frame: { kind: fr.kind, lon: fr.lon, lat: fr.lat } });
  }

  private answer(m: FromRouter) {
    if (m.kind !== "fit") return;
    const s = this.spot;
    if (!s || s.seq !== m.seq || s.phase !== "fit" || !s.line) return;
    const tol = TOLS[s.tol];
    if (this.last) Object.assign(this.last, { fitMs: this.last.fitMs + m.ms, tol, worst: m.worst });
    if (!m.pis || m.worst > Math.max(WORST_SHARE * tol, WORST_M)) {
      if (s.tol + 1 < TOLS.length) {
        // wider, as a new request so an answer for the narrower one is not taken for it
        s.tol++;
        s.seq = ++this.seq;
        return this.fit(s);
      }
      this.redLine(s);
      return this.finish(s, "none", t("tools.streetsTight"));
    }
    const pis: PiInput[] = [];
    for (let k = 0; k + 3 < m.pis.length; k += 4) pis.push({ x: m.pis[k], y: m.pis[k + 1], level: m.pis[k + 3] as Level, ...(m.pis[k + 2] > 0 ? { radius: m.pis[k + 2] } : {}) });
    const L = s.line.length;
    s.route = { first: [s.line[0], s.line[1]], last: [s.line[L - 2], s.line[L - 1]], pis, line: Array.from(s.line) };
    s.phase = "preview";
    if (this.last) this.last.corners = pis.length;
    const t0 = performance.now();
    const req = this.request(s.route);
    this.client.preview(req, (a) => {
      if (this.spot !== s) return;
      if (this.last) this.last.previewMs = performance.now() - t0;
      this.show(a, req);
      this.finish(s, "done");
    });
  }

  /** The street's line in local units for the ghost, a vertex kept only half a pixel or more from the last. */
  private ghostPts(line: Float64Array): Float32Array {
    const step = 0.5 * unitPerPxAt(this.map.getZoom());
    const [x0, y0] = toLocal(line[0], line[1]);
    const out = [x0, y0, 0];
    let lx = x0, ly = y0;
    for (let k = 2; k < line.length; k += 2) {
      const [x, y] = toLocal(line[k], line[k + 1]);
      if (k + 2 < line.length && Math.hypot(x - lx, y - ly) < step) continue;
      out.push(x, y, 0);
      (lx = x), (ly = y);
    }
    return new Float32Array(out);
  }
}
