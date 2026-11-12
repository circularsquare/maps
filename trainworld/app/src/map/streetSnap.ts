// T-2927: the track tool's snap to streets with Tram picked (notes/T-2921.md 6), called from tools.ts
// while it is on (the switch, flipped while Ctrl is held). After each pointer move (REST_MS), the
// streets between the segment's start and the pointer are searched with a cost for each turn
// (`BusRoads.tramLeg`, `RoadGraph.turnPath`), drawn as the ghost, and fitted as tram corners in the
// router worker (`TrackApi.trace_kind`) at the first tolerance of TOLS that fits; the fitted track's
// preview then replaces the ghost with the usual tag. Until then the last fitted route's ghost and tag
// stay (T-2996), the street's line drawn only where none shows. A pointer with no street within reach is an
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
import { BusRoads, railRoad, type BusRefusal } from "./busRoads";

/** the wait after the pointer moves before the streets are searched, ms; the search and fit take a few ms (notes/T-3016.md) and the workers drop stale asks */
const REST_MS = 0;
/** a pointer this far from the spot asked about is a new spot, px */
const SAME_PX = 4;
/** a search or fit taking longer shows the spinner in the tag, ms */
const SLOW_MS = 300;
/** the fit's tolerances in turn, ground m: the street's line within the first that fits */
const TOLS = [10, 20, 40];
/** T-3129: rail on roads, the fit's tolerances in turn (ground m: the road's line within the first that fits) */
const RAIL_TOLS = [20, 40];
/** a turn costs the time lost slowing for it, as metres at the target speed: `(v - TURN_V)^2 / RAIL_DECEL`
 * with v the target speed in m/s, a right-angle corner taken at about 44 km/h; at least `RAIL_TURN_MIN_M` and
 * at most `RAIL_TURN_MAX_M`, past which a fast line would search the whole city for a way round */
const RAIL_TURN_V = 12.2;
const RAIL_DECEL = 1;
const RAIL_TURN_MIN_M = 10;
const RAIL_TURN_MAX_M = 600;
export const railTurnM = (kmh: number) => Math.max(RAIL_TURN_MIN_M, Math.min(RAIL_TURN_MAX_M, (Math.max(0, kmh / 3.6 - RAIL_TURN_V) ** 2) / RAIL_DECEL));
/** a fit whose worst point is further than this many times its tolerance, and than WORST_M, is not
 * taken. Where it leaves the street it is tram track, priced so (`set_route_street`); WORST_M is a
 * 25 m curve round the sharpest corner the search takes (BACK_DEG, 40 m from the corner) and a bit,
 * roomy for the smallest tram curve (18 m: 29 m from it) */
const WORST_SHARE = 1.5;
const WORST_M = 45;
/** the search's cost for a turn sharper than TURN_DEG where streets meet, m; never turning back
 * more than BACK_DEG at a junction, or within BACK_M of street (across a divided road's median) */
export const TURN_M = 200;
const TURN_DEG = 30;
const BACK_DEG = 135;
/** a turn between BACK_DEG and this costs TURN_M to twice that, more the sharper (T-3035); past it, refused */
const HARD_DEG = 165;
const BACK_M = 60;
/** how far back along the last route's street its end's heading is taken from, ground m */
const COME_M = 15;
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

/** what the snap follows: street tram track on streets, or rail on the main roads (T-3129) */
export type SnapProfile = "tram" | "rail";

const REFUSED: Record<Exclude<BusRefusal, "far">, () => string> = { path: () => t("busDraw.tagPath"), long: () => t("busDraw.tagLong") };

export class StreetSnap {
  private worker: Worker | null = null;
  private seq = 0;
  private from: [number, number] | null = null;
  private spot: Spot | null = null;
  /** T-3263: where the pointer went while the spot was being fitted or priced, asked for when it is done */
  private next: { px: number; py: number; lng: number; lat: number; onTrack: boolean } | null = null;
  private timer = 0;
  private held = false;
  private on = false;
  /** T-2996: what this snap has on the map for the segment: an earlier spot's ghost stays until the
   * new spot's is ready, so the ghost never blanks while the pointer moves */
  private shown: "route" | "line" | "red" | null = null;
  private shownWhy = "";
  private readonly tramRoads: BusRoads;
  private readonly railRoads: BusRoads;
  private profile: SnapProfile = "tram";
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
    /** what rail track on roads is laid for: the target speed (km/h), high speed, quad */
    private railOpts: () => { design: number; hsr: boolean; quad: boolean },
  ) {
    this.tramRoads = new BusRoads(map, () => this.landed());
    this.railRoads = new BusRoads(map, () => this.landed(), railRoad);
  }

  /** the roads the snap follows now */
  get roads(): BusRoads {
    return this.profile === "rail" ? this.railRoads : this.tramRoads;
  }

  private get rail() {
    return this.profile === "rail";
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

  /** The switch (flipped by Ctrl) with Tram in the track tool's hand, or with `profile` rail (T-3129) Snap to roads. */
  setOn(on: boolean, profile: SnapProfile = this.profile) {
    if (on === this.on && profile === this.profile) return;
    this.on = on;
    this.profile = profile;
    if (!on) this.ringAt = null;
    this.drop();
  }

  get isOn() {
    return this.on;
  }

  get snapProfile(): SnapProfile {
    return this.profile;
  }

  /** where the route at the pointer is (checks) */
  get phase() {
    return this.spot?.phase ?? null;
  }

  /** Where a first click at (lng, lat) goes: onto the street within reach, or where it is. */
  landing(lng: number, lat: number): [number, number] {
    const p = this.on ? this.roads.street({ lng, lat }, this.rail) : null;
    return p ? [p.lng, p.lat] : [lng, lat];
  }

  /** The segment's start (lng, lat), or null when there is none. */
  begin(at: [number, number] | null) {
    this.drop();
    this.from = at;
    const e = this.ended;
    // T-2997: carrying on from the street route placed last, never back the way it came
    this.came = at && e && Math.abs(e.at[0] - at[0]) < 1e-7 && Math.abs(e.at[1] - at[1]) < 1e-7 ? e.back : null;
  }

  /** the end of the street route placed last, and a point on its street before that */
  private ended: { at: [number, number]; back: [number, number] } | null = null;
  /** the segment carries on from that route: the point it came from */
  private came: [number, number] | null = null;

  private endOf(r: StreetRoute) {
    const l = r.line, n = l.length;
    const [ex, ey] = [l[n - 2], l[n - 1]], k = Math.cos((ey * Math.PI) / 180);
    let back: [number, number] = [l[0], l[1]];
    for (let i = n - 4; i >= 0; i -= 2) {
      back = [l[i], l[i + 1]];
      if (Math.hypot((l[i] - ex) * k, l[i + 1] - ey) * 111320 >= COME_M) break;
    }
    this.ended = { at: r.last, back };
  }

  /** The route between this segment's clicks follows streets. */
  get active() {
    return this.on && !!this.from && !!this.spot && this.spot.phase !== "off";
  }

  /** The pointer moved (screen px) to (lng, lat); `onTrack` where it is on tram track to end on. */
  move(px: number, py: number, lng: number, lat: number, onTrack: boolean, now = false) {
    if (!this.on) return;
    if (!this.from) {
      const p = this.roads.street({ lng, lat }, this.rail);
      this.ringAt = p ? [p.lng, p.lat] : null;
      return;
    }
    this.ringAt = null;
    if (this.held) return;
    const s = this.spot;
    if (s && Math.hypot(s.px - px, s.py - py) < SAME_PX && s.onTrack === onTrack) {
      this.next = null;
      return;
    }
    // T-3263: a spot being fitted or priced finishes, its ghost shown, and the newest pointer goes next: dropping
    // it at every move left the ghost where the pointer last rested for as long as the pointer moved
    if (!now && s && (s.phase === "fit" || s.phase === "preview")) {
      this.next = { px, py, lng, lat, onTrack };
      return;
    }
    this.next = null;
    this.drop(true);
    this.spot = { seq: ++this.seq, px, py, lng, lat, onTrack, t0: 0, phase: "rest", tol: 0, waiters: [] };
    this.timer = setTimeout(() => this.ask(), REST_MS) as unknown as number;
  }

  /** The route at the pointer is still being found and has taken a while (the tag's spinner). */
  get slow() {
    const s = this.spot;
    return this.active && !!s && s.phase !== "done" && s.phase !== "none" && s.t0 > 0 && performance.now() - s.t0 > SLOW_MS;
  }

  /** The tag's rows while a street route is being found or was refused (null: the usual tag, which
   * while the last route's ghost stands is that route's). */
  tag(): { text: string; bad: boolean; spin: boolean } | null {
    const s = this.spot;
    if (!this.active || !s) return null;
    if (s.phase === "none") return { text: s.why ?? "", bad: true, spin: false };
    if (s.phase === "done" || this.shown === "route") return null;
    return { text: this.shown === "red" ? this.shownWhy : "", bad: this.shown === "red", spin: this.slow };
  }

  /** The second click: waits for the route if it is still being found. */
  async click(px: number, py: number, lng: number, lat: number, onTrack: boolean): Promise<StreetClick> {
    if (!this.on || !this.from) return null;
    // the click's own spot at once, whatever is still being worked out for an earlier one
    this.move(px, py, lng, lat, onTrack, true);
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
    if (s.phase === "none" || !s.route) return { kind: "refuse", text: s.why ?? t(this.rail ? "tools.roadsTight" : "tools.streetsTight") };
    this.endOf(s.route);
    return { kind: "route", route: s.route };
  }

  /** The elevation changed: the route at the pointer fitted again. */
  again() {
    const s = this.spot;
    if (!s || !this.from || this.held) return;
    this.drop(true);
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

  /** The spot dropped; with `keep` its ghost stays on the map until the next spot's replaces it. */
  private drop(keep = false) {
    clearTimeout(this.timer);
    this.next = null;
    const s = this.spot;
    this.spot = null;
    if (s) {
      if (s.phase === "fit") this.post({ kind: "drop", seq: s.seq });
      else if (s.phase === "preview") this.client.dropPreview();
      s.phase = "off";
      for (const w of s.waiters.splice(0)) w();
      if (!keep) this.display(null);
    }
    if (!keep) this.shown = null;
  }

  private get tols() {
    return this.rail ? RAIL_TOLS : TOLS;
  }

  private display(a: PreviewAnswer | null, req?: object) {
    this.shown = a ? "route" : null;
    this.show(a, req);
  }

  private finish(s: Spot, phase: "done" | "none" | "off", why?: string) {
    s.phase = phase;
    s.why = why;
    for (const w of s.waiters.splice(0)) w();
    this.redraw();
    const n = this.next;
    if (n && this.spot === s) {
      this.next = null;
      this.move(n.px, n.py, n.lng, n.lat, n.onTrack);
    }
  }

  /** The red straight from the start to the pointer, saying `why`. */
  private redLine(s: Spot, why: string) {
    // the last route's figures go with its ghost
    if (this.shown === "route") this.display(null);
    this.shown = "red";
    this.shownWhy = why;
    const [ax, ay] = toLocal(this.from![0], this.from![1]), [bx, by] = toLocal(s.lng, s.lat);
    this.overlay.renderer.setPreview(strokesFromPreview(new Float32Array([ax, ay, 0, bx, by, 0]), true, undefined, true, undefined, 255, false, true));
    this.overlay.invalidate();
  }

  private ask(retry = false) {
    const s = this.spot, f = this.from;
    if (!s || !f || (s.phase !== "rest")) return;
    if (!retry) {
      s.t0 = performance.now();
      // the spinner, if it is still being found by then
      setTimeout(() => this.spot === s && this.redraw(), SLOW_MS + 20);
    }
    const c = this.came;
    const turnM = this.rail ? railTurnM(this.railOpts().design) : TURN_M;
    const leg = this.roads.tramLeg({ lng: f[0], lat: f[1] }, { lng: s.lng, lat: s.lat }, turnM, (TURN_DEG * Math.PI) / 180, (BACK_DEG * Math.PI) / 180, BACK_M, c && { lng: c[0], lat: c[1] }, (HARD_DEG * Math.PI) / 180, this.rail);
    // tiles still landing: asked again when they have (`landed`)
    if (!leg.ready) {
      setTimeout(() => this.redraw(), SLOW_MS + 20);
      return;
    }
    this.last = { searchMs: leg.ms, states: leg.states, turns: leg.turns, lengthM: leg.m, fitMs: 0, tol: 0, corners: 0, worst: 0, previewMs: 0 };
    if (leg.refused === "far") {
      // no street by one end: an ordinary segment
      this.finish(s, "off");
      this.display(null);
      return;
    }
    if (leg.refused) {
      const why = REFUSED[leg.refused]();
      this.redLine(s, why);
      this.finish(s, "none", why);
      return;
    }
    const line = [f[0], f[1], ...leg.pts];
    if (s.onTrack) line.push(s.lng, s.lat);
    s.line = new Float64Array(line);
    // the street's line stands in until the fit is ready, unless a fitted route already shows
    if (this.shown !== "route") {
      this.shown = "line";
      this.overlay.renderer.setPreview(strokesFromPreview(this.ghostPts(s.line), false, undefined, true, undefined, 255, false, true));
      this.overlay.invalidate();
    }
    this.fit(s);
  }

  private fit(s: Spot) {
    s.phase = "fit";
    this.redraw();
    const fr = getWorldFrame();
    const rail = this.rail ? this.railOpts() : undefined;
    this.post({ kind: "trace", seq: s.seq, line: s.line!.slice(), tol: this.tols[s.tol], track: STREET_TRAM, level: this.level(), frame: { kind: fr.kind, lon: fr.lon, lat: fr.lat }, rail });
  }

  private answer(m: FromRouter) {
    if (m.kind !== "fit") return;
    const s = this.spot;
    if (!s || s.seq !== m.seq || s.phase !== "fit" || !s.line) return;
    const tol = this.tols[s.tol];
    if (this.last) Object.assign(this.last, { fitMs: this.last.fitMs + m.ms, tol, worst: m.worst });
    if (!m.pis || m.worst > (this.rail ? WORST_SHARE * tol : Math.max(WORST_SHARE * tol, WORST_M))) {
      if (s.tol + 1 < this.tols.length) {
        // wider, as a new request so an answer for the narrower one is not taken for it
        s.tol++;
        s.seq = ++this.seq;
        return this.fit(s);
      }
      const tight = t(this.rail ? "tools.roadsTight" : "tools.streetsTight");
      this.redLine(s, tight);
      return this.finish(s, "none", tight);
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
      this.display(a, req);
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
