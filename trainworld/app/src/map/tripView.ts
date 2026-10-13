// The trip explorer on the map (T-195; docs/interface.md "Trip explorer"): the tool's clicks,
// asking the demand workers about the trip (game/trip.ts), and drawing its routes on a canvas of
// its own over the demand views' (2D: a few dozen points, projected on each MapLibre frame).
//
// Clicks with the trip tool alternate: the first sets where the trip starts, the next where it
// ends, the one after starts a new trip; a click on a station takes the station as that end
// (T-221). The trip stays selected when the tool is left (Esc), so
// it can be read in the inspector, until the selection is cleared.

import { computed, effect } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { clock } from "../game/clock";
import { lngLatToPack, packToLngLat } from "../game/coords";
import { demandView } from "../game/demand";
import { count, km } from "../game/format";
import { NET_INK } from "../game/palette";
import { periodAt } from "../game/periods";
import { askTrip, stationName, stationPoint, tripKey, tripPeriod, tripView, type TripLeg, type TripPeriod, type TripRoute } from "../game/trip";
import { selection } from "../game/ui";
import { world } from "../game/world";
import { addFinder } from "./demandHover";
import { toLngLat, toLocal } from "./geo";
import type { NetworkBuffers } from "./network";
import type { Overlay } from "./overlay";
import { pick } from "./pick";

/** The period the explorer shows: the player's pick, else the one in force on the clock. */
export const shownPeriod = computed(() => tripPeriod.value ?? periodAt(clock.minute.value * 60));

/** Hit distance for the hover, px. */
const HIT_PX = 6;
/** Opacity of the land-coloured veil over the map under a trip's routes. */
const VEIL = 0.6;

interface Drawn {
  /** screen-independent polylines in local units (map/geo.ts), with what the hover says; a ride
   * follows its line's track (T-208) */
  parts: { pts: [number, number][]; colour: string; width: number; dash: boolean; alpha: number; tip: string[] }[];
  stops: { at: [number, number]; big: boolean; tip: string[] }[];
}

export interface TripTool {
  click(e: MapMouseEvent): void;
  /** the same at a map point, screen px (a click on a DOM tag over the map) */
  clickAt(x: number, y: number): void;
  /** a station is under the pointer (screen px), for the cursor */
  stationAt(x: number, y: number): boolean;
}

export function startTripView(map: MlMap, overlay: Overlay): TripTool {
  const canvas = document.createElement("canvas");
  canvas.className = "overlay trip";
  // over the demand views' canvas (it comes right after the network's), under the station names
  (map.getContainer().querySelector("canvas.overlay.demand") ?? overlay.canvas).after(canvas);
  const ctx = canvas.getContext("2d")!;
  let drawn: Drawn | null = null;
  let marks: { from: [number, number]; to: [number, number] | null; fromStation?: string; toStation?: string } | null = null;

  const size = () => {
    const dpr = map.getPixelRatio(), c = map.getCanvas();
    canvas.width = Math.round(c.clientWidth * dpr);
    canvas.height = Math.round(c.clientHeight * dpr);
    canvas.style.width = c.clientWidth + "px";
    canvas.style.height = c.clientHeight + "px";
  };
  size();
  map.on("resize", () => { size(); draw(); });

  /** pack metres (the two ends) to screen */
  const screen = (p: [number, number]) => {
    const [lng, lat] = packToLngLat(p[0], p[1]);
    const s = map.project([lng, lat]);
    return [s.x, s.y] as [number, number];
  };
  /** local units (the drawn routes) to screen */
  const screenLocal = (p: [number, number]) => {
    const s = map.project(toLngLat(p[0], p[1]));
    return [s.x, s.y] as [number, number];
  };

  function draw() {
    const dpr = map.getPixelRatio();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!marks) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // a veil of the land colour over the network while a trip's routes are drawn, so they stand
    // out from the lines they ride
    if (drawn?.parts.length) {
      ctx.globalAlpha = VEIL;
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--map-land").trim() || "#f4f1ea";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const p of drawn?.parts ?? []) {
      const pts = p.pts.map(screenLocal);
      if (pts.length < 2) continue;
      ctx.globalAlpha = p.alpha;
      const path = () => {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (const q of pts.slice(1)) ctx.lineTo(q[0], q[1]);
      };
      if (p.dash) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = p.width + 2;
        path();
        ctx.stroke();
        ctx.strokeStyle = p.colour;
        ctx.lineWidth = p.width;
        path();
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = NET_INK;
        ctx.lineWidth = p.width + 4;
        path();
        ctx.stroke();
        ctx.strokeStyle = p.colour;
        ctx.lineWidth = p.width;
        path();
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    for (const s of drawn?.stops ?? []) {
      const [x, y] = screenLocal(s.at);
      ctx.beginPath();
      ctx.arc(x, y, s.big ? 4.5 : 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = NET_INK;
      ctx.stroke();
    }
    // the two ends: where the trip starts an open ring, where it ends a filled disc
    for (const [p, filled] of [[marks.from, false], [marks.to, true]] as const) {
      if (!p) continue;
      const [x, y] = screen(p);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = filled ? NET_INK : "#ffffff";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = filled ? "#ffffff" : NET_INK;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 8.5, 0, 2 * Math.PI);
      ctx.lineWidth = 1;
      ctx.strokeStyle = NET_INK;
      ctx.stroke();
    }
  }
  overlay.afterMapFrame.push(() => { if (marks) draw(); });
  let raf = 0;
  const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

  // ---- asking: when the trip or the demand result changes ----
  let asked = "";
  effect(() => {
    const sel = selection.value;
    const dv = demandView.value;
    if (sel?.kind !== "trip") {
      if (marks) { marks = null; drawn = null; redraw(); }
      tripView.value = null;
      asked = "";
      return;
    }
    marks = { from: sel.from, to: sel.to, fromStation: sel.fromStation, toStation: sel.toStation };
    if (!sel.to) {
      drawn = null;
      tripView.value = null;
      asked = "";
      redraw();
      return;
    }
    const key = tripKey(sel.from, sel.to);
    const ask = `${key}|${dv?.version}|${dv?.round}`;
    if (ask === asked) return;
    asked = ask;
    // keep the last answer for the same points drawn while the next is on its way
    const last = tripView.peek();
    tripView.value = { key, trip: last?.key === key ? last.trip : null, asking: true };
    if (last?.key !== key) { drawn = null; redraw(); }
    void askTrip(sel.from, sel.to).then((trip) => {
      if (asked !== ask) return;
      // null: the workers moved on; the next result asks again
      const now = tripView.peek();
      tripView.value = { key, trip: trip ?? (now?.key === key ? now.trip : null), asking: !trip };
    });
  });
  // ---- drawing: the shown period's routes ----
  // the network the rides were drawn along: a new one (it can land after `world`) redraws them
  let builtOn: NetworkBuffers | null = null;
  const rebuild = () => {
    const v = tripView.peek(), q = shownPeriod.peek(), w = world.peek();
    const t = v?.trip?.periods[q] ?? null;
    builtOn = overlay.renderer.net ?? null;
    drawn = t && w ? build(t, marks?.from ?? t.from, marks?.to ?? t.to, builtOn) : null;
    redraw();
  };
  effect(() => {
    tripView.value;
    shownPeriod.value;
    world.value;
    rebuild();
  });
  overlay.afterMapFrame.push(() => { if (drawn && (overlay.renderer.net ?? null) !== builtOn) rebuild(); });
  // ?debug=1: what is drawn, for the browser checks
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twTrip = { drawn: () => drawn };

  addFinder(30, (lx, ly) => {
    if (!drawn || !marks) return null;
    const z = map.getZoom();
    const tol = (HIT_PX * 40075016.686) / (512 * Math.pow(2, z));
    const loc = (p: [number, number]) => { const [lng, lat] = packToLngLat(p[0], p[1]); return toLocal(lng, lat); };
    for (const [p, tip, at, st] of [[marks.to, "Where the trip ends", "The trip ends at", marks.toStation], [marks.from, "Where the trip starts", "The trip starts at", marks.fromStation]] as const) {
      if (!p) continue;
      const [x, y] = loc(p);
      const name = stationName(st);
      if (Math.hypot(x - lx, y - ly) <= tol * 1.5) return [name ? `${at} ${name}` : tip];
    }
    for (const s of drawn.stops) {
      const [x, y] = s.at;
      if (Math.hypot(x - lx, y - ly) <= tol) return s.tip;
    }
    // the topmost part first (drawn last)
    for (let i = drawn.parts.length - 1; i >= 0; i--) {
      const pts = drawn.parts[i].pts;
      for (let k = 1; k < pts.length; k++) if (segDist(lx, ly, pts[k - 1], pts[k]) <= tol) return drawn.parts[i].tip;
    }
    return null;
  });

  return {
    click(e) {
      this.clickAt(e.point.x, e.point.y);
    },
    clickAt(x, y) {
      // a click on a station picks the station (T-221): the end is at its position (game/trip.ts
      // stationPoint says why), else the point clicked
      const hit = pick(overlay, x, y, false);
      const station = hit?.kind === "station" ? hit.station : undefined;
      const ll = map.unproject([x, y]);
      const p = lngLatToPack(ll.lng, ll.lat);
      const at: [number, number] = (station && stationPoint(station)) || [p.x, p.y];
      const sel = selection.peek();
      selection.value = sel?.kind === "trip" && !sel.to
        ? { kind: "trip", from: sel.from, to: at, fromStation: sel.fromStation, toStation: station }
        : { kind: "trip", from: at, to: null, fromStation: station };
    },
    stationAt(x, y) {
      return pick(overlay, x, y, false)?.kind === "station";
    },
  };
}

function segDist(x: number, y: number, a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / l2)) : 0;
  return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
}

const minutes = (m: number) => `${Math.max(1, Math.round(m))} min`;

/** Pack metres to local units. */
function packLocal(p: [number, number]): [number, number] {
  const [lng, lat] = packToLngLat(p[0], p[1]);
  return toLocal(lng, lat);
}

/** How far (local units) a ride's track may end from its stations' dots and still be taken as
 * theirs; farther, the drawn network is not the one the solve saw, and the ride is drawn straight. */
const TRACK_TOL = 60;

/**
 * Each line's stroke segments (network.ts `lines`), in path order, by line id: what a ride is
 * drawn along. A segment's `info[0]` is the stop-to-stop stretch of run 0 it is on.
 */
function strokesByLine(net: NetworkBuffers | null): Map<number, number[]> {
  const by = new Map<number, number[]>();
  if (!net?.lines.info) return by;
  const s = net.lines;
  for (let c = 0; c < s.count; c++) {
    const v = by.get(s.colour[c]);
    if (v) v.push(c);
    else by.set(s.colour[c], [c]);
  }
  return by;
}

/**
 * A ride's track between its two stops (T-208): the line's stroke segments over the stretches
 * between stop `k0` and stop `k1` of run 0, joined into one polyline (reversed for run 1, so it
 * runs from where the ride boards), in local units. Null when the drawn network does not have it
 * as the solve saw it (the line not drawn, or the ends not at the stations' dots).
 */
function rideTrack(net: NetworkBuffers | null, segs: number[] | undefined, k0: number, k1: number, a: [number, number] | null, b: [number, number] | null): [number, number][] | null {
  if (!net?.lines.info || !segs || k0 < 0 || k1 < 0 || k0 === k1) return null;
  const s = net.lines, info = net.lines.info;
  const lo = Math.min(k0, k1), hi = Math.max(k0, k1);
  const pts: [number, number][] = [];
  for (const c of segs) {
    const k = info[c * 4];
    if (k < lo || k >= hi) continue;
    if (!pts.length) pts.push([s.seg[c * 4], s.seg[c * 4 + 1]]);
    pts.push([s.seg[c * 4 + 2], s.seg[c * 4 + 3]]);
  }
  if (pts.length < 2) return null;
  if (k0 > k1) pts.reverse();
  const near = (p: [number, number], q: [number, number] | null) => !q || Math.hypot(p[0] - q[0], p[1] - q[1]) <= TRACK_TOL;
  return near(pts[0], a) && near(pts[pts.length - 1], b) ? pts : null;
}

/** The shown period's routes as polylines in local units: the main route on top at full strength,
 * the others thinner and faint under it. Rides follow their line's track when the drawn network
 * has it, else run straight between the stations they call at. */
function build(t: TripPeriod, fromPack: [number, number], toPack: [number, number], net: NetworkBuffers | null): Drawn {
  const w = world.peek()!;
  const st = new Map(w.save.stations.map((s) => [s.id, s]));
  const ln = new Map(w.save.lines.map((l) => [l.id, l]));
  // a station's dot on the track (network.ts), else its position
  const dot = new Map<string, [number, number]>();
  if (net) net.stationIds.forEach((id, i) => dot.set(id, [net.stations[i * 3], net.stations[i * 3 + 1]]));
  const pos = (id: string): [number, number] | null => {
    const d = dot.get(id);
    if (d) return d;
    const s = st.get(id);
    return s ? toLocal(s.lng, s.lat) : null;
  };
  const name = (id: string) => st.get(id)?.name ?? "a station";
  const from = packLocal(fromPack), to = packLocal(toPack);
  const segs = strokesByLine(net);
  const track = (l: TripLeg) => rideTrack(net, segs.get(Number(l.line)), l.k0, l.k1, pos(l.from), pos(l.to));
  const out: Drawn = { parts: [], stops: [] };
  const share = (r: TripRoute) => `${Math.round(100 * r.share)}% of this trip's train riders take this route`;
  const routes = [...t.routes].reverse(); // the main one last, on top
  for (const r of routes) {
    const main = r === t.routes[0];
    const pair = t.pairs[r.pair];
    const alpha = main ? 1 : 0.35, width = main ? 5 : 3;
    const legs = r.legs;
    const first = legs[0], last = legs[legs.length - 1];
    const a = first ? pos(first.from) : null, b = last ? pos(last.to) : null;
    if (a && pair) out.parts.push({ pts: [from, a], colour: NET_INK, width: main ? 2.5 : 1.5, dash: true, alpha, tip: [`Walk ${km(pair.accessM)} to ${name(first.from)}, ${minutes(pair.accessMin)}`, "Straight-line distance, the model's average for the homes near here"] });
    for (const l of legs) out.parts.push(legPart(l, main, alpha, width, ln, pos, name, share(r), track));
    if (b && pair) out.parts.push({ pts: [b, to], colour: NET_INK, width: main ? 2.5 : 1.5, dash: true, alpha, tip: [`Walk ${km(pair.egressM)} from ${name(last.to)}, ${minutes(pair.egressMin)}`, "Straight-line distance, the model's average for the jobs near here"] });
    if (main) {
      for (const l of legs) {
        if (l.kind !== "ride") continue;
        l.stations.forEach((id, k) => {
          const p = pos(id);
          const ends = k === 0 || k === l.stations.length - 1;
          const line = ln.get(l.line)?.name ?? "the line";
          if (p) out.stops.push({ at: p, big: ends, tip: [name(id), k === 0 ? `Board ${line} here` : k === l.stations.length - 1 ? `Leave ${line} here` : `${line} stops here`] });
        });
      }
    }
  }
  return out;
}

function legPart(l: TripLeg, main: boolean, alpha: number, width: number, ln: Map<string, { name: string; colour: string }>, pos: (id: string) => [number, number] | null, name: (id: string) => string, share: string, track: (l: TripLeg) => [number, number][] | null) {
  if (l.kind === "walk") {
    const pts = [pos(l.from), pos(l.to)].filter((p): p is [number, number] => !!p);
    return { pts, colour: NET_INK, width: main ? 2.5 : 1.5, dash: true, alpha, tip: [`Walk from ${name(l.from)} to ${name(l.to)}, ${minutes(l.min)}`, share] };
  }
  const line = ln.get(l.line);
  const pts = track(l) ?? l.stations.map(pos).filter((p): p is [number, number] => !!p);
  const crowd = l.perceived - l.min;
  return {
    pts, colour: line?.colour ?? NET_INK, width, dash: false, alpha,
    tip: [
      `${line?.name ?? "A line"} from ${name(l.from)} to ${name(l.to)}`,
      `${count(l.stops)} ${l.stops === 1 ? "stop" : "stops"}, ${minutes(l.min)} on the train after a ${minutes(l.waitMin)} wait${crowd >= 0.5 ? `; crowding makes it feel ${minutes(l.perceived)}` : ""}`,
      share,
    ],
  };
}
