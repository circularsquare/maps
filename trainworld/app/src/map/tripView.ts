// The trip explorer on the map (T-195; docs/ui-inspectors.md "The trip explorer"): the tool's clicks,
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
import { CITIES } from "../game/cities";
import { lngLatToPack, packToLngLat } from "../game/coords";
import { cityCells, demandView, flows, intercitySelection, viewFinal, viewUpdating } from "../game/demand";
import { dealFlows } from "../game/farChunks";
import { count, distance } from "../game/format";
import { loadMarketRows, rowsAnswer, stationNear } from "../game/marketRule";
import { NET_INK } from "../game/palette";
import { shadeColours } from "../game/shades";
import { periodAt } from "../game/periods";
import type { PurposeFilter, PurposeKey } from "../game/purposes";
import { askMarketTrip, askTrip, averageTrips, busiestPeriod, mergeMarketPeriods, type MarketEnd, kindsOf, openRoute, routeLabel, stationName, stationPoint, tripKey, tripMix, tripOpenedFrom, tripPeriod, tripView, type MarketPeriod, type Trip, type TripLeg, type TripPeriod, type TripPlaces, type TripRoute, type TripShown } from "../game/trip";
import { purposeFilter, selection, shownDay } from "../game/ui";
import { metresBetween } from "../game/walks";
import { stationNames, stopsById, world } from "../game/world";
import { addFinder } from "./demandHover";
import { toLngLat, toLocal, unitPerPxAt } from "./geo";
import { sizeCanvasLike } from "./gl";
import type { NetworkBuffers } from "./network";
import type { Overlay } from "./overlay";
import { pick } from "./pick";

/** The period the explorer shows: the player's pick, else the one with most of the trips shown
 * (T-586), else the one in force on the clock. */
export const shownPeriod = computed(() => tripPeriod.value ?? tripView.value?.busiest ?? periodAt(clock.local.value * 60)); // T-309: local in view

/** Hit distance for the hover, px. */
const HIT_PX = 6;
/** T-979: how long after an empty answer on a final result the trip is asked again, ms. */
const RETRY_MS = 3000;
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
  /** the straight line's hover, while one is drawn (T-1252) */
  let straight: string[] | null = null;
  let marks: { from: [number, number]; to: [number, number] | null; fromStation?: string; toStation?: string } | null = null;

  sizeCanvasLike(canvas, map);
  map.on("resize", () => { sizeCanvasLike(canvas, map); draw(); });

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
      ctx.fillStyle = shadeColours().land;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const shades = shadeColours();
    for (const p of drawn?.parts ?? []) {
      const pts = p.pts.map(screenLocal);
      if (pts.length < 2) continue;
      ctx.globalAlpha = p.alpha;
      const path = () => {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      };
      if (p.dash) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = shades.casing;
        ctx.lineWidth = p.width + 2;
        path();
        ctx.stroke();
        ctx.strokeStyle = p.colour === NET_INK ? shades.mapInk : p.colour;
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
    if (straight && marks.to) {
      const [ax, ay] = screen(marks.from), [bx, by] = screen(marks.to);
      ctx.setLineDash([0.1, 7]);
      for (const [colour, width] of [[shades.casing, 6], [shades.mapInk, 3.5]] as const) {
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = colour;
        ctx.lineWidth = width;
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
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

  // ---- asking: when the trip, the purpose filter or the demand result (and its day) changes ----
  let asked = "";
  effect(() => {
    const sel = selection.value;
    const dv = demandView.value;
    const filter = purposeFilter.value;
    if (sel?.kind !== "trip") {
      if (marks) { marks = null; drawn = null; straight = null; redraw(); }
      tripView.value = null;
      asked = "";
      return;
    }
    marks = { from: sel.from, to: sel.to, fromStation: sel.fromStation, toStation: sel.toStation };
    if (!sel.to) {
      drawn = null;
      straight = null;
      tripView.value = null;
      asked = "";
      redraw();
      return;
    }
    const key = tripKey(sel.from, sel.to);
    const ask = `${key}|${dv?.version}|${dv?.round}|${shownDay.value}|${filter}`;
    if (ask === asked) return;
    asked = ask;
    // keep the last answer for the same points drawn while the next is on its way
    const last = tripView.peek();
    const same = last?.key === key;
    if (!same) { tripPeriod.value = null; openRoute.value = null; }
    tripView.value = same ? { ...last!, asking: true, failed: false } : { key, trip: null, asking: true, mix: null, purpose: "work", busiest: null };
    if (!same) { drawn = null; straight = null; redraw(); }
    const opened = tripOpenedFrom.peek();
    const to = sel.to;
    const run = (again: boolean) => explore(sel.from, to, filter, opened?.trip === sel ? opened.places : null, sel.far, sel.near).then((shown) => {
      if (asked !== ask) return;
      if (shown) tripView.value = { key, ...shown, asking: false };
      // null: the workers moved on and the next result asks again. T-979: on a final result none
      // comes, so it is asked once more (a pack the trip needs may have been opening), then the
      // explorer says there is no answer
      else if (!(viewFinal.peek() && !viewUpdating.peek())) tripView.value = { ...tripView.peek()!, asking: true };
      else if (!again) setTimeout(() => { if (asked === ask) run(true); }, RETRY_MS);
      else tripView.value = { ...tripView.peek()!, asking: false, failed: true };
    }, (err) => {
      console.error("trip explorer:", err);
      if (asked === ask) tripView.value = { ...tripView.peek()!, asking: false, failed: true };
    });
    run(false);
  });
  // ---- drawing: the shown period's routes ----
  // the network the rides were drawn along: a new one (it can land after `world`) redraws them
  let builtOn: NetworkBuffers | null = null;
  const rebuild = () => {
    const v = tripView.peek(), q = shownPeriod.peek(), w = world.peek();
    const t = v?.trip?.periods[q] ?? null;
    // T-866: between zones, the market's routes from the selection's point to the far end's
    const mt = v?.market?.periods[q] ?? null;
    builtOn = overlay.renderer.net ?? null;
    drawn = t && w ? build(t, marks?.from ?? t.from, marks?.to ?? t.to, builtOn) : mt && w && marks?.to ? build(mt, marks.from, marks.to, builtOn) : null;
    // T-1252: answered with no train route in the period shown, a straight line joins the two ends
    const answered = !!v && (!!v.trip || !!v.market || !!v.failed);
    if (answered && marks?.to && !drawn?.parts.length) {
      const anyRoute = (v.trip?.periods ?? v.market?.periods ?? []).some((p) => p?.routes.length);
      const d = metresBetween(...packToLngLat(marks.from[0], marks.from[1]), ...packToLngLat(marks.to[0], marks.to[1]));
      straight = [anyRoute ? "No train route at this time of day" : "No train route between these two points", `${distance(d)} in a straight line`];
    } else straight = null;
    redraw();
  };
  effect(() => {
    tripView.value;
    shownPeriod.value;
    world.value;
    openRoute.value;
    rebuild();
  });
  overlay.afterMapFrame.push(() => { if (drawn && (overlay.renderer.net ?? null) !== builtOn) rebuild(); });
  // ?debug=1: what is drawn, for the browser checks
  if (new URLSearchParams(location.search).get("debug") === "1") (window as any).twTrip = { drawn: () => drawn };

  addFinder(30, (p) => {
    // the routes are on the drawn network's anchor, which is the current one (T-307)
    if ((!drawn && !straight) || !marks) return null;
    const [lx, ly] = toLocal(p.lng, p.lat);
    const z = map.getZoom();
    const tol = HIT_PX * unitPerPxAt(z);
    const loc = (p: [number, number]) => { const [lng, lat] = packToLngLat(p[0], p[1]); return toLocal(lng, lat); };
    for (const [p, tip, at, st] of [[marks.to, "Where the trip ends", "The trip ends at", marks.toStation], [marks.from, "Where the trip starts", "The trip starts at", marks.fromStation]] as const) {
      if (!p) continue;
      const [x, y] = loc(p);
      const name = stationName(st);
      if (Math.hypot(x - lx, y - ly) <= tol * 1.5) return [name ? `${at} ${name}` : tip];
    }
    const parts = drawn?.parts ?? [];
    for (const s of drawn?.stops ?? []) {
      const [x, y] = s.at;
      if (Math.hypot(x - lx, y - ly) <= tol) return s.tip;
    }
    // the topmost part first (drawn last)
    for (let i = parts.length - 1; i >= 0; i--) {
      const pts = parts[i].pts;
      for (let k = 1; k < pts.length; k++) if (segDist(lx, ly, pts[k - 1], pts[k]) <= tol) return parts[i].tip;
    }
    if (straight && marks.to && segDist(lx, ly, loc(marks.from), loc(marks.to)) <= tol) return straight;
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

/**
 * Everything the explorer shows for a trip from `from` to `to` under purpose filter `filter`
 * (T-586): the trips a day between the two places by mode as the view counts them (`places`, the
 * bubble pair it was opened from; else the two cells, as homes to destinations), the purpose with
 * most of them (or the filter's), and its trip read by the model in every period. Null while the
 * workers move on.
 */
async function explore(from: [number, number], to: [number, number], filter: PurposeFilter, places: TripPlaces | null, far?: MarketEnd, near?: MarketEnd): Promise<Omit<TripShown, "key" | "asking"> | null> {
  // T-866: a far end in another zone, picked as one of a bubble selection's far bubbles (T-885:
  // or several, or with a place as the near end too)
  if (far && (places || near)) return exploreMarket(places?.end ?? "home", places?.near ?? new Uint32Array(0), places?.fars ?? [far], near);
  // the commute first: cheap, and it names the two cells
  const commute = await askTrip(from, to);
  if (!commute) return null;
  if (commute.outside) {
    // T-866: the trip tool's end in another zone: the start's cell to the far 6 km zone (or place)
    // nearest the end
    const between = await farEnd(from, to);
    if (between === undefined) return null;
    if (between) return exploreMarket(between.end, between.cells, [between.far], between.near);
    return { trip: commute, mix: null, purpose: "work", busiest: null };
  }
  const p0 = commute.periods.find((p) => p);
  const at = places ?? (p0 ? { end: "home" as const, near: Uint32Array.of(p0.cells[0]), far: Uint32Array.of(p0.cells[1]) } : null);
  const asked = at ? await flows(at.end, at.near) : null;
  if (at && !asked) return null;
  // opened from far bubbles: their trips grouped as the bubbles draw them (T-909)
  const xy = asked && places ? (await cityCells())?.xy : null;
  const f = asked && xy ? dealFlows(asked, filter, xy) : asked;
  const mix = f && at ? tripMix(f, at.far, filter) : null;
  const top = mix?.purposes[0];
  const purpose: PurposeKey = filter !== "all" ? filter : top && top.rail + top.walk + top.drive > 0 ? top.key : "work";
  let trip = commute;
  if (purpose !== "work") {
    const best = await bestKind(from, to, purpose);
    if (best === null) return null;
    if (best) trip = best;
  }
  // T-865: an end of several bubbles: the same purpose over the sampled cell pairs, averaged
  const samples = places?.group?.samples ?? [];
  if (samples.length > 1) {
    const asked = await Promise.all(samples.map((s) => askTrip(s.from, s.to, trip.purpose)));
    if (asked.some((t) => !t)) return null;
    const ok = (asked as Trip[]).map((t, i) => ({ trip: t, w: samples[i].w })).filter((x) => !x.trip.outside && x.trip.periods.some((p) => p));
    const names = new Map((world.peek()?.save.stations ?? []).map((s) => [s.id, s.name]));
    if (ok.length) trip = averageTrips(ok, routeLabel(names));
  }
  return { trip, mix, purpose: trip.purpose < 0 ? "work" : purpose, busiest: f ? busiestPeriod(f, trip.purpose < 0 ? "work" : purpose) : null };
}

/** An everyday purpose's trip between two points; school counts university too, so the kind with
 * more trips between the two areas. Null while the workers move on, undefined where no kind has one. */
async function bestKind(from: [number, number], to: [number, number], purpose: PurposeKey): Promise<Trip | null | undefined> {
  const asked = await Promise.all(kindsOf(purpose).map((k) => askTrip(from, to, k)));
  if (asked.some((t) => !t)) return null;
  const trips = (t: Trip) => Math.max(0, ...t.periods.map((p) => p?.pairTrips ?? 0));
  return (asked as Trip[]).filter((t) => t.periods.some((p) => p)).sort((a, b) => trips(b) - trips(a))[0];
}

/** T-1436: one sampled rider's trip as the explorer reads it, in perceived minutes (its "Effective
 * time" by train and by car), for a trip of purpose `key` in period `q`. */
export interface RiderTrip {
  /** the two ends are in different zones: the regional market's figures, averaged over the zones */
  market: boolean;
  train: number | null;
  car: number | null;
  /** the chances of each mode for this trip (0-1) */
  pTrain: number | null;
  pDrive: number | null;
  pWalk: number | null;
}

/** Undefined while the workers move on (ask again); null where the two points have no answer. */
export async function riderTrip(from: [number, number], to: [number, number], key: PurposeKey, q: number): Promise<RiderTrip | null | undefined> {
  const commute = await askTrip(from, to);
  if (!commute) return undefined;
  const fin = (x: number) => (Number.isFinite(x) ? x : null);
  if (commute.outside) {
    const between = await farEnd(from, to);
    if (between === undefined) return undefined;
    if (!between) return null;
    const shown = await exploreMarket(between.end, between.cells, [between.far], between.near);
    if (!shown) return undefined;
    const m = shown.market?.periods[q];
    if (!m) return null;
    const all = m.rail + m.air + m.car;
    return { market: true, train: fin(m.railPerceived), car: fin(m.carPerceived), pTrain: all > 0 ? m.rail / all : null, pDrive: all > 0 ? m.car / all : null, pWalk: null };
  }
  let trip = commute;
  if (key !== "work" && key !== "longer") {
    const best = await bestKind(from, to, key);
    if (best === null) return undefined;
    if (best) trip = best;
  }
  const t = trip.periods[q] ?? trip.periods.find((p) => p);
  if (!t) return null;
  return { market: false, train: t.railOffered ? fin(t.railPerceived) : null, car: fin(t.roadPerceived), pTrain: t.pTrain, pDrive: t.pDrive, pWalk: t.pWalk };
}

/** T-866: the longer trips between a selection's cells and a far place or 6 km zone, every period,
 * as the regional market chooses them; the period shown first is the one with most of them. */
async function exploreMarket(end: "home" | "work", cells: Uint32Array, fars: MarketEnd[], near?: MarketEnd): Promise<Omit<TripShown, "key" | "asking"> | null> {
  // T-1148: whole places where no train is in play (no station near one end of every pair): the
  // market rows answer at once, with no worker
  const stations = world.peek()?.save.stations ?? [];
  const bare = near && near.gid < 0 && fars.every((f) => f.gid < 0) ? [near, ...fars].map((e) => e.place).filter((p) => !stationNear(p, stations)) : [];
  const noRail = near && (bare.includes(near.place) || fars.every((f) => bare.includes(f.place)));
  const rows = noRail && (await loadMarketRows()) ? fars.map((f) => rowsAnswer(near, f, shownDay.peek())) : null;
  const asked = rows && rows.every((r) => r) ? rows : await Promise.all(fars.map((f) => askMarketTrip(end, cells, f.place, f.gid, near)));
  if (asked.some((a) => !a)) return null;
  const all = asked as (MarketPeriod | null)[][];
  // T-885: a group of far ends as one
  const names = new Map((world.peek()?.save.stations ?? []).map((s) => [s.id, s.name]));
  const periods = all.length === 1 ? all[0] : all[0].map((_, q) => mergeMarketPeriods(all.map((a) => a[q]), routeLabel(names)));
  let busiest: number | null = null;
  // T-880: every kind of trip between the two ends counts
  const n = (p: MarketPeriod) => p.trips + p.commute.legs + p.everyday.legs;
  periods.forEach((p, q) => { if (p && n(p) > 0 && (busiest === null || n(p) > n(periods[busiest]!))) busiest = q; });
  return { trip: null, mix: null, purpose: "longer", busiest, market: { far: fars[0], periods, end, ...(fars.length > 1 ? { fars } : {}), ...(near ? { near } : {}), ...(asked === rows ? { unbuilt: bare } : {}) } };
}

/** How far a trip tool's end may be from a far 6 km zone's centre and still be read as in it, m. */
const FAR_ZONE_M = 6000;
/** The same for a far place's mean trip end, m. */
const FAR_PLACE_M = 60000;

/**
 * T-866, T-885: a trip tool trip with an end outside the focused zone's cells. One end in them: its
 * own cell (as homes for the start, destinations for the end) and the far 6 km zone with longer
 * trips with it nearest the other end, else the nearest such place. Neither: the two zones' places
 * whose centres are nearest the ends. Null when nothing fits; undefined while the workers move on.
 */
async function farEnd(from: [number, number], to: [number, number]): Promise<{ end: "home" | "work"; cells: Uint32Array; far: MarketEnd; near?: MarketEnd } | null | undefined> {
  const cells = await cityCells();
  if (!cells) return undefined;
  const cellAt = (pt: [number, number]) => {
    let best = -1, bestD = 600 * 600;
    for (let c = 0; c < cells.n; c++) {
      const d = (cells.xy[2 * c] - pt[0]) ** 2 + (cells.xy[2 * c + 1] - pt[1]) ** 2;
      if (d <= bestD) (bestD = d), (best = c);
    }
    return best;
  };
  const a = cellAt(from), b = a < 0 ? cellAt(to) : -1;
  if (a < 0 && b < 0) {
    const placeAt = (pt: [number, number]) => {
      let best = -1, bestD = FAR_PLACE_M;
      for (const c of CITIES) {
        const p = lngLatToPack(c.center[0], c.center[1]), d = Math.hypot(p.x - pt[0], p.y - pt[1]);
        if (d <= bestD) (bestD = d), (best = c.place);
      }
      return best;
    };
    const pa = placeAt(from), pb = placeAt(to);
    return pa >= 0 && pb >= 0 && pa !== pb ? { end: "home", cells: new Uint32Array(0), near: { place: pa, gid: -1 }, far: { place: pb, gid: -1 } } : null;
  }
  const end = a >= 0 ? "home" : "work", near = Uint32Array.of(a >= 0 ? a : b), other = a >= 0 ? to : from;
  const ic = await intercitySelection(end, near);
  if (!ic) return undefined;
  const flowsOf = end === "home" ? ic.starts : ic.ends;
  const dist = (p: { x: number; y: number }) => Math.hypot(p.x - other[0], p.y - other[1]);
  const zone = flowsOf.zones.filter((z) => z.all > 0).sort((p, q) => dist(p) - dist(q))[0];
  if (zone && dist(zone) <= FAR_ZONE_M) return { end, cells: near, far: { place: zone.place, gid: zone.gid } };
  const place = flowsOf.places.filter((p) => p.all > 0 && Number.isFinite(p.x)).sort((p, q) => dist(p) - dist(q))[0];
  return place && dist(place) <= FAR_PLACE_M ? { end, cells: near, far: { place: place.place, gid: -1 } } : null;
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
 * drawn along. A segment's `stretch` is the stop-to-stop stretch of run 0 it is on.
 */
function strokesByLine(net: NetworkBuffers | null): Map<number, number[]> {
  const by = new Map<number, number[]>();
  if (!net?.lines.stretch) return by;
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
function rideTrack(net: NetworkBuffers | null, segs: number[] | undefined, k0: number, k1: number, a: [number, number] | null, b: [number, number] | null, wrapN = 0): [number, number][] | null {
  if (!net?.lines.stretch || !segs || k0 < 0 || k1 < 0 || k0 === k1) return null;
  const s = net.lines, stretch = s.stretch!;
  const span = (lo: number, hi: number) => {
    const pts: [number, number][] = [];
    for (const c of segs) {
      const k = stretch[c];
      if (k < lo || k >= hi) continue;
      if (!pts.length) pts.push([s.seg[c * 4], s.seg[c * 4 + 1]]);
      pts.push([s.seg[c * 4 + 2], s.seg[c * 4 + 3]]);
    }
    return pts;
  };
  let pts: [number, number][];
  if (wrapN) {
    // T-798: round a circle through its first stop, from the higher stop to the closing one and on
    // from the first; run 1 the same way reversed
    const [from, to] = k0 > k1 ? [k0, k1] : [k1, k0];
    pts = [...span(from, wrapN - 1), ...span(0, to).slice(1)];
    if (k0 < k1) pts.reverse();
  } else {
    pts = span(Math.min(k0, k1), Math.max(k0, k1));
    if (k0 > k1) pts.reverse();
  }
  if (pts.length < 2) return null;
  const near = (p: [number, number], q: [number, number] | null) => !q || Math.hypot(p[0] - q[0], p[1] - q[1]) <= TRACK_TOL;
  return near(pts[0], a) && near(pts[pts.length - 1], b) ? pts : null;
}

/** The shown period's routes as polylines in local units: the main route on top at full strength,
 * the others thinner and faint under it. Rides follow their line's track when the drawn network
 * has it, else run straight between the stations they call at. */
function build(t: Pick<TripPeriod, "routes" | "pairs"> | MarketPeriod, fromPack: [number, number], toPack: [number, number], net: NetworkBuffers | null): Drawn {
  const w = world.peek()!;
  const st = stopsById(w), names = stationNames(w);
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
  const name = (id: string) => names.get(id) ?? "a station";
  const from = packLocal(fromPack), to = packLocal(toPack);
  const segs = strokesByLine(net);
  const track = (l: TripLeg) => rideTrack(net, segs.get(Number(l.line)), l.k0, l.k1, pos(l.from), pos(l.to), l.wrapN ?? 0);
  const out: Drawn = { parts: [], stops: [] };
  const share = (r: TripRoute) => `${Math.round(100 * r.share)}% of this trip's train riders take this route`;
  // the main one on top, then the route opened in the inspector (T-870) over it
  const label = routeLabel(names);
  const open = openRoute.peek();
  const isOpen = (r: TripRoute) => open !== null && r !== t.routes[0] && label(r) === open;
  const routes = [...t.routes].reverse().sort((a, b) => Number(isOpen(a)) - Number(isOpen(b)));
  for (const r of routes) {
    const shown = isOpen(r);
    const main = r === t.routes[0] || shown;
    const pair = t.pairs[r.pair];
    const alpha = main ? 1 : 0.35, width = r === t.routes[0] ? 5 : shown ? 4 : 3;
    const legs = r.legs;
    const first = legs[0], last = legs[legs.length - 1];
    const a = first ? pos(first.from) : null, b = last ? pos(last.to) : null;
    // T-866: a market trip's ends are areas, reached on foot or by car, with felt minutes only
    const area = pair && !Number.isFinite(pair.accessM);
    // T-865: an averaged trip's pair walks from its own cells
    const start = pair?.from ? packLocal(pair.from) : from, end = pair?.to ? packLocal(pair.to) : to;
    if (a && pair) out.parts.push({ pts: [start, a], colour: NET_INK, width: main ? 2.5 : 1.5, dash: true, alpha, tip: area
      ? [`To ${name(first.from)}`, `On foot or by car, felt as ${minutes(pair.accessPerceived)} on average over the area`]
      : [`${pair.accessRide ? "Drive" : "Walk"} ${distance(pair.accessM)} to ${name(first.from)}, ${minutes(pair.accessMin)}`, pair.accessRide ? "Straight-line distance from the homes near here" : "Straight-line distance from this home"] });
    for (const l of legs) out.parts.push(legPart(l, main, alpha, width, ln, pos, name, share(r), track, !!w.lineStats[l.line]?.bus));
    if (b && pair) out.parts.push({ pts: [b, end], colour: NET_INK, width: main ? 2.5 : 1.5, dash: true, alpha, tip: area
      ? [`From ${name(last.to)}`, `On foot or dropped off, felt as ${minutes(pair.egressPerceived)} on average over the area`]
      : [`Walk ${distance(pair.egressM)} from ${name(last.to)}, ${minutes(pair.egressMin)}`, "Straight-line distance to this place"] });
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

function legPart(l: TripLeg, main: boolean, alpha: number, width: number, ln: Map<string, { name: string; colour: string }>, pos: (id: string) => [number, number] | null, name: (id: string) => string, share: string, track: (l: TripLeg) => [number, number][] | null, bus = false) {
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
      `${count(l.stops)} ${l.stops === 1 ? "stop" : "stops"}, ${minutes(l.min)} on the ${bus ? "bus" : "train"} after a ${minutes(l.waitMin)} wait${crowd >= 0.5 ? `; crowding makes it feel ${minutes(l.perceived)}` : ""}`,
      share,
    ],
  };
}
