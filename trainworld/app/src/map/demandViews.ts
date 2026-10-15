// The demand views (T-084, T-097, SPEC 8), wired to the data. Queries are asked for when a view
// needs them, never pushed (DECISIONS 2026-10-09): a view asks again when `demandView` changes,
// keeps showing its last answer meanwhile, and treats a null answer as "the workers moved on".
//
// - Network layers (line width, station size, train fill): map/demandLayers.ts, rebuilt when the
//   network, the world, the demand result, the period in force or a setting changes.
// - Commuter bubbles and a bubble selection: map/commuterView.ts; they catch up over a game day
//   after a new result (T-156, map/catchUp.ts).
// - Longer trips of a selection, and one bubble per city zoomed far out: map/intercityView.ts.
// - A selected station's riders: `stationRiders(id)`, drawn as discs over the network by
//   map/demandOverlay.ts and listed in the station inspector from `stationView`.
// - Hovering any of these discs shows how many people they stand for (map/demandHover.ts).

import { computed, effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity } from "../game/cities";
import { clock } from "../game/clock";
import { complexes, selectedScope } from "../game/complexView";
import { packToLngLat } from "../game/coords";
import { cellModes, cityCells, demandByDay, demandView, stationRiders, type CityCells, type DemandView, type StationRiders } from "../game/demand";
import { count, wholeDistance } from "../game/format";
import { CATCHMENT_COLOUR, DESTINATION_ALPHA, DESTINATION_COLOUR, rgb } from "../game/palette";
import { compass } from "../game/places";
import { periodAt } from "../game/periods";
import { tripOpenedFrom } from "../game/trip";
import type { Selection } from "../game/types";
import { display, purposeFilter, selection, stopEditing } from "../game/ui";
import { world } from "../game/world";
import { DISC } from "../workers/commutersProtocol";
import { commuterStats, startCommuterView } from "./commuterView";
import { buildLayers } from "./demandLayers";
import { anchorVersion } from "./anchorVersion";
import { addFinder, discAt, hoverAbove, hoverAt, pointerAt, startHover } from "./demandHover";
import { DemandOverlay } from "./demandOverlay";
import { anchor, kAtY, toLocal, type Anchor } from "./geo";
import { startIntercityView } from "./intercityView";
import type { Overlay } from "./overlay";

/** Which end of a selected station's riders the map and inspector show. */
export const stationSide = signal<"home" | "work">("home");
export interface Place {
  name: string;
  riders: number;
}
/** The selected station's riders, as the inspector lists them; null while unknown. `scope`: the
 * stations counted, `StationRiders.scope` (its complex's in complex view, T-146). */
export interface StationView { stationId: string; scope: string; home: number; work: number; walked: number; rode: number; places: Place[] }
export const stationView = signal<StationView | null>(null);

/** The period in force on the clock; changes five times a game day. */
const period = computed(() => periodAt(clock.local.value * 60)); // T-309: the local time in view

/** Timings for ?debug=1 (`window.twViews`) and the notes. */
export const viewStats = { layersMs: 0, layerBuilds: 0, stationMs: 0, commuters: commuterStats };

/** A disc with the area of a cell (H3 resolution 9), m. */
const CELL_R = 183;

export interface DemandViews {
  /** call after `renderer.setNetwork` */
  networkChanged(): void;
  /** a click with the select tool: a bubble selection, or undefined to let the network pick */
  click(px: number, py: number, add: boolean): Selection | undefined;
  /** the hover tag's lines at a point (for probes) */
  hover(px: number, py: number): string[] | null;
  overlay: DemandOverlay;
}

export function startDemandViews(overlay: Overlay, map: MlMap): DemandViews {
  const renderer = overlay.renderer;
  const view = new DemandOverlay(map, overlay);
  startHover(map);
  // The selected station shows above the bubbles (T-189): cut out of them, redrawn when the
  // network draws a new selection, and picked before them.
  const project = (x: number, y: number) => overlay.toScreen(x, y);
  view.cutouts = () => renderer.selectedShapes(project);
  renderer.onSelectionDrawn = () => view.invalidate();
  const onSelected = (px: number, py: number) => renderer.selectedStationAt(px, py, project) >= 0;
  hoverAbove.at = onSelected;

  // ---- network layers ----
  let netSeen: unknown = null;
  let inputs: unknown[] = [];
  const rebuild = () => {
    const n = renderer.net, w = world.peek();
    if (!n || !w) return;
    const d = display.peek();
    // stop editing (T-233) draws plain widths and no station circles: the demand is out of the way
    const editing = stopEditing.peek();
    const on = { lineLoad: d.trackColour === "traffic" && !editing, stationRiders: d.stationRiders && !editing, trainLoad: d.trainLoad, trainLoadBasis: d.trainLoadBasis,
      complexes: d.complexView ? complexes.peek() : null, purpose: purposeFilter.peek(), days: demandByDay.peek().map((x) => x.view) };
    // an edit changes the network and then the world: one build for both when nothing else moved
    const now = [n, w, demandView.peek(), period.peek(), on.lineLoad, on.stationRiders, on.trainLoad, on.trainLoadBasis, on.complexes, on.purpose];
    if (now.every((v, i) => v === inputs[i])) return;
    inputs = now;
    const t0 = performance.now();
    renderer.setLayers(buildLayers(n, w, demandView.peek(), period.peek(), on));
    viewStats.layersMs = performance.now() - t0;
    viewStats.layerBuilds++;
    netSeen = n;
    overlay.invalidate();
  };
  // After the current task: the clock client sets the world and then hands over the network, so
  // an edit builds once, with both.
  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      rebuild();
    });
  };
  effect(() => {
    world.value;
    demandView.value;
    period.value;
    const d = display.value;
    void [d.stationRiders, d.trainLoad, d.trainLoadBasis, d.trackColour, d.complexView];
    purposeFilter.value;
    stopEditing.value;
    schedule();
  });

  // ---- commuter bubbles ----
  const commuters = startCommuterView(map, view);
  // ---- longer trips of a selection or a station (T-118) ----
  const intercity = startIntercityView(view, map);

  // ---- a selected station's riders ----
  let focusSeen = activeCity.peek();
  let cellsP: Promise<CityCells | null> | null = null;
  let stationAsk = 0;
  let last: { r: StationRiders; dv: DemandView | null } | null = null;
  let marks: { near: Float32Array; far: Float32Array; nearN: Float32Array; farN: Float32Array; side: "home" | "work"; anchor: Anchor } | null = null;
  const clear = () => {
    marks = null;
    view.setDiscs("stationNear", null);
    view.setDiscs("stationFar", null);
  };
  // a station's catchment counts commuters: off the map while the purpose filter shows another
  // purpose (T-158); the inspector keeps its numbers
  const commutesShown = () => purposeFilter.peek() === "all" || purposeFilter.peek() === "work";
  /** The catchment's discs, relative to the draw anchor in force (T-317: redone when it moves). */
  const drawMarks = (r: StationRiders, cells: CityCells | null) => {
    const side = stationSide.peek();
    const a = anchor();
    marks = cells && commutesShown() ? { ...marksFor(r, side, cells, a), side, anchor: a } : null;
    view.setDiscs("stationNear", marks?.near ?? null, a);
    view.setDiscs("stationFar", marks?.far ?? null, a);
  };
  const show = (r: StationRiders, cells: CityCells | null) => {
    drawMarks(r, cells);
    stationView.value = describe(r, stationSide.peek(), cells);
  };
  effect(() => {
    const focus = activeCity.value;
    if (focus !== focusSeen) { focusSeen = focus; cellsP = null; stationAsk++; last = null; clear(); stationView.value = null; }
    const sel = selection.value;
    const dv = demandView.value;
    stationSide.value;
    purposeFilter.value;
    // with nothing shown yet the riders are still asked for (T-327): the answer is in hand when the
    // first estimate shows
    if (sel?.kind !== "station") {
      stationAsk++;
      last = null;
      stationView.value = null;
      clear();
      return;
    }
    const id = sel.station;
    // the station, or its complex in complex view (T-146)
    const scope = selectedScope.value || id, members = scope.split(",");
    if (last && last.r.stationId === id && last.r.scope === scope && last.dv === dv) {
      // only the side or the purpose filter changed (or a display setting that does not matter here)
      const r = last.r;
      if (!!marks !== commutesShown() || (marks && marks.side !== stationSide.peek())) void (cellsP ??= cityCells()).then((cells) => show(r, cells));
      return;
    }
    if (!last || last.r.stationId !== id || last.r.scope !== scope) {
      stationView.value = null;
      clear();
    }
    const ask = ++stationAsk;
    void (async () => {
      const t0 = performance.now();
      const [r, cells] = await Promise.all([stationRiders(id, 20, members), (cellsP ??= cityCells())]);
      // null: the workers moved on (the next demandView asks again), or not a station demand knows
      if (ask !== stationAsk || !r) return;
      viewStats.stationMs = performance.now() - t0;
      const fresh = !last || last.r.stationId !== id;
      last = { r, dv };
      if (fresh) stationSide.value = r.home.total >= r.work.total ? "home" : "work";
      show(r, cells);
    })();
  });
  effect(() => {
    anchorVersion.value;
    const l = last;
    if (!l || !marks) return;
    void (cellsP ??= cityCells()).then((cells) => { if (last === l) drawMarks(l.r, cells); });
  });
  addFinder(20, (p) => {
    if (!marks) return null;
    const live = marks.side === "home";
    let i = discAt(marks.near, p, marks.anchor);
    if (i >= 0) return [`${count(marks.nearN[i])} riders ${live ? "live" : "work"} here`];
    i = discAt(marks.far, p, marks.anchor);
    if (i >= 0) return [`${count(marks.farN[i])} of them ${live ? "work" : "live"} here`, "Click to explore this trip"];
    return null;
  });

  /**
   * A click on a far-end disc of a station's catchment opens the trip explorer for the pair
   * (T-208): from the catchment cell with the most of the station's riders to the far zone's cell
   * with the most rail legs at that end (the cached all-purpose answer: the riders' own cells in
   * the 2 km zone are not known here, and the zone's centre can fall where nobody is). True when
   * the click was on one.
   */
  const stationTrip = (px: number, py: number): boolean => {
    const sel = selection.peek();
    const m = marks, l = last;
    if (!m || !l || sel?.kind !== "station") return false;
    const p = pointerAt(map, px, py);
    if (discAt(m.near, p, m.anchor) >= 0) return false;
    const k = discAt(m.far, p, m.anchor);
    if (k < 0) return false;
    const side = m.side;
    const own = (side === "home" ? l.r.home : l.r.work).cells[0];
    const zone = (side === "home" ? l.r.homeRidersWorkIn : l.r.workRidersLiveIn).zones[k];
    void Promise.all([(cellsP ??= cityCells()), cellModes("all")]).then(([cells, modes]) => {
      if (!cells || own === undefined || selection.peek() !== sel) return;
      const legs = side === "home" ? modes?.workRail : modes?.homeRail;
      let best = -1, bestN = -1;
      for (let c = 0; c < cells.n; c++) {
        if (cells.zone[c] !== zone) continue;
        const n = legs ? legs[c] : 0;
        if (n > bestN) (bestN = n), (best = c);
      }
      if (best < 0) return;
      const at = (c: number): [number, number] => [cells.xy[c * 2], cells.xy[c * 2 + 1]];
      const trip: Selection = side === "home" ? { kind: "trip", from: at(own), to: at(best) } : { kind: "trip", from: at(best), to: at(own) };
      tripOpenedFrom.value = { trip, from: sel, what: `From ${world.peek()?.save.stations.find((s) => s.id === sel.station)?.name ?? "the station"}'s riders` };
      selection.value = trip;
    });
    return true;
  };

  return {
    networkChanged() {
      if (renderer.net !== netSeen) schedule();
    },
    click(px, py, add) {
      if (onSelected(px, py)) return undefined; // the network picks the selected station
      const picked = commuters.click(px, py, add) ?? intercity.click(px, py);
      if (picked !== undefined) return picked;
      if (stationTrip(px, py)) return selection.peek() ?? undefined;
      // Station catchments and longer-trip discs describe the current selection. They are
      // drawn over the network, so a click on them retains that selection and is consumed.
      return hoverAt(map, px, py) ? selection.peek() ?? undefined : undefined;
    },
    hover: (px, py) => hoverAt(map, px, py),
    overlay: view,
  };
}

/** Pack metres to local units of anchor `a` (T-317). */
function local(x: number, y: number, a: Anchor): [number, number] {
  const [lng, lat] = packToLngLat(x, y);
  return toLocal(lng, lat, a);
}

/**
 * Discs for a station's riders, drawn over the network: the cells at its end opaque on top, the
 * busiest the area of a cell; the zones at the far end filled and partly see-through below them,
 * the busiest half a zone wide. No minimum size. Smaller ones last, so they stay visible.
 */
function marksFor(r: StationRiders, side: "home" | "work", cells: CityCells, a: Anchor) {
  const near = side === "home" ? r.home : r.work;
  const far = side === "home" ? r.homeRidersWorkIn : r.workRidersLiveIn;
  const nc = rgb(CATCHMENT_COLOUR).map((v) => v / 255), fc = rgb(DESTINATION_COLOUR).map((v) => v / 255);
  const nearD = new Float32Array(near.cells.length * DISC), farD = new Float32Array(far.zones.length * DISC);
  const maxNear = near.riders[0] || 1, maxFar = far.riders[0] || 1;
  // lists come most first, which is biggest first: smaller discs draw over bigger ones
  for (let k = 0; k < far.zones.length; k++) {
    const z = far.zones[k];
    const [x, y] = local(cells.zoneXY[z * 2], cells.zoneXY[z * 2 + 1], a);
    farD.set([x, y, (cells.zoneM / 2) * Math.sqrt(far.riders[k] / maxFar) * kAtY(y, a), fc[0], fc[1], fc[2], DESTINATION_ALPHA, 0], k * DISC);
  }
  for (let k = 0; k < near.cells.length; k++) {
    const c = near.cells[k];
    const [x, y] = local(cells.xy[c * 2], cells.xy[c * 2 + 1], a);
    nearD.set([x, y, CELL_R * Math.sqrt(near.riders[k] / maxNear) * kAtY(y, a), nc[0], nc[1], nc[2], 1, 1], k * DISC);
  }
  return { near: nearD, far: farD, nearN: near.riders, farN: far.riders };
}

/**
 * The far end of a station's riders as places a player knows: each zone named by the station
 * nearest its centre within 3 km (zones sharing one merged), else by distance and direction from
 * the selected station.
 */
function describe(r: StationRiders, side: "home" | "work", cells: CityCells | null) {
  const far = side === "home" ? r.homeRidersWorkIn : r.workRidersLiveIn;
  const w = world.peek();
  const here = w?.save.stations.find((s) => s.id === r.stationId);
  const byName = new Map<string, number>();
  for (let k = 0; k < far.zones.length; k++) {
    const z = far.zones[k];
    let name = `Zone ${z}`;
    if (cells) {
      const x = cells.zoneXY[z * 2], y = cells.zoneXY[z * 2 + 1];
      let best: { name: string; d: number } | null = null;
      for (const s of w?.save.stations ?? []) {
        const d = Math.hypot(s.x - x, s.y - y);
        if (!best || d < best.d) best = { name: s.name, d };
      }
      if (best && best.d <= 3000) name = best.name;
      else if (here) name = `${wholeDistance(Math.hypot(x - here.x, y - here.y))} ${compass(x - here.x, y - here.y)}`;
    }
    byName.set(name, (byName.get(name) ?? 0) + far.riders[k]);
  }
  const places = [...byName].map(([name, riders]) => ({ name, riders })).sort((a, b) => b.riders - a.riders);
  return { stationId: r.stationId, scope: r.scope, home: r.home.total, work: r.work.total, walked: r.walked, rode: r.rode, places };
}
