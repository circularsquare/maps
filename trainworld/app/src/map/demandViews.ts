// The demand views (T-084, T-097, SPEC 8), wired to the data. Queries are asked for when a view
// needs them, never pushed (DECISIONS 2026-10-09): a view asks again when `demandView` changes,
// keeps showing its last answer meanwhile, and treats a null answer as "the workers moved on".
//
// - Network layers (line width, station use circles, train fill): map/demandLayers.ts, rebuilt when the
//   network, the world, the demand result, the period in force or a setting changes.
// - Commuter bubbles and a bubble selection: map/commuterView.ts; they catch up over a game day
//   after a new result (T-156, map/catchUp.ts).
// - Longer trips of a selection, and one bubble per city zoomed far out: map/intercityView.ts.
// - A selected station's riders: `stationRiders(id)`, listed in the station inspector from
//   `stationView`. A station selection leaves the map as it was (T-519).
// - Hovering any of these discs shows how many people they stand for (map/demandHover.ts).

import { computed, effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { activeCity } from "../game/cities";
import { clock } from "../game/clock";
import { complexes, selectedScope } from "../game/complexView";
import { cityCells, demandByDay, demandView, stationRiders, type CityCells, type DemandView, type StationRiders } from "../game/demand";
import { wholeDistance } from "../game/format";
import { compass } from "../game/places";
import { periodAt } from "../game/periods";
import { stationUse, useFigure } from "../game/stationUse";
import type { Selection } from "../game/types";
import { display, purposeFilter, selection, stopEditing } from "../game/ui";
import { world } from "../game/world";
import { startAirportMarks } from "./airportMarks";
import { commuterStats, startCommuterView } from "./commuterView";
import { buildLayers } from "./demandLayers";
import { hoverAbove, hoverAt, startHover } from "./demandHover";
import { DemandOverlay } from "./demandOverlay";
import { startIntercityView } from "./intercityView";
import { openGreyCityAt, startGreyCities } from "./greyCities";
import type { Overlay } from "./overlay";

/** Which end of a selected station's riders the inspector lists. */
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

export interface DemandViews {
  /** call after `renderer.setNetwork` */
  networkChanged(): void;
  /** a click with the select tool: a bubble selection, or undefined to let the network pick;
   * `under`: the bubble even beneath the selected station (the resting hover's list, T-732) */
  click(px: number, py: number, add: boolean, under?: boolean): Selection | undefined;
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
    const dv = demandView.peek(), mode = editing ? "none" : d.stationUse;
    const on = { lineLoad: d.trackColour === "traffic" && !editing, stationFigure: mode === "none" ? null : (id: string) => useFigure(dv, mode, id),
      trainLoad: d.trainLoad, trainLoadBasis: d.trainLoadBasis,
      complexes: d.complexView ? complexes.peek() : null, purpose: purposeFilter.peek(), days: demandByDay.peek().map((x) => x.view) };
    // an edit changes the network and then the world: one build for both when nothing else moved
    const now = [n, w, dv, period.peek(), on.lineLoad, mode, mode === "origins" || mode === "transfers" ? stationUse.peek() : null, on.trainLoad, on.trainLoadBasis, on.complexes, on.purpose];
    if (now.every((v, i) => v === inputs[i])) return;
    inputs = now;
    const t0 = performance.now();
    renderer.setLayers(buildLayers(n, w, dv, period.peek(), on));
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
    void [d.stationUse, d.trainLoad, d.trainLoadBasis, d.trackColour, d.complexView];
    stationUse.value;
    purposeFilter.value;
    stopEditing.value;
    schedule();
  });

  // ---- commuter bubbles ----
  const commuters = startCommuterView(map, view);
  startAirportMarks(map, overlay);
  // ---- longer trips of a selection (T-118) ----
  const intercity = startIntercityView(view, map);
  startGreyCities(view, map); // T-729

  // ---- a selected station's riders, for the inspector ----
  let focusSeen = activeCity.peek();
  let cellsP: Promise<CityCells | null> | null = null;
  let stationAsk = 0;
  let last: { r: StationRiders; dv: DemandView | null } | null = null;
  const show = (r: StationRiders, cells: CityCells | null) => {
    stationView.value = describe(r, stationSide.peek(), cells);
  };
  effect(() => {
    const focus = activeCity.value;
    if (focus !== focusSeen) { focusSeen = focus; cellsP = null; stationAsk++; last = null; stationView.value = null; }
    const sel = selection.value;
    const dv = demandView.value;
    stationSide.value;
    // with nothing shown yet the riders are still asked for (T-327): the answer is in hand when the
    // first estimate shows
    if (sel?.kind !== "station") {
      stationAsk++;
      last = null;
      stationView.value = null;
      return;
    }
    const id = sel.station;
    // the station, or its complex in complex view (T-146)
    const scope = selectedScope.value || id, members = scope.split(",");
    if (last && last.r.stationId === id && last.r.scope === scope && last.dv === dv) {
      // only the side changed
      const r = last.r;
      void (cellsP ??= cityCells()).then((cells) => { if (last?.r === r) show(r, cells); });
      return;
    }
    if (!last || last.r.stationId !== id || last.r.scope !== scope) stationView.value = null;
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

  return {
    networkChanged() {
      if (renderer.net !== netSeen) schedule();
    },
    click(px, py, add, under = false) {
      if (!under && onSelected(px, py)) return undefined; // the network picks the selected station
      if (openGreyCityAt(px, py)) return selection.peek() ?? undefined;
      const picked = commuters.click(px, py, add) ?? intercity.click(px, py);
      if (picked !== undefined) return picked;
      // Longer-trip discs describe the current selection. They are drawn over the network, so a
      // click on them retains that selection and is consumed.
      return hoverAt(map, px, py) ? selection.peek() ?? undefined : undefined;
    },
    hover: (px, py) => hoverAt(map, px, py),
    overlay: view,
  };
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
