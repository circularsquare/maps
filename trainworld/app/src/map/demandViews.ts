// The demand views (T-084, T-097, SPEC 8), wired to the data. Queries are asked for when a view
// needs them, never pushed (DECISIONS 2026-10-09): a view asks again when `demandView` changes,
// keeps showing its last answer meanwhile, and treats a null answer as "the workers moved on".
//
// - Network layers (line width, train fill): map/demandLayers.ts, rebuilt when the network, the
//   world, the demand result, the period in force or a setting changes.
// - Station ridership circles, over the bubbles: map/stationCircles.ts.
// - Commuter bubbles and a bubble selection: map/commuterView.ts.
// - Longer trips of a selection, and one bubble per city zoomed far out: map/intercityView.ts.
// - Hovering any of these discs shows how many people they stand for (map/demandHover.ts).

import { computed, effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { stopKey } from "../game/complexView";
import { demandByDay, demandView } from "../game/demand";
import { periodAt } from "../game/periods";
import type { Selection } from "../game/types";
import { display, purposeFilter, selection, stopEditing } from "../game/ui";
import { world } from "../game/world";
import { startAirportMarks } from "./airportMarks";
import { startAttractionMarks } from "./attractionMarks";
import { commuterStats, startCommuterView } from "./commuterView";
import { buildLayers, busWidths } from "./demandLayers";
import { hoverAbove, hoverAt, startHover } from "./demandHover";
import { DemandOverlay } from "./demandOverlay";
import { startIntercityView } from "./intercityView";
import type { Overlay } from "./overlay";
import { StationCircles, stationCircles } from "./stationCircles";

/** The period in force on the clock; changes five times a game day. */
const period = computed(() => periodAt(clock.local.value * 60)); // T-309: the local time in view

/** Timings for ?debug=1 (`window.twViews`) and the notes. */
export const viewStats = { layersMs: 0, layerBuilds: 0, commuters: commuterStats };

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
  // the station ridership circles, over the bubbles (T-1448)
  const circles = stationCircles.now = new StationCircles(map, overlay, view.canvas);
  startHover(map);
  // The selected station shows above the bubbles (T-189): cut out of them, redrawn when the
  // network draws a new selection, and picked before them.
  const project = (x: number, y: number) => overlay.toScreen(x, y);
  view.cutouts = () => renderer.selectedShapes(project);
  renderer.onSelectionDrawn = () => view.invalidate();
  const onSelected = (px: number, py: number) => renderer.selectedStationAt(px, py, project) >= 0;
  // the bubbles' hover and click give way to the selected station and the station circles
  const above = (px: number, py: number) => onSelected(px, py) || !!circles.at(px, py);
  hoverAbove.at = above;

  // ---- network layers ----
  let netSeen: unknown = null;
  let inputs: unknown[] = [];
  const rebuild = () => {
    const n = renderer.net, w = world.peek();
    if (!n || !w) return;
    const d = display.peek();
    // stop editing (T-233) draws plain widths: the demand is out of the way
    const editing = stopEditing.peek();
    const dv = demandView.peek();
    const on = { lineLoad: d.trackColour === "traffic" && !editing, trainLoad: d.trainLoad,
      stopKey, purpose: purposeFilter.peek(), days: demandByDay.peek().map((x) => x.view) };
    // an edit changes the network and then the world: one build for both when nothing else moved
    const now = [n, w, dv, period.peek(), on.lineLoad, on.trainLoad, on.purpose, demandByDay.peek()];
    if (now.every((v, i) => v === inputs[i])) return;
    inputs = now;
    const t0 = performance.now();
    renderer.setLayers(buildLayers(n, w, dv, period.peek(), on));
    renderer.buses.setTraffic(on.lineLoad ? busWidths(w, dv, on) : null);
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
    demandByDay.value;
    period.value;
    const d = display.value;
    void [d.trainLoad, d.trackColour];
    purposeFilter.value;
    stopEditing.value;
    schedule();
  });

  // ---- commuter bubbles ----
  const commuters = startCommuterView(map, view);
  startAirportMarks(map, overlay);
  startAttractionMarks(map, overlay);
  // ---- longer trips of a selection (T-118) ----
  const intercity = startIntercityView(view, map);

  return {
    networkChanged() {
      if (renderer.net !== netSeen) schedule();
      circles.invalidate();
    },
    click(px, py, add, under = false) {
      if (!under && above(px, py)) return undefined; // the network picks the selected station
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
