// The demand views (T-084, T-097, SPEC 8), wired to the data. Queries are asked for when a view
// needs them, never pushed (DECISIONS 2026-10-09): a view asks again when `demandView` changes,
// keeps showing its last answer meanwhile, and treats a null answer as "the workers moved on".
//
// - Network layers (line width, station size, train fill): map/demandLayers.ts, rebuilt when the
//   network, the world, the demand result, the period in force or a setting changes.
// - Commuter bubbles and a bubble selection: map/commuterView.ts.
// - A selected station's riders: `stationRiders(id)`, drawn as discs over the network by
//   map/demandOverlay.ts and listed in the station inspector from `stationView`.
// - Hovering any of these discs shows how many people they stand for (map/demandHover.ts).

import { computed, effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { packToLngLat } from "../game/coords";
import { cityCells, demandView, stationRiders, type CityCells, type DemandView, type StationRiders } from "../game/demand";
import { count } from "../game/format";
import { CATCHMENT_COLOUR, DESTINATION_ALPHA, DESTINATION_COLOUR, rgb } from "../game/palette";
import { compass } from "../game/places";
import { periodAt } from "../game/periods";
import type { Selection } from "../game/types";
import { display, selection } from "../game/ui";
import { world } from "../game/world";
import { DISC } from "../workers/commutersProtocol";
import { commuterStats, startCommuterView } from "./commuterView";
import { buildLayers } from "./demandLayers";
import { addFinder, discAt, hoverAt, startHover } from "./demandHover";
import { DemandOverlay } from "./demandOverlay";
import { MERC_K, toLocal } from "./geo";
import type { Overlay } from "./overlay";

/** Which end of a selected station's riders the map and inspector show. */
export const stationSide = signal<"home" | "work">("home");
export interface Place {
  name: string;
  riders: number;
}
/** The selected station's riders, as the inspector lists them; null while unknown. */
export const stationView = signal<{ stationId: string; home: number; work: number; places: Place[] } | null>(null);

/** The period in force on the clock; changes five times a game day. */
const period = computed(() => periodAt(clock.minute.value * 60));

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

  // ---- network layers ----
  let netSeen: unknown = null;
  let inputs: unknown[] = [];
  const rebuild = () => {
    const n = renderer.net, w = world.peek();
    if (!n || !w) return;
    const d = display.peek();
    const on = { lineLoad: d.trackColour === "traffic", stationRiders: d.stationRiders, trainLoad: d.trainLoad, trainLoadBasis: d.trainLoadBasis };
    // an edit changes the network and then the world: one build for both when nothing else moved
    const now = [n, w, demandView.peek(), period.peek(), on.lineLoad, on.stationRiders, on.trainLoad, on.trainLoadBasis];
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
    void [d.stationRiders, d.trainLoad, d.trainLoadBasis, d.trackColour];
    schedule();
  });

  // ---- commuter bubbles ----
  const commuters = startCommuterView(map, overlay, view);

  // ---- a selected station's riders ----
  let cellsP: Promise<CityCells | null> | null = null;
  let stationAsk = 0;
  let last: { r: StationRiders; dv: DemandView } | null = null;
  let marks: { near: Float32Array; far: Float32Array; nearN: Float32Array; farN: Float32Array; side: "home" | "work" } | null = null;
  const clear = () => {
    marks = null;
    view.setDiscs("stationNear", null);
    view.setDiscs("stationFar", null);
  };
  const show = (r: StationRiders, cells: CityCells | null) => {
    const side = stationSide.peek();
    marks = cells ? { ...marksFor(r, side, cells), side } : null;
    view.setDiscs("stationNear", marks?.near ?? null);
    view.setDiscs("stationFar", marks?.far ?? null);
    stationView.value = describe(r, side, cells);
  };
  effect(() => {
    const sel = selection.value;
    const dv = demandView.value;
    stationSide.value;
    if (sel?.kind !== "station" || !dv) {
      stationAsk++;
      last = null;
      stationView.value = null;
      clear();
      return;
    }
    const id = sel.station;
    if (last && last.r.stationId === id && last.dv === dv) {
      // only the side changed
      const r = last.r;
      void (cellsP ??= cityCells()).then((cells) => show(r, cells));
      return;
    }
    if (!last || last.r.stationId !== id) {
      stationView.value = null;
      clear();
    }
    const ask = ++stationAsk;
    void (async () => {
      const t0 = performance.now();
      const [r, cells] = await Promise.all([stationRiders(id, 20), (cellsP ??= cityCells())]);
      // null: the workers moved on (the next demandView asks again), or not a station demand knows
      if (ask !== stationAsk || !r) return;
      viewStats.stationMs = performance.now() - t0;
      const fresh = !last || last.r.stationId !== id;
      last = { r, dv };
      if (fresh) stationSide.value = r.home.total >= r.work.total ? "home" : "work";
      show(r, cells);
    })();
  });
  addFinder(20, (x, y) => {
    if (!marks) return null;
    const live = marks.side === "home";
    let i = discAt(marks.near, x, y);
    if (i >= 0) return [`${count(marks.nearN[i])} riders ${live ? "live" : "work"} here`];
    i = discAt(marks.far, x, y);
    if (i >= 0) return [`${count(marks.farN[i])} of them ${live ? "work" : "live"} here`];
    return null;
  });

  return {
    networkChanged() {
      if (renderer.net !== netSeen) schedule();
    },
    click: commuters.click,
    hover: (px, py) => hoverAt(map, px, py),
    overlay: view,
  };
}

/** Pack metres to local units. */
function local(x: number, y: number): [number, number] {
  const [lng, lat] = packToLngLat(x, y);
  return toLocal(lng, lat);
}

/**
 * Discs for a station's riders, drawn over the network: the cells at its end opaque on top, the
 * busiest the area of a cell; the zones at the far end filled and partly see-through below them,
 * the busiest half a zone wide. No minimum size. Smaller ones last, so they stay visible.
 */
function marksFor(r: StationRiders, side: "home" | "work", cells: CityCells) {
  const near = side === "home" ? r.home : r.work;
  const far = side === "home" ? r.homeRidersWorkIn : r.workRidersLiveIn;
  const nc = rgb(CATCHMENT_COLOUR).map((v) => v / 255), fc = rgb(DESTINATION_COLOUR).map((v) => v / 255);
  const nearD = new Float32Array(near.cells.length * DISC), farD = new Float32Array(far.zones.length * DISC);
  const maxNear = near.riders[0] || 1, maxFar = far.riders[0] || 1;
  // lists come most first, which is biggest first: smaller discs draw over bigger ones
  for (let k = 0; k < far.zones.length; k++) {
    const z = far.zones[k];
    const [x, y] = local(cells.zoneXY[z * 2], cells.zoneXY[z * 2 + 1]);
    farD.set([x, y, (cells.zoneM / 2) * Math.sqrt(far.riders[k] / maxFar) * MERC_K, fc[0], fc[1], fc[2], DESTINATION_ALPHA, 0], k * DISC);
  }
  for (let k = 0; k < near.cells.length; k++) {
    const c = near.cells[k];
    const [x, y] = local(cells.xy[c * 2], cells.xy[c * 2 + 1]);
    nearD.set([x, y, CELL_R * Math.sqrt(near.riders[k] / maxNear) * MERC_K, nc[0], nc[1], nc[2], 1, 1], k * DISC);
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
      else if (here) name = `${Math.max(1, Math.round(Math.hypot(x - here.x, y - here.y) / 1000))} km ${compass(x - here.x, y - here.y)}`;
    }
    byName.set(name, (byName.get(name) ?? 0) + far.riders[k]);
  }
  const places = [...byName].map(([name, riders]) => ({ name, riders })).sort((a, b) => b.riders - a.riders);
  return { stationId: r.stationId, home: r.home.total, work: r.work.total, places };
}
