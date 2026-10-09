// Commuter bubbles (T-097, SPEC 8), Anita's Subway Builder style view: where commuters live (or
// work), as bubbles sized by people and coloured by how they travel, aggregated to H3 parents by
// zoom. A click on a bubble, or a long press and a drag over several, selects them (a
// `commuters` selection, so Escape and an empty click clear it): the map then shows the selected
// bubbles outlined and where those people work (or live) as bubbles, and the inspector lists them.
//
// Data is asked for, not pushed: `cellModes()` when the demand result changes (a solve's first
// estimate and its refined result, not the crowding rounds between), `flows(end, cells)` when the
// selection or the demand result changes. Whole-city aggregation runs in
// workers/commuters.worker.ts; the selection's few thousand cells are summed here.

import { effect, signal } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { cellModes, cityCells, demandUpdating, demandView, flows, type CellModes, type DemandView } from "../game/demand";
import { count } from "../game/format";
import { BUBBLE_ALPHA } from "../game/palette";
import { compass } from "../game/places";
import type { Selection } from "../game/types";
import { display, selection, tool } from "../game/ui";
import { world } from "../game/world";
import { packOrigin } from "../workers/clockClient";
import { BUBBLE_DENSITY, RES_AREA_M2, type Bubbles, type FromCommuters, type Ready, type Summed, type ToCommuters } from "../workers/commutersProtocol";
import { aggregate, discs, type Groups } from "./bubbles";
import { addFinder, discAt, pointerLocal } from "./demandHover";
import { bubbleLevel, type DemandOverlay } from "./demandOverlay";
import { MERC_K, toLocal } from "./geo";
import { pick } from "./pick";
import type { Overlay } from "./overlay";

/** The legend: commuters by mode, the whole city, for the bubbles drawn. */
export const commuterLegend = signal<{ end: "home" | "work"; totals: [number, number, number] } | null>(null);

export interface Place {
  name: string;
  riders: number;
}
/** A bubble selection as the inspector shows it. `far` is null while its far end is being asked
 * for,. */
export const commuterPick = signal<{ sel: Selection; totals: [number, number, number]; far: { totals: [number, number, number]; places: Place[] } | null } | null>(null);


export const commuterStats = { cellsMs: 0, workerMs: 0, roundTripMs: 0, builds: 0, uploadMs: 0, selectMs: 0, flowsMs: 0, farMs: 0, farCells: 0, sumMs: 0 };

const pct = (v: number, all: number) => (all > 0 ? Math.round((100 * v) / all) : 0);
const modeLine = (t: number, w: number, d: number) => {
  const all = t + w + d;
  return `Train ${pct(t, all)}%, walking ${pct(w, all)}%, driving ${pct(d, all)}%`;
};

export function startCommuterView(map: MlMap, overlay: Overlay, view: DemandOverlay) {
  let city: Ready | null = null;
  let worker: Worker | null = null;
  let workerReady: Promise<boolean> | null = null;
  let nextId = 1;
  const pending = new Map<number, (b: any) => void>();
  const startWorker = (): Promise<boolean> =>
    (workerReady ??= (async () => {
      const cells = await cityCells();
      if (!cells) {
        workerReady = null;
        return false;
      }
      worker = new Worker(new URL("../workers/commuters.worker.ts", import.meta.url), { type: "module" });
      const ready = new Promise<boolean>((done) => {
        worker!.onmessage = (e: MessageEvent<FromCommuters>) => {
          const m = e.data;
          if (m.kind === "ready") {
            city = m;
            commuterStats.cellsMs = m.ms;
            done(true);
          } else {
            pending.get(m.id)?.(m);
            pending.delete(m.id);
          }
        };
      });
      const o = packOrigin.peek() ?? { lon: -73.985, lat: 40.758 };
      const msg: ToCommuters = { kind: "cells", xy: cells.xy.slice(), h3: cells.h3.slice(), origin: { lon: o.lon, lat: o.lat } };
      worker.postMessage(msg, [msg.xy.buffer, msg.h3.buffer]);
      return ready;
    })());

  // ---- the city's bubbles ----
  let shown: { dv: DemandView; end: "home" | "work"; bubbles: Bubbles; modes: CellModes } | null = null;
  let built: { dv: DemandView; modes: CellModes; ends: Map<string, Bubbles> } | null = null;
  let wanted = 0;
  effect(() => {
    const d = display.value;
    const dv = demandView.value;
    const updating = demandUpdating.value;
    const sel = selection.value;
    // the city's bubbles give way to a selection's own (and to a station's catchment)
    view.setBubblesVisible(d.commuters && sel?.kind !== "commuters" && sel?.kind !== "station");
    if (!d.commuters || !dv) return;
    const end = d.commuterEnd;
    // a solve's first estimate, and its refined result once the crowding rounds are done
    if (shown && shown.dv.version === dv.version && shown.end === end && (shown.dv === dv || updating)) return;
    const ask = ++wanted;
    void (async () => {
      let b = built?.dv === dv ? built.ends.get(end) : undefined;
      let modes = built?.dv === dv ? built.modes : null;
      if (!b || !modes) {
        if (!(await startWorker())) return;
        const t0 = performance.now();
        modes = modes ?? (await cellModes());
        // null: the workers moved on; the next demandView asks again and the last bubbles stay
        if (!modes || ask !== wanted) return;
        const m = modes;
        const id = nextId++;
        const pickEnd = end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
        b = await new Promise<Bubbles>((done) => {
          pending.set(id, done);
          const msg: ToCommuters = { kind: "build", id, end, train: pickEnd[0].slice(), walk: pickEnd[1].slice(), drive: pickEnd[2].slice() };
          worker!.postMessage(msg, [msg.train.buffer, msg.walk.buffer, msg.drive.buffer]);
        });
        commuterStats.workerMs = b.ms;
        commuterStats.roundTripMs = performance.now() - t0;
        commuterStats.builds++;
        if (built?.dv !== dv) built = { dv, modes: m, ends: new Map() };
        built.ends.set(end, b);
      }
      if (ask !== wanted) return;
      shown = { dv, end, bubbles: b, modes };
      view.setBubbles(b.levels.map((l) => l.discs));
      commuterStats.uploadMs = view.uploadMs;
      commuterLegend.value = { end, totals: b.totals };
      if (selection.peek()?.kind === "commuters") refreshPick();
    })();
  });

  // the size slider in the settings pane scales every commuter bubble on the GPU (T-099)
  effect(() => view.setBubbleSize(display.value.bubbleSize));

  // ---- hover ----
  const endWord = (end: "home" | "work") => (end === "home" ? "live" : "work");
  addFinder(10, (x, y) => {
    if (!shown || !display.peek().commuters) return null;
    const sel = selection.peek();
    if (sel?.kind === "station") return null;
    if (sel?.kind === "commuters") {
      let i = discAt(nearDiscs, x, y, view.bubbleSize);
      if (i >= 0 && near) return [`${count(near.train[i] + near.walk[i] + near.drive[i])} commuters ${endWord(sel.end)} here`, modeLine(near.train[i], near.walk[i], near.drive[i])];
      i = discAt(farDiscs, x, y, view.bubbleSize);
      if (i >= 0 && far) return [`${count(far.train[i] + far.walk[i] + far.drive[i])} of them ${endWord(sel.end === "home" ? "work" : "home")} here`, modeLine(far.train[i], far.walk[i], far.drive[i])];
      return null;
    }
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const i = discAt(L.discs, x, y, view.bubbleSize);
    if (i < 0) return null;
    return [`${count(L.train[i] + L.walk[i] + L.drive[i])} commuters ${endWord(shown.end)} here`, modeLine(L.train[i], L.walk[i], L.drive[i])];
  });

  // ---- selecting bubbles ----
  /** The city's bubbles at the zoom on screen whose disc contains (or, for a box, whose centre is
   * in) the given area: their groups. */
  const groupsAt = (x: number, y: number): number[] => {
    if (!shown) return [];
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const i = discAt(L.discs, x, y, view.bubbleSize);
    return i < 0 ? [] : [L.group[i]];
  };
  const groupsIn = (x0: number, y0: number, x1: number, y1: number): number[] => {
    if (!shown) return [];
    const L = shown.bubbles.levels[bubbleLevel(map.getZoom())];
    const out: number[] = [];
    for (let i = 0; i < L.group.length; i++) {
      const x = L.discs[i * 8], y = L.discs[i * 8 + 1];
      if (x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1)) out.push(L.group[i]);
    }
    return out;
  };
  const selectGroups = (groups: number[], add: boolean): Selection | null => {
    if (!shown || !city || !groups.length) return null;
    const g = city.levels[bubbleLevel(map.getZoom())];
    const cells = new Set<number>();
    const cur = selection.peek();
    if (add && cur?.kind === "commuters" && cur.end === shown.end) for (const c of cur.cells) cells.add(c);
    for (const q of groups) for (let k = g.off[q]; k < g.off[q + 1]; k++) cells.add(g.members[k]);
    const prevAreas = add && cur?.kind === "commuters" ? cur.areas : 0;
    return { kind: "commuters", end: shown.end, cells: [...cells], areas: prevAreas + groups.length };
  };

  /** A click in the select tool: a bubble selection, or undefined to let the network have it. */
  const click = (px: number, py: number, add: boolean): Selection | undefined => {
    if (!display.peek().commuters || !shown || selection.peek()?.kind === "station") return undefined;
    // stations and trains stay clickable through the bubbles
    const net = pick(overlay, px, py, display.peek().trains);
    if (net && (net.kind === "station" || net.kind === "train")) return undefined;
    const [x, y] = pointerLocal(map, px, py);
    const s = selectGroups(groupsAt(x, y), add);
    return s ?? undefined;
  };

  // Long press, then drag: a box; the bubbles whose centres are in it are selected.
  const box = document.createElement("div");
  box.className = "demand-box";
  box.style.display = "none";
  map.getContainer().appendChild(box);
  let press: { x: number; y: number; timer: number; on: boolean; add: boolean } | null = null;
  const LONG_MS = 400, STILL_PX = 5;
  const canvas = map.getCanvasContainer();
  const rel = (e: MouseEvent) => {
    const r = map.getContainer().getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };
  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || tool.peek() !== "select" || !display.peek().commuters || !shown) return;
    const [x, y] = rel(e);
    const p = { x, y, on: false, add: e.shiftKey, timer: 0 };
    p.timer = window.setTimeout(() => {
      p.on = true;
      map.dragPan.disable();
      map.getCanvas().style.cursor = "crosshair";
      box.style.display = "";
      Object.assign(box.style, { left: x + "px", top: y + "px", width: "0px", height: "0px" });
    }, LONG_MS);
    press = p;
  });
  window.addEventListener("mousemove", (e) => {
    const p = press;
    if (!p) return;
    const [x, y] = rel(e);
    if (!p.on) {
      if (Math.hypot(x - p.x, y - p.y) > STILL_PX) {
        clearTimeout(p.timer);
        press = null;
      }
      return;
    }
    Object.assign(box.style, { left: Math.min(x, p.x) + "px", top: Math.min(y, p.y) + "px", width: Math.abs(x - p.x) + "px", height: Math.abs(y - p.y) + "px" });
  });
  window.addEventListener("mouseup", (e) => {
    const p = press;
    press = null;
    if (!p) return;
    clearTimeout(p.timer);
    if (!p.on) return;
    box.style.display = "none";
    map.dragPan.enable();
    map.getCanvas().style.cursor = "";
    const [x, y] = rel(e);
    if (Math.hypot(x - p.x, y - p.y) <= STILL_PX) return; // a long press in place: the click decides
    const [ax, ay] = pointerLocal(map, p.x, p.y), [bx, by] = pointerLocal(map, x, y);
    const s = selectGroups(groupsIn(ax, ay, bx, by), p.add);
    if (s) selection.value = s;
  });

  // ---- the selection on the map and in the inspector ----
  let near: Groups | null = null, far: Groups | null = null;
  let nearDiscs: Float32Array | null = null, farDiscs: Float32Array | null = null;
  /** the far end summed into every level (in the worker: up to ~80k cells) */
  let farCells: Groups[] | null = null;
  let flowAsk = 0;
  let lastFlowKey: unknown = null;

  /** Near and far discs at the level on screen. */
  const drawPick = () => {
    const sel = selection.peek();
    if (!city || !shown || sel?.kind !== "commuters") {
      near = far = nearDiscs = farDiscs = null;
      view.setDiscs("near", null);
      view.setDiscs("far", null);
      return;
    }
    const level = bubbleLevel(map.getZoom());
    const g = city.levels[level];
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    const t = Float32Array.from(sel.cells, (c) => mt[c]), w = Float32Array.from(sel.cells, (c) => mw[c]), d = Float32Array.from(sel.cells, (c) => md[c]);
    near = aggregate(g, city.cellX, city.cellY, sel.cells, t, w, d);
    nearDiscs = discs(near, 1 / BUBBLE_DENSITY[sel.end], BUBBLE_ALPHA, 2);
    view.setDiscs("near", nearDiscs);
    if (farCells) {
      far = farCells[level];
      // sized among themselves: the biggest about the area of a bubble's hexagon
      let max = 0;
      for (let i = 0; i < far.group.length; i++) max = Math.max(max, far.train[i] + far.walk[i] + far.drive[i]);
      farDiscs = discs(far, max > 0 ? (0.6 * RES_AREA_M2[level]) / max : 0, BUBBLE_ALPHA, 1);
      view.setDiscs("far", farDiscs);
    } else {
      far = farDiscs = null;
      view.setDiscs("far", null);
    }
  };
  view.onLevel = () => drawPick();

  /** The far end's places, named by the station nearest them. */
  const places = (fc: Groups[]): Place[] => {
    if (!city) return [];
    const a = fc[2]; // resolution 7, about 5 km²: places a player knows
    const st = (world.peek()?.save.stations ?? []).map((s) => ({ name: s.name, at: toLocal(s.lng, s.lat) }));
    const sel = selection.peek();
    let cx = 0, cy = 0, n = 0;
    if (sel?.kind === "commuters") for (const c of sel.cells.slice(0, 2000)) (cx += city.cellX[c]), (cy += city.cellY[c]), n++;
    cx /= n || 1;
    cy /= n || 1;
    const byName = new Map<string, number>();
    // the 40 biggest areas are plenty for the 8 places listed, and keep this a millisecond
    const top = Array.from(a.group, (_, i) => i)
      .sort((p, q) => a.train[q] + a.walk[q] + a.drive[q] - (a.train[p] + a.walk[p] + a.drive[p]))
      .slice(0, 40);
    for (const i of top) {
      let best = Infinity, name = "";
      for (const s of st) {
        const dd = Math.hypot(s.at[0] - a.x[i], s.at[1] - a.y[i]);
        if (dd < best) (best = dd), (name = s.name);
      }
      if (best > 3000 * MERC_K) name = `${Math.max(1, Math.round(Math.hypot(a.x[i] - cx, a.y[i] - cy) / MERC_K / 1000))} km ${compass(a.x[i] - cx, -(a.y[i] - cy))}`;
      byName.set(name, (byName.get(name) ?? 0) + a.train[i] + a.walk[i] + a.drive[i]);
    }
    return [...byName].map(([name, riders]) => ({ name, riders })).sort((p, q) => q.riders - p.riders);
  };

  const refreshPick = () => {
    const sel = selection.peek();
    if (sel?.kind !== "commuters" || !shown) return;
    const t0 = performance.now();
    const m = shown.modes;
    const [mt, mw, md] = sel.end === "home" ? [m.homeRail, m.homeWalk, m.homeDrive] : [m.workRail, m.workWalk, m.workDrive];
    const totals: [number, number, number] = [0, 0, 0];
    for (const c of sel.cells) (totals[0] += mt[c]), (totals[1] += mw[c]), (totals[2] += md[c]);
    const prev = commuterPick.peek();
    const keepFar = prev?.sel === sel ? prev.far : null;
    commuterPick.value = { sel, totals, far: keepFar };
    drawPick();
    commuterStats.selectMs = performance.now() - t0;
    // the far end: asked again when the selection or the demand result changes
    const key = [sel, demandView.peek()];
    if ((lastFlowKey as unknown[] | null)?.every((v, i) => v === key[i])) return;
    lastFlowKey = key;
    const ask = ++flowAsk;
    const t1 = performance.now();
    void flows(sel.end, Uint32Array.from(sel.cells)).then(async (f) => {
      // null: the workers moved on; the next demand result asks again
      if (ask !== flowAsk || !f || selection.peek() !== sel || !worker) return;
      commuterStats.flowsMs = performance.now() - t1;
      commuterStats.farCells = f.cells.length;
      const ft: [number, number, number] = [f.rail - f.cutRail, f.walk - f.cutWalk, f.drive - f.cutDrive];
      // summed into the bubble levels in the worker, not here
      const id = nextId++;
      const summed = await new Promise<Summed>((done) => {
        pending.set(id, done);
        const msg: ToCommuters = { kind: "sum", id, cells: f.cells, train: f.farRail, walk: f.farWalk, drive: f.farDrive };
        worker!.postMessage(msg);
      });
      if (ask !== flowAsk || selection.peek() !== sel) return;
      const t2 = performance.now();
      farCells = summed.levels;
      commuterPick.value = { sel, totals, far: { totals: ft, places: places(farCells) } };
      drawPick();
      commuterStats.farMs = performance.now() - t2;
      commuterStats.sumMs = summed.ms;
    });
  };
  effect(() => {
    const sel = selection.value;
    demandView.value;
    if (sel?.kind !== "commuters") {
      flowAsk++;
      farCells = null;
      lastFlowKey = null;
      commuterPick.value = null;
      drawPick();
      return;
    }
    if (commuterPick.peek()?.sel !== sel) {
      farCells = null;
      lastFlowKey = null;
    }
    refreshPick();
  });
  // a new end in the settings drops a selection of the other end
  effect(() => {
    const d = display.value;
    const sel = selection.peek();
    if (sel?.kind === "commuters" && (!d.commuters || sel.end !== d.commuterEnd)) selection.value = null;
  });

  return { click };
}
