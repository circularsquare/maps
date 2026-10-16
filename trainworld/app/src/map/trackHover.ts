// Hovering track (T-157): a tag by the pointer says what the current track display draws there:
// the line's name (Line), its forecast riders a day on that stop-to-stop stretch (Line + traffic
// thickness), and always the elevation and speed limit there plus the lines on the track (T-473,
// T-526). It shows at once. The tag follows the map's pick order (T-133): with the select tool,
// demand bubbles, capacity tags and editing handles above the network hide it; with a build tool
// the demand views are beneath the network, and the tag sits above the pointer, clear of the
// tool's own tag. Trains, stations, platform crowds and station demand circles have tags of their
// own (T-194, networkTips.ts).

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { demandView, lineLoadsFor, type DemandView } from "../game/demand";
import { purposeName, type PurposeFilter } from "../game/purposes";
import type { LineInput, WorldView } from "../game/types";
import { ask, display, isBuildTool, purposeFilter, selection, shownDayPhrase, tool } from "../game/ui";
import { world } from "../game/world";
import { hoverAt } from "./demandHover";
import { hoverHeld } from "../ui/blurHold";
import { live } from "./live";
import type { NetworkBuffers } from "./network";
import { circleTip, stationTip, trainTip } from "./networkTips";
import type { Overlay } from "./overlay";
import { pick, SLOP_PX, topSeg } from "./pick";
import { elevationFact, factRow, lineBullets, speedFact } from "./trackTagRows";

/** A line of the tag: plain text, or a row of facts with icons. */
type TipLine = string | HTMLElement;

/** A forecast count, rounded so it does not claim more precision than a forecast has. */
export function riders(n: number): string {
  const r = n >= 1000 ? Math.round(n / 100) * 100 : n >= 100 ? Math.round(n / 10) * 10 : Math.round(n);
  return r.toLocaleString("en-US");
}

/** Daily riders on run 0's stretch `k` of a line, both directions: every period's load summed;
 * null when demand has no answer for it that fits its stops. `purpose`: the map's purpose
 * filter (T-158), that purpose's riders only. */
export function dailyRiders(dv: DemandView | null, lineId: string, stops: number, k: number, purpose: PurposeFilter = "all"): number | null {
  const loads = lineLoadsFor(dv, lineId, purpose);
  if (!loads?.length || stops < 2 || k < 0 || k > stops - 2) return null;
  let sum = 0;
  for (const p of loads) {
    if (p.length !== 2 * (stops - 1)) return null;
    // run 1 runs the stops backwards: its stretch j covers run 0's stretch n - 2 - j
    sum += p[k] + p[stops - 1 + (stops - 2 - k)];
  }
  return sum;
}

/** The tag's lines for a line stroke segment `i`. */
function lineTip(n: NetworkBuffers, w: WorldView | null, dv: DemandView | null, traffic: boolean, i: number): string[] {
  const num = n.lines.colour[i];
  const l = w?.save.lines.find((x) => x.num === num);
  const name = l?.name ?? `Line ${num}`;
  const status = l ? w?.lineStats[l.id]?.status : undefined;
  if (status === "planned") return [name, "Not running yet"];
  if (status === "broken") return [name, "Not running"];
  if (!l || !traffic) return [name];
  if (!dv) return [name, "Calculating demand"];
  const k = n.lines.info ? n.lines.info[i * 4] : -1;
  const purpose = purposeFilter.peek();
  const day = dailyRiders(dv, l.id, l.stops.length, k, purpose);
  if (day === null) return [name, "Calculating demand"];
  const count = day < 0.5 ? "No forecast riders on this stretch" : `${riders(day)} forecast riders ${shownDayPhrase.peek()} in both directions`;
  // the purpose filter (T-158) says what it counted
  return purpose === "all" ? [name, count] : [name, count, `${purposeName(purpose)} trips only`];
}

/** The tag's lines at a point (CSS px in the map), or null: what the current track display draws
 * on the line or track `pick` finds there; for a train, a station, a platform's crowd or a station
 * demand circle, what they stand for (T-194, networkTips.ts). Nothing for junctions, nor for the
 * station circles while a build tool has the demand views beneath the network. */
export function trackTip(ov: Overlay, px: number, py: number, zoom = 12): TipLine[] | null {
  const n = ov.renderer.net;
  if (!n || !ov.camMatrix) return null;
  const d = display.peek();
  const hit = pick(ov, px, py, d.trains);
  const w = world.peek();
  if (w && hit?.kind === "train") return trainTip(ov, w, hit.line, hit.profile, hit.dep);
  if (w && hit?.kind === "station") {
    const i = n.stationIds.indexOf(hit.station);
    return i < 0 ? null : stationTip(ov, w, i);
  }
  // the demand circles are drawn under the network
  if (!hit) return w && !isBuildTool(tool.peek()) ? circleTip(ov, w, px, py, zoom) : null;
  const slop = SLOP_PX * SLOP_PX + 1;
  if (hit.kind === "line") {
    const L = n.lines;
    // a selected line's highlight is drawn above the others (pick.ts)
    const sel = selection.peek();
    // as wide as drawn (traffic thickness, T-227), as pick() takes it
    const wide = ov.renderer.lineWidths, widePx = ov.renderer.slotPx;
    let [i] = sel?.kind === "line" ? topSeg(ov, L.seg, L.count, px, py, slop, undefined, 0, (j) => String(L.colour[j]) === sel.line, wide, widePx) : [-1];
    if (i < 0) [i] = topSeg(ov, L.seg, L.count, px, py, slop, undefined, 0, (j) => String(L.colour[j]) === hit.line, wide, widePx);
    if (i < 0) return null;
    // T-473: the track under the line says its elevation and speed too
    return [...lineTip(n, w, demandView.peek(), d.trackColour === "traffic", i), ...trackFacts(ov, L.edge[i], px, py, slop)];
  }
  if (hit.kind !== "track") return null;
  const facts = trackFacts(ov, hit.edge, px, py, slop);
  if (!facts.length) return null;
  // T-473: always the elevation, speed and lines on it, whatever the track display; no lines row
  // where none runs (T-526). Track under a bundle of lines loses to the lines, except at its edges.
  const on = linesOn(w, hit.edge);
  if (on.length) return [...facts, lineBullets(on)];
  return !w && lineOnEdge(n, hit.edge) ? [...facts, "A line runs here"] : facts;
}

/** The lines whose path uses an edge, in the save's order. */
function linesOn(w: WorldView | null, edge: number): LineInput[] {
  const e = w?.edges.find((x) => x.id === edge);
  return e ? w!.save.lines.filter((l) => e.lines.includes(l.id)) : [];
}

/** The elevation and speed limit at a point of an edge's track stroke, a row each (T-473,
 * T-526); none off the stroke. */
function trackFacts(ov: Overlay, edge: number, px: number, py: number, slop: number): HTMLElement[] {
  const T = ov.renderer.net!.track;
  const [i] = topSeg(ov, T.seg, T.count, px, py, slop, undefined, 0, (j) => T.edge[j] === edge);
  if (i < 0) return [];
  const rows = [factRow(elevationFact(T.level[i]))];
  if (T.speed?.[i] !== undefined) rows.push(factRow(speedFact(T.speed[i])));
  return rows;
}

function lineOnEdge(n: NetworkBuffers, edge: number): boolean {
  for (let i = 0; i < n.lines.count; i++) if (n.lines.edge[i] === edge) return true;
  return false;
}

/** The tag in the map: shown while the pointer is on track, moved and refilled on moves. */
export function startTrackHover(map: MlMap, ov: Overlay) {
  const box = map.getContainer();
  const tag = document.createElement("div");
  tag.className = "tool-tag demand-tag track-tag";
  tag.style.display = "none";
  box.appendChild(tag);
  let shown = false;
  let last = "";
  /** the pointer, CSS px in the map; null when it is not over the map's canvas */
  let at: [number, number] | null = null;

  const hide = () => {
    shown = false;
    tag.style.display = "none";
  };
  /** The tag's lines at the pointer, or null where something above the track is, or a tool
   * without it is on (the stop editor and the trip tool). */
  const tipAt = (px: number, py: number): TipLine[] | null => {
    const t = tool.peek();
    if (ask.peek()) return null;
    if (t === "select") {
      if (live.nodeDrag?.dragging) return null;
      if (hoverAt(map, px, py)) return null; // a demand bubble above the network
      if (live.nodeDrag?.hit(px, py)?.movable) return null; // a blueprint node handle
    } else if (!isBuildTool(t)) return null;
    return trackTip(ov, px, py, map.getZoom());
  };
  const show = (lines: TipLine[], px: number, py: number) => {
    const key = lines.map((s) => (typeof s === "string" ? s : s.outerHTML)).join("\n");
    if (key !== last) {
      tag.replaceChildren(...lines.map((s) => (typeof s === "string" ? Object.assign(document.createElement("div"), { textContent: s }) : s)));
      last = key;
    }
    tag.style.display = "";
    // a build tool's own tag is below right of the pointer, so this one goes above
    tag.style.transform = isBuildTool(tool.peek())
      ? `translate(${Math.round(px + 14)}px, ${Math.round(py - 10)}px) translateY(-100%)`
      : `translate(${Math.round(px + 14)}px, ${Math.round(py + 14)}px)`;
    shown = true;
  };
  const update = () => {
    if (!at) return hide();
    const t = tipAt(at[0], at[1]);
    if (t) show(t, at[0], at[1]);
    else hide();
  };

  // On the container, not MapLibre's events: capacity tags and handles are DOM targets above the
  // canvas that MapLibre does not hear, and they must hide the tag too.
  box.addEventListener("mousemove", (e) => {
    const r = box.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    // anything but the bare map canvas (a capacity tag, a handle, a control) is above the track
    if (e.buttons || e.target !== map.getCanvas()) {
      at = null;
      return hide();
    }
    at = [px, py];
    update();
  });
  box.addEventListener("mousedown", () => hide());
  box.addEventListener("mouseleave", () => {
    if (hoverHeld()) return;
    at = null;
    hide();
  });
  box.addEventListener("wheel", () => hide(), { passive: true });
  map.on("movestart", hide);
  // What is drawn changed under a resting pointer: say what is there now. After the current task,
  // as the clock client sets the world before handing the renderer its network.
  effect(() => {
    world.value;
    demandView.value;
    display.value;
    purposeFilter.value;
    selection.value;
    tool.value;
    ask.value;
    clock.minute.value; // trains move and crowds change under a resting pointer (T-194)
    if (shown) queueMicrotask(() => shown && update());
  });
}
