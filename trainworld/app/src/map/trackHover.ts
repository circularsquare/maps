// Hovering track (T-157): rest the pointer on a stretch of track for DWELL_MS and a tag by the
// pointer says what the current track display draws there: the line's name (Line), its forecast
// riders a day on that stop-to-stop stretch (Line + traffic thickness), the level (Height) or the
// geometry's speed limit (Max speed). The tag is the demand views' (demandHover.ts) and follows the
// map's pick order (T-133): demand bubbles, capacity tags and editing handles above the network
// hide it; trains, stations, platform crowds and station demand circles have tags of their own
// (T-194, networkTips.ts). Select tool only: the other tools have tags of their own.
// Once shown it follows the pointer along the track and updates; leaving the track hides it, and
// the next one needs another rest.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { clock } from "../game/clock";
import { demandView, lineLoadsFor, type DemandView } from "../game/demand";
import { levelLabel, speed } from "../game/format";
import { purposeName, type PurposeFilter } from "../game/purposes";
import type { WorldView } from "../game/types";
import { ask, display, purposeFilter, selection, tool } from "../game/ui";
import { world } from "../game/world";
import { hoverAt } from "./demandHover";
import { live } from "./live";
import type { NetworkBuffers } from "./network";
import { circleTip, stationTip, trainTip } from "./networkTips";
import type { Overlay } from "./overlay";
import { pick, SLOP_PX, topSeg } from "./pick";

/** How long the pointer rests on track before the tag shows, ms (Anita 2026-10-12: about 100, responsive). */
export const DWELL_MS = 100;

/** What the Height display's colours mean, by level (as the Build tab names them). */
const LEVEL_NAME: Record<number, string> = {
  3: "High viaduct",
  2: "Viaduct",
  1: "Viaduct",
  0: "Ground level",
  [-1]: "Cut and cover",
  [-2]: "Tunnel",
  [-3]: "Deep tunnel",
};

/** A forecast count, rounded so it does not claim more precision than a forecast has. */
export function riders(n: number): string {
  const r = n >= 1000 ? Math.round(n / 100) * 100 : n >= 100 ? Math.round(n / 10) * 10 : Math.round(n);
  return r.toLocaleString("en-US");
}

/** The Height display's text for a stroke level (rounded height / 8 m, as the shader colours it). */
export function levelText(level: number): string {
  const l = Math.max(-3, Math.min(3, Math.round(level)));
  return l === 0 ? "Ground level" : `${LEVEL_NAME[l]} at level ${levelLabel(l)}`;
}

export function speedText(kmh: number): string {
  return `Max speed ${speed(kmh)}`;
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
  const count = day < 0.5 ? "No forecast riders on this stretch" : `${riders(day)} forecast riders a day in both directions`;
  // the purpose filter (T-158) says what it counted
  return purpose === "all" ? [name, count] : [name, count, `${purposeName(purpose)} trips only`];
}

/** The tag's lines at a point (CSS px in the map), or null: what the current track display draws
 * on the line or track `pick` finds there; for a train, a station, a platform's crowd or a station
 * demand circle, what they stand for (T-194, networkTips.ts). Nothing for junctions. */
export function trackTip(ov: Overlay, px: number, py: number, zoom = 12): string[] | null {
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
  if (!hit) return w ? circleTip(ov, w, px, py, zoom) : null;
  const slop = SLOP_PX * SLOP_PX + 1;
  if (hit.kind === "line") {
    const L = n.lines;
    // a selected line's highlight is drawn above the others (pick.ts)
    const sel = selection.peek();
    // as wide as drawn (traffic thickness, T-227), as pick() takes it
    const wide = ov.renderer.lineWidths, widePx = ov.renderer.slotPx;
    let [i] = sel?.kind === "line" ? topSeg(ov, L.seg, L.count, px, py, slop, undefined, 0, (j) => String(L.colour[j]) === sel.line, wide, widePx) : [-1];
    if (i < 0) [i] = topSeg(ov, L.seg, L.count, px, py, slop, undefined, 0, (j) => String(L.colour[j]) === hit.line, wide, widePx);
    return i < 0 ? null : lineTip(n, w, demandView.peek(), d.trackColour === "traffic", i);
  }
  if (hit.kind !== "track") return null;
  const T = n.track;
  const [i] = topSeg(ov, T.seg, T.count, px, py, slop, undefined, 0, (j) => T.edge[j] === hit.edge);
  if (i < 0) return null;
  if (d.trackColour === "height") return [levelText(T.level[i])];
  if (d.trackColour === "speed") return T.speed?.[i] !== undefined ? [speedText(T.speed[i])] : null;
  // Line views: bare track. Track under a bundle of lines loses to the lines, except at its edges.
  const used = w ? (w.edges.find((e) => e.id === hit.edge)?.lines.length ?? 0) > 0 : lineOnEdge(n, hit.edge);
  return used ? null : ["No line runs here"];
}

function lineOnEdge(n: NetworkBuffers, edge: number): boolean {
  for (let i = 0; i < n.lines.count; i++) if (n.lines.edge[i] === edge) return true;
  return false;
}

/** The tag in the map: shown after the pointer rests on track, moved and refilled on moves. */
export function startTrackHover(map: MlMap, ov: Overlay) {
  const box = map.getContainer();
  const tag = document.createElement("div");
  tag.className = "tool-tag demand-tag track-tag";
  tag.style.display = "none";
  box.appendChild(tag);
  let timer = 0;
  let shown = false;
  let last = "";
  /** the pointer, CSS px in the map; null when it is not over the map's canvas */
  let at: [number, number] | null = null;

  const hide = () => {
    if (timer) clearTimeout(timer);
    timer = 0;
    shown = false;
    tag.style.display = "none";
  };
  /** The tag's lines at the pointer, or null where something above the track is, or a tool is on. */
  const tipAt = (px: number, py: number): string[] | null => {
    if (tool.peek() !== "select" || ask.peek() || live.nodeDrag?.dragging) return null;
    if (hoverAt(map, px, py)) return null; // a demand bubble above the network
    if (live.nodeDrag?.hit(px, py)?.movable) return null; // a blueprint node handle
    return trackTip(ov, px, py, map.getZoom());
  };
  const show = (lines: string[], px: number, py: number) => {
    const html = lines.join("\n");
    if (html !== last) {
      tag.replaceChildren(...lines.map((s) => Object.assign(document.createElement("div"), { textContent: s })));
      last = html;
    }
    tag.style.display = "";
    tag.style.transform = `translate(${Math.round(px + 14)}px, ${Math.round(py + 14)}px)`;
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
    if (shown) return update();
    if (timer) clearTimeout(timer);
    timer = window.setTimeout(() => {
      timer = 0;
      update();
    }, DWELL_MS);
  });
  box.addEventListener("mousedown", () => hide());
  box.addEventListener("mouseleave", () => {
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
