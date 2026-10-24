// The resting hover (T-732): when the pointer rests 250 ms on the map with the select tool, a list
// under the hover tag names what else is under it: at most one demand bubble, one station and
// three lines, the thing the tag already describes counted in those limits. Clicking an entry
// selects that thing. The query runs once per rest, not per move (2-4 ms on the real save, T-724).
//
// While the list is up, the pointer may cross the stretch between where it rested and the list
// (the hull of that point and the tag with its list) without the tags moving, so the list can be
// reached; leaving that stretch drops the list and the tags follow the pointer again.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import type { Selection, WorldView } from "../game/types";
import { ask, display, selection, tool } from "../game/ui";
import { world } from "../game/world";
import { hoverHeld } from "../ui/blurHold";
import { complexOf } from "../game/complexView";
import { hoverAt, hoverDisc } from "./demandHover";
import { setListHot, type Hot } from "./hovered";
import { markAt } from "./gradeMarks";
import { TAG_OFFSET_PX } from "./labelLayout";
import { live } from "./live";
import type { Overlay } from "./overlay";
import { linesAt, pick, SLOP_PX } from "./pick";
import { lineBullet } from "./trackTagRows";

const REST_MS = 250;
const MAX_LINES = 3;

/** One entry: its row and what clicking it selects. */
interface Entry {
  row: HTMLElement;
  select: () => Selection;
  /** what pointing at it lights on the map (T-741) */
  hot: Partial<Hot>;
}

/** where the pointer rested, CSS px in the map, while the list is up */
let restAt: [number, number] | null = null;
/** the corners of the tag and list together, below right of `restAt` */
let box: { l: number; t: number; r: number; b: number } | null = null;

/** Whether the pointer at (px, py) is on its way from the rest point to the list, or on it: the
 * hover tags then stay as they are. */
export function restHolds(px: number, py: number): boolean {
  if (!restAt || !box) return false;
  const [x0, y0] = restAt;
  if (Math.hypot(px - x0, py - y0) <= 4) return true;
  if (px >= box.l && px <= box.r && py >= box.t && py <= box.b) return true;
  // the hull of the rest point and the box's far corners: rest point, top right, bottom right, bottom left
  const poly: [number, number][] = [[x0, y0], [box.r, box.t], [box.r, box.b], [box.l, box.b]];
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** `bubbleSelect`: what clicking the demand bubble at a point selects (main.ts `demandOnTop`). */
export function startRestList(map: MlMap, ov: Overlay, bubbleSelect: (x: number, y: number) => Selection | undefined) {
  const container = map.getContainer();
  const el = document.createElement("div");
  el.className = "tool-tag demand-tag rest-list";
  el.style.display = "none";
  container.appendChild(el);
  let timer = 0;

  const hide = () => {
    clearTimeout(timer);
    if (!restAt) return;
    restAt = box = null;
    el.style.display = "none";
    el.replaceChildren();
    setListHot(null);
  };

  /** The entries at a point, without the one the hover tag already describes. */
  const entriesAt = (px: number, py: number): Entry[] => {
    const w = world.peek(), n = ov.renderer.net;
    if (!w || !n || !ov.camMatrix) return [];
    // with the select tool the bubbles are above the network, so the tag shows a bubble when there
    // is one, unless the selected station is drawn over it
    const bubbleOnTop = hoverAt(map, px, py);
    const hit = bubbleOnTop ? null : pick(ov, px, py, display.peek().trains);
    const marked = hit?.kind === "line" && n.marks && markAt(ov, n.marks, px, py, map.getZoom()) >= 0;
    const out: Entry[] = [];
    const under = bubbleOnTop ? null : hoverAt(map, px, py, true);
    const disc = hoverDisc();
    const text = under?.find((s) => s && !s.startsWith("<"));
    if (text) out.push({ row: row(text, Object.assign(document.createElement("span"), { className: "rest-bub" })), select: () => bubbleSelect(px, py) ?? null, hot: { disc } });
    const si = ov.renderer.stationAt(px, py, (x, y) => ov.toScreen(x, y), SLOP_PX);
    const sid = si >= 0 ? n.stationIds[si] : null;
    const s = sid !== null ? w.save.stations.find((x) => x.id === sid) : undefined;
    if (s && !(hit?.kind === "station" && hit.station === s.id)) {
      out.push({ row: row(s.name, Object.assign(document.createElement("span"), { className: "rest-stn" })), select: () => ({ kind: "station", station: s.id }), hot: { stations: complexOf(s.id) } });
    }
    for (const id of linesAt(ov, px, py, selection.peek(), MAX_LINES)) {
      if (hit?.kind === "line" && hit.line === id && !marked) continue;
      const l = w.save.lines.find((x) => x.id === id) ?? w.save.lines.find((x) => String(x.num) === id);
      if (!l) continue;
      out.push({ row: row(l.name, lineBullet(l, true)), select: () => ({ kind: "line", line: id }), hot: { line: l.num } });
    }
    return out;
  };

  const rest = (px: number, py: number) => {
    if (tool.peek() !== "select" || ask.peek() || live.nodeDrag?.dragging) return;
    const entries = entriesAt(px, py);
    if (!entries.length) return;
    el.replaceChildren(...entries.map((e) => {
      e.row.onclick = (ev) => {
        ev.stopPropagation();
        const sel = e.select();
        hide();
        if (sel) selection.value = sel;
      };
      e.row.onmouseenter = () => setListHot(e.hot);
      e.row.onmouseleave = () => setListHot(null);
      return e.row;
    }));
    // under whichever hover tag is up, flush with it
    let top = py + TAG_OFFSET_PX;
    let right = px + TAG_OFFSET_PX;
    for (const t of container.querySelectorAll<HTMLElement>(".demand-tag:not(.rest-list)")) {
      if (t.style.display === "none" || !t.offsetHeight) continue;
      top = Math.max(top, py + TAG_OFFSET_PX + t.offsetHeight - 1);
      right = Math.max(right, px + TAG_OFFSET_PX + t.offsetWidth);
    }
    el.style.display = "";
    el.style.transform = `translate(${Math.round(px + TAG_OFFSET_PX)}px, ${Math.round(top)}px)`;
    restAt = [px, py];
    restKey = networkKey(world.peek()!);
    box = { l: px + TAG_OFFSET_PX, t: py + TAG_OFFSET_PX, r: Math.max(right, px + TAG_OFFSET_PX + el.offsetWidth), b: top + el.offsetHeight };
  };

  container.addEventListener("mousemove", (e) => {
    const r = container.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (restHolds(px, py)) return;
    hide();
    if (e.buttons || e.target !== map.getCanvas()) return;
    timer = window.setTimeout(() => rest(px, py), REST_MS);
  });
  // a press on the list is the list's own, never the map's (no drag, no pick beneath it)
  for (const k of ["mousedown", "pointerdown", "dblclick"] as const) el.addEventListener(k, (e) => e.stopPropagation());
  container.addEventListener("mousedown", hide);
  container.addEventListener("wheel", hide, { passive: true });
  container.addEventListener("mouseleave", () => hoverHeld() || hide());
  map.on("movestart", hide);
  effect(() => {
    selection.value;
    tool.value;
    display.value;
    ask.value;
    hide();
  });
  // Most world changes (demand settling, the clock's live figures) leave the network as it was; only
  // an edit that changes what is under the pointer drops the list (T-753). The key is built only
  // while the list is up.
  effect(() => {
    const w = world.value;
    if (restAt && w && networkKey(w) !== restKey) hide();
  });
}

/** the network's shape when the list came up */
let restKey = "";

/** What decides what is under a point: track, which lines run where, stations and their places,
 * line names. */
function networkKey(w: WorldView): string {
  const e = w.edges.map((x) => `${x.id}:${x.a}:${x.b}:${x.lengthM}:${x.lines.join(",")}`).join("|");
  const s = w.save.stations.map((x) => `${x.id}:${x.x}:${x.y}:${x.heading}:${x.length}:${x.name}`).join("|");
  const l = w.save.lines.map((x) => `${x.id}:${x.num}:${x.name}`).join("|");
  return `${e}/${s}/${l}`;
}

function row(text: string, lead?: HTMLElement): HTMLDivElement {
  const d = document.createElement("div");
  d.className = "rest-entry";
  if (lead) d.append(lead);
  d.append(Object.assign(document.createElement("span"), { textContent: text }));
  return d;
}
