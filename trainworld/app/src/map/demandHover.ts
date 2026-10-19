// Hovering the demand views' discs (T-097): a small tag by the pointer with the number of people
// in the bubble under it. Each view registers a finder; the topmost layer's answer wins.

import type { Map as MlMap } from "maplibre-gl";
import { DISC } from "../workers/commutersProtocol";
import { EARTH_C, toLocal, type Anchor } from "./geo";
import { hoverHeld } from "../ui/blurHold";
import { restHolds } from "./hoverRest";

/** Where the pointer is on the ground. Each disc set is in the local units of its own draw anchor
 * (T-317, map/geo.ts), so a finder turns the pointer into those (`discAt`, `pointerLocal`). */
export interface Pointer {
  lng: number;
  lat: number;
  /** local units per CSS px at the current zoom; without it discs have no minimum pick radius */
  lpx?: number;
}
/** A disc drawn smaller than this radius (CSS px) is still hit within it (T-566). */
export const MIN_PICK_PX = 8;
/** A finder gets the pointer and answers the tag's lines, or null. */
export type Finder = (p: Pointer) => string[] | null;
const finders: { order: number; f: Finder }[] = [];

/** Higher `order` is drawn higher and asked first. */
export function addFinder(order: number, f: Finder) {
  finders.push({ order, f });
  finders.sort((a, b) => b.order - a.order);
}

/** The topmost disc of a set (drawn last, built relative to anchor `a`) containing the pointer,
 * or -1; radii times `scale`. A disc drawn under MIN_PICK_PX counts as that wide, unless the
 * topmost disc containing the pointer is drawn above it; where those wider areas overlap the
 * nearest centre wins (T-566). */
export function discAt(discs: Float32Array | null, p: Pointer, a: Anchor, scale = 1): number {
  if (!discs) return -1;
  const [x, y] = toLocal(p.lng, p.lat, a);
  const n = discs.length / DISC, minR = (p.lpx ?? 0) * MIN_PICK_PX;
  let top = -1;
  for (let i = n - 1; i >= 0; i--) {
    const dx = discs[i * DISC] - x, dy = discs[i * DISC + 1] - y, r = discs[i * DISC + 2] * scale;
    if (dx * dx + dy * dy <= r * r) {
      top = i;
      break;
    }
  }
  if (!minR) return top;
  let best = top, bestD = Infinity;
  for (let i = n - 1; i > top; i--) {
    const r = discs[i * DISC + 2] * scale;
    if (!(r > 0) || r >= minR || !(discs[i * DISC + 6] > 0)) continue;
    const dx = discs[i * DISC] - x, dy = discs[i * DISC + 1] - y, d = dx * dx + dy * dy;
    if (d <= minR * minR && d < bestD) (best = i), (bestD = d);
  }
  return best;
}

/** Pointer (CSS px in the map) to a place on the ground. */
export function pointerAt(map: MlMap, px: number, py: number): Pointer {
  const ll = map.unproject([px, py]);
  return { lng: ll.lng, lat: ll.lat, lpx: EARTH_C / (512 * 2 ** map.getZoom()) };
}

/** The pointer in local units of anchor `a`. */
export function pointerLocal(p: Pointer, a: Anchor): [number, number] {
  return toLocal(p.lng, p.lat, a);
}

/** Points where something drawn above the discs is (the selected station, T-189): no disc there. */
export const hoverAbove: { at: ((px: number, py: number) => boolean) | null } = { at: null };

/** `under`: the bubble even beneath what is drawn above the discs (the resting hover's list, T-732). */
export function hoverAt(map: MlMap, px: number, py: number, under = false): string[] | null {
  if (!finders.length || (!under && hoverAbove.at?.(px, py))) return null;
  const p = pointerAt(map, px, py);
  for (const { f } of finders) {
    const t = f(p);
    if (t) return t;
  }
  return null;
}

/** The tag itself: one DOM element in the map, moved and filled on mouse moves. */
export function startHover(map: MlMap) {
  const tag = document.createElement("div");
  tag.className = "tool-tag demand-tag";
  tag.style.display = "none";
  map.getContainer().appendChild(tag);
  let last = "";
  const hide = () => {
    tag.style.display = "none";
    last = "";
  };
  const show = (e: MouseEvent, px: number, py: number) => {
    if (restHolds(px, py)) return; // on the way to the resting hover's list (T-732)
    if (e.buttons) return hide();
    const t = hoverAt(map, px, py);
    if (!t) return hide();
    const html = t.filter(Boolean).map((s) => (s.startsWith("<div") ? s : `<div>${s}</div>`)).join("");
    if (html !== last) {
      tag.innerHTML = html;
      last = html;
    }
    tag.style.display = "";
    tag.style.transform = `translate(${Math.round(px + 14)}px, ${Math.round(py + 14)}px)`;
  };
  // after the move's other handlers, so a build tool's tag is up to date (main.ts, T-526)
  map.on("mousemove", (e) => queueMicrotask(() => show(e.originalEvent, e.point.x, e.point.y)));
  // A lower capacity tag is a DOM target even when an upper demand disc covers it. MapLibre
  // canvas events need not fire there, so use the actual pointer for those targets too.
  map.getContainer().addEventListener("mousemove", (e) => {
    if (!(e.target as HTMLElement | null)?.closest?.(".cap-mark")) return;
    const r = map.getContainer().getBoundingClientRect();
    show(e, e.clientX - r.left, e.clientY - r.top);
  });
  map.getContainer().addEventListener("mouseleave", () => hoverHeld() || hide());
  map.on("movestart", hide);
}
