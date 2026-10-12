// Hovering the demand views' discs (T-097): a small tag by the pointer with the number of people
// in the bubble under it. Each view registers a finder; the topmost layer's answer wins.

import type { Map as MlMap } from "maplibre-gl";
import { DISC } from "../workers/commutersProtocol";
import { toLocal } from "./geo";

/** A finder gets the pointer in local units and answers the tag's lines, or null. */
export type Finder = (x: number, y: number) => string[] | null;
const finders: { order: number; f: Finder }[] = [];

/** Higher `order` is drawn higher and asked first. */
export function addFinder(order: number, f: Finder) {
  finders.push({ order, f });
  finders.sort((a, b) => b.order - a.order);
}

/** The topmost disc of a set (drawn last) containing the point, or -1; radii times `scale`. */
export function discAt(discs: Float32Array | null, x: number, y: number, scale = 1): number {
  if (!discs) return -1;
  for (let i = discs.length / DISC - 1; i >= 0; i--) {
    const dx = discs[i * DISC] - x, dy = discs[i * DISC + 1] - y, r = discs[i * DISC + 2] * scale;
    if (dx * dx + dy * dy <= r * r) return i;
  }
  return -1;
}

/** Pointer (CSS px in the map) to local units. */
export function pointerLocal(map: MlMap, px: number, py: number): [number, number] {
  const ll = map.unproject([px, py]);
  return toLocal(ll.lng, ll.lat);
}

export function hoverAt(map: MlMap, px: number, py: number): string[] | null {
  if (!finders.length) return null;
  const [x, y] = pointerLocal(map, px, py);
  for (const { f } of finders) {
    const t = f(x, y);
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
    if (e.buttons) return hide();
    const t = hoverAt(map, px, py);
    if (!t) return hide();
    const html = t.map((s) => `<div>${s}</div>`).join("");
    if (html !== last) {
      tag.innerHTML = html;
      last = html;
    }
    tag.style.display = "";
    tag.style.transform = `translate(${Math.round(px + 14)}px, ${Math.round(py + 14)}px)`;
  };
  map.on("mousemove", (e) => show(e.originalEvent, e.point.x, e.point.y));
  // A lower capacity tag is a DOM target even when an upper demand disc covers it. MapLibre
  // canvas events need not fire there, so use the actual pointer for those targets too.
  map.getContainer().addEventListener("mousemove", (e) => {
    if (!(e.target as HTMLElement | null)?.closest?.(".cap-mark")) return;
    const r = map.getContainer().getBoundingClientRect();
    show(e, e.clientX - r.left, e.clientY - r.top);
  });
  map.getContainer().addEventListener("mouseleave", hide);
  map.on("movestart", hide);
}
