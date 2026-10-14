// Hovering the demand views' discs (T-097): a small tag by the pointer with the number of people
// in the bubble under it. Each view registers a finder; the topmost layer's answer wins.

import type { Map as MlMap } from "maplibre-gl";
import { DISC } from "../workers/commutersProtocol";
import { closestStations, minLabel, walkGridEpoch } from "../game/walks";
import { world } from "../game/world";
import { HOME_ANCHOR, toLngLat, toLocal } from "./geo";

/** A finder gets the pointer in local units of the home anchor (the demand views stay on it for
 * now, T-307, map/geo.ts) and answers the tag's lines, or null. */
export type Finder = (x: number, y: number) => string[] | null;
const finders: { order: number; f: Finder }[] = [];

/** Higher `order` is drawn higher and asked first. */
export function addFinder(order: number, f: Finder) {
  finders.push({ order, f });
  finders.sort((a, b) => b.order - a.order);
}

/** The centre of the disc `discAt` found last (local units): the tag lists the stations closest to it. */
let lastDisc: [number, number] | null = null;

/** The topmost disc of a set (drawn last) containing the point, or -1; radii times `scale`. */
export function discAt(discs: Float32Array | null, x: number, y: number, scale = 1): number {
  if (!discs) return -1;
  for (let i = discs.length / DISC - 1; i >= 0; i--) {
    const dx = discs[i * DISC] - x, dy = discs[i * DISC + 1] - y, r = discs[i * DISC + 2] * scale;
    if (dx * dx + dy * dy <= r * r) {
      lastDisc = [discs[i * DISC], discs[i * DISC + 1]];
      return i;
    }
  }
  return -1;
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
let closest: { key: string; lines: string[] } | null = null;
/** T-312: a bubble's 3 closest stations with their walking minutes (game/walks.ts), built or
 * blueprint; kept while the pointer stays on the bubble. */
function closestLines(at: [number, number]): string[] {
  const w = world.peek();
  if (!w) return [];
  const key = `${at[0].toFixed(1)},${at[1].toFixed(1)},${w.version},${walkGridEpoch.peek()}`;
  if (closest?.key === key) return closest.lines;
  const [lng, lat] = toLngLat(at[0], at[1], HOME_ANCHOR);
  const near = closestStations(lng, lat, w.save.stations);
  const lines = near.length
    ? ["Closest stations", ...near.map((n) => `${esc(n.station.name)}${n.station.built ? "" : " (blueprint)"}, ${n.metres === null ? "no walk" : minLabel(n.metres)}`)]
    : [];
  closest = { key, lines };
  return lines;
}

/** Pointer (CSS px in the map) to local units of the home anchor. */
export function pointerLocal(map: MlMap, px: number, py: number): [number, number] {
  const ll = map.unproject([px, py]);
  return toLocal(ll.lng, ll.lat, HOME_ANCHOR);
}

/** Points where something drawn above the discs is (the selected station, T-189): no disc there. */
export const hoverAbove: { at: ((px: number, py: number) => boolean) | null } = { at: null };

export function hoverAt(map: MlMap, px: number, py: number): string[] | null {
  if (!finders.length || hoverAbove.at?.(px, py)) return null;
  const [x, y] = pointerLocal(map, px, py);
  for (const { f } of finders) {
    lastDisc = null;
    const t = f(x, y);
    if (t) return lastDisc ? [...t, ...closestLines(lastDisc)] : t;
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
