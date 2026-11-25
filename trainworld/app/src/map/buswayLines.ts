import { world } from "../game/world";
import { toLngLat } from "./geo";
import type { Overlay } from "./overlay";
import { hidden, levelHidden } from "./visibility";

/** T-3378: the busways (track of kind 4, constructed or planned) as lng, lat lines for the bus router, worked out again
 * when the world's edges or the drawn network change. */
let buswayMemo: { edges: unknown; net: unknown; mask: number; out: { key: string; lines: number[][] } } | null = null;
/** `visibleOnly`: busways wholly on a switched-off level are left out (what a station beside one can snap to) */
export function buswaysOf(overlay: Overlay, visibleOnly = false): { key: string; lines: number[][] } {
  const w = world.peek(), n = overlay.renderer.net;
  if (!w || !n) return { key: "", lines: [] };
  const mask = visibleOnly ? hidden.levels : 0;
  if (buswayMemo && buswayMemo.edges === w.edges && buswayMemo.net === n && buswayMemo.mask === mask) return buswayMemo.out;
  const lines: number[][] = [], keys: string[] = [];
  for (const e of w.edges) {
    if (e.kind !== 4 || (mask && levelHidden(e.levelMin) && levelHidden(e.levelMax))) continue;
    const pts = n.edgePts.get(e.id);
    if (!pts || pts.length < 6) continue;
    const ll: number[] = [];
    for (let k = 0; k + 2 < pts.length; k += 3) ll.push(...toLngLat(pts[k], pts[k + 1]));
    lines.push(ll);
    keys.push(`${e.id}:${e.lengthM}:${ll[0]},${ll[1]}`);
  }
  buswayMemo = { edges: w.edges, net: n, mask, out: { key: keys.join("|"), lines } };
  return buswayMemo.out;
}
