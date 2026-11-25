import { world } from "../game/world";
import { toLngLat } from "./geo";
import type { Overlay } from "./overlay";

/** T-3378: the busways (track of kind 4, constructed or planned) as lng, lat lines for the bus router, worked out again
 * when the world's edges or the drawn network change. */
let buswayMemo: { edges: unknown; net: unknown; out: { key: string; lines: number[][] } } | null = null;
export function buswaysOf(overlay: Overlay): { key: string; lines: number[][] } {
  const w = world.peek(), n = overlay.renderer.net;
  if (!w || !n) return { key: "", lines: [] };
  if (buswayMemo && buswayMemo.edges === w.edges && buswayMemo.net === n) return buswayMemo.out;
  const lines: number[][] = [], keys: string[] = [];
  for (const e of w.edges) {
    if (e.kind !== 4) continue;
    const pts = n.edgePts.get(e.id);
    if (!pts || pts.length < 6) continue;
    const ll: number[] = [];
    for (let k = 0; k + 2 < pts.length; k += 3) ll.push(...toLngLat(pts[k], pts[k + 1]));
    lines.push(ll);
    keys.push(`${e.id}:${e.lengthM}:${ll[0]},${ll[1]}`);
  }
  buswayMemo = { edges: w.edges, net: n, out: { key: keys.join("|"), lines } };
  return buswayMemo.out;
}
