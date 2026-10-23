// T-800: every registered zone's coarse outline, `packs/outlines.json` (pipeline/zone_outlines.py),
// so the map has an outline everywhere before any zone's own boundary has been fetched. Read once:
// the outlines and the realms' shade (T-1006) share it.
import type { MultiPolygon } from "geojson";
import { dataUrl } from "../game/dataBase";

export type OutlineFile = { q: number; zones: Record<string, number[][]> };

let read: Promise<Map<string, MultiPolygon>> | null = null;
let text: string | null = null;

/** The coarse footprint of each zone in the file, by zone id; empty when the file is unavailable. */
export function coarseOutlines(): Promise<Map<string, MultiPolygon>> {
  return (read ??= load());
}

/** The file as fetched, once (the shade's worker reads its own copy, T-1026); null after that, or
 * before the file is in. */
export function takeOutlineText(): string | null {
  const t = text;
  text = null;
  return t;
}

/** One ring of the file as lon, lat pairs, flat and open. */
export function decodeRing(flat: number[], q: number): Float64Array {
  const out = new Float64Array(flat.length);
  let x = 0, y = 0;
  for (let i = 0; i < flat.length; i += 2) { out[i] = (x += flat[i]) / q; out[i + 1] = (y += flat[i + 1]) / q; }
  return out;
}

async function load(): Promise<Map<string, MultiPolygon>> {
  const out = new Map<string, MultiPolygon>();
  try {
    const r = await fetch(dataUrl("outlines.json"));
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const body = await r.text();
    const { q, zones } = JSON.parse(body) as OutlineFile;
    for (const id in zones) {
      out.set(id, {
        type: "MultiPolygon",
        coordinates: zones[id].map((flat) => {
          let x = flat[0], y = flat[1];
          const ring = [[x / q, y / q]];
          for (let i = 2; i < flat.length; i += 2) ring.push([(x += flat[i]) / q, (y += flat[i + 1]) / q]);
          ring.push([...ring[0]]);
          return [ring];
        }),
      });
    }
    text = body;
  } catch (error) {
    console.warn("Coarse zone outlines unavailable", error);
  }
  return out;
}
