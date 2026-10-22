// T-800: every registered zone's coarse outline, `packs/outlines.json` (pipeline/zone_outlines.py),
// so the map has an outline everywhere before any zone's own boundary has been fetched.
import type { MultiPolygon } from "geojson";
import { dataUrl } from "../game/dataBase";

type File = { q: number; zones: Record<string, number[][]> };

/** The coarse footprint of each zone in the file, by zone id; empty when the file is unavailable. */
export async function coarseOutlines(): Promise<Map<string, MultiPolygon>> {
  const out = new Map<string, MultiPolygon>();
  try {
    const r = await fetch(dataUrl("outlines.json"));
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const { q, zones } = (await r.json()) as File;
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
  } catch (error) {
    console.warn("Coarse zone outlines unavailable", error);
  }
  return out;
}
