// Each registered city's exact footprint, `packs/<id>.boundary.geojson`, fetched once per page
// (T-172): the demand pool's reach test (workers/cityReach.ts), the map's outlines and the focused
// city's shade (map/cityBoundaries.ts) all read the same files, which were fetched two or three
// times each at startup. A failed fetch is forgotten, so the next caller tries again.

import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import { dataUrl } from "./dataBase";

export type Boundary = FeatureCollection<Polygon | MultiPolygon>;

const cache = new Map<string, Promise<Boundary>>();

export function cityBoundary(id: string): Promise<Boundary> {
  let p = cache.get(id);
  if (!p) {
    p = fetch(dataUrl(`${id}.boundary.geojson`)).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      return (await r.json()) as Boundary;
    });
    cache.set(id, p);
    p.catch(() => { if (cache.get(id) === p) cache.delete(id); });
  }
  return p;
}
