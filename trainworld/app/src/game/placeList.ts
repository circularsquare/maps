// Each city's place list from OpenStreetMap (T-177, T-065): towns, villages and neighbourhoods
// inside the pack's footprint, `packs/<id>.places.json` (pipeline/places.py). The search box finds
// them, and a station placed where the basemap has no street names takes the nearest one's name.
// Fetched the first time either needs them, never at startup; a failed fetch is forgotten, so the
// next caller tries again.

import { signal } from "@preact/signals";
import { CITIES, type CityId } from "./cities";
import { normalise } from "./search";

export interface Place {
  name: string;
  /** OSM's place kind: city, town, borough, suburb, quarter, village, neighbourhood, hamlet */
  kind: string;
  lng: number;
  lat: number;
  /** OSM's population, 0 when it gives none */
  pop: number;
  /** the pack whose footprint holds it */
  city: CityId;
  /** normalised name and other names (alt_name, short_name...), for matching */
  keys: string[];
}

interface PlaceFile { format: number; kinds: string[]; places: [string, number, number, number, number, string[]?][] }

const files = new Map<CityId, Promise<Place[]>>();
/** Every place list loaded so far, one array per city; replaced as each arrives. */
export const places = signal<ReadonlyMap<CityId, Place[]>>(new Map());

function load(id: CityId): Promise<Place[]> {
  let p = files.get(id);
  if (!p) {
    p = fetch(new URL(`packs/${id}.places.json`, document.baseURI)).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      const f = (await r.json()) as PlaceFile;
      const list = f.places.map(([name, k, lng, lat, pop, others]): Place => ({
        name, kind: f.kinds[k] ?? "neighbourhood", lng, lat, pop, city: id,
        keys: [normalise(name), ...(others ?? []).map(normalise)],
      }));
      places.value = new Map([...places.peek(), [id, list]]);
      return list;
    });
    files.set(id, p);
    p.catch(() => { if (files.get(id) === p) files.delete(id); });
  }
  return p;
}

/** Fetch every registered city's list (each once). Resolves when all have arrived or failed. */
export function loadPlaces(): Promise<void> {
  return Promise.all(CITIES.map((c) => load(c.id).catch(() => []))).then(() => undefined);
}

/** Kinds a station may be named after, nearest first within this many metres. */
const NAMING_M: Record<string, number> = {
  neighbourhood: 2500, quarter: 3000, suburb: 4000, borough: 4000, hamlet: 3000, village: 4000, town: 5000, city: 6000,
};

/** Places near a point that a station there could be named after, nearest first (loads the lists
 * first). Distance is measured to the place's label point. */
export async function placesNear(lng: number, lat: number): Promise<{ place: Place; d: number; dx: number; dy: number }[]> {
  await loadPlaces();
  const kx = 111320 * Math.cos((lat * Math.PI) / 180), ky = 110540;
  const out: { place: Place; d: number; dx: number; dy: number }[] = [];
  for (const list of places.peek().values())
    for (const p of list) {
      const dx = (lng - p.lng) * kx, dy = (lat - p.lat) * ky;
      if (Math.abs(dx) > 6000 || Math.abs(dy) > 6000) continue;
      const d = Math.hypot(dx, dy);
      if (d <= (NAMING_M[p.kind] ?? 2500)) out.push({ place: p, d, dx, dy });
    }
  return out.sort((a, b) => a.d - b.d);
}
