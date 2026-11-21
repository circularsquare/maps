// T-3913: where a bus is in town for BRT (the sim's `streets::brt_in_town`: a plain bus there moves under BRT's floor),
// as polygons per pack, sent by the clock worker as each pack's street speeds land (`TrackApi.last_town_outline`). The
// bus tool tints the ground by them. Out of town wherever no pack's speeds are in, as in the sim.

import { signal } from "@preact/signals";

const polygons = new Map<string, number[][][][]>();
/** bumped whenever the polygons change */
export const townVersion = signal(0);

export function addTownOutline(id: string, o: Float64Array) {
  const list: number[][][][] = [];
  let at = 1;
  for (let p = 0; p < o[0]; p++) {
    const rings: number[][][] = [];
    const n = o[at++];
    for (let r = 0; r < n; r++) {
      const k = o[at++];
      const ring: number[][] = [];
      for (let i = 0; i < k; i++, at += 2) ring.push([o[at], o[at + 1]]);
      rings.push(ring);
    }
    list.push(rings);
  }
  polygons.set(id, list);
  townVersion.value++;
}

export function clearTownOutlines() {
  if (!polygons.size) return;
  polygons.clear();
  townVersion.value++;
}

/** Every pack's polygons as one MultiPolygon, or null with none. */
export function townGeometry(): { type: "MultiPolygon"; coordinates: number[][][][] } | null {
  const coordinates = [...polygons.values()].flat();
  return coordinates.length ? { type: "MultiPolygon", coordinates } : null;
}
