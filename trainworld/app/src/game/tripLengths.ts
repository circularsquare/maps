// T-2365: a zone's trips a day by straight-line length and mode, for the city view's trip-length
// graph: the demand workers' bins (sim `lengths.rs`) added up over their periods.

import type { DemandDay } from "./days";

/** One value per bin, in mode order. */
export interface ModeBins { trips: Float64Array; rail: Float64Array; walk: Float64Array; none: Float64Array; drive: Float64Array }

export interface TripLengths {
  version: number;
  day: DemandDay;
  /** upper edges of the bins, km (the last bin has none) */
  edges: Float32Array;
  /** trips a day per bin, all purposes of the zone's local trips (its split's), commutes and outings */
  n: ModeBins;
  /** the same trips' straight-line km a day per bin */
  km: ModeBins;
}

/** The bins from eight blocks (trips, rail, walking, no way, then their km) added up; driving is
 * the trips less the other three, as the split has it. */
export function tripLengthsOf(version: number, day: DemandDay, edges: Float32Array, sum: Float64Array): TripLengths {
  const nb = edges.length + 1;
  const modes = (o: number): ModeBins => {
    const b = (k: number) => sum.slice((o + k) * nb, (o + k + 1) * nb);
    const trips = b(0), rail = b(1), walk = b(2), none = b(3);
    const drive = trips.map((t, i) => t - rail[i] - walk[i] - none[i]);
    return { trips, rail, walk, none, drive };
  };
  return { version, day, edges, n: modes(0), km: modes(4) };
}
