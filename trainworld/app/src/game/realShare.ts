// T-2145: the train share of trips within a city (the city view's Within bar) that its real network
// gets in this game's own demand, for the cities with a real network save (data/packs/real/<id>.save),
// marked on that bar. Precomputed: each save loaded into a new game with every zone opened, read at
// the demand final (artifacts/trainworld/t2143/realshare-<id>.json, 2026-11-03). Rerun after a
// demand change moves them: notes/T-2145.md.

import type { CityId } from "./cities";
import { AVERAGE_DAY, PLAIN_KIND, type DemandDay } from "./days";

/** % on a weekday, Saturday, Sunday, and the average day */
const REAL_SHARE: Partial<Record<CityId, [number, number, number, number]>> = {
  nyc: [17.1, 16, 15.9, 16.8],
  lon: [21.7, 20, 20, 21.3],
  par: [24.5, 23.8, 23.9, 24.3],
  tyo: [23.1, 21.4, 21.3, 22.7],
  seo: [24.2, 22, 21.7, 23.7],
  tor: [7.7, 7.5, 7.4, 7.6],
  sdj: [9.6, 9.7, 9.8, 9.7],
};

/** The real network's train share (%) in city `id` on demand day `day`, or null where there is none. */
export function realShare(id: string, day: DemandDay): number | null {
  const s = REAL_SHARE[id as CityId];
  if (!s) return null;
  // a holiday's days take the plain day they stand in for
  return day === AVERAGE_DAY ? s[3] : s[Math.min(2, PLAIN_KIND[day])];
}
