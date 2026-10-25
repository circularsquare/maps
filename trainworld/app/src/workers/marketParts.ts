// The regional market's `part_summary` on another kind of day (T-260, T-1229). Plain data and
// arithmetic, so a Node check can import it (checks/t1177-cross-everyday.mjs).

import { EVERYDAY_KEYS, type PurposeKey } from "../game/purposes";

/** The numbers `IntercityApi.part_summary` gives: trips and rail of the longer trips, legs and rail
 * legs of the commutes between zones, legs, rail legs, walking legs and walking legs with no rail
 * of the everyday outings across a zone line (T-1175), then those four for each everyday purpose
 * (T-1229), `EVERYDAY_KEYS` order. */
export const PARTS_LEN = 8 + 4 * EVERYDAY_KEYS.length;

/**
 * One period's `part_summary` as it is on a kind of day: the longer trips by their factor `fl`, the
 * commutes by work's `fw`, and each purpose's everyday outings by that purpose's own (`factorOf`);
 * the three-purpose-wide figures (4 to 7) are then the purposes' sums. Parts without the per-purpose
 * numbers (older answers) keep the everyday figures as they are.
 */
export function marketPartsOn(parts: number[], fl: number, fw: number, factorOf: (key: PurposeKey) => number): number[] {
  const out = parts.map((v, i) => v * (i < 2 ? fl : i < 4 ? fw : 1));
  if (parts.length >= PARTS_LEN) {
    EVERYDAY_KEYS.forEach((key, k) => {
      const f = factorOf(key);
      for (let j = 0; j < 4; j++) out[8 + 4 * k + j] *= f;
    });
    for (let j = 0; j < 4; j++) out[4 + j] = EVERYDAY_KEYS.reduce((s, _, k) => s + out[8 + 4 * k + j], 0);
  }
  return out;
}
