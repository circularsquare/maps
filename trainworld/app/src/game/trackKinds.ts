// The kinds of track and what differs between them: a display copy of sim/src/track/params.rs
// (`TrackKind`, `TRACK_SPECS`; T-2921 section 5), which owns the real table.

export type TrackKind = "regular" | "highSpeed" | "streetTram" | "tram";

export interface TrackSpec {
  /** the number a save stores in edge flags bits 8-10 (0 and 1 are also written as bit 6) */
  id: number;
  /** nodes join only within a family: 0 rail, 1 tram */
  family: number;
  vCapKmh: number;
  minRadius: number;
  /** price multiplier per level, index level + 3, in the units of the sim's `LEVEL_MULT` */
  levelMult: readonly number[];
  /** what terrain (hills, mountains) and what water add to the level's multiplier are scaled by (high
   * speed's are shown as a factor of their own instead; tram terrain T-3004) */
  terrain: number;
  water: number;
  /** upkeep over ground rail's */
  upkeep: number;
  roadCrossings: boolean;
}

const LEVEL_MULT = [2.0, 1.6, 1.0, 0.5, 1.0, 1.5, 2.0] as const;
const HSR_COST = 1.5;

export const TRACK_SPECS: Record<TrackKind, TrackSpec> = {
  regular: { id: 0, family: 0, vCapKmh: 160, minRadius: 25, levelMult: LEVEL_MULT, terrain: 1, water: 1, upkeep: 1, roadCrossings: true },
  highSpeed: { id: 1, family: 0, vCapKmh: 320, minRadius: 50, levelMult: LEVEL_MULT.map(m => m * HSR_COST), terrain: 1, water: 1, upkeep: HSR_COST, roadCrossings: true },
  streetTram: { id: 2, family: 1, vCapKmh: 60, minRadius: 18, levelMult: [1.6, 1.2, 0.6, 0.1, 0.6, 1.1, 1.6], terrain: 0.25, water: 0.5, upkeep: 0.6, roadCrossings: false },
  tram: { id: 3, family: 1, vCapKmh: 80, minRadius: 18, levelMult: [1.6, 1.2, 0.6, 0.5, 0.6, 1.1, 1.6], terrain: 0.25, water: 1, upkeep: 1, roadCrossings: false },
};

/** a tram stop's upkeep a day, real US$ (sim `TrackSpec::station_upkeep`) */
export const TRAM_STOP_UPKEEP = 50;

/** Level multiplier of a kind at a level (-3 to +3). */
export const kindLevelMult = (k: TrackKind, level: number) => TRACK_SPECS[k].levelMult[level + 3];
