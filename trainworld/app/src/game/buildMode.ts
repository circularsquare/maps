// T-3693: the mode (regular, high speed, tram, bus) and track count (single, double, quad) the map shows: the
// track tool's aura and pointer mark, and the halo round the ghost and a selected or hovered line or piece, a
// ring a track in the mode's colour (`MODE_HALO_COLOURS`).

import type { MODE_HALO_COLOURS } from "./palette";
import { highSpeed, quadTrack, singleTrack, tramTrack } from "./ui";

export type HaloMode = keyof typeof MODE_HALO_COLOURS;

/** The track tool's mode and count now; `peek` reads without subscribing (from a map frame's effects). */
export const buildMode = (peek = false): { mode: HaloMode; count: 1 | 2 | 4 } => {
  const v = <T,>(s: { value: T; peek(): T }) => (peek ? s.peek() : s.value);
  return { mode: v(tramTrack) ? "tram" : v(highSpeed) ? "hsr" : "rail", count: v(quadTrack) ? 4 : v(singleTrack) ? 1 : 2 };
};
