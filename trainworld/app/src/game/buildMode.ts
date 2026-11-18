// T-3693, T-3714: the track type (regular, high speed, tram) and count (single, double, quad) being built: the
// track tool's aura and its placing tag's track count icon take the type's colour (`TRACK_TYPE_COLOURS`).

import type { TRACK_TYPE_COLOURS } from "./palette";
import { highSpeed, quadTrack, singleTrack, tramTrack } from "./ui";

export type BuildKind = keyof typeof TRACK_TYPE_COLOURS;

/** The track tool's mode and count now; `peek` reads without subscribing (from a map frame's effects). */
export const buildMode = (peek = false): { mode: BuildKind; count: 1 | 2 | 4 } => {
  const v = <T,>(s: { value: T; peek(): T }) => (peek ? s.peek() : s.value);
  return { mode: v(tramTrack) ? "tram" : v(highSpeed) ? "hsr" : "rail", count: v(quadTrack) ? 4 : v(singleTrack) ? 1 : 2 };
};
