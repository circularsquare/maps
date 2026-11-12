// T-3142: the half-filled dot of a stop served one way (T-3041), for the rail, bus and ferry tools.

import type { Overlay } from "./overlay";
import { rightHand } from "../game/world";

/** (the sim plane runs north with y, the map frame south, T-3139) The screen angle (degrees clockwise from screen right) of the plane heading `heading` (radians, as
 * `LineStats.oneWayAt` gives it) at plane point (x, y), as the map is drawn now. */
export function headingAtPlane(ov: Overlay, x: number, y: number, heading: number): number | null {
  const a = ov.toScreen(x, y), b = ov.toScreen(x + 10 * Math.cos(heading), y - 10 * Math.sin(heading));
  const dx = b[0] - a[0], dy = b[1] - a[1];
  return dx || dy ? (Math.atan2(dy, dx) * 180) / Math.PI : null;
}

/** A CSS background filling the half of a disc on the side of the way `heading` (screen degrees) goes that
 * vehicles keep to (`rightHand`, T-3297) in the colour `--c`, the other half white; "" for a stop served both ways. */
export function halfFill(heading: number | null): string {
  if (heading === null) return "";
  return `linear-gradient(${(heading + (rightHand.peek() ? 180 : 0)).toFixed(1)}deg, #fff 50%, var(--c) 50%)`;
}

/** The same as an SVG linearGradient's vector for a disc at (cx, cy): from the white half to the coloured one. */
export function halfFillVector(heading: number): { x1: number; y1: number; x2: number; y2: number } {
  // the coloured half is on the right-hand normal on screen (y down), (-dy, dx), with keep-left the left (T-3297)
  const r = (heading * Math.PI) / 180, k = rightHand.peek() ? 1 : -1, nx = -Math.sin(r) * k, ny = Math.cos(r) * k;
  return { x1: 0.5 - 0.5 * nx, y1: 0.5 - 0.5 * ny, x2: 0.5 + 0.5 * nx, y2: 0.5 + 0.5 * ny };
}
