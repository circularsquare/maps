// T-3111: which way the trains of a one-way stop head, on the map's own frame.

import type { Overlay } from "./overlay";

type Net = NonNullable<Overlay["renderer"]["net"]>;

/** The screen angle (degrees clockwise from screen right, y down) of the world heading `heading` (radians
 * counter-clockwise from east, `LineStats.oneWayAt`, T-3122) at station `id` as the map draws it now, so it turns
 * with a rotated map; null when the station is not in the network. */
export function headingOnScreen(ov: Overlay, net: Net, id: string, heading: number, index?: ReadonlyMap<string, number>): number | null {
  const i = index ? index.get(id) : net.stationIds.indexOf(id);
  if (i === undefined || i < 0) return null;
  const x = net.stations[i * 3], y = net.stations[i * 3 + 1];
  // the network's local units run south with y (mercator), the sim's world frame north (T-3139)
  const a = ov.toScreen(x, y), b = ov.toScreen(x + 10 * Math.cos(heading), y - 10 * Math.sin(heading));
  const dx = b[0] - a[0], dy = b[1] - a[1];
  return dx || dy ? (Math.atan2(dy, dx) * 180) / Math.PI : null;
}
