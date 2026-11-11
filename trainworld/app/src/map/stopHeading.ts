// T-3111: which way the trains of a one-way stop head, on the map's own frame.

import type { Overlay } from "./overlay";

type Net = NonNullable<Overlay["renderer"]["net"]>;

/** The screen angle (degrees clockwise from screen right, y down) from station `from` to station `to`
 * as the map draws them now, so it turns with a rotated map; null when either is not in the network. */
export function headingOnScreen(ov: Overlay, net: Net, from: string, to: string, index?: ReadonlyMap<string, number>): number | null {
  const at = (s: string): [number, number] | null => {
    const i = index ? index.get(s) : net.stationIds.indexOf(s);
    return i === undefined || i < 0 ? null : ov.toScreen(net.stations[i * 3], net.stations[i * 3 + 1]);
  };
  const a = at(from), b = at(to);
  if (!a || !b) return null;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  return dx || dy ? (Math.atan2(dy, dx) * 180) / Math.PI : null;
}
