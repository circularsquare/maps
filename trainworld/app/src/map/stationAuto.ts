// T-4105: the station tool's Auto choice: the kind of station the pointer is over, found before the click so the ghost and
// tag show it. Rule, first match wins: track under the pointer (train, or tram only where no train track is within
// OVERLAP_PX of the nearest), a road a bus may use or a bus route, a ferry route, water or a shore a port could take;
// open ground is a train station.

import { autoKind, stationChoice, tool, type StationKind } from "../game/ui";
import { freeStarted } from "./stationFree";
import { OVERLAP_PX } from "./toolParts";
import { roadUnder } from "./stationTool";
import type { Tools } from "./tools";

type At = { px: number; py: number; lng: number; lat: number };

/** the last answer to whether the pointer is on water or a shore a port could take, and whether one is being asked */
let wet: { lng: number; lat: number; on: boolean } | null = null;
let asking = false;

function wetAt(tk: Tools, p: At): boolean | undefined {
  if (wet && wet.lng === p.lng && wet.lat === p.lat) return wet.on;
  if (!asking) {
    asking = true;
    void tk.ferry.water.probe([p.lng, p.lat]).then((r) => {
      asking = false;
      if (r) wet = { lng: p.lng, lat: p.lat, on: r.water || r.shore };
      if (tool.peek() !== "station") return;
      tk.autoPick();
      tk.place();
    });
  }
  return undefined;
}

function kindAt(tk: Tools, p: At): StationKind | undefined {
  tk.autoProbe = true;
  let s;
  try {
    s = tk.snap(p.px, p.py, { lng: p.lng, lat: p.lat });
  } finally {
    tk.autoProbe = false;
  }
  if (s.kind !== "free") {
    const spots = tk.alts.length ? tk.alts : [s];
    const d = spots.map((a) => { const [x, y] = tk.screen(a.lx, a.ly); return Math.hypot(x - p.px, y - p.py); });
    const near = Math.min(...d);
    return spots.some((a, i) => d[i] <= near + OVERLAP_PX && !tk.tramSpot(a)) ? "rail" : "tram";
  }
  if (roadUnder(tk, p.lng, p.lat)) return "bus";
  if (roadUnder(tk, p.lng, p.lat, true)) return "port";
  const on = wetAt(tk, p);
  return on === undefined ? undefined : on ? "port" : "rail";
}

/** With Auto picked, set the kind the pointer (or `at`) is over; it stays while the first platform end of a free station is down. */
export function autoPick(tk: Tools, at?: At) {  if (tool.peek() !== "station" || stationChoice.peek() !== "auto" || freeStarted()) return;
  const p = at ?? tk.pointer;
  if (!p) return;
  const k = kindAt(tk, p);
  if (k && k !== autoKind.peek()) autoKind.value = k;
}
