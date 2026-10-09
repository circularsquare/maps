// T-005's hand-made New York network as a NetworkSnapshot, for `?demandTest=1`: the demand pool
// solves on it instead of the clock worker's network, so the mode split and the timings can be
// checked before (or without) building anything, and compared with notes/T-005.md and T-020.md.
// A port of `synth_network` in sim/src/demand/network.rs (25 lines, 434 stations): keep the two
// in step if either changes.

import type { NetworkSnapshot } from "./protocol";

type Kind = "subway" | "light" | "commuter" | "orbital";
const KIND: Record<Kind, { speed: number; tph: [number, number, number]; cars: number; spacing: (r: number) => number }> = {
  // trains an hour from network.rs's headways (morning peak, midday, night)
  subway: { speed: 35, tph: [15, 7.5, 3], cars: 10, spacing: (r) => (r < 8 ? 0.9 : 1.4) },
  light: { speed: 25, tph: [60 / 7, 5, 2], cars: 3, spacing: () => 1.1 },
  commuter: { speed: 55, tph: [5, 2, 1], cars: 11, spacing: () => 5.0 },
  orbital: { speed: 35, tph: [10, 6, 3], cars: 7, spacing: () => 1.8 },
};

/** Waypoints in km from Times Square (the pack origin), from real places. */
const LINES: [string, Kind, [number, number][]][] = [
  ["Broadway-7 Av", "subway", [[6.3, 13.6], [4.6, 8.5], [2.2, 3.0], [0.0, 0.0], [-2.0, -5.8], [-0.4, -7.3], [3.0, -10.0], [6.5, -11.5]]],
  ["Lexington", "subway", [[12.0, 16.0], [8.0, 11.4], [4.0, 5.1], [0.7, -0.7], [-1.6, -5.5], [-0.4, -7.3], [4.0, -8.5], [8.0, -10.0]]],
  ["8 Av", "subway", [[5.5, 11.0], [3.5, 6.0], [-0.3, -0.5], [-2.3, -4.5], [-0.6, -7.0], [5.0, -8.0], [11.0, -9.5], [19.4, -17.0]]],
  ["6 Av-Queens Blvd", "subway", [[14.8, -6.4], [9.0, -2.5], [3.3, -1.2], [0.3, -0.3], [-1.2, -4.0], [-0.9, -8.0], [-0.5, -13.0], [0.3, -20.1]]],
  ["Broadway BMT", "subway", [[5.5, 1.8], [3.3, -1.2], [0.1, 0.1], [-1.8, -5.0], [-1.2, -7.5], [-3.2, -13.8]]],
  ["Flushing", "subway", [[13.1, 0.1], [6.5, -0.5], [3.3, -1.2], [0.6, -0.6], [-0.6, -0.2]]],
  ["Second Av-Bronx", "subway", [[11.0, 16.0], [9.0, 9.0], [3.8, 4.2], [1.5, -1.5], [-1.0, -5.0]]],
  ["Canarsie", "subway", [[-1.8, -2.6], [1.2, -2.4], [3.5, -6.0], [6.0, -8.5], [9.0, -11.0]]],
  ["Brooklyn-Queens", "subway", [[-0.4, -7.3], [2.0, -4.5], [3.3, -1.2], [5.2, 2.4]]],
  ["PATH Newark", "subway", [[-15.1, -2.7], [-10.5, -3.0], [-6.6, -2.8], [-3.8, -2.6], [-2.2, -5.6]]],
  ["Hudson light rail", "light", [[-12.0, -11.0], [-8.5, -7.0], [-6.6, -3.5], [-4.5, 0.5], [-4.0, 6.0]]],
  ["Staten Island", "light", [[-7.5, -12.8], [-13.0, -18.0], [-22.3, -27.4]]],
  ["Hudson", "commuter", [[0.7, -0.7], [4.0, 5.1], [6.3, 13.6], [7.2, 19.8], [8.9, 48.0], [9.0, 70.0]]],
  ["Harlem", "commuter", [[0.7, -0.7], [4.0, 5.1], [9.5, 14.0], [17.7, 30.6], [30.8, 70.0]]],
  ["New Haven", "commuter", [[0.7, -0.7], [4.0, 5.1], [11.0, 13.0], [17.1, 17.0], [37.3, 32.1], [60.0, 42.0], [89.0, 60.0]]],
  ["LIRR Main", "commuter", [[-0.7, -0.9], [3.3, -1.2], [8.0, -3.5], [14.8, -6.4], [29.0, -1.9], [38.5, 1.0], [74.2, 5.6]]],
  ["LIRR Babylon", "commuter", [[-0.7, -0.9], [14.8, -6.4], [25.0, -9.0], [40.0, -8.5], [55.6, -6.4], [75.0, -2.0]]],
  ["LIRR Port Washington", "commuter", [[-0.7, -0.9], [3.3, -1.2], [13.1, 0.1], [25.1, 7.9]]],
  ["NJ Northeast Corridor", "commuter", [[-0.7, -0.9], [-7.6, 0.3], [-15.1, -2.7], [-19.4, -10.1], [-38.8, -29.0], [-64.8, -60.0]]],
  ["NJ Coast", "commuter", [[-0.7, -0.9], [-7.6, 0.3], [-15.1, -2.7], [-19.4, -10.1], [-24.0, -26.0], [-10.0, -40.0], [-0.4, -51.0]]],
  ["NJ Morris", "commuter", [[-0.7, -0.9], [-7.6, 0.3], [-15.1, -2.7], [-25.0, 1.0], [-41.7, 4.3], [-48.4, 13.6]]],
  ["NJ Bergen", "commuter", [[-0.7, -0.9], [-7.6, 0.3], [-10.0, 10.0], [-15.6, 17.7], [-13.9, 39.0]]],
  ["Triboro", "orbital", [[-3.2, -13.8], [2.0, -12.0], [6.0, -8.5], [10.0, -4.0], [9.5, 2.0], [8.0, 11.4], [6.3, 13.6]]],
  ["Outer ring", "orbital", [[-19.4, -10.1], [-15.1, -2.7], [-12.0, 8.0], [-4.9, 14.2], [7.2, 19.8], [17.1, 17.0], [25.1, 7.9], [29.0, -1.9], [27.0, -12.0], [17.3, -12.5]]],
  ["Bronx-Queens", "orbital", [[0.0, 13.0], [8.0, 11.4], [13.0, 6.0], [13.1, 0.1], [14.8, -6.4], [17.3, -12.5]]],
];

const MERGE_KM = 0.35;

export function demandTestSnapshot(version = 1): NetworkSnapshot {
  const st: { x: number; y: number }[] = [];
  const place = (x: number, y: number, stops: number[]) => {
    let s = st.findIndex((p) => Math.hypot(p.x / 1000 - x, p.y / 1000 - y) < MERGE_KM);
    if (s < 0) {
      // network.rs keeps station positions in f32 metres
      st.push({ x: Math.fround(x * 1000), y: Math.fround(y * 1000) });
      s = st.length - 1;
    }
    if (!stops.includes(s)) stops.push(s);
  };
  const lines: NetworkSnapshot["lines"] = LINES.map(([name, kind, pts], id) => {
    const k = KIND[kind];
    const stops: number[] = [];
    place(pts[0][0], pts[0][1], stops);
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const len = Math.hypot(x1 - x0, y1 - y0);
      const midR = Math.hypot((x0 + x1) / 2, (y0 + y1) / 2);
      const n = Math.max(1, Math.round(len / k.spacing(midR)));
      for (let j = 1; j < n; j++) {
        const x = x0 + ((x1 - x0) * j) / n, y = y0 + ((y1 - y0) * j) / n;
        if (kind === "commuter" && Math.hypot(x, y) < 9) continue; // express through the inner area
        place(x, y, stops);
      }
      place(x1, y1, stops);
    }
    // run time per hop from straight-line distance x 1.15 at the line's average speed (dwell included)
    const hops = stops.slice(1).map((s, i) => (Math.hypot(st[s].x - st[stops[i]].x, st[s].y - st[stops[i]].y) * 1.15) / 1000 / k.speed * 3600);
    const back = hops.slice().reverse();
    const zero = stops.map(() => 0);
    return {
      id, name, stops, tph: k.tph,
      hopS: [[hops, hops, hops], [back, back, back]],
      dwellS: [[zero, zero, zero], [zero, zero, zero]],
      turnaroundS: 180, cars: k.cars,
    };
  });
  return {
    version,
    city: "nyc",
    stations: st.map((p, id) => ({ id, name: `Test ${id}`, x: p.x, y: p.y, level: 0, platform: 400 })),
    lines,
  };
}
