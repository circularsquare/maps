// Which side of its platform a station's annexes go (T-1416): the waiting room and the park and
// ride tile are drawn joined to the box on one side (stationGeometry.ts `stationAnnexes`). Here,
// in the clock worker, each station with one picks the side with fewer other platforms and tracks
// where the annexes would be, so in a dense complex they stay off the neighbours; on a tie, the
// outside of the platform's bend. The pick sets ST_FLIP in the station's flags.

import { PLATFORM_POINTS, ST_FLIP, ST_PARK, ST_ROOM } from "./stationGeometry";

/** Where the annexes reach, m from the platform's centre line and along it from its middle: about
 * the drawn size between zoom 14.5 and 16, where a box is 16-30 m wide on the ground. */
const ACROSS_M = [14, 24, 34], ALONG_M = [-28, -10, 10, 28];
/** Too close: another platform's centre line, or another track */
const NEAR_STATION_M = 14, NEAR_TRACK_M = 5;
const CELL_M = 40;

export function annexSides(
  st: { id: number; lat: number }[],
  stations: Float32Array,
  stationLine: Float32Array,
  stationShape: Float32Array,
  edgePts: Map<number, Float32Array>,
  waysAt: Map<number, number[]>,
) {
  const want: number[] = [];
  for (let i = 0; i < st.length; i++) if (stations[i * 3 + 2] & (ST_ROOM | ST_PARK)) want.push(i);
  if (!want.length) return;
  const perM = 1 / Math.cos(st[want[0]].lat * Math.PI / 180), cell = CELL_M * perM;
  // segments by grid cell: x0, y0, x1, y1, then the owner (edge id, or -1 - station index)
  const seg: number[] = [], grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => cx * 1_000_003 + cy;
  const add = (x0: number, y0: number, x1: number, y1: number, owner: number) => {
    const k = seg.length / 5;
    seg.push(x0, y0, x1, y1, owner);
    for (let cx = Math.floor(Math.min(x0, x1) / cell); cx <= Math.floor(Math.max(x0, x1) / cell); cx++)
      for (let cy = Math.floor(Math.min(y0, y1) / cell); cy <= Math.floor(Math.max(y0, y1) / cell); cy++) {
        const g = grid.get(key(cx, cy));
        if (g) g.push(k); else grid.set(key(cx, cy), [k]);
      }
  };
  // only what lies near a station that wants annexes
  const reach = (ACROSS_M[2] + 40) * perM, near = new Set<number>();
  for (const i of want) {
    const cx = Math.floor(stations[i * 3] / cell), cy = Math.floor(stations[i * 3 + 1] / cell), r = Math.ceil(reach / cell) + 4;
    for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) near.add(key(cx + dx, cy + dy));
  }
  const isNear = (x: number, y: number) => near.has(key(Math.floor(x / cell), Math.floor(y / cell)));
  for (const [id, p] of edgePts)
    for (let k = 0; k + 5 < p.length; k += 3) if (isNear(p[k], p[k + 1])) add(p[k], p[k + 1], p[k + 3], p[k + 4], id);
  for (let i = 0; i < st.length; i++) {
    const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS;
    if (!isNear(x, y)) continue;
    for (let j = 0; j + 1 < PLATFORM_POINTS; j++) add(x + stationLine[o + j * 2], y + stationLine[o + j * 2 + 1], x + stationLine[o + j * 2 + 2], y + stationLine[o + j * 2 + 3], -1 - i);
  }
  const m = PLATFORM_POINTS >> 1;
  for (const i of want) {
    const own = new Set(waysAt.get(st[i].id) ?? []);
    const x = stations[i * 3], y = stations[i * 3 + 1], o = i * 2 * PLATFORM_POINTS;
    const mx = x + stationLine[o + m * 2], my = y + stationLine[o + m * 2 + 1];
    let dx = stationLine[o + m * 2 + 2] - stationLine[o + m * 2 - 2], dy = stationLine[o + m * 2 + 3] - stationLine[o + m * 2 - 1];
    const dl = Math.hypot(dx, dy) || 1;
    dx /= dl; dy /= dl;
    const score = (sg: number) => {
      let bad = 0;
      for (const a of ALONG_M) for (const c of ACROSS_M) {
        const px = mx + (dx * a - dy * sg * c) * perM, py = my + (dy * a + dx * sg * c) * perM;
        let worst = 0;
        const cx = Math.floor(px / cell), cy = Math.floor(py / cell), ks: number[] = [];
        for (let u = -1; u <= 1; u++) for (let v = -1; v <= 1; v++) ks.push(...(grid.get(key(cx + u, cy + v)) ?? []));
        for (const k of ks) {
          const owner = seg[k * 5 + 4];
          if (owner === -1 - i || own.has(owner)) continue;
          const d = distSeg(px, py, seg[k * 5], seg[k * 5 + 1], seg[k * 5 + 2], seg[k * 5 + 3]) / perM;
          if (owner < 0 && d < NEAR_STATION_M) worst = 2;
          else if (owner >= 0 && d < NEAR_TRACK_M) worst = Math.max(worst, 1);
        }
        bad += worst;
      }
      return bad;
    };
    // + is to the left of the platform's way; it bends towards + when phi > 0, so its outside is -
    // (a straight one's too, T-1287's side)
    const outside = stationShape[i * 4 + 2] < 0 ? 1 : -1;
    const a = score(outside), b = score(-outside);
    const side = b < a ? -outside : outside;
    if (side > 0) stations[i * 3 + 2] += ST_FLIP;
  }
}

function distSeg(px: number, py: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
  return Math.hypot(x0 + t * dx - px, y0 + t * dy - py);
}
