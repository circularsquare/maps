// T-1435, where people drive: the biggest car flows as arcs, ranked by how much a train would win
// (docs/ui-demand-views.md "Where people drive"). Inside a zone the pairs are squares of its grid,
// from the workers' tables (sim api/drives.rs, everyday/drives.rs); zoomed out they are the market's
// corridors between zones. Display only: demand is untouched.

/** The straight-line train a drive is set against (DECISIONS 2026-10-26, T-1435), as the sim's
 * `TRAIN_ACCESS_MIN` and `TRAIN_KMH`: minutes to reach a station, wait and leave the far one, and
 * its speed over the straight line. */
export const TRAIN_ACCESS_MIN = 15, TRAIN_KMH = 100;
export const trainGain = (carMin: number, km: number) => Math.max(0, carMin - TRAIN_ACCESS_MIN - (km * 60) / TRAIN_KMH);

/** The model's drive between two places `km` apart in a straight line (sim kernel `drive_min_f` at
 * its default parameters: road 1.3 times the line, the first 2 km at each end on local streets, the
 * next 8 km on main roads, the rest at 110 km/h), with both ends at the density where street speeds
 * are halfway between town and city (5,000 people and jobs a km²): the page holds no density for a
 * whole zone. */
export function corridorDriveMin(km: number): number {
  const road = km * 1.3, local = 60 / (16.5 + (44 - 16.5) * 0.5), main = 60 / (16.5 + (88 - 16.5) * 0.5);
  const e = Math.min(0.5 * road, 2), mid = road - 2 * e, k = Math.min(0.5 * mid, 8);
  return 2 * e * local + 2 * k * main + ((mid - 2 * k) * 60) / 110;
}

/** A zone's squares (sim `DriveSquares::packed`): their side in zones and zone size, and per
 * square its people's and jobs' centre (`x` longitude, `y` latitude) and its place in the grid. */
export interface DriveSquares { side: number; zoneM: number; n: number; x: Float64Array; y: Float64Array; sx: Int32Array; sy: Int32Array }

/** What the levels are made from: the tables with no rail (`commute` 3 blocks, `everyday` 2, of
 * `n (n - 1) / 2` pairs) and, per pair, the car legs rail takes from each. */
export interface DriveInput { squares: DriveSquares; commute: Float32Array; everyday: Float32Array | null; commuteRail: Float32Array; everydayRail: Float32Array }

/** Floats per arc: its two ends (longitude, latitude), car legs a day, minutes saved by the train. */
export const ARC = 6;
/** One grouping of the squares: `sideM` metres a square, its arcs by car legs times minutes saved,
 * the biggest first (at most `KEEP`). */
export interface DriveLevel { sideM: number; arcs: Float64Array }
/** Squares merged 2 x 2 at each level above the first. */
export const DRIVE_LEVELS = 4;
const KEEP = 2500;

/** The arcs at every level (in the sums worker): each pair's car legs, rail's taken off, and the
 * minutes its car trips would save by train, their mean with no rail. */
export function driveLevels(inp: DriveInput): DriveLevel[] {
  const s = inp.squares, n = s.n, P = (n * (n - 1)) / 2;
  const cb = inp.commute, eb = inp.everyday, cr = inp.commuteRail, er = inp.everydayRail;
  const car = new Float64Array(P), cg = new Float64Array(P);
  for (let k = 0; k < P; k++) {
    const t = cb[k], c0 = cb[P + k];
    let c = c0 > 0 ? Math.max(0, c0 - (t > 0 ? (cr[k] * c0) / t : 0)) : 0;
    let g = c0 > 0 ? c * (cb[2 * P + k] / c0) : 0;
    if (eb) {
      const e0 = eb[k];
      if (e0 > 0) {
        const e = Math.max(0, e0 - er[k]);
        c += e;
        g += e * (eb[P + k] / e0);
      }
    }
    car[k] = c;
    cg[k] = g;
  }
  const out: DriveLevel[] = [];
  for (let l = 0; l < DRIVE_LEVELS; l++) {
    const f = 2 ** l, ids = new Map<string, number>(), grp = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const key = `${Math.floor(s.sx[i] / f)},${Math.floor(s.sy[i] / f)}`;
      let g = ids.get(key);
      if (g === undefined) ids.set(key, (g = ids.size));
      grp[i] = g;
    }
    const m = ids.size, at = new Map<number, number>();
    // per level pair: car, car x gain, and its ends weighed by car legs
    const acc: number[] = [];
    for (let i = 0, k = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++, k++) {
        const c = car[k];
        if (!(c > 0)) continue;
        let a = i, b = j, ga = grp[i], gb = grp[j];
        if (ga === gb) continue;
        if (ga > gb) [a, b, ga, gb] = [b, a, gb, ga];
        const key = ga * m + gb;
        let o = at.get(key);
        if (o === undefined) { at.set(key, (o = acc.length)); acc.push(0, 0, 0, 0, 0, 0); }
        acc[o] += c; acc[o + 1] += cg[k];
        acc[o + 2] += c * s.x[a]; acc[o + 3] += c * s.y[a];
        acc[o + 4] += c * s.x[b]; acc[o + 5] += c * s.y[b];
      }
    }
    const order: number[] = [];
    for (let o = 0; o < acc.length; o += 6) if (acc[o + 1] > 0) order.push(o);
    order.sort((p, q) => acc[q + 1] - acc[p + 1]);
    const keep = order.slice(0, KEEP), arcs = new Float64Array(keep.length * ARC);
    keep.forEach((o, i) => {
      const c = acc[o];
      arcs.set([acc[o + 2] / c, acc[o + 3] / c, acc[o + 4] / c, acc[o + 5] / c, c, acc[o + 1] / c], i * ARC);
    });
    out.push({ sideM: s.side * s.zoneM * f, arcs });
  }
  return out;
}
