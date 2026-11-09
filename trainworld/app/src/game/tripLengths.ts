// T-2365: a zone's trips a day by straight-line length and mode, for the city view's trip-length
// graph: the demand workers' bins (sim `lengths.rs`) added up over their periods, and the trips to
// other places binned the same way by the distance between the two places.

import type { DemandDay } from "./days";
import { EVERYDAY_KEYS, PURPOSES, shownKey, type PurposeKey } from "./purposes";

/** One value per bin, in mode order. */
export interface ModeBins { trips: Float64Array; rail: Float64Array; walk: Float64Array; none: Float64Array; drive: Float64Array }

export interface TripLengths {
  version: number;
  day: DemandDay;
  /** upper edges of the bins, km (the last bin has none) */
  edges: Float32Array;
  /** trips a day per bin, all purposes of the zone's local trips (its split's), commutes and outings */
  n: ModeBins;
  /** the same trips' straight-line km a day per bin */
  km: ModeBins;
  /** T-2790: the same trips per bin by purpose as shown (work the commutes, university with
   * school), which add up to `n.trips` bin by bin; a purpose with no trips may be missing */
  purposes: Partial<Record<PurposeKey, Float64Array>>;
}

/** The bins from eight blocks (trips, rail, walking, no way, then their km) added up; driving is
 * the trips less the other three, as the split has it. `work`: the commutes' trips per bin; `kinds`:
 * one block of bins per everyday purpose, in `EVERYDAY_KEYS`' order. */
export function tripLengthsOf(version: number, day: DemandDay, edges: Float32Array, sum: Float64Array, work: Float64Array, kinds: Float64Array): TripLengths {
  const nb = edges.length + 1;
  const modes = (o: number): ModeBins => {
    const b = (k: number) => sum.slice((o + k) * nb, (o + k + 1) * nb);
    const trips = b(0), rail = b(1), walk = b(2), none = b(3);
    const drive = trips.map((t, i) => t - rail[i] - walk[i] - none[i]);
    return { trips, rail, walk, none, drive };
  };
  const purposes: TripLengths["purposes"] = { work };
  EVERYDAY_KEYS.forEach((key, k) => {
    if ((k + 1) * nb > kinds.length) return;
    const to = (purposes[shownKey(key)] ??= new Float64Array(nb));
    for (let b = 0; b < nb; b++) to[b] += kinds[k * nb + b];
  });
  return { version, day, edges, n: modes(0), km: modes(4), purposes };
}

// ---------------------------------------------------------------- the graph's layers

/** The graph's layers, bottom to top. */
export const LAYERS = ["train", "walk", "drive", "none", "plane"] as const;
export type Layer = (typeof LAYERS)[number];

/** Bin edges past the kernel's last (300 km), for the trips to other places. */
export const FAR_EDGES_KM = [400, 500, 700, 1000, 1500, 2000, 3000, 5000];
/** The first bin's lower end and the last's upper end on the log axis, km (the bins themselves
 * start at 0 and have no end). */
const FIRST_LO_KM = 0.125, LAST_HI_KM = 10000;

/** T-2790: the graph's layers by purpose, bottom to top: the purposes as the filter orders them,
 * inter-city trips last. */
export const PURPOSE_LAYERS: readonly PurposeKey[] = PURPOSES.map((p) => p.key);

/** A zone's trips a day by length, within the zone and to other places together, per layer: by
 * mode (`LAYERS`) or by purpose (`PURPOSE_LAYERS`). */
export interface LengthLayers<K extends string = Layer> {
  /** the layers, bottom to top */
  keys: readonly K[];
  /** upper edges of the bins, km: the kernel's, then `FAR_EDGES_KM` */
  edges: number[];
  /** trips a day per bin, one array per layer */
  n: Record<K, Float64Array>;
  /** of each bin's trips, those to and from other places */
  far: Float64Array;
}

/** One corridor's trips a day: its two places, all trips, by rail, by plane (the rest by car). */
export interface FarTrips { km: number; trips: number; rail: number; air: number }

/** The local bins and the corridors' trips on one set of bins: local train, walking, driving and No
 * route as the kernel gives them; each corridor at its places' distance, its rail with the train,
 * its car trips with driving, its flights as planes. */
export function lengthLayers(local: TripLengths | null, far: FarTrips[]): LengthLayers {
  const L = emptyLayers(LAYERS, local);
  const { edges, n, far: farN } = L;
  if (local) {
    for (let i = 0; i < local.n.trips.length; i++) {
      n.train[i] += local.n.rail[i];
      n.walk[i] += local.n.walk[i];
      n.drive[i] += Math.max(0, local.n.drive[i]);
      n.none[i] += local.n.none[i];
    }
  }
  for (const f of far) {
    if (!(f.trips > 0)) continue;
    const b = binOf(edges, f.km);
    n.train[b] += f.rail;
    n.plane[b] += f.air;
    n.drive[b] += Math.max(0, f.trips - f.rail - f.air);
    farN[b] += f.trips;
  }
  return L;
}

/** T-2790: the same trips by purpose: the local bins by purpose, each corridor at its places'
 * distance as inter-city trips. */
export function purposeLayers(local: TripLengths | null, far: FarTrips[]): LengthLayers<PurposeKey> {
  const L = emptyLayers(PURPOSE_LAYERS, local);
  if (local) {
    for (const key of PURPOSE_LAYERS) {
      const v = local.purposes[key];
      if (v) for (let i = 0; i < v.length; i++) L.n[key][i] += v[i];
    }
  }
  for (const f of far) {
    if (!(f.trips > 0)) continue;
    const b = binOf(L.edges, f.km);
    L.n.longer[b] += f.trips;
    L.far[b] += f.trips;
  }
  return L;
}

/** Layers `keys` with no trips yet, on the kernel's bins and `FAR_EDGES_KM`. */
function emptyLayers<K extends string>(keys: readonly K[], local: TripLengths | null): LengthLayers<K> {
  const kernel = local ? Array.from(local.edges, (e) => Math.round(e * 1000) / 1000) : KERNEL_EDGES_KM;
  const edges = [...kernel, ...FAR_EDGES_KM];
  const nb = edges.length + 1;
  const n = Object.fromEntries(keys.map((l) => [l, new Float64Array(nb)])) as Record<K, Float64Array>;
  return { keys, edges, n, far: new Float64Array(nb) };
}

/** sim `LEN_EDGES_KM`, for a graph with no local bins yet */
const KERNEL_EDGES_KM = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 60, 70, 80, 90, 100, 125, 150, 200, 300];

/** The bin of a distance, as sim `len_bin`. */
export function binOf(edges: number[], km: number): number {
  let lo = 0, hi = edges.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (edges[m] <= km) lo = m + 1; else hi = m;
  }
  return lo;
}

/** Bin `i`'s two ends, km, as the log axis draws them. */
export const binRange = (edges: number[], i: number): [number, number] =>
  [i === 0 ? FIRST_LO_KM : edges[i - 1], i < edges.length ? edges[i] : LAST_HI_KM];

/** The trips of every layer in bins `from` to `to - 1`, and those to other places. */
export function binSums<K extends string>(L: LengthLayers<K>, from: number, to: number): Record<K, number> & { all: number; far: number } {
  const layers = Object.fromEntries(L.keys.map((l) => [l, 0])) as Record<K, number>;
  let far = 0;
  for (let i = from; i < to; i++) {
    for (const l of L.keys) layers[l] += L.n[l][i];
    far += L.far[i];
  }
  let all = 0;
  for (const l of L.keys) all += layers[l];
  return Object.assign(layers, { all, far });
}

/** Trips this long or longer (flights, most of them) are drawn together over the axis's last
 * stretch, from here to `AXIS_END_KM`, and are one band in the hover: spread over their true
 * lengths out to 10,000 km they would be a hairline taking a third of the graph's width. */
export const OVER_KM = 1000;
const AXIS_END_KM = 2000;

/** The hover's bands, km: each a run of whole bins (every one is a bin edge); the last, past
 * `OVER_KM`, has no end. */
export const BAND_EDGES_KM = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, OVER_KM];

/** The band holding `km`: its ends (0 below the first, Infinity past the last) and its bins. */
export function bandAt(edges: number[], km: number): { lo: number; hi: number; from: number; to: number } {
  const k = binOf(BAND_EDGES_KM, km);
  const lo = k === 0 ? 0 : BAND_EDGES_KM[k - 1], hi = k < BAND_EDGES_KM.length ? BAND_EDGES_KM[k] : Infinity;
  return { lo, hi, from: lo === 0 ? 0 : binOf(edges, lo), to: hi === Infinity ? edges.length + 1 : binOf(edges, hi) };
}

/** Bin `i`'s two ends as drawn: every bin from `OVER_KM` on over `OVER_KM` to `AXIS_END_KM`. */
const drawnRange = (edges: number[], i: number): [number, number] => {
  const r = binRange(edges, i);
  return r[0] >= OVER_KM ? [OVER_KM, AXIS_END_KM] : r;
};

/** The axis's span, log10 km: from the first bin to the end of the last bin with trips (at least
 * 100 km), so a zone with nothing far keeps its local trips wide. */
export function axisSpan<K extends string>(L: LengthLayers<K>): [number, number] {
  let last = 0;
  for (let i = 0; i <= L.edges.length; i++) if (L.keys.some((l) => L.n[l][i] > 0)) last = i;
  return [Math.log10(FIRST_LO_KM), Math.log10(Math.max(100, drawnRange(L.edges, last)[1]))];
}

/** Bins narrower than this many decades are drawn as smooth as this (T-2365: "gentle smoothing"). */
export const SMOOTH_DECADES = 0.05;

/**
 * Each layer as trips a day per decade of length at `n` points evenly over [u0, u1] (log10 km), so
 * the area under a layer between two lengths is its trips there: each bin's trips spread evenly over
 * its width on the log axis, then blurred by a normal curve of `SMOOTH_DECADES`, reflected at both
 * ends of the axis so no trip is lost past them.
 */
export function layerCurves<K extends string>(L: LengthLayers<K>, u0: number, u1: number, n: number, sigma = SMOOTH_DECADES): { u: Float64Array; y: Record<K, Float64Array> } {
  const u = Float64Array.from({ length: n }, (_, k) => u0 + ((u1 - u0) * k) / (n - 1));
  const y = Object.fromEntries(L.keys.map((l) => [l, new Float64Array(n)])) as Record<K, Float64Array>;
  const nb = L.edges.length + 1;
  for (let i = 0; i < nb; i++) {
    const [lo, hi] = drawnRange(L.edges, i);
    const a = Math.log10(lo), b = Math.min(Math.log10(hi), u1);
    if (!(b > a)) continue;
    const per = L.keys.map((l) => L.n[l][i] / (b - a));
    if (!per.some((v) => v > 0)) continue;
    for (let k = 0; k < n; k++) {
      // the bin's step blurred, at u and at its mirror images in either end
      const w = stepBlur(a, b, u[k], sigma) + stepBlur(a, b, 2 * u0 - u[k], sigma) + stepBlur(a, b, 2 * u1 - u[k], sigma);
      if (w < 1e-9) continue;
      L.keys.forEach((l, j) => { y[l][k] += per[j] * w; });
    }
  }
  return { u, y };
}

/** A step of 1 over [a, b] blurred by a normal curve of `s`, at `x`. */
function stepBlur(a: number, b: number, x: number, s: number): number {
  return 0.5 * (erf((b - x) / (s * Math.SQRT2)) - erf((a - x) / (s * Math.SQRT2)));
}

/** Abramowitz and Stegun 7.1.26, to 1.5e-7. */
function erf(x: number): number {
  const s = x < 0 ? -1 : 1, t = 1 / (1 + 0.3275911 * Math.abs(x));
  const p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return s * (1 - p * Math.exp(-x * x));
}
