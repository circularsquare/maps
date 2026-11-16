import type { PopView } from "./demandClusters";

export const VIEW_TOLERANCE = 0.01;
/** T-1764: a line clipped to `FEW_STOPS` or fewer stops in a view (a pad at an end) is one the view's
 * riders only cross: its trains an hour, run times and span do not touch the view, and the whole
 * solve that follows every edit brings them in */
const FEW_STOPS = 3;
/** T-1764: per line of the view, 1 where it is such a line, with each stop's line */
function fewStopsOf(v: PopView): { few: Uint8Array; lineOfStop: Uint32Array } | null {
  if (!v.pads.length) return null;
  const n = v.net.lineN, pad = new Set(v.pads), few = new Uint8Array(n.length), lineOfStop = new Uint32Array(v.net.stops.length);
  let o = 0, any = false;
  for (let l = 0; l < n.length; l++) {
    lineOfStop.fill(l, o, o + n[l]);
    if (n[l] <= FEW_STOPS && (pad.has(2 * o) || pad.has(2 * o + n[l]))) { few[l] = 1; any = true; }
    o += n[l];
  }
  return any ? { few, lineOfStop } : null;
}
/** T-1733: trains an hour (and the share of riders with a train due, which follows them) move by a
 * few percent on far lines when a line is built (all of Japan: 2.7% on an Osaka line for a Tokyo
 * line, which made Osaka a third of the local solve); within this a zone keeps its figures until
 * the whole solve */
const TRAINS_TOLERANCE = 0.05;
/** T-1764: `span` is a share (0 to 1) in steps of 0.05, one train more or less in a period's day; a
 * single step, 0.05000001 in float32, is the same view, so a far line's step does not touch its zones */
const SPAN_TOLERANCE = 0.06;

/** T-1733: whether two views give a population the same inputs (so its state solved on one holds
 * for the other); their route nodes in the whole network may differ */
/** each value replaced by the order it first appears in */
function canonical(a: Uint32Array): Uint32Array {
  const seen = new Map<number, number>();
  return a.map((v) => { let k = seen.get(v); if (k === undefined) seen.set(v, (k = seen.size)); return k; });
}
export function sameView(x: PopView, y: PopView): boolean {
  return viewDiff(x, y) === null;
}
export function viewDiff(x: PopView, y: PopView): string | null {
  if (x === y) return null;
  if (x.pads.length !== y.pads.length || x.pads.some((v, k) => v !== y.pads[k])) return "pads";
  const a = x.net as unknown as Record<string, unknown>, b = y.net as unknown as Record<string, unknown>;
  const few = fewStopsOf(x);
  const nr = 2 * x.net.stops.length;
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    let u = a[k], v = b[k];
    // a station of no complex is numbered past every complex (`packService`): the same grouping
    // under other numbers is the same input
    if (k === "complex" && u instanceof Uint32Array && v instanceof Uint32Array) [u, v] = [canonical(u), canonical(v)];
    if (ArrayBuffer.isView(u) && ArrayBuffer.isView(v)) {
      const p = u as unknown as ArrayLike<number>, w = v as unknown as ArrayLike<number>;
      if (p.length !== w.length) return `${k} length ${p.length} ${w.length}`; // i18n-ignore: debug diff
      // run times, trains an hour and the like move by float noise and by small knock-on changes
      // when a line is added beside them (0.1% seen): within VIEW_TOLERANCE they count as the same
      const tol = !(u instanceof Float32Array) || k === "stXY" || k === "catchXY" ? 0 : k === "tph" ? TRAINS_TOLERANCE : k === "span" ? SPAN_TOLERANCE : VIEW_TOLERANCE;
      // lines of few stops in the view: their trains, run times and span are not this view's inputs
      const lineOf = few && (k === "tph" ? (i: number) => Math.floor(i / 3) : k === "times" ? (i: number) => few.lineOfStop[Math.floor(i / 6)] : k === "span" ? (i: number) => few.lineOfStop[Math.floor((i % nr) / 2)] : null);
      for (let i = 0; i < p.length; i++) if (p[i] !== w[i] && !(lineOf && few!.few[lineOf(i)]) && Math.abs(p[i] - w[i]) > tol * Math.max(1, Math.abs(p[i]))) return `${k}[${i}] ${p[i]} ${w[i]}`;
    } else if (JSON.stringify(u) !== JSON.stringify(v)) return `${k} ${JSON.stringify(u)?.slice(0, 80)} ${JSON.stringify(v)?.slice(0, 80)}`;
  }
  return null;
}
