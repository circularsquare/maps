// The city view's trips by length (T-2365): trip length along a log axis, each mode's trips a day
// per decade of length stacked, so a colour's area is its trips; the trips to other places with
// them at their places' distance, planes on top. Under it a strip of each length's mode shares.
// Hover: the length band under the pointer, its trips by mode.

import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { City } from "../../game/cities";
import { demandView, intercityDemand } from "../../game/demand";
import { count } from "../../game/format";
import { axisSpan, bandAt, binSums, LAYERS, layerCurves, lengthLayers, type Layer, type LengthLayers } from "../../game/tripLengths";
import { shownDay } from "../../game/ui";
import { farTrips, lengthsOf, wantLengths } from "../../game/zoneLengths";
import { tipBullet, tipGrid } from "../tip";

const W = 300, L = 1, R = 299, T = 4, B = 86;
/** the share strip under the plot, and the labels under it */
const S0 = B + 3, S1 = B + 13, LABEL_Y = S1 + 12, H = LABEL_Y + 3;
/** points along the axis, about one per 2 units */
const N = 150;
/** a layer thinner than this (units) gets no top edge there */
const EDGE_MIN = 0.6;
/** the labelled lengths, most wanted first: powers of ten, then fives, then twos */
const TICKS = [1, 10, 100, 1000, 0.5, 5, 50, 500, 2, 20, 200];
const NAMES: Record<Layer, string> = { train: "Train", walk: "Walking", drive: "Driving", none: "No route", plane: "Plane" };
const COLOUR_KEY: Record<Layer, "train" | "walk" | "drive" | "none" | "other"> = { train: "train", walk: "walk", drive: "drive", none: "none", plane: "other" };
const fmt = (v: number) => v.toLocaleString("en-US");

/** The trips of the band `lo`-`hi` km as the hover's title. */
const bandTitle = (lo: number, hi: number) =>
  lo === 0 ? `Trips under ${fmt(hi)} km per day` : hi === Infinity ? `Trips over ${fmt(lo)} km per day` : `Trips of ${fmt(lo)}-${fmt(hi)} km per day`;

/** The band's hover: each mode's share and trips, all of them, and those to other places. */
function bandTip(Ls: LengthLayers, km: number): string {
  const b = bandAt(Ls.edges, km);
  const s = binSums(Ls, b.from, b.to);
  // train, walking and driving always; planes where the band has some, No route where it shows as 0.1% or more
  const rows: string[][] = LAYERS.filter((l) => (l === "none" ? s.none >= 5e-4 * s.all : l !== "plane" || s.plane > 0)).map((l) => [
    `${tipBullet("", `var(--mode-${COLOUR_KEY[l]})`)}${NAMES[l]}`, s.all > 0 ? `${((100 * s[l]) / s.all).toFixed(1)}%` : "–", count(s[l]),
  ]);
  rows.push([""], ["All", "", count(s.all)]);
  if (s.far > 0) rows.push(["To other places", "", count(s.far)]);
  return `${bandTitle(b.lo, b.hi)}\n${tipGrid(rows)}`;
}

/** Path of the area between two lines of points. */
function band(xs: number[], top: number[], bottom: number[]): string {
  let d = "";
  for (let k = 0; k < xs.length; k++) d += `${k ? "L" : "M"}${xs[k].toFixed(1)},${top[k].toFixed(2)}`;
  for (let k = xs.length - 1; k >= 0; k--) d += `L${xs[k].toFixed(1)},${bottom[k].toFixed(2)}`;
  return d + "Z";
}
/** Path of a line of points, broken where `keep` is false. */
function edge(xs: number[], ys: number[], keep: (k: number) => boolean): string {
  let d = "", on = false;
  for (let k = 0; k < xs.length; k++) {
    if (!keep(k)) { on = false; continue; }
    d += `${on ? "L" : "M"}${xs[k].toFixed(1)},${ys[k].toFixed(2)}`;
    on = true;
  }
  return d;
}

/** What the pointer does not change: the curves, the strip, the labels and their drawing (kept as one
 * element, so a hover re-renders only the band and the hover rect, T-2439). */
function drawGraph(local: NonNullable<ReturnType<typeof lengthsOf>>, market: ReturnType<typeof intercityDemand.peek>, place: City["place"]) {
  const Ls = lengthLayers(local, market ? farTrips(place, market.corridors) : []);
  const [u0, u1] = axisSpan(Ls);
  const { u, y } = layerCurves(Ls, u0, u1, N);
  const x = (v: number) => L + ((v - u0) / (u1 - u0)) * (R - L);
  const xs = Array.from(u, x);
  // stacked tops, in trips per decade, and the scale that puts the highest at the plot's top
  const tops: number[][] = [];
  let acc = new Array<number>(N).fill(0);
  for (const l of LAYERS) {
    acc = acc.map((a, k) => a + y[l][k]);
    tops.push(acc);
  }
  const peak = Math.max(...acc);
  if (!(peak > 0)) return null;
  const py = (v: number) => B - (v / peak) * (B - T);
  const total = acc;
  // the strip: each point's shares, where there are trips enough to have shares
  const shown = (k: number) => total[k] > peak * 2e-3;
  const sy = (f: number) => S1 - f * (S1 - S0);
  const runs: [number, number][] = [];
  for (let k = 0; k < N; k++) if (shown(k)) { if (runs.length && runs[runs.length - 1][1] === k - 1) runs[runs.length - 1][1] = k; else runs.push([k, k]); }

  // labels in order of preference, each kept only clear of those placed; the longest says "km"
  const ticks = TICKS.filter((t) => Math.log10(t) >= u0 && Math.log10(t) <= u1);
  const lastTick = Math.max(...ticks);
  const labels: { t: number; x: number; text: string; end: boolean; l: number; r: number }[] = [];
  for (const t of ticks) {
    const text = fmt(t) + (t === lastTick ? " km" : ""), w = 6 * text.length, tx = x(Math.log10(t));
    const end = tx + w / 2 > R, l = end ? tx - w : tx - w / 2, r = end ? tx : tx + w / 2;
    if (labels.some((o) => l < o.r + 5 && r > o.l - 5)) continue;
    labels.push({ t, x: tx, text, end, l, r });
  }
  const hx = (km: number) => Math.max(L, Math.min(R, km <= 0 ? L : km === Infinity ? R : x(Math.log10(km))));
  const drawn = (
    <>
      {labels.map((l) => <line class="tg-grid" x1={l.x} x2={l.x} y1={T} y2={B} />)}
      {LAYERS.map((l, j) => (
        <path class="tg-area" style={{ fill: `var(--mode-${COLOUR_KEY[l]})` }} d={band(xs, tops[j].map(py), (j ? tops[j - 1] : total.map(() => 0)).map(py))} />
      ))}
      {/* each layer's top: a gap in the panel's colour, then a line in its own */}
      {LAYERS.map((l, j) => {
        const d = edge(xs, tops[j].map(py), (k) => (y[l][k] / peak) * (B - T) > EDGE_MIN);
        return d && <>
          <path class="tg-gap" d={d} />
          <path class="tg-edge" style={{ stroke: `var(--mode-${COLOUR_KEY[l]})` }} d={d} />
        </>;
      })}
      <line class="tg-axis" x1={L} x2={R} y1={B + 0.5} y2={B + 0.5} />
      {runs.map(([a, b]) => {
        const rx = xs.slice(a, b + 1);
        return LAYERS.map((l, j) => {
          const f = (k: number) => tops[j][k] / total[k], g = (k: number) => (j ? tops[j - 1][k] / total[k] : 0);
          const top = rx.map((_, i) => sy(f(a + i))), bot = rx.map((_, i) => sy(g(a + i)));
          return top.some((t, i) => bot[i] - t > 0.05) && <path class="tg-area" style={{ fill: `var(--mode-${COLOUR_KEY[l]})` }} d={band(rx, top, bot)} />;
        });
      })}
      {labels.map((l) => <text class="tg-label" x={l.x} y={LABEL_Y} text-anchor={l.end ? "end" : "middle"}>{l.text}</text>)}
    </>
  );
  return { Ls, u0, u1, hx, drawn };
}

export function TripGraph({ city }: { city: City }) {
  const dv = demandView.value, day = shownDay.value;
  useEffect(() => { wantLengths(city.id); }, [city.id, dv?.version, dv?.round, day]);
  // the km under the pointer, set only when the pointer moves into another band
  const [hoverKm, setHoverKm] = useState<number | null>(null);
  const band0 = useRef<number | null>(null);
  const local = lengthsOf(city.id);
  const market = intercityDemand.value;
  const g = useMemo(() => (local ? drawGraph(local, market, city.place) : null), [local, market, city.place]);
  if (!g) return null;
  const { Ls, u0, u1, hx, drawn } = g;
  const hb = hoverKm === null ? null : bandAt(Ls.edges, hoverKm);
  const pointer = (e: PointerEvent) => {
    const r = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    const px = Math.max(L, Math.min(R, ((e.clientX - r.left) * W) / r.width));
    const km = Math.pow(10, u0 + ((px - L) / (R - L)) * (u1 - u0));
    const b = bandAt(Ls.edges, km);
    if (band0.current === b.lo) return;
    band0.current = b.lo;
    setHoverKm(km);
  };
  const leave = () => { band0.current = null; setHoverKm(null); };

  return (
    <>
      <div class="h">By trip length</div>
      <svg class="trip-graph" viewBox={`0 0 ${W} ${H}`}>
        {drawn}
        {hb && <rect class="tg-band" x={hx(hb.lo)} y={T} width={Math.max(1, hx(hb.hi) - hx(hb.lo))} height={S1 - T} />}
        <rect class="tg-hover" x={L} y={T} width={R - L} height={S1 - T} data-tip={hoverKm === null ? undefined : bandTip(Ls, hoverKm)}
          onPointerMove={pointer} onPointerDown={pointer} onPointerLeave={leave} />
      </svg>
    </>
  );
}
