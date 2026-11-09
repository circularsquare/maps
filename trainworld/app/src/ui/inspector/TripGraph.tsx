// The city view's trips by length (T-2365): trip length along a log axis, each mode's trips a day
// per decade of length stacked, so a colour's area is its trips; the trips to other places with
// them at their places' distance, planes on top. Under it a strip of each length's mode shares.
// Hover: the length band under the pointer, its trips by mode. A switch draws the same trips by
// purpose instead (T-2790).

import { signal } from "@preact/signals";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { City } from "../../game/cities";
import { demandAskTurn, demandView, intercityDemand } from "../../game/demand";
import { count } from "../../game/format";
import { t, tNumber, type MessageKey } from "../../i18n";
import { purposeColour, purposeName, type PurposeKey } from "../../game/purposes";
import { axisSpan, bandAt, binSums, LAYERS, layerCurves, lengthLayers, purposeLayers, type Layer, type LengthLayers } from "../../game/tripLengths";
import { shownDay } from "../../game/ui";
import { farTrips, lengthsOf, wantLengths } from "../../game/zoneLengths";
import { tipBullet, tipGrid } from "../tip";
import { Segmented } from "../widgets";
import { pct1 } from "./tripText";

/** What the graph's layers are; kept from one zone to the next. */
type LayersBy = "mode" | "purpose";
const layersBy = signal<LayersBy>("mode");

const W = 300, L = 1, R = 299, T = 4, B = 86;
/** the share strip under the plot, and the labels under it */
const S0 = B + 3, S1 = B + 13, LABEL_Y = S1 + 12, H = LABEL_Y + 3;
/** points along the axis, about one per 2 units */
const N = 150;
/** a layer thinner than this (units) gets no top edge there */
const EDGE_MIN = 0.6;
/** the labelled lengths, most wanted first: powers of ten, then fives, then twos */
const TICKS = [1, 10, 100, 1000, 0.5, 5, 50, 500, 2, 20, 200];
const NAME_KEYS = { train: "tripGraph.train", walk: "tripGraph.walking", drive: "tripGraph.driving", none: "tripGraph.noRoute", plane: "tripGraph.plane" } as const satisfies Record<Layer, MessageKey>;
const COLOUR_KEY: Record<Layer, "train" | "walk" | "drive" | "none" | "other"> = { train: "train", walk: "walk", drive: "drive", none: "none", plane: "other" };
const fmt = (v: number) => tNumber(v);

/** The trips of the band `lo`-`hi` km as the hover's title. */
const bandTitle = (lo: number, hi: number) =>
  lo === 0 ? t("tripGraph.under", { km: fmt(hi) }) : hi === Infinity ? t("tripGraph.over", { km: fmt(lo) }) : t("tripGraph.between", { from: fmt(lo), to: fmt(hi) });

/** The band's hover: each mode's share and trips, all of them, and those to other places. */
function modeTip(Ls: LengthLayers, km: number): string {
  const b = bandAt(Ls.edges, km);
  const s = binSums(Ls, b.from, b.to);
  // train, walking and driving always; planes where the band has some, No route where it shows as 0.1% or more
  const rows: string[][] = LAYERS.filter((l) => (l === "none" ? s.none >= 5e-4 * s.all : l !== "plane" || s.plane > 0)).map((l) => [
    `${tipBullet("", `var(--mode-${COLOUR_KEY[l]})`)}${t(NAME_KEYS[l])}`, s.all > 0 ? pct1(s[l] / s.all) : "–", count(s[l]),
  ]);
  rows.push([""], [t("tripGraph.all"), "", count(s.all)]);
  if (s.far > 0) rows.push([t("tripGraph.toOther"), "", count(s.far)]);
  return `${bandTitle(b.lo, b.hi)}\n${tipGrid(rows)}`;
}

/** The band's hover by purpose: each purpose that shows as 0.1% or more, then all of them. */
function purposeTip(Ls: LengthLayers<PurposeKey>, km: number): string {
  const b = bandAt(Ls.edges, km);
  const s = binSums(Ls, b.from, b.to);
  const rows: string[][] = Ls.keys.filter((k) => s[k] > 0 && s[k] >= 5e-4 * s.all).map((k) => [
    `${tipBullet("", purposeColour(k))}${purposeName(k)}`, pct1(s[k] / s.all), count(s[k]),
  ]);
  rows.push([""], [t("tripGraph.all"), "", count(s.all)]);
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
function drawGraph<K extends string>(Ls: LengthLayers<K>, colour: (l: K) => string, tip: (km: number) => string) {
  const keys = Ls.keys;
  const [u0, u1] = axisSpan(Ls);
  const { u, y } = layerCurves(Ls, u0, u1, N);
  const x = (v: number) => L + ((v - u0) / (u1 - u0)) * (R - L);
  const xs = Array.from(u, x);
  // stacked tops, in trips per decade, and the scale that puts the highest at the plot's top
  const tops: number[][] = [];
  let acc = new Array<number>(N).fill(0);
  for (const l of keys) {
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
  const kmSuffix = t("tripGraph.kmSuffix");
  for (const t of ticks) {
    const text = fmt(t) + (t === lastTick ? kmSuffix : ""), w = 6 * text.length, tx = x(Math.log10(t));
    const end = tx + w / 2 > R, l = end ? tx - w : tx - w / 2, r = end ? tx : tx + w / 2;
    if (labels.some((o) => l < o.r + 5 && r > o.l - 5)) continue;
    labels.push({ t, x: tx, text, end, l, r });
  }
  const hx = (km: number) => Math.max(L, Math.min(R, km <= 0 ? L : km === Infinity ? R : x(Math.log10(km))));
  const drawn = (
    <>
      {labels.map((l) => <line class="tg-grid" x1={l.x} x2={l.x} y1={T} y2={B} />)}
      {keys.map((l, j) => (
        <path class="tg-area" style={{ fill: colour(l) }} d={band(xs, tops[j].map(py), (j ? tops[j - 1] : total.map(() => 0)).map(py))} />
      ))}
      {/* each layer's top: a gap in the panel's colour, then a line in its own */}
      {keys.map((l, j) => {
        const d = edge(xs, tops[j].map(py), (k) => (y[l][k] / peak) * (B - T) > EDGE_MIN);
        return d && <>
          <path class="tg-gap" d={d} />
          <path class="tg-edge" style={{ stroke: colour(l) }} d={d} />
        </>;
      })}
      <line class="tg-axis" x1={L} x2={R} y1={B + 0.5} y2={B + 0.5} />
      {runs.map(([a, b]) => {
        const rx = xs.slice(a, b + 1);
        return keys.map((l, j) => {
          const f =(k: number) => tops[j][k] / total[k], g = (k: number) => (j ? tops[j - 1][k] / total[k] : 0);
          const top = rx.map((_, i) => sy(f(a + i))), bot = rx.map((_, i) => sy(g(a + i)));
          return top.some((t, i) => bot[i] - t > 0.05) && <path class="tg-area" style={{ fill: colour(l) }} d={band(rx, top, bot)} />;
        });
      })}
      {labels.map((l) => <text class="tg-label" x={l.x} y={LABEL_Y} text-anchor={l.end ? "end" : "middle"}>{l.text}</text>)}
    </>
  );
  return { edges: Ls.edges, u0, u1, hx, drawn, tip };
}

const modeColour = (l: Layer) => `var(--mode-${COLOUR_KEY[l]})`;

export function TripGraph({ city }: { city: City }) {
  const dv = demandView.value, day = shownDay.value;
  useEffect(() => { wantLengths(city.id); }, [city.id, dv?.version, dv?.round, day, demandAskTurn.value]);
  // the km under the pointer, set only when the pointer moves into another band
  const [hoverKm, setHoverKm] = useState<number | null>(null);
  const band0 = useRef<number | null>(null);
  const local = lengthsOf(city.id);
  const market = intercityDemand.value;
  const by = layersBy.value;
  const g = useMemo(() => {
    if (!local) return null;
    const far = market ? farTrips(city.place, market.corridors) : [];
    if (by === "purpose") {
      const Ls = purposeLayers(local, far);
      return drawGraph(Ls, purposeColour, (km) => purposeTip(Ls, km));
    }
    const Ls = lengthLayers(local, far);
    return drawGraph(Ls, modeColour, (km) => modeTip(Ls, km));
  }, [local, market, city.place, by]);
  if (!g) return null;
  const { edges, u0, u1, hx, drawn, tip } = g;
  const hb = hoverKm === null ? null : bandAt(edges, hoverKm);
  const pointer = (e: PointerEvent) => {
    const r = (e.currentTarget as SVGElement).ownerSVGElement!.getBoundingClientRect();
    const px = Math.max(L, Math.min(R, ((e.clientX - r.left) * W) / r.width));
    const km = Math.pow(10, u0 + ((px - L) / (R - L)) * (u1 - u0));
    const b = bandAt(edges, km);
    if (band0.current === b.lo) return;
    band0.current = b.lo;
    setHoverKm(km);
  };
  const leave = () => { band0.current = null; setHoverKm(null); };

  return (
    <>
      <div class="h">
        {t("tripGraph.title")}
        <span class="right"><Segmented<LayersBy> options={[["mode", t("tripGraph.byMode")], ["purpose", t("tripGraph.byPurpose")]]} value={by} onChange={(v) => (layersBy.value = v)} /></span>
      </div>
      <svg class="trip-graph" viewBox={`0 0 ${W} ${H}`}>
        {drawn}
        {hb && <rect class="tg-band" x={hx(hb.lo)} y={T} width={Math.max(1, hx(hb.hi) - hx(hb.lo))} height={S1 - T} />}
        <rect class="tg-hover" x={L} y={T} width={R - L} height={S1 - T} data-tip={hoverKm === null ? undefined : tip(hoverKm)}
          onPointerMove={pointer} onPointerDown={pointer} onPointerLeave={leave} />
      </svg>
    </>
  );
}
