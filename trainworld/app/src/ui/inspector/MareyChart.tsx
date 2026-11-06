// T-2603: a line's timetable chart (a Marey chart): time across, the line's stops down by their
// distance along it, each train's trip a line. The trips are the clock worker's (`TrackApi.line_trips`,
// the departures the map draws) over a window around now; each trip's curve is its run profile from
// the network buffers (`profileS`), so it bends where the train speeds up, slows and stands. Opened
// from the line panel's schedule heading; it follows the selected line. docs/ui-inspectors.md.
// T-2660: other lines' trains on the stretches they share, and the flat junctions and crossings where
// other routes' trains cross its path (mareyShare.ts). T-2665: the span follows how busy the line is,
// and the wheel zooms it.

import { signal } from "@preact/signals";
import { useEffect, useLayoutEffect, useMemo, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { clockTime } from "../../game/format";
import { t as tr, tList } from "../../i18n";
import { drawnReversed } from "../../game/northUp";
import { nearestStation } from "../../game/places";
import { display, selection, setDisplay } from "../../game/ui";
import { linesById, linesByNum, stationNames, stationsById, world } from "../../game/world";
import type { LineInput, LineStats } from "../../game/types";
import { live } from "../../map/live";
import type { NetworkBuffers } from "../../map/network";
import { queryClock } from "../../workers/clockClient";
import { useClockQuery } from "../useClockQuery";
import { Chip, Icon } from "../widgets";
import { clipPath, gradePasses, sharedStretches, type Pass, type Shared } from "./mareyShare";

/** Whether the chart is open; it shows the selected rail line, and closes when no line is selected. */
export const mareyOpen = signal(false);

const W_PX = 600, GUTTER = 92, PAD_R = 18, TOP = 8, AXIS = 22;
const PLOT_W = W_PX - GUTTER - PAD_R;
/** the window's share before now */
const BEFORE = 0.25;
const HOUR = 3600, QUARTER = 900;
/** a busy line's default span holds about this many departures each way */
const DENSE_DEPARTURES = 12;
/** the wheel's span: each notch times or over this, between a quarter hour and a day */
const ZOOM_STEP = 1.25, MIN_SPAN = QUARTER, MAX_SPAN = 24 * HOUR;
const DEBUG = typeof location !== "undefined" && new URLSearchParams(location.search).has("debug");

/** The chart's span by default, s: two round trips, three of its longest headways, at least 2 hours;
 * on a busy line no more than 12 of its shortest headways or half a round trip (one trip end to end),
 * whichever is longer, at least half an hour; at most 8 hours; in whole quarter hours. */
export function defaultSpan(l: LineInput, st: LineStats): number {
  const tph = Object.values(l.tph).filter((v) => v > 0);
  const rt = st.roundTripS || 0;
  let s = Math.max(2 * rt, 3 * (tph.length ? HOUR / Math.min(...tph) : 0), 2 * HOUR);
  if (tph.length) s = Math.min(s, Math.max(DENSE_DEPARTURES * (HOUR / Math.max(...tph)), 0.5 * rt, HOUR / 2));
  return Math.ceil(Math.min(8 * HOUR, s) / QUARTER - 1e-6) * QUARTER;
}

/** Tick spacing in minutes: the smallest giving at most 8 ticks over the span. */
const tickMinutes = (span: number) => [5, 10, 15, 30, 60, 120, 180, 240].find((m) => span / 60 / m <= 8) ?? 240;

/** Run `r`'s offset `s` as a distance along run 0, through the stops both runs share (run 1's stop
 * `k` is run 0's `n - 1 - k`), straight between them. */
function toRun0(stopS: [number[], number[]], r: number, s: number): number {
  const [a, b] = stopS;
  if (r === 0 || b.length !== a.length || a.length < 2) return r === 0 ? s : (a[a.length - 1] ?? 0) - s;
  const n = b.length;
  if (s <= b[0]) return a[n - 1];
  for (let k = 1; k < n; k++) {
    if (s <= b[k]) {
      const f = b[k] > b[k - 1] ? (s - b[k - 1]) / (b[k] - b[k - 1]) : 1;
      return a[n - k] + (a[n - 1 - k] - a[n - k]) * f;
    }
  }
  return a[0];
}

/** A profile's curve as (trip seconds, offset) points: each phase's start, a few points inside a
 * phase that speeds up or slows, and the trip's end. */
function profilePoints(n: NetworkBuffers, p: number): number[] {
  const first = n.meta[p * 4], count = n.meta[p * 4 + 1], tripS = n.meta[p * 4 + 2];
  const out: number[] = [];
  for (let i = first; i < first + count; i++) {
    const t = n.phases[i * 4], s = n.phases[i * 4 + 1], v = n.phases[i * 4 + 2], a = n.phases[i * 4 + 3];
    const t1 = i + 1 < first + count ? n.phases[(i + 1) * 4] : tripS;
    out.push(t, s);
    if (a !== 0 && t1 > t) {
      const stop = a < 0 ? v / -a : Infinity;
      for (const f of [0.25, 0.5, 0.75]) {
        const tt = Math.min((t1 - t) * f, stop);
        out.push(t + (t1 - t) * f, s + v * tt + 0.5 * a * tt * tt);
      }
    }
  }
  if (count) {
    const i = first + count - 1, t = n.phases[i * 4], s = n.phases[i * 4 + 1], v = n.phases[i * 4 + 2], a = n.phases[i * 4 + 3];
    let tt = Math.max(0, tripS - t);
    if (a < 0) tt = Math.min(tt, v / -a);
    out.push(tripS, s + v * tt + 0.5 * a * tt * tt);
  }
  return out;
}

/** The first trip second at which profile `p` is within a metre of offset `s` (its arrival there). */
function reachTau(pts: number[], s: number): number {
  for (let i = 0; i < pts.length; i += 2) if (pts[i + 1] >= s - 1) return pts[i];
  return pts[pts.length - 2] ?? 0;
}

interface Trip { p: number; dep: number; end: number }
const tripsOf = (d: Float64Array): Trip[] => Array.from({ length: d.length / 4 }, (_, i) => ({ p: d[i * 4], dep: d[i * 4 + 1], end: d[i * 4 + 2] }));

/** What the chart draws besides the line's own trains (T-2660): the lines over its track and the
 * at-grade crossings of its path, by the network build. */
interface Links { shared: Map<number, Shared[]>; passes: Pass[] }

/** A flat junction or crossing on the line's path that another line uses or crosses (`TrackApi.line_marks`). */
interface Flat { crossing: boolean; s: number; tip: string }
function flatsFrom(d: Float64Array, stopS: number[]): Flat[] {
  const lines = world.value ? linesByNum(world.value) : new Map<number, LineInput>();
  const out: Flat[] = [];
  for (let i = 0; i + 5 < d.length; ) {
    const n = d[i + 5], g = d[i + 1], f = d[i + 2];
    if (n > 0) {
      const crossing = d[i] === 5, near = nearestStation(d[i + 3], d[i + 4])?.name ?? "";
      const list = tList(Array.from(d.subarray(i + 6, i + 6 + n), (o) => lines.get(o)?.name ?? tr("lineInspector.lineNum", { num: o })));
      const s = (stopS[g] ?? 0) + f * ((stopS[g + 1] ?? 0) - (stopS[g] ?? 0));
      out.push({ crossing, s, tip: tr(crossing ? "lineInspector.flatCrossing" : "lineInspector.flatJunction", { hasNear: !!near, near, hasOthers: true, list }) });
    }
    i += 6 + n;
  }
  return out;
}

interface Drawn { id: string; d: string }
interface Run { key: string; href: string; x: number; colour?: string; tip: string }
interface Mark { x: number; y: number; colour: string; tip: string }

/** The scrolled part of the chart: the time ticks and every train, built once per answer and moved
 * as a whole by its parent, so a new minute redraws none of it. */
function TripLayer({ width, height, ticks, defs, stands, others, passes, runs, onHover }: {
  width: number; height: number; ticks: number[]; defs: Drawn[]; stands: number[][]; others: Run[]; passes: Mark[]; runs: Run[]; onHover: (k: string | null) => void;
}) {
  return (
    <svg class="marey-layer" width={width} height={height}>
      <defs>{defs.map((p) => <path id={p.id} d={p.d} />)}</defs>
      {ticks.map((x) => <line class="marey-tick" x1={x} x2={x} y1={TOP} y2={height - AXIS} />)}
      {stands.map((s) => <line class="marey-stand" x1={s[0]} x2={s[1]} y1={s[2]} y2={s[2]} />)}
      {others.map((r) => <use class="run other" href={r.href} x={r.x} style={{ "--run": r.colour }} />)}
      {others.map((r) => <use class="run-hit" href={r.href} x={r.x} data-tip={r.tip} onPointerEnter={() => onHover(r.key)} onPointerLeave={() => onHover(null)} />)}
      {passes.map((m) => <circle class="marey-pass" cx={m.x} cy={m.y} r={2.5} style={{ "--run": m.colour }} data-tip={m.tip} />)}
      {runs.map((r) => <use class="run" href={r.href} x={r.x} />)}
      {runs.map((r) => <use class="run-hit" href={r.href} x={r.x} data-tip={r.tip} onPointerEnter={() => onHover(r.key)} onPointerLeave={() => onHover(null)} />)}
    </svg>
  );
}

function MareyBody({ l, st, links }: { l: LineInput; st: LineStats; links: Links | null }) {
  const r0 = DEBUG ? performance.now() : 0;
  const w = world.value!;
  const net = live.overlay?.renderer.net ?? null;
  const [zoom, setZoom] = useState(0);
  const span = Math.min(MAX_SPAN, Math.max(MIN_SPAN, defaultSpan(l, st) * ZOOM_STEP ** zoom));
  const q = span / 4;
  // fetched a quarter window either side of the window, again as now enters the next quarter
  const now = clock.minute.value * 60;
  const slot = Math.floor(now / q);
  const base = (slot - 1) * q;
  const answer = useClockQuery({ what: "lineTrips", line: l.num, t0: base, t1: base + 6 * q }, [l.num, w.version, base, span],
    (d) => ({ line: l.num, base, span, trips: tripsOf(d) }));
  const showOthers = display.value.chartOtherLines;
  const otherKey = showOthers && links ? [...new Set([...links.shared.keys(), ...links.passes.map((p) => p.o)])].join(",") : "";
  const otherNums = useMemo(() => (otherKey ? otherKey.split(",").map(Number) : []), [otherKey]);
  const [otherAnswer, setOtherAnswer] = useState<{ base: number; span: number; byLine: Map<number, Trip[]> } | null>(null);
  useEffect(() => {
    if (!otherNums.length) { setOtherAnswer(null); return; }
    let on = true;
    void Promise.all(otherNums.map((o) => queryClock({ what: "lineTrips", line: o, t0: base, t1: base + 6 * q }))).then((ds) => {
      if (!on) return;
      const byLine = new Map<number, Trip[]>();
      ds.forEach((d, i) => { if (d) byLine.set(otherNums[i], tripsOf(d)); });
      setOtherAnswer({ base, span, byLine });
    });
    return () => { on = false; };
  }, [otherNums, w.version, base, span]);
  const flats = useClockQuery({ what: "lineMarks", line: l.num }, [l.num, w.version], (d) => flatsFrom(d, st.stopS[0])) ?? [];
  const [hover, setHover] = useState<string | null>(null);

  const names = stationNames(w), byId = stationsById(w);
  const n = l.stops.length;
  const flip = drawnReversed(byId.get(l.stops[0]), byId.get(l.stops[n - 1]), !!st.loop);
  const d0 = st.stopS[0][0] ?? 0, d1 = st.stopS[0][n - 1] ?? 1;
  const H = Math.max(120, Math.min(340, n * 20));
  const yOf = (d: number) => {
    const f = d1 > d0 ? (d - d0) / (d1 - d0) : 0;
    return TOP + (flip ? 1 - f : f) * H;
  };
  const k = PLOT_W / span;
  const shift = st.shift ?? 0;
  const local = (t: number) => clockTime(Math.floor((t + shift) / 60));
  const id = `marey-${l.num}`;

  // the line's own trips, built once per answer: each profile's path in pixels from its departure
  const drawn = useMemo(() => {
    const t0 = performance.now();
    if (!answer || answer.line !== l.num || !net) return null;
    const paths = new Map<number, { d: string; pts: number[]; run: number }>();
    for (const t of answer.trips) {
      if (paths.has(t.p) || !net.meta[t.p * 4 + 1]) continue;
      const run = Math.floor(t.p / 3) % 2, pts = profilePoints(net, t.p);
      let d = "";
      for (let i = 0; i < pts.length; i += 2) d += `${i ? "L" : "M"}${(pts[i] * k).toFixed(1)} ${yOf(toRun0(st.stopS, run, pts[i + 1])).toFixed(1)}`;
      paths.set(t.p, { d, pts, run });
    }
    const stands: number[][] = [];
    const runs: Run[] = answer.trips.filter((t) => paths.has(t.p)).map((t, i) => {
      const pr = paths.get(t.p)!, tripS = net.meta[t.p * 4 + 2], stops = st.stopS[pr.run];
      const first = pr.run === 0 ? l.stops[0] : l.stops[n - 1], last = pr.run === 0 ? l.stops[n - 1] : l.stops[0];
      const arrive = t.dep + reachTau(pr.pts, stops[stops.length - 1] ?? 0);
      if (t.end > t.dep + tripS + 1) stands.push([(t.dep + tripS - answer.base) * k, (t.end - answer.base) * k, yOf(toRun0(st.stopS, pr.run, pr.pts[pr.pts.length - 1] ?? 0))]);
      return {
        key: `own${i}`, href: `#${id}-p${t.p}`, x: (t.dep - answer.base) * k,
        tip: tr("mareyChart.tip", { from: names.get(first) ?? "", leave: local(t.dep), to: names.get(last) ?? "", arrive: local(arrive) }),
      };
    });
    (window as unknown as { __mareyMs?: number }).__mareyMs = performance.now() - t0;
    return { defs: [...paths].map(([p, pr]) => ({ id: `${id}-p${p}`, d: pr.d })), stands, runs };
  }, [answer, net, w.version, flip, H, k]);

  // the other lines' trains on the stretches they share, and their passes over at-grade conflicts
  const otherDrawn = useMemo(() => {
    if (!otherAnswer || !answer || otherAnswer.base !== answer.base || !net || !links) return null;
    const byNum = linesByNum(w);
    const defs: Drawn[] = [], runs: Run[] = [], marks: Mark[] = [];
    for (const [o, trips] of otherAnswer.byLine) {
      const ol = byNum.get(o), ost = ol && w.lineStats[ol.id];
      if (!ol || !ost || ost.bus) continue;
      const shared = links.shared.get(o) ?? [], passes = links.passes.filter((p) => p.o === o);
      const pts = new Map<number, { pts: number[]; d: string; run: number; len: number }>();
      for (const t of trips) {
        if (pts.has(t.p) || !net.meta[t.p * 4 + 1]) continue;
        const run = Math.floor(t.p / 3) % 2, len = net.meta[t.p * 4 + 3], raw = profilePoints(net, t.p);
        let d = "";
        if (shared.length) {
          const s0 = run === 0 ? raw : raw.map((v, i) => (i % 2 ? len - v : v));
          for (const sh of shared) d += clipPath(s0, sh.a, sh.b, (t) => t * k, (s) => yOf(sh.c + sh.sg * (s - sh.a)));
        }
        if (d) defs.push({ id: `${id}-o${t.p}`, d });
        pts.set(t.p, { pts: raw, d, run, len });
      }
      const m = ol.stops.length;
      trips.forEach((t, i) => {
        const pr = pts.get(t.p);
        if (!pr) return;
        if (pr.d) {
          const stops = ost.stopS[pr.run];
          const first = pr.run === 0 ? ol.stops[0] : ol.stops[m - 1], last = pr.run === 0 ? ol.stops[m - 1] : ol.stops[0];
          const arrive = t.dep + reachTau(pr.pts, stops[stops.length - 1] ?? 0);
          runs.push({
            key: `o${o}-${i}`, href: `#${id}-o${t.p}`, x: (t.dep - answer.base) * k, colour: ol.colour,
            tip: tr("mareyChart.otherTip", { line: ol.name, from: names.get(first) ?? "", leave: local(t.dep), to: names.get(last) ?? "", arrive: local(arrive) }),
          });
        }
        const reach = pr.pts[pr.pts.length - 1] ?? 0;
        for (const ps of passes) {
          if (ps.run >= 0 && ps.run !== pr.run) continue;
          const s = pr.run === 0 ? ps.sO : pr.len - ps.sO;
          if (s < -1 || s > reach + 1) continue;
          const at = t.dep + reachTau(pr.pts, s);
          marks.push({ x: (at - answer.base) * k, y: yOf(ps.sL), colour: ol.colour, tip: tr("mareyChart.passTip", { line: ol.name, time: local(at) }) });
        }
      });
    }
    return { defs, runs, marks };
  }, [otherAnswer, answer, net, links, w.version, flip, H, k]);

  // the time ticks over the fetched span, on the line's local clock
  const fetchedBase = answer?.base ?? base;
  const step = tickMinutes(span) * 60;
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let t = Math.ceil((fetchedBase + shift) / step) * step - shift; t <= fetchedBase + 6 * q; t += step) out.push(t);
    return out;
  }, [fetchedBase, shift, step, q]);
  const height = TOP + H + AXIS;
  const layer = useMemo(() => (
    <TripLayer width={6 * q * k} height={height} ticks={ticks.map((t) => (t - fetchedBase) * k)} defs={[...(drawn?.defs ?? []), ...(otherDrawn?.defs ?? [])]}
      stands={drawn?.stands ?? []} others={otherDrawn?.runs ?? []} passes={otherDrawn?.marks ?? []} runs={drawn?.runs ?? []} onHover={setHover} />
  ), [drawn, otherDrawn, ticks, height]);
  const lit = hover === null ? null : (drawn?.runs.find((r) => r.key === hover) ?? otherDrawn?.runs.find((r) => r.key === hover) ?? null);

  const view0 = now - BEFORE * span;
  const offset = -(view0 - fetchedBase) * k;
  const nowX = BEFORE * PLOT_W;
  // labels down the side, each kept only where it clears the last one shown (the ends always)
  const order = l.stops.map((_, i) => i).sort((a, b) => yOf(st.stopS[0][a] ?? 0) - yOf(st.stopS[0][b] ?? 0));
  const labels: { i: number; y: number }[] = [];
  for (const i of order) {
    const y = yOf(st.stopS[0][i] ?? 0);
    const end = i === 0 || i === n - 1;
    if (labels.length && y - labels[labels.length - 1].y < 12) {
      if (!end) continue;
      if (labels.length > 1 || !(labels[0].i === 0 || labels[0].i === n - 1)) labels.pop();
    }
    labels.push({ i, y });
  }
  const fit = (s: string) => (s.length > 14 ? s.slice(0, 13) + "…" : s);
  const onWheel = (e: WheelEvent) => {
    if (!e.deltaY) return;
    e.preventDefault();
    const dir = e.deltaY > 0 ? 1 : -1;
    if ((dir > 0 && span >= MAX_SPAN) || (dir < 0 && span <= MIN_SPAN)) return;
    setZoom(zoom + dir);
  };
  useLayoutEffect(() => {
    if (!DEBUG) return;
    const g = window as unknown as { __mareyRender?: number[] };
    (g.__mareyRender ??= []).push(performance.now() - r0);
  });
  return (
    <div class="marey-chart" style={{ width: W_PX + "px", height: height + "px", "--run": l.colour }} onWheel={onWheel}>
      <svg class="marey-base" width={W_PX} height={height}>
        {labels.map(({ i, y }) => (
          <text class="marey-stop" x={GUTTER - 8} y={y + 4}>{fit(names.get(l.stops[i]) ?? "")}</text>
        ))}
        {l.stops.map((_, i) => <line class="marey-grid" x1={GUTTER} x2={GUTTER + PLOT_W} y1={yOf(st.stopS[0][i] ?? 0)} y2={yOf(st.stopS[0][i] ?? 0)} />)}
        {flats.map((f) => <line class="marey-flat" x1={GUTTER} x2={GUTTER + PLOT_W} y1={yOf(f.s)} y2={yOf(f.s)} />)}
        {flats.map((f) => {
          const x = GUTTER + PLOT_W + 9, y = yOf(f.s);
          return (
            <g class={"marey-flat-mark " + (f.crossing ? "crossing" : "junction")} data-tip={f.tip}>
              <circle class="hit" cx={x} cy={y} r={7} />
              {f.crossing ? <path d={`M${x - 3.5} ${y - 3.5}l7 7m0 -7l-7 7`} /> : <rect x={x - 3} y={y - 3} width={6} height={6} transform={`rotate(45 ${x} ${y})`} />}
            </g>
          );
        })}
      </svg>
      <div class="marey-plot" style={{ left: GUTTER + "px", width: PLOT_W + "px", height: height + "px" }}>
        <div class="marey-scroll" style={{ transform: `translateX(${offset.toFixed(1)}px)` }}>
          {layer}
          <svg class="marey-lit" width={6 * q * k} height={height}>
            {lit && <use class={"run on" + (lit.colour ? " other" : "")} href={lit.href} x={lit.x} style={lit.colour ? { "--run": lit.colour } : undefined} />}
          </svg>
        </div>
        <svg class="marey-over" width={PLOT_W} height={height}>
          <rect class="marey-past" x={0} y={TOP - 2} width={nowX} height={H + 4} />
          {ticks.map((t) => {
            const x = (t - fetchedBase) * k + offset;
            return x > 16 && x < PLOT_W - 16 && <text class="marey-time" x={x} y={TOP + H + 15}>{local(t)}</text>;
          })}
          <line class="marey-now" x1={nowX} x2={nowX} y1={TOP - 4} y2={TOP + H + 4} />
        </svg>
      </div>
    </div>
  );
}

/** The chart's panel beside the dock, for the selected rail line while `mareyOpen`. */
export function MareyPanel() {
  const open = mareyOpen.value;
  const sel = selection.value;
  const w = world.value;
  const l = open && w && sel?.kind === "line" ? linesById(w).get(sel.line) : undefined;
  const st = l && w ? w.lineStats[l.id] : undefined;
  const net = live.overlay?.renderer.net ?? null;
  const n = l?.stops.length ?? 0;
  const d0 = st?.stopS[0][0] ?? 0, d1 = st?.stopS[0][n - 1] ?? 0;
  // by the network build and the line's ends only: the stats object is new on every state message
  const links = useMemo<Links | null>(() => {
    if (!net || !l || n < 2) return null;
    return { shared: sharedStretches(net, l.num, d0, d1), passes: gradePasses(net, l.num, d0, d1) };
  }, [net, l?.num, w?.version, d0, d1]);
  // no rail line selected: closed, so it does not come back by itself on the next line
  useEffect(() => { if (open && w && (!sel || sel.kind !== "line" || w.lineStats[sel.line]?.bus)) mareyOpen.value = false; }, [open, sel, w]);
  useEffect(() => {
    if (!open) return;
    // before main.ts's Escape, which would clear the selection too
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      mareyOpen.value = false;
      e.preventDefault();
      e.stopImmediatePropagation();
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [open]);
  if (!l || !st || st.bus || l.stops.length < 2) return null;
  const others = display.value.chartOtherLines;
  const any = !!links && (links.shared.size > 0 || links.passes.length > 0);
  return (
    <div class="marey" role="dialog" aria-label={tr("mareyChart.label", { line: l.name })}>
      <div class="marey-head">
        <Chip line={l} plain />
        <span class="marey-name">{l.name}</span>
        {any && (
          <button type="button" class={"btn icon mini marey-others" + (others ? " on" : "")} aria-pressed={others} data-tip={tr(others ? "mareyChart.hideOthers" : "mareyChart.showOthers")}
            aria-label={tr(others ? "mareyChart.hideOthers" : "mareyChart.showOthers")} onClick={() => setDisplay("chartOtherLines", !others)}><Icon.related /></button>
        )}
        <button type="button" class="btn icon mini marey-close" data-tip={tr("mareyChart.close")} aria-label={tr("mareyChart.closeLabel")} onClick={() => (mareyOpen.value = false)}><Icon.cross /></button>
      </div>
      <MareyBody key={l.num} l={l} st={st} links={links} />
    </div>
  );
}

/** The schedule heading's button that opens and closes the chart. */
export function MareyButton({ st }: { st: LineStats }) {
  if (st.status !== "running" || st.bus) return null;
  const open = mareyOpen.value;
  return (
    <button class={"btn icon mini marey-open" + (open ? " on" : "")} aria-label={tr(open ? "mareyChart.hide" : "mareyChart.show")} aria-pressed={open}
      data-tip={tr(open ? "mareyChart.hide" : "mareyChart.show")} onClick={() => (mareyOpen.value = !open)}><Icon.marey /></button>
  );
}
