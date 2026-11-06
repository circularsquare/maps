// T-2603: a line's timetable chart (a Marey chart): time across, the line's stops down by their
// distance along it, each train's trip a line. The trips are the clock worker's (`TrackApi.line_trips`,
// the departures the map draws) over a window around now; each trip's curve is its run profile from
// the network buffers (`profileS`), so it bends where the train speeds up, slows and stands. Opened
// from the line panel's schedule heading; it follows the selected line. docs/ui-inspectors.md.

import { signal } from "@preact/signals";
import { useEffect, useMemo, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { clockTime } from "../../game/format";
import { drawnReversed } from "../../game/northUp";
import { selection } from "../../game/ui";
import { linesById, stationNames, stationsById, world } from "../../game/world";
import type { LineInput, LineStats } from "../../game/types";
import { live } from "../../map/live";
import type { NetworkBuffers } from "../../map/network";
import { useClockQuery } from "../useClockQuery";
import { Chip, Icon } from "../widgets";

/** Whether the chart is open; it shows the selected rail line, and closes when no line is selected. */
export const mareyOpen = signal(false);

const W_PX = 600, GUTTER = 92, PAD_R = 10, TOP = 8, AXIS = 22;
const PLOT_W = W_PX - GUTTER - PAD_R;
/** the window's share before now */
const BEFORE = 0.25;
const HOUR = 3600;

/** How many hours the chart spans: two round trips, three of its longest headways, at least 2
 * hours and at most 8, in whole hours. */
export function windowHours(l: LineInput, st: LineStats): number {
  const tph = Object.values(l.tph).filter((v) => v > 0);
  const headway = tph.length ? HOUR / Math.min(...tph) : 0;
  const s = Math.max(2 * (st.roundTripS || 0), 3 * headway, 2 * HOUR);
  return Math.min(8, Math.ceil(s / HOUR - 1e-6));
}

/** Tick spacing in minutes: the smallest giving at most 8 ticks over the window. */
const tickMinutes = (hours: number) => [10, 15, 30, 60, 120].find((m) => (hours * 60) / m <= 8) ?? 120;

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

function MareyBody({ l, st }: { l: LineInput; st: LineStats }) {
  const w = world.value!;
  const net = live.overlay?.renderer.net ?? null;
  const hours = windowHours(l, st);
  const span = hours * HOUR, q = span / 4;
  // fetched a quarter window either side of the window, again as now enters the next quarter
  const now = clock.minute.value * 60;
  const slot = Math.floor(now / q);
  const base = (slot - 1) * q;
  const answer = useClockQuery({ what: "lineTrips", line: l.num, t0: base, t1: base + 6 * q }, [l.num, w.version, base, span],
    (d) => ({ line: l.num, base, trips: Array.from({ length: d.length / 4 }, (_, i): Trip => ({ p: d[i * 4], dep: d[i * 4 + 1], end: d[i * 4 + 2] })) }));
  const [hover, setHover] = useState(-1);

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
  const local = (t: number) => clockTime(Math.floor((t + (st.shift ?? 0)) / 60));

  // the trips' curves, built once per answer: each profile's path in pixels from its departure
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
    const runs = answer.trips.filter((t) => paths.has(t.p)).map((t) => {
      const pr = paths.get(t.p)!, tripS = net.meta[t.p * 4 + 2], stops = st.stopS[pr.run];
      const first = pr.run === 0 ? l.stops[0] : l.stops[n - 1], last = pr.run === 0 ? l.stops[n - 1] : l.stops[0];
      const arrive = t.dep + reachTau(pr.pts, stops[stops.length - 1] ?? 0);
      const endY = yOf(toRun0(st.stopS, pr.run, pr.pts[pr.pts.length - 1] ?? 0));
      return {
        ...t, x: (t.dep - answer.base) * k, stand: t.end > t.dep + tripS + 1 ? [(t.dep + tripS - answer.base) * k, (t.end - answer.base) * k, endY] : null,
        tip: `Leaves ${names.get(first) ?? ""} ${local(t.dep)}\nReaches ${names.get(last) ?? ""} ${local(arrive)}`,
      };
    });
    (window as unknown as { __mareyMs?: number }).__mareyMs = performance.now() - t0;
    return { paths, runs };
  }, [answer, net, w.version, flip, H, k]);

  // the time ticks over the fetched span, on the line's local clock
  const fetchedBase = answer?.base ?? base;
  const step = tickMinutes(hours) * 60;
  const ticks: number[] = [];
  const shift = st.shift ?? 0;
  for (let t = Math.ceil((fetchedBase + shift) / step) * step - shift; t <= fetchedBase + 6 * q; t += step) ticks.push(t);
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
  const id = `marey-${l.num}`;
  const fit = (s: string) => (s.length > 14 ? s.slice(0, 13) + "…" : s);
  return (
    <svg class="marey-chart" width={W_PX} height={TOP + H + AXIS} style={{ width: W_PX + "px", height: TOP + H + AXIS + "px", "--run": l.colour }}>
      <defs>
        <clipPath id={`${id}-clip`}><rect x={0} y={0} width={PLOT_W} height={TOP + H + AXIS} /></clipPath>
        {drawn && [...drawn.paths].map(([p, pr]) => <path id={`${id}-p${p}`} d={pr.d} />)}
      </defs>
      {labels.map(({ i, y }) => (
        <text class="marey-stop" x={GUTTER - 8} y={y + 4}>{fit(names.get(l.stops[i]) ?? "")}</text>
      ))}
      <g transform={`translate(${GUTTER} 0)`}>
        {l.stops.map((_, i) => <line class="marey-grid" x1={0} x2={PLOT_W} y1={yOf(st.stopS[0][i] ?? 0)} y2={yOf(st.stopS[0][i] ?? 0)} />)}
        <g clip-path={`url(#${id}-clip)`}>
          <g transform={`translate(${offset.toFixed(1)} 0)`}>
            {ticks.map((t) => {
              const x = (t - fetchedBase) * k, shown = x + offset;
              return <>
                <line class="marey-tick" x1={x} x2={x} y1={TOP} y2={TOP + H} />
                {shown > 16 && shown < PLOT_W - 16 && <text class="marey-time" x={x} y={TOP + H + 15}>{local(t)}</text>}
              </>;
            })}
            {drawn?.runs.map((r) => r.stand && <line class="marey-stand" x1={r.stand[0]} x2={r.stand[1]} y1={r.stand[2]} y2={r.stand[2]} />)}
            {drawn?.runs.map((r, i) => <use class={i === hover ? "run on" : "run"} href={`#${id}-p${r.p}`} x={r.x} />)}
            {drawn?.runs.map((r, i) => <use class="run-hit" href={`#${id}-p${r.p}`} x={r.x} data-tip={r.tip} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(-1)} />)}
          </g>
          <rect class="marey-past" x={0} y={TOP - 2} width={nowX} height={H + 4} />
        </g>
        <line class="marey-now" x1={nowX} x2={nowX} y1={TOP - 4} y2={TOP + H + 4} />
      </g>
    </svg>
  );
}

/** The chart's panel beside the dock, for the selected rail line while `mareyOpen`. */
export function MareyPanel() {
  const open = mareyOpen.value;
  const sel = selection.value;
  const w = world.value;
  const l = open && w && sel?.kind === "line" ? linesById(w).get(sel.line) : undefined;
  const st = l && w ? w.lineStats[l.id] : undefined;
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
  return (
    <div class="marey" role="dialog" aria-label={`${l.name} timetable chart`}>
      <div class="marey-head">
        <Chip line={l} plain />
        <span class="marey-name">{l.name}</span>
        <button type="button" class="btn icon mini marey-close" data-tip="Close" aria-label="Close timetable chart" onClick={() => (mareyOpen.value = false)}><Icon.cross /></button>
      </div>
      <MareyBody l={l} st={st} />
    </div>
  );
}

/** The schedule heading's button that opens and closes the chart. */
export function MareyButton({ st }: { st: LineStats }) {
  if (st.status !== "running" || st.bus) return null;
  const open = mareyOpen.value;
  return (
    <button class={"btn icon mini marey-open" + (open ? " on" : "")} aria-label={open ? "Hide timetable chart" : "Show timetable chart"} aria-pressed={open}
      data-tip={open ? "Hide timetable chart" : "Show timetable chart"} onClick={() => (mareyOpen.value = !open)}><Icon.marey /></button>
  );
}
