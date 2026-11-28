// T-2603: a line's timetable chart (a Marey chart): time across, the line's stops down by their
// distance along it, each train's trip a line. The trips are the clock worker's (`TrackApi.line_trips`,
// the departures the map draws); each trip's curve is its run profile from the network buffers
// (`profileS`), so it bends where the train speeds up, slows and stands. Drawn under the schedule
// panel's timetable (TrainTable.tsx), on its time axis. docs/ui-inspectors.md.
// T-2660: other lines' trains on the stretches they share, and the flat junctions and crossings where
// other routes' trains cross its path (mareyShare.ts). T-3817: the trains are drawn on a canvas, each
// profile's path made once per answer in game seconds across and placed by the table's zoom and
// scroll, so a zoom frame is one redraw of the trains in view (an SVG of a day's trips took about 35-50 ms a zoom frame).

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { clockTime } from "../../game/format";
import { t as tr, tList } from "../../i18n";
import { drawnReversed } from "../../game/northUp";
import { nearestStation } from "../../game/places";
import { display, shade as uiShade } from "../../game/ui";
import { rgb } from "../../game/palette";
import { linesByNum, stationNames, stationsById, world } from "../../game/world";
import type { LineInput, LineStats } from "../../game/types";
import { live } from "../../map/live";
import type { NetworkBuffers } from "../../map/network";
import { queryClock } from "../../workers/clockClient";
import { useClockQuery } from "../useClockQuery";
import { clipPath, gradePasses, sharedStretches, type Pass, type Shared } from "./mareyShare";
import { profileRun } from "../../map/profiles";

const TOP = 8, AXIS = 22;
export const GUTTER = 92, PAD_R = 18;
const DEBUG = typeof location !== "undefined" && new URLSearchParams(location.search).has("debug");

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
function profilePoints(n: NetworkBuffers, p: number, interior = true): number[] {
  const first = n.meta[p * 4], count = n.meta[p * 4 + 1], tripS = n.meta[p * 4 + 2];
  const out: number[] = [];
  for (let i = first; i < first + count; i++) {
    const t = n.phases[i * 4], s = n.phases[i * 4 + 1], v = n.phases[i * 4 + 2], a = n.phases[i * 4 + 3];
    const t1 = i + 1 < first + count ? n.phases[(i + 1) * 4] : tripS;
    out.push(t, s);
    if (interior && a !== 0 && t1 > t) {
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
export interface Links { shared: Map<number, Shared[]>; passes: Pass[] }

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

/** A train's trip: its profile's path (`path`, x in trip seconds, y px), its departure `x` s from the
 * window's start and its length `len` s. `own`: another line drawn at full strength, a line of the
 * charted group (T-3346); `back`: the way the table doesn't show. */
interface Run { key: string; path: Path2D; coarse: Path2D; x: number; len: number; colour?: string; own?: boolean; back?: boolean; tip: string }

/** Below this many px a second a run is drawn from its phases' starts alone: a speed change then spans a
 * pixel or two, and a day of trips draws about a third of the points. */
const COARSE_K = 0.05;
/** Run `r`'s path at scale `k`. */
const pathOf = (r: Run, k: number) => (k < COARSE_K ? r.coarse : r.path);
/** a pass dot: `x` s from the window's start, `y` px */
interface Mark { x: number; y: number; colour: string; tip: string }

/** A line colour as drawn on the chart: on the dark panel lifted toward white, so navy and black lines stay visible. */
function inkOf(c: string, dark: boolean): string {
  if (!dark || !/^#[0-9a-f]{6}$/i.test(c)) return c;
  const [r, g, b] = rgb(c).map((v) => Math.round(v * 0.65 + 255 * 0.35));
  return `rgb(${r},${g},${b})`;
}

/** How a run is stroked: width px and opacity (style.css's old `.run` rules). */
function strokeOf(r: Run, lit = false): [number, number] {
  const full = !r.colour || r.own;
  if (lit) return [full ? 3.2 : 2.4, 1];
  if (r.back) return full ? [1.2, 0.3] : [1, 0.2];
  return full ? [1.6, 1] : [1, 0.55];
}

/** Run `r` placed: its path moved to its departure and stretched to `k` px a second, `x0` the window's start in px. */
const placedPath = (r: Run, k: number, x0: number) => {
  const out = new Path2D();
  out.addPath(pathOf(r, k), new DOMMatrix([k, 0, 0, 1, x0 + r.x * k, 0]));
  return out;
};

/** The chart of line `l`, `width` px wide, under a timetable (T-3469). `win`: the trips asked for, game s
 * (the table's day); `view`: the time at the plot's left edge, game s, and the scale, px a game second,
 * both the table's. `plotH` sets the height of its stops' span (else by its stop count), and
 * `group`'s lines are drawn where they share its track whether or not other lines are shown, at full
 * strength (T-3346). T-3475: `dir` the run of `l` the table shows, drawn in front, the other way dimmed
 * behind; `hot` the train hovered (line@profile@departure), lit over the rest faded, set by `onHot`. T-4555: `picked`
 * the train picked, drawn lit; a click on the chart gives `onPick` the train under it, or null. */
export function MareyBody({ l, st, links, width, plotH, group, win, view, dir, hot, onHot, picked = null, onPick }: {
  l: LineInput; st: LineStats; links: Links | null; width: number; plotH?: number; group?: ReadonlySet<number>;
  win: { t0: number; t1: number }; view: { t0: number; k: number }; dir?: number; hot: string | null; onHot: (k: string | null) => void;
  picked?: string | null; onPick?: (k: string | null) => void;
}) {
  const r0 = DEBUG ? performance.now() : 0;
  const w = world.value!;
  const net = live.overlay?.renderer.net ?? null;
  const PLOT_W = width - GUTTER - PAD_R;
  const { k } = view;
  const span = PLOT_W / k;
  const now = clock.minute.value * 60;
  const base = win.t0;
  const answer = useClockQuery({ what: "lineTrips", line: l.num, t0: win.t0, t1: win.t1 }, [l.num, w.version, win.t0, win.t1],
    (d) => ({ line: l.num, base, trips: tripsOf(d) }));
  const showOthers = display.value.chartOtherLines;
  const otherKey = links ? [...new Set([...links.shared.keys(), ...links.passes.map((p) => p.o)])].filter((o) => showOthers || group?.has(o)).join(",") : "";
  const otherNums = useMemo(() => (otherKey ? otherKey.split(",").map(Number) : []), [otherKey]);
  const [otherAnswer, setOtherAnswer] = useState<{ base: number; byLine: Map<number, Trip[]> } | null>(null);
  useEffect(() => {
    if (!otherNums.length) { setOtherAnswer(null); return; }
    let on = true;
    void Promise.all(otherNums.map((o) => queryClock({ what: "lineTrips", line: o, t0: win.t0, t1: win.t1 }))).then((ds) => {
      if (!on) return;
      const byLine = new Map<number, Trip[]>();
      ds.forEach((d, i) => { if (d) byLine.set(otherNums[i], tripsOf(d)); });
      setOtherAnswer({ base, byLine });
    });
    return () => { on = false; };
  }, [otherNums, w.version, win.t0, win.t1]);
  const flats = useClockQuery({ what: "lineMarks", line: l.num }, [l.num, w.version], (d) => flatsFrom(d, st.stopS[0])) ?? [];

  const names = stationNames(w), byId = stationsById(w);
  const n = l.stops.length;
  const flip = drawnReversed(byId.get(l.stops[0]), byId.get(l.stops[n - 1]), !!st.loop);
  const d0 = st.stopS[0][0] ?? 0, d1 = st.stopS[0][n - 1] ?? 1;
  const H = plotH ?? Math.max(120, Math.min(340, n * 20));
  const yOf = (d: number) => {
    const f = d1 > d0 ? (d - d0) / (d1 - d0) : 0;
    return TOP + (flip ? 1 - f : f) * H;
  };
  const shift = st.shift ?? 0;
  const local = (t: number) => clockTime(Math.floor((t + shift) / 60));

  // the line's own trips, built once per answer: each profile's path from its departure, x in trip seconds
  const drawn = useMemo(() => {
    const t0 = performance.now();
    if (!answer || answer.line !== l.num || !net) return null;
    const paths = new Map<number, { path: Path2D; coarse: Path2D; pts: number[]; run: number }>();
    const pathFrom = (pts: number[], run: number) => {
      let d = "";
      for (let i = 0; i < pts.length; i += 2) d += `${i ? "L" : "M"}${pts[i].toFixed(1)} ${yOf(toRun0(st.stopS, run, pts[i + 1])).toFixed(1)}`;
      return new Path2D(d);
    };
    for (const t of answer.trips) {
      if (paths.has(t.p) || !net.meta[t.p * 4 + 1]) continue;
      const run = profileRun(t.p), pts = profilePoints(net, t.p);
      paths.set(t.p, { path: pathFrom(pts, run), coarse: pathFrom(profilePoints(net, t.p, false), run), pts, run });
    }
    const stands: number[][] = [];
    const runs: Run[] = answer.trips.filter((t) => paths.has(t.p)).map((t) => {
      const pr = paths.get(t.p)!, tripS = net.meta[t.p * 4 + 2], stops = st.stopS[pr.run];
      const first = pr.run === 0 ? l.stops[0] : l.stops[n - 1], last = pr.run === 0 ? l.stops[n - 1] : l.stops[0];
      const arrive = t.dep + reachTau(pr.pts, stops[stops.length - 1] ?? 0);
      if (t.end > t.dep + tripS + 1) stands.push([t.dep + tripS - answer.base, t.end - answer.base, yOf(toRun0(st.stopS, pr.run, pr.pts[pr.pts.length - 1] ?? 0))]);
      return {
        key: `${l.num}@${t.p}@${t.dep}`, path: pr.path, coarse: pr.coarse, x: t.dep - answer.base, len: tripS, back: dir !== undefined && pr.run !== dir,
        tip: tr("mareyChart.tip", { from: names.get(first) ?? "", leave: local(t.dep), to: names.get(last) ?? "", arrive: local(arrive) }),
      };
    });
    if (DEBUG) (window as unknown as { __mareyMs?: number }).__mareyMs = performance.now() - t0;
    return { stands, runs };
  }, [answer, net, w.version, flip, H, dir]);

  // the other lines' trains on the stretches they share, and their passes over at-grade conflicts
  const otherDrawn = useMemo(() => {
    if (!otherAnswer || !answer || otherAnswer.base !== answer.base || !net || !links) return null;
    const byNum = linesByNum(w);
    const runs: Run[] = [], marks: Mark[] = [];
    for (const [o, trips] of otherAnswer.byLine) {
      const ol = byNum.get(o), ost = ol && w.lineStats[ol.id];
      if (!ol || !ost || ost.bus) continue;
      const shared = links.shared.get(o) ?? [], passes = links.passes.filter((p) => p.o === o);
      const pts = new Map<number, { pts: number[]; path: Path2D | null; coarse: Path2D | null; run: number; len: number }>();
      for (const t of trips) {
        if (pts.has(t.p) || !net.meta[t.p * 4 + 1]) continue;
        const run = profileRun(t.p), len = net.meta[t.p * 4 + 3], raw = profilePoints(net, t.p);
        const clipped = (ps: number[]) => {
          let d = "";
          const s0 = run === 0 ? ps : ps.map((v, i) => (i % 2 ? len - v : v));
          for (const sh of shared) d += clipPath(s0, sh.a, sh.b, (t) => t, (s) => yOf(sh.c + (sh.m ?? sh.sg) * (s - sh.a)));
          return d ? new Path2D(d) : null;
        };
        const path = shared.length ? clipped(raw) : null;
        pts.set(t.p, { pts: raw, path, coarse: path && clipped(profilePoints(net, t.p, false)), run, len });
      }
      const m = ol.stops.length;
      // the run of `l` that this line's run 0 goes the way of, by most of the track they share
      const way = shared.reduce((s, x) => s + x.sg * (x.b - x.a), 0) < 0 ? 1 : 0;
      trips.forEach((t) => {
        const pr = pts.get(t.p);
        if (!pr) return;
        if (pr.path) {
          const stops = ost.stopS[pr.run];
          const first = pr.run === 0 ? ol.stops[0] : ol.stops[m - 1], last = pr.run === 0 ? ol.stops[m - 1] : ol.stops[0];
          const arrive = t.dep + reachTau(pr.pts, stops[stops.length - 1] ?? 0);
          runs.push({
            key: `${o}@${t.p}@${t.dep}`, path: pr.path, coarse: pr.coarse ?? pr.path, x: t.dep - answer.base, len: net.meta[t.p * 4 + 2], colour: ol.colour, own: group?.has(o), back: dir !== undefined && (pr.run ^ way) !== dir,
            tip: tr("mareyChart.otherTip", { line: ol.name, from: names.get(first) ?? "", leave: local(t.dep), to: names.get(last) ?? "", arrive: local(arrive) }),
          });
        }
        const reach = pr.pts[pr.pts.length - 1] ?? 0;
        for (const ps of passes) {
          if (ps.run >= 0 && ps.run !== pr.run) continue;
          const s = pr.run === 0 ? ps.sO : pr.len - ps.sO;
          if (s < -1 || s > reach + 1) continue;
          const at = t.dep + reachTau(pr.pts, s);
          marks.push({ x: at - answer.base, y: yOf(ps.sL), colour: ol.colour, tip: tr("mareyChart.passTip", { line: ol.name, time: local(at) }) });
        }
      });
    }
    return { runs, marks };
  }, [otherAnswer, answer, net, links, w.version, flip, H, group, dir]);

  const height = TOP + H + AXIS;
  const dark = uiShade.value === "dark";
  const canvas = useRef<HTMLCanvasElement>(null);
  // read once per size and shade, not every frame: each read would work out the page's style again
  const zoomFor = useRef({ w: 0, z: 1 });
  const bg = useMemo(() => getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() || "#fff", [dark]);
  const runOf = (key: string | null) => (key === null ? null : (drawn?.runs.find((r) => r.key === key) ?? otherDrawn?.runs.find((r) => r.key === key) ?? null));
  const lit = runOf(hot), sel = runOf(picked);
  const view0 = view.t0;
  // px from the plot's left of the window's start
  const x0 = -(view0 - base) * k;
  // the runs in view, in the order drawn: stands under all, then the way not shown, then the way shown
  const inView = (r: Run) => x0 + (r.x + r.len) * k >= -8 && x0 + r.x * k <= PLOT_W + 8;
  const order = (): Run[] => {
    const others = otherDrawn?.runs ?? [], runs = drawn?.runs ?? [];
    return [...others.filter((r) => r.back), ...runs.filter((r) => r.back), ...others.filter((r) => !r.back), ...runs.filter((r) => !r.back)].filter(inView);
  };
  useLayoutEffect(() => {
    const cv = canvas.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx) return;
    const d0 = DEBUG ? performance.now() : 0;
    if (zoomFor.current.w !== PLOT_W) zoomFor.current = { w: PLOT_W, z: ((cv as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1) * devicePixelRatio };
    const z = zoomFor.current.z;
    const cw = Math.round(PLOT_W * z), ch = Math.round(height * z);
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.setTransform(z, 0, 0, z, 0, 0);
    const fade = lit ? 0.35 : 1;
    // runs of one look stroked together
    const stroke = (rs: Run[]) => {
      const batches = new Map<string, { path: Path2D; w: number; a: number; c: string }>();
      for (const r of rs) {
        const [wd, a] = strokeOf(r), c = inkOf(r.colour ?? l.colour, dark), key = `${c}|${wd}|${a}`;
        let bt = batches.get(key);
        if (!bt) batches.set(key, (bt = { path: new Path2D(), w: wd, a, c }));
        bt.path.addPath(pathOf(r, k), new DOMMatrix([k, 0, 0, 1, x0 + r.x * k, 0]));
      }
      for (const bt of batches.values()) {
        ctx.globalAlpha = bt.a * fade;
        ctx.strokeStyle = bt.c;
        ctx.lineWidth = bt.w;
        ctx.stroke(bt.path);
      }
    };
    ctx.lineJoin = "round";
    // a train standing at its terminal for its next departure: dashed
    ctx.globalAlpha = fade;
    ctx.setLineDash([2, 2]);
    ctx.strokeStyle = inkOf(l.colour, dark);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [a, b, y] of drawn?.stands ?? []) {
      const xa = x0 + a * k, xb = x0 + b * k;
      if (xb < 0 || xa > PLOT_W) continue;
      ctx.moveTo(xa, y + 0.5);
      ctx.lineTo(xb, y + 0.5);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    const all = order(), front = all.filter((r) => !r.back), back = all.filter((r) => r.back);
    stroke(back);
    stroke(front.filter((r) => r.colour));
    for (const m of otherDrawn?.marks ?? []) {
      const x = x0 + m.x * k;
      if (x < -4 || x > PLOT_W + 4) continue;
      ctx.globalAlpha = fade;
      ctx.beginPath();
      ctx.arc(x, m.y, 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = inkOf(m.colour, dark);
      ctx.fill();
      ctx.strokeStyle = bg;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    stroke(front.filter((r) => !r.colour));
    for (const r of [sel, lit]) {
      if (!r) continue;
      const [wd] = strokeOf(r, true);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = inkOf(r.colour ?? l.colour, dark);
      ctx.lineWidth = wd;
      ctx.stroke(placedPath(r, k, x0));
    }
    ctx.globalAlpha = 1;
    if (DEBUG) ((window as unknown as { __mareyDraw?: number[] }).__mareyDraw ??= []).push(performance.now() - d0);
    if (DEBUG) (window as unknown as { __mareyRuns?: unknown }).__mareyRuns = { own: all.filter((r) => !r.colour).length, group: all.filter((r) => r.colour && r.own).length, other: all.filter((r) => r.colour && !r.own).length };
  });
  // the train or pass dot under the pointer: the one drawn last there. The canvas always has a data-tip, so
  // the tag (tip.ts) watches it from the first move and shows a change as it is made
  const hitAt = (e: MouseEvent): { key: string | null; tip: string | null } => {
    const cv = canvas.current, ctx = cv?.getContext("2d");
    if (!cv || !ctx) return { key: null, tip: null };
    const z = (cv as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1, r = cv.getBoundingClientRect();
    const px = (e.clientX - r.left) / z, py = (e.clientY - r.top) / z;
    let key: string | null = null, tip: string | null = null;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.lineWidth = 9;
    const all = order();
    const hit = (rs: Run[]) => {
      for (let i = rs.length - 1; i >= 0; i--) {
        const run = rs[i], xa = x0 + run.x * k;
        if (px >= xa - 5 && px <= xa + run.len * k + 5 && ctx.isPointInStroke(placedPath(run, k, x0), px, py)) return run;
      }
      return null;
    };
    // as drawn: this line's way shown on top, then the pass dots, then the rest
    let run = hit(all.filter((r) => !r.back && !r.colour));
    if (!run) tip = passAt(px, py);
    if (!run && !tip) run = hit(all.filter((r) => r.back || r.colour));
    if (run) { key = run.key; tip = run.tip; }
    ctx.restore();
    return { key, tip };
  };
  const onMove = (e: PointerEvent) => {
    const cv = canvas.current;
    if (!cv || e.buttons) return;
    const { key, tip } = hitAt(e);
    cv.setAttribute("data-tip", tip ?? "");
    if (onPick) cv.style.cursor = key === null ? "" : "pointer";
    if (key !== hot) onHot(key);
  };
  const onClick = (e: MouseEvent) => onPick?.(hitAt(e).key);
  const passAt = (px: number, py: number): string | null => {
    for (const m of otherDrawn?.marks ?? []) if (Math.hypot(x0 + m.x * k - px, m.y - py) <= 4.5) return m.tip;
    return null;
  };
  const onLeave = () => { canvas.current?.setAttribute("data-tip", ""); if (hot !== null) onHot(null); };
  const nowAt = (now - view0) * k, nowX = Math.max(0, Math.min(PLOT_W, nowAt));
  // the time ticks over the view, on the line's local clock
  const step = tickMinutes(span) * 60;
  const ticks: number[] = [];
  for (let t = Math.ceil((view0 + shift) / step) * step - shift; t <= view0 + span; t += step) ticks.push(t);
  // labels down the side, each kept only where it clears the last one shown (the ends always)
  const byY = l.stops.map((_, i) => i).sort((a, b) => yOf(st.stopS[0][a] ?? 0) - yOf(st.stopS[0][b] ?? 0));
  const labels: { i: number; y: number }[] = [];
  for (const i of byY) {
    const y = yOf(st.stopS[0][i] ?? 0);
    const end = i === 0 || i === n - 1;
    if (labels.length && y - labels[labels.length - 1].y < 12) {
      if (!end) continue;
      if (labels.length > 1 || !(labels[0].i === 0 || labels[0].i === n - 1)) labels.pop();
    }
    labels.push({ i, y });
  }
  const fit = (s: string) => (s.length > 14 ? s.slice(0, 13) + "…" : s);
  useLayoutEffect(() => {
    if (!DEBUG) return;
    const g = window as unknown as { __mareyRender?: number[] };
    (g.__mareyRender ??= []).push(performance.now() - r0);
  });
  return (
    <div class="marey-chart" style={{ width: width + "px", height: height + "px" }}>
      <svg class="marey-base" width={width} height={height}>
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
        <svg class="marey-ticks" width={PLOT_W} height={height}>
          {ticks.map((t) => <line class="marey-tick" x1={(t - view0) * k} x2={(t - view0) * k} y1={TOP} y2={height - AXIS} />)}
        </svg>
        <canvas class="marey-layer" ref={canvas} data-tip="" style={{ width: PLOT_W + "px", height: height + "px" }} onPointerMove={onMove} onPointerLeave={onLeave} onClick={onPick ? onClick : undefined} />
        <svg class="marey-over" width={PLOT_W} height={height}>
          <rect class="marey-past" x={0} y={TOP - 2} width={nowX} height={H + 4} />
          {ticks.map((t) => {
            const x = (t - view0) * k;
            return x > 16 && x < PLOT_W - 16 && <text class="marey-time" x={x} y={TOP + H + 15}>{local(t)}</text>;
          })}
          {nowAt === nowX && <line class="marey-now" x1={nowX} x2={nowX} y1={TOP - 4} y2={TOP + H + 4} />}
        </svg>
      </div>
    </div>
  );
}

/** Where other lines meet line `l`'s chart: their stretches on its track and their flat crossings of
 * its path, by the network build and the line's ends only (the stats object is new on every state
 * message). */
export function useChartLinks(l: LineInput | undefined, st: LineStats | undefined): Links | null {
  const w = world.value;
  const net = live.overlay?.renderer.net ?? null;
  const n = l?.stops.length ?? 0;
  const d0 = st?.stopS[0][0] ?? 0, d1 = st?.stopS[0][n - 1] ?? 0;
  return useMemo<Links | null>(() => {
    if (!net || !l || n < 2) return null;
    return { shared: sharedStretches(net, l.num, d0, d1), passes: gradePasses(net, l.num, d0, d1) };
  }, [net, l?.num, w?.version, d0, d1]);
}
