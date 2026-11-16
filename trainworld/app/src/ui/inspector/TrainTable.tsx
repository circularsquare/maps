// T-3469: the schedule panel's timetable, laid out as a printed one (a Tokaido Shinkansen timetable
// from Anita's board, 2026-11-16): a column per train under its line's bullet, the stations down the
// side, small times in tight cells, one direction at a time over the day from the last hour. Each column stands
// over its train's departure on the time scale of the chart below, which scrolls with the table, so a
// column sits right above where its train's path starts. The times are the profiles the chart draws
// (as the departures board's check reads them). notes/T-3346.md "Stage 1 as built".

import { signal, type Signal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { drawnReversed } from "../../game/northUp";
import { inkOn, legibleOn } from "../../game/palette";
import { shade as uiShade } from "../../game/ui";
import type { LineInput, LineStats } from "../../game/types";
import { linesById, stationNames, stationsById, world } from "../../game/world";
import { t } from "../../i18n";
import { live } from "../../map/live";
import type { NetworkBuffers } from "../../map/network";
import { queryClock } from "../../workers/clockClient";
import { Chip, Segmented } from "../widgets";
import { GUTTER, MareyBody, PAD_R, type Links } from "./MareyChart";

const DAY = 86400, HOUR = 3600;
/** a column's width, a row's height and the head's height, px */
const COL = 27, ROW = 13, MARK_H = 14;
/** a strip head's letters at most, and the height of each, px */
const STRIP_MAX = 4, STRIP_CH = 11;
/** the furthest a column may stand from its train's time, px: so far, its edge still meets it */
const PUSH = COL;
/** the time a column's width stands for, s, kept between these */
const GAP_MIN = 20, GAP_MAX = 1800;
/** a cell's value for a train that runs through the station without stopping */
const PASS = -1;
/** the time marks' steps, minutes */
const MARK_MIN = [5, 10, 15, 30, 60, 120, 180];
/** a mark's label, local, from the local hour `h0` the table starts at and s on from it: "8:00", "8:30" */
const markText = (h0: number, s: number) => `${(h0 + Math.floor(s / HOUR)) % 24}:${String(Math.floor((s % HOUR) / 60)).padStart(2, "0")}`;

const DEBUG = typeof location !== "undefined" && new URLSearchParams(location.search).has("debug");

/** The run of profile `p`. Stage 2 (T-3346) makes this `profileRun`. */
const runOf = (p: number) => Math.floor(p / 3) % 2;

/** Seconds into a trip at which profile `p` leaves each of its stops at offsets `offs`: as it
 * first moves on after its stand there (at once from the first stop); the last stop's is its
 * arrival. NaN where the profile never gets there. */
function stopTimes(net: NetworkBuffers, p: number, offs: number[]): Float64Array {
  const first = net.meta[p * 4], count = net.meta[p * 4 + 1], tripS = net.meta[p * 4 + 2];
  const out = new Float64Array(offs.length).fill(NaN);
  let k = 0, at = true, arrived = 0;
  for (let q = first; q < first + count && k < offs.length; q++) {
    const tq = net.phases[q * 4], s = net.phases[q * 4 + 1], v = net.phases[q * 4 + 2], a = net.phases[q * 4 + 3];
    const rest = Math.abs(v) < 0.01 && a <= 0;
    if (at) {
      if (!rest) { out[k++] = tq; at = false; }
      continue;
    }
    // a stop run through without a stand: the phase's start, near enough for a minute
    while (k < offs.length - 1 && s > offs[k] + 1.5) out[k++] = tq;
    if (rest && Math.abs(s - offs[k]) < 1.5) { at = true; arrived = tq; }
  }
  if (k === offs.length - 1) out[k] = at ? arrived : tripS;
  return out;
}

/** A line whose trains get columns: its bullet's line (a T-3333 twin shows its parent's), each of
 * its stops' row on the axis line (run 0 order, -1 off it), and whether its run 0 goes the axis's
 * run 1 way. */
interface Src { line: LineInput; bullet: LineInput; st: LineStats; rowOf: number[]; flipped: boolean }

/** The columns' lines: the axis line, then each other line by the stations it shares with the
 * axis (two at least, which set its direction), each with its hidden quad twin. */
function sourcesOf(axis: LineInput, others: readonly LineInput[]): Src[] {
  const w = world.value!;
  const byId = linesById(w);
  const index = new Map<string, number>();
  axis.stops.forEach((s, i) => { if (!index.has(s)) index.set(s, i); });
  const out: Src[] = [];
  const add = (l: LineInput, rowOf: number[], flipped: boolean) => {
    const st = w.lineStats[l.id];
    if (!st || st.bus) return;
    out.push({ line: l, bullet: l, st, rowOf, flipped });
    const twin = st.twin ? byId.get(st.twin) : undefined;
    const tst = twin ? w.lineStats[twin.id] : undefined;
    if (twin && tst && twin.stops.length === l.stops.length) out.push({ line: twin, bullet: l, st: tst, rowOf, flipped });
  };
  add(axis, axis.stops.map((_, i) => i), false);
  for (const o of others) {
    const rowOf = o.stops.map((s) => index.get(s) ?? -1);
    const on = rowOf.filter((r) => r >= 0);
    if (new Set(on).size < 2) continue;
    add(o, rowOf, on[on.length - 1] < on[0]);
  }
  return out;
}

/** A train's name shared with the chart (MareyBody's `hot`): its line, profile and departure. */
const trainKey = (line: number, p: number, dep: number) => `${line}@${p}@${dep}`;

/** One train's column: its bullet, its time at the first row it has, its cells by row shown (NaN
 * blank, PASS run through). */
interface Col { bullet: LineInput; t: number; cells: Float64Array; key: string }

/** The day's trains of `srcs` from `day0`, as columns per direction (0 the axis's run 0 way, rows
 * in that order), each sorted by time. */
function useColumns(srcs: Src[], rows: number, day0: number): [Col[], Col[]] | null {
  const w = world.value!;
  const net = live.overlay?.renderer.net ?? null;
  const key = srcs.map((s) => s.line.num).join(",");
  const [answer, setAnswer] = useState<{ key: string; day0: number; trips: (Float64Array | null)[] } | null>(null);
  useEffect(() => {
    let on = true;
    // from two hours before, for trains that set off then and reach the axis after the day starts
    void Promise.all(srcs.map((s) => queryClock({ what: "lineTrips", line: s.line.num, t0: day0 - 2 * HOUR, t1: day0 + DAY }))).then((trips) => {
      if (on && trips.every((d) => d)) setAnswer({ key, day0, trips });
    });
    return () => { on = false; };
  }, [key, w.version, day0]);
  return useMemo(() => {
    if (!answer || answer.key !== key || !net) return null;
    const out: [Col[], Col[]] = [[], []];
    srcs.forEach((src, si) => {
      const d = answer.trips[si];
      if (!d) return;
      const n = src.line.stops.length;
      const times = new Map<number, Float64Array>();
      for (let i = 0; i + 3 < d.length; i += 4) {
        const p = d[i], dep = d[i + 1];
        if (!net.meta[p * 4 + 1]) continue;
        const run = runOf(p);
        let tt = times.get(p);
        if (!tt) times.set(p, (tt = stopTimes(net, p, src.st.stopS[run] ?? [])));
        const dir = (run === 1) !== src.flipped ? 1 : 0;
        const cells = new Float64Array(rows).fill(NaN);
        let lo = rows, hi = -1;
        for (let k = 0; k < tt.length; k++) {
          const r = src.rowOf[run === 0 ? k : n - 1 - k];
          if (r === undefined || r < 0 || !Number.isFinite(tt[k])) continue;
          const row = dir === 0 ? r : rows - 1 - r;
          if (Number.isFinite(cells[row])) continue;
          cells[row] = dep + tt[k];
          lo = Math.min(lo, row); hi = Math.max(hi, row);
        }
        if (hi < 0) continue;
        for (let r = lo + 1; r < hi; r++) if (!Number.isFinite(cells[r])) cells[r] = PASS;
        const t0 = cells[lo];
        if (t0 < day0 || t0 >= day0 + DAY) continue;
        out[dir].push({ bullet: src.bullet, t: t0, cells, key: trainKey(src.line.num, p, dep) });
      }
    });
    for (const cs of out) cs.sort((a, b) => a.t - b.t);
    return out;
  }, [answer, net, rows]);
}

/** How far columns are pushed from their times at `scale`, px: the 95th percentile, since trains
 * leaving within seconds of each other push one a column whatever the scale. */
function drift(cols: Col[], scale: number): number {
  let last = -Infinity;
  const push = cols.map((c) => {
    const x = c.t * scale;
    last = Math.max(x, last + COL);
    return last - x;
  });
  return push.sort((a, b) => a - b)[Math.floor(push.length * 0.95)] ?? 0;
}

/** px per game second: the most compact scale at which columns are pushed no more than their width
 * from their times (but a few), so the busiest hours pack tight and each column still stands over
 * its train. */
function scaleOf(cols: Col[]): number {
  let lo = GAP_MIN, hi = GAP_MAX;
  if (drift(cols, COL / hi) <= PUSH) return COL / hi;
  for (let i = 0; i < 24; i++) {
    const g = Math.sqrt(lo * hi);
    if (drift(cols, COL / g) <= PUSH) lo = g;
    else hi = g;
  }
  return COL / lo;
}

/** Each column's centre, px from the day's start: at its time, pushed right where it would overlap
 * the one before. */
function placed(cols: Col[], scale: number, day0: number): Float64Array {
  const xs = new Float64Array(cols.length);
  let last = -Infinity;
  cols.forEach((c, i) => { last = xs[i] = Math.max((c.t - day0) * scale, last + COL); });
  return xs;
}

/** First index with xs[i] >= x. */
function lowerBound(xs: Float64Array, x: number): number {
  let a = 0, b = xs.length;
  while (a < b) { const m = (a + b) >> 1; if (xs[m] < x) a = m + 1; else b = m; }
  return a;
}

/** The width of `ref`'s element's content box, px, kept as it changes. */
function useClientWidth(ref: { current: HTMLElement | null }): number {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return w;
}

/** Whether line `l`'s bullet fits a column: one wide letter or two narrow. */
const fits = (l: LineInput) => [...l.letter].reduce((s, ch) => s + ((ch.codePointAt(0) ?? 0) >= 0x2e80 ? 12 : 7), 8) <= COL - 2;

/** A column's head: the line's bullet, or where that is too wide its letters down a strip of its
 * colour, as printed timetables head a train with its service's name. */
function Bullet({ l }: { l: LineInput }) {
  if (fits(l)) return <Chip line={l} size="sm" plain />;
  return <span class="tt-strip" style={{ "--c": l.colour, "--ct": inkOn(l.colour) }}>{[...l.letter].slice(0, STRIP_MAX).map((ch) => <span>{ch}</span>)}</span>;
}

/** A train's column of times, its centre `x` px from the day's start. */
function Column({ c, x, cls, shift, dark }: { c: Col; x: number; cls: string; shift: number; dark: boolean }) {
  return (
    <div class={"tt-col" + cls} data-k={c.key} style={{ left: GUTTER + x - COL / 2 + "px", width: COL + "px", "--ink": legibleOn(c.bullet.colour, dark) }}>
      {Array.from(c.cells, (v) => <span class={v === PASS ? "pass" : undefined}>{v === PASS ? "|" : Number.isFinite(v) ? printed(v, shift) : ""}</span>)}
    </div>
  );
}

/** The hovered train, over a veil that fades the rest: its bullet (`part` "head") or its times. One
 * element's change per hover, whatever the number of columns. */
function Lit({ part, hot, at, cols, xs, shift, dark, now }: {
  part: "head" | "body"; hot: Signal<string | null>; at: Map<string, number>; cols: Col[]; xs: Float64Array; shift: number; dark: boolean; now: number;
}) {
  const i = hot.value === null ? undefined : at.get(hot.value);
  if (i === undefined) return null;
  const c = cols[i];
  return (
    <>
      <div class="tt-veil" />
      {part === "head"
        ? <span class="tt-bullet on" style={{ left: GUTTER + xs[i] - COL / 2 + "px", width: COL + "px", top: MARK_H + "px" }}><Bullet l={c.bullet} /></span>
        : <Column c={c} x={xs[i]} cls={" on" + (c.t < now ? " past" : "")} shift={shift} dark={dark} />}
    </>
  );
}

/** A time as printed in the table, local: "742", "1008". */
function printed(t: number, shift: number): string {
  const m = ((Math.floor((t + shift) / 60) % 1440) + 1440) % 1440;
  return String(Math.floor(m / 60)) + String(m % 60).padStart(2, "0");
}

/** The chart under the table, the only part of the table's tree that redraws on a hover. */
function Chart({ hot, ...rest }: { hot: Signal<string | null> } & Omit<Parameters<typeof MareyBody>[0], "hot" | "onHot">) {
  return <MareyBody {...rest} hot={hot.value} onHot={(k) => { hot.value = k; }} />;
}

/** The timetable of `axis` (its stops as rows) with `others`' trains where they run on its
 * stations, and the chart of `axis` under it. `tools` sit at the end of its direction choice. */
export function TrainTable({ axis, st, others = [], links, group, plotH, tools }: {
  axis: LineInput; st: LineStats; others?: readonly LineInput[]; links: Links | null; group?: ReadonlySet<number>; plotH: number; tools?: ComponentChildren;
}) {
  const w = world.value!;
  const names = stationNames(w), byId = stationsById(w);
  const n = axis.stops.length;
  const shift = st.shift ?? 0;
  const dark = uiShade.value === "dark";
  const now = clock.minute.value * 60;
  // from the hour before this one, a day on: what is coming, with the last hour for context
  const day0 = Math.floor((now + shift) / HOUR) * HOUR - shift - HOUR;
  const otherKey = others.map((o) => o.num).join(",");
  const srcs = useMemo(() => sourcesOf(axis, others), [axis.num, otherKey, w.version]);
  const both = useColumns(srcs, n, day0);
  // the chart's top-to-bottom way first, so the rows read in its order
  const down = drawnReversed(byId.get(axis.stops[0]), byId.get(axis.stops[n - 1]), !!st.loop) ? 1 : 0;
  const [pick, setPick] = useState<number | null>(null);
  const dir = pick ?? down;
  const cols = both?.[dir] ?? [];
  const scale = useMemo(() => scaleOf(cols), [cols]);
  const xs = useMemo(() => placed(cols, scale, day0), [cols, scale, day0]);
  // the train under the pointer, here or on the chart: read only by the parts that draw it, so a
  // hover redraws neither the table nor the chart's trains
  const hot = useMemo(() => signal<string | null>(null), []);
  const colAt = useMemo(() => new Map(cols.map((c, i) => [c.key, i])), [cols]);
  const onOver = (e: PointerEvent) => {
    const k = (e.target as Element).closest?.("[data-k]")?.getAttribute("data-k") ?? null;
    if (hot.value !== k) hot.value = k;
  };
  const onOut = () => { hot.value = null; };
  if (DEBUG) (window as unknown as { __tt?: unknown }).__tt = { cols: cols.length, hourPx: scale * HOUR, drift: drift(cols, scale), h0: ((Math.round((day0 + shift) / HOUR) % 24) + 24) % 24 };
  const width = Math.max(DAY * scale, cols.length ? xs[xs.length - 1] + COL : 0) + COL;

  const scroller = useRef<HTMLDivElement>(null);
  const vw = useClientWidth(scroller);
  const [left, setLeft] = useState(0);
  const frame = useRef(0);
  // the time at the view's left edge, kept as the table's start or scale moves under it
  const leftT = useRef(0);
  const onScroll = () => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const x = scroller.current?.scrollLeft ?? 0;
      leftT.current = day0 + x / scale;
      setLeft(x);
    });
  };
  // a new line or direction: the time now a quarter of the way in
  const placedFor = useRef("");
  useLayoutEffect(() => {
    const el = scroller.current, k = `${axis.num}:${dir}:${otherKey}`;
    if (!el || !both || !vw) return;
    if (placedFor.current !== k) {
      placedFor.current = k;
      leftT.current = now - (0.25 * (vw - GUTTER)) / scale;
    }
    const x = Math.max(0, (leftT.current - day0) * scale);
    if (Math.abs(el.scrollLeft - x) > 1) el.scrollLeft = x;
    setLeft(el.scrollLeft);
  }, [both, dir, vw, scale, day0]);

  const rowNames = Array.from({ length: n }, (_, i) => names.get(axis.stops[dir === 0 ? i : n - 1 - i]) ?? "");
  const label = (d: number) => st.loop
    ? t("schedulePanel.via", { station: names.get(axis.stops[d === 0 ? 1 : n - 2]) ?? "" })
    : t("schedulePanel.to", { station: names.get(axis.stops[d === 0 ? n - 1 : 0]) ?? "" });
  // only the columns near the view are drawn
  const plotW = Math.max(0, vw - GUTTER);
  const a = lowerBound(xs, left - plotW - COL), b = lowerBound(xs, left + 2 * plotW + COL);
  const shown = cols.slice(a, b);
  // time marks from the day's start, as far apart as a label needs
  const h0 = (((Math.round((day0 + shift) / HOUR)) % 24) + 24) % 24;
  const step = (MARK_MIN.find((m) => m * 60 * scale >= 52) ?? 240) * 60;
  const marks: number[] = [];
  for (let s = Math.ceil(Math.max(0, left / scale - step) / step) * step; s <= DAY && s * scale < left + plotW + 60; s += step) marks.push(s);
  const onWheel = (e: WheelEvent) => {
    const el = scroller.current, d = e.deltaX || e.deltaY;
    if (!el || !d) return;
    e.preventDefault();
    el.scrollLeft += d;
  };
  const rowsH = n * ROW;
  const tallest = Math.max(1, ...srcs.map((s) => (fits(s.bullet) ? 1 : Math.min(STRIP_MAX, [...s.bullet.letter].length))));
  const HEAD = MARK_H + (tallest > 1 ? tallest * STRIP_CH + 6 : 20);
  return (
    <div class="tt">
      <div class="sp-tabs">
        <Segmented<number> options={[[down, label(down)], [1 - down, label(1 - down)]]} value={dir} onChange={setPick} />
        {tools}
      </div>
      <div class="tt-scroll" ref={scroller} onScroll={onScroll} style={{ marginRight: PAD_R + "px", maxHeight: `min(${rowsH + HEAD + 14}px, 44vh)` }}>
        <div class="tt-inner" style={{ width: GUTTER + width + "px", height: HEAD + rowsH + "px" }} onPointerOver={onOver} onPointerLeave={onOut}>
          <div class="tt-head" style={{ height: HEAD + "px" }}>
            <div class="tt-corner" style={{ width: GUTTER + "px" }} />
            {marks.map((s) => (
              <span class={"tt-hour" + (s % HOUR ? "" : " full")} style={{ left: GUTTER + s * scale + "px" }}>{markText(h0, s)}</span>
            ))}
            {shown.map((c, i) => (
              <span class="tt-bullet" data-k={c.key} style={{ left: GUTTER + xs[a + i] - COL / 2 + "px", width: COL + "px", top: MARK_H + "px" }}><Bullet l={c.bullet} /></span>
            ))}
            <Lit part="head" hot={hot} at={colAt} cols={cols} xs={xs} shift={shift} dark={dark} now={now} />
          </div>
          <div class="tt-body" style={{ height: rowsH + "px", "--row": ROW + "px" }}>
            <div class="tt-names" style={{ width: GUTTER + "px" }}>{rowNames.map((s) => <span>{s}</span>)}</div>
            <span class="tt-now" style={{ left: GUTTER + (now - day0) * scale + "px" }} />
            {marks.map((s) => <span class={"tt-hour-rule" + (s % HOUR ? "" : " full")} style={{ left: GUTTER + s * scale + "px" }} />)}
            {shown.map((c, i) => <Column c={c} x={xs[a + i]} cls={c.t < now ? " past" : ""} shift={shift} dark={dark} />)}
            <Lit part="body" hot={hot} at={colAt} cols={cols} xs={xs} shift={shift} dark={dark} now={now} />
          </div>
        </div>
      </div>
      <div class="tt-chart" onWheel={onWheel}>
        {vw > 0 && scale > 0 && (
          <Chart hot={hot} key={axis.num} l={axis} st={st} links={links} width={vw + PAD_R} plotH={plotH} group={group} dir={dir}
            view={{ t0: day0 + left / scale, span: plotW / scale }} />
        )}
      </div>
    </div>
  );
}
