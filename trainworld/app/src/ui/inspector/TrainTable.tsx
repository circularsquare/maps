// T-3469: the schedule panel's timetable, laid out as a printed one (a Tokaido Shinkansen timetable
// from Anita's board, 2026-11-16): a column per train under its line's bullet, the stations down the
// side, small times in tight cells, one direction at a time over the day from the last hour. Each column stands
// over its train's departure on the time scale of the chart below, which scrolls with the table, so a
// column sits right above where its train's path starts. The times are the profiles the chart draws
// (as the departures board's check reads them). notes/T-3346.md "Stage 1 as built".
// T-3817: the wheel zooms the table and chart together; only the columns that fit at the zoom are drawn,
// every train's bullet is, the drawn ones in front. notes/T-3817.md.

import { signal, type Signal } from "@preact/signals";
import type { ComponentChildren, VNode } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { clock } from "../../game/clock";
import { stopKey } from "../../game/complexView";
import { drawnReversed } from "../../game/northUp";
import { inkOn, legibleOn } from "../../game/palette";
import { decodeRelated, type RelatedGroup } from "../../game/relatedLines";
import { display, shade as uiShade } from "../../game/ui";
import type { LineInput, LineStats } from "../../game/types";
import { linesById, linesByNum, stationNames, stationsById, world } from "../../game/world";
import { t } from "../../i18n";
import { profileRun } from "../../map/profiles";
import { live } from "../../map/live";
import type { NetworkBuffers } from "../../map/network";
import { queryClock } from "../../workers/clockClient";
import { useClockQuery } from "../useClockQuery";
import { Chip, Segmented } from "../widgets";
import { GUTTER, MareyBody, PAD_R, type Links } from "./MareyChart";
import type { Shared } from "./mareyShare";
import { ONLY_TIP, OnlyIcon } from "./stopOnly";

const DAY = 86400, HOUR = 3600;
/** a column's width, a row's height and the head's height, px */
const COL = 27, ROW = 13, MARK_H = 14;
/** a strip head's letters at most, and the height of each, px */
const STRIP_MAX = 4, STRIP_CH = 11;
/** the share of the day's columns the opening zoom draws, at least */
const OPEN_SHARE = 0.95;
/** the zoom's limits: the span across the plot, s */
const SPAN_MIN = 900, SPAN_MAX = DAY;
/** the wheel: the zoom a notch (100 px of wheel) */
const NOTCH = 1.25;
/** the zoom levels' gaps, s: a level draws columns at least this far apart in time; each level keeps
 * the coarser one's columns and fills the gaps between them, so zooming in only adds columns */
const GAPS = Array.from({ length: 30 }, (_, j) => 10 * Math.SQRT2 ** j);
/** how near a bullet's centre the pointer must be to hover it, px */
const BULLET_HIT = 9;
/** how much lower a related line's bullet sits, px (Anita, 2026-11-20 pad: "like 2px lower") */
const REL_DROP = 2;
/** related bullets behind closer than this to one another are drawn as one, px (a whole day's pile) */
const REL_THIN = 3;
/** a column's class for its rank and time: a related line's times are fainter (style.css `.tt-col.rel`) */
const colClass = (c: Col, now: number) => (c.rank ? " rel" : "") + (c.t < now ? " past" : "");
/** a cell's value for a train that runs through the station without stopping */
const PASS = -1;
/** the time marks' steps, minutes */
const MARK_MIN = [5, 10, 15, 30, 60, 120, 180];
/** a mark's label, local, from the local hour `h0` the table starts at and s on from it: "8:00", "8:30" */
const markText = (h0: number, s: number) => `${(h0 + Math.floor(s / HOUR)) % 24}:${String(Math.floor((s % HOUR) / 60)).padStart(2, "0")}`;

const DEBUG = typeof location !== "undefined" && new URLSearchParams(location.search).has("debug");

/** The run of profile `p`. */
const runOf = (p: number) => profileRun(p);

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
 * its stops' row on the axis line (run 0 order, -1 off it), whether its run 0 goes the axis's
 * run 1 way, and its rank (0: the axis line and its group's; 1: a related line, T-3818). `fill`: the
 * axis rows (run 0 order) its trains may run through, where it runs beside the axis; every row
 * without it. */
interface Src { line: LineInput; bullet: LineInput; st: LineStats; rowOf: number[]; flipped: boolean; rank: number; fill?: Uint8Array }

/** A line related to the table's line (T-3818, T-3846): a line the line panel's related lines draw beside it
 * (`lineRelated`, kind beside) that passes `isRelated`, its stops' rows on the table's line where both stop, the rows it runs
 * beside it, and for the chart its stretches from stop to stop. */
export interface Related { line: LineInput; rowOf: number[]; fill: Uint8Array; stretches: Shared[] }

/** What each line of `groups` has in common with the selected line: the complexes where both stop (a complex
 * counted once) and whether it runs on the same track anywhere. */
export function relatedCommon(groups: readonly RelatedGroup[]): Map<number, { stops: Set<number>; track: boolean }> {
  const out = new Map<number, { stops: Set<number>; track: boolean }>();
  for (const g of groups) {
    let c = out.get(g.line);
    if (!c) out.set(g.line, (c = { stops: new Set(), track: false }));
    if (g.shared) c.track = true;
    for (const m of g.meetings) if (m.stopL >= 0 && m.stopO >= 0) c.stops.add(m.complex);
  }
  return out;
}

/** The timetable's rule for a related line (T-3846, Anita 2026-11-20): it runs on the line's track at any point,
 * or shares at least 3 stops with it and those are at least 40% of either line's `nL`, `nO` stops. */
export function isRelated(c: { stops: Set<number>; track: boolean } | undefined, nL: number, nO: number): boolean {
  if (!c) return false;
  const k = c.stops.size;
  return c.track || (k >= 3 && (k >= 0.4 * nL || k >= 0.4 * nO));
}

/** The lines related to `l` (not its own twin or `skip`'s), and `links` with their stretches added
 * where the chart has no shared track for them, so the chart draws them as it draws other lines. */
export function useRelated(l: LineInput, st: LineStats, links: Links | null, skip?: ReadonlySet<number>): { links: Links | null; related: Related[] } {
  const w = world.value!;
  const groups = useClockQuery({ what: "lineRelated", line: l.num }, [l.num, w.version], (d) => ({ line: l.num, groups: decodeRelated(d) }));
  const skipKey = skip ? [...skip].join(",") : "";
  return useMemo(() => {
    if (!groups || groups.line !== l.num) return { links, related: [] };
    const byNum = linesByNum(w), byId = linesById(w);
    const twin = st.twin ? byId.get(st.twin)?.num : undefined;
    const n = l.stops.length, L0 = st.stopS[0] ?? [];
    const by = new Map<number, Related>();
    const common = relatedCommon(groups.groups);
    for (const g of groups.groups) {
      if (g.kind !== "beside" || g.line === twin || skip?.has(g.line)) continue;
      const o = byNum.get(g.line), ost = o && w.lineStats[o.id];
      // running rail lines only: a planned or stopped line has no timetable
      if (!o || !ost || ost.bus || o.twinOf || ost.status !== "running") continue;
      if (!isRelated(common.get(g.line), n, o.stops.length)) continue;
      let r = by.get(g.line);
      if (!r) by.set(g.line, (r = { line: o, rowOf: o.stops.map(() => -1), fill: new Uint8Array(n), stretches: [] }));
      const both = g.meetings.filter((m) => m.stopL >= 0 && m.stopO >= 0 && m.stopL < n && m.stopO < o.stops.length);
      if (!both.length) continue;
      for (const m of both) if (r.rowOf[m.stopO] < 0) r.rowOf[m.stopO] = m.stopL;
      const ls = both.map((m) => m.stopL);
      for (let i = Math.min(...ls); i <= Math.max(...ls); i++) r.fill[i] = 1;
      // from stop to stop where both stop, the other line's run 0 offsets onto this line's
      const O0 = ost.stopS[0] ?? [];
      const pts = both.map((m) => [O0[m.stopO], L0[m.stopL]]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)).sort((x, y) => x[0] - y[0]);
      for (let i = 0; i + 1 < pts.length; i++) {
        const [a, c] = pts[i], [b, e] = pts[i + 1];
        if (b - a < 1) continue;
        const m = (e - c) / (b - a);
        r.stretches.push({ a, b, c, m, sg: m < 0 ? -1 : 1 });
      }
    }
    const related = [...by.values()].filter((r) => new Set(r.rowOf.filter((x) => x >= 0)).size >= 2);
    if (!links) return { links, related };
    const shared = new Map(links.shared);
    for (const r of related) {
      if (shared.has(r.line.num) || !r.stretches.length) continue;
      shared.set(r.line.num, r.stretches);
      const tw = w.lineStats[r.line.id]?.twin, tl = tw ? byId.get(tw) : undefined;
      if (tl && tl.stops.length === r.line.stops.length && !shared.has(tl.num)) shared.set(tl.num, r.stretches);
    }
    return { links: { shared, passes: links.passes }, related };
  }, [groups, links, w.version, skipKey]);
}

/** The columns' lines: the axis line, then each other line by the stations it shares with the
 * axis (two at least, which set its direction; a station's complex counts as the station), each
 * with its hidden quad twin, then the related lines. */
function sourcesOf(axis: LineInput, others: readonly LineInput[], related: readonly Related[]): Src[] {
  const w = world.value!;
  const byId = linesById(w);
  const index = new Map<string, number>();
  axis.stops.forEach((s, i) => { const k = stopKey(s); if (!index.has(k)) index.set(k, i); });
  const out: Src[] = [];
  const add = (l: LineInput, rowOf: number[], flipped: boolean, rank: number, fill?: Uint8Array) => {
    const st = w.lineStats[l.id];
    if (!st || st.bus) return;
    out.push({ line: l, bullet: l, st, rowOf, flipped, rank, fill });
    const twin = st.twin ? byId.get(st.twin) : undefined;
    const tst = twin ? w.lineStats[twin.id] : undefined;
    if (twin && tst && twin.stops.length === l.stops.length) out.push({ line: twin, bullet: l, st: tst, rowOf, flipped, rank, fill });
  };
  const flippedOf = (rowOf: number[]) => { const on = rowOf.filter((r) => r >= 0); return on[on.length - 1] < on[0]; };
  add(axis, axis.stops.map((_, i) => i), false, 0);
  for (const o of others) {
    const rowOf = o.stops.map((s) => index.get(stopKey(s)) ?? -1);
    if (new Set(rowOf.filter((r) => r >= 0)).size < 2) continue;
    add(o, rowOf, flippedOf(rowOf), 0);
  }
  const have = new Set(out.map((s) => s.line.num));
  for (const r of related) if (!have.has(r.line.num)) add(r.line, r.rowOf, flippedOf(r.rowOf), 1, r.fill);
  return out;
}

/** A train's name shared with the chart (MareyBody's `hot`): its line, profile and departure. */
const trainKey = (line: number, p: number, dep: number) => `${line}@${p}@${dep}`;

/** One train's column: its bullet, its time at the first row it has, its cells by row shown (NaN
 * blank, PASS run through), and its source's rank. */
interface Col { bullet: LineInput; t: number; cells: Float64Array; key: string; rank: number }

/** The trips the table and the chart ask for: from two hours before the day, for trains that set off
 * then and reach the axis after it starts. One question for both, so the clock worker answers it once. */
const windowOf = (day0: number) => ({ t0: day0 - 2 * HOUR, t1: day0 + DAY });

/** The day's trains of `srcs` from `day0`, as columns per direction (0 the axis's run 0 way, rows
 * in that order), each sorted by time. */
function useColumns(srcs: Src[], rows: number, day0: number): [Col[], Col[]] | null {
  const w = world.value!;
  const net = live.overlay?.renderer.net ?? null;
  const key = srcs.map((s) => s.line.num).join(",");
  // each line's trips kept by line, so lines added (the related ones, by the toggle) are asked for
  // alone and the table stays up meanwhile
  const [answer, setAnswer] = useState<{ day0: number; version: number; trips: Map<number, Float64Array> } | null>(null);
  useEffect(() => {
    let on = true;
    const win = windowOf(day0), version = w.version;
    const have = answer && answer.day0 === day0 && answer.version === version ? answer.trips : null;
    const need = srcs.filter((s) => !have?.has(s.line.num));
    if (!need.length) return;
    void Promise.all(need.map((s) => queryClock({ what: "lineTrips", line: s.line.num, t0: win.t0, t1: win.t1 }))).then((trips) => {
      // the line's own must all answer; a related line that doesn't is left out
      if (!on || trips.some((d, i) => !d && need[i].rank === 0)) return;
      setAnswer((was) => {
        const m = new Map(was && was.day0 === day0 && was.version === version ? was.trips : []);
        trips.forEach((d, i) => m.set(need[i].line.num, d ?? new Float64Array(0)));
        return { day0, version, trips: m };
      });
    });
    return () => { on = false; };
  }, [key, w.version, day0]);
  return useMemo(() => {
    if (!answer || !net || srcs.some((s) => s.rank === 0 && !answer.trips.has(s.line.num))) return null;
    const out: [Col[], Col[]] = [[], []];
    srcs.forEach((src) => {
      const d = answer.trips.get(src.line.num);
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
        const fill = src.fill;
        for (let r = lo + 1; r < hi; r++) if (!Number.isFinite(cells[r]) && (!fill || fill[dir === 0 ? r : rows - 1 - r])) cells[r] = PASS;
        const t0 = cells[lo];
        if (t0 < day0 || t0 >= day0 + DAY) continue;
        out[dir].push({ bullet: src.bullet, t: t0, cells, key: trainKey(src.line.num, p, dep), rank: src.rank });
      }
    });
    for (const cs of out) cs.sort((a, b) => a.t - b.t);
    return out;
  }, [answer, net, rows, srcs]);
}

/** First index with `ts[i] >= x`, for times sorted. */
function lowerBound(ts: ArrayLike<number>, x: number): number {
  let a = 0, b = ts.length;
  while (a < b) { const m = (a + b) >> 1; if (ts[m] < x) a = m + 1; else b = m; }
  return a;
}

/** Whether time `t` is at least `g` from every time in `ts` (sorted). */
function clears(ts: number[], t: number, g: number): boolean {
  const i = lowerBound(ts, t);
  return (i >= ts.length || ts[i] - t >= g) && (i === 0 || t - ts[i - 1] >= g);
}

/** The columns drawn at each zoom level (`GAPS`), 1 where drawn: every level keeps the coarser level's
 * columns and adds, left to right, each that is a level's gap from all it has. Rank by rank: a rank
 * fills only the gaps the ranks before it leave at that level, and its coarser columns that crowd
 * them go, so a lower rank never takes a column's place. */
function pickLevels(cols: Col[]): Uint8Array[] {
  const n = cols.length;
  const out = GAPS.map(() => new Uint8Array(n));
  const ranks = [...new Set(cols.map((c) => c.rank))].sort((a, b) => a - b);
  for (const r of ranks) {
    for (let j = GAPS.length - 1; j >= 0; j--) {
      const m = out[j], up = out[j + 1], g = GAPS[j];
      const before: number[] = [];
      for (let i = 0; i < n; i++) if (m[i]) before.push(cols[i].t);
      if (up) for (let i = 0; i < n; i++) if (up[i] && cols[i].rank === r && clears(before, cols[i].t, g)) m[i] = 1;
      const fixed: number[] = [];
      for (let i = 0; i < n; i++) if (m[i]) fixed.push(cols[i].t);
      let p = 0, last = -Infinity;
      for (let i = 0; i < n; i++) {
        const tt = cols[i].t;
        while (p < fixed.length && fixed[p] <= tt) last = Math.max(last, fixed[p++]);
        if (m[i] || cols[i].rank !== r) continue;
        if (tt - last >= g && (p >= fixed.length || fixed[p] - tt >= g)) { m[i] = 1; last = tt; }
      }
    }
  }
  return out;
}

/** The opening zoom, px per game second: the coarsest level that draws nearly every column of the line's
 * own (rank 0; related lines never change it), just past the scale at which its columns stand a column's
 * width apart. */
function openingScale(levels: Uint8Array[], cols: Col[]): number {
  const n = cols.reduce((s, c) => s + (c.rank === 0 ? 1 : 0), 0);
  for (let j = GAPS.length - 1; j >= 0; j--) {
    let k = 0;
    levels[j].forEach((v, i) => { if (v && cols[i].rank === 0) k++; });
    if (k >= OPEN_SHARE * n) return (COL / GAPS[j]) * 1.001;
  }
  return (COL / GAPS[0]) * 1.001;
}

/** The zoom level for `scale`: the finest whose columns stand a column's width apart. */
const levelOf = (scale: number) => Math.min(GAPS.length - 1, Math.max(0, GAPS.findIndex((g) => g * scale >= COL)));

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
 * colour, as printed timetables head a train with its service's name. A thin rim in the page's
 * colour, so bullets over each other read as many. */
function Bullet({ l }: { l: LineInput }) {
  if (fits(l)) return <Chip line={l} size="sm" plain rim />;
  return <span class="tt-strip" style={{ "--c": l.colour, "--ct": inkOn(l.colour) }}>{[...l.letter].slice(0, STRIP_MAX).map((ch) => <span>{ch}</span>)}</span>;
}

/** A train's times down its column, made once and kept as the column moves with the zoom. */
function Cells({ c, shift }: { c: Col; shift: number }) {
  return <>{Array.from(c.cells, (v) => <span class={v === PASS ? "pass" : undefined}>{v === PASS ? "|" : Number.isFinite(v) ? printed(v, shift) : ""}</span>)}</>;
}

/** A train's column of times, its centre `x` px from the table's left; `cells` its `Cells`. */
function Column({ c, x, cls, cells, dark }: { c: Col; x: number; cls: string; cells: ComponentChildren; dark: boolean }) {
  return <div class={"tt-col" + cls} data-k={c.key} style={{ left: x - COL / 2 + "px", width: COL + "px", "--ink": legibleOn(c.bullet.colour, dark) }}>{cells}</div>;
}

/** The hovered train, over a veil that fades the rest: its bullet (`part` "head"), or its times with
 * covers over the drawn columns it lands on (a train not drawn at this zoom shows the same way). One
 * element's change per hover, whatever the number of columns. */
function Lit({ part, hot, at, cols, drawn, scale, x0, shift, dark, now }: {
  part: "head" | "body"; hot: Signal<string | null>; at: Map<string, number>; cols: Col[]; drawn: Uint8Array; scale: number; x0: number; shift: number; dark: boolean; now: number;
}) {
  const i = hot.value === null ? undefined : at.get(hot.value);
  if (i === undefined) return null;
  const c = cols[i], x = x0 + c.t * scale;
  if (part === "head") {
    return (
      <>
        <div class="tt-veil" />
        <span class={"tt-bullet on" + (c.rank ? " rel" : "")} style={{ left: x - COL / 2 + "px", width: COL + "px", top: MARK_H + (c.rank ? REL_DROP : 0) + "px" }}><Bullet l={c.bullet} /></span>
      </>
    );
  }
  const covers: number[] = [];
  for (const d of [-1, 1]) for (let j = i + d; j >= 0 && j < cols.length && Math.abs(cols[j].t - c.t) * scale < COL; j += d) if (drawn[j]) covers.push(j);
  return (
    <>
      {covers.map((j) => <span class="tt-cover" style={{ left: x0 + cols[j].t * scale - COL / 2 - 1 + "px", width: COL + 2 + "px" }} />)}
      <div class="tt-veil" />
      <Column c={c} x={x} cls={" on" + colClass(c, now)} cells={<Cells c={c} shift={shift} />} dark={dark} />
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

/** The element's CSS zoom (the panels are zoomed by the UI scale): pointer offsets are divided by it. */
const zoomOf = (el: HTMLElement) => (el as HTMLElement & { currentCSSZoom?: number }).currentCSSZoom ?? 1;

/** The timetable of `axis` (its stops as rows) with `others`' trains where they run on its
 * stations, `related`'s behind them while other lines are shown (T-3818), and the chart of `axis`
 * under it. `tools` sit at the end of its direction choice. */
export function TrainTable({ axis, st, others = [], related = [], links, group, plotH, tools }: {
  axis: LineInput; st: LineStats; others?: readonly LineInput[]; related?: readonly Related[]; links: Links | null; group?: ReadonlySet<number>; plotH: number; tools?: ComponentChildren;
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
  const rel = display.value.chartOtherLines ? related : [];
  const relKey = rel.map((r) => r.line.num).join(",");
  const srcs = useMemo(() => sourcesOf(axis, others, rel), [axis.num, otherKey, relKey, w.version]);
  const both = useColumns(srcs, n, day0);
  // the chart's top-to-bottom way first, so the rows read in its order
  const down = drawnReversed(byId.get(axis.stops[0]), byId.get(axis.stops[n - 1]), !!st.loop) ? 1 : 0;
  const [pick, setPick] = useState<number | null>(null);
  const dir = pick ?? down;
  const cols = both?.[dir] ?? [];
  const times = useMemo(() => Float64Array.from(cols, (c) => c.t), [cols]);
  const levels = useMemo(() => pickLevels(cols), [cols]);
  const opening = useMemo(() => openingScale(levels, cols), [levels]);

  const scroller = useRef<HTMLDivElement>(null);
  const vw = useClientWidth(scroller);
  const plotW = Math.max(0, vw - GUTTER);
  // the zoom the player chose, kept while the line is the same; the opening zoom until then
  const [chosen, setChosen] = useState<{ for: string; scale: number } | null>(null);
  const lineKey = `${axis.num}:${otherKey}`;
  const lo = plotW > 0 ? plotW / SPAN_MAX : 0, hi = plotW > 0 ? plotW / SPAN_MIN : Infinity;
  // opening no closer than an hour across: trains leaving seconds apart (a quad twin's) never all fit
  const scale = Math.min(hi, Math.max(lo, chosen?.for === lineKey ? chosen.scale : (plotW > 0 ? Math.min(opening, plotW / HOUR) : opening)));
  const level = levelOf(scale);
  const drawn = levels[level] ?? new Uint8Array(0);
  // the train under the pointer, here or on the chart: read only by the parts that draw it, so a
  // hover redraws neither the table nor the chart's trains
  const hot = useMemo(() => signal<string | null>(null), []);
  const colAt = useMemo(() => new Map(cols.map((c, i) => [c.key, i])), [cols]);
  const drag = useRef<{ id: number; x: number; y: number; sl: number; st: number; moved: boolean } | null>(null);
  const onOver = (e: PointerEvent) => {
    if (drag.current?.moved) return;
    const k = (e.target as Element).closest?.(".tt-body [data-k]")?.getAttribute("data-k") ?? null;
    if (k !== null || !(e.target as Element).closest?.(".tt-head")) { if (hot.value !== k) hot.value = k; }
  };
  const onOut = () => { hot.value = null; };
  const width = DAY * scale + COL;
  /** The table's scroll that puts time `t` at the view's left edge at scale `s`, px. */
  const xFor = (t: number, s: number) => Math.max(0, Math.min(GUTTER + DAY * s + COL - vw, (t - day0) * s));

  const [left, setLeft] = useState(0);
  // the scroll last set here, so the table's scroll is set without being read back
  const setX = useRef(-1);
  const frame = useRef(0);
  // the time at the view's left edge, kept as the table's start or scale moves under it
  const leftT = useRef(0);
  const onScroll = () => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => {
      const x = scroller.current?.scrollLeft ?? 0;
      leftT.current = day0 + x / scaleRef.current;
      setX.current = x;
      setLeft(x);
    });
  };
  const scaleRef = useRef(scale);
  scaleRef.current = scale;
  // a new line or direction: the time now a quarter of the way in
  const placedFor = useRef("");
  useLayoutEffect(() => {
    const el = scroller.current, k = `${axis.num}:${dir}:${otherKey}`;
    if (!el || !both || !vw) return;
    if (placedFor.current !== k) {
      placedFor.current = k;
      leftT.current = now - (0.25 * plotW) / scale;
    }
    const x = xFor(leftT.current, scale);
    if (Math.abs(setX.current - x) > 0.5) { el.scrollLeft = x; setX.current = x; }
    if (Math.abs(left - x) > 0.5) setLeft(x);
    // a zoom step's work on the page, from the wheel's frame to the table laid out (with ?debug)
    if (DEBUG && zoomT.current) { void el.offsetWidth; ((window as unknown as { __zoomMs?: number[] }).__zoomMs ??= []).push(performance.now() - zoomT.current); zoomT.current = 0; }
  }, [both, dir, vw, scale, day0]);

  // the wheel zooms about the pointer, one step a frame however many events come
  const wheel = useRef<{ f: number; cx: number; raf: number } | null>(null);
  const zoomT = useRef(0);
  const zoomAt = (f: number, cx: number) => {
    const el = scroller.current;
    if (!el) return;
    const s0 = scaleRef.current, s1 = Math.min(hi, Math.max(lo, s0 * f));
    if (s1 === s0) return;
    if (DEBUG) zoomT.current = performance.now();
    const tAt = day0 + (setX.current + cx - GUTTER) / s0;
    leftT.current = tAt - (cx - GUTTER) / s1;
    // the scale and the scroll it brings in one render, so a zoom frame lays the page out once
    setChosen({ for: lineKey, scale: s1 });
    setLeft(xFor(leftT.current, s1));
  };
  // wheel and pinch both come here: factors multiply, one zoom a frame
  const zoomStep = (f: number, cx: number) => {
    const p = wheel.current ?? (wheel.current = { f: 1, cx, raf: 0 });
    p.f *= f;
    p.cx = cx;
    if (!p.raf) p.raf = requestAnimationFrame(() => { const q = wheel.current!; wheel.current = null; zoomAt(q.f, Math.max(GUTTER, q.cx)); });
  };
  const onWheel = (e: WheelEvent) => {
    const el = scroller.current;
    if (!el) return;
    const z = zoomOf(el), r = el.getBoundingClientRect(), cx = (e.clientX - r.left) / z;
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientWidth : 1;
    const dx = e.deltaX * unit, dy = e.deltaY * unit;
    // over the station names the wheel scrolls the rows, as it does any list
    if (e.currentTarget === el && cx < GUTTER && !e.shiftKey) return;
    e.preventDefault();
    if (e.shiftKey || Math.abs(dx) > Math.abs(dy)) { el.scrollLeft += e.shiftKey ? dx || dy : dx; return; }
    if (!dy) return;
    zoomStep(NOTCH ** (-dy / 100), cx);
  };
  useEffect(() => () => { if (wheel.current) cancelAnimationFrame(wheel.current.raf); }, []);
  // a press and drag moves the table and chart, sideways and down its rows (touch scrolls natively)
  const onDown = (e: PointerEvent) => {
    const el = scroller.current;
    if (!el || e.button !== 0 || e.pointerType === "touch") return;
    // a press on the table's own scrollbars scrolls natively
    if (e.target === el && (e.offsetX >= el.clientWidth || e.offsetY >= el.clientHeight)) return;
    drag.current = { id: e.pointerId, x: e.clientX, y: e.clientY, sl: el.scrollLeft, st: el.scrollTop, moved: false };
  };
  const onMove = (e: PointerEvent) => {
    const d = drag.current, el = scroller.current;
    if (!d || !el || d.id !== e.pointerId) return;
    // released where this element did not hear it
    if (!(e.buttons & 1)) { drag.current = null; el.classList.remove("dragging"); return; }
    const z = zoomOf(el), dx = (e.clientX - d.x) / z, dy = (e.clientY - d.y) / z;
    if (!d.moved && Math.hypot(dx, dy) < 4) return;
    if (!d.moved) {
      d.moved = true;
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      el.classList.add("dragging");
      hot.value = null;
    }
    el.scrollLeft = d.sl - dx;
    el.scrollTop = d.st - dy;
  };
  const onUp = (e: PointerEvent) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    drag.current = null;
    scroller.current?.classList.remove("dragging");
  };
  // two fingers zoom about their midpoint by the change in their distance, through the wheel's step; one
  // finger scrolls natively (the CSS touch-action keeps the page itself from zooming)
  const pinch = useRef(0);
  const span = (e: TouchEvent) => Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  const onTouchStart = (e: TouchEvent) => { pinch.current = e.touches.length === 2 ? span(e) : 0; };
  const onTouchMove = (e: TouchEvent) => {
    const el = scroller.current;
    if (!el || e.touches.length !== 2) return;
    if (e.cancelable) e.preventDefault();
    const d = span(e), prev = pinch.current;
    pinch.current = d;
    if (!prev || !d) return;
    const z = zoomOf(el), r = el.getBoundingClientRect();
    const cx = ((e.touches[0].clientX + e.touches[1].clientX) / 2 - r.left) / z;
    zoomStep(d / prev, cx);
  };
  const onTouchEnd = (e: TouchEvent) => { pinch.current = e.touches.length === 2 ? span(e) : 0; };
  const grab = { onPointerDown: onDown, onPointerMove: onMove, onPointerUp: onUp, onPointerCancel: onUp, onWheel, onTouchStart, onTouchMove, onTouchEnd, onTouchCancel: onTouchEnd };
  // the bullet nearest the pointer in the head row, so one under others is reached too
  const onHeadMove = (e: PointerEvent) => {
    const el = scroller.current;
    if (!el || drag.current?.moved) return;
    const z = zoomOf(el), r = el.getBoundingClientRect();
    const cx = (e.clientX - r.left) / z, cy = (e.clientY - r.top) / z;
    let k: string | null = null, tip = "";
    if (cx >= GUTTER && cy >= MARK_H && cols.length) {
      const tt = day0 + (el.scrollLeft + cx - GUTTER) / scale;
      // the nearest centre, a related line's counted REL_DROP px further, so the line's own win a tie
      let best = Infinity, j = -1;
      for (let i = lowerBound(times, tt - BULLET_HIT / scale); i < cols.length && (times[i] - tt) * scale <= BULLET_HIT; i++) {
        const d = Math.abs(times[i] - tt) * scale + (cols[i].rank ? REL_DROP : 0);
        if (d < best) { best = d; j = i; }
      }
      if (j >= 0) {
        k = cols[j].key;
        // another line's bullet names its line where the bullet's letters don't already
        const b = cols[j].bullet;
        if (b.num !== axis.num && b.name !== b.letter) tip = b.name;
      }
    }
    const head = e.currentTarget as HTMLElement;
    if ((head.getAttribute("data-tip") ?? "") !== tip) head.setAttribute("data-tip", tip);
    if (hot.value !== k) hot.value = k;
  };

  // only the columns near the view are drawn, and of them only the zoom level's
  const a = lowerBound(times, day0 + (left - plotW / 2 - COL) / scale), b = lowerBound(times, day0 + (left + 1.5 * plotW + COL) / scale);
  const shown: number[] = [];
  for (let i = a; i < b; i++) if (drawn[i]) shown.push(i);
  // a column's times and a line's bullet are made once and kept, so a zoom or a scroll moves their
  // boxes and renders none of what is in them
  const cellsMade = useMemo(() => new Map<string, VNode>(), [cols, shift]);
  const cellsOf = (c: Col) => {
    let v = cellsMade.get(c.key);
    if (!v) cellsMade.set(c.key, (v = <Cells c={c} shift={shift} />));
    return v;
  };
  const bulletsMade = useMemo(() => new Map<number, VNode>(), [cols]);
  const bulletOf = (l: LineInput) => {
    let v = bulletsMade.get(l.num);
    if (!v) bulletsMade.set(l.num, (v = <Bullet l={l} />));
    return v;
  };
  // px from the table's left of game second 0
  const x0 = GUTTER - day0 * scale;
  const bullet = (c: Col, cls: string) => <span key={c.key} class={"tt-bullet" + cls + (c.rank ? " rel" : "")} style={{ left: x0 + c.t * scale - COL / 2 + "px", width: COL + "px", top: MARK_H + (c.rank ? REL_DROP : 0) + "px" }}>{bulletOf(c.bullet)}</span>;
  // back to front: related lines' bullets behind the line's own, the drawn columns' in front of both
  const backs: VNode[] = [], fronts: VNode[] = [];
  for (const rank of [1, 0]) {
    // related bullets behind: one per REL_THIN px of the day (by the day's px, so a pan never changes which)
    let last = -Infinity;
    for (let i = a; i < b; i++) {
      if ((cols[i].rank ? 1 : 0) !== rank) continue;
      const bin = Math.floor((cols[i].t * scale) / REL_THIN);
      if (rank && bin === last) continue;
      last = bin;
      backs.push(bullet(cols[i], " back"));
    }
    for (const i of shown) if ((cols[i].rank ? 1 : 0) === rank) fronts.push(bullet(cols[i], ""));
  }
  if (DEBUG) (window as unknown as { __tt?: unknown }).__tt = { cols: cols.length, rel: cols.reduce((s, c) => s + (c.rank ? 1 : 0), 0), relDrawn: shown.reduce((s, i) => s + (cols[i].rank ? 1 : 0), 0), hourPx: scale * HOUR, level, drawn: drawn.reduce((s, v) => s + v, 0), shown: shown.length, h0: ((Math.round((day0 + shift) / HOUR) % 24) + 24) % 24 };

  const rowIds = Array.from({ length: n }, (_, i) => axis.stops[dir === 0 ? i : n - 1 - i]);
  // T-4013: a stop the line's riders only board or only get off at carries its mark
  const rowNames = rowIds.map((id) => {
    const only = st.stopOnly?.[id];
    return <>{names.get(id) ?? ""}{only && <span class="stop-only" data-tip={t(ONLY_TIP[only])}><OnlyIcon kind={only} /></span>}</>;
  });
  const label = (d: number) => st.loop
    ? t("schedulePanel.via", { station: names.get(axis.stops[d === 0 ? 1 : n - 2]) ?? "" })
    : t("schedulePanel.to", { station: names.get(axis.stops[d === 0 ? n - 1 : 0]) ?? "" });
  // time marks from the day's start, as far apart as a label needs
  const h0 = (((Math.round((day0 + shift) / HOUR)) % 24) + 24) % 24;
  const step = (MARK_MIN.find((m) => m * 60 * scale >= 52) ?? 240) * 60;
  const marks: number[] = [];
  for (let s = Math.ceil(Math.max(0, left / scale - step) / step) * step; s <= DAY && s * scale < left + plotW + 60; s += step) marks.push(s);
  const rowsH = n * ROW;
  const tallest = Math.max(1, ...srcs.map((s) => (fits(s.bullet) ? 1 : Math.min(STRIP_MAX, [...s.bullet.letter].length))));
  const HEAD = MARK_H + (tallest > 1 ? tallest * STRIP_CH + 6 : 20) + (srcs.some((s) => s.rank) ? REL_DROP : 0);
  const win = useMemo(() => windowOf(day0), [day0]);
  return (
    <div class="tt">
      <div class="sp-tabs">
        <Segmented<number> options={[[down, label(down)], [1 - down, label(1 - down)]]} value={dir} onChange={setPick} />
        {tools}
      </div>
      <div class="tt-scroll" ref={scroller} onScroll={onScroll} {...grab} style={{ marginRight: PAD_R + "px", maxHeight: `min(${rowsH + HEAD + 14}px, 44vh)` }}>
        <div class="tt-inner" style={{ width: GUTTER + width + "px", height: HEAD + rowsH + "px" }} onPointerOver={onOver} onPointerLeave={onOut}>
          <div class="tt-head" style={{ height: HEAD + "px" }} data-tip="" onPointerMove={onHeadMove}>
            <div class="tt-corner" style={{ width: GUTTER + "px" }} />
            {marks.map((s) => (
              <span class={"tt-hour" + (s % HOUR ? "" : " full")} style={{ left: GUTTER + s * scale + "px" }}>{markText(h0, s)}</span>
            ))}
            {backs}
            {fronts}
            <Lit part="head" hot={hot} at={colAt} cols={cols} drawn={drawn} scale={scale} x0={x0} shift={shift} dark={dark} now={now} />
          </div>
          <div class="tt-body" style={{ height: rowsH + "px", "--row": ROW + "px" }}>
            <div class="tt-names" style={{ width: GUTTER + "px" }}>{rowNames.map((s) => <span>{s}</span>)}</div>
            <span class="tt-now" style={{ left: GUTTER + (now - day0) * scale + "px" }} />
            {marks.map((s) => <span class={"tt-hour-rule" + (s % HOUR ? "" : " full")} style={{ left: GUTTER + s * scale + "px" }} />)}
            {shown.map((i) => <Column key={cols[i].key} c={cols[i]} x={x0 + cols[i].t * scale} cls={colClass(cols[i], now)} cells={cellsOf(cols[i])} dark={dark} />)}
            <Lit part="body" hot={hot} at={colAt} cols={cols} drawn={drawn} scale={scale} x0={x0} shift={shift} dark={dark} now={now} />
          </div>
        </div>
      </div>
      <div class="tt-chart" {...grab}>
        {vw > 0 && scale > 0 && (
          <Chart hot={hot} key={axis.num} l={axis} st={st} links={links} width={vw + PAD_R} plotH={plotH} group={group} dir={dir}
            win={win} view={{ t0: day0 + left / scale, k: scale }} />
        )}
      </div>
    </div>
  );
}
