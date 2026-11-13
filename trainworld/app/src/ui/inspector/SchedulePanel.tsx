// T-3346 stage 1: the schedule panel, centred like Community (notes/T-3346.md). A line's, opened
// from its panel's schedule heading: the schedule in Simple mode on the left (the line panel's own
// table, steppers and dots), its departures board and timetable chart on the right. A group's,
// opened from the group view: the group's chart, the longest line's stops down its side and the
// others drawn where they share its track. docs/ui-panels.md "Schedule panel".

import { signal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { clock, demandAt, lineSeconds } from "../../game/clock";
import { groupName, lineGroups } from "../../game/lineGroups";
import type { LineInput, LineStats } from "../../game/types";
import { display, setDisplay } from "../../game/ui";
import { linesById, listedLines, noteHeadwayLimits, stationNames, stationsById, world } from "../../game/world";
import { t } from "../../i18n";
import { queryClock } from "../../workers/clockClient";
import { GroupPill } from "../GroupPill";
import { useModalDialog } from "../Modal";
import { useClockQuery, useLatestAnswer } from "../useClockQuery";
import { Chip, Icon, Segmented } from "../widgets";
import { limitsFrom } from "./LineInspector";
import { MareyBody, useChartLinks } from "./MareyChart";
import { NO_LIMITS, Schedule, useLinePeriods, vehicleOf } from "./schedule";

/** What the panel shows: a line's schedule, or a group's chart; null closed. */
export type ScheduleShown = { kind: "line"; line: string } | { kind: "group"; group: number };
export const scheduleShown = signal<ScheduleShown | null>(null);

/** The right side's view, kept for the session. */
type View = "board" | "chart";
const view = signal<View>("board");

/** The chart's stops span, px; the right side is the chart's height (as MareyChart's, 30 px more)
 * whichever view shows, so switching moves nothing. */
const PLOT_H = 380;
const VIEW_H = PLOT_H + 30;
/** The board's day starts at this local hour, as Japanese station timetables do. */
const DAY_START_H = 4;

const DAY = 86400, HOUR = 3600;

/** A rail line the panel can show: running, with track and two stops. */
const charted = (st: LineStats | undefined, l: LineInput | undefined): st is LineStats => !!st && !!l && !st.bus && st.status === "running" && l.stops.length >= 2;

// ---------------------------------------------------------------- departures

/** One train leaving the board's station: its run, its stop index in that run, game seconds. */
export interface Departure { run: number; k: number; t: number }

/** T-3346: the one read of a station's departures (`TrackApi.station_departures`: each trip's
 * leaving time there from its own timetable, in the sim). Line `l` and, run on both pairs of quad
 * track (T-3333), its hidden twin at the stop in the same place in its list, merged in time order. */
export async function stationDepartures(l: LineInput, st: LineStats, station: string, t0: number, t1: number): Promise<Departure[] | null> {
  const w = world.value;
  if (!w) return null;
  const byId = stationsById(w);
  const asks: { line: number; node: number }[] = [];
  const node = byId.get(station)?.num;
  if (node !== undefined) asks.push({ line: l.num, node });
  const twin = st.twin ? linesById(w).get(st.twin) : undefined;
  if (twin) {
    const nodes = new Set(l.stops.flatMap((s, i) => (s === station && twin.stops[i] ? [byId.get(twin.stops[i])?.num] : [])));
    for (const n of nodes) if (n !== undefined) asks.push({ line: twin.num, node: n });
  }
  const answers = await Promise.all(asks.map((a) => queryClock({ what: "stationDeps", line: a.line, node: a.node, t0, t1 })));
  if (answers.some((d) => !d)) return null;
  const out: Departure[] = [];
  for (const d of answers) for (let i = 0; i + 2 < d!.length; i += 3) out.push({ run: d![i], k: d![i + 1], t: d![i + 2] });
  return out.sort((a, b) => a.t - b.t || a.run - b.run);
}

/** The board's hours from the day's start, local: [hour, minutes per run]. */
function boardRows(deps: Departure[], shift: number): { hour: number; mins: string[][] }[] {
  const rows = Array.from({ length: 24 }, (_, i) => ({ hour: (DAY_START_H + i) % 24, mins: [[], []] as string[][] }));
  for (const d of deps) {
    const local = (((d.t + shift) % DAY) + DAY) % DAY;
    const h = Math.floor(local / HOUR);
    rows[(h - DAY_START_H + 24) % 24].mins[d.run === 1 ? 1 : 0].push(String(Math.floor((local % HOUR) / 60)).padStart(2, "0"));
  }
  const used = rows.map((r) => r.mins[0].length + r.mins[1].length > 0);
  const a = used.indexOf(true), b = used.lastIndexOf(true);
  return a < 0 ? [] : rows.slice(a, b + 1);
}

/** The station's departures over the next 24 hours from the start of this hour, laid out as a day:
 * the line's current timetable for every hour. */
function Board({ l, st }: { l: LineInput; st: LineStats }) {
  const w = world.value!;
  const names = stationNames(w);
  // the stations in the line's order, each once; a loop's first stop again at its end left out
  const stops = [...new Set(l.stops)];
  const [station, setStation] = useState(stops[0]);
  const at = stops.includes(station) ? station : stops[0];
  const shift = st.shift ?? 0;
  const now = clock.minute.value * 60;
  const t0 = Math.floor((now + shift) / HOUR) * HOUR - shift;
  const deps = useLatestAnswer(() => stationDepartures(l, st, at, t0, t0 + DAY).then((d) => (d ? { at, d } : null)), [l.num, at, w.version, t0]);
  const shown = deps && deps.at === at ? deps.d : null;
  const n = l.stops.length;
  // a run's heading: where its trains go, or on a loop the next stop
  const label = (run: number, k: number) => {
    if (st.loop) {
      const next = run === 0 ? l.stops[k + 1] : l.stops[n - 2 - k];
      return t("schedulePanel.via", { station: names.get(next ?? "") ?? "" });
    }
    return t("schedulePanel.to", { station: names.get(run === 0 ? l.stops[n - 1] : l.stops[0]) ?? "" });
  };
  const runs = [0, 1].filter((r) => shown?.some((d) => d.run === r));
  const heads = runs.map((r) => label(r, shown!.find((d) => d.run === r)!.k));
  const rows = shown ? boardRows(shown, shift) : [];
  const hourNow = Math.floor((((now + shift) % DAY) + DAY) % DAY / HOUR);
  const body = useRef<HTMLDivElement>(null);
  // opened, or another station: the hour now at the top
  useLayoutEffect(() => {
    const el = body.current, row = el?.querySelector<HTMLElement>(".sp-row.now");
    if (el && row) el.scrollTop = row.offsetTop - el.offsetTop - 28;
  }, [at, !!shown]);
  return (
    <>
      <div class="sp-tools">
        <select aria-label={t("schedulePanel.station")} value={at} onChange={(e) => setStation(e.currentTarget.value)}>
          {stops.map((s) => <option value={s}>{names.get(s) ?? ""}</option>)}
        </select>
      </div>
      <div class="sp-board" ref={body} style={{ "--runs": Math.max(1, runs.length) }}>
        {shown && !rows.length && <p class="muted">{t("schedulePanel.noDepartures")}</p>}
        {rows.length > 0 && (
          <div class="sp-grid">
            <div class="sp-row sp-heads">
              <span class="sp-hour" />
              {heads.map((h) => <span class="sp-dir">{h}</span>)}
            </div>
            {rows.map((r) => (
              <div class={"sp-row" + (r.hour === hourNow ? " now" : "")}>
                <span class="sp-hour">{r.hour}</span>
                {runs.map((run) => <span class="sp-mins">{r.mins[run].map((m) => <span>{m}</span>)}</span>)}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- the panel

/** The width of `ref`'s element, px, kept as it changes. */
function useWidth(ref: { current: HTMLElement | null }): number {
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

/** The toggle for other lines' trains on the chart, where any meet it. */
function OthersToggle({ any }: { any: boolean }) {
  if (!any) return null;
  const on = display.value.chartOtherLines;
  const tip = t(on ? "mareyChart.hideOthers" : "mareyChart.showOthers");
  return <button type="button" class={"btn icon mini sp-others" + (on ? " on" : "")} aria-pressed={on} data-tip={tip} aria-label={tip}
    onClick={() => setDisplay("chartOtherLines", !on)}><Icon.related /></button>;
}

function LineBody({ l, st }: { l: LineInput; st: LineStats }) {
  const w = world.value!;
  const rv = vehicleOf(st);
  const limits = useClockQuery({ what: "lineHeld", line: l.num }, [l.num, w.version, rv], (d) => limitsFrom(d, rv)) ?? NO_LIMITS;
  if (limits !== NO_LIMITS) noteHeadwayLimits(l.id, limits.map((x) => x?.s ?? 0));
  const { periods } = useLinePeriods(l.id, l);
  const links = useChartLinks(l, st);
  const right = useRef<HTMLDivElement>(null);
  const width = useWidth(right);
  const v = view.value;
  const any = !!links && (links.shared.size > 0 || links.passes.length > 0);
  return (
    <div class="sp-body">
      <div class="sp-left">
        <Schedule l={l} st={st} now={demandAt(lineSeconds(st))} limits={limits} crowd={periods} label={t("schedulePanel.setBy")} />
      </div>
      <div class="sp-right" ref={right}>
        <div class="sp-tabs">
          <Segmented<View> options={[["board", t("schedulePanel.departures")], ["chart", t("schedulePanel.chart")]]} value={v} onChange={(x) => (view.value = x)} />
          {v === "chart" && <OthersToggle any={any} />}
        </div>
        <div class="sp-view" style={{ height: VIEW_H + "px" }}>
          {v === "board" ? <Board key={l.num} l={l} st={st} />
            : width > 0 && <MareyBody key={l.num} l={l} st={st} links={links} width={width} plotH={PLOT_H} />}
        </div>
      </div>
    </div>
  );
}

function GroupBody({ axis, st, others }: { axis: LineInput; st: LineStats; others: number[] }) {
  const links = useChartLinks(axis, st);
  // kept while the group's lines are the same: the chart redraws its trains when it changes
  const key = others.join(",");
  const group = useMemo<ReadonlySet<number>>(() => new Set(others), [key]);
  const box = useRef<HTMLDivElement>(null);
  const width = useWidth(box);
  const any = !!links && [...links.shared.keys(), ...links.passes.map((p) => p.o)].some((o) => !group.has(o));
  return (
    <div class="sp-group" ref={box}>
      {any && <div class="sp-tabs"><OthersToggle any /></div>}
      {width > 0 && <MareyBody key={axis.num} l={axis} st={st} links={links} width={width} plotH={PLOT_H} group={group} />}
    </div>
  );
}

/** The panel's frame: a modal like the saved games panel's, its title the line's or group's bullet,
 * name and "schedule". */
function Frame({ label, title, mode, children }: { label: string; title: ComponentChildren; mode?: ComponentChildren; children: ComponentChildren }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useModalDialog(dialog, true);
  const close = () => (scheduleShown.value = null);
  return (
    <dialog class="saves-panel sched-panel" ref={dialog} aria-label={label}
      onCancel={(e) => { e.preventDefault(); close(); }}
      onClick={(e) => { if (e.target === e.currentTarget) close(); }}
      onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Escape") { e.preventDefault(); close(); } }}>
      <div class="saves-inner">
        <div class="saves-head sp-title">
          <h2>{title}<span class="sp-word">{t("schedulePanel.schedule")}</span></h2>
          {mode}
          <button class="btn icon sp-close" autofocus aria-label={t("modal.close")} data-tip={t("modal.closeTip")} onClick={close}><Icon.cross /></button>
        </div>
        {children}
      </div>
    </dialog>
  );
}

/** The schedule panel while `scheduleShown`; it closes when its line or group goes. */
export function SchedulePanel() {
  const s = scheduleShown.value;
  const w = world.value;
  let body: ComponentChildren = null;
  if (s && w && s.kind === "line") {
    const l = linesById(w).get(s.line), st = l ? w.lineStats[l.id] : undefined;
    if (l && charted(st, l)) {
      body = (
        <Frame label={t("schedulePanel.label", { line: l.name })} title={<><Chip line={l} plain /><span class="sp-name">{l.name}</span></>}
          mode={<span class="sp-mode"><Segmented options={[["simple", t("schedulePanel.simple")]]} value="simple" onChange={() => {}} /></span>}>
          <LineBody l={l} st={st} />
        </Frame>
      );
    }
  } else if (s && w && s.kind === "group") {
    const ls = lineGroups(listedLines(w)).get(s.group) ?? [];
    const rail = ls.filter((l) => charted(w.lineStats[l.id], l));
    // the longest line's stops down the side
    const axis = rail.reduce<LineInput | undefined>((a, l) => (!a || w.lineStats[l.id].km > w.lineStats[a.id].km ? l : a), undefined);
    if (axis && rail.length >= 2) {
      const name = groupName(ls);
      body = (
        <Frame label={t("schedulePanel.label", { line: name })} title={<><GroupPill ls={ls} /><span class="sp-name">{name}</span></>}>
          <GroupBody axis={axis} st={w.lineStats[axis.id]} others={rail.filter((l) => l !== axis).map((l) => l.num)} />
        </Frame>
      );
    }
  }
  // its line gone, stopped or now a bus line: closed, so it does not come back by itself
  useEffect(() => { if (s && w && !body) scheduleShown.value = null; }, [s, w, !body]);
  return <>{body}</>;
}

/** The schedule panel is hidden for players (v0.3.3, Anita 2026-11-13) unless the address has `?timetables`. */
const TIMETABLES_SHOWN = typeof location !== "undefined" && new URLSearchParams(location.search).has("timetables");

/** The line panel's icon beside its schedule: the schedule panel. */
export function ScheduleButton({ l, st }: { l: LineInput; st: LineStats }) {
  if (!TIMETABLES_SHOWN || !charted(st, l)) return null;
  const tip = t("schedulePanel.open");
  return <button class="btn icon mini" aria-label={tip} data-tip={tip} onClick={() => (scheduleShown.value = { kind: "line", line: l.id })}><Icon.timetable /></button>;
}

/** The group view's icon: the group's chart, for a group of two or more running rail lines. */
export function GroupChartButton({ ls }: { ls: readonly LineInput[] }) {
  const w = world.value;
  if (!TIMETABLES_SHOWN || !w || ls.filter((l) => charted(w.lineStats[l.id], l)).length < 2 || !ls[0].group) return null;
  const tip = t("schedulePanel.groupChart");
  return <button class="btn icon mini related-toggle" aria-label={tip} data-tip={tip} onClick={() => (scheduleShown.value = { kind: "group", group: ls[0].group! })}><Icon.marey /></button>;
}
