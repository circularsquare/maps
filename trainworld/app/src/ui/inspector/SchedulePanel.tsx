// T-3346 stage 1: the schedule panel, centred like Community (notes/T-3346.md). A line's, opened
// from its panel's schedule heading: the schedule in Simple mode on the left (the line panel's own
// table, steppers and dots), its departures board and timetable chart on the right. A group's,
// opened from the group view: a timetable of every station of its lines (T-4276) over the chart of
// the longest line, the others drawn where they share its track. A bus or ferry line's (T-3703) is the left side without the board.
// docs/ui-panels.md "Schedule panel".

import { signal } from "@preact/signals";
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { clock, lineSeconds } from "../../game/clock";
import { periodAt, rowsOf } from "../../game/periods";
import { groupName, lineGroups } from "../../game/lineGroups";
import type { LineInput, LineStats } from "../../game/types";
import { ask, display, setDisplay } from "../../game/ui";
import { linesById, listedLines, noteHeadwayLimits, rowCount, rowHeadwayShown, rowTrainsShown, setPeriods, stationNames, stationsById, world } from "../../game/world";
import { t, tList, tNumber } from "../../i18n";
import { queryClock } from "../../workers/clockClient";
import { GroupPill } from "../GroupPill";
import { useModalDialog } from "../Modal";
import { useClockQuery, useLatestAnswer } from "../useClockQuery";
import { Chip, Icon, Segmented } from "../widgets";
import { limitsFrom } from "./LineInspector";
import { useChartLinks } from "./MareyChart";
import { TrainTable, useRelated } from "./TrainTable";
import { NO_LIMITS, Schedule, useLinePeriods, vehicleOf, vehiclesText } from "./schedule";
import { ONLY_TIP, OnlyIcon } from "./stopOnly";

/** What the panel shows: a line's schedule, or a group's chart; null closed. */
export type ScheduleShown = { kind: "line"; line: string } | { kind: "group"; group: number };
export const scheduleShown = signal<ScheduleShown | null>(null);

/** The chart's stops span under the timetable, px. */
const PLOT_H = 240;
/** The board's day starts at this local hour, as Japanese station timetables do. */
const DAY_START_H = 4;

const DAY = 86400, HOUR = 3600;

/** A line the panel opens for: running, with two stops (T-3703: a bus or ferry line too). */
const scheduled = (st: LineStats | undefined, l: LineInput | undefined): st is LineStats => !!st && !!l && st.status === "running" && l.stops.length >= 2;
/** A line the board and chart can show: a scheduled rail line (the sim keeps no bus or ferry timetable). */
const charted = (st: LineStats | undefined, l: LineInput | undefined): st is LineStats => scheduled(st, l) && !st.bus;

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
  // T-4013: trains still leave a getting-off-only stop, so its departures stay, under its mark
  const only = st.stopOnly?.[at];
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
    if (el && row) el.scrollTop = row.offsetTop - el.offsetTop - (el.querySelector<HTMLElement>(".sp-heads")?.offsetHeight ?? 0);
  }, [at, !!shown]);
  return (
    <>
      <div class="sp-tools">
        <span class="h sp-board-h">{t("schedulePanel.departures")}</span>
        <select aria-label={t("schedulePanel.station")} value={at} onChange={(e) => setStation(e.currentTarget.value)}>
          {stops.map((s) => <option value={s}>{names.get(s) ?? ""}</option>)}
        </select>
        {only && <span class="muted sp-only"><OnlyIcon kind={only} />{t(ONLY_TIP[only])}</span>}
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

/** The toggle for other lines' trains on the chart, where any meet it. */
function OthersToggle({ any }: { any: boolean }) {
  if (!any) return null;
  const on = display.value.chartOtherLines;
  const tip = t(on ? "mareyChart.hideOthers" : "mareyChart.showOthers");
  return <button type="button" class={"btn icon mini sp-others" + (on ? " on" : "")} aria-pressed={on} data-tip={tip} aria-label={tip}
    onClick={() => setDisplay("chartOtherLines", !on)}><Icon.related /></button>;
}

// ---------------------------------------------------------------- periods (T-3542)

const PERIOD_CHOICES: [number, string][] = [[3, "3"], [4, "4"], [5, "5"]];

/** The rows of a line of `from` rows that a switch to `to` joins into one (none when `to` is not fewer):
 * five to three joins morning, midday and afternoon, five to four morning and afternoon, four to three
 * rush and midday. */
function joinedRows(from: number, to: number): number[] {
  if (to >= from) return [];
  const into = rowsOf(to), was = rowsOf(from);
  for (const r of into) {
    const olds = was.flatMap((o, i) => (o.periods.some((p) => r.periods.includes(p)) ? [i] : []));
    if (olds.length > 1) return olds;
  }
  return [];
}
/** "Morning, midday and afternoon" for a line of `n` rows' rows `rs`. */
function rowList(n: number, rs: number[]): string {
  const s = tList(rs.map((r) => t("schedulePanel.rowWord", { row: rowsOf(n)[r].code })));
  return s.charAt(0).toLocaleUpperCase() + s.slice(1);
}

/** Switch to `to` rows, after a question where rows are joined (Anita, 2026-11-17). `line`: this line's
 * own choice (0: the game's), else the game's. `from`: the rows the switch moves from, for the question. */
function switchPeriods(l: LineInput, to: number, line: boolean, from: number, after: number) {
  const send = () => setPeriods(to, line ? l.id : undefined);
  const join = joinedRows(from, after);
  if (!join.length) return send();
  const rows = rowList(from, join), count = join.length;
  let text: string;
  if (!line) text = t("schedulePanel.joinAskGame", { rows, count });
  else {
    const auto = rowHeadwayShown(l.id), trains = rowTrainsShown(l.id) ?? [];
    const h = auto ? Math.min(...join.map((r) => auto[r]).filter((x) => x > 0)) : 0;
    text = auto && Number.isFinite(h) && h > 0 ? t("schedulePanel.joinAskHeadway", { rows, count, headway: t("inspector.min", { n: tNumber(h / 60, { maximumFractionDigits: 2 }) }) })
      : t("schedulePanel.joinAskTrains", { rows, count, trains: vehiclesText(vehicleOf(world.value!.lineStats[l.id]), Math.max(0, ...join.map((r) => trains[r] ?? 0))) });
  }
  ask.value = { text, yes: t("schedulePanel.usePeriods", { n: after }), no: t("schedulePanel.keepPeriods", { n: from }), run: send };
}

/** The window's periods: the game's 3 / 4 / 5 (every line on the game's), "Set for this line" at its end; once the
 * line has its own, a "This line" 3 / 4 / 5 under it with "Same as game" at its end (Anita, 2026-11-19). */
function Periods({ l, st }: { l: LineInput; st: LineStats }) {
  const w = world.value!, game = w.periods, own = st.periodsOwn, n = rowCount(st);
  return (
    <div class="sp-periods">
      <span class="sp-periods-label">{t("schedulePanel.periods")}</span>
      <Segmented<number> options={PERIOD_CHOICES} value={game} onChange={(to) => switchPeriods(l, to, false, game, to)} />
      {own > 0 ? <span /> : (
        <button type="button" class="btn text mini sp-periods-own" data-tip={t("schedulePanel.setForLineTip")}
          onClick={() => setPeriods(n, l.id)}>{t("schedulePanel.setForLine")}</button>
      )}
      {own > 0 && (
        <>
          <span class="sp-periods-label">{t("schedulePanel.thisLine")}</span>
          <Segmented<number> options={PERIOD_CHOICES} value={own} onChange={(to) => switchPeriods(l, to, true, n, to)} />
          <button type="button" class="btn text mini sp-periods-own" onClick={() => switchPeriods(l, 0, true, n, game)}>{t("schedulePanel.useGame")}</button>
        </>
      )}
    </div>
  );
}

function LineBody({ l, st }: { l: LineInput; st: LineStats }) {
  const w = world.value!;
  const rv = vehicleOf(st);
  const limits = useClockQuery({ what: "lineHeld", line: l.num, rows: true }, [l.num, w.version, rv], (d) => limitsFrom(d, rv)) ?? NO_LIMITS;
  if (limits !== NO_LIMITS) noteHeadwayLimits(l.id, limits.map((x) => x?.s ?? 0));
  const { periods } = useLinePeriods(l.id, l);
  const { links, related } = useRelated(l, st, useChartLinks(l, st));
  const any = related.length > 0 || (!!links && (links.shared.size > 0 || links.passes.length > 0));
  return (
    <div class="sp-body">
      <div class="sp-left">
        <Schedule l={l} st={st} period={periodAt(lineSeconds(st))} limits={limits} crowd={periods} label={t("schedulePanel.setBy")} weekends hours above={<Periods l={l} st={st} />} />
        <Board key={l.num} l={l} st={st} />
      </div>
      <div class="sp-right">
        <TrainTable axis={l} st={st} related={related} links={links} plotH={PLOT_H} tools={<OthersToggle any={any} />} />
      </div>
    </div>
  );
}

/** T-3703: a bus or ferry line's: the left side alone, with no track limits (no track) and no board (no timetable).
 * T-3767: its periods and hours as a rail line's. */
function BusBody({ l, st }: { l: LineInput; st: LineStats }) {
  const { periods } = useLinePeriods(l.id, l);
  // T-3378: a BRT line's headways are limited by its lanes' room
  const w = world.value!;
  const lanes = !!st.bus?.brt && !!st.bus.built;
  const limits = useClockQuery(lanes ? { what: "lineHeld", line: l.num, rows: true } : null, [l.num, w.version, lanes], (d) => limitsFrom(d, "bus")) ?? NO_LIMITS;
  return (
    <div class="sp-body">
      <div class="sp-left">
        <Schedule l={l} st={st} period={periodAt(lineSeconds(st))} limits={limits} crowd={periods} label={t("schedulePanel.setBy")} weekends hours above={<Periods l={l} st={st} />} />
      </div>
    </div>
  );
}

function GroupBody({ axis, st, others }: { axis: LineInput; st: LineStats; others: LineInput[] }) {
  // kept while the group's lines are the same: the chart redraws its trains when it changes
  const key = others.map((o) => o.num).join(",");
  const group = useMemo<ReadonlySet<number>>(() => new Set(others.map((o) => o.num)), [key]);
  const { links, related } = useRelated(axis, st, useChartLinks(axis, st), group);
  const any = !!links && [...links.shared.keys(), ...links.passes.map((p) => p.o)].some((o) => !group.has(o));
  return (
    <div class="sp-group">
      <TrainTable axis={axis} st={st} others={others} related={related} links={links} group={group} plotH={PLOT_H} tools={<OthersToggle any={any} />} />
    </div>
  );
}

/** The panel's frame: a modal like the saved games panel's, its title the line's or group's bullet,
 * name and "schedule". */
/** The modes the heading offers; the choice is drawn only once there is more than Simple (Detailed and Exact, T-3346 stages 3 and 6). */
const scheduleModes = (): [string, string][] => [["simple", t("schedulePanel.simple")]];

function Frame({ label, title, mode, narrow = false, children }: { label: string; title: ComponentChildren; mode?: ComponentChildren; narrow?: boolean; children: ComponentChildren }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useModalDialog(dialog, true);
  const close = () => (scheduleShown.value = null);
  // a click outside closes it, but not a drag of the timetable let go outside (its click lands on the dialog too)
  const pressedOutside = useRef(false);
  return (
    <dialog class={"saves-panel sched-panel" + (narrow ? " narrow" : "")} ref={dialog} aria-label={label}
      onCancel={(e) => { e.preventDefault(); close(); }}
      onPointerDown={(e) => { pressedOutside.current = e.target === e.currentTarget; }}
      onClick={(e) => { if (e.target === e.currentTarget && pressedOutside.current) close(); }}
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
    if (l && scheduled(st, l)) {
      body = (
        <Frame label={t("schedulePanel.label", { line: l.name })} title={<><Chip line={l} plain /><span class="sp-name">{l.name}</span></>} narrow={!!st.bus}
          mode={scheduleModes().length > 1 ? <span class="sp-mode"><Segmented options={scheduleModes()} value="simple" onChange={() => {}} /></span> : undefined}>
          {st.bus ? <BusBody l={l} st={st} /> : <LineBody l={l} st={st} />}
        </Frame>
      );
    }
  } else if (s && w && s.kind === "group") {
    const ls = lineGroups(listedLines(w)).get(s.group) ?? [];
    const rail = ls.filter((l) => charted(w.lineStats[l.id], l));
    // the longest line's stops down the side
    const axis = rail.reduce<LineInput | undefined>((a, l) => (!a || w.lineStats[l.id].km > w.lineStats[a.id].km ? l : a), undefined);
    if (axis && rail.length >= 2) {
      const name = groupName(ls, s.group);
      body = (
        <Frame label={t("schedulePanel.label", { line: name })} title={<><GroupPill ls={ls} g={s.group} /><span class="sp-name">{name}</span></>}>
          <GroupBody axis={axis} st={w.lineStats[axis.id]} others={rail.filter((l) => l !== axis)} />
        </Frame>
      );
    }
  }
  // its line gone or stopped: closed, so it does not come back by itself
  useEffect(() => { if (s && w && !body) scheduleShown.value = null; }, [s, w, !body]);
  return <>{body}</>;
}

/** The line panel's icon beside its schedule: the schedule panel (T-3543: a size bigger than the others there). */
export function ScheduleButton({ l, st }: { l: LineInput; st: LineStats }) {
  if (!scheduled(st, l)) return null;
  const tip = t("schedulePanel.open");
  return <button class="btn icon mini sched-open" aria-label={tip} data-tip={tip} onClick={() => (scheduleShown.value = { kind: "line", line: l.id })}><Icon.timetable /></button>;
}

/** The group view's icon: the group's timetable and chart, for a group of two or more running rail lines; the
 * line panel's icon and hover, so it is found where a line's is (T-4276). */
export function GroupChartButton({ ls }: { ls: readonly LineInput[] }) {
  const w = world.value;
  if (!w || ls.filter((l) => charted(w.lineStats[l.id], l)).length < 2 || !ls[0].group) return null;
  const tip = t("schedulePanel.open");
  return <button class="btn icon mini related-toggle" aria-label={tip} data-tip={tip} onClick={() => (scheduleShown.value = { kind: "group", group: ls[0].group! })}><Icon.timetable /></button>;
}
