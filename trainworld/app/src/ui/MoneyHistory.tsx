// The Money tab's history (T-447): revenue (fares) against expenses (running trains plus upkeep,
// T-867, and one-off costs; blue and red, T-1111), then train rides (T-565), per game hour
// over the past 24, per day over the past 30, or per day over every day kept (T-2143: a week or a month
// to a column once there are many) (T-2271; the clock worker keeps the days and 48 hours and saves
// them). Days run on the start zone's local time (`MoneyState.dayShift`) and hours are written in it.
// The day and hour now are drawn so far, moving as fares land (T-284). The network's, or the region the
// Money tab is set to (T-531).

import { signal } from "@preact/signals";
import { clockTime, count, money } from "../game/format";
import { DAY1_OF_YEAR, dateName, dateOf, dateRange, shortDate, weekdayOf, withYear, yearOf } from "../game/days";
import { clock } from "../game/clock";
import { moneyState } from "../game/money";
import { cityById, isCity } from "../game/cities";
import { calendarAt } from "../game/holidays";
import { regionLedger } from "../game/regionMoney";
import { Segmented } from "./widgets";
import { t, tNumber, tRich } from "../i18n";

type Span = "1d" | "30d" | "all";
/** The span shown, kept while the game is open. */
const span = signal<Span>("all");
/** T-2271: columns each span keeps, newest last (hours for 1d) */
const KEEP: Record<Span, number> = { "1d": 24, "30d": 30, all: Infinity };

/** `from`, `to` (game days, `to` not included) and `n` (days counted, today's part so far) for a
 * column of several days, whose figures are then averages per day (T-2143). */
interface Entry { at: number; fares: number; running: number; upkeep?: number; loan?: number; trains: number; build: number; rides?: number; from?: number; to?: number; n?: number }

const W = 300, H = 96, PAD = 3;
/** the space between two days' bars (T-2444) */
const GAP = 1.5;
/** A young game's few entries sit at the right of this many slots, not stretched over the width. */
const MIN_SLOTS = { day: 7, hour: 12 };
/** T-2143: at most this many columns of days; past it a column is a week, then a month, then several months. */
const MAX_COLS = 90;

const oneOff = (e: Entry) => e.trains + e.build;
/** T-867: the cost line, running trains plus upkeep, and loan payments (T-1661, T-2022) */
const run = (e: Entry) => e.running + (e.upkeep ?? 0) + (e.loan ?? 0);
const usd = (v: number) => money(v * 1e6);
const rides = (e: Entry) => e.rides ?? 0;
const exact = (v: number) => tNumber(Math.round(v));

/** "Mon Apr 14" or "Mon Apr 14, 04:00" in hovers; without the weekday under the charts. An hour is
 * the clock's, written `sh` hours ahead in the ledger days' local time. */
function when(e: Entry, hours: boolean, sh: number, long = true): string {
  const at = hours ? e.at + sh : e.at;
  const day = hours ? Math.floor(at / 24) : at;
  const date = long ? dateName(day) : shortDate(day);
  return hours ? `${date}, ${clockTime((((at % 24) + 24) % 24) * 60)}` : date;
}

/** The ledger's entries (the network's, or a region's: T-531) with the day or hour now replaced by the live one. */
function entries(hours: boolean, zone: string | null): Entry[] {
  if (!moneyState.value) return [];
  const r = regionLedger(zone);
  const list: Entry[] = hours ? r.hours.map((h) => ({ ...h, at: h.hour })) : r.ledger.map((d) => ({ ...d, at: d.day }));
  const now = hours ? (r.hour && { ...r.hour, at: r.hour.hour }) : (r.today && { ...r.today, at: r.today.day });
  if (now) {
    const last = list[list.length - 1];
    if (last && last.at === now.at) list[list.length - 1] = now;
    else if (!last || now.at > last.at) list.push(now);
  }
  return list;
}

/** The month count of game day `d` since the calendar's start (game/days.ts: 365-day years). */
const monthIndex = (d: number) => Math.floor((d - 1 + DAY1_OF_YEAR) / 365) * 12 + dateOf(d).month;

/** T-2143: days as they are while they fit in MAX_COLS columns, else grouped by week (from Monday),
 * month, or runs of months, each group's figures a day's average over the days it covers, the day
 * now counted as far as it has gone. Days missing from a region's ledger count as days of nothing. */
function binDays(list: Entry[], shift: number): { cols: Entry[]; binned: boolean } {
  const first = list[0].at, last = list[list.length - 1].at;
  if (last - first + 1 <= MAX_COLS) return { cols: list, binned: false };
  const weeks = Math.floor((last - weekdayOf(last) - (first - weekdayOf(first))) / 7) + 1;
  const months = monthIndex(last) - monthIndex(first) + 1;
  const per = Math.ceil(months / MAX_COLS);
  const key = weeks <= MAX_COLS ? (d: number) => d - weekdayOf(d) : (d: number) => Math.floor(monthIndex(d) / per);
  // the day now's part so far, on the clock in the days' local time; a whole day when the clock is elsewhere
  const part = Math.min(1, Math.max(1 / 24, (clock.minute.peek() + shift / 60) / 1440 - last));
  /** the group's first day, or the history's */
  const start = (d: number) => { let s = d; while (s > first && key(s - 1) === key(d)) s--; return s; };
  const cols: Entry[] = [];
  for (const e of list) {
    const c = cols[cols.length - 1];
    if (c && key(c.at) === key(e.at)) {
      c.fares += e.fares; c.running += e.running; c.upkeep! += e.upkeep ?? 0; c.loan! += e.loan ?? 0;
      c.trains += e.trains; c.build += e.build; c.rides! += rides(e);
    } else cols.push({ at: e.at, from: start(e.at), fares: e.fares, running: e.running, upkeep: e.upkeep ?? 0, loan: e.loan ?? 0, trains: e.trains, build: e.build, rides: rides(e) });
  }
  // a group runs to the next one's start (its days with no entry count), the last to the day now
  for (let i = 0; i < cols.length; i++) {
    const c = cols[i];
    c.to = i + 1 < cols.length ? cols[i + 1].from! : last + 1;
    c.n = c.to - c.from! - (i === cols.length - 1 ? 1 - part : 0);
    for (const k of ["fares", "running", "upkeep", "loan", "trains", "build", "rides"] as const) c[k] = (c[k] ?? 0) / c.n;
  }
  return { cols, binned: true };
}

export function MoneyHistory({ zone }: { zone: string | null }) {
  const hours = span.value === "1d";
  const shift = moneyState.value?.dayShift ?? 0, sh = shift / 3600;
  const all = entries(hours, zone);
  if (!all.length) return null;
  // the past 24 hours or 30 days, the hour or day now among them
  const newest = all[all.length - 1].at;
  const raw = all.filter((e) => e.at > newest - KEEP[span.value]);
  const { cols: list, binned } = span.value === "all" ? binDays(raw, shift) : { cols: raw, binned: false };
  const slots = Math.max(list.length, MIN_SLOTS[hours ? "hour" : "day"]);
  const col = W / slots;
  const x0 = W - list.length * col;
  const cx = (i: number) => x0 + (i + 0.5) * col;
  const last = list.length - 1;
  const bw = Math.max(1, col - GAP);
  /** a binned column's figures are per day */
  const per = binned ? t("moneyHistory.perDay") : "";
  // T-2684: years are labelled once the days shown span more than one
  const firstDay = list[0].from ?? list[0].at, lastDay = list[last].to !== undefined ? list[last].to! - 1 : list[last].at;
  const years = span.value === "all" && !hours && lastDay - firstDay + 1 > 365;
  const dayEnd = (e: Entry) => (e.to !== undefined ? e.to - 1 : e.at);
  const name = (e: Entry, i: number) => {
    let w = binned ? dateRange(e.from!, e.to!) : when(e, hours, sh);
    if (years) w = withYear(w, dayEnd(e));
    return i === last ? t("moneyHistory.soFar", { when: w }) : w;
  };

  /** y on a chart from 0 up to `top` */
  const yScale = (top: number) => (v: number) => H - PAD - (v / top) * (H - 2 * PAD);

  // T-2340, T-2444: each column a fares bar and a running costs bar side by side, the one-off costs (new
  // trains, construction) a paler bar stacked on the running costs bar, scaled to the larger of the first two; a one-off bar
  // past the top is cut with a cap so a build day cannot flatten the others (T-452)
  const top = Math.max(...list.map((e) => Math.max(e.fares, run(e))), 1e-9);
  const y1 = yScale(top);
  /** bar `k` of `n` across a column's width */
  const bx = (i: number, k: number, n: number) => cx(i) - bw / 2 + (k * bw) / n;
  const bar = (cls: string, i: number, k: number, n: number, v: number) => {
    const y = Math.max(PAD, y1(v));
    return <rect class={cls + (i === last ? " now" : "")} x={bx(i, k, n)} y={y} width={bw / n} height={Math.max(0.5, H - PAD - y)} />;
  };
  const tip1 = (e: Entry, i: number) => [
    name(e, i),
    t("moneyHistory.fares", { amount: usd(e.fares) + per }),
    t("moneyHistory.running", { amount: usd(-(e.running + (e.upkeep ?? 0))) + per }),
    t("moneyHistory.runningTrains", { amount: usd(-e.running) + per }),
    t("moneyHistory.upkeep", { amount: usd(-(e.upkeep ?? 0)) + per }),
    ...((e.loan ?? 0) > 0 ? [t("moneyHistory.loan", { amount: usd(-(e.loan ?? 0)) + per })] : []),
    ...(oneOff(e) > 0 ? [t("moneyHistory.oneOff", { amount: usd(-oneOff(e)) + per })] : []),
    ...(e.trains > 0 ? [t("moneyHistory.newTrains", { amount: usd(-e.trains) + per })] : []),
    ...(e.build > 0 ? [t("moneyHistory.construction", { amount: usd(-e.build) + per })] : []),
  ].join("\n");

  // train rides (T-565): bars up from zero
  const rideTop = Math.max(...list.map(rides), 1e-9);
  const y3 = yScale(rideTop);

  // T-264: a tick under each day or hour a holiday or a long holiday's day where the player has built
  const only = zone && isCity(zone) ? cityById(zone).country ?? null : null;
  const marks = binned ? [] : list.map((e) => calendarAt(hours ? e.at * 3600 + 1800 : e.at * 86400 + 43200 - shift, only));

  const first = list[0];
  // T-2487: the titles say which scope they are, being a scroll below the switch that sets it
  const where = zone ? cityById(zone).name : t("moneyHistory.allCities");
  return (
    <>
      <div class="h">
        {t("moneyHistory.history")}
        <span class="right hist-switches">
          <Segmented<Span>
            options={[
              ["1d", t("moneyHistory.span1d")],
              ["30d", t("moneyHistory.span30d")],
              ["all", t("moneyHistory.spanAll")],
            ]}
            value={span.value}
            onChange={(v) => { span.value = v; }}
          />
        </span>
      </div>
      <div class="hist-title small"><span>{tRich("moneyHistory.title", { revenue: <span class="hist-fares">{t("moneyHistory.revenue")}</span>, expenses: <span class="hist-costs">{t("moneyHistory.expenses")}</span>, where })}</span><span class="right muted">{usd(top)}{per}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={H - PAD} y2={H - PAD} />
        {list.map((e, i) => {
          const v = oneOff(e), base = run(e);
          const yTop = Math.max(PAD, y1(base + v)), yBase = y1(base);
          return <>
            {bar("hist-fares", i, 0, 2, e.fares)}
            {bar("hist-costs", i, 1, 2, base)}
            {v > 0 && yBase > yTop && <rect class={"hist-oneoff" + (i === last ? " now" : "")} x={bx(i, 1, 2)} y={yTop} width={bw / 2} height={yBase - yTop} />}
            {base + v > top && <path class="hist-oneoff cap" d={`M${bx(i, 1, 2)},${PAD + 3}L${bx(i, 1.5, 2)},0L${bx(i, 2, 2)},${PAD + 3}Z`} />}
          </>;
        })}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={tip1(e, i)} />)}
      </svg>
      <div class="hist-title small"><span>{t("moneyHistory.ridesTitle", { where })}</span><span class="right muted">{count(rideTop)}{per}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={H - PAD} y2={H - PAD} />
        {list.map((e, i) => {
          const y = y3(rides(e));
          return <rect class={"hist-rides" + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={y} width={bw} height={Math.max(0.5, H - PAD - y)} />;
        })}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={`${name(e, i)}\n${t("moneyHistory.rides", { count: exact(rides(e)) + per })}`} />)}
      </svg>
      {marks.some((m) => m.lines.length) && (
        <svg class="hist-hols" viewBox={`0 0 ${W} 4`} preserveAspectRatio="none">
          {list.map((e, i) => marks[i].lines.length ? <rect class={"hist-hol" + (marks[i].off ? "" : " rush")} x={cx(i) - bw / 2} y={0} width={bw} height={4} /> : null)}
          {list.map((e, i) => marks[i].lines.length ? <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={4} data-tip={[name(e, i), ...marks[i].lines].join("\n")} /> : null)}
        </svg>
      )}
      {years && <div class="hist-years small muted">{list.map((e, i) => {
        const y = yearOf(dayEnd(e));
        if (i > 0 && y === yearOf(dayEnd(list[i - 1]))) return null;
        const at = ((x0 + i * col) / W) * 100;
        return <span class={at > 85 ? "late" : ""} style={{ left: `${at}%` }}>{t("moneyHistory.year", { year: y })}</span>;
      })}</div>}
      <div class="hist-x small muted"><span>{when(first, hours, sh, false)}</span><span>{when(binned ? { ...list[last], at: list[last].to! - 1 } : list[last], hours, sh, false)}</span></div>
    </>
  );
}
