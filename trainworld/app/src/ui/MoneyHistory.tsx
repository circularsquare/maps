// The Money tab's history (T-447): revenue (fares) against expenses (running trains plus upkeep,
// T-867, and one-off costs; blue and red, T-1111), then net, then train rides (T-565), per game hour
// over the past 24, per day over the past 30, or per day over every day kept (T-2143: a week or a month
// to a column once there are many) (T-2271; the clock worker keeps the days and 48 hours and saves
// them). Days run on the start zone's local time (`MoneyState.dayShift`) and hours are written in it.
// The day and hour now are drawn so far, moving as fares land (T-284). The network's, or the region the
// Money tab is set to (T-531).

import { signal } from "@preact/signals";
import { clockTime, count, money } from "../game/format";
import { DAY1_OF_YEAR, dateName, dateOf, dateRange, shortDate, weekdayOf } from "../game/days";
import { clock } from "../game/clock";
import { moneyState } from "../game/money";
import { cityById, isCity } from "../game/cities";
import { calendarAt } from "../game/holidays";
import { regionLedger } from "../game/regionMoney";
import { Segmented } from "./widgets";
import { Money } from "./MoneyText";

type Span = "1d" | "30d" | "all";
/** The span shown, kept while the game is open. */
const span = signal<Span>("all");
/** T-2271: columns each span keeps, newest last (hours for 1d) */
const KEEP: Record<Span, number> = { "1d": 24, "30d": 30, all: Infinity };

/** `from`, `to` (game days, `to` not included) and `n` (days counted, today's part so far) for a
 * column of several days, whose figures are then averages per day (T-2143). */
interface Entry { at: number; fares: number; running: number; upkeep?: number; loan?: number; trains: number; build: number; rides?: number; from?: number; to?: number; n?: number }

const W = 300, H = 96, PAD = 3;
/** A young game's few entries sit at the right of this many slots, not stretched over the width. */
const MIN_SLOTS = { day: 7, hour: 12 };
/** T-2143: at most this many columns of days; past it a column is a week, then a month, then several months. */
const MAX_COLS = 90;

const oneOff = (e: Entry) => e.trains + e.build;
/** T-867: the cost line, running trains plus upkeep, and loan payments (T-1661, T-2022) */
const run = (e: Entry) => e.running + (e.upkeep ?? 0) + (e.loan ?? 0);
const costs = (e: Entry) => run(e) + oneOff(e);
const net = (e: Entry) => e.fares - costs(e);
const usd = (v: number) => money(v * 1e6);
const rides = (e: Entry) => e.rides ?? 0;
const exact = (v: number) => Math.round(v).toLocaleString("en-US");

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
  const bw = Math.max(1, col * 0.7);
  /** a binned column's figures are per day */
  const per = binned ? "/day" : "";
  const name = (e: Entry, i: number) => (binned ? dateRange(e.from!, e.to!) : when(e, hours, sh)) + (i === last ? " so far" : "");

  /** y on a chart from 0 up to `top` */
  const yScale = (top: number) => (v: number) => H - PAD - (v / top) * (H - 2 * PAD);

  // fares and running costs as lines, scaled to the larger of the two; one-off costs (new trains,
  // construction) as bars on the same scale, cut at the top with a cap so a build day cannot
  // flatten the lines (T-452)
  const top = Math.max(...list.map((e) => Math.max(e.fares, run(e))), 1e-9);
  const y1 = yScale(top);
  // the day or hour now is drawn dashed: it is still filling
  const pt = (v: (e: Entry) => number, i: number) => `${cx(i).toFixed(1)},${y1(v(list[i])).toFixed(1)}`;
  const path = (v: (e: Entry) => number) => list.slice(0, last).map((_, i) => `${i ? "L" : "M"}${pt(v, i)}`).join("");
  const tail = (v: (e: Entry) => number) => (last > 0 ? `M${pt(v, last - 1)}L${pt(v, last)}` : "");
  const tip1 = (e: Entry, i: number) => [
    name(e, i),
    `Fares ${usd(e.fares)}${per}`,
    `Running costs ${usd(-(e.running + (e.upkeep ?? 0)))}${per}`,
    `Running trains ${usd(-e.running)}${per}`,
    `Upkeep ${usd(-(e.upkeep ?? 0))}${per}`,
    ...((e.loan ?? 0) > 0 ? [`Loan ${usd(-(e.loan ?? 0))}${per}`] : []),
    ...(oneOff(e) > 0 ? [`One-off costs ${usd(-oneOff(e))}${per}`] : []),
    ...(e.trains > 0 ? [`New trains ${usd(-e.trains)}${per}`] : []),
    ...(e.build > 0 ? [`Construction ${usd(-e.build)}${per}`] : []),
  ].join("\n");

  // net: bars up or down from zero
  const hi = Math.max(0, ...list.map(net)), lo = Math.min(0, ...list.map(net));
  const range = hi - lo || 1e-9;
  const y2 = (v: number) => PAD + ((hi - v) / range) * (H - 2 * PAD);
  const zero = y2(0);

  // train rides (T-565): bars up from zero
  const rideTop = Math.max(...list.map(rides), 1e-9);
  const y3 = yScale(rideTop);

  // T-264: a tick under each day or hour a holiday or a long holiday's day where the player has built
  const only = zone && isCity(zone) ? cityById(zone).country ?? null : null;
  const marks = binned ? [] : list.map((e) => calendarAt(hours ? e.at * 3600 + 1800 : e.at * 86400 + 43200 - shift, only));

  const first = list[0];
  return (
    <>
      <div class="h">
        History
        <span class="right hist-switches">
          <Segmented<Span>
            options={[
              ["1d", "1d"],
              ["30d", "30d"],
              ["all", "All time"],
            ]}
            value={span.value}
            onChange={(v) => { span.value = v; }}
          />
        </span>
      </div>
      <div class="hist-title small"><span><span class="hist-fares">Revenue</span> and <span class="hist-costs">expenses</span></span><span class="right muted">{usd(top)}{per}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={H - PAD} y2={H - PAD} />
        {list.map((e, i) => {
          const v = oneOff(e);
          if (v <= 0) return null;
          const y = Math.max(PAD, y1(v));
          return <>
            <rect class={"hist-oneoff" + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={y} width={bw} height={Math.max(0.5, H - PAD - y)} />
            {v > top && <path class="hist-oneoff cap" d={`M${cx(i) - bw / 2},${PAD + 3}L${cx(i)},0L${cx(i) + bw / 2},${PAD + 3}Z`} />}
          </>;
        })}
        <path class="hist-costs" d={path(run)} />
        <path class="hist-fares" d={path((e) => e.fares)} />
        <path class="hist-costs now" d={tail(run)} />
        <path class="hist-fares now" d={tail((e) => e.fares)} />
        {list.map((e, i) => <circle class="hist-costs dot" cx={cx(i)} cy={y1(run(e))} r={1.6} />)}
        {list.map((e, i) => <circle class="hist-fares dot" cx={cx(i)} cy={y1(e.fares)} r={1.6} />)}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={tip1(e, i)} />)}
      </svg>
      <div class="hist-title small"><span>Net</span><span class="right muted">{usd(hi)}{per}{lo < 0 ? <> to <Money dollars={lo * 1e6} />{per}</> : ""}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={zero} y2={zero} />
        {list.map((e, i) => {
          const v = net(e), y = y2(v);
          return <rect class={(v < 0 ? "hist-loss" : "hist-gain") + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={Math.min(y, zero)} width={bw} height={Math.max(0.5, Math.abs(y - zero))} />;
        })}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={`${name(e, i)}\nNet ${usd(net(e))}${per}`} />)}
      </svg>
      <div class="hist-title small"><span>Train rides</span><span class="right muted">{count(rideTop)}{per}</span></div>
      <svg class="hist" viewBox={`0 0 ${W} ${H}`}>
        <line class="hist-axis" x1={0} x2={W} y1={H - PAD} y2={H - PAD} />
        {list.map((e, i) => {
          const y = y3(rides(e));
          return <rect class={"hist-rides" + (i === last ? " now" : "")} x={cx(i) - bw / 2} y={y} width={bw} height={Math.max(0.5, H - PAD - y)} />;
        })}
        {list.map((e, i) => <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={H} data-tip={`${name(e, i)}\nTrain rides ${exact(rides(e))}${per}`} />)}
      </svg>
      {marks.some((m) => m.lines.length) && (
        <svg class="hist-hols" viewBox={`0 0 ${W} 4`} preserveAspectRatio="none">
          {list.map((e, i) => marks[i].lines.length ? <rect class={"hist-hol" + (marks[i].off ? "" : " rush")} x={cx(i) - bw / 2} y={0} width={bw} height={4} /> : null)}
          {list.map((e, i) => marks[i].lines.length ? <rect class="hist-col" x={x0 + i * col} y={0} width={col} height={4} data-tip={[name(e, i), ...marks[i].lines].join("\n")} /> : null)}
        </svg>
      )}
      <div class="hist-x small muted"><span>{when(first, hours, sh, false)}</span><span>{when(binned ? { ...list[last], at: list[last].to! - 1 } : list[last], hours, sh, false)}</span></div>
    </>
  );
}
