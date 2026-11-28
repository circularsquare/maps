// The parts of a line's panel that rail, bus and ferry lines share (T-2373): the schedule table and
// its hovers, the average speed, and the past 24 hours. The vehicle words come from the line
// (`vehicleOf`); what only a mode has (a train's track limits, crowding dots and copy buttons, a bus's
// Changing to trains row) the panel passes in.

import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { lineMoneyTip, Money } from "../MoneyText";
import { count, duration, headway, money, speed, trainKind } from "../../game/format";
import { boatFleet } from "../../game/boats";
import { lowerWord, t, tList, tNumber, tSeries } from "../../i18n";
import { pctWhole } from "./tripText";
import { crowdColour } from "../../game/palette";
import { CUTS_DEFAULT, hourSpan, levelHours, PERIODS, rowAt, rowHourSpans, rowHoursText, rowsOf, typedHours, type Cuts, type Row as PeriodRow } from "../../game/periods";
import { lineReversed } from "../../game/northUp";
import { crowdIn, dayNote, endNames, hasTrains, PERIOD_STRIDE, pendingOf, pendingRiders, profitKnown, runningToday, today, upkeepToday } from "../../game/today";
import { accountVersion, capacity } from "../../game/boarding";
import { onRails, vehicleCars, vehicleOf, type Vehicle } from "../../game/vehicle";
import { lineSeconds } from "../../game/clock";
import { DAY_KIND_NAME } from "../../game/days";
import { kindIn, lineCountry, rowIn } from "../../game/holidays";
import { ask, display, setDisplay } from "../../game/ui";
import { useClockQuery } from "../useClockQuery";
import { DEMAND_INDEX, DEMANDS, loopHalves, type Demand, type LineInput, type LineStats } from "../../game/types";
import { pendingLabel } from "../../game/lineFigures";
import { headwaysFor, lineKind, makeDefaultHeadway, newHeadway, applyDefaultHeadway } from "../../game/newHeadway";
import { liveMoney, moneyState, type MoneyState } from "../../game/money";
import { demandView, lineFiguresIn } from "../../game/demand";
import { AUTO_SLACK, autoTrainsFor, headwayFromMinutes, nextHeadway, pairRows, refusedForCash, restHeadwayStep, restRowTphShown, rowCount, rowHeadwayShown, rowTrainsShown, rowWeekendShown, knock, setHours, setRowAuto, setRowCount, setRowHeadway, setRowWeekendCount, setRowWeekendShare, setWeekendShareAll, stepRowCount, stepRowHeadway, stepRowWeekendCount, stepRowWeekendHeadway, trainCap, typeRowWeekendHeadway, world, MAX_TYPED_VEHICLES, WEEKEND_SHARE_STEP } from "../../game/world";
import { Icon, Row, Segmented, shakeEl, Stepper } from "../widgets";
import { demandPending, Spinner } from "../Spinner";
import { nudgeKey, nudgesSeen, pendingNudge, stepNudge } from "../../game/nudge";

/** A demand level's name, capitalised: "High". */
export const demandLabel = (d: Demand): string => t("schedule.demandName", { demand: d });

/** When a demand level runs, in the line's local time (the hour spans of game/periods.ts), every
 * kind of day alike (T-2335): "High demand period: 06:00-20:00". */
export function levelTimes(d: Demand): string {
  return t("schedule.levelTimes", { demand: d, hours: tSeries(levelHours(d).map(([a, b]) => hourSpan(a, b))) });
}

// ---------------------------------------------------------------- the vehicle

export { vehicleCars, vehicleOf, type Vehicle };
/** The vehicle's plural noun: "trains". */
export const manyOf = (v: Vehicle): string => t("schedule.many", { vehicle: v });
/** As `manyOf`, capitalised: "Trains". */
const manyCap = (v: Vehicle): string => t("schedule.manyCap", { vehicle: v });

/** "1 bus", "3 buses" */
export const vehiclesText = (v: Vehicle, n: number) => t("schedule.vehicles", { vehicle: v, count: n });

export const carsText = (n: number) => t("inspector.cars", { count: n });

/** What buying `n` of a stock costs, or that the depot's spare ones cover it. */
const buys = (unit: (n: number) => string, n: number, price: number, spare: number): string => {
  const bought = Math.max(0, n - spare);
  return bought > 0 ? t("schedule.buys", { unit: unit(bought), price: money(bought * price * 1e6) }) : t("schedule.uses", { unit: unit(n) });
};
export const carsBought = (k: { carPrice: number; spare: number }, cars: number): string => buys(carsText, cars, k.carPrice, k.spare);
/** T-3892: the same for `n` boats of a size. */
export const boatsBought = (k: { price: number; spare: number }, n: number): string => buys((b) => vehiclesText("boat", b), n, k.price, k.spare);

/** What a schedule change draws on: a train's cars of its kind (T-631), or buses or boats. */
function stockOf(m: MoneyState, st: LineStats, v: Vehicle): { unit: (n: number) => string; per: number; price: number; spare: number } | null {
  if (onRails(v)) {
    if (!st.cars) return null;
    const k = trainKind(m, st.fast, st.tram);
    return { unit: carsText, per: st.cars, price: k.carPrice, spare: k.spare };
  }
  const [owned, needed, price] = v === "boat" ? boatFleet(m, st) : [m.fleetBus, m.busesNeeded, m.busPrice];
  return { unit: (n) => vehiclesText(v, n), per: 1, price, spare: Math.max(0, owned - needed) };
}

/** What a stepper's button leads to: the vehicle count, and for a running line what it needs beyond
 * its busiest level, bought unless the depot has spare ones, or what it frees, which stays owned for
 * any line to use. `others`: the other kind of day's counts (T-3521), as the fleet is the most of
 * either. */
function changeTip(counts: readonly number[], lev: number, to: number, st: LineStats, others: readonly number[] = []): string {
  const v = vehicleOf(st);
  const head = vehiclesText(v, to);
  const m = moneyState.value;
  const k = m && st.status === "running" ? stockOf(m, st, v) : null;
  if (!k) return head;
  const next = [...counts];
  next[lev] = to;
  const more = (Math.max(...next, ...others) - Math.max(...counts, ...others)) * k.per;
  if (more > 0) return head + "\n" + buys(k.unit, more, k.price, k.spare);
  return more < 0 ? head + "\n" + t("schedule.frees", { unit: k.unit(-more) }) : head;
}

// ---------------------------------------------------------------- the schedule table

/** What limits a rail line per level (T-679): its shortest headway in seconds with the other lines
 * as they run, and the spot named as the capacity alerts name it ("Terminus at West"). */
/** `capS`: the shortest headway the train count is capped by, which on single track counts the
 * meeting margin for a lone train too (T-3526); `s` otherwise */
export interface Limit { s: number; what: string; capS: number }
export const NO_LIMITS: (Limit | null)[] = [null, null, null];

/** "Terminus at West takes one train every 4m00s at most", or with no room left "... is full" */
export const limitText = (lim: Limit, vehicle: Vehicle) => t("schedule.limit", { full: !Number.isFinite(lim.s), what: lim.what, time: Number.isFinite(lim.s) ? duration(lim.s) : "", vehicle });

const headwayText = (s: number) => (s > 0 ? t("inspector.min", { n: tNumber(s / 60, { maximumFractionDigits: 2 }) }) : t("schedule.noHeadway"));

/** How long a refused edit's control shakes, ms (T-625; style.css `.shake`) */
const SHAKE_MS = 300;

/** T-3542: a line's figures per row (its own for four or five rows, else its three levels'), as the
 * schedule table reads them. */
interface RowsOf { n: number; rows: PeriodRow[]; trip: readonly number[]; settled: readonly number[]; autoSettled: readonly number[] | null }
function rowsFigs(st: LineStats): RowsOf {
  const r = st.rows, n = rowCount(st);
  return { n, rows: rowsOf(n), trip: r ? r.roundTrip : st.roundTripByLevel, settled: r ? r.trains : st.trainsNeeded, autoSettled: r ? r.auto : st.autoS };
}
/** Row `i`'s trains an hour on weekdays, with a hidden twin's (T-3333). */
function rowTph(l: LineInput, st: LineStats, i: number): number {
  const w = world.value;
  if (st.rows && w) return pairRows(w, st)!.tph[i] ?? 0;
  const d = DEMANDS[i], twin = st.twin ? w?.save.lines.find((x) => x.id === st.twin) : undefined;
  return l.tph[d] + (twin?.tph[d] ?? 0);
}

/** One row's vehicles and headway cells: a count stepper and the headway it gives, or (T-625)
 * the headway stepper and the count it gives. A count an edit was refused for cash is red, and
 * its control shakes once; a headway shorter than `limit` allows is red (T-625, T-679). `lev`: the
 * row (T-3542: of three, four or five). */
function LevelCells({ l, st, lev, auto, limits }: { l: LineInput; st: LineStats; lev: number; auto: readonly number[] | null; limits: (Limit | null)[] }) {
  const v = vehicleOf(st);
  const f = rowsFigs(st);
  const trip = f.trip[lev];
  const limit = limits[lev] ?? null;
  const lim = limit?.s ?? 0;
  const loop = loopHalves(st);
  // T-1276: a headway or count still on its way to the worker (an edit takes seconds on a big
  // network) shows the other cell as it will be, not the world's from an edit or two before
  const autoPending = auto ? auto.map((s, i) => s !== f.autoSettled?.[i]) : null;
  const counts = auto
    ? f.settled.map((n, i) => (autoPending![i] ? autoTrainsFor(f.trip[i], auto[i], limits[i]?.s ?? 0, loop) : n))
    : (rowTrainsShown(l.id) ?? f.settled);
  const countPending = !auto && counts[lev] !== f.settled[lev];
  // T-3521: a weekday change buys or frees only past the weekends' trains
  const others = rowWeekendShown(l.id)?.counts ?? [];
  const r = refusedForCash.value;
  const refused = r && r.id === l.id && r.levs[lev] ? r : null;
  const broke = refused !== null && liveMoney.value.cash < refused.need;
  const shake = refused !== null && performance.now() - refused.at < SHAKE_MS;
  // the tutorial's marks are on a train line's cells
  const guide = !onRails(v) ? undefined : lev === 0 ? "trains" : counts[lev] < 1 ? "levels" : undefined;
  const wrap = (child: ComponentChildren) => <span key={refused?.at ?? 0} class={shake ? "shake" : undefined} data-guide={guide}>{child}</span>;
  const n = <span class={broke ? "bad" : undefined}>{counts[lev]}</span>;
  // T-798: a loop's trains are half each way, so its count steps by two
  const k = loop ? 2 : 1;
  if (!auto) {
    const over = limit !== null && counts[lev] > 0 && counts[lev] - trip / limit.s > 0.01;
    // T-3333: a line on both pairs: its hidden twin's trains an hour too
    const tph = countPending ? (trip > 0 ? (counts[lev] * 3600) / trip : 0) : rowTph(l, st, lev);
    // T-3165: one train more than the track lets run, so one is held and the held messages show
    const cap = trainCap(st, trip, limit?.capS ?? null);
    const full = counts[lev] >= cap;
    const grow = (to: number) => setRowCount(l.id, lev, Math.min(to, cap));
    return (
      <>
        <td>
          {wrap(<Stepper value={n} label={t("schedule.stepperLabel", { vehicle: v })} jump
            onStep={(s) => (s > 0 ? !full && grow(counts[lev] + s * k) : stepRowCount(l.id, lev, s))}
            type={{ text: String(counts[lev]), set: (x) => {
              const to = Math.max(0, Math.round(x)), now = counts[lev];
              if (to === now) return;
              if (to < now) { setRowCount(l.id, lev, to); return; }
              return !full && setRowCount(l.id, lev, Math.min(to, cap)) && to <= cap ? undefined : false;
            } }}
            tipDown={changeTip(counts, lev, Math.max(0, counts[lev] - k), st, others)}
            tipUp={full && limit !== null && cap < MAX_TYPED_VEHICLES ? limitText(limit, v) : changeTip(counts, lev, counts[lev] + k, st, others)} />)}
        </td>
        <td>{over ? <span class="bad" data-tip={t("schedule.limitWait", { text: limitText(limit, v), vehicle: v })}>{headway(tph)}</span> : headway(tph)}</td>
      </>
    );
  }
  const short = limit !== null && auto[lev] > 0 && auto[lev] * AUTO_SLACK < limit.s;
  return (
    <>
      <td>
        <span class={"num" + (broke ? " bad" : "")} data-tip={loop
          ? t("schedule.lapTip", { vehicles: vehiclesText(v, counts[lev]), half: counts[lev] / 2, lap: duration(trip / 2) })
          : st.loop ? t("schedule.lapOneWayTip", { vehicles: vehiclesText(v, counts[lev]), lap: duration(trip) })
          : t("schedule.roundTip", { vehicles: vehiclesText(v, counts[lev]), count: counts[lev], trip: duration(trip) })}>{counts[lev]}</span>
      </td>
      <td>
        {wrap(<Stepper wide jump jumpUnit={t("schedule.minUnit")} label={t("schedule.headwayLabel")} value={short ? <span class="bad">{headwayText(auto[lev])}</span> : headwayText(auto[lev])} onStep={(s) => stepRowHeadway(l.id, lev, s)}
          tipDown={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], -1), lim, loop), st, others)}
          tipUp={changeTip(counts, lev, autoTrainsFor(trip, nextHeadway(auto[lev], 1), lim, loop), st, others)}
          type={{ text: auto[lev] ? String(+(auto[lev] / 60).toFixed(2)) : "0", set: (v) => setRowHeadway(l.id, lev, headwayFromMinutes(v)), tip: short ? t("schedule.limitWait", { text: limitText(limit, v), vehicle: v }) : undefined }} />)}
      </td>
    </>
  );
}

// ---------------------------------------------------------------- weekends (T-3521)

/** One row's weekend cells in the schedule panel: as the weekday row's (`LevelCells`), a
 * count stepper and the headway it runs, or under headways the count and a headway stepper. A step
 * sets the line's weekend counts by hand (`setRowWeekendCount`), with the headway stepped to under headways. The headway is the rate settled for
 * weekends (`restTph`), or for a count still on its way the round trip over it. */
function WeekendCells({ l, st, lev, auto, limits }: { l: LineInput; st: LineStats; lev: number; auto: boolean; limits: (Limit | null)[] }) {
  const v = vehicleOf(st), f = rowsFigs(st);
  const week = rowTrainsShown(l.id) ?? f.settled;
  const counts = rowWeekendShown(l.id)?.counts ?? week;
  const n = counts[lev];
  const trip = f.trip[lev], limit = limits[lev] ?? null;
  // the track holds as many on a weekend as on a weekday: the weekdays' cap (T-3165)
  const most = trainCap(st, trip, limit?.s ?? null);
  const tph = restRowTphShown(l.id, st, lev, n);
  const loop = loopHalves(st), k = loop ? 2 : 1;
  if (auto) {
    const lim = limit?.s ?? 0, h = tph > 0 ? 3600 / tph : 0;
    const short = limit !== null && h > 0 && h * AUTO_SLACK < limit.s;
    const stepTip = (dir: number) => {
      const s = restHeadwayStep(n, dir, trip, lim, loop, h);
      return s === null ? undefined : changeTip(counts, lev, Math.min(s.to, most), st, week);
    };
    return (
      <>
        <td>
          <span class="num" data-tip={loop
            ? t("schedule.lapTip", { vehicles: vehiclesText(v, n), half: n / 2, lap: duration(trip / 2) })
            : st.loop ? t("schedule.lapOneWayTip", { vehicles: vehiclesText(v, n), lap: duration(trip) })
            : t("schedule.roundTip", { vehicles: vehiclesText(v, n), count: n, trip: duration(trip) })}>{n}</span>
        </td>
        <td>
          <Stepper wide jump jumpUnit={t("schedule.minUnit")} label={t("schedule.weekendHeadwayLabel")}
            value={short ? <span class="bad">{headwayText(h)}</span> : headwayText(h)}
            onStep={(s) => stepRowWeekendHeadway(l.id, lev, s, trip, lim, loop, most, h)}
            tipDown={stepTip(-1)} tipUp={stepTip(1)}
            type={{ text: h ? String(+(h / 60).toFixed(2)) : "0", set: (x) => { typeRowWeekendHeadway(l.id, lev, x, trip, lim, loop, most); } }} />
        </td>
      </>
    );
  }
  const over = limit !== null && n > 0 && n - trip / limit.s > 0.01;
  const full = n >= most;
  return (
    <>
      <td>
        <Stepper value={n} label={t("schedule.weekendLabel", { vehicle: v })} jump
          onStep={(s) => (s > 0 ? !full && setRowWeekendCount(l.id, lev, n + s * k, most) : stepRowWeekendCount(l.id, lev, s, most))}
          type={{ text: String(n), set: (x) => {
            const to = Math.max(0, Math.round(x));
            if (to === n) return;
            if (to < n) { setRowWeekendCount(l.id, lev, to, most); return; }
            return !full && setRowWeekendCount(l.id, lev, to, most) && to <= most ? undefined : false;
          } }}
          tipDown={changeTip(counts, lev, Math.max(0, n - k), st, week)}
          tipUp={full && limit !== null && most < MAX_TYPED_VEHICLES ? limitText(limit, v) : changeTip(counts, lev, n + k, st, week)} />
      </td>
      <td>{over ? <span class="bad" data-tip={t("schedule.limitWait", { text: limitText(limit, v), vehicle: v })}>{headway(tph)}</span> : headway(tph)}</td>
    </>
  );
}

/** The Weekends section's heading row: its name, the line's percent of its weekdays (greyed while its
 * counts are set by hand, since it no longer says what runs), and the button that sets every line's
 * weekends to that percent after a question. */
function WeekendHead({ l, cols }: { l: LineInput; cols: number }) {
  const wk = rowWeekendShown(l.id);
  const share = wk?.share ?? 100, custom = !!wk?.custom;
  const pct = pctWhole(share / 100);
  const all = () => {
    ask.value = {
      text: t("schedule.weekendAllAsk", { pct }),
      yes: t("schedule.weekendAll"),
      no: t("toolNotes.cancel"),
      run: () => setWeekendShareAll(share),
    };
  };
  return (
    <tr class="sched-day">
      <th colSpan={cols}>
        <span class="sched-day-row">
          <span data-tip={t("schedule.weekendsTip")}>{t("schedule.weekends")}</span>
          <Stepper value={<span class={custom ? "muted" : undefined}>{pct}</span>} label={t("schedule.weekendShareLabel")}
            tipValue={custom ? t("schedule.weekendShareOff") : undefined}
            onStep={(s) => setRowWeekendShare(l.id, share + Math.sign(s) * WEEKEND_SHARE_STEP)}
            type={{ text: String(share), set: (x) => setRowWeekendShare(l.id, x) }} />
          <button type="button" class="btn text mini" onClick={all}>{t("schedule.weekendAll")}</button>
        </span>
      </th>
    </tr>
  );
}

/** T-3542: row `r`'s name for a line of `n` rows: "High" for three, "Midday" for four or five. */
export function rowLabel(n: number, r: number): string {
  const row = rowsOf(n)[r];
  if (!row) return "";
  return n === 3 ? demandLabel(DEMANDS[r]) : t("schedule.rowName", { row: row.code });
}
/** Row `r`'s hours by the line's hours `c`, its name's hover: "High demand period: 06:00-20:00", "Rush:
 * 06:00-10:00, 16:00-20:00". */
function rowTimes(c: Cuts, n: number, r: number): string {
  const row = rowsOf(n)[r];
  if (!row) return levelTimes(DEMANDS[r]);
  const hours = tSeries(rowHourSpans(c, row));
  return n === 3 ? t("schedule.levelTimes", { demand: DEMANDS[r], hours }) : t("schedule.rowTimes", { row: row.code, hours });
}
/** The line's hours (T-3542 stage 2), the default's for a line read before they were kept. */
const hoursOf = (st: LineStats): Cuts => st.hours ?? CUTS_DEFAULT;

/** T-477, T-515, T-2236, T-2335: a row's name in the schedule table, after its crowding dot
 * on today's kind of day (`crowdIn`, from the boarding account) while `v` has one: the fill graded
 * over the line's stretches, the ring (1 px out with a 1 px gap) at its fullest. One hover for both
 * (T-2334): the row's hours, then the dot's. */
function LevelName({ v, n, r, stops, flip, pending, ends, vehicle, hours }: { v: Float64Array | null; n: number; r: number; stops: number; flip: boolean; pending: boolean; ends?: [string, string]; vehicle: Vehicle; hours: Cuts }) {
  const c = v ? crowdIn(v, rowsOf(n)[r].periods, stops, flip, ends, vehicle) : null;
  return (
    <span data-tip={c ? `${rowTimes(hours, n, r)}\n${c.tip}` : rowTimes(hours, n, r)}>
      {c ? <i class="crowd-dot" style={{ background: c.fill, outlineColor: crowdColour(c.peak) }} data-guide={r === 0 ? "crowd" : undefined} />
        : pending && <Spinner class="row-dot" tip={t("schedule.waitingDemand")} />}
      {rowLabel(n, r)}
    </span>
  );
}

/** The Schedule heading, its Trains (Buses, Boats) or Headway switch, and the table of rows. `period`:
 * the period now in the line's local time (its row is bold); `crowd`: the line's periods for the
 * crowding dots (a rail line's); `copy`: the copy and paste buttons; `label`: the heading's words in
 * place of "Schedule" (the schedule panel's "Set by", T-3346); `hours`: the Hours column (a rail line's
 * schedule window, T-3542); `weekends`: a Weekdays and a Weekends
 * section (the schedule panel's, T-3521; T-3542: the row now by the line's own hours, `period` unread); `above`: rows the schedule panel puts under the heading
 * (T-3542: the periods); `nudge`: with `weekends`, the Offset column (a rail line's, T-3742). */
export function Schedule({ l, st, limits = NO_LIMITS, crowd = null, canAuto = true, copy, label, weekends = false, hours = false, above, nudge = false }: {
  l: LineInput; st: LineStats; period: number; limits?: (Limit | null)[]; crowd?: Float64Array | null; canAuto?: boolean; copy?: ComponentChildren; label?: string; weekends?: boolean; hours?: boolean; above?: ComponentChildren; nudge?: boolean;
}) {
  const v = vehicleOf(st);
  const auto = rowHeadwayShown(l.id);
  const stops = l.stops.length;
  const f = rowsFigs(st), now = rowAt(hoursOf(st), f.n, lineSeconds(st));
  // T-2661: the crowding bar in a level's hover reads west to east
  const w = world.value, flip = !!w && lineReversed(l, w, !!st.loop);
  return (
    <>
      <div class={label ? "h set-by" : "h"}>
        {label ?? t("schedule.title")}
        {canAuto && copy}
        {canAuto && (
          <span class="right">
            <Segmented<boolean>
              options={[
                [false, manyCap(v)],
                [true, t("schedule.headway")],
              ]}
              value={auto !== null}
              onChange={(on) => setRowAuto(l.id, on)}
            />
          </span>
        )}
        {weekends && nudge && <OffsetsToggle />}
      </div>
      {above}
      <table class="sched">
        <tr>
          <th>{f.n === 3 ? t("schedule.demand") : t("schedule.period")}</th>
          {weekends && hours && <th>{t("schedule.hours")}</th>}
          <th>{manyCap(v)}</th>
          <th>{t("schedule.headway")}</th>
          {weekends && nudge && display.value.scheduleOffsets && <th><span data-tip={t("schedule.offsetTip")}>{t("schedule.offset")}</span></th>}
        </tr>
        {!weekends && f.rows.map((_, r) => {
          // T-1783: no dot on a row with no vehicles
          const dot = st.status === "running" && (rowTrainsShown(l.id) ?? f.settled)[r] > 0;
          const p = dot && crowd && lineFiguresIn(demandView.value, l.id, stops) ? crowd : null;
          return (
            <tr class={r === now ? "now" : ""}>
              <td><LevelName v={p} n={f.n} r={r} stops={stops} flip={flip} ends={endNames(l)} vehicle={vehicleOf(st)} pending={dot && !p && demandPending.value} hours={hoursOf(st)} /></td>
              <LevelCells l={l} st={st} lev={r} auto={auto} limits={limits} />
            </tr>
          );
        })}
        {weekends && <DaySections l={l} st={st} now={now} limits={limits} crowd={crowd} auto={auto} flip={flip} hours={hours} nudge={nudge} />}
      </table>
      {weekends && <DefaultHeadways id={l.id} n={f.n} auto={rowHeadwayShown(l.id)} />}
    </>
  );
}

/** T-3543, T-3588: the headways every new line of this line's kind starts with, in the schedule panel: what they
 * are now, the button that makes this line's the default for the kind, and the way back. A bus or ferry line's
 * (T-3703) are its tool's until one is set. T-3767: one headway per row of the line's `n` (`headwaysFor`). */
function DefaultHeadways({ id, n, auto }: { id: string; n: number; auto: readonly number[] | null }) {
  const kind = lineKind(id);
  if (!kind) return null;
  const now = headwaysFor(kind, newHeadway.value, n);
  const same = !!auto && !!now && auto.every((s, i) => s === now[i]);
  return (
    <div class="sched-default">
      <div class="sched-default-head">{t(KIND_TITLE[kind])}</div>
      <div class="muted">{now ? tList(now.map(headwayText)) : t("schedule.defaultNone")}</div>
      <div class="sched-default-btns">
        <button type="button" class="btn text mini" disabled={same} onClick={() => makeDefaultHeadway(id)}>{t("schedule.defaultSet")}</button>
        <button type="button" class="btn text mini" disabled={!now || same} onClick={() => applyDefaultHeadway(id)}>{t("schedule.defaultUse")}</button>
      </div>
    </div>
  );
}
const KIND_TITLE = { rail: "schedule.defaultTitle", tram: "schedule.defaultTitleTram", fast: "schedule.defaultTitleFast", bus: "schedule.defaultTitleBus", ferry: "schedule.defaultTitleFerry" } as const;

/** T-3521: the schedule panel's table body: a Weekdays section and a Weekends section, each a row a
 * row of the line's (T-3542). Today's kind of day bolds its row now (`now`), as the line panel does, and has `crowd`'s dots;
 * the other section's dots are a weekday's or a Saturday's (the boarding account's kinds 0 and 1). */
function DaySections({ l, st, now, limits, crowd, auto, flip, hours, nudge }: { l: LineInput; st: LineStats; now: number; limits: (Limit | null)[]; crowd: Float64Array | null; auto: readonly number[] | null; flip: boolean; hours: boolean; nudge: boolean }) {
  const showOffsets = nudge && display.value.scheduleOffsets;
  const cols = (hours ? 4 : 3) + (showOffsets ? 1 : 0);
  const w = world.value, stops = l.stops.length;
  // T-3742: each row's offset, read again with every state; a step shows at once and stays until a readout holds it
  const v = w?.version ?? 0;
  const nudges = useClockQuery(showOffsets ? { what: "lineNudges", line: l.num } : null, [l.num, v, showOffsets], (d) => ({ v, rows: rowNudges(d) }));
  const restNudges = useClockQuery(showOffsets ? { what: "lineNudges", line: l.num, weekend: true } : null, [l.num, v, showOffsets], (d) => ({ v, rows: rowNudges(d) }));
  useEffect(() => { if (nudges && restNudges) nudgesSeen(l.id, Math.min(nudges.v, restNudges.v)); }, [nudges, restNudges]);
  const today = rowIn(lineCountry(l.stops), Math.floor(lineSeconds(st) / 86400));
  const acct = accountVersion.value;
  const other = 1 - today;
  const asked = useClockQuery(st.status === "running" ? { what: "linePeriods", line: l.num, day: other } : null, [l.num, w?.version, acct, other], (x) => (x.length ? x : null)) ?? null;
  const crowdOf = (row: number) => (row === today ? crowd : asked);
  const f = rowsFigs(st);
  // T-3542: the row whose typed hours left some row too short, and the hours it needs
  const [short, setShort] = useState<{ r: number; need: number } | null>(null);
  useEffect(() => {
    if (!short) return;
    const id = setTimeout(() => setShort(null), SHORT_SHOWN_MS);
    return () => clearTimeout(id);
  }, [short]);
  return (
    <>
      {[0, 1].map((row) => (
        <>
          {row === 0 ? <tr class="sched-day"><th colSpan={cols}>{t("schedule.weekdays")}</th></tr> : <WeekendHead l={l} cols={cols} />}
          {f.rows.map((_, lev) => {
            const shown = row === 0 ? rowTrainsShown(l.id) ?? f.settled : rowWeekendShown(l.id)?.counts ?? f.settled;
            // T-1783: no dot on a row with no vehicles that day
            const dot = st.status === "running" && shown[lev] > 0;
            const c = crowdOf(row);
            const p = dot && c && lineFiguresIn(demandView.value, l.id, stops) ? c : null;
            return (
              <>
              <tr class={lev === now && today === row ? "now" : ""}>
                <td><LevelName v={p} n={f.n} r={lev} stops={stops} flip={flip} ends={endNames(l)} vehicle={vehicleOf(st)} pending={dot && !p && demandPending.value} hours={hoursOf(st)} /></td>
                {hours && <td>{row === 0 && <HoursBox l={l} st={st} n={f.n} r={lev} onShort={(need) => setShort(need ? { r: lev, need } : null)} />}</td>}
                {row === 0 ? <LevelCells l={l} st={st} lev={lev} auto={auto} limits={limits} /> : <WeekendCells l={l} st={st} lev={lev} auto={auto !== null} limits={limits} />}
                {showOffsets && <OffsetCell l={l} lev={lev} n={(row === 1 ? restNudges : nudges)?.rows[lev]} weekend={row === 1} />}
              </tr>
              {row === 0 && short?.r === lev && <tr class="sched-short"><td colSpan={cols} class="bad">{t(short.need > 2 ? "schedule.hoursShortHigh" : "schedule.hoursShort", { count: short.need })}</td></tr>}
              </>
            );
          })}
        </>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- offsets (T-3742)

/** T-4510: the header's button that shows the Offset column (off until a player turns it on). */
function OffsetsToggle() {
  const on = display.value.scheduleOffsets;
  const tip = t(on ? "schedule.offsetsHide" : "schedule.offsetsShow");
  return <button type="button" class={"btn icon mini related-toggle" + (on ? " on" : "")} aria-pressed={on} data-tip={tip} aria-label={tip}
    onClick={() => setDisplay("scheduleOffsets", !on)}><Icon.shift /></button>;
}

/** A row's offset as the track model reads it (`TrackApi.line_nudges`): s, half its headway (the range either way;
 * 0 where it runs nothing), whether the row is held, and the other lines timed with it, which move with it. */
interface RowNudge { offset: number; half: number; held: boolean; others: number[] }
function rowNudges(d: Float64Array): RowNudge[] {
  const out: RowNudge[] = [];
  for (let i = 0; i + 4 <= d.length; i += 4 + d[i + 3]) out.push({ offset: d[i], half: d[i + 1], held: d[i + 2] > 0, others: Array.from(d.subarray(i + 4, i + 4 + d[i + 3])) });
  return out;
}
/** A click's step, s: 1 min, with Shift 15 s, with Ctrl 10 min. */
const NUDGE_STEPS: [number, number, number] = [60, 15, 600];
/** "+4 min", "−1m15s", "0 min". */
function offsetText(s: number): string {
  const r = Math.round(s), a = Math.abs(r);
  const v = a % 60 === 0 ? t("inspector.min", { n: tNumber(a / 60) }) : duration(a);
  return r > 0 ? "+" + v : r < 0 ? "−" + v : v;
}

/** One row's Offset cell: a stepper, each step one edit (`stepNudge`) within half the headway, the value's hover naming
 * the lines timed with this one; a weekend row's is its own (T-4510), shown muted until set. Empty on a row with no trains. */
function OffsetCell({ l, lev, n, weekend }: { l: LineInput; lev: number; n: RowNudge | undefined; weekend: boolean }) {
  if (!n || n.half <= 0) return <td />;
  const p = pendingNudge.value.get(nudgeKey(l.id, lev, weekend));
  const now = p?.target ?? n.offset;
  const end = n.half - 0.5;
  const step = (d: number) => {
    const to = Math.max(-n.half, Math.min(n.half, now + d));
    if (Math.abs(to - now) < 0.5) return false;
    stepNudge(l.id, lev, now, to, weekend);
    return true;
  };
  const lines = world.value?.save.lines;
  const names = n.others.map((o) => lines?.find((x) => x.num === o)?.name).filter((x): x is string => !!x);
  return (
    <td class="sched-offset">
      <Stepper value={<span class={n.held || p ? undefined : "muted"}>{offsetText(now)}</span>} label={t("schedule.offsetLabel")}
        steps={{ by: NUDGE_STEPS, tip: t("schedule.offsetSteps") }}
        tipDown={now <= -end ? t("schedule.offsetEnd") : t("schedule.offsetEarlier")}
        tipUp={now >= end ? t("schedule.offsetEnd") : t("schedule.offsetLater")}
        tipValue={names.length ? t("schedule.offsetTimed", { lines: tList(names), count: names.length }) : undefined}
        onStep={step} />
    </td>
  );
}

/** How long the hours a row's box was refused for show under it, ms. */
const SHORT_SHOWN_MS = 5000;

/** T-3542 stage 2: a row's hours in the schedule window ("6-10"; rush "6-10, 16-20"), typed in whole
 * hours. They set the game's hours, or the line's own once it has its own periods. Hours that leave
 * some row under two hours (a line of three rows' high under six) are refused: the box shakes and
 * `onShort` gets the hours that row needs (0 clears it). */
function HoursBox({ l, st, n, r, onShort }: { l: LineInput; st: LineStats; n: number; r: number; onShort: (need: number) => void }) {
  const cuts = hoursOf(st), row = rowsOf(n)[r];
  const text = row ? rowHoursText(cuts, row) : "";
  const [draft, setDraft] = useState<string | null>(null);
  const done = useRef(false), fresh = useRef(false);
  const box = useRef<HTMLSpanElement>(null);
  const open = (el: HTMLInputElement | null) => {
    if (!el || !fresh.current) return;
    fresh.current = false;
    el.focus();
    el.select();
  };
  const commit = () => {
    if (done.current) return;
    done.current = true;
    const d = draft;
    setDraft(null);
    if (d === null || d.trim() === text) return;
    const got = typedHours(cuts, n, r, d);
    if (!got) return;
    if ("short" in got) { knock(); shakeEl(box.current); onShort(got.short); return; }
    onShort(0);
    setHours(got.cuts, st.periodsOwn > 0 ? l.id : undefined);
  };
  const label = t("schedule.hoursLabel", { row: rowLabel(n, r) });
  return (
    <span ref={box} class="sched-hours" onAnimationEnd={() => box.current?.classList.remove("shake")}>
      {draft === null
        ? <button class="v typed" aria-label={t("widgets.type", { label })} onClick={() => { done.current = false; fresh.current = true; setDraft(text); }}>{text}</button>
        : <input class="v typing" ref={open} value={draft} aria-label={label}
          onInput={(e) => setDraft(e.currentTarget.value)} onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") { done.current = true; setDraft(null); }
          }} />}
    </span>
  );
}

/** The line's average speed end to end: the distance both ways over the round trip minus the turning
 * at the ends, so it counts the stops and the waiting at busy track but not the layover. */
export function AverageSpeed({ st, lev }: { st: LineStats; lev: number }) {
  const rt = st.roundTripByLevel[lev] ?? st.roundTripS;
  // T-798: a loop once round in its lap, minus a balloon's turning
  const s = st.loop ? rt / (loopHalves(st) ? 2 : 1) - (st.loop === "balloon" ? st.turnaroundS : 0) : rt - 2 * st.turnaroundS;
  if (!(s > 0) || !(st.km > 0)) return null;
  const km = st.loop ? st.km : 2 * st.km;
  const kmh = km / (s / 3600);
  return (
    <li>
      <span class="grow">{t("schedule.averageSpeed")}</span>
      <span class="val">{speed(kmh)}</span>
    </li>
  );
}

// ---------------------------------------------------------------- the past 24 hours

/** The line's kind of day now (T-872, T-262: its country decides) and the boarding account's numbers
 * by period on it (`TrackApi.line_periods`, T-477, T-491), asked again as the account changes. */
export function useLinePeriods(id: string, l: LineInput | undefined): { kind: number; periods: Float64Array | null } {
  const w = world.value;
  const kind = kindIn(lineCountry(l?.stops ?? []), Math.floor(lineSeconds(w?.lineStats[id]) / 86400));
  const acct = accountVersion.value;
  const periods = useClockQuery(l ? { what: "linePeriods", line: l.num, day: kind } : null, [l?.num, w?.version, acct, kind], (d) => (d.length ? d : null)) ?? null;
  return { kind, periods };
}

/** T-820, T-2373: the day's rider-seconds aboard over what the trips hold at the vehicle's rated load
 * (T-2186: a train's 160 a car, so past 1 up to crush's 1.5; a bus's 80, which is its crush; a boat's
 * 320, up to 480), each trip taken as its level's time from the first stop to the last; null with no
 * trips. */
function averageLoad(v: Float64Array, st: LineStats): number | null {
  const { rated, crush } = capacity(vehicleCars(st));
  let aboard = 0, room = 0;
  for (let i = 0; i < PERIODS.length; i++) {
    const o = i * PERIOD_STRIDE;
    if (!(v[o] > 0)) continue;
    const times = st.fromStartByLevel[DEMAND_INDEX[PERIODS[i].demand]] ?? st.fromStartS;
    aboard += v[o + 5];
    room += v[o] * (times[times.length - 1] ?? 0) * rated;
  }
  return room > 0 ? Math.min(crush / rated, aboard / room) : null;
}

/** The line's past 24 hours (T-446, game/today.ts): riders, `extra` rows (hidden while the riders are
 * worked out again), the expected profit for today (a whole day, T-2475), the average load, and its vehicles from the boarding account (T-230): how
 * many left full, and for a train line who was left on the platform. Its own component: it re-renders
 * as the counts move. */
export function PastDay({ l, st, periods, kind, extra }: { l: LineInput; st: LineStats; periods: Float64Array | null; kind: number; extra?: ComponentChildren }) {
  const d = today.value.get(l.id);
  const pending = pendingOf(l);
  // T-2678: while pending, the last figure there is (its count, its last one, or demand's), the label beside it
  const shown = pending ? pendingRiders(l) : null;
  if (!d) {
    if (!pending) return null;
    return (
      <>
        <div class="h divided">{shown !== null && <span class="pending-note muted">{pendingLabel(pending)}</span>}</div>
        <ul class="rows"><Row label={t("schedule.riders")}>{shown !== null ? count(shown) : <span class="muted">{pendingLabel(pending)}</span>}</Row></ul>
      </>
    );
  }
  const v = vehicleOf(st);
  const load = periods ? averageLoad(periods, st) : null;
  const run = runningToday(l, st);
  // T-2392: while the riders are worked out again the old figures stay, the label above them
  return (
    <>
      <div class="h divided">{pending && <span class="pending-note muted">{pendingLabel(pending)}</span>}</div>
      <ul class="rows">
        <Row label={t("schedule.riders")}>{count(shown ?? d.riders)}</Row>
        {!pending && extra}
        {profitKnown(l) ? (
          <li data-tip={lineMoneyTip(d.dayFares, run, upkeepToday(l, st), v === "boat" ? "boats" : v === "tram" ? "trams" : v === "bus", dayNote(l, st))}>
            <span class="grow">{t("schedule.profit")}</span>
            <span class="val"><Money dollars={d.dayFares - run - upkeepToday(l, st)} /></span>
          </li>
        ) : hasTrains(l) && <Row label={t("schedule.profit")}><span class="muted">{t("schedule.calculating")}</span></Row>}
        {load !== null && (
          <li data-tip={t("schedule.loadTip", { day: lowerWord(DAY_KIND_NAME[kind]).replace(/ day$/, "") })}>
            <span class="grow">{t("schedule.averageLoad")}</span>
            <span class="val">{pctWhole(load)}</span>
          </li>
        )}
        {d.fullTrains > 0 && <Row label={t("schedule.leavingFull", { vehicle: v })}><span data-tip={t("schedule.departedFull", { full: count(d.fullTrains), all: count(d.trains), vehicle: v })}>{count(d.fullTrains)}/{count(d.trains)}</span></Row>}
        {v !== "bus" && (
          <li data-tip={t("schedule.leftBehindTip", { left: count(d.leftBehind), all: count(d.riders + d.leftBehind) })}>
            <span class="grow">{t("schedule.leftBehind", { vehicle: v })}</span>
            <span class="val">{count(d.leftBehind)}/{count(d.riders + d.leftBehind)}</span>
          </li>
        )}
      </ul>
    </>
  );
}
